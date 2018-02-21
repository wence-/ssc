from __future__ import absolute_import
import numpy
import operator

from firedrake.petsc import PETSc
from firedrake import VectorElement, MixedElement, PCBase

from pyop2 import op2
from pyop2 import base as pyop2
from pyop2 import sequential as seq
from pyop2.datatypes import IntType

from ssc import PatchPC as cPatchPC


class DenseSparsity(object):
    def __init__(self, rset, cset):
        self.shape = (1, 1)
        self._nrows = rset.size
        self._ncols = cset.size
        self._dims = (((1, 1), ), )
        self.dims = self._dims
        self.dsets = rset, cset

    def __getitem__(self, *args):
        return self


class MatArg(seq.Arg):
    def c_addto(self, i, j, buf_name, tmp_name, tmp_decl,
                extruded=None, is_facet=False, applied_blas=False):
        # Override global c_addto to index the map locally rather than globally.
        # Replaces MatSetValuesLocal with MatSetValues
        from pyop2.utils import as_tuple
        rmap, cmap = as_tuple(self.map, op2.Map)
        rset, cset = self.data.sparsity.dsets
        nrows = sum(m.arity*s.cdim for m, s in zip(rmap, rset))
        ncols = sum(m.arity*s.cdim for m, s in zip(cmap, cset))
        rows_str = "%s + n * %s" % (self.c_map_name(0, i), nrows)
        cols_str = "%s + n * %s" % (self.c_map_name(1, j), ncols)

        if extruded is not None:
            raise NotImplementedError("Not for extruded right now")

        if is_facet:
            raise NotImplementedError("Not for interior facets and extruded")

        ret = []
        addto_name = buf_name
        if rmap.vector_index is not None or cmap.vector_index is not None:
            raise NotImplementedError
        ret.append("""MatSetValues(%(mat)s, %(nrows)s, %(rows)s,
                                         %(ncols)s, %(cols)s,
                                         (const PetscScalar *)%(vals)s,
                                         %(insert)s);""" %
                   {'mat': self.c_arg_name(i, j),
                    'vals': addto_name,
                    'nrows': nrows,
                    'ncols': ncols,
                    'rows': rows_str,
                    'cols': cols_str,
                    'insert': "INSERT_VALUES" if self.access == op2.WRITE else "ADD_VALUES"})
        return "\n".join(ret)


class DenseMat(pyop2.Mat):
    def __init__(self, dset):
        self._sparsity = DenseSparsity(dset, dset)
        self.dtype = numpy.dtype(PETSc.ScalarType)

    def __call__(self, access, path):
        path_maps = [arg.map for arg in path]
        path_idxs = [arg.idx for arg in path]
        return MatArg(self, path_maps, path_idxs, access)


class JITModule(seq.JITModule):
    @classmethod
    def _cache_key(cls, *args, **kwargs):
        # No caching
        return None


def matrix_funptr(form):
    from tsfc import compile_form
    from firedrake.tsfc_interface import KernelInfo, Kernel
    test, trial = map(operator.methodcaller("function_space"), form.arguments())
    if test != trial:
        raise NotImplementedError("Only for matching test and trial spaces")
    kernel, = compile_form(form, prefix="subspace_form")

    kinfo = KernelInfo(kernel=Kernel(kernel.ast, kernel.ast.name),
                       integral_type=kernel.integral_type,
                       oriented=kernel.oriented,
                       subdomain_id=kernel.subdomain_id,
                       domain_number=kernel.domain_number,
                       coefficient_map=kernel.coefficient_numbers,
                       needs_cell_facets=False,
                       pass_layer_arg=False)

    if kinfo.subdomain_id != "otherwise":
        raise NotImplementedError("Only for full domain integrals")
    if kinfo.integral_type != "cell":
        raise NotImplementedError("Only for cell integrals")

    # OK, now we've validated the kernel, let's build the callback
    args = []

    toset = op2.Set(1)
    dofset = op2.DataSet(toset, 1)
    arity = sum(m.arity*s.cdim
                for m, s in zip(test.cell_node_map(),
                                test.dof_dset))
    iterset = test.cell_node_map().iterset
    cell_node_map = op2.Map(iterset,
                            toset, arity,
                            values=numpy.zeros(iterset.total_size*arity, dtype=IntType))
    mat = DenseMat(dofset)

    arg = mat(op2.INC, (cell_node_map[op2.i[0]],
                        cell_node_map[op2.i[1]]))
    arg.position = 0
    args.append(arg)

    mesh = form.ufl_domains()[kinfo.domain_number]
    arg = mesh.coordinates.dat(op2.READ, mesh.coordinates.cell_node_map()[op2.i[0]])
    arg.position = 1
    args.append(arg)
    for n in kinfo.coefficient_map:
        c = form.coefficients()[n]
        for (i, c_) in enumerate(c.split()):
            map_ = c_.cell_node_map()
            if map_ is not None:
                map_ = map_[op2.i[0]]
            arg = c_.dat(op2.READ, map_)
            arg.position = len(args)
            args.append(arg)

    iterset = op2.Subset(mesh.cell_set, [0])
    mod = JITModule(kinfo.kernel, iterset, *args)
    return mod._fun, kinfo


def bcdofs(bc, ghost=True):
    # Return the global dofs fixed by a DirichletBC
    # in the numbering given by concatenation of all the
    # subspaces of a mixed function space
    Z = bc.function_space()
    while Z.parent is not None:
        Z = Z.parent

    indices = bc._indices
    offset = 0

    for (i, idx) in enumerate(indices):
        if isinstance(Z.ufl_element(), VectorElement):
            offset += idx
            assert i == len(indices)-1 # assert we're at the end of the chain
            assert Z.sub(idx).value_size == 1
        elif isinstance(Z.ufl_element(), MixedElement):
            if ghost:
                offset += sum(Z.sub(j).dof_count for j in range(idx))
            else:
                offset += sum(Z.sub(j).dof_dset.size * Z.sub(j).value_size for j in range(idx))
        else:
            raise NotImplementedError("How are you taking a .sub?")

        Z = Z.sub(idx)

    bs = Z.value_size
    out = []
    nodes = bc.nodes
    if not ghost:
        nodes = nodes[nodes < Z.dof_dset.size]

    return numpy.concatenate([nodes*bs + j for j in range(bs)]) + offset


def user_construction_op(pc, *args, **kwargs):
    prefix = pc.getOptionsPrefix()
    sentinel = object()
    usercode = PETSc.Options(prefix).getString("pc_patch_construction_python_type", default=sentinel)
    if usercode == sentinel:
        raise ValueError("Must set %spc_patch_construction_python_type" % prefix)

    (modname, funname) = usercode.rsplit('.', 1)
    mod = __import__(modname)
    fun = getattr(mod, funname)
    if isinstance(fun, type):
        fun = fun()
    return fun(pc, *args, **kwargs)


def setup_patch_pc(patch, J, bcs):
    patch = cPatchPC.PC.cast(patch)
    funptr, kinfo = matrix_funptr(J)
    V, _ = map(operator.methodcaller("function_space"), J.arguments())
    mesh = V.ufl_domain()

    if len(bcs) > 0:
        ghost_bc_nodes = numpy.unique(numpy.concatenate([bcdofs(bc, ghost=True) for bc in bcs]))
        global_bc_nodes = numpy.unique(numpy.concatenate([bcdofs(bc, ghost=False) for bc in bcs]))
    else:
        ghost_bc_nodes = numpy.empty(0, dtype=PETSc.IntType)
        global_bc_nodes = numpy.empty(0, dtype=PETSc.IntType)

    op_coeffs = [mesh.coordinates]
    for n in kinfo.coefficient_map:
        op_coeffs.append(J.coefficients()[n])

    op_args = []
    for c in op_coeffs:
        for c_ in c.split():
            op_args.append(c_.dat._data.ctypes.data)
            c_map = c_.cell_node_map()
            if c_map is not None:
                op_args.append(c_map._values.ctypes.data)

    def op(pc, mat, ncell, cells, cell_dofmap):
        funptr(0, ncell, cells, mat.handle,
               cell_dofmap, cell_dofmap, *op_args)
        mat.assemble()
    patch.setDM(mesh._plex)
    patch.setPatchCellNumbering(mesh._cell_numbering)

    offsets = numpy.append([0], numpy.cumsum([W.dof_count for W in V])).astype(PETSc.IntType)
    patch.setPatchDiscretisationInfo([W.dm for W in V],
                                     numpy.array([W.value_size for W in V], dtype=PETSc.IntType),
                                     [W.cell_node_list for W in V],
                                     offsets,
                                     ghost_bc_nodes,
                                     global_bc_nodes)
    patch.setPatchComputeOperator(op)
    patch.setPatchUserConstructionOperator(user_construction_op)
    return patch


class PatchPC(PCBase):

    def initialize(self, pc):
        A, P = pc.getOperators()
        if P.getType() == "python":
            from firedrake.matrix_free.operators import ImplicitMatrixContext
            ctx = P.getPythonContext()
            if ctx is None:
                raise ValueError("No context found on matrix")
            if not isinstance(ctx, ImplicitMatrixContext):
                raise ValueError("Don't know how to get form from %r", ctx)
            J = ctx.a
            bcs = ctx.row_bcs
            if bcs != ctx.col_bcs:
                raise NotImplementedError("Row and column bcs must match")
        else:
            from firedrake.dmhooks import get_appctx
            from firedrake.solving_utils import _SNESContext
            ctx = get_appctx(pc.getDM())
            if ctx is None:
                raise ValueError("No context found on form")
            if not isinstance(ctx, _SNESContext):
                raise ValueError("Don't know how to get form from %r", ctx)
            J = ctx.Jp or ctx.J
            bcs = ctx._problem.bcs

        mesh = J.ufl_domain()
        if mesh.cell_set._extruded:
            raise NotImplementedError("Not implemented on extruded meshes")

        patch = PETSc.PC().create(comm=pc.comm)
        patch.setOptionsPrefix(pc.getOptionsPrefix() + "patch_")
        patch.setOperators(A, P)
        patch.setType("patch")
        patch = setup_patch_pc(patch, J, bcs)
        patch.setFromOptions()
        self.patch = patch

    def update(self, pc):
        self.patch.setUp()

    def apply(self, pc, x, y):
        self.patch.apply(x, y)

    def applyTranspose(self, pc, x, y):
        self.patch.applyTranspose(x, y)

    def view(self, pc, viewer=None):
        self.patch.view(pc, viewer=viewer)

class LineSmoother(object):
    @staticmethod
    def coords(dm, p):
        coordsSection = dm.getCoordinateSection()
        coordsDM = dm.getCoordinateDM()
        dim = coordsDM.getDimension()
        coordsVec = dm.getCoordinatesLocal()
        return dm.getVecClosure(coordsSection, coordsVec, p).reshape(-1, dim).mean(axis=0)

    def sort_entities(self, dm, axis, dir, ndiv):
        # compute
        # [(pStart, (x, y, z)), (pEnd, (x, y, z))]
        entities = [(p, self.coords(dm, p)) for p in range(*dm.getChart())]

        minx = min(entities, key=lambda z: z[1][axis])[1][axis]
        maxx = max(entities, key=lambda z: z[1][axis])[1][axis]
        s = sorted(entities, key=lambda z: z[1][axis], reverse=(dir==-1))

        divisions = numpy.linspace(minx, maxx, ndiv+1)
        (entities, coords) = zip(*s)
        coords = [c[axis] for c in coords]
        indices = numpy.searchsorted(coords[::dir], divisions)

        out = []
        for k in range(ndiv):
            out.append(entities[indices[k]:indices[k+1]])
        out.append(entities[indices[-1]:])

        return out

    def __call__(self, pc):
        dm = pc.getDM()
        prefix = pc.getOptionsPrefix()
        sentinel = object()
        sweeps = PETSc.Options(prefix).getString("pc_patch_construction_ls_sweeps", default=sentinel)
        if sweeps == sentinel:
            raise ValueError("Must set %spc_patch_construction_python_type" % prefix)

        out = []
        for sweep in sweeps.split(':'):
            axis = int(sweep[0])
            dir  = {'+': +1, '-': -1}[sweep[1]]
            ndiv = int(sweep[2:])

            entities = self.sort_entities(dm, axis, dir, ndiv)
            for patch in entities:
                iset = PETSc.IS().createGeneral(patch, comm=PETSc.COMM_SELF)
                out.append(iset)

        return out
