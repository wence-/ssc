from __future__ import absolute_import
import numpy
import operator

from firedrake.petsc import PETSc

from pyop2 import op2
from pyop2 import base as pyop2
from pyop2 import sequential as seq

from ssc import PatchPC


class DenseSparsity(object):
    def __init__(self, rset, cset):
        if isinstance(rset, op2.MixedDataSet) or \
           isinstance(cset, op2.MixedDataSet):
            raise NotImplementedError("Not implemented for mixed sparsities")
        self.shape = (1, 1)
        self._nrows = rset.size
        self._ncols = cset.size
        self._dims = (((rset.cdim, cset.cdim), ), )
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
        maps = as_tuple(self.map, op2.Map)
        nrows = maps[0].split[i].arity
        ncols = maps[1].split[j].arity
        rows_str = "%s + n * %s" % (self.c_map_name(0, i), nrows)
        cols_str = "%s + n * %s" % (self.c_map_name(1, j), ncols)

        if extruded is not None:
            rows_str = extruded + self.c_map_name(0, i)
            cols_str = extruded + self.c_map_name(1, j)

        if is_facet:
            nrows *= 2
            ncols *= 2

        ret = []
        rbs, cbs = self.data.sparsity[i, j].dims[0][0]
        rdim = rbs * nrows
        addto_name = buf_name
        addto = 'MatSetValues'
        if self.data._is_vector_field:
            addto = 'MatSetValuesBlocked'
            rmap, cmap = maps
            rdim, cdim = self.data.dims[i][j]
            if rmap.vector_index is not None or cmap.vector_index is not None:
                raise NotImplementedError
        ret.append("""%(addto)s(%(mat)s, %(nrows)s, %(rows)s,
                                         %(ncols)s, %(cols)s,
                                         (const PetscScalar *)%(vals)s,
                                         %(insert)s);""" %
                   {'mat': self.c_arg_name(i, j),
                    'vals': addto_name,
                    'addto': addto,
                    'nrows': nrows,
                    'ncols': ncols,
                    'rows': rows_str,
                    'cols': cols_str,
                    'insert': "INSERT_VALUES" if self.access == op2.WRITE else "ADD_VALUES"})
        return "\n".join(ret)


class DenseMat(pyop2.Mat):
    def __init__(self, rset, cset):
        self._sparsity = DenseSparsity(rset, cset)
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
    from firedrake.tsfc_interface import compile_form
    test, trial = map(operator.methodcaller("function_space"), form.arguments())
    if test != trial:
        raise NotImplementedError("Only for matching test and trial spaces")
    if len(test) != 1:
        raise NotImplementedError("Not for mixed spaces")
    kernels = compile_form(form, "subspace_form")
    if len(kernels) != 1:
        raise NotImplementedError("Only for single integral")
    kernel = kernels[0]
    kinfo = kernel.kinfo
    if kinfo.subdomain_id != "otherwise":
        raise NotImplementedError("Only for full domain integrals")
    if kinfo.integral_type != "cell":
        raise NotImplementedError("Only for cell integrals")

    # OK, now we've validated the kernel, let's build the callback
    args = []

    mat = DenseMat(test.dof_dset, trial.dof_dset)

    arg = mat(op2.INC, (test.cell_node_map()[op2.i[0]],
                        trial.cell_node_map()[op2.i[1]]))
    arg.position = 0
    args.append(arg)

    mesh = form.ufl_domains()[kinfo.domain_number]
    arg = mesh.coordinates.dat(op2.READ, mesh.coordinates.cell_node_map())
    arg.position = 1
    args.append(arg)
    for n in kinfo.coefficient_map:
        c = form.coefficients()[n]
        for c_ in c.split():
            arg = c_.dat(op2.READ, c_.cell_node_map())
            arg.position = len(args)
            args.append(arg)

    iterset = op2.Subset(mesh.cell_set, [0])
    mod = JITModule(kinfo.kernel, iterset, *args)
    return mod._fun, kinfo


def setup_patch_pc(patch, J, bcs):
    patch = PatchPC.PC.cast(patch)
    funptr, kinfo = matrix_funptr(J)
    V, _ = map(operator.methodcaller("function_space"), J.arguments())
    mesh = V.ufl_domain()

    if len(bcs) > 0:
        bc_nodes = numpy.unique(numpy.concatenate([b.nodes for b in bcs]))
    else:
        bc_nodes = numpy.empty(0, dtype=numpy.int32)

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
    patch.setPatchDMPlex(mesh._plex)
    patch.setPatchDefaultSF(V.dm.getDefaultSF())
    patch.setPatchCellNumbering(mesh._cell_numbering)
    patch.setPatchDiscretisationInfo(V.dm.getDefaultSection(),
                                     V.value_size, V.cell_node_list,
                                     bc_nodes)
    patch.setPatchComputeOperator(op)
    return patch
