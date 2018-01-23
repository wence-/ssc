from firedrake import *
from firedrake.mg.utils import *
from tsfc.fiatinterface import create_element
import firedrake.utils as utils
from numpy import where, unique

def fix_coarse_boundaries(cV, fV):
    # With lots of help from Lawrence Mitchell
    c2fmap = coarse_to_fine_node_map(cV, fV)

    cmesh = cV.mesh()
    ch, level = get_level(cmesh)
    refinements_per_level = ch.refinements_per_level
    element = create_element(cV.ufl_element(), vector_is_mixed=False)
    omap = fV.cell_node_map().values
    c2f, vperm = ch._cells_vperm[int(level*refinements_per_level)]
    indices, _ = get_unique_indices(element,
                                    omap[c2f[0, :], ...].reshape(-1),
                                    vperm[0, :],
                                    offset=None)
    weights = restriction_weights(element)[indices]
    sums = (weights > 0).sum(axis=1)
    indices_to_fix = where(sums <= cmesh.topological_dimension())[0]

    nodelist = unique(c2fmap.values[:, indices_to_fix])

    class FixedDirichletBC(DirichletBC):
        def __init__(self, V, g, nodelist):
            self.nodelist = nodelist
            DirichletBC.__init__(self, V, g, "on_boundary")

        @utils.cached_property
        def nodes(self):
            return self.nodelist

    bc = FixedDirichletBC(fV, (0, 0), nodelist)

    return bc



class DivProlong(PCBase):

    def __init__(self, J, bcs):
        self.J = J
        self.bcs = bcs
        super(DivProlong, self).__init__()

    def initialize(self, pc):
        _, P = pc.getOperators()
        self.context = P.getPythonContext()
        appctx = self.context.appctx

        cmesh = appctx["cmesh"]
        fmesh = appctx["fmesh"]
        vmesh = appctx["vmesh"]
        self.fV = appctx["fV"]
        self.cV = FunctionSpace(cmesh, self.fV.ufl_element())
        self.fix = fix_coarse_boundaries(self.cV, self.fV)
        self.Re = appctx["Re"]
        self.gamma = appctx["gamma"]
        self.exactparams = appctx["exactparams"]
        # self.f = appctx["f"]
        
        pass

    def update(self, pc):
        pass

    def apply(self, pc, x, y):
        # x.copy(y)
        # return

        cuf = Function(self.fV)
        with cuf.dat.vec as x_:
            x.copy(x_)
        tildeu = Function(self.fV)

        from firedrake import inner, grad, dx, div
        Re = self.Re
        gamma = self.gamma
        u = TrialFunction(self.fV)
        def ah(u, v):
            # FIXME: should cell_avg be there?
            return 1.0/Re * inner(grad(u), grad(v))*dx + gamma * inner(div(u), div(v))*dx

        v = TestFunction(self.fV)
        A = assemble(ah(u, v), bcs=self.fix)
        solve(A, tildeu, cuf, bcs=self.fix)
        # print(assemble(inner(tildeu, tildeu)*dx))
        with tildeu.dat.vec as x_:
            x_.copy(y)
        return

        solve(ah(tildeu, v) - ah(cuf, v) == 0, tildeu, self.fix,
              solver_parameters=self.exactparams)
        # print(assemble(inner(tildeu, tildeu)*dx))
        cuf.assign(cuf+tildeu)
        with cuf.dat.vec as x_:
            x_.copy(y)


    def applyTranspose(self, pc, x, y):
        raise NotImplementedError("Haven't coded DivProlong-applyTranspose")

    def view(self, pc, viewer=None):
        pass
        # if viewer is None:
        #     viewer = PETSc.Viewer.STDOUT
        # viewer.printfASCII("Low-order PC\n")
        # viewer.pushASCIITab()
        # self.lo.view(viewer)
        # viewer.popASCIITab()
