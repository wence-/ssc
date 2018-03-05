import firedrake
from firedrake.petsc import PETSc

from ssc.patch import setup_patch_pc
from ssc.lo import P1PC


class SSC(firedrake.PCBase):

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

        combiner = PETSc.PC().create(comm=pc.comm)
        combiner.setOptionsPrefix(pc.getOptionsPrefix() + "ssc_")
        combiner.setOperators(A, P)
        combiner.setType(PETSc.PC.Type.COMPOSITE)
        combiner.addCompositePC("patch")
        combiner.addCompositePC("python")
        patch = combiner.getCompositePC(0)
        patch = setup_patch_pc(patch, J, bcs)
        lo = combiner.getCompositePC(1)
        lo.setPythonContext(P1PC(J, bcs))
        self.combiner = combiner
        self.combiner.setFromOptions()

    def update(self, pc):
        self.combiner.setUp()

    def apply(self, pc, x, y):
        self.combiner.apply(x, y)

    def applyTranspose(self, pc, x, y):
        return self.combiner.applyTranspose(x, y)

    def view(self, pc, viewer=None):
        if viewer is None:
            viewer = PETSc.Viewer.STDOUT
        viewer.printfASCII("Subspace correction PC\n")
        viewer.printfASCII("Combining (composite) PC:\n")
        viewer.pushASCIITab()
        self.combiner.view(viewer)
        viewer.popASCIITab()
