import pytest
from firedrake import *


@pytest.fixture(params=[1, 2, 3],
                ids=["Interval", "Rectangle", "Box"])
def mesh(request):
    distribution = {"overlap_type": (DistributedMeshOverlapType.VERTEX, 1)}
    if request.param == 1:
        return IntervalMesh(10, 5, distribution_parameters=distribution)
    if request.param == 2:
        return RectangleMesh(10, 20, 2, 3, distribution_parameters=distribution)
    if request.param == 3:
        return BoxMesh(5, 3, 5, 1, 2, 3, distribution_parameters=distribution)


@pytest.fixture
def expected(mesh):
    if mesh.geometric_dimension() == 1:
        return [5, 5, 5, 5]
    elif mesh.geometric_dimension() == 2:
        return [10, 12, 12, 12]
    elif mesh.geometric_dimension() == 3:
        return [4, 21, 22, 22]


def test_p_independence(mesh, expected):
    nits = []
    for p in range(1, 5):
        V = FunctionSpace(mesh, "CG", p)

        u = TrialFunction(V)
        v = TestFunction(V)

        a = inner(grad(u), grad(v))*dx

        L = inner(Constant(1), v)*dx

        bcs = DirichletBC(V, zero(V.ufl_element().value_shape()), "on_boundary")

        uh = Function(V)
        problem = LinearVariationalProblem(a, L, uh, bcs=bcs)

        solver = LinearVariationalSolver(problem, solver_parameters={
            "mat_type": "matfree",
            "ksp_type": "cg",
            "pc_type": "python",
            "pc_python_type": "ssc.SSC",
            "ssc_sub_0": {
                "pc_patch_sub_mat_type": "aij",
                "pc_patch_save_operators": True,
                "sub_ksp_type": "preonly",
                "sub_pc_type": "lu",
            },
            "ssc_sub_1": {
                "lo_mat_type": "aij",
                "lo_ksp_type": "preonly",
                "lo_pc_type": "lu",
            },
            "ksp_monitor": True})

        solver.solve()

        nits.append(solver.snes.ksp.getIterationNumber())
    assert (nits == expected)
