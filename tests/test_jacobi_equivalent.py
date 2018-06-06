import pytest
import numpy
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


@pytest.fixture(params=["scalar", "vector", "tensor", "mixed"])
def problem_type(request):
    return request.param


def test_jacobi_equivalence(mesh, problem_type):
    if problem_type == "scalar":
        V = FunctionSpace(mesh, "CG", 1)
    elif problem_type == "vector":
        V = VectorFunctionSpace(mesh, "CG", 1)
    elif problem_type == "tensor":
        V = TensorFunctionSpace(mesh, "CG", 1)
    elif problem_type == "mixed":
        P = FunctionSpace(mesh, "CG", 1)
        Q = VectorFunctionSpace(mesh, "CG", 1)
        R = TensorFunctionSpace(mesh, "CG", 1)
        V = P*Q*R

    shape = V.ufl_element().value_shape()
    rhs = numpy.full(shape, 1, dtype=float)

    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(grad(u), grad(v))*dx

    L = inner(Constant(rhs), v)*dx

    if problem_type == "mixed":
        bcs = [DirichletBC(Q, zero(Q.ufl_element().value_shape()), "on_boundary")
               for Q in V.split()]
    else:
        bcs = DirichletBC(V, zero(V.ufl_element().value_shape()), "on_boundary")

    uh = Function(V)
    problem = LinearVariationalProblem(a, L, uh, bcs=bcs)

    jacobi = LinearVariationalSolver(problem,
                                     solver_parameters={"ksp_type": "richardson",
                                                        "pc_type": "jacobi",
                                                        "ksp_max_it": 5,
                                                        "ksp_convergence_test": "skip",
                                                        "ksp_monitor": True})

    jacobi.snes.ksp.setConvergenceHistory()

    jacobi.solve()

    jacobi_history = jacobi.snes.ksp.getConvergenceHistory()

    patch = LinearVariationalSolver(problem,
                                    solver_parameters={"mat_type": "matfree",
                                                       "ksp_type": "richardson",
                                                       "ksp_max_it": 5,
                                                       "ksp_convergence_test": "skip",
                                                       "pc_type": "python",
                                                       "pc_python_type": "ssc.PatchPC",
                                                       "patch_pc_patch_multiplicative": False,
                                                       "patch_pc_patch_save_operators": True,
                                                       "patch_pc_patch_sub_mat_type": "aij",
                                                       "patch_sub_ksp_type": "preonly",
                                                       "patch_sub_pc_type": "lu",
                                                       "ksp_monitor": True})

    patch.snes.ksp.setConvergenceHistory()

    uh.assign(0)
    patch.solve()

    patch_history = patch.snes.ksp.getConvergenceHistory()

    assert numpy.allclose(jacobi_history, patch_history)
