from utils import matrix_dot_matrix, matrix_transpose
from conjugate_gradient import ConjugateGradientFiniteDifferencePotentialSolver
from choleski import CholeskiDecomposition

if __name__ == "__main__":
    print("\n", end="\n")
    print("# --------------- TEST --------------- #", end="\n")
    print("# -------- Conjugate Gradient -------- #", end="\n")
    print("# ------------------------------------ #", end="\n\n")
    cgfdps = ConjugateGradientFiniteDifferencePotentialSolver(h=0.02)
    potential_history, residual_history, search_history = cgfdps.solve()
    print("A:\n", cgfdps._A, end="\n\n")
    print("b:\n", cgfdps._b, end="\n\n")
    print("result = solve(A, b):\n", potential_history[-1], end="\n\n")
    print("# ------------------------------------ #", end="\n\n")


    print("\n", end="\n")
    print("# --------------- TEST --------------- #", end="\n")
    print("# ------ Choleski Decomposition ------ #", end="\n")
    print("# ------------------------------------ #", end="\n\n")
    chol_d = CholeskiDecomposition()
    # Create a symmetric, real, positive definite matrix.
    A = cgfdps._A
    b = cgfdps._b

    A = matrix_dot_matrix(matrix_transpose(A), A)
    b = matrix_dot_matrix(matrix_transpose(A), b)

    print("A:\n", A, end="\n\n")
    print("b:\n", b, end="\n\n")
    v = chol_d.solve(A=A, b=b)
    print("result = solve(A, b):\n", v, end="\n\n")
    print("# ------------------------------------ #", end="\n\n")
