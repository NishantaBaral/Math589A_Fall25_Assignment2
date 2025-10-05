import numpy as np
import paqlu_decomposition_square as paqlu_square
import paqlu_decomposition_rectangular as paqlu_rectangular

def forward_substitution(L, Pb):
    """ Solve the equation L*y = Pb for y using forward substitution.
        L is a lower triangular matrix.
        Pb is the permuted right-hand side vector.
    """
    m = L.shape[0]
    y = np.zeros(m)
    for i in range(m):
        y[i] = Pb[i] - L[i, :i] @ y[:i]
    return y
def back_substitution(U, y):
    """ Solve the equation U*x = y for x using back substitution.
        U is an upper triangular matrix.
        y is the right-hand side vector.
    """
    n = U.shape[0]
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - U[i, i+1:] @ x[i+1:]) / U[i, i]
    return x

def solve(A, b, tol=1e-6):
    m, n = A.shape
    if m == n:
        P, Q, L, U = paqlu_square.paqlu_decomposition_in_place(A)
        Pb = P @ b
        y = forward_substitution(L, Pb)
        x_perm = back_substitution(U, y)
        x = Q.T @ x_perm
        N = np.zeros(x.shape)
        return x, N
    
    # rectangular
    return False
