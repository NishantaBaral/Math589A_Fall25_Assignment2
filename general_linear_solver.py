import numpy as np
import paqlu_decomposition_rectangular as paqlu_rectangular
import paqlu_decomposition_square as paqlu_square


def solve(A, b):
    m,n = A.shape
    if m == n:
        return square_solver(A, b)
    elif m != n:
        return False
    else:
        raise ValueError("Matrix A must be either square or rectangular.")
    


def forward_substitution(L, b):
    m = L.shape[0]
    y = np.zeros(m)
    for i in range(m):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])
        y[i] /= L[i, i]
    return y

def back_substitution(U, y):
    n = U.shape[0]
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
    return x

def square_solver(A, b):
    m,n = A.shape
    P, Q, L, U = paqlu_square.paqlu_decomposition_in_place(A)
    b_perm = np.dot(P, b)        # permute b according to P
    y = forward_substitution(L, b_perm)  # solve Ly = Pb
    x_perm = back_substitution(U, y)  # solve Ux' = y
    x = np.dot(Q, x_perm)         # unpermute x' according to Q
    nullspace = np.zeros((n, 0), dtype=float)
    return nullspace, x
