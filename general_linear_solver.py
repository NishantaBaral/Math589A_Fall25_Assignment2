import numpy as np
import paqlu_decomposition_rectangular as paqlu_rectangular
import paqlu_decomposition_square as paqlu_square


def solve(A, b):
    m,n = A.shape
    if m == n:
        return square_solver(A, b)
    elif m != n:
        return rectangular_solver(A, b)
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

def rectangular_solver(A, b):
    m,n = A.shape
    P, Q, L, U, rank = paqlu_rectangular.paqlu_decomposition_in_place(A)
    L11 = L[:rank, :rank]
    U11 = U[:rank, :rank]
    U12 = U[:rank, rank:]
    Pb = np.dot(P, b)      # permute b according to P
    y = forward_substitution(L11, Pb[:rank])  # solve Ly = Pb
    x_basic = back_substitution(U11, y)  # solve Ux' = y
    x_perm = np.zeros(n,dtype=float)
    x_perm[:rank] = x_basic

    x_particular = Q @ x_perm  # unpermute x' according to Q

    # Nullspace basis (columns). If r < n:
    if rank < n:
        k = n - rank
        N_perm = np.zeros((n, k), dtype=float)
        # For each free variable e_i in the permuted coordinates:
        for i in range(k):
            # Solve U11 * w = -U12[:, i]
            rhs_ns = -U12[:, i]
            w = back_substitution(U11, rhs_ns)  # length r

            col = np.zeros(n, dtype=float)
            col[:rank] = w           # basic part
            col[rank + i] = 1        # free var = 1
            N_perm[:, i] = col

        nullspace = np.dot(Q, N_perm)  # unpermute nullspace basis
    else:
        nullspace = np.zeros((n, 0), dtype=float)

    return nullspace, x_particular
