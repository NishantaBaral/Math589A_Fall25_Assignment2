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
    x = np.dot(Q.T, x_perm)         # unpermute x' according to Q
    nullspace = np.zeros(n)
    return x, nullspace

def rectangular_solver(A, b):
    m,n = A.shape
    P, Q, L, U, rank = paqlu_rectangular.paqlu_decomposition_in_place(A)
    b_ = np.dot(P, b)        # permute b according to P
    y = forward_substitution(L, b_)  # solve Ly = Pb
    #Take the first 'rank' components of y
    y_pivot = y[:rank]
    #partiition the U matrix to pivot and non-pivot columns
    U_basic = U[:rank, :rank]
    U_free = U[:rank, rank:]
    #Compute particular solution
    x_basic = back_substitution(U_basic, y_pivot)
    #Construct the full solution vector
    x = np.zeros(n)
    x[:rank] = x_basic
    x = np.dot(Q.T, x)         # unpermute x' according to Q
    #Compute the nullspace
    if rank < n:
        nullspace = np.zeros((n, n - rank))
        for i in range(n - rank):
            nullspace[rank + i, i] = 1
            for j in range(rank - 1, -1, -1):
                nullspace[j, i] = -np.dot(U_basic[j, :], nullspace[:rank, i]) / U_basic[j, j]
        nullspace = np.dot(Q.T, nullspace)  # unpermute nullspace according to Q
    else:
        nullspace = np.zeros((n, 0))  # No nullspace if rank == n
    return x, nullspace
