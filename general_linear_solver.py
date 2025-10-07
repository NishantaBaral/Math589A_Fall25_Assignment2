import numpy as np
import paqlu_decomposition_rectangular as paqlu_rectangular
import paqlu_decomposition_square as paqlu_square
from logging_setup import logger



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
    x = np.zeros(n,dtype=float)
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

def get_nullspace(U11, U12, Q, n, rank):
    if rank >= n:
        return np.zeros((n, 0), dtype=float) #full rank, empty nullspace
    
    if rank == 0:
        return Q @ np.eye(n, dtype=float) #no rank , all free variables

    k = n - rank
    N_perm = np.zeros((n, k), dtype=float) #basis vectots
    for j in range(k):
        # solve U11 w = -U12[:, j]
        rhs = -U12[:, j]
        w = np.zeros(rank, dtype=float)
        for i in range(rank-1, -1, -1):
            w[i] = (rhs[i] - np.dot(U11[i, i+1:], w[i+1:])) / U11[i, i]
        col = np.zeros(n, dtype=float)
        col[:rank] = w
        col[rank + j] = 1.0
        N_perm[:, j] = col
    return Q @ N_perm

def get_x_particualar(P,Q,L11,U11,b,rank,n):
    Pb = np.dot(P,b)
    if rank == 0:
        return np.zeros(n, dtype=float)
    y = forward_substitution(L11, Pb[:rank])  # solve Ly = Pb
    x_basic = back_substitution(U11, y)  # solve Ux' = y
    x_perm = np.zeros(n,dtype=float)
    x_perm[:rank] = x_basic
    x_perm[rank:] = 0
    x_particular = Q @ x_perm  # unpermute x' according to Q

    return x_particular

def rectangular_solver(A, b,tol=1e-6):
    m,n = A.shape
    P, Q, L, U, rank = paqlu_rectangular.paqlu_decomposition_in_place(A)

    L11 = L[:rank, :rank]
    U11 = U[:rank, :rank]
    U12 = U[:rank, rank:]

    x_particular = get_x_particualar(P,Q,L11,U11,b,rank,n)
    nullspace = get_nullspace (U11,U12,Q,n,rank)

    return x_particular,nullspace
    

def test():
    A = np.array([[1,2],[4,5],[6,7],[8,9]],dtype=float)
    B = np.array([7,8,9,10],dtype=float)
    x,N = solve(A,B)
    print("x is", x)
    print("nullspace is", N)

if __name__ == "__main__":
    test()