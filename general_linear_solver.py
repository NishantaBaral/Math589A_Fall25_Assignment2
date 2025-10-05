import numpy as np
import paqlu_decomposition_square as paqlu_square
import paqlu_decomposition_rectangular as paqlu_rectangular

def forward_substitution(L,b):
    n = L.shape[0]
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - np.dot(L[i,:i],y[:i])
    return y

def back_substitution(U,y):
    n = U.shape[0]
    x = np.zeros(n)
    for i in range(n-1,-1,-1):
        x[i] = (y[i] - np.dot(U[i,i+1:],x[i+1:]))/U[i,i]
    return x

def rectangular_solve(A, B, tol=1e-12):
    P, Q, L, U = paqlu_rectangular.paqlu_decomposition_in_place(A)
    m, n = A.shape
    rank = U.shape[0]
    Pb = np.dot(P, B)
    y = forward_substitution(L, Pb[:rank])
    x_basic = back_substitution(U, y)
    x = np.zeros(n)
    # Q is a permutation matrix, so Q.T @ [x_basic, 0] puts x_basic in the correct places
    x_full = np.zeros(n)
    x_full[:rank] = x_basic
    x = np.dot(Q.T, x_full)
    # Null space basis
    N = np.zeros((n, n - rank))
    if n > rank:
        N[rank:, :] = np.eye(n - rank)
        N = np.dot(Q.T, N)
    else:
        N = np.zeros((n, 0))
    return x, N

def solve(A,B):
    m,n = A.shape
    if m == n:
        P,Q,L,U = paqlu_square.paqlu_decomposition_in_place(A)
        Pb = np.dot(P,B)
        y = forward_substitution(L,Pb)
        x_perm = back_substitution(U,y)
        x = np.dot(Q.T, x_perm)
        N = np.zeros((n, 0))  # Empty null space with correct shape
        return x, N
    else:
        x, N = rectangular_solve(A,B)
        return x, N