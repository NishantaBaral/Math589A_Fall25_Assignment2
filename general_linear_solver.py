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

def rectangular_solve(A,B,tol=1e-12):
    P,Q,L,U = paqlu_rectangular.paqlu_decomposition_in_place(A)
    y = forward_substitution(L,np.dot(P,B))
    rank = U.shape[0]
    x_basic = back_substitution(U,y[:rank])
    n = A.shape[1]
    x = np.zeros(n)
    x[:rank] = x_basic
    N = np.zeros((n - rank, n))
    for i in range(n - rank):
        N[i, rank + i] = 1
    N = np.dot(Q,N.T).T
    return np.dot(Q,x),N

def solve(A,B):
    m,n = A.shape
    if m == n:
        P,Q,L,U = paqlu_square.paqlu_decomposition_in_place(A)
        Pb = np.dot(P,B)
        y = forward_substitution(L,Pb)
        x_perm = back_substitution(U,y)
        x = np.dot(Q, x_perm)
        N = np.zeros((n, 0))  # Empty null space with correct shape
        return x, N
    else:
        x, N = rectangular_solve(A,B)
        return x, N