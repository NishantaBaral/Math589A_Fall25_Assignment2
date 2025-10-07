import numpy as np

def paqlu_decomposition_simple(A, tol=1e-6):

    m, n = A.shape
    U = A.copy().astype(float)
    P = np.eye(m)
    Q = np.eye(n)
    L = np.eye(m, min(m, n))
    
    rank = 0
    
    for k in range(min(m, n)):
        # Find pivot
        pivot_val = 0
        pivot_i, pivot_j = k, k
        
        for i in range(k, m):
            for j in range(k, n):
                if abs(U[i, j]) > abs(pivot_val):
                    pivot_val = U[i, j]
                    pivot_i, pivot_j = i, j
        
        if abs(pivot_val) < tol:
            break
            
        rank += 1
        
        # Swap rows
        if pivot_i != k:
            U[[k, pivot_i]] = U[[pivot_i, k]]
            P[[k, pivot_i]] = P[[pivot_i, k]]
            if k > 0:
                L[[k, pivot_i], :k] = L[[pivot_i, k], :k]
        
        # Swap columns
        if pivot_j != k:
            U[:, [k, pivot_j]] = U[:, [pivot_j, k]]
            Q[:, [k, pivot_j]] = Q[:, [pivot_j, k]]
        
        # Elimination
        for i in range(k + 1, m):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]
    
    return P, Q, L[:, :rank], U[:rank, :], rank