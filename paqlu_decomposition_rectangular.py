import numpy as np
import math

def paqlu_decomposition_in_place(A, tol=1e-6):
    m = A.shape[0]  # Number of rows
    n = A.shape[1]  # Number of columns
    U = A.copy().astype(float)  # Working copy
    P = np.eye(m)  # Row permutation
    Q = np.eye(n)  # Column permutation
    L = np.eye(m)  # Start with identity
    rank = 0
    
    for k in range(min(m, n)):
        # Find pivot in submatrix U[k:, k:]
        submatrix = U[k:, k:]
        pivot_idx = np.unravel_index(np.argmax(np.abs(submatrix)), submatrix.shape)
        pivot_val = submatrix[pivot_idx]
        
        if np.abs(pivot_val) <= tol:
            break
            
        rank += 1
        
        # Convert relative indices to absolute indices
        pivot_row = k + pivot_idx[0]
        pivot_col = k + pivot_idx[1]
        
        # Swap rows in P and U
        if pivot_row != k:
            P[[k, pivot_row]] = P[[pivot_row, k]]
            U[[k, pivot_row]] = U[[pivot_row, k]]
            # Swap rows in L for columns 0 to k-1
            L[[k, pivot_row], :k] = L[[pivot_row, k], :k]
        
        # Swap columns in Q and U  
        if pivot_col != k:
            Q[:, [k, pivot_col]] = Q[:, [pivot_col, k]]
            U[:, [k, pivot_col]] = U[:, [k, pivot_col]]
        
        # Elimination
        for i in range(k + 1, m):
            if np.abs(U[k, k]) > tol:  # Avoid division by zero
                multiplier = U[i, k] / U[k, k]
                L[i, k] = multiplier
                U[i, k:] = U[i, k:] - multiplier * U[k, k:]
    
    # Extract the proper L and U matrices for rectangular case
    L_final = L[:m, :rank]  # m × rank
    U_final = U[:rank, :n]  # rank × n
    
    return P, Q, L_final, U_final, rank