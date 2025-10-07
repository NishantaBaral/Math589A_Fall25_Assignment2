import numpy as np
import math

def paqlu_decomposition_in_place(A,tol=1e-12):
    m = A.shape[:][0] #Number of rows
    n = A.shape[:][1] #Number of columns
    U = A.copy() #Starting with U as A
    P = np.eye(m) #Starting with P as identity matrix of size m
    Q = np.eye(n) #Starting with Q as identity matrix of size n
    M = np.zeros((m,n)) #Starting with M as zero matrix of size m x n
    rank = 0 #Starting rank as 0

    for k in range (min(m,n)):
        a = U[k:,k:] #Extracting the relevant submatrix
        i_rel, j_rel = np.unravel_index(np.abs(a).argmax(), a.shape) #Finding the index of maximum element in the submatrix
        row_index = k + i_rel
        column_index = k + j_rel
        pivot = U[row_index, column_index]
        if abs(pivot) <= tol:
            break #if the pivot is less than tolerance, we stop
        rank += 1 #Incrementing rank
        P[[k,row_index],:] = P[[row_index,k],:] #Swapping the pivot rows of P with k
        Q[:,[k,column_index]] = Q[:,[column_index,k]] #Swapping the pivot columns of Q with k

        U[[k,row_index],:] = U[[row_index,k],:] #Swapping the pivot rows of U with k
        U[:,[k,column_index]] = U[:,[column_index,k]] #Swapping the pivot columns of U with k
        M[[k, row_index], :k] = M[[row_index, k], :k]  #Swapping the pivot rows of M only upto column k
        
        for i in range(k+1,m):
            multiplier = (U[i][k])/(U[k][k]) #Finding the multiplier
            M[i][k] = multiplier
            U[i, k:] -= multiplier * U[k, k:]

    L = np.zeros((m,rank)) #Initializing L as zero matrix of size m x rank
    L = np.tril(M[:, :rank], 0) #Extracting the lower triangular part of M upto rank
    U = np.triu(U[:rank, :], 0) #Extracting the upper triangular part of U upto rank
    for d in range(rank):
        L[d, d] = 1.0 #Setting the diagonal elements of L to 1

    return P, Q,L, U,rank

