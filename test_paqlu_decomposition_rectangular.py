import numpy as np
import math

def paqlu_decomposition_in_place(A,tol=1e-6):
    m = A.shape[:][0] #Number of rows
    n = A.shape[:][1] #Number of columns
    U = A.copy() #Starting with U as A
    P = np.identity(m) #Starting with P as identity matrix of size m
    Q = np.identity(n) #Starting with Q as identity matrix of size n
    M = np.zeros((m,n)) #Starting with M as zero matrix of size m x n
    rank = 0 #Starting rank as 0

    for k in range (min(m,n)):
        a = U[k:,k:] #Extracting the relevant submatrix
        pivot = a.flat[np.abs(a).argmax()]  #Finding the pivot element
        r = np.argwhere(a == pivot) #Finding the position of pivot element
        if np.abs(pivot) <= tol:
            break #If pivot is zero, break the loop
        rank += 1 #Incrementing rank
        row_index = r[0][0] + k #Getting the actual row index in U
        column_index= r[0][1] + k #Getting the actual column index in U
        P[[k,row_index],:] = P[[row_index,k],:] #Swapping the pivot rows of P with k
        Q[:,[k,column_index]] = Q[:,[column_index,k]] #Swapping the pivot columns of Q with k

        U[[k,row_index],:] = U[[row_index,k],:] #Swapping the pivot rows of U with k
        U[:,[k,column_index]] = U[:,[column_index,k]] #Swapping the pivot columns of U with k
        M[[k, row_index], :k] = M[[row_index, k], :k]  #Swapping the pivot rows of M only upto column k
        
        for i in range(k+1,m):
            multiplier = (U[i][k])/(U[k][k]) #Finding the multiplier
            M[i][k] = multiplier
            for j in range(k,n):
                U[i][j] = U[i][j] - (multiplier*U[k][j]) #Updating the U matrix, Gauss elimination step

    L = np.zeros((m,rank)) #Initializing L as zero matrix of size m x rank
    L = np.tril(M[:, :rank], 0) #Extracting the lower triangular part of M upto rank
    U = np.triu(U[:rank, :], 0) #Extracting the upper triangular part of U upto rank
    for d in range(rank):
        L[d, d] = 1.0 #Setting the diagonal elements of L to 1

    return P, Q,L, U,rank
