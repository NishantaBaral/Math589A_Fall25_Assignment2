
import numpy as np

def plu_decomposition_in_place(A):

    #Getting the dimension of matrix A
    n = A.shape[0] 
    U = A.copy() #Starting with U as A
    P = np.identity(n) #Starting with P as identity matrix
    L = np.zeros((n, n)) #Starting with L as zero matrix

    for k in range (n-1):
        r = k + np.abs(U[k:, k]).argmax() #Finding the pivot row
        pivot = U[r][k] #Finding the pivot element
        
        P[[k,r],:] = P[[r,k],:] #Swapping the pivot rows of P with k
        U[[k,r],:] = U[[r,k],:] #Swapping the pivot rows of U with k
        L[[k,r],k] = L[[r,k],k] #Swapping the pivot rows of L only upto column k

        for i in range(k+1,n):
            L[i][k] = (U[i][k])/(U[k][k]) #Finding the multiplier and storing it in L
            multiplier = L[i][k]
            for j in range(k,n):
                U[i][j] = U[i][j] - (multiplier*U[k][j]) #Updating the U matrix, Gauss elimination step

    for n in range(n):
        L[n][n] = 1 #Setting the diagonal elements of L to 1
    

    # Note: P must be a vector, not array
    return P, A, L, U
