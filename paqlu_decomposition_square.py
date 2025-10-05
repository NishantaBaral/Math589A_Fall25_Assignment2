import numpy as np

def paqlu_decomposition_in_place(A):
    
    #working with square matrix first before translating to rectangular matrix
    n = A.shape[:][0] #Getting the number of columns
    U = A.copy() #Starting with U as A
    P = np.identity(n) #Starting with P as identity matrix
    Q = np.identity(n) #Starting with Q as identity matrix
    L =  np.zeros((n,n)) #Starting with L as zero matrix

    for k in range (n-1):
        a = U[k:,k:] #Extracting the relevant submatrix
        pivot = a.flat[np.abs(a).argmax()]  #Finding the pivot element
        r = np.argwhere(a == pivot) #Finding the position of pivot element
        row_index = r[0][0] + k #Getting the actual row index in U
        column_index= r[0][1] + k #Getting the actual column index in U
        P[[k,row_index],:] = P[[row_index,k],:] #Swapping the pivot rows of P with k
        Q[:,[k,column_index]] = Q[:,[column_index,k]] #Swapping the pivot columns of Q with k

        U[[k,row_index],:] = U[[row_index,k],:] #Swapping the pivot rows of U with k
        U[:,[k,column_index]] = U[:,[column_index,k]] #Swapping the pivot columns of U with k

        L[[k, row_index], :k] = L[[row_index, k], :k] #Swapping the pivot rows of L only upto column k
        

        for i in range(k+1,n):
            L[i][k] = (U[i][k])/(U[k][k]) #Finding the multiplier and storing it in L
            multiplier = L[i][k] #defining the multiplier variable for ease of use
            for j in range(k,n):
                U[i][j] = U[i][j] - (multiplier*U[k][j]) #Updating the U matrix, Gauss elimination step
      

    for n in range(n):
        L[n][n] = 1 #Setting the diagonal elements of L to 1
        
    return P, Q, L, U
