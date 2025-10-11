import numpy as np
from logging_setup import logger

def paqlu_decomposition_in_place(A,tol=1e-6):
    m = A.shape[0] #Number of rows
    logger.debug("Number of rows (m): %d", m)
    n = A.shape[1] #Number of columns
    logger.debug("Number of columns (n): %d", n)
    U = A.copy() #Starting with U as A
    logger.debug("Initial U matrix:\n%s", U)
    P = np.eye(m) #Starting with P as identity matrix of size m
    logger.debug("Initial P matrix (identity):\n%s", P)
    Q = np.eye(n) #Starting with Q as identity matrix of size n
    logger.debug("Initial Q matrix (identity):\n%s", Q)
    M = np.zeros((m,n)) #Starting with M as zero matrix of size m x n
    logger.debug("Initial M matrix (zeros):\n%s", M)
    rank = 0 #Starting rank as 0
    logger.debug("Initial rank: %d", rank)

    for k in range (min(m,n)):
        a = U[k:,k:] #Extracting the relevant submatrix
        logger.debug("Submatrix a at step %d:\n%s", k, a)
        i_rel, j_rel = np.unravel_index(np.abs(a).argmax(), a.shape) #Finding the index of maximum element in the submatrix
        logger.debug("Pivot element position in submatrix: (%d, %d)", i_rel, j_rel)
        row_index = k + i_rel
        logger.debug("Pivot row index in U: %d", row_index)
        column_index = k + j_rel
        logger.debug("Pivot column index in U: %d", column_index)   
        pivot = U[row_index, column_index]
        logger.debug("Pivot element value: %f", pivot)  
        if abs(pivot) <= tol:
            logger.debug("Pivot below tolerance (%f <= %f), stopping.", abs(pivot), tol)
            break #if the pivot is less than tolerance, we stop
        rank += 1 #Incrementing rank
        logger.debug("Updated rank: %d", rank)
        P[[k,row_index],:] = P[[row_index,k],:] #Swapping the pivot rows of P with k
        logger.debug("Updated P matrix after row swap:\n%s", P)
        Q[:,[k,column_index]] = Q[:,[column_index,k]] #Swapping the pivot columns of Q with k
        logger.debug("Updated Q matrix after column swap:\n%s", Q)

        U[[k,row_index],:] = U[[row_index,k],:] #Swapping the pivot rows of U with k
        U[:,[k,column_index]] = U[:,[column_index,k]] #Swapping the pivot columns of U with k
        logger.debug("Updated U matrix after row and column swaps:\n%s", U)
        M[[k, row_index], :k] = M[[row_index, k], :k]  #Swapping the pivot rows of M only upto column k
        logger.debug("Updated M matrix after row swap:\n%s", M)
        
        for i in range(k+1,m):
            logger.debug("Eliminating row %d", i)
            multiplier = (U[i][k])/(U[k][k]) #Finding the multiplier
            logger.debug("Multiplier for row %d: %f", i, multiplier)
            M[i][k] = multiplier
            logger.debug("Updated M matrix with multiplier:\n%s", M)
            U[i, k:] -= multiplier * U[k, k:]
            logger.debug("Updated U matrix after elimination of row %d:\n%s", i, U) 

    L = np.zeros((m,rank)) #Initializing L as zero matrix of size m x rank
    logger.debug("Initial L matrix (zeros):\n%s", L)
    L = np.tril(M[:, :rank], 0) #Extracting the lower triangular part of M upto rank
    logger.debug("Updated L matrix after extracting lower triangular part:\n%s", L)
    U = np.triu(U[:rank, :], 0) #Extracting the upper triangular part of U upto rank
    logger.debug("Updated U matrix after extracting upper triangular part:\n%s", U)
    for d in range(rank):
        L[d, d] = 1.0 #Setting the diagonal elements of L to 1
        logger.debug("Set L[%d, %d] to 1", d, d)
    logger.debug("Final L matrix:\n%s", L)
    logger.debug("Final U matrix:\n%s", U)  
    logger.debug("Final rank: %d", rank)
    logger.debug("Final P matrix:\n%s", P)
    logger.debug("Final Q matrix:\n%s", Q)
    return P, Q,L, U,rank

def solve(A, b):
   return rectangular_solver(A, b)
    
def forward_substitution(L, b):
# Solve Ly = b for y using forward substitution
    logger.debug("Starting forward substitution")
    m = L.shape[1]
    logger.debug("L shape: %s", L.shape)
    logger.debug("b shape: %s", b.shape)
    #m is number of columns of L
    y = np.zeros(m)
    logger.debug("Initial y (zeros): %s", y)
    for i in range(m):
        #We  are computing residuals for row and then dividing by diagonal entry
        y[i] = b[i] - np.dot(L[i, :i], y[:i])
        logger.debug("Computed y at index %d before division: %f", i, y[i])
        y[i] /= L[i, i]
        logger.debug("Updated y at index %d: %f", i, y[i])
    return y

def back_substitution(U, y):
    n = U.shape[0]
    logger.debug("U shape: %s", U.shape)
    logger.debug("y shape: %s", y.shape)
    # n is the number of rows of U
    x = np.zeros(n,dtype=float)
    logger.debug("Initial x (zeros): %s", x)
    for i in range(n-1, -1, -1):
        # We are solving Ux = y for x using back substitution.
        #Compute x[i] from already-known x[i+1:] because U is upper triangular.
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
        logger.debug("Updated x at index %d: %f", i, x[i])
    return x


def get_nullspace(U11, U12, Q, n, rank):
    # Compute the nullspace of the matrix using the PAQLU decomposition components
    if rank >= n:
        logger.debug("Full rank detected: %d >= %d", rank, n)
        #If rank is full, the dimension of nullspace is zero
        return np.zeros((n, 0), dtype=float) #full rank, empty nullspace
    
    if rank == 0:
        logger.debug("Zero rank detected: %d", rank)
        #If rank is zero, all variables are free, the nullspace is R^n
        return Q @ np.eye(n, dtype=float)

    k = n - rank
    logger.debug("Nullspace dimension k: %d", k)
    # Nullspace dimension k. Allocate matrix to hold k basis vectors (each length n).
    N_perm = np.zeros((n, k), dtype=float) #basis vectors
    logger.debug("Initial nullspace basis matrix N_perm (zeros):\n%s", N_perm)
    for j in range(k):
        logger.debug("Computing basis vector for free variable %d", j)
        #For each free variable j, solve U11 w = -U12[:, j] to get corresponding basic variables.
        rhs = -U12[:, j]
        logger.debug("RHS for free variable %d: %s", j, rhs)
        w = np.zeros(rank, dtype=float)
        logger.debug("Initial w (zeros): %s", w)
        for i in range(rank-1, -1, -1):
            w[i] = (rhs[i] - np.dot(U11[i, i+1:], w[i+1:])) / U11[i, i]
            logger.debug("Updated w at index %d: %f", i, w[i])  
        col = np.zeros(n, dtype=float) # basis vector in permuted space
        logger.debug("Initial column vector (zeros): %s", col)
        col[:rank] = w # basic variables
        logger.debug("Column vector after setting basic variables: %s", col)
        col[rank + j] = 1.0 # free variable
        logger.debug("Column vector after setting free variable %d: %s", j, col)
        N_perm[:, j] = col # basis vector j in permuted space
        logger.debug("Updated nullspace basis matrix N_perm:\n%s", N_perm)
        Q_arr = np.asarray(Q) # convert Q to array if it's not already
        logger.debug("Q array:\n%s", Q_arr)
        logger.debug("Q array ndim: %d", Q_arr.ndim)
        logger.debug("N_perm shape: %s", N_perm[Q_arr, :].shape if Q_arr.ndim == 1 else (Q_arr @ N_perm))
    return N_perm[Q_arr, :] if Q_arr.ndim == 1 else Q_arr @ N_perm # apply column permutation to get nullspace in original variable order


def get_x_particular(P,Q,L,L11,U11,b,rank,n):
    b = np.asarray(b, dtype=float).reshape(-1)   
    logger.debug("b reshaped to 1-D array: %s", b)
    P_arr = np.asarray(P)
    Pb = b[P_arr] if P_arr.ndim == 1 else P_arr @ b
    logger.debug("Pb after applying permutation P: %s", Pb)
    # We want to make sure that b is 1-D array and apply row permutation P to b to get Pb.
    #if P is a vector, index b[P]; if matrix, multiply Pb
    if rank == 0:
        logger.debug("Zero rank case in get_x_particular")
        #If A has zero rank: system is consistent only if Pb ≈ 0.
        if np.linalg.norm(Pb, np.inf) > 1e-6:
        # Pb is not approximately zero, so no solution exists.
            return None    
        return np.zeros(n, dtype=float)
    y = forward_substitution(L11, Pb[:rank])  # solve Ly = Pb
    logger.debug("Solution y from forward substitution: %s", y)
     # there are extra rows (overdetermined or rank-deficient), 
     # check that the lower block satisfies the equations: 
     # L21 y must equal the corresponding part of Pb. If not, no solution.
    if L is not None and L.shape[0] > rank:
        L21 = L[rank:, :rank]
        logger.debug("L21 matrix:\n%s", L21)
        logger.debug("L21 @ y: %s", L21 @ y)
        logger.debug("Pb[rank:]: %s", Pb[rank:])
        if L21.size and not np.allclose(L21 @ y, Pb[rank:], atol=1e-6, rtol=0):
            logger.debug("Inconsistency detected in lower block equations.")
            return None  # inconsistent: no particular solution
     #Solve the upper-triangular pivot system U11 x_basic = y.
    x_basic = back_substitution(U11, y) 
    logger.debug("Solution x_basic from back substitution: %s", x_basic)
    #Build the solution in the permuted coordinates:
    #  basic variables = x_basic, free variables = 0 (particular solution).
    x_perm = np.zeros(n,dtype=float)
    logger.debug("Initial x_perm (zeros): %s", x_perm)
    x_perm[:rank] = x_basic
    logger.debug("x_perm after setting basic variables: %s", x_perm)
    x_perm[rank:] = 0
    logger.debug("x_perm after setting free variables to zero: %s", x_perm)
    # Undo column permutation to original coordinates: scatter if Q is a vector; multiply if matrix.
    Q_arr = np.asarray(Q)
    logger.debug("Q array:\n%s", Q_arr)
    logger.debug("Q array ndim: %d", Q_arr.ndim)
    logger.debug("x_perm before applying Q: %s", x_perm)
    if Q_arr.ndim == 1:
        x_particular = np.zeros_like(x_perm)
        x_particular[Q_arr] = x_perm
    else:
        x_particular = Q_arr @ x_perm

    if x_particular is not None and x_particular.ndim == 1:
        x_particular = x_particular.reshape(-1, 1)
    logger.debug("Final particular solution x_particular: %s", x_particular)    
    return x_particular

def rectangular_solver(A, b,tol=1e-6):
    m,n = A.shape
    P, Q, L, U, rank = paqlu_decomposition_in_place(A)

    #Extract relevant submatrices from L and U
    L11 = L[:rank, :rank]
    logger.debug("L11 matrix:\n%s", L11)
    U11 = U[:rank, :rank]
    logger.debug("U11 matrix:\n%s", U11)
    U12 = U[:rank, rank:]
    logger.debug("U12 matrix:\n%s", U12)

    x_particular = get_x_particular(P,Q,L,L11,U11,b,rank,n)
    logger.debug("Particular solution x_particular: %s", x_particular)
    nullspace = get_nullspace (U11,U12,Q,n,rank)
    logger.debug("Nullspace basis:\n%s", nullspace)

    return x_particular,nullspace

def unitest():
    A1 = np.array([
    [1., 0., 2., 0.],
    [0., 0., 1., 1.],
    [2., 0., 0., 1.],
    [0., 0., 0., 0.]], dtype=float)

    b1 = np.array([3., 2., 4., 0.], dtype=float)
    logger.info("Testing rectangular_solver with example matrix A1 and vector b1")
    logger.info("Matrix A1:\n%s", A1)
    logger.info("Vector b1: %s", b1)
    x_particular, nullspace = rectangular_solver(A1, b1)
    logger.info("Particular solution x: %s", x_particular)
    logger.info("Nullspace basis N:\n%s", nullspace)

if __name__ == "__main__":
    unitest()
