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


def rectangular_solver(A, b,tol=1e-6):
    m,n = A.shape
    logger.info(f"Solving system with matrix A of shape {A.shape} and vector b of shape {b.shape}")
    P, Q, L, U, rank = paqlu_rectangular.paqlu_decomposition_in_place(A)
    logger.info(f"P is: {P}")
    logger.info(f"A is: {A}")
    logger.info(f"Q is: {Q}")
    logger.info(f"L is: {L}")
    logger.info(f"U is: {U}")
    logger.info(f"Decomposition complete. Rank: {rank}")
    #handle edge cases
    if rank == 0:
        x_particular = np.zeros(n, dtype=float)
        nullspace = Q @ np.eye(n)   # all free
        return x_particular, nullspace
    
    L11 = L[:rank, :rank]
    logger.info(f"L11 is: {L11}")
    U11 = U[:rank, :rank]
    logger.info(f"U11 is: {U11}")
    U12 = U[:rank, rank:]
    logger.info(f"U12 is: {U12}")   
    Pb = np.dot(P, b)      # permute b according to P
    logger.info(f"Permuted b (Pb) is: {Pb}")
    y = forward_substitution(L11, Pb[:rank])  # solve Ly = Pb
    logger.info(f"Intermediate solution y after forward substitution is: {y}")
    x_basic = back_substitution(U11, y)  # solve Ux' = y
    logger.info(f"Basic solution x_basic after back substitution is: {x_basic}")
    x_perm = np.zeros(n,dtype=float)
    logger.info(f"Initialized x_perm with zeros: {x_perm}")
    x_perm[:rank] = x_basic
    logger.info(f"x_perm after assigning x_basic: {x_perm}")
    x_perm[rank:] = 0

    x_particular = x_perm  # unpermute x' according to Q
    logger.info(f"Particular solution x_particular after unpermuting: {x_particular}")
    # Nullspace basis (columns). If r < n:
    if rank < n:
        k = n - rank
        logger.info(f"Nullspace dimension (k) is: {k}")
        N_perm = np.zeros((n, k), dtype=float)
        logger.info(f"Initialized N_perm with zeros: {N_perm}")
        # For each free variable e_i in the permuted coordinates:
        for i in range(k):
            # Solve U11 * w = -U12[:, i]
            rhs_ns = -U12[:, i]
            logger.info(f"RHS for nullspace computation (rhs_ns) for free variable {i} is: {rhs_ns}")
            w = back_substitution(U11, rhs_ns)  # length r
            logger.info(f"Solution w for free variable {i} is: {w}")

            col = np.zeros(n, dtype=float)
            col[:rank] = w           # basic part
            logger.info(f"Column for nullspace basis before setting free variable {i} is: {col}")
            col[rank + i] = 1        # free var = 1
            logger.info(f"Column for nullspace basis after setting free variable {i} is: {col}")
            N_perm[:, i] = col
            logger.info(f"N_perm after setting column for free variable {i} is: {N_perm}")

        nullspace = np.dot(Q, N_perm)  # unpermute nullspace basis
        logger.info(f"Nullspace basis after unpermuting (nullspace) is: {nullspace}")   
    else:
        nullspace = np.zeros((n, 0), dtype=float)
    return x_particular, nullspace

def test():
    A = np.array([[1,2],[4,5],[6,7],[8,9]],dtype=float)
    B = np.array([7,8,9,10],dtype=float)
    x,N = solve(A,B)
    logger.info(f"Solution x is: {x}")
    logger.info(f"Nullspace N is: {N}")

if __name__ == "__main__":
    test()