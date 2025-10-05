import numpy as np
import paqlu_decomposition_square as paqlu_square
import paqlu_decomposition_rectangular as paqlu_rectangular

def forward_substitution(L, b):
    n = L.shape[0]
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])
    return y

def back_substitution(U, y):
    n = U.shape[0]
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
    return x
def square_solver(A, b):
    P, Q, L, U = paqlu_square.paqlu_decomposition_in_place(A)
    Pb = np.dot(P, b)
    y = forward_substitution(L, Pb)
    x_temp = back_substitution(U, y)
    x = np.dot(Q, x_temp)
    N  = np.zeros((A.shape[1], 0))   # no free variables
    return x, N


def rectangular_solver(A,b,tol=1e-12):
    P,Q,L,U = paqlu_rectangular.paqlu_decomposition_in_place(A)
    #Forward solve Ly = Pb
    y = np.linalg.solve(L, np.dot(P,b))

    #Rank from nonzero rows of U
    nonzero_rows = np.where(np.any(np.abs(U) > tol, axis=1))[0]
    r = 0 if len(nonzero_rows)==0 else (nonzero_rows[-1] + 1)

    # check
    if np.linalg.norm(y[r:], ord=np.inf) > 10*tol:
        return None, None, False, r

    # 4) Blocks
    n = U.shape[1]
    U_top = U[:r,:]
    U11   = U_top[:,:r]
    U12   = U_top[:,r:]
    y_piv = y[:r]

    # Particular and null in z-coordinates
    zB_part = back_substitution(U11, y_piv)
    if n > r:
        NB = -back_substitution(U11, U12)         # r x (n-r)
        N  = Q @ np.vstack([NB, np.eye(n-r)])
    else:
        N  = np.zeros((n,0))

    z_part = np.concatenate([zB_part, np.zeros(n-r)])
    x_part = Q @ z_part

    return x_part, N

def solve(A, b):
    m, n = A.shape
    if m == n:
        return square_solver(A, b)
    elif m > n:
        return rectangular_solver(A, b)
    elif m < n:
        return rectangular_solver(A, b)
    else:
        raise NotImplementedError("solve is a stub; implement parametric solver here.")
    