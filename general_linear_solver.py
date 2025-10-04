import numpy as np
import paqlu_decomposition as paqlu_square
import paqlu_decomposition_rectangular as paqlu_rectangular

def square_solver(A, b):
    P, Q, L, U = paqlu_square.paqlu_decomposition_in_place(A)
    Pb = np.dot(P, b)
    y = np.linalg.solve(L,Pb)
    print(L)
    print(Pb)
    print(y)
    x_temp = np.linalg.solve(U,y)
    x = np.dot(Q, x_temp)
    return x

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
    zB_part = np.linalg.solve(U11, y_piv)
    if n > r:
        NB = -np.linalg.solve(U11, U12)         # r x (n-r)
        N  = Q @ np.vstack([NB, np.eye(n-r)])
    else:
        N  = np.zeros((n,0))

    z_part = np.concatenate([zB_part, np.zeros(n-r)])
    x_part = Q @ z_part

    return x_part, N, True, r
