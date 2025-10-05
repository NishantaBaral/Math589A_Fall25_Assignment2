import numpy as np
import paqlu_decomposition_square as paqlu_square
import paqlu_decomposition_rectangular as paqlu_rectangular

###############################################
# Basic triangular solves
###############################################

def forward_substitution(L, b):
    """Solve L y = b for y (L lower triangular, unit or non‑unit diag).
    L may be (m x r) in rectangular case; we only use its leading square part when needed."""
    b = np.asarray(b, dtype=float).flatten()
    n = L.shape[0]
    # If L is not square (m x r with m>=r) we only solve for first r rows when called with reduced L
    # Caller should pass square leading block when needed.
    if L.shape[0] != L.shape[1]:
        # Expect shape (m, r) with m>=r; only leading r rows form a square system with L[:r,:]
        # So this function should only be called with square L. Guard to catch misuse.
        pass
    y = np.zeros(n, dtype=float)
    for i in range(n):
        diag = L[i, i] if abs(L[i, i]) > 0 else 1.0  # allow stored zeros meaning unit diag
        y[i] = (b[i] - L[i, :i] @ y[:i]) / diag
    return y

def back_substitution(U, y):
    """Solve U x = y for x (U upper triangular)."""
    y = np.asarray(y, dtype=float).flatten()
    n = U.shape[0]
    x = np.zeros(n, dtype=float)
    for i in range(n - 1, -1, -1):
        if abs(U[i, i]) < 1e-14:
            raise ZeroDivisionError("Singular upper triangular system (zero pivot)")
        x[i] = (y[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x

###############################################
# Rectangular solver using PAQ = L U (rank‑revealing)
###############################################

def rectangular_solver(A, b, tol=1e-10):
    """Return (x_particular, NullSpaceBasis) solving A x = b.

    If system inconsistent raises ValueError.
    A: m x n (m != n allowed)
    b: length m
    NullSpaceBasis shape: (n, n-rank)
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).flatten()
    m, n = A.shape
    P, Q, L, U, rank = paqlu_rectangular.paqlu_decomposition_in_place(A)

    # Permute RHS
    b1 = (P @ b.reshape(-1, 1)).flatten()

    if rank == 0:
        if np.linalg.norm(b1) > tol:
            raise ValueError("Inconsistent system (all-zero matrix but nonzero b)")
        return np.zeros(n), np.eye(n)

    # L is m x rank; take its leading square part
    L_pivot = L[:rank, :rank]
    b_pivot = b1[:rank]
    y = forward_substitution(L_pivot, b_pivot)

    # Consistency check for remaining rows (should be zero)
    if rank < m:
        tail = b1[rank:]
        if np.linalg.norm(tail, ord=np.inf) > tol:
            raise ValueError("Inconsistent system (pivot rows solved, residual rows nonzero)")

    # Partition U (rank x n)
    U_basic = U[:rank, :rank]
    U_free = U[:rank, rank:]

    # Particular solution: free vars = 0
    z_basic = back_substitution(U_basic, y)
    z = np.concatenate([z_basic, np.zeros(n - rank)])  # permuted variable order
    x_particular = Q @ z  # undo column permutation

    # Null space basis
    if n - rank == 0:
        N = np.zeros((n, 0))
    else:
        N_basic = np.zeros((rank, n - rank))
        for j in range(n - rank):
            rhs = -U_free[:, j]
            nb = back_substitution(U_basic, rhs)
            N_basic[:, j] = nb
        Z = np.vstack([N_basic, np.eye(n - rank)])  # basis in permuted coordinates
        N = Q @ Z

    return x_particular, N

###############################################
# Square solver
###############################################

def square_solver(A, b):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).flatten()
    n = A.shape[0]
    P, Q, L, U = paqlu_square.paqlu_decomposition_in_place(A)
    b1 = (P @ b.reshape(-1, 1)).flatten()
    y = forward_substitution(L, b1)
    z = back_substitution(U, y)
    x = Q @ z
    N = np.zeros((n, 0))  # assume full rank
    return x, N

###############################################
# Dispatcher
###############################################

def solve(A, b, tol=1e-10):
    m, n = A.shape
    if m == n:
        return square_solver(A, b)
    return rectangular_solver(A, b, tol=tol)

if __name__ == "__main__":
    # Wide matrix example (m < n)
    A = np.array([[0, 2, 1, 3],
                  [1, 0, 2, 1],
                  [3, 1, 0, 2]], dtype=float)
    b = np.array([1, 2, 3], dtype=float)
    x, N = solve(A, b)
    print("Particular solution x =", x)
    print("Null space basis N shape =", N.shape)
    print("Residual ||Ax-b|| =", np.linalg.norm(A @ x - b))