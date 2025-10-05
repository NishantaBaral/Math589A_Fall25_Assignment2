import numpy as np
import paqlu_decomposition_square as paqlu_square
import paqlu_decomposition_rectangular as paqlu_rectangular


# ---------- helpers for 1D/2D RHS handling ----------
def _as_2d(b):
    """Return b as (m,k) and a flag if original was 1D."""
    if b.ndim == 1:
        return b[:, None], True
    return b, False

def _restore_shape(x, was_vec):
    return x.ravel() if was_vec else x


# ---------- triangular solves (vector or multiple RHS) ----------
def forward_substitution(L, b):
    """
    Solve L y = b for y; L is (unit) lower-triangular.
    b: (m,) or (m,k)  ->  y with same trailing shape.
    """
    B, was_vec = _as_2d(b)
    m = L.shape[0]
    Y = np.zeros((m, B.shape[1]), dtype=B.dtype)
    for i in range(m):
        # L is unit-lower in PAQLU; if not, divide by L[i,i].
        Y[i, :] = B[i, :] - L[i, :i] @ Y[:i, :]
        if not np.allclose(L[i, i], 1.0):
            Y[i, :] /= L[i, i]
    return _restore_shape(Y, was_vec)

def back_substitution(U, y):
    """
    Solve U x = y for x; U upper-triangular (r x r).
    y: (r,) or (r,k)  ->  x with same trailing shape.
    """
    Y, was_vec = _as_2d(y)
    r = U.shape[0]
    X = np.zeros((r, Y.shape[1]), dtype=Y.dtype)
    for i in range(r - 1, -1, -1):
        X[i, :] = (Y[i, :] - U[i, i + 1:] @ X[i + 1:, :]) / U[i, i]
    return _restore_shape(X, was_vec)


# ---------- nullspace builder ----------
def _nullspace_from_U(U):
    """
    U: shape (r, n), rank r. Partition U = [U1 | U2] with U1 (r x r) nonsingular.
    Return N_perm = [ -U1^{-1} U2 ; I ]  of shape (n, f) where f = n-r.
    """
    r, n = U.shape
    f = n - r
    if f == 0:
        return np.zeros((n, 0), dtype=U.dtype)

    U1 = U[:, :r]        # (r x r)
    U2 = U[:, r:]        # (r x f)
    T = back_substitution(U1, -U2)  # (r x f)
    N_perm = np.vstack([T, np.eye(f, dtype=U.dtype)])  # (n x f)
    return N_perm


# ---------- rectangular solve (general m x n) ----------
def rectangular_solve(A, b, tol=1e-12):
    """
    Solve A x = b with PA=QLU (rectangular).
    Returns (c, N): c is particular solution (n,) or (n,k); N is (n, f).
    Raises ValueError if inconsistent.
    """
    P, Q, L, U = paqlu_rectangular.paqlu_decomposition_in_place(A)
    B, was_vec = _as_2d(b)
    m, n = A.shape
    r = U.shape[0]  # rank

    # Forward solve: L y = P b
    y = forward_substitution(L, P @ B)
    Y, _ = _as_2d(y)

    # Consistency: lower (m-r) rows must be ~0
    if m > r:
        if np.linalg.norm(Y[r:, :], ord=np.inf) > tol:
            raise ValueError("Linear system is inconsistent: b not in column space of A.")

    # Particular solution in permuted coordinates: choose free vars = 0
    U1 = U[:, :r]
    x1 = back_substitution(U1, Y[:r, :])     # (r, k)
    k = x1.shape[1]
    x_perm = np.zeros((n, k), dtype=B.dtype)
    x_perm[:r, :] = x1

    # *** IMPORTANT: undo column permutation with Q.T (matches your PAQLU convention) ***
    c_full = Q.T @ x_perm                    # (n, k)
    c = _restore_shape(c_full, was_vec)

    # Nullspace basis: N_perm then undo permutation the SAME WAY
    N_perm = _nullspace_from_U(U)            # (n, f)
    N = Q.T @ N_perm                         # (n, f)

    return c, N


# ---------- square solve ----------
def solve(A, b, tol=1e-12):
    """
    Unified entry point. Returns (c, N) with shapes:
      c: (n,) or (n,k)
      N: (n, f), or (n,0) if full column rank
    """
    m, n = A.shape
    if m == n:
        P, Q, L, U = paqlu_square.paqlu_decomposition_in_place(A)
        B, was_vec = _as_2d(b)
        y = forward_substitution(L, P @ B)
        x_perm = back_substitution(U, y)
        Xp, _ = _as_2d(x_perm)

        # square path also uses Q.T to unpermute
        c_full = Q.T @ Xp
        c = _restore_shape(c_full, was_vec)

        N = np.zeros((n, 0), dtype=A.dtype)  # full column rank => empty nullspace
        return c, N

    # rectangular
    return rectangular_solve(A, b, tol=tol)
