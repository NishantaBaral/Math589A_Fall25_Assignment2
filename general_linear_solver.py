import numpy as np
import paqlu_decomposition_square as paqlu_square
import paqlu_decomposition_rectangular as paqlu_rectangular


def _as_2d(b):
    """Return b as (m,k) and a flag telling if original was a vector."""
    if b.ndim == 1:
        return b[:, None], True
    return b, False


def _restore_shape(x, was_vector):
    return x.ravel() if was_vector else x


def forward_substitution(L, b):
    """
    Solve L y = b for y where L is (unit) lower-triangular (m x r) with ones on diag.
    b can be shape (m,) or (m,k). Returns y with same trailing shape as b.
    """
    B, was_vec = _as_2d(b)
    m = L.shape[0]
    y = np.zeros((m, B.shape[1]), dtype=B.dtype)
    for i in range(m):
        # If L is unit lower, L[i,i] == 1. If not, divide by L[i,i].
        y[i, :] = B[i, :] - L[i, :i] @ y[:i, :]
        if not np.allclose(L[i, i], 1.0):
            y[i, :] /= L[i, i]
    return _restore_shape(y, was_vec)


def back_substitution(U, y):
    """
    Solve U x = y for x where U is upper-triangular (r x r).
    y can be shape (r,) or (r,k). Returns x with same trailing shape as y.
    """
    Y, was_vec = _as_2d(y)
    r = U.shape[0]
    x = np.zeros((r, Y.shape[1]), dtype=Y.dtype)
    for i in range(r - 1, -1, -1):
        x[i, :] = (Y[i, :] - U[i, i + 1:] @ x[i + 1:, :]) / U[i, i]
    return _restore_shape(x, was_vec)


def _build_nullspace_from_U(U, tol=1e-12):
    """
    Given U of shape (r x n) with r = rank and leading r x r block nonsingular,
    return N_perm = [ -U1^{-1} U2 ; I ] of shape (n x (n-r)).
    """
    r, n = U.shape
    f = n - r
    if f == 0:
        return np.zeros((n, 0), dtype=U.dtype)

    U1 = U[:, :r]              # (r x r) upper-triangular
    U2 = U[:, r:]              # (r x f)

    # Solve U1 * T = -U2  for T (r x f). Use back-substitution column-wise.
    T_rhs = -U2
    T = back_substitution(U1, T_rhs)  # shape (r, f)

    # N_perm = [T; I_f]
    N_perm = np.vstack([T, np.eye(f, dtype=U.dtype)])
    # (n x f)
    return N_perm


def rectangular_solve(A, b, tol=1e-12):
    """
    Solve A x = b (m x n, rank r) using PA=QLU from rectangular PAQLU.
    Returns (c, N) where c is a particular solution (n,) or (n,k),
    and N is an (n x f) nullspace basis (f = n - r).
    Raises ValueError if inconsistent.
    """
    P, Q, L, U = paqlu_rectangular.paqlu_decomposition_in_place(A)
    B, was_vec = _as_2d(b)

    # Forward solve: L y = P b
    Pb = P @ B
    y = forward_substitution(L, Pb)
    Y, _ = _as_2d(y)

    r = U.shape[0]
    m, n = A.shape

    # Consistency check: last (m-r) rows of y must be ~0
    if m > r:
        resid = np.linalg.norm(Y[r:, :], ord=np.inf)
        if not np.isfinite(resid) or resid > tol:
            raise ValueError("Linear system is inconsistent: b not in column space of A.")

    # Particular solution in permuted coordinates:
    # U1 x1 + U2 x2 = y[:r]; choose x2 = 0  =>  U1 x1 = y[:r]
    U1 = U[:, :r]
    y_top = Y[:r, :]
    x1 = back_substitution(U1, y_top)            # (r,k)
    k = x1.shape[1]
    x_perm = np.zeros((n, k), dtype=B.dtype)
    x_perm[:r, :] = x1

    # Undo column permutation to get solution in original variable order
    c_full = Q @ x_perm                           # (n,k)
    c = _restore_shape(c_full, was_vec)

    # Nullspace basis
    N_perm = _build_nullspace_from_U(U, tol=tol)  # (n,f)
    N = Q @ N_perm                                # undo the permutation

    return c, N


def solve(A, b, tol=1e-12):
    """
    Unified solver: works for square and rectangular A.
    Returns (c, N) per spec. Raises ValueError if inconsistent.
    """
    m, n = A.shape
    if m == n:
        # Square case; still use the PA=QLU logic safely.
        P, Q, L, U = paqlu_square.paqlu_decomposition_in_place(A)
        B, was_vec = _as_2d(b)

        y = forward_substitution(L, P @ B)        # L y = P b
        # Here U is (n x n) upper-triangular
        x_perm = back_substitution(U, y)          # solve U x_perm = y
        Xp, _ = _as_2d(x_perm)
        c_full = Q @ Xp                            # undo column permutation
        c = _restore_shape(c_full, was_vec)

        # Full column rank => empty nullspace with shape (n,0)
        N = np.zeros((n, 0), dtype=A.dtype)
        return c, N
    else:
        return rectangular_solve(A, b, tol=tol)
