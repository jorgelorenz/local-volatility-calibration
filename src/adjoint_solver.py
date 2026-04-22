"""
adjoint_solver.py
-----------------
Solves the discrete adjoint (co-state) equation backward in maturity T.

Derivation (Discretize-then-Optimize)
--------------------------------------
The discrete state system for step n -> n+1 is:

    A^n * C^{n+1} = B^n * C^n  +  bc^n

where A^n = I - theta*dT*L(u_col^n) and B^n = I + (1-theta)*dT*L(u_col^n).

The Lagrangian:
    L = J(C)  +  sum_{n=0}^{N_T-1}  (p^{n+1})^T (A^n C^{n+1} - B^n C^n - bc^n)

Stationarity conditions delta L / delta C^{n+1} = 0:

For n+1 = N_T (terminal time):
    (A^{N_T-1})^T p^{N_T}  =  - dJ/dC^{N_T}

For n+1 = k, 1 <= k <= N_T-1 (internal times):
    (A^{k-1})^T p^k  =  (B^k)^T p^{k+1} - dJ/dC^k

where dJ/dC^k = dK * dT * w[:,k] * (C[:,k] - z[:,k])

The boundary conditions for p are homogeneous Dirichlet (p=0 at K_min, K_max),
consistent with the state boundary conditions.

Algorithm (backward sweep from n = N_T-1 down to 0):
    1. p^{N_T} = 0  (terminal condition on adjoint, distinct from delta L / delta C^{N_T})
       Wait - that is NOT correct. See below.

Correct terminal condition
--------------------------
At n+1 = N_T:
    (A^{N_T-1})^T p^{N_T}  =  - dJ/dC^{N_T}

This gives p^{N_T} as the solution of a tridiagonal system (NOT zero).

Then for k = N_T-1, N_T-2, ..., 1:
    (A^{k-1})^T p^k  =  (B^k)^T p^{k+1} - dJ/dC^k

Public API
----------
solve_adjoint(u, C, z, w, grid, theta=0.5)  ->  p  (shape (N_K+1, N_T+1))
"""

from __future__ import annotations
import numpy as np
from scipy.linalg import solve_banded

from .grid import Grid
from .state_solver import _coeffs, _build_banded, _apply_tridiag


def _build_banded_T(lo, diag, hi):
    """
    Pack the TRANSPOSE of a tridiagonal matrix into scipy banded form.

    For a tridiagonal A with:
      A[i,i-1] = lo[i]   (sub-diagonal)
      A[i,i]   = diag[i]
      A[i,i+1] = hi[i]   (super-diagonal)

    A^T has:
      (A^T)[j,j-1] = A[j-1,j] = hi[j-1]   (sub-diagonal of A^T)
      (A^T)[j,j]   = diag[j]
      (A^T)[j,j+1] = A[j+1,j] = lo[j+1]   (super-diagonal of A^T)

    In scipy banded (1,1) format:
      ab[0,j] = A^T[j-1,j] = lo[j]   -> ab[0,1:] = lo[1:]
      ab[1,j] = diag[j]
      ab[2,j] = A^T[j+1,j] = hi[j]   -> ab[2,:-1] = hi[:-1]
    """
    N = len(diag)
    ab = np.zeros((3, N))
    ab[0, 1:]  = lo[1:]     # super-diagonal of A^T = sub-diagonal of A (shifted correctly)
    ab[1, :]   = diag
    ab[2, :-1] = hi[:-1]    # sub-diagonal of A^T = super-diagonal of A (shifted correctly)
    return ab


def solve_adjoint(
    u: np.ndarray,
    C: np.ndarray,
    z: np.ndarray,
    w: np.ndarray,
    grid: Grid,
    theta: float = 0.5,
    source_override: np.ndarray | None = None,
) -> np.ndarray:
    """
    Solve the discrete adjoint equation backward in T.

    Adjoint equations (for interior nodes):

    Terminal (k = N_T):
        (A^{N_T-1})^T p^{N_T}  =  -dJ/dC^{N_T}

    Internal (k = N_T-1, ..., 1):
        (A^{k-1})^T p^k  =  (B^k)^T p^{k+1} - dJ/dC^k

    where  dJ/dC^k_{int}  =  dK * dT * w_{int,k} * (C_{int,k} - z_{int,k})
    (integration weights included here so that J = 0.5 * dK*dT * sum w*(C-z)^2 )

    Parameters
    ----------
    u               : local variance, shape (N_K+1, N_T+1)
    C               : model call prices (state), shape (N_K+1, N_T+1)
    z               : observed call prices, shape (N_K+1, N_T+1); NaN -> zero
    w               : weight array, shape (N_K+1, N_T+1); NaN -> zero
    grid            : Grid instance
    theta           : theta-scheme (must match state solver)
    source_override : optional precomputed dJ/dC source, shape (N_K+1, N_T+1).
                      If provided, replaces the default w*(C-z) source.
                      This is used for the IV misfit where the source is
                      (iv_model - iv_mkt) / vega instead of w*(C-z).
                      The caller must already include the dK*dT factor if needed.
                      NOTE: source_override should be the RAW source (no dK*dT
                      factor) analogous to w*(C-z); the dK*dT is applied here.

    Returns
    -------
    p     : adjoint variable, shape (N_K+1, N_T+1)
    """
    K   = grid.K
    dK  = grid.dK
    dT  = grid.dT
    N_K = grid.N_K
    N_T = grid.N_T

    r_arr, q_arr = grid.rate_arrays()

    p = np.zeros((N_K + 1, N_T + 1))

    # Source: dJ/dC^k_{int}
    if source_override is not None:
        # source_override is raw (no dK*dT), same convention as w*(C-z)
        raw = np.where(~np.isnan(source_override), source_override, 0.0)
        source = dK * dT * raw
    else:
        # Default: price misfit source = dK*dT * w*(C-z)  (with NaN -> 0)
        diff = C - z
        valid = ~np.isnan(diff) & ~np.isnan(w)
        source = np.where(valid, dK * dT * w * diff, 0.0)   # shape (N_K+1, N_T+1)

    def _matrices_for_step(n):
        """Return A^T band matrix and B matrices for step n -> n+1."""
        T_mid = 0.5 * (grid.T[n] + grid.T[n + 1])
        r = float(grid.r_val(T_mid))
        q = float(grid.q_val(T_mid))
        u_col = 0.5 * (u[:, n] + u[:, n + 1])
        lo, diag, hi = _coeffs(u_col, K, r, q, dK)

        # A = I - theta*dT*L
        A_lo   = -theta * dT * lo
        A_diag =  1.0 - theta * dT * diag
        A_hi   = -theta * dT * hi
        AT_band = _build_banded_T(A_lo, A_diag, A_hi)

        # B = I + (1-theta)*dT*L
        B_lo   = (1.0 - theta) * dT * lo
        B_diag = 1.0 + (1.0 - theta) * dT * diag
        B_hi   = (1.0 - theta) * dT * hi
        return AT_band, B_lo, B_diag, B_hi

    # ---- Terminal step: k = N_T ----
    # (A^{N_T-1})^T p^{N_T} = -dJ/dC^{N_T}
    AT_band, B_lo, B_diag, B_hi = _matrices_for_step(N_T - 1)
    rhs = -source[1:-1, N_T]
    p[1:-1, N_T] = solve_banded((1, 1), AT_band, rhs)
    p[0, N_T] = 0.0
    p[-1, N_T] = 0.0

    # ---- Internal backward sweep: k = N_T-1, ..., 1 ----
    for k in range(N_T - 1, 0, -1):
        # (A^{k-1})^T p^k  =  (B^k)^T p^{k+1} - dJ/dC^k
        # Use matrices for step k-1 -> k  (i.e., n = k-1)
        AT_band, _, _, _ = _matrices_for_step(k - 1)
        # Matrices for step k -> k+1 (to build B^k^T)
        _, B_lo_k, B_diag_k, B_hi_k = _matrices_for_step(k)

        # B^T * p^{k+1}  (transpose of B: sub=hi, super=lo)
        p_next = p[1:-1, k + 1]
        BT_p = B_diag_k * p_next.copy()
        BT_p[1:]  += B_hi_k[:-1]  * p_next[:-1]   # B_hi is super-diag of B -> sub of B^T
        BT_p[:-1] += B_lo_k[1:]   * p_next[1:]    # B_lo is sub-diag of B -> super of B^T

        rhs = BT_p - source[1:-1, k]
        p[1:-1, k] = solve_banded((1, 1), AT_band, rhs)
        p[0, k] = 0.0
        p[-1, k] = 0.0

    # p[:,0] not needed for gradient (no step -1->0)
    return p
