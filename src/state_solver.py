"""
state_solver.py
---------------
Solves the Dupire forward PDE for European call option prices.

PDE (state equation)
--------------------
  dC/dT = (1/2) u(K,T) K^2 d2C/dK2  -  (r(T) - q(T)) K dC/dK  -  q(T) C

  for (K,T) in Omega = (K_min, K_max) x (0, T_max)

  Initial condition:  C(K, 0)     = max(S0 - K, 0)
  Left  BC:           C(K_min, T) = S0*Bq(T) - K_min*Br(T)
  Right BC:           C(K_max, T) = 0

where u(K,T) = sigma^2(K,T) is the local variance (the control parameter).

Discretization
--------------
theta-scheme (Crank-Nicolson by default, theta=0.5):

  (C^{n+1} - C^n) / dT  =  theta * L^n(u) C^{n+1}  +  (1-theta) * L^n(u) C^n

Spatial operator L(u) at interior node i (using centered differences):

  [L(u)C]_i = (1/2) u_i K_i^2 (C_{i+1} - 2C_i + C_{i-1}) / dK^2
             - (r-q) K_i       (C_{i+1} - C_{i-1}) / (2*dK)
             - q C_i

This gives the tridiagonal linear system at each time step:

  A(u^n) C^{n+1} = B(u^n) C^n + rhs_bc

where A, B are (N_K-1) x (N_K-1) tridiagonal matrices acting on the
interior nodes {1, ..., N_K-1}.

Public API
----------
solve_state(u, grid, theta=0.5)  ->  C  (shape (N_K+1, N_T+1))

  u     : local variance array, shape (N_K+1, N_T+1)
  grid  : Grid instance
  theta : implicitness (0=explicit, 0.5=CN, 1=BE).  Default 0.5.

  Returns the full call price surface C[i,n] for all (K_i, T_n).

Internal helpers
----------------
_build_tridiag(u_col, K, r, q, dK, dT, theta)
    Returns (A, B) as scipy banded matrices for one time step.
"""

from __future__ import annotations
import numpy as np
from scipy.linalg import solve_banded

from .grid import Grid


def _coeffs(u_col: np.ndarray, K: np.ndarray,
            r: float, q: float, dK: float) -> tuple:
    """
    Compute tridiagonal coefficients for interior nodes i=1,...,N_K-1.

    Returns (lo, diag, hi) each of shape (N_K-1,) corresponding to
    nodes 1..N_K-1.

    For node i:
      alpha_i = (1/2) u_i K_i^2 / dK^2
      beta_i  = (r-q) K_i / (2*dK)

    Tridiagonal entries of L(u):
      lower_i  = alpha_i + beta_i       (coefficient of C_{i-1})
      center_i = -2*alpha_i - q         (coefficient of C_i)
      upper_i  = alpha_i - beta_i       (coefficient of C_{i+1})
    """
    idx = np.arange(1, len(K) - 1)   # interior indices 1..N_K-1
    Ki = K[idx]
    ui = u_col[idx]

    alpha = 0.5 * ui * Ki**2 / dK**2
    beta  = (r - q) * Ki / (2.0 * dK)

    lo   = alpha + beta       # sub-diagonal  (multiplies C_{i-1})
    diag = -2.0 * alpha - q   # main diagonal (multiplies C_i)
    hi   = alpha - beta       # super-diagonal(multiplies C_{i+1})
    return lo, diag, hi


def _build_banded(lo, diag, hi):
    """
    Pack (lo, diag, hi) into scipy.linalg.solve_banded's ab format:
    shape (3, N) where
      ab[0,1:] = hi   (super-diagonal, first element unused)
      ab[1,:]  = diag
      ab[2,:-1]= lo   (sub-diagonal, last element unused)
    """
    N = len(diag)
    ab = np.zeros((3, N))
    ab[0, 1:]  = hi[:-1]   # scipy convention: ab[0,j] is coeff of x[j] in eq j-1
    ab[1, :]   = diag
    ab[2, :-1] = lo[1:]    # ab[2,j] is coeff of x[j] in eq j+1
    return ab


def _step_matrices(u_col: np.ndarray, K: np.ndarray,
                   r: float, q: float, dK: float, dT: float,
                   theta: float):
    """
    Build the banded matrix A and the dense operator B for one time step.

    System: A * C^{n+1}_int = B * C^n_int  +  rhs_correction

    A = I - theta*dT*L
    B = I + (1-theta)*dT*L

    Returns
    -------
    A_band : (3, N_int) banded form for solve_banded
    B_lo, B_diag, B_hi : dense tridiagonal coefficients of B  (each N_int,)
    """
    lo, diag, hi = _coeffs(u_col, K, r, q, dK)
    N = len(diag)

    # A = I - theta*dT*L
    A_lo   = -theta * dT * lo
    A_diag = 1.0 - theta * dT * diag
    A_hi   = -theta * dT * hi
    A_band = _build_banded(A_lo, A_diag, A_hi)

    # B = I + (1-theta)*dT*L
    B_lo   = (1.0 - theta) * dT * lo
    B_diag = 1.0 + (1.0 - theta) * dT * diag
    B_hi   = (1.0 - theta) * dT * hi

    return A_band, B_lo, B_diag, B_hi


def _apply_tridiag(lo, diag, hi, v):
    """Multiply tridiagonal matrix (lo,diag,hi) by vector v."""
    N = len(diag)
    out = diag * v
    out[1:]  += lo[1:]  * v[:-1]
    out[:-1] += hi[:-1] * v[1:]
    return out


def solve_state(u: np.ndarray, grid: Grid, theta: float = 0.5) -> np.ndarray:
    """
    Solve the Dupire forward PDE for call prices.

    Parameters
    ----------
    u     : local variance  sigma^2(K,T), shape (N_K+1, N_T+1)
    grid  : Grid instance
    theta : theta-scheme parameter (0.5 = Crank-Nicolson, 1.0 = backward Euler)

    Returns
    -------
    C     : call price surface, shape (N_K+1, N_T+1)
            C[i, n]  =  C(K_i, T_n)
    """
    K    = grid.K
    T    = grid.T
    dK   = grid.dK
    dT   = grid.dT
    N_K  = grid.N_K
    N_T  = grid.N_T

    # Precompute BCs and rates
    BC_left  = grid.boundary_left()    # shape (N_T+1,)
    BC_right = grid.boundary_right()   # shape (N_T+1,)
    r_arr, q_arr = grid.rate_arrays()  # shape (N_T+1,)

    # Allocate output
    C = np.zeros((N_K + 1, N_T + 1))

    # Initial condition
    C[:, 0] = grid.initial_condition()
    C[0, 0] = BC_left[0]
    C[-1, 0] = BC_right[0]

    # Time march: n -> n+1
    for n in range(N_T):
        T_mid = 0.5 * (T[n] + T[n + 1])   # evaluate coefficients at mid-time
        r = float(grid.r_val(T_mid))
        q = float(grid.q_val(T_mid))

        # Use average local variance for CN step
        u_col = 0.5 * (u[:, n] + u[:, n + 1])

        A_band, B_lo, B_diag, B_hi = _step_matrices(
            u_col, K, r, q, dK, dT, theta
        )

        # Interior values of current column
        C_int = C[1:-1, n]

        # rhs = B * C_int
        rhs = _apply_tridiag(B_lo, B_diag, B_hi, C_int)

        # Correction for BCs of C^{n+1}:
        # The first interior equation involves C[0,n+1] (left BC)
        # The last  interior equation involves C[-1,n+1] (right BC)
        bc_l_next = BC_left[n + 1]
        bc_r_next = BC_right[n + 1]

        lo_coeff, _, _ = _coeffs(u_col, K, r, q, dK)
        hi_coeff       = _coeffs(u_col, K, r, q, dK)[2]

        rhs[0]  += theta * dT * lo_coeff[0] * bc_l_next   # lo[0] = coeff at i=1 for C_{i-1}=C_0
        rhs[-1] += theta * dT * hi_coeff[-1] * bc_r_next  # hi[-1]= coeff at i=N_K-1 for C_{i+1}=C_NK

        # Also add (1-theta) terms from current BCs
        bc_l_cur = BC_left[n]
        bc_r_cur = BC_right[n]
        rhs[0]  += (1.0 - theta) * dT * lo_coeff[0] * bc_l_cur
        rhs[-1] += (1.0 - theta) * dT * hi_coeff[-1] * bc_r_cur

        # Solve A * C_int_new = rhs
        C_int_new = solve_banded((1, 1), A_band, rhs)

        # Store
        C[0,  n + 1] = bc_l_next
        C[-1, n + 1] = bc_r_next
        C[1:-1, n + 1] = C_int_new

    return C
