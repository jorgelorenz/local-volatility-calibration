"""
slv_pricer.py
-------------
Finite-difference solver for the Stochastic Local Volatility (SLV) backward
pricing PDE using the Craig-Sneyd ADI scheme.

Model (Heston-like SLV)
-----------------------
Asset price and instantaneous variance follow:

    dS = (r - q) S dt  +  L(S, t) sqrt(v) S dW_S
    dv = kappa (theta_v - v) dt  +  xi sqrt(v) dW_v

    corr(dW_S, dW_v) = rho dt

where L(S, t) is the leverage function (2-D surface on the S x T grid).

When L(S, t) = sigma_loc(S, t) / sqrt(v0) with constant v0, the model
recovers the pure local-volatility process dS = sigma_loc S dW_S.

2D backward PDE (tau = T_final - t, forward in tau)
-----------------------------------------------------
    dV/dtau = A_S V  +  A_v V  +  A_mix V  -  r V

where
    A_S   V = 1/2  L^2 v S^2  V_SS  +  (r - q) S  V_S
    A_v   V = 1/2  xi^2 v     V_vv  +  kappa (theta_v - v)  V_v
    A_mix V = rho  xi  L  v  S      V_Sv

Terminal condition (tau=0):   V(S, v, 0) = payoff(S)
Left   BC in S (S = S_min):   V = 0
Right  BC in S (S = S_max):   V = S_max * e^{-q tau} - K * e^{-r tau}
BC in v (v = 0):              Neumann dV/dv = 0
BC in v (v = v_max):          V = Black-Scholes with vol = L * sqrt(v_max)

Craig-Sneyd ADI scheme
-----------------------
Step 1 (S-implicit, explicit in v and cross):
    Y  = V + dt [theta A_S Y  + (1-theta) A_S V  + A_v V  + A_mix V  - r V]
    => (I - theta dt A_S) Y = V + dt [(1-theta) A_S V + A_v V + A_mix V - r V]

Step 2 (v-implicit correction):
    (I - theta dt A_v) (V_new - Y) = theta dt (A_v Y - A_v V)
    => V_new = Y + (I - theta dt A_v)^{-1} [theta dt (A_v Y - A_v V)]

Public API
----------
solve_slv(L, S_nodes, t_grid, T_final, r_func, q_func,
          kappa, theta_v, xi, rho, v0, payoff, K_strike,
          N_v=30, v_max=None, theta_adi=0.5)
    -> V_at_v0  shape (N_S,)
"""

from __future__ import annotations
import numpy as np
from scipy.linalg import solve_banded


# ---------------------------------------------------------------------------
# Rate accessor
# ---------------------------------------------------------------------------

def _make_r(x):
    if callable(x):
        return x
    val = float(x)
    return lambda t: val


# ---------------------------------------------------------------------------
# Banded system helpers
# ---------------------------------------------------------------------------

def _make_banded(lo: np.ndarray, diag: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """
    Pack three arrays of equal length N into scipy solve_banded (1,1) format.
    lo[0] and hi[-1] are ignored (they lie outside the matrix).
    """
    N = len(diag)
    assert len(lo) == N and len(hi) == N
    ab = np.zeros((3, N))
    ab[1, :]   = diag
    ab[0, 1:]  = hi[:-1]    # super-diag: ab[0, j] multiplies x[j], j >= 1
    ab[2, :-1] = lo[1:]     # sub-diag:   ab[2, j] multiplies x[j], j <= N-2
    return ab


def _apply_banded(lo: np.ndarray, diag: np.ndarray, hi: np.ndarray,
                  v: np.ndarray) -> np.ndarray:
    """
    Multiply a tridiagonal matrix (lo, diag, hi) by vector v.
    All three coefficient arrays must have the same length as v.
    lo[0] and hi[-1] are not used.
    """
    out = diag * v
    out[:-1] += hi[:-1] * v[1:]
    out[1:]  += lo[1:]  * v[:-1]
    return out


# ---------------------------------------------------------------------------
# Operator coefficient builders
# ---------------------------------------------------------------------------

def _S_coefficients(S: np.ndarray, vj: float, L_col: np.ndarray,
                    r: float, q: float):
    """
    Return full-length (N_S,) tridiagonal coefficients for A_S at variance vj.

    A_S V = 1/2 L^2 vj S^2 V_SS  +  (r-q) S V_S   (central differences)

    lo[i]   = coefficient of V[i-1]
    diag[i] = coefficient of V[i]
    hi[i]   = coefficient of V[i+1]

    Boundary rows (i=0, i=N_S-1) are set to zero; BCs imposed externally.
    """
    N_S = len(S)
    dS  = S[1] - S[0]

    diff = 0.5 * L_col**2 * vj * S**2 / dS**2
    adv  = (r - q) * S / (2.0 * dS)

    lo   =  diff - adv          # coeff of V[i-1]
    diag = -2.0 * diff          # coeff of V[i]
    hi   =  diff + adv          # coeff of V[i+1]

    # Zero-out boundary rows
    lo[0]    = diag[0]    = hi[0]    = 0.0
    lo[-1]   = diag[-1]   = hi[-1]   = 0.0
    return lo, diag, hi


def _v_coefficients(v: np.ndarray, kappa: float, theta_v: float, xi: float):
    """
    Return full-length (N_v+1,) tridiagonal coefficients for A_v.

    A_v V = 1/2 xi^2 v V_vv  +  kappa (theta_v - v) V_v

    Boundary rows (j=0 Neumann, j=N_v Dirichlet) zeroed; handled externally.
    """
    N = len(v)
    dv   = v[1] - v[0]
    diff = 0.5 * xi**2 * v / dv**2
    adv  = kappa * (theta_v - v) / (2.0 * dv)

    lo   =  diff - adv
    diag = -2.0 * diff
    hi   =  diff + adv

    lo[0]  = diag[0]  = hi[0]  = 0.0
    lo[-1] = diag[-1] = hi[-1] = 0.0
    return lo, diag, hi


# ---------------------------------------------------------------------------
# Cross-derivative (explicit)
# ---------------------------------------------------------------------------

def _cross_deriv(V: np.ndarray, S: np.ndarray, v: np.ndarray,
                 L: np.ndarray, rho: float, xi: float) -> np.ndarray:
    """
    Compute A_mix V = rho xi L[i] v[j] S[i] * d^2V/(dS dv)
    using 2nd-order central differences.
    Returns array shape (N_S, N_v+1); boundary rows/cols set to zero.
    """
    N_S = len(S)
    N_v = len(v) - 1
    dS  = S[1] - S[0]
    dv  = v[1] - v[0]

    out = np.zeros_like(V)
    # Interior points only
    dVdSdv = np.zeros_like(V)
    dVdSdv[1:-1, 1:-1] = (
        V[2:, 2:] - V[:-2, 2:] - V[2:, :-2] + V[:-2, :-2]
    ) / (4.0 * dS * dv)

    for i in range(1, N_S - 1):
        for j in range(1, N_v):
            out[i, j] = rho * xi * L[i] * v[j] * S[i] * dVdSdv[i, j]
    return out


# ---------------------------------------------------------------------------
# Public solver
# ---------------------------------------------------------------------------

def solve_slv(
    L: np.ndarray,
    S_nodes: np.ndarray,
    t_grid: np.ndarray,
    T_final: float,
    r_func,
    q_func,
    kappa: float,
    theta_v: float,
    xi: float,
    rho: float,
    v0: float,
    payoff,
    K_strike: float,
    N_v: int = 30,
    v_max: float | None = None,
    theta_adi: float = 0.5,
) -> np.ndarray:
    """
    Price a European call via the SLV backward PDE using Craig-Sneyd ADI.

    Parameters
    ----------
    L        : leverage surface, shape (N_S, N_T+1)
    S_nodes  : asset-price grid, shape (N_S,)
    t_grid   : physical time grid, shape (N_T+1,), t_grid[-1]=T_final
    T_final  : maturity
    r_func   : callable r(t)->float or scalar
    q_func   : callable q(t)->float or scalar
    kappa    : mean-reversion speed
    theta_v  : long-run variance
    xi       : vol-of-vol
    rho      : correlation in [-1,1]
    v0       : initial variance (interpolate output to this level)
    payoff   : callable(S)->array or array shape (N_S,)
    K_strike : strike for right-S boundary condition
    N_v      : number of v intervals
    v_max    : v upper bound; default = max(v0, theta_v) * 5
    theta_adi: ADI theta (0.5 = second-order)

    Returns
    -------
    V_at_v0 : shape (N_S,) -- option values V(S_i, v0, t=0)
    """
    from .utils import bs_call as _bs_call

    r_fn = _make_r(r_func)
    q_fn = _make_r(q_func)

    S    = np.asarray(S_nodes, dtype=float)
    N_S  = len(S)
    N_T  = len(t_grid) - 1

    if v_max is None:
        v_max = max(v0, theta_v) * 5.0
    v    = np.linspace(0.0, v_max, N_v + 1)
    dv   = v[1] - v[0]

    # ---- terminal condition ------------------------------------------------
    if callable(payoff):
        pay = np.asarray(payoff(S), dtype=float)
    else:
        pay = np.asarray(payoff, dtype=float)

    # V[i_S, j_v]
    V = np.zeros((N_S, N_v + 1))
    for j in range(N_v + 1):
        V[:, j] = pay.copy()

    V[0,  :] = 0.0
    V[-1, :] = np.maximum(S[-1] - K_strike, 0.0)  # intrinsic at tau=0

    # ---- time march (tau: 0 -> T_final) ------------------------------------
    for n in range(N_T - 1, -1, -1):
        t_cur   = t_grid[n]
        t_next  = t_grid[n + 1]
        dt      = t_next - t_cur      # positive step in physical time = tau step
        tau     = T_final - t_cur     # tau at NEW (unknown) time level

        t_mid   = 0.5 * (t_cur + t_next)
        r       = r_fn(t_mid)
        q       = q_fn(t_mid)

        L_mid   = 0.5 * (L[:, n] + L[:, n + 1])   # shape (N_S,)

        # ---- boundary conditions -------------------------------------------
        bc_l_S = 0.0
        bc_r_S = S[-1] * np.exp(-q * tau) - K_strike * np.exp(-r * tau)

        # v_max BC: BS with effective vol = L_mid * sqrt(v_max)
        bc_v_max = np.array([
            _bs_call(S[i], K_strike, max(tau, 1e-12), r, q,
                     max(L_mid[i] * np.sqrt(v_max), 1e-8))
            for i in range(N_S)
        ])
        bc_v_max[0]  = bc_l_S
        bc_v_max[-1] = bc_r_S

        # ---- A_v coefficients (full length N_v+1, BCs zeroed) ---------------
        lo_v, diag_v, hi_v = _v_coefficients(v, kappa, theta_v, xi)

        # Neumann at j=0: one-sided diff, dV/dv=0 => V[i,0] = V[i,1]
        # Implement as: A_v[i,0] = kappa*theta_v*(V[i,1]-V[i,0])/dv
        # (only advection term survives at v=0 since diff ~ v -> 0)

        # ---- explicit cross-derivative -------------------------------------
        cross = _cross_deriv(V, S, v, L_mid, rho, xi)   # (N_S, N_v+1)

        # ---- Step 1: S-implicit for each j ---------------------------------
        Y = V.copy()

        for j in range(N_v + 1):
            vj = v[j]

            # A_v V (explicit, for this j)
            if j == 0:
                # Neumann: use one-sided
                A_v_V = kappa * theta_v * (V[:, 1] - V[:, 0]) / dv
                A_v_V[0] = 0.0; A_v_V[-1] = 0.0
            elif j == N_v:
                A_v_V = np.zeros(N_S)  # BC row
            else:
                A_v_V = (lo_v[j] * V[:, j-1]
                         + diag_v[j] * V[:, j]
                         + hi_v[j]   * V[:, j+1])

            # Explicit RHS
            rhs = V[:, j] + dt * (A_v_V + cross[:, j] - r * V[:, j])

            # Handle boundary rows for S
            rhs[0]  = bc_l_S
            rhs[-1] = bc_v_max[-1] if j == N_v else bc_r_S

            if j == N_v:
                # Dirichlet in v -> no S-solve needed, just set BC
                Y[:, j] = bc_v_max.copy()
                continue

            # A_S coefficients (full N_S)
            lo_s, diag_s, hi_s = _S_coefficients(S, vj, L_mid, r, q)

            # (1-theta)*dt*A_S*V contribution to RHS (interior rows only)
            A_S_V = _apply_banded(lo_s, diag_s, hi_s, V[:, j])

            rhs_int = rhs[1:-1] + (1.0 - theta_adi) * dt * A_S_V[1:-1]
            # BC column corrections
            rhs_int[0]  -= theta_adi * dt * lo_s[1]  * bc_l_S
            rhs_int[-1] += theta_adi * dt * hi_s[-2] * bc_r_S

            # LHS: (I - theta*dt*A_S) for interior
            # A_S lo[i]*V[i-1] + diag[i]*V[i] + hi[i]*V[i+1]
            # LHS coeff: delta_{ij} - theta*dt * A_S_{ij}
            n_int = N_S - 2
            LHS_lo   = np.empty(n_int)
            LHS_diag = np.empty(n_int)
            LHS_hi   = np.empty(n_int)
            LHS_lo[:]   = -theta_adi * dt * lo_s[1:-1]
            LHS_diag[:] = 1.0 - theta_adi * dt * diag_s[1:-1]
            LHS_hi[:]   = -theta_adi * dt * hi_s[1:-1]

            ab = _make_banded(LHS_lo, LHS_diag, LHS_hi)
            Y[1:-1, j] = solve_banded((1, 1), ab, rhs_int)
            Y[0,  j] = bc_l_S
            Y[-1, j] = bc_r_S

        Y[:, N_v] = bc_v_max
        Y[0, :]   = 0.0

        # ---- Step 2: v-implicit correction for each i ----------------------
        V_new = Y.copy()

        for i in range(1, N_S - 1):
            # Interior v nodes: j = 1 .. N_v-1
            n_vint = N_v - 1
            vint   = v[1:-1]
            dv2    = dv**2

            diff_j = 0.5 * xi**2 * vint / dv2
            adv_j  = kappa * (theta_v - vint) / (2.0 * dv)

            # A_v Y[:,interior] and A_v V[:,interior] at node i (scalars per j)
            A_v_Y   = (diff_j - adv_j) * Y[i, :-2] \
                    - 2.0 * diff_j     * Y[i, 1:-1] \
                    + (diff_j + adv_j) * Y[i, 2:]

            A_v_Vold = (diff_j - adv_j) * V[i, :-2] \
                     - 2.0 * diff_j     * V[i, 1:-1] \
                     + (diff_j + adv_j) * V[i, 2:]

            correction_rhs = theta_adi * dt * (A_v_Y - A_v_Vold)
            # Note: A_v_Y already uses Y[i, N_v]=bc_v_max[i] for the last row,
            # so no additional BC correction is needed here.

            # LHS: (I - theta*dt*A_v) for interior v
            LHS_d  = 1.0 + 2.0 * theta_adi * dt * diff_j
            LHS_l  = np.empty(n_vint)
            LHS_h  = np.empty(n_vint)
            LHS_l[:] = -theta_adi * dt * (diff_j - adv_j)   # sub-diag
            LHS_h[:] = -theta_adi * dt * (diff_j + adv_j)   # super-diag

            ab_v = _make_banded(LHS_l, LHS_d, LHS_h)
            delta = solve_banded((1, 1), ab_v, correction_rhs)

            V_new[i, 1:-1] = Y[i, 1:-1] + delta
            V_new[i, 0]    = Y[i, 1]    # Neumann: dV/dv=0 => V[i,0]=V[i,1]
            V_new[i, N_v]  = bc_v_max[i]

        # Enforce global BCs
        V_new[0,  :] = 0.0
        V_new[-1, :] = bc_r_S
        V_new[:, N_v] = bc_v_max
        V = V_new

    # ---- interpolate to v = v0 -----------------------------------------------
    j0 = int(np.searchsorted(v, v0))
    j0 = int(np.clip(j0, 1, N_v))
    alpha   = (v0 - v[j0 - 1]) / (v[j0] - v[j0 - 1])
    V_at_v0 = (1.0 - alpha) * V[:, j0 - 1] + alpha * V[:, j0]

    return V_at_v0
