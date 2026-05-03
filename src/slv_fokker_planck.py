"""
slv_fokker_planck.py
--------------------
2D Fokker-Planck (forward Kolmogorov) PDE solver for the joint density
p(S, v, t) of the Stochastic Local Volatility (SLV) model.

SLV Model
---------
    dS = (r - q) S dt  +  L(S, t) sqrt(v) S dW_S
    dv = kappa (theta_v - v) dt  +  xi sqrt(v) dW_v
    corr(dW_S, dW_v) = rho dt

Fokker-Planck PDE
-----------------
The joint density p(S, v, t) satisfies:

    dp/dt = (1/2) d^2/dS^2 [L^2 v S^2 p]
           - d/dS [(r-q) S p]
           + (1/2) d^2/dv^2 [xi^2 v p]
           - d/dv [kappa(theta_v - v) p]
           - rho xi d^2/dSdv [L sqrt(v) S sqrt(v) p]
           = (1/2) d^2/dS^2 [L^2 v S^2 p]
           - d/dS [(r-q) S p]
           + (1/2) xi^2 d^2/dv^2 [v p]
           - kappa d/dv [(theta_v - v) p]
           - rho xi d^2/dSdv [L v S p]

Discretisation
--------------
Domain:
    S in [S_min, S_max]  (N_S nodes, reuses grid.K)
    v in [0, v_max]      (N_v+1 nodes, uniform)

Boundary conditions:
    p = 0 at S = S_min, S = S_max  (absorbing)
    Neumann dp/dv = 0 at v = 0
    p = 0 at v = v_max

Initial condition:
    p(S, v, 0) = delta(S - S0) * delta(v - v0)
    approximated as a bivariate Gaussian with small std = (dS, dv).

Time stepping: Craig-Sneyd ADI (same structure as slv_pricer.py but
applied to the Fokker-Planck equation).

Step 1 (S-implicit, v and cross explicit):
    (I - theta dt F_S) p* = p^n + dt [(1-theta) F_S p^n + F_v p^n + F_cross p^n]

Step 2 (v-implicit correction):
    (I - theta dt F_v)(p^{n+1} - p*) = theta dt (F_v p* - F_v p^n)

Public API
----------
solve_fokker_planck(L, grid, kappa, theta_v, xi, rho, v0,
                    N_v=40, v_max=None, theta_adi=0.5)
    -> p  shape (N_S, N_v+1, N_T+1)

marginals(p, v_grid)
    -> (p_S, E_v)  each shape (N_S, N_T+1)
       p_S[i,n]  = int p(S_i, v, t_n) dv    (marginal density in S)
       E_v[i,n]  = int v p(S_i, v, t_n) dv / p_S[i,n]  (conditional mean of v)
"""

from __future__ import annotations
import numpy as np
from scipy.linalg import solve_banded

from .grid import Grid


# ---------------------------------------------------------------------------
# Banded helpers (same as slv_pricer.py)
# ---------------------------------------------------------------------------

def _make_banded(lo: np.ndarray, diag: np.ndarray, hi: np.ndarray) -> np.ndarray:
    N  = len(diag)
    ab = np.zeros((3, N))
    ab[1, :]   = diag
    ab[0, 1:]  = hi[:-1]
    ab[2, :-1] = lo[1:]
    return ab


def _apply_banded(lo: np.ndarray, diag: np.ndarray, hi: np.ndarray,
                  v: np.ndarray) -> np.ndarray:
    out        = diag * v
    out[:-1]  += hi[:-1] * v[1:]
    out[1:]   += lo[1:]  * v[:-1]
    return out


# ---------------------------------------------------------------------------
# FP operator coefficients in S
# ---------------------------------------------------------------------------

def _fp_S_coefficients(
    S: np.ndarray,
    vj: float,
    L_col: np.ndarray,
    r: float,
    q: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Conservative finite-difference coefficients for the S-part of the FP operator:

        F_S p = (1/2) d^2/dS^2 [L^2 v S^2 p]  -  d/dS [(r-q) S p]

    Using the conservative (divergence) form with central differences:

        F_S p[i] = (a_{i+1/2} p_{i+1} - (a_{i+1/2}+a_{i-1/2}) p_i + a_{i-1/2} p_{i-1}) / dS^2
                 - (b_{i+1/2} p_{i+1} - b_{i-1/2} p_{i-1}) / (2*dS)

    where:
        a_{i+1/2} = (1/2) * (L[i]^2+L[i+1]^2)/2 * vj * ((S[i]+S[i+1])/2)^2
        b_{i+1/2} = (r-q) * (S[i]+S[i+1])/2

    Returns (lo, diag, hi) each of length N_S (boundary rows zeroed).
    """
    N_S = len(S)
    dS  = S[1] - S[0]

    # Face-centred diffusion and advection coefficients
    # Using arithmetic average of L^2 at faces
    L2 = L_col**2
    S_face = 0.5 * (S[:-1] + S[1:])           # (N_S-1,)
    L2_face = 0.5 * (L2[:-1] + L2[1:])        # (N_S-1,)
    a_face  = 0.5 * L2_face * vj * S_face**2  # diffusion at faces
    b_face  = (r - q) * S_face                 # advection at faces

    lo   = np.zeros(N_S)
    diag = np.zeros(N_S)
    hi   = np.zeros(N_S)

    # Interior nodes i = 1..N_S-2
    # lo[i]  = a_{i-1/2}/dS^2 + b_{i-1/2}/(2*dS)   coeff of p[i-1]
    # diag[i]= -(a_{i+1/2}+a_{i-1/2})/dS^2          coeff of p[i]
    # hi[i]  = a_{i+1/2}/dS^2 - b_{i+1/2}/(2*dS)   coeff of p[i+1]
    for i in range(1, N_S - 1):
        a_R = a_face[i]      # face i+1/2
        a_L = a_face[i - 1]  # face i-1/2
        b_R = b_face[i]
        b_L = b_face[i - 1]
        lo[i]   = a_L / dS**2 + b_L / (2.0 * dS)
        diag[i] = -(a_R + a_L) / dS**2
        hi[i]   = a_R / dS**2 - b_R / (2.0 * dS)

    # Boundary rows zeroed (absorbing BC: p=0 at S_min, S_max)
    lo[0] = diag[0] = hi[0] = 0.0
    lo[-1] = diag[-1] = hi[-1] = 0.0

    return lo, diag, hi


# ---------------------------------------------------------------------------
# FP operator coefficients in v
# ---------------------------------------------------------------------------

def _fp_v_coefficients(
    v: np.ndarray,
    kappa: float,
    theta_v: float,
    xi: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Conservative FD coefficients for the v-part of the FP operator:

        F_v p = (1/2) xi^2 d^2/dv^2 [v p]  -  kappa d/dv [(theta_v-v) p]

    Using conservative form with face-centred averages.
    Returns (lo, diag, hi) each of length N_v+1 (boundary rows zeroed).
    """
    N = len(v)
    dv = v[1] - v[0]

    v_face = 0.5 * (v[:-1] + v[1:])           # (N-1,)
    a_face = 0.5 * xi**2 * v_face             # diffusion at faces
    b_face = -kappa * (theta_v - v_face)       # advection (note sign: d/dv [kappa(th-v)p] = -d/dv[-kappa(th-v)p])

    lo   = np.zeros(N)
    diag = np.zeros(N)
    hi   = np.zeros(N)

    for j in range(1, N - 1):
        a_R = a_face[j]
        a_L = a_face[j - 1]
        b_R = b_face[j]
        b_L = b_face[j - 1]
        lo[j]   = a_L / dv**2 + b_L / (2.0 * dv)
        diag[j] = -(a_R + a_L) / dv**2
        hi[j]   = a_R / dv**2 - b_R / (2.0 * dv)

    # Boundary rows zeroed (handled externally)
    lo[0] = diag[0] = hi[0] = 0.0
    lo[-1] = diag[-1] = hi[-1] = 0.0

    return lo, diag, hi


# ---------------------------------------------------------------------------
# Cross derivative (explicit)
# ---------------------------------------------------------------------------

def _fp_cross(
    p: np.ndarray,
    S: np.ndarray,
    v: np.ndarray,
    L: np.ndarray,
    rho: float,
    xi: float,
) -> np.ndarray:
    """
    Compute the cross term: -rho xi d^2/dSdv [L v S p]

    Uses 2nd-order central differences on interior points.
    Returns array of shape (N_S, N_v+1) with zero boundaries.
    """
    N_S = len(S)
    N_v = len(v) - 1
    dS  = S[1] - S[0]
    dv  = v[1] - v[0]

    # Build f[i,j] = L[i] * v[j] * S[i] * p[i,j]
    LS = L * S                          # shape (N_S,)
    f  = LS[:, None] * v[None, :] * p  # shape (N_S, N_v+1)

    out = np.zeros_like(p)
    # d^2f/dSdv via central differences
    df_dSdv = np.zeros_like(p)
    df_dSdv[1:-1, 1:-1] = (
        f[2:, 2:] - f[:-2, 2:] - f[2:, :-2] + f[:-2, :-2]
    ) / (4.0 * dS * dv)

    out[1:-1, 1:-1] = -rho * xi * df_dSdv[1:-1, 1:-1]
    return out


# ---------------------------------------------------------------------------
# Public solver
# ---------------------------------------------------------------------------

def solve_fokker_planck(
    L: np.ndarray,
    grid: Grid,
    kappa: float,
    theta_v: float,
    xi: float,
    rho: float,
    v0: float,
    N_v: int = 40,
    v_max: float | None = None,
    theta_adi: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve the 2D Fokker-Planck PDE for the SLV joint density p(S, v, t).

    Parameters
    ----------
    L        : leverage surface, shape (N_S, N_T+1), L[i,n] = L(S[i], T[n])
    grid     : Grid instance (S0, K-axis used as S-axis, T-axis, rates)
    kappa    : mean-reversion speed
    theta_v  : long-run variance
    xi       : vol-of-vol
    rho      : correlation in [-1,1]
    v0       : initial variance
    N_v      : number of v intervals (N_v+1 nodes)
    v_max    : v upper bound; default = max(v0, theta_v) * 5
    theta_adi: ADI theta (0.5 = second-order)

    Returns
    -------
    p_S : marginal density in S, shape (N_S, N_T+1)
    E_v : conditional E[v|S,t],   shape (N_S, N_T+1)
    """
    S    = grid.K.copy()      # reuse K-axis as S-axis
    N_S  = len(S)
    N_T  = grid.N_T
    T    = grid.T

    if v_max is None:
        v_max = max(v0, theta_v) * 5.0
    v    = np.linspace(0.0, v_max, N_v + 1)
    dv   = v[1] - v[0]

    # Allocate output
    p_S_all = np.zeros((N_S, N_T + 1))
    E_v_all = np.zeros((N_S, N_T + 1))

    # ---- Initial condition: bivariate Gaussian at (S0, v0) ----------------
    dS   = S[1] - S[0]
    sig_S = dS
    sig_v = dv
    i0 = np.argmin(np.abs(S - grid.S0))
    j0 = np.argmin(np.abs(v - v0))

    p = np.zeros((N_S, N_v + 1))
    for i in range(N_S):
        for j in range(N_v + 1):
            p[i, j] = (
                np.exp(-0.5 * ((S[i] - grid.S0) / sig_S)**2)
                * np.exp(-0.5 * ((v[j] - v0) / sig_v)**2)
            )
    # Normalise
    mass = np.trapz(np.trapz(p, v, axis=1), S)
    if mass > 0:
        p /= mass

    # Enforce absorbing BCs on initial condition
    p[0,  :] = 0.0
    p[-1, :] = 0.0
    p[:, -1] = 0.0

    # Store t=0 marginals
    p_S_all[:, 0] = np.trapz(p, v, axis=1)
    denom         = np.where(p_S_all[:, 0] > 0, p_S_all[:, 0], 1.0)
    E_v_all[:, 0] = np.trapz(v[None, :] * p, v, axis=1) / denom

    # ---- Time march (forward in t) ----------------------------------------
    for n in range(N_T):
        t_cur  = T[n]
        t_next = T[n + 1]
        dt     = t_next - t_cur
        t_mid  = 0.5 * (t_cur + t_next)

        r = float(grid.r_val(t_mid))
        q = float(grid.q_val(t_mid))

        L_mid = 0.5 * (L[:, n] + L[:, n + 1])   # (N_S,)

        # FP v coefficients (full length N_v+1)
        lo_v, diag_v, hi_v = _fp_v_coefficients(v, kappa, theta_v, xi)

        # Explicit cross term
        cross = _fp_cross(p, S, v, L_mid, rho, xi)

        # ---- Step 1: S-implicit for each j ---------------------------------
        p_star = p.copy()

        for j in range(1, N_v):     # interior v nodes only
            vj = v[j]

            # F_v p (explicit)
            F_v_p = (lo_v[j] * p[:, j - 1]
                     + diag_v[j] * p[:, j]
                     + hi_v[j]   * p[:, j + 1])

            # Explicit RHS
            rhs = p[:, j] + dt * ((1.0 - theta_adi) * np.zeros(N_S)
                                  + F_v_p + cross[:, j])

            # F_S coefficients
            lo_s, diag_s, hi_s = _fp_S_coefficients(S, vj, L_mid, r, q)

            # (1-theta)*dt*F_S*p contribution to RHS
            F_S_p = _apply_banded(lo_s, diag_s, hi_s, p[:, j])
            rhs_int = rhs[1:-1] + (1.0 - theta_adi) * dt * F_S_p[1:-1]

            # LHS: (I - theta*dt*F_S) for interior S nodes
            n_int    = N_S - 2
            LHS_lo   = -theta_adi * dt * lo_s[1:-1]
            LHS_diag =  1.0 - theta_adi * dt * diag_s[1:-1]
            LHS_hi   = -theta_adi * dt * hi_s[1:-1]

            ab = _make_banded(LHS_lo, LHS_diag, LHS_hi)
            p_star[1:-1, j] = solve_banded((1, 1), ab, rhs_int)
            # Absorbing BCs
            p_star[0,  j] = 0.0
            p_star[-1, j] = 0.0

        # Boundary v nodes
        p_star[:, 0]  = p_star[:, 1]   # Neumann: dp/dv=0 at v=0 => p[:,0]=p[:,1]
        p_star[:, -1] = 0.0             # Dirichlet: p=0 at v=v_max
        p_star[0,  :] = 0.0
        p_star[-1, :] = 0.0

        # ---- Step 2: v-implicit correction for each S node -----------------
        p_new = p_star.copy()

        for i in range(1, N_S - 1):
            n_vint = N_v - 1
            vint   = v[1:-1]
            dv2    = dv**2

            diff_j = 0.5 * xi**2 * vint / dv2
            adv_j  = -kappa * (theta_v - vint) / (2.0 * dv)

            F_v_pstar = (diff_j - adv_j) * p_star[i, :-2] \
                      - 2.0 * diff_j     * p_star[i, 1:-1] \
                      + (diff_j + adv_j) * p_star[i, 2:]

            F_v_pold  = (diff_j - adv_j) * p[i, :-2] \
                      - 2.0 * diff_j     * p[i, 1:-1] \
                      + (diff_j + adv_j) * p[i, 2:]

            correction_rhs = theta_adi * dt * (F_v_pstar - F_v_pold)

            LHS_d  = 1.0 + 2.0 * theta_adi * dt * diff_j
            LHS_l  = -theta_adi * dt * (diff_j - adv_j)
            LHS_h  = -theta_adi * dt * (diff_j + adv_j)

            LHS_l_arr = np.full(n_vint, LHS_l) if np.ndim(LHS_l) == 0 else LHS_l
            LHS_h_arr = np.full(n_vint, LHS_h) if np.ndim(LHS_h) == 0 else LHS_h

            ab_v  = _make_banded(LHS_l_arr, LHS_d, LHS_h_arr)
            delta = solve_banded((1, 1), ab_v, correction_rhs)

            p_new[i, 1:-1] = p_star[i, 1:-1] + delta
            p_new[i, 0]    = p_new[i, 1]       # Neumann at v=0
            p_new[i, -1]   = 0.0               # Dirichlet at v=v_max

        p_new[0,  :] = 0.0
        p_new[-1, :] = 0.0
        p_new[:, -1] = 0.0
        p_new        = np.maximum(p_new, 0.0)   # density must be non-negative

        p = p_new

        # Store marginals at t_{n+1}
        p_S_all[:, n + 1] = np.trapz(p, v, axis=1)
        denom             = np.where(p_S_all[:, n + 1] > 0,
                                     p_S_all[:, n + 1], 1.0)
        E_v_all[:, n + 1] = np.trapz(v[None, :] * p, v, axis=1) / denom

    return p_S_all, E_v_all
