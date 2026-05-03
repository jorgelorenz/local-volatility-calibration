"""
fem_backward_solver.py
----------------------
Finite-Element Method (FEM) solver for the local-volatility backward PDE
(European option pricing with a given local volatility surface).

PDE (backward pricing equation)
--------------------------------
  -dV/dt = (1/2) sigma^2(S,t) S^2 d^2V/dS^2  +  (r(t)-q(t)) S dV/dS  -  r(t) V

  for (S, t) in Omega_back = (S_min, S_max) x (0, T_final)

  Terminal condition : V(S, T_final) = payoff(S)   (e.g. max(S - K, 0))
  Left  BC           : V(S_min, t) = V_left(t)      (Dirichlet)
  Right BC           : V(S_max, t) = V_right(t)      (Dirichlet)

where sigma(S,t) is the local volatility (input as a 2D array on the FEM
node / time grid).

This is the same PDE as in the Dupire forward equation but posed backward
in time.  We march from T_final -> 0 by making the substitution tau = T-t,
which flips the sign of the time derivative and turns the problem into a
forward-in-tau Dupire-like PDE:

  dV/dtau = (1/2) sigma^2(S, T-tau) S^2 d^2V/dS^2
           + (r(T-tau)-q(T-tau)) S dV/dS  -  r(T-tau) V

FEM Formulation
---------------
Identical P1 (hat-function) FEM in S to fem_state_solver.py, with the
reaction term changed from  q*V  to  r*V  (risk-free rate, not dividend).

Time stepping is backward-Euler / Crank-Nicolson in tau (forward in tau,
backward in physical time t).

Public API
----------
solve_fem_backward(sigma, nodes_S, t_grid, T_final, r_func, q_func,
                   payoff, bc_left_func, bc_right_func, theta=0.5)
    -> V  shape (N_nodes, N_t)

    sigma       : local volatility surface, shape (N_nodes, N_t).
                  sigma[i, n] = sigma(nodes_S[i], t_grid[n]).
    nodes_S     : 1D node array for the S-axis (asset price), shape (N_nodes,)
    t_grid      : time array (physical time, increasing), shape (N_t,)
                  t_grid[0] = 0, t_grid[-1] = T_final.
    T_final     : maturity / horizon
    r_func      : callable r(t) -> float, or scalar float
    q_func      : callable q(t) -> float, or scalar float
    payoff      : callable payoff(S) -> array of shape (N_nodes,), or array
    bc_left_func  : callable bc_left(t) -> float (value at S_min)
    bc_right_func : callable bc_right(t) -> float (value at S_max)
    theta       : implicitness (0.5 = CN, 1.0 = backward Euler)

    Returns V[i, n] = option value at (nodes_S[i], t_grid[n]).
    V[:, -1] = payoff at T_final.
    V[:, 0]  = option value at t=0.

Convenience wrapper
-------------------
solve_fem_backward_grid(sigma2, grid, K_strike, theta=0.5, nodes=None)
    -> V  shape (N_nodes, N_T+1)

FEM adjoint solver
------------------
solve_fem_adjoint(u, C, z, w, grid, theta=0.5, nodes=None, source_override=None)
    -> p  shape (N_nodes, N_T+1)

    Uses a Grid instance (same as FD solver) for compatibility.
    sigma2 : local variance (sigma^2), shape (N_nodes, N_T+1)
    K_strike : strike for European call payoff
    nodes    : FEM node array; None => grid.K

Notes on boundary conditions for European call
----------------------------------------------
  Left  BC: V(S_min, t) = 0               (deep OTM)
  Right BC: V(S_max, t) = S_max*e^{-q*tau} - K*e^{-r*tau}   (deep ITM)

These match the standard Black-Scholes Dirichlet BCs used in the FD solver.
"""

from __future__ import annotations
import numpy as np
from scipy.linalg import solve_banded

from .grid import Grid
from .fem_state_solver import _element_matrices, _tri_to_banded, _apply_tri, _assemble


# ---------------------------------------------------------------------------
# Public solver (general interface)
# ---------------------------------------------------------------------------

def solve_fem_backward(
    sigma: np.ndarray,
    nodes_S: np.ndarray,
    t_grid: np.ndarray,
    T_final: float,
    r_func,
    q_func,
    payoff,
    bc_left_func,
    bc_right_func,
    theta: float = 0.5,
) -> np.ndarray:
    """
    Solve the local-vol backward PDE using P1 FEM in S, theta-scheme in time.

    Time is stored as physical time t increasing from 0 to T_final.
    The solver marches backward: it starts at T_final (terminal condition)
    and steps toward t=0.

    Parameters
    ----------
    sigma         : local volatility, shape (N_nodes, N_t)
    nodes_S       : asset-price nodes, shape (N_nodes,)
    t_grid        : physical time grid, shape (N_t,), t_grid[-1] = T_final
    T_final       : maturity (should equal t_grid[-1])
    r_func        : r(t) -> float  or scalar
    q_func        : q(t) -> float  or scalar
    payoff        : callable(S_array) -> array or pre-computed array (N_nodes,)
    bc_left_func  : callable(t) -> float
    bc_right_func : callable(t) -> float
    theta         : 0.5=CN, 1.0=BE

    Returns
    -------
    V : shape (N_nodes, N_t), V[:, -1] = terminal payoff, V[:, 0] = t=0 prices
    """
    N_nodes = len(nodes_S)
    N_t     = len(t_grid)

    # Rate accessors
    def r_val(t):
        return float(r_func(t)) if callable(r_func) else float(r_func)
    def q_val(t):
        return float(q_func(t)) if callable(q_func) else float(q_func)

    # Allocate
    V = np.zeros((N_nodes, N_t))

    # Terminal condition
    if callable(payoff):
        V[:, -1] = payoff(nodes_S)
    else:
        V[:, -1] = np.asarray(payoff)
    V[0,  -1] = bc_left_func(T_final)
    V[-1, -1] = bc_right_func(T_final)

    # March backward: step from time index n+1 to n  (n = N_t-2 down to 0)
    for n in range(N_t - 2, -1, -1):
        t_cur  = t_grid[n]
        t_next = t_grid[n + 1]
        dt     = t_next - t_cur
        t_mid  = 0.5 * (t_cur + t_next)

        r = r_val(t_mid)
        q = q_val(t_mid)

        # For the backward PDE the reaction is r (not q).
        # We reuse _element_matrices from fem_state_solver but pass r as q
        # (since the reaction term is -r*V here, matching q*V in the forward PDE
        # with the substitution q->r).
        # The sign of the advection coefficient (r-q)*S dV/dS is the same.

        # Average local variance at each node (sigma^2)
        sigma2_nodes = 0.5 * (sigma[:, n]**2 + sigma[:, n + 1]**2)

        # Assemble the backward BS bilinear form using divergence-form rewrite.
        #
        # The symmetric FEM stiffness integral ∫ ½σ²S² V_S φ_S dS satisfies:
        #   (A_diff V)_i = ∫ ½σ²S² V_S φ_i' dS
        #                = -∫ d(½σ²S² V_S)/dS φ_i dS   [IBP, Dirichlet BCs => no boundary]
        #                = -∫ [½σ²S² V_SS + σ²S V_S] φ_i dS
        #                = -M(½σ²S² V_SS) - M(σ²S V_S)
        #
        # So:  M(½σ²S² V_SS) = -A_diff V - M(σ²S V_S)
        #
        # The backward PDE operator is:
        #   L_bck V = ½σ²S² V_SS + (r-q)S V_S - r V
        #           = [-A_diff V/M - σ²S V_S] + (r-q)S V_S - r V
        #
        # For M dV/dτ = -A V (scheme), we need:
        #   A V = -M(L_bck V) = A_diff V + M(σ²S V_S) - M((r-q)S V_S) + M(r V)
        #       = A_diff V + M((σ²-(r-q))S V_S) + M(r V)
        #
        # The advection correction is c_adv = (u_mid - (r-q)) * K_mid, i.e.:
        #   adv_correction_sign = +1,  q_adv = -(r-q)
        #   => c_adv = (u_mid - (r-q)) * K_mid  ✓
        M, A = _assemble(nodes_S, sigma2_nodes, r, q, q_reac=r, q_adv=-(r - q),
                         adv_correction_sign=+1.0)

        LHS_full = M + theta * dt * A
        RHS_mat  = M - (1.0 - theta) * dt * A

        # BCs at current (t_cur) and next (t_next) time
        bc_l_cur  = bc_left_func(t_cur)
        bc_r_cur  = bc_right_func(t_cur)
        bc_l_next = bc_left_func(t_next)
        bc_r_next = bc_right_func(t_next)

        # Interior values at t_next (already computed)
        V_int_next = V[1:-1, n + 1]

        # RHS = RHS_mat_int * V_int_next - LHS_bc_contributions + RHS_bc_contributions
        rhs = _apply_tri(RHS_mat[1:-1, 1:-1], V_int_next)

        lhs_left_col  = LHS_full[1:-1, 0]
        lhs_right_col = LHS_full[1:-1, -1]
        rhs_left_col  = RHS_mat[1:-1, 0]
        rhs_right_col = RHS_mat[1:-1, -1]

        rhs -= lhs_left_col  * bc_l_cur
        rhs -= lhs_right_col * bc_r_cur
        rhs += rhs_left_col  * bc_l_next
        rhs += rhs_right_col * bc_r_next

        LHS_int = LHS_full[1:-1, 1:-1]
        ab = _tri_to_banded(LHS_int)
        V_int_cur = solve_banded((1, 1), ab, rhs)

        V[0,  n] = bc_l_cur
        V[-1, n] = bc_r_cur
        V[1:-1, n] = V_int_cur

    return V


# ---------------------------------------------------------------------------
# Convenience wrapper using Grid (compatible with FD solver interface)
# ---------------------------------------------------------------------------

def solve_fem_backward_grid(
    sigma2: np.ndarray,
    grid: Grid,
    K_strike: float,
    theta: float = 0.5,
    nodes: np.ndarray | None = None,
) -> np.ndarray:
    """
    Solve the backward local-vol PDE for a European call with strike K_strike.

    Uses a Grid instance for rates and time grid (compatible with existing
    infrastructure).

    Parameters
    ----------
    sigma2    : local variance sigma^2(S,t), shape (N_nodes, N_T+1)
    grid      : Grid instance (S0, T, r, q used for BCs and rates)
    K_strike  : option strike
    theta     : implicitness parameter
    nodes     : FEM node array for S-axis; None => grid.K

    Returns
    -------
    V : shape (N_nodes, N_T+1), V[:, -1] = payoff, V[:, 0] = t=0 prices
    """
    if nodes is None:
        nodes = grid.K.copy()

    T_final = grid.T_max
    t_grid  = grid.T   # physical time from 0 to T_max
    r_arr, q_arr = grid.rate_arrays()

    def r_func(t):
        return grid.r_val(t)

    def q_func(t):
        return grid.q_val(t)

    def payoff(S):
        return np.maximum(S - K_strike, 0.0)

    def bc_left_func(t):
        # Deep OTM call at S_min: value ≈ 0
        return 0.0

    def bc_right_func(t):
        # Deep ITM call at S_max: intrinsic discounted
        # tau = remaining time to maturity
        tau = T_final - t
        Bq = grid.discount_q(tau)
        Br = grid.discount_r(tau)
        return nodes[-1] * Bq - K_strike * Br

    # sigma2 is local variance; backward solver expects sigma (vol), not variance
    sigma_surf = np.sqrt(np.maximum(sigma2, 0.0))

    return solve_fem_backward(
        sigma=sigma_surf,
        nodes_S=nodes,
        t_grid=t_grid,
        T_final=T_final,
        r_func=r_func,
        q_func=q_func,
        payoff=payoff,
        bc_left_func=bc_left_func,
        bc_right_func=bc_right_func,
        theta=theta,
    )


# ---------------------------------------------------------------------------
# FEM adjoint solver (for use in FEM-based calibration)
# ---------------------------------------------------------------------------

def solve_fem_adjoint(
    u: np.ndarray,
    C: np.ndarray,
    z: np.ndarray,
    w: np.ndarray,
    grid: Grid,
    theta: float = 0.5,
    nodes: np.ndarray | None = None,
    source_override: np.ndarray | None = None,
) -> np.ndarray:
    """
    Solve the FEM discrete adjoint equation backward in maturity T.

    The adjoint equation is the transpose of the FEM state system.  For each
    interior time step k the system is:

        (M + theta*dT*A^{k-1})^T p^k_int
            = (M - (1-theta)*dT*A^k)^T p^{k+1}_int  -  dJ/dC^k_int

    where  dJ/dC^k_int = dK_eff * dT * w_int[:,k] * (C_int[:,k] - z_int[:,k])
    and dK_eff is the effective FEM node spacing (grid.dK when nodes=grid.K).

    Since M and A are symmetric tridiagonal for P1 FEM, the transpose equals
    the original matrix, so we solve the same systems as the forward state
    solver but driven by the misfit source.

    Parameters
    ----------
    u               : local variance sigma^2, shape (N_nodes, N_T+1)
    C               : FEM state solution (call prices), shape (N_nodes, N_T+1)
    z               : observed call prices, shape (N_nodes, N_T+1); NaN -> 0
    w               : weight array, shape (N_nodes, N_T+1); NaN -> 0
    grid            : Grid instance (T, dT, rates)
    theta           : theta-scheme (must match state solver)
    nodes           : FEM node array; None => grid.K
    source_override : optional raw source override (same shape as C),
                      replaces w*(C-z) if provided.  dK*dT factor is applied
                      internally.

    Returns
    -------
    p : adjoint variable, shape (N_nodes, N_T+1), p=0 at boundary nodes
    """
    if nodes is None:
        nodes = grid.K.copy()

    N_nodes = len(nodes)
    N_T     = grid.N_T
    T       = grid.T
    dT      = grid.dT

    # Effective integration weight in K direction: mean spacing around each node
    # For uniform mesh this equals dK; for non-uniform we use trapezoidal weights.
    h = np.zeros(N_nodes)
    h[0]    = 0.5 * (nodes[1] - nodes[0])
    h[-1]   = 0.5 * (nodes[-1] - nodes[-2])
    h[1:-1] = 0.5 * (nodes[2:] - nodes[:-2])

    # Build source array: dJ/dC^k = h_i * dT * w_i,k * (C_i,k - z_i,k)
    if source_override is not None:
        raw = np.where(~np.isnan(source_override), source_override, 0.0)
    else:
        diff  = C - z
        valid = ~np.isnan(diff) & ~np.isnan(w)
        raw   = np.where(valid, w * diff, 0.0)

    # source[i, k] = h[i] * dT * raw[i, k]
    source = (h[:, None] * dT) * raw   # shape (N_nodes, N_T+1)

    p = np.zeros((N_nodes, N_T + 1))

    def _get_matrices(n: int):
        """Assemble (M, A) for time step n -> n+1."""
        T_mid = 0.5 * (T[n] + T[n + 1])
        r = float(grid.r_val(T_mid))
        q = float(grid.q_val(T_mid))
        u_nodes = 0.5 * (u[:, n] + u[:, n + 1])
        M, A = _assemble(nodes, u_nodes, r, q)
        return M, A

    # ---- Terminal step: k = N_T ----
    # (M + theta*dT*A^{N_T-1})^T p^{N_T}_int = -dJ/dC^{N_T}_int
    # Since M and A are symmetric, ^T = identity here.
    M, A = _get_matrices(N_T - 1)
    LHS_full = M + theta * dT * A
    LHS_int  = LHS_full[1:-1, 1:-1]
    rhs      = -source[1:-1, N_T]
    ab       = _tri_to_banded(LHS_int)
    p[1:-1, N_T] = solve_banded((1, 1), ab, rhs)

    # ---- Internal backward sweep: k = N_T-1, ..., 1 ----
    for k in range(N_T - 1, 0, -1):
        # LHS uses step k-1 -> k
        M_km1, A_km1 = _get_matrices(k - 1)
        LHS_full = M_km1 + theta * dT * A_km1
        LHS_int  = LHS_full[1:-1, 1:-1]
        ab       = _tri_to_banded(LHS_int)

        # RHS contribution: (M - (1-theta)*dT*A^k)^T * p^{k+1}_int
        # = (M - (1-theta)*dT*A^k) * p^{k+1}_int  (symmetric matrices)
        M_k, A_k   = _get_matrices(k)
        RHS_mat    = M_k - (1.0 - theta) * dT * A_k
        p_next_int = p[1:-1, k + 1]

        # Interior-interior block times p_next (BC corrections from columns 0,-1 are zero
        # because adjoint BCs are homogeneous Dirichlet)
        rhs = _apply_tri(RHS_mat[1:-1, 1:-1], p_next_int) - source[1:-1, k]

        p[1:-1, k] = solve_banded((1, 1), ab, rhs)
        # Boundary nodes remain 0 (homogeneous Dirichlet for adjoint)

    return p
