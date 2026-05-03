"""
slv_calibration.py
------------------
Calibration of the leverage function L(S, t) for the Stochastic Local
Volatility (SLV) model using the Fokker-Planck PDE matching approach.

Mathematical Background
-----------------------
Given market call prices -> local vol surface sigma_loc(S, t) via Dupire,
and Heston-like stochastic variance parameters (kappa, theta_v, xi, rho, v0),
the leverage function is recovered from the Dupire matching condition:

    L^2(S, t) = sigma_loc^2(S, t) / E[v | S(t) = S]

where E[v | S(t) = S] is computed by solving the 2D Fokker-Planck equation
for the joint density p(S, v, t) and marginalising:

    E[v | S] = integral v p(S,v,t) dv  /  integral p(S,v,t) dv

Algorithm (iterative fixed-point)
----------------------------------
1. Start with L^0(S,t) = sigma_loc(S,t) / sqrt(theta_v).
2. Outer loop until ||L^{k+1} - L^k||_inf < tol or k >= max_outer_iter:
   a. Solve the 2D FP PDE with current L to get E_v(S, t).
   b. Update: L^{k+1}^2(S,t) = sigma_loc^2(S,t) / E_v(S,t).
   c. Clip L to [L_min, L_max].
3. Return final L.

Public API
----------
calibrate_leverage(sigma_loc, grid, kappa, theta_v, xi, rho, v0,
                   N_v=40, v_max=None, L_bounds=(0.01, 5.0),
                   tol=1e-4, max_outer_iter=10, verbose=True)
    -> SLVCalibrationResult
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field

import numpy as np

from .grid import Grid
from .slv_fokker_planck import solve_fokker_planck


@dataclass
class SLVCalibrationResult:
    """Container for SLV leverage function calibration output."""
    L: np.ndarray               # Leverage surface, shape (N_S, N_T+1)
    sigma_loc: np.ndarray       # Input local vol surface, shape (N_S, N_T+1)
    E_v: np.ndarray             # Conditional E[v|S,t], shape (N_S, N_T+1)
    p_marginal: np.ndarray      # Marginal density p_S(S,t), shape (N_S, N_T+1)
    n_outer_iter: int           # Number of outer iterations performed
    converged: bool             # True if ||L^{k+1}-L^k||_inf < tol
    elapsed_s: float            # Total wall-clock time in seconds
    history: dict = field(default_factory=dict)
    """
    history['L_change'] : list[float]  -- ||L^{k+1}-L^k||_inf at each outer iter
    history['t_iter']   : list[float]  -- wall-clock seconds per outer iter
    """

    def __str__(self) -> str:
        return (
            f"SLVCalibrationResult:\n"
            f"  L range     = [{self.L.min():.4f}, {self.L.max():.4f}]\n"
            f"  E_v range   = [{self.E_v.min():.4f}, {self.E_v.max():.4f}]\n"
            f"  n_outer_iter = {self.n_outer_iter}\n"
            f"  converged   = {self.converged}\n"
            f"  elapsed     = {self.elapsed_s:.1f} s"
        )


def calibrate_leverage(
    sigma_loc: np.ndarray,
    grid: Grid,
    kappa: float,
    theta_v: float,
    xi: float,
    rho: float,
    v0: float,
    N_v: int = 40,
    v_max: float | None = None,
    L_bounds: tuple = (0.01, 5.0),
    tol: float = 1e-4,
    max_outer_iter: int = 10,
    verbose: bool = True,
) -> SLVCalibrationResult:
    """
    Calibrate the SLV leverage function L(S, t) via iterative Fokker-Planck
    matching.

    Parameters
    ----------
    sigma_loc      : local vol surface from LV calibration, shape (N_S, N_T+1)
                     where N_S = grid.N_K+1 (uses grid.K as the S-axis)
    grid           : Grid instance (S0, K/S-axis, T-axis, r, q)
    kappa          : Heston mean-reversion speed
    theta_v        : Heston long-run variance
    xi             : Heston vol-of-vol
    rho            : correlation in [-1, 1]
    v0             : initial variance (must be > 0)
    N_v            : number of v intervals in the Fokker-Planck grid
    v_max          : upper bound for v; default = max(v0, theta_v) * 5
    L_bounds       : (L_min, L_max) box constraints on L
    tol            : convergence tolerance for ||L^{k+1} - L^k||_inf
    max_outer_iter : maximum number of fixed-point iterations
    verbose        : print iteration log

    Returns
    -------
    SLVCalibrationResult
    """
    L_min, L_max = L_bounds
    sigma_loc = np.asarray(sigma_loc, dtype=float)
    sigma2_loc = sigma_loc**2

    N_S  = sigma_loc.shape[0]   # should equal grid.N_K + 1
    N_T1 = grid.N_T + 1

    # ---- Initialise L -------------------------------------------------------
    # L^0 = sigma_loc / sqrt(theta_v)  (flat-vol start: L^2 * theta_v = sigma_loc^2)
    L = np.clip(sigma_loc / np.sqrt(max(theta_v, 1e-10)), L_min, L_max)

    history: dict = {"L_change": [], "t_iter": []}
    t_start = time.perf_counter()

    if verbose:
        print(f"Starting SLV leverage calibration:")
        print(f"  N_S={N_S}, N_T={grid.N_T}, N_v={N_v}")
        print(f"  kappa={kappa}, theta_v={theta_v}, xi={xi}, rho={rho}, v0={v0}")
        print(f"  L bounds: [{L_min}, {L_max}], tol={tol:.1e}, "
              f"max_outer_iter={max_outer_iter}")
        print(f"  {'iter':>5}  {'||ΔL||∞':>12}  {'t_iter':>8}")
        print(f"  {'-'*5}  {'-'*12}  {'-'*8}")

    p_S_final = None
    E_v_final = None
    converged = False
    k = 0

    for k in range(max_outer_iter):
        t_iter_start = time.perf_counter()

        # Solve Fokker-Planck with current L
        p_S, E_v = solve_fokker_planck(
            L=L,
            grid=grid,
            kappa=kappa,
            theta_v=theta_v,
            xi=xi,
            rho=rho,
            v0=v0,
            N_v=N_v,
            v_max=v_max,
            theta_adi=0.5,
        )

        # Update L
        # Clip E_v from below to avoid division by zero
        E_v_safe = np.maximum(E_v, 1e-8)
        L2_new   = sigma2_loc / E_v_safe
        L_new    = np.clip(np.sqrt(np.maximum(L2_new, 0.0)), L_min, L_max)

        # Check convergence
        L_change = float(np.max(np.abs(L_new - L)))
        t_iter_k = time.perf_counter() - t_iter_start

        history["L_change"].append(L_change)
        history["t_iter"].append(t_iter_k)

        if verbose:
            print(f"  {k+1:5d}  {L_change:12.4e}  {t_iter_k:8.2f}s")

        p_S_final = p_S
        E_v_final = E_v
        L         = L_new

        if L_change < tol:
            converged = True
            break

    elapsed = time.perf_counter() - t_start

    if verbose:
        status = "CONVERGED" if converged else "MAX ITER REACHED"
        print(f"\nSLV calibration {status} in {elapsed:.1f}s after {k+1} iteration(s).")
        print(f"  Final ||ΔL||∞ = {history['L_change'][-1]:.4e}")
        print(f"  L range: [{L.min():.4f}, {L.max():.4f}]")

    # Use last computed p_S / E_v (from the last iteration)
    if p_S_final is None:
        # Edge case: max_outer_iter = 0
        p_S_final = np.zeros((N_S, N_T1))
        E_v_final = np.full((N_S, N_T1), theta_v)

    return SLVCalibrationResult(
        L=L,
        sigma_loc=sigma_loc,
        E_v=E_v_final,
        p_marginal=p_S_final,
        n_outer_iter=k + 1,
        converged=converged,
        elapsed_s=elapsed,
        history=history,
    )
