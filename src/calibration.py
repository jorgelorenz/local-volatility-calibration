"""
calibration.py
--------------
Main entry point for local volatility calibration (DtO / L-BFGS-B method).

Algorithm
---------
Minimizes the Tikhonov-regularized objective:

  J_alpha(u) = misfit(F(u), data)  +  (alpha/2)||u - u*||^2_{H^1}

using L-BFGS-B with box constraints u_L <= u <= u_U.

Two misfit types are supported (controlled by misfit_type parameter):
  "price"  : (1/2) dK*dT * sum w*(C - z)^2  (default, vega-weighted price error)
  "iv"     : (1/2) dK*dT * sum (IV_model - IV_mkt)^2  (direct IV error, like calibrate_v2)

At each iteration:
  1. Solve state PDE  ->  C = F(u)
  2. Evaluate J
  3. Build adjoint source (depends on misfit_type)
  4. Solve adjoint PDE  ->  p
  5. Compute gradient  ->  grad_J

The optimizer (scipy L-BFGS-B) uses (J, grad_J) to update u.

Stopping criteria
-----------------
  - ||grad_J||_inf < gtol
  - |Delta J| / max(|J|, 1) < ftol
  - Iteration count >= max_iter

Public API
----------
calibrate(grid, z, w, u_star, alpha, u0=None,
          sigma_bounds=(0.01, 2.0), theta=0.5,
          ftol=1e-12, gtol=1e-8, max_iter=500,
          misfit_type="price",
          iv_mkt=None, r_arr=None, q_arr=None,
          verbose=True, log_every=5)
    -> CalibrationResult

CalibrationResult.history keys
-------------------------------
  J          : list[float]  -- objective value at each recorded iteration
  grad_norm  : list[float]  -- ||grad_J||_inf at each recorded iteration
  delta_J    : list[float]  -- absolute change in J  (nan for first entry)
  t_iter     : list[float]  -- wall-clock seconds for each optimiser iteration
  t_cumul    : list[float]  -- cumulative wall-clock seconds
  t_forward  : list[float]  -- seconds in forward PDE solve
  t_adjoint  : list[float]  -- seconds in adjoint solve
  t_gradient : list[float]  -- seconds in gradient assembly
"""

from __future__ import annotations
from dataclasses import dataclass, field
import time
import numpy as np
from scipy.optimize import minimize

from .grid import Grid
from .state_solver import solve_state
from .adjoint_solver import solve_adjoint
from .objective import evaluate_J, misfit_iv_source
from .gradient import evaluate_gradient


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class CalibrationResult:
    """Container for calibration output."""
    u_opt: np.ndarray              # optimal local variance sigma^2, shape (N_K+1, N_T+1)
    sigma_opt: np.ndarray          # optimal local vol = sqrt(u_opt)
    C_opt: np.ndarray              # model call prices at optimum
    J_final: float                 # final objective value
    grad_norm: float               # final ||grad J||_inf
    n_iter: int                    # number of optimizer iterations
    n_fevals: int                  # number of function evaluations
    success: bool
    message: str
    elapsed_s: float
    history: dict = field(default_factory=dict)
    """
    history contains per-iteration records (recorded every log_every iters):
      J          : list[float]   objective value
      grad_norm  : list[float]   ||grad_J||_inf
      delta_J    : list[float]   |J_k - J_{k-1}|  (nan at first entry)
      t_iter     : list[float]   seconds for this iteration
      t_cumul    : list[float]   cumulative seconds
      t_forward  : list[float]   seconds in forward PDE solve
      t_adjoint  : list[float]   seconds in adjoint solve
      t_gradient : list[float]   seconds in gradient assembly
    """

    def __str__(self):
        return (
            f"CalibrationResult:\n"
            f"  J_final   = {self.J_final:.6e}\n"
            f"  grad_norm = {self.grad_norm:.6e}\n"
            f"  n_iter    = {self.n_iter}\n"
            f"  n_fevals  = {self.n_fevals}\n"
            f"  elapsed   = {self.elapsed_s:.1f} s\n"
            f"  success   = {self.success}\n"
            f"  message   = {self.message}"
        )


# ---------------------------------------------------------------------------
# Main calibration routine
# ---------------------------------------------------------------------------

def calibrate(
    grid: Grid,
    z: np.ndarray,
    w: np.ndarray,
    u_star: np.ndarray,
    alpha: float,
    u0: np.ndarray | None = None,
    sigma_bounds: tuple = (0.01, 2.0),
    theta: float = 0.5,
    ftol: float = 1e-12,
    gtol: float = 1e-8,
    max_iter: int = 500,
    misfit_type: str = "price",
    iv_mkt: np.ndarray | None = None,
    r_arr: np.ndarray | None = None,
    q_arr: np.ndarray | None = None,
    verbose: bool = True,
    log_every: int = 5,
) -> CalibrationResult:
    """
    Calibrate local volatility surface via PDE-constrained inverse problem (L-BFGS-B).

    Parameters
    ----------
    grid         : Grid instance (K, T axes, rates, dividends)
    z            : observed call prices, shape (N_K+1, N_T+1)
                   (NaN entries are ignored in the misfit)
    w            : weight array, shape (N_K+1, N_T+1)
    u_star       : prior local variance sigma*^2, shape (N_K+1, N_T+1)
    alpha        : Tikhonov regularization parameter
    u0           : initial local variance (default: u_star)
    sigma_bounds : (sigma_min, sigma_max) for box constraints on sqrt(u)
    theta        : theta-scheme (0.5 = Crank-Nicolson)
    ftol, gtol   : convergence tolerances for L-BFGS-B
    max_iter     : maximum optimizer iterations
    misfit_type  : "price" (default) or "iv" — type of data misfit
    iv_mkt       : market IV surface (required if misfit_type="iv")
    r_arr, q_arr : rate arrays (required if misfit_type="iv")
    verbose      : print iteration progress to stdout
    log_every    : print / record history every this many iterations

    Returns
    -------
    CalibrationResult
    """
    N_K1 = grid.N_K + 1
    N_T1 = grid.N_T + 1
    n_dof = N_K1 * N_T1

    sigma_lo, sigma_hi = sigma_bounds
    u_lo = sigma_lo ** 2
    u_hi = sigma_hi ** 2

    if u0 is None:
        u0 = u_star.copy()
    u0 = np.clip(u0, u_lo, u_hi)

    if misfit_type == "iv":
        if iv_mkt is None or r_arr is None or q_arr is None:
            raise ValueError("iv_mkt, r_arr, q_arr required for misfit_type='iv'")

    # History tracking
    history: dict = {
        "J":          [],
        "grad_norm":  [],
        "delta_J":    [],
        "t_iter":     [],
        "t_cumul":    [],
        "t_forward":  [],
        "t_adjoint":  [],
        "t_gradient": [],
    }

    t_start     = time.perf_counter()
    iter_count  = [0]
    feval_count = [0]

    # We cache the last known J and grad_norm so the callback can access them
    # without re-evaluating the (expensive) objective.
    last_J          = [float("nan")]
    last_grad_norm  = [float("nan")]
    last_iter_start = [t_start]
    last_t_fwd      = [0.0]
    last_t_adj      = [0.0]
    last_t_grad     = [0.0]

    # ------------------------------------------------------------------
    # Objective + gradient callback
    # ------------------------------------------------------------------
    def fg(u_vec: np.ndarray):
        u = u_vec.reshape(N_K1, N_T1)
        u = np.clip(u, u_lo, u_hi)

        t0 = time.perf_counter()
        C = solve_state(u, grid, theta=theta)
        last_t_fwd[0] = time.perf_counter() - t0

        J = evaluate_J(u, C, z, w, u_star, alpha, grid,
                       misfit_type=misfit_type, iv_mkt=iv_mkt,
                       r_arr=r_arr, q_arr=q_arr)

        t0 = time.perf_counter()
        if misfit_type == "iv":
            src = misfit_iv_source(C, iv_mkt, grid, r_arr, q_arr)
            p = solve_adjoint(u, C, z, w, grid, theta=theta,
                              source_override=src)
        else:
            p = solve_adjoint(u, C, z, w, grid, theta=theta)
        last_t_adj[0] = time.perf_counter() - t0

        t0 = time.perf_counter()
        g = evaluate_gradient(u, C, p, u_star, alpha, grid, theta=theta)
        last_t_grad[0] = time.perf_counter() - t0

        feval_count[0] += 1

        # Update cached values for the callback
        last_J[0]         = float(J)
        last_grad_norm[0] = float(np.max(np.abs(g)))

        return float(J), g.ravel()

    def callback(u_vec):
        iter_count[0] += 1
        k = iter_count[0]

        now       = time.perf_counter()
        t_iter_k  = now - last_iter_start[0]
        t_cumul_k = now - t_start
        last_iter_start[0] = now

        J_k    = last_J[0]
        gn_k   = last_grad_norm[0]
        prev_J = history["J"][-1] if history["J"] else float("nan")
        dJ     = abs(J_k - prev_J) if not np.isnan(prev_J) else float("nan")

        if k % log_every == 0 or k == 1:
            # Record in history
            history["J"].append(J_k)
            history["grad_norm"].append(gn_k)
            history["delta_J"].append(dJ)
            history["t_iter"].append(t_iter_k)
            history["t_cumul"].append(t_cumul_k)
            history["t_forward"].append(last_t_fwd[0])
            history["t_adjoint"].append(last_t_adj[0])
            history["t_gradient"].append(last_t_grad[0])

            if verbose:
                dJ_str = f"{dJ:.3e}" if not np.isnan(dJ) else "   ---  "
                print(
                    f"  iter {k:4d}"
                    f"  |  J={J_k:.6e}"
                    f"  |  \u0394J={dJ_str}"
                    f"  |  \u2016\u2207J\u2016\u221e={gn_k:.3e}"
                    f"  |  fwd={last_t_fwd[0]:.2f}s"
                    f"  |  adj={last_t_adj[0]:.2f}s"
                    f"  |  grad={last_t_grad[0]:.2f}s"
                    f"  |  t_total={t_cumul_k:.1f}s"
                )

    # ------------------------------------------------------------------
    # Run L-BFGS-B
    # ------------------------------------------------------------------
    bounds = [(u_lo, u_hi)] * n_dof

    if verbose:
        print(f"Starting calibration (DtO/L-BFGS-B): N_K={grid.N_K}, N_T={grid.N_T}, "
              f"alpha={alpha:.2e}, max_iter={max_iter}")
        print(f"  misfit_type={misfit_type!r}")
        print(f"  sigma bounds: [{sigma_lo}, {sigma_hi}]")
        print(f"  DOFs: {n_dof}")
        print(f"  {'iter':>6}  {'J':>14}  {'ΔJ':>12}  {'‖∇J‖∞':>12}"
              f"  {'fwd':>7}  {'adj':>7}  {'grad':>7}  {'t_total':>9}")
        print(f"  {'-'*6}  {'-'*14}  {'-'*12}  {'-'*12}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*9}")

    result = minimize(
        fg,
        u0.ravel(),
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        callback=callback,
        options={
            "maxiter": max_iter,
            "ftol":    ftol,
            "gtol":    gtol,
            "maxls":   50,
        },
    )

    elapsed = time.perf_counter() - t_start

    u_opt     = result.x.reshape(N_K1, N_T1)
    u_opt     = np.clip(u_opt, u_lo, u_hi)
    sigma_opt = np.sqrt(u_opt)

    C_opt     = solve_state(u_opt, grid, theta=theta)
    _, g_final = fg(u_opt.ravel())
    grad_norm  = float(np.max(np.abs(g_final.reshape(N_K1, N_T1))))

    if verbose:
        print(f"\nCalibration finished in {elapsed:.1f}s")
        print(f"  J_final   = {result.fun:.6e}")
        print(f"  grad_norm = {grad_norm:.6e}")
        print(f"  n_iter    = {result.nit}")
        print(f"  n_fevals  = {feval_count[0]}")
        print(f"  success   = {result.success}")
        print(f"  message   = {result.message}")

    return CalibrationResult(
        u_opt=u_opt,
        sigma_opt=sigma_opt,
        C_opt=C_opt,
        J_final=float(result.fun),
        grad_norm=grad_norm,
        n_iter=result.nit,
        n_fevals=feval_count[0],
        success=result.success,
        message=str(result.message),
        elapsed_s=elapsed,
        history=history,
    )
