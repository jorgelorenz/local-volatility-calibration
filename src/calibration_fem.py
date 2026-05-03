"""
calibration_fem.py
------------------
Local volatility calibration using the FEM (P1) state and adjoint solvers.

This module mirrors calibration.py but replaces the finite-difference PDE
solvers with the FEM backends from fem_state_solver.py and
fem_backward_solver.py.  The calibration.py file is NOT modified.

Algorithm
---------
Same Tikhonov-regularized L-BFGS-B loop as calibration.py:

  J_alpha(u) = misfit(F_FEM(u), data)  +  (alpha/2)||u - u*||^2_{H^1}

At each iteration:
  1. Solve FEM state PDE  ->  C = F_FEM(u)
  2. Evaluate J
  3. Solve FEM adjoint PDE  ->  p
  4. Compute FEM gradient  ->  grad_J

Public API
----------
calibrate_fem(grid, z, w, u_star, alpha, u0=None,
              sigma_bounds=(0.01, 2.0), theta=0.5,
              nodes=None, use_fenics=False,
              ftol=1e-12, gtol=1e-8, max_iter=500,
              verbose=True, log_every=5)
    -> CalibrationResult
"""

from __future__ import annotations
import time
import numpy as np
from scipy.optimize import minimize

from .grid import Grid
from .calibration import CalibrationResult          # reuse same dataclass
from .fem_state_solver import solve_fem_state
from .fem_backward_solver import solve_fem_adjoint
from .gradient import evaluate_fem_gradient
from .objective import evaluate_J


def calibrate_fem(
    grid: Grid,
    z: np.ndarray,
    w: np.ndarray,
    u_star: np.ndarray,
    alpha: float,
    u0: np.ndarray | None = None,
    sigma_bounds: tuple = (0.01, 2.0),
    theta: float = 0.5,
    nodes: np.ndarray | None = None,
    use_fenics: bool = False,
    ftol: float = 1e-12,
    gtol: float = 1e-8,
    max_iter: int = 500,
    verbose: bool = True,
    log_every: int = 5,
) -> CalibrationResult:
    """
    Calibrate local volatility surface via PDE-constrained inverse problem
    using P1 FEM state and adjoint solvers.

    Parameters
    ----------
    grid         : Grid instance
    z            : observed call prices, shape (N_nodes, N_T+1)
    w            : weight array, shape (N_nodes, N_T+1)
    u_star       : prior local variance, shape (N_nodes, N_T+1)
    alpha        : Tikhonov regularization parameter
    u0           : initial local variance (default: u_star)
    sigma_bounds : (sigma_min, sigma_max) box constraints on sqrt(u)
    theta        : theta-scheme (0.5 = Crank-Nicolson)
    nodes        : FEM node array along K-axis; None => grid.K (uniform)
    use_fenics   : if True, attempt to use FEniCS backends
    ftol, gtol   : L-BFGS-B convergence tolerances
    max_iter     : maximum optimizer iterations
    verbose      : print iteration log to stdout
    log_every    : log every this many iterations

    Returns
    -------
    CalibrationResult  (same dataclass as calibration.calibrate)
    """
    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    if nodes is None:
        nodes = grid.K.copy()

    N_nodes = len(nodes)
    N_T1    = grid.N_T + 1
    n_dof   = N_nodes * N_T1

    sigma_lo, sigma_hi = sigma_bounds
    u_lo = sigma_lo ** 2
    u_hi = sigma_hi ** 2

    if u0 is None:
        u0 = u_star.copy()
    u0 = np.clip(u0, u_lo, u_hi)

    # Optionally use FEniCS
    _state_solver  = solve_fem_state
    _adjoint_solver = solve_fem_adjoint

    if use_fenics:
        try:
            from . import _FENICS_AVAILABLE
            if _FENICS_AVAILABLE:
                from .fem_state_solver_fenics import solve_fem_state_fenics
                from .fem_backward_solver_fenics import solve_fem_backward_grid_fenics
                _state_solver = solve_fem_state_fenics
                # Note: FEniCS adjoint wrapper not yet implemented; fall back.
                if verbose:
                    print("  [FEniCS] State solver available; adjoint uses manual FEM.")
        except Exception:
            if verbose:
                print("  [FEniCS] Not available; using manual FEM backends.")

    # History
    history: dict = {
        "J": [], "grad_norm": [], "delta_J": [],
        "t_iter": [], "t_cumul": [],
        "t_forward": [], "t_adjoint": [], "t_gradient": [],
    }

    t_start          = time.perf_counter()
    iter_count       = [0]
    feval_count      = [0]
    last_J           = [float("nan")]
    last_grad_norm   = [float("nan")]
    last_iter_start  = [t_start]
    last_t_fwd       = [0.0]
    last_t_adj       = [0.0]
    last_t_grad      = [0.0]

    # ------------------------------------------------------------------
    # Objective + gradient (FEM)
    # ------------------------------------------------------------------
    def fg(u_vec: np.ndarray):
        u = u_vec.reshape(N_nodes, N_T1)
        u = np.clip(u, u_lo, u_hi)

        # Forward FEM solve
        t0 = time.perf_counter()
        C  = _state_solver(u, grid, theta=theta, nodes=nodes)
        last_t_fwd[0] = time.perf_counter() - t0

        # Interpolate C back to the uniform grid for evaluate_J if nodes != grid.K
        if not np.allclose(nodes, grid.K):
            from scipy.interpolate import interp1d
            C_uniform = np.zeros((grid.N_K + 1, N_T1))
            for n_t in range(N_T1):
                f = interp1d(nodes, C[:, n_t], kind='linear',
                             bounds_error=False, fill_value=(C[0, n_t], C[-1, n_t]))
                C_uniform[:, n_t] = f(grid.K)
            C_eval = C_uniform
            z_eval = z  # z should be on grid.K
        else:
            C_eval = C
            z_eval = z

        # Objective value (price misfit + Tikhonov)
        # evaluate_J expects arrays on grid.K; if nodes differ we use C_eval
        J = evaluate_J(u if np.allclose(nodes, grid.K) else
                       _interp_u_to_grid(u, nodes, grid),
                       C_eval, z_eval, w, u_star, alpha, grid,
                       misfit_type="price")

        # Adjoint FEM solve
        t0 = time.perf_counter()
        p  = _adjoint_solver(u, C, z_eval if np.allclose(nodes, grid.K)
                             else _interp_z_to_nodes(z, nodes, grid),
                             w if np.allclose(nodes, grid.K)
                             else _interp_z_to_nodes(w, nodes, grid),
                             grid, theta=theta, nodes=nodes)
        last_t_adj[0] = time.perf_counter() - t0

        # FEM gradient
        t0 = time.perf_counter()
        g  = evaluate_fem_gradient(u, C, p, u_star, alpha, grid,
                                   theta=theta, nodes=nodes)
        last_t_grad[0] = time.perf_counter() - t0

        feval_count[0] += 1
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
        print(f"Starting FEM calibration (L-BFGS-B): N_nodes={N_nodes}, "
              f"N_T={grid.N_T}, alpha={alpha:.2e}, max_iter={max_iter}")
        print(f"  sigma bounds: [{sigma_lo}, {sigma_hi}]")
        print(f"  DOFs: {n_dof}")
        print(f"  {'iter':>6}  {'J':>14}  {'ΔJ':>12}  {'‖∇J‖∞':>12}"
              f"  {'fwd':>7}  {'adj':>7}  {'grad':>7}  {'t_total':>9}")
        print(f"  {'-'*6}  {'-'*14}  {'-'*12}  {'-'*12}"
              f"  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*9}")

    result = minimize(
        fg,
        u0.ravel(),
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        callback=callback,
        options={"maxiter": max_iter, "ftol": ftol, "gtol": gtol, "maxls": 50},
    )

    elapsed = time.perf_counter() - t_start

    u_opt     = result.x.reshape(N_nodes, N_T1)
    u_opt     = np.clip(u_opt, u_lo, u_hi)
    sigma_opt = np.sqrt(u_opt)

    C_opt    = _state_solver(u_opt, grid, theta=theta, nodes=nodes)
    _, g_fin = fg(u_opt.ravel())
    grad_norm = float(np.max(np.abs(g_fin.reshape(N_nodes, N_T1))))

    if verbose:
        print(f"\nFEM calibration finished in {elapsed:.1f}s")
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


# ---------------------------------------------------------------------------
# Helpers for non-uniform mesh interpolation
# ---------------------------------------------------------------------------

def _interp_u_to_grid(u: np.ndarray, nodes: np.ndarray, grid: Grid) -> np.ndarray:
    """Interpolate u from custom nodes to grid.K."""
    from scipy.interpolate import interp1d
    N_T1  = grid.N_T + 1
    u_out = np.zeros((grid.N_K + 1, N_T1))
    for n_t in range(N_T1):
        f = interp1d(nodes, u[:, n_t], kind='linear',
                     bounds_error=False,
                     fill_value=(u[0, n_t], u[-1, n_t]))
        u_out[:, n_t] = f(grid.K)
    return u_out


def _interp_z_to_nodes(z: np.ndarray, nodes: np.ndarray, grid: Grid) -> np.ndarray:
    """Interpolate z/w from grid.K to custom FEM nodes."""
    from scipy.interpolate import interp1d
    N_T1  = grid.N_T + 1
    z_out = np.zeros((len(nodes), N_T1))
    for n_t in range(N_T1):
        col = z[:, n_t]
        valid = ~np.isnan(col)
        if np.any(valid):
            f = interp1d(grid.K[valid], col[valid], kind='linear',
                         bounds_error=False, fill_value=np.nan)
            z_out[:, n_t] = f(nodes)
        else:
            z_out[:, n_t] = np.nan
    return z_out
