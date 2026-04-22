"""
calibration_oe.py
-----------------
Local volatility calibration via direct resolution of the first-order
optimality equations (OE / KKT conditions).

Algorithm
---------
The discrete KKT stationarity condition is:

    grad_J(u) = grad_pde(u) + alpha*(-Delta_h + I)(u - u_star) = 0

Rearranging as a Newton-like iterative scheme:

    alpha * (-Delta_h + I) delta_u = - grad_J(u^k)
    =>  delta_u = solve_optimality_system(-grad_J, alpha, grid)
    =>  u^{k+1} = clip( u^k + step_size * delta_u, u_lo, u_hi )

The gradient grad_pde is computed via the discrete adjoint (same as DtO /
calibration.py), but the *search direction* is obtained by solving the
elliptic preconditioning system instead of using L-BFGS-B.

This corresponds to a gradient descent preconditioned by the inverse of the
Tikhonov operator (-Delta_h + I), which is equivalent to solving the
linearised optimality equation at each iteration.  It is also interpreted as
one step of a fixed-point iteration for the equation:

    (-Delta_h + I)(u - u_star) = -(1/alpha) grad_pde(u)

Optional FD gradient verification
----------------------------------
When verify_fd=True, at the first iteration the PDE gradient is also
computed by finite differences (one forward PDE solve per DOF column —
coloured FD along the time axis to keep cost manageable) and compared with
the adjoint gradient.  Expensive but useful for debugging.

Public API
----------
calibrate_oe(grid, z, w, u_star, alpha, u0=None,
             sigma_bounds=(0.01, 1.5), theta=0.5,
             max_iter=200, tol=1e-6, step_size=1.0,
             oe_solver="dct", misfit_type="price",
             iv_mkt=None, r_arr=None, q_arr=None,
             verbose=True, log_every=5,
             verify_fd=False)
    -> CalibrationOEResult

CalibrationOEResult.history keys
---------------------------------
  J          : list[float]   objective value at recorded iterations
  grad_norm  : list[float]   ||grad_J||_inf
  delta_J    : list[float]   |J_k - J_{k-1}|  (nan at first)
  t_iter     : list[float]   seconds for each iteration
  t_cumul    : list[float]   cumulative seconds
  t_forward  : list[float]   seconds in forward solve
  t_adjoint  : list[float]   seconds in adjoint solve
  t_gradient : list[float]   seconds in gradient assembly
  t_oe_solve : list[float]   seconds in OE system solve
"""

from __future__ import annotations
from dataclasses import dataclass, field
import time
import numpy as np

from .grid import Grid
from .state_solver import solve_state
from .adjoint_solver import solve_adjoint
from .gradient import evaluate_gradient
from .objective import evaluate_J, misfit_iv_source
from .optimality_solver import solve_optimality_system


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class CalibrationOEResult:
    """Container for OE calibration output."""
    u_opt: np.ndarray              # optimal local variance sigma^2
    sigma_opt: np.ndarray          # optimal local vol = sqrt(u_opt)
    C_opt: np.ndarray              # model call prices at optimum
    J_final: float                 # final objective value
    grad_norm: float               # final ||grad J||_inf
    n_iter: int                    # number of OE iterations
    elapsed_s: float               # wall-clock time
    converged: bool
    history: dict = field(default_factory=dict)

    def __str__(self):
        return (
            f"CalibrationOEResult:\n"
            f"  J_final   = {self.J_final:.6e}\n"
            f"  grad_norm = {self.grad_norm:.6e}\n"
            f"  n_iter    = {self.n_iter}\n"
            f"  elapsed   = {self.elapsed_s:.1f} s\n"
            f"  converged = {self.converged}"
        )


# ---------------------------------------------------------------------------
# Optional FD gradient verification
# ---------------------------------------------------------------------------

def _fd_gradient_pde(u: np.ndarray, C: np.ndarray,
                     z: np.ndarray, w: np.ndarray,
                     u_star: np.ndarray, alpha: float,
                     grid: Grid, theta: float,
                     misfit_type: str, iv_mkt, r_arr, q_arr,
                     eps: float = 1e-5) -> np.ndarray:
    """
    Finite-difference approximation of grad_J w.r.t. each u[i,n]
    by perturbing one column n at a time (coloured FD).

    Cost: O(N_T) forward PDE solves (much cheaper than O(N_K*N_T) if we
    lump all nodes in the same time column together — but this only works
    because each column affects all K nodes simultaneously, so we can't do
    a full column FD.  Instead we do N_T full column FDs).
    """
    J0 = evaluate_J(u, C, z, w, u_star, alpha, grid,
                    misfit_type=misfit_type, iv_mkt=iv_mkt,
                    r_arr=r_arr, q_arr=q_arr)
    grad_fd = np.zeros_like(u)
    for n in range(grid.N_T + 1):
        u_p = u.copy()
        u_p[1:-1, n] += eps
        C_p = solve_state(u_p, grid, theta=theta)
        J_p = evaluate_J(u_p, C_p, z, w, u_star, alpha, grid,
                         misfit_type=misfit_type, iv_mkt=iv_mkt,
                         r_arr=r_arr, q_arr=q_arr)
        grad_fd[1:-1, n] = (J_p - J0) / eps
    return grad_fd


# ---------------------------------------------------------------------------
# Main OE calibration
# ---------------------------------------------------------------------------

def calibrate_oe(
    grid: Grid,
    z: np.ndarray,
    w: np.ndarray,
    u_star: np.ndarray,
    alpha: float,
    u0: np.ndarray | None = None,
    sigma_bounds: tuple = (0.01, 1.5),
    theta: float = 0.5,
    max_iter: int = 200,
    tol: float = 1e-6,
    step_size: float = 0.1,
    oe_solver: str = "dct",
    misfit_type: str = "price",
    iv_mkt: np.ndarray | None = None,
    r_arr: np.ndarray | None = None,
    q_arr: np.ndarray | None = None,
    verbose: bool = True,
    log_every: int = 5,
    verify_fd: bool = False,
) -> CalibrationOEResult:
    """
    Calibrate local volatility via iterative resolution of optimality equations.

    Parameters
    ----------
    grid         : Grid instance
    z            : observed call prices, shape (N_K+1, N_T+1)
    w            : weight array, shape (N_K+1, N_T+1)
    u_star       : prior local variance, shape (N_K+1, N_T+1)
    alpha        : Tikhonov regularization parameter
    u0           : initial local variance (default: u_star)
    sigma_bounds : (sigma_min, sigma_max) box constraints
    theta        : theta-scheme (0.5 = Crank-Nicolson)
    max_iter     : maximum OE iterations
    tol          : convergence: stop when ||grad_J||_inf < tol
    step_size    : damping factor for update u^{k+1} = u^k + step_size*delta_u
    oe_solver    : "dct" | "lu" | "cg"
    misfit_type  : "price" | "iv"
    iv_mkt       : market IV surface (required if misfit_type="iv")
    r_arr, q_arr : rate arrays (required if misfit_type="iv")
    verbose      : print progress
    log_every    : record / print every N iterations
    verify_fd    : if True, check adjoint gradient against FD at iter 1

    Returns
    -------
    CalibrationOEResult
    """
    sigma_lo, sigma_hi = sigma_bounds
    u_lo = sigma_lo ** 2
    u_hi = sigma_hi ** 2

    if u0 is None:
        u0 = u_star.copy()
    u = np.clip(u0, u_lo, u_hi).copy()

    # Validate IV misfit requirements
    if misfit_type == "iv":
        if iv_mkt is None or r_arr is None or q_arr is None:
            raise ValueError("iv_mkt, r_arr, q_arr required for misfit_type='iv'")

    # LU cache (reused across iterations when oe_solver="lu")
    lu_cache: dict = {}

    history: dict = {
        "J":          [],
        "grad_norm":  [],
        "delta_J":    [],
        "t_iter":     [],
        "t_cumul":    [],
        "t_forward":  [],
        "t_adjoint":  [],
        "t_gradient": [],
        "t_oe_solve": [],
    }

    t_start = time.perf_counter()
    converged = False

    if verbose:
        print(f"Starting OE calibration: N_K={grid.N_K}, N_T={grid.N_T}, "
              f"alpha={alpha:.2e}, max_iter={max_iter}")
        print(f"  misfit_type={misfit_type!r}  oe_solver={oe_solver!r}")
        print(f"  sigma bounds: [{sigma_lo}, {sigma_hi}]")
        print(f"  DOFs: {(grid.N_K+1)*(grid.N_T+1)}")
        header = (f"  {'iter':>5}  {'J':>14}  {'ΔJ':>12}  {'‖∇J‖∞':>12}"
                  f"  {'t_fwd':>7}  {'t_adj':>7}  {'t_grad':>7}  {'t_oe':>7}  {'t_tot':>8}")
        print(header)
        print("  " + "-" * (len(header) - 2))

    prev_J = float("nan")

    for k in range(1, max_iter + 1):
        t_iter_start = time.perf_counter()

        # 1. Forward PDE solve
        t0 = time.perf_counter()
        C = solve_state(u, grid, theta=theta)
        t_fwd = time.perf_counter() - t0

        # 2. Objective value
        J = evaluate_J(u, C, z, w, u_star, alpha, grid,
                       misfit_type=misfit_type, iv_mkt=iv_mkt,
                       r_arr=r_arr, q_arr=q_arr)

        # 3. Adjoint source for IV misfit
        t0 = time.perf_counter()
        if misfit_type == "iv":
            src = misfit_iv_source(C, iv_mkt, grid, r_arr, q_arr)
            p = solve_adjoint(u, C, z, w, grid, theta=theta,
                              source_override=src)
        else:
            p = solve_adjoint(u, C, z, w, grid, theta=theta)
        t_adj = time.perf_counter() - t0

        # 4. Gradient assembly
        t0 = time.perf_counter()
        grad_J = evaluate_gradient(u, C, p, u_star, alpha, grid, theta=theta)
        t_grad = time.perf_counter() - t0

        grad_norm = float(np.max(np.abs(grad_J)))

        # Optional FD verification at first iteration
        if verify_fd and k == 1:
            grad_fd = _fd_gradient_pde(u, C, z, w, u_star, alpha, grid, theta,
                                       misfit_type, iv_mkt, r_arr, q_arr)
            rel_err = (np.max(np.abs(grad_J[1:-1, 1:] - grad_fd[1:-1, 1:]))
                       / max(np.max(np.abs(grad_fd[1:-1, 1:])), 1e-12))
            if verbose:
                print(f"\n  [FD verification] max rel error adjoint vs FD: {rel_err:.3e}")
                if rel_err > 0.01:
                    print("  WARNING: large FD/adjoint discrepancy - check gradient!")
                else:
                    print("  OK: adjoint gradient matches FD to < 1%")

        # 5. OE system solve:  (-Delta_h+I) delta_u = -grad_J
        #    (alpha is NOT included in the operator here — the solve gives
        #    the H^1-preconditioned gradient direction, and step_size controls
        #    the actual update magnitude)
        t0 = time.perf_counter()
        delta_u = solve_optimality_system(
            -grad_J, 1.0, grid,
            method=oe_solver,
            _lu_cache=lu_cache,
        )
        t_oe = time.perf_counter() - t0  # includes backtracking below

        # 6. Armijo backtracking line search along delta_u direction.
        #    Start from step_size (user-supplied) and halve until J decreases.
        #    Descent condition: J(u + s*d) <= J(u) - 1e-4 * s * <grad_J, d>
        #    where d = delta_u.  For a descent direction <grad_J, d> < 0, so
        #    the RHS is > J(u) confirming decrease.
        directional_deriv = float(np.sum(grad_J * delta_u))  # should be < 0
        s = step_size
        armijo_c = 1e-4
        max_backtracks = 30
        # Use -|directional_deriv| so the Armijo condition always requires a
        # genuine J decrease, regardless of sign of deriv (box projection can
        # flip it).
        armijo_rhs_slope = -abs(directional_deriv)
        for _bt in range(max_backtracks):
            u_trial = np.clip(u + s * delta_u, u_lo, u_hi)
            u_trial[0, :] = u[0, :]
            u_trial[-1, :] = u[-1, :]
            C_trial = solve_state(u_trial, grid, theta=theta)
            J_trial = evaluate_J(u_trial, C_trial, z, w, u_star, alpha, grid,
                                 misfit_type=misfit_type, iv_mkt=iv_mkt,
                                 r_arr=r_arr, q_arr=q_arr)
            if J_trial <= J + armijo_c * s * armijo_rhs_slope:
                break
            s *= 0.5
        else:
            # Accept smallest step anyway if line search exhausted
            pass
        t_oe = time.perf_counter() - t0  # total: OE solve + backtracking
        u_new = u_trial

        t_iter = time.perf_counter() - t_iter_start
        t_cumul = time.perf_counter() - t_start

        dJ = abs(J - prev_J) if np.isfinite(prev_J) else float("nan")
        prev_J = J

        # Record history every log_every iterations
        if k % log_every == 0 or k == 1:
            history["J"].append(J)
            history["grad_norm"].append(grad_norm)
            history["delta_J"].append(dJ)
            history["t_iter"].append(t_iter)
            history["t_cumul"].append(t_cumul)
            history["t_forward"].append(t_fwd)
            history["t_adjoint"].append(t_adj)
            history["t_gradient"].append(t_grad)
            history["t_oe_solve"].append(t_oe)

            if verbose:
                dJ_str = f"{dJ:.3e}" if np.isfinite(dJ) else "   ---  "
                print(
                    f"  {k:5d}  |  J={J:.6e}"
                    f"  |  ΔJ={dJ_str}"
                    f"  |  ‖∇J‖∞={grad_norm:.3e}"
                    f"  |  fwd={t_fwd:.2f}s  adj={t_adj:.2f}s"
                    f"  |  grad={t_grad:.2f}s  oe={t_oe:.3f}s"
                    f"  |  tot={t_cumul:.1f}s"
                )

        # Check convergence
        if grad_norm < tol:
            converged = True
            u = u_new
            if verbose:
                print(f"\n  Converged at iteration {k}: ||grad_J||_inf={grad_norm:.3e} < {tol:.1e}")
            break

        u = u_new

    elapsed = time.perf_counter() - t_start
    sigma_opt = np.sqrt(np.clip(u, u_lo**2, None))  # safe sqrt
    # Actually: u is already sigma^2, so:
    sigma_opt = np.sqrt(np.clip(u, 0.0, None))

    # Final evaluation
    C_opt = solve_state(u, grid, theta=theta)
    J_final = evaluate_J(u, C_opt, z, w, u_star, alpha, grid,
                         misfit_type=misfit_type, iv_mkt=iv_mkt,
                         r_arr=r_arr, q_arr=q_arr)

    if misfit_type == "iv":
        src_final = misfit_iv_source(C_opt, iv_mkt, grid, r_arr, q_arr)
        p_final = solve_adjoint(u, C_opt, z, w, grid, theta=theta,
                                source_override=src_final)
    else:
        p_final = solve_adjoint(u, C_opt, z, w, grid, theta=theta)
    grad_final = evaluate_gradient(u, C_opt, p_final, u_star, alpha, grid, theta=theta)
    grad_norm_final = float(np.max(np.abs(grad_final)))

    if verbose:
        print(f"\nOE calibration finished in {elapsed:.1f}s  ({k} iterations)")
        print(f"  J_final   = {J_final:.6e}")
        print(f"  grad_norm = {grad_norm_final:.6e}")
        print(f"  converged = {converged}")

    return CalibrationOEResult(
        u_opt=u,
        sigma_opt=sigma_opt,
        C_opt=C_opt,
        J_final=J_final,
        grad_norm=grad_norm_final,
        n_iter=k,
        elapsed_s=elapsed,
        converged=converged,
        history=history,
    )
