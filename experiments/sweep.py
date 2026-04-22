"""
experiments/sweep.py
--------------------
Incremental parameter-sweep calibration runner.

For each combination of (method, N_K, N_T, alpha) the script:

  1. Calibrates local vol using the selected method and coarse PDE grid.
  2. Re-prices vanillas on a HIGH-RESOLUTION grid (fine) to get reliable IVs.
  3. Writes a text log file to experiments/logs/.
  4. Saves an IV-comparison plot to experiments/plots/.
  5. Appends a row to experiments/summary.csv.

Supported methods
-----------------
  dto-price   : DtO (L-BFGS-B) with price misfit  [default]
  dto-iv      : DtO (L-BFGS-B) with IV misfit
  oe-price    : OE (optimality-equation) with price misfit
  oe-iv       : OE (optimality-equation) with IV misfit
  calib-v2    : Reference implementation (SimulatorEngine.calibrate_v2)

Experiments are INCREMENTAL: if a log file for a given parameter combination
already exists it is skipped (unless --force is passed), so you can interrupt
and resume without re-doing finished runs.

Usage
-----
    # Full sweep on default asset (index 0)
    python experiments/sweep.py

    # Quick sanity check (small grid, few iterations)
    python experiments/sweep.py --quick

    # Specific methods only
    python experiments/sweep.py --methods dto-price oe-price

    # Specific asset, force re-run all
    python experiments/sweep.py --asset-index 1 --force

    # Custom sweep: override alpha list only
    python experiments/sweep.py --alpha 1e-4 5e-4 1e-3

    # Increase fine-grid resolution beyond config default
    python experiments/sweep.py --fine-nk 400 --fine-nt 300

CLI arguments
-------------
--asset-index INT     Asset index in UnRisk JSON (default: config.DEFAULT_ASSET_INDEX)
--methods STR ...     Methods to sweep (default: dto-price dto-iv oe-price oe-iv calib-v2)
--quick               Use tiny grid for fast sanity-check (ignores sweep lists)
--force               Re-run experiments even if log already exists
--alpha FLOAT ...     Override SWEEP_ALPHA_VALUES
--nk INT ...          Override SWEEP_N_K_VALUES
--nt INT ...          Override SWEEP_N_T_VALUES (must match --nk length)
--max-iter INT ...    Override SWEEP_MAX_ITER_LIST (single value applied to all)
--fine-nk INT         Override GRID_FINE_N_K for this run
--fine-nt INT         Override GRID_FINE_N_T for this run
--out-dir PATH        Base output directory (default: experiments/)
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

# Ensure stdout/stderr can handle Unicode (needed on Windows cp1252 consoles)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, RectBivariateSpline

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "help" / "other_calibration_process"))

from src import config as cfg
from src.grid import Grid
from src.market_data import iv_surface_to_call_prices, vega_weights
from src.calibration import calibrate
from src.calibration_oe import calibrate_oe
from src.utils import implied_vol_brentq
from src.diagnostics import plot_convergence, plot_iv_comparison, print_iv_metrics

from unrisk_adapter import read_implied_for_engine, list_equity_assets

ALL_METHODS = ["dto-price", "dto-iv", "oe-price", "oe-iv", "calib-v2"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _effective_rate(curve_arr: np.ndarray, t_val: float) -> float:
    """Linearly interpolate a (years, rate) curve at t_val."""
    ts = curve_arr[:, 0]
    rs = curve_arr[:, 1]
    fn = interp1d(ts, rs, kind="linear", bounds_error=False,
                  fill_value=(float(rs[0]), float(rs[-1])))
    return float(fn(t_val))


def _iv_from_price_safe(S0, K, t, r_eff, q_eff, price):
    """Wrapper that never raises."""
    try:
        from scipy.optimize import brentq
        from scipy.special import ndtr
        t = max(float(t), 1e-12)
        if t < 5.0 / 365.0:
            return np.nan
        K = max(float(K), 1e-12)
        def bs(sig):
            sig = max(sig, 1e-9)
            d1 = (np.log(max(S0, 1e-12) / K) + (r_eff - q_eff + 0.5 * sig * sig) * t) / (sig * np.sqrt(t))
            d2 = d1 - sig * np.sqrt(t)
            return S0 * np.exp(-q_eff * t) * ndtr(d1) - K * np.exp(-r_eff * t) * ndtr(d2)
        df_r = np.exp(-r_eff * t)
        df_q = np.exp(-q_eff * t)
        intrinsic = max(S0 * df_q - K * df_r, 0.0)
        upper = S0 * df_q
        target = float(np.clip(price, intrinsic + 1e-14, upper - 1e-14))
        if target - intrinsic < 1e-5:
            return np.nan
        if bs(1e-6) * bs(5.0) > 0.0:
            return np.nan
        return float(brentq(lambda s: bs(s) - target, 1e-6, 5.0, xtol=1e-12, maxiter=200))
    except Exception:
        return np.nan


def backproject_iv(
    C_opt: np.ndarray,      # (N_K+1, N_T+1)  model call prices (fine grid)
    grid_fine: Grid,
    tenors: np.ndarray,     # market tenors
    strikes: np.ndarray,    # market strikes
    r_curve: np.ndarray,
    q_curve: np.ndarray,
    S0: float,
) -> np.ndarray:
    """
    Interpolate fine-grid model prices onto market (tenor, strike) grid,
    then invert to IVs.

    Returns iv_model: shape (n_T, n_K), NaN where inversion fails.
    """
    # Bilinear spline over PDE grid (skip T=0 column)
    C_spline = RectBivariateSpline(grid_fine.K, grid_fine.T[1:], C_opt[:, 1:], kx=1, ky=1)

    n_t, n_k = len(tenors), len(strikes)
    iv_model = np.full((n_t, n_k), np.nan)

    for i, t in enumerate(tenors):
        r_eff = _effective_rate(r_curve, t)
        q_eff = _effective_rate(q_curve, t)
        for j, k in enumerate(strikes):
            price = float(C_spline(k, t))
            iv_model[i, j] = _iv_from_price_safe(S0, k, t, r_eff, q_eff, price)

    return iv_model


# ---------------------------------------------------------------------------
# Log writer
# ---------------------------------------------------------------------------

def write_log(
    log_path: Path,
    params: dict,
    calib_result,
    iv_metrics: dict,
    fine_grid_params: dict,
    elapsed_validation: float,
) -> None:
    hist = getattr(calib_result, "history", {})
    lines = [
        "=" * 70,
        f"  Experiment log  --  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        "",
        "  Parameters",
        "  ----------",
        f"    Method          : {params.get('method', 'dto-price')}",
        f"    Asset index     : {params['asset_index']}",
        f"    Asset name      : {params['asset_name']}",
        f"    N_K (calib)     : {params['N_K']}",
        f"    N_T (calib)     : {params['N_T']}",
        f"    alpha           : {params['alpha']:.2e}",
        f"    max_iter        : {params['max_iter']}",
        f"    theta           : {params['theta']}",
        f"    sigma_bounds    : {params['sigma_bounds']}",
        "",
        "  Calibration results",
        "  -------------------",
        f"    J_final         : {calib_result.J_final:.6e}",
        f"    grad_norm       : {calib_result.grad_norm:.6e}",
        f"    n_iter          : {calib_result.n_iter}",
    ]

    # DtO-specific fields
    if hasattr(calib_result, "n_fevals"):
        lines.append(f"    n_fevals        : {calib_result.n_fevals}")
        lines.append(f"    success         : {calib_result.success}")
        lines.append(f"    message         : {calib_result.message}")
    if hasattr(calib_result, "converged"):
        lines.append(f"    converged       : {calib_result.converged}")

    lines += [
        f"    elapsed_calib   : {calib_result.elapsed_s:.2f} s",
        "",
        "  J convergence history (every recorded step)",
        "  -------------------------------------------",
    ]

    for idx, (j_val, gn, dj, ti, tc) in enumerate(zip(
        hist.get("J", []),
        hist.get("grad_norm", []),
        hist.get("delta_J", []),
        hist.get("t_iter", []),
        hist.get("t_cumul", []),
    )):
        dj_str = f"{dj:.3e}" if np.isfinite(dj) else "     ---"
        lines.append(
            f"    rec {idx+1:4d}  J={j_val:.6e}  dJ={dj_str}"
            f"  grad={gn:.3e}  t_iter={ti:.2f}s  t_cumul={tc:.1f}s"
        )

    # Sub-timings if present
    if hist.get("t_forward"):
        lines += ["", "  Sub-step timings (mean over recorded steps)", "  -------------------------------------------"]
        for key in ["t_forward", "t_adjoint", "t_gradient", "t_oe_solve"]:
            vals = hist.get(key)
            if vals:
                lines.append(f"    {key:<20}: mean={np.mean(vals):.3f}s  max={np.max(vals):.3f}s")

    lines += [
        "",
        "  Validation (fine grid)",
        "  ----------------------",
        f"    Fine N_K        : {fine_grid_params['N_K']}",
        f"    Fine N_T        : {fine_grid_params['N_T']}",
        f"    elapsed_valid   : {elapsed_validation:.2f} s",
        f"    IV MAE          : {iv_metrics['MAE']:.6f}",
        f"    IV RMSE         : {iv_metrics['RMSE']:.6f}",
        f"    IV max error    : {iv_metrics['max_err']:.6f}",
        f"    n valid points  : {iv_metrics['n']}",
        "",
        "=" * 70,
    ]

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Log written: {log_path}")


# ---------------------------------------------------------------------------
# CSV summary
# ---------------------------------------------------------------------------

def append_csv_row(csv_path: Path, row: dict) -> None:
    fieldnames = [
        "timestamp", "method", "asset_index", "asset_name",
        "NK_calib", "NT_calib", "alpha", "max_iter",
        "J_final", "grad_norm", "n_iter", "n_fevals",
        "IV_MAE", "IV_RMSE", "IV_max_err",
        "t_calibration_s", "t_validation_s",
        "success", "fine_NK", "fine_NT",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


# ---------------------------------------------------------------------------
# calibrate_v2 wrapper
# ---------------------------------------------------------------------------

def _run_calib_v2(
    S0, tenors, strikes, iv_market, r_curve, q_curve,
    N_K, N_T, alpha, max_iter, sigma_bounds, theta,
):
    """
    Run SimulatorEngine.calibrate_v2 and return a duck-typed result object
    with the same interface as CalibrationResult / CalibrationOEResult.
    """
    from SimulatorEngine import DupireSimulator

    sim = DupireSimulator(
        S0=S0,
        r_zero=r_curve,
        q_zero=q_curve,
    )

    t0 = time.perf_counter()
    sim.calibrate_v2(
        vol_matrices=[iv_market],
        strikes_list=[strikes],
        tenors_list=[tenors],
        r_curve=[r_curve],
        q_curve=[q_curve],
        alpha=alpha,
        max_iter=max_iter,
        nK_pde=N_K,
        nT_pde=N_T,
        pde_theta=theta,
        verbose=False,
    )
    elapsed = time.perf_counter() - t0

    # Extract calibrated local vol surface on (T, K) grid
    lv_interp = sim.vol_interpolators_[0]

    # Build a synthetic result object
    class _V2Result:
        pass

    res = _V2Result()
    res.elapsed_s = elapsed
    res.J_final = float("nan")
    res.grad_norm = float("nan")
    res.n_iter = max_iter
    res.n_fevals = max_iter
    res.success = True
    res.message = "calibrate_v2"
    res.converged = True
    res.history = {"J": [], "grad_norm": [], "delta_J": [], "t_iter": [], "t_cumul": []}
    res._lv_interp = lv_interp

    return res


def _build_u_fine_from_lv_interp(lv_interp, grid_fine, sigma_bounds):
    """Evaluate calibrate_v2 local vol interpolator on the fine grid.

    calibrate_v2 stores a spline interpolator whose closure expects a single
    argument of shape (n, 2) with columns [T, K] — NOT two separate scalars.
    Calling lv_interp(T, K) passes two positional args to a one-param function,
    raises TypeError, and the except silently fills every cell with the fallback
    constant.  Fix: pass np.array([[T, K]]) and index the scalar result.
    """
    sigma_lo, sigma_hi = sigma_bounds
    u_fine = np.zeros((grid_fine.N_K + 1, grid_fine.N_T + 1))
    for n in range(grid_fine.N_T + 1):
        for i in range(grid_fine.N_K + 1):
            try:
                sigma_val = float(
                    lv_interp(np.array([[grid_fine.T[n], grid_fine.K[i]]]))[0]
                )
            except Exception:
                sigma_val = (sigma_lo + sigma_hi) / 2.0
            sigma_val = float(np.clip(sigma_val, sigma_lo, sigma_hi))
            u_fine[i, n] = sigma_val ** 2
    return u_fine


# ---------------------------------------------------------------------------
# Core: single experiment
# ---------------------------------------------------------------------------

def run_experiment(
    *,
    method: str,
    asset_index: int,
    asset_name: str,
    S0: float,
    tenors: np.ndarray,
    strikes: np.ndarray,
    iv_market: np.ndarray,
    r_curve: np.ndarray,
    q_curve: np.ndarray,
    # calibration grid
    N_K: int,
    N_T: int,
    alpha: float,
    max_iter: int,
    theta: float,
    sigma_bounds: tuple,
    ftol: float,
    gtol: float,
    # fine validation grid
    fine_N_K: int,
    fine_N_T: int,
    # output
    out_dir: Path,
    log_every: int = 5,
    oe_step_size: float = 0.1,
    oe_solver: str = "lu",
) -> dict:
    """Run one calibration + validation experiment. Returns a result dict."""

    # ------------------------------------------------------------------ setup
    r_fn = interp1d(r_curve[:, 0], r_curve[:, 1], kind="linear",
                    bounds_error=False,
                    fill_value=(r_curve[0, 1], r_curve[-1, 1]))
    q_fn = interp1d(q_curve[:, 0], q_curve[:, 1], kind="linear",
                    bounds_error=False,
                    fill_value=(q_curve[0, 1], q_curve[-1, 1]))

    K_min = float(strikes[0])  * (1.0 - cfg.GRID_K_MARGIN)
    K_max = float(strikes[-1]) * (1.0 + cfg.GRID_K_MARGIN)
    T_max = float(tenors[-1])  * (1.0 + cfg.GRID_T_MARGIN)

    # Coarse calibration grid
    grid_calib = Grid(S0=S0, K_min=K_min, K_max=K_max, T_max=T_max,
                      N_K=N_K, N_T=N_T, r=r_fn, q=q_fn)

    # Interpolate market IVs onto calibration grid
    iv_mkt_KT = iv_market.T  # (n_K, n_T)
    kx = min(3, len(strikes) - 1)
    ky = min(3, len(tenors)  - 1)
    spline = RectBivariateSpline(strikes, tenors, iv_mkt_KT, kx=kx, ky=ky)

    IV_calib = np.zeros((grid_calib.N_K + 1, grid_calib.N_T + 1))
    for n in range(grid_calib.N_T + 1):
        for i in range(grid_calib.N_K + 1):
            IV_calib[i, n] = max(float(spline(grid_calib.K[i], grid_calib.T[n])), 1e-4)
    IV_calib[:, 0] = np.nan

    r_arr_c, q_arr_c = grid_calib.rate_arrays()
    z = iv_surface_to_call_prices(IV_calib, grid_calib.K, grid_calib.T,
                                  S0, r_arr_c, q_arr_c)
    z[:, 0] = grid_calib.initial_condition()
    w = vega_weights(IV_calib, grid_calib.K, grid_calib.T, S0, r_arr_c, q_arr_c)

    sigma_prior = float(np.nanmedian(iv_market))
    u_star = np.full((grid_calib.N_K + 1, grid_calib.N_T + 1), sigma_prior ** 2)

    # ------------------------------------------------------------------ calibrate
    print(f"\n  [{method}]  NK={N_K}  NT={N_T}  alpha={alpha:.2e}  max_iter={max_iter}")

    if method in ("dto-price", "dto-iv"):
        misfit_type = "price" if method == "dto-price" else "iv"
        iv_mkt_arg  = IV_calib if misfit_type == "iv" else None
        r_arg       = r_arr_c  if misfit_type == "iv" else None
        q_arg       = q_arr_c  if misfit_type == "iv" else None
        result = calibrate(
            grid=grid_calib, z=z, w=w, u_star=u_star, alpha=alpha,
            sigma_bounds=sigma_bounds, theta=theta,
            ftol=ftol, gtol=gtol, max_iter=max_iter,
            verbose=True, log_every=log_every,
            misfit_type=misfit_type,
            iv_mkt=iv_mkt_arg, r_arr=r_arg, q_arr=q_arg,
        )

    elif method in ("oe-price", "oe-iv"):
        misfit_type = "price" if method == "oe-price" else "iv"
        iv_mkt_arg  = IV_calib if misfit_type == "iv" else None
        r_arg       = r_arr_c  if misfit_type == "iv" else None
        q_arg       = q_arr_c  if misfit_type == "iv" else None
        result = calibrate_oe(
            grid=grid_calib, z=z, w=w, u_star=u_star, alpha=alpha,
            sigma_bounds=sigma_bounds, theta=theta,
            max_iter=max_iter, tol=gtol,
            step_size=oe_step_size, oe_solver=oe_solver,
            misfit_type=misfit_type,
            iv_mkt=iv_mkt_arg, r_arr=r_arg, q_arr=q_arg,
            verbose=True, log_every=log_every,
        )

    elif method == "calib-v2":
        result = _run_calib_v2(
            S0=S0, tenors=tenors, strikes=strikes,
            iv_market=iv_market, r_curve=r_curve, q_curve=q_curve,
            N_K=N_K, N_T=N_T, alpha=alpha, max_iter=max_iter,
            sigma_bounds=sigma_bounds, theta=theta,
        )
    else:
        raise ValueError(f"Unknown method: {method!r}")

    print(f"  Calibration done: J={result.J_final:.4e}  iters={result.n_iter}"
          f"  t={result.elapsed_s:.1f}s")

    # ------------------------------------------------------------------ fine-grid validation
    print(f"\n  Validation on fine grid (NK={fine_N_K}, NT={fine_N_T}) ...")
    t0_valid = time.perf_counter()

    grid_fine = Grid(S0=S0, K_min=K_min, K_max=K_max, T_max=T_max,
                     N_K=fine_N_K, N_T=fine_N_T, r=r_fn, q=q_fn)

    if method == "calib-v2":
        u_fine = _build_u_fine_from_lv_interp(
            result._lv_interp, grid_fine, sigma_bounds
        )
    else:
        # Re-project calibrated local vol onto fine grid
        sigma_spline = RectBivariateSpline(
            grid_calib.K, grid_calib.T[1:], result.sigma_opt[:, 1:], kx=1, ky=1
        )
        u_fine = np.zeros((grid_fine.N_K + 1, grid_fine.N_T + 1))
        for n in range(grid_fine.N_T + 1):
            for i in range(grid_fine.N_K + 1):
                sigma_val = max(float(sigma_spline(grid_fine.K[i], grid_fine.T[n])),
                                sigma_bounds[0])
                sigma_val = min(sigma_val, sigma_bounds[1])
                u_fine[i, n] = sigma_val ** 2

    # Solve state PDE on fine grid
    from src.state_solver import solve_state
    C_fine = solve_state(u_fine, grid_fine, theta=theta)

    # Back-project to market grid IVs
    iv_model_mkt = backproject_iv(
        C_fine, grid_fine, tenors, strikes, r_curve, q_curve, S0
    )

    elapsed_valid = time.perf_counter() - t0_valid
    print(f"  Validation done in {elapsed_valid:.1f}s")

    # ------------------------------------------------------------------ metrics
    label = f"{method.upper()} (NK={N_K}, NT={N_T}, a={alpha:.0e})"
    print(f"  IV metrics (market vs. model on fine grid) [{method}]:")
    iv_metrics = print_iv_metrics(iv_market, iv_model_mkt, label=label)

    # ------------------------------------------------------------------ plots
    safe_name = asset_name.replace(" ", "_").replace("/", "_")
    method_safe = method.replace("-", "_")
    param_tag = (f"{safe_name}_{method_safe}_NK{N_K}_NT{N_T}"
                 f"_alpha{alpha:.0e}_iter{max_iter}")

    plot_dir  = out_dir / "plots"
    log_dir   = out_dir / "logs"
    plot_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    iv_plot_path   = plot_dir / f"iv_compare_{param_tag}.png"
    conv_plot_path = plot_dir / f"convergence_{param_tag}.png"

    plot_iv_comparison(
        strikes=strikes, tenors=tenors,
        iv_market=iv_market, iv_model=iv_model_mkt,
        label_model=label,
        title=f"{asset_name}  {method}  NK={N_K} NT={N_T} a={alpha:.0e}",
        out_path=str(iv_plot_path),
    )

    if hasattr(result, "history") and result.history.get("J"):
        plot_convergence(
            history=result.history,
            title=f"{asset_name}  {method}  NK={N_K} NT={N_T} a={alpha:.0e}",
            out_path=str(conv_plot_path),
        )

    # ------------------------------------------------------------------ log
    params = {
        "method":        method,
        "asset_index":   asset_index,
        "asset_name":    asset_name,
        "N_K":           N_K,
        "N_T":           N_T,
        "alpha":         alpha,
        "max_iter":      max_iter,
        "theta":         theta,
        "sigma_bounds":  sigma_bounds,
    }
    log_path = log_dir / f"log_{param_tag}.txt"
    write_log(
        log_path=log_path,
        params=params,
        calib_result=result,
        iv_metrics=iv_metrics,
        fine_grid_params={"N_K": fine_N_K, "N_T": fine_N_T},
        elapsed_validation=elapsed_valid,
    )

    n_fevals = getattr(result, "n_fevals", result.n_iter)
    success  = getattr(result, "success", getattr(result, "converged", True))

    return {
        "timestamp":       datetime.now().isoformat(timespec="seconds"),
        "method":          method,
        "asset_index":     asset_index,
        "asset_name":      asset_name,
        "NK_calib":        N_K,
        "NT_calib":        N_T,
        "alpha":           alpha,
        "max_iter":        max_iter,
        "J_final":         result.J_final,
        "grad_norm":       result.grad_norm,
        "n_iter":          result.n_iter,
        "n_fevals":        n_fevals,
        "IV_MAE":          iv_metrics["MAE"],
        "IV_RMSE":         iv_metrics["RMSE"],
        "IV_max_err":      iv_metrics["max_err"],
        "t_calibration_s": result.elapsed_s,
        "t_validation_s":  elapsed_valid,
        "success":         success,
        "fine_NK":         fine_N_K,
        "fine_NT":         fine_N_T,
    }


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Incremental parameter sweep for local-vol calibration."
    )
    parser.add_argument("--asset-index", type=int,
                        default=cfg.DEFAULT_ASSET_INDEX)
    parser.add_argument("--methods", nargs="+", default=ALL_METHODS,
                        choices=ALL_METHODS,
                        help=f"Methods to include (default: all). Choices: {ALL_METHODS}")
    parser.add_argument("--quick", action="store_true",
                        help="Small grid for fast sanity check")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if log already exists")
    parser.add_argument("--alpha", type=float, nargs="+",
                        help="Override SWEEP_ALPHA_VALUES")
    parser.add_argument("--nk", type=int, nargs="+",
                        help="Override SWEEP_N_K_VALUES")
    parser.add_argument("--nt", type=int, nargs="+",
                        help="Override SWEEP_N_T_VALUES")
    parser.add_argument("--max-iter", type=int, nargs="+",
                        help="Maximum iterations (single value or one per NK/NT pair)")
    parser.add_argument("--fine-nk", type=int, default=cfg.GRID_FINE_N_K)
    parser.add_argument("--fine-nt", type=int, default=cfg.GRID_FINE_N_T)
    parser.add_argument("--out-dir", type=str,
                        default=str(REPO_ROOT / "experiments"))
    parser.add_argument("--log-every", type=int, default=cfg.LOG_EVERY_N_ITER,
                        help="Print/record history every N iterations")
    parser.add_argument("--oe-step-size", type=float, default=0.1,
                        help="Step size for OE calibration (default: 0.1)")
    parser.add_argument("--oe-solver", type=str, default="lu",
                        choices=["lu", "cg", "dct"],
                        help="OE linear system solver backend (default: lu)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    # ------------------------------------------------------------------ load market data
    catalog = list_equity_assets(cfg.UNRISK_JSON_PATH)
    print("Available assets:")
    for a in catalog:
        print(f"  {a['index']:2d} | {a['name']} | ccy={a['currency']}")

    raw = read_implied_for_engine(cfg.UNRISK_JSON_PATH, asset_index=args.asset_index)
    asset_name = raw["asset_names"][0]
    S0         = float(raw["S0_list"][0])
    tenors     = np.asarray(raw["tenors_list"][0], dtype=float)
    strikes    = np.asarray(raw["strikes_list"][0], dtype=float)
    iv_market  = np.asarray(raw["vol_matrices"][0], dtype=float)
    r_curve    = raw["r_zero_list"][0]
    q_curve    = raw["q_zero_list"][0]

    print(f"\nLoaded: {asset_name}  S0={S0:.4f}"
          f"  tenors={tenors.size}  strikes={strikes.size}")

    # ------------------------------------------------------------------ build sweep list
    if args.quick:
        NK_list    = [cfg.SWEEP_QUICK_N_K]
        NT_list    = [cfg.SWEEP_QUICK_N_T]
        alpha_list = [cfg.ALPHA]
        iter_list  = [cfg.SWEEP_QUICK_MAX_ITER]
        fine_NK    = min(args.fine_nk, 80)
        fine_NT    = min(args.fine_nt, 60)
        print("\n[--quick mode]  Using small grid for fast sanity check.")
    else:
        NK_list    = args.nk if args.nk else cfg.SWEEP_N_K_VALUES
        NT_list    = args.nt if args.nt else cfg.SWEEP_N_T_VALUES
        alpha_list = args.alpha if args.alpha else cfg.SWEEP_ALPHA_VALUES
        fine_NK    = args.fine_nk
        fine_NT    = args.fine_nt

        if len(NK_list) != len(NT_list):
            parser.error("--nk and --nt must have the same number of values")

        if args.max_iter:
            if len(args.max_iter) == 1:
                iter_list = [args.max_iter[0]] * len(NK_list)
            elif len(args.max_iter) == len(NK_list):
                iter_list = args.max_iter
            else:
                parser.error("--max-iter must be length 1 or same length as --nk")
        else:
            iter_list = [cfg.MAX_ITER] * len(NK_list)

    # Build all (method, NK, NT, alpha, max_iter) combinations
    experiments = [
        (method, nk, nt, alpha, mi)
        for nk, nt, mi in zip(NK_list, NT_list, iter_list)
        for alpha in alpha_list
        for method in args.methods
    ]

    total = len(experiments)
    print(f"\n{'='*60}")
    print(f"  Sweep: {total} experiment(s)"
          f"  fine grid: NK={fine_NK} NT={fine_NT}")
    print(f"  Methods: {args.methods}")
    print(f"{'='*60}")

    csv_path = out_dir / "summary.csv"
    completed = 0
    skipped   = 0

    for exp_idx, (method, NK, NT, alpha, max_iter) in enumerate(experiments, start=1):
        safe_name  = asset_name.replace(" ", "_").replace("/", "_")
        method_safe = method.replace("-", "_")
        param_tag  = (f"{safe_name}_{method_safe}_NK{NK}_NT{NT}"
                      f"_alpha{alpha:.0e}_iter{max_iter}")
        log_path   = out_dir / "logs" / f"log_{param_tag}.txt"

        print(f"\n[{exp_idx}/{total}]  method={method}  NK={NK}  NT={NT}"
              f"  alpha={alpha:.2e}  max_iter={max_iter}")

        if log_path.exists() and not args.force and cfg.SWEEP_SKIP_EXISTING:
            print(f"  SKIP (log exists: {log_path.name})  "
                  f"Pass --force to re-run.")
            skipped += 1
            continue

        try:
            row = run_experiment(
                method=method,
                asset_index=args.asset_index,
                asset_name=asset_name,
                S0=S0,
                tenors=tenors,
                strikes=strikes,
                iv_market=iv_market,
                r_curve=r_curve,
                q_curve=q_curve,
                N_K=NK, N_T=NT,
                alpha=alpha,
                max_iter=max_iter,
                theta=cfg.THETA,
                sigma_bounds=cfg.SIGMA_BOUNDS,
                ftol=cfg.FTOL,
                gtol=cfg.GTOL,
                fine_N_K=fine_NK,
                fine_N_T=fine_NT,
                out_dir=out_dir,
                log_every=args.log_every,
                oe_step_size=args.oe_step_size,
                oe_solver=args.oe_solver,
            )
            append_csv_row(csv_path, row)
            completed += 1
            print(
                f"  DONE  IV_MAE={row['IV_MAE']:.5f}"
                f"  IV_RMSE={row['IV_RMSE']:.5f}"
                f"  t_calib={row['t_calibration_s']:.1f}s"
                f"  t_valid={row['t_validation_s']:.1f}s"
            )
        except KeyboardInterrupt:
            print("\n  Interrupted by user.  Partial results saved.")
            break
        except Exception as exc:
            warnings.warn(f"  Experiment FAILED: {exc}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"  Sweep finished:  {completed} done,  {skipped} skipped")
    print(f"  Summary CSV: {csv_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
