"""
examples/compare_methods.py
---------------------------
Full comparison of three local-volatility calibration methods on real
UnRisk market data (DJ EUROSTOXX 50 and S&P 500):

  v2          -- MCPricing Levenberg-Marquardt + PDE + B-splines  (calibrate_v2)
  UnRisk LR   -- Pre-computed local vol stored in the JSON
  New (Tikh.) -- Tikhonov-regularized PDE inverse problem (this repo)

For each asset the script produces:
  1. Per-tenor IV smile comparison  (3 x n_tenors grid of subplots)
  2. Local-vol surface heatmaps (v2 | Tikhonov)
  3. Error metrics table printed to stdout
  4. Calibration timing bar chart

Usage
-----
    py examples/compare_methods.py [--asset-index 0] [--out-dir examples/out]

Default: run on both assets (indices 0 and 1).
"""

from __future__ import annotations
import sys
import os
import argparse
import time
import warnings
from pathlib import Path

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

from unrisk_adapter import (
    list_equity_assets,
    read_implied_for_engine,
    read_local_for_engine,
)
from SimulatorEngine import DupireSimulator

from src.grid import Grid
from src.market_data import iv_surface_to_call_prices, vega_weights
from src.calibration import calibrate
from src.utils import implied_vol_brentq

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
JSON_PATH = str(
    REPO_ROOT / "help" / "other_calibration_process" / "datos"
    / "EquityBasket_8e55875c81554f3697bd765e672c3704_20260305_170633.json"
)


# ---------------------------------------------------------------------------
# Helpers: Black-Scholes with piecewise-constant (trapezoidal) rate integrals
# ---------------------------------------------------------------------------

def _integrate_curve_at(curve_arr: np.ndarray, t_val: float) -> float:
    """Integrate a step-function rate curve at a single maturity t_val.
    curve_arr is (n, 2): col-0 = year_fraction, col-1 = rate.
    Returns integral (not divided by t_val).
    """
    ts = curve_arr[:, 0]
    rs = curve_arr[:, 1]
    fn = interp1d(ts, rs, kind="linear", bounds_error=False,
                  fill_value=(rs[0], rs[-1]))
    t_query = np.linspace(0.0, float(t_val), 200)
    return float(np.trapezoid(fn(t_query), t_query))


def effective_rates(curve_arr: np.ndarray, tenors: np.ndarray):
    """Return effective (annualised) r_eff and integral for each tenor."""
    ints = np.array([_integrate_curve_at(curve_arr, t) for t in tenors])
    with np.errstate(invalid="ignore", divide="ignore"):
        effs = np.where(tenors > 0, ints / np.maximum(tenors, 1e-12), 0.0)
    return effs, ints


def _bs_call_scalar(S0, K, t, r_eff, q_eff, sigma):
    from scipy.special import ndtr
    t = max(float(t), 1e-12)
    sig = max(float(sigma), 1e-12)
    K = max(float(K), 1e-12)
    d1 = (np.log(max(S0, 1e-12) / K) + (r_eff - q_eff + 0.5 * sig * sig) * t) / (sig * np.sqrt(t))
    d2 = d1 - sig * np.sqrt(t)
    return S0 * np.exp(-q_eff * t) * ndtr(d1) - K * np.exp(-r_eff * t) * ndtr(d2)


def iv_from_price_scalar(S0, K, t, r_eff, q_eff, price):
    """Invert BS call price to IV; returns nan on failure."""
    from scipy.optimize import brentq
    from scipy.special import ndtr
    t = max(float(t), 1e-12)
    if t < 7.0 / 365.0:
        return np.nan
    K = max(float(K), 1e-12)
    df_r = np.exp(-r_eff * t)
    df_q = np.exp(-q_eff * t)
    intrinsic = max(S0 * df_q - K * df_r, 0.0)
    upper = S0 * df_q
    target = float(np.clip(price, intrinsic + 1e-14, upper - 1e-14))
    if target - intrinsic < 1e-4:
        return np.nan
    def f(sig):
        return _bs_call_scalar(S0, K, t, r_eff, q_eff, sig) - target
    if f(1e-6) * f(5.0) > 0.0:
        return np.nan
    try:
        return float(brentq(f, 1e-6, 5.0, xtol=1e-12, maxiter=200))
    except Exception:
        return np.nan


def prices_from_iv_matrix(S0, tenors, strikes, iv_matrix, r_curve, q_curve):
    """iv_matrix: (n_T, n_K) → prices (n_T, n_K) using effective rates."""
    r_eff, _ = effective_rates(r_curve, tenors)
    q_eff, _ = effective_rates(q_curve, tenors)
    prices = np.empty_like(iv_matrix, dtype=float)
    for i, t in enumerate(tenors):
        for j, k in enumerate(strikes):
            prices[i, j] = _bs_call_scalar(S0, k, t, float(r_eff[i]), float(q_eff[i]),
                                           float(iv_matrix[i, j]))
    return prices


def iv_from_prices_matrix(S0, tenors, strikes, prices, r_curve, q_curve):
    """prices: (n_T, n_K) → iv (n_T, n_K)."""
    r_eff, _ = effective_rates(r_curve, tenors)
    q_eff, _ = effective_rates(q_curve, tenors)
    iv = np.full_like(prices, np.nan)
    for i, t in enumerate(tenors):
        for j, k in enumerate(strikes):
            iv[i, j] = iv_from_price_scalar(S0, k, t, float(r_eff[i]), float(q_eff[i]),
                                            float(prices[i, j]))
    return iv


# ---------------------------------------------------------------------------
# New-method (Tikhonov PDE) adapter
# ---------------------------------------------------------------------------

def run_new_method(S0, tenors, strikes, iv_market, r_curve, q_curve, verbose=True):
    """
    Build a Grid from market data, interpolate IV onto it, run calibrate(),
    and back-project model prices onto market (tenor, strike) grid.

    Returns
    -------
    iv_new_on_mkt  : (n_T, n_K) IV implied by new method at market grid points
    sigma_surface  : (N_K+1, N_T+1) local vol surface on PDE grid
    grid           : Grid instance
    result         : CalibrationResult
    elapsed        : calibration time (seconds)
    """
    # Rate callables via linear interpolation
    r_fn = interp1d(r_curve[:, 0], r_curve[:, 1], kind="linear",
                    bounds_error=False, fill_value=(r_curve[0, 1], r_curve[-1, 1]))
    q_fn = interp1d(q_curve[:, 0], q_curve[:, 1], kind="linear",
                    bounds_error=False, fill_value=(q_curve[0, 1], q_curve[-1, 1]))

    K_min = float(strikes[0]) * 0.98
    K_max = float(strikes[-1]) * 1.02
    T_max = float(tenors[-1]) * 1.02

    N_K = 80
    N_T = 60

    grid = Grid(S0=S0, K_min=K_min, K_max=K_max, T_max=T_max,
                N_K=N_K, N_T=N_T, r=r_fn, q=q_fn)

    # Interpolate iv_market (n_T, n_K) → IV_grid (N_K+1, N_T+1)
    # iv_market axes: [tenor_idx, strike_idx] → we need [K_idx, T_idx]
    iv_mkt_KT = iv_market.T  # (n_K, n_T)
    kx = min(3, len(strikes) - 1)
    ky = min(3, len(tenors) - 1)
    spline = RectBivariateSpline(strikes, tenors, iv_mkt_KT, kx=kx, ky=ky)

    IV_grid = np.zeros((grid.N_K + 1, grid.N_T + 1))
    for n in range(grid.N_T + 1):
        for i in range(grid.N_K + 1):
            val = float(spline(grid.K[i], grid.T[n]))
            IV_grid[i, n] = max(val, 1e-4)
    IV_grid[:, 0] = np.nan  # T=0 undefined

    r_arr, q_arr = grid.rate_arrays()
    z = iv_surface_to_call_prices(IV_grid, grid.K, grid.T, S0, r_arr, q_arr)
    z[:, 0] = grid.initial_condition()
    w = vega_weights(IV_grid, grid.K, grid.T, S0, r_arr, q_arr)

    # Prior: flat vol = median of market
    sigma_prior = float(np.nanmedian(iv_market))
    u_star = np.full((grid.N_K + 1, grid.N_T + 1), sigma_prior ** 2)

    alpha = 1e-3

    t0 = time.perf_counter()
    result = calibrate(
        grid=grid, z=z, w=w, u_star=u_star, alpha=alpha,
        sigma_bounds=(0.01, 1.5), theta=0.5,
        ftol=1e-10, gtol=1e-6, max_iter=300, verbose=verbose,
    )
    elapsed = time.perf_counter() - t0

    # Back-project model prices onto market (tenor, strike) grid
    # Model prices C_opt are on (N_K+1, N_T+1) grid; interpolate to market points
    C_opt = result.C_opt  # shape (N_K+1, N_T+1)

    # Use bilinear spline over (K, T) → interpolate at (strikes[j], tenors[i])
    C_spline = RectBivariateSpline(grid.K, grid.T[1:], C_opt[:, 1:], kx=1, ky=1)

    r_eff, _ = effective_rates(r_curve, tenors)
    q_eff, _ = effective_rates(q_curve, tenors)

    iv_new_on_mkt = np.full_like(iv_market, np.nan)
    for i, t in enumerate(tenors):
        for j, k in enumerate(strikes):
            price = float(C_spline(k, t))
            iv_new_on_mkt[i, j] = iv_from_price_scalar(
                S0, k, t, float(r_eff[i]), float(q_eff[i]), price
            )

    return iv_new_on_mkt, result.sigma_opt, grid, result, elapsed


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_iv_smiles(strikes, tenors, iv_market, iv_v2, iv_local_read,
                   iv_new, asset_name, out_path):
    """Per-tenor IV smile subplots: market / v2 / UnRisk-LR / new Tikhonov."""
    n_t = len(tenors)
    n_cols = 3
    n_rows = int(np.ceil(n_t / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5.5 * n_cols, 3.8 * n_rows), squeeze=False)
    for i, t in enumerate(tenors):
        ax = axes[i // n_cols][i % n_cols]
        ax.plot(strikes, iv_market[i, :], "o-",  label="IV market",      lw=1.8, ms=4)
        ax.plot(strikes, iv_v2[i, :],     "s--", label="IV v2 (LM-BS)",  lw=1.5, ms=4)
        ax.plot(strikes, iv_local_read[i, :], "^-.", label="IV UnRisk LR", lw=1.3, ms=4)
        if iv_new is not None:
            ax.plot(strikes, iv_new[i, :], "D:",  label="IV Tikhonov PDE", lw=1.5, ms=4)
        ax.set_title(f"T={t:.3f}y", fontsize=9)
        ax.set_xlabel("Strike", fontsize=8)
        ax.set_ylabel("Implied Vol", fontsize=8)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=7)
        ax.tick_params(labelsize=7)
    for j in range(n_t, n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis("off")
    fig.suptitle(f"{asset_name} — IV smile comparison", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_local_vol_surfaces(grid_new, sigma_new, sim_v2, tenors, strikes,
                             asset_name, out_path):
    """Side-by-side heatmaps: v2 local vol | new Tikhonov local vol."""
    # v2 local vol evaluated on market grid
    sigma_fn_v2 = sim_v2.vol_interpolators_[0]
    n_T = len(tenors)
    n_K = len(strikes)
    lv_v2 = np.zeros((n_T, n_K))
    for i, t in enumerate(tenors):
        for j, k in enumerate(strikes):
            lv_v2[i, j] = float(sigma_fn_v2(k, t))

    # new method: sigma_new is (N_K+1, N_T+1) on PDE grid — sample on market grid
    sigma_new_spline = RectBivariateSpline(grid_new.K, grid_new.T[1:],
                                           sigma_new[:, 1:], kx=1, ky=1)
    lv_new = np.zeros((n_T, n_K))
    for i, t in enumerate(tenors):
        for j, k in enumerate(strikes):
            lv_new[i, j] = float(sigma_new_spline(k, t))

    T_mesh, K_mesh = np.meshgrid(tenors, strikes, indexing="ij")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    vmin = min(lv_v2.min(), lv_new.min())
    vmax = max(lv_v2.max(), lv_new.max())

    for ax, lv, title in [
        (axes[0], lv_v2, "v2 (LM-B-splines)"),
        (axes[1], lv_new, "New (Tikhonov PDE)"),
    ]:
        im = ax.pcolormesh(K_mesh, T_mesh, lv, cmap="RdYlGn_r",
                           vmin=vmin, vmax=vmax, shading="auto")
        plt.colorbar(im, ax=ax, label="Local vol σ(K,T)")
        ax.set_xlabel("Strike K")
        ax.set_ylabel("Maturity T (years)")
        ax.set_title(f"{asset_name}\n{title}", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_timing(methods, times, asset_name, out_path):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    bars = ax.bar(methods, times, color=colors[: len(methods)], edgecolor="k", linewidth=0.6)
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5, f"{t:.1f}s",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Calibration time (s)")
    ax.set_title(f"{asset_name} — calibration timing")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------------

def error_metrics(iv_mkt, iv_model, label):
    mask = np.isfinite(iv_model) & np.isfinite(iv_mkt)
    if mask.sum() == 0:
        return {"label": label, "MAE": np.nan, "RMSE": np.nan, "n": 0}
    diff = iv_model[mask] - iv_mkt[mask]
    return {
        "label": label,
        "MAE":   float(np.mean(np.abs(diff))),
        "RMSE":  float(np.sqrt(np.mean(diff ** 2))),
        "n":     int(mask.sum()),
    }


def print_metrics_table(metrics_list, asset_name):
    print(f"\n{'='*55}")
    print(f"  Error metrics — {asset_name}")
    print(f"{'='*55}")
    print(f"  {'Method':<22}  {'MAE':>10}  {'RMSE':>10}  {'n':>5}")
    print(f"  {'-'*22}  {'-'*10}  {'-'*10}  {'-'*5}")
    for m in metrics_list:
        print(f"  {m['label']:<22}  {m['MAE']:>10.6f}  {m['RMSE']:>10.6f}  {m['n']:>5d}")
    print(f"{'='*55}\n")


# ---------------------------------------------------------------------------
# Per-asset pipeline
# ---------------------------------------------------------------------------

def run_asset(asset_index: int, out_dir: Path, skip_new: bool = False):
    print(f"\n{'#'*60}")
    print(f"  Processing asset index {asset_index}")
    print(f"{'#'*60}")

    raw        = read_implied_for_engine(JSON_PATH, asset_index=asset_index)
    raw_local  = read_local_for_engine(JSON_PATH, asset_index=asset_index)
    asset_name = raw["asset_names"][0]

    S0         = float(raw["S0_list"][0])
    tenors     = np.asarray(raw["tenors_list"][0], dtype=float)
    strikes    = np.asarray(raw["strikes_list"][0], dtype=float)
    iv_market  = np.asarray(raw["vol_matrices"][0], dtype=float)     # (n_T, n_K)
    r_curve    = raw["r_zero_list"][0]                                 # (n, 2)
    q_curve    = raw["q_zero_list"][0]
    lv_local_read = np.asarray(raw_local["vol_matrices"][0], dtype=float)

    print(f"  Asset: {asset_name}  S0={S0:.4f}  "
          f"tenors={tenors.size}  strikes={strikes.size}")

    # ------------------------------------------------------------------ v2
    print("\n--- Running v2 (LM + B-splines) ---")
    sim_v2 = DupireSimulator(S0=S0, r_zero=r_curve, q_zero=q_curve, n_assets=1)
    t0_v2 = time.perf_counter()
    sim_v2.calibrate_v2(
        vol_matrices=iv_market,
        strikes_list=strikes,
        tenors_list=tenors,
        vol_floor=1e-8,
        degree=3,
        n_knots_T=4,
        n_knots_K=8,
        alpha=1e-2,
        lm_lambda0=1e-1,
        max_iter=8,
        tol=1e-6,
        use_log_sigma=True,
        nK_pde=400,
        nT_pde=500,
        pde_theta=1.0,
    )
    t_v2 = time.perf_counter() - t0_v2
    print(f"  v2 done in {t_v2:.1f}s")

    model_prices_v2 = sim_v2._dupire_solve_call_surface(
        S0=S0, tenors=tenors, strikes=strikes,
        sigma_fn=sim_v2.vol_interpolators_[0],
        r_curve=r_curve, q_curve=q_curve,
        vol_floor=1e-8, nK_pde=400, nT_pde=500, theta=1.0,
    )
    iv_v2 = iv_from_prices_matrix(S0, tenors, strikes, model_prices_v2, r_curve, q_curve)

    # ------------------------------------------------------------------ UnRisk LR
    print("\n--- Computing UnRisk local-read IV ---")
    sim_lr = DupireSimulator(
        S0=S0, r_zero=r_curve, q_zero=q_curve,
        vol_matrices=lv_local_read, tenor_grids=tenors, strike_grids=strikes,
        n_assets=1,
    )
    lr_prices = sim_lr._dupire_solve_call_surface(
        S0=S0, tenors=tenors, strikes=strikes,
        sigma_fn=sim_lr.vol_interpolators_[0],
        r_curve=r_curve, q_curve=q_curve,
        vol_floor=1e-8, nK_pde=400, nT_pde=500, theta=1.0,
    )
    iv_local_read = iv_from_prices_matrix(S0, tenors, strikes, lr_prices, r_curve, q_curve)

    # ------------------------------------------------------------------ New method
    iv_new = None
    sigma_new = None
    grid_new = None
    t_new = None

    if not skip_new:
        print("\n--- Running new Tikhonov PDE method ---")
        try:
            iv_new, sigma_new, grid_new, res_new, t_new = run_new_method(
                S0, tenors, strikes, iv_market, r_curve, q_curve, verbose=True
            )
            print(f"  New method done in {t_new:.1f}s  |  J={res_new.J_final:.4e}")
        except Exception as exc:
            warnings.warn(f"New method failed: {exc}")
            iv_new = None

    # ------------------------------------------------------------------ Plots
    safe_name = asset_name.replace(" ", "_").replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)

    # IV smile comparison
    plot_iv_smiles(
        strikes, tenors, iv_market, iv_v2, iv_local_read, iv_new,
        asset_name,
        out_dir / f"iv_compare_{safe_name}.png",
    )

    # Local vol heatmaps (only if new method ran)
    if sigma_new is not None and grid_new is not None:
        plot_local_vol_surfaces(
            grid_new, sigma_new, sim_v2, tenors, strikes,
            asset_name,
            out_dir / f"localvol_surface_{safe_name}.png",
        )

    # Timing bar chart
    methods = ["v2 (LM-BS)"]
    times   = [t_v2]
    if t_new is not None:
        methods.append("New (Tikhonov)")
        times.append(t_new)
    plot_timing(methods, times, asset_name, out_dir / f"timing_{safe_name}.png")

    # ------------------------------------------------------------------ Metrics
    metrics = [
        error_metrics(iv_market, iv_v2,         "v2 (LM-B-splines)"),
        error_metrics(iv_market, iv_local_read,  "UnRisk local-read"),
    ]
    if iv_new is not None:
        metrics.append(error_metrics(iv_market, iv_new, "Tikhonov PDE (new)"))
    print_metrics_table(metrics, asset_name)

    return {
        "asset": asset_name,
        "t_v2":  t_v2,
        "t_new": t_new,
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare local-vol calibration methods on UnRisk market data."
    )
    parser.add_argument(
        "--asset-index", type=int, default=None,
        help="0-based asset index (default: run all assets)"
    )
    parser.add_argument(
        "--out-dir", type=str,
        default=str(REPO_ROOT / "examples" / "out"),
        help="Output directory for plots"
    )
    parser.add_argument(
        "--skip-new", action="store_true",
        help="Skip the new Tikhonov PDE calibration (faster, for v2/LR comparison only)"
    )
    args = parser.parse_args()

    catalog = list_equity_assets(JSON_PATH)
    print("Available assets:")
    for a in catalog:
        print(f"  {a['index']:2d} | {a['name']} | ccy={a['currency']}")

    if args.asset_index is not None:
        indices = [args.asset_index]
    else:
        indices = [a["index"] for a in catalog]

    out_dir = Path(args.out_dir)
    results = []
    for idx in indices:
        res = run_asset(idx, out_dir, skip_new=args.skip_new)
        results.append(res)

    print("\n" + "=" * 55)
    print("  SUMMARY")
    print("=" * 55)
    for r in results:
        print(f"\n  {r['asset']}")
        print(f"    v2 time : {r['t_v2']:.1f}s")
        if r["t_new"] is not None:
            print(f"    new time: {r['t_new']:.1f}s")
        for m in r["metrics"]:
            print(f"    [{m['label']}]  MAE={m['MAE']:.6f}  RMSE={m['RMSE']:.6f}")


if __name__ == "__main__":
    main()
