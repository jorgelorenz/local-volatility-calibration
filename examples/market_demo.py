"""
examples/market_demo.py
-----------------------
Market-data calibration demo: calibrate local volatility from a real
(or realistic synthetic) implied volatility surface.

Two modes
---------
1. --mode synthetic   (default)
   Builds a realistic implied vol surface with skew and term structure,
   converts it to call prices, then calibrates.  No external data needed.

2. --mode file --json PATH --asset NAME
   Loads an UnRisk-format JSON file and calibrates against it.

Usage
-----
    py examples/market_demo.py                          # synthetic market
    py examples/market_demo.py --mode file --json data/mkt.json --asset AAPL
    py examples/market_demo.py --plot                   # add surface plots
"""

import sys
import os
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.grid import Grid
from src.state_solver import solve_state
from src.market_data import (
    iv_surface_to_call_prices,
    vega_weights,
    synthetic_iv_surface,
    load_unrisk_market_data,
)
from src.calibration import calibrate
from src.utils import bs_call, implied_vol_brentq as compute_iv


# ---------------------------------------------------------------------------
# Grid parameters  (typical equity option surface)
# ---------------------------------------------------------------------------
S0    = 100.0
r     = 0.04
q     = 0.01
T_max = 2.0
K_min = 60.0
K_max = 160.0
N_K   = 80
N_T   = 60


# ---------------------------------------------------------------------------
# Realistic synthetic IV surface (skew + term structure)
# ---------------------------------------------------------------------------

def realistic_iv(K: float, T: float) -> float:
    """
    A realistic equity implied vol surface with:
      - Negative skew (downside protection premium)
      - Volatility term structure (short end elevated)
      - Moderate smile curvature

    Formula (approximation of typical equity surface):
      IV(K,T) = atm(T) + skew(T)*m + curve(T)*m^2

    where m = log(K/S0)/sqrt(T) is normalized log-moneyness,
    and:
      atm(T)   = 0.20 + 0.03*exp(-2T)          (term structure: 23% short, 20% long)
      skew(T)  = -0.08 / sqrt(T)                 (skew flattens with maturity)
      curve(T) =  0.04 / sqrt(T)                 (smile curvature)
    """
    if T <= 0:
        return 0.20
    m        = np.log(K / S0) / np.sqrt(T)
    atm_vol  = 0.20 + 0.03 * np.exp(-2.0 * T)
    skew     = -0.08 / np.sqrt(T)
    curve    =  0.04 / np.sqrt(T)
    iv       = atm_vol + skew * m + curve * m**2
    return float(np.clip(iv, 0.05, 1.50))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_synthetic(verbose: bool = True):
    """Calibrate against a synthetic realistic market IV surface."""
    print("=" * 65)
    print("Market local vol calibration  —  synthetic realistic surface")
    print("=" * 65)

    g = Grid(S0=S0, K_min=K_min, K_max=K_max, T_max=T_max,
             N_K=N_K, N_T=N_T, r=r, q=q)
    r_arr, q_arr = g.rate_arrays()

    # Build IV surface
    IV_mkt = synthetic_iv_surface(g.K, g.T, realistic_iv)  # NaN at T=0

    # Convert to call prices
    z = iv_surface_to_call_prices(IV_mkt, g.K, g.T, S0, r_arr, q_arr)
    z[:, 0] = g.initial_condition()

    # Vega-based weights (so misfit is approximately in IV units)
    w = vega_weights(IV_mkt, g.K, g.T, S0, r_arr, q_arr)
    w[:, 0] = 0.0    # T=0 not informative
    w[0,  :] = 0.0   # boundary K_min
    w[-1, :] = 0.0   # boundary K_max

    # Prior: flat ATM vol
    atm_guess = 0.21
    u_star = np.full((N_K + 1, N_T + 1), atm_guess**2)

    # Regularization
    alpha = 5e-5

    if verbose:
        print(f"\nGrid: N_K={N_K}, N_T={N_T}, K=[{K_min},{K_max}], T=[0,{T_max}]")
        print(f"Market IV range: [{np.nanmin(IV_mkt):.3f}, {np.nanmax(IV_mkt):.3f}]")
        print(f"ATM IV at T=0.5: {realistic_iv(S0, 0.5):.3f}")
        print(f"Prior sigma (flat): {atm_guess:.3f}")
        print(f"Alpha: {alpha:.2e}\n")

    result = calibrate(
        grid=g, z=z, w=w, u_star=u_star, alpha=alpha,
        u0=u_star.copy(),
        sigma_bounds=(0.02, 1.2),
        max_iter=500,
        ftol=1e-14,
        gtol=1e-8,
        verbose=verbose,
    )

    print("\n" + str(result))

    # -----------------------------------------------------------------------
    # Post-calibration diagnostics
    # -----------------------------------------------------------------------
    _diagnostics(g, result, z, w, IV_mkt, r_arr, q_arr, verbose)
    return result, g, IV_mkt


def run_file(json_path: str, asset_name: str, verbose: bool = True):
    """Calibrate against real market data from an UnRisk JSON file."""
    print("=" * 65)
    print(f"Market local vol calibration  —  {asset_name}")
    print("=" * 65)

    g = Grid(S0=S0, K_min=K_min, K_max=K_max, T_max=T_max,
             N_K=N_K, N_T=N_T, r=r, q=q)

    z, w, IV_mkt = load_unrisk_market_data(json_path, asset_name, g)
    r_arr, q_arr = g.rate_arrays()

    # Zero-weight boundary and T=0
    w[:, 0] = 0.0
    w[0,  :] = 0.0
    w[-1, :] = 0.0

    # Prior: ATM smile
    atm_guess = float(np.nanmedian(IV_mkt[:, 1:]))
    u_star = np.full((N_K + 1, N_T + 1), atm_guess**2)

    alpha = 5e-5

    if verbose:
        print(f"\nGrid: N_K={N_K}, N_T={N_T}")
        print(f"Market IV range: [{np.nanmin(IV_mkt):.3f}, {np.nanmax(IV_mkt):.3f}]")
        print(f"Prior sigma (median IV): {atm_guess:.3f}")
        print(f"Alpha: {alpha:.2e}\n")

    result = calibrate(
        grid=g, z=z, w=w, u_star=u_star, alpha=alpha,
        u0=u_star.copy(),
        sigma_bounds=(0.02, 1.2),
        max_iter=500,
        ftol=1e-14,
        gtol=1e-8,
        verbose=verbose,
    )

    print("\n" + str(result))
    _diagnostics(g, result, z, w, IV_mkt, r_arr, q_arr, verbose)
    return result, g, IV_mkt


def _diagnostics(g, result, z, w, IV_mkt, r_arr, q_arr, verbose):
    """Print calibration quality metrics."""
    C_opt = result.C_opt

    # Price RMSE
    mask = (w > 0) & np.isfinite(z) & (z > 0.01)
    if mask.sum() > 0:
        diff = C_opt[mask] - z[mask]
        rmse = np.sqrt(np.mean(diff**2))
        rel  = np.abs(diff) / z[mask]
        print(f"\nPrice fit ({mask.sum()} liquid options, price > $0.01):")
        print(f"  RMSE                : ${rmse:.4f}")
        print(f"  Mean rel. error     : {rel.mean()*100:.3f}%")
        print(f"  Max  rel. error     : {rel.max()*100:.3f}%")

    # IV RMSE (back out implied vol from calibrated prices)
    iv_errors = []
    for n in range(1, g.N_T + 1):
        T_n = g.T[n]
        r_n = float(r_arr[n])
        q_n = float(q_arr[n])
        for i in range(1, g.N_K):
            if not mask[i, n]:
                continue
            try:
                iv_cal = compute_iv(C_opt[i, n], S0, g.K[i], T_n, r_n, q_n)
                iv_mkt = float(IV_mkt[i, n])
                if np.isfinite(iv_cal) and np.isfinite(iv_mkt):
                    iv_errors.append(iv_cal - iv_mkt)
            except Exception:
                pass

    if iv_errors:
        iv_err = np.array(iv_errors)
        print(f"\nImplied vol fit (same {len(iv_err)} options):")
        print(f"  RMSE IV error (bps) : {np.sqrt(np.mean(iv_err**2))*1e4:.1f}")
        print(f"  Mean IV error (bps) : {np.mean(iv_err)*1e4:.1f}")
        print(f"  Max |IV error| (bps): {np.max(np.abs(iv_err))*1e4:.1f}")

    # Local vol summary
    sigma_opt = result.sigma_opt
    print(f"\nCalibrated local vol:")
    print(f"  Range : [{sigma_opt.min():.4f}, {sigma_opt.max():.4f}]")
    print(f"  Mean  : {sigma_opt[1:-1, 1:].mean():.4f}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(g, result, IV_mkt, r_arr, q_arr):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("matplotlib not available; skipping plot.")
        return

    sigma_opt = result.sigma_opt
    C_opt     = result.C_opt

    # Back-compute calibrated IV surface
    IV_cal = np.full_like(IV_mkt, np.nan)
    for n in range(1, g.N_T + 1):
        T_n = g.T[n]
        r_n = float(r_arr[n])
        q_n = float(q_arr[n])
        for i in range(1, g.N_K):
            try:
                iv = compute_iv(C_opt[i, n], S0, g.K[i], T_n, r_n, q_n)
                if np.isfinite(iv):
                    IV_cal[i, n] = iv
            except Exception:
                pass

    KK, TT = np.meshgrid(g.K, g.T, indexing="ij")

    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.35)

    # Row 1: IV surfaces
    for col, (data, title) in enumerate([
        (IV_mkt,            "Market IV"),
        (IV_cal,            "Calibrated IV"),
        (IV_cal - IV_mkt,   "IV residual (cal − mkt)"),
    ]):
        ax = fig.add_subplot(gs[0, col])
        c  = ax.contourf(TT[:, 1:], KK[:, 1:], data[:, 1:], levels=20,
                         cmap="RdYlGn" if col < 2 else "bwr")
        plt.colorbar(c, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("T")
        ax.set_ylabel("K")

    # Row 2: local vol surface + cross sections
    ax_lv = fig.add_subplot(gs[1, 0])
    c = ax_lv.contourf(TT[:, 1:], KK[:, 1:], sigma_opt[:, 1:],
                       levels=20, cmap="plasma")
    plt.colorbar(c, ax=ax_lv)
    ax_lv.set_title("Calibrated local vol σ(K,T)")
    ax_lv.set_xlabel("T")
    ax_lv.set_ylabel("K")

    # IV smile cross-section at a few maturities
    ax_smile = fig.add_subplot(gs[1, 1])
    tenors = [0.25, 0.5, 1.0, 2.0]
    for t_target in tenors:
        n = int(np.argmin(np.abs(g.T - t_target)))
        if g.T[n] <= 0:
            continue
        valid = np.isfinite(IV_mkt[:, n]) & np.isfinite(IV_cal[:, n])
        ax_smile.plot(g.K[valid], IV_mkt[valid, n], "o--", ms=3,
                      label=f"mkt T={g.T[n]:.2f}")
        ax_smile.plot(g.K[valid], IV_cal[valid, n], "-",
                      label=f"cal T={g.T[n]:.2f}")
    ax_smile.set_title("IV smile: market vs calibrated")
    ax_smile.set_xlabel("K")
    ax_smile.set_ylabel("IV")
    ax_smile.legend(fontsize=7)

    # Local vol cross section at ATM vs T
    ax_term = fig.add_subplot(gs[1, 2])
    atm_idx = int(np.argmin(np.abs(g.K - S0)))
    ax_term.plot(g.T[1:], sigma_opt[atm_idx, 1:], "b-", label="σ(ATM, T)")
    ax_term.set_title("ATM local vol term structure")
    ax_term.set_xlabel("T")
    ax_term.set_ylabel("σ")
    ax_term.legend()

    plt.suptitle("Local vol calibration — market demo", fontsize=13)
    out_path = "market_demo_result.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"\nPlot saved to {out_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",  default="synthetic", choices=["synthetic", "file"])
    parser.add_argument("--json",  default=None,
                        help="Path to UnRisk JSON file (required for --mode file)")
    parser.add_argument("--asset", default=None,
                        help="Asset name in JSON file")
    parser.add_argument("--plot",  action="store_true")
    args = parser.parse_args()

    if args.mode == "synthetic":
        result, g, IV_mkt = run_synthetic(verbose=True)
    else:
        if args.json is None or args.asset is None:
            parser.error("--mode file requires --json and --asset")
        result, g, IV_mkt = run_file(args.json, args.asset, verbose=True)

    if args.plot:
        r_arr, q_arr = g.rate_arrays()
        plot_results(g, result, IV_mkt, r_arr, q_arr)
