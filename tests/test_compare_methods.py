"""
tests/test_compare_methods.py
-----------------------------
Integration test: compare the new Tikhonov PDE method against synthetic data
and (optionally) against v2 LM on real UnRisk market data.

What is being tested
--------------------
This file contains two test groups:

GROUP A — Synthetic data (always runs, no external files needed)
  Generates a known smooth local vol surface sigma_true(K,T), converts it to
  call prices, and calibrates.  Success criterion: IV MAE < 3% (300 bps).
  
  Data used:
    - 100% synthetic.  No market file.
    - sigma_true(K, T) = 0.20 + 0.05*(K/100-1)² + 0.02*T  (a mild smile+term)
    - Grid: S0=100, K∈[70,150], T∈[0,1], N_K=60, N_T=40
    - r=0.04 constant, q=0.01 constant

GROUP B — Real UnRisk data (skipped if JSON not found)
  Loads DJ EUROSTOXX 50 from the UnRisk JSON, calibrates with new method,
  and checks IV MAE < 5% (500 bps) as a loose smoke test.

  Data used:
    - Real market data from UnRisk JSON (asset_index=0).
    - File path from src.config.UNRISK_JSON_PATH.
    - If the file is missing, all GROUP B tests are skipped.

Output
------
  - IV comparison images saved to tests/out/
  - Convergence plots saved to tests/out/

How to run
----------
  python -m pytest tests/test_compare_methods.py -v        # via pytest
  python tests/test_compare_methods.py                     # standalone
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "help" / "other_calibration_process"))

from src.grid import Grid
from src.market_data import iv_surface_to_call_prices, vega_weights
from src.market_data import synthetic_iv_surface
from src.calibration import calibrate
from src.utils import implied_vol_brentq
from src.diagnostics import plot_iv_comparison, plot_convergence, print_iv_metrics
from src import config as cfg

OUT_DIR = Path(__file__).parent / "out"
OUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers shared by both groups
# ---------------------------------------------------------------------------

def _backproject_iv(C_opt, grid, tenors, strikes, S0, r, q):
    """Interpolate model prices on PDE grid → IV at (tenors, strikes)."""
    from scipy.interpolate import RectBivariateSpline

    C_spline = RectBivariateSpline(grid.K, grid.T[1:], C_opt[:, 1:], kx=1, ky=1)
    n_t, n_k = len(tenors), len(strikes)
    iv = np.full((n_t, n_k), np.nan)
    for i, t in enumerate(tenors):
        for j, k in enumerate(strikes):
            price = float(C_spline(k, t))
            iv[i, j] = implied_vol_brentq(price, S0, k, t, r, q, "call")
    return iv


# ===========================================================================
# GROUP A: Synthetic data tests
# ===========================================================================

class TestSyntheticCalibration:
    """
    Calibrate against a known synthetic IV surface and verify recovery.
    All data is generated in-memory — no external files required.
    """

    @pytest.fixture(scope="class")
    def synthetic_setup(self):
        """Build grid, synthetic IV surface, and run calibration once."""
        r, q = 0.04, 0.01
        S0 = 100.0
        grid = Grid(S0=S0, K_min=70.0, K_max=150.0, T_max=1.0,
                    N_K=60, N_T=40, r=r, q=q)

        # True local vol (also used as BS IV for this synthetic test)
        def sigma_true(K, T):
            return 0.20 + 0.05 * (K / S0 - 1.0) ** 2 + 0.02 * T

        IV_true = synthetic_iv_surface(grid.K, grid.T, sigma_true)
        r_arr, q_arr = grid.rate_arrays()
        z = iv_surface_to_call_prices(IV_true, grid.K, grid.T, S0, r_arr, q_arr)
        z[:, 0] = grid.initial_condition()
        w = vega_weights(IV_true, grid.K, grid.T, S0, r_arr, q_arr)

        sigma_prior = 0.20
        u_star = np.full_like(z, sigma_prior ** 2)

        result = calibrate(
            grid=grid, z=z, w=w, u_star=u_star,
            alpha=1e-3,
            sigma_bounds=(0.01, 1.0),
            theta=0.5,
            ftol=1e-10, gtol=1e-6,
            max_iter=200,
            verbose=True,
            log_every=10,
        )

        # Market tenors/strikes for back-projection
        tenors  = grid.T[5::5]   # subsample interior time nodes
        strikes = grid.K[5::5]   # subsample interior strike nodes
        iv_model = _backproject_iv(result.C_opt, grid, tenors, strikes, S0, r, q)

        # Build market IV at same (tenors, strikes) for comparison
        iv_mkt = np.full((len(tenors), len(strikes)), np.nan)
        for i, t in enumerate(tenors):
            for j, k in enumerate(strikes):
                iv_mkt[i, j] = sigma_true(k, t)

        metrics = print_iv_metrics(iv_mkt, iv_model, label="Synthetic calibration")

        # Save plots
        plot_iv_comparison(
            strikes=strikes, tenors=tenors,
            iv_market=iv_mkt, iv_model=iv_model,
            label_model="Tikhonov PDE (synthetic)",
            title="Synthetic smile+term  —  IV recovery",
            out_path=str(OUT_DIR / "iv_compare_synthetic.png"),
        )
        plot_convergence(
            history=result.history,
            title="Synthetic calibration convergence",
            out_path=str(OUT_DIR / "convergence_synthetic.png"),
        )

        return result, metrics, iv_mkt, iv_model

    def test_calibration_converged(self, synthetic_setup):
        """L-BFGS-B must declare success or reach a small gradient."""
        result, metrics, _, _ = synthetic_setup
        assert result.success or result.grad_norm < 1e-3, (
            f"Calibration did not converge: success={result.success}, "
            f"grad_norm={result.grad_norm:.2e}"
        )

    def test_iv_mae_below_threshold(self, synthetic_setup):
        """IV MAE should be below 3% (300 bps) on synthetic data."""
        _, metrics, _, _ = synthetic_setup
        assert metrics["MAE"] < 0.03, (
            f"IV MAE too large: {metrics['MAE']:.4f} (threshold 0.03)"
        )

    def test_iv_rmse_below_threshold(self, synthetic_setup):
        """IV RMSE should be below 4% on synthetic data."""
        _, metrics, _, _ = synthetic_setup
        assert metrics["RMSE"] < 0.04, (
            f"IV RMSE too large: {metrics['RMSE']:.4f} (threshold 0.04)"
        )

    def test_history_recorded(self, synthetic_setup):
        """Convergence history must be non-empty and consistent."""
        result, _, _, _ = synthetic_setup
        h = result.history
        assert len(h["J"]) > 0, "History 'J' is empty"
        assert len(h["J"]) == len(h["grad_norm"]) == len(h["delta_J"]), (
            "History arrays have inconsistent lengths"
        )

    def test_local_vol_positive(self, synthetic_setup):
        """Calibrated local vol must be strictly positive everywhere."""
        result, _, _, _ = synthetic_setup
        assert np.all(result.sigma_opt > 0), "Negative local vol found"

    def test_plots_created(self):
        """Output plots must have been created by the fixture."""
        assert (OUT_DIR / "iv_compare_synthetic.png").exists()
        assert (OUT_DIR / "convergence_synthetic.png").exists()


# ===========================================================================
# GROUP B: Real UnRisk market data tests
# ===========================================================================

_UNRISK_AVAILABLE = Path(cfg.UNRISK_JSON_PATH).exists()
_skip_unrisk = pytest.mark.skipif(
    not _UNRISK_AVAILABLE,
    reason=f"UnRisk JSON not found at {cfg.UNRISK_JSON_PATH}"
)


class TestMarketCalibration:
    """
    Smoke test on real UnRisk market data (DJ EUROSTOXX 50, asset_index=0).

    Data used:
      REAL market data from the UnRisk JSON file.
      Spot, IV surface, r-curve, q-curve all read from file.
      If the file is missing, all tests in this class are skipped.
    """

    @_skip_unrisk
    @pytest.fixture(scope="class")
    def market_setup(self):
        from unrisk_adapter import read_implied_for_engine
        from scipy.interpolate import interp1d, RectBivariateSpline

        raw        = read_implied_for_engine(cfg.UNRISK_JSON_PATH, asset_index=0)
        asset_name = raw["asset_names"][0]
        S0         = float(raw["S0_list"][0])
        tenors     = np.asarray(raw["tenors_list"][0], dtype=float)
        strikes    = np.asarray(raw["strikes_list"][0], dtype=float)
        iv_market  = np.asarray(raw["vol_matrices"][0], dtype=float)
        r_curve    = raw["r_zero_list"][0]
        q_curve    = raw["q_zero_list"][0]

        r_fn = interp1d(r_curve[:, 0], r_curve[:, 1], kind="linear",
                        bounds_error=False,
                        fill_value=(r_curve[0, 1], r_curve[-1, 1]))
        q_fn = interp1d(q_curve[:, 0], q_curve[:, 1], kind="linear",
                        bounds_error=False,
                        fill_value=(q_curve[0, 1], q_curve[-1, 1]))

        K_min = float(strikes[0])  * 0.98
        K_max = float(strikes[-1]) * 1.02
        T_max = float(tenors[-1])  * 1.02

        grid = Grid(S0=S0, K_min=K_min, K_max=K_max, T_max=T_max,
                    N_K=cfg.GRID_N_K, N_T=cfg.GRID_N_T, r=r_fn, q=q_fn)

        iv_mkt_KT = iv_market.T
        kx = min(3, len(strikes) - 1)
        ky = min(3, len(tenors)  - 1)
        spline = RectBivariateSpline(strikes, tenors, iv_mkt_KT, kx=kx, ky=ky)
        IV_grid = np.zeros((grid.N_K + 1, grid.N_T + 1))
        for n in range(grid.N_T + 1):
            for i in range(grid.N_K + 1):
                IV_grid[i, n] = max(float(spline(grid.K[i], grid.T[n])), 1e-4)
        IV_grid[:, 0] = np.nan

        r_arr, q_arr = grid.rate_arrays()
        z = iv_surface_to_call_prices(IV_grid, grid.K, grid.T, S0, r_arr, q_arr)
        z[:, 0] = grid.initial_condition()
        w = vega_weights(IV_grid, grid.K, grid.T, S0, r_arr, q_arr)

        sigma_prior = float(np.nanmedian(iv_market))
        u_star = np.full_like(z, sigma_prior ** 2)

        result = calibrate(
            grid=grid, z=z, w=w, u_star=u_star,
            alpha=cfg.ALPHA,
            sigma_bounds=cfg.SIGMA_BOUNDS,
            theta=cfg.THETA,
            ftol=cfg.FTOL, gtol=cfg.GTOL,
            max_iter=cfg.MAX_ITER,
            verbose=True,
            log_every=10,
        )

        # Back-project onto market grid
        C_spline = RectBivariateSpline(
            grid.K, grid.T[1:], result.C_opt[:, 1:], kx=1, ky=1
        )
        n_t, n_k = len(tenors), len(strikes)
        iv_model = np.full((n_t, n_k), np.nan)
        for i, t in enumerate(tenors):
            r_eff = float(r_fn(t))
            q_eff = float(q_fn(t))
            for j, k in enumerate(strikes):
                price = float(C_spline(k, t))
                iv_model[i, j] = implied_vol_brentq(price, S0, k, t,
                                                    r_eff, q_eff, "call")

        metrics = print_iv_metrics(iv_market, iv_model,
                                   label=f"Market calibration ({asset_name})")

        safe = asset_name.replace(" ", "_").replace("/", "_")
        plot_iv_comparison(
            strikes=strikes, tenors=tenors,
            iv_market=iv_market, iv_model=iv_model,
            label_model="Tikhonov PDE",
            title=f"{asset_name}  —  IV comparison",
            out_path=str(OUT_DIR / f"iv_compare_market_{safe}.png"),
        )
        plot_convergence(
            history=result.history,
            title=f"{asset_name}  convergence",
            out_path=str(OUT_DIR / f"convergence_market_{safe}.png"),
        )

        return result, metrics, asset_name

    @_skip_unrisk
    def test_market_calibration_runs(self, market_setup):
        """Calibration must complete without exception."""
        result, _, _ = market_setup
        assert result is not None

    @_skip_unrisk
    def test_market_iv_mae_smoke(self, market_setup):
        """Loose smoke test: IV MAE < 5% on real market data."""
        _, metrics, asset_name = market_setup
        assert metrics["MAE"] < 0.05, (
            f"[{asset_name}] IV MAE too large: {metrics['MAE']:.4f} (threshold 0.05)"
        )

    @_skip_unrisk
    def test_market_local_vol_positive(self, market_setup):
        """Calibrated local vol must be strictly positive."""
        result, _, _ = market_setup
        assert np.all(result.sigma_opt > 0)


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Running test_compare_methods.py as standalone script")
    print("=" * 60)

    print("\n[GROUP A] Synthetic calibration ...")
    tc = TestSyntheticCalibration()
    setup = tc.synthetic_setup.__wrapped__(tc) if hasattr(
        tc.synthetic_setup, "__wrapped__") else None

    # Run directly
    import types
    # Manually exercise the fixture
    r, q = 0.04, 0.01
    S0 = 100.0
    grid = Grid(S0=S0, K_min=70.0, K_max=150.0, T_max=1.0,
                N_K=60, N_T=40, r=r, q=q)

    def sigma_true(K, T):
        return 0.20 + 0.05 * (K / S0 - 1.0) ** 2 + 0.02 * T

    IV_true = synthetic_iv_surface(grid.K, grid.T, sigma_true)
    r_arr, q_arr = grid.rate_arrays()
    z = iv_surface_to_call_prices(IV_true, grid.K, grid.T, S0, r_arr, q_arr)
    z[:, 0] = grid.initial_condition()
    w = vega_weights(IV_true, grid.K, grid.T, S0, r_arr, q_arr)
    u_star = np.full_like(z, 0.04)

    result = calibrate(
        grid=grid, z=z, w=w, u_star=u_star,
        alpha=1e-3, sigma_bounds=(0.01, 1.0),
        theta=0.5, ftol=1e-10, gtol=1e-6, max_iter=200,
        verbose=True, log_every=10,
    )

    tenors  = grid.T[5::5]
    strikes = grid.K[5::5]
    iv_model = _backproject_iv(result.C_opt, grid, tenors, strikes, S0, r, q)
    iv_mkt = np.array([[sigma_true(k, t) for k in strikes] for t in tenors])
    metrics = print_iv_metrics(iv_mkt, iv_model, label="Synthetic")

    plot_iv_comparison(
        strikes=strikes, tenors=tenors,
        iv_market=iv_mkt, iv_model=iv_model,
        label_model="Tikhonov PDE (synthetic)",
        title="Synthetic smile+term — IV recovery",
        out_path=str(OUT_DIR / "iv_compare_synthetic.png"),
    )
    plot_convergence(
        history=result.history,
        title="Synthetic calibration convergence",
        out_path=str(OUT_DIR / "convergence_synthetic.png"),
    )

    print(f"\nSynthetic  IV_MAE={metrics['MAE']:.5f}"
          f"  IV_RMSE={metrics['RMSE']:.5f}"
          f"  t={result.elapsed_s:.1f}s")

    if _UNRISK_AVAILABLE:
        print("\n[GROUP B] Market calibration (UnRisk) ...")
        print("  (Run via pytest or experiments/sweep.py for full market test)")
    else:
        print(f"\n[GROUP B] Skipped — UnRisk JSON not found at {cfg.UNRISK_JSON_PATH}")
