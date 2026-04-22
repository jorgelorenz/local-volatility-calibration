"""
market_data.py
--------------
Utilities for converting market implied volatility surfaces to observed
call prices z(K,T) used as data in the inverse problem objective.

Also provides a loader for UnRisk JSON files via the unrisk_adapter module
(help/other_calibration_process/unrisk_adapter.py).

Functions
---------
iv_surface_to_call_prices(IV, K, T, S0, r_arr, q_arr)  -> np.ndarray
    Convert IV(K,T) to call prices via Black-Scholes.

vega_weights(IV, K, T, S0, r_arr, q_arr)  -> np.ndarray
    Compute Vega-based weights w(K,T) = 1/Vega^2 for the objective.

load_unrisk_market_data(json_path, grid, asset_index=0)  -> (np.ndarray, np.ndarray, np.ndarray)
    Load market implied vol surface from an UnRisk JSON file using
    unrisk_adapter, interpolate onto the given Grid, and return
    (z, w, IV_mkt).  The asset is selected by 0-based index (default 0).

list_unrisk_assets(json_path)  -> list[dict]
    Print and return the list of equity assets available in a UnRisk JSON.

synthetic_iv_surface(K, T, sigma_func)  -> np.ndarray
    Build a synthetic IV surface from a known sigma(K,T) function.

Notes
-----
The UnRisk JSON format used by this loader is the one produced by UnRisk
and stored under help/other_calibration_process/datos/.  It follows the
schema:
    market_data.models.equity_models.<asset_name>.implied_volatility_surface
"""

from __future__ import annotations
import sys
import warnings
from pathlib import Path
import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import interp1d

from .grid import Grid
from .utils import bs_call, bs_vega

# ---------------------------------------------------------------------------
# Locate and import unrisk_adapter (lives outside the src package)
# ---------------------------------------------------------------------------

def _get_unrisk_adapter():
    """Lazily import unrisk_adapter, adding its directory to sys.path once."""
    adapter_dir = str(
        Path(__file__).resolve().parent.parent
        / "help" / "other_calibration_process"
    )
    if adapter_dir not in sys.path:
        sys.path.insert(0, adapter_dir)
    try:
        import unrisk_adapter as _ua
        return _ua
    except ImportError as exc:
        raise ImportError(
            "unrisk_adapter not found.  Expected at "
            f"{adapter_dir}/unrisk_adapter.py"
        ) from exc


# ---------------------------------------------------------------------------
# IV -> call prices
# ---------------------------------------------------------------------------

def iv_surface_to_call_prices(
    IV: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    S0: float,
    r_arr: np.ndarray,
    q_arr: np.ndarray,
) -> np.ndarray:
    """
    Convert an implied volatility surface to call prices.

    Parameters
    ----------
    IV    : shape (N_K+1, N_T+1).  IV[i,n] = sigma_BS(K_i, T_n).
            NaN entries are allowed; the corresponding call price is set to NaN.
    K     : strike array, shape (N_K+1,)
    T     : maturity array, shape (N_T+1,)
    S0    : spot
    r_arr : risk-free rates, shape (N_T+1,)
    q_arr : dividend yields, shape (N_T+1,)

    Returns
    -------
    C : call price array, shape (N_K+1, N_T+1)
    """
    C = np.full_like(IV, np.nan)
    for n in range(len(T)):
        t = float(T[n])
        r = float(r_arr[n])
        q = float(q_arr[n])
        for i in range(len(K)):
            iv = float(IV[i, n])
            if np.isnan(iv) or t <= 0.0:
                C[i, n] = max(S0 - float(K[i]), 0.0) if t <= 0.0 else np.nan
            else:
                C[i, n] = bs_call(S0, float(K[i]), t, r, q, iv)
    return C


# ---------------------------------------------------------------------------
# Vega-based weights
# ---------------------------------------------------------------------------

def vega_weights(
    IV: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    S0: float,
    r_arr: np.ndarray,
    q_arr: np.ndarray,
    vega_floor: float = 1e-4,
) -> np.ndarray:
    """
    Compute inverse-Vega-squared weights for the misfit term:
      w(K,T) = 1 / max(Vega(K,T), vega_floor)^2

    These weights normalize the objective so that all strikes/maturities
    contribute roughly equally in *implied volatility* space.

    Returns
    -------
    w : shape (N_K+1, N_T+1), uniform weight 1.0 where IV is NaN or T=0.
    """
    w = np.ones_like(IV)
    for n in range(1, len(T)):
        t = float(T[n])
        r = float(r_arr[n])
        q = float(q_arr[n])
        for i in range(len(K)):
            iv = float(IV[i, n])
            if not np.isnan(iv):
                v = bs_vega(S0, float(K[i]), t, r, q, iv)
                w[i, n] = 1.0 / max(v, vega_floor) ** 2
    return w


# ---------------------------------------------------------------------------
# UnRisk JSON loader
# ---------------------------------------------------------------------------

def list_unrisk_assets(json_path: str) -> list:
    """
    List equity assets available in a UnRisk JSON file.

    Parameters
    ----------
    json_path : path to the UnRisk JSON file

    Returns
    -------
    List of dicts with keys 'index', 'name', 'currency'.
    Also prints the list to stdout for convenience.
    """
    ua = _get_unrisk_adapter()
    assets = ua.list_equity_assets(json_path)
    print(f"Assets in {Path(json_path).name}:")
    for a in assets:
        print(f"  {a['index']:2d} | {a['name']} | ccy={a['currency']}")
    return assets


def load_unrisk_market_data(
    json_path: str,
    grid: Grid,
    asset_index: int = 0,
) -> tuple:
    """
    Load market implied vol surface from an UnRisk JSON file and
    interpolate it onto the given Grid.

    Uses unrisk_adapter.read_implied_for_engine() to parse the JSON, so
    it works with the standard UnRisk schema without any manual asset-name
    annotation.

    Parameters
    ----------
    json_path   : path to the UnRisk JSON file
    grid        : Grid instance (defines K and T axes for the PDE).
                  grid.S0 is overridden by the spot read from the file.
    asset_index : 0-based index of the asset to load (default 0).
                  Call list_unrisk_assets(json_path) to see available assets.

    Returns
    -------
    z      : observed call prices on the grid, shape (N_K+1, N_T+1)
    w      : Vega-inverse-squared weights,     shape (N_K+1, N_T+1)
    IV_mkt : interpolated IV surface,          shape (N_K+1, N_T+1)

    Notes
    -----
    - Risk-free and dividend rates are read directly from the JSON and
      converted to effective annual rates for each grid tenor via linear
      interpolation of the zero-rate curves.
    - Only market tenors/strikes are used for the spline fit; grid nodes
      outside the market range are clamped to boundary values.
    - The grid's r and q attributes are NOT modified; rates are used only
      internally for the BS call-price conversion.  If you want the grid
      to use the market-consistent rates, build the Grid with callable r/q
      after loading the curves (see examples/market_demo.py).
    """
    ua = _get_unrisk_adapter()
    raw = ua.read_implied_for_engine(json_path, asset_index=asset_index)

    asset_name = raw["asset_names"][0]
    S0         = float(raw["S0_list"][0])
    tenors     = np.asarray(raw["tenors_list"][0], dtype=float)   # (n_T,)
    strikes    = np.asarray(raw["strikes_list"][0], dtype=float)  # (n_K,)
    iv_market  = np.asarray(raw["vol_matrices"][0], dtype=float)  # (n_T, n_K)
    r_curve    = raw["r_zero_list"][0]   # (n, 2): col0=years, col1=rate
    q_curve    = raw["q_zero_list"][0]   # (n, 2): col0=years, col1=rate

    # ---- build effective rate arrays on the PDE grid ----
    r_fn = interp1d(r_curve[:, 0], r_curve[:, 1], kind="linear",
                    bounds_error=False,
                    fill_value=(r_curve[0, 1], r_curve[-1, 1]))
    q_fn = interp1d(q_curve[:, 0], q_curve[:, 1], kind="linear",
                    bounds_error=False,
                    fill_value=(q_curve[0, 1], q_curve[-1, 1]))

    r_arr = np.array([float(r_fn(t)) for t in grid.T])
    q_arr = np.array([float(q_fn(t)) for t in grid.T])

    # ---- interpolate IV market surface onto PDE grid ----
    # iv_market shape: (n_T, n_K)  →  need IV_mkt_raw[i_K, i_T] = (n_K, n_T)
    IV_mkt_raw = iv_market.T  # (n_K, n_T)

    kx = min(3, len(strikes) - 1)
    ky = min(3, len(tenors) - 1)
    spline = RectBivariateSpline(strikes, tenors, IV_mkt_raw, kx=kx, ky=ky)

    IV_grid = np.zeros((grid.N_K + 1, grid.N_T + 1))
    for n in range(grid.N_T + 1):
        for i in range(grid.N_K + 1):
            val = float(spline(grid.K[i], grid.T[n]))
            IV_grid[i, n] = max(val, 1e-4)  # floor to avoid negative IV

    IV_grid[:, 0] = np.nan  # T=0: IV undefined

    z = iv_surface_to_call_prices(IV_grid, grid.K, grid.T, S0, r_arr, q_arr)
    w = vega_weights(IV_grid, grid.K, grid.T, S0, r_arr, q_arr)

    # Repair NaN call prices at T=0 with intrinsic values
    z[:, 0] = grid.initial_condition()

    return z, w, IV_grid


# ---------------------------------------------------------------------------
# Synthetic IV surface builder
# ---------------------------------------------------------------------------

def synthetic_iv_surface(
    K: np.ndarray,
    T: np.ndarray,
    sigma_func,
) -> np.ndarray:
    """
    Build a synthetic implied vol surface from a known local/constant vol
    function  sigma_func(K, T)  (assumed to equal the BS implied vol for
    this helper, i.e. the model is Black-Scholes with sigma=sigma_func).

    Parameters
    ----------
    K          : strike array, shape (N_K+1,)
    T          : maturity array, shape (N_T+1,)
    sigma_func : callable sigma_func(K_i, T_n) -> float

    Returns
    -------
    IV : shape (N_K+1, N_T+1); NaN at T=0.
    """
    IV = np.full((len(K), len(T)), np.nan)
    for n in range(1, len(T)):
        for i in range(len(K)):
            IV[i, n] = sigma_func(float(K[i]), float(T[n]))
    return IV
