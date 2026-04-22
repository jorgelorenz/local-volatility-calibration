"""
objective.py
------------
Evaluates the Tikhonov-regularized objective functional.

Two misfit types are supported (chosen via misfit_type argument):

  "price"  (default):
      J(u, C, z) = (1/2) * dK * dT * sum_{i,n} w_{i,n} * (C_{i,n} - z_{i,n})^2
                 + R(u, u_star, alpha)

  "iv":
      J(u, C, iv_mkt) = (1/2) * dK * dT * sum_{i,n} (IV_model_{i,n} - iv_mkt_{i,n})^2
                      + R(u, u_star, alpha)
      where IV_model is obtained by Black-Scholes inversion of C.
      This approach is analogous to calibrate_v2 in SimulatorEngine.py:
      it avoids the vega-weight approximation and directly minimises IV errors.
      NaN entries in iv_mkt (missing data) are treated as zero contribution.

NaN entries in z / w / iv_mkt are always treated as zero contribution.

Public API
----------
misfit_value(C, z, w, grid, misfit_type="price",
             iv_mkt=None, S0=None, r_arr=None, q_arr=None)  -> float
evaluate_J(u, C, z, w, u_star, alpha, grid, misfit_type="price",
           iv_mkt=None, S0=None, r_arr=None, q_arr=None)    -> float

For iv-misfit the adjoint/gradient still runs against the price residual via
vega-chain-rule; the misfit_iv_source() helper returns the effective
dJ/dC source term so that the adjoint equation remains unchanged in form.
"""

from __future__ import annotations
import numpy as np
from scipy.optimize import brentq
from .grid import Grid
from .regularization import tikhonov_value


# ---------------------------------------------------------------------------
# Fast vectorised BS call / vega
# ---------------------------------------------------------------------------

def _bs_call_vec(S0: float, K: np.ndarray, T: np.ndarray,
                 r_arr: np.ndarray, q_arr: np.ndarray,
                 sigma: np.ndarray) -> np.ndarray:
    """Vectorised Black-Scholes call price.  sigma/K/T/r_arr/q_arr all same shape."""
    eps = 1e-12
    sqT = np.sqrt(np.maximum(T, eps))
    sig_sqT = np.maximum(sigma, eps) * sqT
    d1 = (np.log(np.maximum(S0 / np.maximum(K, eps), eps))
          + (r_arr - q_arr + 0.5 * sigma**2) * T) / sig_sqT
    d2 = d1 - sig_sqT
    from scipy.special import ndtr
    Nd1 = ndtr(d1)
    Nd2 = ndtr(d2)
    df_r = np.exp(-r_arr * T)
    df_q = np.exp(-q_arr * T)
    return S0 * df_q * Nd1 - K * df_r * Nd2


def _bs_vega_vec(S0: float, K: np.ndarray, T: np.ndarray,
                 r_arr: np.ndarray, q_arr: np.ndarray,
                 sigma: np.ndarray) -> np.ndarray:
    """Vectorised BS vega: dC/dsigma = S0 * df_q * phi(d1) * sqrt(T)."""
    from scipy.special import ndtr
    from scipy.stats import norm
    eps = 1e-12
    sqT = np.sqrt(np.maximum(T, eps))
    sig_sqT = np.maximum(sigma, eps) * sqT
    d1 = (np.log(np.maximum(S0 / np.maximum(K, eps), eps))
          + (r_arr - q_arr + 0.5 * sigma**2) * T) / sig_sqT
    phi_d1 = norm.pdf(d1)
    df_q = np.exp(-q_arr * T)
    return S0 * df_q * phi_d1 * sqT


def _invert_iv_scalar(price: float, S0: float, K: float, T: float,
                      r: float, q: float) -> float:
    """Invert BS call to IV via Brentq; return NaN on failure."""
    eps = 1e-12
    T = max(T, eps)
    K = max(K, eps)
    df_r = np.exp(-r * T)
    df_q = np.exp(-q * T)
    intrinsic = max(S0 * df_q - K * df_r, 0.0)
    upper = S0 * df_q
    target = float(np.clip(price, intrinsic + 1e-14, upper - 1e-14))
    if target - intrinsic < 1e-6:
        return np.nan
    def bs(sig):
        from scipy.special import ndtr
        sig = max(sig, 1e-9)
        sqT = np.sqrt(T)
        d1 = (np.log(max(S0 / K, 1e-12)) + (r - q + 0.5 * sig**2) * T) / (sig * sqT)
        d2 = d1 - sig * sqT
        return S0 * np.exp(-q * T) * ndtr(d1) - K * np.exp(-r * T) * ndtr(d2) - target
    try:
        if bs(1e-6) * bs(5.0) > 0.0:
            return np.nan
        return float(brentq(bs, 1e-6, 5.0, xtol=1e-10, maxiter=100))
    except Exception:
        return np.nan


def call_prices_to_iv(C: np.ndarray, grid: Grid,
                      r_arr: np.ndarray, q_arr: np.ndarray) -> np.ndarray:
    """
    Invert model call prices C (N_K+1, N_T+1) to implied vol surface.
    Returns iv_model shape (N_K+1, N_T+1); NaN where inversion fails.
    T=0 column is always NaN.
    """
    iv = np.full_like(C, np.nan)
    S0 = grid.S0
    for n in range(1, grid.N_T + 1):
        t = grid.T[n]
        r = float(r_arr[n])
        q = float(q_arr[n])
        for i in range(1, grid.N_K):  # skip boundary nodes
            iv[i, n] = _invert_iv_scalar(float(C[i, n]), S0,
                                          float(grid.K[i]), t, r, q)
    return iv


# ---------------------------------------------------------------------------
# Price misfit
# ---------------------------------------------------------------------------

def misfit_value(
    C: np.ndarray,
    z: np.ndarray,
    w: np.ndarray,
    grid: Grid,
) -> float:
    """
    Price data misfit:  (1/2) integral_Omega w(K,T) (C - z)^2 dK dT

    NaN entries in z or w are skipped.

    Parameters
    ----------
    C, z : call price arrays, shape (N_K+1, N_T+1)
    w    : weight array,       shape (N_K+1, N_T+1)
    grid : Grid instance

    Returns
    -------
    misfit : scalar float
    """
    diff = C - z
    valid = ~np.isnan(diff) & ~np.isnan(w)
    integrand = np.where(valid, w * diff**2, 0.0)
    return float(0.5 * np.sum(integrand) * grid.dK * grid.dT)


# ---------------------------------------------------------------------------
# IV misfit  (analogous to calibrate_v2 approach)
# ---------------------------------------------------------------------------

def misfit_iv_value(
    C: np.ndarray,
    iv_mkt: np.ndarray,
    grid: Grid,
    r_arr: np.ndarray,
    q_arr: np.ndarray,
) -> float:
    """
    IV data misfit:  (1/2) dK * dT * sum_{i,n} (IV_model_{i,n} - iv_mkt_{i,n})^2

    IV_model is obtained by BS inversion of C.  NaN entries in iv_mkt are skipped.

    Parameters
    ----------
    C      : model call prices, shape (N_K+1, N_T+1)
    iv_mkt : market IV surface, shape (N_K+1, N_T+1)  (NaN = missing)
    grid   : Grid instance
    r_arr, q_arr : rate arrays, shape (N_T+1,)

    Returns
    -------
    misfit_iv : scalar float
    """
    iv_model = call_prices_to_iv(C, grid, r_arr, q_arr)
    diff = iv_model - iv_mkt
    valid = ~np.isnan(diff)
    integrand = np.where(valid, diff**2, 0.0)
    return float(0.5 * np.sum(integrand) * grid.dK * grid.dT)


def misfit_iv_source(
    C: np.ndarray,
    iv_mkt: np.ndarray,
    grid: Grid,
    r_arr: np.ndarray,
    q_arr: np.ndarray,
    sigma_iv: np.ndarray | None = None,
    vega_floor: float = 1e-4,
) -> np.ndarray:
    """
    Effective dJ/dC source for the IV misfit, i.e. the right-hand side
    that replaces w*(C-z) in the adjoint equation when misfit_type="iv".

    Using the chain rule:
        d/dC [ (IV_model - IV_mkt)^2 / 2 ]  =  (IV_model - IV_mkt) / Vega

    so the source term is:  source_{i,n} = (iv_model - iv_mkt) / vega

    Parameters
    ----------
    C        : model call prices, shape (N_K+1, N_T+1)
    iv_mkt   : market IV, shape (N_K+1, N_T+1)
    grid     : Grid instance
    r_arr, q_arr : rate arrays
    sigma_iv : optional pre-computed iv_model (avoids re-inversion)
    vega_floor : minimum vega for numerical stability

    Returns
    -------
    source : shape (N_K+1, N_T+1)  — same role as w*(C-z) in price misfit
    """
    if sigma_iv is None:
        sigma_iv = call_prices_to_iv(C, grid, r_arr, q_arr)

    # Build K, T meshgrids for vectorised vega computation
    K2d, T2d = np.meshgrid(grid.K, grid.T, indexing="ij")   # (N_K+1, N_T+1)
    r2d = np.tile(r_arr, (grid.N_K + 1, 1))
    q2d = np.tile(q_arr, (grid.N_K + 1, 1))

    # Vega using model IV (or iv_mkt where model IV is NaN)
    sigma_for_vega = np.where(~np.isnan(sigma_iv), sigma_iv, iv_mkt)
    sigma_for_vega = np.where(~np.isnan(sigma_for_vega), sigma_for_vega, 0.2)

    vega = _bs_vega_vec(grid.S0, K2d, T2d, r2d, q2d, sigma_for_vega)
    vega = np.maximum(vega, vega_floor)

    iv_diff = sigma_iv - iv_mkt  # NaN where either is NaN
    valid = ~np.isnan(iv_diff)
    # source = (iv_diff / vega), zero where invalid / boundary
    source = np.where(valid, iv_diff / vega, 0.0)
    # zero out T=0, boundaries
    source[:, 0] = 0.0
    source[0,  :] = 0.0
    source[-1, :] = 0.0
    return source


# ---------------------------------------------------------------------------
# Unified objective evaluation
# ---------------------------------------------------------------------------

def evaluate_J(
    u: np.ndarray,
    C: np.ndarray,
    z: np.ndarray,
    w: np.ndarray,
    u_star: np.ndarray,
    alpha: float,
    grid: Grid,
    misfit_type: str = "price",
    iv_mkt: np.ndarray | None = None,
    r_arr: np.ndarray | None = None,
    q_arr: np.ndarray | None = None,
) -> float:
    """
    Full objective:  J = misfit + Tikhonov

    Parameters
    ----------
    u            : current local variance,  shape (N_K+1, N_T+1)
    C            : model call prices (from solve_state), shape (N_K+1, N_T+1)
    z            : observed call prices,    shape (N_K+1, N_T+1)
    w            : weight array,            shape (N_K+1, N_T+1)
    u_star       : prior local variance,    shape (N_K+1, N_T+1)
    alpha        : regularization parameter
    grid         : Grid instance
    misfit_type  : "price" or "iv"
    iv_mkt       : market IV surface (required if misfit_type="iv")
    r_arr, q_arr : rate arrays (required if misfit_type="iv")

    Returns
    -------
    J : scalar float
    """
    if misfit_type == "iv":
        if iv_mkt is None or r_arr is None or q_arr is None:
            raise ValueError("iv_mkt, r_arr, q_arr required for misfit_type='iv'")
        mf = misfit_iv_value(C, iv_mkt, grid, r_arr, q_arr)
    else:
        mf = misfit_value(C, z, w, grid)

    return mf + tikhonov_value(u, u_star, alpha, grid)
