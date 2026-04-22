"""
utils.py
--------
Black-Scholes analytical formulas and implied volatility inversion.

Functions
---------
bs_call(S, K, T, r, q, sigma)  -> float
    Black-Scholes European call price.
bs_put(S, K, T, r, q, sigma)   -> float
    Black-Scholes European put price.
bs_vega(S, K, T, r, q, sigma)  -> float
    Vega: dC/dsigma  (same for call and put).
implied_vol_brentq(price, S, K, T, r, q, option_type, ...)  -> float
    Implied volatility via Brent root-finding.
iv_surface_from_prices(C, grid, S0)  -> ndarray
    Vectorized implied vol inversion over the full (K, T) grid.
"""

from __future__ import annotations
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


_SQRT2PI = np.sqrt(2.0 * np.pi)
_IV_LB = 1e-8   # lower bound for sigma in Brent search
_IV_UB = 10.0   # upper bound for sigma in Brent search


# ---------------------------------------------------------------------------
# Black-Scholes formulas
# ---------------------------------------------------------------------------

def _d1d2(S: float, K: float, T: float, r: float, q: float, sigma: float):
    """Return (d1, d2) for Black-Scholes."""
    if T <= 0.0 or sigma <= 0.0 or K <= 0.0:
        return np.nan, np.nan
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


def bs_call(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """
    European call price by Black-Scholes.

    Parameters
    ----------
    S     : spot price
    K     : strike
    T     : time to maturity (years)
    r     : continuously compounded risk-free rate
    q     : continuously compounded dividend yield
    sigma : implied / local volatility (annualised)

    Returns
    -------
    Call price  (>= intrinsic value, >= 0)
    """
    if T <= 0.0:
        return float(max(S * np.exp(-q * 0.0) - K * np.exp(-r * 0.0), 0.0))
    d1, d2 = _d1d2(S, K, T, r, q, sigma)
    return float(S * np.exp(-q * T) * norm.cdf(d1)
                 - K * np.exp(-r * T) * norm.cdf(d2))


def bs_put(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """European put price by Black-Scholes (put-call parity)."""
    call = bs_call(S, K, T, r, q, sigma)
    return float(call - S * np.exp(-q * T) + K * np.exp(-r * T))


def bs_vega(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """
    Vega = dC/dsigma = S * exp(-q*T) * N'(d1) * sqrt(T).

    Identical for calls and puts.
    """
    if T <= 0.0 or sigma <= 0.0:
        return 0.0
    d1, _ = _d1d2(S, K, T, r, q, sigma)
    return float(S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T))


# ---------------------------------------------------------------------------
# Implied volatility inversion
# ---------------------------------------------------------------------------

def implied_vol_brentq(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    option_type: str = "call",
    lb: float = _IV_LB,
    ub: float = _IV_UB,
    tol: float = 1e-10,
) -> float:
    """
    Implied volatility via Brent's method.

    Parameters
    ----------
    price       : observed option price
    S, K, T, r, q : market parameters
    option_type : 'call' or 'put'
    lb, ub      : search bracket for sigma
    tol         : root-finding tolerance

    Returns
    -------
    sigma_iv  or  np.nan if inversion fails / price out of bounds
    """
    if T <= 0.0:
        return np.nan

    if option_type == "call":
        pricer = lambda s: bs_call(S, K, T, r, q, s) - price  # noqa: E731
        intrinsic = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
        upper_bound = S * np.exp(-q * T)
    else:
        pricer = lambda s: bs_put(S, K, T, r, q, s) - price   # noqa: E731
        intrinsic = max(K * np.exp(-r * T) - S * np.exp(-q * T), 0.0)
        upper_bound = K * np.exp(-r * T)

    if price <= intrinsic or price >= upper_bound:
        return np.nan

    try:
        return float(brentq(pricer, lb, ub, xtol=tol, full_output=False))
    except ValueError:
        return np.nan


def iv_surface_from_prices(
    C: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    S0: float,
    r_arr: np.ndarray,
    q_arr: np.ndarray,
) -> np.ndarray:
    """
    Invert call prices to implied volatilities over the (K, T) grid.

    Parameters
    ----------
    C     : call price array, shape (N_K+1, N_T+1)
    K     : strike array, shape (N_K+1,)
    T     : maturity array, shape (N_T+1,)
    S0    : spot
    r_arr : risk-free rate at each T node, shape (N_T+1,)
    q_arr : dividend yield at each T node, shape (N_T+1,)

    Returns
    -------
    IV    : implied vol array, shape (N_K+1, N_T+1)
            np.nan where inversion failed or T=0
    """
    N_K1, N_T1 = C.shape
    IV = np.full_like(C, np.nan)
    for n in range(1, N_T1):           # skip T=0 (IV undefined)
        t = float(T[n])
        r = float(r_arr[n])
        q = float(q_arr[n])
        for i in range(N_K1):
            IV[i, n] = implied_vol_brentq(
                float(C[i, n]), S0, float(K[i]), t, r, q, "call"
            )
    return IV
