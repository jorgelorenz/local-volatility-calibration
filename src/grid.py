"""
grid.py
-------
Defines the computational grid over (strike, maturity) space for the
Dupire PDE discretization.

Domain:  Omega = [K_min, K_max] x [0, T_max]
Axes:
  K[i]  = K_min + i * dK,   i = 0, ..., N_K       (N_K+1 nodes)
  T[n]  = n * dT,            n = 0, ..., N_T       (N_T+1 nodes)

The parameter u = sigma^2(K, T) is stored as a (N_K+1, N_T+1) array
(same shape as the call price surface C).

Notes on boundary conditions for the Dupire forward PDE
--------------------------------------------------------
Left  (K = K_min):  deep ITM call  ->  C ~ S0*B_q(T) - K_min*B_r(T)
Right (K = K_max):  deep OTM call  ->  C ~ 0
Initial (T = 0):    C(K,0) = max(S0 - K, 0)

where B_q(T) = exp(-int_0^T q(s) ds)  and  B_r(T) = exp(-int_0^T r(s) ds).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Union
import numpy as np


ScalarOrCallable = Union[float, Callable[[float], float]]


def _make_callable(x: ScalarOrCallable) -> Callable[[float], float]:
    """Wrap a scalar as a constant function if needed."""
    if callable(x):
        return x
    val = float(x)
    return lambda t: val  # noqa: E731


@dataclass
class Grid:
    """
    Computational grid for the Dupire PDE.

    Parameters
    ----------
    S0 : float
        Spot price at valuation date.
    K_min, K_max : float
        Strike boundaries.  K_min > 0.
    T_max : float
        Maximum maturity.
    N_K : int
        Number of strike intervals  (N_K+1 nodes).
    N_T : int
        Number of time  intervals   (N_T+1 nodes).
    r : float or callable
        Risk-free rate curve r(T).  If float, treated as constant.
    q : float or callable
        Dividend yield curve q(T).  If float, treated as constant.
    """

    S0: float
    K_min: float
    K_max: float
    T_max: float
    N_K: int
    N_T: int
    r: ScalarOrCallable = 0.0
    q: ScalarOrCallable = 0.0

    # Derived quantities (computed post-init)
    K: np.ndarray = field(init=False, repr=False)
    T: np.ndarray = field(init=False, repr=False)
    dK: float = field(init=False, repr=False)
    dT: float = field(init=False, repr=False)

    def __post_init__(self):
        assert self.K_min > 0, "K_min must be strictly positive."
        assert self.K_max > self.K_min
        assert self.T_max > 0
        assert self.N_K >= 2 and self.N_T >= 1

        self.K = np.linspace(self.K_min, self.K_max, self.N_K + 1)
        self.T = np.linspace(0.0, self.T_max, self.N_T + 1)
        self.dK = self.K[1] - self.K[0]
        self.dT = self.T[1] - self.T[0]

        # Wrap rate / dividend as callables
        self._r_func = _make_callable(self.r)
        self._q_func = _make_callable(self.q)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def r_val(self, T: float) -> float:
        """Instantaneous risk-free rate at maturity T."""
        return float(self._r_func(T))

    def q_val(self, T: float) -> float:
        """Instantaneous dividend yield at maturity T."""
        return float(self._q_func(T))

    def discount_r(self, T: float) -> float:
        """Discount factor exp(-int_0^T r(s) ds), via trapezoidal rule."""
        ts = np.linspace(0.0, T, max(int(T / self.dT * 10) + 1, 50))
        rs = np.array([self._r_func(t) for t in ts])
        return float(np.exp(-np.trapezoid(rs, ts)))

    def discount_q(self, T: float) -> float:
        """Dividend discount exp(-int_0^T q(s) ds), via trapezoidal rule."""
        ts = np.linspace(0.0, T, max(int(T / self.dT * 10) + 1, 50))
        qs = np.array([self._q_func(t) for t in ts])
        return float(np.exp(-np.trapezoid(qs, ts)))

    # ------------------------------------------------------------------
    # Precomputed arrays used repeatedly by PDE solvers
    # ------------------------------------------------------------------

    def rate_arrays(self):
        """
        Returns (r_arr, q_arr) of shape (N_T+1,) containing the rates
        evaluated at each time node T[n].
        """
        r_arr = np.array([self.r_val(t) for t in self.T])
        q_arr = np.array([self.q_val(t) for t in self.T])
        return r_arr, q_arr

    def boundary_left(self) -> np.ndarray:
        """
        Left BC: C(K_min, T_n) = S0*B_q(T_n) - K_min*B_r(T_n)
        Returns array of shape (N_T+1,).
        """
        Bq = np.array([self.discount_q(t) for t in self.T])
        Br = np.array([self.discount_r(t) for t in self.T])
        return self.S0 * Bq - self.K_min * Br

    def boundary_right(self) -> np.ndarray:
        """
        Right BC: C(K_max, T_n) = 0.
        Returns array of shape (N_T+1,).
        """
        return np.zeros(self.N_T + 1)

    def initial_condition(self) -> np.ndarray:
        """
        C(K_i, 0) = max(S0 - K_i, 0).
        Returns array of shape (N_K+1,).
        """
        return np.maximum(self.S0 - self.K, 0.0)

    def __repr__(self) -> str:
        return (
            f"Grid(S0={self.S0}, K=[{self.K_min},{self.K_max}], "
            f"T=[0,{self.T_max}], N_K={self.N_K}, N_T={self.N_T}, "
            f"dK={self.dK:.4f}, dT={self.dT:.4f})"
        )
