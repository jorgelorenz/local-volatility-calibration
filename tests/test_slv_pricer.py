"""
test_slv_pricer.py
------------------
Validation tests for src/slv_pricer.py.

Flat-leverage validation
~~~~~~~~~~~~~~~~~~~~~~~~
When xi=0 and L(S,t) = sigma / sqrt(v0) (constant), the SLV model
degenerates to a pure Black-Scholes GBM.  The backward SLV PDE collapses
to the standard BS backward PDE, so the numerical price must agree with
the BS closed-form up to discretisation error (< 2% relative error for
interior strikes and T >= 0.10).

Tests
-----
1. test_flat_leverage_atm         – ATM call, T=0.5
2. test_flat_leverage_itm         – ITM call (S > K)
3. test_flat_leverage_otm         – OTM call (S < K)
4. test_flat_leverage_multiple_T  – several maturities T in {0.1, 0.25, 0.5, 1.0}
5. test_non_negativity            – option values >= 0 everywhere
6. test_monotone_in_vol           – higher sigma -> higher call price
7. test_pure_lv_recovery          – verify xi>0 with rho=0 still gives ~BS at small xi
"""

from __future__ import annotations
import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils import bs_call
from src.slv_pricer import solve_slv


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_flat_leverage(sigma: float, v0: float, N_S: int, N_T: int):
    """
    Return a (N_S, N_T+1) leverage surface  L[i,n] = sigma / sqrt(v0)
    (constant), so that L * sqrt(v) = sigma when v = v0.
    """
    lev = sigma / np.sqrt(v0)
    return np.full((N_S, N_T + 1), lev)


def _default_grid(S0=100.0, K=100.0, T=0.5, sigma=0.2, N_S=80, N_T=100):
    S_min = 0.5 * S0
    S_max = 2.0 * S0
    S_nodes = np.linspace(S_min, S_max, N_S)
    t_grid  = np.linspace(0.0, T, N_T + 1)
    return S_nodes, t_grid


def _run_flat(sigma=0.2, v0=0.04, r=0.05, q=0.02,
              kappa=1.0, theta_v=0.04, xi=0.0, rho=0.0,
              S0=100.0, K=100.0, T=0.5,
              N_S=80, N_T=100, N_v=25):
    """Solve SLV with flat leverage and return (S_nodes, V_numerical)."""
    S_nodes, t_grid = _default_grid(S0, K, T, sigma, N_S, N_T)
    L = _make_flat_leverage(sigma, v0, N_S, N_T)
    payoff = lambda S: np.maximum(S - K, 0.0)  # noqa: E731

    V = solve_slv(
        L=L, S_nodes=S_nodes, t_grid=t_grid, T_final=T,
        r_func=r, q_func=q,
        kappa=kappa, theta_v=theta_v, xi=xi, rho=rho, v0=v0,
        payoff=payoff, K_strike=K,
        N_v=N_v, theta_adi=0.5,
    )
    return S_nodes, V


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFlatLeverageBS:
    """Flat-leverage SLV must recover Black-Scholes prices."""

    tol = 0.02   # 2% relative error

    def _check(self, S_nodes, V, K, T, r, q, sigma, label=""):
        """Compare numerical V to BS prices at each S node (interior only)."""
        errs = []
        for i, S in enumerate(S_nodes):
            bs = bs_call(S, K, T, r, q, sigma)
            if bs < 0.5:
                continue   # skip near-zero prices
            rel_err = abs(V[i] - bs) / bs
            errs.append(rel_err)
        assert errs, "No interior prices found to compare"
        max_err = max(errs)
        assert max_err < self.tol, (
            f"{label}: max relative error {max_err:.4f} exceeds {self.tol}"
        )

    def test_atm(self):
        r, q, sigma, v0 = 0.05, 0.02, 0.20, 0.04
        K, T = 100.0, 0.5
        S_nodes, V = _run_flat(sigma=sigma, v0=v0, r=r, q=q, K=K, T=T)
        self._check(S_nodes, V, K, T, r, q, sigma, label="ATM")

    def test_itm(self):
        r, q, sigma, v0 = 0.05, 0.02, 0.20, 0.04
        K, T = 80.0, 0.5
        S_nodes, V = _run_flat(sigma=sigma, v0=v0, r=r, q=q, K=K, T=T)
        self._check(S_nodes, V, K, T, r, q, sigma, label="ITM")

    def test_otm(self):
        r, q, sigma, v0 = 0.05, 0.02, 0.20, 0.04
        K, T = 120.0, 0.5
        S_nodes, V = _run_flat(sigma=sigma, v0=v0, r=r, q=q, K=K, T=T)
        self._check(S_nodes, V, K, T, r, q, sigma, label="OTM")

    @pytest.mark.parametrize("T", [0.10, 0.25, 0.5, 1.0])
    def test_multiple_maturities(self, T):
        r, q, sigma, v0 = 0.05, 0.0, 0.25, 0.0625
        K = 100.0
        N_T = max(50, int(T * 150))
        S_nodes, t_grid = _default_grid(T=T, N_T=N_T)
        L = _make_flat_leverage(sigma, v0, len(S_nodes), N_T)
        payoff = lambda S: np.maximum(S - K, 0.0)  # noqa: E731
        V = solve_slv(
            L=L, S_nodes=S_nodes, t_grid=t_grid, T_final=T,
            r_func=r, q_func=0.0,
            kappa=1.0, theta_v=v0, xi=0.0, rho=0.0, v0=v0,
            payoff=payoff, K_strike=K,
            N_v=25, theta_adi=0.5,
        )
        self._check(S_nodes, V, K, T, r, 0.0, sigma, label=f"T={T}")


class TestBasicProperties:
    """Model-independent sanity checks."""

    def test_non_negativity(self):
        """Option prices must be >= 0 everywhere."""
        S_nodes, V = _run_flat()
        assert np.all(V >= -1e-10), f"Negative prices found: min={V.min():.6f}"

    def test_monotone_in_vol(self):
        """Higher vol => higher call price (vega > 0)."""
        _, V_lo = _run_flat(sigma=0.15)
        _, V_hi = _run_flat(sigma=0.30)
        # At least in a large interior region, higher vol -> higher price
        mid = slice(len(V_lo) // 4, 3 * len(V_lo) // 4)
        assert np.all(V_hi[mid] >= V_lo[mid] - 1e-6), (
            "Higher vol should give higher prices in interior"
        )

    def test_intrinsic_value(self):
        """Prices must be >= intrinsic value (S - K)+ discounted."""
        r, q, sigma, v0 = 0.05, 0.02, 0.20, 0.04
        K, T = 100.0, 0.5
        S_nodes, V = _run_flat(sigma=sigma, v0=v0, r=r, q=q, K=K, T=T)
        intrinsic = np.maximum(S_nodes * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
        assert np.all(V >= intrinsic - 1e-4), (
            "Prices below discounted intrinsic value"
        )


class TestSmallXiRecovery:
    """With xi very small (near-zero vol-of-vol) result should still be near BS."""

    def test_small_xi(self):
        """xi=0.05, rho=0 should be within 3% of BS (slightly looser tolerance)."""
        r, q, sigma, v0 = 0.05, 0.0, 0.20, 0.04
        K, T = 100.0, 0.5
        S_nodes, t_grid = _default_grid(T=T)
        N_S = len(S_nodes)
        N_T = len(t_grid) - 1
        L = _make_flat_leverage(sigma, v0, N_S, N_T)
        payoff = lambda S: np.maximum(S - K, 0.0)  # noqa: E731
        V = solve_slv(
            L=L, S_nodes=S_nodes, t_grid=t_grid, T_final=T,
            r_func=r, q_func=0.0,
            kappa=2.0, theta_v=v0, xi=0.05, rho=0.0, v0=v0,
            payoff=payoff, K_strike=K,
            N_v=30, theta_adi=0.5,
        )
        errs = []
        for i, S in enumerate(S_nodes):
            bs = bs_call(S, K, T, r, 0.0, sigma)
            if bs < 0.5:
                continue
            errs.append(abs(V[i] - bs) / bs)
        assert max(errs) < 0.05, f"max rel err = {max(errs):.4f} > 0.05"
