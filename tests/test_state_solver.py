"""
tests/test_state_solver.py
--------------------------
Phase 1 validation: flat local vol -> Dupire forward PDE should reproduce
Black-Scholes call prices analytically.

What is being tested
--------------------
For sigma(K,T) = sigma_0  (constant), the Dupire forward Kolmogorov PDE

    dC/dT = (1/2) u K² d²C/dK²  -  (r-q) K dC/dK  -  q C,   u = sigma_0²

has the closed-form solution C_BS(S0, K, T, r, q, sigma_0) given by the
standard Black-Scholes formula.  This test verifies that our finite-difference
solver (solve_state) reproduces that solution within the expected PDE
truncation error O(dK² + dT²).

Data used (100% synthetic — no market files)
---------------------------------------------
  Grid:   S0=100, K∈[50,200] (N_K=300), T∈[0,1] (N_T=100)
  Rates:  r=0.05 (constant), q=0.02 (constant)
  Vol:    sigma_0=0.20 (constant local vol → flat u=0.04 everywhere)
  Reference: Black-Scholes formula  bs_call(S0, K, T, r, q, sigma_0)

Test inventory
--------------
  test_flat_vol_bs_accuracy   Max relative error < 1% for interior strikes/maturities
  test_initial_condition      C(K,0) = max(S0-K, 0) exactly
  test_boundary_right_zero    C(K_max, T) = 0 for all T
  test_boundary_left          C(K_min, T) ≈ S0·e^{-qT} - K_min·e^{-rT}
  test_no_negative_prices     C(K,T) >= 0 everywhere
  test_monotone_in_strike     C(K_i, T) >= C(K_{i+1}, T) (non-arbitrage)

How to run
----------
  python -m pytest tests/test_state_solver.py -v        # via pytest
  python tests/test_state_solver.py                     # standalone
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from src.grid import Grid
from src.state_solver import solve_state
from src.utils import bs_call


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
S0    = 100.0
r     = 0.05
q     = 0.02
sigma = 0.20
T_max = 1.0
K_min = 50.0
K_max = 200.0   # wide domain so both BCs are far from ATM
N_K   = 300
N_T   = 100


@pytest.fixture(scope="module")
def grid_and_C():
    g = Grid(S0=S0, K_min=K_min, K_max=K_max, T_max=T_max,
             N_K=N_K, N_T=N_T, r=r, q=q)
    u = np.full((N_K + 1, N_T + 1), sigma**2)
    C = solve_state(u, g, theta=0.5)
    return g, C


def test_flat_vol_bs_accuracy(grid_and_C):
    """Max relative error in the interior (away from boundaries) < 1%.

    We skip the 20 outermost strike nodes on each side because the Dirichlet
    BC at K_min is the no-arbitrage intrinsic bound (S0*Bq - K_min*Br), not
    the exact BS price, so a small boundary layer error is expected near K_min.
    """
    g, C = grid_and_C
    errors = []
    # Start from T=0.25 to avoid short-maturity grid artefacts
    n_start = max(5, int(round(0.25 / g.dT)))
    for n in range(n_start, N_T + 1, 5):
        T = g.T[n]
        for i in range(30, N_K - 29):     # well away from both BCs
            K = g.K[i]
            c_bs  = bs_call(S0, K, T, r, q, sigma)
            c_pde = C[i, n]
            if c_bs > 0.10:    # skip deep OTM with tiny prices (< 10 cents)
                errors.append(abs(c_pde - c_bs) / c_bs)

    max_err = max(errors)
    print(f"\nFlat-vol test: max relative error = {max_err:.4e}")
    assert max_err < 0.01, f"Max relative error {max_err:.4e} exceeds 1%"


def test_initial_condition(grid_and_C):
    """C(K, 0) = max(S0 - K, 0)."""
    g, C = grid_and_C
    ic = g.initial_condition()
    np.testing.assert_allclose(C[:, 0], ic, atol=1e-10)


def test_boundary_right_zero(grid_and_C):
    """C(K_max, T) = 0 for all T."""
    g, C = grid_and_C
    np.testing.assert_allclose(C[-1, :], 0.0, atol=1e-10)


def test_boundary_left(grid_and_C):
    """C(K_min, T) ~ S0*Bq(T) - K_min*Br(T)."""
    g, C = grid_and_C
    bc_left = g.boundary_left()
    np.testing.assert_allclose(C[0, :], bc_left, rtol=1e-6)


def test_no_negative_prices(grid_and_C):
    """All call prices should be non-negative."""
    g, C = grid_and_C
    assert np.all(C >= -1e-10), f"Negative prices found: min={C.min():.4e}"


def test_monotone_in_strike(grid_and_C):
    """Call prices should decrease (non-strictly) as K increases.

    We only check the interior nodes (away from the left BC where the
    Dirichlet value slightly exceeds the adjacent BS price for small T).
    """
    g, C = grid_and_C
    for n in range(5, N_T + 1, 10):
        diffs = np.diff(C[1:, n])    # skip node 0 which is fixed BC
        assert np.all(diffs <= 1e-4), (
            f"Non-monotone at T={g.T[n]:.2f}: max(diff)={diffs.max():.4e}"
        )


if __name__ == "__main__":
    g = Grid(S0=S0, K_min=K_min, K_max=K_max, T_max=T_max,
             N_K=N_K, N_T=N_T, r=r, q=q)
    u = np.full((N_K + 1, N_T + 1), sigma**2)
    C = solve_state(u, g, theta=0.5)

    print("=== Flat-vol validation against Black-Scholes ===")
    print(f"  S0={S0}, r={r}, q={q}, sigma={sigma}")
    print(f"  Grid: N_K={N_K}, N_T={N_T}, K=[{K_min},{K_max}], T=[0,{T_max}]")
    print()
    print(f"{'K':>8} {'T':>6} {'C_BS':>12} {'C_PDE':>12} {'RelErr':>10}")
    print("-" * 52)
    for T_val in [0.25, 0.5, 1.0]:
        n = int(round(T_val / g.dT))
        for K_val in [80.0, 90.0, 100.0, 110.0, 120.0]:
            i = int(round((K_val - K_min) / g.dK))
            c_bs  = bs_call(S0, K_val, T_val, r, q, sigma)
            c_pde = C[i, n]
            err   = abs(c_pde - c_bs) / max(c_bs, 1e-6)
            print(f"  {K_val:6.0f} {T_val:6.2f} {c_bs:12.4f} {c_pde:12.4f} {err:10.4e}")
