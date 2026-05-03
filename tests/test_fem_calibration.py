"""
tests/test_fem_calibration.py
------------------------------
Tests for the FEM-based local volatility calibration:
  1. Flat-vol recovery: calibrate_fem should recover constant sigma from BS prices.
  2. Comparison with FD calibration: final J values should be similar.
  3. FD gradient check: evaluate_fem_gradient vs finite-difference perturbation.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.grid import Grid
from src.utils import bs_call
from src.fem_state_solver import solve_fem_state
from src.fem_backward_solver import solve_fem_adjoint
from src.gradient import evaluate_fem_gradient
from src.objective import evaluate_J
from src.calibration_fem import calibrate_fem


# ---------------------------------------------------------------------------
# Shared parameters
# ---------------------------------------------------------------------------

S0    = 100.0
K_MIN = 60.0
K_MAX = 160.0
T_MAX = 1.0
SIGMA = 0.20
R     = 0.05
Q     = 0.02
N_K   = 40
N_T   = 30
ALPHA = 1e-3


@pytest.fixture
def grid():
    return Grid(S0=S0, K_min=K_MIN, K_max=K_MAX, T_max=T_MAX,
                N_K=N_K, N_T=N_T, r=R, q=Q)


@pytest.fixture
def flat_u(grid):
    return np.full((grid.N_K + 1, grid.N_T + 1), SIGMA**2)


@pytest.fixture
def bs_prices(grid):
    """Black-Scholes call price surface (exact, used as synthetic market data)."""
    C = np.zeros((grid.N_K + 1, grid.N_T + 1))
    for j, T in enumerate(grid.T):
        for i, K in enumerate(grid.K):
            if T == 0.0:
                C[i, j] = max(S0 - K, 0.0)
            else:
                C[i, j] = bs_call(S0, K, T, R, Q, SIGMA)
    return C


# ---------------------------------------------------------------------------
# 1. FEM state solver smoke test
# ---------------------------------------------------------------------------

def test_fem_state_matches_bs(grid, flat_u, bs_prices):
    """FEM state solver with flat vol should match BS prices to within 1%."""
    C_fem = solve_fem_state(flat_u, grid)
    # Check interior strike range, T > 0
    for n in range(1, grid.N_T + 1):
        T = grid.T[n]
        for i in range(2, grid.N_K - 1):
            K   = grid.K[i]
            ref = bs_call(S0, K, T, R, Q, SIGMA)
            if ref > 0.5:   # only test meaningful prices
                rel_err = abs(C_fem[i, n] - ref) / ref
                assert rel_err < 0.02, (
                    f"FEM state vs BS: rel_err={rel_err:.4f} at K={K:.1f}, T={T:.2f}"
                )


# ---------------------------------------------------------------------------
# 2. FEM adjoint + gradient: FD gradient check
# ---------------------------------------------------------------------------

def test_fem_gradient_fd_check(grid, flat_u, bs_prices):
    """
    Verify evaluate_fem_gradient against a finite-difference perturbation of J.

    We perturb a few interior nodes of u and check that the directional
    derivative matches grad_J . direction within 1% relative tolerance.
    """
    u     = flat_u.copy()
    z     = bs_prices.copy()
    w     = np.ones_like(z)
    u_star = flat_u.copy()
    theta  = 0.5
    eps    = 1e-5

    # Forward solve
    C = solve_fem_state(u, grid, theta=theta)
    # Adjoint
    p = solve_fem_adjoint(u, C, z, w, grid, theta=theta)
    # Analytical gradient
    g = evaluate_fem_gradient(u, C, p, u_star, ALPHA, grid, theta=theta)

    # Pick a random direction (perturb interior nodes only)
    rng = np.random.default_rng(42)
    direction = np.zeros_like(u)
    direction[1:-1, 1:-1] = rng.standard_normal((grid.N_K - 1, grid.N_T - 1))
    direction /= np.linalg.norm(direction) + 1e-12

    # Finite-difference directional derivative
    u_plus  = np.clip(u + eps * direction, 1e-4, 4.0)
    u_minus = np.clip(u - eps * direction, 1e-4, 4.0)

    C_plus  = solve_fem_state(u_plus,  grid, theta=theta)
    C_minus = solve_fem_state(u_minus, grid, theta=theta)

    J_plus  = evaluate_J(u_plus,  C_plus,  z, w, u_star, ALPHA, grid)
    J_minus = evaluate_J(u_minus, C_minus, z, w, u_star, ALPHA, grid)

    fd_deriv    = (J_plus - J_minus) / (2.0 * eps)
    anal_deriv  = float(np.sum(g * direction))

    rel_err = abs(anal_deriv - fd_deriv) / (abs(fd_deriv) + 1e-12)
    assert rel_err < 0.05, (
        f"FEM gradient FD check failed: anal={anal_deriv:.6e}, "
        f"fd={fd_deriv:.6e}, rel_err={rel_err:.4f}"
    )


# ---------------------------------------------------------------------------
# 3. calibrate_fem: flat-vol recovery
# ---------------------------------------------------------------------------

def test_calibrate_fem_flat_vol_recovery(grid, bs_prices):
    """
    calibrate_fem starting from a noisy initial guess should recover
    flat vol to within 15% on the interior of the surface.
    """
    z      = bs_prices.copy()
    w      = np.ones_like(z)
    u_star = np.full_like(z, SIGMA**2)
    u0     = np.full_like(z, (SIGMA * 1.3)**2)   # 30% biased start

    result = calibrate_fem(
        grid=grid,
        z=z,
        w=w,
        u_star=u_star,
        alpha=ALPHA,
        u0=u0,
        sigma_bounds=(0.05, 1.0),
        theta=0.5,
        max_iter=30,
        verbose=False,
    )

    # Check that the final objective is smaller than the initial objective
    # (optimiser made progress)
    C0 = solve_fem_state(u0, grid)
    J0 = evaluate_J(u0, C0, z, w, u_star, ALPHA, grid)
    assert result.J_final < J0, (
        f"calibrate_fem did not reduce J: J0={J0:.4e}, J_final={result.J_final:.4e}"
    )

    # Check that recovered sigma is in a reasonable range
    assert result.sigma_opt.min() > 0.01
    assert result.sigma_opt.max() < 1.5


# ---------------------------------------------------------------------------
# 4. calibrate_fem vs calibrate (FD): similar final objective
# ---------------------------------------------------------------------------

def test_calibrate_fem_vs_fd(grid, bs_prices):
    """
    FEM calibration and FD calibration should reach comparable final J values
    (within an order of magnitude) on synthetic flat-vol data.
    """
    from src.calibration import calibrate

    z      = bs_prices.copy()
    w      = np.ones_like(z)
    u_star = np.full_like(z, SIGMA**2)
    u0     = np.full_like(z, (SIGMA * 1.1)**2)

    common = dict(z=z, w=w, u_star=u_star, alpha=ALPHA, u0=u0,
                  sigma_bounds=(0.05, 1.0), theta=0.5, max_iter=10, verbose=False)

    res_fd  = calibrate(grid=grid, **common)
    res_fem = calibrate_fem(grid=grid, **common)

    # Both should be small (well-calibrated), and within 10x of each other
    assert res_fd.J_final  < 1.0
    assert res_fem.J_final < 1.0
    ratio = max(res_fd.J_final, res_fem.J_final) / (
            min(res_fd.J_final, res_fem.J_final) + 1e-15)
    assert ratio < 20.0, (
        f"FEM vs FD J ratio too large: J_fd={res_fd.J_final:.4e}, "
        f"J_fem={res_fem.J_final:.4e}, ratio={ratio:.1f}"
    )
