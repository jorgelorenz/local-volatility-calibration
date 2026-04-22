"""
tests/test_optimality.py
------------------------
Tests for:
  1. optimality_solver.py — all three backends (dct, lu, cg)
  2. calibration_oe.py — basic convergence on synthetic data
  3. IV misfit: misfit_iv_value and misfit_iv_source
"""

import sys
from pathlib import Path
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.grid import Grid
from src.optimality_solver import solve_optimality_system, _apply_operator
from src.objective import misfit_iv_value, misfit_iv_source, _bs_call_vec
from src.calibration_oe import calibrate_oe


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_grid(N_K=30, N_T=20):
    return Grid(S0=100.0, K_min=80.0, K_max=130.0, T_max=1.0,
                N_K=N_K, N_T=N_T, r=0.02, q=0.0)


# ---------------------------------------------------------------------------
# 1. optimality_solver: correctness of all backends
# ---------------------------------------------------------------------------

@pytest.fixture
def small_grid():
    return _make_grid(N_K=20, N_T=15)


def _random_rhs(grid, seed=42):
    rng = np.random.default_rng(seed)
    rhs = rng.standard_normal((grid.N_K + 1, grid.N_T + 1))
    return rhs


def _check_solution(x, rhs, alpha, grid, atol=1e-6):
    """Verify alpha*(-Delta+I) x ≈ rhs."""
    Ax = _apply_operator(x, alpha, grid)
    err = np.max(np.abs(Ax - rhs))
    return err


@pytest.mark.parametrize("method", ["dct", "lu", "cg"])
def test_oe_solver_residual(small_grid, method):
    """All backends should solve the system to reasonable accuracy."""
    alpha = 1e-3
    rhs = _random_rhs(small_grid)
    x = solve_optimality_system(rhs, alpha, small_grid, method=method)
    err = _check_solution(x, rhs, alpha, small_grid)
    assert err < 1e-5, f"method={method}: residual {err:.3e} too large"


def test_oe_solver_dct_lu_agree(small_grid):
    """DCT and LU should give nearly identical results."""
    alpha = 5e-4
    rhs = _random_rhs(small_grid, seed=7)
    cache = {}
    x_dct = solve_optimality_system(rhs, alpha, small_grid, method="dct")
    x_lu  = solve_optimality_system(rhs, alpha, small_grid, method="lu",
                                    _lu_cache=cache)
    diff = np.max(np.abs(x_dct - x_lu))
    assert diff < 1e-8, f"DCT vs LU disagreement: {diff:.3e}"


def test_oe_solver_lu_cache_reuse(small_grid):
    """LU cache should be populated and reused without error."""
    alpha = 1e-3
    rhs1 = _random_rhs(small_grid, seed=1)
    rhs2 = _random_rhs(small_grid, seed=2)
    cache = {}
    x1 = solve_optimality_system(rhs1, alpha, small_grid, method="lu",
                                  _lu_cache=cache)
    x2 = solve_optimality_system(rhs2, alpha, small_grid, method="lu",
                                  _lu_cache=cache)
    # Both should solve correctly
    assert _check_solution(x1, rhs1, alpha, small_grid) < 1e-5
    assert _check_solution(x2, rhs2, alpha, small_grid) < 1e-5


def test_oe_solver_unknown_method(small_grid):
    with pytest.raises(ValueError, match="Unknown OE solver"):
        solve_optimality_system(_random_rhs(small_grid), 1e-3, small_grid,
                                method="banana")


# ---------------------------------------------------------------------------
# 2. IV misfit: basic sanity checks
# ---------------------------------------------------------------------------

def test_misfit_iv_zero_at_true_prices():
    """When C = C_BS(iv_mkt), iv misfit should be ≈ 0."""
    grid = _make_grid(N_K=20, N_T=10)
    r_arr, q_arr = grid.rate_arrays()

    # Build a simple flat IV surface
    iv_mkt = np.full((grid.N_K + 1, grid.N_T + 1), 0.20)
    iv_mkt[:, 0] = np.nan   # T=0: no meaningful IV

    # Build C from BS using iv_mkt
    K2d, T2d = np.meshgrid(grid.K, grid.T, indexing="ij")
    r2d = np.tile(r_arr, (grid.N_K + 1, 1))
    q2d = np.tile(q_arr, (grid.N_K + 1, 1))
    C = _bs_call_vec(grid.S0, K2d, T2d, r2d, q2d, 0.20 * np.ones_like(K2d))
    C[:, 0] = grid.initial_condition()

    mf = misfit_iv_value(C, iv_mkt, grid, r_arr, q_arr)
    assert mf < 1e-6, f"IV misfit at true prices = {mf:.3e} (expected ≈ 0)"


def test_misfit_iv_source_shape():
    """IV source should have same shape as C and be zero at boundaries."""
    grid = _make_grid(N_K=15, N_T=10)
    r_arr, q_arr = grid.rate_arrays()
    iv_mkt = np.full((grid.N_K + 1, grid.N_T + 1), 0.20)
    iv_mkt[:, 0] = np.nan

    K2d, T2d = np.meshgrid(grid.K, grid.T, indexing="ij")
    r2d = np.tile(r_arr, (grid.N_K + 1, 1))
    q2d = np.tile(q_arr, (grid.N_K + 1, 1))
    C = _bs_call_vec(grid.S0, K2d, T2d, r2d, q2d, 0.25 * np.ones_like(K2d))

    src = misfit_iv_source(C, iv_mkt, grid, r_arr, q_arr)
    assert src.shape == C.shape
    assert np.all(src[:, 0] == 0.0)   # T=0
    assert np.all(src[0,  :] == 0.0)  # K_min boundary
    assert np.all(src[-1, :] == 0.0)  # K_max boundary


# ---------------------------------------------------------------------------
# 3. calibrate_oe: synthetic convergence
# ---------------------------------------------------------------------------

def _build_synthetic_data(grid, sigma_true=0.20):
    """Build z, w, iv_mkt from a flat true local vol."""
    from src.state_solver import solve_state
    from src.market_data import vega_weights

    u_true = np.full((grid.N_K + 1, grid.N_T + 1), sigma_true ** 2)
    C_true = solve_state(u_true, grid, theta=0.5)

    r_arr, q_arr = grid.rate_arrays()
    iv_mkt = np.full_like(C_true, sigma_true)
    iv_mkt[:, 0] = np.nan

    from src.market_data import iv_surface_to_call_prices
    z = C_true.copy()
    z[:, 0] = grid.initial_condition()
    w = vega_weights(iv_mkt, grid.K, grid.T, grid.S0, r_arr, q_arr)
    return z, w, iv_mkt, r_arr, q_arr


@pytest.mark.parametrize("misfit_type", ["price", "iv"])
def test_calibrate_oe_convergence_flat(misfit_type):
    """OE calibration should reduce J on flat vol synthetic data."""
    grid = _make_grid(N_K=25, N_T=18)
    z, w, iv_mkt, r_arr, q_arr = _build_synthetic_data(grid, sigma_true=0.20)

    sigma_prior = 0.22
    u_star = np.full_like(z, sigma_prior ** 2)

    kwargs = {}
    if misfit_type == "iv":
        kwargs = dict(iv_mkt=iv_mkt, r_arr=r_arr, q_arr=q_arr)

    result = calibrate_oe(
        grid=grid, z=z, w=w, u_star=u_star, alpha=1e-4,
        max_iter=30, tol=1e-8, step_size=1.0,
        oe_solver="dct", misfit_type=misfit_type,
        verbose=False, log_every=5,
        **kwargs
    )

    assert len(result.history["J"]) > 0
    # J should decrease from initial
    assert result.J_final < result.history["J"][0] + 1e-10
    assert result.sigma_opt.shape == z.shape
    assert np.all(result.sigma_opt > 0)


@pytest.mark.parametrize("oe_solver", ["dct", "lu", "cg"])
def test_calibrate_oe_solvers_agree(oe_solver):
    """All OE solvers should give similar J after 10 iterations."""
    grid = _make_grid(N_K=20, N_T=15)
    z, w, iv_mkt, r_arr, q_arr = _build_synthetic_data(grid)
    u_star = np.full_like(z, 0.22 ** 2)

    result = calibrate_oe(
        grid=grid, z=z, w=w, u_star=u_star, alpha=1e-4,
        max_iter=10, tol=1e-10, step_size=1.0,
        oe_solver=oe_solver, misfit_type="price",
        verbose=False, log_every=10,
    )
    assert result.J_final < 1e3  # sanity: not diverged
    assert result.sigma_opt.shape == z.shape


def test_calibrate_oe_history_sub_timings():
    """History should record sub-step timings."""
    grid = _make_grid(N_K=15, N_T=10)
    z, w, iv_mkt, r_arr, q_arr = _build_synthetic_data(grid)
    u_star = np.full_like(z, 0.22 ** 2)

    result = calibrate_oe(
        grid=grid, z=z, w=w, u_star=u_star, alpha=1e-4,
        max_iter=5, tol=1e-10, step_size=1.0,
        oe_solver="dct", misfit_type="price",
        verbose=False, log_every=1,
    )
    h = result.history
    assert "t_forward" in h and len(h["t_forward"]) > 0
    assert "t_adjoint" in h and len(h["t_adjoint"]) > 0
    assert "t_oe_solve" in h and len(h["t_oe_solve"]) > 0
    assert all(t >= 0 for t in h["t_forward"])
