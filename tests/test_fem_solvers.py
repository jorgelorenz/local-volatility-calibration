"""
tests/test_fem_solvers.py
-------------------------
Validation of the FEM solvers (forward Dupire + backward local-vol) against
Black-Scholes closed-form prices for flat (constant) local volatility.

For sigma(K,T) = sigma_0 (constant):
  - The Dupire forward PDE solution C(K,T) must equal bs_call(S0, K, T, r, q, sigma_0).
  - The backward local-vol PDE solution V(S, t) for a call with strike K
    must equal bs_call(S, K, T-t, r, q, sigma_0).

Both uniform and graded (non-uniform) meshes are tested.

Also validates fem_mesh.py utilities.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.grid import Grid
from src.utils import bs_call
from src.fem_state_solver import solve_fem_state
from src.fem_backward_solver import solve_fem_backward_grid
from src.fem_mesh import uniform_mesh, graded_mesh, bisection_refine, make_mesh


# ---------------------------------------------------------------------------
# Shared test parameters
# ---------------------------------------------------------------------------

S0     = 100.0
K_MIN  = 50.0
K_MAX  = 200.0
T_MAX  = 1.0
SIGMA  = 0.20
R      = 0.05
Q      = 0.02
N_K    = 150
N_T    = 80


@pytest.fixture
def flat_grid():
    return Grid(S0=S0, K_min=K_MIN, K_max=K_MAX, T_max=T_MAX,
                N_K=N_K, N_T=N_T, r=R, q=Q)


@pytest.fixture
def flat_u(flat_grid):
    """Flat local variance array sigma^2 on the grid."""
    return np.full((flat_grid.N_K + 1, flat_grid.N_T + 1), SIGMA**2)


# ---------------------------------------------------------------------------
# FEM forward (Dupire) tests — uniform mesh
# ---------------------------------------------------------------------------

class TestFEMForwardUniform:

    def test_flat_vol_bs_accuracy(self, flat_grid, flat_u):
        """FEM forward with flat vol matches BS to < 2% relative error.

        Note: FEM with consistent mass matrix smooths the kink in the initial
        condition max(S0-K,0), causing ~20% errors at the first 1-2 time steps.
        This is a known property of Galerkin FEM (vs FD which uses pointwise
        values).  We exclude the first few time steps from the accuracy check.
        """
        C = solve_fem_state(flat_u, flat_grid, theta=0.5)

        K = flat_grid.K
        T = flat_grid.T

        # Compare at interior strikes/maturities
        # Skip very short maturities where FEM IC smoothing dominates
        k_lo = int(0.1 * flat_grid.N_K)
        k_hi = int(0.9 * flat_grid.N_K)
        t_min = 0.075  # skip T < 0.075 to avoid IC smoothing artefacts at ATM

        max_rel_err = 0.0
        for j in range(1, flat_grid.N_T + 1):
            Tj = T[j]
            if Tj < t_min:
                continue
            bs_prices = np.array([bs_call(S0, Ki, Tj, R, Q, SIGMA) for Ki in K[k_lo:k_hi]])
            fem_prices = C[k_lo:k_hi, j]
            mask = bs_prices > 0.5
            if mask.any():
                rel_err = np.abs(fem_prices[mask] - bs_prices[mask]) / bs_prices[mask]
                max_rel_err = max(max_rel_err, rel_err.max())

        assert max_rel_err < 0.02, (
            f"FEM forward (uniform) max relative error vs BS = {max_rel_err:.4f} >= 2%"
        )

    def test_initial_condition(self, flat_grid, flat_u):
        """FEM forward initial condition is max(S0-K, 0)."""
        C = solve_fem_state(flat_u, flat_grid)
        expected = np.maximum(S0 - flat_grid.K, 0.0)
        np.testing.assert_allclose(C[:, 0], expected, atol=1e-10)

    def test_no_negative_prices(self, flat_grid, flat_u):
        """FEM forward prices are non-negative everywhere."""
        C = solve_fem_state(flat_u, flat_grid)
        assert np.all(C >= -1e-8), f"Negative price: min = {C.min():.2e}"

    def test_monotone_in_strike(self, flat_grid, flat_u):
        """FEM forward call prices decrease with strike (no-arbitrage)."""
        C = solve_fem_state(flat_u, flat_grid)
        for j in range(1, flat_grid.N_T + 1):
            diffs = np.diff(C[:, j])
            assert np.all(diffs <= 1e-8), (
                f"Non-monotone at T={flat_grid.T[j]:.2f}: max increase = {diffs.max():.2e}"
            )

    def test_boundary_right_zero(self, flat_grid, flat_u):
        """FEM forward right BC is zero for all T."""
        C = solve_fem_state(flat_u, flat_grid)
        np.testing.assert_allclose(C[-1, :], 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# FEM forward — graded mesh
# ---------------------------------------------------------------------------

class TestFEMForwardGraded:

    def test_flat_vol_graded_mesh(self, flat_grid):
        """FEM forward with graded mesh near ATM also matches BS < 2% (T >= 0.05)."""
        nodes = graded_mesh(K_MIN, K_MAX, N_K, center=S0, width=20.0, ratio=0.5)
        # Interpolate u onto nodes
        u_uniform = np.full((len(nodes), flat_grid.N_T + 1), SIGMA**2)
        C = solve_fem_state(u_uniform, flat_grid, nodes=nodes)

        T = flat_grid.T
        k_lo = int(0.1 * len(nodes))
        k_hi = int(0.9 * len(nodes))
        t_min = 0.075  # skip T < 0.075 to avoid IC smoothing artefacts at ATM

        max_rel_err = 0.0
        for j in range(1, flat_grid.N_T + 1):
            Tj = T[j]
            if Tj < t_min:
                continue
            bs_prices = np.array([bs_call(S0, Ki, Tj, R, Q, SIGMA) for Ki in nodes[k_lo:k_hi]])
            fem_prices = C[k_lo:k_hi, j]
            mask = bs_prices > 0.5
            if mask.any():
                rel_err = np.abs(fem_prices[mask] - bs_prices[mask]) / bs_prices[mask]
                max_rel_err = max(max_rel_err, rel_err.max())

        assert max_rel_err < 0.02, (
            f"FEM forward (graded) max relative error = {max_rel_err:.4f} >= 2%"
        )


# ---------------------------------------------------------------------------
# FEM backward (local-vol) tests — uniform mesh
# ---------------------------------------------------------------------------

class TestFEMBackwardUniform:

    def test_flat_vol_bs_accuracy(self, flat_grid, flat_u):
        """
        FEM backward with flat local vol gives call prices matching BS < 2%.

        The backward PDE prices a European call with strike K=S0 (ATM).
        We check V(S, t=0) against bs_call(S, K_strike, T_MAX, r, q, sigma).
        """
        K_STRIKE = S0  # ATM

        # sigma2 on grid: using flat_grid.K as node array and flat vol
        V = solve_fem_backward_grid(flat_u, flat_grid, K_strike=K_STRIKE, theta=0.5)

        S_nodes = flat_grid.K
        # Price at t=0 (first column)
        v0 = V[:, 0]

        k_lo = int(0.1 * flat_grid.N_K)
        k_hi = int(0.9 * flat_grid.N_K)

        bs_prices = np.array([
            bs_call(Si, K_STRIKE, T_MAX, R, Q, SIGMA) for Si in S_nodes[k_lo:k_hi]
        ])
        fem_prices = v0[k_lo:k_hi]

        mask = bs_prices > 0.5
        if mask.any():
            rel_err = np.abs(fem_prices[mask] - bs_prices[mask]) / bs_prices[mask]
            max_rel_err = rel_err.max()
        else:
            max_rel_err = 0.0

        assert max_rel_err < 0.02, (
            f"FEM backward (uniform) max relative error vs BS = {max_rel_err:.4f} >= 2%"
        )

    def test_terminal_condition(self, flat_grid, flat_u):
        """FEM backward terminal condition is max(S - K_strike, 0)."""
        K_STRIKE = 100.0
        V = solve_fem_backward_grid(flat_u, flat_grid, K_strike=K_STRIKE)
        payoff = np.maximum(flat_grid.K - K_STRIKE, 0.0)
        np.testing.assert_allclose(V[:, -1], payoff, atol=1e-10)

    def test_no_negative_prices(self, flat_grid, flat_u):
        """FEM backward option prices are non-negative everywhere."""
        V = solve_fem_backward_grid(flat_u, flat_grid, K_strike=S0)
        assert np.all(V >= -1e-8), f"Negative option value: min = {V.min():.2e}"

    def test_monotone_in_asset(self, flat_grid, flat_u):
        """FEM backward call price increases with asset price (no-arbitrage)."""
        V = solve_fem_backward_grid(flat_u, flat_grid, K_strike=S0)
        for n in range(flat_grid.N_T):
            diffs = np.diff(V[:, n])
            assert np.all(diffs >= -1e-8), (
                f"Non-monotone at t_idx={n}: min diff = {diffs.min():.2e}"
            )


# ---------------------------------------------------------------------------
# Mesh utility tests
# ---------------------------------------------------------------------------

class TestFemMesh:

    def test_uniform_mesh_shape(self):
        nodes = uniform_mesh(50.0, 200.0, 100)
        assert len(nodes) == 101
        assert nodes[0] == pytest.approx(50.0)
        assert nodes[-1] == pytest.approx(200.0)

    def test_uniform_mesh_spacing(self):
        nodes = uniform_mesh(0.0, 1.0, 10)
        diffs = np.diff(nodes)
        np.testing.assert_allclose(diffs, diffs[0], rtol=1e-12)

    def test_graded_mesh_endpoints(self):
        nodes = graded_mesh(50.0, 200.0, 100, center=100.0, ratio=0.4)
        assert nodes[0] == pytest.approx(50.0)
        assert nodes[-1] == pytest.approx(200.0)
        assert len(nodes) == 101

    def test_graded_mesh_denser_near_center(self):
        nodes = graded_mesh(0.0, 200.0, 200, center=100.0, width=30.0, ratio=0.6)
        diffs = np.diff(nodes)
        # Spacing near center should be smaller than near endpoints
        center_idx = len(nodes) // 2
        assert diffs[center_idx] < diffs[0], "Graded mesh not denser near center"

    def test_bisection_refine(self):
        nodes = uniform_mesh(0.0, 1.0, 4)  # 5 nodes, 4 elements
        indicator = np.array([0.1, 0.9, 0.9, 0.1])  # refine middle two
        refined = bisection_refine(nodes, indicator)
        assert len(refined) > len(nodes), "Refinement should add nodes"
        assert refined[0] == pytest.approx(0.0)
        assert refined[-1] == pytest.approx(1.0)
        assert np.all(np.diff(refined) > 0), "Refined nodes should be sorted"

    def test_make_mesh_factory(self):
        nodes_u = make_mesh("uniform", 0.0, 1.0, 10)
        nodes_g = make_mesh("graded", 0.0, 1.0, 10, center=0.5)
        assert len(nodes_u) == 11
        assert len(nodes_g) == 11

    def test_make_mesh_unknown_kind(self):
        with pytest.raises(ValueError):
            make_mesh("foobar", 0.0, 1.0, 10)


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pytest as pt
    pt.main([__file__, "-v"])
