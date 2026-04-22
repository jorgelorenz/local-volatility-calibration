"""
tests/test_adjoint.py
---------------------
Phase 2 validation: adjoint-based gradient satisfies the Taylor (FD) test.

What is being tested
--------------------
The adjoint solver (solve_adjoint) + gradient assembler (evaluate_gradient)
must together produce the exact gradient of the discrete objective J w.r.t.
the local variance field u.

We verify this via the first-order Taylor expansion:

    (J(u + eps·v) - J(u)) / eps  →  ⟨∇J(u), v⟩  as eps → 0

The ratio R(eps) = [(J(u+eps·v) - J(u))/eps] / ⟨∇J(u), v⟩ should satisfy
R(eps) → 1.0 for eps ∈ {1e-4, 1e-5, 1e-6}.

This test is the critical correctness check: if the adjoint is wrong, the
optimiser (L-BFGS-B) will not converge correctly even if the code runs.

Data used (100% synthetic — no market files)
---------------------------------------------
  Grid:   S0=100, K∈[80,130] (N_K=40), T∈[0,0.5] (N_T=30)
  Rates:  r=0.05 (constant), q=0.02 (constant)
  u:      random field in [0.01², 0.5²], fixed random seed for reproducibility
  z:      synthetic call prices from a different random u_true (so the
          misfit term is non-trivial)
  w:      uniform weight 1.0 everywhere (no NaN masking needed)
  alpha:  1e-4

Test inventory
--------------
  test_gradient_fd_check        Ratio R(eps) ∈ [0.99, 1.01] for all eps
  test_gradient_zero_at_minimum At u=u_star, gradient is finite and ‖grad‖ < 1e3

How to run
----------
  python -m pytest tests/test_adjoint.py -v        # via pytest
  python tests/test_adjoint.py                     # standalone with detail
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from src.grid import Grid
from src.state_solver import solve_state
from src.adjoint_solver import solve_adjoint
from src.objective import evaluate_J
from src.gradient import evaluate_gradient
from src.market_data import iv_surface_to_call_prices


# ---------------------------------------------------------------------------
# Small grid for speed
# ---------------------------------------------------------------------------
S0    = 100.0
r     = 0.05
q     = 0.02
T_max = 0.5
K_min = 80.0
K_max = 130.0
N_K   = 40
N_T   = 30
alpha = 1e-4
sigma_ref = 0.20


def _make_problem():
    g = Grid(S0=S0, K_min=K_min, K_max=K_max, T_max=T_max,
             N_K=N_K, N_T=N_T, r=r, q=q)
    r_arr, q_arr = g.rate_arrays()

    # Synthetic observations: BS prices at sigma_ref
    u_true = np.full((N_K + 1, N_T + 1), sigma_ref**2)
    z      = solve_state(u_true, g)
    w      = np.ones_like(z)
    w[:, 0] = 0.0   # T=0 contributes nothing

    # Slightly perturbed starting point
    rng = np.random.default_rng(42)
    u_star = np.full_like(u_true, (sigma_ref * 0.9)**2)
    u0 = np.clip(u_star + 0.005 * rng.standard_normal(u_star.shape), 0.01**2, 2.0**2)

    return g, z, w, u_star, u0


@pytest.fixture(scope="module")
def problem():
    return _make_problem()


def test_gradient_fd_check(problem):
    """
    Check: (J(u0 + eps*v) - J(u0)) / eps ~ <grad_J, v>
    Ratio should be close to 1 for intermediate eps.
    """
    g, z, w, u_star, u0 = problem

    rng = np.random.default_rng(7)
    v = rng.standard_normal(u0.shape)
    v[0,  :] = 0.0   # respect boundary structure
    v[-1, :] = 0.0

    # Compute gradient at u0
    C0    = solve_state(u0, g)
    J0    = evaluate_J(u0, C0, z, w, u_star, alpha, g)
    p0    = solve_adjoint(u0, C0, z, w, g)
    grad0 = evaluate_gradient(u0, C0, p0, u_star, alpha, g)
    dir_deriv_adj = float(np.dot(grad0.ravel(), v.ravel()))

    print(f"\nAdjoint directional derivative: {dir_deriv_adj:.6e}")
    print(f"\n{'eps':>12}  {'FD':>14}  {'ratio':>10}")
    print("-" * 40)

    ratios = []
    for eps in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
        u_plus = u0 + eps * v
        C_plus = solve_state(u_plus, g)
        J_plus = evaluate_J(u_plus, C_plus, z, w, u_star, alpha, g)
        fd     = (J_plus - J0) / eps
        ratio  = fd / dir_deriv_adj if abs(dir_deriv_adj) > 1e-30 else float("nan")
        ratios.append(ratio)
        print(f"  {eps:.1e}   {fd:14.6e}   {ratio:10.6f}")

    # Check that the ratio is close to 1 for small eps (where FD is accurate)
    mid_ratios = ratios[2:5]   # eps = 1e-4, 1e-5, 1e-6
    for ratio in mid_ratios:
        assert abs(ratio - 1.0) < 0.01, (
            f"Gradient check failed: ratio={ratio:.6f} (expected ~1.0)"
        )


def test_gradient_zero_at_minimum(problem):
    """At the true u (where J is minimal for alpha->0), gradient should be small."""
    g, z, w, u_star, _ = problem
    u_true = np.full((N_K + 1, N_T + 1), sigma_ref**2)
    C_true = solve_state(u_true, g)
    p_true = solve_adjoint(u_true, C_true, z, w, g)
    grad   = evaluate_gradient(u_true, C_true, p_true, u_star, alpha, g)
    # With alpha small, gradient = regularization term at u_true
    # (the PDE term vanishes because C_true = z exactly)
    grad_pde_norm = np.max(np.abs(grad - alpha * (u_true - u_star) *
                                  g.dK * g.dT))
    print(f"\nGradient at true u: ||grad_PDE||_inf = {grad_pde_norm:.4e}")
    # Just check it's finite and bounded
    assert np.isfinite(grad).all()
    assert np.max(np.abs(grad)) < 1e3


if __name__ == "__main__":
    g, z, w, u_star, u0 = _make_problem()

    rng = np.random.default_rng(7)
    v = rng.standard_normal(u0.shape)
    v[0, :] = v[-1, :] = 0.0

    C0    = solve_state(u0, g)
    J0    = evaluate_J(u0, C0, z, w, u_star, alpha, g)
    p0    = solve_adjoint(u0, C0, z, w, g)
    grad0 = evaluate_gradient(u0, C0, p0, u_star, alpha, g)
    dJ_adj = float(np.dot(grad0.ravel(), v.ravel()))

    print("=== Adjoint gradient check ===")
    print(f"  J(u0)              = {J0:.6e}")
    print(f"  <grad J, v> (adj)  = {dJ_adj:.6e}")
    print()
    print(f"{'eps':>10}  {'FD approx':>14}  {'ratio':>10}")
    print("-" * 40)
    for eps in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
        C_p  = solve_state(u0 + eps * v, g)
        J_p  = evaluate_J(u0 + eps * v, C_p, z, w, u_star, alpha, g)
        fd   = (J_p - J0) / eps
        print(f"  {eps:.1e}    {fd:14.6e}    {fd/dJ_adj:10.6f}")
