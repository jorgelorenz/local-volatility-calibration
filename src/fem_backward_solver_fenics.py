"""
fem_backward_solver_fenics.py
-----------------------------
FEniCS (legacy fenics-2019 / dolfin) solver for the local-volatility
backward pricing PDE.  Mirrors the public API of fem_backward_solver.py
exactly.

This module is only importable when a compatible FEniCS installation is
present.  The package __init__.py guards the import with try/except.

PDE (backward in time, forward in tau = T_final - t)
----------------------------------------------------
    dV/dtau = (1/2) sigma^2(S, T-tau) S^2 V_SS
              + (r(T-tau) - q(T-tau)) S V_S  -  r(T-tau) V

Weak / bilinear form (same sign convention as fem_backward_solver.py):

    a(sigma^2; V, v) = integral [
          (1/2) sigma^2 S^2  dV/dS dv/dS
        + (sigma^2 - (r-q)) S V dv/dS
        + r V v
    ] dS

Time scheme: theta-method (theta=0.5 => Crank-Nicolson).

Public API
----------
solve_fem_backward_fenics(sigma, nodes_S, t_grid, T_final,
                          r_func, q_func, payoff,
                          bc_left_func, bc_right_func, theta=0.5)
    -> V  shape (N_nodes, N_t)

solve_fem_backward_grid_fenics(sigma2, grid, K_strike, theta=0.5, nodes=None)
    -> V  shape (N_nodes, N_T+1)

Both signatures mirror their counterparts in fem_backward_solver.py.
"""

from __future__ import annotations
import numpy as np

try:
    import fenics as fe
    _FENICS_OK = True
except ImportError:
    _FENICS_OK = False

from .grid import Grid
from .utils import bs_call


def _require_fenics():
    if not _FENICS_OK:
        raise ImportError(
            "FEniCS (dolfin) is not available in this Python environment.  "
            "Run inside a FEniCS-enabled WSL/Docker environment."
        )


# ---------------------------------------------------------------------------
# Re-use mesh builder from forward FEniCS solver
# ---------------------------------------------------------------------------

def _build_mesh_and_space(nodes: np.ndarray):
    n_cells = len(nodes) - 1
    mesh = fe.IntervalMesh(n_cells, float(nodes[0]), float(nodes[-1]))
    coords = mesh.coordinates()
    t_uniform = np.linspace(0.0, 1.0, len(nodes))
    t_coords  = (coords[:, 0] - coords[0, 0]) / (coords[-1, 0] - coords[0, 0])
    coords[:, 0] = np.interp(t_coords, t_uniform, np.sort(nodes))
    V_cg1 = fe.FunctionSpace(mesh, "CG", 1)
    V_dg0 = fe.FunctionSpace(mesh, "DG", 0)
    return mesh, V_cg1, V_dg0


def _cg1_vec_from_nodal(fn, nodal_vals: np.ndarray) -> np.ndarray:
    V      = fn.function_space()
    coords = V.tabulate_dof_coordinates().flatten()
    nodes_sorted = np.sort(np.unique(coords))
    return np.interp(coords, nodes_sorted, nodal_vals)


def _array_to_dg0(arr_mid: np.ndarray, V_dg0) -> "fe.Function":
    f = fe.Function(V_dg0)
    f.vector()[:] = arr_mid
    return f


# ---------------------------------------------------------------------------
# General backward solver
# ---------------------------------------------------------------------------

def solve_fem_backward_fenics(
    sigma: np.ndarray,
    nodes_S: np.ndarray,
    t_grid: np.ndarray,
    T_final: float,
    r_func,
    q_func,
    payoff,
    bc_left_func,
    bc_right_func,
    theta: float = 0.5,
) -> np.ndarray:
    """
    FEniCS P1 FEM backward local-vol PDE solver.

    Parameters
    ----------
    sigma         : local volatility surface, shape (N_nodes, N_t)
    nodes_S       : asset-price node array, shape (N_nodes,)
    t_grid        : physical time, shape (N_t,), t_grid[0]=0, t_grid[-1]=T_final
    T_final       : maturity
    r_func        : callable r(t)->float or scalar
    q_func        : callable q(t)->float or scalar
    payoff        : callable(S)->array or array (N_nodes,)
    bc_left_func  : callable bc_left(t)->float
    bc_right_func : callable bc_right(t)->float
    theta         : implicitness (0.5=CN)

    Returns
    -------
    V : shape (N_nodes, N_t) -- V[:, -1]=payoff, V[:, 0]=price at t=0
    """
    _require_fenics()

    r_fn = (lambda t: float(r_func)) if not callable(r_func) else r_func
    q_fn = (lambda t: float(q_func)) if not callable(q_func) else q_func

    nodes_S = np.asarray(nodes_S, dtype=float)
    N_nodes = len(nodes_S)
    N_t     = len(t_grid)
    node_mids = 0.5 * (nodes_S[:-1] + nodes_S[1:])

    mesh, V_cg1, V_dg0 = _build_mesh_and_space(nodes_S)

    def bc_left_marker(x, on_boundary):
        return on_boundary and fe.near(x[0], nodes_S[0])

    def bc_right_marker(x, on_boundary):
        return on_boundary and fe.near(x[0], nodes_S[-1])

    # Terminal condition
    if callable(payoff):
        pay_vals = np.asarray(payoff(nodes_S), dtype=float)
    else:
        pay_vals = np.asarray(payoff, dtype=float)

    V_out = np.zeros((N_nodes, N_t))
    V_out[:, -1] = pay_vals

    # FEniCS solution at previous tau step (starts at terminal)
    V_n = fe.Function(V_cg1)
    V_n.vector()[:] = _cg1_vec_from_nodal(V_n, pay_vals)
    V_new = fe.Function(V_cg1)

    C_trial = fe.TrialFunction(V_cg1)
    v_test  = fe.TestFunction(V_cg1)

    # March from tau=0 to tau=T_final  (i.e. t from T_final down to 0)
    for n in range(N_t - 2, -1, -1):
        t_cur  = t_grid[n]
        t_next = t_grid[n + 1]
        dtau   = t_next - t_cur      # positive tau step

        t_mid  = 0.5 * (t_cur + t_next)
        r      = r_fn(t_mid)
        q      = q_fn(t_mid)

        # sigma^2 at midpoint time (average columns n and n+1), interpolated to elements
        sig2_mid_nodes = 0.5 * (sigma[:, n]**2 + sigma[:, n + 1]**2)
        sig2_mid_elems = np.interp(node_mids, nodes_S, sig2_mid_nodes)
        sig2_fn = _array_to_dg0(sig2_mid_elems, V_dg0)

        S_coord = fe.SpatialCoordinate(mesh)[0]

        # Bilinear form a(V, v):
        # a = integral [ 1/2 sig2 S^2 V_S v_S + (sig2-(r-q)) S V v_S + r V v ] dS
        a_form = (
            0.5 * sig2_fn * S_coord**2 * fe.dot(fe.grad(C_trial), fe.grad(v_test))
            + (sig2_fn - (r - q)) * S_coord * C_trial * v_test.dx(0)
            + r * C_trial * v_test
        ) * fe.dx

        m_form = C_trial * v_test * fe.dx

        # Theta scheme:  (M + theta*dtau*A) V_new = (M - (1-theta)*dtau*A) V_n
        lhs      = m_form + theta * dtau * a_form
        rhs_form = m_form - (1.0 - theta) * dtau * a_form

        A_mat = fe.assemble(lhs)
        b_vec = fe.assemble(fe.action(rhs_form, V_n))

        # Dirichlet BCs at physical time t_cur (tau = T_final - t_cur)
        bl_val = float(bc_left_func(t_cur))
        br_val = float(bc_right_func(t_cur))
        bc_l = fe.DirichletBC(V_cg1, fe.Constant(bl_val), bc_left_marker)
        bc_r = fe.DirichletBC(V_cg1, fe.Constant(br_val), bc_right_marker)
        for bc in [bc_l, bc_r]:
            bc.apply(A_mat, b_vec)

        fe.solve(A_mat, V_new.vector(), b_vec)

        # Extract sorted nodal values
        dof_coords = V_cg1.tabulate_dof_coordinates().flatten()
        sort_idx   = np.argsort(dof_coords)
        V_vals     = V_new.vector()[sort_idx]
        V_out[:, n] = np.interp(nodes_S, dof_coords[sort_idx], V_vals)

        V_n.assign(V_new)

    return V_out


# ---------------------------------------------------------------------------
# Grid-based convenience wrapper
# ---------------------------------------------------------------------------

def solve_fem_backward_grid_fenics(
    sigma2: np.ndarray,
    grid: Grid,
    K_strike: float,
    theta: float = 0.5,
    nodes: np.ndarray | None = None,
) -> np.ndarray:
    """
    Convenience wrapper matching solve_fem_backward_grid signature.

    Parameters
    ----------
    sigma2   : local variance sigma^2(S,T), shape (N_nodes, N_T+1)
    grid     : Grid instance
    K_strike : strike for European call payoff and right BC
    theta    : implicitness
    nodes    : FEM node array; None => grid.K

    Returns
    -------
    V : shape (N_nodes, N_T+1) -- V[:, -1]=payoff, V[:, 0]=price at t=0
    """
    if nodes is None:
        nodes = grid.K
    nodes = np.asarray(nodes, dtype=float)

    sigma = np.sqrt(np.maximum(sigma2, 0.0))  # shape (N_nodes, N_T+1)
    T_final = grid.T_max
    t_grid  = grid.T                          # shape (N_T+1,)

    payoff = np.maximum(nodes - K_strike, 0.0)

    def bc_left(t):
        return 0.0

    def bc_right(t):
        tau = T_final - t
        r   = grid.r_val(t)
        q   = grid.q_val(t)
        return nodes[-1] * np.exp(-q * tau) - K_strike * np.exp(-r * tau)

    return solve_fem_backward_fenics(
        sigma=sigma,
        nodes_S=nodes,
        t_grid=t_grid,
        T_final=T_final,
        r_func=grid.r_val,
        q_func=grid.q_val,
        payoff=payoff,
        bc_left_func=bc_left,
        bc_right_func=bc_right,
        theta=theta,
    )
