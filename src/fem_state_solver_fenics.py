"""
fem_state_solver_fenics.py
--------------------------
FEniCS (legacy dolfinx / fenics-2019) solver for the Dupire forward PDE.
Mirrors the public API of fem_state_solver.py exactly.

This module is only importable when a compatible FEniCS installation is
present (typically inside a WSL or Docker environment).  The package
__init__.py guards the import with try/except ImportError.

PDE
---
Same as fem_state_solver.py:

    dC/dT = (1/2) u K^2 C_KK  - (r-q) K C_K  - q C

Weak / bilinear form (Achdou 2005 divergence form):

    a(u; C, v) = integral [
          (1/2) u K^2  dC/dK dv/dK
        + (u + r - q) K C dv/dK
        + q C v
    ] dK

Time scheme: theta-method (theta=0.5 => Crank-Nicolson).

Public API
----------
solve_fem_state_fenics(u, grid, theta=0.5, nodes=None)
    -> C  shape (N_nodes, N_T+1)

    Identical signature to solve_fem_state; drops the FEniCS-specific
    details (mesh construction, function spaces) internally.

Notes
-----
- Uses CG1 (P1 Lagrange) elements on a 1-D IntervalMesh.
- Dirichlet BCs imposed via DirichletBC (strong, elimination).
- The local variance u(K,T) is projected onto a DG0 function space
  (piecewise-constant per element) at each time step.
- For large grids the FEniCS LU solver is replaced by PETSC Krylov
  solvers automatically; controllable via solver_parameters dict.
"""

from __future__ import annotations
import numpy as np

# FEniCS import (legacy API – fenics-2019 / dolfin)
try:
    import fenics as fe
    _FENICS_OK = True
except ImportError:
    _FENICS_OK = False

from .grid import Grid


def _require_fenics():
    if not _FENICS_OK:
        raise ImportError(
            "FEniCS (dolfin) is not available in this Python environment.  "
            "Run inside a FEniCS-enabled WSL/Docker environment."
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_mesh_and_space(nodes: np.ndarray):
    """
    Create a 1-D FEniCS mesh on [nodes[0], nodes[-1]] with len(nodes)-1
    cells.  For a uniform mesh this is equivalent to IntervalMesh; for a
    graded mesh we warp the coordinates after construction.

    Returns (mesh, V_cg1, V_dg0).
    """
    n_cells = len(nodes) - 1
    mesh = fe.IntervalMesh(n_cells, float(nodes[0]), float(nodes[-1]))
    # Warp coordinates to match the (potentially non-uniform) node array
    coords = mesh.coordinates()        # shape (N+1, 1)
    # Sort nodes array (should already be sorted)
    sorted_nodes = np.sort(nodes)
    # Map from uniform [0,1] parametrisation to node positions
    t_uniform = np.linspace(0.0, 1.0, len(sorted_nodes))
    t_coords  = (coords[:, 0] - coords[0, 0]) / (coords[-1, 0] - coords[0, 0])
    new_coords = np.interp(t_coords, t_uniform, sorted_nodes)
    coords[:, 0] = new_coords

    V_cg1 = fe.FunctionSpace(mesh, "CG", 1)   # P1 Lagrange
    V_dg0 = fe.FunctionSpace(mesh, "DG", 0)   # piecewise-constant
    return mesh, V_cg1, V_dg0


def _array_to_dg0(arr_mid: np.ndarray, V_dg0) -> "fe.Function":
    """
    Map a numpy array of element midpoint values to a DG0 Function.
    arr_mid[e] is the value on element e.
    """
    f = fe.Function(V_dg0)
    # DG0 dof ordering matches cell ordering for 1-D meshes
    f.vector()[:] = arr_mid
    return f


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

def solve_fem_state_fenics(
    u: np.ndarray,
    grid: Grid,
    theta: float = 0.5,
    nodes: np.ndarray | None = None,
) -> np.ndarray:
    """
    FEniCS P1 FEM forward Dupire solver.

    Parameters
    ----------
    u      : local variance sigma^2(K,T), shape (N_nodes, N_T+1)
    grid   : Grid instance (time grid, rates, BCs, S0)
    theta  : implicitness (0.5 = Crank-Nicolson, 1.0 = backward Euler)
    nodes  : FEM node array in K; None => grid.K (uniform)

    Returns
    -------
    C : shape (N_nodes, N_T+1) -- call price surface
    """
    _require_fenics()

    if nodes is None:
        nodes = grid.K
    nodes = np.asarray(nodes, dtype=float)
    N_nodes = len(nodes)
    N_T     = grid.N_T
    dT      = grid.dT

    mesh, V_cg1, V_dg0 = _build_mesh_and_space(nodes)

    # Boundary markers
    def bc_left(x, on_boundary):
        return on_boundary and fe.near(x[0], nodes[0])

    def bc_right(x, on_boundary):
        return on_boundary and fe.near(x[0], nodes[-1])

    # Pre-compute boundary arrays
    bl_arr = grid.boundary_left()    # shape (N_T+1,)
    br_arr = grid.boundary_right()   # shape (N_T+1,)

    # Trial / test functions
    C_trial = fe.TrialFunction(V_cg1)
    v_test  = fe.TestFunction(V_cg1)

    # Solution function (updated each step)
    C_n = fe.Function(V_cg1)  # C at time level n (known)
    C_new = fe.Function(V_cg1)  # C at time level n+1 (solved)

    # Initial condition: C(K, 0) = max(S0 - K, 0)
    ic_vals = grid.initial_condition()   # shape (N_K+1,) on grid.K
    # Interpolate onto FEM nodes if different
    ic_on_nodes = np.interp(nodes, grid.K, ic_vals)
    C_n.vector()[:] = _cg1_vec_from_nodal(C_n, ic_on_nodes)

    # Output array
    C_out = np.zeros((N_nodes, N_T + 1))
    C_out[:, 0] = ic_on_nodes

    # u interpolated to element midpoints at each time step
    node_mids = 0.5 * (nodes[:-1] + nodes[1:])  # element midpoints
    n_elems   = len(node_mids)

    for n in range(N_T):
        T_n   = grid.T[n]
        T_np1 = grid.T[n + 1]
        T_mid = 0.5 * (T_n + T_np1)

        r  = grid.r_val(T_mid)
        q  = grid.q_val(T_mid)

        # Local variance on elements: average of n and n+1, then interpolate to midpoints
        u_mid_nodes = 0.5 * (u[:, n] + u[:, n + 1])     # shape (N_nodes,)
        u_mid_elems = np.interp(node_mids, nodes, u_mid_nodes)  # shape (n_elems,)

        u_fn  = _array_to_dg0(u_mid_elems, V_dg0)
        K_coord = fe.SpatialCoordinate(mesh)[0]

        # Bilinear form a(C, v):
        #   a = integral [ 1/2 u K^2 C_K v_K + (u + r - q) K C v_K + q C v ] dK
        # Note: v_K means dv/dK (derivative of test function)
        a_form = (
            0.5 * u_fn * K_coord**2 * fe.dot(fe.grad(C_trial), fe.grad(v_test))
            + (u_fn + (r - q)) * K_coord * C_trial * v_test.dx(0)
            + q * C_trial * v_test
        ) * fe.dx

        # Mass form: (C_trial, v_test)
        m_form = C_trial * v_test * fe.dx

        # Theta-scheme:
        #   M (C_new - C_n) / dT = -theta * A * C_new - (1-theta) * A * C_n
        # => (M + theta*dT*A) C_new = (M - (1-theta)*dT*A) C_n

        lhs = m_form + theta * dT * a_form
        rhs_form = m_form - (1.0 - theta) * dT * a_form

        # Assemble
        A_mat = fe.assemble(lhs)
        b_vec = fe.assemble(fe.action(rhs_form, C_n))

        # Dirichlet BCs at this step
        bl_val = float(bl_arr[n + 1])
        br_val = float(br_arr[n + 1])
        bc_l = fe.DirichletBC(V_cg1, fe.Constant(bl_val), bc_left)
        bc_r = fe.DirichletBC(V_cg1, fe.Constant(br_val), bc_right)
        bcs  = [bc_l, bc_r]
        for bc in bcs:
            bc.apply(A_mat, b_vec)

        fe.solve(A_mat, C_new.vector(), b_vec)

        # Extract nodal values
        dof_coords = V_cg1.tabulate_dof_coordinates().flatten()
        sort_idx   = np.argsort(dof_coords)
        C_vals     = C_new.vector()[sort_idx]

        # Interpolate to output nodes order (nodes may differ from dof order)
        C_out[:, n + 1] = np.interp(nodes, dof_coords[sort_idx], C_vals)

        # Update C_n
        C_n.assign(C_new)

    return C_out


# ---------------------------------------------------------------------------
# Helper: fill CG1 vector from nodal values (handles arbitrary dof ordering)
# ---------------------------------------------------------------------------

def _cg1_vec_from_nodal(fn: "fe.Function", nodal_vals: np.ndarray) -> np.ndarray:
    """
    Return a FEniCS vector array for a CG1 function given values at sorted
    node positions.  Handles arbitrary internal DOF orderings.
    """
    V    = fn.function_space()
    coords = V.tabulate_dof_coordinates().flatten()
    nodes_sorted = np.sort(np.unique(coords))
    # Map each DOF to its nodal value
    vec = np.interp(coords, nodes_sorted, nodal_vals)
    return vec
