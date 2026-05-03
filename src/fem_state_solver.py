"""
fem_state_solver.py
-------------------
Finite-Element Method (FEM) solver for the Dupire forward PDE.

PDE (state equation)
--------------------
  dC/dT = (1/2) u(K,T) K^2 d^2C/dK^2  -  (r(T)-q(T)) K dC/dK  -  q(T) C

  on Omega = (K_min, K_max) x (0, T_max)

  Initial condition : C(K, 0) = max(S0 - K, 0)
  Left  BC          : C(K_min, T) = S0*Bq(T) - K_min*Br(T)   (Dirichlet)
  Right BC          : C(K_max, T) = 0                          (Dirichlet)

where u(K,T) = sigma^2(K,T) is the local variance (control parameter).

FEM Formulation (following Achdou 2005 and subject notes)
---------------------------------------------------------
The spatial domain is partitioned into N elements using node array
`nodes` of length N+1.  On each element [K_i, K_{i+1}] we use
piecewise-linear (P1 / hat-function) basis functions phi_i.

Weak form (multiply PDE by test function v, integrate over K, integrate
by parts the second-order term):

  (dC/dT, v) = -a(u; C, v) + l(v)

where the bilinear form a and load l arise from the advection-diffusion
operator:

  a(u; C, v) = integral_Kmin^Kmax [
        (1/2) u(K,T) K^2  dC/dK dv/dK
      + (r-q) K C dv/dK            <-- advection (integrated by parts)
      + q C v
    ] dK

  (Natural Neumann BC at both ends is NOT imposed here; instead Dirichlet
  BCs are enforced by removing the first and last DOF from the system and
  incorporating their values into the RHS.)

Time discretisation: theta-scheme (theta=0.5 is Crank-Nicolson):

  M (C^{n+1} - C^n)/dT = -theta * A^{n+1/2} C^{n+1}
                          - (1-theta) * A^{n+1/2} C^n  + F_bc

where M is the mass matrix, A is the stiffness+advection+reaction matrix.

Rearranged as the linear system solved at each time step:

  (M + theta*dT*A) C^{n+1}_int = (M - (1-theta)*dT*A) C^n_int + rhs_bc

The matrices M and A are assembled element-by-element using 2-point Gauss
quadrature on each element.

Public API
----------
solve_fem_state(u, grid, theta=0.5, nodes=None)
    -> C  shape (len(nodes), N_T+1)

    u      : local variance sigma^2(K,T).
              If nodes is None (uniform mesh = grid.K): shape (N_K+1, N_T+1).
              If nodes is provided: shape (len(nodes), N_T+1), i.e. u must
              already be sampled on the FEM nodes.
    grid   : Grid instance (provides T, dT, rates, BCs, S0)
    theta  : implicitness parameter (0.5 = Crank-Nicolson)
    nodes  : 1D node array for the K-direction.
              None => use grid.K (uniform, matches FD grid for easy comparison)

    Returns C[i, n] = call price at (nodes[i], T_n).

Notes
-----
- Dirichlet BCs are imposed by elimination: the boundary DOFs (first and
  last rows/cols of the system) are removed from the solve and their
  prescribed values are moved to the RHS.
- The element integrals are computed analytically (exact for P1 basis with
  linearly varying coefficients) using the midpoint approximation for the
  local variance u(K,T) on each element at each time step.
- For validation with flat vol the result should match solve_state (FD)
  and bs_call to within ~1% in the interior of the strike range.
"""

from __future__ import annotations
import numpy as np
from scipy.linalg import solve_banded

from .grid import Grid


# ---------------------------------------------------------------------------
# Element-level assembly (P1, two nodes per element)
# ---------------------------------------------------------------------------

def _element_matrices(
    K_L: float, K_R: float,
    u_mid: float,
    r: float, q: float,
    q_reac: float | None = None,
    q_adv: float | None = None,
    adv_correction_sign: float = +1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the 2x2 element mass matrix M_e and stiffness matrix A_e for the
    element [K_L, K_R] with midpoint local variance u_mid = u((K_L+K_R)/2).

    P1 basis on [K_L, K_R]:
        phi_0(K) = (K_R - K) / h,   phi_1(K) = (K - K_L) / h
        dphi_0/dK = -1/h,            dphi_1/dK = +1/h
    where h = K_R - K_L.

    We use the midpoint rule (= 1-point Gauss) for the coefficients that
    vary with K (K^2, K), and exact integration for the phi_i phi_j products.

    Mass matrix M_e[i,j] = integral phi_i phi_j dK  (exact):
        M_e = (h/6) * [[2, 1], [1, 2]]

    Stiffness matrix A_e = diffusion + advection + reaction  (midpoint rule):

      The Dupire operator  L C = (1/2) u K^2 C_KK - (r-q) K C_K - q C
      is in NON-divergence form.  For FEM (IBP), we must rewrite in
      divergence form first:

          d/dK [ (1/2) u K^2 C_K ] = (1/2) u K^2 C_KK + (1/2)(u K^2)_K C_K

      At flat vol (u=const): (u K^2/2)_K = u K.
      For general u(K): (u(K) K^2 / 2)_K ≈ u_mid K_mid  (midpoint approx).

      So:  (1/2) u K^2 C_KK = d/dK[(1/2)u K^2 C_K] - u_mid K_mid C_K

      Substituting into L:
          L C = d/dK[(1/2)u K^2 C_K]
              - [u_mid K_mid + (r-q) K_mid] C_K
              - q C

      This is now in divergence + lower-order form.  The effective advection
      coefficient is: c_adv = -(u_mid K_mid + (r-q)*K_mid)
                             = -(u_mid + (r-q)) * K_mid

      Weak form (IBP of divergence term, Dirichlet BCs => no boundary terms):
          integral L C v dK
          = -integral (1/2) u K^2 C_K v_K dK
            - integral [u_mid K_mid + (r-q)*K_mid] C_K v dK
            - integral q C v dK

      Element matrices:
      (i) Stiffness (symmetric):
          A_diff[i,j] = integral (1/2) u_mid K_mid^2 dphi_j/dK dphi_i/dK dK
                      = u_mid K_mid^2 / (2h) * [[1,-1],[-1,1]]

      (ii) Effective advection (includes IBP correction term):
          c_adv = (u_mid + (r-q)) * K_mid   (with sign absorbed below)
          A_adv[i,j] = c_adv * (dphi_j/dK) * integral phi_i dK
                     = c_adv * (dphi_j/dK) * (h/2)
          dphi_0/dK=-1/h, dphi_1/dK=+1/h:
          A_adv[i,0] = -c_adv * K_mid/2 * 1
          A_adv[i,1] = +c_adv * K_mid/2 * 1
          -- wait: factor already in c_adv --
          A_adv[i,j] = c_adv * dphi_j/dK * h/2
          row i=0,1 both: A_adv[i,0]=-c_adv/2, A_adv[i,1]=+c_adv/2
          So A_adv = (c_adv/2) * [[-1,+1],[-1,+1]]

      (iii) Reaction:
          A_reac[i,j] = q * integral phi_i phi_j dK = q * M_e[i,j]

    Returns
    -------
    M_e : (2,2) mass matrix
    A_e : (2,2) stiffness+advection+reaction matrix

    Parameters q_reac and q_adv allow overriding the reaction and advection
    coefficients independently (used by the backward solver where reaction=r
    and advection=(r-q) but the function signature re-uses r, q).
    If None, defaults to q_reac=q and q_adv=(r-q).

    adv_correction_sign: +1 for forward Dupire (IBP correction adds u_mid K),
    -1 for backward BS PDE (IBP correction subtracts u_mid K).
    Forward Dupire divergence form:  c_adv = +(u_mid + q_adv) * K_mid
    Backward BS PDE divergence form: c_adv = +(q_adv - u_mid) * K_mid
    """
    if q_reac is None:
        q_reac = q
    if q_adv is None:
        q_adv = r - q
    h     = K_R - K_L
    K_mid = 0.5 * (K_L + K_R)

    # Mass matrix (exact for P1)
    M_e = (h / 6.0) * np.array([[2.0, 1.0], [1.0, 2.0]])

    # Diffusion (symmetric stiffness from IBP of divergence form)
    d = 0.5 * u_mid * K_mid**2 / h
    A_diff = d * np.array([[1.0, -1.0], [-1.0, 1.0]])

    # Effective advection: includes both IBP correction (u_mid*K_mid) and
    # original advection -(r-q)*K_mid.
    # For forward Dupire: q_adv = r-q, so c_adv = (u_mid + q_adv)*K_mid
    # For backward PDE:   q_adv = r-q, same formula applies.
    c_adv = (adv_correction_sign * u_mid + q_adv) * K_mid
    A_adv = (c_adv * 0.5) * np.array([[-1.0, +1.0], [-1.0, +1.0]])

    # Reaction: q_reac * ∫ C v dK
    A_reac = q_reac * M_e

    A_e = A_diff + A_adv + A_reac
    return M_e, A_e


# ---------------------------------------------------------------------------
# Global assembly
# ---------------------------------------------------------------------------

def _assemble(
    nodes: np.ndarray,
    u_nodes: np.ndarray,
    r: float,
    q: float,
    q_reac: float | None = None,
    q_adv: float | None = None,
    adv_correction_sign: float = +1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Assemble the global mass matrix M and stiffness matrix A on the mesh
    defined by `nodes` (shape N_nodes,).

    u_nodes : local variance at each node, shape (N_nodes,).
    q_reac  : reaction coefficient (defaults to q).
    q_adv   : advection coefficient (defaults to r-q).
    adv_correction_sign: +1 for forward Dupire, -1 for backward BS PDE.

    Returns M, A each of shape (N_nodes, N_nodes) as dense arrays.
    """
    N_nodes = len(nodes)
    M = np.zeros((N_nodes, N_nodes))
    A = np.zeros((N_nodes, N_nodes))

    for e in range(N_nodes - 1):
        K_L = nodes[e]
        K_R = nodes[e + 1]
        u_mid = 0.5 * (u_nodes[e] + u_nodes[e + 1])  # midpoint approximation

        M_e, A_e = _element_matrices(K_L, K_R, u_mid, r, q,
                                     q_reac=q_reac, q_adv=q_adv,
                                     adv_correction_sign=adv_correction_sign)

        idx = [e, e + 1]
        for i_loc, i_glob in enumerate(idx):
            for j_loc, j_glob in enumerate(idx):
                M[i_glob, j_glob] += M_e[i_loc, j_loc]
                A[i_glob, j_glob] += A_e[i_loc, j_loc]

    return M, A


# ---------------------------------------------------------------------------
# Tridiagonal utilities (M and A are tridiagonal for P1)
# ---------------------------------------------------------------------------

def _tri_to_banded(mat: np.ndarray) -> np.ndarray:
    """
    Pack a tridiagonal matrix into scipy solve_banded (1,1) banded format.
    Shape of output: (3, N).
    """
    N = mat.shape[0]
    ab = np.zeros((3, N))
    ab[1, :]   = np.diag(mat, 0)
    ab[0, 1:]  = np.diag(mat, 1)   # super-diagonal
    ab[2, :-1] = np.diag(mat, -1)  # sub-diagonal
    return ab


def _apply_tri(mat: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Multiply tridiagonal matrix by vector using the 3-diagonal structure."""
    diag = np.diag(mat, 0)
    sup  = np.diag(mat, 1)
    sub  = np.diag(mat, -1)
    out  = diag * v
    out[:-1] += sup * v[1:]
    out[1:]  += sub * v[:-1]
    return out


# ---------------------------------------------------------------------------
# Public solver
# ---------------------------------------------------------------------------

def solve_fem_state(
    u: np.ndarray,
    grid: Grid,
    theta: float = 0.5,
    nodes: np.ndarray | None = None,
) -> np.ndarray:
    """
    Solve the Dupire forward PDE using P1 FEM in strike, theta-scheme in time.

    Parameters
    ----------
    u      : local variance sigma^2(K,T).
             Shape (N_nodes, N_T+1) if nodes is provided,
             or (N_K+1, N_T+1) if nodes is None (uses grid.K).
    grid   : Grid instance
    theta  : implicitness (0.5 = Crank-Nicolson, 1.0 = backward Euler)
    nodes  : FEM node positions along K-axis.
             None  => uniform mesh grid.K (N_K+1 nodes).
             array => custom (possibly non-uniform) node array.

    Returns
    -------
    C : shape (N_nodes, N_T+1)
        C[i, n] = call price at strike nodes[i] and maturity T_n.
    """
    if nodes is None:
        nodes = grid.K.copy()

    N_nodes = len(nodes)
    N_T     = grid.N_T
    T       = grid.T
    dT      = grid.dT

    # Rates
    r_arr, q_arr = grid.rate_arrays()   # shape (N_T+1,)

    # BCs: Dirichlet at nodes[0] and nodes[-1]
    # Left BC:  S0*Bq(T) - K_min*Br(T)
    # Right BC: 0
    # We recompute BCs on the actual K values of the FEM boundary nodes.
    # Use grid's discount factors (which depend only on T, not K).
    K_left  = nodes[0]
    K_right = nodes[-1]

    def bc_left(n: int) -> float:
        Tn = T[n]
        Bq = grid.discount_q(Tn)
        Br = grid.discount_r(Tn)
        return grid.S0 * Bq - K_left * Br

    def bc_right(n: int) -> float:
        return 0.0

    # Allocate solution
    C = np.zeros((N_nodes, N_T + 1))

    # Initial condition: max(S0 - K, 0) at T=0
    C[:, 0] = np.maximum(grid.S0 - nodes, 0.0)
    C[0,  0] = bc_left(0)
    C[-1, 0] = bc_right(0)

    # Time march
    for n in range(N_T):
        T_mid = 0.5 * (T[n] + T[n + 1])
        r = float(grid.r_val(T_mid))
        q = float(grid.q_val(T_mid))

        # Average local variance at each node (CN in time)
        u_nodes = 0.5 * (u[:, n] + u[:, n + 1])

        # Assemble global M and A
        M, A = _assemble(nodes, u_nodes, r, q)

        # System matrices (interior DOFs only: indices 1..N_nodes-2)
        # LHS: (M + theta*dT*A)_int
        # RHS: (M - (1-theta)*dT*A)_int * C_int^n  +  bc_corrections
        LHS_full = M + theta * dT * A
        RHS_mat  = M - (1.0 - theta) * dT * A

        C_int_n = C[1:-1, n]

        # RHS contribution from interior DOFs
        rhs = _apply_tri(RHS_mat[1:-1, 1:-1], C_int_n)

        # BC corrections: columns 0 and -1 of LHS and RHS_mat applied to BC values
        bc_l_next = bc_left(n + 1)
        bc_r_next = bc_right(n + 1)
        bc_l_cur  = bc_left(n)
        bc_r_cur  = bc_right(n)

        # Contribution from left BC column (column 0 of the interior rows)
        lhs_left_col = LHS_full[1:-1, 0]   # shape (N_int,)
        lhs_right_col = LHS_full[1:-1, -1]
        rhs_left_col  = RHS_mat[1:-1, 0]
        rhs_right_col = RHS_mat[1:-1, -1]

        rhs -= lhs_left_col  * bc_l_next
        rhs -= lhs_right_col * bc_r_next
        rhs += rhs_left_col  * bc_l_cur
        rhs += rhs_right_col * bc_r_cur

        # Extract interior block of LHS and solve
        LHS_int = LHS_full[1:-1, 1:-1]
        ab = _tri_to_banded(LHS_int)
        C_int_new = solve_banded((1, 1), ab, rhs)

        # Store
        C[0,  n + 1] = bc_l_next
        C[-1, n + 1] = bc_r_next
        C[1:-1, n + 1] = C_int_new

    return C
