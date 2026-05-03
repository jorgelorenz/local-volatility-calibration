"""
gradient.py
-----------
Gradient of the reduced objective J_alpha(u) via the discrete adjoint.

After solving the state (forward) and adjoint (backward) equations, the
gradient w.r.t. each local variance node u[i,n] is assembled from
contributions of all time steps that depend on u[i,n].

Since u_col^n = (u[:,n] + u[:,n+1]) / 2 is used in step n -> n+1, node
u[i,n] contributes to:
  - step n-1 -> n  (if n >= 1)  via u_col^{n-1}[i]  with weight 1/2
  - step n   -> n+1 (if n < N_T) via u_col^n[i]     with weight 1/2

For step n -> n+1, the gradient w.r.t. u_col[i] (interior i = 1..N_K-1,
affecting interior row j = i-1) is:

  dJ/d(u_col^n[i])  =  (p^{n+1}_{int})^T  *  d(RHS - A*C^{n+1})/du_col[i]
                     =  p^{n+1}_{int}[j]  *  (dB^n/du_i * C^n - dA^n/du_i * C^{n+1})_j

where

  (dA^n/du_i * C^{n+1})_j = theta*dT * (K[i]^2/(2dK^2)) * (C[i-1,n+1] - 2C[i,n+1] + C[i+1,n+1])
  (dB^n/du_i * C^n)_j     = -(1-theta)*dT * (K[i]^2/(2dK^2)) * (C[i-1,n] - 2C[i,n] + C[i+1,n])

Note: the adjoint p already incorporates the dK*dT integration weights (they
were included in the source term dJ/dC in the adjoint solver).

FEM gradient
------------
evaluate_fem_gradient uses the FEM discrete adjoint.  For P1 elements, the
derivative of the stiffness matrix A w.r.t. u at element midpoint e (shared
by nodes i and i+1) is:

  dA_e/d(u_mid_e) = dA_diff_e/d(u_mid_e)
                  = K_mid_e^2 / (2*h_e) * [[1,-1],[-1,1]]

The gradient w.r.t. u[i,n] accumulates over the two adjacent elements (e=i-1
and e=i) and two time levels (n-1->n and n->n+1) via the CN averaging.

Public API
----------
evaluate_gradient(u, C, p, u_star, alpha, grid, theta=0.5)
    -> grad_J : np.ndarray shape (N_K+1, N_T+1)

evaluate_fem_gradient(u, C, p, u_star, alpha, grid, theta=0.5, nodes=None)
    -> grad_J : np.ndarray shape (N_nodes, N_T+1)
"""

from __future__ import annotations
import numpy as np
from .grid import Grid
from .regularization import tikhonov_gradient


def evaluate_gradient(
    u: np.ndarray,
    C: np.ndarray,
    p: np.ndarray,
    u_star: np.ndarray,
    alpha: float,
    grid: Grid,
    theta: float = 0.5,
) -> np.ndarray:
    """
    Gradient of J_alpha w.r.t. u via the discrete adjoint.

    Parameters
    ----------
    u      : local variance, shape (N_K+1, N_T+1)
    C      : model call prices (state solution), shape (N_K+1, N_T+1)
    p      : adjoint variable from solve_adjoint, shape (N_K+1, N_T+1)
    u_star : prior local variance, shape (N_K+1, N_T+1)
    alpha  : regularization parameter
    grid   : Grid instance
    theta  : theta-scheme parameter (must match state/adjoint solvers)

    Returns
    -------
    grad_J : shape (N_K+1, N_T+1)
    """
    K   = grid.K
    dK  = grid.dK
    dT  = grid.dT
    N_K = grid.N_K
    N_T = grid.N_T

    grad_pde = np.zeros((N_K + 1, N_T + 1))

    # For each time step n -> n+1, accumulate gradient w.r.t. u_col[i] = (u[i,n]+u[i,n+1])/2
    # for i = 1,...,N_K-1.  Row j = i-1 in the interior system.
    for n in range(N_T):
        # Interior nodes (full-array indices i=1..N_K-1, interior-system row j=i-1=0..N_K-2)
        # K values at these nodes
        ii = np.arange(1, N_K)      # shape (N_K-1,)
        Ki = K[ii]                   # shape (N_K-1,)
        da_du = Ki**2 / (2.0 * dK**2)  # d(alpha_i)/d(u_col[i])

        # Second differences of C at time n+1 and n, centered at i:
        # (C[i-1] - 2*C[i] + C[i+1])  in full array
        C_next = C[:, n + 1]
        C_cur  = C[:, n]

        sec_diff_next = C_next[:-2] - 2.0 * C_next[1:-1] + C_next[2:]   # shape (N_K-1,)
        sec_diff_cur  = C_cur[:-2]  - 2.0 * C_cur[1:-1]  + C_cur[2:]    # shape (N_K-1,)

        # dA/du_col[i] * C^{n+1} contribution to row j=i-1:
        dA_C = theta * dT * da_du * sec_diff_next

        # dB/du_col[i] * C^n contribution to row j=i-1:
        dB_C = -(1.0 - theta) * dT * da_du * sec_diff_cur

        # Adjoint at interior row j=i-1 at time n+1:
        p_next = p[1:-1, n + 1]  # shape (N_K-1,)

        # Gradient of J w.r.t. u_col[i]:
        # dJ/d(u_col[i]) = p^{n+1}[j] * (dB_C[j] - dA_C[j])
        # = p_next[i-1] * (dB_C[i-1] - dA_C[i-1])
        contrib = p_next * (dB_C - dA_C)   # shape (N_K-1,)

        # Chain rule: d(u_col[i])/d(u[i,n]) = 0.5 and
        #             d(u_col[i])/d(u[i,n+1]) = 0.5
        grad_pde[1:-1, n]     += 0.5 * contrib
        grad_pde[1:-1, n + 1] += 0.5 * contrib

    # Regularization gradient
    grad_reg = tikhonov_gradient(u, u_star, alpha, grid)

    grad_J = grad_pde + grad_reg

    # Zero out boundary nodes (not optimized)
    grad_J[0,  :] = 0.0
    grad_J[-1, :] = 0.0

    return grad_J


def evaluate_fem_gradient(
    u: np.ndarray,
    C: np.ndarray,
    p: np.ndarray,
    u_star: np.ndarray,
    alpha: float,
    grid: Grid,
    theta: float = 0.5,
    nodes: np.ndarray | None = None,
) -> np.ndarray:
    """
    FEM gradient of J_alpha w.r.t. u via the discrete adjoint.

    For P1 FEM with element e spanning [nodes[e], nodes[e+1]], the
    stiffness matrix A_e depends on u_mid_e = (u[e,n] + u[e+1,n])/2.
    The derivative of A_e w.r.t. u_mid_e is:

        dA_diff_e / d(u_mid_e) = K_mid_e^2 / (2*h_e) * [[1,-1],[-1,1]]

    The gradient of J w.r.t. u[i,n] sums contributions from all elements
    adjacent to node i (elements e=i-1 and e=i), at time steps n-1->n and
    n->n+1 (via the CN half-weight).

    For each time step n -> n+1 and each element e:
        dL/d(u_mid_e) = dT * p^{n+1} . (dA_e/du_mid) . (theta*C^{n+1} + (1-theta)*C^n)

    where dA_e/du_mid has two contributions:
        dA_diff/du_mid = K_mid^2/(2*h_e) * [[1,-1],[-1,1]]
        dA_adv /du_mid = K_mid/2          * [[-1,+1],[-1,+1]]

    The +sign on (1-theta)*C^n comes from differentiating the Lagrangian:
        d(LHS)/du = +theta*dT*dA,  d(RHS)/du = -(1-theta)*dT*dA
        => -d(RHS)/du * C^n = +(1-theta)*dT*dA*C^n

    Chain rule: d(u_mid_e)/d(u[e,n]) = d(u_mid_e)/d(u[e+1,n]) = 0.25
    (0.5 from spatial avg over element nodes, 0.5 from CN time avg)

    Parameters
    ----------
    u      : local variance, shape (N_nodes, N_T+1)
    C      : FEM state solution, shape (N_nodes, N_T+1)
    p      : FEM adjoint solution, shape (N_nodes, N_T+1)
    u_star : prior local variance, shape (N_nodes, N_T+1)
    alpha  : regularization parameter
    grid   : Grid instance
    theta  : theta-scheme parameter (must match state/adjoint solvers)
    nodes  : FEM node array; None => grid.K

    Returns
    -------
    grad_J : shape (N_nodes, N_T+1)
    """
    if nodes is None:
        nodes = grid.K.copy()

    N_nodes = len(nodes)
    N_T     = grid.N_T
    dT      = grid.dT

    grad_pde = np.zeros((N_nodes, N_T + 1))

    for n in range(N_T):
        # Loop over elements e = 0 .. N_nodes-2
        for e in range(N_nodes - 1):
            h_e    = nodes[e + 1] - nodes[e]
            K_mid  = 0.5 * (nodes[e] + nodes[e + 1])

            # Derivative of A_e w.r.t. u_mid_e (element midpoint variance):
            #
            # dA_diff_e/du_mid = K_mid^2/(2*h) * [[1,-1],[-1,1]]
            # dA_adv_e/du_mid  = K_mid^2/2     * [[-1,+1],[-1,+1]]
            #                    (from IBP correction: c_adv = (u_mid + q_adv)*K_mid,
            #                     dc_adv/du_mid = K_mid, A_adv = c_adv/2*[[-1,+1],[-1,+1]])
            #
            # So (dA_e/du_mid) * v   (element vector action, indices 0=node e, 1=node e+1):
            #   row 0: K_mid^2/(2h)*(v0-v1) + K_mid^2/2*(-v0+v1)
            #   row 1: K_mid^2/(2h)*(v1-v0) + K_mid^2/2*(-v0+v1)
            #
            # In matrix form with x = (C[e], C[e+1]):
            #   (dA_e/du) * x = [  K_mid^2/(2h)*(C[e]-C[e+1]) - K_mid^2/2*(C[e]-C[e+1]),
            #                      K_mid^2/(2h)*(C[e+1]-C[e]) - K_mid^2/2*(C[e]-C[e+1]) ]
            #
            # Let dC = C[e] - C[e+1]:
            #   (dA_e/du) * x = K_mid^2 * dC * [ 1/(2h) - 1/2,
            #                                     -1/(2h) - 1/2 ]

            # Effective C for this step: from Lagrangian differentiation,
            #   d(LHS)/du = +theta*dT*dA,  d(RHS)/du = -(1-theta)*dT*dA
            #   dL/du_mid = p^{n+1} . dA_e/du_mid . (theta*C^{n+1} + (1-theta)*C^n)
            Ce_eff  = theta * C[e,     n + 1] + (1.0 - theta) * C[e,     n]
            Ce1_eff = theta * C[e + 1, n + 1] + (1.0 - theta) * C[e + 1, n]

            # (dA_e/du_mid) @ Ceff:
            #   dA_diff/du_mid = K_mid^2/(2h) * [[1,-1],[-1,1]]
            #   dA_adv /du_mid = K_mid/2       * [[-1,+1],[-1,+1]]
            #     (because c_adv = (u_mid + q_adv)*K_mid => dc_adv/du_mid = K_mid,
            #      and A_adv = (c_adv/2)*[[-1,+1],[-1,+1]])
            dC_eff  = Ce_eff - Ce1_eff
            r0 = K_mid**2 / (2.0 * h_e) * dC_eff + K_mid / 2.0 * (-Ce_eff  + Ce1_eff)
            r1 = K_mid**2 / (2.0 * h_e) * (-dC_eff) + K_mid / 2.0 * (-Ce_eff  + Ce1_eff)

            pe0 = p[e,     n + 1]
            pe1 = p[e + 1, n + 1]

            total_contrib = dT * (pe0 * r0 + pe1 * r1)

            # Chain rule:
            #   u_mid_e = 0.5*(u[e]+u[e+1])  (spatial average over element nodes)
            #   u_avg[i] = 0.5*(u[i,n]+u[i,n+1])  (CN time average)
            #   So d(u_mid_e)/d(u[e,n]) = d(u_mid_e)/d(u[e+1,n]) = 0.5 * 0.5 = 0.25
            grad_pde[e,     n]     += 0.25 * total_contrib
            grad_pde[e + 1, n]     += 0.25 * total_contrib
            grad_pde[e,     n + 1] += 0.25 * total_contrib
            grad_pde[e + 1, n + 1] += 0.25 * total_contrib

    # Regularization (uses grid.dK and grid.dT, which match the uniform mesh;
    # for custom nodes the regularization is still computed on the uniform grid
    # since u_star and u have shape matching nodes when nodes=grid.K).
    # We use a node-spacing-aware version for non-uniform meshes.
    if nodes is grid.K or np.allclose(nodes, grid.K):
        grad_reg = tikhonov_gradient(u, u_star, alpha, grid)
    else:
        # Generic: finite-difference regularization using actual node spacings
        grad_reg = _fem_tikhonov_gradient(u, u_star, alpha, nodes, dT)

    grad_J = grad_pde + grad_reg

    # Zero out boundary nodes (not optimized)
    grad_J[0,  :] = 0.0
    grad_J[-1, :] = 0.0

    return grad_J


def _fem_tikhonov_gradient(
    u: np.ndarray,
    u_star: np.ndarray,
    alpha: float,
    nodes: np.ndarray,
    dT: float,
) -> np.ndarray:
    """
    H^1 Tikhonov gradient for a non-uniform node mesh.

    Uses trapezoidal weights in K and uniform dT in time.
    """
    N_nodes, N_T1 = u.shape
    d = u - u_star

    # Trapezoidal weights for K integration
    h = np.zeros(N_nodes)
    h[0]    = 0.5 * (nodes[1] - nodes[0])
    h[-1]   = 0.5 * (nodes[-1] - nodes[-2])
    h[1:-1] = 0.5 * (nodes[2:] - nodes[:-2])

    # L^2 contribution: alpha * h_i * dT * d[i,n]
    g = alpha * dT * h[:, None] * d

    # K-direction gradient: finite differences with local spacing
    lap_K = np.zeros_like(d)
    for i in range(1, N_nodes - 1):
        hL = nodes[i]     - nodes[i - 1]
        hR = nodes[i + 1] - nodes[i]
        lap_K[i, :] = alpha * dT * h[i] * (
            -(d[i + 1, :] - d[i, :]) / hR + (d[i, :] - d[i - 1, :]) / hL
        ) / (0.5 * (hL + hR))
    lap_K[0,  :] = 0.0
    lap_K[-1, :] = 0.0

    # T-direction gradient: uniform spacing
    dT2    = dT * dT
    lap_T  = np.zeros_like(d)
    lap_T[:, 1:-1] = alpha * dT * h[:, None] * (
        2.0 * d[:, 1:-1] - d[:, :-2] - d[:, 2:]
    ) / dT2
    lap_T[:, 0 ] = alpha * dT * h * (d[:, 0]  - d[:, 1])  / dT2
    lap_T[:, -1] = alpha * dT * h * (d[:, -1] - d[:, -2]) / dT2

    return g + lap_K + lap_T
