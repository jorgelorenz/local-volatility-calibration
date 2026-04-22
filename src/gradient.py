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

Public API
----------
evaluate_gradient(u, C, p, u_star, alpha, grid, theta=0.5)
    -> grad_J : np.ndarray shape (N_K+1, N_T+1)
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
