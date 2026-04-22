"""
regularization.py
-----------------
Tikhonov regularization for the local variance parameter u = sigma^2.

We use an H^1(Omega) semi-norm regularization:

  R(u) = (alpha/2) * integral_Omega [ (u - u*)^2  +  |grad(u - u*)|^2 ] dK dT

Discretized on the (N_K+1) x (N_T+1) grid with trapezoidal weights:

  R_h(u) = (alpha/2) * dK * dT * sum_{i,n} w_{i,n} *
               [ (u_{i,n} - u*_{i,n})^2
                 + ((u_{i+1,n} - u_{i,n})/dK)^2
                 + ((u_{i,n+1} - u_{i,n})/dT)^2 ]

The gradient of R_h with respect to u_{i,n} is derived analytically.

Public API
----------
tikhonov_value(u, u_star, alpha, grid)  -> float
tikhonov_gradient(u, u_star, alpha, grid)  -> np.ndarray (same shape as u)
"""

from __future__ import annotations
import numpy as np
from .grid import Grid


def tikhonov_value(
    u: np.ndarray,
    u_star: np.ndarray,
    alpha: float,
    grid: Grid,
) -> float:
    """
    Evaluate the Tikhonov regularization term R(u).

    Parameters
    ----------
    u, u_star : local variance arrays, shape (N_K+1, N_T+1)
    alpha     : regularization parameter
    grid      : Grid instance

    Returns
    -------
    R : scalar float
    """
    dK = grid.dK
    dT = grid.dT
    d  = u - u_star

    # L^2 part
    L2 = np.sum(d**2) * dK * dT

    # dK gradient (forward differences in K direction)
    dK_grad = ((d[1:, :] - d[:-1, :]) / dK)**2
    GK = np.sum(dK_grad) * dK * dT

    # dT gradient (forward differences in T direction)
    dT_grad = ((d[:, 1:] - d[:, :-1]) / dT)**2
    GT = np.sum(dT_grad) * dK * dT

    return float(0.5 * alpha * (L2 + GK + GT))


def tikhonov_gradient(
    u: np.ndarray,
    u_star: np.ndarray,
    alpha: float,
    grid: Grid,
) -> np.ndarray:
    """
    Gradient of the Tikhonov regularization term with respect to u.

    grad_R[i,n] = alpha * dK * dT *
        [  (u[i,n] - u*[i,n])
           - (u[i+1,n]-u[i,n])/dK^2 + (u[i,n]-u[i-1,n])/dK^2     (in K)
           - (u[i,n+1]-u[i,n])/dT^2 + (u[i,n]-u[i,n-1])/dT^2 ]   (in T)

    (Discrete negative Laplacian of d = u - u* with homogeneous Neumann BCs.)

    Returns
    -------
    grad_R : shape (N_K+1, N_T+1)
    """
    dK  = grid.dK
    dT  = grid.dT
    dK2 = dK * dK
    dT2 = dT * dT
    d   = u - u_star

    # L^2 contribution
    g = d.copy()

    # K-direction: negative Laplacian via second differences
    # Interior: -Laplace_K d[i] = (2*d[i] - d[i-1] - d[i+1]) / dK^2
    lap_K = np.zeros_like(d)
    lap_K[1:-1, :] = (2.0*d[1:-1, :] - d[:-2, :] - d[2:, :]) / dK2
    # Neumann at boundaries: one-sided
    lap_K[0,  :] = (d[0,  :] - d[1,  :]) / dK2
    lap_K[-1, :] = (d[-1, :] - d[-2, :]) / dK2

    # T-direction: negative Laplacian via second differences
    lap_T = np.zeros_like(d)
    lap_T[:, 1:-1] = (2.0*d[:, 1:-1] - d[:, :-2] - d[:, 2:]) / dT2
    lap_T[:, 0 ] = (d[:, 0 ] - d[:, 1 ]) / dT2
    lap_T[:, -1] = (d[:, -1] - d[:, -2]) / dT2

    grad_R = alpha * dK * dT * (g + lap_K + lap_T)
    return grad_R
