"""
fem_mesh.py
-----------
Mesh generation utilities for the 1D FEM solvers (strike / asset-price axis).

All solvers in fem_state_solver.py and fem_backward_solver.py accept a 1D
node array `nodes` produced by one of the functions below.  This decouples
mesh construction from the PDE assembly.

Available mesh types
--------------------
uniform_mesh(a, b, N)
    Uniform partition of [a, b] into N elements (N+1 nodes).

graded_mesh(a, b, N, center, width, ratio)
    Concentrates nodes near `center` (e.g. ATM strike or current asset price)
    by blending a uniform mesh with a logistic grading function.
    `ratio`  in (0, 1]: fraction of the N nodes allocated to the region
    [center-width, center+width].  ratio=1 puts all refinement at center.

bisection_refine(nodes, indicator)
    Adaptive refinement: bisects each element where indicator[i] > threshold.
    indicator is a per-element array of length len(nodes)-1.
    Returns a new (finer) sorted node array.

make_mesh(kind, a, b, N, **kwargs)
    Convenience factory: kind in {"uniform", "graded"}.
"""

from __future__ import annotations
import numpy as np


# ---------------------------------------------------------------------------
# Uniform mesh
# ---------------------------------------------------------------------------

def uniform_mesh(a: float, b: float, N: int) -> np.ndarray:
    """
    Return N+1 uniformly spaced nodes on [a, b].

    Parameters
    ----------
    a, b : endpoints of the interval  (a < b)
    N    : number of elements  (number of nodes = N+1)

    Returns
    -------
    nodes : shape (N+1,), dtype float64
    """
    return np.linspace(a, b, N + 1)


# ---------------------------------------------------------------------------
# Graded mesh  (concentration near ATM / current price)
# ---------------------------------------------------------------------------

def graded_mesh(
    a: float,
    b: float,
    N: int,
    center: float | None = None,
    width: float | None = None,
    ratio: float = 0.5,
) -> np.ndarray:
    """
    Graded mesh on [a, b] with node concentration near `center`.

    The construction maps a uniform parameter t in [0, 1] through a smooth
    grading function g(t) that stretches the spacing away from the center
    region.  Specifically:

        phi(t) = t + ratio * f(t)

    where f(t) is a bump function supported near t_c = (center - a)/(b - a),
    then phi is normalised to [0,1] and mapped to [a, b].

    For `ratio=0` the result is a uniform mesh.  For `ratio` close to 1
    roughly half the nodes are placed within [center-width, center+width].

    Parameters
    ----------
    a, b     : interval endpoints
    N        : number of elements  (N+1 nodes returned)
    center   : point of refinement; defaults to midpoint (a+b)/2
    width    : half-width of the refined region; defaults to (b-a)/6
    ratio    : controls refinement strength in [0, 1)

    Returns
    -------
    nodes : shape (N+1,) sorted array of node coordinates
    """
    if center is None:
        center = 0.5 * (a + b)
    if width is None:
        width = (b - a) / 6.0

    ratio = float(np.clip(ratio, 0.0, 0.99))

    t = np.linspace(0.0, 1.0, N + 1)
    t_c = (center - a) / (b - a)
    w   = width / (b - a)

    # Smooth bump: positive on (t_c - w, t_c + w), zero outside
    # Use a quartic bell: bump(s) = max(0, 1 - (s/w)^2)^2
    s = (t - t_c) / max(w, 1e-12)
    bump = np.maximum(0.0, 1.0 - s**2)**2
    # Normalise bump so integral over [0,1] ≈ 1
    bump_integral = np.trapz(bump, t)
    if bump_integral > 1e-14:
        bump /= bump_integral

    # Grading: build CDF of density rho(t) = 1 + ratio * bump(t).
    # rho is large near t_c => the inverse CDF places nodes densely there.
    # Compute cumulative integral (CDF) of rho, then interpolate the inverse:
    # uniform phi-values [0,1] -> t-values -> x-values.
    rho = 1.0 + ratio * bump
    phi = np.cumsum(rho) * (t[1] - t[0])  # forward CDF (left-Riemann)
    phi -= phi[0]
    phi /= phi[-1]   # normalise to [0, 1]

    # Nodes placed at uniform intervals of phi => invert CDF
    phi_uniform = np.linspace(0.0, 1.0, N + 1)
    t_nodes = np.interp(phi_uniform, phi, t)   # inverse CDF

    nodes = a + t_nodes * (b - a)
    nodes[0]  = a
    nodes[-1] = b
    return nodes


# ---------------------------------------------------------------------------
# Bisection refinement  (adaptive)
# ---------------------------------------------------------------------------

def bisection_refine(
    nodes: np.ndarray,
    indicator: np.ndarray,
    threshold: float | None = None,
) -> np.ndarray:
    """
    Refine a 1D mesh by bisecting elements where indicator > threshold.

    Parameters
    ----------
    nodes     : sorted node array, shape (M+1,)
    indicator : per-element refinement indicator, shape (M,)
    threshold : bisect element i if indicator[i] > threshold.
                Defaults to mean(indicator).

    Returns
    -------
    new_nodes : sorted node array (finer)
    """
    nodes = np.asarray(nodes, dtype=float)
    indicator = np.asarray(indicator, dtype=float)
    if threshold is None:
        threshold = float(np.mean(indicator))

    new_nodes = list(nodes)
    for i, val in enumerate(indicator):
        if val > threshold:
            midpoint = 0.5 * (nodes[i] + nodes[i + 1])
            new_nodes.append(midpoint)

    return np.sort(np.unique(new_nodes))


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def make_mesh(
    kind: str,
    a: float,
    b: float,
    N: int,
    **kwargs,
) -> np.ndarray:
    """
    Factory for named mesh types.

    Parameters
    ----------
    kind : "uniform" | "graded"
    a, b : interval endpoints
    N    : number of elements
    **kwargs : passed to the specific mesh function

    Returns
    -------
    nodes : shape (N+1,)  (or larger if graded adds extra nodes)
    """
    kind = kind.lower()
    if kind == "uniform":
        return uniform_mesh(a, b, N)
    elif kind == "graded":
        return graded_mesh(a, b, N, **kwargs)
    else:
        raise ValueError(f"Unknown mesh kind: {kind!r}. Choose 'uniform' or 'graded'.")
