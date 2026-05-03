"""
optimality_solver.py
--------------------
Solver for the Tikhonov preconditioner system arising in the optimality-
equation (OE) calibration algorithm.

Problem
-------
The discrete first-order optimality condition for the calibration problem is:

    grad_J(u) = grad_pde(u) + grad_R(u) = 0

where grad_R(u) = alpha * (-Delta_h + I)(u - u_star)  (discrete H^1 gradient).

Rearranging as a Newton-like step delta_u = u_new - u:

    alpha * (-Delta_h + I) delta_u = - grad_J(u)
    =>  (-Delta_h + I) delta_u = -grad_J(u) / alpha

This is a 2D discrete Poisson equation on the (N_K+1) x (N_T+1) grid with
homogeneous Neumann boundary conditions.

The operator  L = -Delta_h + I  with one-sided Neumann BCs decomposes as a
tensor product:
    L = L_K (x) I_T + I_K (x) L_T + I_{KT}

Three solver backends are provided, selectable at runtime:

  "dct" (default):
      Uses 2D DCT-II to diagonalise the operator exactly.

      The 1D Neumann Laplacian with one-sided boundary differences:
          row 0   : (u_0 - u_1) / h^2
          interior: (2*u_i - u_{i-1} - u_{i+1}) / h^2
          row N   : (u_N - u_{N-1}) / h^2

      is a symmetric tridiagonal matrix whose eigenvectors are exactly the
      DCT-II basis vectors cos(pi*k*(2i+1)/(2*(N+1))) and whose eigenvalues
      are the analytic formula:

          lambda_k = 2*(1 - cos(pi*k/(N+1))) / h^2,   k = 0,...,N

      (This is NOT the same as DCT-I, which corresponds to a different
      symmetric stencil with mirrored BCs.  DCT-II is correct here because
      the one-sided stencil makes the boundary row coefficients 1/h^2 on the
      diagonal instead of 2/h^2, which is equivalent to a half-cell shift —
      exactly the geometry for which DCT-II is the eigenbasis.)

      In 2D the operator decomposes as a Kronecker product, so the 2D solve
      reduces to:
          1. Apply 2D DCT-II to rhs  ->  rhs_hat
          2. Divide pointwise by  alpha*(1 + lambda_K[i] + lambda_T[j])
          3. Apply 2D inverse DCT-II (= normalised DCT-III)  ->  solution

      Complexity: O(N log N) per solve, no matrix assembly or factorisation.

  "lu":
      Assembles the sparse matrix L explicitly and factorises it once with
      scipy.sparse.linalg.splu.  The factorisation is cached per (grid, alpha)
      pair so repeated solves with the same operator are O(N).
      Setup cost: O(N^1.5) for a 2D sparse matrix.

  "cg":
      Conjugate Gradient with a Jacobi (diagonal) preconditioner.
      Matrix-free: applies (-Delta_h + I) directly.  Cost: O(sqrt(kappa)*N)
      per solve where kappa is the condition number.

Public API
----------
solve_optimality_system(rhs, alpha, grid, method="dct",
                        cg_tol=1e-8, cg_maxiter=None,
                        _lu_cache=None)
    -> delta_u : np.ndarray shape (N_K+1, N_T+1)

_lu_cache is an optional mutable dict used to persist the LU factorisation
across repeated calls; pass the same dict every call to avoid recomputation.
"""

from __future__ import annotations
import numpy as np
from .grid import Grid


# ---------------------------------------------------------------------------
# Matrix-free operator (shared by CG and DCT eigenvalue computation)
# ---------------------------------------------------------------------------

def _apply_operator(x: np.ndarray, alpha: float, grid: Grid) -> np.ndarray:
    """
    Matrix-free application of alpha*(-Delta_h + I) x on the 2D grid.

    Uses one-sided Neumann BCs:
      boundary rows: (-Delta u)_0 = (u_0 - u_1) / h^2   (main=1/h^2)
      interior rows: (-Delta u)_i = (2*u_i - u_{i-1} - u_{i+1}) / h^2
    """
    dK2 = grid.dK**2
    dT2 = grid.dT**2

    # -Delta_K x  (one-sided Neumann)
    lap_K = np.zeros_like(x)
    lap_K[1:-1, :] = (2.0*x[1:-1, :] - x[:-2, :] - x[2:, :]) / dK2
    lap_K[0,  :]   = (x[0, :] - x[1, :]) / dK2
    lap_K[-1, :]   = (x[-1, :] - x[-2, :]) / dK2

    # -Delta_T x  (one-sided Neumann)
    lap_T = np.zeros_like(x)
    lap_T[:, 1:-1] = (2.0*x[:, 1:-1] - x[:, :-2] - x[:, 2:]) / dT2
    lap_T[:, 0]    = (x[:, 0] - x[:, 1]) / dT2
    lap_T[:, -1]   = (x[:, -1] - x[:, -2]) / dT2

    return alpha * (x + lap_K + lap_T)


# ---------------------------------------------------------------------------
# DCT-II backend  (exact O(N log N) solver)
# ---------------------------------------------------------------------------

def _dct2_eigenvalues_1d(N: int, h: float) -> np.ndarray:
    """
    Analytic eigenvalues of the 1D Neumann Laplacian (-Delta_h) with
    one-sided boundary differences, for a grid of N+1 nodes with spacing h.

    The matrix is symmetric tridiagonal:
        diag  = [1/h^2, 2/h^2, ..., 2/h^2, 1/h^2]
        off   = [-1/h^2, ..., -1/h^2]

    Its eigenvalues are:
        lambda_k = 2*(1 - cos(pi*k/(N+1))) / h^2,   k = 0, 1, ..., N

    and its eigenvectors are the DCT-II basis vectors, i.e. the matrix is
    exactly diagonalised by scipy.fft.dct(type=2) (with appropriate
    normalisation).

    Note: k=0 gives lambda_0=0 (the constant mode, corresponding to the
    null space of the Neumann Laplacian).  After adding the identity (+I)
    the smallest eigenvalue of (-Delta_h + I) is 1 > 0, so the system is
    always non-singular.
    """
    k = np.arange(N + 1)
    return 2.0 * (1.0 - np.cos(np.pi * k / (N + 1))) / h**2


def _solve_dct(rhs: np.ndarray, alpha: float, grid: Grid) -> np.ndarray:
    """
    Solve  alpha * (-Delta_h + I) x = rhs  via 2D DCT-II.

    Algorithm
    ---------
    The 2D operator on the (N_K+1) x (N_T+1) grid decomposes as a Kronecker
    product:

        L = alpha * ((-Delta_K + I_K) (x) I_T + I_K (x) (-Delta_T) + I)
          -- wait, more carefully:
        alpha*(-Delta_h + I) = alpha*(I + L_K (x) I_T + I_K (x) L_T)

    where L_K, L_T are the 1D Neumann Laplacians.  Both L_K and L_T are
    diagonalised by the DCT-II transform (same eigenbasis), so in 2D the
    operator is diagonalised by the 2D DCT-II:

        Eigenvalue at mode (i,j): alpha * (1 + lambda_K[i] + lambda_T[j])

    Steps:
        1. rhs_hat = DCT2(rhs)            (2D DCT-II, unnormalised)
        2. x_hat   = rhs_hat / Lambda     (pointwise division)
        3. x       = IDCT2(x_hat)         (2D inverse DCT-II = DCT-III/2N)

    scipy.fft.dctn / idctn with type=2 implement exactly this transform.
    Complexity: O(N_K * N_T * log(N_K * N_T)).
    """
    from scipy.fft import dctn, idctn

    # 1D eigenvalues of the Neumann Laplacian for each dimension
    lam_K = _dct2_eigenvalues_1d(grid.N_K, grid.dK)  # shape (N_K+1,)
    lam_T = _dct2_eigenvalues_1d(grid.N_T, grid.dT)  # shape (N_T+1,)

    # 2D eigenvalue grid: Lambda[i,j] = alpha*(1 + lam_K[i] + lam_T[j])
    Lambda = alpha * (1.0 + lam_K[:, None] + lam_T[None, :])  # (N_K+1, N_T+1)

    # Transform, divide, inverse-transform
    rhs_hat = dctn(rhs, type=2, norm="ortho")
    x_hat   = rhs_hat / Lambda
    x       = idctn(x_hat, type=2, norm="ortho")
    return x


# ---------------------------------------------------------------------------
# Sparse LU backend
# ---------------------------------------------------------------------------

def _build_laplacian_neumann_1d(N: int, h: float):
    """
    Build the 1D negative Laplacian with Neumann BCs as a sparse matrix.
    Size (N+1) x (N+1), using one-sided differences at boundaries.

    Stencil:
      row 0   : (u_0 - u_1) / h^2          -> main=1/h^2, off=-1/h^2
      interior: (2*u_i - u_{i-1} - u_{i+1}) / h^2
      row N   : (u_N - u_{N-1}) / h^2      -> main=1/h^2, off=-1/h^2
    """
    from scipy.sparse import diags

    n = N + 1
    main_diag = np.full(n, 2.0 / h**2)
    main_diag[0]  = 1.0 / h**2
    main_diag[-1] = 1.0 / h**2
    off_diag = np.full(n - 1, -1.0 / h**2)

    return diags([off_diag, main_diag, off_diag], [-1, 0, 1], format="csr")


def _build_operator_sparse(alpha: float, grid: Grid):
    """
    Build the sparse matrix for alpha * (-Delta_h + I) on the 2D grid.
    Uses Kronecker product structure: L = L_K (x) I_T + I_K (x) L_T + I
    """
    from scipy.sparse import eye, kron

    N_K, N_T = grid.N_K, grid.N_T
    dK, dT   = grid.dK, grid.dT

    L_K = _build_laplacian_neumann_1d(N_K, dK)
    L_T = _build_laplacian_neumann_1d(N_T, dT)
    I_K = eye(N_K + 1, format="csr")
    I_T = eye(N_T + 1, format="csr")
    I   = eye((N_K + 1) * (N_T + 1), format="csr")

    # (-Delta_h + I) = kron(L_K, I_T) + kron(I_K, L_T) + I
    L2d = kron(L_K, I_T, format="csr") + kron(I_K, L_T, format="csr") + I
    return alpha * L2d


def _solve_lu(rhs: np.ndarray, alpha: float, grid: Grid,
              cache: dict) -> np.ndarray:
    """
    Solve  alpha*(-Delta_h+I) x = rhs  via sparse LU (factorisation cached).
    """
    from scipy.sparse.linalg import splu

    key = ("lu", grid.N_K, grid.N_T, grid.dK, grid.dT, alpha)
    if key not in cache:
        A = _build_operator_sparse(alpha, grid)
        cache[key] = splu(A.tocsc())
    lu = cache[key]
    x_flat = lu.solve(rhs.ravel(order="C"))
    return x_flat.reshape(rhs.shape, order="C")


# ---------------------------------------------------------------------------
# CG backend
# ---------------------------------------------------------------------------

def _solve_cg(rhs: np.ndarray, alpha: float, grid: Grid,
              tol: float = 1e-8, maxiter: int | None = None) -> np.ndarray:
    """
    Solve  alpha*(-Delta_h+I) x = rhs  via Conjugate Gradient.
    Jacobi preconditioner (diagonal).
    """
    from scipy.sparse.linalg import LinearOperator, cg

    n = rhs.size

    def matvec(xv):
        return _apply_operator(xv.reshape(rhs.shape), alpha, grid).ravel()

    A_op = LinearOperator((n, n), matvec=matvec, dtype=float)

    # Jacobi preconditioner: diagonal of alpha*(-Delta_h+I)
    dK2 = grid.dK**2
    dT2 = grid.dT**2
    # Diagonal entries of -Delta_K (one-sided Neumann: 1/h^2 at boundaries)
    diag_K = np.full(grid.N_K + 1, 2.0 / dK2)
    diag_K[0]  = 1.0 / dK2
    diag_K[-1] = 1.0 / dK2
    diag_T = np.full(grid.N_T + 1, 2.0 / dT2)
    diag_T[0]  = 1.0 / dT2
    diag_T[-1] = 1.0 / dT2
    diag = 1.0 + diag_K[:, None] + diag_T[None, :]
    prec_diag = alpha * diag

    def precond(xv):
        return xv / prec_diag.ravel()

    M_op = LinearOperator((n, n), matvec=precond, dtype=float)

    maxiter = maxiter or 4 * n
    x_flat, info = cg(A_op, rhs.ravel(), M=M_op, rtol=tol, maxiter=maxiter)
    if info > 0:
        import warnings
        warnings.warn(f"OE CG did not converge (info={info})", RuntimeWarning)
    return x_flat.reshape(rhs.shape)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def solve_optimality_system(
    rhs: np.ndarray,
    alpha: float,
    grid: Grid,
    method: str = "dct",
    cg_tol: float = 1e-8,
    cg_maxiter: int | None = None,
    _lu_cache: dict | None = None,
) -> np.ndarray:
    """
    Solve  alpha * (-Delta_h + I) delta_u = rhs

    i.e. the Tikhonov preconditioner step in the OE calibration loop.

    Parameters
    ----------
    rhs        : right-hand side, shape (N_K+1, N_T+1)
    alpha      : Tikhonov regularisation parameter
    grid       : Grid instance (determines dK, dT, N_K, N_T)
    method     : solver backend: "dct" | "lu" | "cg"
                 "dct" uses the exact DCT-II spectral solver (O(N log N)).
                 "lu"  uses sparse LU with factorisation caching.
                 "cg"  uses Conjugate Gradient (matrix-free).
    cg_tol     : tolerance for CG (used only when method="cg")
    cg_maxiter : max CG iterations (None = 4*N)
    _lu_cache  : optional mutable dict for LU factorisation reuse

    Returns
    -------
    delta_u : shape (N_K+1, N_T+1)
    """
    if _lu_cache is None:
        _lu_cache = {}

    method = method.lower()
    if method == "dct":
        return _solve_dct(rhs, alpha, grid)
    elif method == "lu":
        return _solve_lu(rhs, alpha, grid, _lu_cache)
    elif method == "cg":
        return _solve_cg(rhs, alpha, grid, tol=cg_tol, maxiter=cg_maxiter)
    else:
        raise ValueError(f"Unknown OE solver method: {method!r}. "
                         f"Choose from 'dct', 'lu', 'cg'.")

