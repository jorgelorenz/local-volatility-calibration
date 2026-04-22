"""
config.py
---------
Central configuration for the local volatility calibrator.

All tunable parameters are defined here so they can be adjusted in one
place without hunting through individual scripts.

Sections
--------
  GRID_*         -- PDE finite-difference grid (calibration)
  GRID_FINE_*    -- High-resolution grid used for final validation only
  THETA          -- Theta-scheme parameter (0.5 = Crank-Nicolson)
  ALPHA          -- Tikhonov regularisation strength
  VEGA_FLOOR     -- Minimum vega for weight computation
  SIGMA_BOUNDS   -- Box constraints on local vol during optimisation
  FTOL/GTOL/...  -- L-BFGS-B convergence settings
  ASSET_*        -- Default asset / UnRisk JSON path
  SWEEP_*        -- Parameter grid for experiments/sweep.py
  LOGGING_*      -- Verbosity and log-file settings

Usage
-----
    from src.config import ALPHA, GRID_N_K, GRID_N_T   # individual values
    from src import config                              # module-level access
    config.print_config()                               # print active config
"""

from __future__ import annotations
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Absolute path to the repository root (two levels up from this file).
REPO_ROOT: str = str(Path(__file__).resolve().parent.parent)

# Default UnRisk JSON data file.  Override by setting UNRISK_JSON_PATH in the
# calling script or via the UNRISK_JSON_PATH environment variable.
UNRISK_JSON_PATH: str = os.environ.get(
    "UNRISK_JSON_PATH",
    str(
        Path(REPO_ROOT)
        / "help"
        / "other_calibration_process"
        / "datos"
        / "EquityBasket_8e55875c81554f3697bd765e672c3704_20260305_170633.json"
    ),
)

# Default asset to load from the JSON (0-based index).
#   0 → DJ EUROSTOXX 50
#   1 → S&P 500
DEFAULT_ASSET_INDEX: int = 0

# ---------------------------------------------------------------------------
# Calibration PDE grid
# ---------------------------------------------------------------------------
# These control the speed vs. accuracy of the calibration solve.
# Increasing N_K / N_T improves accuracy but raises cost O(N_K * N_T) per
# forward/adjoint PDE solve.

GRID_N_K: int = 80
"""Number of strike nodes for the calibration grid."""

GRID_N_T: int = 60
"""Number of time nodes for the calibration grid."""

GRID_K_MARGIN: float = 0.02
"""Fractional extension of the strike axis beyond market min/max.
E.g. 0.02 means K_min = min_market_strike * 0.98."""

GRID_T_MARGIN: float = 0.02
"""Fractional extension of the time axis beyond the last market tenor."""

# ---------------------------------------------------------------------------
# Validation / final-check PDE grid  (high resolution, used only for the
# final IV back-out after calibration — not during the optimisation loop)
# ---------------------------------------------------------------------------
# These values should be large enough that PDE discretisation error is
# negligible compared to the calibration error you are diagnosing.

GRID_FINE_N_K: int = 300
"""Strike nodes for the fine-grid final validation solve."""

GRID_FINE_N_T: int = 200
"""Time nodes for the fine-grid final validation solve."""

# ---------------------------------------------------------------------------
# Numerical scheme
# ---------------------------------------------------------------------------

THETA: float = 0.5
"""Theta-scheme parameter for the Crank-Nicolson discretisation.
  0.5  → Crank-Nicolson (second-order in time, unconditionally stable)
  1.0  → Fully implicit (first-order in time, more damping)
  0.0  → Explicit (conditionally stable — avoid for large grids)
"""

# ---------------------------------------------------------------------------
# Tikhonov regularisation
# ---------------------------------------------------------------------------

ALPHA: float = 1e-3
"""H¹ Tikhonov regularisation parameter α.

Controls the trade-off between data fit (misfit term) and smoothness of
the calibrated local variance surface:

  J_α(u) = ½‖F(u) − z‖²_{H,w}  +  (α/2)‖u − u*‖²_{H¹}

Larger α → smoother surface, less sensitive to noisy market quotes.
Smaller α → closer fit to market IVs, but may over-fit or produce
            spurious spikes in the local vol surface.

Typical range: 1e-5 (aggressive fit) to 1e-2 (heavy smoothing).
"""

ALPHA_PRIOR_FLAT: bool = True
"""If True, the prior u* = median(market IV)² (flat surface).
If False, the caller must supply u_star explicitly."""

# ---------------------------------------------------------------------------
# Objective function weights
# ---------------------------------------------------------------------------

VEGA_FLOOR: float = 1e-4
"""Minimum Black-Scholes vega used when computing w(K,T) = 1/Vega².

Prevents infinite weights for deep OTM/ITM options where Vega → 0.
Reducing this value gives more weight to wings but may cause numerical
instability in the gradient."""

# ---------------------------------------------------------------------------
# L-BFGS-B optimiser settings
# ---------------------------------------------------------------------------

SIGMA_BOUNDS: tuple[float, float] = (0.01, 1.5)
"""Box constraints on local volatility σ(K,T) in [sigma_min, sigma_max].
The optimisation variable is u = σ², so the actual bounds passed to
L-BFGS-B are (sigma_min², sigma_max²).

sigma_min > 0 ensures no negative local variance.
sigma_max limits extreme values that can arise in poorly constrained regions.
"""

FTOL: float = 1e-10
"""Relative change in J below which L-BFGS-B declares convergence.
|ΔJ| / max(|J|, 1) < ftol  →  stop."""

GTOL: float = 1e-6
"""Gradient infinity-norm below which L-BFGS-B declares convergence.
‖∇J‖∞ < gtol  →  stop."""

MAX_ITER: int = 300
"""Maximum number of L-BFGS-B iterations."""

LBFGSB_MAXLS: int = 50
"""Maximum number of line-search function evaluations per L-BFGS-B step."""

LBFGSB_M: int = 10
"""Number of (s, y) pairs stored in the L-BFGS-B limited-memory Hessian.
Larger m → better Hessian approximation, more memory per iteration."""

# ---------------------------------------------------------------------------
# Logging / verbosity
# ---------------------------------------------------------------------------

VERBOSE: bool = True
"""Print calibration progress to stdout."""

LOG_EVERY_N_ITER: int = 5
"""Print a progress line every N iterations (only when VERBOSE=True)."""

LOG_TO_FILE: bool = False
"""If True, duplicate stdout progress to a log file (see experiments/)."""

# ---------------------------------------------------------------------------
# Experiments / parameter sweep  (used by experiments/sweep.py)
# ---------------------------------------------------------------------------

SWEEP_N_K_VALUES: list[int] = [40, 60, 80]
"""Calibration grid N_K values to sweep."""

SWEEP_N_T_VALUES: list[int] = [30, 40, 60]
"""Calibration grid N_T values to sweep (paired 1-to-1 with SWEEP_N_K_VALUES)."""

SWEEP_ALPHA_VALUES: list[float] = [1e-4, 1e-3, 1e-2]
"""Regularisation parameter values to sweep."""

SWEEP_MAX_ITER_LIST: list[int] = [50, 100, 200, 300]
"""Maximum iteration counts to sweep (useful for studying convergence speed)."""

SWEEP_ASSET_INDICES: list[int] = [0]
"""Asset indices to include in the sweep (default: only first asset)."""

SWEEP_SKIP_EXISTING: bool = True
"""If True, skip a sweep experiment whose log file already exists (cache)."""

SWEEP_QUICK_N_K: int = 40
"""N_K used in --quick mode (fast sanity check, not a real sweep)."""

SWEEP_QUICK_N_T: int = 30
"""N_T used in --quick mode."""

SWEEP_QUICK_MAX_ITER: int = 30
"""max_iter used in --quick mode."""

# ---------------------------------------------------------------------------
# Objective function type
# ---------------------------------------------------------------------------

MISFIT_TYPE: str = "price"
"""Type of misfit used in the objective function.

  "price"  (default)  : misfit in call prices,  w(K,T) * (C_model - C_mkt)^2
                         with vega-based weights w = 1/Vega^2.
  "iv"                : misfit directly in implied volatility,
                         (IV_model(K,T) - IV_mkt(K,T))^2.
                         IV_model is obtained by BS inversion of C_model at
                         each call to the objective (more expensive but avoids
                         the vega-weighting approximation).
"""

# ---------------------------------------------------------------------------
# Optimality-Equation (OE) solver settings
# ---------------------------------------------------------------------------

OE_SOLVER: str = "dct"
"""Method used to solve the Tikhonov preconditioner system
(-Delta_h + I) delta_u = rhs  inside the OE calibration loop.

  "dct"  (default)  : Discrete Cosine Transform (exact for Neumann BCs,
                       O(N log N), fastest).
  "lu"              : Sparse LU factorisation (robust, factorised once and
                       reused; O(N^1.5) setup, then O(N) per solve).
  "cg"              : Conjugate Gradient iterative solver (matrix-free,
                       O(sqrt(kappa)*N) per solve where kappa is the condition
                       number of -Delta_h+I).
"""

OE_MAX_ITER: int = 200
"""Maximum number of outer iterations for the OE calibration loop."""

OE_TOL: float = 1e-6
"""Convergence tolerance for the OE loop: stop when ||grad_J||_inf < OE_TOL."""

OE_STEP_SIZE: float = 1.0
"""Step-size (damping) for the OE Newton-like update:
  u^{k+1} = clip( u^k + OE_STEP_SIZE * delta_u, u_lo, u_hi )
Values < 1 give more conservative (smaller) steps."""

OE_VERIFY_GRADIENT_FD: bool = False
"""If True, additionally compute grad_pde by finite differences at the first
iteration and compare with the adjoint gradient.  Useful for debugging but
expensive: requires O(N_K * N_T) extra PDE solves."""

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def print_config() -> None:
    """Print all active configuration values to stdout."""
    import sys

    module = sys.modules[__name__]
    print("=" * 60)
    print("  Active configuration  (src/config.py)")
    print("=" * 60)

    sections = [
        ("Paths",                  ["REPO_ROOT", "UNRISK_JSON_PATH", "DEFAULT_ASSET_INDEX"]),
        ("Calibration grid",       ["GRID_N_K", "GRID_N_T", "GRID_K_MARGIN", "GRID_T_MARGIN"]),
        ("Fine-grid (validation)", ["GRID_FINE_N_K", "GRID_FINE_N_T"]),
        ("Numerical scheme",       ["THETA"]),
        ("Tikhonov regularisation",["ALPHA", "ALPHA_PRIOR_FLAT"]),
        ("Objective weights",      ["VEGA_FLOOR"]),
        ("Objective / misfit",     ["MISFIT_TYPE"]),
        ("L-BFGS-B",               ["SIGMA_BOUNDS", "FTOL", "GTOL", "MAX_ITER",
                                    "LBFGSB_MAXLS", "LBFGSB_M"]),
        ("OE solver",              ["OE_SOLVER", "OE_MAX_ITER", "OE_TOL",
                                    "OE_STEP_SIZE", "OE_VERIFY_GRADIENT_FD"]),
        ("Logging",                ["VERBOSE", "LOG_EVERY_N_ITER", "LOG_TO_FILE"]),
        ("Sweep",                  ["SWEEP_N_K_VALUES", "SWEEP_N_T_VALUES",
                                    "SWEEP_ALPHA_VALUES", "SWEEP_MAX_ITER_LIST",
                                    "SWEEP_ASSET_INDICES", "SWEEP_SKIP_EXISTING",
                                    "SWEEP_QUICK_N_K", "SWEEP_QUICK_N_T",
                                    "SWEEP_QUICK_MAX_ITER"]),
    ]

    for section_name, keys in sections:
        print(f"\n  [{section_name}]")
        for key in keys:
            val = getattr(module, key, "<not found>")
            print(f"    {key:<30} = {val!r}")

    print("=" * 60)
