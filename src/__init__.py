"""
Local Volatility Calibration as a PDE-Constrained Inverse Problem.

Based on the Engl (2001) framework: minimize a Tikhonov-regularized objective
whose state equation is the Dupire forward PDE for call option prices.

Modules
-------
grid                      : Grid dataclass (strike/maturity axes, rate/dividend curves)
utils                     : Black-Scholes pricing, IV inversion, Vega computation
market_data               : Load implied vol surface -> observed call prices z
state_solver              : Solve Dupire forward PDE  (state equation, forward in T)
adjoint_solver            : Solve adjoint PDE         (backward in T)
regularization            : Tikhonov regularization matrices and evaluations
objective                 : Evaluate J(u, C, z, alpha)
gradient                  : Evaluate gradient nabla J via discrete adjoint
calibration               : Main optimizer loop (L-BFGS-B)
fem_mesh                  : FEM mesh utilities (uniform, graded, bisection refinement)
fem_state_solver          : FEM P1 forward Dupire solver (manual assembly)
fem_backward_solver       : FEM P1 backward local-vol solver (manual assembly)
slv_pricer                : SLV 2D backward PDE solver (Craig-Sneyd ADI)
fem_state_solver_fenics   : FEM P1 forward Dupire solver  (FEniCS, optional)
fem_backward_solver_fenics: FEM P1 backward local-vol solver (FEniCS, optional)
"""

from .grid import Grid
from .utils import bs_call, bs_vega, implied_vol_brentq
from .market_data import iv_surface_to_call_prices, load_unrisk_market_data
from .state_solver import solve_state
from .adjoint_solver import solve_adjoint
from .regularization import tikhonov_value, tikhonov_gradient
from .objective import evaluate_J
from .gradient import evaluate_gradient
from .calibration import calibrate
from .fem_mesh import uniform_mesh, graded_mesh, bisection_refine, make_mesh
from .fem_state_solver import solve_fem_state
from .fem_backward_solver import solve_fem_backward, solve_fem_backward_grid
from .slv_pricer import solve_slv

# FEniCS solvers are optional (require a FEniCS-enabled environment)
try:
    from .fem_state_solver_fenics import solve_fem_state_fenics
    from .fem_backward_solver_fenics import (
        solve_fem_backward_fenics,
        solve_fem_backward_grid_fenics,
    )
    _FENICS_AVAILABLE = True
except ImportError:
    _FENICS_AVAILABLE = False

__all__ = [
    # Core
    "Grid",
    "bs_call", "bs_vega", "implied_vol_brentq",
    "iv_surface_to_call_prices", "load_unrisk_market_data",
    "solve_state",
    "solve_adjoint",
    "tikhonov_value", "tikhonov_gradient",
    "evaluate_J",
    "evaluate_gradient",
    "calibrate",
    # FEM (manual)
    "uniform_mesh", "graded_mesh", "bisection_refine", "make_mesh",
    "solve_fem_state",
    "solve_fem_backward", "solve_fem_backward_grid",
    # SLV
    "solve_slv",
    # FEniCS (optional)
    "solve_fem_state_fenics",
    "solve_fem_backward_fenics",
    "solve_fem_backward_grid_fenics",
    "_FENICS_AVAILABLE",
]
