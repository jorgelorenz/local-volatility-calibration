"""
Local Volatility Calibration as a PDE-Constrained Inverse Problem.

Based on the Engl (2001) framework: minimize a Tikhonov-regularized objective
whose state equation is the Dupire forward PDE for call option prices.

Modules
-------
grid            : Grid dataclass (strike/maturity axes, rate/dividend curves)
utils           : Black-Scholes pricing, IV inversion, Vega computation
market_data     : Load implied vol surface -> observed call prices z
state_solver    : Solve Dupire forward PDE  (state equation, forward in T)
adjoint_solver  : Solve adjoint PDE         (backward in T)
regularization  : Tikhonov regularization matrices and evaluations
objective       : Evaluate J(u, C, z, alpha)
gradient        : Evaluate gradient nabla J via discrete adjoint
calibration     : Main optimizer loop (L-BFGS-B)
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

__all__ = [
    "Grid",
    "bs_call", "bs_vega", "implied_vol_brentq",
    "iv_surface_to_call_prices", "load_unrisk_market_data",
    "solve_state",
    "solve_adjoint",
    "tikhonov_value", "tikhonov_gradient",
    "evaluate_J",
    "evaluate_gradient",
    "calibrate",
]
