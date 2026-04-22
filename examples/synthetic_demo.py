"""
examples/synthetic_demo.py
--------------------------
Synthetic calibration demo: recover a known local volatility surface.

Setup
-----
We choose a "true" local vol surface sigma_true(K,T) (either flat or
a smile/skew pattern), generate synthetic call prices by solving the
Dupire forward PDE, then run the calibration starting from a flat
initial guess and check how well we recover sigma_true.

Usage
-----
    py examples/synthetic_demo.py [--case flat|smile] [--plot]
"""

import sys
import os
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.grid import Grid
from src.state_solver import solve_state
from src.calibration import calibrate


# ---------------------------------------------------------------------------
# Grid parameters
# ---------------------------------------------------------------------------
S0    = 100.0
r     = 0.05
q     = 0.02
T_max = 1.0
K_min = 70.0
K_max = 140.0
N_K   = 60
N_T   = 40


# ---------------------------------------------------------------------------
# True local variance surfaces
# ---------------------------------------------------------------------------

def flat_local_vol(K_grid, T_grid, sigma=0.20):
    """Flat local vol (BS world)."""
    return np.full((len(K_grid), len(T_grid)), sigma**2)


def smile_local_vol(K_grid, T_grid):
    """
    Mild smile: quadratic in log-moneyness.
    sigma(K,T) = 0.20 + 0.05*(log(K/100))^2 - 0.02*(T-0.5)
    (clipped to [0.05, 0.80])
    """
    K_arr = np.asarray(K_grid)
    T_arr = np.asarray(T_grid)
    KK, TT = np.meshgrid(K_arr, T_arr, indexing="ij")
    lm = np.log(KK / S0)
    sigma = 0.20 + 0.05 * lm**2 - 0.02 * (TT - 0.5)
    sigma = np.clip(sigma, 0.05, 0.80)
    return sigma**2


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(case: str = "flat", verbose: bool = True):
    print("=" * 60)
    print(f"Synthetic local vol calibration — case: {case}")
    print("=" * 60)

    # Build grid
    g = Grid(S0=S0, K_min=K_min, K_max=K_max, T_max=T_max,
             N_K=N_K, N_T=N_T, r=r, q=q)

    # True local variance
    if case == "flat":
        u_true = flat_local_vol(g.K, g.T, sigma=0.20)
    elif case == "smile":
        u_true = smile_local_vol(g.K, g.T)
    else:
        raise ValueError(f"Unknown case: {case}")

    # Synthetic "observations": call prices from the true PDE solution
    z = solve_state(u_true, g)

    # Weights: ignore T=0 (initial condition) and boundary nodes
    w = np.ones_like(z)
    w[:, 0] = 0.0      # T=0 slice: no information
    w[0,  :] = 0.0     # K_min boundary
    w[-1, :] = 0.0     # K_max boundary

    # Prior / initial guess: flat vol slightly off from truth
    sigma_init = 0.22 if case == "flat" else 0.21
    u_star = np.full_like(u_true, sigma_init**2)
    u0     = u_star.copy()

    # Regularization parameter
    alpha = 1e-4

    if verbose:
        print(f"\nGrid: N_K={N_K}, N_T={N_T}, K=[{K_min},{K_max}], T=[0,{T_max}]")
        print(f"True sigma range: [{np.sqrt(u_true.min()):.3f}, {np.sqrt(u_true.max()):.3f}]")
        print(f"Initial sigma: {sigma_init:.3f}")
        print(f"Alpha: {alpha:.2e}")
        print()

    # Run calibration
    result = calibrate(
        grid=g,
        z=z,
        w=w,
        u_star=u_star,
        alpha=alpha,
        u0=u0,
        sigma_bounds=(0.01, 1.5),
        max_iter=300,
        ftol=1e-14,
        gtol=1e-8,
        verbose=verbose,
    )

    print("\n" + str(result))

    # -----------------------------------------------------------------------
    # Validation metrics
    # -----------------------------------------------------------------------
    C_opt = result.C_opt
    u_opt = result.u_opt

    # Price relative error (interior nodes only, T>0)
    mask = (w > 0) & (z > 0.05)   # only liquid options
    if mask.sum() > 0:
        rel_err = np.abs(C_opt[mask] - z[mask]) / z[mask]
        print(f"\nPrice fit (interior, price > $0.05):")
        print(f"  Mean relative error : {rel_err.mean():.4e}")
        print(f"  Max  relative error : {rel_err.max():.4e}")

    # Local vol recovery error (only for flat case where we know the answer)
    if case == "flat":
        sigma_opt  = result.sigma_opt
        sigma_true = np.sqrt(u_true)
        sigma_err  = np.abs(sigma_opt[1:-1, 1:] - sigma_true[1:-1, 1:])
        print(f"\nLocal vol recovery (interior, T>0):")
        print(f"  Mean |sigma_opt - sigma_true| : {sigma_err.mean():.4e}")
        print(f"  Max  |sigma_opt - sigma_true| : {sigma_err.max():.4e}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="flat", choices=["flat", "smile"])
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    result = run(case=args.case, verbose=True)

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            from src.grid import Grid

            g = Grid(S0=S0, K_min=K_min, K_max=K_max, T_max=T_max,
                     N_K=N_K, N_T=N_T, r=r, q=q)

            if args.case == "flat":
                u_true = flat_local_vol(g.K, g.T, sigma=0.20)
            else:
                u_true = smile_local_vol(g.K, g.T)

            sigma_true = np.sqrt(u_true)
            sigma_opt  = result.sigma_opt

            KK, TT = np.meshgrid(g.K, g.T, indexing="ij")

            fig, axes = plt.subplots(1, 3, figsize=(15, 4))

            im0 = axes[0].contourf(TT, KK, sigma_true, levels=20, cmap="RdYlGn")
            axes[0].set_title("True local vol σ(K,T)")
            axes[0].set_xlabel("T"); axes[0].set_ylabel("K")
            plt.colorbar(im0, ax=axes[0])

            im1 = axes[1].contourf(TT, KK, sigma_opt, levels=20, cmap="RdYlGn")
            axes[1].set_title("Calibrated local vol σ(K,T)")
            axes[1].set_xlabel("T"); axes[1].set_ylabel("K")
            plt.colorbar(im1, ax=axes[1])

            im2 = axes[2].contourf(TT, KK, np.abs(sigma_opt - sigma_true),
                                   levels=20, cmap="hot_r")
            axes[2].set_title("|σ_opt - σ_true|")
            axes[2].set_xlabel("T"); axes[2].set_ylabel("K")
            plt.colorbar(im2, ax=axes[2])

            plt.tight_layout()
            plt.savefig("synthetic_demo_result.png", dpi=120, bbox_inches="tight")
            print("\nPlot saved to synthetic_demo_result.png")
            plt.show()

        except ImportError:
            print("matplotlib not available; skipping plot.")
