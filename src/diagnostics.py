"""
diagnostics.py
--------------
Visualisation and diagnostic utilities for calibration results.

Functions
---------
plot_convergence(history, title='', out_path=None, show=False)
    Three-panel convergence chart: J(iter), ||grad_J||_inf(iter),
    t_iter(iter).  Optionally saves to disk and/or displays interactively.

plot_iv_comparison(strikes, tenors, iv_market, iv_model,
                   label_model='Model', title='', out_path=None, show=False)
    Per-tenor IV smile subplots comparing market quotes with model IVs.
    Useful for both the new Tikhonov method and the v2 (LM) method.

print_iv_metrics(iv_market, iv_model, label='')
    Print MAE / RMSE / max-error between two IV surfaces (NaN-safe).
"""

from __future__ import annotations
import numpy as np


# ---------------------------------------------------------------------------
# Convergence plot
# ---------------------------------------------------------------------------

def plot_convergence(
    history: dict,
    title: str = "",
    out_path: str | None = None,
    show: bool = False,
) -> None:
    """
    Plot calibration convergence from a CalibrationResult.history dict.

    Parameters
    ----------
    history  : dict with keys 'J', 'grad_norm', 'delta_J', 't_iter', 't_cumul'
               (as produced by calibration.calibrate()).
    title    : suptitle string (e.g. asset name + parameter summary).
    out_path : if given, save the figure to this path (PNG/PDF/SVG).
    show     : if True, call plt.show() (interactive display).

    Notes
    -----
    The x-axis is the *record index* (every log_every-th iteration), not the
    raw iteration number.  If you log every 5 iterations, tick 10 on the
    x-axis corresponds to iteration 50.
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    J         = np.asarray(history.get("J", []), dtype=float)
    gn        = np.asarray(history.get("grad_norm", []), dtype=float)
    dJ        = np.asarray(history.get("delta_J", []), dtype=float)
    t_iter    = np.asarray(history.get("t_iter", []), dtype=float)
    t_cumul   = np.asarray(history.get("t_cumul", []), dtype=float)

    if J.size == 0:
        print("diagnostics.plot_convergence: history is empty, nothing to plot.")
        return

    iters = np.arange(1, J.size + 1)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Panel 1: Objective J
    ax = axes[0]
    ax.semilogy(iters, J, "o-", color="#4C72B0", lw=1.5, ms=3)
    ax.set_xlabel("Recorded iteration")
    ax.set_ylabel("J (log scale)")
    ax.set_title("Objective J(u)")
    ax.grid(True, which="both", alpha=0.25)

    # Panel 2: Gradient norm
    ax = axes[1]
    ax.semilogy(iters, gn, "s-", color="#C44E52", lw=1.5, ms=3)
    ax.set_xlabel("Recorded iteration")
    ax.set_ylabel("‖∇J‖∞ (log scale)")
    ax.set_title("Gradient norm ‖∇J‖∞")
    ax.grid(True, which="both", alpha=0.25)

    # Panel 3: Time per iteration
    ax = axes[2]
    if t_iter.size > 0:
        ax.bar(iters, t_iter, color="#55A868", width=0.7, edgecolor="k", linewidth=0.4)
        # Overlay cumulative time as a line on twin axis
        ax2 = ax.twinx()
        ax2.plot(iters, t_cumul, "D--", color="#8172B2", lw=1.2, ms=3,
                 label="Cumulative")
        ax2.set_ylabel("Cumulative time (s)", color="#8172B2")
        ax2.tick_params(axis="y", labelcolor="#8172B2")
        ax2.legend(loc="upper left", fontsize=8)
    ax.set_xlabel("Recorded iteration")
    ax.set_ylabel("Time per iteration (s)")
    ax.set_title("Iteration timing")
    ax.grid(True, alpha=0.25)

    suptitle = "Calibration convergence"
    if title:
        suptitle += f"  —  {title}"
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        print(f"  Convergence plot saved: {out_path}")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# IV comparison plot
# ---------------------------------------------------------------------------

def plot_iv_comparison(
    strikes: np.ndarray,
    tenors: np.ndarray,
    iv_market: np.ndarray,
    iv_model: np.ndarray,
    label_model: str = "Model",
    title: str = "",
    out_path: str | None = None,
    show: bool = False,
) -> None:
    """
    Per-tenor IV smile subplots: market vs. one model.

    Parameters
    ----------
    strikes    : shape (n_K,)
    tenors     : shape (n_T,)
    iv_market  : shape (n_T, n_K)
    iv_model   : shape (n_T, n_K)  — NaN where not available
    label_model: legend label for the model curve
    title      : figure suptitle
    out_path   : save path (optional)
    show       : display interactively
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_t   = len(tenors)
    n_cols = min(3, n_t)
    n_rows = int(np.ceil(n_t / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5.5 * n_cols, 3.8 * n_rows), squeeze=False)

    for i, t in enumerate(tenors):
        ax = axes[i // n_cols][i % n_cols]
        ax.plot(strikes, iv_market[i, :], "o-",  label="Market IV",
                lw=1.8, ms=4, color="#4C72B0")
        ax.plot(strikes, iv_model[i, :],  "s--", label=label_model,
                lw=1.5, ms=4, color="#C44E52")

        # Annotate error stats
        mask = np.isfinite(iv_model[i, :]) & np.isfinite(iv_market[i, :])
        if mask.sum() > 0:
            diff = iv_model[i, mask] - iv_market[i, mask]
            mae  = float(np.mean(np.abs(diff)))
            ax.set_title(f"T={t:.3f}y   MAE={mae:.4f}", fontsize=9)
        else:
            ax.set_title(f"T={t:.3f}y", fontsize=9)

        ax.set_xlabel("Strike", fontsize=8)
        ax.set_ylabel("Implied Vol", fontsize=8)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=7)
        ax.tick_params(labelsize=7)

    # Hide unused subplots
    for j in range(n_t, n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis("off")

    suptitle = "IV smile comparison"
    if title:
        suptitle = f"{title}  —  {suptitle}"
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        print(f"  IV comparison plot saved: {out_path}")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------------

def print_iv_metrics(
    iv_market: np.ndarray,
    iv_model: np.ndarray,
    label: str = "",
) -> dict:
    """
    Compute and print MAE / RMSE / max-error between two IV surfaces.

    Parameters
    ----------
    iv_market : reference IV surface (any shape, NaN-safe)
    iv_model  : model IV surface (same shape)
    label     : display name for the model

    Returns
    -------
    dict with keys 'label', 'MAE', 'RMSE', 'max_err', 'n'
    """
    iv_mkt = np.asarray(iv_market, dtype=float)
    iv_mod = np.asarray(iv_model,  dtype=float)
    mask   = np.isfinite(iv_mod) & np.isfinite(iv_mkt)
    n      = int(mask.sum())

    if n == 0:
        metrics = {"label": label, "MAE": np.nan, "RMSE": np.nan,
                   "max_err": np.nan, "n": 0}
    else:
        diff = iv_mod[mask] - iv_mkt[mask]
        metrics = {
            "label":   label,
            "MAE":     float(np.mean(np.abs(diff))),
            "RMSE":    float(np.sqrt(np.mean(diff ** 2))),
            "max_err": float(np.max(np.abs(diff))),
            "n":       n,
        }

    lbl = f"[{label}]" if label else ""
    print(
        f"  {lbl:<24}  MAE={metrics['MAE']:.6f}"
        f"  RMSE={metrics['RMSE']:.6f}"
        f"  max={metrics['max_err']:.6f}"
        f"  n={metrics['n']}"
    )
    return metrics
