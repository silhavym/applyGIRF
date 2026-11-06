"""
Plot k-space before (nominal) and after (GIRF-predicted) gradient waveforms.
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def plot_kspace(out, save=None, show=True):
    """`out` is the dict built in apply.py (t_x, kx_nom, kx_act, ...)."""
    if not show:
        matplotlib.use("Agg")
    axes = [a for a in "xyz" if f"k{a}_nom" in out]
    n = len(axes)
    has_traj = n >= 2

    fig = plt.figure(figsize=(4.5 * n, 9 if has_traj else 4.5))
    rows = 2 if has_traj else 1
    gs = fig.add_gridspec(rows, n)

    # Top row: k(t) per axis
    for i, a in enumerate(axes):
        ax = fig.add_subplot(gs[0, i])
        t = out[f"t_{a}"] * 1e3
        ax.plot(t, out[f"k{a}_nom"], "--", label="nominal")
        ax.plot(t, out[f"k{a}_act"], label="GIRF")
        ax.set(xlabel="time (ms)", ylabel=f"k{a} (cycles/m)", title=f"k{a}(t)")
        ax.legend()
        ax.grid(alpha=0.3)

    # Bottom: trajectory in k-space
    if has_traj:
        m = min(len(out[f"k{a}_nom"]) for a in axes)
        nom = [out[f"k{a}_nom"][:m] for a in axes]
        act = [out[f"k{a}_act"][:m] for a in axes]
        if n == 3:
            ax = fig.add_subplot(gs[1, :], projection="3d")
            ax.plot(*nom, "--", lw=0.8, label="nominal")
            ax.plot(*act, lw=0.8, label="GIRF")
            ax.set_zlabel(f"k{axes[2]} (cycles/m)")
        else:
            ax = fig.add_subplot(gs[1, :])
            ax.plot(*nom, "--", lw=0.8, label="nominal")
            ax.plot(*act, lw=0.8, label="GIRF")
            ax.set_aspect("equal", adjustable="datalim")
            ax.grid(alpha=0.3)
        ax.set_xlabel(f"k{axes[0]} (cycles/m)")
        ax.set_ylabel(f"k{axes[1]} (cycles/m)")
        ax.set_title("k-space trajectory")
        ax.legend()

    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150)
    if show:
        plt.show()
    plt.close(fig)