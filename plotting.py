"""
Plot gradient-integrated k-space and the ADC-sampled MRSI trajectories.
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def plot_kspace(out, save=None, show=True):
    """Plot continuous nominal and GIRF-predicted k(t), plus the full trajectory."""
    if not show:
        matplotlib.use("Agg")
    axes = [a for a in "xyz" if f"k{a}_nom" in out]
    n = len(axes)
    has_traj = n >= 2
    fig = plt.figure(figsize=(4.5 * n, 9 if has_traj else 4.5))
    rows = 2 if has_traj else 1
    gs = fig.add_gridspec(rows, n)

    for i, axis in enumerate(axes):
        ax = fig.add_subplot(gs[0, i])
        t = out[f"t_{axis}"] * 1e3
        ax.plot(t, out[f"k{axis}_nom"], "--", label="nominal")
        ax.plot(t, out[f"k{axis}_act"], label="GIRF")
        ax.set(xlabel="time (ms)", ylabel=f"k{axis} (cycles/m)", title=f"k{axis}(t)")
        ax.legend()
        ax.grid(alpha=0.3)

    if has_traj:
        m = min(len(out[f"k{axis}_nom"]) for axis in axes)
        nom = [out[f"k{axis}_nom"][:m] for axis in axes]
        act = [out[f"k{axis}_act"][:m] for axis in axes]
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
        ax.set_title("Continuous gradient-integrated trajectory")
        ax.legend()

    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=160)
    if show:
        plt.show()
    plt.close(fig)


def plot_kfile_trajectories(nominal, girf, save_prefix=None, show=True):
    """
    Plot ADC-sampled nominal and GIRF k-files separately and overlaid.
    Each TR is plotted independently so leaf assignments remain visible.
    """
    if not show:
        matplotlib.use("Agg")

    def _plot_one(data, title, save_path):
        fig, ax = plt.subplots(figsize=(7, 7))
        for tr in np.unique(data["TR"]):
            mask = data["TR"] == tr
            ax.plot(data["Kx"][mask], data["Ky"][mask], lw=0.9,
                    label=f"TR {tr}" if len(np.unique(data["TR"])) <= 12 else None)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("Kx (cycles/mm)")
        ax.set_ylabel("Ky (cycles/mm)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        if len(np.unique(data["TR"])) <= 12:
            ax.legend(title="Rosette leaf / TR")
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=160)
        if show:
            plt.show()
        plt.close(fig)

    if nominal is not None:
        _plot_one(nominal, "Nominal ADC-sampled rosette trajectory",
                  f"{save_prefix}_nominal.png" if save_prefix else None)
    if girf is not None:
        _plot_one(girf, "GIRF-corrected ADC-sampled rosette trajectory",
                  f"{save_prefix}_girf.png" if save_prefix else None)

    if nominal is not None and girf is not None:
        fig, ax = plt.subplots(figsize=(7, 7))
        for tr in np.unique(nominal["TR"]):
            mask = nominal["TR"] == tr
            ax.plot(nominal["Kx"][mask], nominal["Ky"][mask], "--",
                    lw=0.9, label=f"Nominal TR {tr}")
        for tr in np.unique(girf["TR"]):
            mask = girf["TR"] == tr
            ax.plot(girf["Kx"][mask], girf["Ky"][mask], lw=1.0,
                    label=f"GIRF TR {tr}")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("Kx (cycles/mm)")
        ax.set_ylabel("Ky (cycles/mm)")
        ax.set_title("Nominal vs GIRF-corrected ADC-sampled trajectory")
        ax.grid(alpha=0.3)
        if len(np.unique(nominal["TR"])) <= 6:
            ax.legend(fontsize="small")
        fig.tight_layout()
        if save_prefix:
            fig.savefig(f"{save_prefix}_overlay.png", dpi=160)
        if show:
            plt.show()
        plt.close(fig)
