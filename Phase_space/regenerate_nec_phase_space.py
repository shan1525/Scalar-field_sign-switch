#!/usr/bin/env python3
"""Regenerate the one-column dark-energy phase-plane figure.

The smooth ECDM and SSCDM tracks are computed from the same density
prescriptions and finite-difference operator used by phi_recons.py.  Exact
Ladder-Lambda-CDM is represented by its plateau points and distributional
impulse arrows; it is not assigned regulator-dependent vertical connectors.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np

import phi_recons as pr

import matplotlib as mpl

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    # Final-size typography for a 3.4-inch journal column.
    "font.size": 8.0,
    "axes.labelsize": 9.0,
    "axes.titlesize": 9.0,
    "legend.fontsize": 7.0,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.linewidth": 0.85,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": False,
    "ytick.right": False,
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "pdf.use14corefonts": False,
})


def zero_crossing(z: np.ndarray, y: np.ndarray, q: np.ndarray) -> tuple[float, float]:
    """Linearly interpolate the first zero of y and the corresponding q."""
    hits = np.flatnonzero(y[:-1] * y[1:] <= 0.0)
    if hits.size == 0:
        raise RuntimeError("No zero crossing found in the plotted interval.")
    j = int(hits[0])
    if y[j + 1] == y[j]:
        return float(z[j]), float(q[j])
    f = -y[j] / (y[j + 1] - y[j])
    return float(z[j] + f * (z[j + 1] - z[j])), float(q[j] + f * (q[j + 1] - q[j]))


def smooth_track(z_full: np.ndarray, rho_full: np.ndarray, z_max: float) -> dict[str, np.ndarray | float | int]:
    """Differentiate the complete saved grid, then retain the plotted interval."""
    _, _, drho_dz_full = pr.compute_S_and_D(z_full, rho_full)
    pressure_full = -rho_full + (1.0 + z_full) * drho_dz_full / 3.0
    keep = z_full <= z_max
    z = z_full[keep]
    rho = rho_full[keep]
    pressure = pressure_full[keep]
    imin = int(np.argmin(pressure))
    zzero, pzero = zero_crossing(z, rho, pressure)
    return {
        "z": z,
        "rho": rho,
        "p": pressure,
        "imin": imin,
        "zmin": float(z[imin]),
        "pmin": float(pressure[imin]),
        "zzero": zzero,
        "pzero": pzero,
    }


def add_rising_time_arrow(ax: plt.Axes, rho: np.ndarray, p: np.ndarray, fraction: float) -> None:
    """Add a forward-time arrow on a branch with increasing pressure."""
    stride = max(1, len(rho) // 14)
    candidates = np.flatnonzero(p[stride:] > p[:-stride])
    if candidates.size == 0:
        return
    i = int(candidates[int(np.clip(fraction, 0.0, 1.0) * (candidates.size - 1))])
    j = min(i + stride, len(rho) - 1)
    ax.annotate(
        "", xy=(rho[j], p[j]), xytext=(rho[i], p[i]),
        arrowprops=dict(arrowstyle="-|>", color="black", lw=0.95, mutation_scale=8.5),
        zorder=7,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("dark_energy_phase_space_corrected.pdf"))
    parser.add_argument("--z-max", type=float, default=5.0)
    parser.add_argument("--z-hi-grid", type=float, default=100.0)
    parser.add_argument("--nz", type=int, default=30000)
    parser.add_argument("--zdag", type=float, default=1.8)
    parser.add_argument("--eta", type=float, default=5.0)
    parser.add_argument("--dx-ss", type=float, default=0.4)
    parser.add_argument("--dz-ladder", type=float, default=0.15)
    parser.add_argument("--n-ladder", type=int, default=8)
    args = parser.parse_args()


    om_r0 = pr.omega_r0_from_Tcmb(h=0.7, Tcmb=2.7255, Neff=3.046)
    om_de0 = 1.0 - 0.31 - om_r0
    z_full = pr.build_grid(args.z_hi_grid, 0.0, args.nz)
    rho_e_full = pr.rho_de_norm_ecdm(z_full, om_de0, args.zdag, args.eta)
    rho_s_full = pr.rho_de_norm_sscdm(z_full, om_de0, args.zdag, args.dx_ss)
    ecdm = smooth_track(z_full, rho_e_full, args.z_max)
    sscdm = smooth_track(z_full, rho_s_full, args.z_max)

    print(f"Omega_r0={om_r0:.12g}; Omega_de0={om_de0:.12g}")
    for name, tr in (("ECDM", ecdm), ("SSCDM", sscdm)):
        print(
            f"{name}: p_min/rho_c0={tr['pmin']:.8f} at z={tr['zmin']:.8f}; "
            f"rho=0 at z={tr['zzero']:.8f}, p/rho_c0={tr['pzero']:.8f}"
        )

    # Numerical release checks, allowing for the documented grid-level rounding.
    expected = {"ECDM": (-1.41253, 1.63694, -1.29748), "SSCDM": (-2.89298, 1.75854, -2.82973)}
    for name, tr in (("ECDM", ecdm), ("SSCDM", sscdm)):
        ep, ez, ep0 = expected[name]
        if not (abs(float(tr["pmin"]) - ep) < 3e-3 and abs(float(tr["zmin"]) - ez) < 5e-3 and abs(float(tr["pzero"]) - ep0) < 3e-3):
            raise RuntimeError(f"{name} benchmark check failed; confirm phi_recons.py and grid settings.")

    # Keep the requested journal-column width but add height and side margins.
    fig, ax = plt.subplots(figsize=(3.4, 3.9), constrained_layout=True)

    # Reference boundaries and lightly shaded dynamical regimes.
    xx = np.linspace(-1.8, 1.8, 700)
    ylo, yhi = -3.4, 2.2
    xn = xx[xx <= 0.0]
    xp = xx[xx >= 0.0]
    ax.fill_between(xn, -xn, yhi, color="#eaf4fb", alpha=0.88, zorder=-4)
    ax.fill_between(xp, -xp, yhi, color="#fff7dc", alpha=0.88, zorder=-4)
    ax.fill_between(xn, ylo, -xn, color="#fde9e5", alpha=0.88, zorder=-4)
    ax.fill_between(xp, ylo, -xp, color="#f9dddd", alpha=0.88, zorder=-4)
    ax.fill_between(xx, ylo, -xx / 3.0, color="#f7cfc7", alpha=0.16, zorder=-3)
    ax.plot(xx, -xx, color="black", lw=0.9, ls="-", zorder=1)
    ax.plot(xx, -xx / 3.0, color="#b22222", lw=0.8, ls="-", zorder=1)
    ax.text(-1.27, 1.4, r"$\mathrm{NECB}$",
            rotation=-35, fontsize=7.2, color="black", ha="center", va="center")
    ax.text(1.02, -0.46, r"$\mathcal{M}_{\rm de}=0$", rotation=-9,
            fontsize=7.0, color="#9d1c1c", ha="left", va="bottom")

    styles = {
        "ECDM": dict(color="#1b9e77", lw=1.5, ls="-"),
        "SSCDM": dict(color="#7570b3", lw=1.55, ls=(0, (5.0, 2.2))),
    }
    endpoint_offsets_pt = {"ECDM": -1.8, "SSCDM": 1.8}
    for name, tr in (("ECDM", ecdm), ("SSCDM", sscdm)):
        rho = np.asarray(tr["rho"])
        pressure = np.asarray(tr["p"])
        st = styles[name]
        ax.plot(rho, pressure, label=name, zorder=4, **st)
        # The coincident z=5 and z=0 markers receive equal-and-opposite display
        # offsets in points; their underlying phase-plane coordinates are unchanged.
        endpoint_transform = ax.transData + mtransforms.ScaledTranslation(endpoint_offsets_pt[name] / 72.0, 0.0, fig.dpi_scale_trans)
        ax.plot(rho[0], pressure[0], marker="^", ms=4.5, mfc="white", mec=st["color"], mew=0.9, transform=endpoint_transform, zorder=6)
        ax.plot(rho[-1], pressure[-1], marker="o", ms=4.3, mfc="white", mec=st["color"], mew=0.9, transform=endpoint_transform, zorder=6)
        ax.plot(0.0, float(tr["pzero"]), marker="D", ms=4.0, mfc="white", mec=st["color"], mew=0.9, zorder=6)
        k = int(tr["imin"])
        ax.plot(rho[k], pressure[k], marker="s", ms=4.2, mfc="white", mec=st["color"], mew=0.9, zorder=6)
        add_rising_time_arrow(ax, rho, pressure, 0.42 if name == "ECDM" else 0.70)

    # Exact Ladder-Lambda-CDM: plateau values lie on the NEC boundary.
    n = np.arange(1, args.n_ladder + 1, dtype=float)
    z_steps = args.zdag + (n - 0.5 * (args.n_ladder + 1.0)) * args.dz_ladder
    rho_plateau = om_de0 * np.linspace(-1.0, 1.0, args.n_ladder + 1)
    p_plateau = -rho_plateau
    ax.plot(
        rho_plateau, p_plateau, ls="none", marker="*", ms=5.0, mew=0.65,
        mfc="#e66101", mec="black", color="#e66101",
        label=r"L$\Lambda$CDM plateaux", zorder=5,
    )
    for x0, y0 in zip(rho_plateau[1:-1], p_plateau[1:-1]):
        ax.annotate(
            "", xy=(x0, max(-3.15, y0 - 0.42)), xytext=(x0, y0 - 0.04),
            arrowprops=dict(arrowstyle="-|>", lw=0.75, color="#e66101", mutation_scale=6),
            zorder=3,
        )
    ax.text(-1.68, -2.96, r"$\mathrm{negative\ Dirac\ pressure}$" + "\n" + r"$\mathrm{impulses\ (schematic)}$",
            color="#b34b00", fontsize=6.2, ha="left", va="bottom")

    # Compact marker key suitable for the final one-column size.
    marker_handles = [
        plt.Line2D([], [], marker="^", ls="none", mfc="white", mec="black", ms=4, label=r"$z=5$"),
        plt.Line2D([], [], marker="o", ls="none", mfc="white", mec="black", ms=4, label=r"$z=0$"),
        plt.Line2D([], [], marker="D", ls="none", mfc="white", mec="black", ms=3.7, label=r"$\rho_{\rm de}=0$"),
        plt.Line2D([], [], marker="s", ls="none", mfc="white", mec="black", ms=3.8, label=r"$p_{\rm de,min}$"),
    ]
    handles, labels = ax.get_legend_handles_labels()
    model_pairs = [(h, lab) for h, lab in zip(handles, labels)
                   if lab in ("ECDM", "SSCDM", r"L$\Lambda$CDM plateaux")]
    leg1 = ax.legend([p[0] for p in model_pairs], [p[1] for p in model_pairs],
                     loc="upper right", frameon=True, framealpha=0.88,
                     handlelength=2.0, borderpad=0.25, labelspacing=0.22,
                     prop={"size": 6.8})
    ax.add_artist(leg1)
    ax.legend(handles=marker_handles, loc="lower right", ncol=2, frameon=True,
              framealpha=0.82, handletextpad=0.3, columnspacing=0.55,
              borderpad=0.25, labelspacing=0.25, prop={"size": 6.5})

    ax.text(0.76, 0.72, r"$\mathrm{p-quintessence}$", transform=ax.transAxes,
            color="#b97800", ha="center", fontsize=7.2, fontweight="bold")
    ax.text(0.23, 0.78, r"$\mathrm{n-quintessence}$", transform=ax.transAxes,
            color="#1769aa", ha="center", fontsize=7.2, fontweight="bold")
    ax.text(0.18, 0.20, r"$\mathrm{n-phantom}$", transform=ax.transAxes,
            color="#c51b29", ha="center", fontsize=7.2, fontweight="bold")
    ax.text(0.79, 0.20, r"$\mathrm{p-phantom}$", transform=ax.transAxes,
            color="#c51b29", ha="center", fontsize=7.2, fontweight="bold")
    ax.text(-1.60, 0.67, r"$\mathcal{M}_{\rm de}>0$" + "\n" + r"$\mathrm{focusing}$" + "\n" + r"$\mathrm{contribution}$",
            color="0.40", ha="left", va="center", fontsize=6.1, linespacing=0.92)
    ax.text(1.28, -1.30, r"$\mathcal{M}_{\rm de}<0$" + "\n" + r"$\mathrm{(Repulsive)}$",
            color="0.45", ha="center", va="center", fontsize=6.6)
    # Cartesian coordinate axes through the origin, as in the original figure.
    xmin, xmax = -1.8, 1.8
    ymin, ymax = -3.4, 2.2
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.spines["left"].set_position(("data", 0.0))
    ax.spines["bottom"].set_position(("data", 0.0))
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.set_xticks([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
    # Omit the y=0 label so that the two zero labels do not collide at the origin.
    ax.set_yticks([-3.0, -2.0, -1.0, 1.0])
    ax.tick_params(width=0.7, length=3.0, pad=2.0)

    # Arrowheads and axis labels at the positive ends.
    ax.plot(xmax, 0.0, marker=">", ms=4.5, color="black", clip_on=False, zorder=20)
    ax.plot(0.0, ymax, marker="^", ms=4.5, color="black", clip_on=False, zorder=20)
    ax.text(xmax - 0.05, 0.12, r"$\rho_{\rm de}/\rho_{\rm c0}$",
            ha="right", va="bottom", fontsize=8.7)
    ax.text(-0.06, ymax - 0.05, r"$p_{\rm de}/\rho_{\rm c0}$",
            ha="right", va="top", fontsize=8.7)
    ax.grid(False)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".png"), dpi=600, bbox_inches="tight")
    print(f"Saved {args.out} and {args.out.with_suffix('.png')}")
    print("Ladder step grid:", ", ".join(f"{v:.3f}" for v in z_steps))


if __name__ == "__main__":
    main()
