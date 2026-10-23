"""Figure for the strain-measure model-form check (JNE revision, R2-1).

Rows: the four scalar strain measures (von Mises, max principal, max areal,
orientation-averaged areal).  Columns: (a) measure on the focal x-z plane,
(b) firing map (spikes per voxel) at the canonical K_A*alpha ("raw"),
(c) firing map with the measure rescaled to the von Mises focal peak
("matched").  Reads <sim_dir>/strain_variants/ written by
run_strain_measure_variants.py and writes fig_strain_variants.{png,pdf}
into the sim dir and the manuscript figures dir.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sim_dir = REPO / "data" / "halle_dacc"
vd = sim_dir / "strain_variants"
out_fig_dir = REPO / "figures"

meas = np.load(vd / "strain_measures.npz")
summary = json.loads((vd / "summary.json").read_text())
dx_mm = float(meas["dx_m"]) * 1e3
labels = {
    "vm": r"von Mises $\varepsilon_{\mathrm{eq}}$ (submitted)",
    "e1d": r"max. principal $\varepsilon_1$ (deviatoric)",
    "eAmaxd": r"max. areal $\varepsilon_A^{\max}$ (deviatoric)",
    "eAavgd": r"orient.-avg. areal $\langle\varepsilon_A^{+}\rangle$ (deviatoric)",
    "e1": r"max. principal $\varepsilon_1$ (raw tensor)",
    "eAmax": r"max. areal $\varepsilon_A^{\max}$ (raw tensor)",
    "eAavg": r"orient.-avg. areal $\langle\varepsilon_A^{+}\rangle$ (raw tensor)",
}
order = ["vm", "e1", "eAmax", "eAavg"]

# focal voxel from the control tension field
ctrl = np.load(vd / "neural_vm_raw.npz")
ix, iy, iz = np.unravel_index(int(np.argmax(ctrl["tension_3d"])), ctrl["tension_3d"].shape)
nx, ny, nz = ctrl["tension_3d"].shape
extent = [0, nz * dx_mm, nx * dx_mm, 0]   # z (lateral) horizontal, x (depth) vertical

vmax_meas = float(meas["vm"].max())
vmax_spk = max(float(np.load(vd / f"neural_{k}_{m}.npz")["spike_count_map"].max())
               for k in order for m in (["raw"] if k == "vm" else ["raw", "matched"]))

short = {"vm": "von Mises $\\varepsilon_{\\mathrm{eq}}$\n(default)", "e1": "max. principal $\\varepsilon_1$",
         "eAmax": "max. areal $\\varepsilon_A^{\\max}$", "eAavg": "orient.-avg. areal $\\langle\\varepsilon_A^+\\rangle$"}
fig, axes = plt.subplots(3, len(order), figsize=(10.5, 9.6), constrained_layout=True)
for c, k in enumerate(order):
    f = meas[k][:, iy, :]
    im0 = axes[0, c].imshow(f / vmax_meas, extent=extent, cmap="viridis", vmin=0, vmax=1, aspect="equal")
    axes[0, c].set_title(short[k] + f"\npeak {float(meas[k].max()):.2e}", fontsize=8.5)
    for r, mode in enumerate(("raw", "matched"), start=1):
        key = f"{k}_{mode}"
        if k == "vm" and mode == "matched":
            axes[r, c].axis("off"); axes[r, c].text(0.5, 0.5, "(identical\nto above)", ha="center", va="center",
                                                    transform=axes[r, c].transAxes, fontsize=8)
            continue
        d = np.load(vd / f"neural_{key}.npz")
        sp = d["spike_count_map"][:, iy, :].astype(float)
        sp = np.ma.masked_where(sp <= 0, sp)
        axes[r, c].imshow(np.zeros_like(f), extent=extent, cmap="Greys", vmin=0, vmax=1, aspect="equal")
        im1 = axes[r, c].imshow(sp, extent=extent, cmap="inferno", vmin=0, vmax=vmax_spk, aspect="equal")
        h = summary["runs"][key]
        txt = f"{h['n_firing_voxels']} vox, {h['total_spikes']} spk\niso-25 {h['iso25_volume_mm3']/1e3:.1f}e3 mm$^3$"
        if "dice_vs_control" in h:
            txt += f"\nDice {h['dice_vs_control']:.2f}"
            if "dice_vs_control_excl_shell" in h:
                txt += f" ({h['dice_vs_control_excl_shell']:.2f} deep)"
        axes[r, c].set_title(txt, fontsize=7.5)
    for r in range(3):
        axes[r, c].plot(iz * dx_mm, ix * dx_mm, "c+", ms=7, mew=1.1)
        axes[r, c].tick_params(labelsize=7)
        if r == 2: axes[r, c].set_xlabel("z (mm)", fontsize=8)
        if c == 0: axes[r, c].set_ylabel(["strain field\nx, depth (mm)", "canonical $K_A\\alpha$\nx, depth (mm)", "peak-matched\nx, depth (mm)"][r], fontsize=8)
        if c > 0: axes[r, c].set_yticklabels([])
fig.colorbar(im0, ax=axes[0, :], shrink=0.8, pad=0.01, label="strain / von Mises peak")
fig.colorbar(im1, ax=axes[1:, :], shrink=0.6, pad=0.01, label="spikes per voxel (20 ms)")
for p in (sim_dir / "fig_strain_variants.png", out_fig_dir / "fig_strain_variants.png",
          out_fig_dir / "fig_strain_variants.pdf",
          REPO / "docs" / "jne_submission_verhagen" / "supplementary_figures" / "fig_strain_variants.pdf"):
    if not p.parent.is_dir():          # the manuscript folder exists only in the dev repo
        continue
    fig.savefig(p, dpi=170)
    print("wrote", p)
