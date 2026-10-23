"""Figure + LaTeX snippet for the (K_A*alpha, g_Piezo1) response map (JNE revision, R2-3).

Reads data/halle_dacc/response_map/response_map.json and writes
  figures/fig_response_map.{png,pdf}   (manuscript figures dir)
  data/response_map_numbers.tex        (\\newcommand macros with the headline numbers)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

REPO = Path(__file__).resolve().parent.parent
rm = json.loads((REPO / "data/halle_dacc/response_map/response_map.json").read_text())
meta = rm["_meta"]
alphas, gps, gts = meta["alphas"], meta["g_piezo"], meta["g_trek"]
ka_alpha = [meta["K_A"] * a for a in alphas]                # N/m per unit strain
fig_dir = REPO / "figures"


def grid(g_trek, field):
    z = np.full((len(gps), len(alphas)), np.nan)
    for i, gp in enumerate(gps):
        for j, a in enumerate(alphas):
            k = f"trek{g_trek:g}_alpha{a:g}_gp{gp:g}"
            if k in rm:
                z[i, j] = rm[k][field]
    return z


fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
vol = {gt: grid(gt, "iso25_volume_mm3") / 1e3 for gt in gts}
vmax = np.nanmax([np.nanmax(v) for v in vol.values()])
floor = 0.05                                                   # 10^3 mm^3, below one voxel-ish for log display
for ax, gt in zip(axes, gts):
    z = vol[gt]
    zplot = np.where(z > 0, z, np.nan)
    im = ax.pcolormesh(np.arange(len(alphas) + 1), np.arange(len(gps) + 1), zplot,
                       cmap="viridis", norm=LogNorm(vmin=floor, vmax=vmax), shading="flat")
    # annotate every cell with the volume (or "0")
    for i in range(len(gps)):
        for j in range(len(alphas)):
            v = z[i, j]
            if np.isnan(v):
                continue
            txt = "0" if v == 0 else (f"{v:.1f}" if v >= 0.95 else f"{v:.2f}")
            ax.text(j + 0.5, i + 0.5, txt, ha="center", va="center", fontsize=7.5,
                    color="white" if (v == 0 or v < 0.3 * vmax) else "black")
    ax.set_xticks(np.arange(len(alphas)) + 0.5)
    ax.set_xticklabels([f"{k:g}\n($\\alpha$={a:g})" for k, a in zip(ka_alpha, alphas)], fontsize=8)
    ax.set_yticks(np.arange(len(gps)) + 0.5); ax.set_yticklabels([f"{g:g}" for g in gps], fontsize=8)
    ax.set_xlabel(r"$K_A\alpha$ (N/m per unit strain)")
    ax.set_ylabel(r"$\bar g_{\mathrm{Piezo1}}$ (mS/cm$^2$)")
    ax.set_title(rf"$\bar g_{{\mathrm{{TREK\text{{-}}1}}}} = {gt:g}$ mS/cm$^2$" + (" (canonical)" if gt == 0.04 else " (library default)"), fontsize=10)
    # literature lower end of alpha (200) as a vertical line at the left edge of that column
    ax.axvline(1.0, color="w", lw=1.0, ls=":")
    ax.text(1.02, len(gps) - 0.1, r"$\alpha\approx200$ (planar bilayer)", color="w", fontsize=7, va="top")
    # markers
    if gt == 0.04:
        ax.plot(alphas.index(1000.0) + 0.5, gps.index(0.10) + 0.5, marker="*", ms=16, mfc="crimson", mec="k")
        ax.text(alphas.index(1000.0) + 0.5, gps.index(0.10) + 0.95, "canonical", ha="center", va="bottom", fontsize=8, color="crimson", fontweight="bold")
    if gt == 0.02:
        ax.plot(alphas.index(1000.0) + 0.5, gps.index(0.005) + 0.5, marker="s", ms=11, mfc="none", mec="crimson", mew=2)
        ax.text(alphas.index(1000.0) + 0.5, gps.index(0.005) + 0.95, "library default", ha="center", va="bottom", fontsize=8, color="crimson", fontweight="bold")
        ax.axhline(1.0, color="crimson", lw=1.0, ls="--")
cb = fig.colorbar(im, ax=axes, shrink=0.9, pad=0.02)
cb.set_label(r"iso-25 % firing volume ($10^3$ mm$^3$); blank = no firing")
for p in (fig_dir / "fig_response_map.png", fig_dir / "fig_response_map.pdf", REPO / "data/halle_dacc/fig_response_map.png"):
    fig.savefig(p, dpi=170)
    print("wrote", p)

# ---- numbers for the text ----
def get(gt, a, gp):
    return rm.get(f"trek{gt:g}_alpha{a:g}_gp{gp:g}")
can = get(0.04, 1000.0, 0.10); lib = get(0.02, 1000.0, 0.005); lib_trek04 = get(0.04, 1000.0, 0.005)
lines = []
def mac(name, val): lines.append(f"\\newcommand{{\\{name}}}{{{val}}}")
mac("RMcanFire", f"{can['n_firing_voxels']:,}".replace(",", "\\,"))
mac("RMlibFire", f"{lib['n_firing_voxels']:,}".replace(",", "\\,"))
mac("RMlibSpikes", f"{lib['total_spikes']:,}".replace(",", "\\,"))
mac("RMlibIso", f"{lib['iso25_volume_mm3']:.0f}")
mac("RMlibTpk", f"{lib['peak_tension_mNm']:.2f}")
mac("RMlibTrekFourFire", f"{lib_trek04['n_firing_voxels']:,}".replace(",", "\\,"))
# threshold conductance at canonical K_A alpha for each TREK panel
for gt, tag in ((0.04, "Four"), (0.02, "Two")):
    fires = [gp for gp in gps if get(gt, 1000.0, gp) and get(gt, 1000.0, gp)["n_firing_voxels"] > 0]
    mac(f"RMminGpTrek{tag}", f"{min(fires):g}" if fires else "none")
# range of iso-25 volume over the literature bracket at canonical g
vals = [get(0.04, a, 0.10)["iso25_volume_mm3"] for a in alphas if get(0.04, a, 0.10)]
mac("RMcanIsoMin", f"{min(vals)/1e3:.1f}"); mac("RMcanIsoMax", (f"{max(vals)/1e3:.0f}" if max(vals) >= 1e4 else f"{max(vals)/1e3:.1f}"))
(REPO / "data/response_map_numbers.tex").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
