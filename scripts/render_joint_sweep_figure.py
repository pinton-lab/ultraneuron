"""Figure for the joint-scaling mechanism sweep (JNE revision, R2-4).

Panel (a): total spike count vs focal peak pressure (bottom axis) with the
matching peak strain (top axis) for scenarios S1-S5, joint scaling.
Panel (b): the same for the strain-only design (pressure and temperature held
at the canonical field), for comparison / supplement.
Panel (c): firing-map slices (spikes per voxel, focal y-z plane) for S1-S5 at
the canonical level s = 1, joint scaling.

Also writes data/joint_sweep_numbers.tex with macros for the manuscript text.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sd = REPO / "data/halle_dacc/joint_sweep"
fig_dir = REPO / "figures"
J = json.loads((sd / "results_joint.json").read_text())
S = json.loads((sd / "results_strain.json").read_text()) if (sd / "results_strain.json").exists() else None
meta = J["_meta"]; s_levels = meta["s_levels"]; scen = meta["scenarios"]
colors = dict(S1="tab:blue", S2="tab:orange", S3="tab:green", S4="tab:red", S5="tab:purple")
labels = dict(S1="S1 1-comp", S2="S2 3-comp", S3="S3 +NICE", S4="S4 +inact/Ca/SK", S5="S5 full")


def series(R, sc, key="total_spikes"):
    return np.array([R[f"{sc}_s{s:g}"][key] for s in s_levels], dtype=float)


ppp = np.array([J[f"S2_s{s:g}"]["focal_ppp_mpa"] for s in s_levels])
eps = np.array([J[f"S2_s{s:g}"]["eps_peak"] for s in s_levels])

fig = plt.figure(figsize=(15, 8.6))
gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.05], hspace=0.42)
top = gs[0].subgridspec(1, 2, wspace=0.28)
for k, (R, title) in enumerate(((J, "(a) joint scaling: $p$, $I$, $\\varepsilon$, $\\Delta T$ from one source amplitude"),
                                (S, "(b) strain-axis design: $\\varepsilon$ rescaled, $p$ and $T$ at the canonical field"))):
    ax = fig.add_subplot(top[0, k])
    if R is None:
        ax.text(0.5, 0.5, "strain-mode results pending", ha="center", va="center", transform=ax.transAxes); continue
    for sc in scen:
        ax.plot(ppp, series(R, sc), marker="o", lw=1.6, color=colors[sc], label=labels[sc],
                ls="--" if sc in ("S3", "S5") else "-", ms=5 if sc in ("S3", "S5") else 6)
    ax.set_xscale("log"); ax.set_yscale("symlog", linthresh=10)
    ax.set_ylim(-1, 5e5)
    ax.set_xlabel("focal peak pressure PPP (MPa)" + ("" if k == 0 else "  [equivalent level; pressure not applied]"))
    ax.set_ylabel(f"total spikes ({meta['duration_ms']:.0f} ms window)")
    ax.axvline(ppp[s_levels.index(1.0)], color="k", ls="--", lw=0.8)
    ax.text(ppp[s_levels.index(1.0)] * 1.03, 5, "canonical", fontsize=8, rotation=90, va="bottom")
    ax.grid(alpha=0.3, which="both"); ax.set_title(title, fontsize=10)
    sec = ax.secondary_xaxis("top", functions=(lambda p: (p / ppp[s_levels.index(1.0)]) ** 2 * eps[s_levels.index(1.0)],
                                               lambda e: np.sqrt(e / eps[s_levels.index(1.0)]) * ppp[s_levels.index(1.0)]))
    sec.set_xlabel(r"peak strain $\varepsilon_{\mathrm{eq}}^{\max}$")
    if k == 0:
        ax.legend(fontsize=8, frameon=False, loc="upper left")
        # annotate S3-S2 and S5-S4 relative differences at each level
        s2, s3, s4, s5 = (series(J, x) for x in ("S2", "S3", "S4", "S5"))
        for i, p in enumerate(ppp):
            if s2[i] >= 100:
                ax.annotate(f"NICE {100*(s3[i]-s2[i])/s2[i]:+.0f}%", (p, s2[i]), textcoords="offset points",
                            xytext=(0, 9), ha="center", fontsize=6.5, color="tab:green")

# panel (c): slices at s = 1 (joint)
bot = gs[1].subgridspec(1, len(scen), wspace=0.06)
maps = {sc: np.load(sd / f"firing_joint_{sc}_s1.npz")["spike_count_map"] for sc in scen}
ref = maps["S2"]; pk = np.unravel_index(int(np.argmax(ref)), ref.shape) if ref.max() > 0 else tuple(n // 2 for n in ref.shape)
ix, jy, kz = pk
vmax = max(int(m.max()) for m in maps.values()) or 1
dx_mm = meta["voxel_mm3"] ** (1 / 3)
ny, nz = ref.shape[1], ref.shape[2]
for i, sc in enumerate(scen):
    ax = fig.add_subplot(bot[0, i])
    half = int(round(20.0 / dx_mm))
    y0, y1 = max(jy - half, 0), min(jy + half, ny); z0, z1 = max(kz - half, 0), min(kz + half, nz)
    im = ax.imshow(maps[sc][ix, y0:y1, z0:z1].T, cmap="hot", vmin=0, vmax=vmax,
                   extent=(y0 * dx_mm, y1 * dx_mm, z0 * dx_mm, z1 * dx_mm), origin="lower", aspect="equal")
    h = J[f"{sc}_s1"]
    ax.set_title(f"{labels[sc]}\n{h['n_firing_voxels']} vox, {h['total_spikes']} spk", fontsize=8.5)
    ax.set_xlabel("y (mm)"); ax.set_ylabel("z (mm)" if i == 0 else "")
    if i: ax.set_yticks([])
cb = fig.colorbar(im, ax=fig.axes[2:], location="right", shrink=0.8, pad=0.02); cb.set_label("spikes / voxel")
fig.text(0.5, 0.445, f"(c) firing maps on the focal y-z plane at the canonical level (PPP = {ppp[s_levels.index(1.0)]:.2f} MPa, joint scaling), 40 mm window around the focus",
         ha="center", fontsize=10)
for p in (fig_dir / "fig9_joint_mechanism_sweep.pdf", fig_dir / "fig9_joint_mechanism_sweep.png", sd / "fig9_joint_mechanism_sweep.png"):
    fig.savefig(p, dpi=170, bbox_inches="tight"); print("wrote", p)

# ---- numbers for the text
def pct(a, b):
    return 100.0 * (a - b) / b if b else float("nan")
mac = {}
i1 = s_levels.index(1.0); itop = len(s_levels) - 1
s1, s2, s3, s4, s5 = (series(J, x) for x in scen)
mac["JSppTop"] = f"{ppp[itop]:.2f}"; mac["JSepsTop"] = f"{eps[itop]:.1e}".replace("e-0", "\\times10^{-").replace("e-", "\\times10^{-") + "}"
mac["JSdTtop"] = f"{meta['dT_focus_canonical_K'] * s_levels[itop]**2 * 1e3:.0f}"
mac["JSniceCan"] = f"{pct(s3[i1], s2[i1]):+.0f}"; mac["JSniceTop"] = f"{pct(s3[itop], s2[itop]):+.0f}"
mac["JSinactCan"] = f"{pct(s4[i1], s2[i1]):+.0f}"; mac["JSinactTop"] = f"{pct(s4[itop], s2[itop]):+.0f}"
mac["JStrpCan"] = f"{pct(s5[i1], s4[i1]):+.0f}"; mac["JStrpTop"] = f"{pct(s5[itop], s4[itop]):+.0f}"
mac["JSfullCan"] = f"{pct(s5[i1], s2[i1]):+.0f}"; mac["JSfullTop"] = f"{pct(s5[itop], s2[itop]):+.0f}"
mac["JSsTwoCan"] = f"{int(s2[i1]):,}".replace(",", "\\,"); mac["JSsThreeCan"] = f"{int(s3[i1]):,}".replace(",", "\\,")
mac["JSsFourCan"] = f"{int(s4[i1]):,}".replace(",", "\\,"); mac["JSsFiveCan"] = f"{int(s5[i1]):,}".replace(",", "\\,")
mac["JSsOneMax"] = f"{int(s1.max()):,}".replace(",", "\\,")
# first level with any S2 firing
first = next((f"{ppp[i]:.2f}" for i in range(len(s_levels)) if s2[i] > 0), "none"); mac["JSthreshPPP"] = first
if S is not None:
    t2, t3 = series(S, "S2"), series(S, "S3")
    mac["SSniceTop"] = f"{pct(t3[itop], t2[itop]):+.1f}"
(REPO / "data/joint_sweep_numbers.tex").write_text(
    "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in mac.items()) + "\n")
print(json.dumps(mac, indent=1))
