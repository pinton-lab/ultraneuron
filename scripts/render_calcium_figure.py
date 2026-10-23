"""Supplementary figure: calcium outputs of the canonical Halle/dACC run (rest start).

Reads data/halle_dacc/neural_results.npz, data/halle_dacc/calcium/somatic_session.npz and
data/halle_dacc/calcium/piezo_ca_canonical.npz; writes figures/fig_calcium_session.{png,pdf}
(manuscript figures dir) and data/halle_dacc/calcium/fig_calcium_session.png.

Top row: focal-plane maps of spikes per burst, Piezo1-pool peak [Ca] (soma) and somatic-pool
plateau [Ca] over the theta-burst session. Bottom row: Piezo1-pool [Ca] in dendrite and soma of the
focal voxel during one burst, and the somatic-pool [Ca] of the focal voxel over the 80 s session.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
D = REPO / "data/halle_dacc"
neu = np.load(D / "neural_results.npz")
som = np.load(D / "calcium/somatic_session.npz")
pz = np.load(D / "calcium/piezo_ca_canonical.npz")

DX = 1.232                                   # analysis grid spacing, mm
fx, fy, fz = (int(v) for v in som["focal_voxel"])
counts = neu["spike_count_map"].astype(float)
plateau = som["plateau_map_uM"].astype(float)
pz_soma = pz["ca_peak_map"][1].astype(float)
ca_rest = float(som["Ca_rest_uM"])


def plane(a):
    """Axial (beam) x lateral plane through the focal voxel, beam axis vertical."""
    return a[:, :, fz]


fig = plt.figure(figsize=(11.5, 7.2), constrained_layout=True)
gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1])
ext = [0, counts.shape[1] * DX, counts.shape[0] * DX, 0]
maps = [
    (counts, "Spikes per 20 ms burst", "viridis", None, None),
    (pz_soma, r"Piezo1 pool, soma peak ($\mu$M)", "magma", ca_rest, None),
    (plateau, r"Somatic pool, plateau ($\mu$M)", "magma", ca_rest, None),
]
for j, (arr, title, cmap, rest, _) in enumerate(maps):
    ax = fig.add_subplot(gs[0, j])
    a = plane(arr).copy()
    if rest is not None:
        a = np.where(np.isfinite(a), a, rest)
    im = ax.imshow(a, cmap=cmap, extent=ext, aspect="equal", interpolation="nearest")
    ax.plot((fy + 0.5) * DX, (fx + 0.5) * DX, marker="+", color="w", ms=9, mew=1.5)
    ax.set_title(f"({'abc'[j]}) {title}", fontsize=9.5, loc="left")
    ax.set_xlabel("lateral (mm)")
    if j == 0:
        ax.set_ylabel("axial, along the beam (mm)")
    ax.set_xlim((fy - 18) * DX, (fy + 18) * DX)
    ax.set_ylim((fx + 30) * DX, (fx - 30) * DX)
    fig.colorbar(im, ax=ax, shrink=0.85)

# (d) Piezo1-pool [Ca] during one burst at the focal voxel
ax = fig.add_subplot(gs[1, 0:1])
row = int(np.flatnonzero((pz["spike_voxel_idx"] == [fx, fy, fz]).all(axis=1))[0])
t_b = pz["ca_trace_t_ms"]
for c, name, col in ((0, "dendrite", "#1f77b4"), (1, "soma", "#d62728")):
    ax.plot(t_b, pz["ca_trace_firing"][row, c], color=col, lw=1.6, label=name)
s0, s1 = pz["spike_offsets"][row], pz["spike_offsets"][row + 1]
for ts in pz["spike_times_ms"][s0:s1]:
    ax.axvline(ts, color="0.6", lw=0.7, ls=":")
ax.set_xlabel("time in burst (ms)")
ax.set_ylabel(r"$[\mathrm{Ca}^{2+}]$ ($\mu$M)")
ax.set_title(r"(d) Piezo1-pool $[\mathrm{Ca}^{2+}]$, focal voxel ($\tau$ = %d ms)" % float(pz["tau_decay_ms"]),
             fontsize=9.5, loc="left")
ax.legend(frameon=False, fontsize=8, loc="upper left")

# (e) somatic-pool [Ca] over the theta-burst session at the focal voxel
ax = fig.add_subplot(gs[1, 1:3])
t_s = som["t_ms"] / 1e3
ax.plot(t_s, som["focal_trace_uM"], color="#d62728", lw=0.8)
n_b, per = int(som["n_bursts"]), float(som["period_ms"]) / 1e3
ax.axvspan(0, n_b * per, color="0.92", zorder=0)
ax.axhline(ca_rest, color="0.5", lw=0.8, ls="--")
ax.set_xlabel("time from session onset (s)")
ax.set_ylabel(r"$[\mathrm{Ca}^{2+}]$ ($\mu$M)")
ax.set_title(r"(e) Somatic-pool $[\mathrm{Ca}^{2+}]$, focal voxel, %d bursts of %d ms every %d ms "
             r"($\tau$ = %.1f s, %.2f $\mu$M per spike)" % (n_b, float(som["burst_ms"]), float(som["period_ms"]),
                                                            float(som["tau_ms"]) / 1e3, float(som["dCa_per_spike_uM"])),
             fontsize=9.5, loc="left")
ins = ax.inset_axes([0.2, 0.12, 0.3, 0.42])
m = t_s <= 2.0
ins.plot(t_s[m], som["focal_trace_uM"][m], color="#d62728", lw=0.9)
ins.set_xlabel("s", fontsize=7)
ins.tick_params(labelsize=7)
ins.set_title("first 2 s", fontsize=7)

out = [REPO / "figures/fig_calcium_session.png",
       REPO / "figures/fig_calcium_session.pdf",
       D / "calcium/fig_calcium_session.png"]
for p in out:
    fig.savefig(p, dpi=170)
    print("wrote", p)
