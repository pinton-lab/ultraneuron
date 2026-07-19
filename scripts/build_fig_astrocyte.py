"""Compose the astrocytic TRPA1→Ca²⁺→glutamate→NMDA mechanism figure.

Three panels from ``data/<sim>/astrocyte_results.npz``:
  (a) focal-voxel relay timecourse — the slow Ca²⁺ / glutamate rise
      (hundreds of ms), the pathway's defining slowness.
  (b) astrocytic-NMDA excitation drive vs cumulative sonication on-time —
      the duty-cumulative signature that distinguishes this pathway from
      the per-pulse channel mechanisms (S1–S5).
  (c) per-voxel astrocytic-NMDA excitation map on the focal plane.

Reads ``HALLE_SIM_NAME`` (default halle_dacc), parallel to build_fig{3..7}.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
SIM = os.environ.get("HALLE_SIM_NAME", "halle_dacc")
DATA = REPO_ROOT / "data" / SIM / "astrocyte_results.npz"
OUT_DIR = REPO_ROOT / "figures"


def main() -> None:
    d = np.load(DATA, allow_pickle=True)
    t_ms = d["focal_t_ms"]
    Ca = d["focal_Ca_uM"]
    glu = d["focal_glu_uM"]
    ontime_s = d["ontime_s"]
    ontime_drive = d["ontime_drive"]
    exc = d["excitation_map"]
    focal_idx = tuple(int(x) for x in d["focal_idx"])
    focal_T = float(d["focal_tension_mNm"])
    duty = float(d["duty_cycle"])
    session_s = float(d["session_s"])
    tau_ca = float(d["tau_Ca_ms"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)

    # (a) relay timecourse — slow Ca²⁺ / glutamate rise
    ax = axes[0]
    ax.plot(t_ms / 1e3, Ca, color="tab:red", lw=2, label=r"astrocytic [Ca$^{2+}$] ($\mu$M)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel(r"[Ca$^{2+}$]$_\mathrm{astro}$ ($\mu$M)", color="tab:red")
    ax.tick_params(axis="y", labelcolor="tab:red")
    ax2 = ax.twinx()
    ax2.plot(t_ms / 1e3, glu, color="tab:blue", lw=2, ls="--")
    ax2.set_ylabel(r"glutamate ($\mu$M)", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    ax.axvline(tau_ca / 1e3, color="grey", ls=":", lw=1)
    ax.set_title(f"(a) Focal relay timecourse\n(slow: $\\tau_{{Ca}}$ = {tau_ca:.0f} ms)")

    # (b) excitation vs cumulative on-time — duty-cumulative signature
    ax = axes[1]
    ax.semilogx(ontime_s, ontime_drive, color="tab:purple", lw=2, marker="o", ms=3)
    ax.axvline(session_s, color="k", ls="--", lw=1,
               label=f"session = {session_s:.0f} s")
    ax.set_xlabel("cumulative sonication on-time (s)")
    ax.set_ylabel("astrocytic-NMDA drive (0–1)")
    ax.set_title(f"(b) Duty-cumulative drive\n(duty = {duty:.2f}, focal T = {focal_T:.1f} mN/m)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3, which="both")

    # (c) per-voxel excitation map — focal x–z plane (through focal y).
    # Depth (x) on the vertical axis with 0 at the top (radiology
    # convention, matching the acoustic/thermal/elastic ortho figures).
    ax = axes[2]
    sl = exc[:, focal_idx[1], :]  # (nx=depth, nz=elevation)
    im = ax.imshow(sl, origin="upper", aspect="auto", cmap="magma",
                   vmin=0.0, vmax=max(float(exc.max()), 1e-6))
    ax.plot(focal_idx[2], focal_idx[0], "c*", ms=10, mec="white")
    ax.set_xlabel("z (elevation, voxels)")
    ax.set_ylabel("x (depth, voxels)")
    ax.set_title("(c) Astrocytic-NMDA excitation map\n(focal x–z plane; depth on y, 0 at top)")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("excitation drive (0–1)")

    for ext in ("pdf", "png"):
        out = OUT_DIR / f"fig9_astrocyte_pathway.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
