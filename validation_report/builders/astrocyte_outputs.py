"""Validation-report section for the astrocytic TRPA1 → Ca²⁺ → glutamate
→ NMDA relay (Oh et al. 2019).

Reads ``astrocyte_results.npz`` (written by ``scripts/run_astrocyte_sim.py``)
and renders the three-panel mechanism figure plus a small metrics block.
If the npz is absent the section is silently skipped (the template gates on
``ctx.astrocyte_outputs is defined``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _render_figure(d, fig_path: Path) -> None:
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

    ax = axes[0]
    ax.plot(t_ms / 1e3, Ca, color="tab:red", lw=2)
    ax.set_xlabel("time (s)")
    ax.set_ylabel(r"[Ca$^{2+}$]$_\mathrm{astro}$ ($\mu$M)", color="tab:red")
    ax.tick_params(axis="y", labelcolor="tab:red")
    ax2 = ax.twinx()
    ax2.plot(t_ms / 1e3, glu, color="tab:blue", lw=2, ls="--")
    ax2.set_ylabel(r"glutamate ($\mu$M)", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    ax.axvline(tau_ca / 1e3, color="grey", ls=":", lw=1)
    ax.set_title(f"(a) Focal relay timecourse ($\\tau_{{Ca}}$ = {tau_ca:.0f} ms)")

    ax = axes[1]
    ax.semilogx(ontime_s, ontime_drive, color="tab:purple", lw=2, marker="o", ms=3)
    ax.axvline(session_s, color="k", ls="--", lw=1, label=f"session = {session_s:.0f} s")
    ax.set_xlabel("cumulative sonication on-time (s)")
    ax.set_ylabel("astrocytic-NMDA drive (0–1)")
    ax.set_title(f"(b) Duty-cumulative drive (duty = {duty:.2f})")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3, which="both")

    # Depth (x) on the vertical axis with 0 at the top (radiology
    # convention, matching the acoustic/thermal/elastic ortho figures).
    ax = axes[2]
    sl = exc[:, focal_idx[1], :]  # (nx=depth, nz=elevation)
    im = ax.imshow(sl, origin="upper", aspect="auto", cmap="magma",
                   vmin=0.0, vmax=max(float(exc.max()), 1e-6))
    ax.plot(focal_idx[2], focal_idx[0], "c*", ms=10, mec="white")
    ax.set_xlabel("z (elevation, voxels)")
    ax.set_ylabel("x (depth, voxels)")
    ax.set_title("(c) Excitation map (focal x–z plane; depth on y, 0 at top)")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("drive (0–1)")

    fig.savefig(fig_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build(cfg, fig_dir: Path) -> dict[str, Any]:
    """Render the astrocyte-relay section (skipped if no astrocyte_results.npz)."""
    repo_root = Path(__file__).resolve().parents[2]
    sim_dir = (repo_root / cfg.OUT_DIR).parent
    if not sim_dir.is_absolute():
        sim_dir = (repo_root / sim_dir).resolve()

    npz_path = sim_dir / "astrocyte_results.npz"
    if not npz_path.exists():
        print(f"  [astrocyte_outputs] no {npz_path.name}; section skipped "
              f"(run scripts/run_astrocyte_sim.py first)")
        return {}

    d = np.load(npz_path, allow_pickle=True)
    fig_name = "astrocyte_pathway.png"
    _render_figure(d, fig_dir / fig_name)

    ctx = {
        "astrocyte_outputs": {
            "present": True,
            "figure": fig_name,
            "focal_tension_mNm": float(d["focal_tension_mNm"]),
            "peak_drive": float(d["peak_drive"]),
            "n_recruited": int(d["n_recruited"]),
            "drive_threshold": float(d["drive_threshold"]),
            "tau_Ca_ms": float(d["tau_Ca_ms"]),
            "trpa1_T_half_mNm": float(d["trpa1_T_half_mNm"]),
            "focal_depolarisation_mV": float(d["focal_depolarisation_mV"]),
            "focal_n_spikes": int(d["focal_n_spikes"]),
            "duty_cycle": float(d["duty_cycle"]),
            "session_s": float(d["session_s"]),
        }
    }
    print(f"  [astrocyte_outputs] focal T={ctx['astrocyte_outputs']['focal_tension_mNm']:.2f} mN/m, "
          f"peak drive={ctx['astrocyte_outputs']['peak_drive']:.3f}, "
          f"recruited={ctx['astrocyte_outputs']['n_recruited']}")
    return ctx
