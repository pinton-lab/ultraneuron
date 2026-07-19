"""Validation-report section for mechanosensitive synaptic / voltage-gated
transmission (M6) --- presynaptic release modulation (Tyler 2012), NMDA
stretch enhancement (Maneshi 2017), and the voltage-gated Na+/Ca2+ shift
(Kubanek 2016).

Reads ``synaptic_results.npz`` (written by ``scripts/run_synaptic_sim.py``)
and renders the three-panel mechanism figure plus a small metrics block.
If the npz is absent the section is silently skipped (the template gates on
``ctx.synaptic_outputs is defined``).
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
    V = d["focal_V_mV"]
    glu = d["focal_glu_uM"]
    sweep_T = d["sweep_T_mNm"]
    sweep_syn = d["sweep_synaptic_gain"]
    sweep_nmda = d["sweep_nmda_stretch"]
    sweep_nav = d["sweep_nav_gain"]
    sweep_combined = d["sweep_combined_gain"]
    sweep_spikes = d["sweep_spikes"]
    gain_map = d["gain_map"]
    focal_idx = tuple(int(x) for x in d["focal_idx"])
    focal_T = float(d["focal_tension_mNm"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)

    # (a) focal relay timecourse: post-synaptic V + cleft glutamate
    ax = axes[0]
    ax.plot(t_ms, V, color="tab:blue", lw=1.2)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("post-synaptic $V_m$ (mV)", color="tab:blue")
    ax.tick_params(axis="y", labelcolor="tab:blue")
    ax2 = ax.twinx()
    ax2.plot(t_ms, glu, color="tab:green", lw=2, ls="--")
    ax2.set_ylabel(r"cleft glutamate ($\mu$M)", color="tab:green")
    ax2.tick_params(axis="y", labelcolor="tab:green")
    ax.set_title(f"(a) Focal relay timecourse (T = {focal_T:.2f} mN/m)")

    # (b) dose-response: scalar gains + relay spike count vs tension
    ax = axes[1]
    ax.plot(sweep_T, sweep_syn, color="tab:purple", lw=1.8, marker="o", ms=3,
            label=r"presyn. $P_r(T)/P_r(0)$")
    ax.plot(sweep_T, sweep_nmda, color="tab:red", lw=1.8, marker="s", ms=3,
            label=r"NMDA $f_\mathrm{stretch}(T)$")
    ax.plot(sweep_T, sweep_nav, color="tab:orange", lw=1.8, marker="^", ms=3,
            label=r"Na$^+$ gain")
    ax.plot(sweep_T, sweep_combined, color="k", lw=2.2, label="combined")
    ax.set_xlabel("peak membrane tension (mN/m)")
    ax.set_ylabel("pathway gain ($\\geq 1$)")
    ax.set_title("(b) Mechanism dose-response")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(sweep_T, sweep_spikes, color="tab:gray", lw=1.4, ls=":", marker="x", ms=4)
    ax2.set_ylabel("relay spike count", color="tab:gray")
    ax2.tick_params(axis="y", labelcolor="tab:gray")

    # (c) combined-gain map on the focal x-z plane.
    # Depth (x) on the vertical axis with 0 at the top (radiology
    # convention, matching the acoustic/thermal/elastic ortho figures).
    ax = axes[2]
    sl = gain_map[:, focal_idx[1], :]  # (nx=depth, nz=elevation)
    im = ax.imshow(sl, origin="upper", aspect="auto", cmap="magma",
                   vmin=1.0, vmax=max(float(gain_map.max()), 1.0 + 1e-6))
    ax.plot(focal_idx[2], focal_idx[0], "c*", ms=10, mec="white")
    ax.set_xlabel("z (elevation, voxels)")
    ax.set_ylabel("x (depth, voxels)")
    ax.set_title("(c) Combined-gain map (focal x–z plane; depth on y, 0 at top)")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("combined pathway gain")

    fig.savefig(fig_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build(cfg, fig_dir: Path) -> dict[str, Any]:
    """Render the synaptic-mechanism section (skipped if no synaptic_results.npz)."""
    repo_root = Path(__file__).resolve().parents[2]
    sim_dir = (repo_root / cfg.OUT_DIR).parent
    if not sim_dir.is_absolute():
        sim_dir = (repo_root / sim_dir).resolve()

    npz_path = sim_dir / "synaptic_results.npz"
    if not npz_path.exists():
        print(f"  [synaptic_outputs] no {npz_path.name}; section skipped "
              f"(run scripts/run_synaptic_sim.py first)")
        return {}

    d = np.load(npz_path, allow_pickle=True)
    fig_name = "synaptic_pathway.png"
    _render_figure(d, fig_dir / fig_name)

    ctx = {
        "synaptic_outputs": {
            "present": True,
            "figure": fig_name,
            "focal_tension_mNm": float(d["focal_tension_mNm"]),
            "peak_combined_gain": float(d["peak_combined_gain"]),
            "n_recruited": int(d["n_recruited"]),
            "gain_threshold": float(d["gain_threshold"]),
            "focal_depolarisation_mV": float(d["focal_depolarisation_mV"]),
            "focal_n_spikes": int(d["focal_n_spikes"]),
            "Pr_baseline": float(d["Pr_baseline"]),
            "Pr_max": float(d["Pr_max"]),
            "k_mechano_nav": float(d["k_mechano_nav"]),
            "alpha_NMDA": float(d["alpha_NMDA"]),
        }
    }
    print(f"  [synaptic_outputs] focal T={ctx['synaptic_outputs']['focal_tension_mNm']:.2f} mN/m, "
          f"peak gain={ctx['synaptic_outputs']['peak_combined_gain']:.2f}, "
          f"recruited={ctx['synaptic_outputs']['n_recruited']}")
    return ctx
