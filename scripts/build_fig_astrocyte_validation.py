"""Astrocytic TRPA1-relay validation figure — the Oh et al. 2019 controls.

Three calibration anchors, computed directly from the relay primitives
(reproducible by re-running; the same scenarios are asserted in
tests/test_astrocyte.py):

  (a) TRPA1 knockout / antagonist (g_TRPA1 -> 0) abolishes the relay:
      astrocytic Ca2+ / glutamate stay at rest.
  (b) AP5 (NMDA g_max -> 0) abolishes neuronal excitation while the
      astrocytic Ca2+ / glutamate transients persist unchanged.
  (c) The relay is slow: astrocytic Ca2+ rises over hundreds of ms, ~5
      orders of magnitude slower than the 500 kHz carrier cycle (2 us).
"""
from __future__ import annotations

import copy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
import sys

sys.path.insert(0, str(REPO_ROOT))

from ultraneuron import (  # noqa: E402
    TRPA1,
    AstrocyteRelay,
    NMDAReceptor,
    simulate_astrocyte_relay,
)

OUT_DIR = REPO_ROOT / "figures"
DRIVE_T = 10.0   # mN/m — focal-tension drive, well above TRPA1 T_half (6 mN/m)
CARRIER_HZ = 500e3


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)

    # (a) TRPA1 knockout abolishes the relay
    intact = AstrocyteRelay()
    ko = AstrocyteRelay()
    ko.trpa1 = copy.deepcopy(TRPA1)
    ko.trpa1.g_density = 0.0
    tci = intact.relay_timecourse(DRIVE_T, dt=1.0, duration_ms=1500.0)
    tck = ko.relay_timecourse(DRIVE_T, dt=1.0, duration_ms=1500.0)
    ax = axes[0]
    ax.plot(tci["t"] / 1e3, tci["glutamate"], color="tab:blue", lw=2, label="intact")
    ax.plot(tck["t"] / 1e3, tck["glutamate"], color="tab:red", lw=2, ls="--",
            label=r"TRPA1 KO ($\bar{g}=0$)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel(r"glutamate ($\mu$M)")
    ax.set_title("(a) TRPA1 knockout abolishes relay\n(necessary sensor; Oh 2019)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (b) AP5 abolishes neuronal excitation; relay (Ca/glu) persists
    res_intact = simulate_astrocyte_relay(peak_tension_mNm=DRIVE_T, dt=0.025,
                                          duration_ms=500.0)
    res_ap5 = simulate_astrocyte_relay(peak_tension_mNm=DRIVE_T,
                                       nmda=NMDAReceptor(g_max=0.0),
                                       dt=0.025, duration_ms=500.0)
    ax = axes[1]
    ax.plot(res_intact["t"] / 1e3, res_intact["V_neuron"], color="tab:green",
            lw=1.5, label="neuron V, intact")
    ax.plot(res_ap5["t"] / 1e3, res_ap5["V_neuron"], color="tab:gray",
            lw=1.5, label="neuron V, AP5")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("neuron $V_m$ (mV)", color="black")
    ax2 = ax.twinx()
    # Glutamate is identical intact vs AP5 (relay is upstream of NMDA).
    ax2.plot(res_intact["t"] / 1e3, res_intact["glutamate"], color="tab:blue",
             lw=2, ls=":", label="glutamate (both)")
    ax2.set_ylabel(r"glutamate ($\mu$M)", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    ax.set_title("(b) AP5 abolishes excitation,\nrelay persists (Oh 2019)")
    lines1, lab1 = ax.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lab1 + lab2, fontsize=7, loc="center right")

    # (c) Slow timescale: Ca rise vs the carrier cycle
    ax = axes[2]
    Ca = tci["Ca"]
    peak = Ca.max()
    half = intact.calcium.Ca_rest + 0.5 * (peak - intact.calcium.Ca_rest)
    idx = int(np.argmax(Ca >= half))
    t_half_ms = tci["t"][idx]
    ax.plot(tci["t"], Ca, color="tab:red", lw=2)
    ax.axvline(t_half_ms, color="grey", ls="--", lw=1,
               label=f"$t_{{1/2}}$ = {t_half_ms:.0f} ms")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel(r"[Ca$^{2+}$]$_\mathrm{astro}$ ($\mu$M)")
    carrier_us = 1e6 / CARRIER_HZ
    ax.set_title(f"(c) Slow relay: $t_{{1/2}}$ $\\approx${t_half_ms:.0f} ms\n"
                 f"vs {carrier_us:.0f} $\\mu$s carrier cycle "
                 f"($\\sim$10$^5\\times$ slower)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    for ext in ("pdf", "png"):
        out = OUT_DIR / f"fig_astrocyte_validation.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"wrote {out}")
    plt.close(fig)

    # Print the asserted anchors for the caption / sanity.
    print(f"  (a) glu intact peak={tci['glutamate'].max():.2f} uM, "
          f"KO peak={tck['glutamate'].max():.3f} uM")
    print(f"  (b) depol intact={res_intact['depolarisation']:.1f} mV, "
          f"AP5={res_ap5['depolarisation']:.2f} mV; "
          f"glu identical={np.allclose(res_intact['glutamate'], res_ap5['glutamate'])}")
    print(f"  (c) Ca t_half={t_half_ms:.0f} ms vs carrier {carrier_us:.0f} us")


if __name__ == "__main__":
    main()
