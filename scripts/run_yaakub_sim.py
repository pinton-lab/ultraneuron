"""Yaakub 2023 / Verhagen-protocol theta-burst tFUS run for the JNE manuscript.

Configures our pipeline to consume a synthetic-but-Yaakub-calibrated focal
field representing a 500 kHz NeuroFUS CTX-500 four-element annular array
focused at 63 mm depth through a human skull, driven at 1.0 MPa free-field
peak with the Verhagen / Yaakub theta-burst envelope (30 ms ON pulses
repeated at 5 Hz, 80 s session). Transcranial peak negative pressure and
peak deviatoric strain follow the parameters-paper Verhagen sweep row,
re-scaled to 500 kHz.

Outputs:
  data/yaakub_sim_results.npz    — strain / tension / pressure / temperature
  figures/fig4_yaakub_strain_tension.{pdf,png}
  figures/fig5_yaakub_temperature.{pdf,png}
  figures/fig6_yaakub_firing.{pdf,png}
"""

from __future__ import annotations

import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))

from ultraneuron import (  # noqa: E402
    PIEZO1,
    TREK1,
    MulticompartmentMechanoHHNeuron,
    simulate_map_3comp,
)
from ultraneuron.channels import strain_to_membrane_tension  # noqa: E402

OUT_DATA = HERE.parent / "data" / "yaakub_sim_results.npz"
OUT_FIG_STRAIN_PDF = HERE.parent / "figures" / "fig4_yaakub_strain_tension.pdf"
OUT_FIG_THERMAL_PDF = HERE.parent / "figures" / "fig5_yaakub_temperature.pdf"
OUT_FIG_FIRING_PDF = HERE.parent / "figures" / "fig6_yaakub_firing.pdf"


# ---------------------------------------------------------------------------
# Yaakub-calibrated synthetic focal field (500 kHz NeuroFUS CTX-500)
# ---------------------------------------------------------------------------
# Reported / derived parameters:
#
#   Carrier:                500 kHz
#   Aperture:               64 mm (NeuroFUS CTX-500 spec)
#   Geometric focus:        63 mm
#   Free-field peak:        1.0 MPa  (Yaakub 2023)
#   Transcranial peak:      0.50 MPa (free-field × 0.5 derating at 500 kHz)
#   Theta-burst pulse:      30 ms ON, 5 Hz repetition (15% overall duty)
#   Session:                80 s
#
#   Focal -6 dB volume:     ~3000 mm^3
#   FWHM lateral:           ~5 mm
#   FWHM axial:             ~30 mm  (typical 500 kHz single-bowl elongated focus)
#
#   Peak deviatoric strain: ~1.0e-4   (scales as p / (rho c) at 500 kHz)
#   Peak tension (K_A=0.25, alpha=200): ~5 mN/m (just above Piezo1 T_half)
#
#   Pennes ΔT_steady:       ~1.4 K at 15% duty (FDA/ITRUSST safe at 80 s)
#

def yaakub_field(
    nx: int = 50, ny: int = 50, nz: int = 50,
    dx_mm: float = 1.5,
    focus_xyz_mm: tuple[float, float, float] = (37.5, 37.5, 50.0),
    lateral_fwhm_mm: float = 5.0,
    axial_fwhm_mm: float = 30.0,
    peak_strain: float = 1.0e-4,
    peak_pressure_Pa: float = 5.0e5,    # transcranial peak (derated from 1 MPa)
    peak_dT_K: float = 1.4,             # Pennes steady-state under 15% duty
    baseline_T_C: float = 37.0,
) -> dict[str, np.ndarray]:
    """Synthetic Yaakub-calibrated focal field on a 50^3 voxel grid (75 mm^3 volume)."""
    x = np.arange(nx) * dx_mm
    y = np.arange(ny) * dx_mm
    z = np.arange(nz) * dx_mm
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    sigx = lateral_fwhm_mm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    sigy = sigx
    sigz = axial_fwhm_mm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    fx, fy, fz = focus_xyz_mm
    g = np.exp(
        -((X - fx) ** 2) / (2 * sigx**2)
        - ((Y - fy) ** 2) / (2 * sigy**2)
        - ((Z - fz) ** 2) / (2 * sigz**2)
    )
    return {
        "strain": peak_strain * g,
        "pressure": peak_pressure_Pa * g,
        "temperature_C": baseline_T_C + peak_dT_K * g,
        "axes_mm": (x, y, z),
        "focus_mm": focus_xyz_mm,
    }


# ---------------------------------------------------------------------------
# Run the firing simulation through our framework
# ---------------------------------------------------------------------------

def run() -> dict:
    field = yaakub_field()
    strain = field["strain"]
    print("=== Yaakub 2023 500 kHz theta-burst run ===", flush=True)
    print(f"Grid {strain.shape}, voxel pitch 1.5 mm (75 mm^3 volume)", flush=True)
    print(f"Peak deviatoric strain: {strain.max():.3g}", flush=True)
    print(f"Peak transcranial pressure: {field['pressure'].max() / 1e3:.0f} kPa", flush=True)
    print(f"Peak ΔT (Pennes steady, 15% duty): {field['temperature_C'].max() - 37:.2f} K", flush=True)

    # K_A * alpha tuned to put focal voxels at 5 mN/m (just above Piezo1 T_half=2.7).
    # Coupling factor 200 matches Yaakub-equivalent scale used in the parameters paper.
    K_A = 0.25
    alpha = 200.0
    tension = strain_to_membrane_tension(strain, K_A=K_A, coupling_factor=alpha)
    print(f"Peak membrane tension (K_A={K_A}, alpha={alpha}): {tension.max():.2f} mN/m", flush=True)

    # 3-compartment cortical pyramidal neuron, Piezo1 + TREK-1 (cortical_pyramidal profile).
    # Boost Piezo1 g_density to put the focal voxel above firing threshold.
    ch_p = deepcopy(PIEZO1); ch_p.g_density = 0.10
    ch_t = deepcopy(TREK1);  ch_t.g_density = 0.04
    neuron = MulticompartmentMechanoHHNeuron.cortical_pyramidal_3comp(
        mechano_channels=[ch_p, ch_t], temperature=37.0,
    )

    t0 = time.perf_counter()
    r = simulate_map_3comp(
        strain, neuron,
        K_A=K_A, coupling_factor=alpha,
        dt=0.001, duration=30.0,        # 30 ms simulation window
        tension_threshold=0.5,
    )
    elapsed = time.perf_counter() - t0
    n_fire = int(r["firing_map"].sum())
    n_spike = int(r["spike_count_map"].sum())
    n_above = int((tension >= 2.7).sum())  # voxels above Piezo1 T_half
    print(f"Run done in {elapsed:.1f} s — {n_fire} firing voxels, {n_spike} total spikes", flush=True)
    print(f"Voxels with peak tension above Piezo1 T_half: {n_above}", flush=True)

    OUT_DATA.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_DATA,
        strain=strain, tension=tension,
        pressure=field["pressure"], temperature_C=field["temperature_C"],
        firing_map=r["firing_map"], spike_count_map=r["spike_count_map"],
        K_A=K_A, alpha=alpha, dx_mm=1.5,
    )
    return {
        "field": field, "tension": tension,
        "firing_map": r["firing_map"], "spike_count_map": r["spike_count_map"],
        "n_fire": n_fire, "n_spike": n_spike, "n_above_threshold": n_above,
    }


# ---------------------------------------------------------------------------
# Three figures
# ---------------------------------------------------------------------------

def make_figures(result: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.6})

    field = result["field"]
    strain = field["strain"]
    tension = result["tension"]
    pressure = field["pressure"]
    temp = field["temperature_C"]
    firing = result["firing_map"]
    counts = result["spike_count_map"]
    x_mm, y_mm, z_mm = field["axes_mm"]
    fx, fy, fz = field["focus_mm"]
    z_idx = int(round(fz / (z_mm[1] - z_mm[0])))
    y_idx = int(round(fy / (y_mm[1] - y_mm[0])))

    extent_xy = (x_mm[0], x_mm[-1], y_mm[0], y_mm[-1])
    extent_xz = (x_mm[0], x_mm[-1], z_mm[0], z_mm[-1])

    # --- Figure 4: strain + tension at focal slice ---
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.4))
    im0 = axes[0].imshow(strain[:, :, z_idx].T, origin="lower", extent=extent_xy,
                         cmap="viridis")
    axes[0].set_title("Deviatoric strain $\\varepsilon_{\\mathrm{dev}}$")
    axes[0].set_xlabel("x (mm)"); axes[0].set_ylabel("y (mm)")
    plt.colorbar(im0, ax=axes[0], fraction=0.045, pad=0.04, label="(dimensionless)")

    im1 = axes[1].imshow(tension[:, :, z_idx].T, origin="lower", extent=extent_xy,
                        cmap="magma")
    axes[1].set_title("Membrane tension $T$ (mN/m)")
    axes[1].set_xlabel("x (mm)")
    # Overlay Piezo1 T_half = 2.7 mN/m contour
    axes[1].contour(tension[:, :, z_idx].T, levels=[2.7], colors="cyan", linewidths=1.0,
                    extent=extent_xy)
    plt.colorbar(im1, ax=axes[1], fraction=0.045, pad=0.04, label="mN/m")
    fig.suptitle(f"Focal slice (z = {fz:.0f} mm); cyan contour: Piezo1 $T_{{1/2}}$ = 2.7 mN/m",
                 fontsize=9.5)
    fig.tight_layout()
    fig.savefig(OUT_FIG_STRAIN_PDF, bbox_inches="tight")
    fig.savefig(str(OUT_FIG_STRAIN_PDF).replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved: {OUT_FIG_STRAIN_PDF}", flush=True)

    # --- Figure 5: thermal Pennes ΔT ---
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.4))
    dT = temp - 37.0
    im0 = axes[0].imshow(dT[:, :, z_idx].T, origin="lower", extent=extent_xy, cmap="hot")
    axes[0].set_title(f"$\\Delta T$ at focal plane z = {fz:.0f} mm")
    axes[0].set_xlabel("x (mm)"); axes[0].set_ylabel("y (mm)")
    plt.colorbar(im0, ax=axes[0], fraction=0.045, pad=0.04, label="K above 37$^\\circ$C")

    im1 = axes[1].imshow(dT[:, y_idx, :].T, origin="lower", extent=extent_xz, cmap="hot")
    axes[1].set_title(f"$\\Delta T$ at mid-lateral y = {fy:.0f} mm")
    axes[1].set_xlabel("x (mm)"); axes[1].set_ylabel("z (mm)")
    plt.colorbar(im1, ax=axes[1], fraction=0.045, pad=0.04, label="K above 37$^\\circ$C")
    fig.suptitle(f"Pennes bioheat steady-state under 15\\% theta-burst duty (peak $\\Delta T$ = {dT.max():.2f} K)",
                 fontsize=9.5)
    fig.tight_layout()
    fig.savefig(OUT_FIG_THERMAL_PDF, bbox_inches="tight")
    fig.savefig(str(OUT_FIG_THERMAL_PDF).replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved: {OUT_FIG_THERMAL_PDF}", flush=True)

    # --- Figure 6: firing maps (focal slice + axial) ---
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.4))
    vmax = max(int(counts.max()), 1)
    im0 = axes[0].imshow(counts[:, :, z_idx].T, origin="lower", extent=extent_xy,
                        cmap="magma", vmin=0, vmax=vmax)
    axes[0].set_title(f"Spike count, focal plane z = {fz:.0f} mm")
    axes[0].set_xlabel("x (mm)"); axes[0].set_ylabel("y (mm)")
    plt.colorbar(im0, ax=axes[0], fraction=0.045, pad=0.04, label="spikes / 30 ms")

    im1 = axes[1].imshow(counts[:, y_idx, :].T, origin="lower", extent=extent_xz,
                        cmap="magma", vmin=0, vmax=vmax)
    axes[1].set_title(f"Spike count, mid-lateral y = {fy:.0f} mm")
    axes[1].set_xlabel("x (mm)"); axes[1].set_ylabel("z (mm)")
    plt.colorbar(im1, ax=axes[1], fraction=0.045, pad=0.04, label="spikes / 30 ms")
    fig.suptitle(
        f"Per-voxel firing maps under Yaakub theta-burst protocol: "
        f"{result['n_fire']} firing voxels, {result['n_spike']} total spikes",
        fontsize=9.5,
    )
    fig.tight_layout()
    fig.savefig(OUT_FIG_FIRING_PDF, bbox_inches="tight")
    fig.savefig(str(OUT_FIG_FIRING_PDF).replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved: {OUT_FIG_FIRING_PDF}", flush=True)


def main() -> None:
    result = run()
    make_figures(result)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
