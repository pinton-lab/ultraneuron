"""TIPS-transcranial mechanism-comparison strain sweep for the JNE manuscript.

Sweeps peak deviatoric strain across a published-relevant range and runs
all five pathway-combination scenarios at each strain level, so we can
read out where each mechanism candidate begins to diverge from the
others. The pressure amplitude tracks the strain via a fixed
strain-to-pressure ratio that mirrors the linear-elastic regime
(ε_dev ≈ 0.01 ↔ P_ac ≈ 150 kPa from a typical shear FDTD run).

Outputs:
  data/tips_sweep_results.npz       — per-scenario, per-strain firing/count maps
  figures/fig4_tips_mechanism_comparison.{pdf,png}   — Figure 4 (replaces the previous version)

Runtime: ~3-6 minutes on numpy.
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
    TRPV1,
    TRPV4,
    CalciumDynamics,
    KCaChannel,
    MulticompartmentMechanoHHNeuron,
    NICEModel,
    simulate_map,
)
from ultraneuron.channels import strain_to_membrane_tension  # noqa: E402
from ultraneuron.neuron import MechanoHHNeuron  # noqa: E402

OUT_DATA = HERE.parent / "data" / "tips_sweep_results.npz"
OUT_FIG_PDF = HERE.parent / "figures" / "fig4_tips_mechanism_comparison.pdf"
OUT_FIG_PNG = HERE.parent / "figures" / "fig4_tips_mechanism_comparison.png"


# ---------------------------------------------------------------------------
# Synthetic TIPS focal field
# ---------------------------------------------------------------------------

def synthetic_field(
    peak_strain: float,
    nx: int = 30, ny: int = 30, nz: int = 30,
    dx_mm: float = 2.0,
    focus_xyz_mm: tuple[float, float, float] = (30.0, 30.0, 40.0),
    lateral_fwhm_mm: float = 6.0,
    axial_fwhm_mm: float = 18.0,
    strain_to_pressure_Pa_per_unit: float = 1.5e5 / 0.012,  # 12.5 MPa per unit ε
) -> dict[str, np.ndarray]:
    x = np.arange(nx) * dx_mm
    y = np.arange(ny) * dx_mm
    z = np.arange(nz) * dx_mm
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    sigx = lateral_fwhm_mm / (2 * np.sqrt(2 * np.log(2)))
    sigy = sigx
    sigz = axial_fwhm_mm / (2 * np.sqrt(2 * np.log(2)))
    fx, fy, fz = focus_xyz_mm
    gauss = np.exp(
        -((X - fx) ** 2) / (2 * sigx**2)
        - ((Y - fy) ** 2) / (2 * sigy**2)
        - ((Z - fz) ** 2) / (2 * sigz**2)
    )
    return {
        "strain": peak_strain * gauss,
        "pressure": peak_strain * strain_to_pressure_Pa_per_unit * gauss,
        "axes_mm": (x, y, z),
    }


# ---------------------------------------------------------------------------
# Scenario builders. Piezo1 is the only mechano channel here so that the
# sweep cleanly isolates the effect of each opt-in extension. (TREK1 was
# the reason the previous-run S1 = 0; including only Piezo1 + the cell-type
# channel-library defaults — boosted via g_density — keeps the comparison
# focused on Phase-2/3/4 contributions.)
# ---------------------------------------------------------------------------

def piezo1_boosted():
    ch = deepcopy(PIEZO1)
    ch.g_density = 0.05  # boosted ~10x to bring the focal voxel above threshold
    return ch


def _multicomp(*, with_nice=False, with_inact_ca=False, with_trp=False):
    kwargs = dict(
        mechano_channels=[piezo1_boosted()],
        temperature=37.0,
    )
    if with_nice:
        kwargs["nice_model"] = NICEModel()
        kwargs["nice_compartments"] = (0, 1, 2)
    if with_inact_ca:
        kwargs["piezo_inactivation"] = True
        kwargs["calcium_dynamics"] = CalciumDynamics()
        kwargs["kca_channels"] = [KCaChannel(g_density=0.3)]
        kwargs["kca_compartments"] = (1,)
    if with_trp:
        kwargs["trp_channels"] = [deepcopy(TRPV1), deepcopy(TRPV4)]
        kwargs["trp_compartments"] = (1,)
    return MulticompartmentMechanoHHNeuron.cortical_pyramidal_3comp(**kwargs)


def _run_singlecomp(strain, pressure, dt, duration):
    neuron = MechanoHHNeuron.cortical_pyramidal(
        temperature=37.0, mechano_channels=[piezo1_boosted()],
    )
    r = simulate_map(strain, neuron, dt=dt, duration=duration, tension_threshold=0.5)
    return r["firing_map"], r["spike_count_map"]


def _run_multicomp_phase1(strain, pressure, dt, duration):
    return _run_per_voxel_pressure(_multicomp(), strain, np.zeros_like(pressure), dt, duration)


def _run_multicomp_NICE(strain, pressure, dt, duration):
    return _run_per_voxel_pressure(_multicomp(with_nice=True), strain, pressure, dt, duration)


def _run_multicomp_M5(strain, pressure, dt, duration):
    return _run_per_voxel_pressure(_multicomp(with_inact_ca=True), strain, np.zeros_like(pressure), dt, duration)


def _run_full(strain, pressure, dt, duration):
    return _run_per_voxel_pressure(_multicomp(with_nice=True, with_inact_ca=True, with_trp=True), strain, pressure, dt, duration)


def _run_per_voxel_pressure(neuron, strain, pressure, dt, duration):
    nx, ny, nz = strain.shape
    tension = strain_to_membrane_tension(strain, K_A=0.25, coupling_factor=1.0)
    mask = tension >= 0.5
    firing = np.zeros((nx, ny, nz), dtype=bool)
    counts = np.zeros((nx, ny, nz), dtype=np.int32)
    if not np.any(mask):
        return firing, counts
    tensions_flat = tension[mask].astype(np.float64)
    pressure_flat = pressure[mask].astype(np.float64)
    n_steps = int(duration / dt)
    sc = neuron._simulate_batch(
        tensions_flat, dt=dt, n_steps=n_steps,
        I_stim_per_compartment=np.array([0.0, 0.0, 0.0]),
        primary_compartment=1,
        pressure_envelope=pressure_flat,
    )
    counts[mask] = sc
    firing[mask] = sc > 0
    return firing, counts


SCENARIOS = [
    ("S1_singlecomp", "single-compartment\nPhase 1", _run_singlecomp, 0.01),
    ("S2_multicomp", "3-compartment\nPhase 1", _run_multicomp_phase1, 0.001),
    ("S3_NICE", "3-comp + NICE\n(Phase 2)", _run_multicomp_NICE, 0.001),
    ("S4_M5", "3-comp + M5\n(Phase 3)", _run_multicomp_M5, 0.001),
    ("S5_full", "Full pathway\n(Phase 1+2+3+4)", _run_full, 0.001),
]


# ---------------------------------------------------------------------------
# Sweep + figure
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== JNE Phase 4b: TIPS mechanism strain sweep ===", flush=True)

    strain_levels = np.array([0.005, 0.010, 0.015, 0.020, 0.030, 0.040])
    duration = 30.0  # ms

    n_scen = len(SCENARIOS)
    n_strain = len(strain_levels)
    n_firing = np.zeros((n_scen, n_strain), dtype=int)
    n_spikes = np.zeros((n_scen, n_strain), dtype=int)
    # Save the firing maps at the *middle* strain level for the slice panel
    middle_idx = len(strain_levels) // 2
    middle_maps = {}
    middle_field = None

    for j, peak_strain in enumerate(strain_levels):
        field = synthetic_field(peak_strain=float(peak_strain))
        if j == middle_idx:
            middle_field = field
        print(
            f"strain peak={peak_strain:.3f}, pressure peak={field['pressure'].max() / 1e3:.0f} kPa",
            flush=True,
        )
        for i, (key, label, runner, dt) in enumerate(SCENARIOS):
            t0 = time.perf_counter()
            firing, counts = runner(
                field["strain"], field["pressure"], dt=dt, duration=duration,
            )
            elapsed = time.perf_counter() - t0
            nf = int(firing.sum())
            ns = int(counts.sum())
            n_firing[i, j] = nf
            n_spikes[i, j] = ns
            print(
                f"  {key}: {nf:3d} firing voxels, {ns:4d} spikes, {elapsed:.1f}s",
                flush=True,
            )
            if j == middle_idx:
                middle_maps[key] = counts

    # --- Save data ---
    OUT_DATA.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_DATA,
        strain_levels=strain_levels,
        scenario_keys=np.array([s[0] for s in SCENARIOS]),
        scenario_labels=np.array([s[1] for s in SCENARIOS]),
        n_firing=n_firing,
        n_spikes=n_spikes,
        middle_strain=strain_levels[middle_idx],
        middle_strain_field=middle_field["strain"] if middle_field else np.zeros((1,)),
        **{f"middle_count__{k}": v for k, v in middle_maps.items()},
    )
    print(f"Saved: {OUT_DATA}", flush=True)

    # --- Figure ---
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6})

    fig = plt.figure(figsize=(7.0, 6.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.2, 1.0], hspace=0.45)

    # Top: total spike count vs peak strain across scenarios.
    ax = fig.add_subplot(gs[0])
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]
    markers = ["o", "s", "^", "D", "v"]
    for i, (key, label, _runner, _dt) in enumerate(SCENARIOS):
        ax.plot(
            strain_levels, n_spikes[i],
            marker=markers[i], linewidth=1.2, color=colors[i],
            label=label.replace("\n", " "),
        )
    ax.set_xscale("log")
    ax.set_xlabel(r"Peak deviatoric strain $\varepsilon_{\mathrm{dev}}$")
    ax.set_ylabel("Total spikes (30 ms window)")
    ax.set_title("(a) Mechanism comparison across strain regime")
    ax.legend(loc="upper left", fontsize=7.5, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Mark the strain at which the Phase-1 single-compartment model
    # first fires, as a visual cue for the firing threshold.
    if (n_spikes[0] > 0).any():
        thr_idx = int(np.argmax(n_spikes[0] > 0))
        ax.axvline(strain_levels[thr_idx], color="grey", linestyle="--", linewidth=0.6, alpha=0.6)

    # Bottom: firing-map slices at the middle strain level
    gs_bot = gs[1].subgridspec(1, n_scen, wspace=0.20)
    middle_field = synthetic_field(peak_strain=float(strain_levels[middle_idx]))
    z_idx = int(round(middle_field["strain"].shape[2] // 2 + 5))
    x_mm, y_mm, _ = middle_field["axes_mm"]
    extent = (x_mm[0], x_mm[-1], y_mm[0], y_mm[-1])
    vmax_local = max(int(middle_maps[k][:, :, z_idx].max()) for k in middle_maps) or 1
    for i, (key, label, _runner, _dt) in enumerate(SCENARIOS):
        ax_m = fig.add_subplot(gs_bot[0, i])
        if key in middle_maps:
            im = ax_m.imshow(
                middle_maps[key][:, :, z_idx].T,
                origin="lower", extent=extent, vmin=0, vmax=vmax_local, cmap="magma",
            )
        else:
            ax_m.imshow(np.zeros((middle_field["strain"].shape[0], middle_field["strain"].shape[1])),
                        origin="lower", extent=extent, vmin=0, vmax=1, cmap="magma")
        ax_m.set_title(label, fontsize=7.5)
        if i == 0:
            ax_m.set_ylabel("y (mm)")
        ax_m.set_xlabel("x (mm)")
    cbar = fig.colorbar(im, ax=fig.get_axes()[1:], fraction=0.018, pad=0.02,
                        label="spike count")
    fig.text(0.04, 0.40,
             f"(b) Firing-map slice at z = {z_idx*2} mm "
             f"(focal plane), $\\varepsilon$ = {strain_levels[middle_idx]:.3f}",
             fontsize=8.5)

    fig.savefig(OUT_FIG_PDF, bbox_inches="tight")
    fig.savefig(OUT_FIG_PNG, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved: {OUT_FIG_PDF}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
