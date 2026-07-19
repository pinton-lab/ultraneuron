"""TIPS-transcranial mechanism-comparison run for the JNE manuscript.

Generates a synthetic but TIPS-typical focal field (Gaussian deviatoric
strain, matching pressure envelope, mild thermal rise) and runs the
multi-compartment neuron under five pathway configurations to compare
predicted firing maps. The strain-tension and pressure values are
chosen to put the focal voxel above Piezo1 activation but well within
the model's validated regime; the synthetic field stands in for the
real shear-FDTD output until the upstream simulator is hooked up.

Outputs:
  data/tips_sim_results.npz   — firing/spike-count maps per scenario
  figures/fig4_tips_mechanism_comparison.{pdf,png}   — Figure 4

Runtime: ~1–3 minutes on numpy at the grid sizes below.
"""

from __future__ import annotations

import os
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np

# Make the in-tree package importable when run from the repo root.
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))

from ultraneuron import (  # noqa: E402
    PIEZO1,
    TRAAK,
    TREK1,
    TRPV1,
    TRPV4,
    CalciumDynamics,
    KCaChannel,
    MulticompartmentMechanoHHNeuron,
    NICEModel,
    simulate_map,
    simulate_map_3comp,
)
from ultraneuron.neuron import MechanoHHNeuron  # noqa: E402

OUT_DATA = HERE.parent / "data" / "tips_sim_results.npz"
OUT_FIG_PDF = HERE.parent / "figures" / "fig4_tips_mechanism_comparison.pdf"
OUT_FIG_PNG = HERE.parent / "figures" / "fig4_tips_mechanism_comparison.png"


# ---------------------------------------------------------------------------
# Synthetic TIPS focal field (deviatoric strain, pressure, temperature)
# ---------------------------------------------------------------------------

def synthetic_tips_field(
    nx: int = 30, ny: int = 30, nz: int = 30,
    dx_mm: float = 2.0,
    focus_xyz_mm: tuple[float, float, float] = (30.0, 30.0, 40.0),
    lateral_fwhm_mm: float = 6.0,
    axial_fwhm_mm: float = 18.0,
    peak_strain: float = 0.012,
    peak_pressure_Pa: float = 1.5e5,
    peak_dT_K: float = 0.4,
    baseline_T_C: float = 37.0,
) -> dict[str, np.ndarray]:
    """Return strain (deviatoric), pressure-amplitude, and absolute-T fields
    on a (nx, ny, nz) Cartesian grid. The fields share a Gaussian focal
    profile typical of a curved-bowl TIPS transducer."""
    x = np.arange(nx) * dx_mm
    y = np.arange(ny) * dx_mm
    z = np.arange(nz) * dx_mm
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    sigx = lateral_fwhm_mm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    sigy = sigx
    sigz = axial_fwhm_mm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    fx, fy, fz = focus_xyz_mm
    gauss = np.exp(
        -((X - fx) ** 2) / (2 * sigx**2)
        - ((Y - fy) ** 2) / (2 * sigy**2)
        - ((Z - fz) ** 2) / (2 * sigz**2)
    )
    return {
        "strain": peak_strain * gauss,
        "pressure": peak_pressure_Pa * gauss,
        "temperature_C": baseline_T_C + peak_dT_K * gauss,
        "axes_mm": (x, y, z),
        "focus_mm": focus_xyz_mm,
    }


# ---------------------------------------------------------------------------
# Mechanism scenarios
# ---------------------------------------------------------------------------

def scenarios() -> list[dict]:
    """Five scenario specs. Each returns a callable mapping
    (strain_3d, pressure_3d, temperature_3d) -> {firing, count} maps."""
    return [
        {
            "key": "S1_singlecomp_phase1",
            "label": "Single compartment\n(Phase 1 baseline)",
        },
        {
            "key": "S2_multicomp_phase1",
            "label": "3-compartment\n(Phase 1)",
        },
        {
            "key": "S3_multicomp_NICE",
            "label": "3-comp + NICE\n(Phase 2 alone)",
        },
        {
            "key": "S4_multicomp_M5",
            "label": "3-comp + Piezo1 inact.\n+ Ca/SK (Phase 3, M5)",
        },
        {
            "key": "S5_full",
            "label": "Full pathway\n(Phase 1+2+3+4)",
        },
    ]


def _run_singlecomp(strain, pressure, temperature_C, dt, duration):
    ch = deepcopy(PIEZO1)
    ch.g_density = 0.05  # boosted for visibility (matches technical report)
    ch2 = deepcopy(TREK1)
    ch2.g_density = 0.04
    neuron = MechanoHHNeuron.cortical_pyramidal(
        temperature=37.0, mechano_channels=[ch, ch2],
    )
    r = simulate_map(strain, neuron, dt=dt, duration=duration, tension_threshold=0.5)
    return {"firing": r["firing_map"], "count": r["spike_count_map"]}


def _build_multicomp(*, with_nice=False, with_inact_ca=False, with_trp=False):
    ch_p = deepcopy(PIEZO1)
    ch_p.g_density = 0.05
    ch_t = deepcopy(TREK1)
    ch_t.g_density = 0.04
    kwargs = dict(
        mechano_channels=[ch_p, ch_t],
        temperature=37.0,
    )
    if with_nice:
        kwargs["nice_model"] = NICEModel()
        kwargs["nice_compartments"] = (0, 1, 2)
    if with_inact_ca:
        kwargs["piezo_inactivation"] = True
        kwargs["calcium_dynamics"] = CalciumDynamics()
        sk = KCaChannel(g_density=0.3)
        kwargs["kca_channels"] = [sk]
        kwargs["kca_compartments"] = (1,)
    if with_trp:
        trpv1 = deepcopy(TRPV1)
        trpv4 = deepcopy(TRPV4)
        kwargs["trp_channels"] = [trpv1, trpv4]
        kwargs["trp_compartments"] = (1,)
    return MulticompartmentMechanoHHNeuron.cortical_pyramidal_3comp(**kwargs)


def _run_multicomp_phase1(strain, pressure, temperature_C, dt, duration):
    n = _build_multicomp()
    r = simulate_map_3comp(
        strain, n, dt=dt, duration=duration, tension_threshold=0.5,
    )
    return {"firing": r["firing_map"], "count": r["spike_count_map"]}


def _run_multicomp_NICE(strain, pressure, temperature_C, dt, duration):
    n = _build_multicomp(with_nice=True)
    # Constant-pressure batch path: pass per-voxel pressure (1-D over masked voxels).
    # simulate_map_3comp handles only constant-pressure NICE in its current API,
    # so we feed the field unchanged and rely on the batch routine's per-voxel
    # constant envelope branch (added in Phase 2).
    # Note: simulate_map_3comp internally calls _simulate_batch with scalar
    # pressure by default; to vary per voxel we extend below.
    return _run_per_voxel_pressure(n, strain, pressure, dt, duration)


def _run_multicomp_M5(strain, pressure, temperature_C, dt, duration):
    n = _build_multicomp(with_inact_ca=True)
    r = simulate_map_3comp(
        strain, n, dt=dt, duration=duration, tension_threshold=0.5,
    )
    return {"firing": r["firing_map"], "count": r["spike_count_map"]}


def _run_full(strain, pressure, temperature_C, dt, duration):
    n = _build_multicomp(with_nice=True, with_inact_ca=True, with_trp=True)
    return _run_per_voxel_pressure(n, strain, pressure, dt, duration)


def _run_per_voxel_pressure(neuron, strain, pressure, dt, duration):
    """Run multi-compartment batch with per-voxel constant pressure envelope."""
    from ultraneuron.channels import strain_to_membrane_tension

    nx, ny, nz = strain.shape
    tension = strain_to_membrane_tension(strain, K_A=0.25, coupling_factor=1.0)
    mask = tension >= 0.5
    firing = np.zeros((nx, ny, nz), dtype=bool)
    counts = np.zeros((nx, ny, nz), dtype=np.int32)
    if not np.any(mask):
        return {"firing": firing, "count": counts}
    tensions_flat = tension[mask].astype(np.float64)
    pressure_flat = pressure[mask].astype(np.float64)
    n_steps = int(duration / dt)
    I_arr = np.array([0.0, 0.0, 0.0])  # no extrinsic current; mechano is the drive
    sc = neuron._simulate_batch(
        tensions_flat,
        dt=dt,
        n_steps=n_steps,
        I_stim_per_compartment=I_arr,
        primary_compartment=1,
        pressure_envelope=pressure_flat,  # (N,) per-voxel constant
    )
    counts[mask] = sc
    firing[mask] = sc > 0
    return {"firing": firing, "count": counts}


SCENARIO_RUNNERS = {
    "S1_singlecomp_phase1": _run_singlecomp,
    "S2_multicomp_phase1": _run_multicomp_phase1,
    "S3_multicomp_NICE": _run_multicomp_NICE,
    "S4_multicomp_M5": _run_multicomp_M5,
    "S5_full": _run_full,
}


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(field, results, out_pdf, out_png):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nx, ny, nz = field["strain"].shape
    z_focal_idx = int(round(field["focus_mm"][2] / (field["axes_mm"][2][1] - field["axes_mm"][2][0])))
    scen_specs = scenarios()

    fig, axes = plt.subplots(2, 5, figsize=(13, 5.5),
                             gridspec_kw={"height_ratios": [3, 1]})
    plt.rcParams["font.size"] = 9

    # Top row: firing-map slices
    vmax = max((r["count"][:, :, z_focal_idx].max() for r in results.values()), default=1)
    vmax = max(int(vmax), 1)

    x_mm, y_mm, z_mm = field["axes_mm"]
    extent = (x_mm[0], x_mm[-1], y_mm[0], y_mm[-1])

    for j, spec in enumerate(scen_specs):
        ax = axes[0, j]
        cmap = ax.imshow(
            results[spec["key"]]["count"][:, :, z_focal_idx].T,
            origin="lower", extent=extent, vmin=0, vmax=vmax, cmap="magma",
        )
        ax.set_title(spec["label"], fontsize=8)
        ax.set_xlabel("lateral x (mm)" if j == 0 else "")
        if j == 0:
            ax.set_ylabel("lateral y (mm)")
    cbar = fig.colorbar(cmap, ax=axes[0, :].ravel().tolist(),
                        fraction=0.025, pad=0.02, label="spike count / 30 ms")

    # Bottom: per-scenario firing-voxel count bar
    ax_bar = fig.add_subplot(2, 1, 2)
    keys = [s["key"] for s in scen_specs]
    labels = [s["label"].replace("\n", " ") for s in scen_specs]
    n_firing = [int(results[k]["firing"].sum()) for k in keys]
    bars = ax_bar.bar(range(len(keys)), n_firing, color="#4C72B0")
    for bar, val in zip(bars, n_firing):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(n_firing) * 0.01,
                    f"{val}", ha="center", fontsize=8)
    ax_bar.set_xticks(range(len(keys)))
    ax_bar.set_xticklabels(labels, fontsize=7)
    ax_bar.set_ylabel("Firing voxels (count)")
    ax_bar.set_title("Total firing voxels under each pathway combination", fontsize=9)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    # Hide the original axes[1, :] row that we replaced with the bar chart
    for ax in axes[1, :]:
        ax.set_visible(False)

    fig.suptitle(
        f"TIPS-typical focal field (peak ε_dev = {field['strain'].max():.3g}, "
        f"peak P = {field['pressure'].max() / 1e3:.0f} kPa)",
        fontsize=10,
    )
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== JNE Phase 4: TIPS mechanism-comparison run ===", flush=True)
    field = synthetic_tips_field()
    print(f"Grid: {field['strain'].shape}", flush=True)
    print(
        f"  peak ε_dev    = {field['strain'].max():.3g}",
        flush=True,
    )
    print(
        f"  peak pressure = {field['pressure'].max() / 1e3:.1f} kPa",
        flush=True,
    )
    print(
        f"  peak ΔT       = {field['temperature_C'].max() - 37.0:.2f} K",
        flush=True,
    )

    dt = 0.001  # ms — multi-compartment requirement
    duration = 30.0  # ms

    results = {}
    for spec in scenarios():
        t0 = time.perf_counter()
        print(f"  running {spec['key']} ({spec['label'].replace(chr(10), ' ')}) ...", flush=True)
        runner = SCENARIO_RUNNERS[spec["key"]]
        # single-compartment can use a coarser dt
        local_dt = 0.01 if spec["key"] == "S1_singlecomp_phase1" else dt
        out = runner(
            field["strain"], field["pressure"], field["temperature_C"],
            dt=local_dt, duration=duration,
        )
        results[spec["key"]] = out
        elapsed = time.perf_counter() - t0
        n_fire = int(out["firing"].sum())
        n_spike = int(out["count"].sum())
        print(
            f"    done in {elapsed:.1f}s — {n_fire} firing voxels, "
            f"{n_spike} total spikes",
            flush=True,
        )

    OUT_DATA.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_DATA,
        strain=field["strain"],
        pressure=field["pressure"],
        temperature_C=field["temperature_C"],
        **{f"{k}__firing": v["firing"] for k, v in results.items()},
        **{f"{k}__count": v["count"] for k, v in results.items()},
    )
    print(f"Saved data: {OUT_DATA}", flush=True)

    OUT_FIG_PDF.parent.mkdir(parents=True, exist_ok=True)
    make_figure(field, results, OUT_FIG_PDF, OUT_FIG_PNG)
    print(f"Saved figure: {OUT_FIG_PDF}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
