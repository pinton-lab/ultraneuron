"""Mechanism-comparison sweep on the real Halle-skull / F3->left-caudate
shear-FDTD strain field.

Replaces the synthetic Gaussian focal field used in the original
tips_sim mechanism-comparison sweep. The real anatomical pattern
(skull aberration, focal asymmetry, dendrite-heavy distribution) is
preserved; the exposure-axis sweep is implemented by uniformly scaling
the field so the focal voxel hits a chosen target peak deviatoric
strain. This isolates the mechanism axis at the threshold region
without re-running the upstream Fullwave + shear-FDTD pipeline.

Output
------
data/halle_mechanism_sweep_results.npz
figures/fig8_halle_mechanism_comparison.{pdf,png}
"""
from __future__ import annotations

import os
import sys
import time
from copy import deepcopy
from pathlib import Path

import matplotlib

# Compute backend for the per-voxel HH integration.  Override at the
# shell with AMN_BACKEND=numba (CPU) or AMN_BACKEND=cuda; default 'auto' = GPU when visible.
_BACKEND = "auto"  # resolved by ultraneuron.cuda_backend.resolve_backend (AMN_BACKEND env override)

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))

from ultraneuron import (  # noqa: E402
    PIEZO1,
    TREK1,
    MulticompartmentMechanoHHNeuron,
    simulate_map,
    simulate_map_3comp,
    simulate_map_3comp_with_spike_times,
)
from ultraneuron.calcium import CalciumDynamics, KCaChannel  # noqa: E402
from ultraneuron.neuron import MechanoHHNeuron  # noqa: E402
from ultraneuron.nice import NICEModel  # noqa: E402
from ultraneuron.thermosensors import TRPV1, TRPV4  # noqa: E402

# Inline copy of validation_report/builders/elastic_outputs._strain_tensor
# (relative-import unfriendly when invoked as a script).
def _strain_tensor(u, dx, dy, dz):
    ux, uy, uz = u[..., 0], u[..., 1], u[..., 2]
    duxdx, duxdy, duxdz = np.gradient(ux, dx, dy, dz, edge_order=1)
    duydx, duydy, duydz = np.gradient(uy, dx, dy, dz, edge_order=1)
    duzdx, duzdy, duzdz = np.gradient(uz, dx, dy, dz, edge_order=1)
    e_xx, e_yy, e_zz = duxdx, duydy, duzdz
    e_xy = 0.5 * (duxdy + duydx)
    e_xz = 0.5 * (duxdz + duzdx)
    e_yz = 0.5 * (duydz + duzdy)
    e_vol = e_xx + e_yy + e_zz
    em = e_vol / 3.0
    dxx, dyy, dzz = e_xx - em, e_yy - em, e_zz - em
    e_dev_norm2 = (dxx*dxx + dyy*dyy + dzz*dzz
                   + 2.0*(e_xy*e_xy + e_xz*e_xz + e_yz*e_yz))
    return {"e_eq": np.sqrt((2.0/3.0) * e_dev_norm2)}


_SIM_NAME = os.environ.get("HALLE_SIM_NAME", "halle_caudate")
SIM_DIR = REPO_ROOT / "data" / _SIM_NAME
OUT_NPZ = REPO_ROOT / "data" / f"{_SIM_NAME.replace('halle_', 'halle_')}_mechanism_sweep_results.npz" if _SIM_NAME != "halle_caudate" else REPO_ROOT / "data" / "halle_mechanism_sweep_results.npz"
OUT_FIG = REPO_ROOT / "figures" / (f"fig8_{_SIM_NAME}_mechanism_comparison" if _SIM_NAME != "halle_caudate" else "fig8_halle_mechanism_comparison")


# Target peak ε_eq levels (dimensionless), spanning sub- to supra-threshold.
# Target peak ε_eq levels chosen so that at α=1000 the focal-voxel
# tension T = K_A·α·ε spans 0.5–25 mN/m: from sub-threshold (0.5 ≈ 0.18×T_½)
# through the Boltzmann knee (4–9 mN/m, the canonical working point) up
# to saturation (25 mN/m). The current Halle / caudate canonical run sits
# at ε ≈ 2×10⁻⁵ (4.92 mN/m, 1.8×T_½), the third sweep level.
TARGET_EPS = np.array([2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4])
DT_MS = 0.001


def _config_push_window_ms(sim_name: str, default_ms: float = 30.0) -> float:
    """Neural integration window (ms) from the sim's NEUROMOD_SEQUENCE
    (arf_push_window_s, falling back to tone_burst_duration_s), so the
    mechanism sweep integrates over the same protocol ON window as the
    canonical run (Yaakub: 20 ms; Legon configs: 30 ms)."""
    cfg_path = REPO_ROOT / "validation_report" / "configs" / f"{sim_name}.py"
    if not cfg_path.exists():
        return default_ms
    import importlib.util
    spec = importlib.util.spec_from_file_location(f"_cfg_{sim_name}", cfg_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    seq = getattr(mod, "NEUROMOD_SEQUENCE", {}) or {}
    win_s = seq.get("arf_push_window_s",
                    seq.get("tone_burst_duration_s", default_ms * 1e-3))
    return float(win_s) * 1e3


DURATION_MS = _config_push_window_ms(_SIM_NAME)
NICE_PRESSURE_PA = 500e3  # constant pressure for NICE compartments (S3, S5)


# =====================================================================
# Field loader
# =====================================================================


def _load_real_strain() -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Return (strain_3d, brain_mask, grid) at peak displacement."""
    grid_npz = np.load(SIM_DIR / "grid_params.npz", allow_pickle=True)
    dx = float(grid_npz["dx"]) * int(grid_npz["mod_x"])
    dy = float(grid_npz["dy"]) * int(grid_npz["mod_y"])
    dz = float(grid_npz["dz"]) * int(grid_npz["mod_z"])
    c_map = np.load(SIM_DIR / "c_ds.npy")
    skull = c_map > 1600.0
    disp_4d = np.load(SIM_DIR / "displacement_4d.npy", mmap_mode="r")
    absmax_per_frame = np.abs(disp_4d).reshape(disp_4d.shape[0], -1).max(axis=1)
    peak_t_idx = int(np.argmax(absmax_per_frame))
    u_peak = np.asarray(disp_4d[peak_t_idx], dtype=np.float64)
    strain = _strain_tensor(u_peak, dx, dy, dz)
    eeq = strain["e_eq"].astype(np.float32)
    eeq[skull] = 0.0  # skull saturation artefacts → drop
    grid = dict(dx=dx, dy=dy, dz=dz, peak_t_idx=peak_t_idx)
    return eeq, ~skull, grid


# =====================================================================
# Scenario builders (mirror run_tips_sim.py but force g_Piezo1 = 0.05
# so the focal voxel sits near threshold across the sweep)
# =====================================================================


_G_PIEZO1 = 0.05
_G_TREK1 = 0.04


def _build_multicomp(*, with_nice=False, with_inact_ca=False, with_trp=False):
    ch_p = deepcopy(PIEZO1); ch_p.g_density = _G_PIEZO1
    ch_t = deepcopy(TREK1);  ch_t.g_density = _G_TREK1
    kwargs = dict(mechano_channels=[ch_p, ch_t], temperature=37.0)
    if with_inact_ca:
        kwargs["piezo_inactivation"] = True
        kwargs["calcium_dynamics"] = CalciumDynamics()
        kwargs["kca_channels"] = [KCaChannel(g_density=0.3)]
        kwargs["kca_compartments"] = (1,)
    if with_nice:
        kwargs["nice_model"] = NICEModel()
        kwargs["nice_compartments"] = (0, 1, 2)
    if with_trp:
        kwargs["trp_channels"] = [deepcopy(TRPV1), deepcopy(TRPV4)]
        kwargs["trp_compartments"] = (1,)
    return MulticompartmentMechanoHHNeuron.cortical_pyramidal_3comp(**kwargs)


_COUPLING_FACTOR = 1000.0  # α (cortical-pyramidal-calibrated; matches Table 1)
_K_A = 0.25


def _S1(strain):  # 1-comp Piezo1+TREK1
    ch_p = deepcopy(PIEZO1); ch_p.g_density = _G_PIEZO1
    ch_t = deepcopy(TREK1);  ch_t.g_density = _G_TREK1
    n = MechanoHHNeuron.cortical_pyramidal(temperature=37.0,
                                            mechano_channels=[ch_p, ch_t])
    r = simulate_map(strain, n, dt=DT_MS, duration=DURATION_MS,
                     tension_threshold=0.5,
                     K_A=_K_A, coupling_factor=_COUPLING_FACTOR,
                     backend=_BACKEND)
    return r["firing_map"], r["spike_count_map"]


def _S2(strain):  # 3-comp Piezo1+TREK1
    n = _build_multicomp()
    r = simulate_map_3comp(strain, n, dt=DT_MS, duration=DURATION_MS,
                           tension_threshold=0.5,
                           K_A=_K_A, coupling_factor=_COUPLING_FACTOR,
                           backend=_BACKEND)
    return r["firing_map"], r["spike_count_map"]


def _S3(strain):  # 3-comp + NICE
    n = _build_multicomp(with_nice=True)
    r = simulate_map_3comp(strain, n, dt=DT_MS, duration=DURATION_MS,
                           tension_threshold=0.5,
                           K_A=_K_A, coupling_factor=_COUPLING_FACTOR,
                           backend=_BACKEND)
    return r["firing_map"], r["spike_count_map"]


def _S4(strain):  # 3-comp + Piezo1 inact + Ca + SK
    n = _build_multicomp(with_inact_ca=True)
    r = simulate_map_3comp(strain, n, dt=DT_MS, duration=DURATION_MS,
                           tension_threshold=0.5,
                           K_A=_K_A, coupling_factor=_COUPLING_FACTOR,
                           backend=_BACKEND)
    return r["firing_map"], r["spike_count_map"]


def _S5(strain):  # full pathway
    n = _build_multicomp(with_nice=True, with_inact_ca=True, with_trp=True)
    r = simulate_map_3comp(strain, n, dt=DT_MS, duration=DURATION_MS,
                           tension_threshold=0.5,
                           K_A=_K_A, coupling_factor=_COUPLING_FACTOR,
                           backend=_BACKEND)
    return r["firing_map"], r["spike_count_map"]


SCENARIOS = [
    ("S1", "Single compartment\n(Phase 1)", _S1, "tab:blue"),
    ("S2", "3-compartment\n(Phase 1)", _S2, "tab:orange"),
    ("S3", "3-comp + NICE\n(Phase 2)", _S3, "tab:green"),
    ("S4", "3-comp + inact + Ca/SK\n(Phase 3, M5)", _S4, "tab:red"),
    ("S5", "Full pathway\n(1+2+3+4)", _S5, "tab:purple"),
]


# =====================================================================
# Sweep + render
# =====================================================================


def main() -> None:
    print("[halle-sweep] loading real Halle-caudate strain field...", flush=True)
    eeq_real, brain_mask, grid = _load_real_strain()
    real_peak = float(eeq_real.max())
    print(f"  field shape {eeq_real.shape}, real peak ε_eq = {real_peak:.3e}",
          flush=True)
    print(f"  voxel pitch dx={grid['dx']*1e3:.3f} mm "
          f"dy={grid['dy']*1e3:.3f} mm dz={grid['dz']*1e3:.3f} mm",
          flush=True)
    print(f"  brain voxels: {brain_mask.sum()}", flush=True)

    # Per-scenario, per-target_eps total spike count + firing map at middle target.
    n_eps = len(TARGET_EPS)
    n_sc = len(SCENARIOS)
    spike_totals = np.zeros((n_sc, n_eps), dtype=np.int64)
    firing_maps_mid = {}
    mid_idx = n_eps // 2  # middle scaling for the firing-map snapshot panel

    for j, eps in enumerate(TARGET_EPS):
        scale = float(eps) / real_peak
        strain_scaled = (eeq_real * scale).astype(np.float32)
        for i, (key, label, runner, _color) in enumerate(SCENARIOS):
            t0 = time.perf_counter()
            firing, count = runner(strain_scaled)
            elapsed = time.perf_counter() - t0
            n_fire = int(firing.sum())
            n_spike = int(count.sum())
            spike_totals[i, j] = n_spike
            print(f"  ε={eps:.4f}  {key:3s}: {n_fire:5d} firing voxels, "
                  f"{n_spike:5d} spikes ({elapsed:.1f}s)", flush=True)
            if j == mid_idx:
                firing_maps_mid[key] = count

    np.savez_compressed(
        OUT_NPZ,
        target_eps=TARGET_EPS,
        spike_totals=spike_totals,
        scenarios=np.array([k for k, _, _, _ in SCENARIOS]),
        labels=np.array([l for _, l, _, _ in SCENARIOS]),
        real_peak_eeq=real_peak,
        focal_eps_at_mid=TARGET_EPS[mid_idx],
        **{f"firing_count_{k}": firing_maps_mid[k] for k, _, _, _ in SCENARIOS},
    )
    print(f"[halle-sweep] wrote {OUT_NPZ}", flush=True)

    # ----- Render fig8 -----
    fig = plt.figure(figsize=(15, 5.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.6], wspace=0.20)

    # Panel (a): total spike count vs peak ε
    ax = fig.add_subplot(gs[0, 0])
    for i, (key, label, _, color) in enumerate(SCENARIOS):
        ax.plot(TARGET_EPS, spike_totals[i], marker="o",
                lw=1.6, color=color, label=key)
    ax.set_xlabel(r"Peak deviatoric strain $\varepsilon_{\mathrm{eq}}^{\max}$")
    ax.set_ylabel(f"Total spike count ({DURATION_MS:.0f} ms window)")
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=10)
    ax.grid(alpha=0.3, which="both")
    ax.set_title("(a) Mechanism axis on real Halle-caudate field",
                  fontsize=11)
    ax.legend(loc="upper left", fontsize=8, frameon=False)

    # Panel (b): firing-map slices at middle ε on focal y-z plane
    # Take the slice through the brain-restricted argmax of the S2 map
    ref = firing_maps_mid["S2"]
    if ref.max() > 0:
        peak_idx = np.unravel_index(int(np.argmax(ref)), ref.shape)
    else:
        peak_idx = tuple(s // 2 for s in ref.shape)
    ix = peak_idx[0]
    dy = grid["dy"] * 1e3; dz = grid["dz"] * 1e3
    ny, nz = ref.shape[1], ref.shape[2]
    ext = (0, nz * dz, 0, ny * dy)

    inner_gs = gs[0, 1].subgridspec(1, n_sc, wspace=0.05)
    vmax = max(int(firing_maps_mid[k].max()) for k, _, _, _ in SCENARIOS) or 1
    for i, (key, label, _, _color) in enumerate(SCENARIOS):
        ax = fig.add_subplot(inner_gs[0, i])
        sl = firing_maps_mid[key][ix, :, :].T
        im = ax.imshow(sl, cmap="hot", vmin=0, vmax=vmax,
                       extent=ext, origin="lower", aspect="equal")
        ax.set_title(key, fontsize=10)
        if i == 0:
            ax.set_ylabel("y (mm)")
        else:
            ax.set_yticks([])
        ax.set_xlabel("z (mm)")
    cbar = fig.colorbar(im, ax=fig.axes[1:],
                        location="right", shrink=0.85, pad=0.02)
    cbar.set_label("Spikes / voxel")
    fig.suptitle(
        rf"Mechanism-comparison sweep on the real Halle-caudate "
        rf"$\varepsilon_{{\mathrm{{eq}}}}$ field "
        rf"(peak focal strain rescaled across {n_eps} levels; panel b "
        rf"shown at $\varepsilon_{{\mathrm{{eq}}}}^{{\max}} = "
        rf"{TARGET_EPS[mid_idx]:.1e}$ on focal y-z plane)",
        fontsize=11,
    )
    fig.savefig(f"{OUT_FIG}.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{OUT_FIG}.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[halle-sweep] wrote {OUT_FIG}.{{png,pdf}}", flush=True)


if __name__ == "__main__":
    main()
