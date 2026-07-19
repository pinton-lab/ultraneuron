"""Generic FDTD-consuming neural runner.

Loads a Fullwave + shear-FDTD output directory (``displacement.npy``,
``c_ds.npy``, ``grid_params.npz``), evaluates the multi-pathway
mechano-Hodgkin-Huxley neuron at every above-threshold brain voxel,
and saves a single-burst result npz next to the upstream artefacts:

    <SIM_DATA_DIR>/neural_results.npz

Mirrors the canonical Yaakub run from the manuscript: 3-compartment
cortical pyramidal neuron, Piezo1 + TREK-1 channels, K_A = 0.25 N/m,
coupling factor alpha = 200, 30 ms simulation window at constant
peak tension.  Spike times are recovered for the firing subset and
packed CSR-style for downstream rasters / PSTH / latency rendering.

Usage
-----
    python -m scripts.run_neural_sim \\
        --config validation_report/configs/halle_caudate.py

or with a dotted module path.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
import time
from copy import deepcopy
from pathlib import Path
from types import ModuleType

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))

from ultraneuron import (  # noqa: E402
    PIEZO1,
    TREK1,
    MulticompartmentMechanoHHNeuron,
    simulate_map_3comp_with_spike_times,
)
from ultraneuron.strain import compute_strain_invariants  # noqa: E402


# Manuscript-calibrated defaults (Sec. 3.1, Yaakub theta-burst run).
_DEFAULTS = {
    "K_A": 0.25,                     # N/m, lipid bilayer area expansion modulus
    "coupling_factor": 1000.0,       # tissue->membrane lumped multi-scale gain
                                     # (calibrated to give manuscript's peak tension
                                     # ~5 mN/m at the manuscript's 0.5 MPa transcranial
                                     # focal PNP. Literature alpha=200 from Salahshoor
                                     # was calibrated for hippocampal/different cell
                                     # types; cortical-pyramidal in Halle skull gives
                                     # ~5x lower strain than the manuscript reports,
                                     # so alpha~1000 is needed to recover T_max = 5 mN/m)
    "tension_threshold": 0.5,        # mN/m, skip voxels below this (Piezo1 ~2.7)
    "duration_ms": 30.0,             # 30 ms post-pulse window
    "dt_ms": 0.001,                  # 3-comp stability margin
    "piezo1_g_density": 0.10,        # mS/cm^2 (manuscript value for Yaakub)
    "trek1_g_density": 0.04,         # mS/cm^2
}


def _load_config(arg: str) -> ModuleType:
    p = Path(arg)
    if p.exists() and p.suffix == ".py":
        spec = importlib.util.spec_from_file_location("nrm_config", p)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load config from {p}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    return importlib.import_module(arg)


def _resolve_sim_dir(cfg: ModuleType) -> Path:
    """SIM_DATA_DIR is repo-relative; OUT_DIR is the validation_report
    folder underneath it.  Pick whichever is present, return absolute."""
    if hasattr(cfg, "SIM_DATA_DIR"):
        d = Path(cfg.SIM_DATA_DIR)
    else:
        d = Path(cfg.OUT_DIR).parent
    return d if d.is_absolute() else (REPO_ROOT / d).resolve()


def _strain_field_from_displacement(disp: np.ndarray, dx: float, dy: float,
                                    dz: float) -> np.ndarray:
    """Deviatoric (von Mises) strain from a 3-component displacement
    field; matches the elastic-outputs builder convention so the
    validation report and the neural runner see the same scalar."""
    ux, uy, uz = disp[..., 0], disp[..., 1], disp[..., 2]
    duxdx, duxdy, duxdz = np.gradient(ux, dx, dy, dz, edge_order=1)
    duydx, duydy, duydz = np.gradient(uy, dx, dy, dz, edge_order=1)
    duzdx, duzdy, duzdz = np.gradient(uz, dx, dy, dz, edge_order=1)
    e_xx = duxdx; e_yy = duydy; e_zz = duzdz
    e_xy = 0.5 * (duxdy + duydx)
    e_xz = 0.5 * (duxdz + duzdx)
    e_yz = 0.5 * (duydz + duzdy)
    e_vol = e_xx + e_yy + e_zz
    em = e_vol / 3.0
    dxx = e_xx - em; dyy = e_yy - em; dzz = e_zz - em
    e_dev_norm2 = (dxx * dxx + dyy * dyy + dzz * dzz
                   + 2.0 * (e_xy * e_xy + e_xz * e_xz + e_yz * e_yz))
    return np.sqrt((2.0 / 3.0) * e_dev_norm2).astype(np.float32)


def _build_neuron(cfg: ModuleType, scenario: str = "S2"
                   ) -> MulticompartmentMechanoHHNeuron:
    """Build the multi-compartment neuron for the requested mechanism scenario.

    ``scenario`` ∈ {S1, S2, S3, S4, S5}:
      * S1 — single-compartment baseline (only valid for the
        mechanism-comparison sweep in run_halle_mechanism_sweep.py;
        rejected here since this script is for the multi-comp pipeline).
      * S2 — 3-comp + Piezo1 (+TREK-1 if cfg.NEURAL_USE_TREK1) — DEFAULT
        canonical Yaakub run; sufficient at sub-MPa carrier and 30 ms
        tone-bursts at body temperature.
      * S3 — S2 + intramembrane cavitation (NICE).  Use above ~1 MPa
        focal carrier where ⟨ΔC_m⟩/C_{m,0} > 5%.
      * S4 — S2 + Piezo1 inactivation + Ca²⁺ pool + SK counter-current.
        Use for protocols > ~100 ms ON window or to model
        Legon-style high-pressure inhibition.
      * S5 — full pathway (S2+S3+S4 + TRPV1 + TRPV4).  Use above ~3 K
        focal heating where TRPV1 starts to engage.
    """
    from ultraneuron.calcium import CalciumDynamics, KCaChannel
    from ultraneuron.nice import NICEModel
    from ultraneuron.thermosensors import TRPV1, TRPV4

    if scenario == "S1":
        msg = ("scenario=S1 is single-compartment; use the canonical "
               "single-compartment pipeline (e.g. run_halle_mechanism_sweep.py) "
               "rather than this multi-compartment runner.")
        raise SystemExit(msg)

    g_p1 = float(getattr(cfg, "NEURAL_PIEZO1_G_DENSITY",
                         _DEFAULTS["piezo1_g_density"]))
    g_trek1 = float(getattr(cfg, "NEURAL_TREK1_G_DENSITY",
                             _DEFAULTS["trek1_g_density"]))
    use_trek1 = bool(getattr(cfg, "NEURAL_USE_TREK1", True))

    p1 = deepcopy(PIEZO1); p1.g_density = g_p1
    channels = [p1]
    if use_trek1:
        t1 = deepcopy(TREK1); t1.g_density = g_trek1
        channels.append(t1)

    kwargs: dict = dict(mechano_channels=channels, temperature=37.0)

    # Phase-2: NICE intramembrane cavitation
    if scenario in ("S3", "S5"):
        kwargs["nice_model"] = NICEModel()
        kwargs["nice_compartments"] = (0, 1, 2)

    # Phase-3: Piezo1 inactivation + Ca pool + SK counter-current
    if scenario in ("S4", "S5"):
        # Piezo1 inactivation gate (Coste 2010, τ_inact ≈ 12 ms)
        p1.tau_inact = 12.0
        kwargs["piezo_inactivation"] = True
        kwargs["calcium_dynamics"] = CalciumDynamics()
        sk = KCaChannel(
            name="SK", g_density=0.3, Ca_half=0.5, n_hill=4.0, E_K=-100.0,
        )
        kwargs["kca_channels"] = [sk]
        kwargs["kca_compartments"] = (1,)

    # Phase-4: TRPV1 / TRPV4 thermosensors
    if scenario == "S5":
        kwargs["trp_channels"] = [deepcopy(TRPV1), deepcopy(TRPV4)]
        kwargs["trp_compartments"] = (1,)

    return MulticompartmentMechanoHHNeuron.cortical_pyramidal_3comp(**kwargs)


def _git_sha() -> str:
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True,
                    help="Dotted module path or filesystem path to a config .py")
    ap.add_argument("--sim-dir", default=None,
                    help="Override sim data directory (else cfg.SIM_DATA_DIR)")
    ap.add_argument("--duration-ms", type=float, default=None,
                    help="Neural integration / post-pulse window (ms). Defaults "
                         "to the config's arf_push_window_s (the protocol ON "
                         "window), falling back to tone_burst_duration_s, then "
                         f"{_DEFAULTS['duration_ms']:.0f} ms.")
    ap.add_argument("--dt-ms", type=float, default=_DEFAULTS["dt_ms"])
    ap.add_argument("--integrator", choices=["rk4", "rl"], default="rk4",
                    help="Per-voxel ODE integrator: 'rk4' (default, canonical "
                         "reference) or 'rl' (Rush-Larsen / exponential-Euler; "
                         "~3x faster, L-stable, S2-only, ~1%% below the RK4 "
                         "reference — for fast sweeps/previews).")
    ap.add_argument("--K-A", type=float, default=_DEFAULTS["K_A"])
    ap.add_argument("--alpha", type=float, dest="coupling_factor",
                    default=_DEFAULTS["coupling_factor"])
    ap.add_argument("--threshold-mNm", type=float,
                    default=_DEFAULTS["tension_threshold"])
    ap.add_argument("--scenario", choices=["S2", "S3", "S4", "S5"],
                    default=None,
                    help="Mechanism scenario: S2=3-comp+Piezo1[+TREK1] "
                         "(default canonical), S3=+NICE, S4=+inact+Ca+SK, "
                         "S5=full pathway including TRPV1/TRPV4. "
                         "Falls back to cfg.NEURAL_SCENARIO or 'S2'.")
    args = ap.parse_args()

    cfg = _load_config(args.config)
    sim_dir = (Path(args.sim_dir).resolve() if args.sim_dir
               else _resolve_sim_dir(cfg))
    print(f"[run_neural_sim] sim_dir = {sim_dir}", flush=True)

    # Neural integration window tracks the protocol ON window so the
    # single-push response is integrated over the acoustic burst rather
    # than a fixed constant (Yaakub: 20 ms; Legon configs: 30 ms).
    if args.duration_ms is None:
        seq = getattr(cfg, "NEUROMOD_SEQUENCE", {}) or {}
        win_s = seq.get("arf_push_window_s",
                        seq.get("tone_burst_duration_s",
                                _DEFAULTS["duration_ms"] * 1e-3))
        args.duration_ms = float(win_s) * 1e3
    print(f"  neural integration window = {args.duration_ms:.1f} ms", flush=True)

    # ---- Load FDTD products ------------------------------------------
    disp = np.load(sim_dir / "displacement.npy").astype(np.float32)
    c_map = np.load(sim_dir / "c_ds.npy").astype(np.float32)
    grid = np.load(sim_dir / "grid_params.npz")
    # Displacement / c_ds live on the downsampled validation grid; the
    # native FDTD pitch is multiplied by the mod_? decimation factors.
    dx = float(grid["dx"]) * int(grid["mod_x"])
    dy = float(grid["dy"]) * int(grid["mod_y"])
    dz = float(grid["dz"]) * int(grid["mod_z"])
    print(f"  displacement {disp.shape} max={np.abs(disp).max():.3e} m", flush=True)
    print(f"  voxel pitch dx={dx*1e3:.3f} mm dy={dy*1e3:.3f} mm dz={dz*1e3:.3f} mm",
          flush=True)

    # ---- Strain → tension --------------------------------------------
    strain = _strain_field_from_displacement(disp, dx, dy, dz)
    print(f"  deviatoric strain max={strain.max():.3e}", flush=True)

    # Restrict to brain (skull voxels saturate strain artefactually)
    skull = c_map > 1600.0
    strain_brain = strain.copy()
    strain_brain[skull] = 0.0

    scenario = args.scenario or str(getattr(cfg, "NEURAL_SCENARIO", "S2"))
    neuron = _build_neuron(cfg, scenario=scenario)
    print(f"  scenario: {scenario}", flush=True)
    print(f"  neuron: 3-comp cortical, channels="
          f"{[c.name for c in neuron.mechano_channels]}, "
          f"piezo1 g={neuron.mechano_channels[0].g_density:.3f}", flush=True)

    # ---- Run --------------------------------------------------------
    t0 = time.perf_counter()
    res = simulate_map_3comp_with_spike_times(
        strain_brain, neuron,
        K_A=args.K_A,
        coupling_factor=args.coupling_factor,
        dt=args.dt_ms,
        duration=args.duration_ms,
        tension_threshold=args.threshold_mNm,
        integrator=args.integrator,
    )
    elapsed = time.perf_counter() - t0
    n_fire = int(res["firing_map"].sum())
    n_spike = int(res["spike_count_map"].sum())
    print(f"  done in {elapsed:.1f} s — {n_fire} firing voxels, "
          f"{n_spike} total spikes (single push)", flush=True)
    print(f"  peak per-voxel tension {res['tension_3d'].max():.2f} mN/m", flush=True)

    # ---- Sequence schedule (for replay across theta groups) -----------
    seq = getattr(cfg, "NEUROMOD_SEQUENCE", {}) or {}
    group_period_s = float(seq.get(
        "group_period_s", seq.get("pulse_period_s", 0.2)))
    session_s = float(seq.get("session_duration_s", 80.0))
    n_groups = max(int(round(session_s / group_period_s)), 1)
    prf_hz = float(seq.get("pulse_repetition_freq_hz", 5.0))
    theta_hz = float(seq.get("theta_freq_hz", prf_hz))

    # ---- Save -------------------------------------------------------
    out_path = sim_dir / "neural_results.npz"
    np.savez_compressed(
        out_path,
        # 3D maps
        T_peak_mNm=res["tension_3d"].astype(np.float32),
        firing_map=res["firing_map"],
        spike_count_map=res["spike_count_map"],
        latency_map=res["latency_map"].astype(np.float32),
        rate_map=(res["spike_count_map"].astype(np.float32) /
                  (args.duration_ms * 1e-3)).astype(np.float32),
        # CSR spike times (single burst)
        spike_voxel_idx=res["spike_voxel_idx"].astype(np.int32),
        spike_offsets=res["spike_offsets"].astype(np.int64),
        spike_times_ms=res["spike_times_ms"].astype(np.float32),
        # Driver / provenance
        dt_ms=np.float64(args.dt_ms),
        duration_ms=np.float64(args.duration_ms),
        T_pulse_ms=np.float64(args.duration_ms),  # constant-T window
        threshold_mNm=np.float64(args.threshold_mNm),
        K_A=np.float64(args.K_A),
        coupling_factor=np.float64(args.coupling_factor),
        piezo1_g_density=np.float64(neuron.mechano_channels[0].g_density),
        n_mechano_channels=np.int32(len(neuron.mechano_channels)),
        primary_compartment=np.int32(1),
        group_period_ms=np.float64(group_period_s * 1e3),
        n_groups_session=np.int32(n_groups),
        prf_hz=np.float64(prf_hz),
        theta_hz=np.float64(theta_hz),
        integrator=str(args.integrator),
        sim_git_sha=str(_git_sha()),
        sim_name=str(getattr(cfg, "SIM_NAME", "unknown")),
    )
    print(f"  wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
