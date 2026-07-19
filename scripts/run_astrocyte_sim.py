"""Astrocytic TRPA1 → Ca²⁺ → glutamate → NMDA relay on a canonical run.

Scores the non-Piezo (astrocytic) mechanism of Oh et al. 2019 on the same
per-voxel peak-tension field that drives the canonical neural pipeline,
and produces the data behind the astrocyte mechanism figure / validation-
report section.

Inputs (from ``<SIM_DATA_DIR>/neural_results.npz``):
  - ``T_peak_mNm`` : per-voxel peak membrane tension [mN/m].

Outputs (``<SIM_DATA_DIR>/astrocyte_results.npz``):
  - ``excitation_map``   : per-voxel astrocytic-NMDA drive (0–1) at the
                           protocol duty / session on-time.
  - ``focal_t_ms``, ``focal_Ca``, ``focal_glu`` : focal-voxel relay
                           timecourse (the slow Ca²⁺ / glutamate rise).
  - ``ontime_s``, ``ontime_drive`` : excitation_drive vs cumulative
                           sonication on-time at the focal tension — the
                           duty-cumulative signature distinguishing this
                           pathway from the per-pulse channel mechanisms.
  - scalar metadata: focal tension, duty, session, eligible / "firing"
    voxel counts, focal depolarisation & spikes from a short forward sim.

Usage:
    python scripts/run_astrocyte_sim.py --config validation_report/configs/halle_dacc.py
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
import sys

sys.path.insert(0, str(REPO_ROOT))

from ultraneuron import (  # noqa: E402
    AstrocyteRelay,
    NMDAReceptor,
    simulate_astrocyte_relay,
)


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
    d = Path(cfg.SIM_DATA_DIR) if hasattr(cfg, "SIM_DATA_DIR") else Path(cfg.OUT_DIR).parent
    return d if d.is_absolute() else (REPO_ROOT / d).resolve()


def _excitation_map(
    relay: AstrocyteRelay,
    nmda: NMDAReceptor,
    T_peak: np.ndarray,
    duty: float,
    on_time_s: float,
    v_ref_mV: float = -50.0,
) -> np.ndarray:
    """Vectorised per-voxel excitation_drive over a peak-tension field.

    Mirrors ``AstrocyteRelay.excitation_drive`` exactly, but evaluated on
    the whole array at once (the scalar method is for the per-case
    inference contract; this is the spatial map for the figure).
    """
    cal = relay.calcium
    i_ca = (
        relay.trpa1.g_density
        * np.asarray(relay.trpa1.open_probability(T_peak), dtype=np.float64)
        * relay.trpa1.Ca_permeability_fraction
        * (relay.trpa1.E_rev - relay.V_astro)
    )
    j_avg = cal.flux_factor * i_ca * float(duty)
    fill = 1.0 - np.exp(-(on_time_s * 1e3) / cal.tau_decay)
    ca = cal.Ca_rest + j_avg * cal.tau_decay * fill
    glu = relay.glutamate_from_ca(ca)
    return (
        np.asarray(nmda.glutamate_binding(glu), dtype=np.float64)
        * float(nmda.mg_unblock(v_ref_mV))
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--sim-dir", default=None)
    ap.add_argument("--drive-threshold", type=float, default=0.05,
                    help="Astrocytic-NMDA drive above which a voxel is 'recruited'.")
    args = ap.parse_args()

    cfg = _load_config(args.config)
    sim_dir = (Path(args.sim_dir).resolve() if args.sim_dir
               else _resolve_sim_dir(cfg))
    print(f"[run_astrocyte_sim] sim_dir = {sim_dir}", flush=True)

    nr = np.load(sim_dir / "neural_results.npz", allow_pickle=True)
    T_peak = np.asarray(nr["T_peak_mNm"], dtype=np.float64)

    seq = getattr(cfg, "NEUROMOD_SEQUENCE", {}) or {}
    duty = float(seq.get("duty_cycle", 0.15))
    session_s = float(seq.get("session_duration_s", 80.0))

    relay = AstrocyteRelay()
    nmda = NMDAReceptor()
    print(f"  duty = {duty}, session = {session_s} s, "
          f"TRPA1 T_half = {relay.trpa1.T_half} mN/m, tau_Ca = {relay.calcium.tau_decay} ms",
          flush=True)

    # --- per-voxel astrocytic-NMDA excitation map ---
    excitation_map = _excitation_map(relay, nmda, T_peak, duty, session_s)
    focal_idx = np.unravel_index(int(np.argmax(T_peak)), T_peak.shape)
    focal_T = float(T_peak[focal_idx])
    drive_thr = float(args.drive_threshold)
    n_recruited = int((excitation_map >= drive_thr).sum())
    print(f"  focal tension = {focal_T:.2f} mN/m at {focal_idx}; "
          f"peak drive = {excitation_map.max():.3f}; "
          f"recruited (>{drive_thr}) = {n_recruited:,} voxels", flush=True)

    # --- focal relay timecourse (slow Ca²⁺ / glutamate rise) ---
    # Drive at the focal peak tension; show ~10 tau to capture the rise.
    tc = relay.relay_timecourse(focal_T, dt=1.0, duration_ms=10.0 * relay.calcium.tau_decay)

    # --- excitation_drive vs cumulative on-time (the duty-cumulative signature) ---
    ontime_s = np.geomspace(1e-2, session_s, 40)
    ontime_drive = np.array([
        relay.excitation_drive(focal_T, duty_cycle=duty, on_time_s=float(t))
        for t in ontime_s
    ])

    # --- short forward sim at the focal voxel (relay → NMDA → neuron) ---
    fwd = simulate_astrocyte_relay(
        peak_tension_mNm=focal_T, dt=0.025, duration_ms=500.0,
    )
    print(f"  focal forward sim (500 ms): depol = {fwd['depolarisation']:.1f} mV, "
          f"spikes = {fwd['n_spikes']}", flush=True)

    out = sim_dir / "astrocyte_results.npz"
    np.savez_compressed(
        out,
        excitation_map=excitation_map.astype(np.float32),
        T_peak_mNm=T_peak.astype(np.float32),
        focal_idx=np.array(focal_idx, dtype=np.int32),
        focal_tension_mNm=np.float64(focal_T),
        focal_t_ms=tc["t"],
        focal_Ca_uM=tc["Ca"],
        focal_glu_uM=tc["glutamate"],
        ontime_s=ontime_s,
        ontime_drive=ontime_drive,
        peak_drive=np.float64(excitation_map.max()),
        n_recruited=np.int64(n_recruited),
        drive_threshold=np.float64(drive_thr),
        duty_cycle=np.float64(duty),
        session_s=np.float64(session_s),
        focal_depolarisation_mV=np.float64(fwd["depolarisation"]),
        focal_n_spikes=np.int64(fwd["n_spikes"]),
        tau_Ca_ms=np.float64(relay.calcium.tau_decay),
        trpa1_T_half_mNm=np.float64(relay.trpa1.T_half),
    )
    print(f"  wrote {out}", flush=True)


if __name__ == "__main__":
    main()
