"""Piezo1-pool passive Ca readout for the canonical halle_dacc S2 run (rest start).

Re-runs the canonical S2 neural stage (3-comp cortical pyramidal, Piezo1 + TREK-1,
K_A = 0.25 N/m, alpha = 1000, 20 ms ON window, dt = 1 us, RK4, mechano gates from
true rest) with the single-pool CalciumDynamics model attached as a passive readout
(``simulate_map_3comp_with_spike_times(calcium_readout=True, ca_trace_dt_ms=...)``).
S2 has no KCa channel, so the pool has no path back to V: spike counts and spike
times must be bit-identical to data/halle_dacc/neural_results.npz (checked here).

Writes data/halle_dacc/calcium/piezo_ca_canonical.npz:
  ca_peak_map      float32 (3, nx, ny, nz)  per-compartment peak [Ca] over the
                   window [uM] (0 dendrite, 1 soma, 2 AIS); NaN below threshold
  ca_peak_firing   float32 (n_fire, 3)      aligned with spike_voxel_idx
  ca_trace_firing  float32 (n_fire, 3, n_t) decimated firing-voxel traces [uM]
  ca_trace_t_ms    float64 (n_t,)
  spike_voxel_idx, spike_offsets, spike_times_ms (copy of the CSR, for alignment)
  Ca_rest_uM, tau_decay_ms, flux_factor, ...

    python scripts/compute_piezo_ca_readout.py [--ca-trace-dt-ms 0.1]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(HERE))

from ultraneuron import simulate_map_3comp_with_spike_times  # noqa: E402
from ultraneuron.calcium import CalciumDynamics  # noqa: E402
from ultraneuron.cuda_backend import resolve_backend  # noqa: E402
from run_neural_sim import (  # noqa: E402
    _build_neuron, _load_config, _resolve_sim_dir, _strain_field_from_displacement,
)

COMP = ("dendrite", "soma", "AIS")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="validation_report/configs/halle_dacc.py")
    ap.add_argument("--ca-trace-dt-ms", type=float, default=0.1)
    ap.add_argument("--backend", default="auto")
    args = ap.parse_args()

    cfg = _load_config(str(REPO / args.config) if not Path(args.config).is_absolute() else args.config)
    sim_dir = _resolve_sim_dir(cfg)
    nr = np.load(sim_dir / "neural_results.npz")
    dt, dur = float(nr["dt_ms"]), float(nr["duration_ms"])
    K_A, alpha, thr = float(nr["K_A"]), float(nr["coupling_factor"]), float(nr["threshold_mNm"])

    disp = np.load(sim_dir / "displacement.npy").astype(np.float32)
    c_map = np.load(sim_dir / "c_ds.npy").astype(np.float32)
    grid = np.load(sim_dir / "grid_params.npz")
    dx = float(grid["dx"]) * int(grid["mod_x"])
    dy = float(grid["dy"]) * int(grid["mod_y"])
    dz = float(grid["dz"]) * int(grid["mod_z"])
    strain = _strain_field_from_displacement(disp, dx, dy, dz)
    strain[c_map > 1600.0] = 0.0

    neuron = _build_neuron(cfg, scenario="S2")
    backend = resolve_backend(args.backend)
    pool = CalciumDynamics()
    print(f"[piezo_ca] backend={backend} dt={dt} ms window={dur} ms "
          f"pool: rest {pool.Ca_rest} uM tau {pool.tau_decay} ms flux {pool.flux_factor}", flush=True)
    t0 = time.perf_counter()
    res = simulate_map_3comp_with_spike_times(
        strain, neuron, K_A=K_A, coupling_factor=alpha, dt=dt, duration=dur,
        tension_threshold=thr, backend=backend, integrator="rk4",
        calcium_readout=pool, ca_trace_dt_ms=args.ca_trace_dt_ms,
    )
    wall = time.perf_counter() - t0

    # Passive readout must leave the spike outputs untouched
    same = {k: bool(np.array_equal(res[k], nr[k])) for k in
            ("spike_count_map", "spike_voxel_idx", "spike_offsets", "spike_times_ms")}
    print(f"  done in {wall:.1f} s; identical to neural_results.npz: {same}", flush=True)
    if not all(same.values()):
        raise SystemExit("passive Ca readout changed the spike outputs")
    assert bool(res["ca_passive"])

    out_dir = sim_dir / "calcium"; out_dir.mkdir(exist_ok=True)
    pk = res["ca_peak_firing"]
    stats = {c: {"median": float(np.median(pk[:, i])), "min": float(pk[:, i].min()),
                 "max": float(pk[:, i].max())} for i, c in enumerate(COMP)}
    sha = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                                  text=True).strip()
    np.savez_compressed(
        out_dir / "piezo_ca_canonical.npz",
        ca_peak_map=res["ca_peak_map"].astype(np.float32),
        ca_peak_firing=pk.astype(np.float32),
        ca_trace_firing=res["ca_trace_firing"].astype(np.float32),
        ca_trace_t_ms=np.asarray(res["ca_trace_t_ms"], dtype=np.float64),
        compartments=np.array(COMP),
        spike_voxel_idx=res["spike_voxel_idx"], spike_offsets=res["spike_offsets"],
        spike_times_ms=res["spike_times_ms"],
        Ca_rest_uM=pool.Ca_rest, tau_decay_ms=pool.tau_decay, flux_factor=pool.flux_factor,
        dt_ms=dt, duration_ms=dur, ca_trace_dt_ms=float(args.ca_trace_dt_ms),
        mechano_init="rest", backend=str(backend), git_sha=sha,
    )
    (out_dir / "piezo_ca_canonical.json").write_text(json.dumps(
        {"n_fire": int(len(pk)), "peak_firing_uM": stats,
         "pool": {"Ca_rest_uM": pool.Ca_rest, "tau_decay_ms": pool.tau_decay,
                  "flux_factor": pool.flux_factor},
         "ca_trace_dt_ms": args.ca_trace_dt_ms, "wall_s": wall, "backend": str(backend),
         "spike_outputs_identical_to_neural_results": same, "git_sha": sha}, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
