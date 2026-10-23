"""Time course of the focus -> ring prediction (JNE revision, R2-M8).

``scripts/render_focus_ring.py`` counts spikes per voxel over one 20 ms ON
window started from rest (the canonical initial condition since b141c0a).
Piezo1 inactivation needs ~10 ms to build from rest, so the high-tension
core (T >= T_1/2,i = 4.5 mN/m) still fires early in the burst under S4.
This driver resolves the spike trains in time to ask whether the core is
quiescent within a well-defined window:

1. ``single``: the 20 ms burst from rest, S2 and S4 (spike times per voxel).
2. ``bursts``: theta-burst periods 1..``--n-bursts`` (20 ms ON / 180 ms OFF,
   tension relaxing exponentially with the visco-elastic tau in the OFF
   time, ``tension_drive``), full state carried over, burst 1 from rest.
3. ``long``: one ``--long-ms`` ON window at constant peak tension from rest
   (the 50/100 ms windows are its prefixes; chaining/prefixes are exact for
   a constant drive). Outside the Yaakub protocol (20 ms ON).

Voxels: every eligible voxel with T >= ``--t-min`` mN/m (below the 20 ms
firing threshold 2.21 mN/m, so near-threshold voxels that fire only in a
long ON window are captured). Canonical field, conductances and coupling as
``render_focus_ring.py`` (K_A 0.25 N/m, alpha from the config, dt 1 us, RK4).

Output: ``<sim_dir>/focus_ring/focus_ring_timecourse_raw.npz`` (spike trains,
CSR) consumed by ``scripts/render_focus_ring_timecourse.py``.

Usage: CUDA_VISIBLE_DEVICES=0 python scripts/run_focus_ring_timecourse.py \\
           --config validation_report/configs/halle_dacc.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT)); sys.path.insert(0, str(HERE))
from ultraneuron.channels import strain_to_membrane_tension  # noqa: E402
from ultraneuron.cuda_backend import resolve_backend  # noqa: E402
from ultraneuron.multicompartment import simulate_voxels_3comp, tension_drive  # noqa: E402
from run_neural_sim import _build_neuron, _git_sha, _load_config, _resolve_sim_dir, _strain_field_from_displacement  # noqa: E402


def _pack(offsets_list, times_list):
    """Concatenate per-run CSR (offsets, times) along a leading run axis."""
    return np.stack(offsets_list).astype(np.int64), np.concatenate(times_list).astype(np.float64) if times_list else np.zeros(0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="validation_report/configs/halle_dacc.py")
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--t-min", type=float, default=1.8)
    ap.add_argument("--n-bursts", type=int, default=8)
    ap.add_argument("--long-ms", type=float, default=200.0)
    args = ap.parse_args()
    t_all = time.perf_counter()
    cfg = _load_config(args.config); sim_dir = _resolve_sim_dir(cfg)
    out_dir = sim_dir / "focus_ring"; out_dir.mkdir(exist_ok=True)
    backend = resolve_backend(args.backend)
    seq = getattr(cfg, "NEUROMOD_SEQUENCE", {}) or {}
    t_on = float(seq.get("arf_push_window_s", 0.02)) * 1e3
    period = float(seq.get("pulse_period_s", 0.2)) * 1e3
    tau_relax = float(json.loads((sim_dir / "validation_report" / "manifest.json").read_text())
                      ["displacement_outputs"]["relax_tau_ms"])
    dt = 0.001
    disp = np.load(sim_dir / "displacement.npy").astype(np.float32); c = np.load(sim_dir / "c_ds.npy")
    grid = np.load(sim_dir / "grid_params.npz"); dx = float(grid["dx"]) * int(grid["mod_x"])
    eps = _strain_field_from_displacement(disp, dx, dx, dx); eps[c > 1600] = 0.0
    K_A, alpha = float(cfg.NEURAL_K_A), float(cfg.NEURAL_COUPLING_FACTOR)
    T3 = strain_to_membrane_tension(eps, K_A, alpha)
    sel = T3 >= args.t_min
    vox_idx = np.argwhere(sel).astype(np.int32)
    T = T3[sel].astype(np.float64)
    print(f"[timecourse] backend={backend} N={T.size} (T>={args.t_min}) ON={t_on} period={period} "
          f"tau_relax={tau_relax:.2f} K_A={K_A} alpha={alpha}", flush=True)
    npz = {"voxel_idx": vox_idx, "T_mNm": T, "dt_ms": dt, "t_on_ms": t_on, "period_ms": period,
           "tau_relax_ms": tau_relax, "shape": np.array(T3.shape)}
    timing = {}
    drv = tension_drive(dt, t_on, period - t_on, 1, off_mode="exp", tau_relax_ms=tau_relax)
    for sc in ("S2", "S4"):
        n = _build_neuron(cfg, sc)
        # 1. single 20 ms burst from rest
        t0 = time.perf_counter()
        r = simulate_voxels_3comp(T, n, dt=dt, duration=t_on, backend=backend, return_state=False)
        npz[f"single_{sc}_off"], npz[f"single_{sc}_t"] = r["spike_offsets"], r["spike_times_ms"]
        timing[f"single_{sc}"] = time.perf_counter() - t0
        print(f"  {sc} single: {int(r['spike_counts'].sum())} spikes, {int((r['spike_counts'] > 0).sum())} firing "
              f"({timing[f'single_{sc}']:.0f} s)", flush=True)
        # 2. chained theta-burst periods
        t0 = time.perf_counter(); st = None; offs, ts = [], []
        for b in range(args.n_bursts):
            r = simulate_voxels_3comp(T, n, dt=dt, duration=period, backend=backend, drive=drv,
                                      initial_state=st, return_state=True, max_spikes_per_voxel=int(period) + 8)
            st = r["final_state"]; offs.append(r["spike_offsets"]); ts.append(r["spike_times_ms"])
            on = r["spike_times_ms"] < t_on
            print(f"    {sc} burst {b + 1}: {int(on.sum())} ON spikes, {int((~on).sum())} OFF spikes", flush=True)
        npz[f"bursts_{sc}_off"], npz[f"bursts_{sc}_t"] = _pack(offs, ts)
        timing[f"bursts_{sc}"] = time.perf_counter() - t0
        # 3. long ON window from rest
        t0 = time.perf_counter()
        r = simulate_voxels_3comp(T, n, dt=dt, duration=args.long_ms, backend=backend, return_state=False,
                                  max_spikes_per_voxel=int(args.long_ms) + 8)
        npz[f"long_{sc}_off"], npz[f"long_{sc}_t"] = r["spike_offsets"], r["spike_times_ms"]
        timing[f"long_{sc}"] = time.perf_counter() - t0
        print(f"  {sc} long {args.long_ms} ms: {int(r['spike_counts'].sum())} spikes ({timing[f'long_{sc}']:.0f} s)", flush=True)
    timing["total"] = time.perf_counter() - t_all
    npz["long_ms"] = args.long_ms; npz["n_bursts"] = args.n_bursts
    npz["meta"] = json.dumps({"git_sha": _git_sha(), "backend": backend, "K_A": K_A, "alpha": alpha,
                              "t_min_mNm": args.t_min, "timing_s": timing, "config": args.config})
    np.savez_compressed(out_dir / "focus_ring_timecourse_raw.npz", **npz)
    print(json.dumps(timing, indent=1))


if __name__ == "__main__":
    main()
