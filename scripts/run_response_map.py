"""Response map of the canonical Halle/dACC firing volume over (K_A*alpha, g_Piezo1)
(JNE revision, R2-3), plus the single run at the Table S1 library defaults.

Grid (S2 neuron, same strain field, window and dt as the canonical run):
    K_A*alpha  in {50, 100, 150, 250, 375, 500} N/m per unit strain
               (K_A = 0.25 N/m fixed; alpha = 200 ... 2000)
    g_Piezo1   in {0.005, 0.01, 0.02, 0.05, 0.10, 0.20} mS/cm^2
    g_TREK1    in {0.04 (canonical), 0.02 (library default)}   -> two panels

Special points:
    canonical        (250, 0.10) at g_TREK1 = 0.04   -> must reproduce 5468 / 20416
    library default  (250, 0.005) at g_TREK1 = 0.02  -> the run Referee 2 asked for

Outputs under <sim_dir>/response_map/:
    response_map.json    per-point headline numbers (resumable)
    firing_<key>.npz     firing / spike-count maps for the special points only

Usage:
    python scripts/run_response_map.py --config validation_report/configs/halle_dacc.py [--resume]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HERE))

from ultraneuron import simulate_map_3comp_with_spike_times  # noqa: E402
from run_neural_sim import (  # noqa: E402
    _DEFAULTS, _build_neuron, _load_config, _resolve_sim_dir,
    _strain_field_from_displacement,
)

KA = 0.25
ALPHAS = [200.0, 400.0, 600.0, 1000.0, 1500.0, 2000.0]        # K_A*alpha = 50 ... 500 mN/m
G_PIEZO = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20]
G_TREK = [0.04, 0.02]
SPECIAL = {("0.04", 1000.0, 0.10): "canonical", ("0.02", 1000.0, 0.005): "library_default"}


def key_of(g_trek, alpha, g_p):
    return f"trek{g_trek:g}_alpha{alpha:g}_gp{g_p:g}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--sim-dir", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dt-ms", type=float, default=_DEFAULTS["dt_ms"])
    ap.add_argument("--threshold-mNm", type=float, default=_DEFAULTS["tension_threshold"])
    args = ap.parse_args()

    cfg = _load_config(args.config)
    sim_dir = Path(args.sim_dir).resolve() if args.sim_dir else _resolve_sim_dir(cfg)
    out_dir = sim_dir / "response_map"; out_dir.mkdir(exist_ok=True)
    seq = getattr(cfg, "NEUROMOD_SEQUENCE", {}) or {}
    duration_ms = float(seq.get("arf_push_window_s", seq.get("tone_burst_duration_s", 0.02))) * 1e3

    disp = np.load(sim_dir / "displacement.npy").astype(np.float32)
    c_map = np.load(sim_dir / "c_ds.npy").astype(np.float32)
    grid = np.load(sim_dir / "grid_params.npz")
    dx = float(grid["dx"]) * int(grid["mod_x"]); dy = float(grid["dy"]) * int(grid["mod_y"]); dz = float(grid["dz"]) * int(grid["mod_z"])
    strain = _strain_field_from_displacement(disp, dx, dy, dz)
    strain[c_map > 1600.0] = 0.0
    vox_mm3 = (dx * 1e3) ** 3
    print(f"[response_map] sim_dir={sim_dir} window={duration_ms:.1f} ms peak eps={strain.max():.3e}", flush=True)

    out_json = out_dir / "response_map.json"
    results = json.loads(out_json.read_text()) if (args.resume and out_json.exists()) else {}
    results.setdefault("_meta", {"K_A": KA, "alphas": ALPHAS, "g_piezo": G_PIEZO, "g_trek": G_TREK,
                                 "duration_ms": duration_ms, "dt_ms": args.dt_ms, "threshold_mNm": args.threshold_mNm,
                                 "voxel_mm3": vox_mm3, "scenario": "S2", "peak_eps": float(strain.max())})

    # order: special points first, then the rest
    points = [(gt, a, gp) for gt in G_TREK for a in ALPHAS for gp in G_PIEZO]
    points.sort(key=lambda p: 0 if (f"{p[0]:g}", p[1], p[2]) in SPECIAL else 1)
    t_start = time.perf_counter()
    for n, (g_trek, alpha, g_p) in enumerate(points):
        key = key_of(g_trek, alpha, g_p)
        if key in results:
            continue
        ns = SimpleNamespace(NEURAL_PIEZO1_G_DENSITY=g_p, NEURAL_TREK1_G_DENSITY=g_trek, NEURAL_USE_TREK1=True)
        neuron = _build_neuron(ns, scenario="S2")
        t0 = time.perf_counter()
        res = simulate_map_3comp_with_spike_times(
            strain, neuron, K_A=KA, coupling_factor=alpha, dt=args.dt_ms, duration=duration_ms,
            tension_threshold=args.threshold_mNm, integrator="rk4")
        fire = res["firing_map"].astype(bool)
        rate = res["spike_count_map"].astype(np.float64) / (duration_ms * 1e-3)
        rmax = float(rate.max())
        h = {
            "g_trek1": g_trek, "alpha": alpha, "K_A_alpha_mNm": KA * alpha * 1e3, "g_piezo1": g_p,
            "n_firing_voxels": int(fire.sum()), "total_spikes": int(res["spike_count_map"].sum()),
            "peak_rate_hz": rmax, "firing_volume_mm3": float(fire.sum() * vox_mm3),
            "iso25_volume_mm3": float((rate >= 0.25 * rmax).sum() * vox_mm3) if rmax > 0 else 0.0,
            "iso50_volume_mm3": float((rate >= 0.5 * rmax).sum() * vox_mm3) if rmax > 0 else 0.0,
            "peak_tension_mNm": float(res["tension_3d"].max()),
            "n_voxels_above_threshold": int((res["tension_3d"] >= args.threshold_mNm).sum()),
            "elapsed_s": time.perf_counter() - t0,
        }
        sp = SPECIAL.get((f"{g_trek:g}", alpha, g_p))
        if sp:
            h["special"] = sp
            np.savez_compressed(out_dir / f"firing_{sp}.npz", firing_map=res["firing_map"],
                                spike_count_map=res["spike_count_map"], tension_3d=res["tension_3d"].astype(np.float32),
                                latency_map=res["latency_map"].astype(np.float32))
        results[key] = h
        out_json.write_text(json.dumps(results, indent=1))
        print(f"{n+1:2d}/{len(points)} {key:32s} fire={h['n_firing_voxels']:5d} spikes={h['total_spikes']:6d} "
              f"iso25={h['iso25_volume_mm3']:7.0f} Tpk={h['peak_tension_mNm']:.2f} ({h['elapsed_s']:.0f}s)"
              + (f"  <-- {sp}" if sp else ""), flush=True)
    print(f"[response_map] done in {(time.perf_counter()-t_start)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
