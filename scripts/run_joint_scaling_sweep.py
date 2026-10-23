"""Mechanism-comparison sweep with joint scaling of pressure, strain and temperature
(JNE revision, R2-4; replaces the strain-only sweep of run_halle_mechanism_sweep.py).

A single source-amplitude factor s multiplies the canonical Halle/dACC field:

    p(x)     -> s   * p_can(x)        carrier pressure amplitude (NICE input)
    I, F_ARF -> s^2                    (linear acoustics, I ~ p^2)
    eps(x)   -> s^2 * eps_can(x)       (linear viscoelastic shear response)
    dT(x)    -> s^2 * dT_can(x)        (linear bioheat)

so the strain axis of the earlier sweep is recovered but the intramembrane-
cavitation module now sees the pressure that goes with each strain level and
the thermosensors see the heating that goes with it.  Levels are chosen in s
so that the canonical exposure (s = 1, focal PPP 0.50 MPa) is one of them and
the top level is 1.0 MPa, the stated validity limit of the SONIC reduction.

Mode ``strain`` reproduces the submitted design (strain rescaled, pressure held
at the canonical field, temperature at 37 C) on the corrected field, for the
supplementary strain-axis sensitivity study.

Temperature enters the neuron as a per-run scalar (see manuscript S2.2.3); we
use the focal end-of-session rise scaled by s^2 for every voxel, which is an
upper bound on the field.

Outputs under data/<sim>/joint_sweep/:  results_<mode>.json, firing_<mode>_<S>_s<...>.npz
Usage:  python scripts/run_joint_scaling_sweep.py --config validation_report/configs/halle_dacc.py
                                                  [--mode joint|strain] [--resume]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT)); sys.path.insert(0, str(HERE))

from ultraneuron import (  # noqa: E402
    PIEZO1, TREK1, MechanoHHNeuron, MulticompartmentMechanoHHNeuron,
    simulate_map, simulate_map_3comp,
)
from ultraneuron.calcium import CalciumDynamics, KCaChannel  # noqa: E402
from ultraneuron.cuda_backend import resolve_backend  # noqa: E402
from ultraneuron.nice import NICEModel  # noqa: E402
from ultraneuron.thermosensors import TRPV1, TRPV4  # noqa: E402
from run_neural_sim import _load_config, _resolve_sim_dir, _strain_field_from_displacement  # noqa: E402

S_LEVELS = [0.30, 0.45, 0.65, 1.00, 1.40, 2.00]          # focal PPP = 0.15 ... 1.00 MPa
G_PIEZO1, G_TREK1, K_A, ALPHA, DT_MS, T_THR = 0.05, 0.04, 0.25, 1000.0, 0.001, 0.5
SCENARIOS = ["S1", "S2", "S3", "S4", "S5"]
T_BODY = 37.0


def build(scenario: str, temperature: float):
    ch_p = deepcopy(PIEZO1); ch_p.g_density = G_PIEZO1
    ch_t = deepcopy(TREK1); ch_t.g_density = G_TREK1
    if scenario == "S1":
        return MechanoHHNeuron.cortical_pyramidal(temperature=temperature, mechano_channels=[ch_p, ch_t])
    kw = dict(mechano_channels=[ch_p, ch_t], temperature=temperature)
    if scenario in ("S4", "S5"):
        ch_p.tau_inact = 12.0
        kw.update(piezo_inactivation=True, calcium_dynamics=CalciumDynamics(),
                  kca_channels=[KCaChannel(name="SK", g_density=0.3, Ca_half=0.5, n_hill=4.0, E_K=-100.0)],
                  kca_compartments=(1,))
    if scenario in ("S3", "S5"):
        kw.update(nice_model=NICEModel(), nice_compartments=(0, 1, 2))
    if scenario == "S5":
        kw.update(trp_channels=[deepcopy(TRPV1), deepcopy(TRPV4)], trp_compartments=(1,))
    return MulticompartmentMechanoHHNeuron.cortical_pyramidal_3comp(**kw)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", choices=["joint", "strain"], default="joint")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--backend", default="auto")
    args = ap.parse_args()
    cfg = _load_config(args.config); sim_dir = _resolve_sim_dir(cfg)
    out_dir = sim_dir / "joint_sweep"; out_dir.mkdir(exist_ok=True)
    seq = getattr(cfg, "NEUROMOD_SEQUENCE", {}) or {}
    duration_ms = float(seq.get("arf_push_window_s", seq.get("tone_burst_duration_s", 0.02))) * 1e3
    backend = resolve_backend(args.backend)

    disp = np.load(sim_dir / "displacement.npy").astype(np.float32)
    c_map = np.load(sim_dir / "c_ds.npy"); grid = np.load(sim_dir / "grid_params.npz")
    dx = float(grid["dx"]) * int(grid["mod_x"])
    eps = _strain_field_from_displacement(disp, dx, dx, dx); eps[c_map > 1600] = 0.0
    p_can = np.load(sim_dir / "pressure_peak_map.npy").astype(np.float64); p_can[c_map > 1600] = 0.0
    dT_focus = float(json.loads((sim_dir / "bioheat" / "thermal_metrics.json").read_text())["dT_focus_K"])
    eps_peak = float(eps.max()); p_peak = float(p_can.max()); vox_mm3 = (dx * 1e3) ** 3
    print(f"[joint_sweep] mode={args.mode} backend={backend} eps_peak={eps_peak:.3e} p_peak={p_peak/1e6:.3f} MPa "
          f"dT_focus={dT_focus*1e3:.0f} mK window={duration_ms:.0f} ms", flush=True)

    out_json = out_dir / f"results_{args.mode}.json"
    res = json.loads(out_json.read_text()) if (args.resume and out_json.exists()) else {}
    res["_meta"] = dict(mode=args.mode, s_levels=S_LEVELS, scenarios=SCENARIOS, g_piezo1=G_PIEZO1, g_trek1=G_TREK1,
                        K_A=K_A, alpha=ALPHA, dt_ms=DT_MS, duration_ms=duration_ms, eps_peak_canonical=eps_peak,
                        p_peak_canonical_pa=p_peak, dT_focus_canonical_K=dT_focus, voxel_mm3=vox_mm3, backend=backend)
    for s in S_LEVELS:
        eps_s = (eps * s * s).astype(np.float32)
        if args.mode == "joint":
            P_s = p_can * s; T_s = T_BODY + dT_focus * s * s
        else:
            P_s = p_can; T_s = T_BODY
        for sc in SCENARIOS:
            key = f"{sc}_s{s:g}"
            if key in res:
                continue
            n = build(sc, T_s); t0 = time.perf_counter()
            if sc == "S1":
                r = simulate_map(eps_s, n, dt=DT_MS, duration=duration_ms, tension_threshold=T_THR,
                                 K_A=K_A, coupling_factor=ALPHA, backend=backend)
            else:
                r = simulate_map_3comp(eps_s, n, dt=DT_MS, duration=duration_ms, tension_threshold=T_THR,
                                       K_A=K_A, coupling_factor=ALPHA, backend=backend,
                                       pressure_envelope=P_s if sc in ("S3", "S5") else 0.0)
            fire = r["firing_map"].astype(bool); cnt = r["spike_count_map"]
            res[key] = dict(scenario=sc, s=s, focal_ppp_mpa=p_peak * s / 1e6, eps_peak=eps_peak * s * s,
                            focal_tension_mNm=K_A * ALPHA * eps_peak * s * s * 1e3, temperature_C=T_s,
                            n_firing_voxels=int(fire.sum()), total_spikes=int(cnt.sum()),
                            firing_volume_mm3=float(fire.sum() * vox_mm3), elapsed_s=time.perf_counter() - t0)
            np.savez_compressed(out_dir / f"firing_{args.mode}_{key}.npz", spike_count_map=cnt.astype(np.int16))
            out_json.write_text(json.dumps(res, indent=1))
            print(f"  s={s:<4g} PPP={res[key]['focal_ppp_mpa']:.2f} MPa eps={res[key]['eps_peak']:.2e} T={T_s:.2f}C "
                  f"{sc}: {res[key]['n_firing_voxels']:5d} vox {res[key]['total_spikes']:6d} spk ({res[key]['elapsed_s']:.0f}s)", flush=True)
    print("[joint_sweep] done", flush=True)


if __name__ == "__main__":
    main()
