"""Mechanosensitive synaptic / voltage-gated transmission (M6) on a canonical run.

Scores the sixth candidate mechanism --- mechanosensitive synaptic and
voltage-gated transmission --- on the same per-voxel peak-tension field that
drives the canonical neural pipeline, and produces the data behind the M6
mechanism figure / validation-report section.

The mechanism bundles three literature-grounded effects, each with a
closed-form per-case scalar gain (the inference-pipeline contract):
  - presynaptic release modulation  Pr(T)/Pr(0)      (Tyler 2018)
  - post-synaptic NMDA stretch       f_stretch(T)     (Maneshi 2017)
  - voltage-gated Na+/Ca2+ shift      nav_current_gain(T) (Kubanek 2016)

Inputs (from ``<SIM_DATA_DIR>/neural_results.npz``):
  - ``T_peak_mNm`` : per-voxel peak membrane tension [mN/m].

Outputs (``<SIM_DATA_DIR>/synaptic_results.npz``):
  - ``gain_map``       : per-voxel combined pathway gain (>= 1).
  - focal-voxel relay timecourse (V, glutamate, Pr) from
    ``simulate_synaptic_relay``.
  - tension sweep of the three scalar gains, the combined gain, and the
    relay spike count --- the dose-response behind the figure.
  - scalar metadata (focal tension, recruited-voxel count, focal spikes /
    depolarisation, model parameters).

Usage:
    python scripts/run_synaptic_sim.py --config validation_report/configs/halle_dacc.py
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))

from ultraneuron import (  # noqa: E402
    K_MECHANO_NAV,
    MechanoSynapse,
    nav_current_gain,
    nmdar_stretch_factor,
    simulate_synaptic_relay,
    synaptic_gain,
)
from ultraneuron.neuron import (  # noqa: E402
    NAV_ACT_SLOPE,
    NAV_ACT_VHALF,
    NAV_GAIN_EVAL_V,
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


def _nav_gain_map(T: np.ndarray) -> np.ndarray:
    """Vectorised ``nav_current_gain`` over a peak-tension field.

    Mirrors :func:`ultraneuron.nav_current_gain` exactly using the
    same module constants, evaluated on the whole array at once.
    """
    Tc = np.maximum(np.asarray(T, dtype=np.float64), 0.0)
    shift = K_MECHANO_NAV * Tc

    def _m_inf(v_half):
        x = (NAV_GAIN_EVAL_V - v_half) / NAV_ACT_SLOPE
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500.0, 500.0)))

    return _m_inf(NAV_ACT_VHALF - shift) / _m_inf(NAV_ACT_VHALF)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--sim-dir", default=None)
    ap.add_argument("--gain-threshold", type=float, default=1.5,
                    help="Combined pathway gain above which a voxel is 'recruited'.")
    args = ap.parse_args()

    cfg = _load_config(args.config)
    sim_dir = (Path(args.sim_dir).resolve() if args.sim_dir
               else _resolve_sim_dir(cfg))
    print(f"[run_synaptic_sim] sim_dir = {sim_dir}", flush=True)

    nr = np.load(sim_dir / "neural_results.npz", allow_pickle=True)
    T_peak = np.asarray(nr["T_peak_mNm"], dtype=np.float64)

    seq = getattr(cfg, "NEUROMOD_SEQUENCE", {}) or {}
    duty = float(seq.get("duty_cycle", 0.15))
    session_s = float(seq.get("session_duration_s", 80.0))

    synapse = MechanoSynapse()
    print(f"  Pr {synapse.Pr_baseline}->{synapse.Pr_max}, "
          f"k_mechano_nav = {K_MECHANO_NAV} mV/(mN/m), "
          f"alpha_NMDA = {synapse.nmda.alpha_NMDA}", flush=True)

    # --- per-voxel combined pathway-gain map (closed-form, vectorised) ---
    syn_map = np.asarray(synapse.release_probability(T_peak), dtype=np.float64) / synapse.Pr_baseline
    nmda_map = np.asarray(synapse.nmda.stretch_factor(T_peak), dtype=np.float64)
    nav_map = _nav_gain_map(T_peak)
    gain_map = syn_map * nmda_map * nav_map

    focal_idx = np.unravel_index(int(np.argmax(T_peak)), T_peak.shape)
    focal_T = float(T_peak[focal_idx])
    gain_thr = float(args.gain_threshold)
    n_recruited = int((gain_map >= gain_thr).sum())
    print(f"  focal tension = {focal_T:.2f} mN/m at {focal_idx}; "
          f"peak combined gain = {gain_map.max():.3f}; "
          f"recruited (>{gain_thr}) = {n_recruited:,} voxels", flush=True)

    # --- focal relay timecourse (presynaptic Pr -> glutamate -> AMPA/NMDA) ---
    fwd = simulate_synaptic_relay(
        peak_tension_mNm=focal_T, dt=0.025, duration=120.0, us_onset=10.0,
    )
    print(f"  focal forward sim (120 ms): depol = {fwd['peak_depolarisation']:.1f} mV, "
          f"spikes = {fwd['spike_count']}", flush=True)

    # --- dose-response: scalar gains + relay spike count vs tension ---
    sweep_T = np.linspace(0.0, max(focal_T * 1.1, 1.0), 16)
    sweep_syn = np.array([synaptic_gain(float(t)) for t in sweep_T])
    sweep_nmda = np.array([nmdar_stretch_factor(float(t)) for t in sweep_T])
    sweep_nav = np.array([nav_current_gain(float(t)) for t in sweep_T])
    sweep_combined = sweep_syn * sweep_nmda * sweep_nav
    sweep_spikes = np.array([
        simulate_synaptic_relay(peak_tension_mNm=float(t), dt=0.025,
                                duration=120.0, us_onset=10.0)["spike_count"]
        for t in sweep_T
    ])

    out = sim_dir / "synaptic_results.npz"
    np.savez_compressed(
        out,
        gain_map=gain_map.astype(np.float32),
        T_peak_mNm=T_peak.astype(np.float32),
        focal_idx=np.array(focal_idx, dtype=np.int32),
        focal_tension_mNm=np.float64(focal_T),
        focal_t_ms=fwd["t"].astype(np.float32),
        focal_V_mV=fwd["V"].astype(np.float32),
        focal_glu_uM=fwd["glutamate"].astype(np.float32),
        focal_Pr=fwd["Pr"].astype(np.float32),
        focal_n_spikes=np.int64(fwd["spike_count"]),
        focal_depolarisation_mV=np.float64(fwd["peak_depolarisation"]),
        sweep_T_mNm=sweep_T,
        sweep_synaptic_gain=sweep_syn,
        sweep_nmda_stretch=sweep_nmda,
        sweep_nav_gain=sweep_nav,
        sweep_combined_gain=sweep_combined,
        sweep_spikes=sweep_spikes.astype(np.int64),
        peak_combined_gain=np.float64(gain_map.max()),
        n_recruited=np.int64(n_recruited),
        gain_threshold=np.float64(gain_thr),
        duty_cycle=np.float64(duty),
        session_s=np.float64(session_s),
        Pr_baseline=np.float64(synapse.Pr_baseline),
        Pr_max=np.float64(synapse.Pr_max),
        k_mechano_nav=np.float64(K_MECHANO_NAV),
        alpha_NMDA=np.float64(synapse.nmda.alpha_NMDA),
    )
    print(f"  wrote {out}", flush=True)


if __name__ == "__main__":
    main()
