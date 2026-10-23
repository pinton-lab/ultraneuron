"""Pennes bioheat solve for the validation pipeline.

Consumes a Fullwave run-dir (the same one bowl_transcranial_500khz.py
writes) and produces the thermal artefacts the validation report needs:

    delta_T_final.npy      (nx, ny, nz)         float32, K
    delta_T_frames.npy     (nx, ny, nz, n_t)    float32, K  (captured snapshots)
    delta_T_frame_times.npy (n_t,)              float64, s
    delta_T_history_t.npy  (n_t,)               float64, s  (= frame_times)
    delta_T_history_peak.npy (n_t,)             float32, K  (max over space)
    cem43.npy              (nx, ny, nz)         float32, min
    thermal_indices.json   peak_idx, focus_idx  (3-tuples in voxel coords)
    thermal_metrics.json   scalar metrics + protocol provenance

Heat-source convention: Q = 2 alpha I, with the same silence x cw pulse
normalisation that acoustic_outputs uses to convert the saved <p^2>
window into CW-equivalent ISPPA.

Time integration (default, ``--burst-resolved``): the actual stimulation
protocol is integrated burst by burst -- Q = Q_cw * duty_in_burst during
each ON window (group_on_s) and Q = 0 during the OFF remainder of each
group period, for n_groups groups.  The explicit step is dt_on inside the
ON window and dt_off in the OFF window (both far below the stability
limit), so the within-burst temperature excursion is resolved.  The focal
and spatial-peak temperature are recorded at every step
(focus_trace_*.npy / peak_trace_*.npy); full frames are captured at the
end of the ON window every ``frame_every_groups`` groups for CEM43 and
the report figures.  ``--duty-averaged`` restores the earlier collapsed
source Q_eff = duty_session * Q_cw applied continuously.

Usage:
    python run_bioheat.py --config <path-to-validation-config.py>

The config file is the same one report_builder.py uses; this script
reads ``cfg.SIM_DATA_DIR`` (or ``cfg.OUT_DIR.parent``) for the run-dir
and ``cfg.NEUROMOD_SEQUENCE`` for the protocol parameters.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np

from fullwave.bioheat import compute_heat_source
from fullwave.bioheat.pennes_solver import compute_stable_dt, pennes_solve


THIS_DIR = Path(__file__).resolve().parent
# scripts/ -> repo root
REPO_ROOT = THIS_DIR.parent


def _load_config(arg: str) -> ModuleType:
    p = Path(arg)
    if p.exists() and p.suffix == ".py":
        spec = importlib.util.spec_from_file_location("bh_config", p)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load config from {p}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    return importlib.import_module(arg)


def _resolve_sim_dir(cfg: ModuleType) -> Path:
    if hasattr(cfg, "SIM_DATA_DIR"):
        sim_dir = Path(cfg.SIM_DATA_DIR)
    else:
        sim_dir = Path(cfg.OUT_DIR).parent
    if not sim_dir.is_absolute():
        sim_dir = (REPO_ROOT / sim_dir).resolve()
    return sim_dir


def _load_run(run_dir: Path) -> dict:
    grid = np.load(run_dir / "grid_params.npz", allow_pickle=True)
    pmap = np.load(run_dir / "propagation_map.npy", mmap_mode="r")
    n_tic = int(grid["n_tic"])
    smt = (int(grid["sampling_modulus_time"])
           if "sampling_modulus_time" in grid.files else 1)
    n_source_steps = max(n_tic // smt, 1)
    silence_correction = pmap.shape[0] / n_source_steps
    cw_correction = float(grid["cw_correction"])
    pulse_norm = silence_correction * cw_correction
    return dict(
        pmap=pmap,
        c=np.load(run_dir / "c_ds.npy"),
        rho=np.load(run_dir / "rho_ds.npy"),
        alpha=np.load(run_dir / "alpha_ds.npy"),
        f0=float(grid["f0"]),
        dx=float(grid["dx"]) * float(grid["mod_x"]),
        dy=float(grid["dy"]) * float(grid["mod_y"]),
        dz=float(grid["dz"]) * float(grid["mod_z"]),
        pulse_norm=pulse_norm,
    )


def _tissue_maps(c_map: np.ndarray) -> dict:
    """Pennes property maps. Brain background; skull (c>1600) gets reduced
    conductivity, density, specific heat, and zero perfusion."""
    k = np.full_like(c_map, 0.51)
    c = np.full_like(c_map, 3630.0)
    rho = np.full_like(c_map, 1040.0)
    wb = np.full_like(c_map, 9240.0)
    skull = c_map > 1600.0
    k[skull] = 0.32
    c[skull] = 1300.0
    rho[skull] = 1900.0
    wb[skull] = 0.0
    return {"k": k, "c": c, "rho": rho, "wb": wb}


def _heat_source(run: dict, absorption_fraction_skull: float = 0.3) -> np.ndarray:
    """Q = 2 alpha I in W/m^3, CW-equivalent, with skull-only absorption scaling.

    Pinton et al. 2012 (Med Phys 39:299) showed that in the skull bone only
    a fraction of the attenuated acoustic energy is converted to heat — the
    remainder is lost to scattering and mode conversion to shear, neither of
    which deposits heat in the longitudinal-FDTD voxel. Soft tissue is
    nearly purely absorptive, so the fraction applies only where c > 1600.
    """
    pmap = np.asarray(run["pmap"]).astype(np.float32)
    pressure_4d = pmap.transpose(1, 2, 3, 0)
    q = compute_heat_source(
        pressure_4d=pressure_4d,
        c_map=run["c"],
        rho_map=run["rho"],
        alpha_coeff_map=run["alpha"],
        f0=run["f0"],
        gate=slice(None),
    )
    q = q * float(run["pulse_norm"])
    skull = run["c"] > 1600.0
    q[skull] *= float(absorption_fraction_skull)
    return q


def _cem43_from_frames(frames: np.ndarray, times: np.ndarray,
                       T_baseline_C: float = 37.0) -> np.ndarray:
    """Sapareto-Dewey CEM43 (min) trapezoidal-integrated over saved frames."""
    T_C = T_baseline_C + frames
    R = np.where(T_C < 43.0, 0.25, 0.5)
    integrand = R ** (43.0 - T_C)
    dt_min = np.diff(times) / 60.0
    mean_int = 0.5 * (integrand[..., 1:] + integrand[..., :-1])
    return (mean_int * dt_min[None, None, None, :]).sum(axis=-1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True,
                    help="Validation config .py path or dotted module")
    ap.add_argument("--sim-dir", default=None,
                    help="Override the run-dir derived from cfg")
    ap.add_argument("--t-max-s", type=float, default=None,
                    help="Cap simulated session duration (s); "
                         "defaults to min(session_duration_s, 80)")
    ap.add_argument("--duty-averaged", action="store_true",
                    help="Collapse the protocol into a continuous duty-averaged source "
                         "(pre-revision behaviour). Default: integrate the burst schedule.")
    ap.add_argument("--dt-on-s", type=float, default=2e-3,
                    help="Explicit time step inside each ON window [s] (default 2 ms).")
    ap.add_argument("--dt-off-s", type=float, default=20e-3,
                    help="Explicit time step inside each OFF window [s] (default 20 ms).")
    ap.add_argument("--frame-every-groups", type=int, default=10,
                    help="Capture a full temperature frame at the end of the ON window "
                         "every N groups (default 10 -> every 2 s for a 200 ms period).")
    ap.add_argument("--absorption-fraction-skull", type=float, default=None,
                    help="Fraction of attenuated acoustic energy in skull "
                         "that is converted to heat (Pinton 2012). "
                         "Default 0.3, or cfg.SKULL_ABSORPTION_FRACTION if set. "
                         "Applies only to skull voxels (c > 1600).")
    args = ap.parse_args()

    cfg = _load_config(args.config)
    sim_dir = (Path(args.sim_dir).resolve() if args.sim_dir
               else _resolve_sim_dir(cfg))
    print(f"[run_bioheat] sim_dir = {sim_dir}", flush=True)

    seq = getattr(cfg, "NEUROMOD_SEQUENCE", {}) or {}
    # The macro ON/OFF duty is carried entirely by group_on_s / group_period_s
    # below. ``duty_cycle_in_burst`` is the *intra-burst* duty (the fraction
    # of the ON window the carrier is actually on); for a CW-within-burst
    # protocol like Yaakub it is 1.0. It must NOT fall back to the config's
    # ``duty_cycle`` field, which already encodes the macro duty
    # (group_on/group_period) — doing so double-counts the duty (e.g.
    # 0.15 x 0.15 = 0.0225 instead of 0.15) and under-heats by ~6.7x.
    duty_in_burst = float(seq.get("duty_cycle_in_burst", 1.0))
    group_period_s = float(seq.get("group_period_s",
                                   seq.get("pulse_period_s", 0.2)))
    group_on_s = float(seq.get("group_on_s",
                               seq.get("tone_burst_duration_s", 30e-3)))
    sess_default = float(seq.get("session_duration_s", 80.0))
    t_max = float(args.t_max_s if args.t_max_s is not None
                  else min(sess_default, 80.0))
    n_groups = max(int(round(t_max / group_period_s)), 1)

    if args.absorption_fraction_skull is not None:
        absorption_fraction_skull = float(args.absorption_fraction_skull)
    else:
        absorption_fraction_skull = float(
            getattr(cfg, "SKULL_ABSORPTION_FRACTION", 0.3))

    run = _load_run(sim_dir)
    q_cw = _heat_source(run, absorption_fraction_skull=absorption_fraction_skull)
    duty_session = duty_in_burst * group_on_s / group_period_s
    q_eff = q_cw * duty_session
    t_total = n_groups * group_period_s

    tm = _tissue_maps(run["c"])
    dx, dy, dz = run["dx"], run["dy"], run["dz"]
    dt_heat = compute_stable_dt(tm["k"], tm["c"], tm["rho"], dx, dy, dz)
    capture_every = max(dt_heat * 2, t_total / 40.0)

    print(f"  absorption_fraction_skull = {absorption_fraction_skull:.3f}  "
          f"(Pinton 2012; soft tissue uses full alpha)", flush=True)
    print(f"  Q_cw peak = {q_cw.max():.3e} W/m^3", flush=True)
    print(f"  duty_session = {duty_session:.4f}, "
          f"Q_eff peak = {q_eff.max():.3e} W/m^3", flush=True)
    print(f"  t_total = {t_total:.1f} s, dt_heat = {dt_heat:.3f} s, "
          f"n_steps ~= {int(t_total / dt_heat)}", flush=True)

    # Focus voxel (peak <p^2> in brain) is needed before the solve for the trace.
    skull = run["c"] > 1600.0
    pmap = np.asarray(run["pmap"])
    p2_mean = (pmap.astype(np.float64) ** 2).mean(axis=0)
    p2_brain = np.where(skull, -np.inf, p2_mean)
    focus_idx = (np.unravel_index(int(np.argmax(p2_brain)), p2_brain.shape)
                 if np.isfinite(p2_brain).any() else tuple(x // 2 for x in run["c"].shape))

    u0 = np.zeros_like(run["c"])
    burst_resolved = not args.duty_averaged
    if burst_resolved:
        # ---- integrate the actual ON/OFF protocol -------------------------------
        dt_on = min(float(args.dt_on_s), dt_heat, group_on_s)
        t_off_s = max(group_period_s - group_on_s, 0.0)
        dt_off = min(float(args.dt_off_s), dt_heat, t_off_s) if t_off_s > 0 else 0.0
        q_on = q_cw * duty_in_burst
        q_zero = np.zeros_like(q_cw)
        trace_t: list[float] = []; trace_focus: list[float] = []; trace_peak: list[float] = []
        frames_l: list[np.ndarray] = []; frame_times_l: list[float] = []
        state = {"t0": 0.0}

        def _rec(step: int, t: float, T: np.ndarray) -> None:
            trace_t.append(state["t0"] + t)
            trace_focus.append(float(T[focus_idx]))
            trace_peak.append(float(T.max()))

        print(f"  burst-resolved: {n_groups} groups x ({group_on_s*1e3:.0f} ms ON @ dt={dt_on*1e3:.1f} ms"
              f" + {t_off_s*1e3:.0f} ms OFF @ dt={dt_off*1e3:.1f} ms); "
              f"~{n_groups*(int(np.ceil(group_on_s/dt_on)) + (int(np.ceil(t_off_s/dt_off)) if dt_off else 0))} steps",
              flush=True)
        T_cur = u0
        T_end_of_last_on = None
        for g in range(n_groups):
            r_on = pennes_solve(u0=T_cur, q=q_on, k_map=tm["k"], c_map=tm["c"], rho_map=tm["rho"],
                                wb_map=tm["wb"], dx=dx, dy=dy, dz=dz, dt=dt_on, t_max=group_on_s,
                                capture_every=0.0, callback=_rec)
            T_cur = r_on.temperature; state["t0"] += group_on_s
            if (g % max(int(args.frame_every_groups), 1) == 0) or g == n_groups - 1:
                frames_l.append(T_cur.astype(np.float32).copy()); frame_times_l.append(state["t0"])
            if g == n_groups - 1:
                T_end_of_last_on = T_cur.copy()
            if t_off_s > 0:
                r_off = pennes_solve(u0=T_cur, q=q_zero, k_map=tm["k"], c_map=tm["c"], rho_map=tm["rho"],
                                     wb_map=tm["wb"], dx=dx, dy=dy, dz=dz, dt=dt_off, t_max=t_off_s,
                                     capture_every=0.0, callback=_rec)
                T_cur = r_off.temperature; state["t0"] += t_off_s
            if (g + 1) % max(n_groups // 8, 1) == 0:
                print(f"    group {g+1}/{n_groups}  t={state['t0']:.1f} s  "
                      f"dT_focus={trace_focus[-1]*1e3:.1f} mK  dT_peak={trace_peak[-1]*1e3:.1f} mK", flush=True)
        delta_T_final = T_cur
        frames = np.stack(frames_l, axis=-1)                 # (nx, ny, nz, n_t) like pennes frames
        frame_times = np.asarray(frame_times_l, dtype=np.float64)
        history_t = np.asarray(trace_t, dtype=np.float64)
        history_peak = np.asarray(trace_peak, dtype=np.float32)
        focus_trace = np.asarray(trace_focus, dtype=np.float32)
        # within-burst excursion at the focus over the last group period
        last = history_t >= (t_total - group_period_s - 1e-9)
        ripple_focus = float(focus_trace[last].max() - focus_trace[last].min()) if last.any() else float("nan")
        ripple_peak = float(history_peak[last].max() - history_peak[last].min()) if last.any() else float("nan")
        dT_focus_end_on = float(T_end_of_last_on[focus_idx]) if T_end_of_last_on is not None else float("nan")
        dT_peak_end_on = float(T_end_of_last_on.max()) if T_end_of_last_on is not None else float("nan")
    else:
        # ---- collapsed duty-averaged source (pre-revision behaviour) ----------
        result = pennes_solve(
            u0=u0, q=q_eff, k_map=tm["k"], c_map=tm["c"], rho_map=tm["rho"],
            wb_map=tm["wb"], dx=dx, dy=dy, dz=dz, dt=dt_heat,
            t_max=t_total, capture_every=capture_every,
        )
        delta_T_final = result.temperature
        frames = result.frames
        frame_times = getattr(result, "times", None)
        if frame_times is None:
            frame_times = getattr(result, "frame_times", None)
        if frames is not None and frame_times is not None:
            history_t = np.asarray(frame_times, dtype=np.float64)
            history_peak = frames.max(axis=(0, 1, 2)).astype(np.float32)
        else:
            history_t = np.array([0.0, t_total], dtype=np.float64)
            history_peak = np.array([0.0, float(delta_T_final.max())], dtype=np.float32)
            frames = None
            frame_times = None
        focus_trace = None
        ripple_focus = ripple_peak = dT_focus_end_on = dT_peak_end_on = float("nan")

    peak_idx = np.unravel_index(int(np.argmax(delta_T_final)), delta_T_final.shape)
    if (~skull).any():
        brain_dT = np.where(skull, -np.inf, delta_T_final)
        brain_peak_idx = np.unravel_index(int(np.argmax(brain_dT)), brain_dT.shape)
    else:
        brain_peak_idx = peak_idx
    if frames is not None and frame_times is not None:
        cem43 = _cem43_from_frames(frames, np.asarray(frame_times, dtype=np.float64))
    else:
        cem43 = np.zeros_like(delta_T_final, dtype=np.float32)

    # Single-burst conduction-free upper bound for sanity row
    rho_c = tm["rho"] * tm["c"]
    dT_per_burst_field = (q_cw * duty_in_burst) * group_on_s / rho_c
    dT_per_burst_peak = float(dT_per_burst_field.max())

    metrics = {
        "peak_dT_final_K": float(delta_T_final.max()),
        "peak_dT_history_K": float(history_peak.max()),
        "dT_focus_K": float(delta_T_final[focus_idx]),
        "dT_per_burst_K": dT_per_burst_peak,
        "burst_resolved": bool(burst_resolved),
        "dt_on_s": (float(dt_on) if burst_resolved else None),
        "dt_off_s": (float(dt_off) if burst_resolved else None),
        "dT_focus_end_of_last_on_K": dT_focus_end_on,
        "dT_peak_end_of_last_on_K": dT_peak_end_on,
        "dT_focus_within_burst_ripple_K": ripple_focus,
        "dT_peak_within_burst_ripple_K": ripple_peak,
        "cem43_peak_min": float(cem43.max()),
        "dT_brain_peak_K": (float(delta_T_final[~skull].max())
                            if (~skull).any() else 0.0),
        "dT_skull_peak_K": (float(delta_T_final[skull].max())
                            if skull.any() else 0.0),
        "duty_in_burst": duty_in_burst,
        "duty_session": duty_session,
        "group_period_s": group_period_s,
        "group_on_s": group_on_s,
        "n_groups": n_groups,
        "t_total_s": t_total,
        "dt_heat_s": float(dt_heat),
        "absorption_fraction_skull": absorption_fraction_skull,
    }

    out_dir = sim_dir / "bioheat"
    out_dir.mkdir(exist_ok=True)
    np.save(out_dir / "delta_T_final.npy", delta_T_final.astype(np.float32))
    if frames is not None:
        np.save(out_dir / "delta_T_frames.npy", frames.astype(np.float32))
        np.save(out_dir / "delta_T_frame_times.npy",
                np.asarray(frame_times, dtype=np.float64))
    np.save(out_dir / "delta_T_history_t.npy", history_t)
    np.save(out_dir / "delta_T_history_peak.npy", history_peak)
    np.save(out_dir / "cem43.npy", cem43.astype(np.float32))
    if focus_trace is not None:
        np.save(out_dir / "focus_trace_t.npy", history_t)
        np.save(out_dir / "focus_trace_dT.npy", focus_trace)
        np.save(out_dir / "peak_trace_dT.npy", history_peak)
    (out_dir / "thermal_indices.json").write_text(json.dumps({
        "peak_idx": [int(i) for i in peak_idx],
        "brain_peak_idx": [int(i) for i in brain_peak_idx],
        "focus_idx": [int(i) for i in focus_idx],
    }, indent=2))
    (out_dir / "thermal_metrics.json").write_text(
        json.dumps(metrics, indent=2)
    )

    print(f"  peak dT = {metrics['peak_dT_final_K']:.4f} K  "
          f"(brain {metrics['dT_brain_peak_K']:.4f}, "
          f"skull {metrics['dT_skull_peak_K']:.4f})", flush=True)
    print(f"  CEM43 peak = {metrics['cem43_peak_min']:.3e} min", flush=True)
    print(f"  wrote {out_dir}/", flush=True)


if __name__ == "__main__":
    main()
