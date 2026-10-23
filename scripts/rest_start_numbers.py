"""File-sourced numbers for the rest-start canonical halle_dacc run.

Since ``mechano_init="rest"`` became the default, the canonical neural
stage (``data/halle_dacc/neural_results.npz``, S2, 3-comp neuron, Piezo1 +
TREK-1, K_A 0.25 N/m, dt 0.001 ms RK4) starts every voxel from true rest.
``burst_periodicity.json`` describes the older run whose period 1 used the
legacy peak-tension init; this driver does not touch it and instead writes

* ``data/halle_dacc/calcium/rest_start_numbers.json`` and
* ``data/halle_dacc/calcium/rest_start_numbers.tex`` (LaTeX macros)

containing

1. single-burst statistics read from ``neural_results.npz``: first-spike
   latency over firing voxels, peak firing rate, firing voxels and spikes;
2. a rest-start periodicity re-run: the canonical firing voxels plus the
   ``--n-silent`` highest-tension eligible-but-silent voxels, chained over
   20 ms ON / 180 ms OFF periods (exponential OFF relaxation with the
   manifest ``relax_tau_ms``), period 1 starting from
   ``rest_initial_state`` (the default), for S2 (passive Ca readout) and
   S4 (Piezo1 inactivation + Ca + SK). S4 (or any scenario whose last two
   bursts still differ) is extended up to ``--max-periods``.

Helpers (inputs, train comparison, state layout) are imported from
``scripts/run_burst_periodicity.py`` without modifying it.

    python scripts/rest_start_numbers.py   # picks the GPU with most free memory
"""
from __future__ import annotations

import os
import subprocess


def _pick_gpu() -> None:
    """Expose only the GPU with the most free memory (unless already set)."""
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free",
             "--format=csv,noheader,nounits"], text=True)
        rows = [tuple(int(x) for x in ln.split(",")) for ln in out.strip().splitlines()]
        os.environ["CUDA_VISIBLE_DEVICES"] = str(max(rows, key=lambda r: r[1])[0])
    except Exception:
        pass


_pick_gpu()

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))

from ultraneuron.cuda_backend import resolve_backend  # noqa: E402
from ultraneuron.multicompartment import (  # noqa: E402
    rest_initial_state,
    simulate_voxels_3comp,
    state_layout,
    tension_drive,
)
from scripts.run_burst_periodicity import (  # noqa: E402
    csr_to_steps,
    load_canonical_inputs,
    relax_tau_ms,
)
from scripts.run_neural_sim import (  # noqa: E402
    _DEFAULTS,
    _build_neuron,
    _load_config,
    _resolve_sim_dir,
)

COMP = ("dend", "soma", "ais")


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


# --------------------------------------------------------------------------
def single_burst_stats(canon) -> dict:
    lat = canon["latency_map"]
    fire = canon["firing_map"]
    cnt = canon["spike_count_map"]
    rate = canon["rate_map"]
    dur = float(canon["duration_ms"])
    lf = lat[fire].astype(np.float64)
    # cross-check: latency_map == first CSR spike time of each firing voxel
    off = canon["spike_offsets"]; t = canon["spike_times_ms"]
    first = t[off[:-1]]
    vi = canon["spike_voxel_idx"]
    lat_csr_ok = bool(np.array_equal(lat[vi[:, 0], vi[:, 1], vi[:, 2]], first))
    imax = np.unravel_index(int(np.argmax(rate)), rate.shape)
    return {
        "source": "data/halle_dacc/neural_results.npz",
        "n_firing_voxels": int(fire.sum()),
        "n_spikes_total": int(cnt.sum()),
        "on_window_ms": dur,
        "latency_encoding": "latency_map = time of first spike [ms] for firing voxels, "
                            "NaN for non-firing voxels (initialised NaN in "
                            "simulate_map_3comp_with_spike_times)",
        "latency_nan_outside_firing": bool(np.all(np.isnan(lat[~fire]))
                                           and np.all(np.isfinite(lat[fire]))),
        "latency_map_equals_first_csr_spike": lat_csr_ok,
        "first_spike_latency_ms": {
            "median": float(np.median(lf)), "p10": float(np.percentile(lf, 10)),
            "p90": float(np.percentile(lf, 90)), "min": float(lf.min()),
            "max": float(lf.max()), "mean": float(lf.mean()),
        },
        "rate_definition": "rate_map = spike_count_map / (duration_ms*1e-3), i.e. "
                           "spikes in the single 20 ms ON window divided by 0.020 s "
                           "(scripts/run_neural_sim.py); not an instantaneous ISI rate",
        "peak_rate_hz": float(rate.max()),
        "peak_rate_spike_count": int(cnt[imax]),
        "peak_rate_T_mNm": float(canon["T_peak_mNm"][imax]),
        "median_rate_firing_hz": float(np.median(rate[fire])),
        "median_spikes_per_firing_voxel": float(np.median(cnt[fire])),
        "max_spikes_per_voxel": int(cnt.max()),
        "dt_ms": float(canon["dt_ms"]),
        "sim_git_sha": str(canon["sim_git_sha"]),
        "integrator": str(canon["integrator"]),
    }


def train_match(a: list, b: list, dt: float) -> dict:
    """Exact / within-one-step equality of b vs a, per voxel."""
    n = len(a)
    exact = np.zeros(n, bool); one = np.zeros(n, bool)
    shift = np.full(n, np.nan)
    for i in range(n):
        if len(a[i]) == len(b[i]):
            if len(a[i]) == 0:
                exact[i] = one[i] = True; shift[i] = 0.0
            else:
                d = np.abs(b[i] - a[i])
                exact[i] = not d.any(); one[i] = bool(d.max() <= 1)
                shift[i] = float(d.max()) * dt
    same = np.isfinite(shift)
    return {"exact": exact, "one": one, "shift": shift,
            "frac_exact": float(exact.mean()), "frac_within_one_step": float(one.mean()),
            "frac_same_count": float(same.mean()),
            "n_not_exact": int((~exact).sum()), "n_count_differs": int((~same).sum()),
            "max_shift_ms_same_count": float(np.nanmax(shift)) if same.any() else 0.0}


def strip(m: dict) -> dict:
    return {k: v for k, v in m.items() if k not in ("exact", "one", "shift")}


def state_vs_rest(S: np.ndarray, R: np.ndarray, n_ch: int, ch_names) -> dict:
    lay = state_layout(n_ch)
    D = np.abs(S - R)
    out = {"V_mV": {COMP[c]: {"max": float(D[:, lay["V"].start + c].max()),
                              "median": float(np.median(D[:, lay["V"].start + c]))}
                    for c in range(3)}}
    hh = {g: float(D[:, lay[g]].max()) for g in ("m", "h", "n")}
    out["HH_gate_max_abs"] = {"max": max(hh.values()), **hh}
    w = D[:, lay["w"]]; hm = D[:, lay["h_mech"]]
    per = {}
    for c in range(3):
        for k in range(n_ch):
            per[f"w_{ch_names[k]}_{COMP[c]}"] = float(w[:, c * n_ch + k].max())
            per[f"h_mech_{ch_names[k]}_{COMP[c]}"] = float(hm[:, c * n_ch + k].max())
    out["mechano_gate_max_abs"] = {"max": float(max(w.max(), hm.max())),
                                   "w_max": float(w.max()), "h_mech_max": float(hm.max()),
                                   "per_channel_compartment": per}
    out["Ca_uM_max_abs"] = {COMP[c]: float(D[:, lay["Ca"].start + c].max()) for c in range(3)}
    return out


def chain(T, nr, drv, dt, period, t_on, backend, n_min, n_max, extend):
    n_on = int(round(t_on / dt))
    st = None
    rec = {"on": [], "off_spikes": [], "start": [], "ca_peak": []}
    p = 0
    while True:
        st0 = rest_initial_state(nr, T.size, calcium=True) if st is None else st
        r = simulate_voxels_3comp(T, nr, dt=dt, duration=period, backend=backend,
                                  drive=drv, initial_state=st0, return_state=True,
                                  calcium_readout=True,
                                  max_spikes_per_voxel=int(period) + 8)
        steps = csr_to_steps(r["spike_offsets"], r["spike_times_ms"], dt)
        on = [s[s <= n_on] for s in steps]
        rec["on"].append(on)
        rec["off_spikes"].append(int(sum(len(s) - len(o) for s, o in zip(steps, on))))
        rec["start"].append(st0)
        rec["ca_peak"].append(r["ca_peak_uM"])
        st = r["final_state"]
        p += 1
        if p < n_min:
            continue
        if p >= n_max or not extend:
            break
        if train_match(rec["on"][-2], rec["on"][-1], dt)["frac_exact"] == 1.0:
            break
    rec["n_periods"] = p
    return rec


# --------------------------------------------------------------------------
def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="validation_report/configs/halle_dacc.py")
    ap.add_argument("--backend", default="auto", choices=["auto", "cuda", "numba"])
    ap.add_argument("--n-silent", type=int, default=1000)
    ap.add_argument("--n-periods", type=int, default=3)
    ap.add_argument("--max-periods", type=int, default=8)
    args = ap.parse_args()

    t_start = time.perf_counter()
    cfg = _load_config(args.config)
    sim_dir = _resolve_sim_dir(cfg)
    out_dir = sim_dir / "calcium"
    backend = resolve_backend(args.backend)
    K_A = float(getattr(cfg, "NEURAL_K_A", _DEFAULTS["K_A"]))
    alpha = float(getattr(cfg, "NEURAL_COUPLING_FACTOR", _DEFAULTS["coupling_factor"]))
    thr = _DEFAULTS["tension_threshold"]
    dt = _DEFAULTS["dt_ms"]
    seq = getattr(cfg, "NEUROMOD_SEQUENCE", {}) or {}
    t_on = float(seq.get("arf_push_window_s", 0.02)) * 1e3
    period = float(seq.get("pulse_period_s", 0.2)) * 1e3
    tau = relax_tau_ms(sim_dir)
    print(f"[rest_start] backend={backend} CUDA_VISIBLE_DEVICES="
          f"{os.environ.get('CUDA_VISIBLE_DEVICES')} ON={t_on} period={period} "
          f"tau={tau:.2f}", flush=True)

    canon = np.load(sim_dir / "neural_results.npz")
    summary: dict = {
        "description": __doc__.split("\n\n")[0],
        "provenance": {
            "git_sha": git_sha(),
            "generating_script": "scripts/rest_start_numbers.py",
            "date": datetime.now().astimezone().isoformat(timespec="seconds"),
            "mechano_init": "rest",
            "period1_initial_state": "rest_initial_state(neuron, N, calcium=True) "
                                     "(= default initial_state=None)",
            "config": str(args.config),
            "input_files": [str((sim_dir / f).relative_to(REPO_ROOT)) for f in (
                "neural_results.npz", "displacement.npy", "c_ds.npy",
                "grid_params.npz", "validation_report/manifest.json")],
            "backend": backend,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "not_overwritten": "data/halle_dacc/calcium/burst_periodicity.json "
                               "(legacy peak_steady period-1 init)",
        },
        "params": {"K_A": K_A, "alpha": alpha, "threshold_mNm": thr, "dt_ms": dt,
                   "t_on_ms": t_on, "period_ms": period, "off_mode": "exp",
                   "tau_relax_ms": tau,
                   "tau_relax_source": "validation_report/manifest.json:"
                                       "displacement_outputs.relax_tau_ms"},
        "single_burst": single_burst_stats(canon),
    }
    sb = summary["single_burst"]
    print(f"  single burst: {sb['n_firing_voxels']} voxels, {sb['n_spikes_total']} spikes, "
          f"latency median {sb['first_spike_latency_ms']['median']:.3f} ms, "
          f"peak rate {sb['peak_rate_hz']:.0f} Hz", flush=True)

    # ---------------- voxel set ------------------------------------------
    strain, tension = load_canonical_inputs(cfg, sim_dir, K_A, alpha)
    del strain
    mask = tension >= thr
    elig_lin = np.flatnonzero(mask.reshape(-1))
    elig_T = tension.reshape(-1)[elig_lin].astype(np.float64)
    fire_lin = np.ravel_multi_index(canon["spike_voxel_idx"].astype(np.int64).T,
                                    tension.shape)
    silent = ~np.isin(elig_lin, fire_lin)
    sil_lin = elig_lin[silent][np.argsort(-elig_T[silent], kind="stable")[:args.n_silent]]
    vox_lin = np.concatenate([fire_lin, sil_lin])
    T_vox = tension.reshape(-1)[vox_lin].astype(np.float64)
    group = np.concatenate([np.ones(fire_lin.size, bool), np.zeros(sil_lin.size, bool)])
    canon_steps = csr_to_steps(canon["spike_offsets"], canon["spike_times_ms"], dt)
    canon_steps += [np.zeros(0, np.int64)] * sil_lin.size
    T_canon_npz = canon["T_peak_mNm"].reshape(-1)[vox_lin]
    summary["voxels"] = {
        "n_canonical_firing": int(fire_lin.size),
        "n_silent_highest_tension": int(sil_lin.size),
        "n_eligible": int(elig_lin.size),
        "firing_T_range_mNm": [float(T_vox[group].min()), float(T_vox[group].max())],
        "silent_T_range_mNm": [float(T_vox[~group].min()), float(T_vox[~group].max())],
        "tension_matches_npz_T_peak_max_abs": float(np.abs(T_vox - T_canon_npz).max()),
    }

    drv = tension_drive(dt, t_on, period - t_on, 1, off_mode="exp", tau_relax_ms=tau)
    res = {}
    for scen in ("S2", "S4"):
        nr = _build_neuron(cfg, scen)
        n_ch = len(nr.mechano_channels)
        ch_names = [c.name for c in nr.mechano_channels]
        t0 = time.perf_counter()
        ch = chain(T_vox, nr, drv, dt, period, t_on, backend, args.n_periods,
                   args.max_periods, extend=(scen == "S4"))
        wall = time.perf_counter() - t0
        n = ch["n_periods"]
        on = ch["on"]
        lay = state_layout(n_ch)
        m: dict = {"n_periods_run": n, "wall_s": wall, "channels": ch_names,
                   "on_spikes_per_burst": [int(sum(len(s) for s in o)) for o in on],
                   "on_spikes_per_burst_canonical_firing": [
                       int(sum(len(s) for s, g in zip(o, group) if g)) for o in on],
                   "on_spikes_per_burst_silent_set": [
                       int(sum(len(s) for s, g in zip(o, group) if not g)) for o in on],
                   "on_firing_voxels_per_burst": [int(sum(len(s) > 0 for s in o)) for o in on],
                   "off_window_spikes_per_period": ch["off_spikes"]}
        if scen == "S2":
            m["burst1_vs_canonical_npz"] = strip(train_match(canon_steps, on[0], dt))
        for p in range(1, n):
            m[f"burst{p+1}_vs_burst1"] = strip(train_match(on[0], on[p], dt))
            m[f"burst{p+1}_vs_burst{p}"] = strip(train_match(on[p - 1], on[p], dt))
        # reference = analytic rest state (period-1 start); it differs from the
        # integrated zero-tension fixed point by ~2 uV in V, ~1e-6 in HH gates
        R = rest_initial_state(nr, T_vox.size, calcium=True)
        m["state_burst2_start_vs_rest"] = state_vs_rest(ch["start"][1], R, n_ch, ch_names)
        if n >= 3:
            m["state_burst3_start_vs_rest"] = state_vs_rest(ch["start"][2], R, n_ch, ch_names)
        ca_rest = float(nr.calcium_dynamics.Ca_rest if nr.calcium_dynamics else 0.05)
        ex_start = ch["start"][1][:, lay["Ca"]] - ca_rest
        ex_peak = ch["ca_peak"][0] - ca_rest
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.where(ex_peak > 1e-12, ex_start / ex_peak, np.nan)
        m["Ca_rest_uM"] = ca_rest
        def _fs(x):
            x = x[np.isfinite(x)]
            if x.size == 0:  # no Ca influx in this compartment (peak excess 0)
                return {"median": None, "max": None, "n": 0}
            return {"median": float(np.median(x)), "max": float(x.max()), "n": int(x.size)}
        m["Ca_residue_burst2_start_over_burst1_peak_excess"] = {
            "all_voxels": {COMP[i]: _fs(frac[:, i]) for i in range(3)},
            "canonical_firing": {COMP[i]: _fs(frac[group, i]) for i in range(3)},
            "note": "(Ca(burst-2 start) - Ca_rest) / (burst-1 period Ca peak - Ca_rest); "
                    "AIS has no Ca influx (no Piezo1 there), so its ratio is undefined",
        }
        m["burst1_ca_peak_uM_median_firing"] = {
            COMP[i]: float(np.median(ch["ca_peak"][0][group, i])) for i in range(3)}
        # bursts until periodic: first burst p such that bursts p..n all equal burst n
        def _settle(key):
            for p in range(n):
                if all(train_match(on[q], on[-1], dt)[key].all() for q in range(p, n)):
                    return p + 1
            return None
        last_eq = train_match(on[-2], on[-1], dt)["frac_exact"] == 1.0
        m["periodic"] = {
            "last_two_bursts_identical": bool(last_eq),
            "first_burst_identical_to_all_later_exact": _settle("exact") if last_eq else None,
            "first_burst_identical_to_all_later_within_one_step": _settle("one"),
            "note": "burst index (1-based); 1 = burst 1 already periodic; None = "
                    "not periodic within n_periods_run",
        }
        res[scen] = m
        print(f"  {scen}: {n} periods, ON spikes {m['on_spikes_per_burst']}, "
              f"b2==b1 {m['burst2_vs_burst1']['frac_exact']:.4f}, settled "
              f"{m['periodic']['first_burst_identical_to_all_later_exact']} "
              f"({wall:.0f} s)", flush=True)
    summary["periodicity"] = res
    summary["wall_s_total"] = time.perf_counter() - t_start

    (out_dir / "rest_start_numbers.json").write_text(json.dumps(summary, indent=1))
    write_tex(summary, out_dir / "rest_start_numbers.tex")
    print(f"[rest_start] wrote {out_dir/'rest_start_numbers.json'} and .tex", flush=True)


# --------------------------------------------------------------------------
_NUM = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six",
        7: "Seven", 8: "Eight"}


def write_tex(s: dict, path: Path) -> None:
    sb = s["single_burst"]; L = sb["first_spike_latency_ms"]
    mac = {
        "RestFiringVoxels": f"{sb['n_firing_voxels']}",
        "RestSpikes": f"{sb['n_spikes_total']}",
        "RestLatencyMedian": f"{L['median']:.2f}",
        "RestLatencyPTen": f"{L['p10']:.2f}",
        "RestLatencyPNinety": f"{L['p90']:.2f}",
        "RestLatencyMin": f"{L['min']:.2f}",
        "RestLatencyMax": f"{L['max']:.2f}",
        "RestPeakRate": f"{sb['peak_rate_hz']:.0f}",
        "RestPeakRateSpikes": f"{sb['peak_rate_spike_count']}",
        "RestTauRelax": f"{s['params']['tau_relax_ms']:.1f}",
        "RestSilentVoxels": f"{s['voxels']['n_silent_highest_tension']}",
    }
    for scen, m in s["periodicity"].items():
        P = f"Rest{scen.replace('2', 'Two').replace('4', 'Four')}"
        for p, v in enumerate(m["on_spikes_per_burst"], start=1):
            if p <= 3:
                mac[f"{P}SpikesBurst{_NUM[p]}"] = f"{v}"
        for p in (2, 3):
            k = f"burst{p}_vs_burst1"
            if k in m:
                mac[f"{P}Burst{_NUM[p]}ExactPct"] = f"{100 * m[k]['frac_exact']:.2f}"
                mac[f"{P}Burst{_NUM[p]}OneStepPct"] = f"{100 * m[k]['frac_within_one_step']:.2f}"
                mac[f"{P}Burst{_NUM[p]}MaxShift"] = f"{m[k]['max_shift_ms_same_count']:.3f}"
        st = m["state_burst2_start_vs_rest"]
        for c, C in zip(COMP, ("Dend", "Soma", "AIS")):
            mac[f"{P}VResMax{C}"] = f"{st['V_mV'][c]['max']:.3g}"
            mac[f"{P}VResMedian{C}"] = f"{st['V_mV'][c]['median']:.3g}"
        mac[f"{P}HHGateRes"] = f"{st['HH_gate_max_abs']['max']:.1e}"
        mac[f"{P}MechGateRes"] = f"{st['mechano_gate_max_abs']['max']:.1e}"
        ca = m["Ca_residue_burst2_start_over_burst1_peak_excess"]["all_voxels"]
        for c, C in zip(COMP, ("Dend", "Soma", "AIS")):
            if ca[c]["median"] is not None:
                mac[f"{P}CaResidue{C}"] = f"{ca[c]['median']:.3f}"
        per = m["periodic"]["first_burst_identical_to_all_later_exact"]
        mac[f"{P}PeriodicBurst"] = str(per) if per is not None else "none"
        mac[f"{P}PeriodsRun"] = f"{m['n_periods_run']}"
    for k in mac:
        assert k.isalpha(), k
    pv = s["provenance"]
    lines = [
        "% generated by scripts/rest_start_numbers.py from data/halle_dacc/neural_results.npz "
        "and a rest-start periodicity re-run",
        f"% generated_at = {pv['date']}  git = {pv['git_sha'][:7]}",
        "% rest start (mechano_init='rest'); latency in ms, rate in Hz (spikes / 20 ms ON), "
        "residuals |x - x_rest| at the start of burst 2 (V in mV), Ca residue = "
        "median fraction of the burst-1 peak excess",
    ]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in mac.items()]
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
