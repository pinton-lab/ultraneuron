"""Burst-to-burst periodicity of the canonical theta-burst neural response.

The canonical neural stage (``scripts/run_neural_sim.py``; S2 scenario,
3-compartment Pospischil neuron, Piezo1 + TREK-1, K_A 0.25 N/m, alpha
1000, dt 0.001 ms RK4) starts every voxel from rest and integrates one
20 ms ON window at constant peak tension; the session (400 bursts, 20 ms
ON every 200 ms) is then assumed to repeat that single-burst response.
This driver tests that assumption by chaining successive 200 ms periods
with the full per-voxel state (V, HH gates, mechano activation and
inactivation, [Ca]) carried over, using
``multicompartment.simulate_voxels_3comp``.

Stages
------
A. Passive-Ca invariance: the canonical whole-field map with and without
   a passive Ca readout (no KCa feedback) on the default backend (CUDA
   when visible), compared with the stored ``neural_results.npz`` and,
   with ``--cpu-parity``, with the Numba CPU kernel.
B. True resting state: one voxel at zero tension integrated for 2 s from
   the canonical initial state (V = V_rest = -70 mV, gates at steady state).
C. Periodicity: for every canonical firing voxel plus the ``--n-silent``
   eligible-but-silent voxels with the highest tension (closest to the
   firing threshold), ``--n-periods`` consecutive 200 ms periods (20 ms ON
   at constant peak tension, 180 ms OFF with the tension relaxing
   exponentially with the visco-elastic tau, or dropping to zero with
   ``--off-modes step``), for scenarios S2 (passive Ca readout) and S4
   (Piezo1 inactivation + Ca + SK). Period 1 starts from the canonical
   initial state, so its ON window reproduces the canonical run exactly.
   (Since the default became ``mechano_init="rest"``, the legacy
   peak-tension init is requested explicitly with
   ``mechano_init="peak_steady"`` throughout this driver, so it still
   reproduces its stored outputs.)
   A separate 20 ms ON window started from the true resting state
   (mechano gates closed, as they are at the start of every later burst)
   isolates the effect of the canonical initial condition.
D. Approach to periodic steady state: ``--long-periods`` periods for a
   voxel sample (all voxels whose bursts 2..n differ from burst n, plus a
   random sample), per scenario/off-mode.

Outputs: ``<sim_dir>/calcium/burst_periodicity.json`` (summary) and
``burst_periodicity.npz`` (per-voxel arrays).

Usage
-----
    CUDA_VISIBLE_DEVICES=0 python scripts/run_burst_periodicity.py \\
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
sys.path.insert(0, str(REPO_ROOT))

from ultraneuron.channels import strain_to_membrane_tension  # noqa: E402
from ultraneuron.cuda_backend import resolve_backend  # noqa: E402
from ultraneuron.multicompartment import (  # noqa: E402
    peak_steady_initial_state,
    simulate_map_3comp_with_spike_times,
    simulate_voxels_3comp,
    state_layout,
    tension_drive,
)
from scripts.run_neural_sim import (  # noqa: E402
    _DEFAULTS,
    _build_neuron,
    _git_sha,
    _load_config,
    _resolve_sim_dir,
    _strain_field_from_displacement,
)


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------
def load_canonical_inputs(cfg, sim_dir: Path, K_A: float, alpha: float):
    """Reproduce the canonical brain-masked strain field and tension map."""
    disp = np.load(sim_dir / "displacement.npy").astype(np.float32)
    c_map = np.load(sim_dir / "c_ds.npy").astype(np.float32)
    grid = np.load(sim_dir / "grid_params.npz")
    dx = float(grid["dx"]) * int(grid["mod_x"])
    dy = float(grid["dy"]) * int(grid["mod_y"])
    dz = float(grid["dz"]) * int(grid["mod_z"])
    strain = _strain_field_from_displacement(disp, dx, dy, dz)
    strain[c_map > 1600.0] = 0.0
    tension = strain_to_membrane_tension(strain, K_A, alpha)
    return strain, tension


def relax_tau_ms(sim_dir: Path) -> float:
    man = json.loads((sim_dir / "validation_report" / "manifest.json").read_text())
    return float(man["displacement_outputs"]["relax_tau_ms"])


def csr_to_steps(offsets, times_ms, dt):
    steps = np.rint(np.asarray(times_ms, dtype=np.float64) / dt).astype(np.int64)
    return [steps[offsets[i]:offsets[i + 1]] for i in range(len(offsets) - 1)]


# --------------------------------------------------------------------------
# Train comparison
# --------------------------------------------------------------------------
def compare_trains(a: list, b: list, dt: float, group: np.ndarray | None = None):
    """Per-voxel comparison of two lists of spike-step arrays."""
    n = len(a)
    ca = np.array([len(x) for x in a]); cb = np.array([len(x) for x in b])
    same_count = ca == cb
    identical = np.zeros(n, dtype=bool)
    max_shift = np.full(n, np.nan)
    lat_shift = np.full(n, np.nan)
    for i in range(n):
        if len(a[i]) and len(b[i]):
            lat_shift[i] = (b[i][0] - a[i][0]) * dt
        if same_count[i]:
            if ca[i] == 0:
                identical[i] = True; max_shift[i] = 0.0
            else:
                d = np.abs(b[i] - a[i])
                identical[i] = not d.any()
                max_shift[i] = d.max() * dt

    def _summ(sel):
        k = int(sel.sum())
        if k == 0:
            return {"n": 0}
        ms = max_shift[sel & same_count]
        ls = lat_shift[sel]
        dc = (cb - ca)[sel]
        return {
            "n": k,
            "frac_identical_train": float(identical[sel].mean()),
            "frac_same_count": float(same_count[sel].mean()),
            "n_count_up": int((dc > 0).sum()),
            "n_count_down": int((dc < 0).sum()),
            "delta_count_hist": {str(int(v)): int((dc == v).sum())
                                 for v in np.unique(dc)},
            "spikes_a": int(ca[sel].sum()), "spikes_b": int(cb[sel].sum()),
            "n_firing_a": int((ca[sel] > 0).sum()),
            "n_firing_b": int((cb[sel] > 0).sum()),
            "max_abs_shift_ms_same_count": (float(np.nanmax(ms))
                                            if np.isfinite(ms).any() else 0.0),
            "median_abs_shift_ms_same_count_nonidentical": (
                float(np.nanmedian(ms[ms > 0])) if (ms > 0).any() else 0.0),
            "first_latency_shift_ms": {
                "min": float(np.nanmin(ls)) if np.isfinite(ls).any() else None,
                "max": float(np.nanmax(ls)) if np.isfinite(ls).any() else None,
                "median": float(np.nanmedian(ls)) if np.isfinite(ls).any() else None,
            },
        }

    out = {"all": _summ(np.ones(n, dtype=bool))}
    if group is not None:
        out["canonical_firing"] = _summ(group == 1)
        out["silent_near_threshold"] = _summ(group == 0)
    return out, {"identical": identical, "same_count": same_count,
                 "max_shift_ms": max_shift, "latency_shift_ms": lat_shift}


def state_diff_summary(S: np.ndarray, R: np.ndarray, n_ch: int, ch_names, sel=None):
    """max / median |S - R| per state group (R broadcastable to S)."""
    lay = state_layout(n_ch)
    comp = ("dend", "soma", "ais")
    D = np.abs(S - R)
    if sel is not None:
        D = D[sel]
    out = {}
    for key in ("V", "m", "h", "n", "Ca"):
        sl = lay[key]
        for c in range(3):
            col = D[:, sl.start + c]
            out[f"{key}_{comp[c]}"] = {"max": float(col.max()),
                                       "median": float(np.median(col))}
    for key in ("w", "h_mech"):
        sl = lay[key]
        for c in range(3):
            for k in range(n_ch):
                col = D[:, sl.start + c * n_ch + k]
                out[f"{key}_{ch_names[k]}_{comp[c]}"] = {
                    "max": float(col.max()), "median": float(np.median(col))}
    return out


# --------------------------------------------------------------------------
# Chained periods
# --------------------------------------------------------------------------
def run_chain(T, neuron, n_periods, drive_period, dt, period_ms, backend,
              init=None, calcium_readout=True, t_on_ms=20.0):
    """Chain ``n_periods`` windows; return per-period trains and states."""
    st = init
    rec = {"steps": [], "on_steps": [], "counts": [], "on_counts": [],
           "state_start": [], "ca_peak": []}
    n_on = int(round(t_on_ms / dt))
    for p in range(n_periods):
        if st is None:
            st0 = peak_steady_initial_state(
                neuron, T, calcium=bool(calcium_readout) or neuron.calcium_dynamics is not None)
        else:
            st0 = st
        r = simulate_voxels_3comp(
            T, neuron, dt=dt, duration=period_ms, backend=backend,
            drive=drive_period, initial_state=st, return_state=True,
            mechano_init="peak_steady",  # period 1 = legacy canonical init
            calcium_readout=calcium_readout,
            max_spikes_per_voxel=int(period_ms) + 8,
        )
        steps = csr_to_steps(r["spike_offsets"], r["spike_times_ms"], dt)
        on_steps = [s[s <= n_on] for s in steps]
        rec["steps"].append(steps)
        rec["on_steps"].append(on_steps)
        rec["counts"].append(r["spike_counts"].copy())
        rec["on_counts"].append(np.array([len(s) for s in on_steps]))
        rec["state_start"].append(st0)
        rec["ca_peak"].append(r.get("ca_peak_uM"))
        st = r["final_state"]
    rec["state_end"] = st
    return rec


def pack_csr(list_of_steps, dt):
    off = np.concatenate([[0], np.cumsum([len(s) for s in list_of_steps])]).astype(np.int64)
    t = (np.concatenate(list_of_steps).astype(np.float64) * dt if off[-1]
         else np.zeros(0)).astype(np.float32)
    return off, t


# --------------------------------------------------------------------------
def main() -> None:  # noqa: C901, PLR0915
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="validation_report/configs/halle_dacc.py")
    ap.add_argument("--backend", choices=["auto", "cuda", "numba"], default="auto")
    ap.add_argument("--scenarios", nargs="+", default=["S2", "S4"])
    ap.add_argument("--off-modes", nargs="+", default=["exp", "step"],
                    choices=["exp", "step"])
    ap.add_argument("--n-periods", type=int, default=3)
    ap.add_argument("--n-silent", type=int, default=1000)
    ap.add_argument("--long-periods", type=int, default=20)
    ap.add_argument("--long-sample", type=int, default=300)
    ap.add_argument("--cpu-parity", action="store_true",
                    help="Also run the whole canonical map on the Numba CPU kernel.")
    ap.add_argument("--skip-invariance", action="store_true")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    t_start = time.perf_counter()
    cfg = _load_config(args.config)
    sim_dir = _resolve_sim_dir(cfg)
    out_dir = Path(args.out_dir) if args.out_dir else sim_dir / "calcium"
    out_dir.mkdir(parents=True, exist_ok=True)
    backend = resolve_backend(args.backend)
    K_A = float(getattr(cfg, "NEURAL_K_A", _DEFAULTS["K_A"]))
    alpha = float(getattr(cfg, "NEURAL_COUPLING_FACTOR", _DEFAULTS["coupling_factor"]))
    thr = _DEFAULTS["tension_threshold"]
    dt = _DEFAULTS["dt_ms"]
    seq = getattr(cfg, "NEUROMOD_SEQUENCE", {}) or {}
    t_on = float(seq.get("arf_push_window_s", 0.02)) * 1e3
    period = float(seq.get("pulse_period_s", 0.2)) * 1e3
    t_off = period - t_on
    tau_relax = relax_tau_ms(sim_dir)
    print(f"[periodicity] sim_dir={sim_dir} backend={backend} K_A={K_A} "
          f"alpha={alpha} ON={t_on} ms period={period} ms tau_relax={tau_relax:.2f} ms",
          flush=True)

    canon = np.load(sim_dir / "neural_results.npz")
    strain, tension = load_canonical_inputs(cfg, sim_dir, K_A, alpha)
    mask = tension >= thr
    elig_idx = np.argwhere(mask)
    elig_T = tension[mask].astype(np.float64)
    summary: dict = {
        "description": __doc__.split("\n\n")[0],
        "config": str(args.config), "sim_dir": str(sim_dir),
        "git_sha": _git_sha(), "backend": backend,
        "params": {"K_A": K_A, "alpha": alpha, "threshold_mNm": thr,
                   "dt_ms": dt, "t_on_ms": t_on, "period_ms": period,
                   "tau_relax_ms": tau_relax,
                   "tau_relax_source": "validation_report/manifest.json:"
                                       "displacement_outputs.relax_tau_ms",
                   "n_eligible": int(elig_T.size)},
        "timing_s": {},
    }
    npz: dict = {}

    # ---------------- Stage A: passive-Ca invariance ----------------------
    neuron_s2 = _build_neuron(cfg, "S2")
    ch_names = [c.name for c in neuron_s2.mechano_channels]
    if not args.skip_invariance:
        inv = {}
        runs = {}
        for label, be, ca in [("cuda_noCa", backend, False),
                              ("cuda_Ca", backend, True)] + (
                [("numba_Ca", "numba", True)] if args.cpu_parity else []):
            t0 = time.perf_counter()
            r = simulate_map_3comp_with_spike_times(
                strain, neuron_s2, K_A=K_A, coupling_factor=alpha, dt=dt,
                duration=t_on, tension_threshold=thr, backend=be,
                calcium_readout=ca, ca_trace_dt_ms=(0.1 if ca else None),
                mechano_init="peak_steady",  # compared with legacy npz
            )
            runs[label] = r
            summary["timing_s"][f"A_{label}"] = time.perf_counter() - t0
            print(f"  A {label}: {int(r['firing_map'].sum())} firing, "
                  f"{int(r['spike_count_map'].sum())} spikes "
                  f"({summary['timing_s'][f'A_{label}']:.1f} s)", flush=True)

        def _same(r1, r2):
            return bool(np.array_equal(r1["spike_count_map"], r2["spike_count_map"])
                        and np.array_equal(r1["spike_voxel_idx"], r2["spike_voxel_idx"])
                        and np.array_equal(r1["spike_offsets"], r2["spike_offsets"])
                        and np.array_equal(r1["spike_times_ms"], r2["spike_times_ms"]))
        canon_r = {k: canon[k] for k in ("spike_count_map", "spike_voxel_idx",
                                         "spike_offsets", "spike_times_ms")}
        for k, r in runs.items():
            inv[f"{k}_vs_canonical_npz_bit_identical"] = _same(r, canon_r)
            inv[f"{k}_n_firing"] = int(r["firing_map"].sum())
            inv[f"{k}_n_spikes"] = int(r["spike_count_map"].sum())
        inv["Ca_vs_noCa_bit_identical"] = _same(runs["cuda_Ca"], runs["cuda_noCa"])
        rc = runs["cuda_Ca"]
        if "numba_Ca" in runs:
            rn = runs["numba_Ca"]
            inv["numba_vs_cuda_spikes_bit_identical"] = _same(rn, rc)
            fin = np.isfinite(rc["ca_peak_map"])
            inv["numba_vs_cuda_ca_peak_max_rel_diff"] = float(np.max(
                np.abs(rn["ca_peak_map"][fin] / rc["ca_peak_map"][fin] - 1.0)))
            inv["numba_vs_cuda_ca_trace_max_rel_diff"] = float(np.max(
                np.abs(rn["ca_trace_firing"] / rc["ca_trace_firing"] - 1.0)))
        cpf = rc["ca_peak_firing"]
        cpm = rc["ca_peak_map"]
        inv["ca_peak_uM"] = {
            comp: {"firing_median": float(np.median(cpf[:, c])),
                   "firing_max": float(cpf[:, c].max()),
                   "firing_min": float(cpf[:, c].min()),
                   "eligible_max": float(np.nanmax(cpm[c])),
                   "eligible_silent_median": float(np.nanmedian(
                       cpm[c][mask & ~rc["firing_map"]]))}
            for c, comp in enumerate(("dendrite", "soma", "ais"))}
        inv["Ca_rest_uM"] = 0.05
        inv["ca_passive"] = bool(rc["ca_passive"])
        summary["A_passive_calcium_invariance"] = inv
        npz["A_ca_peak_map"] = cpm.astype(np.float32)
        npz["A_ca_peak_firing"] = cpf
        npz["A_ca_trace_firing"] = rc["ca_trace_firing"].astype(np.float32)
        npz["A_ca_trace_t_ms"] = rc["ca_trace_t_ms"]
        npz["A_spike_voxel_idx"] = rc["spike_voxel_idx"]

    # ---------------- Voxel set ------------------------------------------
    fire_vox = canon["spike_voxel_idx"].astype(np.int64)
    fire_lin = np.ravel_multi_index(fire_vox.T, tension.shape)
    elig_lin = np.ravel_multi_index(elig_idx.T, tension.shape)
    silent = ~np.isin(elig_lin, fire_lin)
    sil_order = np.argsort(-elig_T[silent], kind="stable")[: args.n_silent]
    sil_lin = elig_lin[silent][sil_order]
    vox_lin = np.concatenate([fire_lin, sil_lin])
    vox_idx = np.stack(np.unravel_index(vox_lin, tension.shape), axis=1)
    T_vox = tension.reshape(-1)[vox_lin].astype(np.float64)
    group = np.concatenate([np.ones(fire_lin.size, np.int8),
                            np.zeros(sil_lin.size, np.int8)])
    n_vox = T_vox.size
    canon_steps = csr_to_steps(canon["spike_offsets"], canon["spike_times_ms"], dt)
    canon_steps += [np.zeros(0, np.int64)] * sil_lin.size
    summary["voxels"] = {
        "n_canonical_firing": int(fire_lin.size),
        "n_silent_near_threshold": int(sil_lin.size),
        "silent_tension_range_mNm": [float(T_vox[group == 0].min()),
                                     float(T_vox[group == 0].max())]
        if sil_lin.size else None,
        "firing_tension_range_mNm": [float(T_vox[group == 1].min()),
                                     float(T_vox[group == 1].max())],
    }
    npz["voxel_idx"] = vox_idx.astype(np.int32)
    npz["voxel_T_peak_mNm"] = T_vox
    npz["voxel_group"] = group

    # ---------------- Stage B: true resting state ------------------------
    rest = {}
    for scen in args.scenarios:
        nr = _build_neuron(cfg, scen)
        t0 = time.perf_counter()
        r1 = simulate_voxels_3comp(np.zeros(1), nr, dt=dt, duration=1000.0,
                                   backend="numba", calcium_readout=True,
                                   record_spike_times=False)
        r2 = simulate_voxels_3comp(np.zeros(1), nr, dt=dt, duration=1000.0,
                                   backend="numba", calcium_readout=True,
                                   initial_state=r1["final_state"],
                                   record_spike_times=False)
        rest[scen] = r2["final_state"][0]
        ci = peak_steady_initial_state(nr, np.zeros(1), calcium=True)[0]
        lay = state_layout(len(nr.mechano_channels))
        summary.setdefault("B_rest_state", {})[scen] = {
            "V_rest_true_mV": rest[scen][lay["V"]].tolist(),
            "V_rest_nominal_mV": nr.V_rest,
            "drift_1s_to_2s_max_abs": float(np.abs(r2["final_state"] - r1["final_state"]).max()),
            "rest_minus_canonical_init_T0": state_diff_summary(
                rest[scen][None], ci[None], len(nr.mechano_channels), ch_names),
            "spikes_at_rest": int(r1["spike_counts"][0] + r2["spike_counts"][0]),
        }
        summary["timing_s"][f"B_{scen}"] = time.perf_counter() - t0
        npz[f"B_rest_state_{scen}"] = rest[scen]

    # ---------------- Stage C: chained periods ---------------------------
    for scen in args.scenarios:
        nr = _build_neuron(cfg, scen)
        n_ch = len(nr.mechano_channels)
        # legacy canonical ON window (mechano gates at peak-tension steady
        # state) for this scenario (S2: = npz)
        t0 = time.perf_counter()
        r_can = simulate_voxels_3comp(T_vox, nr, dt=dt, duration=t_on,
                                      backend=backend, calcium_readout=True,
                                      mechano_init="peak_steady")
        can_steps = csr_to_steps(r_can["spike_offsets"], r_can["spike_times_ms"], dt)
        # ON window from the true resting state (gates closed at onset)
        st_rest = np.tile(rest[scen], (n_vox, 1))
        r_rest = simulate_voxels_3comp(T_vox, nr, dt=dt, duration=t_on,
                                       backend=backend, calcium_readout=True,
                                       initial_state=st_rest)
        rest_steps = csr_to_steps(r_rest["spike_offsets"], r_rest["spike_times_ms"], dt)
        sc_sum: dict = {"n_ch": n_ch, "channels": ch_names}
        if scen == "S2":
            sc_sum["from_rest_vs_canonical_npz"] = compare_trains(
                canon_steps, can_steps, dt, group)[0]
        sc_sum["true_rest_start_vs_canonical_init"] = compare_trains(
            can_steps, rest_steps, dt, group)[0]
        npz[f"C_{scen}_canon_on_counts"] = r_can["spike_counts"]
        npz[f"C_{scen}_restStart_on_counts"] = r_rest["spike_counts"]
        o, t = pack_csr(rest_steps, dt)
        npz[f"C_{scen}_restStart_offsets"] = o; npz[f"C_{scen}_restStart_times_ms"] = t

        for mode in args.off_modes:
            drv = tension_drive(dt, t_on, t_off, 1, off_mode=mode, tau_relax_ms=tau_relax)
            t1 = time.perf_counter()
            ch = run_chain(T_vox, nr, args.n_periods, drv, dt, period, backend,
                           calcium_readout=True, t_on_ms=t_on)
            el = time.perf_counter() - t1
            summary["timing_s"][f"C_{scen}_{mode}"] = el
            key = f"{scen}_{mode}"
            m: dict = {"wall_s": el}
            m["burst1_on_vs_from_rest"] = compare_trains(
                can_steps, ch["on_steps"][0], dt, group)[0]
            for p in range(1, args.n_periods):
                m[f"burst{p+1}_vs_burst1_on"] = compare_trains(
                    ch["on_steps"][0], ch["on_steps"][p], dt, group)[0]
                m[f"burst{p+1}_vs_burst{p}_on"] = compare_trains(
                    ch["on_steps"][p - 1], ch["on_steps"][p], dt, group)[0]
                m[f"burst{p+1}_vs_true_rest_start_on"] = compare_trains(
                    rest_steps, ch["on_steps"][p], dt, group)[0]
                if scen == "S2":
                    m[f"burst{p+1}_vs_canonical_npz"] = compare_trains(
                        canon_steps, ch["on_steps"][p], dt, group)[0]
            m["off_window_spikes_per_period"] = [
                int((ch["counts"][p] - ch["on_counts"][p]).sum())
                for p in range(args.n_periods)]
            m["on_spikes_per_period"] = [int(c.sum()) for c in ch["on_counts"]]
            m["on_firing_voxels_per_period"] = [int((c > 0).sum()) for c in ch["on_counts"]]
            # State at the start of burst 2 relative to canonical init / true rest
            S2s = ch["state_start"][1]
            S1s = ch["state_start"][0]
            m["state_burst2_start_minus_canonical_init"] = state_diff_summary(
                S2s, S1s, n_ch, ch_names)
            m["state_burst2_start_minus_true_rest"] = state_diff_summary(
                S2s, rest[scen][None], n_ch, ch_names)
            if args.n_periods >= 3:
                m["state_burst3_start_minus_burst2_start"] = state_diff_summary(
                    ch["state_start"][2], S2s, n_ch, ch_names)
            lay = state_layout(n_ch)
            ca_rest = float((nr.calcium_dynamics.Ca_rest if nr.calcium_dynamics
                             else 0.05))
            ca_ex_start = S2s[:, lay["Ca"]] - ca_rest
            ca_ex_peak = ch["ca_peak"][0] - ca_rest
            with np.errstate(divide="ignore", invalid="ignore"):
                frac = np.where(ca_ex_peak > 1e-12, ca_ex_start / ca_ex_peak, np.nan)
            m["Ca_excess_at_burst2_start_over_burst1_peak_excess"] = {
                c: {"median": float(np.nanmedian(frac[:, i])),
                    "max": float(np.nanmax(frac[:, i]))}
                for i, c in enumerate(("dend", "soma", "ais"))}
            m["ca_peak_uM_per_period_firing_median_soma"] = [
                float(np.median(cp[group == 1, 1])) for cp in ch["ca_peak"]]
            summary.setdefault("C_periodicity", {})[key] = m
            for p in range(args.n_periods):
                o, t = pack_csr(ch["steps"][p], dt)
                npz[f"C_{key}_p{p+1}_offsets"] = o
                npz[f"C_{key}_p{p+1}_times_ms"] = t
                npz[f"C_{key}_p{p+1}_counts"] = ch["counts"][p]
                npz[f"C_{key}_p{p+1}_on_counts"] = ch["on_counts"][p]
                npz[f"C_{key}_p{p+1}_state_start"] = ch["state_start"][p].astype(np.float32)
                npz[f"C_{key}_p{p+1}_ca_peak"] = ch["ca_peak"][p].astype(np.float32)
            npz[f"C_{key}_state_end"] = ch["state_end"].astype(np.float32)
            print(f"  C {key}: ON spikes/period {m['on_spikes_per_period']}, "
                  f"burst2==burst1 {m['burst2_vs_burst1_on']['all']['frac_identical_train']:.4f}, "
                  f"burst2==rest-start "
                  f"{m['burst2_vs_true_rest_start_on']['all']['frac_identical_train']:.4f} "
                  f"({el:.1f} s)", flush=True)

            # ------------- Stage D: approach to steady state -------------
            last = args.n_periods - 1
            diff = ~compare_trains(ch["on_steps"][last - 1],
                                   ch["on_steps"][last], dt)[1]["identical"]
            if args.long_periods > args.n_periods and (diff.any() or scen == "S4"):
                rng = np.random.default_rng(0)
                fire_rows = np.where(group == 1)[0]
                rnd = rng.choice(fire_rows, size=min(args.long_sample, fire_rows.size),
                                 replace=False)
                rows = np.unique(np.concatenate([np.where(diff)[0], rnd]))
                t2 = time.perf_counter()
                chL = run_chain(T_vox[rows], nr, args.long_periods, drv, dt, period,
                                backend, calcium_readout=True, t_on_ms=t_on)
                elL = time.perf_counter() - t2
                summary["timing_s"][f"D_{scen}_{mode}"] = elL
                ident_prev = []
                ident_last = []
                spikes = []
                st_dist = []
                for p in range(args.long_periods):
                    spikes.append(int(chL["on_counts"][p].sum()))
                    if p > 0:
                        ident_prev.append(float(compare_trains(
                            chL["on_steps"][p - 1], chL["on_steps"][p], dt)[1]
                            ["identical"].mean()))
                    ident_last.append(float(compare_trains(
                        chL["on_steps"][p], chL["on_steps"][-1], dt)[1]
                        ["identical"].mean()))
                    st_dist.append(float(np.abs(chL["state_start"][p]
                                                - chL["state_end"]).max()))
                # first period from which every later burst is identical
                settled = None
                for p in range(args.long_periods):
                    if all(v == 1.0 for v in ident_last[p:]):
                        settled = p + 1
                        break
                summary.setdefault("D_steady_state", {})[key] = {
                    "n_voxels": int(rows.size),
                    "n_voxels_burst_n_minus1_ne_n": int(diff.sum()),
                    "n_periods": args.long_periods,
                    "on_spikes_per_period": spikes,
                    "frac_identical_to_previous_burst": ident_prev,
                    "frac_identical_to_last_burst": ident_last,
                    "max_abs_state_start_minus_final_state": st_dist,
                    "first_burst_identical_to_all_later_bursts": settled,
                    "wall_s": elL,
                }
                npz[f"D_{key}_rows"] = rows
                npz[f"D_{key}_on_counts"] = np.stack(chL["on_counts"])
                print(f"  D {key}: {rows.size} voxels x {args.long_periods} periods, "
                      f"settled at burst {settled} ({elL:.1f} s)", flush=True)

        summary.setdefault("C_scenario_info", {})[scen] = sc_sum
        summary["timing_s"][f"C_{scen}_reference_windows"] = time.perf_counter() - t0

    summary["timing_s"]["total"] = time.perf_counter() - t_start
    (out_dir / "burst_periodicity.json").write_text(json.dumps(summary, indent=1))
    np.savez_compressed(out_dir / "burst_periodicity.npz", **npz)
    print(f"[periodicity] wrote {out_dir/'burst_periodicity.json'} "
          f"({summary['timing_s']['total']:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
