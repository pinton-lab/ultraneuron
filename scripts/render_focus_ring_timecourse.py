"""Render the focus -> ring time course (JNE revision, R2-M8).

Reads ``<sim_dir>/focus_ring/focus_ring_timecourse_raw.npz`` written by
``scripts/run_focus_ring_timecourse.py`` and writes

* ``<sim_dir>/focus_ring/focus_ring_timecourse.json`` (all numbers),
* ``<sim_dir>/focus_ring/fig_focus_ring_timecourse.{png,pdf}``,
* ``data/focus_ring_timecourse_numbers.tex`` (macros, prefix ``FR``).

Regions (as ``render_focus_ring.py``): core = T >= T_1/2,i = 4.5 mN/m;
shell = voxels firing under S2 in the 20 ms burst from rest with T < 4.5;
all = core + shell (the S2 firing set; every core voxel fires under S2).
"Retained" = S4 spikes / S2 spikes of the same region and window.

Usage: python scripts/render_focus_ring_timecourse.py --config validation_report/configs/halle_dacc.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT)); sys.path.insert(0, str(HERE))
from run_neural_sim import _load_config, _resolve_sim_dir  # noqa: E402

T_HALF_I = 4.5
C_CORE, C_SHELL = "#eb6834", "#2a78d6"
SUBWIN = [(0, 5), (5, 10), (10, 15), (15, 20), (0, 10), (10, 20), (0, 20)]


def trains(off, t):
    return [t[off[i]:off[i + 1]] for i in range(len(off) - 1)]


def counts(tr, a, b):
    """Spikes with a < t <= b (a spike time is the END of the step that crossed;
    t = 20.000 ms belongs to the ON window, as in the canonical spike counts)."""
    return np.array([int(((x > a) & (x <= b)).sum()) for x in tr])


def region_stats(c2, c4, regions):
    """Per-region statistics of two count vectors (S2, S4) over one window."""
    out = {}
    for name, m in regions.items():
        n = int(m.sum()); s2, s4 = int(c2[m].sum()), int(c4[m].sum())
        out[name] = {"n_vox": n,
                     "S2": {"spikes": s2, "mean_per_vox": s2 / n, "median_per_vox": float(np.median(c2[m])),
                            "frac_silent": float((c2[m] == 0).mean())},
                     "S4": {"spikes": s4, "mean_per_vox": s4 / n, "median_per_vox": float(np.median(c4[m])),
                            "frac_silent": float((c4[m] == 0).mean())},
                     "S4_retained_vs_S2": (s4 / s2) if s2 else None}
    return out


def last_spike_summary(tr, m):
    ls = np.array([x[-1] if len(x) else np.nan for x in tr])[m]
    fired = np.isfinite(ls)
    q = np.nanpercentile(ls, [10, 25, 50, 75, 90, 100]) if fired.any() else [np.nan] * 6
    return {"n_fired": int(fired.sum()), "pct_ms": dict(zip(["p10", "p25", "p50", "p75", "p90", "max"],
                                                            [float(v) for v in q]))}, ls


def rate(tr, m, edges):
    """Population spike rate per voxel (Hz) of the voxels in ``m``."""
    t = np.concatenate([x for x, k in zip(tr, m) if k]) if m.any() else np.zeros(0)
    h, _ = np.histogram(t, edges)
    return h / (m.sum() * np.diff(edges) * 1e-3)


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="validation_report/configs/halle_dacc.py")
    args = ap.parse_args()
    cfg = _load_config(args.config); sim_dir = _resolve_sim_dir(cfg); out_dir = sim_dir / "focus_ring"
    z = np.load(out_dir / "focus_ring_timecourse_raw.npz")
    meta = json.loads(str(z["meta"]))
    T = z["T_mNm"]; t_on = float(z["t_on_ms"]); period = float(z["period_ms"]); L = float(z["long_ms"])
    nb = int(z["n_bursts"])
    single = {sc: trains(z[f"single_{sc}_off"], z[f"single_{sc}_t"]) for sc in ("S2", "S4")}
    c20 = {sc: counts(single[sc], 0, t_on) for sc in single}
    # consistency with render_focus_ring.py maps
    maps = np.load(out_dir / "focus_ring_maps.npz"); vi = z["voxel_idx"]
    for sc in ("S2", "S4"):
        ref = maps[f"spikes_{sc}"][vi[:, 0], vi[:, 1], vi[:, 2]]
        assert np.array_equal(ref, c20[sc]) and int(ref.sum()) == int(maps[f"spikes_{sc}"].sum()), sc
    core = T >= T_HALF_I
    shell = (c20["S2"] > 0) & ~core
    allf = core | shell
    outside = ~allf
    regions = {"core": core, "shell": shell, "all": allf}
    res: dict = {"description": "Time course of the focus->ring prediction (S2 vs S4), canonical halle_dacc field",
                 "meta": meta, "T_half_i_mNm": T_HALF_I, "n_sim_voxels": int(T.size),
                 "regions": {k: int(v.sum()) for k, v in regions.items()},
                 "shell_T_range_mNm": [float(T[shell].min()), float(T[shell].max())],
                 "check_counts_match_focus_ring_maps": True}

    # ---------------- 1. single burst from rest ---------------------------
    s1 = {}
    for a, b in SUBWIN:
        s1[f"{a}-{b}ms"] = region_stats(counts(single["S2"], a, b), counts(single["S4"], a, b), regions)
    lst = {}
    for sc in ("S2", "S4"):
        for rn in ("core", "shell"):
            lst[f"{sc}_{rn}"], _ = last_spike_summary(single[sc], regions[rn])
    # fraction of core voxels with any spike at or after t (S4)
    tgrid = np.arange(0, t_on + 0.01, 0.5)
    act_after = {}
    for sc in ("S2", "S4"):
        for rn in ("core", "shell"):
            ls = np.array([x[-1] if len(x) else -1.0 for x in single[sc]])[regions[rn]]
            act_after[f"{sc}_{rn}"] = [float((ls >= t).mean()) for t in tgrid]
    t_core_quiet = {f"frac{int(100*f)}": float(tgrid[np.argmax(np.array(act_after["S4_core"]) <= 1 - f)])
                    if (np.array(act_after["S4_core"]) <= 1 - f).any() else None for f in (0.5, 0.9, 0.99, 1.0)}
    res["single_burst"] = {"subwindows": s1, "last_spike_time": lst, "t_grid_ms": tgrid.tolist(),
                           "frac_active_at_or_after_t": act_after,
                           "S4_core_time_after_which_frac_silent_ms": t_core_quiet}

    # ---------------- 2. chained bursts ----------------------------------
    bursts = {}
    for sc in ("S2", "S4"):
        off, t = z[f"bursts_{sc}_off"], z[f"bursts_{sc}_t"]
        base = 0; bursts[sc] = []
        for b in range(nb):
            o = off[b] + base; base = int(off[b][-1]) + base
            bursts[sc].append(trains(o, t))
    per_burst = []
    for b in range(nb):
        e = {"burst": b + 1}
        for lab, (a, bb) in {"on_0-20ms": (0, t_on), "on_0-10ms": (0, 10), "on_10-20ms": (10, t_on),
                             "off_20-200ms": (t_on, period), "period_0-200ms": (0, period)}.items():
            e[lab] = region_stats(counts(bursts["S2"][b], a, bb), counts(bursts["S4"][b], a, bb), regions)
        c2 = counts(bursts["S2"][b], 0, t_on); c4 = counts(bursts["S4"][b], 0, t_on)
        e["core_shell_ratio_on"] = {sc: float(c[core].mean() / c[shell].mean()) for sc, c in (("S2", c2), ("S4", c4))}
        e["new_firing_outside_on"] = {sc: int((c[outside] > 0).sum()) for sc, c in (("S2", c2), ("S4", c4))}
        e["S4_identical_to_previous"] = (None if b == 0 else float(np.mean(
            [np.array_equal(x, y) for x, y in zip(bursts["S4"][b], bursts["S4"][b - 1])])))
        per_burst.append(e)
    res["bursts"] = {"protocol": f"{t_on:.0f} ms ON / {period - t_on:.0f} ms OFF, exp OFF relaxation tau "
                                 f"{float(z['tau_relax_ms']):.2f} ms, state carried over, burst 1 from rest",
                     "per_burst": per_burst}
    reb = np.concatenate([x[x > t_on] for x, k in zip(bursts["S4"][-1], core) if k])
    res["bursts"]["S4_core_off_rebound_last_burst"] = {
        "n_spikes": int(reb.size), "frac_core_vox_firing": float(np.mean([(x > t_on).any() for x, k in zip(bursts["S4"][-1], core) if k])),
        "time_from_onset_ms_pct": dict(zip(["p0", "p10", "p50", "p90", "p100"], np.percentile(reb, [0, 10, 50, 90, 100]).tolist()))}

    # ---------------- 3. long ON windows ---------------------------------
    long_ = {sc: trains(z[f"long_{sc}_off"], z[f"long_{sc}_t"]) for sc in ("S2", "S4")}
    lw = {}
    for Lw in (50.0, 100.0, 200.0):
        if Lw > L:
            continue
        e = {"late_15ms_to_end": region_stats(counts(long_["S2"], 15, Lw), counts(long_["S4"], 15, Lw), regions),
             "whole": region_stats(counts(long_["S2"], 0, Lw), counts(long_["S4"], 0, Lw), regions),
             "last_20ms": region_stats(counts(long_["S2"], Lw - 20, Lw), counts(long_["S4"], Lw - 20, Lw), regions),
             "new_firing_outside": {sc: int((counts(long_[sc], 0, Lw)[outside] > 0).sum()) for sc in long_}}
        lw[f"{int(Lw)}ms"] = e
    long_last = {f"{sc}_{rn}": last_spike_summary(long_[sc], regions[rn])[0] for sc in long_ for rn in ("core", "shell")}
    res["long_on"] = {"last_spike_time": long_last, "note": "Single ON window at constant peak tension from rest; OUTSIDE the Yaakub protocol "
                              "(20 ms ON). 50/100 ms windows are prefixes of the 200 ms run.", "windows": lw}
    (out_dir / "focus_ring_timecourse.json").write_text(json.dumps(res, indent=1))

    # ---------------- figure ---------------------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    edges = np.arange(0, 60.01, 2.0); ctr = 0.5 * (edges[1:] + edges[:-1])

    def rate_panel(a, tr2, tr4, title, xmax=60):
        for rn, col in (("core", C_CORE), ("shell", C_SHELL)):
            a.plot(ctr, rate(tr2, regions[rn], edges), color=col, lw=1.2, ls="--", label=f"{rn}, S2")
            a.plot(ctr, rate(tr4, regions[rn], edges), color=col, lw=2.0, label=f"{rn}, S4")
        a.axvspan(0, t_on, color="0.92", zorder=0); a.text(t_on / 2, 0.97, "ON", transform=a.get_xaxis_transform(),
                                                           ha="center", va="top", fontsize=8, color="0.4")
        a.set_xlim(0, xmax); a.set_xlabel("time from burst onset (ms)"); a.set_ylabel("spike rate per voxel (Hz)")
        a.set_title(title, fontsize=10, loc="left"); a.grid(alpha=0.25, lw=0.5)
    rate_panel(ax[0, 0], bursts["S2"][0], bursts["S4"][0], "(a) burst 1 (from rest), 2 ms bins")
    ax[0, 0].legend(fontsize=8, frameon=False)
    rate_panel(ax[0, 1], bursts["S2"][-1], bursts["S4"][-1], f"(b) burst {nb} (periodic steady state)")
    bi = np.arange(1, nb + 1)
    a = ax[1, 0]
    for rn, col in (("core", C_CORE), ("shell", C_SHELL)):
        a.plot(bi, [100 * e["on_0-20ms"][rn]["S4_retained_vs_S2"] for e in per_burst], color=col, lw=2.0, marker="o",
               ms=5, label=f"{rn}, ON 0-20 ms")
        a.plot(bi, [100 * e["on_10-20ms"][rn]["S4_retained_vs_S2"] for e in per_burst], color=col, lw=1.2, ls=":",
               marker="s", ms=4, label=f"{rn}, ON 10-20 ms")
        a.plot(bi, [100 * e["period_0-200ms"][rn]["S4_retained_vs_S2"] for e in per_burst], color=col, lw=1.2,
               ls="--", marker="^", ms=4, label=f"{rn}, period incl. OFF")
    a.set_xlabel("burst"); a.set_ylabel("S4 spikes / S2 spikes (%)"); a.set_xticks(bi)
    a.set_ylim(0, 105); a.legend(fontsize=7, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 0.45), ncol=2); a.grid(alpha=0.25, lw=0.5)
    a.set_title("(c) spikes retained under inactivation, per burst", fontsize=10, loc="left")
    a = ax[1, 1]; e2 = np.arange(0, L + 0.01, 5.0); c2_ = 0.5 * (e2[1:] + e2[:-1])
    for rn, col in (("core", C_CORE), ("shell", C_SHELL)):
        a.plot(c2_, rate(long_["S2"], regions[rn], e2), color=col, lw=1.2, ls="--", label=f"{rn}, S2")
        a.plot(c2_, rate(long_["S4"], regions[rn], e2), color=col, lw=2.0, label=f"{rn}, S4")
    a.axvline(t_on, color="0.5", lw=0.8, ls=":"); a.text(t_on + 2, 0.97, "20 ms (protocol ON)",
                                                          transform=a.get_xaxis_transform(), fontsize=8, va="top", color="0.4")
    a.set_xlim(0, L); a.set_xlabel("time from onset (ms)"); a.set_ylabel("spike rate per voxel (Hz)")
    a.set_title(f"(d) single {L:.0f} ms ON from rest (outside protocol)", fontsize=10, loc="left")
    a.grid(alpha=0.25, lw=0.5)
    for p in (out_dir / "fig_focus_ring_timecourse.png", out_dir / "fig_focus_ring_timecourse.pdf"):
        fig.savefig(p, dpi=170); print("wrote", p)

    # ---------------- macros ---------------------------------------------
    sb = res["single_burst"]["subwindows"]; ss = per_burst[-1]; l200 = lw.get("200ms", {}); l100 = lw.get("100ms", {})
    pct = lambda x: f"{100 * x:.0f}"  # noqa: E731
    one = lambda x: f"{x:.1f}"  # noqa: E731
    mac = {
        "FRcoreSilentFirstHalf": pct(sb["0-10ms"]["core"]["S4"]["frac_silent"]),
        "FRcoreSilentLateHalf": pct(sb["10-20ms"]["core"]["S4"]["frac_silent"]),
        "FRcoreSilentLastQuarter": pct(sb["15-20ms"]["core"]["S4"]["frac_silent"]),
        "FRcoreSilentLateHalfSTwo": pct(sb["10-20ms"]["core"]["S2"]["frac_silent"]),
        "FRshellSilentLateHalf": pct(sb["10-20ms"]["shell"]["S4"]["frac_silent"]),
        "FRcoreRetainedFirstHalf": pct(sb["0-10ms"]["core"]["S4_retained_vs_S2"]),
        "FRcoreRetainedLateHalf": pct(sb["10-20ms"]["core"]["S4_retained_vs_S2"]),
        "FRshellRetainedFirstHalf": pct(sb["0-10ms"]["shell"]["S4_retained_vs_S2"]),
        "FRshellRetainedLateHalf": pct(sb["10-20ms"]["shell"]["S4_retained_vs_S2"]),
        "FRcoreRetainedLastQuarter": pct(sb["15-20ms"]["core"]["S4_retained_vs_S2"]),
        "FRshellRetainedLastQuarter": pct(sb["15-20ms"]["shell"]["S4_retained_vs_S2"]),
        "FRcoreLastSpikeMedian": one(lst["S4_core"]["pct_ms"]["p50"]),
        "FRcoreLastSpikeNinety": one(lst["S4_core"]["pct_ms"]["p90"]),
        "FRshellLastSpikeMedian": one(lst["S4_shell"]["pct_ms"]["p50"]),
        "FRcoreQuietNinety": one(t_core_quiet["frac90"]) if t_core_quiet["frac90"] is not None else "--",
        "FRcoreSilentSteadyBurst": pct(ss["on_0-20ms"]["core"]["S4"]["frac_silent"]),
        "FRcoreSilentSteadyLateHalf": pct(ss["on_10-20ms"]["core"]["S4"]["frac_silent"]),
        "FRcoreRetainedSteady": pct(ss["on_0-20ms"]["core"]["S4_retained_vs_S2"]),
        "FRshellRetainedSteady": pct(ss["on_0-20ms"]["shell"]["S4_retained_vs_S2"]),
        "FRcoreShellRatioSFourSteady": f"{ss['core_shell_ratio_on']['S4']:.2f}",
        "FRcoreShellRatioSTwoSteady": f"{ss['core_shell_ratio_on']['S2']:.2f}",
        "FRnBursts": str(nb),
        "FRcoreReboundMedian": one(res["bursts"]["S4_core_off_rebound_last_burst"]["time_from_onset_ms_pct"]["p50"] - t_on),
        "FRcoreSilentSteadyOff": pct(ss["off_20-200ms"]["core"]["S4"]["frac_silent"]),
        "FRcoreRetainedSteadyPeriod": pct(ss["period_0-200ms"]["core"]["S4_retained_vs_S2"]),
        "FRshellRetainedSteadyPeriod": pct(ss["period_0-200ms"]["shell"]["S4_retained_vs_S2"]),
    }
    if l200:
        mac.update({"FRcoreLastSpikeLongMax": one(long_last["S4_core"]["pct_ms"]["max"]),
                    "FRcoreSilentLongLastTwenty": pct(l200["last_20ms"]["core"]["S4"]["frac_silent"]),
                    "FRshellRetainedLongLastTwenty": pct(l200["last_20ms"]["shell"]["S4_retained_vs_S2"]),"FRcoreSilentLongTwoHundred": pct(l200["late_15ms_to_end"]["core"]["S4"]["frac_silent"]),
                    "FRcoreRetainedLongTwoHundred": pct(l200["late_15ms_to_end"]["core"]["S4_retained_vs_S2"]),
                    "FRshellRetainedLongTwoHundred": pct(l200["late_15ms_to_end"]["shell"]["S4_retained_vs_S2"])})
    if l100:
        mac.update({"FRcoreSilentLongHundred": pct(l100["late_15ms_to_end"]["core"]["S4"]["frac_silent"]),
                    "FRcoreRetainedLongHundred": pct(l100["late_15ms_to_end"]["core"]["S4_retained_vs_S2"]),
                    "FRshellRetainedLongHundred": pct(l100["late_15ms_to_end"]["shell"]["S4_retained_vs_S2"])})
    (REPO_ROOT / "data/focus_ring_timecourse_numbers.tex").write_text(
        "% generated by scripts/render_focus_ring_timecourse.py\n"
        + "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in mac.items()) + "\n")
    print(json.dumps(mac, indent=1))


if __name__ == "__main__":
    main()
