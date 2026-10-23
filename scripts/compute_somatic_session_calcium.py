"""Session somatic [Ca](x, t) for the canonical halle_dacc run (rest start).

Reads the canonical one-burst CSR spike times in data/halle_dacc/neural_results.npz
(S2, mechano gates from true rest: 3531 firing voxels, 12250 spikes, 20 ms burst),
repeats them at each of the 400 bursts of the Yaakub theta-burst session
(20 ms / 200 ms, 80 s) and evaluates the slow somatic pool
(ultraneuron.calcium.SomaticCalcium defaults: rest 0.05 uM, tau 700 ms,
0.08 uM/spike, clamp 10 uM). With the rest start the periodic train is exact, not an
assumption: every burst starts from rest because the tension relaxes during the 180 ms
OFF time (scripts/run_burst_periodicity.py, chained 200 ms periods with full state
carry-over reproduce the 12250 / 3531 burst exactly). The trace uses the analytic
geometric series; the summary maps are cross-checked against the direct event-driven
recursion over the full 4.90 M-spike session train. Compared with the sister
project's gvmp ch6 numbers (/kulm/scratch/gv_multiphysics_movies/data/ch6_neuromod,
read only) when present (those were computed from the legacy peak-steady spikes, so
the per-voxel comparison is skipped when the firing sets differ).

Headline numbers (2026-10-06, rest start): plateau median 1.17 uM (0.33-1.45),
peak median 1.33 uM (0.37-1.64), return to within 10 % of rest 3.70 s (2.72-3.85);
focal voxel (54, 32, 30), 5 spikes/burst: plateau 1.45 uM, peak 1.64 uM, return 3.85 s.
(Legacy peak-steady: 3590 voxels, plateau 1.17 (0.33-1.73), peak 1.33 (0.37-1.96).)

Writes data/halle_dacc/calcium/somatic_session.{npz,json}; the npz carries the
per-firing-voxel summaries, the 3-D plateau / peak / end / return-time maps on the
neural grid (NaN outside the firing set), the decimated traces and the focal trace.
The legacy peak-steady result is kept as somatic_session_periodic_peak_steady.*.

    python scripts/compute_somatic_session_calcium.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from ultraneuron.calcium import (  # noqa: E402
    PulsedProtocol,
    SomaticCalcium,
    session_somatic_calcium,
)

SIM = REPO / "data" / "halle_dacc"
OUT = SIM / "calcium"
GVMP = Path("/kulm/scratch/gv_multiphysics_movies/data/ch6_neuromod")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    nr = np.load(SIM / "neural_results.npz")
    vidx, off, st = nr["spike_voxel_idx"], nr["spike_offsets"], nr["spike_times_ms"]
    nper = np.diff(off)
    Tpk = nr["T_peak_mNm"]
    assert np.array_equal(nr["spike_count_map"][tuple(vidx.T)], nper)
    proto = PulsedProtocol(
        burst_ms=float(nr["T_pulse_ms"]),
        period_ms=float(nr["group_period_ms"]),
        n_bursts=int(nr["n_groups_session"]),
    )
    pool = SomaticCalcium()
    t = np.arange(-1000.0, proto.session_ms + 10000.0 + 5.0, 10.0)

    t0 = time.time()
    ana = session_somatic_calcium(off, st, proto, pool, t_ms=t, method="auto")
    t_ana = time.time() - t0
    t0 = time.time()
    dirc = session_somatic_calcium(off, st, proto, pool, t_ms=t[-1200:], method="direct")
    t_dir = time.time() - t0
    agree = {
        f: float(np.max(np.abs(getattr(ana, f) - getattr(dirc, f))))
        for f in ("peak", "plateau", "ca_end", "t_return_ms")
    }
    agree["trace_tail_max_abs_uM"] = float(np.max(np.abs(ana.ca[:, -1200:] - dirc.ca)))

    # focal voxel = peak tension among firing voxels
    i_foc = int(np.argmax(Tpk[tuple(vidx.T)]))
    foc = [int(v) for v in vidx[i_foc]]

    def stats(x):
        return {"median": float(np.median(x)), "min": float(x.min()), "max": float(x.max()),
                "p05": float(np.percentile(x, 5)), "p95": float(np.percentile(x, 95))}

    by_n = {
        int(n): {"n_vox": int((nper == n).sum()),
                 "plateau_uM": float(np.median(ana.plateau[nper == n])),
                 "peak_median_uM": float(np.median(ana.peak[nper == n])),
                 "analytic_plateau_uM": pool.Ca_rest + n * pool.dCa_per_spike * pool.tau_ms / proto.period_ms}
        for n in np.unique(nper)
    }

    num = {
        "description": "Slow somatic Ca pool over the 400-burst Yaakub theta-burst session; "
                       "one simulated burst (rest start) repeated, exact because every burst "
                       "starts from rest (run_burst_periodicity.py)",
        "model": {"Ca_rest_uM": pool.Ca_rest, "tau_ms": pool.tau_ms,
                  "dCa_per_spike_uM": pool.dCa_per_spike, "Ca_max_uM": pool.Ca_max},
        "protocol": {"burst_ms": proto.burst_ms, "period_ms": proto.period_ms,
                     "n_bursts": proto.n_bursts, "session_ms": proto.session_ms},
        "n_fire": int(len(vidx)), "n_spikes_burst": int(st.size),
        "spikes_per_voxel_burst": {"mean": float(nper.mean()), "max": int(nper.max()), "min": int(nper.min())},
        "plateau_uM": stats(ana.plateau),
        "peak_uM": stats(ana.peak),
        "ca_end_uM": stats(ana.ca_end),
        "t_return_10pct_s": stats(ana.t_return_ms / 1e3),
        "by_spikes_per_burst": by_n,
        "focal_voxel": foc, "focal_spikes_per_burst": int(nper[i_foc]),
        "focal_T_peak_mNm": float(Tpk[tuple(foc)]),
        "focal_peak_uM": float(ana.peak[i_foc]), "focal_plateau_uM": float(ana.plateau[i_foc]),
        "focal_t_return_s": float(ana.t_return_ms[i_foc] / 1e3),
        "method_trace": ana.method,
        "analytic_vs_direct_max_abs": agree,
        "wall_s": {"analytic": t_ana, "direct_summaries": t_dir},
        "definitions": {
            "plateau": "exact time-mean of [Ca] over the last burst period [79.8, 80.0] s",
            "peak": "exact maximum over the session (just after a spike)",
            "t_return": "time after session end (80 s) at which [Ca]-Ca_rest <= 0.1*Ca_rest",
        },
        "source_spikes": "data/halle_dacc/neural_results.npz",
        "sim_git_sha": str(nr["sim_git_sha"]),
    }
    try:
        num["git_sha"] = subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        pass

    if (GVMP / "numbers.json").exists():
        g = json.loads((GVMP / "numbers.json").read_text())
        cmp_ = {"gvmp_late_mean_median_uM": g["ca_late_mean_median_uM"],
                "gvmp_late_mean_range_uM": [g["ca_late_mean_min_uM"], g["ca_late_mean_max_uM"]],
                "gvmp_peak_median_uM": g["ca_peak_median_uM"], "gvmp_peak_max_uM": g["ca_peak_max_uM"],
                "gvmp_focal_voxel": g["focal_voxel"], "gvmp_slow_peak_focal_uM": g["ca_slow_peak_focal_uM"],
                "gvmp_late_window": "60-80 s grid mean (1 ms Euler-type grid)"}
        if (GVMP / "calcium.npz").exists():
            gz = np.load(GVMP / "calcium.npz")
            if np.array_equal(gz["vidx"], vidx):
                cmp_["per_voxel_max_abs_diff_plateau_uM"] = float(np.max(np.abs(gz["ca_late"] - ana.plateau)))
                cmp_["per_voxel_max_rel_diff_plateau"] = float(np.max(np.abs(gz["ca_late"] / ana.plateau - 1)))
                cmp_["per_voxel_max_abs_diff_peak_uM"] = float(np.max(np.abs(gz["ca_peak"] - ana.peak)))
                cmp_["per_voxel_max_rel_diff_peak"] = float(np.max(np.abs(gz["ca_peak"] / ana.peak - 1)))
        num["gvmp_comparison"] = cmp_

    shape = tuple(int(v) for v in nr["spike_count_map"].shape)

    def vol(x):
        m = np.full(shape, np.nan, dtype=np.float32)
        m[tuple(vidx.T)] = x
        return m

    num["mechano_init"] = "rest"
    num["maps_3d"] = "plateau_map_uM, peak_map_uM, ca_end_map_uM, t_return_map_s (NaN outside firing set)"
    np.savez_compressed(
        OUT / "somatic_session.npz",
        plateau_map_uM=vol(ana.plateau), peak_map_uM=vol(ana.peak),
        ca_end_map_uM=vol(ana.ca_end), t_return_map_s=vol(ana.t_return_ms / 1e3),
        focal_voxel=np.array(foc, dtype=np.int32), mechano_init="rest",
        t_ms=t, ca_uM=ana.ca, spike_voxel_idx=vidx, spikes_per_burst=nper,
        shape=np.array(nr["spike_count_map"].shape),
        peak_uM=ana.peak, plateau_uM=ana.plateau, ca_end_uM=ana.ca_end,
        t_return_ms=ana.t_return_ms, focal_row=i_foc, focal_trace_uM=ana.ca[i_foc],
        Ca_rest_uM=pool.Ca_rest, tau_ms=pool.tau_ms, dCa_per_spike_uM=pool.dCa_per_spike,
        Ca_max_uM=pool.Ca_max, burst_ms=proto.burst_ms, period_ms=proto.period_ms,
        n_bursts=proto.n_bursts,
    )
    (OUT / "somatic_session.json").write_text(json.dumps(num, indent=1))
    print(json.dumps(num, indent=1))


if __name__ == "__main__":
    main()
