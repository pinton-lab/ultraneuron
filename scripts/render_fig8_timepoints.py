"""Re-render fig8 panel (b) as a 5x5 grid: 5 timepoints (rows) x
5 scenarios (cols).  Runs only the canonical ε working point to capture
CSR spike times per voxel for each scenario, then builds cumulative
spike-count snapshots on the focal y-z plane at 5 within-burst times.
"""
from __future__ import annotations

import os
import sys
import time
from copy import deepcopy
from pathlib import Path

import matplotlib

# Compute backend for the per-voxel HH integration.  Override at the
# shell with AMN_BACKEND=numba (CPU) or AMN_BACKEND=cuda; default 'auto' = GPU when visible.
_BACKEND = "auto"  # resolved by ultraneuron.cuda_backend.resolve_backend (AMN_BACKEND env override)

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))

from ultraneuron import (  # noqa: E402
    PIEZO1, TREK1, TRPV1, TRPV4,
    CalciumDynamics, KCaChannel,
    MulticompartmentMechanoHHNeuron, NICEModel,
    simulate_map, simulate_map_3comp_with_spike_times,
)
from ultraneuron.neuron import MechanoHHNeuron  # noqa: E402
from ultraneuron.channels import strain_to_membrane_tension  # noqa: E402

# Reuse the sweep script's strain loader + scenario builders to keep
# everything aligned.
sys.path.insert(0, str(HERE))
from run_halle_mechanism_sweep import (  # noqa: E402
    _load_real_strain, _build_multicomp,
    _G_PIEZO1, _G_TREK1, _K_A, _COUPLING_FACTOR, DURATION_MS, DT_MS,
)

OUT_FIG = REPO_ROOT / "figures" / "fig8_halle_mechanism_comparison"
SWEEP_NPZ = REPO_ROOT / "data" / "halle_mechanism_sweep_results.npz"
CUM_CACHE_NPZ = REPO_ROOT / "data" / "fig8_cumulative_snapshots.npz"

TARGET_EPS_FOR_PANEL = 2.0e-5            # canonical working point
# Five evenly-spaced within-burst times spanning the ON window
# (DURATION_MS tracks the protocol: Yaakub 20 ms → [4,8,12,16,20]).
SNAPSHOT_TIMES_MS = [round(DURATION_MS * k / 5.0, 1) for k in range(1, 6)]
SCENARIOS = ["S1", "S2", "S3", "S4", "S5"]


def _run_with_spike_times(scenario: str, strain: np.ndarray) -> dict:
    """Return {firing_map, spike_count_map, spike_voxel_idx, spike_offsets,
    spike_times_ms} for a single scenario."""
    if scenario == "S1":
        ch_p = deepcopy(PIEZO1); ch_p.g_density = _G_PIEZO1
        ch_t = deepcopy(TREK1);  ch_t.g_density = _G_TREK1
        n = MechanoHHNeuron.cortical_pyramidal(temperature=37.0,
                                                mechano_channels=[ch_p, ch_t])
        # 1-comp: simulate_map returns counts only; we approximate spike
        # times by recording at uniform spacing across the burst.  S1
        # contributes only ~68 spikes at this ε so the approximation is
        # acceptable for the visualisation.
        from ultraneuron import simulate_map
        r = simulate_map(strain, n, dt=DT_MS, duration=DURATION_MS,
                         tension_threshold=0.5,
                         K_A=_K_A, coupling_factor=_COUPLING_FACTOR)
        nx, ny, nz = strain.shape
        sc = r["spike_count_map"]
        # Convert per-voxel spike count → uniform spike times
        idxs, sts, offs = [], [], [0]
        for ix, iy, iz in zip(*np.where(sc > 0)):
            n_sp = int(sc[ix, iy, iz])
            ts = np.linspace(DURATION_MS / max(n_sp + 1, 1),
                             DURATION_MS, n_sp, endpoint=False)
            idxs.append((int(ix), int(iy), int(iz)))
            sts.extend(ts.tolist())
            offs.append(offs[-1] + n_sp)
        return {
            "firing_map": r["firing_map"],
            "spike_count_map": sc,
            "spike_voxel_idx": np.array(idxs, dtype=np.int32) if idxs
                                else np.zeros((0, 3), dtype=np.int32),
            "spike_offsets": np.array(offs, dtype=np.int64),
            "spike_times_ms": np.array(sts, dtype=np.float32),
        }

    n_kw = dict(
        with_nice=(scenario in ("S3", "S5")),
        with_inact_ca=(scenario in ("S4", "S5")),
        with_trp=(scenario == "S5"),
    )
    n = _build_multicomp(**n_kw)
    r = simulate_map_3comp_with_spike_times(
        strain, n, dt=DT_MS, duration=DURATION_MS,
        tension_threshold=0.5,
        K_A=_K_A, coupling_factor=_COUPLING_FACTOR,
        backend=_BACKEND,
    )
    return r


def _cumulative_spikes_yz(r: dict, x_slice: int, ny: int, nz: int,
                          t_ms: float) -> np.ndarray:
    out = np.zeros((ny, nz), dtype=np.int32)
    sv = r["spike_voxel_idx"]
    so = r["spike_offsets"]
    st = r["spike_times_ms"]
    for k in range(sv.shape[0]):
        ix, iy, iz = int(sv[k, 0]), int(sv[k, 1]), int(sv[k, 2])
        if ix != x_slice:
            continue
        ts = st[so[k]:so[k + 1]]
        out[iy, iz] = int((ts <= t_ms).sum())
    return out


def main() -> None:
    if CUM_CACHE_NPZ.exists():
        print(f"[fig8-tp] loading cumulative snapshot cache from {CUM_CACHE_NPZ}",
              flush=True)
        cache = np.load(CUM_CACHE_NPZ)
        cum = cache["cum"]
        ext = tuple(cache["ext"].tolist())
        peak_idx = tuple(int(x) for x in cache["peak_idx"])
    else:
        print("[fig8-tp] no cache; running scenarios...", flush=True)
        eeq, _, grid = _load_real_strain()
        real_peak = float(eeq.max())
        scale = TARGET_EPS_FOR_PANEL / real_peak
        strain = (eeq * scale).astype(np.float32)
        print(f"  scaled to peak ε = {TARGET_EPS_FOR_PANEL:.2e} "
              f"(scale={scale:.2f})", flush=True)
        results: dict[str, dict] = {}
        for s in SCENARIOS:
            t0 = time.perf_counter()
            results[s] = _run_with_spike_times(s, strain)
            nfire = int(results[s]["firing_map"].sum())
            nsp = int(results[s]["spike_count_map"].sum())
            print(f"  {s}: {nfire} firing voxels, {nsp} spikes "
                  f"({time.perf_counter()-t0:.1f}s)", flush=True)
        ref = results["S2"]["spike_count_map"]
        peak_idx = (np.unravel_index(int(np.argmax(ref)), ref.shape)
                    if ref.max() > 0 else tuple(s // 2 for s in ref.shape))
        ix = peak_idx[0]
        nx, ny, nz = ref.shape
        dy = grid["dy"] * 1e3
        dz = grid["dz"] * 1e3
        ext = (0, nz * dz, 0, ny * dy)
        cum = np.zeros((len(SNAPSHOT_TIMES_MS), len(SCENARIOS), ny, nz),
                       dtype=np.float32)
        for ti, t_ms in enumerate(SNAPSHOT_TIMES_MS):
            for si, s in enumerate(SCENARIOS):
                cum[ti, si] = _cumulative_spikes_yz(
                    results[s], ix, ny, nz, t_ms,
                )
        np.savez_compressed(CUM_CACHE_NPZ, cum=cum,
                            ext=np.array(ext, dtype=np.float64),
                            peak_idx=np.array(peak_idx, dtype=np.int32))
        print(f"  cached cumulative stack to {CUM_CACHE_NPZ}", flush=True)

    ny, nz = cum.shape[2], cum.shape[3]
    vmax = max(int(cum.max()), 1)

    # Reload sweep npz for panel (a)
    d = np.load(SWEEP_NPZ, allow_pickle=True)
    target_eps = d["target_eps"]
    spike_totals = d["spike_totals"]
    sweep_scenarios = d["scenarios"]
    n_eps = len(target_eps)

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    # Compose figure: panel (a) on the left (wider), 5×5 panel (b) on the right.
    # Panel (a) is 2× the width it had previously (width_ratio 4.4 vs the
    # earlier 2.2).  Figure height set so that panel (b)'s 5×5 cells are
    # roughly square at print and don't carry extra inter-row whitespace
    # from imshow's aspect="equal" constraint: each panel-b column is
    # ~1.56"/5 wide → height ≈ 5×1.56 + headers/xlabels ≈ 9.5".
    fig = plt.figure(figsize=(22, 8.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[4.4, 2.4], wspace=0.12)

    ax = fig.add_subplot(gs[0, 0])
    for i, key in enumerate(sweep_scenarios):
        ax.plot(target_eps, spike_totals[i], marker="o", lw=2.4, ms=9,
                color=colors[i], label=str(key))
    ax.axvline(TARGET_EPS_FOR_PANEL, color="0.5", lw=1.2, ls="--",
               label=f"panel (b): ε={TARGET_EPS_FOR_PANEL:.0e}")
    ax.set_xlabel(r"Peak deviatoric strain $\varepsilon_{\mathrm{eq}}^{\max}$",
                  fontsize=16)
    ax.set_ylabel(f"Total spike count ({DURATION_MS:.0f} ms ON window)", fontsize=16)
    ax.tick_params(labelsize=14)
    ax.set_xscale("log")
    # Symlog with linthresh=1 keeps the 0 baseline legible while showing
    # several decades above it; clip the negative half so the axis only
    # shows non-negative spike counts.
    ax.set_yscale("symlog", linthresh=1)
    ax.set_ylim(0, float(spike_totals.max()) * 1.5)
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper left", fontsize=13, frameon=False)

    inner = gs[0, 1].subgridspec(
        len(SNAPSHOT_TIMES_MS), len(SCENARIOS), wspace=0.12, hspace=0.0,
    )
    im = None
    for ti, t_ms in enumerate(SNAPSHOT_TIMES_MS):
        for si, s in enumerate(SCENARIOS):
            ax = fig.add_subplot(inner[ti, si])
            sl = cum[ti, si].T
            im = ax.imshow(sl, cmap="hot", vmin=0, vmax=vmax,
                           extent=ext, origin="lower", aspect="equal")
            if ti == 0:
                ax.set_title(s, fontsize=17, fontweight="bold")
            if si == 0:
                ax.set_ylabel(f"$t = {t_ms:.0f}$ ms\ny, lat. (mm)",
                              fontsize=13)
            else:
                ax.set_yticks([])
            if ti == len(SNAPSHOT_TIMES_MS) - 1:
                ax.set_xlabel("z (mm)", fontsize=13)
                ax.set_xticks([0, 25, 50, 75])
            else:
                ax.set_xticks([])
            ax.tick_params(labelsize=11)
            # Per-panel cumulative-spike total at this time
            n_at_t = int(cum[ti, si].sum())
            ax.text(0.97, 0.97, f"{n_at_t}",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=12, color="white",
                    bbox=dict(facecolor="black", alpha=0.4, pad=3.0,
                              edgecolor="none"))

    cbar = fig.colorbar(im,
                        ax=[fig.axes[1 + i] for i in range(len(SNAPSHOT_TIMES_MS) * len(SCENARIOS))],
                        location="right", shrink=0.85, pad=0.02)
    cbar.set_label("Cumulative spikes / voxel", fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    # Suptitle deliberately omitted — descriptive content lives in the
    # manuscript figure caption.
    fig.savefig(f"{OUT_FIG}.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{OUT_FIG}.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig8-tp] wrote {OUT_FIG}.{{png,pdf}}", flush=True)


if __name__ == "__main__":
    main()
