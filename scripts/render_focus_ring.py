"""Focus -> ring redistribution under Piezo1 inactivation (JNE revision, R2-M8).

Runs the canonical Halle/dACC field (corrected shear projection) with the canonical
conductances (g_Piezo1 = 0.10, g_TREK1 = 0.04 mS/cm^2, K_A*alpha = 250 N/m) under
S2 (no inactivation) and S4 (Piezo1 inactivation + Ca pool + SK), and renders spikes
per voxel on three orthogonal planes through the focal voxel, with the T_{1/2,i}
contour.  Also writes data/focus_ring_numbers.tex.

Usage: python scripts/render_focus_ring.py --config validation_report/configs/halle_dacc.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT)); sys.path.insert(0, str(HERE))
from ultraneuron import simulate_map_3comp  # noqa: E402
from run_neural_sim import _build_neuron, _load_config, _resolve_sim_dir, _strain_field_from_displacement  # noqa: E402

T_HALF_I = 4.5   # Piezo1 inactivation half-tension, mN/m
T_HALF = 2.7


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = _load_config(args.config); sim_dir = _resolve_sim_dir(cfg)
    out_dir = sim_dir / "focus_ring"; out_dir.mkdir(exist_ok=True)
    seq = getattr(cfg, "NEUROMOD_SEQUENCE", {}) or {}
    dur = float(seq.get("arf_push_window_s", 0.02)) * 1e3
    disp = np.load(sim_dir / "displacement.npy").astype(np.float32); c = np.load(sim_dir / "c_ds.npy")
    grid = np.load(sim_dir / "grid_params.npz"); dx = float(grid["dx"]) * int(grid["mod_x"])
    eps = _strain_field_from_displacement(disp, dx, dx, dx); eps[c > 1600] = 0.0
    K_A, alpha = float(cfg.NEURAL_K_A), float(cfg.NEURAL_COUPLING_FACTOR)
    res = {}
    for sc in ("S2", "S4"):
        n = _build_neuron(cfg, sc); t0 = time.perf_counter()
        r = simulate_map_3comp(eps, n, K_A=K_A, coupling_factor=alpha, dt=0.001, duration=dur, tension_threshold=0.5)
        res[sc] = r; print(sc, int(r["firing_map"].sum()), int(r["spike_count_map"].sum()), f"{time.perf_counter()-t0:.0f}s", flush=True)
    T = res["S2"]["tension_3d"]; pk = np.unravel_index(int(np.argmax(T)), T.shape)
    f2, f4 = res["S2"]["firing_map"].astype(bool), res["S4"]["firing_map"].astype(bool)
    core = T >= T_HALF_I
    stats = dict(
        n_fire_S2=int(f2.sum()), n_fire_S4=int(f4.sum()), spikes_S2=int(res["S2"]["spike_count_map"].sum()),
        spikes_S4=int(res["S4"]["spike_count_map"].sum()),
        n_core_voxels=int(core.sum()), n_core_fire_S2=int((f2 & core).sum()), n_core_fire_S4=int((f4 & core).sum()),
        n_shell_fire_S2=int((f2 & ~core).sum()), n_shell_fire_S4=int((f4 & ~core).sum()),
        n_silenced_by_inact=int((f2 & ~f4).sum()), n_new_under_inact=int((~f2 & f4).sum()),
        T_peak_mNm=float(T.max()), voxel_mm3=(dx * 1e3) ** 3)
    stats["frac_core_silenced"] = 1 - stats["n_core_fire_S4"] / max(stats["n_core_fire_S2"], 1)
    stats["frac_shell_retained"] = stats["n_shell_fire_S4"] / max(stats["n_shell_fire_S2"], 1)
    stats["frac_fire_change"] = stats["n_fire_S4"] / stats["n_fire_S2"] - 1
    (out_dir / "focus_ring_stats.json").write_text(json.dumps(stats, indent=1))
    np.savez_compressed(out_dir / "focus_ring_maps.npz", spikes_S2=res["S2"]["spike_count_map"], spikes_S4=res["S4"]["spike_count_map"],
                        tension=T.astype(np.float32), focal_idx=np.array(pk))
    print(json.dumps(stats, indent=1))

    # ---- figure: 2 rows (S2, S4) x 3 planes, 30 mm window around the focus
    half = int(round(15.0 / (dx * 1e3)))
    vmax = max(int(res[s]["spike_count_map"].max()) for s in ("S2", "S4")) or 1
    fig, axes = plt.subplots(2, 3, figsize=(11, 7.2), constrained_layout=True)
    planes = [("x--y ($z$ = focal)", lambda a: a[:, :, pk[2]], (0, 1)), ("x--z ($y$ = focal)", lambda a: a[:, pk[1], :], (0, 2)),
              ("y--z ($x$ = focal)", lambda a: a[pk[0], :, :], (1, 2))]
    for r_, sc, title in ((0, "S2", "inactivation OFF (S2, canonical)"), (1, "S4", "inactivation ON (S4: Piezo1 inactivation + Ca$^{2+}$/SK)")):
        for c_, (pl, sl, ax_ids) in enumerate(planes):
            ax = axes[r_, c_]
            spk = sl(res[sc]["spike_count_map"]).astype(float); ten = sl(T)
            i0, i1 = pk[ax_ids[0]] - half, pk[ax_ids[0]] + half; j0, j1 = pk[ax_ids[1]] - half, pk[ax_ids[1]] + half
            i0, j0 = max(i0, 0), max(j0, 0)
            sp = np.ma.masked_where(spk[i0:i1, j0:j1] <= 0, spk[i0:i1, j0:j1])
            ext = (j0 * dx * 1e3, j1 * dx * 1e3, i1 * dx * 1e3, i0 * dx * 1e3)
            ax.imshow(np.zeros_like(spk[i0:i1, j0:j1]), cmap="Greys", vmin=0, vmax=1, extent=ext)
            im = ax.imshow(sp, cmap="inferno", vmin=0, vmax=vmax, extent=ext)
            yy, xx = np.mgrid[i0:i1, j0:j1]
            ax.contour(xx * dx * 1e3, yy * dx * 1e3, ten[i0:i1, j0:j1], levels=[T_HALF_I], colors="cyan", linestyles="--", linewidths=1.2)
            ax.contour(xx * dx * 1e3, yy * dx * 1e3, ten[i0:i1, j0:j1], levels=[T_HALF], colors="white", linestyles=":", linewidths=0.9)
            ax.plot(pk[ax_ids[1]] * dx * 1e3, pk[ax_ids[0]] * dx * 1e3, "w+", ms=9, mew=1.2)
            lab = ["x (mm)", "y (mm)", "z (mm)"]
            ax.set_xlabel(lab[ax_ids[1]]); ax.set_ylabel(lab[ax_ids[0]])
            ax.set_title((title + "\n" if c_ == 0 else "") + pl, fontsize=9)
    fig.colorbar(im, ax=axes, shrink=0.6, label="spikes per voxel (20 ms)")
    for p in (REPO_ROOT / "figures/fig_focus_ring.pdf", REPO_ROOT / "figures/fig_focus_ring.png", out_dir / "fig_focus_ring.png"):
        fig.savefig(p, dpi=170); print("wrote", p)
    mac = {"FRfireSTwo": f"{stats['n_fire_S2']:,}".replace(",", "\\,"), "FRfireSFour": f"{stats['n_fire_S4']:,}".replace(",", "\\,"),
           "FRcoreSilenced": f"{100*stats['frac_core_silenced']:.0f}", "FRshellRetained": f"{100*stats['frac_shell_retained']:.0f}",
           "FRfireChange": f"{100*stats['frac_fire_change']:+.0f}", "FRcoreVox": f"{stats['n_core_voxels']:,}".replace(",", "\\,"),
           "FRspikesSTwo": f"{stats['spikes_S2']:,}".replace(",", "\\,"), "FRspikesSFour": f"{stats['spikes_S4']:,}".replace(",", "\\,")}
    (REPO_ROOT / "data/focus_ring_numbers.tex").write_text("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in mac.items()) + "\n")


if __name__ == "__main__":
    main()
