"""Model-form check for the strain -> membrane-tension step (JNE revision, R2-1).

The submitted manuscript drives the membrane tension with the von Mises
equivalent strain eps_eq.  Referee 2 asked for the firing map to be recomputed
with (i) the maximum principal tensile strain and (ii) an orientation-resolved
areal (bilayer area-dilation) measure.  This script recomputes the canonical
Halle/dACC S2 firing map with four scalar strain measures built from the same
end-of-pulse displacement field (``displacement.npy`` = frame at t = 20 ms):

    vm      von Mises equivalent strain            (as submitted; control)
    e1      maximum principal strain, max(lambda_1, 0)
    eAmax   maximum areal dilation over membrane orientation,
            max_n [tr(eps) - n.eps.n] = tr(eps) - lambda_3 = lambda_1 + lambda_2 (+ tr terms)
    eAavg   orientation-averaged positive-part areal strain,
            < max(tr(eps) - n.eps.n, 0) >_n over the unit sphere
            (randomly oriented membranes; Fibonacci-sphere quadrature)

For every measure two neural runs are made:

    raw      same K_A * alpha as the canonical run (model-form spread as-is)
    matched  measure rescaled so its focal (brain) peak equals the vm peak,
             isolating the *shape* difference from the overall scale, which is
             what K_A * alpha would absorb in a calibration.

Outputs (under <sim_dir>/strain_variants/):
    strain_measures.npz            the four 3-D measure fields + eigenvalues + tr(eps)
    neural_<measure>_<mode>.npz    per-run firing / spike / rate / tension maps
    summary.json                   headline numbers per run + Dice vs control

Usage:
    python scripts/run_strain_measure_variants.py --config validation_report/configs/halle_dacc.py
    python scripts/run_strain_measure_variants.py --config ... --measures-only   # fast diagnostics
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
sys.path.insert(0, str(HERE))

from ultraneuron import simulate_map_3comp_with_spike_times  # noqa: E402
from run_neural_sim import (  # noqa: E402
    _DEFAULTS, _build_neuron, _load_config, _resolve_sim_dir,
    _strain_field_from_displacement,
)
from rescale_propagation_map import brain_mask_from_c  # noqa: E402

MEASURES = ("vm", "e1", "eAmax", "eAavg")


def strain_tensor(disp: np.ndarray, dx: float, dy: float, dz: float) -> np.ndarray:
    """Infinitesimal strain tensor, shape (..., 3, 3), same finite differences as
    run_neural_sim._strain_field_from_displacement."""
    ux, uy, uz = disp[..., 0], disp[..., 1], disp[..., 2]
    duxdx, duxdy, duxdz = np.gradient(ux, dx, dy, dz, edge_order=1)
    duydx, duydy, duydz = np.gradient(uy, dx, dy, dz, edge_order=1)
    duzdx, duzdy, duzdz = np.gradient(uz, dx, dy, dz, edge_order=1)
    e = np.empty(disp.shape[:-1] + (3, 3), dtype=np.float64)
    e[..., 0, 0] = duxdx; e[..., 1, 1] = duydy; e[..., 2, 2] = duzdz
    e[..., 0, 1] = e[..., 1, 0] = 0.5 * (duxdy + duydx)
    e[..., 0, 2] = e[..., 2, 0] = 0.5 * (duxdz + duzdx)
    e[..., 1, 2] = e[..., 2, 1] = 0.5 * (duydz + duzdy)
    return e


def fibonacci_sphere(m: int) -> np.ndarray:
    i = np.arange(m) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / m)
    theta = np.pi * (1.0 + 5 ** 0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi),
                     np.sin(theta) * np.sin(phi),
                     np.cos(phi)], axis=1)  # (m, 3)


def areal_measures(lam: np.ndarray, mask: np.ndarray, n_dir: int = 600,
                   chunk: int = 20000) -> tuple[np.ndarray, np.ndarray]:
    """lam: (..., 3) principal strains ascending.  Returns (eAmax, eAavg) fields.

    In the principal frame n.eps.n = sum_i lam_i n_i^2, so the areal strain of a
    membrane patch with normal n is eps_A(n) = tr(eps) - sum_i lam_i n_i^2.
    """
    tr = lam.sum(axis=-1)
    eAmax = np.clip(tr - lam[..., 0], 0.0, None)          # normal along most compressive axis
    eAavg = np.zeros_like(tr)
    n2 = fibonacci_sphere(n_dir) ** 2                       # (m, 3)
    idx = np.flatnonzero(mask.ravel())
    lam_f = lam.reshape(-1, 3); tr_f = tr.ravel(); out = eAavg.ravel()
    for s in range(0, idx.size, chunk):
        ii = idx[s:s + chunk]
        nn = lam_f[ii] @ n2.T                                # (chunk, m)
        ea = tr_f[ii][:, None] - nn
        out[ii] = np.clip(ea, 0.0, None).mean(axis=1)
    return eAmax, out.reshape(tr.shape)


def build_measures(sim_dir: Path) -> dict:
    disp = np.load(sim_dir / "displacement.npy").astype(np.float32)
    c_map = np.load(sim_dir / "c_ds.npy").astype(np.float32)
    grid = np.load(sim_dir / "grid_params.npz")
    dx = float(grid["dx"]) * int(grid["mod_x"])
    dy = float(grid["dy"]) * int(grid["mod_y"])
    dz = float(grid["dz"]) * int(grid["mod_z"])
    # Eligible voxels: the pipeline's brain mask (recompute_arf.py /
    # rescale_propagation_map.py convention).  c <= 1600 m/s alone also admits
    # the water coupling medium between the bowl and the skull (c = 1540 m/s
    # there as in brain), where the raw-tensor areal measure picks up a
    # domain-edge finite-difference residual at x = 0-1 and fires spuriously.
    skull = c_map > 1600.0
    brain = brain_mask_from_c(c_map)

    vm = _strain_field_from_displacement(disp, dx, dy, dz).astype(np.float64)
    e = strain_tensor(disp, dx, dy, dz)
    lam = np.linalg.eigvalsh(e)                              # ascending (..., 3)
    tr = lam.sum(axis=-1)
    e1 = np.clip(lam[..., 2], 0.0, None)
    eAmax, eAavg = areal_measures(lam, brain)
    # Deviatoric-enforced variants: the reduced shear solver is solenoidal in
    # the continuum (K >> mu makes the physical volumetric strain ~1e-5 of the
    # shear strain), so the trace of the discrete field is a numerical residual.
    # Remove it before forming the tensile / areal measures.
    lam_d = lam - tr[..., None] / 3.0
    e1d = np.clip(lam_d[..., 2], 0.0, None)
    eAmaxd, eAavgd = areal_measures(lam_d, brain)

    # cross-check: von Mises from eigenvalues must equal the finite-difference version
    dev = lam - tr[..., None] / 3.0
    vm_eig = np.sqrt((2.0 / 3.0) * (dev ** 2).sum(axis=-1))
    assert np.allclose(vm_eig[brain], vm[brain], rtol=1e-4, atol=1e-12), "von Mises mismatch"

    fields = {"vm": vm, "e1": e1, "eAmax": eAmax, "eAavg": eAavg,
              "e1d": e1d, "eAmaxd": eAmaxd, "eAavgd": eAavgd}
    for k in fields:
        fields[k] = np.where(brain, fields[k], 0.0).astype(np.float32)

    # diagnostics
    vmax = float(fields["vm"].max())
    hot = brain & (fields["vm"] > 0.25 * vmax)              # focal region for ratio stats
    diag = {
        "dx_mm": dx * 1e3,
        "n_brain_voxels": int(brain.sum()),
        "n_coupling_water_voxels_excluded": int((~skull & ~brain).sum()),
        "vm_peak": vmax,
        "trace_abs_max_over_vm_peak": float(np.abs(tr[brain]).max() / vmax),
        "trace_rms_over_vm_rms_focal": float(np.sqrt((tr[hot] ** 2).mean()) / np.sqrt((vm[hot] ** 2).mean())),
        "n_focal_voxels_ratio_stats": int(hot.sum()),
    }
    for k in ("e1", "eAmax", "eAavg", "e1d", "eAmaxd", "eAavgd"):
        r = fields[k][hot] / fields["vm"][hot]
        diag[f"{k}_over_vm_focal_median"] = float(np.median(r))
        diag[f"{k}_over_vm_focal_p5_p95"] = [float(np.percentile(r, 5)), float(np.percentile(r, 95))]
        diag[f"{k}_peak"] = float(fields[k].max())
        diag[f"{k}_peak_over_vm_peak"] = float(fields[k].max() / vmax)
    return {"fields": fields, "lam": lam.astype(np.float32), "tr": tr.astype(np.float32),
            "brain": brain, "skull": skull, "diag": diag, "dx": dx}


def run_neural(field: np.ndarray, neuron, args, duration_ms: float) -> dict:
    t0 = time.perf_counter()
    res = simulate_map_3comp_with_spike_times(
        field, neuron, K_A=args.K_A, coupling_factor=args.coupling_factor,
        dt=args.dt_ms, duration=duration_ms, tension_threshold=args.threshold_mNm,
        integrator="rk4",
    )
    res["elapsed_s"] = time.perf_counter() - t0
    return res


SKULL_SHELL: np.ndarray | None = None   # brain voxels within 2 voxels of skull (set in main)


def headline(res: dict, duration_ms: float, vox_mm3: float, ref_fire: np.ndarray | None) -> dict:
    fire = res["firing_map"].astype(bool)
    rate = res["spike_count_map"].astype(np.float64) / (duration_ms * 1e-3)
    out = {
        "n_firing_voxels": int(fire.sum()),
        "total_spikes": int(res["spike_count_map"].sum()),
        "peak_rate_hz": float(rate.max()),
        "firing_volume_mm3": float(fire.sum() * vox_mm3),
        "iso50_volume_mm3": float((rate >= 0.5 * rate.max()).sum() * vox_mm3) if rate.max() > 0 else 0.0,
        "iso25_volume_mm3": float((rate >= 0.25 * rate.max()).sum() * vox_mm3) if rate.max() > 0 else 0.0,
        "peak_tension_mNm": float(res["tension_3d"].max()),
        "n_voxels_above_threshold": int((res["tension_3d"] >= 0.5).sum()),
        "elapsed_s": float(res["elapsed_s"]),
    }
    if SKULL_SHELL is not None:
        out["n_firing_in_skull_shell"] = int(np.logical_and(fire, SKULL_SHELL).sum())
        if ref_fire is not None:
            deep = ~SKULL_SHELL
            a, b = fire & deep, ref_fire & deep
            out["dice_vs_control_excl_shell"] = float(2 * (a & b).sum() / max(a.sum() + b.sum(), 1))
    if ref_fire is not None:
        inter = np.logical_and(fire, ref_fire).sum()
        out["dice_vs_control"] = float(2 * inter / max(fire.sum() + ref_fire.sum(), 1))
        out["jaccard_vs_control"] = float(inter / max(np.logical_or(fire, ref_fire).sum(), 1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--sim-dir", default=None)
    ap.add_argument("--measures-only", action="store_true")
    ap.add_argument("--K-A", type=float, default=_DEFAULTS["K_A"])
    ap.add_argument("--alpha", type=float, dest="coupling_factor", default=_DEFAULTS["coupling_factor"])
    ap.add_argument("--threshold-mNm", type=float, default=_DEFAULTS["tension_threshold"])
    ap.add_argument("--dt-ms", type=float, default=_DEFAULTS["dt_ms"])
    ap.add_argument("--modes", default="raw,matched")
    ap.add_argument("--measures", default="e1,eAmax,eAavg,e1d,eAmaxd,eAavgd")
    ap.add_argument("--resume", action="store_true", help="keep runs already in summary.json")
    args = ap.parse_args()

    cfg = _load_config(args.config)
    sim_dir = Path(args.sim_dir).resolve() if args.sim_dir else _resolve_sim_dir(cfg)
    out_dir = sim_dir / "strain_variants"; out_dir.mkdir(exist_ok=True)
    seq = getattr(cfg, "NEUROMOD_SEQUENCE", {}) or {}
    duration_ms = float(seq.get("arf_push_window_s", seq.get("tone_burst_duration_s", 0.02))) * 1e3
    print(f"[strain_variants] sim_dir={sim_dir}  window={duration_ms:.1f} ms", flush=True)

    m = build_measures(sim_dir)
    np.savez_compressed(out_dir / "strain_measures.npz", **m["fields"], lam=m["lam"], tr=m["tr"],
                        brain=m["brain"], dx_m=np.float64(m["dx"]))
    print(json.dumps(m["diag"], indent=1), flush=True)
    if args.measures_only:
        return

    global SKULL_SHELL
    from scipy.ndimage import binary_dilation
    SKULL_SHELL = binary_dilation(m["skull"], iterations=2) & m["brain"]

    neuron = _build_neuron(cfg, scenario="S2")
    vox_mm3 = (m["dx"] * 1e3) ** 3
    vm_peak = float(m["fields"]["vm"].max())
    summary = {"diagnostics": m["diag"], "runs": {}, "config": {
        "K_A": args.K_A, "alpha": args.coupling_factor, "threshold_mNm": args.threshold_mNm,
        "dt_ms": args.dt_ms, "duration_ms": duration_ms, "scenario": "S2",
        "piezo1_g": float(neuron.mechano_channels[0].g_density),
        "voxel_mm3": vox_mm3}}

    prev = json.loads((out_dir / "summary.json").read_text())["runs"] if (args.resume and (out_dir / "summary.json").exists()) else {}
    # control
    if "vm_raw" in prev and (out_dir / "neural_vm_raw.npz").exists():
        ref_fire = np.load(out_dir / "neural_vm_raw.npz")["firing_map"].astype(bool)
        summary["runs"]["vm_raw"] = prev["vm_raw"]
    else:
        res = run_neural(m["fields"]["vm"], neuron, args, duration_ms)
        ref_fire = res["firing_map"].astype(bool)
        summary["runs"]["vm_raw"] = headline(res, duration_ms, vox_mm3, None)
        np.savez_compressed(out_dir / "neural_vm_raw.npz", firing_map=res["firing_map"],
                            spike_count_map=res["spike_count_map"], latency_map=res["latency_map"].astype(np.float32),
                            tension_3d=res["tension_3d"].astype(np.float32))
    summary["runs"]["vm_raw"]["n_firing_in_skull_shell"] = int((ref_fire & SKULL_SHELL).sum())
    print("vm_raw", json.dumps(summary["runs"]["vm_raw"]), flush=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))

    modes = args.modes.split(",")
    for meas in args.measures.split(","):
        for mode in modes:
            key = f"{meas}_{mode}"
            if key in prev and (out_dir / f"neural_{key}.npz").exists():
                summary["runs"][key] = prev[key]
                d = np.load(out_dir / f"neural_{key}.npz"); fire = d["firing_map"].astype(bool)
                summary["runs"][key]["n_firing_in_skull_shell"] = int((fire & SKULL_SHELL).sum())
                a, b = fire & ~SKULL_SHELL, ref_fire & ~SKULL_SHELL
                summary["runs"][key]["dice_vs_control_excl_shell"] = float(2 * (a & b).sum() / max(a.sum() + b.sum(), 1))
                print(key, "(cached)", flush=True)
                (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
                continue
            f = m["fields"][meas].astype(np.float32)
            scale = 1.0
            if mode == "matched":
                scale = vm_peak / float(f.max()); f = (f * scale).astype(np.float32)
            res = run_neural(f, neuron, args, duration_ms)
            summary["runs"][key] = headline(res, duration_ms, vox_mm3, ref_fire)
            summary["runs"][key]["scale_applied"] = scale
            np.savez_compressed(out_dir / f"neural_{key}.npz", firing_map=res["firing_map"],
                                spike_count_map=res["spike_count_map"],
                                latency_map=res["latency_map"].astype(np.float32),
                                tension_3d=res["tension_3d"].astype(np.float32))
            print(key, json.dumps(summary["runs"][key]), flush=True)
            (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print("[strain_variants] done", flush=True)


if __name__ == "__main__":
    main()
