"""Grid-convergence check for the shear-FDTD + firing-map stages (JNE revision, R2-6).

The canonical pipeline runs the shear solver and the neural map on the 4x-decimated
acoustic grid (1.232 mm).  Here the same radiation-force field is (i) trilinearly
interpolated to a 2x finer grid (0.616 mm) and (ii) sub-sampled to a 2x coarser
grid (2.464 mm, stride 2, the sensor's decimation semantics); the shear solver is run to the end of the 20 ms ON window on each
grid with identical physics, the equivalent strain is formed with the same central
differences, and the canonical S2 neuron is run on each strain field.  Reported per
grid: peak eps_eq, peak tension, firing voxels, firing volume (mm^3), iso-25 % and
iso-50 % volumes, spike count.  This tests the discretisation of the elastic and
neural stages given the ARF; the acoustic decimation itself is discussed in the text.

Uses the JAX shear solver on the GPU (shearwave.shear_fdtd_jax).
Usage: python scripts/run_shear_convergence.py --config validation_report/configs/halle_dacc.py
                                                [--factors 0.5,2] [--resume]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT)); sys.path.insert(0, str(HERE))

from ultraneuron import simulate_map_3comp  # noqa: E402
from ultraneuron.cuda_backend import resolve_backend  # noqa: E402
from run_neural_sim import _build_neuron, _load_config, _resolve_sim_dir, _strain_field_from_displacement  # noqa: E402

E_YOUNG, NU, ETA, CFL, VEL_DAMP = 13.74e3, 0.499999, 0.5, 0.1, 0.02
DT_REF_S = 3.323664639882993e-05   # canonical shear step at 1.232 mm; vel_damp is per step, so it is rescaled to keep the e-folding time fixed
K_A, ALPHA, T_THR, DT_MS = 0.25, 1000.0, 0.5, 0.001


def resample(field: np.ndarray, factor: float, order: int) -> np.ndarray:
    """factor 2 -> 2x finer grid of size 2n-1 per axis (original samples preserved on even
    indices, trilinear in between, via map_coordinates); factor 0.5 -> 2x coarser by strided
    sub-sampling (offset 0), the same decimation semantics as the acoustic sensor's mod_x."""
    from scipy.ndimage import map_coordinates
    if factor == 1.0:
        return field
    if factor > 1:
        f = int(round(factor))
        axes = [np.linspace(0, n - 1, (n - 1) * f + 1) for n in field.shape]
        grids = np.meshgrid(*axes, indexing="ij")
        coords = np.stack([g.ravel() for g in grids], axis=0)
        out = map_coordinates(field, coords, order=order, mode="nearest").reshape(grids[0].shape)
        return out
    f = int(round(1.0 / factor))
    return field[::f, ::f, ::f]


def run_shear(b0: np.ndarray, dx: float, rho0: float, mu: float, t_end: float, env_period: float, env_on: float):
    from shearwave import shear_fdtd_jax
    cs = math.sqrt(mu / rho0)
    dt_el = CFL * dx / (cs * math.sqrt(3.0)); dt_visc = rho0 * dx ** 2 / (2.0 * 3 * ETA)
    dt = min(dt_el, 0.5 * dt_visc)
    n_steps = int(math.ceil(t_end / dt))
    t = np.arange(n_steps) * dt
    env = np.where((t % env_period) <= env_on, 1.0, 0.0).astype(np.float32)
    t0 = time.perf_counter()
    vel_damp = 1.0 - (1.0 - VEL_DAMP) ** (dt / DT_REF_S)     # same e-folding time (~1.7 ms) on every grid
    u, v = shear_fdtd_jax(bx=b0.astype(np.float32), by=None, bz=None, rho=rho0, mu=mu, dX=dx, dY=dx, dZ=dx,
                          dT=dt, n_steps=n_steps, opts={"vel_damp": vel_damp, "eta": ETA, "env_t": env, "verbose": False})
    return np.asarray(u, dtype=np.float32), dict(dt_s=dt, n_steps=n_steps, elapsed_s=time.perf_counter() - t0, vel_damp=vel_damp)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--factors", default="1,0.5,2")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    cfg = _load_config(args.config); sim_dir = _resolve_sim_dir(cfg)
    out_dir = sim_dir / "grid_convergence"; out_dir.mkdir(exist_ok=True)
    seq = getattr(cfg, "NEUROMOD_SEQUENCE", {}) or {}
    t_on = float(seq.get("arf_push_window_s", 0.02)); t_period = float(seq.get("pulse_period_s", 0.2))
    grid = np.load(sim_dir / "grid_params.npz"); dx0 = float(grid["dx"]) * int(grid["mod_x"])
    b0 = np.load(sim_dir / "radiation_force.npy").astype(np.float64)
    c_ds = np.load(sim_dir / "c_ds.npy"); rho_ds = np.load(sim_dir / "rho_ds.npy")
    mu = E_YOUNG / (2.0 * (1.0 + NU)); rho0 = float(np.median(rho_ds))
    backend = resolve_backend("auto"); neuron = _build_neuron(cfg, "S2")
    out_json = out_dir / "grid_convergence.json"
    res = json.loads(out_json.read_text()) if (args.resume and out_json.exists()) else {}
    print(f"[grid_convergence] dx0={dx0*1e3:.3f} mm  mu={mu:.0f} Pa rho0={rho0:.0f}  backend={backend}", flush=True)
    for fac in [float(x) for x in args.factors.split(",")]:
        key = f"factor_{fac:g}"
        if key in res:
            print("cached", key, res[key], flush=True); continue
        dx = dx0 / fac
        b = resample(b0, fac, order=1)
        skull = resample((c_ds > 1600).astype(np.float64), fac, order=0) > 0.5
        b = np.ascontiguousarray(b); skull = np.ascontiguousarray(skull)
        b[skull] = 0.0
        print(f"  {key}: grid {b.shape} dx={dx*1e3:.3f} mm  ARF peak {b.max():.1f} (canonical {b0.max():.1f})", flush=True)
        u, info = run_shear(b, dx, rho0, mu, t_end=t_on, env_period=t_period, env_on=t_on)
        eps = _strain_field_from_displacement(u, dx, dx, dx); eps[skull] = 0.0
        t0 = time.perf_counter()
        r = simulate_map_3comp(eps, neuron, K_A=K_A, coupling_factor=ALPHA, dt=DT_MS, duration=t_on * 1e3,
                               tension_threshold=T_THR, backend=backend)
        fire = r["firing_map"].astype(bool); cnt = r["spike_count_map"].astype(np.float64)
        rate = cnt / (t_on); vox = (dx * 1e3) ** 3; rmax = rate.max()
        res[key] = dict(factor=fac, dx_mm=dx * 1e3, grid=list(b.shape), n_voxels=int(b.size), voxel_mm3=vox,
                        shear_dt_us=info["dt_s"] * 1e6, shear_steps=info["n_steps"], shear_elapsed_s=info["elapsed_s"], vel_damp=info["vel_damp"],
                        mean_spikes_per_firing_voxel=float(cnt.sum() / max(fire.sum(), 1)),
                        peak_ux_um=float(np.abs(u[..., 0]).max() * 1e6), peak_eps=float(eps.max()),
                        peak_tension_mNm=float(r["tension_3d"].max()),
                        n_firing_voxels=int(fire.sum()), total_spikes=int(cnt.sum()), firing_volume_mm3=float(fire.sum() * vox),
                        iso25_volume_mm3=float((rate >= 0.25 * rmax).sum() * vox) if rmax > 0 else 0.0,
                        iso50_volume_mm3=float((rate >= 0.5 * rmax).sum() * vox) if rmax > 0 else 0.0,
                        neural_elapsed_s=time.perf_counter() - t0)
        np.savez_compressed(out_dir / f"strain_{key}.npz", eps=eps.astype(np.float32), dx_m=dx,
                            spike_count_map=r["spike_count_map"].astype(np.int16))
        out_json.write_text(json.dumps(res, indent=1))
        print("  ->", json.dumps({k: v for k, v in res[key].items() if k not in ("grid",)}), flush=True)
    print("[grid_convergence] done", flush=True)


if __name__ == "__main__":
    main()
