"""Re-run only the shear-FDTD step with the corrected ARF input.

Consumes ``radiation_force.npy`` (after ``recompute_arf.py``) and writes
back all downstream artefacts that the validation report and neural
simulation consume:

    displacement.npy
    displacement_peak.npy
    displacement_4d.npy
    displacement_4d_times.npy
    velocity.npy
    snapshots_xy.npy
    snapshots_xz.npy
    disp_trace.npy
    shear_params.npz

Matches the Yaakub-protocol shear-FDTD setup in
``bowl_transcranial_500khz.py`` (1 pulse, 30 ms ON / 170 ms OFF,
Kelvin-Voigt eta = 0.5 Pa.s, mu = 4580 Pa, rho = median of rho_ds).
"""
from pathlib import Path
import math
import sys

import numpy as np

from shearwave import shear_fdtd_staggered

if len(sys.argv) > 1:
    SIM_DIR = Path(sys.argv[1])
else:
    SIM_DIR = Path(__file__).resolve().parent.parent / "data" / "halle_caudate"
print(f"Loading run: {SIM_DIR}")

grid = np.load(SIM_DIR / "grid_params.npz", allow_pickle=True)
mod_x = int(grid["mod_x"]); mod_y = int(grid["mod_y"]); mod_z = int(grid["mod_z"])
dx_full = float(grid["dx"]); dy_full = float(grid["dy"]); dz_full = float(grid["dz"])
dx2 = dx_full * mod_x
dy2 = dy_full * mod_y
dz2 = dz_full * mod_z

c_ds = np.load(SIM_DIR / "c_ds.npy")
rho_ds = np.load(SIM_DIR / "rho_ds.npy")
b0 = np.load(SIM_DIR / "radiation_force.npy")  # corrected ARF
print(f"  b0 max={b0.max():.3e} N/m^3, non-zero={(b0>0).sum():,}")

# Shear material parameters (matches bowl_transcranial_500khz.py)
e_young = 13.74e3
nu_poisson = 0.499999
mu = e_young / (2.0 * (1.0 + nu_poisson))
rho0 = float(np.median(rho_ds))
cs = math.sqrt(mu / rho0)
eta_pa_s = 0.5
print(f"  mu={mu:.1f} Pa, cs={cs:.3f} m/s, rho0={rho0:.1f} kg/m^3, eta={eta_pa_s} Pa.s")

# Time-stepping
cfl = 0.1
dt_elastic = cfl * min(dx2, dy2, dz2) / (cs * math.sqrt(3))
dx_min = min(dx2, dy2, dz2)
dt_viscous = rho0 * dx_min**2 / (2.0 * 3 * eta_pa_s)
dt_shear = min(dt_elastic, 0.5 * dt_viscous)
print(f"  dt_shear={dt_shear:.6e} s")

# Yaakub 1-pulse protocol: ON window from the sim's NEUROMOD_SEQUENCE
# (arf_push_window_s → tone_burst_duration_s) so the shear push matches
# the protocol ON window (Yaakub: 20 ms ON / 180 ms OFF; Legon: 30/170).
def _config_push_window_s(sim_name, default_s=30.0e-3):
    cfg_path = (Path(__file__).resolve().parent.parent
                / "validation_report" / "configs" / f"{sim_name}.py")
    if not cfg_path.exists():
        return default_s
    import importlib.util
    spec = importlib.util.spec_from_file_location(f"_cfg_{sim_name}", cfg_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    seq = getattr(mod, "NEUROMOD_SEQUENCE", {}) or {}
    return float(seq.get("arf_push_window_s",
                         seq.get("tone_burst_duration_s", default_s)))


t_pulse_on = _config_push_window_s(SIM_DIR.name)
t_period = 200.0e-3
print(f"  t_pulse_on = {t_pulse_on*1e3:.1f} ms (from {SIM_DIR.name} config)")
n_steps = math.ceil(t_period / dt_shear)
t = np.arange(n_steps, dtype=np.float64) * dt_shear
env_t = np.where((t % t_period) <= t_pulse_on, 1.0, 0.0).astype(np.float32)
n_on = int(env_t.sum())
print(f"  n_steps={n_steps}, ON={n_on}, DC={100*n_on/n_steps:.1f}%")

# Snapshot cadence
n_frames = 100
save_every = max(1, n_steps // n_frames)

peak_idx = tuple(int(i) for i in np.unravel_index(b0.argmax(), b0.shape))
print(f"  peak ARF voxel: {peak_idx}")

ny_mid_ds = b0.shape[1] // 2
nz_mid_ds = b0.shape[2] // 2

disp_trace = []
snapshots_xy = []
snapshots_xz = []
u_full_4d = []
u_full_times = []
u_peak_mag = np.zeros(b0.shape, dtype=np.float32)


def record(it, u_center):
    mag = np.linalg.norm(u_center, axis=-1)
    np.maximum(u_peak_mag, mag, out=u_peak_mag)
    disp_trace.append(u_center[peak_idx[0], peak_idx[1], peak_idx[2], 0])
    if it % save_every == 0:
        snapshots_xy.append(u_center[:, :, nz_mid_ds, 0].copy())
        snapshots_xz.append(u_center[:, ny_mid_ds, :, 0].copy())
        u_full_4d.append(u_center.astype(np.float32).copy())
        u_full_times.append(it * dt_shear)


print("\nRunning shear-FDTD...")
u, v = shear_fdtd_staggered(
    bx=b0, by=None, bz=None,
    rho=rho0, mu=mu,
    dX=dx2, dY=dy2, dZ=dz2,
    dT=dt_shear, n_steps=n_steps,
    opts={
        "vel_damp": 0.02,
        "eta": eta_pa_s,
        "env_t": env_t,
        "verbose": True,
        "progress_every": max(1, n_steps // 10),
    },
    callback=record,
)

disp_trace = np.array(disp_trace, dtype=np.float64)
print(f"Max |u_x| (final field): {np.abs(u[..., 0]).max() * 1e6:.3f} um")
print(f"Frames recorded: {len(snapshots_xy)}")

u_4d = np.stack(u_full_4d, axis=0)
u_4d_times = np.array(u_full_times, dtype=np.float64)
peak_t_idx = int(np.argmax(np.linalg.norm(u_4d, axis=-1).max(axis=(1, 2, 3))))
print(f"Peak-displacement frame: {peak_t_idx} (t={u_4d_times[peak_t_idx]*1e3:.1f} ms)")

np.save(SIM_DIR / "displacement.npy", u_4d[peak_t_idx])
np.save(SIM_DIR / "displacement_peak.npy", u_peak_mag)
np.save(SIM_DIR / "displacement_4d.npy", u_4d)
np.save(SIM_DIR / "displacement_4d_times.npy", u_4d_times)
np.save(SIM_DIR / "velocity.npy", v)
np.save(SIM_DIR / "snapshots_xy.npy", np.stack(snapshots_xy))
np.save(SIM_DIR / "snapshots_xz.npy", np.stack(snapshots_xz))
np.save(SIM_DIR / "disp_trace.npy", disp_trace)
np.savez(SIM_DIR / "shear_params.npz",
         dt_shear=dt_shear, save_every=save_every, n_steps=n_steps,
         mu=mu, rho0=rho0, cs=cs, eta=eta_pa_s,
         peak_idx=np.array(peak_idx))
# Also write Yaakub envelope params used here for downstream consumers
# that rely on grid_params.npz for the protocol timing.
grid_keys = {k: grid[k] for k in grid.files}
grid_keys["t_pulse_on"] = np.float64(t_pulse_on)
grid_keys["t_period"] = np.float64(t_period)
grid_keys["dt_shear"] = np.float64(dt_shear)
grid_keys["n_steps_shear"] = np.int64(n_steps)
np.savez(SIM_DIR / "grid_params.npz", **grid_keys)

print(f"Saved shear-FDTD outputs to {SIM_DIR}")
