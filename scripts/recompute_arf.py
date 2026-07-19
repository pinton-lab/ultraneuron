"""Re-compute the brain-masked ARF for an existing transcranial run.

Uses the fixed brain-mask logic (mask only down to the inner table of
the FIRST contiguous skull run, not to the deepest skull voxel in the
column).  This corrects the vertical-stripe artifact in
``radiation_force.npy`` for runs that already saved
``propagation_map.npy``.
"""
from pathlib import Path
import sys

import numpy as np

from shearwave import compute_radiation_force

if len(sys.argv) > 1:
    SIM_DIR = Path(sys.argv[1])
else:
    SIM_DIR = Path(__file__).resolve().parent.parent / "data" / "halle_caudate"

print(f"Loading run: {SIM_DIR}")
grid = np.load(SIM_DIR / "grid_params.npz", allow_pickle=True)
n_tic = int(grid["n_tic"])
samp_t = int(grid["sampling_modulus_time"])
cw_corr = float(grid["cw_correction"])
f0 = float(grid["f0"])

pmap = np.load(SIM_DIR / "propagation_map.npy")  # (nt, nx, ny, nz)
c = np.load(SIM_DIR / "c_ds.npy")
rho = np.load(SIM_DIR / "rho_ds.npy")
alpha = np.load(SIM_DIR / "alpha_ds.npy")
print(f"  pmap {pmap.shape}, c/rho/alpha {c.shape}, f0={f0:.0f} Hz")

# Pressure indexing: bowl_transcranial saves (nt, nx, ny, nz) but the
# shearwave kernel expects (nx, ny, nz, nt).
pressure_4d = np.moveaxis(pmap, 0, -1)

n_source_steps = n_tic // samp_t
nt_recorded = pressure_4d.shape[-1]
silence_corr = nt_recorded / n_source_steps
pulse_norm = silence_corr * cw_corr
print(f"  n_tic={n_tic}, n_source_steps={n_source_steps}, "
      f"silence_corr={silence_corr:.2f}, cw_corr={cw_corr:.2f}, "
      f"pulse_norm={pulse_norm:.2f}")

b0 = compute_radiation_force(pressure_4d, c, rho, alpha, f0, gate=slice(None))
print(f"  raw ARF max = {b0.max():.3e} N/m^3 (pre-pulse_norm)")
b0 *= pulse_norm
print(f"  post-pulse_norm ARF max = {b0.max():.3e} N/m^3")

# Corrected brain mask: only mask the coupling medium + outer skull
# layer (the FIRST contiguous skull run from the bowl side).  Do NOT
# extend the mask to the contralateral skull.
skull_mask = c > 1600.0
brain_mask = ~skull_mask
for j in range(c.shape[1]):
    for k in range(c.shape[2]):
        col = skull_mask[:, j, k]
        skull_indices = np.where(col)[0]
        if len(skull_indices) == 0:
            continue
        first = skull_indices[0]
        i = first
        while i + 1 < col.size and col[i + 1]:
            i += 1
        brain_mask[: i + 1, j, k] = False

n_zeroed_buggy = ((~brain_mask) & (b0 > 0)).sum()
b0[~brain_mask] = 0.0
print(f"  voxels zeroed by corrected mask: {(~brain_mask).sum():,} "
      f"(of which {n_zeroed_buggy:,} would have been zero under the "
      f"buggy .max() mask too)")
print(f"  post-mask ARF max = {b0.max():.3e} N/m^3, "
      f"non-zero voxels = {(b0 > 0).sum():,}")

out = SIM_DIR / "radiation_force.npy"
np.save(out, b0.astype(np.float32))
print(f"wrote {out}")
