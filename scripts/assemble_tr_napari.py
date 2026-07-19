#!/usr/bin/env python3
"""Assemble the micro-CT time-reversal raw fields (written by
fullwave2_transcranial_aperturegrowth/export_tr_fields_napari.m) into the .npy
layout that scripts/napari_wave_propagation.py expects.

Produces, in data/tr_microct/:
  skull_fullres_c.npy, skull_fullres_meta.json, grid_params.npz,
  propagation_map.npy  (outward frames then inward frames, concatenated),
  phase_info.json      (n_out = #outward frames, for phase labelling).
"""
import json
import pathlib

import numpy as np

D = pathlib.Path(__file__).resolve().parents[1] / "data" / "tr_microct"
meta = json.loads((D / "meta.json").read_text())
nX, nY, nZ = meta["nX"], meta["nY"], meta["nZ"]

# Skull speed map (MATLAB column-major -> Fortran order)
c = np.fromfile(D / "skull_c.f32", dtype="<f4").reshape((nX, nY, nZ), order="F")
np.save(D / "skull_fullres_c.npy", c.astype("float32"))
(D / "skull_fullres_meta.json").write_text(
    json.dumps({"dx_mm": meta["dx_mm"], "dy_mm": meta["dx_mm"], "dz_mm": meta["dx_mm"]})
)
np.savez(
    D / "grid_params.npz",
    mod_x=meta["modX"], mod_y=meta["modY"], mod_z=meta["modZ"],
    dx=meta["dx_m"], sampling_modulus_time=meta["modT"],
)
print(f"skull_fullres_c.npy {c.shape}  dx={meta['dx_mm']:.4f} mm  mod={meta['modX']}")


def load_field(prefix):
    d = json.loads((D / f"{prefix}_dims.json").read_text())
    a = np.fromfile(D / f"{prefix}.f32", dtype="<f4").reshape(
        (d["nX2"], d["nY2"], d["nZ2"], d["nT2"]), order="F"
    )
    return a.transpose(3, 0, 1, 2)  # -> (nt, nx, ny, nz)


phases, labels = [], []
out = load_field("p3d_out")
phases.append(out)
labels.append(("outward (target->source)", out.shape[0]))
n_out = out.shape[0]
print(f"outward field {out.shape}")

if (D / "p3d_in_dims.json").exists():
    inw = load_field("p3d_in")
    phases.append(inw)
    labels.append(("inward (source->target)", inw.shape[0]))
    print(f"inward  field {inw.shape}")
else:
    print("inward field not present yet -> outward-only propagation_map")

prop = np.concatenate(phases, axis=0).astype("float32")
np.save(D / "propagation_map.npy", prop)
(D / "phase_info.json").write_text(
    json.dumps({"n_out": int(n_out), "n_total": int(prop.shape[0]), "labels": labels})
)
# drop stale cumulative cache so napari rebuilds it for the new field
for f in ("cumulative_intensity.npy",):
    p = D / f
    if p.exists():
        p.unlink()
print(f"propagation_map.npy {prop.shape}  (n_out={n_out})")
print("ASSEMBLE DONE")
