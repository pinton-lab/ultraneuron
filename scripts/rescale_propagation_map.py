"""Linearly rescale propagation_map.npy so the in-brain peak matches a target.

After the FDTD run for halle_dacc (or any halle_* config), the simulated
in-brain focal PNP usually doesn't land exactly on the paper's reported
value. Because acoustic propagation is linear, we can scale the entire
pressure history by a single factor to hit a target peak in-brain.

The mask convention matches recompute_arf.py: c > 1600 m/s is skull;
everything beyond the first contiguous skull run from the bowl side
counts as brain.

Default target: 0.5 MPa, matching Yaakub 2023's reported transcranial PNP.
Overwrites propagation_map.npy in place (downstream stages then pick up
the rescaled field automatically). Records the scale factor and observed
peaks to rescale_provenance.json for traceability.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def brain_mask_from_c(c: np.ndarray, skull_threshold_mps: float = 1600.0) -> np.ndarray:
    """Replicates recompute_arf.py's brain-mask convention.

    Treats c > skull_threshold as skull; brain = ~skull, then zeros out
    everything from the bowl-side face through the first contiguous skull
    run (so the coupling medium and the outer skull table both count as
    'not brain').
    """
    skull_mask = c > skull_threshold_mps
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
    return brain_mask


def main(sim_dir: Path, target_pa: float, dry_run: bool = False) -> None:
    pmap_path = sim_dir / "propagation_map.npy"
    c_path = sim_dir / "c_ds.npy"
    if not pmap_path.exists():
        raise FileNotFoundError(pmap_path)
    if not c_path.exists():
        raise FileNotFoundError(c_path)

    pmap = np.load(pmap_path)
    c = np.load(c_path)
    print(f"pmap shape {pmap.shape} dtype {pmap.dtype}")
    print(f"c    shape {c.shape}")

    # pmap is (nt, nx, ny, nz). Reduce time first so the mask broadcast is cheap.
    p_abs_max_over_t = np.abs(pmap).max(axis=0)  # (nx, ny, nz)
    print(f"|p|_max over time: shape {p_abs_max_over_t.shape}, "
          f"global max {p_abs_max_over_t.max():.3e} Pa")

    bmask = brain_mask_from_c(c)
    print(f"brain mask: {bmask.sum():,} voxels of {bmask.size:,} "
          f"({100 * bmask.sum() / bmask.size:.1f}%)")

    in_brain_peak = float(p_abs_max_over_t[bmask].max())
    print(f"in-brain |p|_max = {in_brain_peak:.4e} Pa "
          f"= {in_brain_peak * 1e-6:.4f} MPa")

    scale = target_pa / in_brain_peak
    print(f"target = {target_pa * 1e-6:.3f} MPa  =>  scale = {scale:.4f}")

    rescaled_peak = in_brain_peak * scale
    print(f"after rescale, in-brain peak = {rescaled_peak * 1e-6:.4f} MPa "
          f"(target was {target_pa * 1e-6:.4f} MPa)")

    provenance = {
        "target_in_brain_peak_Pa": target_pa,
        "observed_in_brain_peak_Pa_before_rescale": in_brain_peak,
        "scale_factor_applied": scale,
        "rescaled_in_brain_peak_Pa": rescaled_peak,
        "global_pmap_peak_Pa_before_rescale": float(p_abs_max_over_t.max()),
        "global_pmap_peak_Pa_after_rescale": float(p_abs_max_over_t.max() * scale),
        "brain_mask_voxels": int(bmask.sum()),
        "brain_mask_total_voxels": int(bmask.size),
        "skull_threshold_mps": 1600.0,
    }

    if dry_run:
        print()
        print("--- DRY RUN: not modifying propagation_map.npy ---")
        print(json.dumps(provenance, indent=2))
        return

    pmap *= np.float32(scale)
    np.save(pmap_path, pmap)
    print(f"\nWrote rescaled propagation_map.npy to {pmap_path}")

    (sim_dir / "rescale_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(f"Wrote rescale provenance to {sim_dir / 'rescale_provenance.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sim_dir", type=Path,
                    help="Directory containing propagation_map.npy and c_ds.npy")
    ap.add_argument("--target-mpa", type=float, default=0.5,
                    help="Target in-brain peak |p| in MPa (default: 0.5)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Compute the scale factor without writing files")
    args = ap.parse_args()
    main(args.sim_dir, args.target_mpa * 1e6, dry_run=args.dry_run)
