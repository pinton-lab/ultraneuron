"""Smoke variant of `ctx500_placement_3d.py` re-targeting the left
caudate (where the F3-perpendicular CTX-500 focus actually lands).

The transducer placement is unchanged from the dlPFC version: bowl is
positioned perpendicular to the skull at F3, F=63 mm, R=32 mm. Only
the labelled "target" is swapped — the focus point and bowl pose
remain identical to the dlPFC placement, so this figure shows the
SAME geometry annotated with the structure the focus actually hits.

Caudate ROI is rendered from the Harvard-Oxford SUBCORTICAL atlas via
a runtime patch to `visualize_placement_3d.build_and_warp_target_roi`.
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import nibabel as nib

from ultraneuron.registration import ensure_registration_importable

REG_DIR = str(ensure_registration_importable())

import visualize_placement_3d as viz
import halle_placement as hp
from halle_placement import place_on_skull

OUT_DIR = (Path(__file__).resolve().parent.parent
           / "data" / "halle_transcranial")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "fig_ctx500_placement_3d_caudate.png"

F_MM = 63.0
R_MM = 32.0
SCALP = "F3"
TARGET = "Caudate_left"

# Caudate MNI RAS (head of left caudate; matches the focus we computed
# from the F3-perpendicular CTX-500 placement).
CAUDATE_LEFT_MNI = (-12.0, 15.0, 11.0)


def _patch_targets_and_renderer():
    """Add Caudate_left to the upstream MNI target dict and patch the
    renderer's ROI builder to use the subcortical atlas."""
    hp.ANATOMICAL_TARGETS_MNI[TARGET] = CAUDATE_LEFT_MNI
    viz.ANATOMICAL_TARGETS_MNI = hp.ANATOMICAL_TARGETS_MNI

    original_builder = viz.build_and_warp_target_roi

    def caudate_aware_builder(target_name):
        if target_name != TARGET:
            return original_builder(target_name)

        import ants
        from nilearn import datasets

        halle_path = os.path.join(REG_DIR, "halle_ct_1mm.nii.gz")
        fixed = ants.image_read(halle_path)
        fwd_path = os.path.join(REG_DIR, "ants_syn_forward.txt")
        with open(fwd_path) as f:
            fwd = [l.strip() for l in f if l.strip()]

        ho = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-1mm")
        atlas_img = (nib.load(ho.maps)
                     if isinstance(ho.maps, str) else ho.maps)
        atlas_data = atlas_img.get_fdata().astype(np.int32)
        try:
            label_idx = ho.labels.index("Left Caudate")
        except ValueError:
            print("  WARNING: 'Left Caudate' not in subcortical labels; "
                  "falling back to sphere")
            return viz._sphere_roi(np.asarray(CAUDATE_LEFT_MNI),
                                    fixed, fwd, 12.0)

        mask = (atlas_data == label_idx).astype(np.float32)
        print(f"  Left Caudate (subcortical): "
              f"{int(mask.sum())} MNI voxels")

        with tempfile.NamedTemporaryFile(suffix=".nii.gz",
                                          delete=False) as tf:
            nib.save(nib.Nifti1Image(mask, atlas_img.affine), tf.name)
            moving = ants.image_read(tf.name)
            warped = ants.apply_transforms(
                fixed=fixed, moving=moving, transformlist=fwd,
                interpolator="nearestNeighbor",
            )
            os.unlink(tf.name)
        out = warped.numpy() > 0.5
        print(f"  warped ROI: {int(out.sum())} voxels in Halle space")
        return out

    viz.build_and_warp_target_roi = caudate_aware_builder


def main() -> None:
    _patch_targets_and_renderer()
    p = place_on_skull(
        scalp_site=SCALP,
        target_name=TARGET,
        focal_length_mm=F_MM,
        bowl_radius_mm=R_MM,
    )
    p["target_name"] = TARGET
    print("Placement (caudate target, F3 perpendicular):")
    for k, v in p.items():
        print(f"  {k}: {v}")

    viz.OUT_PATH = str(OUT_PATH)
    viz.render(p)


if __name__ == "__main__":
    main()
