"""Placement-3D rendering for the canonical Yaakub/Halle dACC re-aim.

CTX-500 bowl (F=63 mm, R=32 mm) with auto-pick placement on the outer-skull
mesh (scalp_site=None) aimed at the left anterior cingulate (ACC_left, MNI
RAS = (-7, 35, 20)). The Harvard-Oxford cortical anterior-cingulate label
is warped to subject space for the target ROI.
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
           / "data" / "halle_dacc")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "fig_ctx500_placement_3d_dacc.png"
FIG2_PATH = (Path(__file__).resolve().parent.parent
             / "figures" / "fig2_placement_3d.png")
FIG2_PATH.parent.mkdir(parents=True, exist_ok=True)

F_MM = 63.0
R_MM = 32.0
TARGET = "ACC_left"


def _patch_renderer_for_acc():
    """Patch the renderer's ROI builder to use the Harvard-Oxford cortical
    'Cingulate Gyrus, anterior division' label."""
    original_builder = viz.build_and_warp_target_roi

    def acc_aware_builder(target_name):
        if target_name != TARGET:
            return original_builder(target_name)

        import ants
        from nilearn import datasets

        halle_path = os.path.join(REG_DIR, "halle_ct_1mm.nii.gz")
        fixed = ants.image_read(halle_path)
        fwd_path = os.path.join(REG_DIR, "ants_syn_forward.txt")
        with open(fwd_path) as f:
            fwd = [l.strip() for l in f if l.strip()]

        ho = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-1mm")
        atlas_img = (nib.load(ho.maps)
                     if isinstance(ho.maps, str) else ho.maps)
        atlas_data = atlas_img.get_fdata().astype(np.int32)
        try:
            label_idx = ho.labels.index(
                "Cingulate Gyrus, anterior division"
            )
        except ValueError:
            print("  WARNING: 'Cingulate Gyrus, anterior division' not in "
                  "cortical labels; falling back to sphere")
            return viz._sphere_roi(
                np.asarray(hp.ANATOMICAL_TARGETS_MNI[TARGET]),
                fixed, fwd, 12.0,
            )

        # Restrict to left hemisphere (x < 0 in MNI RAS).
        mask3d = (atlas_data == label_idx)
        # Build per-voxel RAS-x without flattening (avoids any view/copy
        # ambiguity from boolean-indexing on a reshape-view).
        ii, jj, kk = np.indices(mask3d.shape)
        aff = atlas_img.affine
        ras_x = (aff[0, 0] * ii + aff[0, 1] * jj
                 + aff[0, 2] * kk + aff[0, 3])
        left = ras_x < 0
        mask = (mask3d & left).astype(np.float32)
        print(f"  Left anterior cingulate (cortical): "
              f"{int(mask.sum())} MNI voxels "
              f"(of {int(mask3d.sum())} bilateral total)")

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

    viz.build_and_warp_target_roi = acc_aware_builder


def main() -> None:
    _patch_renderer_for_acc()
    p = place_on_skull(
        scalp_site=None,              # auto-pick perpendicular aim
        target_name=TARGET,
        focal_length_mm=F_MM,
        bowl_radius_mm=R_MM,
        use_patch_normal=True,
    )
    p["target_name"] = TARGET
    print("Placement (ACC_left, auto-pick):")
    for k, v in p.items():
        print(f"  {k}: {v}")

    viz.OUT_PATH = str(OUT_PATH)
    viz.render(p)

    # Also drop a copy at figures/fig2_placement_3d.png so manuscript builds
    # pick it up without further wiring.
    import shutil
    if OUT_PATH.exists():
        shutil.copyfile(OUT_PATH, FIG2_PATH)
        print(f"Copied to {FIG2_PATH}")


if __name__ == "__main__":
    main()
