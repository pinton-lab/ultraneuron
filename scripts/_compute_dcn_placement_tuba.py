"""Compute a TUBA-constrained placement for the left dentate (DCN) and
emit it in the LEGACY halle_placement dict format that
bowl_transcranial_500khz.py consumes.

Run with the tuba venv:
    <tuba-venv>/bin/python scripts/_compute_dcn_placement_tuba.py

Rationale: the repo's own halle_placement.place_on_skull has no anatomical
guards and auto-picks a sub-Frankfurt suboccipital patch (transducer in the
neck, ~2 mm on-axis bone) for the dentate. TUBA's place_on_skull restricts
to the cranial vault (above Frankfurt), runs a beam air-check, and applies a
depth penalty so the focus actually reaches the target. For the dentate the
best valid window is a far-lateral retromastoid aim (perp residual ~13.7 mm).

Coordinate mapping (verified by reproducing the legacy unconstrained result
bit-for-bit): legacy NRRD-voxel-mm positions == TUBA's *_nrrd_voxel_mm keys;
legacy direction vectors == TUBA's RAS vectors with the x-component negated.
"""
import json
from pathlib import Path

from tuba.species.human import place_on_skull

DENTATE_MNI_RAS = (-15.0, -57.0, -33.0)
OUT = Path(__file__).resolve().parent.parent / "data" / "halle_dcn" / "placement_tuba_legacy.json"


def _negx(v):
    return [-float(v[0]), float(v[1]), float(v[2])]


def _normal_2d(beam3):
    import math
    nx, ny = beam3[0], beam3[1]
    n = math.hypot(nx, ny)
    return [nx / n, ny / n]


def _tangent_2d(beam3):
    n2 = _normal_2d(beam3)
    return [-n2[1], n2[0]]


def main():
    p = place_on_skull(
        target_mni_ras=DENTATE_MNI_RAS,
        focal_length_mm=63.0,
        bowl_radius_mm=32.0,
        restrict_to_cranial_vault=True,
        enable_air_check=True,
        depth_penalty_weight=1.0,
        max_apex_to_target_mm=72.0,
        verbose=True,
    )
    legacy = {
        "xdc_center_lps": list(p["xdc_center_lps_nrrd_voxel_mm"]),
        "scalp_contact_lps": list(p["scalp_contact_lps_nrrd_voxel_mm"]),
        "focus_lps": list(p["focus_lps_nrrd_voxel_mm"]),
        "target_lps": list(p["target_lps_nrrd_voxel_mm"]),
        "normal_3d": _negx(p["normal_3d"]),
        "tangent_3d": _negx(p["tangent_3d"]),
        "bitangent_3d": _negx(p["bitangent_3d"]),
        "beam_dir_3d": _negx(p["beam_dir_3d"]),
        # Vestigial 2.5D fields the report builder still reads: the x-y
        # projection of the beam direction (the build step overwrites these
        # after re-aiming the beam at the target).
        "normal": _normal_2d(_negx(p["beam_dir_3d"])),
        "tangent": _tangent_2d(_negx(p["beam_dir_3d"])),
        "z_mm": float(p["xdc_center_lps_nrrd_voxel_mm"][2]),
        "perp_residual_mm": float(p["perp_residual_mm"]),
        "scalp_target_dist_mm": float(p["scalp_target_dist_mm"]),
        "apex_target_dist_mm": float(p["apex_target_dist_mm"]),
        "bowl_depth_mm": float(p["bowl_depth_mm"]),
        "focal_length_mm": 63.0,
        "bowl_radius_mm": 32.0,
        "aim_focus_at_target": True,
        "patch_normal_used": True,
        "eeg_seed_lps": None,
        "eeg_to_placement_mm": None,
        "target_name": "dentate_left",
        "_provenance": "tuba.species.human.place_on_skull "
                       "restrict_to_cranial_vault=True enable_air_check=True "
                       "depth_penalty_weight=1.0 max_apex_to_target_mm=72.0",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(legacy, indent=2))
    print(f"\nWrote legacy-format placement to {OUT}")
    for k in ("xdc_center_lps", "scalp_contact_lps", "focus_lps", "target_lps",
              "normal_3d", "beam_dir_3d", "perp_residual_mm", "apex_target_dist_mm"):
        print(f"  {k}: {legacy[k]}")


if __name__ == "__main__":
    main()
