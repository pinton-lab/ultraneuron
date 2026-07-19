"""One-off driver: run bowl_transcranial_500khz.main for the DCN (dentate) re-aim.

The repo's legacy halle_placement.place_on_skull has no anatomical guards and,
for the deep/inferior dentate, auto-picks a sub-Frankfurt suboccipital patch
that puts the transducer in the neck (apex below the scanned volume) with the
central beam grazing only ~2 mm of bone. To avoid that we inject a
TUBA-constrained placement instead: tuba.species.human.place_on_skull with
restrict_to_cranial_vault=True + beam air-check + depth penalty, then re-aimed
at the dentate (bowl tilted ~17 deg off the skull normal; coupling residual
~13.7 mm — the honest cost of hitting a deep cerebellar nucleus with a 63 mm
CTX-500). The placement is precomputed in the tuba venv by
_compute_dcn_placement_tuba.py and re-aimed at target; here we just load it and
monkeypatch place_on_skull so ONLY this driver invocation is affected (legacy
halle_placement.py and the dACC pipeline are untouched).

Outputs land in data/halle_dcn/ for the validation_report.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import bowl_transcranial_500khz as btx  # noqa: E402
from bowl_transcranial_500khz import main  # noqa: E402

from ultraneuron.paths import halle_skull_nrrd  # noqa: E402

PLACEMENT_JSON = REPO_ROOT / "data" / "halle_dcn" / "placement_tuba_legacy.json"


def _tuba_placement(scalp_site=None, target_name=None,
                    focal_length_mm=63.0, bowl_radius_mm=32.0):
    """Drop-in replacement for place_on_skull that returns the precomputed
    TUBA-constrained, dentate-aimed placement (legacy dict format)."""
    if not PLACEMENT_JSON.exists():
        raise FileNotFoundError(
            f"{PLACEMENT_JSON} not found — run\n"
            f"  <tuba-venv>/bin/python "
            f"scripts/_compute_dcn_placement_tuba.py\n"
            f"first (it is then re-aimed at target by the build step).")
    d = json.loads(PLACEMENT_JSON.read_text())
    print(f"[run_halle_dcn_fdtd] using TUBA placement from {PLACEMENT_JSON.name}: "
          f"{d.get('_provenance', '')}")
    print(f"  bowl tilt off skull normal = "
          f"{d.get('bowl_tilt_off_skull_normal_deg')} deg, "
          f"coupling residual = {d.get('perp_residual_mm')} mm")
    return d


# Inject for this process only.
btx.place_on_skull = _tuba_placement


if __name__ == "__main__":
    main(
        nrrd_path=str(halle_skull_nrrd()),
        out_dir=str(REPO_ROOT / "data" / "halle_dcn"),
        target_name="dentate_left",
        scalp_site=None,                 # ignored; placement is precomputed
        domain_size_m=(130e-3, 80e-3, 80e-3),
        download=False,
        skip_acoustic=False,
        skip_shear=False,
        use_velocity=False,
        theta_burst=False,
    )
