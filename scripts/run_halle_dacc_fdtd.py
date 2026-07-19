"""One-off driver: run bowl_transcranial_500khz.main for the dACC re-aim.

Calls the canonical FDTD pipeline with auto-pick placement (scalp_site=None)
at the manuscript-canonical surface drive (source_p0 = 0.10 MPa, hardcoded
in bowl_transcranial_500khz.py:921). Outputs land in data/halle_dacc/ for
the validation_report.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from bowl_transcranial_500khz import main  # noqa: E402

from ultraneuron.paths import halle_skull_nrrd  # noqa: E402


if __name__ == "__main__":
    main(
        nrrd_path=str(halle_skull_nrrd()),
        out_dir=str(REPO_ROOT / "data" / "halle_dacc"),
        target_name="ACC_left",
        scalp_site=None,                 # auto-pick perpendicular aim
        domain_size_m=(130e-3, 80e-3, 80e-3),
        download=False,
        skip_acoustic=False,
        skip_shear=False,
        use_velocity=False,
        theta_burst=False,
    )
