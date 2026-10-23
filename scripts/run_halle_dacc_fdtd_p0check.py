"""R2-7 check: re-run the Halle/dACC Fullwave FDTD at the *rescaled* source amplitude
(0.10 MPa x 1.0868, the factor applied post hoc to land the in-brain PPP at 0.50 MPa)
into a separate directory, so the nonlinear re-run can be compared voxel by voxel
with the linearly rescaled canonical field.  Acoustic stage only (skip_shear=True).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; REPO_ROOT = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REPO_ROOT))
from bowl_transcranial_500khz import main  # noqa: E402
from ultraneuron.paths import halle_skull_nrrd  # noqa: E402
prov = json.loads((REPO_ROOT / "data/halle_dacc/rescale_provenance.json").read_text())
scale = float(prov["scale_factor_applied"])
out = REPO_ROOT / "data" / "halle_dacc_p0check"; out.mkdir(exist_ok=True)
(out / "P0CHECK.json").write_text(json.dumps({"source_p0_pa": 0.10e6 * scale, "scale_vs_canonical": scale}, indent=1))
if __name__ == "__main__":
    main(nrrd_path=str(halle_skull_nrrd()), out_dir=str(out), target_name="ACC_left", scalp_site=None,
         domain_size_m=(130e-3, 80e-3, 80e-3), download=False, skip_acoustic=False, skip_shear=True,
         use_velocity=False, theta_burst=False, source_p0=0.10e6 * scale)
