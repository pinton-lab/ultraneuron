"""R2-M3: quantify how well the SyN-warped MNI152 brain fits the Halle skull cavity.

The Harvard-Oxford ROI used to aim the transducer is carried into the specimen frame by
an ANTs SyN registration (Mattes MI) of the MNI152 T1 to the specimen CT, computed in the
sister ``neuromod_parameters`` repository.  Landmark residuals recorded there are not
informative (the specimen has no soft tissue, so scalp landmarks are ill-defined), so the
fit is quantified here by the distance from the warped brain surface to the nearest bone
voxel of the specimen, compared with the ~2-4 mm CSF/meningeal gap of a real head, and
by the fraction of warped brain voxels that fall inside bone.

Writes data/halle_dacc/registration_fit.json and docs/.../data/registration_numbers.tex.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from ultraneuron.registration import registration_dir  # noqa: E402

reg = registration_dir(required=True)
ct = nib.load(reg / "halle_ct_1mm.nii.gz"); hu = ct.get_fdata(); vox = np.abs(np.diag(ct.affine)[:3])
brain = nib.load(reg / "mni_brain_in_halle.nii.gz").get_fdata() > 0.5
bone = hu > 700
dist = ndi.distance_transform_edt(~bone, sampling=vox)
surf = brain & ~ndi.binary_erosion(brain)
d = dist[surf]
z = np.nonzero(surf)[2]; zc = np.percentile(np.nonzero(brain)[2], 40); du = d[z > zc]
val = json.loads((reg / "mni_to_halle_syn_validation.json").read_text())
out = {
    "method": "ANTs SyN (Mattes MI), MNI152 T1 -> specimen CT at 1 mm; sister repo neuromod_parameters/registration",
    "warped_brain_volume_mL": float(brain.sum() * np.prod(vox) / 1e3),
    "brain_voxels_inside_bone_pct": float(100 * (brain & bone).sum() / brain.sum()),
    "surface_to_bone_mm": {"median": float(np.median(d)), "mean": float(d.mean()), "p5": float(np.percentile(d, 5)),
                           "p95": float(np.percentile(d, 95)), "max": float(d.max())},
    "vault_surface_to_bone_mm": {"median": float(np.median(du)), "p95": float(np.percentile(du, 95))},
    "held_out_Cz_landmark_mm": val.get("held_out_Cz_mm"),
    "note_on_landmark_residuals": "scalp landmarks (nasion, inion, preauricular) are ill-defined on a soft-tissue-free specimen; their recorded residuals are not used",
}
(REPO / "data/halle_dacc/registration_fit.json").write_text(json.dumps(out, indent=1))
mac = {"RegSurfMedian": f"{out['surface_to_bone_mm']['median']:.1f}", "RegSurfPninetyfive": f"{out['surface_to_bone_mm']['p95']:.1f}",
       "RegInsideBone": f"{out['brain_voxels_inside_bone_pct']:.1f}", "RegBrainVol": f"{out['warped_brain_volume_mL']:.0f}",
       "RegCz": f"{out['held_out_Cz_landmark_mm']:.0f}" if out["held_out_Cz_landmark_mm"] else "--"}
(REPO / "data/registration_numbers.tex").write_text("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in mac.items()) + "\n")
print(json.dumps(out, indent=1)); print(mac)
