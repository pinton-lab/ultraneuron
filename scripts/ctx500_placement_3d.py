"""3D + slice rendering of the CTX-500 / DLPFC_left placement on the Halle skull.

Same style as Figure 4 in the neuromod_parameters progress report
(top row: orthogonal slice views with bowl outline drawn IN PLANE; bottom row:
three 3D camera angles with bowl wireframe + skull/brain/target meshes).

Reuses visualize_placement_3d.py from the neuromod_parameters registration
module (located at runtime via ultraneuron.registration; set
$NEUROMOD_PARAMETERS_DIR or keep neuromod_parameters as a sibling directory)
with our CTX-500 spec (F=63 mm, R=32 mm) and the F3 -> DLPFC_left placement.
"""
import os
import sys
from pathlib import Path

from ultraneuron.registration import ensure_registration_importable

REG_DIR = str(ensure_registration_importable())

import visualize_placement_3d as viz
from halle_placement import place_on_skull

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "halle_transcranial"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "fig_ctx500_placement_3d.png"

# CTX-500 spec
F_MM = 63.0
R_MM = 32.0
SCALP = "F3"
TARGET = "DLPFC_left"


def main() -> None:
    p = place_on_skull(
        scalp_site=SCALP,
        target_name=TARGET,
        focal_length_mm=F_MM,
        bowl_radius_mm=R_MM,
    )
    p["target_name"] = TARGET
    print("Placement:")
    for k, v in p.items():
        print(f"  {k}: {v}")

    # Redirect the renderer's hard-coded output path
    viz.OUT_PATH = str(OUT_PATH)
    viz.render(p)


if __name__ == "__main__":
    main()
