"""Compose Figure 7 (Boltzmann gating curves) from the validation-report
constituent ``neural_gating.png``.  Single-panel; this script just
re-emits as PDF/PNG into the manuscript figures directory.
"""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

HERE = Path(__file__).resolve().parent
FIG_DIR = HERE.parent / "data" / os.environ.get("HALLE_SIM_NAME", "halle_caudate") / "validation_report" / "figures"
OUT_DIR = HERE.parent / "figures"

img = mpimg.imread(FIG_DIR / "neural_gating.png")
W = 14.0
H = W / (img.shape[1] / img.shape[0])

fig = plt.figure(figsize=(W, H))
ax = fig.add_subplot(111)
ax.imshow(img); ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

for ext in ("pdf", "png"):
    out = OUT_DIR / f"fig7_neural_gating.{ext}"
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.0)
    print(f"wrote {out}")
plt.close(fig)
