"""Compose Figure 6 (within-burst recruitment dynamics).

Layout:
    Top:    6-row tension + cumulative-spike snapshots
            (neural_snapshot_row.png)
    Bottom: single-burst spike raster + 3 ROI panels + depth histogram
            (neural_raster.png)
"""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec

HERE = Path(__file__).resolve().parent
FIG_DIR = HERE.parent / "data" / os.environ.get("HALLE_SIM_NAME", "halle_caudate") / "validation_report" / "figures"
OUT_DIR = HERE.parent / "figures"

snaps = mpimg.imread(FIG_DIR / "neural_snapshot_row.png")
raster = mpimg.imread(FIG_DIR / "neural_raster.png")

W = 14.0
h_top = W / (snaps.shape[1] / snaps.shape[0])
h_bot = W / (raster.shape[1] / raster.shape[0])
H = h_top + h_bot

fig = plt.figure(figsize=(W, H))
gs = GridSpec(
    2, 1, figure=fig,
    height_ratios=[h_top, h_bot],
    left=0.0, right=1.0, top=1.0, bottom=0.0,
    hspace=0.02,
)
ax_top = fig.add_subplot(gs[0]); ax_top.imshow(snaps);  ax_top.axis("off")
ax_bot = fig.add_subplot(gs[1]); ax_bot.imshow(raster); ax_bot.axis("off")

for ext in ("pdf", "png"):
    out = OUT_DIR / f"fig6_neural_snapshots_raster.{ext}"
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.05)
    print(f"wrote {out}")
plt.close(fig)
