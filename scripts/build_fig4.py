"""Compose Figure 4 (mechanical pipeline: ARF + displacement snapshots).

Layout:
    Row 1: 3 orthogonal ARF slices (elastic_arf_ortho.png)
    Row 2: 5 times x 2 planes displacement snapshots (fig_disp_snapshots.png)
"""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec

HERE = Path(__file__).resolve().parent
FIG_DIR = HERE.parent / "data" / os.environ.get("HALLE_SIM_NAME", "halle_caudate") / "validation_report" / "figures"
OUT_DIR = HERE.parent / "figures"

arf = mpimg.imread(FIG_DIR / "elastic_arf_ortho.png")
disp = mpimg.imread(FIG_DIR / "fig_disp_snapshots.png")

W = 14.0
h_top = W / (arf.shape[1] / arf.shape[0])
h_bot = W / (disp.shape[1] / disp.shape[0])
H = h_top + h_bot

fig = plt.figure(figsize=(W, H))
gs = GridSpec(
    2, 1, figure=fig,
    height_ratios=[h_top, h_bot],
    left=0.0, right=1.0, top=1.0, bottom=0.0,
    hspace=0.02,
)
ax_top = fig.add_subplot(gs[0]); ax_top.imshow(arf);  ax_top.axis("off")
ax_bot = fig.add_subplot(gs[1]); ax_bot.imshow(disp); ax_bot.axis("off")

for ext in ("pdf", "png"):
    out = OUT_DIR / f"fig4_arf_displacement.{ext}"
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.05)
    print(f"wrote {out}")
plt.close(fig)
