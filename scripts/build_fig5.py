"""Compose Figure 5 (from mechanical drive to per-voxel firing).

Layout:
    Row 1: tension ortho slices (elastic_tension_ortho.png)
    Row 2: first-spike latency ortho (neural_latency_ortho.png)
    Row 3: firing-rate iso-surfaces 3-D (neural_rate_3d.png)
"""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec

HERE = Path(__file__).resolve().parent
FIG_DIR = HERE.parent / "data" / os.environ.get("HALLE_SIM_NAME", "halle_caudate") / "validation_report" / "figures"
OUT_DIR = HERE.parent / "figures"

tension = mpimg.imread(FIG_DIR / "elastic_tension_ortho.png")
latency = mpimg.imread(FIG_DIR / "neural_latency_ortho.png")
rate_3d = mpimg.imread(FIG_DIR / "neural_rate_3d.png")

W = 14.0
h_top = W / (tension.shape[1] / tension.shape[0])
h_mid = W / (latency.shape[1] / latency.shape[0])
h_bot = W / (rate_3d.shape[1] / rate_3d.shape[0])
H = h_top + h_mid + h_bot

fig = plt.figure(figsize=(W, H))
gs = GridSpec(
    3, 1, figure=fig,
    height_ratios=[h_top, h_mid, h_bot],
    left=0.0, right=1.0, top=1.0, bottom=0.0,
    hspace=0.02,
)
ax1 = fig.add_subplot(gs[0]); ax1.imshow(tension); ax1.axis("off")
ax2 = fig.add_subplot(gs[1]); ax2.imshow(latency); ax2.axis("off")
ax3 = fig.add_subplot(gs[2]); ax3.imshow(rate_3d); ax3.axis("off")

for ext in ("pdf", "png"):
    out = OUT_DIR / f"fig5_neural_latency_iso.{ext}"
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.05)
    print(f"wrote {out}")
plt.close(fig)
