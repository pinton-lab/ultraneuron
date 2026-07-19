"""Compose Figure 3 (acoustic FDTD output + parallel thermal branch).

Layout:
    Row 1: 3 orthogonal intensity slices (acoustic_ortho_intensity.png)
    Row 2: 3 orthogonal DeltaT slices (thermal_ortho_dT.png) +
           single per-voxel DeltaT trace plot (thermal_time_history.png)
"""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec

HERE = Path(__file__).resolve().parent
FIG_DIR = HERE.parent / "data" / os.environ.get("HALLE_SIM_NAME", "halle_caudate") / "validation_report" / "figures"
OUT_DIR = HERE.parent / "figures"

_acoustic_path = FIG_DIR / "acoustic_ortho_intensity.png"
if not _acoustic_path.exists():
    # Newer validation_report builds emit a combined ortho + 3D panel.
    _acoustic_path = FIG_DIR / "acoustic_intensity_combined.png"
acoustic = mpimg.imread(_acoustic_path)
thermal_ortho = mpimg.imread(FIG_DIR / "thermal_ortho_dT.png")
thermal_traces = mpimg.imread(FIG_DIR / "thermal_time_history.png")

# Row heights match panel aspect ratios so each row fills its width
# without distortion.
W = 14.0
h_top = W / (acoustic.shape[1] / acoustic.shape[0])
# Bottom row: ortho slices on left (~60%), trace plot on right (~40%)
w_bot_left = 0.62 * W
w_bot_right = 0.38 * W
h_bot_left = w_bot_left / (thermal_ortho.shape[1] / thermal_ortho.shape[0])
h_bot_right = w_bot_right / (thermal_traces.shape[1] / thermal_traces.shape[0])
h_bot = max(h_bot_left, h_bot_right)
H = h_top + h_bot

fig = plt.figure(figsize=(W, H))
gs = GridSpec(
    2, 2, figure=fig,
    height_ratios=[h_top, h_bot],
    width_ratios=[0.62, 0.38],
    left=0.0, right=1.0, top=1.0, bottom=0.0,
    wspace=0.0, hspace=0.02,
)

ax_top = fig.add_subplot(gs[0, :])
ax_top.imshow(acoustic); ax_top.axis("off")

ax_bl = fig.add_subplot(gs[1, 0])
ax_bl.imshow(thermal_ortho); ax_bl.axis("off")

ax_br = fig.add_subplot(gs[1, 1])
ax_br.imshow(thermal_traces); ax_br.axis("off")

for ext in ("pdf", "png"):
    out = OUT_DIR / f"fig3_acoustic_intensity.{ext}"
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.05)
    print(f"wrote {out}")
plt.close(fig)
