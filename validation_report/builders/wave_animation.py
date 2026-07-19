"""Validation-report builder: 3-D propagation animations for all four domains.

For each requested field (pressure, shear, temperature, neural) this renders
scripts/napari_wave_propagation.py ONCE with two sinks:

  * a standalone MP4 (``data/<sim>/<field>_3d.mp4``), and
  * a numbered PNG frame sequence (``<fig_dir>/<field>anim/frame*.png``)

and returns the context for a LaTeX ``\\animategraphics`` figure embedded in the
matching report section. Opt-in (report_builder --with-animation) because it is
expensive (one napari render per field) and needs a display + GPU. Fields whose
required input is missing are skipped.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

# builders/ -> validation_report/ -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
WAVE_SCRIPT = REPO_ROOT / "scripts" / "napari_wave_propagation.py"
SKULL_SCRIPT = REPO_ROOT / "scripts" / "napari_skull_domain.py"

# Per-field: standalone MP4 name, frame-subdir (relative to figures/), the input
# file that must exist, and the tuned render flags (matching the hero clips).
FIELD_SPECS: dict[str, dict] = {
    "pressure": {
        "mp4": "wave_propagation_3d.mp4", "frames": "pressanim",
        "requires": "propagation_map.npy",
        "extra": ["--clim-frac", "0.15", "--cum-clim-frac", "4", "--zoom", "0.55",
                  "--rotate-deg", "120", "--rotate-axis", "0",
                  "--rotate-deg2", "90", "--rotate-axis2", "1", "--time-end", "320"],
    },
    "shear": {
        "mp4": "shear_propagation_3d.mp4", "frames": "shearanim",
        "requires": "displacement_4d.npy",
        "extra": ["--no-cumulative", "--clim-frac", "0.5", "--zoom", "0.6",
                  "--cam-angles", "8,-12,84", "--time-end", "24",
                  "--target-roi", "--rotate-deg", "0", "--rotate-deg2", "0"],  # static
    },
    "temperature": {
        "mp4": "temperature_3d.mp4", "frames": "tempanim",
        "requires": "bioheat/delta_T_frames.npy",
        "extra": ["--no-cumulative", "--clim-frac", "1.0", "--gamma", "0.4",
                  "--zoom", "0.6", "--cam-angles", "8,-12,84",
                  "--target-roi", "--rotate-deg", "0", "--rotate-deg2", "0"],  # static
    },
    "neural": {
        "mp4": "neural_firing_3d.mp4", "frames": "neuralanim",
        "requires": "neural_results.npz",
        "extra": ["--target-roi", "--clim-frac", "0.4", "--zoom", "0.6",
                  "--cam-angles", "8,-12,84", "--point-size-mm", "0.63",
                  "--rotate-deg", "0", "--rotate-deg2", "0"],  # static
    },
}


def _run(cmd, timeout, retries=1):
    """Run a subprocess; retry on failure (napari/vispy occasionally raises a
    transient 'CanvasBackendDesktop has been deleted' teardown error)."""
    last = None
    for attempt in range(retries + 1):
        proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout)
        if proc.returncode == 0:
            return proc
        last = "\n".join((proc.stdout + proc.stderr).splitlines()[-40:])
        if attempt < retries:
            print(f"[wave_animation] render failed (rc={proc.returncode}), retrying...", flush=True)
    raise RuntimeError(
        f"command failed after {retries + 1} tries:\n  {' '.join(map(str, cmd))}\n"
        f"--- last output ---\n{last}"
    )


def build(
    cfg,
    fig_dir: Path,
    *,
    fields=("pressure", "shear", "temperature", "neural"),
    n_frames: int = 120,
    fps: int = 10,
    size: int = 1000,
    embed_size: int = 480,
    mp4_fps: int = 24,
) -> dict[str, Any]:
    """Render the requested field animations. Returns {'animations': {field: ctx}}.

    Each field renders ONCE at ``size`` px -> a full-quality standalone MP4 +
    a PNG frame sequence; the embedded PDF frames are then downsampled to
    ``embed_size`` px (so the MP4 stays sharp while the PDF stays reasonable).
    ``fps`` is the in-PDF \\animategraphics playback rate (independent of the MP4
    speed, ``mp4_fps``); both use the same ``n_frames`` frames.
    """
    config_path = str(Path(cfg.__file__).resolve())
    sim_dir = REPO_ROOT / cfg.SIM_DATA_DIR

    if not (sim_dir / "skull_fullres_c.npy").exists():
        print(f"[wave_animation] building full-res skull cache for {cfg.SIM_NAME} ...", flush=True)
        _run([sys.executable, str(SKULL_SCRIPT), "--config", config_path, "--build-only"],
             timeout=900)

    animations: dict[str, Any] = {}
    for field in fields:
        spec = FIELD_SPECS.get(field)
        if spec is None:
            print(f"[wave_animation] unknown field {field!r}, skipping", flush=True)
            continue
        if not (sim_dir / spec["requires"]).exists():
            print(f"[wave_animation] skip {field}: missing {spec['requires']}", flush=True)
            continue

        frames_dir = Path(fig_dir) / spec["frames"]
        mp4_path = sim_dir / spec["mp4"]
        print(f"[wave_animation] rendering {field}: {n_frames} frames @ {size}px "
              f"-> {mp4_path.name} + figures/{spec['frames']}/", flush=True)
        cmd = [
            sys.executable, str(WAVE_SCRIPT), "--config", config_path, "--field", field,
            "--render", str(mp4_path),
            "--frames-dir", str(frames_dir), "--frames-prefix", "frame",
            "--render-fps", str(mp4_fps), "--render-seconds", f"{n_frames / mp4_fps:.6f}",
            "--render-size", f"{size}x{size}",
            *spec["extra"],
        ]
        _run(cmd, timeout=1800)
        produced = sorted(frames_dir.glob("frame*.png"))
        if len(produced) < n_frames:
            raise RuntimeError(
                f"[wave_animation] {field}: expected {n_frames} frames, found "
                f"{len(produced)} in {frames_dir}"
            )
        # Downsample the embedded PDF frames (the MP4 keeps the full ``size``).
        if embed_size and embed_size < size:
            from PIL import Image
            for p in produced:
                im = Image.open(p)
                if im.size[0] != embed_size:
                    im.convert("RGB").resize((embed_size, embed_size),
                                             Image.LANCZOS).save(p)
        animations[field] = {
            "available": True,
            "prefix": f"{spec['frames']}/frame",  # relative to graphicspath figures/
            "first": 0,
            "last": n_frames - 1,
            "n_frames": n_frames,
            "fps": fps,
            "duration_s": n_frames / fps,
            "mp4": spec["mp4"],
        }

    return {"animations": animations}
