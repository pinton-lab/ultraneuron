#!/usr/bin/env python3
"""Render the simulated acoustic pressure wave propagating through the
full-resolution skull, in napari.

Loads the FDTD pressure field the acoustic solver already saved
(``propagation_map.npy``, shape ``(nt, nx', ny', nz')`` on the
mod_x/mod_y/mod_z-decimated sensor grid) and overlays it as a time-animated
volume on the full-resolution skull medium (``skull_fullres_c.npy``, built by
``napari_skull_domain.py``).

Grid reconciliation (the mod_x/mod_y/mod_z mismatch)
---------------------------------------------------
The pressure is saved on the spatially-decimated grid (``mod ~= 4``) while the
skull is full resolution. Two ways to put them in one space:

* ``--interp gpu`` (default): keep the coarse field and set the napari layer
  ``scale`` to ``mod*dx``. napari/vispy upload it as a 3-D texture and the GPU
  trilinearly interpolates during volume rendering, so it looks smooth and
  co-registers with the skull in world (mm) coordinates -- no multi-GB
  upsampled array needed (a full 3-D upsample of all 603 frames would be ~69 GB).
* ``--interp cpu``: actually resample every frame onto the full sim grid with
  ``scipy.ndimage.zoom`` (order=1), lazily via dask so only the frame on screen
  is materialised. This is the literal "interpolate back to the sim grid"; use
  it if you need the field voxel-aligned to the skull (e.g. for export).

Runs under the system ``python3`` (napari / PyQt6 / vispy / scipy / dask).

Examples
--------
    python3 scripts/napari_wave_propagation.py --config validation_report/configs/halle_dacc.py
    python3 scripts/napari_wave_propagation.py --config ... --interp cpu
    python3 scripts/napari_wave_propagation.py --config ... --no-gui --screenshot /tmp/wave.png
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
BONE_C_THRESHOLD = 1600.0


def load_config(config_path: str):
    path = Path(config_path)
    if not path.is_absolute():
        path = REPO / path
    spec = importlib.util.spec_from_file_location("sim_config", path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    return cfg


def pressure_diverging_colormap():
    """Diverging blue/red colormap, transparent at zero pressure.

    The transparent middle band means only wavefronts (|p| away from 0) are
    opaque, so the propagating field reads cleanly over the skull instead of
    fogging the whole volume.
    """
    from napari.utils import Colormap

    colors = np.array(
        [
            [0.10, 0.30, 1.00, 1.00],  # strong rarefaction -> opaque blue
            [0.35, 0.55, 1.00, 0.00],  # mild negative      -> transparent
            [1.00, 0.55, 0.35, 0.00],  # mild positive      -> transparent
            [1.00, 0.25, 0.10, 1.00],  # strong compression -> opaque red
        ]
    )
    controls = [0.0, 0.46, 0.54, 1.0]
    return Colormap(colors=colors, controls=controls, name="pressure_diverging")


def pressure_hot_colormap():
    """Symmetric 'hot' colormap for the signed pressure field.

    Transparent at p=0, glowing hot (dark-red -> orange -> yellow -> white)
    toward BOTH compression and rarefaction extremes, so every wavefront reads
    as a hot band (the |p| envelope) rather than the blue/red phase.
    """
    from napari.utils import Colormap

    colors = np.array(
        [
            [1.00, 1.00, 1.00, 1.00],  # 0.00 max rarefaction -> white hot
            [1.00, 0.95, 0.40, 0.92],  # 0.10 yellow
            [1.00, 0.45, 0.00, 0.60],  # 0.28 orange
            [0.55, 0.05, 0.00, 0.10],  # 0.45 dark red, fading
            [0.00, 0.00, 0.00, 0.00],  # 0.50 zero -> transparent
            [0.55, 0.05, 0.00, 0.10],  # 0.55 dark red, fading
            [1.00, 0.45, 0.00, 0.60],  # 0.72 orange
            [1.00, 0.95, 0.40, 0.92],  # 0.90 yellow
            [1.00, 1.00, 1.00, 1.00],  # 1.00 max compression -> white hot
        ]
    )
    controls = [0.0, 0.10, 0.28, 0.45, 0.50, 0.55, 0.72, 0.90, 1.0]
    return Colormap(colors=colors, controls=controls, name="pressure_hot")


def pressure_cool_colormap():
    """Symmetric 'cool' colormap for the signed pressure field.

    Transparent at p=0, glowing cool (deep-blue -> cyan -> white) toward BOTH
    compression and rarefaction extremes, so every wavefront reads as a cool
    band (the |p| envelope). Pairs with the warm cumulative glow.
    """
    from napari.utils import Colormap

    colors = np.array(
        [
            [1.00, 1.00, 1.00, 1.00],  # 0.00 max rarefaction -> white
            [0.50, 0.95, 1.00, 0.92],  # 0.10 pale cyan
            [0.00, 0.55, 1.00, 0.60],  # 0.28 blue
            [0.00, 0.10, 0.50, 0.10],  # 0.45 deep blue, fading
            [0.00, 0.00, 0.00, 0.00],  # 0.50 zero -> transparent
            [0.00, 0.10, 0.50, 0.10],  # 0.55 deep blue, fading
            [0.00, 0.55, 1.00, 0.60],  # 0.72 blue
            [0.50, 0.95, 1.00, 0.92],  # 0.90 pale cyan
            [1.00, 1.00, 1.00, 1.00],  # 1.00 max compression -> white
        ]
    )
    controls = [0.0, 0.10, 0.28, 0.45, 0.50, 0.55, 0.72, 0.90, 1.0]
    return Colormap(colors=colors, controls=controls, name="pressure_cool")


PRESSURE_CMAPS = {
    "cool": pressure_cool_colormap,
    "hot": pressure_hot_colormap,
    "diverging": pressure_diverging_colormap,
}


def intensity_glow_colormap():
    """Warm sequential glow, transparent at zero -> bright at high.

    Used for the cumulative-intensity (time-integrated p^2) background, so the
    deposited-energy trail along the beam path and focus glows without fogging
    the empty volume.
    """
    from napari.utils import Colormap

    colors = np.array(
        [
            [0.00, 0.00, 0.00, 0.00],  # 0           -> fully transparent
            [0.55, 0.00, 0.10, 0.12],  # low         -> faint dark red
            [1.00, 0.45, 0.00, 0.45],  # mid         -> orange
            [1.00, 1.00, 0.75, 0.95],  # high        -> bright yellow-white
        ]
    )
    controls = [0.0, 0.12, 0.5, 1.0]
    return Colormap(colors=colors, controls=controls, name="intensity_glow")


def intensity_cool_colormap():
    """Cool teal/cyan glow, transparent at zero -> bright at high.

    Use for the cumulative background when the pressure layer is 'hot', so the
    warm wave reads clearly against a cool persistent background.
    """
    from napari.utils import Colormap

    colors = np.array(
        [
            [0.00, 0.00, 0.00, 0.00],  # 0    -> fully transparent
            [0.00, 0.20, 0.45, 0.12],  # low  -> faint blue
            [0.00, 0.65, 0.85, 0.45],  # mid  -> cyan
            [0.70, 1.00, 1.00, 0.95],  # high -> bright cyan-white
        ]
    )
    controls = [0.0, 0.12, 0.5, 1.0]
    return Colormap(colors=colors, controls=controls, name="intensity_cool")


CUM_CMAPS = {"glow": intensity_glow_colormap, "cool": intensity_cool_colormap}


def displacement_colormap():
    """Perceptual (viridis) colormap for the displacement magnitude |u|.

    Multi-hue so the magnitude gradient reads as COLOUR (purple->blue->green->
    yellow) rather than a near-uniform single-hue blob, with the alpha
    suppressed below ~35% of the contrast limit so the broad ARF push doesn't
    fill the volume with an opaque halo that hides the skull.
    """
    import matplotlib

    from napari.utils import Colormap

    n = 256
    rgba = matplotlib.colormaps["viridis"](np.linspace(0, 1, n))  # (n, 4)
    alpha = np.clip((np.linspace(0, 1, n) - 0.35) / 0.5, 0, 1) ** 1.2
    rgba[:, 3] = alpha * 0.95
    return Colormap(colors=rgba, controls=np.linspace(0, 1, n), name="displacement_viridis")


def temperature_colormap():
    """Hot (inferno) colormap for the (non-negative) temperature rise ΔT.

    black->purple->red->orange->yellow->white, with the alpha suppressed below
    ~30% of the contrast limit so the broad low-ΔT diffusion halo doesn't fill
    the volume (the rise concentrates in the skull and at the focus).
    """
    import matplotlib

    from napari.utils import Colormap

    n = 256
    rgba = matplotlib.colormaps["inferno"](np.linspace(0, 1, n))
    alpha = np.clip((np.linspace(0, 1, n) - 0.30) / 0.5, 0, 1) ** 1.1
    rgba[:, 3] = alpha * 0.95
    return Colormap(colors=rgba, controls=np.linspace(0, 1, n), name="temperature_inferno")


def neural_colormap():
    """Classic 'fire' (hot) colormap for neural firing activity.

    black->red->yellow->white. Alpha ramps from near-zero quickly so the faint
    persistent 'has-fired' floor is just visible while fresh spikes flash bright.
    """
    import matplotlib

    from napari.utils import Colormap

    n = 256
    rgba = matplotlib.colormaps["hot"](np.linspace(0, 1, n))
    rgba[:, 3] = np.clip((np.linspace(0, 1, n) - 0.08) / 0.30, 0, 1) * 0.97
    return Colormap(colors=rgba, controls=np.linspace(0, 1, n), name="neural_hot")


def overlay_colorbar(img, napari_cmap, clim, label, gamma=1.0):
    """Composite a vertical colorbar (PIL) onto an RGB frame.

    clim = (lo, hi) in DISPLAY units. Bar colours are premultiplied by the
    colormap alpha over black, so the bar matches what is actually visible in
    the volume (transparent low end -> black). ``gamma`` matches the layer's
    gamma: the colour at value-fraction f is colormap(f**gamma), so the bar is
    nonlinear (low values get more colour range) while the tick VALUES stay
    linear -- the colorbar then reads correctly under a gamma stretch.
    """
    from PIL import Image, ImageDraw, ImageFont

    H, W = img.shape[:2]
    im = Image.fromarray(np.ascontiguousarray(img[..., :3]).astype(np.uint8))
    draw = ImageDraw.Draw(im)
    bw = max(16, W // 45)
    bh = int(0.50 * H)
    x1 = W - int(0.05 * W)
    x0 = x1 - bw
    y0 = int(0.24 * H)
    vals = np.linspace(1.0, 0.0, bh)                       # top = hi, bottom = lo
    rgba = napari_cmap.map(vals ** gamma)                  # (bh, 4); gamma-matched
    rgb = (rgba[:, :3] * rgba[:, 3:4] * 255).astype(np.uint8)   # premultiplied over black
    im.paste(Image.fromarray(np.repeat(rgb[:, None, :], bw, axis=1)), (x0, y0))
    draw.rectangle([x0, y0, x1, y0 + bh], outline=(220, 220, 220), width=1)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", max(12, H // 55))
    except Exception:
        font = ImageFont.load_default()
    for frac, val in ((0.0, clim[1]), (0.5, 0.5 * (clim[0] + clim[1])), (1.0, clim[0])):
        yy = int(y0 + frac * bh)
        draw.line([x1, yy, x1 + 5, yy], fill=(220, 220, 220), width=1)
        draw.text((x1 + 8, yy - 7), f"{val:.2g}", fill=(235, 235, 235), font=font)
    draw.text((x0 - 2, y0 - max(18, H // 38)), label, fill=(235, 235, 235), font=font)
    return np.asarray(im)


def load_field(sim_dir, field, *, rebuild=False):
    """Load the time-resolved scalar field to render, with display metadata.

    pressure : signed acoustic pressure p (``propagation_map.npy``), oscillatory.
    shear    : displacement magnitude |u| reduced from ``displacement_4d.npy``
               (nt,nx,ny,nz,3), cached as ``displacement_mag.npy``; non-negative.

    Returns (mmap (nt,nx',ny',nz'), spec) where ``spec.kind`` is 'signed' or
    'magnitude'.
    """
    if field == "pressure":
        mmap = np.load(sim_dir / "propagation_map.npy", mmap_mode="r")
        spec = {
            "kind": "signed", "unit": "Pa", "layer": "pressure [Pa]",
            "title": "pressure wave", "cum_cache": "cumulative_intensity.npy",
            "cum_layer": "cumulative intensity sum(p^2) [Pa^2]",
            "cbar_scale": 1e-3, "cbar_label": "p (kPa)",
        }
        return mmap, spec
    if field == "shear":
        magp = sim_dir / "displacement_mag.npy"
        if rebuild or not magp.exists():
            d = np.load(sim_dir / "displacement_4d.npy", mmap_mode="r")  # (nt,nx,ny,nz,3)
            out = np.empty(d.shape[:4], dtype=np.float32)
            t0 = time.time()
            for t in range(d.shape[0]):
                u = np.asarray(d[t], dtype=np.float64)
                out[t] = np.sqrt((u ** 2).sum(axis=-1))
            np.save(magp, out)
            print(f"[shear] built |u| {out.shape} in {time.time()-t0:.1f}s "
                  f"-> {magp.name} (peak {out.max():.3e} m)", flush=True)
        mmap = np.load(magp, mmap_mode="r")
        spec = {
            "kind": "magnitude", "unit": "m", "layer": "displacement |u| [m]",
            "title": "shear displacement", "cum_cache": "cumulative_shear_energy.npy",
            "cum_layer": "cumulative motion sum(|u|^2) [m^2]",
            "cbar_scale": 1e6, "cbar_label": "|u| (µm)",
        }
        return mmap, spec
    if field == "temperature":
        # delta_T_frames is (nx, ny, nz, nt) with time LAST -> move to (nt, ...).
        arr = np.load(sim_dir / "bioheat" / "delta_T_frames.npy")  # ~48 MB
        arr = np.ascontiguousarray(arr.transpose(3, 0, 1, 2))
        spec = {
            "kind": "magnitude", "unit": "K", "layer": "ΔT [K]",
            "title": "temperature rise", "cum_cache": "cumulative_temperature.npy",
            "cum_layer": "cumulative ΔT", "cbar_scale": 1.0, "cbar_label": "ΔT (K)",
        }
        return arr, spec
    raise SystemExit(f"unknown --field {field!r} (use 'pressure', 'shear' or 'temperature')")


def frame_times(field, sim_dir, cfg, gp, nt):
    """Per-frame physical time (seconds), or None if unavailable.

    shear    : the solver's saved ``displacement_4d_times.npy``.
    pressure : frame_idx * sampling_modulus_time * dt, dt = CFL*dx/c0 (the field
               is saved every ``sampling_modulus_time`` FDTD steps).
    """
    if field == "shear":
        tpath = sim_dir / "displacement_4d_times.npy"
        if tpath.exists():
            t = np.load(tpath)
            return t[:nt] if len(t) >= nt else None
    if field == "temperature":
        tpath = sim_dir / "bioheat" / "delta_T_frame_times.npy"
        if tpath.exists():
            t = np.load(tpath)
            return t[:nt] if len(t) >= nt else None
        return None
    dx = float(gp["dx"])
    cfl = float(getattr(cfg, "CFL", 0.2))
    c0 = float(getattr(cfg, "C0", 1540.0))
    smt = int(gp["sampling_modulus_time"]) if "sampling_modulus_time" in gp else 1
    return np.arange(nt) * smt * (cfl * dx / c0)


def timestamp_format(times_s):
    """Pick a display unit (µs/ms/s) and decimal count so adjacent frames are
    distinguishable but not spuriously precise."""
    tmax = float(np.max(times_s))
    if tmax < 1e-3:
        scale, unit = 1e6, "µs"
    elif tmax < 1.0:
        scale, unit = 1e3, "ms"
    else:
        scale, unit = 1.0, "s"
    dts = np.abs(np.diff(times_s))
    dt_disp = float(np.median(dts)) * scale if dts.size else 1.0
    dec = int(np.clip(1 - np.floor(np.log10(dt_disp)) if dt_disp > 0 else 1, 0, 4))
    return scale, unit, dec


def overlay_timestamp(img, text):
    """Draw a timestamp (PIL) at the top-left of an RGB frame."""
    from PIL import Image, ImageDraw, ImageFont

    H, W = img.shape[:2]
    im = Image.fromarray(np.ascontiguousarray(img[..., :3]).astype(np.uint8))
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", max(14, H // 36))
    except Exception:
        font = ImageFont.load_default()
    draw.text((int(0.03 * W), int(0.03 * H)), text, fill=(245, 245, 245), font=font)
    return np.asarray(im)


def get_cumulative(sim_dir, field_mmap, cache_name, *, rebuild=False):
    """Running time-integral of field^2 (sum_{tau<=t} field[tau]^2).

    Advances with the wavefront but persists as a background. Works for either
    signed pressure (-> sum p^2, time-integrated intensity) or a non-negative
    magnitude such as |u| (-> sum |u|^2, time-integrated motion energy). Cached
    to disk (~field size; a few seconds to build).
    """
    path = sim_dir / cache_name
    if path.exists() and not rebuild:
        print(f"[cumulative] loading cache {path.name}", flush=True)
        return np.load(path, mmap_mode="r")
    nt = field_mmap.shape[0]
    out = np.empty(field_mmap.shape, dtype=np.float32)
    acc = np.zeros(field_mmap.shape[1:], dtype=np.float64)  # float64 accumulator
    t0 = time.time()
    for t in range(nt):
        acc += np.asarray(field_mmap[t], dtype=np.float64) ** 2
        out[t] = acc
    np.save(path, out)
    print(f"[cumulative] built sum(field^2) {out.shape} in {time.time()-t0:.1f}s "
          f"-> {path.name} (final max {acc.max():.3e})", flush=True)
    return np.load(path, mmap_mode="r")


def _axis_weights(coarse_n, mod, full_n):
    """Linear-interp weight matrix (full_n x coarse_n) for one axis.

    The decimated field stores ``coarse[i] = full[mod*i]``, so full-res voxel j
    sits at coarse-index ``j/mod``. Linear-interpolate between the two bracketing
    coarse samples; clamp (constant hold) past the last covered sample
    (full indices > mod*(coarse_n-1), which were never sampled).
    """
    pos = np.arange(full_n) / mod  # coarse-index position of each full voxel
    lo = np.minimum(np.floor(pos).astype(int), coarse_n - 1)
    hi = np.minimum(lo + 1, coarse_n - 1)
    frac = (pos - np.floor(pos)).astype(np.float32)
    W = np.zeros((full_n, coarse_n), dtype=np.float32)
    rows = np.arange(full_n)
    W[rows, lo] += 1.0 - frac
    W[rows, hi] += frac
    return W


def _resample_modaware(frame, weights):
    """Separable linear upsample of a 3-D frame onto the full sim grid."""
    out = frame
    for axis, W in enumerate(weights):
        out = np.moveaxis(out, axis, 0)
        out = np.tensordot(W, out, axes=([1], [0]))
        out = np.moveaxis(out, 0, axis)
    return np.ascontiguousarray(out, dtype=np.float32)


def build_volume_data(mmap, skull_shape, voxel_mm, mod, interp, *, label, smooth_sigma=0.0):
    """Return (data, scale) placing a coarse 4-D field in the skull's world space.

    The decimated field samples full-res voxel ``mod*i`` at coarse index ``i``
    (mod = mod_x/mod_y/mod_z from grid_params), so registration uses ``mod`` --
    NOT the shape ratio nx/coarse (which is off when nx isn't divisible by mod).

    smooth_sigma : optional Gaussian anti-alias (in coarse voxels) applied per
    frame before display -- tames the near-Nyquist faceting at the cost of a
    little sharpness.

    interp : 'gpu' (coarse + scale=mod*dx, GPU interpolates the texture)
             'cpu' (mod-aware separable linear resample onto the full sim grid,
                    lazy per frame via dask).
    """
    nt = mmap.shape[0]
    coarse_shape = mmap.shape[1:]

    def _frame(t):
        f = np.asarray(mmap[t], dtype=np.float32)
        if smooth_sigma > 0:
            from scipy.ndimage import gaussian_filter
            f = gaussian_filter(f, smooth_sigma)
        return f

    if interp == "cpu":
        import dask
        import dask.array as da

        target = tuple(int(s) for s in skull_shape)
        weights = [_axis_weights(coarse_shape[i], mod[i], target[i]) for i in range(3)]

        def upsample(t):
            return _resample_modaware(_frame(t), weights)

        delayed_up = dask.delayed(upsample)
        frames = [
            da.from_delayed(delayed_up(t), shape=target, dtype=np.float32)
            for t in range(nt)
        ]
        data = da.stack(frames, axis=0)
        scale = (1.0, *voxel_mm)
        print(f"[{label}] interp=cpu  mod-aware resample {coarse_shape} -> {target} "
              f"(mod {mod}); coarse[i] <-> full[mod*i]"
              f"{f'  smooth sigma={smooth_sigma}' if smooth_sigma>0 else ''}", flush=True)
    else:  # gpu
        pvox = tuple(mod[i] * voxel_mm[i] for i in range(3))  # exact: mod*dx
        scale = (1.0, *pvox)
        if smooth_sigma > 0:
            import dask
            import dask.array as da

            delayed_sm = dask.delayed(_frame)
            data = da.stack(
                [da.from_delayed(delayed_sm(t), shape=coarse_shape, dtype=np.float32)
                 for t in range(nt)],
                axis=0,
            )
        else:
            data = mmap
        print(f"[{label}] interp=gpu  coarse {coarse_shape} @ scale "
              f"{tuple(round(v,4) for v in pvox)} mm (= mod {mod} * dx)"
              f"{f'  smooth sigma={smooth_sigma}' if smooth_sigma>0 else ''}", flush=True)
    return data, scale


def render_video(viewer, args, nt, cbar=None, ts=None, frame_update=None):
    """Deterministic frame loop: slow camera orbit while the time slider sweeps.

    Matches the pattern in fullwave_doppler_superres/napari_tracks_movie.py --
    rotate the Euler axes by a fraction of --rotate-deg(/2) per frame, traverse
    the propagation frames [time_start, time_end] linearly, screenshot the
    canvas, and write to:
      * an H.264 MP4 via imageio-ffmpeg (--render), and/or
      * a numbered PNG sequence (--frames-dir) for LaTeX \\animategraphics
        (files ``<prefix>0.png .. <prefix>N-1.png``, no zero-padding).
    """
    import imageio.v2 as imageio
    from qtpy.QtCore import QCoreApplication

    w, h = (int(s) for s in args.render_size.lower().split("x"))
    fps = args.render_fps
    n_frames = max(1, int(round(args.render_seconds * fps)))
    t0 = max(0, args.time_start)
    t1 = (nt - 1) if args.time_end < 0 else min(args.time_end, nt - 1)

    rot2 = f" + {args.rotate_deg2:.0f} deg (axis {args.rotate_axis2})" if args.rotate_deg2 else ""
    sinks = [s for s in (args.render, f"{args.frames_dir}/{args.frames_prefix}*.png"
                         if args.frames_dir else None) if s]
    print(f"[render] {n_frames} frames @ {fps} fps ({args.render_seconds:.1f}s), "
          f"{w}x{h}, orbit {args.rotate_deg:.0f} deg (axis {args.rotate_axis}){rot2}, "
          f"time {t0}->{t1} -> {', '.join(sinks)}", flush=True)

    writer = None
    if args.render:
        writer = imageio.get_writer(
            args.render, fps=fps, codec="libx264", quality=8,
            macro_block_size=8, pixelformat="yuv420p",
        )
    frames_dir = None
    if args.frames_dir:
        frames_dir = Path(args.frames_dir)
        frames_dir.mkdir(parents=True, exist_ok=True)
        for old in frames_dir.glob(f"{args.frames_prefix}*.png"):
            old.unlink()

    # Realise the canvas at the render size, THEN re-fit. main's reset_view ran
    # while the window was still at its initial (small) size, and resizing the
    # window changes napari's fit -- without re-fitting here the render comes out
    # zoomed out. Re-establish the intended framing at the actual render canvas.
    viewer.window.resize(w + 220, h + 120)
    for _ in range(8):
        QCoreApplication.processEvents()
        time.sleep(0.05)
    saved_angles = tuple(viewer.camera.angles)
    saved_center = tuple(viewer.camera.center)
    viewer.reset_view()                      # fit at the render canvas size
    viewer.camera.angles = saved_angles
    if args.center_focus:
        viewer.camera.center = saved_center
    viewer.camera.zoom *= args.zoom
    for _ in range(3):
        QCoreApplication.processEvents()

    a0 = list(viewer.camera.angles)
    zoom0 = float(viewer.camera.zoom)
    end_angles = ([float(x) for x in args.cam_angles_end.split(",")]
                  if args.cam_angles_end else None)
    try:
        for i in range(n_frames):
            frac = i / max(1, n_frames - 1)
            if end_angles is not None:                      # explicit start->end sweep
                a = [a0[k] + (end_angles[k] - a0[k]) * frac for k in range(3)]
            else:                                           # additive orbit
                a = list(a0)
                a[args.rotate_axis] = (a0[args.rotate_axis] + args.rotate_deg * frac) % 360
                if args.rotate_deg2:
                    a[args.rotate_axis2] = (a0[args.rotate_axis2] + args.rotate_deg2 * frac) % 360
            viewer.camera.angles = tuple(a)
            viewer.camera.zoom = zoom0 * (args.zoom_in ** frac)  # zoom in over the clip
            t_idx = int(round(t0 + frac * (t1 - t0)))
            viewer.dims.set_current_step(0, t_idx)
            if frame_update is not None:
                frame_update(t_idx)
            QCoreApplication.processEvents()
            img = viewer.screenshot(canvas_only=True, flash=False, size=(h, w))
            if img.shape[-1] == 4:
                img = img[..., :3]
            if cbar is not None:
                img = overlay_colorbar(img, cbar["cmap"], cbar["clim"], cbar["label"], cbar.get("gamma", 1.0))
            if ts is not None:
                val = float(ts["times"][t_idx]) * ts["scale"]
                img = overlay_timestamp(img, f"t = {val:.{ts['dec']}f} {ts['unit']}")
            if writer is not None:
                writer.append_data(img)
            if frames_dir is not None:
                imageio.imwrite(frames_dir / f"{args.frames_prefix}{i}.png", img)
            if (i + 1) % fps == 0:
                print(f"  frame {i+1}/{n_frames} ({100*(i+1)/n_frames:.0f}%, "
                      f"t={t_idx})", flush=True)
    finally:
        if writer is not None:
            writer.close()
    print(f"[render] done ({n_frames} frames)", flush=True)


def target_center_grid(sim_dir, shape, voxel_mm):
    """Full-grid voxel coords (ci, cj, ck) of the placement target, by inverting
    the report's slab->LPS transform."""
    pj = json.loads((sim_dir / "placement.json").read_text())
    apex = np.array(pj["xdc_center_lps"])
    P = np.array(pj.get("target_lps", pj["focus_lps"]))
    n = np.array(pj["normal_3d"])
    t = np.array(pj["tangent_3d"])
    b = np.array(pj["bitangent_3d"])
    v = P - apex
    ci = float(v @ n) / voxel_mm[0]
    cj = float(v @ t) / voxel_mm[1] + shape[1] / 2.0
    ck = float(v @ b) / voxel_mm[2] + shape[2] / 2.0
    return ci, cj, ck, pj.get("target_name", "?")


def target_roi_mask(sim_dir, shape, voxel_mm, radius_mm):
    """Target ROI on the full sim grid: the warped anatomical ROI
    (``target_roi_slab.npy``, built from the registration atlas) if available,
    else a sphere marker at the placement target.
    """
    cache = sim_dir / "target_roi_slab.npy"
    if cache.exists():
        m = np.load(cache)
        if tuple(m.shape) == tuple(shape):
            print(f"[target] warped anatomical ROI {int((m>0.5).sum())} voxels "
                  f"(target_roi_slab.npy)", flush=True)
            return m.astype(np.float32)
    return target_sphere_mask(sim_dir, shape, voxel_mm, radius_mm)


def target_sphere_mask(sim_dir, shape, voxel_mm, radius_mm):
    """Boolean sphere at the placement target, in full sim-grid voxels.

    A faint static marker of the intended target region (e.g. ACC_left).
    """
    ci, cj, ck, name = target_center_grid(sim_dir, shape, voxel_mm)
    rad_vox = radius_mm / voxel_mm[0]
    ii = np.arange(shape[0], dtype=np.float32)[:, None, None]
    jj = np.arange(shape[1], dtype=np.float32)[None, :, None]
    kk = np.arange(shape[2], dtype=np.float32)[None, None, :]
    mask = ((ii - ci) ** 2 + (jj - cj) ** 2 + (kk - ck) ** 2) <= rad_vox ** 2
    print(f"[target] {name} sphere r={radius_mm:.1f} mm at grid "
          f"({ci:.0f},{cj:.0f},{ck:.0f}); {int(mask.sum())} voxels", flush=True)
    return mask.astype(np.float32)


def peak_field(field_mmap, n_sample=24):
    """Peak |field| and the frame it occurs in (sampled, for clim/initial frame)."""
    nt = field_mmap.shape[0]
    idx = np.unique(np.linspace(0, nt - 1, n_sample).astype(int))
    per = np.array([float(np.abs(field_mmap[i]).max()) for i in idx])
    return per.max(), int(idx[int(per.argmax())])


def build_neural_scene(sim_dir, *, nt=120, tau_ms=1.0, point_floor=0.10, cum_sigma=3.0,
                       rebuild=False):
    """From neural_results.npz, build per-neuron flash data + a smooth cumulative
    firing-density background, on the coarse grid.

    Returns (points_ijk (V,3), point_alpha (nt,V), cum (nt,nx,ny,nz), times_s).
    Each firing voxel becomes a flat point that flashes (exponential decay,
    tau_ms) at its spike times with a faint persistent floor; ``cum`` is the
    running spike-count density, Gaussian-smoothed (cum_sigma) into a smooth
    glowing background that builds up and persists. Cached (savez_compressed).
    """
    cache = sim_dir / "neural_firing_cache.npz"
    if cache.exists() and not rebuild:
        z = np.load(cache)
        return z["points"], z["alpha"], z["cum"], z["times_s"]
    from scipy.ndimage import gaussian_filter

    d = np.load(sim_dir / "neural_results.npz")
    vox = d["spike_voxel_idx"].astype(np.int64)          # (V, 3) coarse-grid idx
    off = d["spike_offsets"]
    st = d["spike_times_ms"].astype(np.float64)          # (S,)
    dur = float(d["duration_ms"])
    shape = tuple(int(s) for s in d["firing_map"].shape)  # (106, 65, 65)
    V = len(vox)
    spk_pt = np.repeat(np.arange(V), np.diff(off))        # spike -> firing-voxel
    times_ms = np.linspace(0.0, dur, nt)

    # Per-point flash alpha (nt, V): decayed sum of past spikes + persistent floor.
    alpha = np.zeros((nt, V), np.float32)
    for f, t in enumerate(times_ms):
        m = st <= t
        if not m.any():
            continue
        a = np.zeros(V)
        np.add.at(a, spk_pt[m], np.exp(-(t - st[m]) / tau_ms))
        fired = np.zeros(V, bool)
        fired[spk_pt[m]] = True
        alpha[f] = np.clip(np.maximum(a, point_floor * fired), 0.0, 1.0)

    # Smooth cumulative firing-density background (nt, nx, ny, nz).
    n = int(np.prod(shape))
    lin = (vox[:, 0] * shape[1] * shape[2] + vox[:, 1] * shape[2] + vox[:, 2])
    spk_lin = lin[spk_pt]
    order = np.argsort(st)
    st_s, lin_s = st[order], spk_lin[order]
    cum = np.zeros((nt, n), np.float32)
    running = np.zeros(n, np.float32)
    si = 0
    for f, t in enumerate(times_ms):
        while si < len(st_s) and st_s[si] <= t:
            running[lin_s[si]] += 1.0
            si += 1
        cum[f] = running
    cum = cum.reshape(nt, *shape)
    for f in range(nt):
        cum[f] = gaussian_filter(cum[f], cum_sigma)        # smooth background

    times_s = times_ms * 1e-3
    np.savez_compressed(cache, points=vox, alpha=alpha, cum=cum, times_s=times_s)
    print(f"[neural] {V} firing voxels, {len(st)} spikes; alpha+cum; "
          f"cached {cache.name}", flush=True)
    return vox, alpha, cum, times_s


def finalize_render(viewer, args, nt, first_frame, *, cbar=None, ts=None, frame_update=None):
    """Shared tail: render an MP4/frames, save a screenshot, or open the GUI.

    ``frame_update(t_idx)`` (optional) is called each frame to update non-image
    layers (e.g. the flashing neural Points layer) alongside the time slider.
    """
    import napari

    if args.render or args.frames_dir:
        render_video(viewer, args, nt, cbar=cbar, ts=ts, frame_update=frame_update)
        viewer.close()
        return
    if args.screenshot:
        from qtpy.QtWidgets import QApplication

        app = QApplication.instance()
        if frame_update is not None:
            frame_update(int(first_frame))
        for _ in range(12):
            if app is not None:
                app.processEvents()
            time.sleep(0.1)
        img = viewer.screenshot(canvas_only=True, flash=False, size=(1000, 1000))
        if img.shape[-1] == 4:
            img = img[..., :3]
        if cbar is not None:
            img = overlay_colorbar(img, cbar["cmap"], cbar["clim"], cbar["label"], cbar.get("gamma", 1.0))
        if ts is not None:
            val = float(ts["times"][int(first_frame)]) * ts["scale"]
            img = overlay_timestamp(img, f"t = {val:.{ts['dec']}f} {ts['unit']}")
        import matplotlib.image as mpimg

        mpimg.imsave(args.screenshot, img)
        print(f"[screenshot] saved {args.screenshot} (frame {first_frame})", flush=True)
    if args.no_gui:
        viewer.close()
    else:
        napari.run()


def run_neural(args, cfg, sim_dir, c, voxel_mm, mod):
    """Neural-firing visualization: soft flat flashing-point neurons over a
    smooth cumulative firing-density background, in the skull."""
    import napari

    pts, alpha, cum, times_s = build_neural_scene(sim_dir, rebuild=args.rebuild_field)
    nt = cum.shape[0]
    print(f"[skull] {c.shape} @ {voxel_mm[0]:.3f} mm | [neural] {len(pts)} points "
          f"| modulo={mod}", flush=True)

    viewer = napari.Viewer(
        ndisplay=3, title=f"{getattr(cfg,'SIM_NAME','sim')} — neural firing through skull"
    )
    viewer.add_image(
        (np.asarray(c) > BONE_C_THRESHOLD).astype(np.float32),
        name=f"skull bone (c>{BONE_C_THRESHOLD:.0f})", scale=voxel_mm,
        colormap="gray", rendering="iso", iso_threshold=0.5, opacity=0.25,
    )
    if args.target_roi:
        viewer.add_image(
            target_roi_mask(sim_dir, c.shape, voxel_mm, args.target_radius_mm),
            name=f"target ROI ({args.target_radius_mm:.0f} mm)", scale=voxel_mm,
            colormap="green", rendering="iso", iso_threshold=0.5,
            opacity=args.target_opacity, blending="translucent",
        )
    # Smooth cumulative firing-density background (warm glow that builds up).
    cdata, cscale = build_volume_data(cum, c.shape, voxel_mm, mod, args.interp, label="cum-firing")
    cfin = cum[-1]
    nz = cfin[cfin > 0]
    chi = (float(np.percentile(nz, 99.0)) if nz.size else 1.0) * args.cum_clim_frac
    viewer.add_image(
        cdata, name="cumulative firing density", scale=cscale,
        colormap=intensity_glow_colormap(), contrast_limits=(0.0, chi),
        rendering="attenuated_mip", attenuation=0.02, blending="additive",
        interpolation3d=args.interp3d,
    )
    # Soft flat flashing-point neurons (one per firing voxel) on top. Translucent
    # + semi-transparent + NO spherical shading => soft flat flashes (not solid
    # balls); coords are coarse-grid indices, scale=mod*dx places them on the
    # full skull grid (point i -> full voxel mod*i).
    pt_scale = tuple(mod[i] * voxel_mm[i] for i in range(3))
    base = np.array([0.6, 0.95, 1.0], np.float32)  # cyan
    size_data = args.point_size_mm / pt_scale[0]

    def face_for(frame):
        fc = np.zeros((len(pts), 4), np.float32)
        fc[:, :3] = base
        fc[:, 3] = alpha[min(frame, nt - 1)] * args.point_alpha
        return fc

    layer = viewer.add_points(
        pts, name="neurons (firing)", scale=pt_scale, size=size_data,
        face_color=face_for(0), border_width=0.0, blending="translucent",
    )

    def frame_update(t_idx):
        layer.face_color = face_for(int(t_idx))

    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = "mm"
    viewer.reset_view()
    viewer.camera.angles = tuple(float(a) for a in args.cam_angles.split(","))
    viewer.camera.zoom *= args.zoom
    if args.center_focus:
        ci, cj, ck, _ = target_center_grid(sim_dir, c.shape, voxel_mm)
        viewer.camera.center = (ci * voxel_mm[0], cj * voxel_mm[1], ck * voxel_mm[2])
    first = int(args.frame) if args.frame is not None else 0
    viewer.dims.set_current_step(0, first)
    frame_update(first)

    ts = None
    if not args.no_timestamp:
        tscale, tunit, tdec = timestamp_format(times_s)
        ts = {"times": times_s, "scale": tscale, "unit": tunit, "dec": tdec}

    finalize_render(viewer, args, nt, first, cbar=None, ts=ts, frame_update=frame_update)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="validation_report/configs/halle_dacc.py")
    ap.add_argument("--field", choices=["pressure", "shear", "temperature", "neural"],
                    default="pressure",
                    help="which field to render: 'pressure' (acoustic p, signed), "
                         "'shear' (displacement |u|), 'temperature' (bioheat ΔT), or "
                         "'neural' (flashing neurons + cumulative firing density)")
    ap.add_argument("--point-size-mm", type=float, default=0.63,
                    help="neuron point diameter in mm (--field neural)")
    ap.add_argument("--point-alpha", type=float, default=0.6,
                    help="per-neuron flash opacity (--field neural); <1 keeps the "
                         "translucent flashes soft (flat, not solid balls)")
    ap.add_argument("--rebuild-field", action="store_true",
                    help="recompute the derived field cache (e.g. displacement_mag.npy)")
    ap.add_argument("--interp", choices=["gpu", "cpu"], default="gpu")
    ap.add_argument("--pressure-cmap", choices=list(PRESSURE_CMAPS), default="cool",
                    help="pressure wave colormap: 'cool' (symmetric cyan |p| envelope, "
                         "pairs with warm cumulative), 'hot', or 'diverging' (blue/red phase)")
    ap.add_argument("--clim-frac", type=float, default=0.5,
                    help="contrast limit as fraction of peak |field| (lower => fainter "
                         "values more visible)")
    ap.add_argument("--gamma", type=float, default=1.0,
                    help="display gamma for the live field (<1 lifts low values so a "
                         "weak focal signal isn't buried under a strong peak, e.g. 0.4 "
                         "for temperature where the skull dominates ΔT)")
    ap.add_argument("--frame", type=int, default=None,
                    help="time frame to display first (default: focal-arrival peak)")
    ap.add_argument("--zoom", type=float, default=0.8,
                    help="zoom factor relative to reset_view fit (lower = more out; "
                         "use ~0.6 for renders so the skull doesn't clip while rotating)")
    ap.add_argument("--cam-angles", default="15,-40,70",
                    help="initial camera Euler angles 'a0,a1,a2' in degrees "
                         "(near-face-on slight tilt e.g. '8,-12,84')")
    ap.add_argument("--cam-angles-end", default=None,
                    help="end camera Euler angles 'a0,a1,a2'; if set, the render "
                         "sweeps cam-angles -> cam-angles-end (overrides --rotate-deg)")
    ap.add_argument("--zoom-in", type=float, default=1.0,
                    help="zoom-in factor over the clip (>1 zooms in toward the view "
                         "centre; e.g. 2.5)")
    ap.add_argument("--center-focus", action="store_true",
                    help="centre the camera on the placement target/focus (so "
                         "zoom-in/orbit pivots on the focal volume)")
    ap.add_argument("--no-cumulative", action="store_true",
                    help="skip the cumulative time-integrated intensity overlay")
    ap.add_argument("--cum-cmap", choices=list(CUM_CMAPS), default=None,
                    help="cumulative background colormap (default: 'cool' when "
                         "pressure is hot, else 'glow' -- so the layers contrast)")
    ap.add_argument("--cum-clim-frac", type=float, default=1.0,
                    help="scale the cumulative-intensity upper contrast limit "
                         "(default uses the 99th pct of the final frame)")
    ap.add_argument("--rebuild-cum", action="store_true",
                    help="recompute the cached cumulative_intensity.npy")
    ap.add_argument("--interp3d", choices=["cubic", "linear", "nearest"], default="cubic",
                    help="napari 3-D texture interpolation (cubic = smoothest, "
                         "reduces the near-Nyquist faceting; default cubic)")
    ap.add_argument("--smooth", type=float, default=0.0,
                    help="anti-alias: Gaussian sigma (in coarse voxels) applied per "
                         "pressure frame before rendering (e.g. 0.6); 0 = off")
    ap.add_argument("--target-roi", action="store_true",
                    help="overlay a faint static sphere at the placement target (ROI marker)")
    ap.add_argument("--target-radius-mm", type=float, default=6.0,
                    help="radius of the target sphere marker (mm)")
    ap.add_argument("--target-opacity", type=float, default=0.3,
                    help="opacity of the target ROI overlay")
    ap.add_argument("--no-colorbar", action="store_true",
                    help="do not overlay a colorbar on screenshots/renders")
    ap.add_argument("--no-timestamp", action="store_true",
                    help="do not overlay a per-frame physical-time stamp")
    ap.add_argument("--no-gui", action="store_true")
    ap.add_argument("--screenshot", default=None)
    # --- video render (slow-rotating fly-around while time advances) ---
    ap.add_argument("--render", default=None,
                    help="render an MP4 to this path (slow camera orbit + time sweep)")
    ap.add_argument("--frames-dir", default=None,
                    help="also/instead write a numbered PNG sequence to this dir "
                         "(for LaTeX \\animategraphics in the validation report)")
    ap.add_argument("--frames-prefix", default="frame",
                    help="basename for the PNG sequence (<prefix>0.png ...)")
    ap.add_argument("--render-seconds", type=float, default=12.0)
    ap.add_argument("--render-fps", type=int, default=30)
    ap.add_argument("--render-size", default="1000x1000", help="WxH pixels")
    ap.add_argument("--rotate-deg", type=float, default=360.0,
                    help="total camera rotation over the clip (deg); lower = slower")
    ap.add_argument("--rotate-axis", type=int, default=0, choices=[0, 1, 2],
                    help="primary Euler axis to orbit around")
    ap.add_argument("--rotate-deg2", type=float, default=0.0,
                    help="optional simultaneous rotation on a second axis (deg); 0 = off")
    ap.add_argument("--rotate-axis2", type=int, default=1, choices=[0, 1, 2],
                    help="second Euler axis (used when --rotate-deg2 != 0)")
    ap.add_argument("--time-start", type=int, default=0,
                    help="first propagation frame to sweep from")
    ap.add_argument("--time-end", type=int, default=-1,
                    help="last propagation frame to sweep to (-1 = last)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    sim_dir = REPO / cfg.SIM_DATA_DIR

    c = np.load(sim_dir / "skull_fullres_c.npy", mmap_mode="r")
    meta = json.loads((sim_dir / "skull_fullres_meta.json").read_text())
    voxel_mm = (meta["dx_mm"], meta["dy_mm"], meta["dz_mm"])

    gp = np.load(sim_dir / "grid_params.npz")
    mod = (int(gp["mod_x"]), int(gp["mod_y"]), int(gp["mod_z"]))

    if args.field == "neural":
        run_neural(args, cfg, sim_dir, c, voxel_mm, mod)
        return

    p, spec = load_field(sim_dir, args.field, rebuild=args.rebuild_field)
    print(f"[skull] {c.shape} @ {voxel_mm[0]:.3f} mm | [{args.field}] {p.shape} {p.dtype} "
          f"| modulo (mod_x,y,z)={mod}", flush=True)

    peak, peak_frame = peak_field(p)
    clim_val = args.clim_frac * peak
    first_frame = args.frame if args.frame is not None else peak_frame
    if spec["kind"] == "signed":  # oscillatory -> symmetric colormap, transparent middle
        clim = (-clim_val, clim_val)
        live_cmap = PRESSURE_CMAPS[args.pressure_cmap]()
    else:  # magnitude (>= 0) -> sequential, transparent over the low range
        clim = (0.0, clim_val)
        live_cmap = (temperature_colormap() if args.field == "temperature"
                     else displacement_colormap())
    print(f"[{args.field}] peak |{spec['unit']}| ~ {peak:.3e} {spec['unit']} "
          f"(frame ~{peak_frame}); clim = {clim[0]:.2e}..{clim[1]:.2e} {spec['unit']}", flush=True)

    pdata, pscale = build_volume_data(p, c.shape, voxel_mm, mod, args.interp,
                                      label=args.field, smooth_sigma=args.smooth)

    cum = cum_clim = cdata = cscale = None
    if not args.no_cumulative:
        cum = get_cumulative(sim_dir, p, spec["cum_cache"], rebuild=args.rebuild_cum)
        cum_final = np.asarray(cum[-1])
        nz = cum_final[cum_final > 0]
        cum_hi = (float(np.percentile(nz, 99.0)) if nz.size else 1.0) * args.cum_clim_frac
        cum_clim = (0.0, cum_hi)
        cdata, cscale = build_volume_data(cum, c.shape, voxel_mm, mod, args.interp, label="cumulative")
        print(f"[cumulative] clim = (0, {cum_hi:.2e}) (persistent background)", flush=True)

    import napari

    viewer = napari.Viewer(
        ndisplay=3,
        title=f"{getattr(cfg,'SIM_NAME','sim')} — {spec['title']} through skull",
    )
    # Faint skull bone iso for spatial context.
    viewer.add_image(
        (np.asarray(c) > BONE_C_THRESHOLD).astype(np.float32),
        name=f"skull bone (c>{BONE_C_THRESHOLD:.0f})",
        scale=voxel_mm,
        colormap="gray",
        rendering="iso",
        iso_threshold=0.5,
        opacity=0.25,
    )
    # Faint static target ROI marker (sphere at the placement target). Additive
    # blending so the green glow shows through even when the (opaque) field
    # volume sits in front of it at the focus.
    if args.target_roi:
        viewer.add_image(
            target_roi_mask(sim_dir, c.shape, voxel_mm, args.target_radius_mm),
            name="target ROI",
            scale=voxel_mm,
            colormap="green",
            rendering="iso",
            iso_threshold=0.5,
            opacity=args.target_opacity,
            blending="translucent",
        )
    # Cumulative time-integral (persistent glowing background, added below the
    # live field so the field reads on top). Pick a warm glow unless the live
    # field is the warm 'hot' pressure (then go cool) so the two layers contrast.
    if cdata is not None:
        live_is_warm = spec["kind"] == "signed" and args.pressure_cmap == "hot"
        cum_cmap = args.cum_cmap or ("cool" if live_is_warm else "glow")
        viewer.add_image(
            cdata,
            name=spec["cum_layer"],
            scale=cscale,
            colormap=CUM_CMAPS[cum_cmap](),
            contrast_limits=cum_clim,
            rendering="attenuated_mip",
            attenuation=0.02,
            blending="additive",
            interpolation3d=args.interp3d,
        )
    # Live field (4-D: time + volume).
    viewer.add_image(
        pdata,
        name=spec["layer"],
        scale=pscale,
        colormap=live_cmap,
        contrast_limits=clim,
        gamma=args.gamma,
        rendering="translucent",
        blending="translucent",
        interpolation3d=args.interp3d,
    )
    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = "mm"
    viewer.reset_view()
    viewer.camera.angles = tuple(float(a) for a in args.cam_angles.split(","))
    viewer.camera.zoom *= args.zoom
    # Jump to the focal-arrival frame (axis 0 is time).
    viewer.dims.set_current_step(0, int(first_frame))

    # Colorbar (overlaid on screenshots/renders) for the live field, in display units.
    cbar = None
    if not args.no_colorbar:
        sc = spec["cbar_scale"]
        cbar = {"cmap": live_cmap, "clim": (clim[0] * sc, clim[1] * sc),
                "label": spec["cbar_label"], "gamma": args.gamma}

    # Per-frame physical-time stamp.
    ts = None
    if not args.no_timestamp:
        times_s = frame_times(args.field, sim_dir, cfg, gp, p.shape[0])
        if times_s is not None:
            tscale, tunit, tdec = timestamp_format(times_s)
            ts = {"times": times_s, "scale": tscale, "unit": tunit, "dec": tdec}
            print(f"[time] frame dt ~ {np.median(np.abs(np.diff(times_s)))*tscale:.{tdec}f} "
                  f"{tunit}; stamping each frame", flush=True)

    finalize_render(viewer, args, p.shape[0], first_frame, cbar=cbar, ts=ts)


if __name__ == "__main__":
    main()
