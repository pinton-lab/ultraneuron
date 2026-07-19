#!/usr/bin/env python3
"""Render the full-resolution skull inside the FDTD simulation domain in napari.

This rebuilds the *exact* medium the acoustic solver used — fullwave
``MediumBuilder`` + ``MicroCTSkullDomain`` driven by the persisted
``placement.json`` — but at the full simulation resolution (no ``mod_x``
downsampling; the solver saves only the 4x-decimated ``c_ds`` maps). The
sound-speed map (and a bone mask) are then shown as a 3-D volume in napari with
physical (mm) voxel spacing.

Unlike the rest of the pipeline this runs under the *system* ``python3``, which
is the interpreter that carries napari / PyQt6 / vispy / pynrrd as well as a
fullwave install (``$FULLWAVE_ROOT``, default ``~/fullwave25-speckle``). The
repo ``.venv`` has none of these.

The rebuilt full-res fields are cached next to the sim data
(``skull_fullres_c.npy`` / ``_rho.npy`` + ``_meta.json``) so subsequent runs —
and future movie work — load instantly. Use ``--rebuild`` to force a fresh
extraction from the 2.8 GB microCT NRRD.

Examples
--------
    # build (or load cache) + open the live napari GUI
    python3 scripts/napari_skull_domain.py --config validation_report/configs/halle_dacc.py

    # headless: build/cache + save a verification screenshot, no GUI
    python3 scripts/napari_skull_domain.py --config validation_report/configs/halle_dacc.py \
        --no-gui --screenshot /tmp/skull_dacc.png

    # just (re)build the cache, no napari at all
    python3 scripts/napari_skull_domain.py --config validation_report/configs/halle_dacc.py \
        --build-only --rebuild
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

FULLWAVE_ROOT = os.environ.get("FULLWAVE_ROOT", str(Path.home() / "fullwave25-speckle"))
REPO = Path(__file__).resolve().parents[1]

# Skull-bone sound-speed cutoff used throughout the pipeline for overlays
# (e.g. bowl_transcranial_500khz.py: ``c_ds > 1600``).
BONE_C_THRESHOLD = 1600.0


def load_config(config_path: str):
    """Import a validation-report config module from its file path."""
    path = Path(config_path)
    if not path.is_absolute():
        path = REPO / path
    spec = importlib.util.spec_from_file_location("sim_config", path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    return cfg


def build_fullres_medium(cfg):
    """Rebuild the full-resolution (sound_speed, density) medium on the sim grid.

    Mirrors the medium-build block in scripts/bowl_transcranial_500khz.py
    (Grid -> Placement -> MicroCTSkullDomain -> MediumBuilder -> exponential),
    but keeps the full grid instead of decimating by mod_x/mod_y/mod_z.
    """
    if FULLWAVE_ROOT not in sys.path:
        sys.path.insert(0, FULLWAVE_ROOT)
    import fullwave
    from fullwave.medium_builder import MediumBuilder
    from fullwave.medium_builder.presets import BackgroundDomain, MicroCTSkullDomain
    from fullwave.utils.slab_extraction import Placement

    ds = tuple(cfg.DOMAIN_SIZE)
    grid = fullwave.Grid(
        ds, cfg.F0, ds[0] / cfg.C0 * 2, c0=cfg.C0, ppw=cfg.PPW, cfl=cfg.CFL
    )
    print(f"[grid] {grid.nx} x {grid.ny} x {grid.nz}  dx={grid.dx*1e3:.4f} mm", flush=True)

    sim_dir = REPO / cfg.SIM_DATA_DIR
    pj = json.loads((sim_dir / "placement.json").read_text())
    if all(k in pj for k in ("normal_3d", "tangent_3d", "bitangent_3d")):
        placement = Placement(
            xdc_center_lps=tuple(pj["xdc_center_lps"]),
            normal=tuple(pj["normal_3d"]),
            tangent=tuple(pj["tangent_3d"]),
            bitangent=tuple(pj["bitangent_3d"]),
            z_mm=pj["z_mm"],
        )
    else:  # legacy 2-D pose
        placement = Placement(
            xdc_center_lps=tuple(pj["xdc_center_lps"]),
            normal=tuple(pj["normal"]),
            tangent=tuple(pj["tangent"]),
            z_mm=pj["z_mm"],
        )

    med = cfg.MEDIUM
    print(f"[skull] extracting from {med['nrrd_path']}", flush=True)
    skull = MicroCTSkullDomain(
        grid=grid,
        nrrd_path=med["nrrd_path"],
        placement=placement,
        lower_threshold=med["lower_threshold"],
        upper_threshold=med["upper_threshold"],
        speed_range=tuple(med["speed_range"]),
        density_range=tuple(med["density_range"]),
        blur_sigma_m=med.get("blur_sigma_m", 0.0),
    )
    mb = MediumBuilder(grid=grid)
    mb.register_domain(
        BackgroundDomain(grid=grid, background_property_name=med.get("background", "tissue"))
    )
    mb.register_domain(skull)
    medium = mb.run()
    medium_exp = medium.build_exponential()

    c = np.ascontiguousarray(medium_exp.sound_speed, dtype=np.float32)
    rho = np.ascontiguousarray(medium_exp.density, dtype=np.float32)
    return grid, c, rho


def get_fields(cfg, *, rebuild: bool):
    """Return (c, rho, voxel_mm) from cache, rebuilding if needed."""
    sim_dir = REPO / cfg.SIM_DATA_DIR
    c_path = sim_dir / "skull_fullres_c.npy"
    rho_path = sim_dir / "skull_fullres_rho.npy"
    meta_path = sim_dir / "skull_fullres_meta.json"

    if not rebuild and c_path.exists() and rho_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        print(f"[cache] loading {c_path.name} {tuple(meta['shape'])}", flush=True)
        c = np.load(c_path)
        rho = np.load(rho_path)
        return c, rho, (meta["dx_mm"], meta["dy_mm"], meta["dz_mm"])

    t0 = time.time()
    grid, c, rho = build_fullres_medium(cfg)
    voxel_mm = (grid.dx * 1e3, grid.dy * 1e3, grid.dz * 1e3)
    np.save(c_path, c)
    np.save(rho_path, rho)
    meta_path.write_text(
        json.dumps(
            {
                "shape": list(c.shape),
                "dx_mm": voxel_mm[0],
                "dy_mm": voxel_mm[1],
                "dz_mm": voxel_mm[2],
                "sim_data_dir": cfg.SIM_DATA_DIR,
                "sim_name": getattr(cfg, "SIM_NAME", "?"),
            },
            indent=2,
        )
    )
    print(f"[cache] built + saved in {time.time()-t0:.1f}s -> {c_path.name}", flush=True)
    return c, rho, voxel_mm


def describe(c, rho, voxel_mm):
    bone = c > BONE_C_THRESHOLD
    nx, ny, nz = c.shape
    print(
        f"[fields] c {c.shape}  range {c.min():.0f}-{c.max():.0f} m/s | "
        f"rho range {rho.min():.0f}-{rho.max():.0f} kg/m^3",
        flush=True,
    )
    print(
        f"[bone]  {bone.sum():,} voxels ({100*bone.mean():.2f}% of domain) above "
        f"{BONE_C_THRESHOLD:.0f} m/s",
        flush=True,
    )
    print(
        f"[domain] {nx*voxel_mm[0]:.1f} x {ny*voxel_mm[1]:.1f} x {nz*voxel_mm[2]:.1f} mm "
        f"@ {voxel_mm[0]:.3f} mm/voxel",
        flush=True,
    )


def render(c, rho, voxel_mm, *, sim_name, no_gui, screenshot):
    import napari

    scale = tuple(voxel_mm)  # axes are (x, y, z) voxels -> mm
    # NB: the 3-D vispy/GL canvas only paints once it is realized on a display,
    # so we always create a shown window (DISPLAY is available) and just close it
    # afterwards in headless/screenshot mode. A purely off-screen (show=False)
    # 3-D canvas renders black.
    viewer = napari.Viewer(
        ndisplay=3,
        title=f"{sim_name} — full-res skull in simulation domain",
    )
    # Sound-speed volume: tissue (~1540) fades out, bone (1600-2900) lights up.
    viewer.add_image(
        c,
        name="sound speed [m/s]",
        scale=scale,
        colormap="gray",
        contrast_limits=(BONE_C_THRESHOLD, float(c.max())),
        rendering="attenuated_mip",
        attenuation=0.04,
    )
    # Crisp bone iso-surface on top.
    viewer.add_image(
        (c > BONE_C_THRESHOLD).astype(np.float32),
        name=f"bone mask (c>{BONE_C_THRESHOLD:.0f})",
        scale=scale,
        colormap="gray",
        rendering="iso",
        iso_threshold=0.5,
        opacity=0.6,
    )
    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = "mm"
    viewer.reset_view()
    # Open at an oblique 3-D view (not face-on down the beam axis).
    viewer.camera.angles = (15, -40, 70)
    viewer.camera.zoom = viewer.camera.zoom * 0.8

    if screenshot:
        # Pump the Qt event loop so the GL canvas realizes and paints before grab.
        from qtpy.QtWidgets import QApplication

        app = QApplication.instance()
        for _ in range(10):
            if app is not None:
                app.processEvents()
            time.sleep(0.1)
        img = viewer.screenshot(canvas_only=True, flash=False, size=(1000, 1000))
        import matplotlib.image as mpimg

        mpimg.imsave(screenshot, img)
        print(f"[screenshot] saved {screenshot}", flush=True)

    if no_gui:
        viewer.close()
    else:
        napari.run()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="validation_report/configs/halle_dacc.py")
    p.add_argument("--rebuild", action="store_true", help="force fresh NRRD extraction")
    p.add_argument("--build-only", action="store_true", help="build/cache only, no napari")
    p.add_argument("--no-gui", action="store_true", help="do not open the GUI (offscreen)")
    p.add_argument("--screenshot", default=None, help="save a PNG render to this path")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    c, rho, voxel_mm = get_fields(cfg, rebuild=args.rebuild)
    describe(c, rho, voxel_mm)

    if args.build_only:
        return
    render(
        c,
        rho,
        voxel_mm,
        sim_name=getattr(cfg, "SIM_NAME", "sim"),
        no_gui=args.no_gui,
        screenshot=args.screenshot,
    )


if __name__ == "__main__":
    main()
