"""CTX-500 bowl geometry visualization on the Halle skull grid.

Builds a NeuroFUS CTX-500 spec bowl (64 mm aperture, 63 mm ROC, 4 annular
elements, no central hole) on the same 65 x 66 x 66 mm domain used by the
sparse-array transcranial pipeline, then renders:

  fig_a_bowl_surface.png   — bowl source mask: y-z face, x-y projection, x-z projection
  fig_a2_bowl_over_skull.png — bowl source mask overlaid on the skull medium (x-y and x-z slices)
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

import fullwave
from fullwave.medium_builder import MediumBuilder
from fullwave.medium_builder.presets import BackgroundDomain, MicroCTSkullDomain
from fullwave.transducers.tips import create_tips_transducer
from fullwave.utils.slab_extraction import Placement

from ultraneuron.paths import halle_skull_nrrd
from ultraneuron.registration import ensure_registration_importable

ensure_registration_importable()
from halle_placement import place_on_skull

# CTX-500 spec (Yaakub 2023 / Sonic Concepts NeuroFUS)
CTX500_OUTER_RADIUS_M = 32e-3   # 64 mm aperture
CTX500_INNER_RADIUS_M = 0.0     # no central hole
CTX500_ROC_M = 63e-3            # geometric focus
CTX500_N_ELEMENTS = 4           # annular axial-steering subdivision
CTX500_N_LAYERS = 3


def main(nrrd_path: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    f0 = 500e3
    c0 = 1540
    domain_size = (65e-3, 66e-3, 66e-3)
    duration = domain_size[0] / c0 * 2
    grid = fullwave.Grid(domain_size, f0, duration, c0=c0, ppw=10, cfl=0.2)
    grid.print_info()

    bowl = create_tips_transducer(
        grid=grid,
        n_layers=CTX500_N_LAYERS,
        roc_m=CTX500_ROC_M,
        inner_radius_m=CTX500_INNER_RADIUS_M,
        outer_radius_m=CTX500_OUTER_RADIUS_M,
        n_elements=CTX500_N_ELEMENTS,
    )

    coords = bowl.source_coords_px
    print(
        f"Bowl: {coords.shape[0]} source voxels across {bowl.n_elements} annular elements\n"
        f"  depth (x):       {coords[:, 0].min()} -> {coords[:, 0].max()} px "
        f"({coords[:, 0].min() * grid.dx * 1e3:.1f} -> {coords[:, 0].max() * grid.dx * 1e3:.1f} mm)\n"
        f"  lateral (y):     {coords[:, 1].min()} -> {coords[:, 1].max()} px\n"
        f"  elevational(z):  {coords[:, 2].min()} -> {coords[:, 2].max()} px"
    )

    # ---------------- fig_a: bowl alone -----------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    yz_face = bowl.source_mask.any(axis=0)
    elem_yz = np.zeros((grid.ny, grid.nz), dtype=np.int32)
    elem_yz[coords[:, 1], coords[:, 2]] = bowl.element_ids
    axes[0].imshow(elem_yz.T, origin="lower", cmap="tab10",
                   extent=[0, grid.ny * grid.dy * 1e3, 0, grid.nz * grid.dz * 1e3])
    axes[0].set_title("Bowl footprint (y-z, colored by element)")
    axes[0].set_xlabel("lateral y [mm]")
    axes[0].set_ylabel("elevational z [mm]")
    axes[0].set_aspect("equal")

    xy_proj = bowl.source_mask.any(axis=2)
    axes[1].imshow(xy_proj.T, origin="lower", cmap="gray_r", aspect="auto",
                   extent=[0, grid.nx * grid.dx * 1e3, 0, grid.ny * grid.dy * 1e3])
    axes[1].set_title("Bowl x-y projection")
    axes[1].set_xlabel("depth x [mm]")
    axes[1].set_ylabel("lateral y [mm]")

    xz_proj = bowl.source_mask.any(axis=1)
    axes[2].imshow(xz_proj.T, origin="lower", cmap="gray_r", aspect="auto",
                   extent=[0, grid.nx * grid.dx * 1e3, 0, grid.nz * grid.dz * 1e3])
    axes[2].set_title("Bowl x-z projection")
    axes[2].set_xlabel("depth x [mm]")
    axes[2].set_ylabel("elevational z [mm]")

    fig.suptitle(f"(a) CTX-500 bowl — 64 mm aperture, 63 mm ROC, 4 annular elements ({coords.shape[0]} voxels)")
    fig.tight_layout()
    out_a = out_dir / "fig_a_bowl_surface.png"
    fig.savefig(out_a, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_a}")

    # ---------------- fig_a2: bowl over skull ------------
    p = place_on_skull(
        scalp_site="F3",
        target_name="DLPFC_left",
        focal_length_mm=CTX500_ROC_M * 1e3,
        bowl_radius_mm=CTX500_OUTER_RADIUS_M * 1e3,
    )
    print("Placement (DLPFC_left @ F3):")
    for k in ("xdc_center_lps", "scalp_contact_lps", "focus_lps", "target_lps",
              "perp_residual_mm", "scalp_target_dist_mm", "apex_target_dist_mm",
              "bowl_depth_mm", "eeg_to_placement_mm"):
        print(f"  {k}: {p[k]}")
    placement = Placement(
        xdc_center_lps=tuple(p["xdc_center_lps"]),
        normal=tuple(p["normal"]),
        tangent=tuple(p["tangent"]),
        z_mm=p["z_mm"],
    )
    skull_domain = MicroCTSkullDomain(
        grid=grid,
        nrrd_path=nrrd_path,
        placement=placement,
        lower_threshold=700,
        upper_threshold=1973,
        speed_range=(1540, 2900),
        density_range=(1000, 2200),
        blur_sigma_m=0e-4,
    )
    background_domain = BackgroundDomain(grid=grid, background_property_name="tissue")
    mb = MediumBuilder(grid=grid)
    mb.register_domain(background_domain)
    mb.register_domain(skull_domain)
    medium = mb.run()

    c_map = np.asarray(medium.sound_speed)
    iy_mid = grid.ny // 2
    iz_mid = grid.nz // 2

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].imshow(c_map[:, :, iz_mid].T, origin="lower", cmap="bone", aspect="equal",
                   extent=[0, grid.nx * grid.dx * 1e3, 0, grid.ny * grid.dy * 1e3])
    bowl_xy = bowl.source_mask[:, :, :].any(axis=2)
    axes[0].contour(bowl_xy.T, levels=[0.5], colors="red", linewidths=1.0,
                    extent=[0, grid.nx * grid.dx * 1e3, 0, grid.ny * grid.dy * 1e3])
    axes[0].axhline(grid.ny * grid.dy * 1e3 / 2, color="cyan", lw=0.5, alpha=0.5)
    axes[0].axvline(CTX500_ROC_M * 1e3, color="orange", lw=0.7, alpha=0.7,
                    linestyle="--", label=f"geom. focus (ROC={CTX500_ROC_M*1e3:.0f} mm)")
    axes[0].set_title("Bowl + skull — x-y slice (mid z)")
    axes[0].set_xlabel("depth x [mm]")
    axes[0].set_ylabel("lateral y [mm]")
    axes[0].legend(loc="lower right", fontsize=8)

    axes[1].imshow(c_map[:, iy_mid, :].T, origin="lower", cmap="bone", aspect="equal",
                   extent=[0, grid.nx * grid.dx * 1e3, 0, grid.nz * grid.dz * 1e3])
    bowl_xz = bowl.source_mask[:, :, :].any(axis=1)
    axes[1].contour(bowl_xz.T, levels=[0.5], colors="red", linewidths=1.0,
                    extent=[0, grid.nx * grid.dx * 1e3, 0, grid.nz * grid.dz * 1e3])
    axes[1].axhline(grid.nz * grid.dz * 1e3 / 2, color="cyan", lw=0.5, alpha=0.5)
    axes[1].axvline(CTX500_ROC_M * 1e3, color="orange", lw=0.7, alpha=0.7,
                    linestyle="--", label=f"geom. focus (ROC={CTX500_ROC_M*1e3:.0f} mm)")
    axes[1].set_title("Bowl + skull — x-z slice (mid y)")
    axes[1].set_xlabel("depth x [mm]")
    axes[1].set_ylabel("elevational z [mm]")
    axes[1].legend(loc="lower right", fontsize=8)

    fig.suptitle("(a2) CTX-500 bowl over Halle micro-CT skull")
    fig.tight_layout()
    out_a2 = out_dir / "fig_a2_bowl_over_skull.png"
    fig.savefig(out_a2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_a2}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nrrd-path", default=str(halle_skull_nrrd()))
    parser.add_argument("--out-dir", default="data/halle_transcranial")
    args = parser.parse_args()
    main(nrrd_path=args.nrrd_path, out_dir=Path(args.out_dir))
