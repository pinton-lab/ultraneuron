"""Section 4 — Elastic outputs.

Builds per-voxel maps of three quantities derived from the static
shear-FDTD equilibrium:

  * Acoustic radiation force density (ARF) — read from
    ``radiation_force.npy``; the body force F = 2 alpha I / c that
    drives the elastostatic solve.
  * Equivalent deviatoric (von Mises) strain — computed from the
    saved 3-component displacement field via central differences on
    the strain tensor; isolates shape change from volumetric change.
  * Membrane tension proxy — K_a * |epsilon_vol| with K_a the lipid
    bilayer area-expansion modulus (default 0.4 N/m); a scalar
    surrogate for the bilayer-stretch term that drives capacitance
    modulation in the NICE / Plaksin–Kimmel cavitation model.

Each field is rendered with the same view set used in the acoustic
outputs section: three orthogonal plane slices through the peak
voxel with a cyan skull contour overlay (depth on the vertical axis,
zero at the top — radiology convention) and a three-camera 3-D
iso-surface rendering.  Metrics include peak magnitude, brain-only
peak, and focal-region centroid.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

# Make the top-level package importable when running this builder from
# inside validation_report (repo isn't pip-installed).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from . import acoustic_outputs as _ao


# =====================================================================
# Field derivations
# =====================================================================


def _arf_field(run_dir: Path) -> np.ndarray:
    """Static ARF body-force density (N/m^3)."""
    return np.load(run_dir / "radiation_force.npy").astype(np.float32)


def _strain_tensor(u: np.ndarray, dx: float, dy: float, dz: float
                   ) -> dict[str, np.ndarray]:
    """Symmetric small-strain tensor from a 3-component displacement
    field ``u`` of shape (nx, ny, nz, 3).  Returns the six independent
    components, the trace (volumetric strain ε_vol = ∇·u), and the
    von Mises equivalent strain.
    """
    ux = u[..., 0]
    uy = u[..., 1]
    uz = u[..., 2]
    # Central differences (numpy gradient handles boundaries with 1st-order)
    duxdx, duxdy, duxdz = np.gradient(ux, dx, dy, dz, edge_order=1)
    duydx, duydy, duydz = np.gradient(uy, dx, dy, dz, edge_order=1)
    duzdx, duzdy, duzdz = np.gradient(uz, dx, dy, dz, edge_order=1)
    e_xx = duxdx
    e_yy = duydy
    e_zz = duzdz
    e_xy = 0.5 * (duxdy + duydx)
    e_xz = 0.5 * (duxdz + duzdx)
    e_yz = 0.5 * (duydz + duzdy)
    e_vol = e_xx + e_yy + e_zz
    # Deviatoric components and von Mises equivalent strain.
    em = e_vol / 3.0
    dxx = e_xx - em
    dyy = e_yy - em
    dzz = e_zz - em
    e_dev_norm2 = (dxx * dxx + dyy * dyy + dzz * dzz
                   + 2.0 * (e_xy * e_xy + e_xz * e_xz + e_yz * e_yz))
    e_eq = np.sqrt((2.0 / 3.0) * e_dev_norm2)
    return {
        "e_xx": e_xx, "e_yy": e_yy, "e_zz": e_zz,
        "e_xy": e_xy, "e_xz": e_xz, "e_yz": e_yz,
        "e_vol": e_vol, "e_eq": e_eq,
    }


def _passive_depolarization_field(
    tension_mNm: np.ndarray,
    *,
    g_L_mS_cm2: float = 0.0167,
    E_L_mV: float = -70.3,
    V_rest_mV: float = -70.0,
    g_piezo1_mS_cm2: float = 0.10,
    g_trek1_mS_cm2: float = 0.04,
    use_trek1: bool = True,
) -> np.ndarray:
    """Per-voxel passive-membrane steady-state depolarisation $\\Delta V$ (mV).

    Gates are held at their local mechano-driven equilibrium; the membrane
    is allowed to settle to its conductance-weighted mean reversal:

        V_ss = (g_L·E_L + sum_i g_i·P_o,i(T)·E_rev,i) / (g_L + sum_i g_i·P_o,i(T))
        ΔV   = V_ss − V_rest

    No spiking, no Hodgkin–Huxley dynamics — the field generalises across
    cell types because the firing-zone for any spike threshold drops out
    as the iso-surface ΔV = V_thresh − V_rest, set by the cell substrate
    rather than the simulation.  Defaults: Pospischil cortical leak,
    Piezo1 (E_rev=0 mV) ḡ=0.10, TREK-1 (E_rev=−90 mV) ḡ=0.04 mS/cm² —
    matches the canonical Yaakub-on-Halle config.
    """
    from ultraneuron.channel_library import PIEZO1 as _Piezo1
    from ultraneuron.channel_library import TREK1 as _Trek1

    Po_p = np.asarray(_Piezo1.open_probability(tension_mNm), dtype=np.float64)
    g_p = g_piezo1_mS_cm2
    E_p = float(_Piezo1.E_rev)

    if use_trek1:
        Po_t = np.asarray(_Trek1.open_probability(tension_mNm), dtype=np.float64)
        g_t = g_trek1_mS_cm2
        E_t = float(_Trek1.E_rev)
    else:
        Po_t = np.zeros_like(Po_p)
        g_t = 0.0
        E_t = 0.0

    g_total = g_L_mS_cm2 + g_p * Po_p + g_t * Po_t
    V_ss = (g_L_mS_cm2 * E_L_mV
            + g_p * Po_p * E_p
            + g_t * Po_t * E_t) / np.maximum(g_total, 1e-12)
    return (V_ss - V_rest_mV).astype(np.float32)


def _membrane_tension(e_eq: np.ndarray, K_A: float = 0.25,
                      coupling_factor: float = 1000.0) -> np.ndarray:
    """Membrane tension T = K_A * alpha * eps_dev (mN/m), Eq. 2 of the
    manuscript.  Same convention as ``ultraneuron.channels.
    strain_to_membrane_tension`` and the neural_outputs builder, so the
    elastic Section 5 tension figure is consistent with the neural
    Section 7 tension overlays.

    K_A defaults to 0.25 N/m (Rawicz et al. 2000 lipid-bilayer area
    expansion modulus).  ``coupling_factor`` is the dimensionless
    multi-scale gain alpha relating mm-scale tissue deviatoric strain
    to local membrane area change; default 1000 matches the cortical-
    pyramidal calibration used for the manuscript-canonical Yaakub run.
    Output is in mN/m (factor 1e3 from K_A in N/m).
    """
    return (K_A * coupling_factor * e_eq * 1e3).astype(np.float32)


# =====================================================================
# Generic scalar-field renderers (ortho + 3-D iso) reused per quantity
# =====================================================================


def _peak_voxel_brain(field: np.ndarray, c_map: np.ndarray) -> tuple:
    """Argmax restricted to the brain (c <= 1600 m/s) so peaks pinned
    to skull voxels (which dominate ARF) don't drag the slice planes
    to the bowl/skull interface."""
    brain = c_map <= 1600.0
    if not brain.any():
        return np.unravel_index(int(np.argmax(field)), field.shape)
    masked = np.where(brain, field, -np.inf)
    return np.unravel_index(int(np.argmax(masked)), field.shape)


def _ortho_scalar(field: np.ndarray, run: dict[str, Any], peak_idx: tuple,
                  out_path: Path, *, cmap: str, label: str, title: str,
                  vmin: float | None = None, vmax: float | None = None
                  ) -> None:
    """Three orthogonal slices of ``field`` through ``peak_idx`` with a
    cyan skull contour overlay; depth on the y-axis with 0 on top.

    The skull contour uses the c > 1600 m/s threshold (matches Fig 3
    and Table 2 conventions) and a thin, semi-transparent line so it
    annotates without occluding the underlying field.
    """
    skull = run["c"] > 1600.0
    nx, ny, nz = field.shape
    dx, dy, dz = run["dx"] * 1e3, run["dy"] * 1e3, run["dz"] * 1e3
    Lx, Ly, Lz = nx * dx, ny * dy, nz * dz
    pk = peak_idx
    if vmax is None:
        vmax = float(np.percentile(field, 99.9))
        if vmax <= 0:
            vmax = max(float(field.max()), 1e-12)
    if vmin is None:
        vmin = 0.0 if field.min() >= 0 else float(np.percentile(field, 0.1))

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), constrained_layout=True)

    ax = axes[0]
    im = ax.imshow(field[:, :, pk[2]], cmap=cmap, origin="upper",
                   vmin=vmin, vmax=vmax, aspect="equal",
                   extent=(0, Ly, Lx, 0))
    ax.contour(np.linspace(0, Ly, ny), np.linspace(0, Lx, nx),
               skull[:, :, pk[2]], levels=[0.5], colors="cyan", linewidths=0.4, alpha=0.6)
    ax.set_xlabel("y (mm)"); ax.set_ylabel("x, depth (mm)")
    ax.set_title(f"x-y plane (z={pk[2] * dz:.1f} mm)")

    ax = axes[1]
    ax.imshow(field[:, pk[1], :], cmap=cmap, origin="upper",
              vmin=vmin, vmax=vmax, aspect="equal",
              extent=(0, Lz, Lx, 0))
    ax.contour(np.linspace(0, Lz, nz), np.linspace(0, Lx, nx),
               skull[:, pk[1], :], levels=[0.5], colors="cyan", linewidths=0.4, alpha=0.6)
    ax.set_xlabel("z (mm)"); ax.set_ylabel("x, depth (mm)")
    ax.set_title(f"x-z plane (y={pk[1] * dy:.1f} mm)")

    ax = axes[2]
    ax.imshow(field[pk[0], :, :].T, cmap=cmap, origin="lower",
              vmin=vmin, vmax=vmax, aspect="equal",
              extent=(0, Ly, 0, Lz))
    ax.contour(np.linspace(0, Ly, ny), np.linspace(0, Lz, nz),
               skull[pk[0], :, :].T, levels=[0.5], colors="cyan",
               linewidths=0.6)
    ax.set_xlabel("y (mm)"); ax.set_ylabel("z (mm)")
    ax.set_title(f"y-z plane (x={pk[0] * dx:.1f} mm)")

    cbar = fig.colorbar(im, ax=axes, location="right", shrink=0.85, pad=0.02)
    cbar.set_label(label)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _iso_3d_scalar_slab(field: np.ndarray, run: dict[str, Any], out_path: Path,
                        *, title: str, fractions=(0.50, 0.25),
                        colors=("red", "orange"), alphas=(0.85, 0.45),
                        ) -> None:
    """Slab-coordinate fallback for iso-surfaces of ``field`` (used
    when no placement metadata is available)."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from skimage.measure import marching_cubes

    fmax = float(field.max())
    if fmax <= 0:
        # Nothing to contour; emit a blank placeholder so LaTeX still finds it.
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.text(0.5, 0.5, "field is identically zero", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        return

    skull = run["c"] > 1600.0
    dx, dy, dz = run["dx"] * 1e3, run["dy"] * 1e3, run["dz"] * 1e3
    nx, ny, nz = field.shape

    surfaces = []
    for frac, color, alpha in zip(fractions, colors, alphas):
        thresh = fmax * frac
        try:
            verts, faces, _, _ = marching_cubes(field, level=thresh)
        except (RuntimeError, ValueError):
            continue
        surfaces.append((verts * np.array([dx, dy, dz]), faces, color, alpha,
                         frac, thresh))

    skull_idx = np.argwhere(skull)
    if len(skull_idx) > 6000:
        rng = np.random.default_rng(0)
        skull_idx = skull_idx[rng.choice(len(skull_idx), 6000, replace=False)]
    skull_mm = skull_idx * np.array([dx, dy, dz])

    views = (
        ("Side (along $-y$)",  dict(elev=0,  azim=-90)),
        ("Front (along $-x$)", dict(elev=0,  azim=0)),
        ("Top (along $-z$)",   dict(elev=90, azim=-90)),
    )
    fig = plt.figure(figsize=(12, 4.2))
    for k, (vtitle, vk) in enumerate(views):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        if len(skull_mm):
            ax.scatter(skull_mm[:, 0], skull_mm[:, 1], skull_mm[:, 2],
                       s=1.4, c="dimgray", alpha=0.30)
        for verts_mm, faces, color, alpha, _frac, _thr in surfaces:
            mesh = Poly3DCollection(verts_mm[faces], facecolor=color,
                                     edgecolor="none", alpha=alpha)
            ax.add_collection3d(mesh)
        ax.set_xlim(0, nx * dx); ax.set_ylim(0, ny * dy); ax.set_zlim(0, nz * dz)
        ax.set_box_aspect((nx * dx, ny * dy, nz * dz))
        ax.set_xlabel("x, depth (mm)", labelpad=2)
        ax.set_ylabel("y, lateral (mm)", labelpad=2)
        ax.set_zlabel("z, elevation (mm)", labelpad=2)
        ax.tick_params(labelsize=7)
        ax.view_init(**vk)
        ax.set_title(vtitle, fontsize=10)

    legend_bits = ", ".join(
        f"{c}: {f:.0%} of peak ({thr:.3g})"
        for (_, _, c, _, f, thr) in surfaces
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _iso_3d_scalar(field: np.ndarray, run: dict[str, Any], out_path: Path,
                   *, title: str, fractions=(0.50, 0.25, 0.10),
                   colors=("#d62728", "#ff7f0e", "#ffd92f"),
                   alphas=(1.00, 0.65, 0.30),
                   cfg=None) -> None:
    """LPS-frame 3-D rendering of iso-surfaces of ``field`` patterned
    on the thermal Fig.~10: full skull point cloud, pial brain mesh,
    target ROI, bowl wireframe, beam axis, and apex/scalp/target/focus
    markers — three views (Lateral / Anterior / Superior) — with
    iso-surfaces of the field overlaid.  Falls back to the slab-
    coordinate renderer when no placement metadata is available.
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from skimage.measure import marching_cubes

    fmax = float(field.max())
    if fmax <= 0 or cfg is None:
        _iso_3d_scalar_slab(field, run, out_path, title=title,
                             fractions=fractions[:2], colors=colors[:2],
                             alphas=alphas[:2])
        return

    from ultraneuron.registration import ensure_registration_importable

    ensure_registration_importable()
    import visualize_placement_3d as viz  # noqa: E402
    from halle_placement import place_on_skull  # noqa: E402

    plc = cfg.PLACEMENT
    p = place_on_skull(
        scalp_site=plc["scalp_site"],
        target_name=plc["target_name"],
        focal_length_mm=plc["focal_length_mm"],
        bowl_radius_mm=plc["bowl_radius_mm"],
    )
    p["target_name"] = plc["target_name"]

    apex = np.asarray(p["xdc_center_lps"])
    target = np.asarray(p["target_lps"])
    focus = np.asarray(p["focus_lps"])
    scalp = np.asarray(p["scalp_contact_lps"])
    beam = np.asarray(p["beam_dir_3d"])
    F = float(p["focal_length_mm"])
    R = float(p["bowl_radius_mm"])

    bone_v, _ = viz.load_outer_bone_surface()
    brain_v, brain_f, _ = viz.warped_brain_mesh()
    target_roi = viz.build_and_warp_target_roi(plc["target_name"])
    pts3d = viz.bowl_points(apex, beam, F, R)
    bquads = viz.bowl_quads(pts3d)

    rng = np.random.default_rng(0)
    bone_pts = bone_v   # full bone surface (no subsampling), matches Fig. 2
    brain_f_plot = (brain_f[rng.choice(len(brain_f), 12000, replace=False)]
                    if len(brain_f) > 12000 else brain_f)
    if target_roi.any():
        roi_v, roi_f, _, _ = marching_cubes(target_roi.astype(np.uint8),
                                             level=0.5)
        roi_f_plot = (roi_f[rng.choice(len(roi_f), 6000, replace=False)]
                      if len(roi_f) > 6000 else roi_f)
    else:
        roi_v = np.zeros((0, 3))
        roi_f_plot = np.zeros((0, 3), dtype=int)

    # Slab-grid -> LPS transform from the placement (matches Fig 7/Fig 10).
    n_hat = np.asarray(p.get("normal_3d", beam))
    n_hat = n_hat / (np.linalg.norm(n_hat) + 1e-12)
    if "tangent_3d" in p and "bitangent_3d" in p:
        t_hat = np.asarray(p["tangent_3d"])
        b_hat = np.asarray(p["bitangent_3d"])
    else:
        ref = (np.array([0.0, 0.0, 1.0]) if abs(n_hat[2]) < 0.9
               else np.array([1.0, 0.0, 0.0]))
        t_hat = np.cross(n_hat, ref)
        t_hat /= (np.linalg.norm(t_hat) + 1e-12)
        b_hat = np.cross(n_hat, t_hat)
    t_hat = t_hat / (np.linalg.norm(t_hat) + 1e-12)
    b_hat = b_hat / (np.linalg.norm(b_hat) + 1e-12)

    nx, ny, nz = field.shape
    dx, dy, dz = run["dx"], run["dy"], run["dz"]   # metres
    lat_off = ny / 2.0
    elev_off = nz / 2.0

    def slab_to_lps(verts_idx):
        return (apex
                + (verts_idx[:, 0] * dx * 1e3)[:, None] * n_hat
                + ((verts_idx[:, 1] - lat_off) * dy * 1e3)[:, None] * t_hat
                + ((verts_idx[:, 2] - elev_off) * dz * 1e3)[:, None] * b_hat)

    surfaces = []
    for frac, color, alpha in zip(fractions, colors, alphas):
        thresh = fmax * frac
        if thresh <= 0:
            continue
        try:
            verts, faces, _, _ = marching_cubes(field, level=thresh)
        except (RuntimeError, ValueError):
            continue
        surfaces.append((slab_to_lps(verts), faces, color, alpha,
                          frac, thresh))

    views = [
        ("Lateral (from patient left)", dict(elev=0,  azim=180)),
        ("Anterior (from face)",        dict(elev=0,  azim=-90)),
        ("Superior (from above)",       dict(elev=90, azim=-90)),
    ]
    fig = plt.figure(figsize=(18, 6))
    for k, (vtitle, vk) in enumerate(views):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        ax.scatter(bone_pts[:, 0], bone_pts[:, 1], bone_pts[:, 2],
                   s=1.2, c="dimgray", alpha=0.32)
        ax.add_collection3d(Poly3DCollection(brain_v[brain_f_plot],
                                              facecolor="orange",
                                              edgecolor="none", alpha=0.20))
        if len(roi_f_plot) > 0:
            ax.add_collection3d(Poly3DCollection(roi_v[roi_f_plot],
                                                  facecolor="lime",
                                                  edgecolor="none",
                                                  alpha=0.75))
        ax.add_collection3d(Poly3DCollection(bquads, facecolor="steelblue",
                                              edgecolor="k", linewidth=0.2,
                                              alpha=0.85))
        for verts_lps, faces, color, alpha, _frac, _thr in surfaces:
            ax.add_collection3d(Poly3DCollection(verts_lps[faces],
                                                  facecolor=color,
                                                  edgecolor="none",
                                                  alpha=alpha))
        ax.plot(*list(zip(apex, focus)), color="red", lw=2.5)
        ax.scatter(*apex,   color="blue", s=70,  edgecolor="k", zorder=5)
        ax.scatter(*scalp,  color="cyan", s=60,  edgecolor="k", marker="s",
                    zorder=5)
        ax.scatter(*target, color="lime", s=120, edgecolor="k", zorder=5)
        ax.scatter(*focus,  color="red",  s=120, marker="*", edgecolor="k",
                    zorder=5)
        ax.view_init(**vk)
        ax.set(xlabel="L (mm)", ylabel="P (mm)", zlabel="S (mm)", title=vtitle)
        ax.set_xlim(0, bone_v[:, 0].max())
        ax.set_ylim(0, bone_v[:, 1].max())
        ax.set_zlim(0, bone_v[:, 2].max())
        ax.set_box_aspect([float(np.ptp(bone_v[:, i])) for i in range(3)])

    legend_bits = ", ".join(
        f"{c}: {f:.0%} of peak ({thr:.3g})"
        for (_, _, c, _, f, thr) in surfaces
    )
    fig.tight_layout(rect=(0, 0, 0.93, 0.98))
    if surfaces:
        from matplotlib.colors import BoundaryNorm, ListedColormap
        from matplotlib.cm import ScalarMappable
        ordered = sorted(surfaces, key=lambda s: s[5])  # by threshold
        cb_colors = [s[2] for s in ordered]
        cb_thrs = [s[5] for s in ordered]
        cb_fracs = [s[4] for s in ordered]
        # Geometric bin boundaries so each iso-fraction maps to one swatch.
        edges = [cb_thrs[0] / 1.5] + [
            float(np.sqrt(a * b)) for a, b in zip(cb_thrs, cb_thrs[1:])
        ] + [cb_thrs[-1] * 1.5]
        cmap = ListedColormap(cb_colors)
        norm = BoundaryNorm(edges, ncolors=len(cb_colors))
        cax = fig.add_axes([0.945, 0.18, 0.012, 0.62])
        cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap),
                             cax=cax, ticks=cb_thrs)
        cbar.ax.set_yticklabels(
            [f"{thr:.3g}\n({f:.0%})" for thr, f in zip(cb_thrs, cb_fracs)],
            fontsize=8,
        )
        cbar.set_label(f"{title.split('(')[0].strip()} (peak {fmax:.3g})",
                        fontsize=8)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _strain_time_snapshots(sim_dir: Path, run: dict[str, Any],
                            out_path: Path) -> None:
    """Six-panel temporal evolution of the deviatoric in-plane strain
    proxy, patterned on Fig.~6 (pressure snapshots).  Top row: x-y
    mid-z slice; bottom row: x-z mid-y slice.  Three timepoints: early
    (rising flank), peak (frame of global max strain), and late (decay).
    The strain proxy is the magnitude of the spatial gradient of the
    saved displacement-magnitude snapshot ``snapshots_xy.npy`` /
    ``snapshots_xz.npy``: |∇|u||, which has units of [m/m] and rises
    where the shear wave produces sharp displacement gradients.
    """
    sxy_path = sim_dir / "snapshots_xy.npy"
    sxz_path = sim_dir / "snapshots_xz.npy"
    if not (sxy_path.exists() and sxz_path.exists()):
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.text(0.5, 0.5, "snapshots_xy.npy / snapshots_xz.npy missing",
                ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        return

    sxy = np.load(sxy_path)              # (nt, nx, ny)
    sxz = np.load(sxz_path)              # (nt, nx, nz)

    # Spatial gradient magnitude per frame; voxel spacing in mm (dx, dy)
    # for xy and (dx, dz) for xz.
    dx = run["dx"] * 1e3
    dy = run["dy"] * 1e3
    dz = run["dz"] * 1e3
    gxy_t = np.zeros_like(sxy)
    gxz_t = np.zeros_like(sxz)
    for k in range(sxy.shape[0]):
        gx, gy = np.gradient(sxy[k], dx, dy, edge_order=1)
        gxy_t[k] = np.sqrt(gx * gx + gy * gy)
    for k in range(sxz.shape[0]):
        gx, gz = np.gradient(sxz[k], dx, dz, edge_order=1)
        gxz_t[k] = np.sqrt(gx * gx + gz * gz)

    nt = sxy.shape[0]
    # Peak frame from global max of either plane.
    peak_per_frame = np.maximum(gxy_t.reshape(nt, -1).max(axis=1),
                                  gxz_t.reshape(nt, -1).max(axis=1))
    peak_frame = int(np.argmax(peak_per_frame))
    # Early frame: midway between t=0 and the peak.  Late: midway
    # between peak and the end of the saved window.
    early_frame = max(int(peak_frame * 0.5), 1)
    late_frame = min(int(peak_frame + (nt - 1 - peak_frame) * 0.5), nt - 1)
    frames = [("early", early_frame), ("peak", peak_frame),
              ("late", late_frame)]
    dt_save = run.get("dt_save")
    if dt_save is None:
        dt_save = 0.0

    nx_xy, ny_xy = sxy.shape[1], sxy.shape[2]
    nx_xz, nz_xz = sxz.shape[1], sxz.shape[2]
    Lx_xy, Ly_xy = nx_xy * dx, ny_xy * dy
    Lx_xz, Lz_xz = nx_xz * dx, nz_xz * dz
    vmax = float(np.percentile(np.concatenate([gxy_t.ravel(), gxz_t.ravel()]),
                                99.5))
    if vmax <= 0:
        vmax = max(float(gxy_t.max()), float(gxz_t.max()), 1e-12)

    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4),
                              constrained_layout=True)
    for col, (label, fr) in enumerate(frames):
        peak_xy = float(gxy_t[fr].max())
        peak_xz = float(gxz_t[fr].max())
        peak_now = max(peak_xy, peak_xz)
        t_us = (fr * dt_save * 1e6) if dt_save else fr

        ax = axes[0, col]
        im = ax.imshow(gxy_t[fr], cmap="viridis", origin="upper",
                       vmin=0, vmax=vmax, aspect="equal",
                       extent=(0, Ly_xy, Lx_xy, 0))
        title_t = (f"{label}: $t={t_us:.1f}$ µs (frame {fr})\n"
                   f"peak $|\\nabla|u|| = {peak_now:.2e}$ /m"
                   if dt_save else
                   f"{label}: frame {fr}, peak $|\\nabla|u|| ="
                   f" {peak_now:.2e}$ /m")
        ax.set_title(title_t, fontsize=9)
        ax.set_xlabel("y (mm)")
        ax.set_ylabel("x, depth (mm)")

        ax = axes[1, col]
        ax.imshow(gxz_t[fr], cmap="viridis", origin="upper",
                  vmin=0, vmax=vmax, aspect="equal",
                  extent=(0, Lz_xz, Lx_xz, 0))
        ax.set_xlabel("z (mm)")
        ax.set_ylabel("x, depth (mm)")

    cbar = fig.colorbar(im, ax=axes, location="right", shrink=0.85, pad=0.02)
    cbar.set_label(r"$|\nabla|u||$ (1/m)")
    fig.suptitle(
        "Transcranial: deviatoric-strain proxy (gradient magnitude of "
        "shear-wave displacement) — early / peak / late.  "
        "Top row: x-y mid-z slice; bottom row: x-z mid-y slice.",
        fontsize=10,
    )
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _eeq_time_snapshots(sim_dir: Path, run: dict[str, Any], cfg,
                        out_path: Path) -> dict[str, Any]:
    """True deviatoric strain ε_eq at five representative times across
    one Yaakub PRF period, computed from the full 3-D shear-FDTD
    displacement trajectory ``displacement_4d.npy``.

    Caches the five orthogonal-plane stacks as
    ``eeq_snapshots_cache.npz`` next to the upstream artefacts so
    subsequent rebuilds skip the 3-D gradient pass.
    """
    cache_path = sim_dir / "eeq_snapshots_cache.npz"
    seq = getattr(cfg, "NEUROMOD_SEQUENCE", {}) or {}
    t_pulse_on = float(seq.get("tone_burst_duration_s", 30e-3))
    t_period = float(seq.get("pulse_period_s", 0.2))
    target_times_s = [
        0.25 * t_pulse_on,
        0.50 * t_pulse_on,
        t_pulse_on,
        t_pulse_on + 0.4 * (t_period - t_pulse_on),
        0.99 * t_period,
    ]

    # Invalidate the ε_eq cache when it predates the displacement field it
    # was derived from — otherwise a regenerated displacement_4d.npy leaves
    # a stale cache that silently renders wrong strain/tension figures.
    disp_4d_path = sim_dir / "displacement_4d.npy"
    cache_fresh = cache_path.exists() and (
        not disp_4d_path.exists()
        or cache_path.stat().st_mtime >= disp_4d_path.stat().st_mtime
    )
    if cache_fresh:
        with np.load(cache_path) as data:
            eeq_xy = data["eeq_xy"]
            eeq_xz = data["eeq_xz"]
            t_used_ms = list(map(float, data["t_used_ms"]))
            iy_mid = int(data["iy_mid"]); iz_mid = int(data["iz_mid"])
    else:
        disp_4d_path = sim_dir / "displacement_4d.npy"
        times_path = sim_dir / "displacement_4d_times.npy"
        if not (disp_4d_path.exists() and times_path.exists()):
            fig, ax = plt.subplots(figsize=(9, 3))
            ax.text(0.5, 0.5, "displacement_4d.npy / times missing",
                    ha="center", va="center")
            ax.axis("off")
            fig.savefig(out_path, dpi=160)
            plt.close(fig)
            return {"present": False}
        u_4d = np.load(disp_4d_path, mmap_mode="r")  # (nt, nx, ny, nz, 3)
        t_4d = np.load(times_path)
        nt, nx, ny, nz, _ = u_4d.shape
        iy_mid = ny // 2
        iz_mid = nz // 2
        frame_idx = [int(np.argmin(np.abs(t_4d - tt)))
                     for tt in target_times_s]
        t_used_ms = [float(t_4d[i] * 1e3) for i in frame_idx]
        eeq_xy = np.zeros((5, nx, ny), dtype=np.float32)
        eeq_xz = np.zeros((5, nx, nz), dtype=np.float32)
        for k, fi in enumerate(frame_idx):
            u = np.asarray(u_4d[fi], dtype=np.float64)
            strain = _strain_tensor(u, run["dx"], run["dy"], run["dz"])
            eeq_xy[k] = strain["e_eq"][:, :, iz_mid].astype(np.float32)
            eeq_xz[k] = strain["e_eq"][:, iy_mid, :].astype(np.float32)
        np.savez_compressed(
            cache_path,
            eeq_xy=eeq_xy, eeq_xz=eeq_xz,
            t_used_ms=np.asarray(t_used_ms, dtype=np.float64),
            iy_mid=np.int32(iy_mid), iz_mid=np.int32(iz_mid),
        )

    dx = run["dx"] * 1e3; dy = run["dy"] * 1e3; dz = run["dz"] * 1e3
    nx, ny_xy = eeq_xy.shape[1], eeq_xy.shape[2]
    nz_xz = eeq_xz.shape[2]
    vmax = float(np.percentile(
        np.concatenate([eeq_xy.ravel(), eeq_xz.ravel()]), 99.5,
    ))
    if vmax <= 0:
        vmax = max(float(eeq_xy.max()), float(eeq_xz.max()), 1e-12)
    skull = run["c"] > 1600.0
    skull_xy = skull[:, :, iz_mid]
    skull_xz = skull[:, iy_mid, :]

    fig, axes = plt.subplots(2, 5, figsize=(15, 6.0))
    ext_xy = (0, ny_xy * dy, nx * dx, 0)
    ext_xz = (0, nz_xz * dz, nx * dx, 0)
    for col in range(5):
        ax = axes[0, col]
        im = ax.imshow(eeq_xy[col], cmap="viridis", vmin=0, vmax=vmax,
                       extent=ext_xy, aspect="equal")
        ax.contour(np.linspace(ext_xy[0], ext_xy[1], skull_xy.shape[1]),
                   np.linspace(ext_xy[3], ext_xy[2], skull_xy.shape[0]),
                   skull_xy, levels=[0.5], colors="cyan", linewidths=0.5)
        ax.set_title(f"t = {t_used_ms[col]:.1f} ms", fontsize=10)
        if col == 0:
            ax.set_ylabel("x, depth (mm)\nXY-plane")
        ax.set_xlabel("y (mm)" if col == 0 else "")

        ax = axes[1, col]
        ax.imshow(eeq_xz[col], cmap="viridis", vmin=0, vmax=vmax,
                  extent=ext_xz, aspect="equal")
        ax.contour(np.linspace(ext_xz[0], ext_xz[1], skull_xz.shape[1]),
                   np.linspace(ext_xz[3], ext_xz[2], skull_xz.shape[0]),
                   skull_xz, levels=[0.5], colors="cyan", linewidths=0.5)
        if col == 0:
            ax.set_ylabel("x, depth (mm)\nXZ-plane")
        ax.set_xlabel("z (mm)" if col == 0 else "")

    fig.subplots_adjust(left=0.06, right=0.92, top=0.90, bottom=0.10,
                        wspace=0.10, hspace=0.20)
    cax = fig.add_axes([0.93, 0.12, 0.012, 0.76])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(r"$\varepsilon_{\mathrm{eq}}$ (m/m)")
    fig.suptitle(
        r"Deviatoric strain $\varepsilon_{\mathrm{eq}}$ snapshots — "
        r"XY (top) and XZ (bottom) at five times spanning one Yaakub "
        r"PRF period",
        fontsize=11, fontweight="bold",
    )
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    return {
        "present": True,
        "t_used_ms": t_used_ms,
        "eeq_field_max": float(max(eeq_xy.max(), eeq_xz.max())),
        "eeq_field_max_per_snapshot": [
            float(max(eeq_xy[k].max(), eeq_xz[k].max())) for k in range(5)
        ],
    }


def _tension_time_snapshots(sim_dir: Path, run: dict[str, Any], cfg,
                            out_path: Path, *,
                            K_A: float, coupling_factor: float,
                            ) -> dict[str, Any]:
    """Membrane tension T = K_A · α · ε_eq snapshots, derived from the
    cached deviatoric-strain stacks written by ``_eeq_time_snapshots``.

    Output is in mN/m (Yaakub-canonical scale; manuscript value Piezo1
    threshold ~0.5 mN/m).
    """
    cache_path = sim_dir / "eeq_snapshots_cache.npz"
    if not cache_path.exists():
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.text(0.5, 0.5, "ε_eq cache missing — run elastic_outputs first",
                ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        return {"present": False}

    with np.load(cache_path) as data:
        eeq_xy = data["eeq_xy"]
        eeq_xz = data["eeq_xz"]
        t_used_ms = list(map(float, data["t_used_ms"]))
        iy_mid = int(data["iy_mid"]); iz_mid = int(data["iz_mid"])

    # T = K_A · α · ε_eq, with output in mN/m. The product K_A·α is the
    # tissue→bilayer multi-scale scaling that lifts voxel-scale strain to
    # membrane areal strain (manuscript Eq for tension).
    scale_mNm = K_A * coupling_factor * 1e3  # K_A in N/m → mN/m via 1e3
    T_xy = (eeq_xy * scale_mNm).astype(np.float32)
    T_xz = (eeq_xz * scale_mNm).astype(np.float32)

    dx = run["dx"] * 1e3; dy = run["dy"] * 1e3; dz = run["dz"] * 1e3
    nx, ny_xy = T_xy.shape[1], T_xy.shape[2]
    nz_xz = T_xz.shape[2]
    vmax = float(np.percentile(
        np.concatenate([T_xy.ravel(), T_xz.ravel()]), 99.5,
    ))
    if vmax <= 0:
        vmax = max(float(T_xy.max()), float(T_xz.max()), 1e-12)
    skull = run["c"] > 1600.0
    skull_xy = skull[:, :, iz_mid]
    skull_xz = skull[:, iy_mid, :]

    fig, axes = plt.subplots(2, 5, figsize=(15, 6.0))
    ext_xy = (0, ny_xy * dy, nx * dx, 0)
    ext_xz = (0, nz_xz * dz, nx * dx, 0)
    threshold_mNm = 0.5  # Piezo1 firing threshold (manuscript)
    for col in range(5):
        ax = axes[0, col]
        im = ax.imshow(T_xy[col], cmap="magma", vmin=0, vmax=vmax,
                       extent=ext_xy, aspect="equal")
        ax.contour(np.linspace(ext_xy[0], ext_xy[1], skull_xy.shape[1]),
                   np.linspace(ext_xy[3], ext_xy[2], skull_xy.shape[0]),
                   skull_xy, levels=[0.5], colors="cyan", linewidths=0.5)
        if T_xy[col].max() > threshold_mNm:
            ax.contour(np.linspace(ext_xy[0], ext_xy[1], T_xy[col].shape[1]),
                       np.linspace(ext_xy[3], ext_xy[2], T_xy[col].shape[0]),
                       T_xy[col], levels=[threshold_mNm],
                       colors="lime", linewidths=0.7)
        ax.set_title(f"t = {t_used_ms[col]:.1f} ms", fontsize=10)
        if col == 0:
            ax.set_ylabel("x, depth (mm)\nXY-plane")
        ax.set_xlabel("y (mm)" if col == 0 else "")

        ax = axes[1, col]
        ax.imshow(T_xz[col], cmap="magma", vmin=0, vmax=vmax,
                  extent=ext_xz, aspect="equal")
        ax.contour(np.linspace(ext_xz[0], ext_xz[1], skull_xz.shape[1]),
                   np.linspace(ext_xz[3], ext_xz[2], skull_xz.shape[0]),
                   skull_xz, levels=[0.5], colors="cyan", linewidths=0.5)
        if T_xz[col].max() > threshold_mNm:
            ax.contour(np.linspace(ext_xz[0], ext_xz[1], T_xz[col].shape[1]),
                       np.linspace(ext_xz[3], ext_xz[2], T_xz[col].shape[0]),
                       T_xz[col], levels=[threshold_mNm],
                       colors="lime", linewidths=0.7)
        if col == 0:
            ax.set_ylabel("x, depth (mm)\nXZ-plane")
        ax.set_xlabel("z (mm)" if col == 0 else "")

    fig.subplots_adjust(left=0.06, right=0.92, top=0.90, bottom=0.10,
                        wspace=0.10, hspace=0.20)
    cax = fig.add_axes([0.93, 0.12, 0.012, 0.76])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(r"$T = K_A\,\alpha\,\varepsilon_{\mathrm{eq}}$ (mN/m)")
    fig.suptitle(
        rf"Membrane tension snapshots — XY (top) and XZ (bottom) at five "
        rf"times spanning one Yaakub PRF period.  $K_A = {K_A:.2f}$ N/m, "
        rf"$\alpha = {coupling_factor:.0f}$.  Lime contour: Piezo1 "
        rf"threshold (0.5 mN/m).",
        fontsize=10, fontweight="bold",
    )
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    return {
        "present": True,
        "t_used_ms": t_used_ms,
        "K_A": float(K_A),
        "coupling_factor": float(coupling_factor),
        "tension_field_max_mNm": float(max(T_xy.max(), T_xz.max())),
        "tension_field_max_per_snapshot_mNm": [
            float(max(T_xy[k].max(), T_xz[k].max())) for k in range(5)
        ],
        "n_above_threshold_per_snapshot": [
            int(np.sum(T_xy[k] > threshold_mNm)
                + np.sum(T_xz[k] > threshold_mNm))
            for k in range(5)
        ],
    }


# =====================================================================
# Builder entry-point
# =====================================================================


def build(cfg, fig_dir: Path) -> dict[str, Any]:
    """Render the elastic-outputs section: ARF + deviatoric strain +
    membrane tension cross-sections and 3-D iso-surfaces."""
    repo_root = Path(__file__).resolve().parents[2]
    sim_dir = (repo_root / cfg.OUT_DIR).parent
    if not sim_dir.is_absolute():
        sim_dir = (repo_root / sim_dir).resolve()

    # Reuse the acoustic-outputs loader for c_map / dx-dy-dz / pmap; we
    # only need the geometry + skull mask here, plus the displacement
    # field which lives next to the propagation map.
    run = _ao._load_run(sim_dir)
    # Match the neural-section / manuscript Eq. 2 convention:
    # T = K_A * alpha * eps_dev (mN/m), with K_A=0.25 N/m and alpha=1000
    # for the cortical-pyramidal calibration. Configurable via cfg.
    K_A = float(getattr(cfg, "MEMBRANE_K_A_NM",
                        getattr(cfg, "NEURAL_K_A", 0.25)))
    coupling_factor = float(getattr(cfg, "NEURAL_COUPLING_FACTOR", 1000.0))

    arf = _arf_field(sim_dir)
    disp = np.load(sim_dir / "displacement.npy").astype(np.float32)
    strain = _strain_tensor(disp, run["dx"], run["dy"], run["dz"])
    e_eq = strain["e_eq"]
    tension = _membrane_tension(e_eq, K_A=K_A, coupling_factor=coupling_factor)

    # The upstream FDTD ARF computation zeroes entire x-columns
    # (rays along the propagation direction) at scattered (y, z)
    # grid cells where the bowl source-mask rasterizes to zero —
    # these show up as vertical stripes through the slice planes.
    # Only ~20% of (y, z) columns survive, so an "enclosed-hole"
    # in-paint fails: the bad region is too large for binary
    # closing/fill to bound.  Instead, fill every weak column with
    # a Gaussian-weighted average over its strong (y, z)
    # neighbours, computed per x-slice.  Cells with no nearby
    # strong data (bowl exterior, post-focus shadow) get a near-
    # zero weight so they are left untouched.
    from scipy import ndimage as _ndi
    col_max = arf.max(axis=0)                 # (ny, nz)
    pmax = float(col_max.max())
    valid_yz = (col_max >= 0.01 * pmax).astype(np.float32)
    if 0.0 < valid_yz.sum() < valid_yz.size:
        sigma_yz = 2.5  # voxels — wide enough to span the focal stripes
        weight = _ndi.gaussian_filter(valid_yz, sigma=sigma_yz)
        arf_smooth = np.empty_like(arf)
        for xi in range(arf.shape[0]):
            arf_smooth[xi] = _ndi.gaussian_filter(
                arf[xi] * valid_yz, sigma=sigma_yz
            )
        # Cells flagged "bad" with enough nearby strong neighbours
        # are filled; truly remote cells (weight ~ 0) stay zero.
        fill_mask = (valid_yz == 0) & (weight > 0.05)
        ys, zs = np.where(fill_mask)
        if ys.size:
            arf = arf.copy()
            arf[:, ys, zs] = (arf_smooth[:, ys, zs]
                                / weight[ys, zs][None, :])
            arf = arf.astype(np.float32)

    # Strain and membrane tension are masked to brain-only: finite-
    # difference gradients across the elastic discontinuity at the
    # skull-brain boundary produce spurious peaks otherwise. ARF is
    # left unmasked (skull absorption is the physical source of the
    # body force) but the colormap is set from a brain-only
    # percentile so the focal beam dominates the rendering.
    skull = run["c"] > 1600.0
    brain = ~skull
    e_eq = np.where(brain, e_eq, 0.0).astype(np.float32)
    tension = np.where(brain, tension, 0.0).astype(np.float32)

    fields = [
        ("arf",     arf,     "ARF density",           "N/m$^3$",  "magma",   r"$|\mathbf{F}_{\mathrm{ARF}}|$"),
        ("strain",  e_eq,    "deviatoric strain",     "—",        "viridis", r"$\varepsilon_{\mathrm{eq}}$ (von Mises)"),
        ("tension", tension, "membrane tension",      "mN/m",     "cividis", rf"$T = K_A\,\alpha\,\varepsilon_{{\mathrm{{eq}}}}$, $K_A={K_A}$ N/m, $\alpha={coupling_factor:.0f}$"),
    ]

    figures: dict[str, str] = {}
    metrics: dict[str, Any] = {
        "membrane_K_A_N_per_m": K_A,
        "membrane_coupling_factor": coupling_factor,
    }
    for key, field, name, units, cmap, math_label in fields:
        peak_idx = _peak_voxel_brain(field, run["c"])
        peak_val = float(field.max())
        peak_brain = float(field[brain].max()) if brain.any() else 0.0
        peak_skull = float(field[skull].max()) if skull.any() else 0.0
        peak_focal_val = float(field[peak_idx])

        title_ortho = (f"Transcranial: {name} with skull overlay "
                       f"({math_label} in {units})")
        title_iso = f"Transcranial: {name} ({math_label})"

        # Use the brain-only full max for vmax: ARF in skull saturates
        # the colormap on absorption otherwise; strain/tension are
        # already brain-zeroed so brain-max == global-max for them.
        if brain.any():
            vmax_field = float(field[brain].max())
            if vmax_field <= 0:
                vmax_field = max(float(field.max()), 1e-12)
        else:
            vmax_field = None

        fig_ortho = fig_dir / f"elastic_{key}_ortho.png"
        fig_3d = fig_dir / f"elastic_{key}_3d.png"
        _ortho_scalar(field, run, peak_idx, fig_ortho,
                      cmap=cmap, label=f"{math_label} ({units})",
                      title=title_ortho, vmax=vmax_field)
        _iso_3d_scalar(field, run, fig_3d, title=title_iso, cfg=cfg)

        figures[f"{key}_ortho"] = fig_ortho.name
        figures[f"{key}_3d"] = fig_3d.name
        metrics[key] = {
            "peak": peak_val,
            "peak_brain": peak_brain,
            "peak_skull": peak_skull,
            "peak_focal_voxel": peak_focal_val,
            "peak_idx": [int(i) for i in peak_idx],
            "units": units,
        }

    # True ε_eq tensor strain at five times spanning one PRF period,
    # computed from the full 3-D shear-FDTD trajectory and cached.
    fig_eeq_time = fig_dir / "elastic_eeq_time.png"
    eeq_time_meta = _eeq_time_snapshots(sim_dir, run, cfg, fig_eeq_time)
    figures["eeq_time"] = fig_eeq_time.name
    metrics["eeq_time"] = eeq_time_meta

    # Membrane tension snapshots derived from the cached ε_eq slices.
    fig_tension_time = fig_dir / "elastic_tension_time.png"
    K_A = float(getattr(cfg, "NEURAL_K_A", 0.25))
    # Default must match the static tension map and the neural sim (1000);
    # a 200 default here silently made the time-snapshot tension panel 5x
    # smaller than the rest of the report whenever a config omitted
    # NEURAL_COUPLING_FACTOR.
    coupling_factor = float(getattr(cfg, "NEURAL_COUPLING_FACTOR", 1000.0))
    tension_time_meta = _tension_time_snapshots(
        sim_dir, run, cfg, fig_tension_time,
        K_A=K_A, coupling_factor=coupling_factor,
    )
    figures["tension_time"] = fig_tension_time.name
    metrics["tension_time"] = tension_time_meta

    # Passive-membrane depolarisation field — cell-type-independent
    # readout that lets the firing-zone for any spike threshold drop out
    # as an iso-surface ΔV = V_thresh − V_rest.  Computed analytically
    # from the static-equilibrium tension field.
    g_p = float(getattr(cfg, "NEURAL_PIEZO1_G_DENSITY", 0.10))
    g_t = float(getattr(cfg, "NEURAL_TREK1_G_DENSITY", 0.04))
    use_trek1 = bool(getattr(cfg, "NEURAL_USE_TREK1", True))
    dV = _passive_depolarization_field(
        tension, g_piezo1_mS_cm2=g_p, g_trek1_mS_cm2=g_t, use_trek1=use_trek1,
    )
    dV_brain = dV.copy()
    dV_brain[run["c"] > 1600.0] = 0.0
    pk_dv = _peak_voxel_brain(dV_brain, run["c"])
    fig_dv_ortho = fig_dir / "elastic_depolarization_ortho.png"
    _ortho_scalar(
        dV, run, pk_dv, fig_dv_ortho,
        cmap="RdBu_r",
        label=r"$\Delta V_{\mathrm{passive}}$ (mV)",
        title=("Passive-membrane $\\Delta V_{\\mathrm{passive}}$ — "
               "ortho slices through brain-restricted peak voxel"),
        vmin=-float(np.abs(dV_brain).max()),
        vmax=float(np.abs(dV_brain).max()),
    )
    fig_dv_3d = fig_dir / "elastic_depolarization_3d.png"
    _iso_3d_scalar(
        dV_brain, run, fig_dv_3d,
        title=("Passive-membrane depolarisation iso-surfaces "
               "(red 50\\%, orange 25\\%, yellow 10\\% of brain peak)"),
        cfg=cfg,
    )
    figures["depolarization_ortho"] = fig_dv_ortho.name
    figures["depolarization_3d"] = fig_dv_3d.name
    metrics["depolarization"] = {
        "delta_V_peak_mV": float(dV.max()),
        "delta_V_peak_brain_mV": float(dV_brain.max()),
        "delta_V_min_mV": float(dV.min()),
        "g_leak_mS_cm2": 0.0167,
        "E_L_mV": -70.3,
        "V_rest_mV": -70.0,
        "g_piezo1_mS_cm2": g_p,
        "g_trek1_mS_cm2": g_t,
        "use_trek1": use_trek1,
        "peak_idx": [int(i) for i in pk_dv],
    }

    return {
        "elastic_outputs": metrics,
        "figures_elastic_outputs": figures,
    }
