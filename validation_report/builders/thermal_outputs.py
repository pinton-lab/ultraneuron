"""Section 3 — Thermal outputs.

Pure loader/plotter for the bioheat outputs produced by
``scripts/run_bioheat.py``.  The Pennes solve itself lives in the
simulation pipeline; this module reads ``<sim_dir>/bioheat/`` and
renders the figures used in the validation report (orthogonal slices,
3-D iso-surfaces, time history) plus the metrics table.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from . import acoustic_outputs as _ao


# =====================================================================
# Loader for sim-pipeline bioheat outputs
# =====================================================================


def _load_bioheat(sim_dir: Path) -> dict[str, Any]:
    """Read the artefacts written by ``scripts/run_bioheat.py``.

    Raises a clear error if the bioheat step has not been run.
    """
    bh = sim_dir / "bioheat"
    if not bh.exists():
        raise FileNotFoundError(
            f"No bioheat outputs at {bh}. Run\n"
            f"  python scripts/run_bioheat.py "
            f"--config <validation_config.py>\n"
            f"first; the validation report no longer recomputes the "
            f"Pennes solve."
        )
    delta_T_final = np.load(bh / "delta_T_final.npy")
    history_t = np.load(bh / "delta_T_history_t.npy")
    history_peak = np.load(bh / "delta_T_history_peak.npy")
    frames_path = bh / "delta_T_frames.npy"
    times_path = bh / "delta_T_frame_times.npy"
    frames = np.load(frames_path) if frames_path.exists() else None
    frame_times = np.load(times_path) if times_path.exists() else None
    indices = json.loads((bh / "thermal_indices.json").read_text())
    metrics = json.loads((bh / "thermal_metrics.json").read_text())
    return {
        "delta_T_final": delta_T_final,
        "frames": frames,
        "frame_times": frame_times,
        "history_t": history_t,
        "history_peak": history_peak,
        "peak_idx": tuple(indices["peak_idx"]),
        "brain_peak_idx": tuple(indices.get("brain_peak_idx",
                                             indices["focus_idx"])),
        "focus_idx": tuple(indices["focus_idx"]),
        "metrics": metrics,
    }



# =====================================================================
# Figures: orthogonal slices, 3-D iso-surfaces, time history
# =====================================================================


def _ortho_plot(delta_T: np.ndarray, run: dict[str, Any], peak_idx: tuple,
                out_path: Path, title_prefix: str = "Transcranial") -> None:
    """Three orthogonal plane views of DeltaT through the peak voxel,
    with depth (x) on the y-axis and 0 on top (radiology convention).
    """
    skull = run["c"] > 1600.0
    nx, ny, nz = delta_T.shape
    dx, dy, dz = run["dx"] * 1e3, run["dy"] * 1e3, run["dz"] * 1e3
    Lx, Ly, Lz = nx * dx, ny * dy, nz * dz
    pk = peak_idx
    vmax = float(np.percentile(delta_T, 99.9))
    if vmax <= 0:
        vmax = max(float(delta_T.max()), 1e-3)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), constrained_layout=True)

    # x-y plane (mid z = peak z): rows = depth x, cols = lateral y
    ax = axes[0]
    im = ax.imshow(delta_T[:, :, pk[2]], cmap="hot", origin="upper",
                   vmin=0, vmax=vmax, aspect="equal",
                   extent=(0, Ly, Lx, 0))
    ax.contour(np.linspace(0, Ly, ny), np.linspace(0, Lx, nx),
               skull[:, :, pk[2]], levels=[0.5], colors="cyan",
               linewidths=0.6)
    ax.set_xlabel("y (mm)"); ax.set_ylabel("x, depth (mm)")
    ax.set_title(f"x-y plane (z={pk[2] * dz:.1f} mm)")

    # x-z plane (mid y = peak y)
    ax = axes[1]
    ax.imshow(delta_T[:, pk[1], :], cmap="hot", origin="upper",
              vmin=0, vmax=vmax, aspect="equal",
              extent=(0, Lz, Lx, 0))
    ax.contour(np.linspace(0, Lz, nz), np.linspace(0, Lx, nx),
               skull[:, pk[1], :], levels=[0.5], colors="cyan",
               linewidths=0.6)
    ax.set_xlabel("z (mm)"); ax.set_ylabel("x, depth (mm)")
    ax.set_title(f"x-z plane (y={pk[1] * dy:.1f} mm)")

    # y-z plane (mid x = peak x): keep natural orientation
    ax = axes[2]
    ax.imshow(delta_T[pk[0], :, :].T, cmap="hot", origin="lower",
              vmin=0, vmax=vmax, aspect="equal",
              extent=(0, Ly, 0, Lz))
    ax.contour(np.linspace(0, Ly, ny), np.linspace(0, Lz, nz),
               skull[pk[0], :, :].T, levels=[0.5], colors="cyan",
               linewidths=0.6)
    ax.set_xlabel("y (mm)"); ax.set_ylabel("z (mm)")
    ax.set_title(f"y-z plane (x={pk[0] * dx:.1f} mm)")

    cbar = fig.colorbar(im, ax=axes, location="right", shrink=0.85, pad=0.02)
    cbar.set_label(r"$\Delta T$ (K)")
    fig.suptitle(f"{title_prefix}: thermal-rise field with skull overlay",
                 fontsize=10)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _iso_3d_slab(delta_T: np.ndarray, run: dict[str, Any],
                 out_path: Path, title_prefix: str = "Transcranial") -> None:
    """Slab-coordinate fallback for sims without placement metadata."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from skimage.measure import marching_cubes

    Tmax = float(delta_T.max())
    skull = run["c"] > 1600.0
    dx, dy, dz = run["dx"] * 1e3, run["dy"] * 1e3, run["dz"] * 1e3
    nx, ny, nz = delta_T.shape

    surfaces = []
    for frac, color, alpha, label in (
        (0.50, "red", 0.85, "50 % peak"),
        (0.25, "orange", 0.45, "25 % peak"),
    ):
        thresh = Tmax * frac
        try:
            verts, faces, _, _ = marching_cubes(delta_T, level=thresh)
        except (RuntimeError, ValueError):
            continue
        verts_mm = verts * np.array([dx, dy, dz])
        surfaces.append((verts_mm, faces, color, alpha, label))

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
        for verts_mm, faces, color, alpha, _lbl in surfaces:
            mesh = Poly3DCollection(verts_mm[faces], facecolor=color,
                                     edgecolor="none", alpha=alpha)
            ax.add_collection3d(mesh)
        ax.set_xlim(0, nx * dx)
        ax.set_ylim(0, ny * dy)
        ax.set_zlim(0, nz * dz)
        ax.set_box_aspect((nx * dx, ny * dy, nz * dz))
        ax.set_xlabel("x, depth (mm)", labelpad=2)
        ax.set_ylabel("y, lateral (mm)", labelpad=2)
        ax.set_zlabel("z, elevation (mm)", labelpad=2)
        ax.tick_params(labelsize=7)
        ax.view_init(**vk)
        ax.set_title(vtitle, fontsize=10)

    fig.suptitle(
        f"{title_prefix}: $\\Delta T$ iso-surfaces in 3-D — "
        f"red: 50 \\% of peak ({Tmax * 0.5:.3f} K); "
        f"orange: 25 \\% of peak ({Tmax * 0.25:.3f} K); "
        f"grey: skull ($c>1600$ m/s)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _iso_3d(delta_T: np.ndarray, run: dict[str, Any], out_path: Path,
            cfg=None, title_prefix: str = "Transcranial") -> None:
    """Three-camera 3-D rendering of the DeltaT iso-surfaces, patterned
    on the acoustic ``_intensity_3d`` figure (Fig 8): full skull point
    cloud, pial brain mesh, target ROI, bowl wireframe, beam axis, and
    apex/scalp/target/focus markers in LPS coordinates — with the
    50 % / 25 % peak thermal-rise iso-surfaces overlaid.  Falls back to
    the slab-coord renderer when no placement metadata is supplied.
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from skimage.measure import marching_cubes

    if cfg is None:
        _iso_3d_slab(delta_T, run, out_path, title_prefix)
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

    # Slab-grid -> LPS transform (matches _intensity_3d in
    # acoustic_outputs.py).  Lateral/elevational axes centered on apex.
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

    Tmax = float(delta_T.max())
    nx, ny, nz = delta_T.shape
    dx, dy, dz = run["dx"], run["dy"], run["dz"]   # metres
    lat_off = ny / 2.0
    elev_off = nz / 2.0

    def slab_to_lps(verts_idx):
        return (apex
                + (verts_idx[:, 0] * dx * 1e3)[:, None] * n_hat
                + ((verts_idx[:, 1] - lat_off) * dy * 1e3)[:, None] * t_hat
                + ((verts_idx[:, 2] - elev_off) * dz * 1e3)[:, None] * b_hat)

    surfaces = []
    for frac, color, alpha in (
        (0.50, "#d62728", 1.00),  # 50 % peak — solid crimson
        (0.25, "#ff7f0e", 0.65),  # 25 % peak — orange shell
        (0.10, "#ffd92f", 0.30),  # 10 % peak — yellow halo
    ):
        thresh = Tmax * frac
        if thresh <= 0:
            continue
        try:
            verts, faces, _, _ = marching_cubes(delta_T, level=thresh)
        except (RuntimeError, ValueError):
            continue
        surfaces.append((slab_to_lps(verts), faces, color, alpha, frac))

    views = [
        ("Lateral (from patient left)", dict(elev=0,  azim=180)),
        ("Anterior (from face)",        dict(elev=0,  azim=-90)),
        ("Superior (from above)",       dict(elev=90, azim=-90)),
    ]
    fig = plt.figure(figsize=(18, 6))
    for k, (title, vk) in enumerate(views):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        ax.scatter(bone_pts[:, 0], bone_pts[:, 1], bone_pts[:, 2],
                   s=1.0, c="dimgray", alpha=0.12, depthshade=False)
        ax.add_collection3d(Poly3DCollection(brain_v[brain_f_plot],
                                              facecolor="#ffb478",
                                              edgecolor="none", alpha=0.12))
        if len(roi_f_plot) > 0:
            ax.add_collection3d(Poly3DCollection(roi_v[roi_f_plot],
                                                  facecolor="lime",
                                                  edgecolor="none",
                                                  alpha=0.75))
        ax.add_collection3d(Poly3DCollection(bquads, facecolor="steelblue",
                                              edgecolor="k", linewidth=0.2,
                                              alpha=0.85))
        for verts_lps, faces, color, alpha, _frac in surfaces:
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
        ax.set(xlabel="L (mm)", ylabel="P (mm)", zlabel="S (mm)", title=title)
        ax.set_xlim(0, bone_v[:, 0].max())
        ax.set_ylim(0, bone_v[:, 1].max())
        ax.set_zlim(0, bone_v[:, 2].max())
        ax.set_box_aspect([float(np.ptp(bone_v[:, i])) for i in range(3)])

    fig.suptitle(
        f"{title_prefix}: 3-D placement + $\\Delta T$ iso-surfaces "
        f"(crimson 50\\,\\% = {Tmax*0.5:.3f} K, orange 25\\,\\% = "
        f"{Tmax*0.25:.3f} K, yellow 10\\,\\% = {Tmax*0.10:.3f} K) — "
        f"{plc['scalp_site']} → {plc['target_name']}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _time_history_plot(history_t: np.ndarray, history_peak: np.ndarray,
                       frames: np.ndarray | None, frame_times: np.ndarray | None,
                       peak_idx: tuple, brain_peak_idx: tuple,
                       focus_idx: tuple, out_path: Path,
                       title_prefix: str = "Transcranial") -> None:
    """Single-panel time history with three traces: global peak voxel
    (typically skull), brain-tissue peak voxel, and the acoustic focus
    voxel.  Per-voxel traces are taken from captured frames; if frames
    are unavailable, the global-peak trace falls back to the spatial-peak
    history series.
    """
    fig, ax = plt.subplots(figsize=(7, 3.6), constrained_layout=True)

    if frames is not None and frame_times is not None and len(frame_times):
        trace_peak = frames[peak_idx[0], peak_idx[1], peak_idx[2], :]
        trace_brain = frames[brain_peak_idx[0], brain_peak_idx[1],
                             brain_peak_idx[2], :]
        trace_focus = frames[focus_idx[0], focus_idx[1], focus_idx[2], :]
        ax.plot(frame_times, trace_peak, color="crimson", lw=1.4,
                label=f"global peak voxel, skull "
                      f"(peak {trace_peak.max() * 1e3:.0f} mK)")
        ax.plot(frame_times, trace_brain, color="darkorange", lw=1.4,
                label=f"brain-tissue peak voxel "
                      f"(peak {trace_brain.max() * 1e3:.0f} mK)")
        ax.plot(frame_times, trace_focus, color="navy", lw=1.4,
                label=f"acoustic-focus voxel "
                      f"(peak {trace_focus.max() * 1e3:.0f} mK)")
    else:
        ax.plot(history_t, history_peak, color="crimson", lw=1.4,
                label=f"global spatial-peak "
                      f"(max {float(np.max(history_peak)) * 1e3:.0f} mK)")

    ax.set_xlabel("time (s)")
    ax.set_ylabel(r"$\Delta T$ (K)")
    ax.set_title(f"{title_prefix}: per-voxel $\\Delta T$ history through "
                 f"the theta-burst session")
    ax.legend(loc="best", fontsize=9, frameon=False)
    ax.grid(alpha=0.3)

    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# =====================================================================
# Builder entry-point
# =====================================================================


def build(cfg, fig_dir: Path) -> dict[str, Any]:
    """Load bioheat outputs from the run-dir and emit metrics + figures."""
    repo_root = Path(__file__).resolve().parents[2]
    sim_dir = (repo_root / cfg.OUT_DIR).parent
    if not sim_dir.is_absolute():
        sim_dir = (repo_root / sim_dir).resolve()

    run = _ao._load_run(sim_dir)
    therm = _load_bioheat(sim_dir)

    fig_ortho = fig_dir / "thermal_ortho_dT.png"
    fig_3d = fig_dir / "thermal_3d_dT.png"
    fig_hist = fig_dir / "thermal_time_history.png"
    _ortho_plot(therm["delta_T_final"], run, therm["peak_idx"], fig_ortho)
    _iso_3d(therm["delta_T_final"], run, fig_3d, cfg=cfg)
    _time_history_plot(
        therm["history_t"], therm["history_peak"],
        therm["frames"], therm["frame_times"],
        therm["peak_idx"], therm["brain_peak_idx"], therm["focus_idx"],
        fig_hist,
    )

    return {
        "thermal_outputs": therm["metrics"],
        "figures_thermal_outputs": {
            "ortho": fig_ortho.name,
            "iso_3d": fig_3d.name,
            "time_history": fig_hist.name,
        },
    }
