"""Section 2 — Acoustic outputs.

Reads a finished Fullwave acoustic FDTD run (per the directory layout under
``data/<sim_name>/``) and computes the standard exposure metrics expected
in a TUS validation report: PNP, PPP, mechanical index, thermal indices,
ISPPA / ISPTA, ARF density, focal --6 dB volume, FWHMs along the principal
axes, peak particle velocity, and ARF impulse per pulse.

Bioheat-derived (∆T) and elastic-derived (strain, strain rate, strain
gradient) quantities are intentionally out of scope here; they will be
added in subsequent sections built on top of the bioheat and shear-FDTD
outputs.

The builder also renders the three orthogonal-plane views of intensity
through the focal voxel (with the skull mask contoured on top) and a
3-D maximum-intensity-projection rendering of the focal lobe for
spatial context.

The function returns a dict suitable for merging into the report context.
A second (``Homogeneous``) column is added when the config's
``ACOUSTIC_OUTPUTS_HOMOGENEOUS_DIR`` attribute resolves to an existing
companion run; otherwise the metrics table is rendered with one column.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np


# =====================================================================
# Field loaders
# =====================================================================


def _load_run(run_dir: Path) -> dict[str, Any]:
    """Load the arrays and grid parameters needed for metric computation."""
    grid = np.load(run_dir / "grid_params.npz", allow_pickle=True)
    pmap = np.load(run_dir / "propagation_map.npy", mmap_mode="r")

    # Pulse normalization that recovers the CW-equivalent ISPPA from a
    # tone-burst recording: silence_correction lifts <p^2> over the saved
    # window to <p^2> over the active pulse fraction, and cw_correction
    # accounts for the Gaussian envelope of the burst (peak vs CW).
    n_tic = int(grid["n_tic"])
    smt = (int(grid["sampling_modulus_time"])
           if "sampling_modulus_time" in grid.files else 1)
    n_source_steps = max(n_tic // smt, 1)
    silence_correction = pmap.shape[0] / n_source_steps
    cw_correction = float(grid["cw_correction"])
    pulse_norm = silence_correction * cw_correction

    out = dict(
        pmap=pmap,
        vfield=np.load(run_dir / "velocity.npy"),
        arf=np.load(run_dir / "radiation_force.npy"),
        c=np.load(run_dir / "c_ds.npy"),
        rho=np.load(run_dir / "rho_ds.npy"),
        alpha=np.load(run_dir / "alpha_ds.npy"),
        f0=float(grid["f0"]),
        dx=float(grid["dx"]) * float(grid["mod_x"]),
        dy=float(grid["dy"]) * float(grid["mod_y"]),
        dz=float(grid["dz"]) * float(grid["mod_z"]),
        dt_save=float(grid["dt"]) * smt if "dt" in grid.files else None,
        mod_x=int(grid["mod_x"]),
        mod_y=int(grid["mod_y"]),
        mod_z=int(grid["mod_z"]),
        n_tic=n_tic,
        sampling_modulus_time=smt,
        n_source_steps=n_source_steps,
        cw_correction=cw_correction,
        silence_correction=silence_correction,
        pulse_norm=pulse_norm,
    )
    return out


# =====================================================================
# Per-voxel metric fields
# =====================================================================


def _pnp_ppp_fields(pmap: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return PNP (most-negative pressure) and PPP (most-positive) per voxel.

    Uses the full saved window: with deeper domains the wave reaches the
    focus in the first half of the snapshot record and the latter half
    contains only post-focal ring-down, so a half-cut underestimates PNP.
    """
    arr = np.asarray(pmap)
    return arr.min(axis=0), arr.max(axis=0)


def _intensity_field(p_amp: np.ndarray, rho: np.ndarray,
                     c: np.ndarray) -> np.ndarray:
    """Spatial-peak-pulse-average intensity per voxel (W/m²).

    For a steady-state CW-equivalent pressure amplitude ``p_amp`` (Pa),
    the time-averaged intensity is :math:`I = p_amp^2 / (2 \\rho c)`.
    """
    return (p_amp ** 2) / (2.0 * rho * c)


def _smooth_intensity_field(pmap: np.ndarray, rho: np.ndarray,
                            c: np.ndarray, pulse_norm: float = 1.0,
                            ) -> np.ndarray:
    """Pulse-normalised intensity field for visualisation.

    Computes ``<p^2>/(rho c)`` over the saved propagation_map window
    (the canonical convention used by ``compute_isppa``), multiplied by
    the silence x cw_correction product so the result is the CW-
    equivalent ISPPA in W/m^2.  A 3x3x3 max filter + sigma=1 voxel
    Gaussian smoothing fills the staggered-zero aliasing pattern that
    arises from the spatial mod_x/y/z decimation in the saved pmap.
    """
    from scipy.ndimage import gaussian_filter, maximum_filter

    pressure_4d = np.asarray(pmap).transpose(1, 2, 3, 0)
    isppa = (pressure_4d ** 2).mean(axis=3) / (rho * c)
    isppa = maximum_filter(isppa, size=3)
    isppa = gaussian_filter(isppa, sigma=1.0)
    return isppa * pulse_norm


def _peak_pressure_amplitude(pnp: np.ndarray, ppp: np.ndarray) -> np.ndarray:
    """Peak pressure amplitude per voxel: ``max(|PNP|, PPP)``."""
    return np.maximum(np.abs(pnp), ppp)


# =====================================================================
# Aggregate exposure metrics
# =====================================================================


def _compute_metrics(run: dict[str, Any], duty_session: float,
                     pulse_duration_s: float) -> dict[str, float]:
    """Compute every scalar listed in the validation table for a run.

    Excludes bioheat-derived and elastic-derived rows (∆T, strain,
    strain rate, strain gradient) by design.
    """
    pmap = run["pmap"]
    pnp_field, ppp_field = _pnp_ppp_fields(pmap)
    p_amp = _peak_pressure_amplitude(pnp_field, ppp_field)
    I = _intensity_field(p_amp, run["rho"], run["c"])
    arf = run["arf"]
    vfield = run["vfield"]
    f0 = run["f0"]
    dx, dy, dz = run["dx"], run["dy"], run["dz"]
    voxel_vol_m3 = dx * dy * dz

    pnp_pa = float(np.abs(pnp_field).max())
    ppp_pa = float(ppp_field.max())
    f0_mhz = f0 * 1e-6

    # Mechanical index (FDA): MI = |PNP|_MPa / sqrt(f0_MHz)
    mi = (pnp_pa * 1e-6) / np.sqrt(f0_mhz)

    # Spatial-peak intensities
    isppa_w_per_m2 = float(I.max())
    isppa_w_per_cm2 = isppa_w_per_m2 * 1e-4
    ispta_w_per_cm2 = isppa_w_per_cm2 * duty_session

    # Thermal indices: TI = W_acoustic / W_deg, with the IEC 62359 / ITRUSST
    # soft-tissue and cranial-bone reference powers (W_deg = 210 mW soft;
    # 40 mW cranial at typical scan apertures). Use TIS = ISPTA·A_focal /
    # 210 mW with A_focal = π·(FWHM_lat/2)·(FWHM_elev/2). Compute FWHM
    # below; we set provisional values here and update if needed.

    # --- −6 dB focal volume + FWHMs ----------------------------------
    Imax = isppa_w_per_m2
    half_mask = I >= 0.5 * Imax
    n_half = int(half_mask.sum())
    minus_6dB_volume_mm3 = n_half * voxel_vol_m3 * 1e9

    # FWHMs through the peak voxel along each axis
    peak_idx = np.unravel_index(int(np.argmax(I)), I.shape)
    fwhm = {}
    for axis_name, axis_idx, ds in (
        ("x", 0, dx), ("y", 1, dy), ("z", 2, dz),
    ):
        # Pull the 1-D profile through the peak voxel along this axis
        sl = [peak_idx[0], peak_idx[1], peak_idx[2]]
        sl[axis_idx] = slice(None)
        profile = I[tuple(sl)]
        above = profile >= 0.5 * Imax
        if above.any():
            idxs = np.where(above)[0]
            fwhm_mm = (idxs[-1] - idxs[0] + 1) * ds * 1e3
        else:
            fwhm_mm = 0.0
        fwhm[axis_name] = fwhm_mm

    mean_I_in_minus_6dB = float(I[half_mask].mean()) if n_half else 0.0
    mean_I_in_minus_6dB_w_cm2 = mean_I_in_minus_6dB * 1e-4

    # --- Thermal indices (rough, ITRUSST-style) ----------------------
    fwhm_lat_m = fwhm["y"] * 1e-3
    fwhm_elev_m = fwhm["z"] * 1e-3
    a_focal_m2 = np.pi * 0.25 * fwhm_lat_m * fwhm_elev_m
    p_acoustic_w = ispta_w_per_cm2 * 1e4 * a_focal_m2
    tis = p_acoustic_w / 0.210     # 210 mW reference for soft tissue
    tic = p_acoustic_w / 0.040     # 40 mW reference for cranial bone

    # --- Particle velocity / ARF impulse -----------------------------
    v_mag = np.sqrt((vfield ** 2).sum(axis=-1))
    peak_v_m_per_s = float(v_mag.max())
    arf_peak_n_per_m3 = float(arf.max())
    arf_impulse_per_pulse_pa_s = arf_peak_n_per_m3 * pulse_duration_s

    return dict(
        pnp_mpa=pnp_pa * 1e-6,
        ppp_mpa=ppp_pa * 1e-6,
        mi=float(mi),
        ti_soft=float(tis),
        ti_cranial=float(tic),
        isppa_w_cm2=float(isppa_w_per_cm2),
        ispta_w_cm2=float(ispta_w_per_cm2),
        arf_n_per_m3=arf_peak_n_per_m3,
        minus_6db_vol_mm3=float(minus_6dB_volume_mm3),
        fwhm_x_mm=float(fwhm["x"]),
        fwhm_y_mm=float(fwhm["y"]),
        fwhm_z_mm=float(fwhm["z"]),
        mean_I_w_cm2=float(mean_I_in_minus_6dB_w_cm2),
        peak_v_m_per_s=peak_v_m_per_s,
        arf_impulse_pa_s=float(arf_impulse_per_pulse_pa_s),
        peak_idx=peak_idx,
    )


# =====================================================================
# Pulse-normalization figure: shows what the gating constants do
# =====================================================================


def _pulse_norm_plot(run: dict[str, Any], out_path: Path,
                     title_prefix: str = "Transcranial") -> None:
    """Diagnostic figure for the pulse-normalization constants.

    Left panel: spatial-mean ``<p^2>(t)`` over the saved propagation
    map, shown on a log scale.  The red shaded band marks the active
    pulse window (samples where the spatial-mean exceeds 10 % of its
    peak); the silence correction lifts ``<p^2>`` from the full-window
    average to the active-window average.

    Right panel: the per-voxel time history at the spatial-peak voxel,
    annotated with the burst envelope (peak amplitude in red) and the
    CW-equivalent amplitude (peak / cw_correction in orange) so the
    cw-correction factor is visible.
    """
    pmap = np.asarray(run["pmap"]).astype(np.float32)
    nt = pmap.shape[0]
    dt_save = run.get("dt_save")
    # spatial-mean p²(t)
    mean_p2 = (pmap ** 2).mean(axis=(1, 2, 3))
    thresh = 0.1 * mean_p2.max()
    gate_mask = mean_p2 >= thresh
    n_active = int(gate_mask.sum())
    silence = nt / max(n_active, 1)
    cw_corr = float(run["cw_correction"])
    pulse_norm_full = silence * cw_corr

    # peak-voxel time history (use the ARF-derived focal location)
    I = _smooth_intensity_field(run["pmap"], run["rho"], run["c"], pulse_norm=run["pulse_norm"])
    pk = np.unravel_index(int(np.argmax(I)), I.shape)
    ts = pmap[:, pk[0], pk[1], pk[2]]
    ts_peak = float(np.abs(ts).max())

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), constrained_layout=True)

    if dt_save is not None and dt_save > 0:
        t_axis = np.arange(nt) * dt_save * 1e6  # µs
        x_label = r"time ($\mu$s)"
    else:
        t_axis = np.arange(nt)
        x_label = "saved sample index"
    ax = axes[0]
    ax.semilogy(t_axis, np.maximum(mean_p2, 1e-6), color="k", lw=1.0)
    if gate_mask.any():
        first = t_axis[int(np.argmax(gate_mask))]
        last = t_axis[int(nt - 1 - np.argmax(gate_mask[::-1]))]
        ax.axvspan(first, last, color="red", alpha=0.18,
                   label=f"active pulse window (n={n_active})")
    ax.axhline(thresh, color="red", lw=0.8, ls="--",
               label=r"10 \% threshold")
    ax.set_xlabel(x_label)
    ax.set_ylabel(r"$\langle p^2 \rangle_{xyz}$ (Pa$^2$)")
    ax.set_title(f"Spatial-mean $p^2(t)$  —  silence corr. "
                 f"$= {nt}/{n_active}={silence:.2f}$")
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[1]
    ax.plot(t_axis, ts * 1e-6, color="k", lw=0.8)
    ax.axhline(ts_peak * 1e-6, color="red", lw=0.8, ls="--",
               label=f"burst peak {ts_peak * 1e-6:.2f} MPa")
    ax.axhline(ts_peak * 1e-6 / cw_corr, color="orange", lw=0.8, ls="--",
               label=f"CW-eq. peak / cw\\_corr = {ts_peak * 1e-6 / cw_corr:.2f} MPa")
    ax.set_xlabel(x_label)
    ax.set_ylabel("p (MPa)")
    ax.set_title(f"Time history at spatial peak  —  cw\\_correction $= {cw_corr:.2f}$")
    ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(f"{title_prefix}: pulse normalization "
                 f"$= \\mathrm{{silence}} \\times \\mathrm{{cw}} "
                 f"= {silence:.2f} \\times {cw_corr:.2f} = {pulse_norm_full:.2f}$",
                 fontsize=10)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# =====================================================================
# Figures: orthogonal slices and 3-D rendering
# =====================================================================


def _ortho_plot(run: dict[str, Any], out_path: Path,
                title_prefix: str = "Transcranial") -> None:
    """Three orthogonal plane views of intensity through the focal voxel,
    with the skull mask contoured on top.
    """
    I = _smooth_intensity_field(run["pmap"], run["rho"], run["c"], pulse_norm=run["pulse_norm"])
    I_db = 10.0 * np.log10(np.maximum(I / I.max(), 1e-4))

    skull_mask = run["c"] > 1700.0  # speed-of-sound discriminator
    pk = np.unravel_index(int(np.argmax(I)), I.shape)
    nx, ny, nz = I.shape
    dx, dy, dz = run["dx"] * 1e3, run["dy"] * 1e3, run["dz"] * 1e3
    extents = {
        "xy": (0, ny * dy, nx * dx, 0),
        "xz": (0, nz * dz, nx * dx, 0),
        "yz": (0, nz * dz, ny * dy, 0),
    }

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), constrained_layout=True)

    # x-y slice (z = focal)
    ax = axes[0]
    im = ax.imshow(I_db[:, :, pk[2]], cmap="inferno", vmin=-30, vmax=0,
                   extent=extents["xy"], aspect="equal")
    ax.contour(np.linspace(0, ny * dy, ny), np.linspace(0, nx * dx, nx),
               skull_mask[:, :, pk[2]], levels=[0.5], colors="white",
               linewidths=0.6)
    ax.set_xlabel("y (mm)"); ax.set_ylabel("x, depth (mm)")
    ax.set_title(f"x-y plane (z={pk[2] * dz:.1f} mm)")

    # x-z slice (y = focal)
    ax = axes[1]
    ax.imshow(I_db[:, pk[1], :], cmap="inferno", vmin=-30, vmax=0,
              extent=extents["xz"], aspect="equal")
    ax.contour(np.linspace(0, nz * dz, nz), np.linspace(0, nx * dx, nx),
               skull_mask[:, pk[1], :], levels=[0.5], colors="white",
               linewidths=0.6)
    ax.set_xlabel("z (mm)"); ax.set_ylabel("x, depth (mm)")
    ax.set_title(f"x-z plane (y={pk[1] * dy:.1f} mm)")

    # y-z slice (x = focal)
    ax = axes[2]
    ax.imshow(I_db[pk[0], :, :], cmap="inferno", vmin=-30, vmax=0,
              extent=extents["yz"], aspect="equal")
    ax.contour(np.linspace(0, nz * dz, nz), np.linspace(0, ny * dy, ny),
               skull_mask[pk[0], :, :], levels=[0.5], colors="white",
               linewidths=0.6)
    ax.set_xlabel("z (mm)"); ax.set_ylabel("y (mm)")
    ax.set_title(f"y-z plane (x={pk[0] * dx:.1f} mm)")

    cbar = fig.colorbar(im, ax=axes, location="right", shrink=0.85, pad=0.02)
    cbar.set_label("Intensity (dB re max)")
    fig.suptitle(f"{title_prefix}: focal-plane intensity with skull overlay",
                 fontsize=10)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _intensity_3d(run: dict[str, Any], out_path: Path, cfg=None,
                  title_prefix: str = "Transcranial") -> None:
    """Three-camera 3-D rendering matching the bottom row of Figure 1
    (placement_3d): full skull point cloud, pial brain mesh, target ROI,
    bowl wireframe, beam axis, and apex/scalp/target/focus markers in LPS
    coordinates — with the focal-zone -3 dB / -6 dB intensity iso-surfaces
    overlaid (slab voxel indices transformed into LPS via the placement).
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from skimage.measure import marching_cubes

    from ultraneuron.registration import ensure_registration_importable

    ensure_registration_importable()
    import visualize_placement_3d as viz  # noqa: E402
    from halle_placement import place_on_skull  # noqa: E402

    if cfg is None:
        # No placement context — fall back to slab-coord rendering.
        _intensity_3d_slab(run, out_path, title_prefix)
        return

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
    # Full bone surface (no subsampling); per-point alpha kept low so the
    # focal iso-surfaces still read through the denser skull.
    bone_pts = bone_v
    brain_f_plot = (brain_f[rng.choice(len(brain_f), 12000, replace=False)]
                    if len(brain_f) > 12000 else brain_f)

    if target_roi.any():
        roi_v, roi_f, _, _ = marching_cubes(target_roi.astype(np.uint8), level=0.5)
        roi_f_plot = (roi_f[rng.choice(len(roi_f), 6000, replace=False)]
                      if len(roi_f) > 6000 else roi_f)
    else:
        roi_v = np.zeros((0, 3))
        roi_f_plot = np.zeros((0, 3), dtype=int)

    # --- Build slab-grid -> LPS transform from the placement ---
    # Slab voxel (i, j, k) at downsampled spacing (dx_ds, dy_ds, dz_ds)
    # maps to LPS via:
    #   LPS = xdc_center_lps + (i*dx_ds)*normal
    #         + (j*dy_ds - lat_offset)*tangent
    #         + (k*dz_ds - elev_offset)*bitangent
    # offsets center the lateral/elevational axes on the apex.
    n_hat = np.asarray(p.get("normal_3d", beam))
    n_hat = n_hat / (np.linalg.norm(n_hat) + 1e-12)
    if "tangent_3d" in p and "bitangent_3d" in p:
        t_hat = np.asarray(p["tangent_3d"])
        b_hat = np.asarray(p["bitangent_3d"])
    else:
        # Build an orthonormal frame from beam direction
        ref = np.array([0.0, 0.0, 1.0]) if abs(n_hat[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        t_hat = np.cross(n_hat, ref); t_hat /= (np.linalg.norm(t_hat) + 1e-12)
        b_hat = np.cross(n_hat, t_hat)
    t_hat = t_hat / (np.linalg.norm(t_hat) + 1e-12)
    b_hat = b_hat / (np.linalg.norm(b_hat) + 1e-12)

    I = _smooth_intensity_field(run["pmap"], run["rho"], run["c"],
                                pulse_norm=run["pulse_norm"])
    Imax = float(I.max())
    nx, ny, nz = I.shape
    dx, dy, dz = run["dx"], run["dy"], run["dz"]   # metres
    lat_off = ny / 2.0
    elev_off = nz / 2.0

    def slab_to_lps(verts_idx):
        # verts_idx: (N, 3) in (i, j, k) slab indices
        return (apex
                + (verts_idx[:, 0] * dx * 1e3)[:, None] * n_hat
                + ((verts_idx[:, 1] - lat_off) * dy * 1e3)[:, None] * t_hat
                + ((verts_idx[:, 2] - elev_off) * dz * 1e3)[:, None] * b_hat)

    # Iso-surfaces in slab-index space, then transform to LPS.
    # Outer halo (-9 dB) is added so the focal lobe reads at distance even
    # when partially occluded by skull points.
    surfaces = []
    for thresh_db, color, alpha in ((-3.0, "#d62728", 1.00),   # solid crimson
                                    (-6.0, "#ff7f0e", 0.65),   # orange shell
                                    (-9.0, "#ffd92f", 0.30)):  # yellow halo
        try:
            verts, faces, _, _ = marching_cubes(I, level=Imax * 10.0 ** (thresh_db / 10.0))
        except (RuntimeError, ValueError):
            continue
        surfaces.append((slab_to_lps(verts), faces, color, alpha, thresh_db))

    views = [
        ("Lateral (from patient left)", dict(elev=0,  azim=180)),
        ("Anterior (from face)",        dict(elev=0,  azim=-90)),
        ("Superior (from above)",       dict(elev=90, azim=-90)),
    ]
    fig = plt.figure(figsize=(18, 6))
    for k, (title, vk) in enumerate(views):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        # Translucent skull so the iso-surface inside reads clearly.
        ax.scatter(bone_pts[:, 0], bone_pts[:, 1], bone_pts[:, 2],
                   s=1.0, c="dimgray", alpha=0.12, depthshade=False)
        ax.add_collection3d(Poly3DCollection(brain_v[brain_f_plot],
                                             facecolor="#ffb478",
                                             edgecolor="none", alpha=0.12))
        if len(roi_f_plot) > 0:
            ax.add_collection3d(Poly3DCollection(roi_v[roi_f_plot],
                                                 facecolor="lime",
                                                 edgecolor="none", alpha=0.75))
        ax.add_collection3d(Poly3DCollection(bquads, facecolor="steelblue",
                                             edgecolor="k", linewidth=0.2,
                                             alpha=0.85))
        for verts_lps, faces, color, alpha, _db in surfaces:
            ax.add_collection3d(Poly3DCollection(verts_lps[faces],
                                                 facecolor=color,
                                                 edgecolor="none",
                                                 alpha=alpha))
        ax.plot(*list(zip(apex, focus)), color="red", lw=2.5)
        ax.scatter(*apex,   color="blue", s=70,  edgecolor="k", zorder=5)
        ax.scatter(*scalp,  color="cyan", s=60,  edgecolor="k", marker="s", zorder=5)
        ax.scatter(*target, color="lime", s=120, edgecolor="k", zorder=5)
        ax.scatter(*focus,  color="red",  s=120, marker="*", edgecolor="k", zorder=5)
        ax.view_init(**vk)
        ax.set(xlabel="L (mm)", ylabel="P (mm)", zlabel="S (mm)", title=title)
        ax.set_xlim(0, bone_v[:, 0].max())
        ax.set_ylim(0, bone_v[:, 1].max())
        ax.set_zlim(0, bone_v[:, 2].max())
        ax.set_box_aspect([float(np.ptp(bone_v[:, i])) for i in range(3)])

    fig.suptitle(f"{title_prefix}: 3-D placement + focal iso-surfaces "
                 f"(crimson −3 dB, orange −6 dB, yellow −9 dB) — "
                 f"{plc['scalp_site']} → {plc['target_name']}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _intensity_3d_slab(run: dict[str, Any], out_path: Path,
                       title_prefix: str = "Transcranial") -> None:
    """Slab-coordinate fallback for sims without placement metadata."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from skimage.measure import marching_cubes

    I = _smooth_intensity_field(run["pmap"], run["rho"], run["c"],
                                pulse_norm=run["pulse_norm"])
    Imax = float(I.max())
    skull = run["c"] > 1700.0
    dx, dy, dz = run["dx"] * 1e3, run["dy"] * 1e3, run["dz"] * 1e3
    nx, ny, nz = I.shape
    surfaces = []
    for thresh_db, color, alpha in ((-3.0, "red", 0.85), (-6.0, "orange", 0.45)):
        try:
            verts, faces, _, _ = marching_cubes(I, level=Imax * 10.0 ** (thresh_db / 10.0))
        except (RuntimeError, ValueError):
            continue
        surfaces.append((verts * np.array([dx, dy, dz]), faces, color, alpha))
    skull_idx = np.argwhere(skull)
    if len(skull_idx) > 6000:
        rng = np.random.default_rng(0)
        skull_idx = skull_idx[rng.choice(len(skull_idx), 6000, replace=False)]
    skull_mm = skull_idx * np.array([dx, dy, dz])
    views = (("Side", dict(elev=0, azim=-90)),
             ("Front", dict(elev=0, azim=0)),
             ("Top",   dict(elev=90, azim=-90)))
    fig = plt.figure(figsize=(12, 4.2))
    for k, (vt, vk) in enumerate(views):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        ax.scatter(skull_mm[:, 0], skull_mm[:, 1], skull_mm[:, 2],
                   s=1.4, c="dimgray", alpha=0.30)
        for verts_mm, faces, color, alpha in surfaces:
            ax.add_collection3d(Poly3DCollection(verts_mm[faces],
                                                 facecolor=color,
                                                 edgecolor="none", alpha=alpha))
        ax.set_xlim(0, nx * dx); ax.set_ylim(0, ny * dy); ax.set_zlim(0, nz * dz)
        ax.set_box_aspect((nx * dx, ny * dy, nz * dz))
        ax.set(xlabel="x (mm)", ylabel="y (mm)", zlabel="z (mm)", title=vt)
        ax.view_init(**vk)
    fig.suptitle(f"{title_prefix}: focal-zone iso-surfaces (slab coords)",
                 fontsize=10)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# =====================================================================
# Combined intensity figure: ortho slices (top) + 3D iso-surfaces (bottom)
# =====================================================================


def _intensity_combined(run: dict[str, Any], out_path: Path, cfg=None,
                        title_prefix: str = "Transcranial") -> None:
    """Single 2x3 figure replacing the separate ortho + 3D intensity panels.

    Top row: orthogonal intensity slices through the focal voxel (xy/xz/yz)
    in dB re max, with skull mask contoured in white.
    Bottom row: three 3D camera angles (Lateral/Anterior/Superior) of the
    full skull point cloud, brain mesh, target ROI, bowl wireframe, and
    nested -3/-6/-9 dB intensity iso-surfaces transformed into LPS.
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from skimage.measure import marching_cubes

    from ultraneuron.registration import ensure_registration_importable

    ensure_registration_importable()
    import visualize_placement_3d as viz  # noqa: E402
    from halle_placement import place_on_skull  # noqa: E402

    I = _smooth_intensity_field(run["pmap"], run["rho"], run["c"],
                                pulse_norm=run["pulse_norm"])
    Imax = float(I.max())
    I_db = 10.0 * np.log10(np.maximum(I / Imax, 1e-4))
    skull_mask = run["c"] > 1700.0
    pk = np.unravel_index(int(np.argmax(I)), I.shape)
    nx, ny, nz = I.shape
    dx, dy, dz = run["dx"] * 1e3, run["dy"] * 1e3, run["dz"] * 1e3

    fig = plt.figure(figsize=(18, 10))

    # ---- Top row: orthogonal intensity slices ---------------------------
    ext = {"xy": (0, ny * dy, nx * dx, 0),
           "xz": (0, nz * dz, nx * dx, 0),
           "yz": (0, nz * dz, ny * dy, 0)}
    for col, (slc, ax_title, ext_key, contour, xlabel, ylabel) in enumerate([
        (I_db[:, :, pk[2]], f"x-y plane (z={pk[2]*dz:.1f} mm)", "xy",
         skull_mask[:, :, pk[2]], "y (mm)", "x, depth (mm)"),
        (I_db[:, pk[1], :], f"x-z plane (y={pk[1]*dy:.1f} mm)", "xz",
         skull_mask[:, pk[1], :], "z (mm)", "x, depth (mm)"),
        (I_db[pk[0], :, :], f"y-z plane (x={pk[0]*dx:.1f} mm)", "yz",
         skull_mask[pk[0], :, :], "z (mm)", "y (mm)"),
    ]):
        ax = fig.add_subplot(2, 3, col + 1)
        im = ax.imshow(slc, cmap="inferno", vmin=-30, vmax=0,
                       extent=ext[ext_key], aspect="equal")
        h, w = contour.shape
        ax.contour(np.linspace(ext[ext_key][0], ext[ext_key][1], w),
                   np.linspace(ext[ext_key][3], ext[ext_key][2], h),
                   contour, levels=[0.5], colors="white", linewidths=0.6)
        ax.set(xlabel=xlabel, ylabel=ylabel, title=ax_title)

    # Reserve right-margin in the gridspec for an explicit cax — done up front
    # so neither aspect="equal" on 2D panels nor 3D axes can collide with it.
    fig.subplots_adjust(left=0.05, right=0.90, top=0.92, bottom=0.06,
                        wspace=0.30, hspace=0.30)
    cax = fig.add_axes([0.915, 0.56, 0.012, 0.34])  # [l, b, w, h] fig coords
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Intensity (dB re max)")

    # ---- Bottom row: 3D scene with iso-surfaces ------------------------
    if cfg is None:
        for k in range(3):
            ax = fig.add_subplot(2, 3, 4 + k)
            ax.text(0.5, 0.5, "(no placement context)", ha="center",
                    va="center", transform=ax.transAxes)
            ax.axis("off")
        fig.suptitle(f"{title_prefix}: focal-plane intensity + 3D iso-surfaces",
                     fontsize=12, fontweight="bold")
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return

    plc = cfg.PLACEMENT
    p = place_on_skull(scalp_site=plc["scalp_site"],
                       target_name=plc["target_name"],
                       focal_length_mm=plc["focal_length_mm"],
                       bowl_radius_mm=plc["bowl_radius_mm"])
    p["target_name"] = plc["target_name"]
    apex = np.asarray(p["xdc_center_lps"])
    target = np.asarray(p["target_lps"])
    focus = np.asarray(p["focus_lps"])
    scalp = np.asarray(p["scalp_contact_lps"])
    beam = np.asarray(p["beam_dir_3d"])
    F = float(p["focal_length_mm"]); R = float(p["bowl_radius_mm"])

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
        roi_v, roi_f, _, _ = marching_cubes(target_roi.astype(np.uint8), level=0.5)
        roi_f_plot = (roi_f[rng.choice(len(roi_f), 6000, replace=False)]
                      if len(roi_f) > 6000 else roi_f)
    else:
        roi_v = np.zeros((0, 3)); roi_f_plot = np.zeros((0, 3), dtype=int)

    n_hat = np.asarray(p.get("normal_3d", beam))
    n_hat = n_hat / (np.linalg.norm(n_hat) + 1e-12)
    if "tangent_3d" in p and "bitangent_3d" in p:
        t_hat = np.asarray(p["tangent_3d"])
        b_hat = np.asarray(p["bitangent_3d"])
    else:
        ref = (np.array([0., 0., 1.]) if abs(n_hat[2]) < 0.9
               else np.array([1., 0., 0.]))
        t_hat = np.cross(n_hat, ref); t_hat /= (np.linalg.norm(t_hat) + 1e-12)
        b_hat = np.cross(n_hat, t_hat)
    t_hat /= (np.linalg.norm(t_hat) + 1e-12)
    b_hat /= (np.linalg.norm(b_hat) + 1e-12)

    dx_m, dy_m, dz_m = run["dx"], run["dy"], run["dz"]
    lat_off, elev_off = ny / 2.0, nz / 2.0

    def slab_to_lps(verts_idx):
        return (apex
                + (verts_idx[:, 0] * dx_m * 1e3)[:, None] * n_hat
                + ((verts_idx[:, 1] - lat_off) * dy_m * 1e3)[:, None] * t_hat
                + ((verts_idx[:, 2] - elev_off) * dz_m * 1e3)[:, None] * b_hat)

    surfaces = []
    for thresh_db, color, alpha in ((-3.0, "#d62728", 1.00),
                                    (-6.0, "#ff7f0e", 0.65),
                                    (-9.0, "#ffd92f", 0.30)):
        try:
            verts, faces, _, _ = marching_cubes(I, level=Imax * 10.0 ** (thresh_db / 10.0))
        except (RuntimeError, ValueError):
            continue
        surfaces.append((slab_to_lps(verts), faces, color, alpha))

    views = [("Lateral (from patient left)", dict(elev=0,  azim=180)),
             ("Anterior (from face)",        dict(elev=0,  azim=-90)),
             ("Superior (from above)",       dict(elev=90, azim=-90))]
    for k, (title, vk) in enumerate(views):
        ax = fig.add_subplot(2, 3, 4 + k, projection="3d")
        ax.scatter(bone_pts[:, 0], bone_pts[:, 1], bone_pts[:, 2],
                   s=1.0, c="dimgray", alpha=0.12, depthshade=False)
        ax.add_collection3d(Poly3DCollection(brain_v[brain_f_plot],
                                             facecolor="#ffb478",
                                             edgecolor="none", alpha=0.12))
        if len(roi_f_plot) > 0:
            ax.add_collection3d(Poly3DCollection(roi_v[roi_f_plot],
                                                 facecolor="lime",
                                                 edgecolor="none", alpha=0.75))
        ax.add_collection3d(Poly3DCollection(bquads, facecolor="steelblue",
                                             edgecolor="k", linewidth=0.2,
                                             alpha=0.85))
        for verts_lps, faces, color, alpha in surfaces:
            ax.add_collection3d(Poly3DCollection(verts_lps[faces],
                                                 facecolor=color,
                                                 edgecolor="none",
                                                 alpha=alpha))
        ax.plot(*list(zip(apex, focus)), color="red", lw=2.5)
        ax.scatter(*apex,   color="blue", s=70,  edgecolor="k", zorder=5)
        ax.scatter(*scalp,  color="cyan", s=60,  edgecolor="k", marker="s", zorder=5)
        ax.scatter(*target, color="lime", s=120, edgecolor="k", zorder=5)
        ax.scatter(*focus,  color="red",  s=120, marker="*", edgecolor="k", zorder=5)
        ax.view_init(**vk)
        ax.set(xlabel="L (mm)", ylabel="P (mm)", zlabel="S (mm)", title=title)
        ax.set_xlim(0, bone_v[:, 0].max())
        ax.set_ylim(0, bone_v[:, 1].max())
        ax.set_zlim(0, bone_v[:, 2].max())
        ax.set_box_aspect([float(np.ptp(bone_v[:, i])) for i in range(3)])

    fig.suptitle(f"{title_prefix}: focal intensity — orthogonal slices (top) "
                 f"and 3D iso-surfaces in LPS (bottom; crimson -3 dB, "
                 f"orange -6 dB, yellow -9 dB) - {plc['scalp_site']} -> "
                 f"{plc['target_name']}", fontsize=11)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# =====================================================================
# Pressure snapshots: early / focal / post-focal
# =====================================================================


def _pressure_snapshots(run: dict[str, Any], out_path: Path,
                        title_prefix: str = "Transcranial") -> dict[str, Any]:
    """Render three pressure snapshots — early, focal, post-focal — as
    XY (mid-z) and XZ (mid-y) slices through the propagation map.

    Focal frame is selected as the timestep of global peak |p|; flanking
    frames are placed at ±18 % of the recorded length around it. Skull
    contour is overlaid using the saved c_ds map (c > 1600 m/s).
    """
    pmap = np.asarray(run["pmap"]).astype(np.float32)
    nt_rec, nx_ds, ny_ds, nz_ds = pmap.shape
    dx_ds = run["dx"]
    dy_ds = run["dy"]
    dz_ds = run["dz"]
    dt_save = run.get("dt_save")
    if dt_save is None:
        # Fall back to one wavelength period guess if dt_save is missing
        dt_save = 1.0 / run["f0"] / 2.0

    peak_per_t = np.abs(pmap).reshape(nt_rec, -1).max(axis=1)
    t_focal = int(np.argmax(peak_per_t))
    flank = max(1, int(round(0.18 * nt_rec)))
    t_early = max(0, t_focal - flank)
    t_post = min(nt_rec - 1, t_focal + flank)
    times = [("early", t_early), ("focal", t_focal), ("post-focal", t_post)]

    iy_mid = ny_ds // 2
    iz_mid = nz_ds // 2

    def _comp(x):  # sign-preserving cube-root compression for visibility
        return np.sign(x) * np.abs(x) ** (1.0 / 3.0)

    skull = np.asarray(run["c"]) > 1600.0
    skull_xy = skull[:, :, iz_mid]
    skull_xz = skull[:, iy_mid, :]

    ext_xy = [0, ny_ds * dy_ds * 1e3, nx_ds * dx_ds * 1e3, 0]
    ext_xz = [0, nz_ds * dz_ds * 1e3, nx_ds * dx_ds * 1e3, 0]

    vmax = max(_comp(np.abs(pmap[t])).max() for _, t in times)
    if vmax == 0:
        vmax = 1.0

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for col, (label, t) in enumerate(times):
        ax = axes[0, col]
        im = ax.imshow(_comp(pmap[t, :, :, iz_mid]), cmap="RdBu_r",
                       aspect="equal", vmin=-vmax, vmax=vmax, extent=ext_xy)
        ax.contour(np.linspace(ext_xy[0], ext_xy[1], ny_ds),
                   np.linspace(ext_xy[3], ext_xy[2], nx_ds),
                   skull_xy, levels=[0.5], colors="k", linewidths=0.7)
        ax.set(title=f"{label}: t = {t * dt_save * 1e6:.1f} µs (frame {t})\n"
                     f"peak |p| = {np.abs(pmap[t]).max() * 1e-6:.2f} MPa",
               xlabel="lateral y [mm]", ylabel="depth x [mm]")

        ax = axes[1, col]
        ax.imshow(_comp(pmap[t, :, iy_mid, :]), cmap="RdBu_r",
                  aspect="equal", vmin=-vmax, vmax=vmax, extent=ext_xz)
        ax.contour(np.linspace(ext_xz[0], ext_xz[1], nz_ds),
                   np.linspace(ext_xz[3], ext_xz[2], nx_ds),
                   skull_xz, levels=[0.5], colors="k", linewidths=0.7)
        ax.set(xlabel="elevational z [mm]", ylabel="depth x [mm]")

    fig.colorbar(im, ax=list(axes.flat), shrink=0.7, pad=0.02,
                 label=r"sgn(p) $|p|^{1/3}$ [Pa$^{1/3}$]")
    fig.suptitle(f"{title_prefix}: pressure snapshots — XY (top) / XZ (bottom)",
                 fontsize=14, fontweight="bold")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "t_early_us": float(t_early * dt_save * 1e6),
        "t_focal_us": float(t_focal * dt_save * 1e6),
        "t_post_us": float(t_post * dt_save * 1e6),
        "frame_early": int(t_early),
        "frame_focal": int(t_focal),
        "frame_post": int(t_post),
        "peak_mpa_early": float(np.abs(pmap[t_early]).max() * 1e-6),
        "peak_mpa_focal": float(np.abs(pmap[t_focal]).max() * 1e-6),
        "peak_mpa_post": float(np.abs(pmap[t_post]).max() * 1e-6),
        "global_peak_mpa": float(peak_per_t.max() * 1e-6),
    }


# =====================================================================
# Builder entry-point
# =====================================================================


def build(cfg, fig_dir: Path) -> dict[str, Any]:
    """Compute acoustic-output metrics and render the validation figures.

    Reads from ``cfg.OUT_DIR`` parents, expecting the simulation run
    directory layout described above.  Optionally compares against a
    homogeneous companion run if ``cfg.ACOUSTIC_OUTPUTS_HOMOGENEOUS_DIR``
    is defined and exists.
    """
    repo_root = Path(__file__).resolve().parents[2]
    sim_dir = (repo_root / cfg.OUT_DIR).parent
    if not sim_dir.is_absolute():
        sim_dir = (repo_root / sim_dir).resolve()

    # Pulse duration: prefer config-derived sequence info if available,
    # otherwise fall back to n_tic * dt.
    seq = getattr(cfg, "NEUROMOD_SEQUENCE", None)
    if seq and "tone_burst_duration_s" in seq:
        pulse_s = float(seq["tone_burst_duration_s"])
        duty_session = float(seq.get("duty_cycle_in_burst", 1.0))
    else:
        pulse_s = 7.68e-6
        duty_session = 1.0

    transcranial = _load_run(sim_dir)
    metrics_t = _compute_metrics(transcranial, duty_session, pulse_s)

    homo_metrics = None
    homo_dir_attr = getattr(cfg, "ACOUSTIC_OUTPUTS_HOMOGENEOUS_DIR", None)
    if homo_dir_attr is not None:
        homo_dir = Path(homo_dir_attr)
        if not homo_dir.is_absolute():
            homo_dir = (repo_root / homo_dir).resolve()
        if homo_dir.exists():
            homo_run = _load_run(homo_dir)
            homo_metrics = _compute_metrics(homo_run, duty_session, pulse_s)

    # Figures
    fig_intensity = fig_dir / "acoustic_intensity_combined.png"
    fig_pulse_norm = fig_dir / "acoustic_pulse_norm.png"
    fig_snapshots = fig_dir / "acoustic_pressure_snapshots.png"
    _intensity_combined(transcranial, fig_intensity, cfg=cfg)
    _pulse_norm_plot(transcranial, fig_pulse_norm)
    snapshot_info = _pressure_snapshots(transcranial, fig_snapshots)

    return {
        "acoustic_outputs": {
            "transcranial": metrics_t,
            "homogeneous": homo_metrics,
            "duty_cycle_session": duty_session,
            "pulse_duration_s": pulse_s,
            "nt_saved": int(transcranial["pmap"].shape[0]),
            "n_tic": int(transcranial["n_tic"]),
            "sampling_modulus_time": int(transcranial["sampling_modulus_time"]),
            "n_source_steps": int(transcranial["n_source_steps"]),
            "silence_correction": float(transcranial["silence_correction"]),
            "cw_correction": float(transcranial["cw_correction"]),
            "pulse_norm": float(transcranial["pulse_norm"]),
            "snapshots": snapshot_info,
        },
        "figures_acoustic_outputs": {
            "intensity_combined": fig_intensity.name,
            "pulse_norm": fig_pulse_norm.name,
            "pressure_snapshots": fig_snapshots.name,
        },
    }
