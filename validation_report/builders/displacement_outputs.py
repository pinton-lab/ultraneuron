"""Section — Shear-wave displacement (transient).

Time-resolved counterpart to elastic_outputs.  Where elastic_outputs
reports the static equilibrium driven by a constant ARF, this section
shows the transient creep-and-relax response over one Yaakub PRF
period (single tone-burst ON window followed by relaxation).

Reads from ``data/<sim_name>/``:
  * ``snapshots_xy.npy`` / ``snapshots_xz.npy`` — (n_frames, nx, ny|nz)
    arrays of |u_x| recorded on the orthogonal mid-planes through the
    peak ARF voxel by the shear-FDTD solver.
  * ``disp_trace.npy`` — (n_steps,) trace of |u_x| at the peak ARF
    voxel sampled at every shear time-step.
  * ``grid_params.npz`` — provides shear-grid spacing and the Yaakub
    envelope parameters (t_pulse_on, t_period, dt_shear).
  * ``c_ds.npy`` — for a skull contour overlay on the snapshot panels.

Metrics: peak |u_x|, time-to-peak, on-window-end displacement, fitted
visco-elastic relaxation time constant tau (single exponential, fit
on the OFF window).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np


# =====================================================================
# Data loaders
# =====================================================================


def _load_shear(run_dir: Path) -> dict[str, Any]:
    """Load shear-FDTD outputs and grid parameters."""
    grid = np.load(run_dir / "grid_params.npz", allow_pickle=True)
    snaps_xy = np.load(run_dir / "snapshots_xy.npy")  # (n_frames, nx, ny)
    snaps_xz = np.load(run_dir / "snapshots_xz.npy")  # (n_frames, nx, nz)
    trace = np.load(run_dir / "disp_trace.npy")       # (n_steps,)
    c_map = np.load(run_dir / "c_ds.npy")             # (nx, ny, nz)

    # Shear FDTD spacing matches the downsampled validation grid.
    dx_shear = float(grid["dx"]) * int(grid["mod_x"])
    dy_shear = float(grid["dy"]) * int(grid["mod_y"])
    dz_shear = float(grid["dz"]) * int(grid["mod_z"])

    # Yaakub envelope parameters saved by the bowl script (best effort).
    keys = grid.files
    t_pulse_on = float(grid["t_pulse_on"]) if "t_pulse_on" in keys else None
    t_period = float(grid["t_period"]) if "t_period" in keys else None
    dt_shear = float(grid["dt_shear"]) if "dt_shear" in keys else None
    n_steps_shear = (int(grid["n_steps_shear"])
                     if "n_steps_shear" in keys else trace.shape[0])

    return dict(
        snaps_xy=snaps_xy,
        snaps_xz=snaps_xz,
        trace=trace,
        c_map=c_map,
        dx=dx_shear,
        dy=dy_shear,
        dz=dz_shear,
        t_pulse_on=t_pulse_on,
        t_period=t_period,
        dt_shear=dt_shear,
        n_steps_shear=n_steps_shear,
    )


# =====================================================================
# Metrics
# =====================================================================


def _fit_relax_tau(t: np.ndarray, u: np.ndarray) -> float | None:
    """Single-exponential decay fit u(t) = A exp(-(t - t0) / tau).

    ``t``, ``u`` are restricted to the OFF window (after t_pulse_on).
    Falls back to ``None`` if the data is too noisy / non-monotone.
    """
    if t.size < 4 or u.size < 4:
        return None
    u0 = float(u[0])
    if u0 <= 0:
        return None
    # Trim to the segment that's still positive — log() needs u > 0.
    floor = max(1e-3 * u0, 1e-12)
    mask = u > floor
    if mask.sum() < 4:
        return None
    t_fit = t[mask] - t[0]
    log_u = np.log(u[mask])
    # Linear regression: log(u) = log(A) - t/tau -> slope = -1/tau.
    slope, _intercept = np.polyfit(t_fit, log_u, 1)
    if slope >= 0 or not np.isfinite(slope):
        return None
    return float(-1.0 / slope)


def _compute_metrics(shear: dict[str, Any], cfg) -> dict[str, Any]:
    trace = np.asarray(shear["trace"], dtype=np.float64)
    n = trace.size
    dt = shear["dt_shear"]
    t_pulse_on = shear["t_pulse_on"]
    t_period = shear["t_period"]

    # Prefer values from the cfg's neuromodulation sequence — the bowl
    # script doesn't currently persist these into grid_params.npz, so
    # the cfg is the authoritative source.
    seq = getattr(cfg, "NEUROMOD_SEQUENCE", {}) or {}
    if t_pulse_on is None and "tone_burst_duration_s" in seq:
        t_pulse_on = float(seq["tone_burst_duration_s"])
    if t_period is None and "pulse_period_s" in seq:
        t_period = float(seq["pulse_period_s"])

    if dt is None:
        # Reconstruct from period if available; else assume uniform spacing
        # over (0, t_period) and fall back to 200 ms.
        if t_period is not None:
            dt = float(t_period) / max(n - 1, 1)
        else:
            dt = 0.2 / max(n - 1, 1)
    if t_period is None:
        t_period = dt * (n - 1)
    if t_pulse_on is None:
        t_pulse_on = 0.30 * t_period  # generic fallback

    t = np.arange(n) * dt
    abs_u = np.abs(trace)
    pk_idx = int(np.argmax(abs_u))
    pk_val = float(abs_u[pk_idx])
    t_peak = float(t[pk_idx])

    # Displacement at end of ON window
    on_end_idx = int(np.searchsorted(t, t_pulse_on))
    on_end_idx = max(0, min(on_end_idx, n - 1))
    u_on_end = float(abs_u[on_end_idx])

    # Relaxation tau on the OFF window
    off_mask = t > t_pulse_on
    if off_mask.any():
        tau = _fit_relax_tau(t[off_mask], abs_u[off_mask])
    else:
        tau = None

    # Field max from the snapshot stacks
    field_max_xy = float(np.max(np.abs(shear["snaps_xy"])))
    field_max_xz = float(np.max(np.abs(shear["snaps_xz"])))
    field_max = max(field_max_xy, field_max_xz)

    return dict(
        peak_ux_um=pk_val * 1e6,
        time_to_peak_ms=t_peak * 1e3,
        u_on_end_um=u_on_end * 1e6,
        relax_tau_ms=(tau * 1e3) if tau is not None else None,
        field_max_um=field_max * 1e6,
        t_pulse_on_ms=float(t_pulse_on) * 1e3,
        t_period_ms=float(t_period) * 1e3,
        n_steps_shear=int(n),
        dt_shear_us=float(dt) * 1e6,
    )


# =====================================================================
# Figures
# =====================================================================


def _plot_trace(shear: dict[str, Any], metrics: dict[str, Any],
                out_path: Path, title_prefix: str) -> None:
    """Trace at peak ARF voxel with envelope shade and key annotations."""
    trace = np.asarray(shear["trace"], dtype=np.float64)
    n = trace.size
    dt = metrics["dt_shear_us"] * 1e-6
    t_ms = np.arange(n) * dt * 1e3
    u_um = np.abs(trace) * 1e6

    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(t_ms, u_um, color="tab:blue", lw=1.6,
            label=r"$|u_x|$ at peak ARF voxel")

    # Shade the ON window
    ax.axvspan(0.0, metrics["t_pulse_on_ms"], color="0.85",
               alpha=0.6, lw=0)

    # Peak marker + dashed leader
    ax.axvline(metrics["time_to_peak_ms"], color="k",
               lw=0.6, ls="--", alpha=0.5)
    ax.scatter([metrics["time_to_peak_ms"]], [metrics["peak_ux_um"]],
               s=24, color="k", zorder=5)
    ax.annotate(
        rf"peak {metrics['peak_ux_um']:.2f} $\mu$m at "
        rf"{metrics['time_to_peak_ms']:.1f} ms",
        xy=(metrics["time_to_peak_ms"], metrics["peak_ux_um"]),
        xytext=(8, -2), textcoords="offset points",
        fontsize=9, va="top",
    )
    if metrics["relax_tau_ms"] is not None:
        ax.text(0.97, 0.85,
                rf"$\tau_{{relax}} = {metrics['relax_tau_ms']:.1f}$ ms",
                transform=ax.transAxes, ha="right", fontsize=10,
                bbox=dict(facecolor="white", edgecolor="0.7", pad=2))

    ax.set_xlim(0.0, metrics["t_period_ms"])
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel(r"Displacement $|u_x|$ ($\mu$m)")
    ax.set_title(f"{title_prefix}: shear-wave displacement at peak ARF voxel")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_snapshots(shear: dict[str, Any], metrics: dict[str, Any],
                    out_path: Path, title_prefix: str) -> None:
    """5-time x 2-plane (XY top, XZ bottom) snapshots of |u_x|."""
    snaps_xy = np.asarray(shear["snaps_xy"], dtype=np.float32)
    snaps_xz = np.asarray(shear["snaps_xz"], dtype=np.float32)
    n_frames = snaps_xy.shape[0]
    t_period = metrics["t_period_ms"] * 1e-3
    t_pulse_on = metrics["t_pulse_on_ms"] * 1e-3
    frame_dt = t_period / max(n_frames - 1, 1)

    # 5 representative times: 25 % / 50 % / 100 % into ON, then 2 OFF samples
    target_times = [
        0.25 * t_pulse_on,
        0.5 * t_pulse_on,
        t_pulse_on,
        t_pulse_on + 0.4 * (t_period - t_pulse_on),
        t_period - frame_dt,
    ]
    frame_idx = [int(round(tt / frame_dt)) for tt in target_times]
    frame_idx = [max(0, min(i, n_frames - 1)) for i in frame_idx]
    frame_t_ms = [i * frame_dt * 1e3 for i in frame_idx]

    nx = snaps_xy.shape[1]
    ny = snaps_xy.shape[2]
    nz = snaps_xz.shape[2]
    dx = shear["dx"] * 1e3
    dy = shear["dy"] * 1e3
    dz = shear["dz"] * 1e3

    vmax = max(np.abs(snaps_xy).max(), np.abs(snaps_xz).max()) * 1e6
    if vmax <= 0:
        vmax = 1.0

    skull = (shear["c_map"] > 1600.0)
    iy_mid = ny // 2
    iz_mid = nz // 2

    fig, axes = plt.subplots(2, len(frame_idx), figsize=(15, 6.0),
                             sharex="row", sharey="row")
    ext_xy = (0, ny * dy, nx * dx, 0)  # depth top-down
    ext_xz = (0, nz * dz, nx * dx, 0)

    for col, (fi, tms) in enumerate(zip(frame_idx, frame_t_ms)):
        # XY (top row)
        ax = axes[0, col]
        sl_xy = np.abs(snaps_xy[fi]) * 1e6
        im = ax.imshow(sl_xy, cmap="hot", vmin=0, vmax=vmax,
                       extent=ext_xy, aspect="equal")
        sm_xy = skull[:, :, iz_mid]
        ax.contour(np.linspace(ext_xy[0], ext_xy[1], sm_xy.shape[1]),
                   np.linspace(ext_xy[3], ext_xy[2], sm_xy.shape[0]),
                   sm_xy, levels=[0.5], colors="cyan", linewidths=0.5)
        ax.set_title(f"t = {tms:.1f} ms", fontsize=10)
        if col == 0:
            ax.set_ylabel("x, depth (mm)\nXY-plane")
        ax.set_xlabel("y (mm)" if col == 0 else "")

        # XZ (bottom row)
        ax = axes[1, col]
        sl_xz = np.abs(snaps_xz[fi]) * 1e6
        ax.imshow(sl_xz, cmap="hot", vmin=0, vmax=vmax,
                  extent=ext_xz, aspect="equal")
        sm_xz = skull[:, iy_mid, :]
        ax.contour(np.linspace(ext_xz[0], ext_xz[1], sm_xz.shape[1]),
                   np.linspace(ext_xz[3], ext_xz[2], sm_xz.shape[0]),
                   sm_xz, levels=[0.5], colors="cyan", linewidths=0.5)
        if col == 0:
            ax.set_ylabel("x, depth (mm)\nXZ-plane")
        ax.set_xlabel("z (mm)" if col == 0 else "")

    fig.subplots_adjust(left=0.06, right=0.92, top=0.96, bottom=0.10,
                        wspace=0.10, hspace=0.20)
    cax = fig.add_axes([0.93, 0.12, 0.012, 0.82])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(r"$|u_x|$ ($\mu$m)")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# =====================================================================
# Public API
# =====================================================================


def build(cfg, fig_dir: Path) -> dict[str, Any]:
    """Build the displacement-outputs section context."""
    repo_root = Path(__file__).resolve().parents[2]
    sim_dir = Path(cfg.SIM_DATA_DIR)
    if not sim_dir.is_absolute():
        sim_dir = (repo_root / sim_dir).resolve()

    shear = _load_shear(sim_dir)
    metrics = _compute_metrics(shear, cfg)

    title_prefix = getattr(cfg, "SIM_TITLE", cfg.SIM_NAME)
    trace_png = fig_dir / "fig_disp_trace.png"
    snaps_png = fig_dir / "fig_disp_snapshots.png"
    _plot_trace(shear, metrics, trace_png, title_prefix)
    _plot_snapshots(shear, metrics, snaps_png, title_prefix)

    return {
        "displacement_outputs": metrics,
        "figures_displacement_outputs": {
            "trace": trace_png.name,
            "snapshots": snaps_png.name,
        },
    }
