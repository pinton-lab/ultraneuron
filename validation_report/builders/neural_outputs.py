"""Section 5 - Neural outputs.

Drives a population HH-style mechanosensitive neuron model on every
brain voxel whose membrane tension exceeds threshold and renders the
spatio-temporal response in seven figures that mirror the section
grammar already established by the acoustic/thermal/elastic sections:

    fig:neural_latency       three-plane ortho, time-to-first-spike
    fig:neural_rate          three-plane ortho, peak firing rate
    fig:neural_3d            3-D iso-surfaces of firing rate
    fig:neural_snapshots     tension <-> cumulative-spike snapshot row
    fig:neural_raster        depth-sorted spike raster + ROI key
    fig:neural_psth          population PSTH + recruitment + session env.
    fig:neural_distrib       rate distribution + spike-train PSD

The temporal envelope is the within-burst tension waveform
``T(t) = T_peak * sin(pi t / T_pulse)`` with ``T_pulse = group_on_s``;
the tension field comes from the elastic-outputs derivation
``T_peak = K_a * |eps_vol|``.  Each above-threshold voxel hosts one
point neuron driven by its local time-varying tension trace; spike
times are accumulated over the configured number of theta groups and
fed to the population analyses.
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
from . import elastic_outputs as _el


# =====================================================================
# Tunables (kept conservative so the section runs in seconds, not hours)
# =====================================================================

_TENSION_THRESHOLD = 0.5          # mN/m, skip voxels below this peak T
_DT_MS = 0.025                     # neuron integrator step (HH stable, batch-friendly)
_DURATION_MS = 30.0                # per-voxel sim window (covers one burst + tail)
_N_GROUPS = 3                      # theta groups represented in raster / PSTH
_RASTER_N = 120                    # neurons drawn in raster
_RNG_SEED = 0


# =====================================================================
# Field derivations
# =====================================================================


def _peak_tension_field(sim_dir: Path, run: dict[str, Any], K_a: float,
                        coupling_gain: float) -> np.ndarray:
    """Reuse the elastic builder's tension proxy: T = K_a * |eps_vol|,
    scaled by a tissue->membrane coupling gain so the static-elastostatic
    ε_vol (which is voxel-scale, ~1e-5) is lifted into the bilayer-scale
    areal strain."""
    disp = np.load(sim_dir / "displacement.npy").astype(np.float32)
    strain = _el._strain_tensor(disp, run["dx"], run["dy"], run["dz"])
    return _el._membrane_tension(strain["e_vol"] * coupling_gain, K_a=K_a) * 1e3


def _half_sin_envelope(t_ms: np.ndarray, T_pulse_ms: float) -> np.ndarray:
    """Within-burst tension envelope normalised to peak 1."""
    env = np.zeros_like(t_ms)
    inside = (t_ms >= 0) & (t_ms <= T_pulse_ms)
    env[inside] = np.sin(np.pi * t_ms[inside] / T_pulse_ms)
    return env


def _load_npz_results(npz_path: Path) -> dict[str, Any]:
    """Load `neural_results.npz` written by ``scripts/run_neural_sim.py``
    and reshape it into the dict shape consumed by the figure renderers.

    The npz holds the 3-compartment manuscript pipeline output
    (Piezo1+TREK1 strain channels on a Pinsky-Rinzel skeleton); reusing
    it avoids re-simulating with the validation builder's single-comp
    fallback and keeps the report consistent with the manuscript.
    """
    with np.load(npz_path) as data:
        T_peak = data["T_peak_mNm"].astype(np.float32)
        firing_map = data["firing_map"].astype(bool)
        spike_count_map = data["spike_count_map"].astype(np.int32)
        latency_map = data["latency_map"].astype(np.float32)
        rate_map = data["rate_map"].astype(np.float32)
        spike_voxel_idx = data["spike_voxel_idx"].astype(np.int32)
        spike_offsets = data["spike_offsets"].astype(np.int64)
        spike_times_ms = data["spike_times_ms"].astype(np.float32)
        duration_ms = float(data["duration_ms"])
        T_pulse_ms = float(data["T_pulse_ms"])
        threshold_mNm = float(data["threshold_mNm"])
        piezo1_g = float(data["piezo1_g_density"])
        K_A_npz = float(data["K_A"]) if "K_A" in data.files else 0.25
        coupling_factor_npz = (float(data["coupling_factor"])
                               if "coupling_factor" in data.files else 200.0)

    spike_times_per_voxel: dict[tuple[int, int, int], np.ndarray] = {}
    for k in range(spike_voxel_idx.shape[0]):
        ix, iy, iz = (int(spike_voxel_idx[k, 0]),
                      int(spike_voxel_idx[k, 1]),
                      int(spike_voxel_idx[k, 2]))
        st = spike_times_ms[spike_offsets[k]:spike_offsets[k + 1]]
        if st.size:
            spike_times_per_voxel[(ix, iy, iz)] = np.asarray(st, dtype=np.float32)

    eligible_mask = T_peak >= threshold_mNm
    eligible_idx = np.argwhere(eligible_mask).astype(np.int32)
    T_peak_flat = T_peak[eligible_mask].astype(np.float32)

    return {
        "T_peak": T_peak,
        "spike_count_map": spike_count_map,
        "latency_map": latency_map,
        "rate_map": rate_map,
        "spike_times_per_voxel": spike_times_per_voxel,
        "voxel_idx_per_neuron": eligible_idx,
        "T_peak_per_neuron": T_peak_flat,
        "duration_ms": duration_ms,
        "T_pulse_ms": T_pulse_ms,
        "threshold_mNm": threshold_mNm,
        "piezo1_g_density_mS_cm2": piezo1_g,
        "K_A_npz": K_A_npz,
        "coupling_factor_npz": coupling_factor_npz,
        "n_firing_voxels": int(firing_map.sum()),
        "n_eligible_voxels": int(eligible_mask.sum()),
        "_source": "neural_results.npz (3-comp manuscript pipeline)",
    }


def _simulate_population_3comp(
    sim_dir: Path,
    run: dict[str, Any],
    cfg,
    *,
    K_A: float,
    coupling_factor: float,
    dt_ms: float,
    duration_ms: float,
    threshold_mNm: float,
) -> dict[str, Any]:
    """Manuscript-grade 3-compartment recompute path.

    Mirrors ``scripts/run_neural_sim.py``: builds a Pinsky-Rinzel
    cortical-pyramidal neuron with Piezo1 (+ optional TREK1), passes the
    deviatoric strain through ``simulate_map_3comp_with_spike_times``
    (which now uses the numba JIT batched kernel for the canonical
    feature set), and reshapes the result into the dict shape consumed
    by this module's figure renderers — identical to ``_load_npz_results``.

    Used as the fallback when ``neural_results.npz`` is absent; produces
    the same scientific output as the upstream pipeline so the two
    code paths can never diverge.
    """
    from copy import deepcopy
    from ultraneuron import (  # noqa: E402
        PIEZO1, TREK1, MulticompartmentMechanoHHNeuron,
        simulate_map_3comp_with_spike_times,
    )

    disp = np.load(sim_dir / "displacement.npy").astype(np.float32)
    strain = _el._strain_tensor(disp, run["dx"], run["dy"], run["dz"])
    e_eq = strain["e_eq"].astype(np.float32)

    # Restrict to brain — skull voxels saturate strain artefactually
    skull = run["c"] > 1600.0
    e_eq[skull] = 0.0

    g_p1 = float(getattr(cfg, "NEURAL_PIEZO1_G_DENSITY", 0.10))
    use_trek1 = bool(getattr(cfg, "NEURAL_USE_TREK1", True))
    p1 = deepcopy(PIEZO1); p1.g_density = g_p1
    channels = [p1]
    if use_trek1:
        t1 = deepcopy(TREK1)
        t1.g_density = float(getattr(cfg, "NEURAL_TREK1_G_DENSITY", 0.04))
        channels.append(t1)
    neuron = MulticompartmentMechanoHHNeuron.cortical_pyramidal_3comp(
        mechano_channels=channels, temperature=37.0,
    )

    res = simulate_map_3comp_with_spike_times(
        e_eq, neuron,
        K_A=K_A, coupling_factor=coupling_factor,
        dt=dt_ms, duration=duration_ms,
        tension_threshold=threshold_mNm,
    )

    T_peak = res["tension_3d"].astype(np.float32)
    spike_voxel_idx = res["spike_voxel_idx"].astype(np.int32)
    spike_offsets = res["spike_offsets"].astype(np.int64)
    spike_times_ms = res["spike_times_ms"].astype(np.float32)
    spike_times_per_voxel: dict[tuple[int, int, int], np.ndarray] = {}
    for k in range(spike_voxel_idx.shape[0]):
        ix, iy, iz = (int(spike_voxel_idx[k, 0]),
                      int(spike_voxel_idx[k, 1]),
                      int(spike_voxel_idx[k, 2]))
        st = spike_times_ms[spike_offsets[k]:spike_offsets[k + 1]]
        if st.size:
            spike_times_per_voxel[(ix, iy, iz)] = np.asarray(
                st, dtype=np.float32,
            )

    eligible_mask = T_peak >= threshold_mNm
    eligible_idx = np.argwhere(eligible_mask).astype(np.int32)
    T_peak_flat = T_peak[eligible_mask].astype(np.float32)
    rate_map = (res["spike_count_map"].astype(np.float32)
                / max(duration_ms * 1e-3, 1e-9))

    return {
        "T_peak": T_peak,
        "spike_count_map": res["spike_count_map"].astype(np.int32),
        "latency_map": res["latency_map"].astype(np.float32),
        "rate_map": rate_map,
        "spike_times_per_voxel": spike_times_per_voxel,
        "voxel_idx_per_neuron": eligible_idx,
        "T_peak_per_neuron": T_peak_flat,
        "duration_ms": float(duration_ms),
        "T_pulse_ms": float(duration_ms),
        "threshold_mNm": float(threshold_mNm),
        "piezo1_g_density_mS_cm2": g_p1,
        "n_firing_voxels": int(res["firing_map"].sum()),
        "n_eligible_voxels": int(eligible_mask.sum()),
        "_source": "in-builder 3-comp recompute (manuscript-equivalent)",
    }


# =====================================================================
# Figure renderers
# =====================================================================


def _ortho_neural(field: np.ndarray, run: dict[str, Any], peak_idx: tuple,
                  out_path: Path, *, cmap: str, label: str, title: str,
                  vmin: float | None = None, vmax: float | None = None,
                  reverse_cmap: bool = False) -> None:
    """Three-plane ortho with cyan skull contour, matching the section
    grammar from elastic_outputs._ortho_scalar but with NaN-aware
    rendering (so latency NaNs come out transparent)."""
    skull = run["c"] > 1600.0
    nx, ny, nz = field.shape
    dx, dy, dz = run["dx"] * 1e3, run["dy"] * 1e3, run["dz"] * 1e3
    Lx, Ly, Lz = nx * dx, ny * dy, nz * dz
    pk = peak_idx

    finite = field[np.isfinite(field)]
    if vmax is None:
        vmax = float(np.percentile(finite, 99.9)) if finite.size else 1.0
    if vmin is None:
        vmin = float(np.nanmin(field)) if finite.size else 0.0
    if vmax <= vmin:
        vmax = vmin + 1.0

    cmap_obj = plt.get_cmap(cmap + "_r" if reverse_cmap else cmap).copy()
    cmap_obj.set_bad(color=(1.0, 1.0, 1.0, 0.0))

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), constrained_layout=True)

    ax = axes[0]
    im = ax.imshow(field[:, :, pk[2]], cmap=cmap_obj, origin="upper",
                   vmin=vmin, vmax=vmax, aspect="equal",
                   extent=(0, Ly, Lx, 0))
    ax.contour(np.linspace(0, Ly, ny), np.linspace(0, Lx, nx),
               skull[:, :, pk[2]], levels=[0.5], colors="cyan", linewidths=0.6)
    ax.set_xlabel("y (mm)"); ax.set_ylabel("x, depth (mm)")
    ax.set_title(f"x-y plane (z={pk[2] * dz:.1f} mm)")

    ax = axes[1]
    ax.imshow(field[:, pk[1], :], cmap=cmap_obj, origin="upper",
              vmin=vmin, vmax=vmax, aspect="equal",
              extent=(0, Lz, Lx, 0))
    ax.contour(np.linspace(0, Lz, nz), np.linspace(0, Lx, nx),
               skull[:, pk[1], :], levels=[0.5], colors="cyan", linewidths=0.6)
    ax.set_xlabel("z (mm)"); ax.set_ylabel("x, depth (mm)")
    ax.set_title(f"x-z plane (y={pk[1] * dy:.1f} mm)")

    ax = axes[2]
    ax.imshow(field[pk[0], :, :].T, cmap=cmap_obj, origin="lower",
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


def _tension_slice_4d(sim_dir: Path, x_slice: int, K_A: float,
                      coupling_factor: float
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Time-resolved tension field on a single x slice.

    Computes deviatoric strain per saved frame from displacement_4d.npy
    on the focal y-z plane (x = ``x_slice``), then converts to membrane
    tension via T = K_A * alpha * eps_dev.  Returns (T_slice_t, t_ms)
    with shapes (n_frames, ny, nz) and (n_frames,).

    Beats the synthetic half-sine envelope used previously: that
    approximation flattens the 5-ms shear-wave build-up into a smooth
    push-off shape and underpredicts what the neuron actually sees in
    the second half of the pulse.
    """
    u4 = np.load(sim_dir / "displacement_4d.npy", mmap_mode="r")  # (T,nx,ny,nz,3)
    times = np.load(sim_dir / "displacement_4d_times.npy")
    dxg = float(np.load(sim_dir / "grid_params.npz")["dx"])
    mod_x = int(np.load(sim_dir / "grid_params.npz")["mod_x"])
    pitch = dxg * mod_x

    # We only need three voxel-stencils centred on x_slice for d/dx, but
    # compute on the full volume per frame and slice — n_frames is small
    # (~100) so this is cheap.
    n_t = u4.shape[0]
    ny, nz = u4.shape[2], u4.shape[3]
    T_slice = np.empty((n_t, ny, nz), dtype=np.float32)
    for it in range(n_t):
        u = np.asarray(u4[it], dtype=np.float32)  # (nx, ny, nz, 3)
        ux, uy, uz = u[..., 0], u[..., 1], u[..., 2]
        e_xx = np.gradient(ux, pitch, axis=0)
        e_yy = np.gradient(uy, pitch, axis=1)
        e_zz = np.gradient(uz, pitch, axis=2)
        e_xy = 0.5 * (np.gradient(ux, pitch, axis=1) + np.gradient(uy, pitch, axis=0))
        e_xz = 0.5 * (np.gradient(ux, pitch, axis=2) + np.gradient(uz, pitch, axis=0))
        e_yz = 0.5 * (np.gradient(uy, pitch, axis=2) + np.gradient(uz, pitch, axis=1))
        e_vol = (e_xx + e_yy + e_zz) / 3.0
        e_dev_xx = e_xx - e_vol
        e_dev_yy = e_yy - e_vol
        e_dev_zz = e_zz - e_vol
        e_eq = np.sqrt(2/3 * (e_dev_xx**2 + e_dev_yy**2 + e_dev_zz**2
                               + 2 * (e_xy**2 + e_xz**2 + e_yz**2)))
        T_slice[it] = (K_A * coupling_factor * e_eq[x_slice] * 1e3).astype(np.float32)

    return T_slice, times * 1e3  # convert s -> ms


def _cumulative_spikes_slice(sim: dict[str, Any], x_slice: int,
                             ny: int, nz: int, t_ms: float) -> np.ndarray:
    """Per-voxel cumulative spike count on the y-z slice at time t_ms,
    computed from exact CSR spike-time data (no linear-rate proxy).
    """
    out = np.zeros((ny, nz), dtype=np.int32)
    spv = sim["spike_times_per_voxel"]
    for (ix, iy, iz), st in spv.items():
        if ix != x_slice:
            continue
        out[iy, iz] = int((st <= t_ms).sum())
    return out


def _cumulative_spikes_three_planes(sim: dict[str, Any], peak_idx: tuple,
                                    nx: int, ny: int, nz: int,
                                    t_ms: float
                                    ) -> tuple[np.ndarray, np.ndarray,
                                                np.ndarray]:
    """Cumulative spike counts on three orthogonal planes through the
    brain-restricted peak voxel: x-y (z=peak_z), x-z (y=peak_y),
    y-z (x=peak_x).  Returns three int32 arrays.
    """
    px, py, pz = peak_idx
    out_xy = np.zeros((nx, ny), dtype=np.int32)
    out_xz = np.zeros((nx, nz), dtype=np.int32)
    out_yz = np.zeros((ny, nz), dtype=np.int32)
    spv = sim["spike_times_per_voxel"]
    for (ix, iy, iz), st in spv.items():
        n_sp = int((st <= t_ms).sum())
        if n_sp == 0:
            continue
        if iz == pz:
            out_xy[ix, iy] = n_sp
        if iy == py:
            out_xz[ix, iz] = n_sp
        if ix == px:
            out_yz[iy, iz] = n_sp
    return out_xy, out_xz, out_yz


def _snapshot_row(T_peak: np.ndarray, sim: dict[str, Any], run: dict[str, Any],
                  peak_idx: tuple, sim_dir: Path,
                  K_A: float, coupling_factor: float,
                  out_path: Path,
                  T_half_mNm: float = 2.7) -> None:
    """Headline figure: tension on three orthogonal planes (x-y, x-z,
    y-z, top three rows) plus the cumulative spike count on the y-z
    focal slice (bottom row), sampled at five timepoints across the
    within-burst window.  The Piezo1 half-activation contour
    $T = T_{1/2}$ is overlaid on every tension panel.

    Three planes give a 3-D-aware view of how tension propagates and
    how the firing zone tracks it; without all three the lateral and
    elevational extent of the focal lobe is invisible.
    """
    nx, ny, nz = T_peak.shape
    dx, dy, dz = run["dx"] * 1e3, run["dy"] * 1e3, run["dz"] * 1e3
    pk = peak_idx
    skull = run["c"] > 1600.0

    T_pulse_ms = sim["T_pulse_ms"]

    # Time-resolved tension on the full volume (one frame at a time)
    T_xy_t, T_xz_t, T_yz_t, t_ms = _tension_three_planes_4d(
        sim_dir, peak_idx, K_A, coupling_factor,
    )
    target_ms = np.linspace(T_pulse_ms / 5, T_pulse_ms, 5)
    snap_idx = [int(np.argmin(np.abs(t_ms - tt))) for tt in target_ms]
    snapshot_ms = t_ms[snap_idx]
    T_xy_snap = T_xy_t[snap_idx]   # (5, nx, ny)
    T_xz_snap = T_xz_t[snap_idx]   # (5, nx, nz)
    T_yz_snap = T_yz_t[snap_idx]   # (5, ny, nz)
    n_snap = len(snapshot_ms)

    # Cumulative spike count on three orthogonal planes (matched to the
    # tension panels) so the firing zone is legible from every viewing axis.
    cum_xy_snap = np.empty((n_snap, nx, ny), dtype=np.float32)
    cum_xz_snap = np.empty((n_snap, nx, nz), dtype=np.float32)
    cum_yz_snap = np.empty((n_snap, ny, nz), dtype=np.float32)
    for k, t in enumerate(snapshot_ms):
        cxy, cxz, cyz = _cumulative_spikes_three_planes(
            sim, peak_idx, nx, ny, nz, t,
        )
        cum_xy_snap[k] = cxy
        cum_xz_snap[k] = cxz
        cum_yz_snap[k] = cyz

    Tmax = float(max(T_xy_snap.max(), T_xz_snap.max(), T_yz_snap.max(),
                     T_half_mNm * 1.2))
    Cmax = max(float(cum_xy_snap.max()), float(cum_xz_snap.max()),
               float(cum_yz_snap.max()), 1.0)

    skull_xy = skull[:, :, pk[2]]                  # (nx, ny)
    skull_xz = skull[:, pk[1], :]                  # (nx, nz)
    skull_yz = skull[pk[0], :, :].T                # (nz, ny)

    fig, axes = plt.subplots(6, n_snap, figsize=(2.6 * n_snap, 11.6),
                             constrained_layout=True)

    ext_xy = (0, ny * dy, nx * dx, 0)              # depth-down convention
    ext_xz = (0, nz * dz, nx * dx, 0)
    ext_yz = (0, ny * dy, 0, nz * dz)

    for i, t in enumerate(snapshot_ms):
        # Row 0 — tension on x-y plane (z = peak_z)
        ax = axes[0, i]
        im_t = ax.imshow(T_xy_snap[i], cmap="magma",
                         vmin=0, vmax=Tmax, aspect="equal", extent=ext_xy)
        if T_xy_snap[i].max() >= T_half_mNm:
            ax.contour(np.linspace(0, ny * dy, ny),
                       np.linspace(0, nx * dx, nx),
                       T_xy_snap[i], levels=[T_half_mNm], colors="red", linewidths=1.0)
        if skull_xy.any():
            ax.contour(np.linspace(0, ny * dy, ny),
                       np.linspace(0, nx * dx, nx),
                       skull_xy.astype(float),
                       levels=[0.5], colors="cyan", linewidths=0.5)
        ax.set_title(f"t = {t:.1f} ms", fontsize=9)
        if i == 0:
            ax.set_ylabel("x, depth (mm)\nx-y (z=peak)")

        # Row 1 — tension on x-z plane (y = peak_y)
        ax = axes[1, i]
        ax.imshow(T_xz_snap[i], cmap="magma",
                  vmin=0, vmax=Tmax, aspect="equal", extent=ext_xz)
        if T_xz_snap[i].max() >= T_half_mNm:
            ax.contour(np.linspace(0, nz * dz, nz),
                       np.linspace(0, nx * dx, nx),
                       T_xz_snap[i], levels=[T_half_mNm], colors="red", linewidths=1.0)
        if skull_xz.any():
            ax.contour(np.linspace(0, nz * dz, nz),
                       np.linspace(0, nx * dx, nx),
                       skull_xz.astype(float),
                       levels=[0.5], colors="cyan", linewidths=0.5)
        if i == 0:
            ax.set_ylabel("x, depth (mm)\nx-z (y=peak)")

        # Row 2 — tension on y-z plane (x = peak_x)
        ax = axes[2, i]
        ax.imshow(T_yz_snap[i], cmap="magma", origin="lower",
                  vmin=0, vmax=Tmax, aspect="equal", extent=ext_yz)
        if T_yz_snap[i].max() >= T_half_mNm:
            ax.contour(np.linspace(0, ny * dy, ny),
                       np.linspace(0, nz * dz, nz),
                       T_yz_snap[i], levels=[T_half_mNm], colors="red", linewidths=1.0)
        if skull_yz.any():
            ax.contour(np.linspace(0, ny * dy, ny),
                       np.linspace(0, nz * dz, nz),
                       skull_yz.astype(float),
                       levels=[0.5], colors="cyan", linewidths=0.5)
        if i == 0:
            ax.set_ylabel("z, elev. (mm)\ny-z (x=peak)")

        # Row 3 — cumulative spike count on x-y plane (z=peak_z)
        ax = axes[3, i]
        im_c = ax.imshow(cum_xy_snap[i], cmap="viridis",
                         vmin=0, vmax=Cmax, aspect="equal", extent=ext_xy)
        if skull_xy.any():
            ax.contour(np.linspace(0, ny * dy, ny),
                       np.linspace(0, nx * dx, nx),
                       skull_xy.astype(float),
                       levels=[0.5], colors="white", linewidths=0.4, alpha=0.5)
        if i == 0:
            ax.set_ylabel("x, depth (mm)\nx-y cum. spikes")

        # Row 4 — cumulative spike count on x-z plane (y=peak_y)
        ax = axes[4, i]
        ax.imshow(cum_xz_snap[i], cmap="viridis",
                  vmin=0, vmax=Cmax, aspect="equal", extent=ext_xz)
        if skull_xz.any():
            ax.contour(np.linspace(0, nz * dz, nz),
                       np.linspace(0, nx * dx, nx),
                       skull_xz.astype(float),
                       levels=[0.5], colors="white", linewidths=0.4, alpha=0.5)
        if i == 0:
            ax.set_ylabel("x, depth (mm)\nx-z cum. spikes")

        # Row 5 — cumulative spike count on y-z plane (x=peak_x)
        ax = axes[5, i]
        ax.imshow(cum_yz_snap[i].T, cmap="viridis", origin="lower",
                  vmin=0, vmax=Cmax, aspect="equal", extent=ext_yz)
        if skull_yz.any():
            ax.contour(np.linspace(0, ny * dy, ny),
                       np.linspace(0, nz * dz, nz),
                       skull_yz.astype(float),
                       levels=[0.5], colors="white", linewidths=0.4, alpha=0.5)
        ax.set_xlabel("y, lat. (mm)")
        if i == 0:
            ax.set_ylabel("z, elev. (mm)\ny-z cum. spikes")

    cbar_t = fig.colorbar(im_t, ax=axes[:3, :].ravel().tolist(), location="right",
                          shrink=0.7, pad=0.02)
    cbar_t.set_label(f"Tension $T(t)$ (mN/m); red contour: $T_{{1/2}} = {T_half_mNm}$")
    cbar_c = fig.colorbar(im_c, ax=axes[3:, :].ravel().tolist(), location="right",
                          shrink=0.7, pad=0.02)
    cbar_c.set_label("Cumulative spikes / voxel")

    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _tension_three_planes_4d(sim_dir: Path, peak_idx: tuple,
                              K_A: float, coupling_factor: float
                              ) -> tuple[np.ndarray, np.ndarray,
                                          np.ndarray, np.ndarray]:
    """Time-resolved tension on three orthogonal planes through the
    brain-restricted peak voxel.

    Returns ``(T_xy_t, T_xz_t, T_yz_t, t_ms)`` with shapes
    ``(n_frames, nx, ny)``, ``(n_frames, nx, nz)``, ``(n_frames, ny, nz)``
    and ``(n_frames,)``.  Compute the full ε_eq volume per frame and slice
    three ways — n_frames is small (~100) so the per-frame gradient cost
    is what dominates, not the per-plane slicing.
    """
    u4 = np.load(sim_dir / "displacement_4d.npy", mmap_mode="r")
    times = np.load(sim_dir / "displacement_4d_times.npy")
    grid = np.load(sim_dir / "grid_params.npz")
    pitch_x = float(grid["dx"]) * int(grid["mod_x"])
    pitch_y = float(grid["dy"]) * int(grid["mod_y"])
    pitch_z = float(grid["dz"]) * int(grid["mod_z"])

    n_t, nx, ny, nz, _ = u4.shape
    px, py, pz = peak_idx
    T_xy = np.empty((n_t, nx, ny), dtype=np.float32)
    T_xz = np.empty((n_t, nx, nz), dtype=np.float32)
    T_yz = np.empty((n_t, ny, nz), dtype=np.float32)
    for it in range(n_t):
        u = np.asarray(u4[it], dtype=np.float32)
        ux, uy, uz = u[..., 0], u[..., 1], u[..., 2]
        e_xx = np.gradient(ux, pitch_x, axis=0)
        e_yy = np.gradient(uy, pitch_y, axis=1)
        e_zz = np.gradient(uz, pitch_z, axis=2)
        e_xy = 0.5 * (np.gradient(ux, pitch_y, axis=1)
                      + np.gradient(uy, pitch_x, axis=0))
        e_xz = 0.5 * (np.gradient(ux, pitch_z, axis=2)
                      + np.gradient(uz, pitch_x, axis=0))
        e_yz = 0.5 * (np.gradient(uy, pitch_z, axis=2)
                      + np.gradient(uz, pitch_y, axis=1))
        e_vol = (e_xx + e_yy + e_zz) / 3.0
        e_dev_xx = e_xx - e_vol
        e_dev_yy = e_yy - e_vol
        e_dev_zz = e_zz - e_vol
        e_eq = np.sqrt(2/3 * (e_dev_xx**2 + e_dev_yy**2 + e_dev_zz**2
                              + 2 * (e_xy**2 + e_xz**2 + e_yz**2)))
        T = (K_A * coupling_factor * e_eq * 1e3).astype(np.float32)
        T_xy[it] = T[:, :, pz]
        T_xz[it] = T[:, py, :]
        T_yz[it] = T[px, :, :]
    return T_xy, T_xz, T_yz, times * 1e3


def _gating_curves(sim: dict[str, Any], use_trek1: bool, T_pulse_ms: float,
                   T_max_observed_mNm: float, T_threshold_mNm: float,
                   out_path: Path) -> None:
    """Overlay the Piezo1 (and optional TREK-1) Boltzmann gating curves
    used in this sim with literature anchors and the per-sim working
    point.  A per-sim sanity check that the gating params haven't drifted
    from the published values, and that the focal-voxel tension sits on
    the part of the curve where it should drive firing.
    """
    from ultraneuron.channel_library import (  # noqa: E402
        PIEZO1 as _Piezo1, TREK1 as _TREK1,
    )

    T_grid = np.linspace(0.0, max(T_max_observed_mNm * 1.6, 8.0), 600)

    # Piezo1 activation, inactivation, and effective drive (act × inact)
    p_act = _Piezo1.open_probability(T_grid)
    p_inact = _Piezo1.inactivation_steady_state(T_grid)
    p_drive = np.asarray(p_act) * np.asarray(p_inact)

    fig, ax = plt.subplots(figsize=(8.0, 4.6), constrained_layout=True)

    # Lewis & Grandl 2015 / Coste 2010 literature confidence band on
    # the Piezo1 activation T_half (2.4 - 3.0 mN/m at ~25°C).
    ax.axvspan(2.4, 3.0, color="dimgray", alpha=0.12,
               label="Piezo1 $T_{1/2}$ literature band\n(Coste 2010 / Lewis 2015: 2.4-3.0 mN/m)")

    ax.plot(T_grid, p_act, color="#1f77b4", lw=1.8,
            label=f"Piezo1 activation $P_o(T)$  "
                  f"($T_{{1/2}}={_Piezo1.T_half}$, $\\Delta A={_Piezo1.delta_A}$ nm²)")
    if _Piezo1.tau_inact is not None:
        ax.plot(T_grid, p_inact, color="#1f77b4", lw=1.0, ls=":",
                label=f"Piezo1 inactivation $i_\\infty(T)$  "
                      f"($T_{{1/2,i}}={_Piezo1.inact_T_half}$)")
        ax.plot(T_grid, p_drive, color="#1f77b4", lw=1.4, ls="--",
                label="Piezo1 effective drive $P_o\\!\\cdot\\!i_\\infty$")

    if use_trek1:
        p_trek = _TREK1.open_probability(T_grid)
        ax.plot(T_grid, p_trek, color="#ff7f0e", lw=1.8,
                label=f"TREK-1 activation $P_o(T)$  "
                      f"($T_{{1/2}}={_TREK1.T_half}$, $\\Delta A={_TREK1.delta_A}$ nm²)")

    # Threshold line (eligibility cutoff in run_neural_sim)
    ax.axvline(T_threshold_mNm, color="black", lw=0.6, ls="--", alpha=0.5)
    ax.text(T_threshold_mNm, 1.04, f"sim eligibility cutoff\n({T_threshold_mNm} mN/m)",
            ha="center", va="bottom", fontsize=7, color="black")

    # Per-sim working point
    if T_max_observed_mNm > 0:
        ax.axvline(T_max_observed_mNm, color="crimson", lw=1.0)
        ax.text(T_max_observed_mNm, 1.04,
                f"this run\n$T_{{\\max}}={T_max_observed_mNm:.2f}$ mN/m",
                ha="center", va="bottom", fontsize=8, color="crimson")
        # Read off Po(T_max) and effective drive at the working point
        po_at = float(np.interp(T_max_observed_mNm, T_grid, p_act))
        drv_at = float(np.interp(T_max_observed_mNm, T_grid, p_drive))
        txt = (f"At $T_{{\\max}}$:  $P_o$ = {po_at:.2f}\n"
               f"effective drive = {drv_at:.2f}")
        ax.text(0.98, 0.02, txt, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8, bbox=dict(facecolor="white", edgecolor="0.6",
                                      lw=0.4, alpha=0.85, pad=2))

    ax.set_xlabel("Membrane tension $T$ (mN/m)")
    ax.set_ylabel("Open probability $P_o$ (steady state)")
    ax.set_xlim(0, T_grid[-1])
    ax.set_ylim(-0.02, 1.12)
    ax.grid(alpha=0.3)
    ax.legend(loc="center left", fontsize=7, frameon=False,
              bbox_to_anchor=(0.0, 0.55))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _representative_neurons(sim: dict[str, Any], run: dict[str, Any],
                            sim_dir: Path, K_A: float, coupling_factor: float,
                            T_pulse_ms: float, T_half_mNm: float,
                            peak_idx: tuple, out_path: Path) -> None:
    """Six-panel "representative cells" figure.

    Selection (from the firing subset):
      0) focal-centre   — highest T_peak in brain (peak_idx)
      1) median firer   — voxel whose spike-count is the firing-set median
      2) threshold      — lowest T_peak among firing voxels
      3) earliest firer — smallest first-spike latency
      4) latest firer   — largest first-spike latency

    Top-left panel locates the five voxels on the focal y-z slice.  The
    five trace panels each show membrane tension T(t) computed from the
    saved displacement_4d field, with the T_half threshold line and
    spike-time vlines overlaid; annotation box reports depth, distance
    from focus, T_peak, n_spikes, and first-spike latency.
    """
    spv = sim["spike_times_per_voxel"]
    T_peak = sim["T_peak"]
    spike_count_map = sim["spike_count_map"]
    latency_map = sim["latency_map"]

    if not spv:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.text(0.5, 0.5, "no firing voxels", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        return

    fire_keys = list(spv.keys())
    fire_idx = np.array(fire_keys, dtype=np.int32)
    fire_T_peak = T_peak[fire_idx[:, 0], fire_idx[:, 1], fire_idx[:, 2]]
    fire_counts = spike_count_map[fire_idx[:, 0], fire_idx[:, 1], fire_idx[:, 2]]
    fire_latency = latency_map[fire_idx[:, 0], fire_idx[:, 1], fire_idx[:, 2]]

    # ---- Selection ----
    sel: list[tuple[str, tuple[int, int, int]]] = []
    sel.append(("focal centre", tuple(int(i) for i in peak_idx)))

    median_count = int(np.median(fire_counts))
    cand = np.where(fire_counts == median_count)[0]
    sel.append(("median firer", tuple(int(i) for i in fire_idx[cand[0]])))

    lo_idx = int(np.argmin(fire_T_peak))
    sel.append(("threshold grazer", tuple(int(i) for i in fire_idx[lo_idx])))

    early_idx = int(np.argmin(fire_latency))
    sel.append(("earliest firer", tuple(int(i) for i in fire_idx[early_idx])))

    late_idx = int(np.argmax(fire_latency))
    sel.append(("latest firer", tuple(int(i) for i in fire_idx[late_idx])))

    # Drop duplicates (keep first occurrence; preserve order)
    seen = set()
    sel = [(n, v) for (n, v) in sel
           if not (v in seen or seen.add(v))]
    n_neur = len(sel)

    # ---- Compute tension traces from displacement_4d ----
    u4 = np.load(sim_dir / "displacement_4d.npy", mmap_mode="r")
    times = np.load(sim_dir / "displacement_4d_times.npy")
    grid = np.load(sim_dir / "grid_params.npz")
    pitch = float(grid["dx"]) * int(grid["mod_x"])

    n_t = u4.shape[0]
    traces_T = []  # tension trace per selected voxel (mN/m), len n_t
    for _, (ix, iy, iz) in sel:
        # central differences at the voxel; pad at edges
        Tt = np.empty(n_t, dtype=np.float32)
        for it in range(n_t):
            u = np.asarray(u4[it], dtype=np.float32)  # (nx,ny,nz,3)
            ux = u[..., 0]; uy = u[..., 1]; uz = u[..., 2]
            def _d(a, axis, idx):
                a_p = np.take(a, min(idx + 1, a.shape[axis] - 1), axis=axis)
                a_m = np.take(a, max(idx - 1, 0), axis=axis)
                # 2-point centred difference; clamp at boundary -> half-step
                step = pitch * (2 if 0 < idx < a.shape[axis] - 1 else 1)
                return (a_p - a_m) / step
            e_xx = _d(ux, 0, ix)[iy, iz]
            e_yy = _d(uy, 1, iy)[ix, iz]
            e_zz = _d(uz, 2, iz)[ix, iy]
            e_xy = 0.5 * (_d(ux, 1, iy)[ix, iz] + _d(uy, 0, ix)[iy, iz])
            e_xz = 0.5 * (_d(ux, 2, iz)[ix, iy] + _d(uz, 0, ix)[iy, iz])
            e_yz = 0.5 * (_d(uy, 2, iz)[ix, iy] + _d(uz, 1, iy)[ix, iz])
            e_vol = (e_xx + e_yy + e_zz) / 3.0
            e_dev_xx = e_xx - e_vol
            e_dev_yy = e_yy - e_vol
            e_dev_zz = e_zz - e_vol
            e_eq = float(np.sqrt(2/3 * (e_dev_xx**2 + e_dev_yy**2 + e_dev_zz**2
                                          + 2 * (e_xy**2 + e_xz**2 + e_yz**2))))
            Tt[it] = K_A * coupling_factor * e_eq * 1e3
        traces_T.append(Tt)
    t_ms = times * 1e3

    # ---- Layout ----
    n_cols = max(3, (n_neur + 2) // 2)        # adapt grid; default 3 cols
    n_rows = 2
    fig = plt.figure(figsize=(4.0 * n_cols, 2.6 * n_rows),
                     constrained_layout=True)
    gs = fig.add_gridspec(n_rows, n_cols)

    # Anatomy thumbnail (top-left): focal y-z slice at peak_idx[0]
    dx, dy, dz = run["dx"] * 1e3, run["dy"] * 1e3, run["dz"] * 1e3
    ax_anat = fig.add_subplot(gs[0, 0])
    T_slice = T_peak[peak_idx[0], :, :].T
    ny, nz = T_slice.shape[1], T_slice.shape[0]
    ax_anat.imshow(T_slice, cmap="magma", origin="lower",
                   extent=(0, T_slice.shape[1] * dy, 0, T_slice.shape[0] * dz),
                   aspect="equal")
    palette = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e"]
    focus_yz = (peak_idx[1] * dy, peak_idx[2] * dz)
    for k, ((label, (ix, iy, iz)), col) in enumerate(zip(sel, palette)):
        ax_anat.plot(iy * dy, iz * dz, marker="o", mec="white",
                     mfc=col, ms=9, lw=0)
        ax_anat.annotate(str(k + 1), (iy * dy, iz * dz),
                         textcoords="offset points", xytext=(7, 4),
                         fontsize=8, color="white",
                         bbox=dict(boxstyle="circle,pad=0.1",
                                   fc=col, ec="white", lw=0.5))
    ax_anat.set(xlabel="y, lat. (mm)", ylabel="z, elev. (mm)",
                title=f"Selected voxels on focal y-z slice (x={peak_idx[0]*dx:.1f} mm)")

    # 5 trace panels (or fewer if duplicates collapsed)
    panel_slots = [(r, c) for r in range(n_rows) for c in range(n_cols)
                   if not (r == 0 and c == 0)][:n_neur]
    for k, (slot, (label, (ix, iy, iz)), col) in enumerate(
            zip(panel_slots, sel, palette)):
        r, c = slot
        ax = fig.add_subplot(gs[r, c])
        Tt = traces_T[k]
        ax.fill_between(t_ms, 0, Tt, color=col, alpha=0.20, lw=0)
        ax.plot(t_ms, Tt, color=col, lw=1.2)
        ax.axhline(T_half_mNm, color="black", lw=0.6, ls="--", alpha=0.5)
        # Spike-time vlines
        st = spv.get((ix, iy, iz), np.zeros(0))
        T_max_local = max(float(Tt.max()), T_half_mNm * 1.2)
        for t_sp in st:
            ax.vlines(t_sp, 0, T_max_local, color="black", lw=0.4, alpha=0.7)
        ax.set_xlim(0, max(T_pulse_ms + 5.0, t_ms[-1]))
        ax.set_ylim(0, T_max_local * 1.05)
        ax.set_xlabel("Time (ms)" if r == n_rows - 1 else "")
        ax.set_ylabel("T(t) (mN/m)" if c == 0 else "")
        ax.grid(alpha=0.3)
        # Annotation
        focus_lps_y = peak_idx[1] * dy
        focus_lps_z = peak_idx[2] * dz
        d_focus_mm = np.sqrt((iy * dy - focus_lps_y) ** 2
                             + (iz * dz - focus_lps_z) ** 2)
        depth_mm = ix * dx
        T_pk_local = float(Tt.max())
        n_sp = len(st)
        first_sp = float(st.min()) if n_sp else float("nan")
        ax.set_title(
            f"({k + 1}) {label}\n"
            f"depth={depth_mm:.1f} mm, $r_\\perp$={d_focus_mm:.1f} mm, "
            f"$T_{{\\max}}$={T_pk_local:.2f} mN/m, "
            f"$n_{{\\mathrm{{sp}}}}$={n_sp}, "
            f"$t_1$={first_sp:.1f} ms",
            fontsize=8,
        )

    fig.suptitle("Representative cells — tension and spikes "
                 f"($T_{{1/2}} = {T_half_mNm:.1f}$ mN/m, dashed)",
                 fontsize=10)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _firing_volume_vs_time(sim: dict[str, Any], run: dict[str, Any],
                           out_path: Path, T_pulse_ms: float
                           ) -> dict[str, float]:
    """Two-panel time series of firing volume:
    (a) cumulative recruitment volume V(t) — total brain volume that
        has fired at least one spike by time t (mm^3, monotone).
    (b) instantaneous active volume — volume of voxels spiking in a
        rolling 1 ms window, the volumetric analogue of a population
        PSTH (mm^3 active at time t).
    Returns the headline volumetric metrics for the manifest.
    """
    spv = sim["spike_times_per_voxel"]
    dx, dy, dz = run["dx"] * 1e3, run["dy"] * 1e3, run["dz"] * 1e3
    voxel_vol_mm3 = dx * dy * dz

    t_max_ms = max(T_pulse_ms + 5.0, sim["duration_ms"])
    t_grid = np.linspace(0.0, t_max_ms, 401)        # 0.075 ms cadence
    win_ms = 1.0                                     # rolling window for (b)

    # First-spike time per voxel; cumulative count = #voxels with first
    # spike <= t.
    if spv:
        first_spike = np.array([st.min() for st in spv.values()
                                if len(st) > 0], dtype=np.float32)
        # Flatten all spike-times into a single array for the rolling-window
        all_st = np.concatenate(list(spv.values()))
    else:
        first_spike = np.zeros(0, dtype=np.float32)
        all_st = np.zeros(0, dtype=np.float32)

    if first_spike.size:
        first_spike_sorted = np.sort(first_spike)
        cum_count = np.searchsorted(first_spike_sorted, t_grid, side="right")
    else:
        cum_count = np.zeros_like(t_grid, dtype=np.int64)
    cum_vol_mm3 = cum_count * voxel_vol_mm3

    # Instantaneous: number of distinct voxels with at least one spike
    # in [t-win/2, t+win/2]. Per-voxel-per-window counting is overkill;
    # population spike density is a defensible proxy when each voxel
    # rarely fires more than once per window (true here at PRF=5 Hz).
    if all_st.size:
        edges = np.concatenate([t_grid - win_ms / 2, [t_grid[-1] + win_ms / 2]])
        spk_per_bin, _ = np.histogram(all_st, bins=edges)
        # Normalise to a "voxel-equivalent" count by dividing by the
        # mean spikes-per-firing-voxel-per-window (1 by design).
        inst_vol_mm3 = spk_per_bin * voxel_vol_mm3
    else:
        inst_vol_mm3 = np.zeros_like(t_grid)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4),
                             constrained_layout=True)

    ax = axes[0]
    ax.plot(t_grid, cum_vol_mm3, color="crimson", lw=1.4)
    ax.set_xlabel("Time within burst (ms)")
    ax.set_ylabel(r"Cumulative firing volume $V(t)$ (mm$^3$)")
    ax.set_title("(a) Volume of recruited neurons")
    ax.axvspan(0, T_pulse_ms, alpha=0.08, color="navy",
               label=f"pulse ON ({T_pulse_ms:.0f} ms)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    cum_final = float(cum_vol_mm3[-1])
    cum_at_pulse_end = float(np.interp(T_pulse_ms, t_grid, cum_vol_mm3))
    ax.text(0.02, 0.95,
            f"V(T_pulse) = {cum_at_pulse_end:.1f} mm$^3$\n"
            f"V(t_end)   = {cum_final:.1f} mm$^3$",
            transform=ax.transAxes, va="top", fontsize=8,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8))

    ax = axes[1]
    ax.plot(t_grid, inst_vol_mm3, color="navy", lw=1.0)
    ax.set_xlabel("Time within burst (ms)")
    ax.set_ylabel(r"Spike-active volume per " f"{win_ms:.0f}" r"-ms window (mm$^3$)")
    ax.set_title("(b) Instantaneous active volume")
    ax.axvspan(0, T_pulse_ms, alpha=0.08, color="navy")
    ax.grid(alpha=0.3)
    if inst_vol_mm3.size and inst_vol_mm3.max() > 0:
        peak_t = float(t_grid[int(np.argmax(inst_vol_mm3))])
        peak_v = float(inst_vol_mm3.max())
        ax.axvline(peak_t, color="crimson", lw=0.8, ls="--", alpha=0.6)
        ax.text(0.98, 0.95,
                f"peak = {peak_v:.1f} mm$^3$ at t = {peak_t:.1f} ms",
                transform=ax.transAxes, va="top", ha="right", fontsize=8,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8))

    fig.suptitle("Firing volume vs. time within the burst", fontsize=10)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    return {
        "firing_volume_at_pulse_end_mm3": cum_at_pulse_end
            if first_spike.size else 0.0,
        "firing_volume_final_mm3": cum_final,
        "firing_volume_peak_active_mm3": float(inst_vol_mm3.max())
            if inst_vol_mm3.size else 0.0,
        "voxel_volume_mm3": voxel_vol_mm3,
    }


def _firing_iso_3d(rate_map: np.ndarray, run: dict[str, Any],
                   out_path: Path, cfg=None) -> None:
    """Re-uses the elastic 3-D iso renderer.

    Passes ``cfg`` so the renderer uses the LPS-frame whole-skull
    point cloud + pial brain mesh + bowl wireframe pattern (matching
    the acoustic / thermal / elastic 3-D figures), not the slab-only
    fallback.
    """
    _el._iso_3d_scalar(
        rate_map, run, out_path,
        title="Firing rate iso-surfaces",
        fractions=(0.50, 0.25),
        cfg=cfg,
    )


def _raster_with_key(sim: dict[str, Any], run: dict[str, Any], peak_idx: tuple,
                     out_path: Path, n_groups: int, group_period_ms: float,
                     n_draw: int) -> None:
    """Raster + 3-plane ROI key + depth distribution.

    Time axis is windowed to ``[0, T_pulse_ms · 1.1]`` (the active push
    window plus 10% padding) rather than the full PRF period: spike
    activity outside the ON window is sparse and uninformative for the
    figure, and the wider window compresses the burst into a thin
    vertical band.  ROI key is shown on three orthogonal slices (x-y,
    x-z, y-z) through the brain-restricted peak voxel so the spatial
    extent of recruitment is legible regardless of viewing axis.
    """
    spike_times = sim["spike_times_per_voxel"]
    if not spike_times:
        fig, ax = plt.subplots(figsize=(9, 3.8))
        ax.text(0.5, 0.5, "no firing voxels", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        return

    rng = np.random.default_rng(_RNG_SEED)
    keys = list(spike_times.keys())
    if len(keys) > n_draw:
        keys = [keys[i] for i in rng.choice(len(keys), n_draw, replace=False)]

    dx_mm = run["dx"] * 1e3
    dy_mm = run["dy"] * 1e3
    dz_mm = run["dz"] * 1e3
    depths_mm = np.array([k[0] * dx_mm for k in keys])
    order = np.argsort(depths_mm)
    keys_sorted = [keys[i] for i in order]
    depths_sorted = depths_mm[order]

    # Window to the active modulation time (T_pulse + 10%) — not hard-coded.
    T_pulse_ms = float(sim.get("T_pulse_ms", 30.0))
    raster_window_ms = T_pulse_ms * 1.1

    fig = plt.figure(figsize=(13, 4.4), constrained_layout=True)
    gs = fig.add_gridspec(1, 5, width_ratios=[2.5, 0.85, 0.85, 0.85, 0.7])
    ax_raster = fig.add_subplot(gs[0, 0])
    ax_xy = fig.add_subplot(gs[0, 1])
    ax_xz = fig.add_subplot(gs[0, 2])
    ax_yz = fig.add_subplot(gs[0, 3])
    ax_hist = fig.add_subplot(gs[0, 4])

    cmap = plt.get_cmap("plasma")
    d_lo, d_hi = float(depths_sorted.min()), float(depths_sorted.max())
    norm = (depths_sorted - d_lo) / max(d_hi - d_lo, 1e-9)

    # Single-burst raster (no theta replication — window is one burst only).
    for row, (k, depth, c) in enumerate(zip(keys_sorted, depths_sorted, norm)):
        st = np.asarray(spike_times[k])
        if len(st) == 0:
            continue
        st = st[st <= raster_window_ms]
        if len(st) == 0:
            continue
        ax_raster.vlines(st, row - 0.45, row + 0.45,
                         color=cmap(c), lw=0.5)

    ax_raster.axvspan(0, T_pulse_ms, color="0.92", alpha=0.6, lw=0)
    ax_raster.set_xlim(0, raster_window_ms)
    # Invert y-axis so shallow depth is on top, deep is on the bottom
    # (radiology convention; matches the depth-down convention used on
    # all spatial slice figures in the report).
    ax_raster.set_ylim(len(keys_sorted) - 0.5, -0.5)
    ax_raster.set_xlabel("Time within burst (ms)")
    ax_raster.set_ylabel("Neuron (sorted by depth, shallow $\\to$ deep)")
    ax_raster.set_title(
        f"(a) Spike raster, {len(keys_sorted)} neurons "
        f"(active window 0–{T_pulse_ms:.0f} ms shaded)", fontsize=9,
    )
    ax_raster.text(raster_window_ms * 1.005, 0, f"{d_lo:.0f} mm", fontsize=7,
                   color="0.3", ha="left", va="top")
    ax_raster.text(raster_window_ms * 1.005, len(keys_sorted) - 1,
                   f"{d_hi:.0f} mm", fontsize=7, color="0.3", ha="left", va="bottom")

    # 3-plane ROI key.  Each panel shows the firing-voxel scatter on the
    # mid-plane through the brain-restricted peak voxel (x-y at z=peak,
    # x-z at y=peak, y-z at x=peak), with the skull contour overlaid.
    nx, ny, nz = sim["spike_count_map"].shape
    skull = run["c"] > 1600.0
    xs = np.array([k[0] * dx_mm for k in keys_sorted])
    ys = np.array([k[1] * dy_mm for k in keys_sorted])
    zs = np.array([k[2] * dz_mm for k in keys_sorted])

    # x-y plane (z = peak); axes y (lateral) horiz, x (depth) vert
    sm_xy = skull[:, :, peak_idx[2]]
    ax_xy.contour(np.linspace(0, ny * dy_mm, ny),
                  np.linspace(0, nx * dx_mm, nx),
                  sm_xy, levels=[0.5], colors="cyan", linewidths=0.5)
    ax_xy.scatter(ys, xs, c=norm, cmap=cmap, s=8, edgecolor="none")
    ax_xy.set_xlim(0, ny * dy_mm); ax_xy.set_ylim(nx * dx_mm, 0)
    ax_xy.set_aspect("equal")
    ax_xy.set_xlabel("y, lat. (mm)"); ax_xy.set_ylabel("x, depth (mm)")
    ax_xy.set_title(f"(b1) x-y (z={peak_idx[2]*dz_mm:.0f} mm)", fontsize=8)

    # x-z plane (y = peak); axes z horiz, x vert
    sm_xz = skull[:, peak_idx[1], :]
    ax_xz.contour(np.linspace(0, nz * dz_mm, nz),
                  np.linspace(0, nx * dx_mm, nx),
                  sm_xz, levels=[0.5], colors="cyan", linewidths=0.5)
    ax_xz.scatter(zs, xs, c=norm, cmap=cmap, s=8, edgecolor="none")
    ax_xz.set_xlim(0, nz * dz_mm); ax_xz.set_ylim(nx * dx_mm, 0)
    ax_xz.set_aspect("equal")
    ax_xz.set_xlabel("z, elev. (mm)"); ax_xz.set_ylabel("")
    ax_xz.set_title(f"(b2) x-z (y={peak_idx[1]*dy_mm:.0f} mm)", fontsize=8)

    # y-z plane (x = peak); axes y horiz, z vert
    sm_yz = skull[peak_idx[0], :, :].T
    ax_yz.contour(np.linspace(0, ny * dy_mm, ny),
                  np.linspace(0, nz * dz_mm, nz),
                  sm_yz, levels=[0.5], colors="cyan", linewidths=0.5)
    ax_yz.scatter(ys, zs, c=norm, cmap=cmap, s=8, edgecolor="none")
    ax_yz.set_xlim(0, ny * dy_mm); ax_yz.set_ylim(0, nz * dz_mm)
    ax_yz.set_aspect("equal")
    ax_yz.set_xlabel("y, lat. (mm)"); ax_yz.set_ylabel("z, elev. (mm)")
    ax_yz.set_title(f"(b3) y-z (x={peak_idx[0]*dx_mm:.0f} mm)", fontsize=8)

    # depth histogram
    ax_hist.hist(depths_sorted, bins=15, orientation="horizontal",
                 color="0.4", edgecolor="white")
    ax_hist.invert_yaxis()
    ax_hist.set_xlabel("count")
    ax_hist.set_ylabel("Depth (mm)")
    ax_hist.set_title("(c) Depth dist.", fontsize=9)

    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _psth_recruitment_session(sim: dict[str, Any], out_path: Path,
                               group_period_ms: float, n_groups_session: int
                               ) -> None:
    """Three-panel: (a) PSTH, (b) recruitment by depth band, (c) per-group
    spike total over the full session schedule.

    The PSTH and recruitment x-axis is windowed to the active modulation
    time (``T_pulse_ms · 1.1``) rather than the full PRF period so the
    burst structure is legible; spike activity outside the ON window is
    sparse (residual late firing from voxels at the threshold edge) and
    would otherwise compress the burst into a thin band.
    """
    spike_times = sim["spike_times_per_voxel"]

    T_pulse_ms = float(sim.get("T_pulse_ms", 30.0))
    psth_window_ms = T_pulse_ms * 1.1

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), constrained_layout=True)

    # (a) PSTH over the active push window + 10% padding
    bin_ms = 0.25
    edges = np.arange(0, psth_window_ms + bin_ms, bin_ms)
    centers = 0.5 * (edges[:-1] + edges[1:])
    if spike_times:
        all_st = np.concatenate(list(spike_times.values()))
        h, _ = np.histogram(all_st, bins=edges)
        rate = h / (bin_ms * 1e-3) / max(len(spike_times), 1)
    else:
        rate = np.zeros_like(centers)
    axes[0].axvspan(0, T_pulse_ms, color="0.92", alpha=0.6, lw=0)
    axes[0].fill_between(centers, 0, rate, color="#2E5984", alpha=0.7)
    axes[0].set_xlim(0, psth_window_ms)
    axes[0].set_xlabel("Time within burst (ms)")
    axes[0].set_ylabel("Rate (Hz / neuron)")
    axes[0].set_title(f"(a) Population PSTH (active window 0–{T_pulse_ms:.0f} ms)",
                      fontsize=9)

    # (b) Recruitment by depth band — five quintile-based bands derived
    # from the firing-voxel depth distribution so each band carries an
    # equal fraction of the firing population (much more informative than
    # arbitrary mm cuts that may land entirely above or below the
    # firing zone).
    fine_bin = 0.5
    edges_f = np.arange(0, psth_window_ms + fine_bin, fine_bin)
    centers_f = 0.5 * (edges_f[:-1] + edges_f[1:])
    dx_mm = sim.get("_dx_mm", 1.0)

    band_colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]
    bands: list = []
    if spike_times:
        all_depths = np.sort(np.array([k[0] * dx_mm
                                       for k in spike_times.keys()]))
        if len(all_depths) >= 5:
            qcuts = np.percentile(all_depths, [20, 40, 60, 80])
            edges = [float(all_depths.min()) - 1e-6,
                     float(qcuts[0]), float(qcuts[1]),
                     float(qcuts[2]), float(qcuts[3]),
                     float(all_depths.max()) + 1e-6]
            for k in range(5):
                lo, hi = edges[k], edges[k + 1]
                bands.append((f"Q{k+1}", f"{lo:.0f}-{hi:.0f} mm",
                              lo, hi, band_colors[k]))
        else:
            # Few firing voxels — fall back to a single "all" band
            bands = [("all", f"{all_depths.min():.0f}-{all_depths.max():.0f} mm",
                      float(all_depths.min()) - 1e-6,
                      float(all_depths.max()) + 1e-6, band_colors[0])]
    for label, sub, lo, hi, color in bands:
        if not spike_times:
            continue
        in_band = [(k, st) for k, st in spike_times.items()
                   if lo <= k[0] * dx_mm < hi]
        if not in_band:
            continue
        active = np.zeros(len(centers_f), dtype=int)
        for k, st in in_band:
            h, _ = np.histogram(st, bins=edges_f)
            active += (h > 0).astype(int)
        axes[1].plot(centers_f, active / len(in_band), color=color, lw=1.0,
                     label=f"{label} {sub} (n={len(in_band)})")
    axes[1].axvspan(0, T_pulse_ms, color="0.92", alpha=0.6, lw=0)
    axes[1].set_xlim(0, psth_window_ms)
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_xlabel("Time within group (ms)")
    axes[1].set_ylabel("Active fraction")
    axes[1].set_title("(b) Recruitment by depth band", fontsize=9)
    axes[1].legend(fontsize=7, frameon=False)

    # (c) Session envelope: per-group spike total (constant if stationary;
    # placeholder — real session sim would add fatigue / drift)
    if spike_times:
        per_group = sum(len(st) for st in spike_times.values())
    else:
        per_group = 0
    g = np.arange(n_groups_session)
    totals = np.full(n_groups_session, per_group, dtype=float)
    axes[2].plot(g, totals, color="0.3", lw=0.8)
    axes[2].set_xlim(0, n_groups_session - 1)
    axes[2].set_ylim(0, max(per_group * 1.2, 1.0))
    axes[2].set_xlabel("Theta group #")
    axes[2].set_ylabel("Spikes / group (population)")
    axes[2].set_title(f"(c) Session: {n_groups_session} groups",
                      fontsize=9)

    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _rate_dist_psd(sim: dict[str, Any], out_path: Path,
                   group_period_ms: float, n_groups_session: int,
                   prf_hz: float, theta_hz: float) -> None:
    """(a) per-neuron firing-rate distribution; (b) population spike-train PSD."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), constrained_layout=True)

    rates = sim["rate_map"][np.isfinite(sim["latency_map"])]
    if rates.size:
        axes[0].hist(rates, bins=40, color="#2E5984", edgecolor="white")
    axes[0].set_xlabel("Per-voxel firing rate (Hz)")
    axes[0].set_ylabel("voxel count")
    axes[0].set_title("(a) Rate distribution (firing voxels)", fontsize=9)
    axes[0].set_yscale("log")

    # Build a population spike-count signal sampled at 2 kHz over the
    # full session and compute its PSD.
    spike_times = sim["spike_times_per_voxel"]
    fs = 2000.0
    T_session_s = n_groups_session * group_period_ms * 1e-3
    n_samp = int(fs * T_session_s)
    sig = np.zeros(n_samp, dtype=np.float32)
    if spike_times:
        for st_burst in spike_times.values():
            for g in range(n_groups_session):
                ts = (st_burst + g * group_period_ms) * 1e-3
                idxs = np.clip((ts * fs).astype(int), 0, n_samp - 1)
                np.add.at(sig, idxs, 1.0)
    if sig.sum() > 0:
        from numpy.fft import rfft, rfftfreq
        # Welch-ish: mean of segment PSDs
        seg = 1 << 13
        if n_samp >= seg:
            n_seg = n_samp // seg
            P = np.zeros(seg // 2 + 1)
            for k in range(n_seg):
                x = sig[k * seg:(k + 1) * seg] - sig[k * seg:(k + 1) * seg].mean()
                X = rfft(x * np.hanning(seg))
                P += np.abs(X) ** 2
            P /= n_seg
            f = rfftfreq(seg, d=1.0 / fs)
        else:
            X = rfft(sig - sig.mean())
            P = np.abs(X) ** 2
            f = rfftfreq(n_samp, d=1.0 / fs)
        nz = (f > 0)
        axes[1].loglog(f[nz], P[nz], color="0.3", lw=0.8)
        axes[1].axvline(theta_hz, color="#C44E52", lw=0.6, ls="--",
                        label=f"theta {theta_hz:.0f} Hz")
        axes[1].axvline(prf_hz, color="#55A868", lw=0.6, ls="--",
                        label=f"PRF {prf_hz:.0f} Hz")
        axes[1].legend(fontsize=7, frameon=False)
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Power (a.u.)")
    axes[1].set_title("(b) Population spike-train PSD", fontsize=9)

    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# =====================================================================
# Builder entry-point
# =====================================================================


def build(cfg, fig_dir: Path) -> dict[str, Any]:
    """Render the neural-outputs section."""
    repo_root = Path(__file__).resolve().parents[2]
    sim_dir = (repo_root / cfg.OUT_DIR).parent
    if not sim_dir.is_absolute():
        sim_dir = (repo_root / sim_dir).resolve()

    run = _ao._load_run(sim_dir)
    K_a = float(getattr(cfg, "MEMBRANE_K_A_NM",
                        getattr(cfg, "NEURAL_K_A", 0.25)))
    coupling_factor = float(getattr(cfg, "NEURAL_COUPLING_FACTOR", 200.0))

    seq = getattr(cfg, "NEUROMOD_SEQUENCE", {})
    group_on_s = float(seq.get("group_on_s", 5e-3))
    group_period_s = float(seq.get("group_period_s", 0.2))
    session_s = float(seq.get("session_duration_s", 80.0))
    prf_hz = float(seq.get("pulse_repetition_freq_hz", 1e3))
    theta_hz = float(seq.get("theta_freq_hz", 5.0))
    n_groups_session = max(int(round(session_s / group_period_s)), 1)

    # Prefer the manuscript-grade 3-comp pipeline output if it has been
    # produced upstream by ``scripts/run_neural_sim.py``; otherwise
    # recompute via the same 3-comp pipeline in-process (numba-fast),
    # so the report and the manuscript can never disagree on the
    # neuron model.
    npz_path = sim_dir / "neural_results.npz"
    if npz_path.exists():
        sim = _load_npz_results(npz_path)
        T_peak = sim["T_peak"]
        piezo1_g = float(sim["piezo1_g_density_mS_cm2"])
        use_trek1 = True
    else:
        sim = _simulate_population_3comp(
            sim_dir, run, cfg,
            K_A=K_a, coupling_factor=coupling_factor,
            dt_ms=0.001,
            duration_ms=max(_DURATION_MS, group_on_s * 1e3 + 5.0),
            threshold_mNm=_TENSION_THRESHOLD,
        )
        T_peak = sim["T_peak"]
        piezo1_g = float(sim["piezo1_g_density_mS_cm2"])
        use_trek1 = bool(getattr(cfg, "NEURAL_USE_TREK1", True))
    sim["_dx_mm"] = run["dx"] * 1e3

    # Peak voxel — restrict to brain
    skull = run["c"] > 1600.0
    brain = ~skull
    masked = np.where(brain, sim["rate_map"], -np.inf)
    if np.isfinite(masked).any() and masked.max() > 0:
        peak_idx = np.unravel_index(int(np.argmax(masked)), sim["rate_map"].shape)
    else:
        peak_idx = _el._peak_voxel_brain(T_peak, run["c"])

    figures: dict[str, str] = {}

    # 4.2 latency ortho
    fig_path = fig_dir / "neural_latency_ortho.png"
    _ortho_neural(sim["latency_map"], run, peak_idx, fig_path,
                  cmap="viridis", label="Time-to-first-spike (ms)",
                  title="First-spike latency (brain peak)",
                  reverse_cmap=True)
    figures["latency_ortho"] = fig_path.name

    # 4.3 peak rate ortho
    fig_path = fig_dir / "neural_rate_ortho.png"
    _ortho_neural(sim["rate_map"], run, peak_idx, fig_path,
                  cmap="magma", label="Firing rate (Hz)",
                  title="Per-voxel firing rate (brain peak)")
    figures["rate_ortho"] = fig_path.name

    # 4.4 firing iso 3D
    fig_path = fig_dir / "neural_rate_3d.png"
    _firing_iso_3d(sim["rate_map"], run, fig_path, cfg=cfg)
    figures["rate_3d"] = fig_path.name

    # 4.5 linked tension <-> spike snapshots
    fig_path = fig_dir / "neural_snapshot_row.png"
    # Use the K_A / alpha that the actual run used (recorded in the npz),
    # not cfg defaults which may have drifted.
    _snapshot_row(T_peak, sim, run, peak_idx, sim_dir,
                  K_A=sim.get("K_A_npz", K_a),
                  coupling_factor=sim.get("coupling_factor_npz", coupling_factor),
                  out_path=fig_path)
    figures["snapshot_row"] = fig_path.name

    # 4.6 raster + ROI key
    fig_path = fig_dir / "neural_raster.png"
    _raster_with_key(sim, run, peak_idx, fig_path,
                     n_groups=_N_GROUPS, group_period_ms=group_period_s * 1e3,
                     n_draw=_RASTER_N)
    figures["raster"] = fig_path.name

    # 4.7 PSTH + recruitment + session envelope
    fig_path = fig_dir / "neural_psth.png"
    _psth_recruitment_session(sim, fig_path,
                              group_period_ms=group_period_s * 1e3,
                              n_groups_session=n_groups_session)
    figures["psth"] = fig_path.name

    # 4.8 rate dist + spike PSD
    fig_path = fig_dir / "neural_distrib.png"
    _rate_dist_psd(sim, fig_path,
                   group_period_ms=group_period_s * 1e3,
                   n_groups_session=n_groups_session,
                   prf_hz=prf_hz, theta_hz=theta_hz)
    figures["distrib"] = fig_path.name

    # 4.9 firing volume vs time
    fig_path = fig_dir / "neural_volume_t.png"
    vol_metrics = _firing_volume_vs_time(sim, run, fig_path,
                                         T_pulse_ms=sim["T_pulse_ms"])
    figures["volume_t"] = fig_path.name

    # 4.11 gating curves: Piezo1 (and TREK-1 if enabled) Boltzmann overlay
    fig_path = fig_dir / "neural_gating.png"
    _gating_curves(
        sim, use_trek1=use_trek1, T_pulse_ms=sim["T_pulse_ms"],
        T_max_observed_mNm=float(sim["T_peak"].max()),
        T_threshold_mNm=float(sim["threshold_mNm"]),
        out_path=fig_path,
    )
    figures["gating"] = fig_path.name

    # 4.10 representative neurons (5 cells spanning the firing-zone phenomenology)
    fig_path = fig_dir / "neural_representative.png"
    _representative_neurons(
        sim, run, sim_dir,
        K_A=sim.get("K_A_npz", K_a),
        coupling_factor=sim.get("coupling_factor_npz", coupling_factor),
        T_pulse_ms=sim["T_pulse_ms"],
        T_half_mNm=2.7,                # Piezo1 half-activation (manuscript)
        peak_idx=peak_idx,
        out_path=fig_path,
    )
    figures["representative"] = fig_path.name

    # ----- metrics -----
    finite_lat = sim["latency_map"][np.isfinite(sim["latency_map"])]
    rates_firing = sim["rate_map"][np.isfinite(sim["latency_map"])]
    n_firing = int(sim["n_firing_voxels"])
    n_active = int((sim["spike_count_map"] > 0).sum())
    target_extent_50 = float((sim["rate_map"] >= 0.5 * sim["rate_map"].max()).sum())
    target_extent_25 = float((sim["rate_map"] >= 0.25 * sim["rate_map"].max()).sum())
    voxel_vol_mm3 = (run["dx"] * run["dy"] * run["dz"]) * 1e9

    metrics = {
        "tension_threshold_mNm": _TENSION_THRESHOLD,
        "membrane_K_a_N_per_m": K_a,
        "neural_coupling_factor": coupling_factor,
        "piezo1_g_density_mS_cm2": piezo1_g,
        "use_trek1": use_trek1,
        "T_peak_max_mNm": float(T_peak.max()),
        "n_eligible_voxels": int(sim.get("n_eligible_voxels", n_firing)),
        "n_voxels_above_threshold": n_firing,
        "n_voxels_with_spike": n_active,
        "active_fraction_burst": (n_active / max(n_firing, 1)),
        "peak_rate_hz": float(np.nanmax(sim["rate_map"])),
        "median_latency_ms": float(np.median(finite_lat)) if finite_lat.size else float("nan"),
        "p10_latency_ms": float(np.percentile(finite_lat, 10)) if finite_lat.size else float("nan"),
        "p90_latency_ms": float(np.percentile(finite_lat, 90)) if finite_lat.size else float("nan"),
        "spikes_per_neuron_burst_mean": float(np.mean(sim["spike_count_map"][sim["spike_count_map"] > 0]))
            if n_active else 0.0,
        "iso50_volume_mm3": target_extent_50 * voxel_vol_mm3,
        "iso25_volume_mm3": target_extent_25 * voxel_vol_mm3,
        **vol_metrics,
        "T_pulse_ms": sim["T_pulse_ms"],
        "duration_ms": sim["duration_ms"],
        "group_period_ms": group_period_s * 1e3,
        "n_groups_session": n_groups_session,
        "peak_idx": [int(i) for i in peak_idx],
    }

    return {
        "neural_outputs": metrics,
        "figures_neural_outputs": figures,
    }
