"""Section 1 — Acoustic inputs.

Pure loader/plotter for the acoustic inputs the simulation pipeline
already wrote: medium maps (c/rho/alpha/beta_ds.npy), placement.json,
grid_params.npz. The grid + bowl mask are still built locally because
they are cheap geometry — but no NRRD load, no MediumBuilder, no
halle_placement re-run.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

import fullwave
from fullwave.transducers.tips import create_tips_transducer

from ultraneuron.paths import redact_path


def _stable_key(obj) -> str:
    """SHA1 of a JSON-serialized config slice. Stable across runs."""
    import hashlib
    payload = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def _placement_cache_key(plc: dict) -> str:
    return _stable_key({
        "scalp_site": plc["scalp_site"],
        "target_name": plc["target_name"],
        "focal_length_mm": plc["focal_length_mm"],
        "bowl_radius_mm": plc["bowl_radius_mm"],
    })


def _medium_cache_key(cfg: ModuleType, plc_key: str) -> str:
    return _stable_key({
        "domain": list(cfg.DOMAIN_SIZE),
        "f0": cfg.F0, "c0": cfg.C0, "ppw": cfg.PPW,
        "cfl": cfg.CFL, "duration": cfg.DURATION,
        "medium": cfg.MEDIUM,
        "placement_key": plc_key,
    })


def _load_placement(sim_dir: Path, plc: dict) -> tuple[dict, str]:
    """Load the placement dict written by the sim driver."""
    key = _placement_cache_key(plc)
    p_path = sim_dir / "placement.json"
    if not p_path.exists():
        raise FileNotFoundError(
            f"No placement.json at {p_path}. The simulation pipeline "
            f"now writes this; rerun bowl_transcranial_500khz.py (or "
            f"backfill the file) before building the validation report."
        )
    return json.loads(p_path.read_text()), key


def _to_jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {k: _to_jsonable(v) for k, v in x.items()}
    return x


def build(cfg: ModuleType, fig_dir: Path) -> dict:
    fig_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = fig_dir.parent / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    src = cfg.SOURCE
    plc = cfg.PLACEMENT
    med = cfg.MEDIUM

    # --- 1.1 Grid ---
    grid = fullwave.Grid(
        cfg.DOMAIN_SIZE, cfg.F0, cfg.DURATION,
        c0=cfg.C0, ppw=cfg.PPW, cfl=cfg.CFL,
    )
    grid_info = {
        "f0_hz": float(cfg.F0),
        "c0_mps": float(cfg.C0),
        "ppw": int(cfg.PPW),
        "cfl": float(cfg.CFL),
        "dx_m": float(grid.dx), "dy_m": float(grid.dy), "dz_m": float(grid.dz),
        "nx": int(grid.nx), "ny": int(grid.ny), "nz": int(grid.nz),
        "domain_x_mm": float(grid.nx * grid.dx * 1e3),
        "domain_y_mm": float(grid.ny * grid.dy * 1e3),
        "domain_z_mm": float(grid.nz * grid.dz * 1e3),
        "duration_s": float(cfg.DURATION),
        "dt_s": float(grid.dt),
        "nt": int(grid.nt),
    }

    # --- 1.2 Placement (loaded from sim pipeline output) ---
    sim_dir = Path(getattr(cfg, "SIM_DATA_DIR", Path(cfg.OUT_DIR).parent))
    if not sim_dir.is_absolute():
        # builders/ -> validation_report/ ->
        # docs/ -> repo root
        repo_root = Path(__file__).resolve().parents[2]
        sim_dir = (repo_root / sim_dir).resolve()
    p, plc_key = _load_placement(sim_dir, plc)
    print(f"  placement loaded from {sim_dir.name}/placement.json ({plc_key})")
    placement_info = {
        "scalp_site": plc["scalp_site"],
        "target_name": plc["target_name"],
        "focal_length_mm": float(plc["focal_length_mm"]),
        "bowl_radius_mm": float(plc["bowl_radius_mm"]),
        "xdc_center_lps": _to_jsonable(p["xdc_center_lps"]),
        "scalp_contact_lps": _to_jsonable(p["scalp_contact_lps"]),
        "focus_lps": _to_jsonable(p["focus_lps"]),
        "target_lps": _to_jsonable(p["target_lps"]),
        "perp_residual_mm": float(p["perp_residual_mm"]),
        "scalp_target_dist_mm": float(p["scalp_target_dist_mm"]),
        "apex_target_dist_mm": float(p["apex_target_dist_mm"]),
        "bowl_depth_mm": float(p["bowl_depth_mm"]),
        "eeg_to_placement_mm": (
            float(p["eeg_to_placement_mm"])
            if p.get("eeg_to_placement_mm") is not None else None
        ),
        "beam_dir_3d": _to_jsonable(p["beam_dir_3d"]),
        "normal_2d": _to_jsonable(p["normal"]),
        "tangent_2d": _to_jsonable(p["tangent"]),
        "z_mm": float(p["z_mm"]),
    }

    # --- 1.3 Bowl on slab grid ---
    bowl = create_tips_transducer(
        grid=grid,
        n_layers=src["n_layers"],
        roc_m=src["roc_m"],
        inner_radius_m=src["inner_radius_m"],
        outer_radius_m=src["outer_radius_m"],
        n_elements=src["n_elements"],
    )
    coords = bowl.source_coords_px
    source_info = {
        "type": src["type"],
        "n_layers": int(src["n_layers"]),
        "roc_mm": float(src["roc_m"] * 1e3),
        "outer_radius_mm": float(src["outer_radius_m"] * 1e3),
        "inner_radius_mm": float(src["inner_radius_m"] * 1e3),
        "n_elements": int(src["n_elements"]),
        "p0_mpa": float(src["p0_pa"] * 1e-6),
        "ncycles": int(src["ncycles"]),
        "drop_off": float(src["drop_off"]),
        "n_source_voxels": int(coords.shape[0]),
    }

    # --- 1.3 figure: bowl footprint (3 panels) ---
    # Convention: when depth (x) is shown, it goes on the y-axis with
    # 0 at the top (sonography / radiology convention).  The face-on
    # y-z aperture panel is unaffected.
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    elem_yz = np.zeros((grid.ny, grid.nz), dtype=np.int32)
    elem_yz[coords[:, 1], coords[:, 2]] = bowl.element_ids
    axes[0].imshow(elem_yz.T, origin="lower", cmap="tab10",
                   extent=[0, grid.ny * grid.dy * 1e3, 0, grid.nz * grid.dz * 1e3])
    axes[0].set(title="Bowl footprint (y-z, by element)",
                xlabel="lateral y [mm]", ylabel="elevational z [mm]", aspect="equal")
    # x-y projection: lateral y on x-axis, depth x on y-axis (0 at top).
    proj_xy = bowl.source_mask.any(axis=2)  # shape (nx, ny), [depth, lateral]
    axes[1].imshow(proj_xy, origin="upper", cmap="gray_r",
                   extent=[0, grid.ny * grid.dy * 1e3,
                           grid.nx * grid.dx * 1e3, 0])
    axes[1].set(title="Bowl y-x projection",
                xlabel="lateral y [mm]", ylabel="depth x [mm]")
    # x-z projection: elevational z on x-axis, depth x on y-axis (0 at top).
    proj_xz = bowl.source_mask.any(axis=1)  # shape (nx, nz), [depth, elev]
    axes[2].imshow(proj_xz, origin="upper", cmap="gray_r",
                   extent=[0, grid.nz * grid.dz * 1e3,
                           grid.nx * grid.dx * 1e3, 0])
    axes[2].set(title="Bowl z-x projection",
                xlabel="elevational z [mm]", ylabel="depth x [mm]")
    fig.suptitle(f"Bowl geometry — F={source_info['roc_mm']:.0f} mm, "
                 f"R={source_info['outer_radius_mm']:.0f} mm, "
                 f"{source_info['n_elements']} elements "
                 f"({source_info['n_source_voxels']} voxels)")
    fig.tight_layout()
    fig_bowl = fig_dir / "fig_inputs_bowl_geometry.png"
    fig.savefig(fig_bowl, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- 1.4 Medium maps (loaded from sim pipeline output) ---
    needed = {n: sim_dir / f"{n}_ds.npy" for n in ("c", "rho", "alpha", "beta")}
    missing = [n for n, pth in needed.items() if not pth.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing medium maps in {sim_dir}: {missing}. The simulation "
            f"pipeline (bowl_transcranial_500khz.py) now writes these "
            f"alongside propagation_map.npy; rerun the FDTD step (or "
            f"backfill) before building the validation report."
        )
    c_map = np.load(needed["c"])
    rho_map = np.load(needed["rho"])
    alpha_map = np.load(needed["alpha"])
    beta_map = np.load(needed["beta"])
    print(f"  medium loaded from {sim_dir.name}/{{c,rho,alpha,beta}}_ds.npy "
          f"shape={c_map.shape}")

    medium_info = {
        "type": med["type"],
        "nrrd_path": redact_path(med["nrrd_path"]),
        "lower_threshold": int(med["lower_threshold"]),
        "upper_threshold": int(med["upper_threshold"]),
        "speed_range_mps": _to_jsonable(med["speed_range"]),
        "density_range_kgm3": _to_jsonable(med["density_range"]),
        "blur_sigma_m": float(med["blur_sigma_m"]),
        "background": med["background"],
        "c_min": float(c_map.min()), "c_max": float(c_map.max()),
        "rho_min": float(rho_map.min()), "rho_max": float(rho_map.max()),
        "alpha_min": float(alpha_map.min()), "alpha_max": float(alpha_map.max()),
        "beta_min": float(beta_map.min()), "beta_max": float(beta_map.max()),
        "skull_voxel_count": int(((c_map > med["speed_range"][0] + 1).sum())),
    }

    # Slice indices match the loaded medium array's shape (which may be
    # downsampled relative to grid.{nx,ny,nz} when loaded from *_ds.npy).
    iy = c_map.shape[1] // 2
    iz = c_map.shape[2] // 2

    # --- 1.5 3D placement (cached PNG keyed on placement) ---
    fig_placement_3d = fig_dir / "fig_inputs_placement_3d.png"
    cached_3d = cache_dir / f"placement_3d_{plc_key}.png"
    if cached_3d.exists():
        import shutil as _shutil
        _shutil.copyfile(cached_3d, fig_placement_3d)
        print(f"  3D placement cache HIT ({plc_key})")
    else:
        from ultraneuron.registration import ensure_registration_importable
        try:
            ensure_registration_importable()
            import visualize_placement_3d as viz  # noqa: E402
            viz.OUT_PATH = str(cached_3d)
            import matplotlib.figure as _mf
            _orig_suptitle = _mf.Figure.suptitle
            new_title = (
                f"3D placement: {plc['scalp_site']} → {plc['target_name']} on Halle skull "
                f"(perpendicular-to-skull, F={plc['focal_length_mm']:.0f} mm, "
                f"R={plc['bowl_radius_mm']:.0f} mm)"
            )

            def _patched_suptitle(self, _t, *a, **kw):
                kw["fontsize"] = 22
                kw["fontweight"] = "bold"
                return _orig_suptitle(self, new_title, *a, **kw)

            _mf.Figure.suptitle = _patched_suptitle
            viz.render(p)
            import shutil as _shutil
            _shutil.copyfile(cached_3d, fig_placement_3d)
        except Exception as e:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, f"3D placement unavailable:\n{e}",
                    ha="center", va="center", wrap=True)
            ax.axis("off")
            fig.savefig(fig_placement_3d, dpi=150, bbox_inches="tight")
            plt.close(fig)
        finally:
            _mf.Figure.suptitle = _orig_suptitle

    # --- 1.6 Combined medium maps (c, rho, alpha as 3 rows × 3 ortho-slice cols) ---
    # Depth (x) is plotted on the vertical axis with depth=0 at the top
    # (radiology convention).  Lateral and elevational planes (without
    # a depth axis) keep their natural orientation.
    ix = c_map.shape[0] // 2
    Lx = grid.nx * grid.dx * 1e3
    Ly = grid.ny * grid.dy * 1e3
    Lz = grid.nz * grid.dz * 1e3
    ext_xy_depth = [0, Ly, Lx, 0]   # x (depth) vertical, 0 on top
    ext_xz_depth = [0, Lz, Lx, 0]
    ext_yz = [0, Ly, 0, Lz]
    rows = [
        (c_map, "sound speed c", "m/s", "bone"),
        (rho_map, "density ρ", "kg/m³", "cividis"),
        (alpha_map, "attenuation α", "dB/(MHz·cm)", "magma"),
        (beta_map, "nonlinearity β", "dimensionless", "viridis"),
    ]
    fig, axes = plt.subplots(4, 3, figsize=(15, 17))
    for r, (m3, lbl, units, cmap) in enumerate(rows):
        # x-y plane (mid z): rows = depth x, cols = lateral y
        im = axes[r, 0].imshow(m3[:, :, iz], origin="upper", cmap=cmap,
                               aspect="equal", extent=ext_xy_depth)
        axes[r, 0].set(title=f"{lbl} — x-y (mid z)",
                       xlabel="lateral y [mm]", ylabel="depth x [mm]")
        plt.colorbar(im, ax=axes[r, 0], label=units)
        # x-z plane (mid y): rows = depth x, cols = elevational z
        im = axes[r, 1].imshow(m3[:, iy, :], origin="upper", cmap=cmap,
                               aspect="equal", extent=ext_xz_depth)
        axes[r, 1].set(title=f"{lbl} — x-z (mid y)",
                       xlabel="elevational z [mm]", ylabel="depth x [mm]")
        plt.colorbar(im, ax=axes[r, 1], label=units)
        # y-z plane (mid x): no depth axis; keep natural orientation
        im = axes[r, 2].imshow(m3[ix, :, :].T, origin="lower", cmap=cmap,
                               aspect="equal", extent=ext_yz)
        axes[r, 2].set(title=f"{lbl} — y-z (mid x)",
                       xlabel="lateral y [mm]", ylabel="elevational z [mm]")
        plt.colorbar(im, ax=axes[r, 2], label=units)
    fig.suptitle("Acoustic medium: sound speed, density, attenuation, nonlinearity",
                 fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig_medium = fig_dir / "fig_inputs_medium.png"
    fig.savefig(fig_medium, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- 1.7 Excitation waveform ---
    worst_pulse = fullwave.utils.pulse.gaussian_modulated_sinusoidal_signal(
        nt=grid.nt,
        f0=cfg.F0,
        duration=cfg.DURATION,
        ncycles=src["ncycles"],
        drop_off=src["drop_off"],
        p0=src["p0_pa"],
        i_layer=0,
        dt_for_layer_delay=grid.dt,
        cfl_for_layer_delay=grid.cfl,
        delay_sec=0.0,
    )
    threshold = np.abs(worst_pulse).max() * 1e-6
    active_idx = np.nonzero(np.abs(worst_pulse) > threshold)[0]
    n_tic = int(active_idx[-1] + 1) if len(active_idx) > 0 else int(grid.nt)
    p_src = worst_pulse[:n_tic]
    cw_correction = float(0.5 * np.max(p_src ** 2) / np.mean(p_src ** 2))

    t = np.arange(n_tic) * grid.dt * 1e6  # µs
    P = np.fft.rfft(p_src)
    f = np.fft.rfftfreq(n_tic, d=grid.dt) * 1e-3  # kHz
    spec = np.abs(P) / np.abs(P).max()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(t, p_src * 1e-6, lw=1.2)
    axes[0].set(title=f"Excitation pulse (ncycles={src['ncycles']}, "
                      f"drop_off={src['drop_off']}, p0={source_info['p0_mpa']:.2f} MPa)",
                xlabel="time [µs]", ylabel="pressure [MPa]")
    axes[0].grid(alpha=0.3)
    axes[1].plot(f, 20 * np.log10(spec + 1e-12), lw=1.2)
    axes[1].axvline(cfg.F0 * 1e-3, color="orange", ls="--", lw=0.8,
                    label=f"f0 = {cfg.F0*1e-3:.0f} kHz")
    axes[1].set(title="Excitation spectrum",
                xlabel="frequency [kHz]", ylabel="|P(f)| [dB, normalized]",
                xlim=(0, cfg.F0 * 4e-3), ylim=(-80, 5))
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig_exc = fig_dir / "fig_inputs_excitation.png"
    fig.savefig(fig_exc, dpi=150, bbox_inches="tight")
    plt.close(fig)

    excitation_info = {
        "ncycles": int(src["ncycles"]),
        "drop_off": float(src["drop_off"]),
        "p0_mpa": float(src["p0_pa"] * 1e-6),
        "n_tic": n_tic,
        "n_tic_duration_us": float(n_tic * grid.dt * 1e6),
        "cw_correction": cw_correction,
        "n_total_sources": int(bowl.source_mask.sum()),
    }

    # --- Neuromodulation sequence (from cfg) ---
    seq = getattr(cfg, "NEUROMOD_SEQUENCE", None)
    sequence_info = None
    fig_sequence_name = None
    if seq is not None:
        # 3-panel nested-zoom figure for the Yaakub-style human protocol:
        # (1) Carrier zoom — first ~30 µs of the carrier inside the tone-burst.
        # (2) One pulse period — 80 ms ON / 120 ms OFF rectangular envelope.
        # (3) Pulse train — 5 periods (1 s) of the 5 Hz tone-burst train.
        f0 = float(seq["carrier_freq_hz"])
        t_burst = float(seq["tone_burst_duration_s"])
        prf = float(seq["pulse_repetition_freq_hz"])
        pulse_period = float(seq.get("pulse_period_s", 1.0 / prf))
        duty = float(seq.get("duty_cycle", t_burst / pulse_period))
        n_periods_shown = 5

        # Panel 1: zoom into the carrier — first 30 µs of the tone-burst,
        # since the full 80 ms × 500 kHz = 40000 cycles can't be plotted.
        t1_window = min(t_burst, 30e-6)
        t1 = np.linspace(0, t1_window, 1500)
        carrier = np.sin(2 * np.pi * f0 * t1)

        # Panel 2: one pulse period — 80 ms ON, 120 ms OFF. Use a step
        # envelope (rectangle), not a smooth window, since Yaakub gates
        # the carrier with a hard rectangular window.
        t2 = np.linspace(0, pulse_period, 4000)
        env2 = (t2 < t_burst).astype(float)

        # Panel 3: pulse train across n_periods_shown periods.
        t3 = np.linspace(0, n_periods_shown * pulse_period, 8000)
        env3 = np.zeros_like(t3)
        for k in range(n_periods_shown):
            tk = k * pulse_period
            env3[(t3 >= tk) & (t3 < tk + t_burst)] = 1.0

        fig, axes = plt.subplots(3, 1, figsize=(13, 7.5))

        # --- Panel 1: carrier zoom ---
        axes[0].plot(t1 * 1e6, carrier, lw=0.9, color="C0")
        axes[0].axhline(0, color="k", lw=0.4, alpha=0.3)
        axes[0].set(title=f"(1) Carrier zoom — first {t1_window*1e6:.0f} µs of the "
                          f"{t_burst*1e3:.0f} ms tone-burst "
                          f"({f0*1e-3:.0f} kHz carrier; full burst = "
                          f"{t_burst*f0:.0f} cycles)",
                    xlabel="time [µs]", ylabel="amplitude")
        axes[0].grid(alpha=0.3)

        # --- Panel 2: one pulse period ---
        axes[1].fill_between(t2 * 1e3, 0, env2, color="C0", step="post",
                             alpha=0.55, linewidth=0)
        axes[1].plot(t2 * 1e3, env2, lw=1.0, color="C0", drawstyle="steps-post")
        axes[1].axvspan(0, t_burst * 1e3, color="C0", alpha=0.10)
        axes[1].set(title=f"(2) One pulse period — {t_burst*1e3:.0f} ms ON / "
                          f"{(pulse_period-t_burst)*1e3:.0f} ms OFF "
                          f"(duty cycle {duty*100:.0f}%)",
                    xlabel="time [ms]", ylabel="envelope",
                    xlim=(0, pulse_period * 1e3), ylim=(-0.05, 1.15))
        axes[1].grid(alpha=0.3)

        # --- Panel 3: pulse train ---
        axes[2].fill_between(t3 * 1e3, 0, env3, color="C0", step="post",
                             alpha=0.55, linewidth=0)
        axes[2].plot(t3 * 1e3, env3, lw=1.0, color="C0", drawstyle="steps-post")
        n_total = int(seq.get("n_pulses_per_session",
                              round(prf * float(seq["session_duration_s"]))))
        axes[2].set(title=f"(3) Pulse train — {prf:.0f} Hz PRF "
                          f"({n_periods_shown} periods shown; session "
                          f"{seq['session_duration_s']:.0f} s = "
                          f"{n_total} tone-bursts)",
                    xlabel="time [ms]", ylabel="envelope",
                    xlim=(0, n_periods_shown * pulse_period * 1e3),
                    ylim=(-0.05, 1.15))
        axes[2].grid(alpha=0.3)

        fig.suptitle("Neuromodulation sequence (Yaakub 2023 human profile) "
                     "— nested zoom",
                     fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig_sequence = fig_dir / "fig_inputs_sequence.png"
        fig.savefig(fig_sequence, dpi=150, bbox_inches="tight")
        plt.close(fig)
        fig_sequence_name = fig_sequence.name

        sequence_info = {
            "profile": seq.get("profile", "custom"),
            "envelope": seq["envelope"],
            "carrier_khz": float(seq["carrier_freq_hz"] * 1e-3),
            "ncycles_per_pulse": int(seq["ncycles_per_pulse"]),
            "tone_burst_ms": float(seq["tone_burst_duration_s"] * 1e3),
            "prf_hz": float(seq["pulse_repetition_freq_hz"]),
            "pulse_period_ms": float(pulse_period * 1e3),
            "duty_cycle_pct": float(duty * 100),
            "isi_ms": float(seq["isi_s"] * 1e3),
            "session_s": float(seq["session_duration_s"]),
            "n_pulses_per_session": n_total,
            "arf_push_ms": float(seq["arf_push_window_s"] * 1e3),
            "i_sppa_w_cm2": float(seq["i_sppa_w_cm2"]),
        }

    # Reproducibility manifest stamp
    import datetime
    import platform
    import socket
    import subprocess

    def _git_sha(path: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", "-C", path, "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, timeout=2,
            )
            if out.returncode == 0:
                return out.stdout.strip()
        except Exception:
            return None
        return None

    from ultraneuron.registration import neuromod_parameters_root

    repro = {
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "fullwave_path": redact_path(Path(fullwave.__file__).parent),
        "fullwave_git_sha": _git_sha(str(Path(fullwave.__file__).parents[1])),
        "neuromod_git_sha": _git_sha(str(neuromod_parameters_root() or "")),
        "amun_git_sha": _git_sha(str(Path(__file__).resolve().parents[2])),
        "generated_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
    }

    return {
        "sim_name": cfg.SIM_NAME,
        "sim_title": cfg.SIM_TITLE,
        "grid": grid_info,
        "source": source_info,
        "placement": placement_info,
        "medium": medium_info,
        "excitation": excitation_info,
        "sequence": sequence_info,
        "repro": repro,
        "figures": {
            "bowl_geometry": fig_bowl.name,
            "placement_3d": fig_placement_3d.name,
            "medium": fig_medium.name,
            "excitation": fig_exc.name,
            "sequence": fig_sequence_name,
        },
    }


def write_manifest(ctx: dict, out_path: Path) -> None:
    out_path.write_text(json.dumps(ctx, indent=2, default=_to_jsonable))
