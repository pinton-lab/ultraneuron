"""Time every stage of the canonical Halle/dACC (Yaakub) pipeline on GPU 0.

Re-runs the canonical chain exactly as the canonical outputs in
``data/halle_dacc/`` were produced, into a SCRATCH directory (the canonical
files are only read, never written):

  1. fullwave      Fullwave 2 acoustic FDTD via ``bowl_transcranial_500khz.main``
                   (same arguments as ``run_halle_dacc_fdtd.py``: auto-pick aim at
                   ACC_left, 130x80x80 mm domain, source_p0 = 0.10 MPa,
                   skip_shear=True).  The report figures that main() renders after
                   the acoustic save are skipped (not part of the compute).
  2. arf_bioheat   ``rescale_propagation_map.main`` (in-brain PPP -> 0.50 MPa),
                   ``recompute_arf.py`` (brain-masked ARF), ``run_bioheat.py``
                   (burst-resolved Pennes, default dt_on/dt_off, config protocol).
  3. shear         Kelvin-Voigt shear FDTD with the parameters of
                   ``rerun_shear_fdtd.py`` (1 x 200 ms period, 20 ms ON), but on the
                   GPU/JAX solver ``shearwave.shear_fdtd_jax``.  The full 6018-step
                   run is the timed stage; a 601-step run gives the displacement at
                   the canonical peak frame (it=600, t=19.94 ms) for the HH stage.
  4. neural        ``simulate_map_3comp_with_spike_times`` with the
                   ``run_neural_sim.py`` canonical S2 arguments, backend='auto'
                   (-> CUDA); called twice so the Numba-CUDA JIT compile can be
                   separated from the warm run.

Each stage runs in its own worker subprocess (CUDA_VISIBLE_DEVICES=0,
XLA_PYTHON_CLIENT_PREALLOCATE=false) while the orchestrator samples the worker
process tree's host RSS (psutil) and per-process GPU memory (nvidia-smi) and
verifies that no process of this run appears on any GPU other than physical
GPU 0.

Outputs
  <scratch>/stage_<name>.json          per-stage worker report
  <scratch>/timing_gpu.json            merged report (also copied to --timing-json)
  <scratch>/perf_table.tex             tabular body (also copied to --perf-tex)

Usage
  python scripts/time_canonical_pipeline.py --scratch-dir /path/to/scratch
        [--stages fullwave,arf_bioheat,shear,neural]
        [--timing-json data/halle_dacc/timing_gpu.json]
        [--perf-tex data/perf_table.tex]
"""
from __future__ import annotations

import os

# GPU 0 only; must be set before anything initialises CUDA / JAX.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import platform  # noqa: E402
import resource  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
CANON_DIR = REPO_ROOT / "data" / "halle_dacc"
CONFIG = REPO_ROOT / "validation_report" / "configs" / "halle_dacc.py"
CANON_FIRING_VOXELS = 3531   # S2, mechano gates from rest (default since b141c0a; peak_steady: 3590)
CANON_TOTAL_SPIKES = 12250   # (peak_steady: 13381)
STAGES = ("fullwave", "arf_bioheat", "shear", "neural")
GPU_INDEX = 0  # physical index; GPU 1 must not be touched


# =============================================================================
# helpers
# =============================================================================
def _ru_maxrss_gb(who: int) -> float:
    return resource.getrusage(who).ru_maxrss * 1024 / 1e9  # Linux: KiB


def _nvsmi(args: list[str]) -> str:
    return subprocess.run(["nvidia-smi", *args], capture_output=True, text=True,
                          check=False).stdout


def _gpu_uuids() -> dict[int, dict]:
    out = _nvsmi(["--query-gpu=index,uuid,name,memory.total,driver_version",
                  "--format=csv,noheader,nounits"])
    res = {}
    for line in out.strip().splitlines():
        idx, uuid, name, mem, drv = [s.strip() for s in line.split(",")]
        res[int(idx)] = dict(uuid=uuid, name=name, memory_total_mib=int(mem), driver=drv)
    return res


# =============================================================================
# worker stages (run inside the subprocess)
# =============================================================================
def _stage_fullwave(scratch: Path) -> dict:
    sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REPO_ROOT))
    t_import = time.perf_counter()
    import bowl_transcranial_500khz as btm  # noqa: PLC0415
    from fullwave.solver import launcher as fw_launcher  # noqa: PLC0415

    from ultraneuron.paths import halle_skull_nrrd  # noqa: PLC0415
    import_s = time.perf_counter() - t_import

    # Never write report figures to the shared paper/figures dir; skip the
    # post-save plotting (fig b, c, bc, mp4, ARF/fig d) -- not part of the compute.
    btm.DOCS_FIGURES_DIR = None
    btm.plot_acoustical_maps = lambda *a, **k: None

    class _StopAfterAcoustic(Exception):
        pass

    t_saved = {}

    def _stop(*a, **k):
        t_saved["t"] = time.perf_counter()
        raise _StopAfterAcoustic

    btm.plot_pressure_snapshots = _stop  # first call after "Saved acoustic outputs"

    solver = {}
    _orig_run = fw_launcher.Launcher.run

    def _timed_run(self, simulation_dir, **kw):
        solver["cuda_device_id_env"] = self.cuda_device_id
        solver["binary"] = str(self._path_fullwave_simulation_bin)
        t0 = time.perf_counter()
        try:
            return _orig_run(self, simulation_dir, **kw)
        finally:
            solver["solver_wall_s"] = time.perf_counter() - t0
            log = Path(simulation_dir) / "fw2_execution.log"
            if log.exists():
                txt = log.read_text(errors="replace").splitlines()
                solver["fw2_log_head"] = txt[:40]
                solver["fw2_log_tail"] = txt[-10:]

    fw_launcher.Launcher.run = _timed_run

    t0 = time.perf_counter()
    try:
        btm.main(
            nrrd_path=str(halle_skull_nrrd()),
            out_dir=str(scratch),
            target_name="ACC_left",
            scalp_site=None,
            domain_size_m=(130e-3, 80e-3, 80e-3),
            download=False,
            skip_acoustic=False,
            skip_shear=True,
            use_velocity=False,
            theta_burst=False,
            source_p0=0.10e6,
        )
    except _StopAfterAcoustic:
        pass
    t_end = t_saved.get("t", time.perf_counter())
    stage_wall = t_end - t0

    import numpy as np  # noqa: PLC0415
    gp = np.load(scratch / "grid_params.npz")
    pm = np.load(scratch / "propagation_map.npy", mmap_mode="r")
    return dict(
        stage_wall_s=stage_wall,
        timed_wall_s=stage_wall,
        import_s=import_s,
        solver_wall_s=solver.get("solver_wall_s"),
        solver=solver,
        hardware="GPU",
        hardware_detail=(f"Fullwave 2 CUDA binary, launcher CUDA_VISIBLE_DEVICES="
                         f"{solver.get('cuda_device_id_env')} (physical GPU {GPU_INDEX})"),
        grid=[int(gp["nx"]), int(gp["ny"]), int(gp["nz"])],
        propagation_map_shape=list(pm.shape),
        note=("stage = placement + NRRD load + slab extraction + MediumBuilder + "
              "Fullwave input write + GPU solver + sensor load/reshape/save; "
              "medium build is CPU (single process)"),
    )


def _stage_arf_bioheat(scratch: Path) -> dict:
    sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REPO_ROOT))
    import runpy  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    import rescale_propagation_map as rpm  # noqa: PLC0415
    import run_bioheat  # noqa: PLC0415
    sub = {}
    t_stage = time.perf_counter()

    t0 = time.perf_counter()
    rpm.main(scratch, 0.5e6)
    sub["rescale_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    argv = sys.argv
    try:
        sys.argv = ["recompute_arf.py", str(scratch)]
        runpy.run_path(str(HERE / "recompute_arf.py"), run_name="__main__")
    finally:
        sys.argv = argv
    sub["arf_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    try:
        sys.argv = ["run_bioheat.py", "--config", str(CONFIG), "--sim-dir", str(scratch)]
        run_bioheat.main()
    finally:
        sys.argv = argv
    sub["bioheat_s"] = time.perf_counter() - t0
    stage_wall = time.perf_counter() - t_stage

    # physics cross-checks vs canonical (read-only)
    prov = json.loads((scratch / "rescale_provenance.json").read_text())
    prov_c = json.loads((CANON_DIR / "rescale_provenance.json").read_text())
    tm = json.loads((scratch / "bioheat" / "thermal_metrics.json").read_text())
    tm_c = json.loads((CANON_DIR / "bioheat" / "thermal_metrics.json").read_text())
    b0 = np.load(scratch / "radiation_force.npy")
    b0_c = np.load(CANON_DIR / "radiation_force.npy")
    checks = dict(
        scale_factor=prov["scale_factor_applied"],
        scale_factor_canonical=prov_c["scale_factor_applied"],
        in_brain_peak_before_rescale_Pa=prov["observed_in_brain_peak_Pa_before_rescale"],
        in_brain_peak_before_rescale_Pa_canonical=prov_c["observed_in_brain_peak_Pa_before_rescale"],
        arf_max=float(b0.max()), arf_max_canonical=float(b0_c.max()),
        arf_max_abs_diff_rel=float(np.abs(b0 - b0_c).max() / b0_c.max()),
        dT_peak_K=tm["peak_dT_final_K"], dT_peak_K_canonical=tm_c["peak_dT_final_K"],
        dT_focus_K=tm["dT_focus_K"], dT_focus_K_canonical=tm_c["dT_focus_K"],
        burst_resolved=tm["burst_resolved"],
    )
    return dict(stage_wall_s=stage_wall, timed_wall_s=stage_wall, sub_s=sub,
                hardware="CPU",
                hardware_detail=("NumPy (single process): compute_radiation_force and "
                                 "fullwave.bioheat.pennes_solve operate on NumPy arrays; "
                                 "no CuPy/JAX path is used by recompute_arf.py or run_bioheat.py"),
                checks=checks)


def _stage_shear(scratch: Path) -> dict:
    sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REPO_ROOT))
    import importlib.util  # noqa: PLC0415

    import jax  # noqa: PLC0415
    import jax.monitoring  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    from shearwave import shear_fdtd_jax  # noqa: PLC0415
    from shearwave.solver import project_vector_field  # noqa: PLC0415

    compile_events: list[tuple[str, float]] = []

    def _listener(event, duration, **kw):
        if "compil" in event or "lower" in event or "trac" in event:
            compile_events.append((event, float(duration)))

    jax.monitoring.register_event_duration_secs_listener(_listener)
    devs = jax.devices()
    dev_info = [dict(id=d.id, platform=d.platform, kind=getattr(d, "device_kind", "")) for d in devs]

    t_stage = time.perf_counter()
    t0 = time.perf_counter()
    grid = np.load(scratch / "grid_params.npz", allow_pickle=True)
    dx2 = float(grid["dx"]) * int(grid["mod_x"])
    dy2 = float(grid["dy"]) * int(grid["mod_y"])
    dz2 = float(grid["dz"]) * int(grid["mod_z"])
    rho_ds = np.load(scratch / "rho_ds.npy")
    b0 = np.load(scratch / "radiation_force.npy")
    load_s = time.perf_counter() - t0

    # --- identical parameters to rerun_shear_fdtd.py ---
    e_young = 13.74e3; nu_poisson = 0.499999
    mu = e_young / (2.0 * (1.0 + nu_poisson))
    rho0 = float(np.median(rho_ds)); cs = math.sqrt(mu / rho0); eta_pa_s = 0.5
    cfl = 0.1
    dt_elastic = cfl * min(dx2, dy2, dz2) / (cs * math.sqrt(3))
    dx_min = min(dx2, dy2, dz2)
    dt_viscous = rho0 * dx_min**2 / (2.0 * 3 * eta_pa_s)
    dt_shear = min(dt_elastic, 0.5 * dt_viscous)
    spec = importlib.util.spec_from_file_location("_cfg", CONFIG)
    cfg = importlib.util.module_from_spec(spec); spec.loader.exec_module(cfg)
    seq = cfg.NEUROMOD_SEQUENCE
    t_pulse_on = float(seq.get("arf_push_window_s", seq.get("tone_burst_duration_s", 30e-3)))
    t_period = 200.0e-3
    n_steps = math.ceil(t_period / dt_shear)
    t = np.arange(n_steps, dtype=np.float64) * dt_shear
    env_t = np.where((t % t_period) <= t_pulse_on, 1.0, 0.0).astype(np.float32)
    canon_sp = np.load(CANON_DIR / "shear_params.npz")
    save_every = max(1, n_steps // 100)
    peak_it = 10 * save_every  # canonical peak-displacement frame index 10 -> it=600
    opts = {"vel_damp": 0.02, "eta": eta_pa_s, "verbose": False}

    def _mem_peak():
        try:
            return int(devs[0].memory_stats().get("peak_bytes_in_use", 0))
        except Exception:  # noqa: BLE001
            return None

    # one-off CPU Helmholtz projection (also done inside every shear_fdtd_jax call)
    t0 = time.perf_counter()
    z = np.zeros_like(b0, dtype=np.float32)
    project_vector_field(b0.astype(np.float32), z, z, dx2, dy2, dz2, 1e-5, 200)
    proj_s = time.perf_counter() - t0

    # (A) displacement at the canonical peak frame: u after peak_it+1 updates
    n_a = peak_it + 1
    compile_events.clear()
    t0 = time.perf_counter()
    u_a, _v_a = shear_fdtd_jax(bx=b0, by=None, bz=None, rho=rho0, mu=mu, dX=dx2, dY=dy2, dZ=dz2,
                               dT=dt_shear, n_steps=n_a, opts={**opts, "env_t": env_t[:n_a]})
    run_a_s = time.perf_counter() - t0
    comp_a = sum(d for e, d in compile_events if "backend_compile" in e)
    trace_a = sum(d for e, d in compile_events if "backend_compile" not in e)

    # (B) the full canonical 200 ms period (6018 steps), ~100 snapshots like canonical
    stride = next(s for s in range(save_every, 0, -1) if n_steps % s == 0)
    compile_events.clear()
    t0 = time.perf_counter()
    u_b, v_b, snaps = shear_fdtd_jax(bx=b0, by=None, bz=None, rho=rho0, mu=mu, dX=dx2, dY=dy2,
                                     dZ=dz2, dT=dt_shear, n_steps=n_steps,
                                     opts={**opts, "env_t": env_t}, snapshot_stride=stride)
    run_b_s = time.perf_counter() - t0
    comp_b = sum(d for e, d in compile_events if "backend_compile" in e)
    trace_b = sum(d for e, d in compile_events if "backend_compile" not in e)
    ev_b = sorted({e for e, _ in compile_events})
    gpu_peak = _mem_peak()

    t0 = time.perf_counter()
    np.save(scratch / "displacement.npy", u_a.astype(np.float32))
    np.save(scratch / "displacement_4d.npy", snaps.astype(np.float32))
    np.save(scratch / "displacement_4d_times.npy", (np.arange(1, snaps.shape[0] + 1) * stride - 1) * dt_shear)
    np.save(scratch / "velocity.npy", v_b)
    np.savez(scratch / "shear_params.npz", dt_shear=dt_shear, save_every=stride, n_steps=n_steps,
             mu=mu, rho0=rho0, cs=cs, eta=eta_pa_s, peak_it=peak_it)
    save_s = time.perf_counter() - t0
    stage_wall = time.perf_counter() - t_stage

    disp_c = np.load(CANON_DIR / "displacement.npy")
    timed = load_s + run_b_s + save_s
    return dict(
        stage_wall_s=stage_wall,
        timed_wall_s=timed,
        timed_compile_s=comp_b,
        timed_trace_s=trace_b,
        timed_ex_compile_s=timed - comp_b - trace_b,
        load_s=load_s, save_s=save_s,
        cpu_helmholtz_projection_s=proj_s,
        full_run=dict(n_steps=n_steps, snapshot_stride=stride, n_snapshots=int(snaps.shape[0]),
                      wall_s=run_b_s, backend_compile_s=comp_b, trace_lower_s=trace_b,
                      events=ev_b),
        peak_frame_run=dict(n_steps=n_a, wall_s=run_a_s, backend_compile_s=comp_a,
                            trace_lower_s=trace_a,
                            note="auxiliary 601-step run for the HH input (not in the table time)"),
        jax_devices=dev_info, jax_version=jax.__version__,
        jax_default_backend=jax.default_backend(),
        jax_peak_bytes_in_use=gpu_peak,
        hardware="GPU",
        hardware_detail=f"JAX/XLA on {dev_info}",
        params=dict(dt_shear=dt_shear, dt_shear_canonical=float(canon_sp["dt_shear"]),
                    n_steps=n_steps, n_steps_canonical=int(canon_sp["n_steps"]),
                    n_on=int(env_t.sum()), t_pulse_on=t_pulse_on, mu=mu, rho0=rho0),
        checks=dict(
            peak_ux_um=float(np.abs(u_a[..., 0]).max() * 1e6),
            peak_ux_um_canonical=float(np.abs(disp_c[..., 0]).max() * 1e6),
            disp_max_abs_diff_rel=float(np.abs(u_a - disp_c).max() / np.abs(disp_c).max()),
        ),
    )


def _stage_neural(scratch: Path) -> dict:
    sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REPO_ROOT))
    import numpy as np  # noqa: PLC0415

    from ultraneuron import simulate_map_3comp_with_spike_times  # noqa: PLC0415
    from ultraneuron.cuda_backend import resolve_backend  # noqa: PLC0415
    from run_neural_sim import (  # noqa: PLC0415
        _DEFAULTS,
        _build_neuron,
        _load_config,
        _strain_field_from_displacement,
    )

    t_stage = time.perf_counter()
    cfg = _load_config(str(CONFIG))
    seq = cfg.NEUROMOD_SEQUENCE
    duration_ms = float(seq.get("arf_push_window_s", seq.get("tone_burst_duration_s", 0.03))) * 1e3
    t0 = time.perf_counter()
    disp = np.load(scratch / "displacement.npy").astype(np.float32)
    c_map = np.load(scratch / "c_ds.npy").astype(np.float32)
    grid = np.load(scratch / "grid_params.npz")
    dx = float(grid["dx"]) * int(grid["mod_x"]); dy = float(grid["dy"]) * int(grid["mod_y"])
    dz = float(grid["dz"]) * int(grid["mod_z"])
    strain = _strain_field_from_displacement(disp, dx, dy, dz)
    strain[c_map > 1600.0] = 0.0
    prep_s = time.perf_counter() - t0
    backend = resolve_backend("auto")
    dev = {}
    if backend == "cuda":
        from numba import cuda  # noqa: PLC0415
        d = cuda.get_current_device()
        dev = dict(name=d.name.decode() if isinstance(d.name, bytes) else str(d.name),
                   id=int(d.id), n_visible=len(cuda.gpus),
                   compute_capability=list(d.compute_capability))
    kw = dict(K_A=_DEFAULTS["K_A"], coupling_factor=_DEFAULTS["coupling_factor"],
              dt=_DEFAULTS["dt_ms"], duration=duration_ms,
              tension_threshold=_DEFAULTS["tension_threshold"], integrator="rk4", backend=backend)
    runs = []
    res = None
    for label in ("cold", "warm"):
        neuron = _build_neuron(cfg, scenario="S2")
        t0 = time.perf_counter()
        res = simulate_map_3comp_with_spike_times(strain, neuron, **kw)
        runs.append(dict(label=label, wall_s=time.perf_counter() - t0,
                         n_fire=int(res["firing_map"].sum()),
                         n_spikes=int(res["spike_count_map"].sum())))
    t0 = time.perf_counter()
    np.savez_compressed(scratch / "neural_results.npz", firing_map=res["firing_map"],
                        spike_count_map=res["spike_count_map"],
                        T_peak_mNm=res["tension_3d"].astype(np.float32), backend=backend)
    save_s = time.perf_counter() - t0
    stage_wall = time.perf_counter() - t_stage
    canon = np.load(CANON_DIR / "neural_results.npz")
    fm_c = canon["firing_map"].astype(bool); fm = res["firing_map"].astype(bool)
    cold, warm = runs
    compile_s = max(cold["wall_s"] - warm["wall_s"], 0.0)
    timed = prep_s + cold["wall_s"] + save_s  # one canonical invocation, cold (incl. JIT)
    return dict(
        stage_wall_s=stage_wall,
        timed_wall_s=timed,
        timed_compile_s=compile_s,
        timed_ex_compile_s=timed - compile_s,
        prep_s=prep_s, save_s=save_s, runs=runs,
        backend_resolved=backend, cuda_device=dev,
        hardware="GPU" if backend == "cuda" else "CPU",
        hardware_detail=f"Numba-CUDA RK4 kernel, backend='auto' -> {backend!r}, device {dev}",
        params=dict(kw, scenario="S2", neuron="3-comp cortical Piezo1+TREK-1"),
        checks=dict(
            n_firing=warm["n_fire"], n_firing_canonical_npz=int(fm_c.sum()),
            n_firing_canonical_reported=CANON_FIRING_VOXELS,
            n_spikes=warm["n_spikes"], n_spikes_canonical=int(canon["spike_count_map"].sum()),
            match_reported=bool(warm["n_fire"] == CANON_FIRING_VOXELS),
            voxels_only_new=int((fm & ~fm_c).sum()), voxels_only_canonical=int((~fm & fm_c).sum()),
            peak_tension_mNm=float(res["tension_3d"].max()),
            peak_tension_mNm_canonical=float(canon["T_peak_mNm"].max()),
        ),
    )


def worker_main(stage: str, scratch: Path, out_json: Path) -> None:
    fn = {"fullwave": _stage_fullwave, "arf_bioheat": _stage_arf_bioheat,
          "shear": _stage_shear, "neural": _stage_neural}[stage]
    rep = fn(scratch)
    rep["worker_ru_maxrss_self_gb"] = _ru_maxrss_gb(resource.RUSAGE_SELF)
    rep["worker_ru_maxrss_children_gb"] = _ru_maxrss_gb(resource.RUSAGE_CHILDREN)
    rep["env"] = {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES",
                                                  "XLA_PYTHON_CLIENT_PREALLOCATE", "AMN_BACKEND")}
    out_json.write_text(json.dumps(rep, indent=1, default=str))


# =============================================================================
# orchestrator
# =============================================================================
class Monitor(threading.Thread):
    """Sample host RSS of a process tree and its GPU memory (nvidia-smi)."""

    def __init__(self, pid: int, gpus: dict[int, dict], period: float = 0.5):
        super().__init__(daemon=True)
        import psutil  # noqa: PLC0415
        self.ps = psutil
        self.root = psutil.Process(pid)
        self.period = period
        self.uuid_to_idx = {v["uuid"]: k for k, v in gpus.items()}
        self.stop_evt = threading.Event()
        self.peak_rss = 0
        self.peak_gpu_proc = {}           # gpu index -> peak MiB summed over our pids
        self.peak_gpu_total = {}          # gpu index -> peak memory.used (whole device)
        self.pids_seen_on_gpu = {}        # gpu index -> set(pid)
        self.proc_names = {}

    def _tree(self):
        try:
            procs = [self.root, *self.root.children(recursive=True)]
        except self.ps.NoSuchProcess:
            return []
        return procs

    def run(self):
        while not self.stop_evt.is_set():
            procs = self._tree()
            rss = 0; pids = set()
            for p in procs:
                try:
                    rss += p.memory_info().rss; pids.add(p.pid)
                    self.proc_names.setdefault(p.pid, p.name())
                except self.ps.Error:
                    pass
            self.peak_rss = max(self.peak_rss, rss)
            out = _nvsmi(["--query-compute-apps=pid,gpu_uuid,used_memory",
                          "--format=csv,noheader,nounits"])
            per = {}
            for line in out.strip().splitlines():
                parts = [s.strip() for s in line.split(",")]
                if len(parts) != 3 or not parts[0].isdigit():
                    continue
                pid, uuid, mem = int(parts[0]), parts[1], parts[2]
                if pid in pids:
                    gi = self.uuid_to_idx.get(uuid, -1)
                    per[gi] = per.get(gi, 0) + (int(mem) if mem.isdigit() else 0)
                    self.pids_seen_on_gpu.setdefault(gi, set()).add(pid)
            for gi, m in per.items():
                self.peak_gpu_proc[gi] = max(self.peak_gpu_proc.get(gi, 0), m)
            tot = _nvsmi(["-i", str(GPU_INDEX), "--query-gpu=memory.used",
                          "--format=csv,noheader,nounits"]).strip()
            if tot.isdigit():
                self.peak_gpu_total[GPU_INDEX] = max(self.peak_gpu_total.get(GPU_INDEX, 0), int(tot))
            self.stop_evt.wait(self.period)


def run_stage(stage: str, scratch: Path, gpus: dict) -> dict:
    out_json = scratch / f"stage_{stage}.json"
    log = scratch / f"stage_{stage}.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"; env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env.pop("AMN_BACKEND", None)
    base_total = _nvsmi(["-i", str(GPU_INDEX), "--query-gpu=memory.used",
                         "--format=csv,noheader,nounits"]).strip()
    print(f"[timing] stage {stage} -> {log}", flush=True)
    t0 = time.perf_counter()
    with log.open("w") as fh:
        p = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "--worker", stage,
                              "--scratch-dir", str(scratch), "--worker-json", str(out_json)],
                             stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=str(REPO_ROOT))
        mon = Monitor(p.pid, gpus); mon.start()
        rc = p.wait()
        mon.stop_evt.set(); mon.join()
    wall = time.perf_counter() - t0
    if rc != 0:
        raise RuntimeError(f"stage {stage} failed (rc={rc}); see {log}")
    rep = json.loads(out_json.read_text())
    other = {gi: sorted(p_) for gi, p_ in mon.pids_seen_on_gpu.items() if gi != GPU_INDEX}
    rep["monitor"] = dict(
        subprocess_wall_s=wall,
        peak_host_rss_tree_gb=mon.peak_rss / 1e9,
        peak_gpu_mem_proc_mib=mon.peak_gpu_proc.get(GPU_INDEX, 0),
        peak_gpu_mem_device_mib=mon.peak_gpu_total.get(GPU_INDEX, 0),
        gpu_baseline_device_mib=int(base_total) if base_total.isdigit() else None,
        pids_on_gpu0=sorted(mon.pids_seen_on_gpu.get(GPU_INDEX, set())),
        proc_names_on_gpu0=sorted({mon.proc_names.get(x, "?") for x in mon.pids_seen_on_gpu.get(GPU_INDEX, set())}),
        pids_on_other_gpus=other,
        sample_period_s=mon.period,
    )
    rep["peak_host_rss_gb"] = max(mon.peak_rss / 1e9, rep["worker_ru_maxrss_self_gb"],
                                  rep["worker_ru_maxrss_children_gb"])
    rep["peak_gpu_mem_gb"] = mon.peak_gpu_proc.get(GPU_INDEX, 0) * 1.048576e6 / 1e9
    (scratch / f"stage_{stage}.json").write_text(json.dumps(rep, indent=1, default=str))
    print(f"[timing]   {stage}: {wall:.1f} s subprocess, RSS {rep['peak_host_rss_gb']:.2f} GB, "
          f"GPU {rep['peak_gpu_mem_gb']:.2f} GB, other-GPU pids {other}", flush=True)
    return rep


def _fmt_t(s: float) -> str:
    return f"{s:.0f} s" if s < 90 else f"{s / 60:.1f} min"


def _fmt_gb(g: float) -> str:
    return f"{g:.1f}" if g >= 0.95 else f"{g:.2f}"


def render_tex(stages: dict, total: dict) -> str:
    def mem(r):
        s = f"{_fmt_gb(r['peak_host_rss_gb'])} GB RAM"
        if r.get("peak_gpu_mem_gb", 0) > 0.01:
            s += f", {_fmt_gb(r['peak_gpu_mem_gb'])} GB GPU"
        return s

    def wt(r):
        s = _fmt_t(r["timed_wall_s"])
        c = r.get("timed_compile_s", 0.0) + r.get("timed_trace_s", 0.0)
        if c >= 0.5:
            s += f" (incl.\\ {c:.0f} s JIT)"
        return s

    hw = {
        "fullwave": "GPU (CUDA); property maps on CPU",
        "arf_bioheat": "CPU (NumPy)",
        "shear": "GPU (JAX)",
        "neural": "GPU (Numba-CUDA)" if stages.get("neural", {}).get("backend_resolved") == "cuda" else "CPU (Numba)",
    }
    names = {
        "fullwave": "Fullwave~2 (acoustic)",
        "arf_bioheat": "Radiation force and Pennes bioheat",
        "shear": "Shear-FDTD (Kelvin--Voigt)",
        "neural": "Hodgkin--Huxley neurons",
    }
    lines = ["% Auto-generated by scripts/time_canonical_pipeline.py -- do not edit by hand.",
             f"% {total['timestamp']}  host {total['host']}  GPU {total['gpu_name']} (physical GPU {GPU_INDEX} only)",
             "% Memory = peak host RSS of the stage process tree / peak GPU memory of its processes (nvidia-smi).",
             "% Wall-time = one canonical invocation incl. data load/save and one-off JIT compile (shown in parentheses).",
             "Stage & Hardware & Memory & Wall-time \\\\", "\\mr"]
    for k in STAGES:
        if k in stages:
            r = stages[k]
            lines.append(f"{names[k]} & {hw[k]} & {mem(r)} & {wt(r)} \\\\")
    lines.append("\\mr")
    lines.append(f"Total (peak RAM $\\sim {total['peak_host_rss_gb']:.0f}$~GB) & & & "
                 f"$\\sim {total['wall_s'] / 60:.0f}$ min \\\\")
    return "\n".join(lines) + "\n"


def orchestrate(args) -> None:
    scratch = Path(args.scratch_dir).resolve()
    canon = CANON_DIR.resolve()
    if scratch == canon or canon in scratch.parents or scratch in canon.parents:
        raise SystemExit(f"refusing to run in/over the canonical directory: {scratch}")
    scratch.mkdir(parents=True, exist_ok=True)
    gpus = _gpu_uuids()
    stages_req = [s.strip() for s in args.stages.split(",") if s.strip()]
    stages: dict[str, dict] = {}
    for s in STAGES:
        f = scratch / f"stage_{s}.json"
        if s not in stages_req and f.exists():  # reuse an earlier timed stage
            rep = json.loads(f.read_text())
            if "monitor" in rep:
                stages[s] = rep
    for s in STAGES:
        if s in stages_req:
            stages[s] = run_stage(s, scratch, gpus)

    try:
        import psutil  # noqa: PLC0415
        n_cpu = psutil.cpu_count(logical=True)
    except Exception:  # noqa: BLE001
        n_cpu = os.cpu_count()
    total = dict(
        wall_s=sum(r["timed_wall_s"] for r in stages.values()),
        wall_ex_compile_s=sum(r.get("timed_ex_compile_s", r["timed_wall_s"]) for r in stages.values()),
        peak_host_rss_gb=max(r["peak_host_rss_gb"] for r in stages.values()),
        peak_gpu_mem_gb=max(r.get("peak_gpu_mem_gb", 0.0) for r in stages.values()),
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        host=platform.node(), n_cpu_logical=n_cpu,
        gpu_name=gpus[GPU_INDEX]["name"], gpu_uuid=gpus[GPU_INDEX]["uuid"],
        driver=gpus[GPU_INDEX]["driver"],
        any_process_on_other_gpu=any(r["monitor"]["pids_on_other_gpus"] for r in stages.values()),
    )
    nc = stages.get("neural", {}).get("checks", {})
    report = dict(
        description=("Per-stage timing of the canonical Halle/dACC Yaakub pipeline, re-run on "
                     "physical GPU 0 into a scratch directory (canonical outputs untouched)."),
        scratch_dir=str(scratch), config=str(CONFIG.relative_to(REPO_ROOT)),
        canonical_dir=str(CANON_DIR.relative_to(REPO_ROOT)),
        total=total, stages=stages,
        firing_sanity=dict(n_firing=nc.get("n_firing"), canonical=CANON_FIRING_VOXELS,
                           match=nc.get("match_reported")),
        definitions=dict(
            timed_wall_s="one canonical invocation of the stage incl. data load/save and one-off JIT",
            timed_compile_s="JIT compile inside timed_wall_s (JAX: XLA backend_compile events; "
                            "Numba-CUDA: cold minus warm call)",
            peak_host_rss_gb="max(sampled process-tree RSS, worker ru_maxrss self/children)",
            peak_gpu_mem_gb="peak nvidia-smi used_memory summed over the stage's processes on GPU 0",
        ),
    )
    out = scratch / "timing_gpu.json"
    out.write_text(json.dumps(report, indent=1, default=str))
    tex = render_tex(stages, total)
    (scratch / "perf_table.tex").write_text(tex)
    if args.timing_json:
        Path(args.timing_json).write_text(out.read_text())
    if args.perf_tex:
        Path(args.perf_tex).write_text(tex)
    print(tex)
    print(f"[timing] wrote {out}" + (f", {args.timing_json}" if args.timing_json else "")
          + (f", {args.perf_tex}" if args.perf_tex else ""), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scratch-dir", default=str(REPO_ROOT / "data" / "halle_dacc_timing_gpu"))
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--timing-json", default=str(CANON_DIR / "timing_gpu.json"))
    ap.add_argument("--perf-tex", default=str(REPO_ROOT / "data" / "perf_table.tex"))
    ap.add_argument("--worker", choices=STAGES, help=argparse.SUPPRESS)
    ap.add_argument("--worker-json", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.worker:
        worker_main(args.worker, Path(args.scratch_dir).resolve(), Path(args.worker_json))
    else:
        orchestrate(args)


if __name__ == "__main__":
    main()
