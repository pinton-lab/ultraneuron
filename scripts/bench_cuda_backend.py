"""CUDA vs Numba benchmark for the 3-compartment mechanism-sweep workload.

Replicates the per-scenario voxel count and timestep budget of
``run_halle_mechanism_sweep.py`` (5000 voxels × 30000 steps at
dt = 0.001 ms, Piezo1 + TREK-1, cortical-pyramidal 3-comp).

Usage:
    python -m scripts.bench_cuda_backend
"""
from __future__ import annotations

import time

import numpy as np

from ultraneuron import MulticompartmentMechanoHHNeuron
from ultraneuron.channel_library import get_channel
from ultraneuron.cuda_backend import is_cuda_available


def _build_neuron() -> MulticompartmentMechanoHHNeuron:
    return MulticompartmentMechanoHHNeuron.cortical_pyramidal_3comp(
        mechano_channels=[get_channel("Piezo1"), get_channel("TREK-1")],
    )


def _time_one(N: int, n_steps: int, backend: str) -> tuple[float, np.ndarray]:
    neuron = _build_neuron()
    rng = np.random.default_rng(42)
    tensions = rng.uniform(0.5, 4.0, size=N)
    I_arr = np.array([0.0, 6.0, 0.0])
    # Warmup (Numba JIT / CUDA compile + first launch)
    neuron._simulate_batch(
        tensions[: min(32, N)], dt=0.001, n_steps=500,
        I_stim_per_compartment=I_arr, backend=backend,
    )
    t0 = time.perf_counter()
    sc = neuron._simulate_batch(
        tensions, dt=0.001, n_steps=n_steps,
        I_stim_per_compartment=I_arr, backend=backend,
    )
    return time.perf_counter() - t0, sc


def main() -> None:
    n_steps = 30000  # 30 ms @ dt=0.001 ms
    sizes = [1000, 5000, 20000]
    print(
        f"{'N':>7} {'Numba (s)':>10} {'CUDA (s)':>10} "
        f"{'speedup':>8} {'M vox·step/s (CUDA)':>22} {'match':>6}",
    )
    cuda_ok = is_cuda_available()
    if not cuda_ok:
        print("CUDA unavailable — skipping GPU rows.")
    for N in sizes:
        t_cpu, sc_cpu = _time_one(N, n_steps, "numba")
        if cuda_ok:
            t_gpu, sc_gpu = _time_one(N, n_steps, "cuda")
            speedup = t_cpu / t_gpu
            rate = N * n_steps / t_gpu / 1e6
            match = bool(np.array_equal(sc_cpu, sc_gpu))
            print(
                f"{N:>7d} {t_cpu:>10.2f} {t_gpu:>10.2f} "
                f"{speedup:>8.1f} {rate:>22.1f} {str(match):>6}",
            )
        else:
            print(f"{N:>7d} {t_cpu:>10.2f}  (no CUDA)")


if __name__ == "__main__":
    main()
