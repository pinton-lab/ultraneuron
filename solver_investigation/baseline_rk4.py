"""Reproduce the RK4 dt-convergence table on halle_dacc (20 ms window)."""
from __future__ import annotations

import time

import numpy as np
from _common import build_neuron, eligible_tensions

DURATION = 20.0  # ms

def run(dt, backend="numba"):
    T = eligible_tensions()
    neuron = build_neuron()
    n_steps = int(DURATION / dt)
    I_arr = np.array([0.0, 0.0, 0.0])
    t0 = time.perf_counter()
    sc = neuron._simulate_batch(
        T, dt, n_steps, I_arr, primary_compartment=1, backend=backend,
    )
    el = time.perf_counter() - t0
    return int((sc > 0).sum()), int(sc.sum()), el


if __name__ == "__main__":
    import sys
    backend = sys.argv[1] if len(sys.argv) > 1 else "numba"
    dts = [float(x) for x in sys.argv[2:]] or [0.01, 0.005, 0.0025, 0.001, 0.0005]
    print(f"backend={backend}  duration={DURATION} ms")
    print(f"{'dt (ms)':>10} {'firing':>8} {'spikes':>8} {'wall (s)':>10}")
    for dt in dts:
        fire, spk, el = run(dt, backend)
        print(f"{dt:>10.5f} {fire:>8d} {spk:>8d} {el:>10.2f}", flush=True)
