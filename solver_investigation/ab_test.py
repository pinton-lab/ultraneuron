"""A/B: Rush-Larsen vs RK4 reference (6372 firing / 23260 spikes @ dt=0.001)."""
from __future__ import annotations

import time

import numpy as np
import rl_prototype as rl
from _common import build_neuron, eligible_tensions

DURATION = 20.0
REF_FIRE, REF_SPK = 6372, 23260


def main():
    import sys
    dts = [float(x) for x in sys.argv[1:]] or [0.001, 0.0025, 0.005, 0.01, 0.02, 0.05]
    T = eligible_tensions()
    neuron = build_neuron()
    print(f"Rush-Larsen (exp-Euler), {T.size} voxels, {DURATION} ms")
    print(f"reference RK4 dt=0.001: {REF_FIRE} firing / {REF_SPK} spikes\n")
    print(f"{'dt (ms)':>10} {'firing':>8} {'spikes':>8} {'wall(s)':>9} "
          f"{'d_fire%':>8} {'d_spk%':>8}")
    for dt in dts:
        t0 = time.perf_counter()
        sc = rl.run(neuron, T, dt, DURATION)
        el = time.perf_counter() - t0
        fire = int((sc > 0).sum()); spk = int(sc.sum())
        df = 100.0 * (fire - REF_FIRE) / REF_FIRE
        ds = 100.0 * (spk - REF_SPK) / REF_SPK
        print(f"{dt:>10.5f} {fire:>8d} {spk:>8d} {el:>9.2f} "
              f"{df:>+8.2f} {ds:>+8.2f}", flush=True)


if __name__ == "__main__":
    main()
