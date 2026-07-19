"""Single-cell: compare RL1 spike count & first-spike latency vs the
reference neuron.simulate() (RK4 internal) at a few tensions."""
import numpy as np
from _common import build_neuron
import rl_prototype as rl

nrn = build_neuron()
dt, dur = 0.001, 20.0
for Tval in [1.0, 3.0, 6.5]:
    # reference scalar simulate (RK4) at constant tension
    r = nrn.simulate(duration=dur, dt=dt, tension_trace=Tval, I_stim=0.0)
    ref_n = int(r["spike_count_per_compartment"][1])
    st = r["spike_times_per_compartment"][1]
    ref_lat = float(st[0]) if len(st) else np.nan
    # RL single voxel (pass length-1 tension array)
    sc = rl.run(nrn, np.array([Tval]), dt, dur)
    print(f"T={Tval:>4} mN/m  ref(RK4): {ref_n:>3} spikes lat={ref_lat:6.3f} ms"
          f"   |  RL: {int(sc[0]):>3} spikes")
