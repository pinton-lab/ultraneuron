import time, sys
import numpy as np
import rl2_prototype as rl2
from _common import build_neuron, eligible_tensions
DUR=20.0; RF,RS=6372,23260
dts=[float(x) for x in sys.argv[1:]] or [0.001,0.0025,0.005,0.01,0.02]
T=eligible_tensions(); nrn=build_neuron()
print(f"RL2 midpoint, {T.size} voxels, ref {RF}/{RS}")
print(f"{'dt':>9}{'firing':>8}{'spikes':>8}{'wall':>8}{'dF%':>8}{'dS%':>8}")
for dt in dts:
    t0=time.perf_counter(); sc=rl2.run(nrn,T,dt,DUR); el=time.perf_counter()-t0
    f=int((sc>0).sum()); s=int(sc.sum())
    print(f"{dt:>9.5f}{f:>8d}{s:>8d}{el:>8.2f}{100*(f-RF)/RF:>+8.2f}{100*(s-RS)/RS:>+8.2f}",flush=True)
