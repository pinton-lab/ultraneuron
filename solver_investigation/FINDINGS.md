# ODE-integrator investigation: per-voxel 3-compartment mechano-HH map

**Question.** Can a different integrator let the per-voxel HH map use a
coarser `dt` (or run faster) without changing the converged spike-count
results (halle_dacc: **6372 firing voxels / 23260 total spikes**, 20 ms
window, dt=0.001 ms)?

**Short answer.** Yes for *speed at a sub-1 % tolerance*; no for *bit-exact
reproduction at coarser dt*. A **Rush–Larsen / exponential-Euler** solver is
L-stable where RK4 catastrophically collapses, runs **~3× faster than RK4 at
the same dt=0.001** (20 s vs 62 s, numba/72-core), reproduces the headline
numbers to **−0.20 % firing / −0.95 % spikes**, and degrades *smoothly* at
coarse dt (up to ~19× faster at dt=0.005 for −1.4 %/−6 %). Bit-exact 6372/23260
is **not** reproducible faster than RK4, because the binding constraint is
temporal *resolution* of the Na⁺ upstroke, not integration order or stability.

All numbers below: halle_dacc, S2 (Piezo1+TREK1), 124 020 above-threshold
voxels, 20 ms window, numba backend, 72-core CPU. Harness in this directory
(`_common.py`, `baseline_rk4.py`, `rl_prototype.py`, `rl2_prototype.py`,
`ab_test.py`). Nothing in the committed pipeline was modified.

---

## Why RK4 has a hard floor at dt=0.001 — and why it is a *resolution*, not a *stability*, floor

The stiffness is dominated by the **AIS compartment**. Its axial coupling
conductance to the soma is `g_soma_ais · (A_soma/A_ais) = 50 · 20 =
1000 mS/cm²`, on top of `g_Na = 2000 mS/cm²`. During the upstroke the AIS
membrane time constant `C/G` drops to **≲ 0.001 ms**. RK4 (explicit) needs
`dt` well inside that to track the regenerative upstroke; at dt=0.0025 it
silently *under-integrates* the Na⁺ feedback and the soma never crosses
threshold in most voxels — spike totals collapse from 23 260 → 37 (not a
blow-up; a missed-crossing). This matches the documented cliff.

Crucially, the stiff term (`1000 + g_Na·m³h`) is **linear in V** — a
conductance. That is exactly what an exponential integrator absorbs exactly,
which is why Rush–Larsen helps here.

```
RK4 (numba)        dt(ms)   firing   spikes    wall(s)
                   0.00100    6372    23260      62      <- reference plateau
                   0.00250      37       37      18      <- collapse (resolution)
                   0.00500     —        —         —      (fastmath div-by-0 in kernel)
```

---

## Candidate 1 — Rush–Larsen / exponential-Euler  ★ recommended

Per step, with all rate coefficients frozen at the current state: update HH
gates `m,h,n` and mechano gates `w` by their exact exponential solution
`x ← x∞ + (x−x∞)e^{−dt/τ}`, then update each compartment voltage by the exact
exponential solution of the linearised membrane equation
`C dV/dt = −G_tot(V−V∞)`, where `G_tot` collects every conductance
multiplying V (ionic + mechano + axial) and `V∞ = B/G_tot`. Gates-then-voltage
ordering captures the regenerative upstroke within a step; axial neighbours are
frozen (Gauss–Jacobi), but the stiff diagonal is taken exactly ⇒ **L-stable at
any dt**. Physics is otherwise identical to `_derivs_3comp_scalar` (pure method
swap; verified per-cell — see below).

```
Rush-Larsen (RL1)  dt(ms)   firing   spikes    wall(s)  dFire%   dSpk%   speedup*
                   0.00025    6369    23202      55      -0.05   -0.25    1.1x
                   0.00050    6364    23154      29      -0.13   -0.46    2.1x
                   0.00100    6359    23040      20      -0.20   -0.95    3.1x   <-- same dt as RK4
                   0.00250    6335    22683     6.3      -0.58   -2.48    9.8x
                   0.00500    6285    21875     3.2      -1.37   -5.95     19x
                   0.01000    6195    21442     1.6      -2.78   -7.82     39x
                   0.02000    5981    18205     0.9      -6.14  -21.73     69x
                   0.05000    5578     9119    0.35     -12.46  -60.80    177x
* speedup vs RK4 @ dt=0.001 (62 s).
```

- **Converges to the same true solution** as RK4 (6369/23202 at dt=0.00025 →
  6372/23260), confirming a correct method — it is first-order, so it
  approaches the plateau from below with visible O(dt) bias in the spike total.
- **No cliff.** Degradation is smooth and monotone — the opposite of RK4's
  collapse. This is the key qualitative win: results stay *interpretable* at
  any dt.
- **Per-step cost** is ~3× lower than RK4 (one coefficient evaluation + ~16
  `exp()` vs RK4's four stages ≈ 72 `exp()`), so it is faster *even at equal
  dt*.

**Per-cell faithfulness (single voxel, dt=0.001):**
```
T=1.0 mN/m  RK4: 0 spikes      | RL: 0      (sub-threshold)
T=3.0 mN/m  RK4: 5 spikes lat 2.177 ms | RL: 5
T=6.5 mN/m  RK4: 5 spikes lat 2.255 ms | RL: 5  (focal voxel)
```
The <1 % batch discrepancy comes entirely from voxels sitting *on* the firing
threshold, where ±1 spike/voxel accumulates — not from any systematic shape
error.

## Candidate 1b — second-order (midpoint) Rush–Larsen  RL2

Midpoint predictor (half-step GRL1 → state_mid; full step with coefficients
evaluated at state_mid). 2nd-order; also unlags the axial coupling. Cost ≈ 2
coefficient evals/step ≈ RK4 at equal dt.

```
RL2 (midpoint)     dt(ms)   firing   spikes    wall(s)  dFire%   dSpk%   speedup*
                   0.00100    6371    23224      71      -0.02   -0.15   0.9x  (firing off by 1!)
                   0.00250    6363    23110      26      -0.14   -0.64   2.4x
                   0.00500    6351    22900      14      -0.33   -1.55   4.4x  <-- best ~1% point
                   0.01000    6317    22316     7.1      -0.86   -4.06   8.7x
                   0.02000    6228    21580     3.5      -2.26   -7.22    18x
```

RL2 buys *accuracy*, not speed, at fine dt (no faster than RK4 at 0.001). Its
value is the **coarse-dt accuracy band**: dt=0.005 gives −0.33 %/−1.55 % at
4.4×, the best accuracy-per-wall-clock in the ~1 % regime.

## Candidate 2 — adaptive step-size (RK45 / CVODE-BDF) — not recommended for the batch

Ideal *per neuron* (big steps in the quiescent stretch, tiny only during the
~1 ms spike). But the ~6300 firing voxels spike at **staggered latencies
spread across the 20 ms window**, so at essentially every instant *some* voxel
is mid-upstroke. A batched shared-step scheme (one step, max error across the
batch) is therefore throttled to the worst voxel at all times and never lifts
off the minimum step — no batch-level gain, while per-voxel-divergent steps
would shatter the SIMD/GPU lockstep (hard constraint). **Useful only for a
single-cell validation path, not the map.**

## Candidate 3 — higher-order explicit (RK8, etc.) — ruled out

A sharp upstroke needs temporal resolution regardless of order, and higher
order does nothing for the L-stability problem. Confirmed marginal: the gain
in this regime comes from making the *linear stiff term exact* (exponential),
not from more stages.

---

## Backends

| backend | RK4 dt=0.001 | note |
|---|---|---|
| numba (72-core) | **62 s** → 6372/23260 | reference |
| cuda (1 GPU) | 76 s → 6372/23260 (incl. JIT) | this box is CPU-strong; numba wins |

RL was prototyped and measured on **numba**. Porting to CUDA is mechanical
(uniform step, no control-flow divergence, identical per-step arithmetic
reduction), so the ~3×-per-step reduction is expected to carry over — but this
was **not measured**; treat the CUDA-RL speedup as projected, not confirmed.

---

## Recommendation

**Implement Rush–Larsen (RL1) behind a flag** (e.g. `--integrator rl`,
default `rk4`), keeping RK4 as the reference so every committed result stays
bit-reproducible.

- **Fast canonical setting:** RL1 at **dt=0.001** — 3.1× faster than RK4,
  L-stable, reproduces headline numbers to −0.20 % firing / −0.95 % spikes.
  Drop-in for parameter sweeps and previews where ~1 % is acceptable.
- **Coarse exploratory sweeps:** RL1 or RL2 at **dt=0.005** — 4–19× faster,
  documented few-% negative bias (RL2 tighter: −0.33 %/−1.55 %).
- **Do not** chase bit-exact 6372/23260 with a solver swap: that is a
  temporal-resolution requirement on the AIS upstroke, independent of method.
  RK4 @ dt=0.001 remains the canonical reference for published figures.
- **Do not** pursue adaptive stepping for the batched map; reserve it for
  single-cell validation only.

A 5-line caveat worth honoring if RL is adopted: report results as
"RL @ dt, ≈X % below the RK4 reference," never as reproducing the reference
exactly — the bias is small but systematic.
