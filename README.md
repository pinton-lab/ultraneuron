# ultraneuron

[![tests](https://github.com/pinton-lab/ultraneuron/actions/workflows/tests.yml/badge.svg)](https://github.com/pinton-lab/ultraneuron/actions/workflows/tests.yml)

An open-source framework for predicting ultrasound neuromodulation: from a transcranial acoustic field to per-voxel neural firing maps registered to anatomy.

The pipeline couples heterogeneous nonlinear full-wave acoustic propagation, viscoelastic shear-wave displacement, Pennes bioheat thermal diffusion, a bilayer-mechanics conversion from tissue strain to membrane tension, and a multi-compartment Hodgkin–Huxley neuron carrying interchangeable mechanism modules:

- direct strain-driven mechanosensitive-channel gating (Piezo1, TRAAK, TREK-1, TREK-2);
- Plaksin–Krasovitski intramembrane cavitation (NICE / SONIC reduction);
- Piezo1 inactivation with intracellular Ca²⁺ pool and SK counter-current;
- TRPV1 / TRPV4 thermosensors coupled to the bioheat field;
- an astrocytic TRPA1 → glutamate → NMDA relay and a mechanosensitive synaptic / voltage-gated transmission pathway, each exposed as a closed-form per-case excitation scalar.

As a falsifiable cross-check it also includes the Heimburg–Jackson electromechanical **soliton** as an alternative (non-Hodgkin–Huxley) mechanism, scored per case as a closed-form excitability scalar; because biological membranes melt below body temperature, focal heating *reduces* its excitability — the opposite sign to the thermal pathways, and the cleanest experimental discriminator.

Every numerical parameter is classified by source (literature-anchored, calibrated, assumed, or derived) and the framework ships with a ±25 % sensitivity sweep that identifies the dominant uncertainties.

The accompanying manuscript (under review at *Journal of Neural Engineering*) is hosted on arXiv (link TBD).  This repository contains the code, the run configurations and the result summaries. The scripts in `scripts/` regenerate the manuscript's figures and tables locally (into `figures/` and `data/*.tex`), but those outputs and the manuscript itself are kept separate from the code and are not tracked here.

## Install

Install with [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv pip install -e .         # core (numpy only)
uv pip install -e .[fast]   # + numba CPU JIT (recommended for production)
uv pip install -e .[gpu]    # + numba.cuda backend (requires NVIDIA driver + CUDA toolkit)
uv pip install -e .[docs]   # + matplotlib / scipy for figure regeneration
uv pip install -e .[dev]    # everything + pytest / ruff
```

Or with plain `pip`: replace `uv pip` with `pip` in any of the above.

The CUDA backend is opt-in at the call site via a `backend="cuda"` kwarg on `simulate_map`, `simulate_map_3comp`, and `MulticompartmentMechanoHHNeuron._simulate_batch`, or via the `AMN_BACKEND=cuda` environment variable in the bundled driver scripts. The default is `"numba"` (CPU); it falls back to `"numba"` with a warning if CUDA is unavailable.

## Tests

```bash
pytest tests/
```

259 tests cover each pathway (Piezo1/K2P/TRP channels, NICE leaflet ODE, Ca pool + SK, three-compartment AIS-first initiation, Q₁₀ scaling, the Heimburg–Jackson soliton excitability and its falsifiable temperature sign, source-classification table integrity, CUDA-vs-Numba backend equivalence).

## Quick start

```python
import numpy as np
from ultraneuron import (
    MulticompartmentMechanoHHNeuron, PIEZO1, TREK1,
    simulate_map_3comp,
)

# Displacement field on a Cartesian grid (nx, ny, nz, 3)
u = np.load("displacement.npy")  # from any upstream FDTD solver

# Strain → membrane tension → per-voxel firing
neuron = MulticompartmentMechanoHHNeuron.cortical_pyramidal_3comp(
    mechano_channels=[PIEZO1, TREK1], temperature=37.0,
)
result = simulate_map_3comp(
    strain_eq=strain_field, neuron=neuron,
    dt=0.001, duration=30.0,
    tension_threshold=0.5,
    K_A=0.25, coupling_factor=1000.0,
    backend="cuda",  # or "numba" for CPU
)
firing_map = result["firing_map"]
spike_count_map = result["spike_count_map"]
```

The framework treats upstream wave-propagation, elastic, and bioheat solvers as black-box producers of displacement / pressure / temperature fields on a Cartesian grid. Any compatible solver output can be substituted via a thin loader.

## Reproducing the manuscript

The canonical demonstration is a theta-burst sonication through the Halle micro-CT human-skull specimen, auto-pick placement aimed at the left dorsal anterior cingulate cortex (one of the two Yaakub 2023 deep-cortical targets). The pipeline runs in stages, each producing the inputs for the next:

| Stage | Script | Output |
|---|---|---|
| 1. Acoustic FDTD | `scripts/run_halle_dacc_fdtd.py` (wraps `bowl_transcranial_500khz.py` for the dACC aim) | `data/halle_dacc/propagation_map.npy`, `radiation_force.npy`, plus the shear-FDTD outputs (`displacement_4d.npy`, `snapshots_*.npy`) |
| 1a. Surface-drive calibration (optional) | `scripts/calibrate_ctx500_surface_p0.py` | Reports free-field bowl gain and the calibrated p_0 for any free-field focal target |
| 1b. In-brain rescale | `scripts/rescale_propagation_map.py data/halle_dacc` | Linearly rescales the FDTD pressure history so the in-brain peak matches the paper's reported transcranial PNP (0.5 MPa) |
| 2. ARF recompute | `scripts/recompute_arf.py data/halle_dacc` | Re-derives `radiation_force.npy` from the (rescaled) pressure history with the corrected brain mask |
| 3. Shear-FDTD re-run | `scripts/rerun_shear_fdtd.py data/halle_dacc` | Re-derives the displacement field from the rescaled ARF |
| 4. Pennes bioheat | `scripts/run_bioheat.py --config validation_report/configs/halle_dacc.py` | `data/halle_dacc/bioheat/delta_T_final.npy` and per-voxel ΔT history |
| 5. Per-voxel neural simulation | `scripts/run_neural_sim.py --config validation_report/configs/halle_dacc.py` | `data/halle_dacc/neural_results.npz` |
| 6. Validation report | `validation_report/report_builder.py --config validation_report/configs/halle_dacc.py` | `data/halle_dacc/validation_report/validation_report_halle_dacc.pdf` and the constituent figures used by the manuscript |
| 7. Mechanism sweep | `HALLE_SIM_NAME=halle_dacc scripts/run_halle_mechanism_sweep.py` | `data/halle_dacc_mechanism_sweep_results.npz` and `figures/fig8_halle_dacc_mechanism_comparison.pdf` |
| 8. Fig 8 timepoints panel | `scripts/render_fig8_timepoints.py` | Replaces Fig 8 with the 5×5 timepoints grid (canonical version in the manuscript) |
| 9. Sensitivity sweep | `HALLE_SIM_NAME=halle_dacc scripts/run_sensitivity.py` | `data/sensitivity_results_halle_dacc.npz` and `data/sensitivity_table_halle_dacc.tex` |
| 10. Manuscript figure composites | `scripts/build_fig{3,4,5,6,7}.py` | The composite PDFs/PNGs in `figures/` |

The original F3 → left-caudate target is still reproducible from `validation_report/configs/halle_caudate.py` (set `HALLE_SIM_NAME=halle_caudate` for stages 7 and 9, or omit the env var since `halle_caudate` is the default for backwards compatibility).
The manuscript, its LaTeX sources and its rendered figures and tables are not tracked in this repository.

### Data

The simulation outputs (`*.npy`, `*.npz`, `*.dat`, ~7 GB) are deliberately not tracked in this repository. Users regenerate them by running the stages above against the Halle micro-CT skull specimen (Stages 1–5 produce the canonical run; Stages 6–9 produce the manuscript figures and tables). Stages 1 and 4 require the upstream `fullwave` and `shearwave` solvers; the cellular stages (5 onwards) depend only on the contents of this repository plus the saved displacement and temperature fields.

### GPU acceleration

The CUDA backend (`ultraneuron/cuda_backend.py`) accelerates the per-voxel HH integration by roughly an order of magnitude on a single NVIDIA RTX A6000. The driver scripts (`run_sensitivity.py`, `run_halle_mechanism_sweep.py`, `render_fig8_timepoints.py`) read `AMN_BACKEND` from the environment and default to CUDA. Set `AMN_BACKEND=numba` to fall back to CPU.

Reference wall-clocks for the canonical Halle / caudate sweep on Numba CPU vs CUDA (RTX A6000):

| Stage | Numba CPU | CUDA |
|---|---|---|
| Mechanism sweep (30 sims) | ~49 min | ~7 min |
| Sensitivity sweep (13 sims) | ~32 min | ~5 min (est.) |

The two backends agree to within numerical tolerance on the unit-test suite.

## Citation

If you use this framework, please cite the accompanying manuscript and the software (see `CITATION.cff`). A BibTeX-ready entry will be added when the manuscript appears.

## License

Apache 2.0 — see `LICENSE`.
