"""Provenance-driven sensitivity sweep for the JNE manuscript Results §3.3.

Holds the TIPS focal field constant at peak ε_dev = 0.020 (mid-range
of the strain sweep) and the S5 full-pathway scenario as the baseline,
then perturbs one parameter at a time by ±25% and reports the
fractional change in total spike count over the 30 ms window.

Outputs:
  data/sensitivity_results.npz    — baseline + per-parameter shifts
  data/sensitivity_table.tex      — LaTeX table for Results §3.3
"""

from __future__ import annotations

import argparse
import re
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

# Compute backend for the per-voxel HH integration.  Override at the
# shell with AMN_BACKEND=numba (CPU) or AMN_BACKEND=cuda; default 'auto' = GPU when visible.
_BACKEND = "auto"  # resolved by ultraneuron.cuda_backend.resolve_backend (AMN_BACKEND env override)

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))

from ultraneuron import (  # noqa: E402
    PIEZO1,
    TREK1,
    TRPV1,
    TRPV4,
    K_MECHANO_NAV,
    CalciumDynamics,
    KCaChannel,
    MechanoHHNeuron,
    MechanoSynapse,
    MulticompartmentMechanoHHNeuron,
    NICEModel,
    NMDAReceptor,
    SolitonModel,
    simulate_synaptic_relay,
)
from ultraneuron.channels import strain_to_membrane_tension  # noqa: E402

_SIM_NAME = os.environ.get("HALLE_SIM_NAME", "halle_caudate")
_SUFFIX = "" if _SIM_NAME == "halle_caudate" else f"_{_SIM_NAME}"
OUT_DATA = HERE.parent / "data" / f"sensitivity_results{_SUFFIX}.npz"
OUT_TEX = HERE.parent / "data" / f"sensitivity_table{_SUFFIX}.tex"

# Canonical (no-suffix) output targets — the files the manuscript \input{}s.
CANONICAL_DATA = HERE.parent / "data" / "sensitivity_results.npz"
CANONICAL_TEX = HERE.parent / "data" / "sensitivity_table.tex"

# Properly-cased field display name for the table caption (the raw
# "halle_<x>".replace("halle_","") gives lowercase "dacc"; map the known
# targets to their display form).
_DISPLAY_NAME = {
    "halle_caudate": "caudate",
    "halle_dacc": "dACC",
    "halle_pcc": "PCC",
}.get(_SIM_NAME, _SIM_NAME.replace("halle_", ""))


REAL_SIM_DIR = HERE.parent / "data" / _SIM_NAME
COUPLING_FACTOR = 1000.0  # α (cortical-pyramidal-calibrated; manuscript Eq. 5)


def _config_push_window_ms(sim_name: str, default_ms: float = 30.0) -> float:
    """Neural integration window (ms) from the sim's NEUROMOD_SEQUENCE
    (arf_push_window_s → tone_burst_duration_s), so the sensitivity sweep
    integrates over the same protocol ON window as the canonical run
    (Yaakub: 20 ms; Legon configs: 30 ms)."""
    cfg_path = (HERE.parent / "validation_report" / "configs" / f"{sim_name}.py")
    if not cfg_path.exists():
        return default_ms
    import importlib.util
    spec = importlib.util.spec_from_file_location(f"_cfg_{sim_name}", cfg_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    seq = getattr(mod, "NEUROMOD_SEQUENCE", {}) or {}
    win_s = seq.get("arf_push_window_s",
                    seq.get("tone_burst_duration_s", default_ms * 1e-3))
    return float(win_s) * 1e3


PUSH_WINDOW_MS = _config_push_window_ms(_SIM_NAME)


def _strain_tensor_eeq(u: np.ndarray, dx: float, dy: float, dz: float) -> np.ndarray:
    """Per-voxel von Mises deviatoric strain ε_eq from a 3-component
    displacement field (mirrors validation_report._strain_tensor)."""
    ux, uy, uz = u[..., 0], u[..., 1], u[..., 2]
    e_xx = np.gradient(ux, dx, axis=0)
    e_yy = np.gradient(uy, dy, axis=1)
    e_zz = np.gradient(uz, dz, axis=2)
    e_xy = 0.5 * (np.gradient(ux, dy, axis=1) + np.gradient(uy, dx, axis=0))
    e_xz = 0.5 * (np.gradient(ux, dz, axis=2) + np.gradient(uz, dx, axis=0))
    e_yz = 0.5 * (np.gradient(uy, dz, axis=2) + np.gradient(uz, dy, axis=1))
    e_vol = (e_xx + e_yy + e_zz) / 3.0
    dxx, dyy, dzz = e_xx - e_vol, e_yy - e_vol, e_zz - e_vol
    return np.sqrt((2.0 / 3.0)
                   * (dxx**2 + dyy**2 + dzz**2
                      + 2.0 * (e_xy**2 + e_xz**2 + e_yz**2))).astype(np.float32)


def halle_caudate_field() -> dict:
    """Load the canonical Halle / caudate end-of-pulse strain field
    (the same field that drives the manuscript Fig 4–6 / Table 1
    canonical run), and a per-voxel pressure envelope proxy for NICE.

    Strain comes from displacement_4d.npy at the peak-displacement
    frame (constant ARF push during the 30 ms ON window — within the
    static-equilibrium approximation the per-voxel ε field is the
    relevant input to the sensitivity sweep at the working point
    sensed by the neurons).  Pressure for NICE is set to a per-voxel
    constant equal to the FDTD focal PNP scaled by the local strain
    fraction (so the focal voxel gets the full pressure and far-field
    voxels get a fraction of it; this is what the per-push NICE input
    would look like if we sampled it at the peak instant).
    """
    grid = np.load(REAL_SIM_DIR / "grid_params.npz", allow_pickle=True)
    dx = float(grid["dx"]) * int(grid["mod_x"])
    dy = float(grid["dy"]) * int(grid["mod_y"])
    dz = float(grid["dz"]) * int(grid["mod_z"])
    u4 = np.load(REAL_SIM_DIR / "displacement_4d.npy", mmap_mode="r")
    abs_per_frame = np.abs(u4).reshape(u4.shape[0], -1).max(axis=1)
    peak_t_idx = int(np.argmax(abs_per_frame))
    u_peak = np.asarray(u4[peak_t_idx], dtype=np.float64)
    eeq = _strain_tensor_eeq(u_peak, dx, dy, dz)
    # Mask skull voxels to zero (large gradient artefact at brain-skull
    # boundary)
    c_map = np.load(REAL_SIM_DIR / "c_ds.npy")
    eeq[c_map > 1600.0] = 0.0
    eeq_peak = float(eeq.max())
    # Per-voxel pressure proxy: scale focal PNP (~0.5 MPa transcranial)
    # by the local strain fraction
    p_focal_pa = 0.5e6
    pressure = (eeq / max(eeq_peak, 1e-12)) * p_focal_pa
    return {"strain": eeq.astype(np.float32),
            "pressure": pressure.astype(np.float32),
            "peak_eeq": eeq_peak}


def build_neuron(*, K_A: float = 0.25, dendrite_scale: float = 5.0,
                 piezo_T_half: float = 2.7, ais_g_Na: float = 2000.0,
                 sk_g_density: float = 0.3, sk_Ca_half: float = 0.5):
    ch_p = deepcopy(PIEZO1); ch_p.g_density = 0.10; ch_p.T_half = piezo_T_half
    ch_t = deepcopy(TREK1);  ch_t.g_density = 0.04
    sk = KCaChannel(g_density=sk_g_density, Ca_half=sk_Ca_half)
    n = MulticompartmentMechanoHHNeuron.cortical_pyramidal_3comp(
        mechano_channels=[ch_p, ch_t],
        temperature=37.0,
        dendrite_piezo_scale=dendrite_scale,
        nice_model=NICEModel(),
        nice_compartments=(0, 1, 2),
        piezo_inactivation=True,
        calcium_dynamics=CalciumDynamics(),
        kca_channels=[sk], kca_compartments=(1,),
        trp_channels=[deepcopy(TRPV1), deepcopy(TRPV4)], trp_compartments=(1,),
    )
    n.geometry.ais.g_Na = ais_g_Na
    return n, K_A


def run_total_spikes(neuron, K_A, field, dt=0.001, duration=PUSH_WINDOW_MS):
    tension = strain_to_membrane_tension(
        field["strain"], K_A=K_A, coupling_factor=COUPLING_FACTOR,
    )
    mask = tension >= 0.5
    if not np.any(mask):
        return 0
    n_steps = int(duration / dt)
    sc = neuron._simulate_batch(
        tension[mask].astype(np.float64),
        dt=dt, n_steps=n_steps,
        I_stim_per_compartment=np.array([0.0, 0.0, 0.0]),
        primary_compartment=1,
        pressure_envelope=field["pressure"][mask].astype(np.float64),
        backend=_BACKEND,
    )
    return int(sc.sum())


# ---------------------------------------------------------------------------
# Mechanosensitive synaptic / voltage-gated transmission (M6) sub-pathway
# ---------------------------------------------------------------------------
# The M6 mechanism does not feed the multi-compartment strain-driven spike
# count above; it is a separate synaptic + voltage-gated pathway scored on its
# own readout — the relay focal spike count at the field's focal tension. Its
# sensitivity is reported in a labelled sub-block of the same table.

def focal_tension_from_neural() -> float:
    """Peak membrane tension [mN/m] at the focal voxel of the canonical run.

    Read from ``<sim>/neural_results.npz`` (``T_peak_mNm``), so the M6 block
    needs neither the 542 MB displacement field nor a CUDA sweep.
    """
    nr = np.load(REAL_SIM_DIR / "neural_results.npz", allow_pickle=True)
    return float(np.asarray(nr["T_peak_mNm"], dtype=np.float64).max())


# Integration window for the M6 relay focal-spike readout. Stated in the
# sensitivity-table sub-header so the baseline count is interpretable (it is
# window-dependent; the validation-report driver uses a shorter 120 ms window).
_M6_RELAY_DURATION_MS = 300.0


def run_relay_focal_spikes(
    focal_T: float,
    duration: float = _M6_RELAY_DURATION_MS,
    *,
    Pr_max: float = 0.6,
    T_half_pre: float = 4.0,
    alpha_NMDA: float = 1.0,
    T_half_stretch: float = 5.0,
    k_mechano_nav: float = K_MECHANO_NAV,
    glu_scale: float = 30.0,
    g_ampa: float = 0.08,
) -> int:
    """Relay focal spike count via ``simulate_synaptic_relay`` at ``focal_T``."""
    syn = MechanoSynapse(
        Pr_max=Pr_max,
        T_half_pre=T_half_pre,
        glu_scale=glu_scale,
        g_ampa=g_ampa,
        nmda=NMDAReceptor(g_max=0.45, alpha_NMDA=alpha_NMDA, T_half_stretch=T_half_stretch),
    )
    post = MechanoHHNeuron.cortical_pyramidal(temperature=37.0, k_mechano_nav=k_mechano_nav)
    r = simulate_synaptic_relay(
        peak_tension_mNm=float(focal_T), synapse=syn, post_neuron=post,
        dt=0.025, duration=duration, us_onset=10.0,
    )
    return int(r["spike_count"])


def m6_sensitivity_rows(focal_T: float) -> tuple[int, list[dict]]:
    """Compute the M6 baseline and per-parameter ±25% sensitivity rows."""
    m6_base = run_relay_focal_spikes(focal_T)
    print(f"M6 baseline (relay focal spikes at T={focal_T:.2f} mN/m): {m6_base}", flush=True)
    perturbations = [
        ("Presynaptic $P_{r,\\max}$", "Estimate", "Pr_max", 0.6),
        ("Presynaptic $T_{1/2}^{\\mathrm{pre}}$ (mN/m)", "Estimate", "T_half_pre", 4.0),
        ("NMDA stretch $\\alpha_{\\mathrm{NMDA}}$", "Calibrated", "alpha_NMDA", 1.0),
        ("NMDA $T_{1/2}^{\\mathrm{s}}$ (mN/m)", "Estimate", "T_half_stretch", 5.0),
        ("Na$^+$ shift $k_{\\mathrm{mechano}}$ (mV per mN/m)", "Calibrated", "k_mechano_nav", 0.3),
        ("Glutamate scale $G_s$ ($\\mu$M)", "Calibrated", "glu_scale", 30.0),
        ("AMPA $\\bar{g}$ (mS/cm$^2$)", "Calibrated", "g_ampa", 0.08),
    ]
    rows = []
    for name, source, kwarg, baseline_val in perturbations:
        results = {}
        for sign, label in ((+1, "high"), (-1, "low")):
            shifted = baseline_val * (1.0 + sign * 0.25)
            n_sp = run_relay_focal_spikes(focal_T, **{kwarg: shifted})
            print(f"  M6 {name} {label} ({sign:+d}25% → {shifted:.3g}): {n_sp} spikes", flush=True)
            results[label] = n_sp
        delta_low = (results["low"] - m6_base) / max(m6_base, 1) * 100
        delta_high = (results["high"] - m6_base) / max(m6_base, 1) * 100
        rows.append({
            "name": name, "source": source,
            "low": results["low"], "high": results["high"],
            "delta_low": delta_low, "delta_high": delta_high,
            "abs_max_delta": max(abs(delta_low), abs(delta_high)),
        })
    return m6_base, sorted(rows, key=lambda r: -r["abs_max_delta"])


def _fmt_int(n: int) -> str:
    """Format an integer with LaTeX thin-space thousand separators."""
    return f"{int(n):,}".replace(",", "\\,")


_M6_BEGIN = "        % M6-BEGIN (synaptic / voltage-gated; scripts/run_sensitivity.py)"
_M6_END = "        % M6-END"


def _m6_table_lines(focal_T: float, m6_base: int, m6_rows: list[dict]) -> list[str]:
    """LaTeX rows for the M6 sub-block (delimited so re-runs are idempotent)."""
    out = [
        _M6_BEGIN,
        "        \\midrule",
        "        \\multicolumn{6}{@{}p{\\linewidth}@{}}{\\textit{Synaptic / voltage-gated transmission "
        "(M6), relative to " + _fmt_int(m6_base)
        + " relay focal spikes over a " + f"{_M6_RELAY_DURATION_MS:.0f}"
        + "~ms window at the focal voxel ($T = "
        + f"{focal_T:.1f}" + "$~mN/m):}} \\\\",
        "        \\midrule",
    ]
    for r in m6_rows:
        out.append(
            f"        {r['name']} & {r['source']} & {_fmt_int(r['low'])} & "
            f"{_fmt_int(r['high'])} & {r['delta_low']:+.0f} & {r['delta_high']:+.0f} \\\\",
        )
    out.append(_M6_END)
    return out


def _inject_m6_into_table(path: Path, focal_T: float, m6_base: int, m6_rows: list[dict]) -> None:
    """Insert (or replace) the M6 sub-block before ``\\bottomrule`` in an existing table."""
    import re
    text = path.read_text()
    text = re.sub(r"\n *% M6-BEGIN.*?% M6-END", "", text, flags=re.S)  # drop any prior block
    block = "\n".join(_m6_table_lines(focal_T, m6_base, m6_rows))
    text = re.sub(r"( *\\bottomrule)", lambda m: block + "\n" + m.group(1), text, count=1)
    path.write_text(text)


# ---------------------------------------------------------------------------
# Heimburg-Jackson soliton (alternative thermodynamic-membrane mechanism) block
# ---------------------------------------------------------------------------
# The soliton hypothesis is an *alternative* to the Piezo1/HH pathway, not a
# stage of it. Its readout is a closed-form scalar excitability
# ξ = κ_rel(T)·|P/(ρc²)|, not a spike count, so it gets its own labelled
# sub-block (like M6) scored at the focal operating point. It is cheap
# (no field, no CUDA), so it runs in both the full and --skip-s5 paths.

SOLITON_P_FOCAL_PA = 0.5e6   # transcranial focal PNP [Pa] (matches halle_caudate_field)
SOLITON_FOCAL_T_C = 37.0     # body-temperature operating point [°C]

_SOLITON_BEGIN = "        % SOLITON-BEGIN (alternative mechanism; scripts/run_sensitivity.py)"
_SOLITON_END = "        % SOLITON-END"


def soliton_excitability(*, T_m: float = 30.0, sigma_T: float = 3.0,
                         c_p_max: float = 10.0) -> float:
    """Soliton excitability scalar at the focal operating point [dimensionless]."""
    return SolitonModel(T_m=T_m, sigma_T=sigma_T, c_p_max=c_p_max).excitability(
        SOLITON_P_FOCAL_PA, SOLITON_FOCAL_T_C,
    )


def soliton_sensitivity_rows() -> tuple[float, list[dict]]:
    """Baseline excitability and per-parameter ±25% sensitivity rows."""
    base = soliton_excitability()
    print(f"Soliton baseline (excitability at P={SOLITON_P_FOCAL_PA/1e6:.2f} MPa, "
          f"T={SOLITON_FOCAL_T_C:.0f} °C): {base:.3e}", flush=True)
    perturbations = [
        ("Transition temp $T_m$ ($^\\circ$C)", "Literature", "T_m", 30.0),
        ("Peak half-width $\\sigma_T$ ($^\\circ$C)", "Estimate", "sigma_T", 3.0),
        ("Excess-$c_p$ ratio $c_p^{\\max}$", "Estimate", "c_p_max", 10.0),
    ]
    rows = []
    for name, source, kwarg, baseline_val in perturbations:
        results = {}
        for sign, label in ((+1, "high"), (-1, "low")):
            shifted = baseline_val * (1.0 + sign * 0.25)
            val = soliton_excitability(**{kwarg: shifted})
            print(f"  Soliton {name} {label} ({sign:+d}25% → {shifted:.3g}): "
                  f"ξ = {val:.3e}", flush=True)
            results[label] = val
        delta_low = (results["low"] - base) / base * 100
        delta_high = (results["high"] - base) / base * 100
        rows.append({
            "name": name, "source": source,
            "low": results["low"], "high": results["high"],
            "delta_low": delta_low, "delta_high": delta_high,
            "abs_max_delta": max(abs(delta_low), abs(delta_high)),
        })
    return base, sorted(rows, key=lambda r: -r["abs_max_delta"])


def _fmt_exc(x: float) -> str:
    """Format a small dimensionless excitability in LaTeX scientific notation."""
    if x == 0:
        return "0"
    exp = int(np.floor(np.log10(abs(x))))
    mant = x / (10.0 ** exp)
    return f"{mant:.2f}{{\\times}}10^{{{exp}}}"


def _soliton_table_lines(base: float, rows: list[dict]) -> list[str]:
    """LaTeX rows for the soliton sub-block (delimited so re-runs are idempotent)."""
    out = [
        _SOLITON_BEGIN,
        "        \\midrule",
        "        \\multicolumn{6}{@{}p{\\linewidth}@{}}{\\textit{Heimburg-Jackson soliton "
        "(alternative thermodynamic-membrane mechanism), $-25\\%$/$+25\\%$ "
        "columns are the dimensionless excitability $\\xi = \\kappa_{\\mathrm{rel}}"
        "(T)\\,|P/\\rho c^2|$ relative to $\\xi_0 = $ $" + _fmt_exc(base)
        + "$ at the focal voxel ($P = "
        + f"{SOLITON_P_FOCAL_PA/1e6:.2f}" + "$~MPa, $T = "
        + f"{SOLITON_FOCAL_T_C:.0f}" + "^\\circ$C). The $+25\\%$ $T_m$ shift "
        "reaches $T_m = 37.5^\\circ$C $\\geq$ body temperature, where the "
        "heating-reduces-excitability sign no longer holds; its large "
        "$\\Delta_{\\!+}$ reflects $T_m$ crossing the operating point rather "
        "than a physical regime:}} \\\\",
        "        \\midrule",
    ]
    for r in rows:
        out.append(
            f"        {r['name']} & {r['source']} & ${_fmt_exc(r['low'])}$ & "
            f"${_fmt_exc(r['high'])}$ & {r['delta_low']:+.0f} & {r['delta_high']:+.0f} \\\\",
        )
    out.append(_SOLITON_END)
    return out


def _inject_soliton_into_table(path: Path, base: float, rows: list[dict]) -> None:
    """Insert (or replace) the soliton sub-block before ``\\bottomrule``."""
    import re
    text = path.read_text()
    text = re.sub(r"\n *% SOLITON-BEGIN.*?% SOLITON-END", "", text, flags=re.S)
    block = "\n".join(_soliton_table_lines(base, rows))
    text = re.sub(r"( *\\bottomrule)", lambda m: block + "\n" + m.group(1), text, count=1)
    path.write_text(text)



_DOMINANCE_SENTENCE = (" The Piezo1 half-activation tension $T_{1/2}$ and the strain-tension coupling $K_A$ jointly "
                       "dominate sensitivity by setting where the focal-voxel tension lies relative to the Boltzmann knee.")
_SCALAR_CAPTION = (
    "\\caption{Source-classified sensitivity of the closed-form scalars that score the mechanisms outside the "
    "S1--S5 strain sweep (the S5 spike-count sensitivity is main-text Table~\\ref{tab:sensitivity}). Each row shifts "
    "a single parameter by $\\pm 25\\%$ from its default, sorted by absolute deviation within each block. Upper "
    "block, synaptic\\,/\\,voltage-gated transmission (mechanism 6), relay focal spike count; lower block, "
    "Heimburg--Jackson soliton, dimensionless excitability $\\xi$. Abbreviations: S1--S5, mechanism scenarios of main-text Table~\\ref{tab:scenarios}; AMPA, "
    "$\\alpha$-amino-3-hydroxy-5-methyl-4-isoxazolepropionic acid receptor; NMDA, $N$-methyl-D-aspartate receptor.}")


def split_scalar_blocks(main_tex: Path) -> Path:
    """Move the M6 and soliton sub-blocks out of the S5 table into data/sensitivity_table_scalar*.tex.

    Pure text operation on an existing table (no simulation); keeps the remaining main-table text
    byte-identical apart from the removed blocks and the dominance sentence of the caption.
    """
    text = main_tex.read_text()
    blocks = [m.group(0).lstrip("\n") for pat in (r"\n *% M6-BEGIN.*?% M6-END", r"\n *% SOLITON-BEGIN.*?% SOLITON-END")
              for m in [re.search(pat, text, flags=re.S)] if m]
    scalar_tex = main_tex.with_name(main_tex.name.replace("sensitivity_table", "sensitivity_table_scalar"))
    if not blocks:
        return scalar_tex
    head = [r"% Auto-generated by scripts/run_sensitivity.py (split from " + main_tex.name + "); do not edit by hand.",
            r"\begin{table}[htbp]", r"\begin{revblock}[E]", _SCALAR_CAPTION.replace(r"\caption{", r"\caption{\revtag{L} ", 1), r"\label{tab:sensitivity_scalar}",
            r"\footnotesize", r"\centering", r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
            r"Parameter & Source & $-25\%$ value & $+25\%$ value & $\Delta_{\!-}\,(\%)$ & $\Delta_{\!+}\,(\%)$ \\"]
    tail = [r"\bottomrule", r"\end{tabular}", r"\end{revblock}", r"\end{table}"]
    scalar_tex.write_text("\n".join(head + blocks + tail) + "\n")
    text = re.sub(r"\n *% M6-BEGIN.*?% M6-END", "", text, flags=re.S)
    text = re.sub(r"\n *% SOLITON-BEGIN.*?% SOLITON-END", "", text, flags=re.S)
    main_tex.write_text(text.replace(_DOMINANCE_SENTENCE, ""))
    return scalar_tex


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-s5", action="store_true",
                    help="Reuse the cached S5 block from the results npz and the existing "
                         "table; only (re)compute and inject the M6 sub-block. Avoids the "
                         "heavy multi-compartment CUDA sweep and 542 MB field load.")
    ap.add_argument("--canonical", action="store_true",
                    help="Write the no-suffix canonical files (data/sensitivity_table.tex, "
                         "data/sensitivity_results.npz) — the ones the manuscript \\input{}s — "
                         "regardless of HALLE_SIM_NAME. Use with HALLE_SIM_NAME=halle_dacc to "
                         "regenerate the dACC manuscript table.")
    ap.add_argument("--split-existing", action="store_true",
                    help="No computation: move the M6 and soliton sub-blocks of the existing table "
                         "into data/sensitivity_table_scalar*.tex (supplementary table).")
    args = ap.parse_args()

    out_tex = CANONICAL_TEX if args.canonical else OUT_TEX
    if args.split_existing:
        print("wrote", split_scalar_blocks(out_tex), flush=True)
        return
    out_data = CANONICAL_DATA if args.canonical else OUT_DATA

    # --- M6-only fast path: keep the validated S5 numbers / caption untouched ---
    if args.skip_s5:
        if not out_data.exists() or not out_tex.exists():
            raise SystemExit(
                f"--skip-s5 needs a prior full build; missing {out_data.name} or {out_tex.name}."
            )
        focal_T = focal_tension_from_neural()
        m6_base, m6_rows = m6_sensitivity_rows(focal_T)
        _inject_m6_into_table(out_tex, focal_T, m6_base, m6_rows)
        sol_base, sol_rows = soliton_sensitivity_rows()
        _inject_soliton_into_table(out_tex, sol_base, sol_rows)
        split_scalar_blocks(out_tex)
        print(f"Injected M6 and soliton sub-blocks into {out_tex}", flush=True)
        print("DONE", flush=True)
        return

    print(f"=== JNE Phase 5a: sensitivity sweep (Halle {_DISPLAY_NAME} real field) ===", flush=True)
    field = halle_caudate_field()
    print(f"  field shape {field['strain'].shape}, peak ε_eq = "
          f"{field['peak_eeq']:.3e}, α = {COUPLING_FACTOR}", flush=True)

    # Baseline (everything at default)
    baseline_neuron, baseline_K_A = build_neuron()
    t0 = time.perf_counter()
    n_base = run_total_spikes(baseline_neuron, baseline_K_A, field)
    print(f"baseline (defaults): {n_base} spikes, {time.perf_counter()-t0:.1f}s", flush=True)

    # Per-parameter perturbations: each row shifts a single parameter by +25% and -25%
    perturbations = [
        ("$K_A\\alpha$ (N/m per unit strain; via $K_A$)", "Assumption", "K_A", 0.25, 0.25),
        ("Dendrite Piezo1 scale", "Literature", "dendrite_scale", 5.0, 0.25),
        ("Piezo1 $T_{1/2}$ (mN/m)", "Literature", "piezo_T_half", 2.7, 0.25),
        ("AIS Na density (mS/cm$^2$)", "Literature", "ais_g_Na", 2000.0, 0.25),
        ("SK $\\bar{g}$ (mS/cm$^2$)", "Literature", "sk_g_density", 0.3, 0.25),
        ("SK $[\\mathrm{Ca}^{2+}]_{1/2}$ ($\\mu$M)", "Literature", "sk_Ca_half", 0.5, 0.25),
    ]
    rows = []
    for name, source, kwarg, baseline_val, frac in perturbations:
        results = {}
        for sign, label in ((+1, "high"), (-1, "low")):
            shifted = baseline_val * (1.0 + sign * frac)
            kw = {kwarg: shifted}
            n_shift, K_A_shift = build_neuron(**kw)
            K_A_for_call = K_A_shift if kwarg == "K_A" else 0.25
            t0 = time.perf_counter()
            n_spikes = run_total_spikes(n_shift, K_A_for_call, field)
            print(
                f"  {name} {label} ({sign:+d}25% → {shifted:.3g}): {n_spikes} spikes, "
                f"{time.perf_counter()-t0:.1f}s",
                flush=True,
            )
            results[label] = n_spikes
        delta_high = (results["high"] - n_base) / max(n_base, 1) * 100
        delta_low = (results["low"] - n_base) / max(n_base, 1) * 100
        rows.append({
            "name": name, "source": source,
            "low": results["low"], "high": results["high"],
            "delta_low": delta_low, "delta_high": delta_high,
            "abs_max_delta": max(abs(delta_low), abs(delta_high)),
        })

    rows_sorted = sorted(rows, key=lambda r: -r["abs_max_delta"])

    # M6 synaptic / voltage-gated sub-block (scored on the relay focal spikes)
    focal_T = focal_tension_from_neural()
    m6_base, m6_rows = m6_sensitivity_rows(focal_T)

    # Soliton (alternative thermodynamic-membrane mechanism) sub-block
    sol_base, sol_rows = soliton_sensitivity_rows()

    # Save data (S5 block + M6 block)
    out_data.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_data,
        baseline=n_base,
        names=np.array([r["name"] for r in rows_sorted]),
        sources=np.array([r["source"] for r in rows_sorted]),
        low_counts=np.array([r["low"] for r in rows_sorted]),
        high_counts=np.array([r["high"] for r in rows_sorted]),
        delta_low_pct=np.array([r["delta_low"] for r in rows_sorted]),
        delta_high_pct=np.array([r["delta_high"] for r in rows_sorted]),
        m6_baseline=m6_base,
        m6_focal_tension_mNm=focal_T,
        m6_names=np.array([r["name"] for r in m6_rows]),
        m6_sources=np.array([r["source"] for r in m6_rows]),
        m6_low_counts=np.array([r["low"] for r in m6_rows]),
        m6_high_counts=np.array([r["high"] for r in m6_rows]),
        m6_delta_low_pct=np.array([r["delta_low"] for r in m6_rows]),
        m6_delta_high_pct=np.array([r["delta_high"] for r in m6_rows]),
        soliton_baseline_excitability=sol_base,
        soliton_P_focal_pa=SOLITON_P_FOCAL_PA,
        soliton_focal_T_C=SOLITON_FOCAL_T_C,
        soliton_names=np.array([r["name"] for r in sol_rows]),
        soliton_sources=np.array([r["source"] for r in sol_rows]),
        soliton_low=np.array([r["low"] for r in sol_rows]),
        soliton_high=np.array([r["high"] for r in sol_rows]),
        soliton_delta_low_pct=np.array([r["delta_low"] for r in sol_rows]),
        soliton_delta_high_pct=np.array([r["delta_high"] for r in sol_rows]),
    )
    print(f"Saved: {out_data}", flush=True)

    # LaTeX table
    lines = [
        "% Auto-generated by scripts/run_sensitivity.py — do not edit by hand.",
        "\\begin{table}",
        "    \\caption{Source-classified sensitivity at the S5 full-pathway baseline on the real Halle / " + _DISPLAY_NAME + " end-of-pulse strain field ($\\varepsilon_{\\mathrm{eq}}^{\\max} = " + (f"{field['peak_eeq']:.2e}").replace("e-0", "{\\times}10^{-") + "}$, $\\alpha = %d$, %s baseline spikes). Each row shifts a single parameter by $\\pm 25\\%%$ from its default and reports total spike count and fractional change, sorted by absolute deviation. The Piezo1 half-activation tension $T_{1/2}$ and the strain-tension coupling $K_A$ jointly dominate sensitivity by setting where the focal-voxel tension lies relative to the Boltzmann knee. The baseline is the S5 full-pathway scenario at the canonical conductances ($\\bar g_{\\mathrm{Piezo1}} = 0.10$, $\\bar g_{\\mathrm{TREK\\text{-}1}} = 0.04$\\,mS/cm$^2$, $K_A\\alpha = 250$\\,N/m per unit strain); it differs from the S2 canonical run of Table~\\ref{tab:yaakub_summary} on the same field by the S5-versus-S2 effect of \\S\\ref{sec:results_mechanism_sweep} (Piezo1 inactivation and SK), so the two tables share the field and the working point but not the scenario. \\rev[E]{Abbreviations: dACC, dorsal anterior cingulate cortex; SK, small-conductance Ca\\textsuperscript{2+}-activated K\\textsuperscript{+} channel.}}" % (int(COUPLING_FACTOR), _fmt_int(n_base)),
        "    \\label{tab:sensitivity}",
        "    \\footnotesize",
        "    \\centering",
        "    \\begin{tabular}{@{}lccccc@{}}",
        "        \\toprule",
        "        Parameter & Source & $-25\\%$ count & $+25\\%$ count & $\\Delta_{\\!-}\\,(\\%)$ & $\\Delta_{\\!+}\\,(\\%)$ \\\\",
        "        \\midrule",
    ]
    for r in rows_sorted:
        lines.append(
            f"        {r['name']} & {r['source']} & {_fmt_int(r['low'])} & {_fmt_int(r['high'])} & {r['delta_low']:+.0f} & {r['delta_high']:+.0f} \\\\",
        )
    lines += _m6_table_lines(focal_T, m6_base, m6_rows)
    lines += _soliton_table_lines(sol_base, sol_rows)
    lines += [
        "        \\bottomrule",
        "    \\end{tabular}",
        "\\end{table}",
    ]
    out_tex.write_text("\n".join(lines) + "\n")
    split_scalar_blocks(out_tex)
    print(f"Saved: {out_tex}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
