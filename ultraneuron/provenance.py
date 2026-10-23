"""Per-parameter provenance metadata for the channel and pathway library.

Phase-5 cleanup: every numerical parameter used in the framework is
classified by where it comes from. This makes the absolute-rate
predictions traceable to specific papers (or to declared modelling
choices), which lets us state uncertainty bounds on a per-parameter
basis when comparing mechanism candidates.

This module is data-only and does not change the channel / pathway
APIs.  Provenance entries are queried by name via
:func:`provenance_for`; a structured human-readable summary can be
obtained with :func:`format_provenance_report`.

Source classification
---------------------
* ``LITERATURE`` — value is fit to (or directly read from) a published
  experimental measurement. The reference cites the paper and figure
  / table where the value lives.
* ``ASSUMPTION`` — value is a deliberate modelling choice not pinned by
  data. Examples: the lumped lipid-bilayer area-expansion modulus
  K_A in the strain-to-tension conversion; the Pinsky-Rinzel-style
  area ratios in the multi-compartment skeleton.
* ``CALIBRATED`` — value chosen so that the model reproduces a
  qualitative literature observation (e.g. NICE leaflet inertia
  picked so that |Z| at 100 kPa matches the SONIC operating range).
  A note explaining the calibration target is included.
* ``DERIVED`` — value computed from other parameters (e.g. ω_0 from
  f_resonant). Listed for completeness but not a free variable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ParamSource(str, Enum):
    LITERATURE = "literature"
    ASSUMPTION = "assumption"
    CALIBRATED = "calibrated"
    DERIVED = "derived"
    ESTIMATE = "estimate"  # best-guess value from sparse/indirect data (weaker than ASSUMPTION)


@dataclass
class Provenance:
    """Provenance entry for a single parameter."""

    source: ParamSource
    reference: str = ""  # paper citation (free-form short string)
    notes: str = ""      # one-line explanation


# ---------------------------------------------------------------------------
# MechanosensitiveChannel parameter provenance
# ---------------------------------------------------------------------------

PIEZO1_PROVENANCE: dict[str, Provenance] = {
    "T_half": Provenance(
        ParamSource.LITERATURE,
        reference="Coste et al. 2010, Science 330:55-60; Lewis & Grandl 2015",
        notes="Mid-point of activation across multiple cell types",
    ),
    "delta_A": Provenance(
        ParamSource.LITERATURE,
        reference="Coste et al. 2010 (Boltzmann fit to single-channel records)",
        notes="Effective gating area change from ΔA·T fit",
    ),
    "g_max": Provenance(
        ParamSource.LITERATURE,
        reference="Coste et al. 2010 (single-channel slope)",
        notes="Single-channel conductance ~25 pS in physiological gradients",
    ),
    "g_density": Provenance(
        ParamSource.ASSUMPTION,
        reference="—",
        notes="Sparse expression assumption; user typically overrides per cell type",
    ),
    "E_rev": Provenance(
        ParamSource.LITERATURE,
        reference="Coste et al. 2010 (non-selective cation, ~0 mV)",
    ),
    "tau_0": Provenance(
        ParamSource.ASSUMPTION,
        reference="—",
        notes="Single time constant subsuming activation kinetics; superseded by tau_act when set",
    ),
    "tau_act": Provenance(
        ParamSource.LITERATURE,
        reference="Coste et al. 2010 Fig. 2 (activation rise τ ~1-2 ms)",
        notes="Phase-5 fixed-tau replacement for the bell-shaped τ_w(T)",
    ),
    "tau_inact": Provenance(
        ParamSource.LITERATURE,
        reference="Coste et al. 2010 Fig. 2 (inactivation τ 10-15 ms)",
    ),
    "inact_T_half": Provenance(
        ParamSource.ASSUMPTION,
        reference="—",
        notes="Set above activation T_half; not directly fit in Coste 2010",
    ),
    "inact_delta_A": Provenance(
        ParamSource.ASSUMPTION,
        reference="—",
        notes="Comparable steepness to activation; placeholder pending dedicated fit",
    ),
    "Ca_permeability_fraction": Provenance(
        ParamSource.LITERATURE,
        reference="Coste et al. 2010 (P_Ca / P_Na ~5)",
        notes="~10% of cation current is Ca²⁺ in physiological gradients",
    ),
}

TRAAK_PROVENANCE: dict[str, Provenance] = {
    "T_half": Provenance(
        ParamSource.LITERATURE,
        reference="Brohawn et al. 2014, PNAS 111:3614-3619",
    ),
    "delta_A": Provenance(ParamSource.LITERATURE, reference="Brohawn et al. 2014"),
    "g_max": Provenance(ParamSource.LITERATURE, reference="Brohawn et al. 2014"),
    "g_density": Provenance(
        ParamSource.ASSUMPTION,
        notes="Estimated from neural expression levels; user-tunable",
    ),
    "E_rev": Provenance(
        ParamSource.DERIVED,
        reference="K⁺ Nernst at physiological [K]",
        notes="Nernst potential of K+ at physiological concentrations ([K]o 3 mM, [K]i 140 mM, 37 C), -90 to -100 mV; not a fitted parameter",
    ),
    "tau_0": Provenance(
        ParamSource.ASSUMPTION,
        notes="Legacy bell-shape baseline; superseded by tau_act on the library preset",
    ),
    "tau_act": Provenance(
        ParamSource.LITERATURE,
        reference="Brohawn et al. 2014 (single-channel mechanical activation rise)",
        notes="Phase-5 fixed-tau replacement for the bell-shaped τ_w(T)",
    ),
    "tau_inact": Provenance(
        ParamSource.DERIVED,
        notes="K2P channels don't inactivate; field defaults to None (gate held at 1)",
    ),
    "inact_T_half": Provenance(
        ParamSource.DERIVED, notes="Not applicable; tau_inact is None",
    ),
    "inact_delta_A": Provenance(
        ParamSource.DERIVED, notes="Not applicable; tau_inact is None",
    ),
    "Ca_permeability_fraction": Provenance(
        ParamSource.DERIVED, notes="K2P channels are K⁺-selective; no Ca²⁺ flux",
    ),
}

TREK1_PROVENANCE: dict[str, Provenance] = {
    "T_half": Provenance(
        ParamSource.LITERATURE, reference="Honoré 2007, Nat Rev Neurosci 8:251-261"
    ),
    "delta_A": Provenance(ParamSource.LITERATURE, reference="Honoré 2007"),
    "g_max": Provenance(ParamSource.LITERATURE, reference="Honoré 2007"),
    "g_density": Provenance(
        ParamSource.ASSUMPTION, notes="Cortical pyramidal neuron expression range"
    ),
    "E_rev": Provenance(ParamSource.DERIVED, reference="K⁺ Nernst", notes="Nernst potential of K+ at physiological concentrations ([K]o 3 mM, [K]i 140 mM, 37 C), -90 to -100 mV; not a fitted parameter"),
    "tau_0": Provenance(
        ParamSource.ASSUMPTION,
        notes="Legacy bell-shape baseline; superseded by tau_act on the library preset",
    ),
    "tau_act": Provenance(
        ParamSource.LITERATURE,
        reference="Honoré 2007 (mid-range of 3-10 ms activation τ reported in review)",
        notes="Phase-5 fixed-tau replacement for the bell-shaped τ_w(T)",
    ),
    "tau_inact": Provenance(
        ParamSource.DERIVED,
        notes="K2P channels don't inactivate; field defaults to None",
    ),
    "inact_T_half": Provenance(
        ParamSource.DERIVED, notes="Not applicable; tau_inact is None",
    ),
    "inact_delta_A": Provenance(
        ParamSource.DERIVED, notes="Not applicable; tau_inact is None",
    ),
    "Ca_permeability_fraction": Provenance(
        ParamSource.DERIVED, notes="K2P channels are K⁺-selective; no Ca²⁺ flux",
    ),
}

TREK2_PROVENANCE: dict[str, Provenance] = {
    "T_half": Provenance(ParamSource.LITERATURE, reference="Kang & Kim 2024 review"),
    "delta_A": Provenance(ParamSource.LITERATURE, reference="Kang & Kim 2024"),
    "g_max": Provenance(ParamSource.LITERATURE, reference="Kang & Kim 2024"),
    "g_density": Provenance(
        ParamSource.ASSUMPTION,
        notes="Cortical pyramidal neuron expression range",
    ),
    "E_rev": Provenance(ParamSource.DERIVED, reference="K⁺ Nernst", notes="Nernst potential of K+ at physiological concentrations ([K]o 3 mM, [K]i 140 mM, 37 C), -90 to -100 mV; not a fitted parameter"),
    "tau_0": Provenance(
        ParamSource.ASSUMPTION,
        notes="Legacy bell-shape baseline; superseded by tau_act on the library preset",
    ),
    "tau_act": Provenance(
        ParamSource.LITERATURE,
        reference="Kang & Kim 2024 (TREK-2 kinetics; similar to TREK-1)",
        notes="Phase-5 fixed-tau replacement for the bell-shaped τ_w(T)",
    ),
    "tau_inact": Provenance(
        ParamSource.DERIVED,
        notes="K2P channels don't inactivate; field defaults to None",
    ),
    "inact_T_half": Provenance(
        ParamSource.DERIVED, notes="Not applicable; tau_inact is None",
    ),
    "inact_delta_A": Provenance(
        ParamSource.DERIVED, notes="Not applicable; tau_inact is None",
    ),
    "Ca_permeability_fraction": Provenance(
        ParamSource.DERIVED, notes="K2P channels are K⁺-selective; no Ca²⁺ flux",
    ),
}


# ---------------------------------------------------------------------------
# Strain-to-tension conversion
# ---------------------------------------------------------------------------

STRAIN_TO_TENSION_PROVENANCE: dict[str, Provenance] = {
    "K_A": Provenance(
        ParamSource.LITERATURE,
        reference=(
            "Rawicz et al. 2000 Biophys J 79:328 (micropipette aspiration, "
            "K_A = 0.23-0.27 N/m for PC bilayers; 0.1-0.5 N/m across lipid "
            "compositions); Sukharev & Sachs 2012"
        ),
        notes=(
            "Default 0.25 N/m is the centre of the measured range for lipid "
            "bilayers. Enters the model only through the product K_A*alpha, "
            "which is the single experimentally identifiable quantity and "
            "inherits the ASSUMPTION status of alpha."
        ),
    ),
    "coupling_factor": Provenance(
        ParamSource.ASSUMPTION,
        notes=(
            "Dimensionless multi-scale amplification α relating voxel-scale "
            "tissue strain to bilayer areal strain. Purely mechanical: it "
            "contains no channel-expression content (expression is carried "
            "by the per-compartment conductance densities and the "
            "dendrite:soma Piezo1 density ratio). Bracket [200, 2000]: lower "
            "end from planar-bilayer / continuum estimates (Rawicz 2000, "
            "Sukharev 2012, Salahshoor 2020), upper end from cytoskeletal "
            "and viscoelastic stress concentration; not calibrated on neurons "
            "in tissue under ultrasound. Code default 1.0; manuscript working "
            "point α=1000 (K_A*alpha = 250 N/m per unit strain), chosen so "
            "that the focal tension exceeds both the Piezo1 activation and "
            "inactivation half-tensions. Only the product K_A*alpha is identifiable."
        ),
    ),
}


# ---------------------------------------------------------------------------
# NICE bilayer model
# ---------------------------------------------------------------------------

NICE_PROVENANCE: dict[str, Provenance] = {
    "f_carrier": Provenance(
        ParamSource.DERIVED,
        reference="Typical TUS carrier (Legon, Tyler)",
        notes="Protocol input: the carrier frequency of the simulated exposure (500 kHz here), not a model parameter",
    ),
    "f_resonant": Provenance(
        ParamSource.LITERATURE,
        reference="Plaksin et al. 2014 (leaflet resonance ~5 MHz)",
    ),
    "damping_ratio": Provenance(
        ParamSource.ASSUMPTION,
        notes="ζ = 0.1 (Q ≈ 5); Plaksin-consistent moderate damping",
    ),
    "m_eff": Provenance(
        ParamSource.CALIBRATED,
        reference="SONIC operating range, Lemaire et al. 2019",
        notes="Calibrated so |Z|(100 kPa) ≈ 0.6 nm → ⟨ΔC_m⟩/C_m0 ≈ 5%",
    ),
    "delta_0": Provenance(
        ParamSource.LITERATURE,
        reference="Lipid bilayer half-thickness (Plaksin 2014)",
    ),
    "C_m0": Provenance(
        ParamSource.LITERATURE,
        reference="Gentet, Stuart & Clements 2000 Biophys J 79:314 (0.9 uF/cm^2); Hille 2001, Ion Channels of Excitable Membranes, 3rd ed.",
    ),
    "lj_strength": Provenance(
        ParamSource.CALIBRATED,
        notes=(
            "Dimensionless prefactor for the LJ-like wall on Z<0; "
            "calibrated so the wall contribution stays <5% of linear "
            "restoring at |Z| < 0.1·δ_0"
        ),
    ),
    "st_strength": Provenance(
        ParamSource.CALIBRATED,
        notes=(
            "Dimensionless prefactor for the surface-tension wall on Z>0; "
            "default 1.0 doubles asymptotic restoring on the expansion "
            "side, extending the quantitative validity range to ~1 MPa "
            "(Plaksin & Krasovitski 2014 Fig. 2)."
        ),
    ),
}


# ---------------------------------------------------------------------------
# Heimburg-Jackson soliton (thermodynamic membrane) model
# ---------------------------------------------------------------------------

SOLITON_PROVENANCE: dict[str, Provenance] = {
    "T_m": Provenance(
        ParamSource.LITERATURE,
        reference="Heimburg 2007, Thermal Biophysics of Membranes; Heimburg & Jackson 2005, PNAS 102:9790",
        notes=(
            "Membrane melting transition. Biological membranes melt "
            "~10-15°C below body temperature; default 30°C is "
            "representative (lung-surfactant T_m ≈ 30°C, the system "
            "Heimburg & Jackson fit). Must be < 37°C for the "
            "heating-reduces-excitability sign."
        ),
    ),
    "sigma_T": Provenance(
        ParamSource.ESTIMATE,
        reference="Heimburg 2007 (broadened multicomponent-membrane transition)",
        notes=(
            "Half-width of the c_p / compressibility peak, a few °C. "
            "Biological transitions are broad; specific value tuned."
        ),
    ),
    "c_p_max": Provenance(
        ParamSource.ESTIMATE,
        reference="Heimburg 2007 (κ_A^T ∝ Δc_p proportionality)",
        notes=(
            "Dimensionless peak-to-baseline excess-compressibility ratio "
            "encoding Heimburg's c_p ↔ compressibility divergence near "
            "T_m. Tuned amplitude (default 10×)."
        ),
    ),
    "rho": Provenance(
        ParamSource.DERIVED,
        reference="Soft-tissue/water density; matches repo FDTD RHO0 = 1000 kg/m³",
        notes="Taken from the acoustic medium of the run (rho0 = 1000 kg/m^3)",
    ),
    "c": Provenance(
        ParamSource.DERIVED,
        reference="Soft-tissue speed of sound; matches repo FDTD C0 = 1540 m/s",
        notes="Taken from the acoustic medium of the run (c0 = 1540 m/s)",
    ),
    "peak_shape": Provenance(
        ParamSource.ASSUMPTION,
        notes="Gaussian (default) vs Lorentzian shape for the normalised c_p peak.",
    ),
}


# ---------------------------------------------------------------------------
# Calcium / SK pathway
# ---------------------------------------------------------------------------

CALCIUM_DYNAMICS_PROVENANCE: dict[str, Provenance] = {
    "Ca_rest": Provenance(
        ParamSource.LITERATURE,
        reference="Maravall et al. 2000 Biophys J 78:2655 (resting [Ca2+]i 30-60 nM, neocortical pyramidal cells)",
    ),
    "tau_decay": Provenance(
        ParamSource.LITERATURE,
        reference="Mainen & Sejnowski 1996; Borg-Graham 1999",
        notes="Single-pool decay τ in cortical models, range 30-80 ms",
    ),
    "flux_factor": Provenance(
        ParamSource.CALIBRATED,
        notes=(
            "Lumped sub-membrane shell volume × buffering fraction × "
            "1/(z·F). Default 0.5 lies within published "
            "current-to-concentration coupling factors used in cortical "
            "Ca²⁺-pool models."
        ),
    ),
}

SOMATIC_CALCIUM_PROVENANCE: dict[str, Provenance] = {
    # Ca_rest is not listed: SomaticCalcium inherits CalciumDynamics.Ca_rest
    # (see CALCIUM_DYNAMICS_PROVENANCE["Ca_rest"]).
    "tau_ms": Provenance(
        ParamSource.ASSUMPTION,
        notes=(
            "Decay time constant of the slow somatic (bulk) Ca2+ read-out pool, 700 ms, "
            "chosen within the 0.5-1 s range of a somatic bulk pool; not fitted. "
            "Read-out only, with no feedback on the membrane"
        ),
    ),
    "dCa_per_spike": Provenance(
        ParamSource.ASSUMPTION,
        notes=(
            "Free somatic [Ca2+] increment per action potential, 0.08 μM; assumed in the "
            "range quoted for cortical somatic single-spike transients, not fitted"
        ),
    ),
    "Ca_max": Provenance(
        ParamSource.ASSUMPTION,
        notes=(
            "Saturation clamp of the somatic pool, 10 μM (None = unbounded); assumed, not "
            "fitted; inactive for the canonical session (peak below 2 μM)"
        ),
    ),
}

KCA_CHANNEL_PROVENANCE: dict[str, Provenance] = {
    "Ca_half": Provenance(
        ParamSource.LITERATURE, reference="Köhler et al. 1996 (SK channel)"
    ),
    "n_hill": Provenance(
        ParamSource.LITERATURE, reference="Hirschberg et al. 1998"
    ),
    "g_density": Provenance(
        ParamSource.LITERATURE,
        reference="Sah 1996 Trends Neurosci 19:150 (cortical pyramidal SK density 0.1-1.0 mS/cm^2)",
        notes="Default 0.3 mS/cm^2 chosen within the cited range; a modelling choice within a literature bracket, like the AIS Na density",
    ),
    "E_K": Provenance(
        ParamSource.DERIVED, reference="K⁺ Nernst, matches Pospischil RS"
    , notes="Nernst potential of K+ at physiological concentrations ([K]o 3 mM, [K]i 140 mM, 37 C), -90 to -100 mV; not a fitted parameter"),
}


# ---------------------------------------------------------------------------
# TRP thermosensors
# ---------------------------------------------------------------------------

TRPV1_PROVENANCE: dict[str, Provenance] = {
    "T_half": Provenance(
        ParamSource.LITERATURE,
        reference="Caterina et al. 1997, 2000 (heat activation threshold)",
    ),
    "k": Provenance(
        ParamSource.CALIBRATED,
        notes=(
            "Boltzmann steepness chosen so Po(47)/Po(37) > 20 matching "
            "the literature Q10 ≥ 20 floor (Caterina 1997)."
        ),
    ),
    "g_density": Provenance(
        ParamSource.ASSUMPTION,
        notes="Conservative mid-range density when expressed",
    ),
    "E_rev": Provenance(
        ParamSource.LITERATURE,
        reference="Caterina et al. 1997 Nature 389:816 (non-selective cation, Ca2+-preferring, E_rev ~ 0 to +10 mV)",
    ),
}

TRPV4_PROVENANCE: dict[str, Provenance] = {
    "T_half": Provenance(
        ParamSource.LITERATURE,
        reference="Watanabe et al. 2002 (warm-activation threshold)",
        notes="Within the 27-35°C literature range",
    ),
    "k": Provenance(
        ParamSource.CALIBRATED,
        notes="Q10 ≈ 10 around threshold, less steep than TRPV1",
    ),
    "g_density": Provenance(
        ParamSource.ASSUMPTION,
        notes="Conservative mid-range density when expressed",
    ),
    "E_rev": Provenance(
        ParamSource.LITERATURE, reference="Guler et al. 2002 J Neurosci 22:6408 (non-selective cation, E_rev ~ 0 mV)"
    ),
}


# ---------------------------------------------------------------------------
# Astrocytic TRPA1 → Ca²⁺ → glutamate → NMDA relay (Oh et al. 2019)
# ---------------------------------------------------------------------------

TRPA1_PROVENANCE: dict[str, Provenance] = {
    "T_half": Provenance(
        ParamSource.ESTIMATE,
        reference="Oh et al. 2019, Curr Biol 29:3386 (US activates astrocytic TRPA1)",
        notes=(
            "Half-activation tension for TRPA1 mechanogating is not directly "
            "quantified; set between Piezo1 (2.7) and TRAAK (5.0) mN/m as a "
            "best guess pending a dedicated tension-clamp fit."
        ),
    ),
    "delta_A": Provenance(
        ParamSource.ESTIMATE,
        notes="Gating-area sensitivity guessed comparable to other MS channels (~15 nm²).",
    ),
    "g_max": Provenance(
        ParamSource.LITERATURE,
        reference="Karashima et al. 2010; Wang et al. 2008 (TRPA1 single-channel ~tens of pS)",
    ),
    "g_density": Provenance(
        ParamSource.ASSUMPTION,
        notes="Astrocytic TRPA1 surface density; user overrides per preparation.",
    ),
    "E_rev": Provenance(
        ParamSource.LITERATURE,
        reference="Karashima et al. 2010 (non-selective cation, E_rev ≈ 0 mV)",
    ),
    "tau_0": Provenance(
        ParamSource.ASSUMPTION,
        notes="Legacy bell-shape baseline; superseded by tau_act.",
    ),
    "tau_act": Provenance(
        ParamSource.ESTIMATE,
        notes=(
            "Channel gating is fast (ms); the slow relay signature lives in the "
            "downstream Ca/glutamate dynamics, not in TRPA1 gating."
        ),
    ),
    "Ca_permeability_fraction": Provenance(
        ParamSource.LITERATURE,
        reference="Karashima et al. 2010; Wang et al. 2008 (P_Ca/P_Na ≈ 5–8)",
        notes="~15–20% of TRPA1 cation current is Ca²⁺ at physiological gradients.",
    ),
}

ASTROCYTE_RELAY_PROVENANCE: dict[str, Provenance] = {
    "V_astro": Provenance(
        ParamSource.LITERATURE,
        reference="Bazargani & Attwell 2016, Nat Neurosci (astrocyte V_rest ≈ -80 mV)",
        notes="Sets the Ca²⁺ driving force through TRPA1 (non-spiking cell).",
    ),
    "tau_decay": Provenance(
        ParamSource.LITERATURE,
        reference="Bazargani & Attwell 2016; Oh et al. 2019 (astrocytic Ca²⁺ transients decay over 100s of ms)",
        notes="Slow relay: hundreds of ms, not per-carrier-cycle.",
    ),
    "flux_factor": Provenance(
        ParamSource.CALIBRATED,
        notes=(
            "Chosen so the astrocytic Ca²⁺ pool peaks ~1 μM at full US drive "
            "(physiological gliotransmission regime), given TRPA1 g_density, "
            "Ca permeability, and the -80 mV driving force."
        ),
    ),
    "Ca_rest": Provenance(
        ParamSource.LITERATURE,
        reference="Bazargani & Attwell 2016 (resting astrocytic [Ca²⁺] ~50–100 nM)",
    ),
    "glu_Ca_half": Provenance(
        ParamSource.ESTIMATE,
        reference="Parpura & Haydon 2000 (Ca-dependent gliotransmission threshold)",
        notes="Half-release [Ca²⁺] a few hundred nM; tuned for gliotransmission onset.",
    ),
    "glu_n_hill": Provenance(
        ParamSource.ESTIMATE,
        notes="Cooperative Ca²⁺-triggered release; Hill coefficient ~2–3 (vesicular sensor).",
    ),
    "glu_max": Provenance(
        ParamSource.ESTIMATE,
        notes="Peak cleft glutamate scale [μM] from gliotransmission; order-of-magnitude.",
    ),
}

NMDA_PROVENANCE: dict[str, Provenance] = {
    "g_max": Provenance(
        ParamSource.ASSUMPTION,
        notes="Post-synaptic NMDA conductance density; user overrides per cell type.",
    ),
    "K_glu": Provenance(
        ParamSource.LITERATURE,
        reference="Patneau & Mayer 1990 (NMDA glutamate EC50 ~1–3 μM)",
    ),
    "E_rev": Provenance(
        ParamSource.LITERATURE,
        reference="Jahr & Stevens 1990 J Neurosci 10:3178 (E_rev ~ 0 mV); Hille 2001, Ion Channels of Excitable Membranes, 3rd ed.",
    ),
    "Mg": Provenance(
        ParamSource.LITERATURE,
        reference="Jahr & Stevens 1990 (physiological extracellular [Mg²⁺] ≈ 1 mM)",
    ),
    "mg_block_v_scale": Provenance(
        ParamSource.LITERATURE,
        reference="Jahr & Stevens 1990 (voltage-dependent Mg²⁺ block: 16.13 mV, 3.57 mM)",
        notes="g(V) ∝ 1/(1 + [Mg]·exp(-V/16.13)/3.57).",
    ),
    "alpha_NMDA": Provenance(
        ParamSource.CALIBRATED,
        reference="Maneshi et al. 2017, Sci Rep (membrane stretch activates NMDARs); Singh et al. 2012",
        notes=(
            "Max fractional stretch enhancement of NMDA current. Default 1.0 → "
            "current roughly doubles at saturating stretch, matching Maneshi 2017 "
            "(stretch ≈ doubled NMDA current). f_stretch(∞) = 1 + alpha_NMDA."
        ),
    ),
    "T_half_stretch": Provenance(
        ParamSource.ESTIMATE,
        reference="Maneshi et al. 2017",
        notes=(
            "Half-enhancement membrane tension [mN/m]. Mapped to the MS-channel "
            "tension range pending a dedicated stretch-clamp NMDA fit."
        ),
    ),
    "k_stretch": Provenance(
        ParamSource.ESTIMATE,
        notes="Boltzmann slope [mN/m] of the stretch sigmoid; comparable steepness to MS channels.",
    ),
}


# ---------------------------------------------------------------------------
# Mechanosensitive voltage-gated Na⁺/Ca²⁺ shift (Kubanek et al. 2016)
# ---------------------------------------------------------------------------

MECHANO_NAV_PROVENANCE: dict[str, Provenance] = {
    "k_mechano": Provenance(
        ParamSource.CALIBRATED,
        reference="Kubanek et al. 2016, Sci Rep (US shifts Nav/Cav voltage dependence)",
        notes=(
            "Hyperpolarising activation-shift sensitivity [mV per (mN/m)]. "
            "K_MECHANO_NAV = 0.3 is calibrated so a few-mN/m tension yields a "
            "~mV-scale activation shift and a tens-of-percent Na⁺ current change, "
            "the order Kubanek report (not a directly tabulated value). The "
            "neuron field k_mechano_nav defaults to 0 (pathway off) to preserve "
            "the baseline neuron; set it to K_MECHANO_NAV to enable."
        ),
    ),
    "v_half0": Provenance(
        ParamSource.DERIVED,
        reference="Representative mammalian Nav activation midpoint (Hodgkin-Huxley-type)",
        notes="Activation midpoint of the HH/Pospischil Na+ current used by the neuron (m_inf of the RS parameterisation); the mechano-shift is applied relative to it",
    ),
    "k_a": Provenance(
        ParamSource.DERIVED,
        reference="Hodgkin-Huxley-type Nav activation Boltzmann slope",
        notes="Activation Boltzmann slope of the HH/Pospischil Na+ current used by the neuron",
    ),
    "v_eval": Provenance(
        ParamSource.ASSUMPTION,
        notes="Near-threshold reference voltage [mV] at which nav_current_gain reads the open fraction.",
    ),
}


# ---------------------------------------------------------------------------
# Mechanosensitive presynaptic release + relay synapse (Tyler 2012)
# ---------------------------------------------------------------------------

MECHANO_SYNAPSE_PROVENANCE: dict[str, Provenance] = {
    "Pr_baseline": Provenance(
        ParamSource.LITERATURE,
        reference="Tyler 2008; central-synapse release probability range 0.1–0.3",
        notes="Resting vesicle release probability at zero tension.",
    ),
    "Pr_max": Provenance(
        ParamSource.ESTIMATE,
        reference="Tyler 2012 (mechanical perturbation raises release probability)",
        notes="Saturating release probability at high presynaptic tension.",
    ),
    "T_half_pre": Provenance(
        ParamSource.ESTIMATE,
        reference="Tyler 2012",
        notes="Half-modulation presynaptic tension [mN/m]; in the MS-channel range.",
    ),
    "k_pre": Provenance(
        ParamSource.ESTIMATE,
        notes="Boltzmann slope [mN/m] of the release-probability sigmoid.",
    ),
    "glu_scale": Provenance(
        ParamSource.CALIBRATED,
        notes=(
            "Sustained cleft glutamate [μM] at full release; calibrated with g_ampa "
            "so the resting (baseline-Pr) state is sub-threshold and US-driven "
            "release brings the post neuron to firing."
        ),
    ),
    "tau_glu": Provenance(
        ParamSource.LITERATURE,
        reference="Clements et al. 1992 Science 258:1498 (cleft glutamate clears within ~1-2 ms)",
        notes="First-order glutamate pool decay [ms].",
    ),
    "g_ampa": Provenance(
        ParamSource.CALIBRATED,
        notes="Post-synaptic AMPA conductance density [mS/cm²]; calibrated with glu_scale.",
    ),
    "E_ampa": Provenance(
        ParamSource.LITERATURE,
        reference="Hille 2001, Ion Channels of Excitable Membranes, 3rd ed. (AMPA non-selective cation, E_rev ~ 0 mV)",
    ),
    "K_ampa": Provenance(
        ParamSource.LITERATURE,
        reference="Patneau & Mayer 1990 J Neurosci 10:2385 (AMPA glutamate EC50 ~ 0.3-0.5 mM)",
        notes="AMPA glutamate half-saturation [μM].",
    ),
}


# ---------------------------------------------------------------------------
# Multi-compartment geometry
# ---------------------------------------------------------------------------

CORTICAL_3COMP_PROVENANCE: dict[str, Provenance] = {
    "g_Na_dend": Provenance(
        ParamSource.LITERATURE, reference="Stuart & Sakmann 1995"
    ),
    "g_Na_soma": Provenance(
        ParamSource.LITERATURE, reference="Mainen & Sejnowski 1996"
    ),
    "g_Na_ais": Provenance(
        ParamSource.LITERATURE,
        reference="Kole 2008 (lower end of 2000-5000 mS/cm² range)",
        notes="Default chosen for RK4 stability at dt = 1 μs; user can raise to 5000",
    ),
    "area_ratios": Provenance(
        ParamSource.ASSUMPTION,
        notes="Pinsky-Rinzel-style lumped ratios A_d:A_s:A_a = 5:1:0.05",
    ),
    "g_dend_soma": Provenance(
        ParamSource.ASSUMPTION,
        notes="Lumped axial conductance, mS/cm² of soma",
    ),
    "g_soma_ais": Provenance(
        ParamSource.ASSUMPTION,
        notes="Tighter coupling than dend-soma, reflecting AIS proximity",
    ),
    "dendrite_piezo_scale": Provenance(
        ParamSource.LITERATURE,
        reference="Lewis & Grandl 2015; Dalghi 2019",
        notes="Qualitative dendrite-heavy Piezo1 expression",
    ),
}


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, dict[str, Provenance]] = {
    "Piezo1": PIEZO1_PROVENANCE,
    "TRAAK": TRAAK_PROVENANCE,
    "TREK-1": TREK1_PROVENANCE,
    "TREK-2": TREK2_PROVENANCE,
    "TRPV1": TRPV1_PROVENANCE,
    "TRPV4": TRPV4_PROVENANCE,
    "strain_to_tension": STRAIN_TO_TENSION_PROVENANCE,
    "NICEModel": NICE_PROVENANCE,
    "SolitonModel": SOLITON_PROVENANCE,
    "CalciumDynamics": CALCIUM_DYNAMICS_PROVENANCE,
    "KCaChannel": KCA_CHANNEL_PROVENANCE,
    "SomaticCalcium": SOMATIC_CALCIUM_PROVENANCE,
    "cortical_pyramidal_3comp": CORTICAL_3COMP_PROVENANCE,
    "TRPA1": TRPA1_PROVENANCE,
    "AstrocyteRelay": ASTROCYTE_RELAY_PROVENANCE,
    "NMDAReceptor": NMDA_PROVENANCE,
    "MechanoNav": MECHANO_NAV_PROVENANCE,
    "MechanoSynapse": MECHANO_SYNAPSE_PROVENANCE,
}


def provenance_for(name: str) -> dict[str, Provenance]:
    """Return the provenance map for a named channel / pathway.

    Parameters
    ----------
    name : str
        One of the keys in the registry (case-sensitive). Use channel
        ``.name`` field for mechanosensitive / TRP channels.

    Raises
    ------
    KeyError
        If *name* is not registered.
    """
    if name not in _REGISTRY:
        msg = (
            f"No provenance registered for '{name}'. "
            f"Available: {sorted(_REGISTRY)}"
        )
        raise KeyError(msg)
    return _REGISTRY[name]


def registered_names() -> list[str]:
    """Return the sorted list of names with provenance entries."""
    return sorted(_REGISTRY)


def format_provenance_report(name: str) -> str:
    """Return a human-readable provenance summary for *name*.

    The output groups parameters by source category and lists each
    parameter with its reference and notes. Useful for paper
    appendices and uncertainty audits.
    """
    pmap = provenance_for(name)
    by_src: dict[ParamSource, list[tuple[str, Provenance]]] = {s: [] for s in ParamSource}
    for k, p in pmap.items():
        by_src[p.source].append((k, p))

    lines = [f"Provenance report: {name}", "=" * (len(name) + 20)]
    for src in (
        ParamSource.LITERATURE,
        ParamSource.CALIBRATED,
        ParamSource.ASSUMPTION,
        ParamSource.ESTIMATE,
        ParamSource.DERIVED,
    ):
        items = by_src[src]
        if not items:
            continue
        lines.append("")
        lines.append(f"  [{src.value.upper()}]")
        for param_name, prov in sorted(items):
            head = f"    {param_name:<28}"
            ref_part = f" ref: {prov.reference}" if prov.reference else ""
            note_part = f" — {prov.notes}" if prov.notes else ""
            lines.append(head + ref_part + note_part)
    return "\n".join(lines)
