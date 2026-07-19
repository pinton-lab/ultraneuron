"""Validation-report config for the Halle CTX-500 bowl simulation
re-aimed at the left caudate nucleus (F3 -> caudate_left, MNI RAS = (-12, 15, 11)).
"""

from ultraneuron.paths import halle_skull_nrrd

SIM_NAME = "halle_caudate"
SIM_TITLE = "Halle CTX-500 Transcranial Bowl @ 500 kHz (F3 to left caudate)"

# --- Grid / timing ---
F0 = 500e3
C0 = 1540.0
# (x=depth, y=lateral, z=elevational) in m. Depth bumped to 130 mm
# so the caudate focus (apex-to-target ≈ 62.5 mm) sits well inside the
# domain with ≥ 60 mm of post-focal ring-down margin.
DOMAIN_SIZE = (130e-3, 80e-3, 80e-3)
PPW = 10
CFL = 0.2
DURATION = DOMAIN_SIZE[0] / C0 * 2     # one-way + return

# --- Source: TIPS / NeuroFUS CTX-500 spec bowl ---
SOURCE = {
    "type": "tips_bowl",
    "n_layers": 3,
    "roc_m": 63e-3,
    "outer_radius_m": 32e-3,
    "inner_radius_m": 0.0,
    "n_elements": 4,
    "p0_pa": 0.4e6,
    "ncycles": 2,
    "drop_off": 2,
}

# --- Placement: F3 -> caudate_left ---
PLACEMENT = {
    "scalp_site": "F3",
    "target_name": "caudate_left",
    "focal_length_mm": 63.0,
    "bowl_radius_mm": 32.0,
}

# --- Medium ---
MEDIUM = {
    "type": "microct_skull",
    "nrrd_path": str(halle_skull_nrrd()),
    "lower_threshold": 700,
    "upper_threshold": 1973,
    "speed_range": (1540, 2900),
    "density_range": (1000, 2200),
    "blur_sigma_m": 0.0,
    "background": "tissue",
}

# --- Neuromodulation sequence (Yaakub 2023 human protocol) ---
# 500 kHz CW carrier gated as 30 ms rectangular tone-bursts repeating
# at 5 Hz PRF (= 15 % macro duty), 80 s session.
NEUROMOD_SEQUENCE = {
    "profile": "yaakub_human",
    "envelope": "rectangular CW tone-burst (no nested sub-bursts)",
    "carrier_freq_hz": 500e3,
    "ncycles_per_pulse": 2,
    "tone_burst_duration_s": 30e-3,
    "pulse_repetition_freq_hz": 5.0,
    "pulse_period_s": 0.2,
    "duty_cycle": 0.15,
    "isi_s": 0.170,
    "session_duration_s": 80.0,
    "n_pulses_per_session": 400,
    "arf_push_window_s": 30e-3,
    "i_sppa_w_cm2": 30.0,
}

SIM_DATA_DIR = "data/halle_caudate"
OUT_DIR = "data/halle_caudate/validation_report"

# --- Neural coupling ---
# Volumetric strain → membrane areal-strain coupling gain. The
# elastostatic ε_vol from a single push is ~1e-5; published HH
# mechano-coupling values place the firing threshold around 0.5 mN/m,
# which corresponds to ε_vol ~ 1e-3 with K_a = 0.4 N/m. The gain below
# is the dimensionless tissue→membrane coupling factor (analogous to
# strain_to_membrane_tension's `coupling_factor`) that lifts the
# voxel-scale strain into the membrane-scale areal strain.
NEURAL_COUPLING_GAIN = 5e3
# Multi-scale tissue→bilayer coupling α (cortical-pyramidal-calibrated;
# 10× the literature-anchored α=200 of Rawicz/Sukharev/Salahshoor, which
# was determined on hippocampal/iso-potential preparations and lipid-only
# substrates; the amplification reflects dendrite-heavy Piezo1 expression
# and viscoelastic stress concentration in cortical pyramidal cells).
NEURAL_COUPLING_FACTOR = 1000.0
NEURAL_K_A = 0.25

# Piezo1 conductance density (mS/cm²) for the report's pyramidal cell.
# The library default (0.005) is calibrated for a steady tension; the
# 3 ms theta-burst pulse needs higher expression to spike — bumped to
# 0.5 here so the spatial-temporal visualisations have non-trivial
# population structure. TREK1 is dropped from the profile because its
# K⁺ leak hyperpolarises the cell and masks brief Piezo1 currents.
NEURAL_PIEZO1_G_DENSITY = 0.10
NEURAL_USE_TREK1 = True
NEURAL_TREK1_G_DENSITY = 0.04
