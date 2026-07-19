"""Validation-report config for the Halle CTX-500 bowl simulation
re-aimed at the left anterior cingulate cortex (dACC region, ACC_left,
MNI RAS = (-7, 35, 20)). This matches Yaakub 2023 (Nat Commun 14, 5318)
which sonicated the dACC (60 mm focal depth) and PCC (69 mm depth) as
the two canonical deep-cortical targets.

Differs from halle_caudate.py in placement only — pulse train, transducer,
medium, and neural coupling are unchanged. Placement uses the auto-pick
mode of halle_placement.place_on_skull (scalp_site=None) which scans all
outer-skull vertices for the best perpendicular aim. The nearest 10-20
cap landmark to the auto-picked apex is AF3 (~28 mm offset, F3 at ~37 mm).
"""

from ultraneuron.paths import halle_skull_nrrd

SIM_NAME = "halle_dacc"
SIM_TITLE = "Halle CTX-500 Transcranial Bowl @ 500 kHz (auto-aimed at left dACC)"

# --- Grid / timing ---
F0 = 500e3
C0 = 1540.0
DOMAIN_SIZE = (130e-3, 80e-3, 80e-3)
PPW = 10
CFL = 0.2
DURATION = DOMAIN_SIZE[0] / C0 * 2

# --- Source: TIPS / NeuroFUS CTX-500 spec bowl ---
SOURCE = {
    "type": "tips_bowl",
    "n_layers": 3,
    "roc_m": 63e-3,
    "outer_radius_m": 32e-3,
    "inner_radius_m": 0.0,
    "n_elements": 4,
    "p0_pa": 0.4e6,  # placeholder — overwritten by post-FDTD recalibration
    "ncycles": 2,
    "drop_off": 2,
}

# --- Placement: auto-pick perpendicular aim at ACC_left (dACC) ---
PLACEMENT = {
    "scalp_site": None,
    "target_name": "ACC_left",
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
# 500 kHz CW carrier gated as 20 ms rectangular tone-bursts repeating
# at 5 Hz PRF (= 10 % macro duty), 80 s session.
NEUROMOD_SEQUENCE = {
    "profile": "yaakub_human",
    "envelope": "rectangular CW tone-burst (no nested sub-bursts)",
    "carrier_freq_hz": 500e3,
    "ncycles_per_pulse": 2,
    "tone_burst_duration_s": 20e-3,
    "pulse_repetition_freq_hz": 5.0,
    "pulse_period_s": 0.2,
    "duty_cycle": 0.10,
    "isi_s": 0.180,
    "session_duration_s": 80.0,
    "n_pulses_per_session": 400,
    "arf_push_window_s": 20e-3,
    "i_sppa_w_cm2": 30.0,
}

SIM_DATA_DIR = "data/halle_dacc"
OUT_DIR = "data/halle_dacc/validation_report"

# --- Neural coupling (inherited from halle_caudate) ---
NEURAL_COUPLING_GAIN = 5e3
NEURAL_COUPLING_FACTOR = 1000.0
NEURAL_K_A = 0.25

NEURAL_PIEZO1_G_DENSITY = 0.10
NEURAL_USE_TREK1 = True
NEURAL_TREK1_G_DENSITY = 0.04
