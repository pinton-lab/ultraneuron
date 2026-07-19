"""Validation-report config for the Halle CTX-500 bowl simulation.

A config is a plain Python module that exposes module-level constants the
builder reads. To validate a different sim, copy this file and edit the
constants — the builder + template stay unchanged.
"""

from ultraneuron.paths import halle_skull_nrrd

SIM_NAME = "halle_bowl"
SIM_TITLE = "Halle CTX-500 Transcranial Bowl @ 500 kHz"

# --- Grid / timing ---
F0 = 500e3
C0 = 1540.0
DOMAIN_SIZE = (65e-3, 66e-3, 66e-3)   # (x=depth, y=lateral, z=elevational) in m
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

# --- Placement: F3 -> DLPFC_left ---
PLACEMENT = {
    "scalp_site": "F3",
    "target_name": "DLPFC_left",
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

# --- Neuromodulation sequence (Yaakub 2023 human TUS protocol) ---
# Yaakub SN et al., Brain Stimulation 2023: 500 kHz CW carrier gated as
# 30 ms rectangular tone-bursts repeating at 5 Hz PRF (= 15 % macroscopic
# duty cycle), administered for 80 s total. No nested intra-burst
# modulation — the carrier is continuous within each 30 ms ON window.
# Per-push FDTD captures a single tone-burst.
NEUROMOD_SEQUENCE = {
    "profile": "yaakub_human",
    "envelope": "rectangular CW tone-burst (no nested sub-bursts)",
    "carrier_freq_hz": 500e3,
    "ncycles_per_pulse": 2,                 # used by per-push FDTD source only
    "tone_burst_duration_s": 30e-3,         # 30 ms continuous tone-burst
    "pulse_repetition_freq_hz": 5.0,        # 5 Hz pulse-train rate
    "pulse_period_s": 0.2,                  # 1 / PRF
    "duty_cycle": 0.15,                     # 30 ms ON / 200 ms period
    "isi_s": 0.170,                         # 200 ms - 30 ms
    "session_duration_s": 80.0,             # 80 s session (400 tone-bursts)
    "n_pulses_per_session": 400,            # session_s × PRF
    "arf_push_window_s": 30e-3,             # FDTD ARF integration window = one tone-burst
    "i_sppa_w_cm2": 30.0,                   # in-brain target ISPPA (Yaakub 2023)
}

# Where the sim's saved arrays live (propagation_map.npy, snapshots_*.npy, etc.)
SIM_DATA_DIR = "data/halle_bowl"

# Output destination (relative to repo root)
OUT_DIR = "data/halle_bowl/validation_report"
