"""Unit conventions and conversions for simulation outputs.

JSON-summary keys follow the flat ``<name>_<unit>`` form, where
``<unit>`` is an SI form rendered with underscores in place of slashes
and carets:

* ``peak_dT_K``                   --- temperature delta in kelvin
* ``peak_pressure_Pa``            --- pressure in pascal
* ``tension_peak_N_per_m``        --- surface tension in newton per metre
* ``Q_peak_W_per_m3``             --- volumetric heating rate
* ``current_density_A_per_m2``    --- current density

Biophysics-convention units (``mV``, ``uA_per_cm2``, ``mN_per_m``,
``ms``, ``uF_per_cm2``) are allowed, but only inside a
``metrics.biophysics.*`` namespace where the consumer (the HH solver)
natively uses those units.

The :func:`to_SI` / :func:`from_SI` helpers convert between the two
conventions at the boundary; :func:`validate_key` checks that a key
matches the ``<name>_<unit>`` pattern at JSON-write time.
"""

from __future__ import annotations

import re
from typing import Final

# ---------------------------------------------------------------------
# Conversion factors --- multiply value-in-biophysics by this to get SI
# ---------------------------------------------------------------------

_TO_SI: Final[dict[str, tuple[str, float]]] = {
    # Voltage
    "mV": ("V", 1e-3),
    # Membrane capacitance
    "uF_per_cm2": ("F_per_m2", 1e-2),  # 1 μF/cm² = 0.01 F/m²
    # Conductance density
    "mS_per_cm2": ("S_per_m2", 10.0),  # 1 mS/cm² = 10 S/m²
    # Current density
    "uA_per_cm2": ("A_per_m2", 1e-2),
    # Membrane tension / surface tension
    "mN_per_m": ("N_per_m", 1e-3),
    # Time
    "ms": ("s", 1e-3),
    "us": ("s", 1e-6),
    # Concentration (intracellular Ca²⁺ pool uses μM)
    "uM": ("mol_per_m3", 1e-3),  # 1 μM = 1e-6 mol/L = 1e-3 mol/m³
    # Pressure
    "kPa": ("Pa", 1e3),
    "MPa": ("Pa", 1e6),
    # Length
    "mm": ("m", 1e-3),
    "um": ("m", 1e-6),
    "nm": ("m", 1e-9),
    # Frequency
    "kHz": ("Hz", 1e3),
    "MHz": ("Hz", 1e6),
    # Temperature: K stays K; °C is an offset, handled separately by callers.
}


def to_SI(value: float, biophysics_unit: str) -> float:
    """Convert *value* from a biophysics unit to its canonical SI unit.

    Parameters
    ----------
    value : float
        Numerical value in ``biophysics_unit``.
    biophysics_unit : str
        Source unit (e.g. ``"mV"``, ``"mS_per_cm2"``).

    Returns
    -------
    float
        Same physical quantity expressed in SI.

    Raises
    ------
    KeyError
        If ``biophysics_unit`` is not in the conversion table.
    """
    if biophysics_unit not in _TO_SI:
        msg = (
            f"Unknown biophysics unit '{biophysics_unit}'. "
            f"Known: {sorted(_TO_SI)}"
        )
        raise KeyError(msg)
    _, factor = _TO_SI[biophysics_unit]
    return value * factor


def from_SI(value: float, target_unit: str) -> float:
    """Convert *value* from SI to a target biophysics unit.

    Inverse of :func:`to_SI`.
    """
    if target_unit not in _TO_SI:
        msg = (
            f"Unknown target unit '{target_unit}'. "
            f"Known: {sorted(_TO_SI)}"
        )
        raise KeyError(msg)
    _, factor = _TO_SI[target_unit]
    return value / factor


# ---------------------------------------------------------------------
# Key-pattern validator
# ---------------------------------------------------------------------

# Allowed unit forms that may appear after the trailing underscore in a
# JSON key. Matches both SI (Pa, K, m, s, Hz, ...) and biophysics
# (mV, uA_per_cm2, mN_per_m, ms, uM, uF_per_cm2, mS_per_cm2).
_UNIT_TOKEN = (
    r"(?:"
    # SI base + selected prefixed
    r"Pa|kPa|MPa|"
    r"K|"  # kelvin (no °C — use K)
    r"m|mm|um|nm|"
    r"s|ms|us|"
    r"Hz|kHz|MHz|"
    r"V|mV|"
    r"A|mA|uA|"
    r"S|mS|"
    r"F|uF|"
    r"W|"
    r"N|mN|"
    r"mol|uM|"
    # Compound: <unit>_per_<unit><pow>
    r"|[a-zA-Z]+_per_[a-zA-Z]+[0-9]*"
    # Pure numeric / dimensionless markers
    r"|count|fraction|ratio|index"
    r")"
)
# Full key: lowercase or mixed-case name, underscore, unit token.
_KEY_RE = re.compile(rf"^[A-Za-z][A-Za-z0-9_]*_{_UNIT_TOKEN}$")


def validate_key(key: str) -> None:
    """Raise ``ValueError`` if ``key`` does not match ``<name>_<unit>``.

    Allowed: ``peak_dT_K``, ``Q_peak_W_per_m3``, ``tension_peak_N_per_m``,
    ``firing_count``, ``coupling_alpha_ratio``.

    Disallowed: ``peakDT``, ``peak_dT`` (no unit), ``my-key_K`` (hyphen),
    ``peak dT_K`` (space).
    """
    if not isinstance(key, str) or not key:
        msg = f"Key must be a non-empty string, got {key!r}."
        raise ValueError(msg)
    if not _KEY_RE.match(key):
        msg = (
            f"Key '{key}' does not match the '<name>_<unit>' convention. "
            "Examples of valid keys: 'peak_dT_K', 'tension_peak_N_per_m', "
            "'Q_peak_W_per_m3', 'firing_count'."
        )
        raise ValueError(msg)
