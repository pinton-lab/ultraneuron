"""Temperature-gated TRP channels (TRPV1, TRPV4).

These channels open in response to absolute membrane temperature, with
very steep dose-response curves (Q10 ≥ 20 near the activation threshold
for TRPV1).  At Legon-style ultrasound focal heating (<0.1 K rise above
37°C body temperature) TRPV1 (T_half ≈ 43°C) is essentially closed.
TRPV4 (T_half ≈ 30°C) is partially open at body temperature and acts as
a tonic depolarising leak; high-intensity protocols (Lee 2016, Verhagen
2019) producing >5 K rises can push TRPV1 into significant activation.

Both channels use the same Boltzmann form:

    P_o(T) = 1 / (1 + exp(-(T - T_half) / k))

with literature-derived ``T_half`` and a steepness ``k`` chosen so the
curve matches reported Q10 values (small k → steep Q10).

This module deliberately omits TRPV4's mild shear-stress sensitivity:
ultrasound shear-strain is already captured by the Piezo1/K2P channels
in the channel library, and adding a parallel shear-coupling here would
double-count.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class TRPThermosensorChannel:
    """Boltzmann temperature-gated TRP channel.

    Parameters
    ----------
    name : str
        Identifier (``"TRPV1"``, ``"TRPV4"``).
    T_half : float
        Half-activation temperature [°C].
    k : float
        Boltzmann steepness [°C]. Smaller k → steeper Q10. For TRPV1
        (Q10 ≈ 20), k ≈ 1.5°C; for TRPV4 (Q10 ≈ 10), k ≈ 3°C.
    g_density : float
        Maximum conductance density [mS/cm²].
    E_rev : float
        Reversal potential [mV]. Non-selective cation channels: ~0–20 mV.
    description : str
        Optional provenance note.
    """

    name: str
    T_half: float  # °C
    k: float       # °C
    g_density: float  # mS/cm²
    E_rev: float = 0.0  # mV
    description: str = ""

    def open_probability(
        self,
        T: NDArray[np.floating] | float,
    ) -> NDArray[np.floating] | float:
        """Boltzmann temperature activation, in [0, 1]."""
        T_arr = np.asarray(T, dtype=np.float64)
        # Stable sigmoid for ((T - T_half)/k)
        x = (T_arr - self.T_half) / self.k
        # 1 / (1 + exp(-x)); use the same numerical pattern as channels._safe_sigmoid
        out = np.where(
            x >= 0,
            1.0 / (1.0 + np.exp(-np.minimum(x, 500.0))),
            np.exp(np.maximum(x, -500.0)) / (1.0 + np.exp(np.maximum(x, -500.0))),
        )
        return float(out) if out.ndim == 0 else out

    def current(
        self,
        T: NDArray[np.floating] | float,
        V_m: NDArray[np.floating] | float,
    ) -> NDArray[np.floating] | float:
        """Macroscopic TRP current density [μA/cm²].

        Sign convention matches MechanosensitiveChannel.current:
        positive = inward (depolarising for cations with E_rev > V_m).
        """
        return self.g_density * self.open_probability(T) * (self.E_rev - V_m)


# ---------------------------------------------------------------------------
# Literature presets
# ---------------------------------------------------------------------------

TRPV1 = TRPThermosensorChannel(
    name="TRPV1",
    T_half=43.0,  # °C — Caterina et al. 1997, 2000
    k=1.5,        # steep Q10 ≈ 20 near threshold
    g_density=0.05,  # mS/cm² — variable; conservative literature mid-range
    E_rev=10.0,   # mV — slightly Ca²⁺-preferring non-selective cation
    description="Heat-activated cation channel, capsaicin receptor. Caterina 1997.",
)

TRPV4 = TRPThermosensorChannel(
    name="TRPV4",
    T_half=30.0,  # °C — Watanabe 2002 (warm-activation 24–34°C range; midpoint)
    k=3.0,        # Q10 ≈ 10 around threshold; less steep than TRPV1
    g_density=0.02,  # mS/cm² — lower than TRPV1 typical density
    E_rev=0.0,    # mV — non-selective cation
    description="Warm-activated cation channel, partially open at body temp. Watanabe 2002.",
)
