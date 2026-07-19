"""Intracellular Ca²⁺ pool and Ca²⁺-gated K⁺ channels.

Provides the second inhibitory pathway distinct from K2P (TRAAK/TREK):
Piezo1 admits a Ca²⁺ flux which charges a single-pool intracellular
[Ca²⁺]_i, and a small-conductance Ca²⁺-activated K⁺ channel (SK) opens
in proportion to [Ca²⁺]_i, producing a hyperpolarising counter-current
that can convert sustained Piezo1 activation into net inhibition (the
mechanism candidate M5 for Legon-style high-pressure → inhibition
outcomes).

Single-pool single-time-constant model:

    d[Ca²⁺]_i/dt = -k_flux · I_Ca - ([Ca²⁺]_i - [Ca²⁺]_rest) / τ_Ca

where I_Ca is the inward Ca²⁺ current density [μA/cm²] (positive into
the cell, but we follow the channel-current sign convention where
inward = positive in our HH equations) and k_flux converts current
density to concentration rate [μM·cm²/(μA·ms)] by lumping membrane
surface-to-volume ratio, valence (z = 2), Faraday constant, and
fraction not buffered:

    k_flux = 1 / (z · F · d_shell)
           ≈ 5.18e-3 · (1 - β) / d_shell[μm]   [μM·cm²/(μA·ms)]

with d_shell ≈ 0.1 μm (sub-membrane Ca²⁺ shell) and buffering fraction
β ≈ 0.99 (only ~1% of incoming Ca²⁺ is free, the rest binds to
buffers). Default ``flux_factor`` = 0.5 μM·cm²/(μA·ms) — within the
range of values used in published cortical models (Mainen-Sejnowski
1996, Borg-Graham 1999).

The Ca²⁺ pool is modelled per-compartment (one [Ca²⁺]_i value per
compartment per voxel).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class CalciumDynamics:
    """Single-pool intracellular Ca²⁺ tracker.

    Parameters
    ----------
    Ca_rest : float
        Resting intracellular Ca²⁺ concentration [μM]. Default 0.05 μM
        (50 nM, typical for cortical pyramidal cells).
    tau_decay : float
        First-order decay time constant back to ``Ca_rest`` [ms].
        Default 50 ms (lumps SERCA pump, plasma-membrane extrusion,
        and slow diffusion to bulk cytoplasm).
    flux_factor : float
        Conversion from inward Ca²⁺ current [μA/cm²] to concentration
        rate [μM/ms]. Default 0.5 — a typical lumped value for a
        ~100 nm sub-membrane shell with ~99% buffering.
    """

    Ca_rest: float = 0.05  # μM (50 nM)
    tau_decay: float = 50.0  # ms
    flux_factor: float = 0.5  # μM·cm²/(μA·ms)


@dataclass
class KCaChannel:
    """Small-conductance Ca²⁺-activated K⁺ channel (SK-like).

    Hill-type activation by intracellular Ca²⁺:

        P_o([Ca]) = [Ca]^n / ([Ca]^n + Ca_half^n)

    Current density (per compartment):

        I_KCa = g_density · P_o · (E_K - V)

    Parameters
    ----------
    name : str
        Identifier (e.g. ``"SK"``, ``"BK"``).
    Ca_half : float
        Half-activation [Ca²⁺]_i [μM]. SK channels: ~0.5 μM
        (Köhler 1996); BK channels: voltage-dependent and not modelled
        here.
    n_hill : float
        Hill coefficient — activation curve steepness. SK: ~4
        (Köhler 1996, Hirschberg 1998).
    g_density : float
        Maximum conductance density [mS/cm²]. SK in cortical pyramidal
        cells: 0.1–1.0 mS/cm² depending on layer (Sah 1996, Stocker 2004).
    E_K : float
        K⁺ reversal potential [mV]. Default -100 (matches Pospischil RS).
    description : str
        Optional provenance note.
    """

    name: str = "SK"
    Ca_half: float = 0.5  # μM (Köhler 1996)
    n_hill: float = 4.0
    g_density: float = 0.3  # mS/cm² — mid-range cortical SK density
    E_K: float = -100.0  # mV
    description: str = "Small-conductance Ca²⁺-activated K⁺ channel (SK)"
    # Per-compartment density scale (parallels MechanosensitiveChannel API)
    # Set at neuron-construction time when the channel is attached to a
    # specific 3-compartment geometry; left unfilled here.
    compartment_scale: list[float] = field(default_factory=list)

    def open_probability(
        self,
        Ca_i: NDArray[np.floating] | float,
    ) -> NDArray[np.floating] | float:
        """Hill-type Ca²⁺ activation."""
        Ca = np.asarray(Ca_i, dtype=np.float64)
        # Avoid 0**negative for very low [Ca]; clamp to tiny positive
        Ca_safe = np.maximum(Ca, 1e-9)
        x_n = Ca_safe**self.n_hill
        return x_n / (x_n + self.Ca_half**self.n_hill)
