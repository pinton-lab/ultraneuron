"""Glutamate-gated NMDA receptor with voltage-dependent Mg²⁺ block.

This is the post-synaptic readout of the astrocytic ultrasound-
neuromodulation relay (Oh et al. 2019): ultrasound opens astrocytic
TRPA1 → astrocytic Ca²⁺ rises → Ca²⁺-triggered glutamate release →
glutamate binds neuronal NMDA receptors → NMDA current depolarises the
neuron. The NMDA conductance is the product of a glutamate-binding
saturation term and the Jahr–Stevens voltage-dependent Mg²⁺ unblock:

    g_NMDA(V, [glu]) = g_max · [glu]/([glu] + K_glu)
                              · 1 / (1 + [Mg]·exp(-V/16.13)/3.57)

Blocking the receptor (the AP5 condition in Oh 2019) is modelled by
setting ``g_max = 0``: glutamate release upstream is unaffected, but no
neuronal excitation results.

Mechanosensitivity (Maneshi et al. 2017, Sci Rep; Singh et al. 2012):
membrane stretch shifts NMDA-receptor gating, raising the open
probability / current at fixed glutamate and voltage. This is captured
here by a multiplicative, tension-dependent enhancement factor
``f_stretch(T)`` applied on top of the glutamate-binding × Mg²⁺-unblock
product:

    g_NMDA(V, [glu], T) = g_max · [glu]/([glu] + K_glu)
                              · 1 / (1 + [Mg]·exp(-V/16.13)/3.57)
                              · f_stretch(T)

``f_stretch`` is a saturating sigmoid normalised so that
``f_stretch(0) = 1`` exactly (zero tension → baseline NMDA, preserving
the astrocyte-relay behaviour) and ``f_stretch(∞) = 1 + alpha_NMDA``.
With the default ``alpha_NMDA ≈ 1`` the current roughly doubles at
saturating stretch, matching Maneshi 2017.

Units follow the repo convention:
    - Voltage: mV
    - Conductance: mS/cm²
    - Current density: μA/cm² (positive = inward / depolarising)
    - Glutamate: μM
    - Membrane tension: mN/m
    - [Mg²⁺]: mM
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class NMDAReceptor:
    """Glutamate-gated NMDA receptor conductance (Jahr–Stevens Mg²⁺ block).

    Parameters
    ----------
    name : str
        Identifier.
    g_max : float
        Maximum conductance density when fully bound and unblocked
        [mS/cm²]. Set to 0 to model NMDA antagonism (AP5).
    K_glu : float
        Half-saturating glutamate concentration (EC50) [μM].
        NMDA EC50 ≈ 1–3 μM (Patneau & Mayer 1990).
    E_rev : float
        Reversal potential [mV]. NMDA is a non-selective cation channel,
        ~0 mV.
    Mg : float
        Extracellular Mg²⁺ concentration [mM]. Physiological ≈ 1 mM
        (Jahr & Stevens 1990).
    alpha_NMDA : float
        Maximum fractional stretch-enhancement of NMDA current
        (dimensionless). ``f_stretch(∞) = 1 + alpha_NMDA``. Default 1.0
        → current doubles at saturating stretch (Maneshi et al. 2017).
        Set to 0 to disable mechanosensitivity (pure glutamate/voltage
        NMDA).
    T_half_stretch : float
        Half-enhancement membrane tension [mN/m] — the tension at which
        the stretch sigmoid reaches its half-maximum. Default 5.0.
    k_stretch : float
        Boltzmann slope of the stretch sigmoid [mN/m]. Default 1.5.
    """

    name: str = "NMDA"
    g_max: float = 0.5      # mS/cm²
    K_glu: float = 2.0      # μM
    E_rev: float = 0.0      # mV
    Mg: float = 1.0         # mM
    alpha_NMDA: float = 1.0       # max fractional stretch enhancement (Maneshi 2017)
    T_half_stretch: float = 5.0   # mN/m
    k_stretch: float = 1.5        # mN/m

    def glutamate_binding(
        self,
        glutamate: NDArray[np.floating] | float,
    ) -> NDArray[np.floating] | float:
        """Saturating glutamate-bound fraction in [0, 1]."""
        glu = np.maximum(np.asarray(glutamate, dtype=np.float64), 0.0)
        out = glu / (glu + self.K_glu)
        return float(out) if out.ndim == 0 else out

    def stretch_factor(
        self,
        tension: NDArray[np.floating] | float,
    ) -> NDArray[np.floating] | float:
        r"""Mechanosensitive stretch-enhancement factor (Maneshi 2017).

        .. math::

            f_\mathrm{stretch}(T) = 1 + \alpha_\mathrm{NMDA}\,
                \frac{\sigma\!\big(\tfrac{T - T_{1/2}}{k}\big)
                      - \sigma\!\big(\tfrac{-T_{1/2}}{k}\big)}
                     {1 - \sigma\!\big(\tfrac{-T_{1/2}}{k}\big)}

        with :math:`\sigma` the logistic function. The normalisation
        guarantees ``f_stretch(0) = 1`` exactly (baseline NMDA at zero
        tension) and ``f_stretch(T → ∞) = 1 + alpha_NMDA``. Monotonic
        non-decreasing in tension. Negative tensions are clamped to 0.

        Parameters
        ----------
        tension : ndarray or float
            Membrane tension [mN/m].

        Returns
        -------
        ndarray or float
            Enhancement factor ``≥ 1``.
        """
        T = np.maximum(np.asarray(tension, dtype=np.float64), 0.0)

        def _sigmoid(x: NDArray[np.floating]) -> NDArray[np.floating]:
            return np.where(
                x >= 0,
                1.0 / (1.0 + np.exp(-np.minimum(x, 500.0))),
                np.exp(np.maximum(x, -500.0)) / (1.0 + np.exp(np.maximum(x, -500.0))),
            )

        s0 = _sigmoid(np.asarray(-self.T_half_stretch / self.k_stretch))
        s = _sigmoid((T - self.T_half_stretch) / self.k_stretch)
        # Normalise so f(0) = 1 exactly and f(∞) = 1 + alpha_NMDA.
        out = 1.0 + self.alpha_NMDA * (s - s0) / (1.0 - s0)
        return float(out) if out.ndim == 0 else out

    def mg_unblock(
        self,
        V_m: NDArray[np.floating] | float,
    ) -> NDArray[np.floating] | float:
        """Voltage-dependent Mg²⁺ unblock fraction (Jahr & Stevens 1990).

        Approaches 0 at hyperpolarised potentials (channel blocked) and
        1 at depolarised potentials (Mg²⁺ expelled).
        """
        V = np.asarray(V_m, dtype=np.float64)
        out = 1.0 / (1.0 + self.Mg * np.exp(-V / 16.13) / 3.57)
        return float(out) if out.ndim == 0 else out

    def conductance(
        self,
        V_m: NDArray[np.floating] | float,
        glutamate: NDArray[np.floating] | float,
        tension: NDArray[np.floating] | float = 0.0,
    ) -> NDArray[np.floating] | float:
        """NMDA conductance density [mS/cm²].

        The optional ``tension`` [mN/m] applies the mechanosensitive
        stretch enhancement (Maneshi 2017). Default 0 → baseline NMDA.
        """
        return (
            self.g_max
            * self.glutamate_binding(glutamate)
            * self.mg_unblock(V_m)
            * self.stretch_factor(tension)
        )

    def current(
        self,
        V_m: NDArray[np.floating] | float,
        glutamate: NDArray[np.floating] | float,
        tension: NDArray[np.floating] | float = 0.0,
    ) -> NDArray[np.floating] | float:
        """Macroscopic NMDA current density [μA/cm²].

        Sign convention matches MechanosensitiveChannel.current:
        positive = inward (depolarising for E_rev > V_m). The optional
        ``tension`` [mN/m] applies the stretch enhancement (default 0).
        """
        return self.conductance(V_m, glutamate, tension) * (self.E_rev - V_m)


def nmdar_stretch_factor(
    peak_tension_mNm: float,
    alpha_NMDA: float = 1.0,
    T_half_stretch: float = 5.0,
    k_stretch: float = 1.5,
) -> float:
    """Post-synaptic NMDA stretch-enhancement factor (Maneshi 2017).

    DOWNSTREAM SCALAR ENTRY POINT for the ``neuromod_parameters`` pipeline.

    Thin wrapper around :meth:`NMDAReceptor.stretch_factor` using the
    canonical Maneshi-calibrated defaults, so a single peak-tension
    scalar maps to a multiplicative NMDA-current gain without
    instantiating the receptor.

    Parameters
    ----------
    peak_tension_mNm : float
        Peak membrane tension [mN/m].
    alpha_NMDA, T_half_stretch, k_stretch : float
        Stretch-sigmoid parameters (see :class:`NMDAReceptor`).

    Returns
    -------
    float
        Enhancement factor ``≥ 1``; ``1.0`` at zero tension, →
        ``1 + alpha_NMDA`` (≈ 2, current doubling) at saturating stretch.
    """
    receptor = NMDAReceptor(
        alpha_NMDA=alpha_NMDA,
        T_half_stretch=T_half_stretch,
        k_stretch=k_stretch,
    )
    return float(receptor.stretch_factor(float(peak_tension_mNm)))
