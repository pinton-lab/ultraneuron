"""Mechanosensitive ion channel model for ultrasound neuromodulation.

Implements Boltzmann gating of mechanosensitive channels driven by
membrane tension derived from tissue-level strain fields.

Units follow biophysics convention:
    - Membrane tension: mN/m
    - Voltage: mV
    - Conductance: mS/cm^2
    - Current density: uA/cm^2
    - Time: ms
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def strain_to_membrane_tension(
    deviatoric_strain: NDArray[np.floating] | float,
    K_A: float = 0.25,
    coupling_factor: float = 1.0,
) -> NDArray[np.floating] | float:
    r"""Convert deviatoric strain to effective membrane tension.

    Models the membrane as an elastic sheet embedded in the tissue
    continuum. Tension arises from area dilation of the membrane
    driven by deviatoric tissue strain:

    .. math::

        T = K_A \cdot \alpha \cdot \varepsilon_{\mathrm{dev}}

    Parameters
    ----------
    deviatoric_strain : ndarray or float
        Deviatoric strain norm (dimensionless), as returned by
        ``compute_strain_invariants(...)['deviatoric']``.
    K_A : float
        Area expansion modulus of the lipid bilayer [N/m].
        Typical range 0.1 – 0.5 N/m. Default 0.25 N/m.
    coupling_factor : float
        Dimensionless scaling relating far-field tissue strain to
        local membrane area change. Default 1.0 (no amplification).

    Returns
    -------
    tension : ndarray or float
        Membrane tension [mN/m].

    """
    # K_A is in N/m, output in mN/m -> multiply by 1e3
    return K_A * coupling_factor * deviatoric_strain * 1e3


@dataclass
class MechanosensitiveChannel:
    """Boltzmann-gated mechanosensitive ion channel.

    All mechanosensitive channels share the same two-state Boltzmann
    activation model; they differ only in their biophysical parameters.
    An optional inactivation gate and Ca²⁺ permeability fraction can be
    set per channel to support more detailed neuron models (e.g. the
    ``piezo_inactivation`` and ``kca_dynamics`` opt-ins on
    :class:`MulticompartmentMechanoHHNeuron`); when those are not
    enabled by the neuron, the additional fields have no effect on
    integration and the channel reduces to the single-gate Boltzmann
    behaviour of the Phase-1 model.

    Parameters
    ----------
    name : str
        Human-readable identifier (e.g. ``"Piezo1"``).
    T_half : float
        Half-activation tension [mN/m].
    delta_A : float
        Sensitivity parameter — effective area change at the gate [nm^2].
        Larger values produce steeper activation curves.
    g_max : float
        Maximum single-channel conductance [pS]. Used to scale the
        macroscopic conductance density.
    g_density : float
        Macroscopic conductance density when fully open [mS/cm^2].
    E_rev : float
        Reversal potential [mV].
    tau_0 : float
        Baseline gating time constant [ms].
    channel_type : str
        ``"cation"`` (depolarising, e.g. Piezo1) or
        ``"potassium"`` (hyperpolarising, e.g. TRAAK/TREK).
    description : str
        Optional note on provenance / literature reference.
    tau_inact : float or None
        Inactivation time constant [ms] for an optional second
        Boltzmann gate ``i``. ``None`` (default) → no inactivation,
        identical to Phase-1 behaviour. Piezo1 uses ~12 ms (Coste 2010).
    inact_T_half : float or None
        Half-tension for the inactivation gate [mN/m]. Defaults to
        ``T_half`` when ``None``.
    inact_delta_A : float or None
        Sensitivity of the inactivation gate [nm²]. Defaults to
        ``delta_A`` when ``None``.
    Ca_permeability_fraction : float
        Fraction of channel current carried by Ca²⁺ at physiological
        ionic gradients (in [0, 1]). Default 0 (channel doesn't pass
        Ca²⁺). Piezo1 uses ~0.1 (P_Ca/P_Na ≈ 5; ~10% of cation current
        is Ca²⁺ in physiological gradients).
    tau_act : float or None
        Phase-5 fixed activation time constant [ms]. When set, replaces
        the bell-shaped ``tau_0 / (4 P_o (1 − P_o) + 0.01)`` formula
        with a tension-independent τ. ``None`` (default) preserves the
        original bell-shaped τ for backward compatibility. Piezo1 uses
        2 ms (Coste 2010 activation rise).

    """

    name: str
    T_half: float  # mN/m
    delta_A: float  # nm^2
    g_max: float  # pS
    g_density: float  # mS/cm^2
    E_rev: float  # mV
    tau_0: float = 1.0  # ms
    channel_type: str = "cation"
    description: str = ""
    tau_inact: float | None = None
    inact_T_half: float | None = None
    inact_delta_A: float | None = None
    Ca_permeability_fraction: float = 0.0
    tau_act: float | None = None
    _kT: float = field(default=4.114, init=False, repr=False)  # kT at ~25 °C [pN·nm]

    def open_probability(
        self,
        T: NDArray[np.floating] | float,
    ) -> NDArray[np.floating] | float:
        r"""Steady-state open probability (Boltzmann).

        .. math::

            P_o(T) = \frac{1}{1 + \exp\!\bigl[-\Delta A\,(T - T_{1/2}) / k_BT\bigr]}

        Parameters
        ----------
        T : ndarray or float
            Membrane tension [mN/m].

        Returns
        -------
        P_o : ndarray or float
            Open probability in [0, 1].

        """
        # delta_A in nm^2, T in mN/m.
        # Energy = delta_A [nm^2] * T [mN/m]
        # 1 mN/m = 1e-3 N/m = 1e-3 J/m^2 = 1e-3 * 1e18 pN/nm^2 * 1e-18 = 1 pN/nm
        # So energy in pN·nm = delta_A * (T - T_half)  (already consistent)
        exponent = -self.delta_A * (T - self.T_half) / self._kT
        return _safe_sigmoid(exponent)

    def inactivation_steady_state(
        self,
        T: NDArray[np.floating] | float,
    ) -> NDArray[np.floating] | float:
        r"""Steady-state inactivation gate ``i_∞(T)`` (Boltzmann).

        Returns 1.0 (no inactivation) when ``tau_inact`` is ``None``.
        Otherwise:

        .. math::

            i_\infty(T) = \frac{1}{1 + \exp\!\bigl[\Delta A_i \,(T - T_{1/2,i}) / k_BT\bigr]}

        — the gate closes (multiplicative factor → 0) as tension rises.
        """
        if self.tau_inact is None:
            return 1.0 if np.isscalar(T) else np.ones_like(np.asarray(T, dtype=np.float64))
        T_h = self.T_half if self.inact_T_half is None else self.inact_T_half
        dA = self.delta_A if self.inact_delta_A is None else self.inact_delta_A
        # Inactivation closes with rising T: opposite sign to activation
        exponent = dA * (T - T_h) / self._kT
        return _safe_sigmoid(exponent)

    def current(
        self,
        T: NDArray[np.floating] | float,
        V_m: NDArray[np.floating] | float = -65.0,
    ) -> NDArray[np.floating] | float:
        """Macroscopic current density through this channel population.

        Parameters
        ----------
        T : ndarray or float
            Membrane tension [mN/m].
        V_m : ndarray or float
            Membrane potential [mV]. Default -65 mV (resting).

        Returns
        -------
        I : ndarray or float
            Current density [uA/cm^2].  Positive = inward (depolarising
            for cations with E_rev > V_m).

        """
        P_o = self.open_probability(T)
        return self.g_density * P_o * (self.E_rev - V_m)

    def time_constant(
        self,
        T: NDArray[np.floating] | float,
    ) -> NDArray[np.floating] | float:
        """Activation gate time constant [ms].

        Returns ``self.tau_act`` (tension-independent) if set; otherwise
        falls back to the original bell-shaped formula
        ``tau_0 / (4·P_o(1-P_o) + 0.01)`` peaking at T_half. The
        bell-shaped form is a model assumption (slowest at extreme
        tensions); fixing tau_act to a single literature value (e.g.
        2 ms for Piezo1, Coste 2010) is recommended for studies that
        depend on accurate post-pulse channel-closing dynamics.

        Parameters
        ----------
        T : ndarray or float
            Membrane tension [mN/m].

        Returns
        -------
        tau : ndarray or float
            Time constant [ms].

        """
        if self.tau_act is not None:
            tau_const = float(self.tau_act)
            T_arr = np.asarray(T)
            return tau_const if T_arr.ndim == 0 else np.full_like(T_arr, tau_const, dtype=np.float64)
        P_o = self.open_probability(T)
        denominator = 4.0 * P_o * (1.0 - P_o) + 0.01
        return self.tau_0 / denominator


def _safe_sigmoid(x: NDArray[np.floating] | float) -> NDArray[np.floating] | float:
    """Numerically stable sigmoid: 1 / (1 + exp(x))."""
    x = np.asarray(x, dtype=np.float64)
    # Standard sigmoid: sigma(x) = 1 / (1 + exp(x))
    # For large positive x, exp(x) dominates -> sigma ~ 0
    # For large negative x, exp(x) ~ 0 -> sigma ~ 1
    # Stable form: for x >= 0: sigma = exp(-x) / (1 + exp(-x))
    #              for x < 0:  sigma = 1 / (1 + exp(x))
    result = np.where(
        x >= 0,
        np.exp(-np.minimum(x, 500.0)) / (1.0 + np.exp(-np.minimum(x, 500.0))),
        1.0 / (1.0 + np.exp(np.maximum(x, -500.0))),
    )
    return float(result) if result.ndim == 0 else result
