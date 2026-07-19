"""Astrocytic TRPA1 → Ca²⁺ → glutamate → NMDA ultrasound-neuromodulation relay.

Implements the non-Piezo mechanism of Oh et al. 2019 (Curr Biol 29:3386,
"Ultrasonic Neuromodulation via Astrocytic TRPA1"):

    ultrasound --(mechanical)--> TRPA1 opens in ASTROCYTES
              --> astrocytic Ca²⁺ influx (TRPA1 is Ca²⁺-permeable)
              --> Ca²⁺-triggered glutamate release (Hill in [Ca²⁺])
              --> glutamate binds neuronal NMDA receptors
              --> NMDA current depolarises the neuron --> excitation

Two signature features of the pathway are honoured:

* **TRPA1 is the necessary sensor.** Setting the TRPA1 conductance to
  zero (knockout / HC-030031 / A-967079 antagonist condition) abolishes
  the relay output. The mechanical gate reuses the repo's Boltzmann-in-
  tension :class:`~ultraneuron.channels.MechanosensitiveChannel`.
* **The relay is SLOW.** Astrocytic Ca²⁺ and gliotransmission operate on
  100s-of-ms-to-seconds timescales (``tau_decay`` ≈ 300 ms), much slower
  than the per-carrier-cycle Piezo/HH channels. The astrocytic Ca²⁺ pool
  therefore *accumulates* over the sonication, so the relay output grows
  with duty cycle × on-time rather than tracking the carrier.

Single-pool astrocyte + one target neuron — the same complexity tier as
``nice.py`` / ``calcium.py``. No spatial glia network.

Downstream interface contract (consumed by the neuromod_parameters
inference pipeline): :meth:`AstrocyteRelay.excitation_drive` returns a
cheap per-case scalar from peak membrane tension, duty cycle, and
on-time — see its docstring for the exact signature and units.

Units follow the repo convention:
    - Membrane tension: mN/m
    - Voltage: mV
    - Conductance: mS/cm²
    - Current density: μA/cm² (positive = inward)
    - Ca²⁺ / glutamate: μM
    - Time: ms (on-time arg is in s, per the pipeline contract)
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from .calcium import CalciumDynamics
from .channel_library import TRPA1
from .channels import MechanosensitiveChannel
from .neuron import MechanoHHNeuron
from .nmda import NMDAReceptor

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _default_trpa1() -> MechanosensitiveChannel:
    return copy.deepcopy(TRPA1)


def _default_astro_calcium() -> CalciumDynamics:
    # Slow astrocytic pool: hundreds of ms decay (Bazargani & Attwell 2016),
    # flux_factor calibrated so the pool peaks ~1 μM at full US drive.
    return CalciumDynamics(Ca_rest=0.05, tau_decay=300.0, flux_factor=0.011)


@dataclass
class AstrocyteRelay:
    """Single-pool astrocyte: TRPA1 Ca²⁺ influx → Hill glutamate release.

    Parameters
    ----------
    trpa1 : MechanosensitiveChannel
        The astrocytic mechanosensor (default: library ``TRPA1`` copy).
        Its ``g_density`` is the knockout knob — set to 0 to abolish the
        relay (Oh 2019 TRPA1-KO / antagonist condition).
    calcium : CalciumDynamics
        Astrocytic Ca²⁺ pool (slow ``tau_decay``; ``flux_factor``
        calibrated for a ~1 μM peak at full drive).
    V_astro : float
        Astrocyte resting potential [mV]; sets the Ca²⁺ driving force
        through TRPA1. Astrocytes are non-spiking, ~-80 mV.
    glu_Ca_half : float
        Half-release astrocytic [Ca²⁺] for gliotransmission [μM].
    glu_n_hill : float
        Hill coefficient of the Ca²⁺-triggered release.
    glu_max : float
        Peak cleft glutamate output [μM] at saturating [Ca²⁺].
    """

    trpa1: MechanosensitiveChannel = field(default_factory=_default_trpa1)
    calcium: CalciumDynamics = field(default_factory=_default_astro_calcium)
    V_astro: float = -80.0       # mV
    glu_Ca_half: float = 0.3     # μM
    glu_n_hill: float = 3.0
    glu_max: float = 10.0        # μM

    # ------------------------------------------------------------------
    # Building blocks
    # ------------------------------------------------------------------
    def trpa1_ca_current(
        self,
        tension_mNm: NDArray[np.floating] | float,
    ) -> NDArray[np.floating] | float:
        """Astrocytic TRPA1 Ca²⁺ current density [μA/cm²], inward-positive.

        ``I_Ca = g · P_o(T) · f_Ca · (E_rev − V_astro)`` where ``f_Ca`` is
        the channel's Ca²⁺-permeability fraction.
        """
        p_o = self.trpa1.open_probability(tension_mNm)
        return (
            self.trpa1.g_density
            * p_o
            * self.trpa1.Ca_permeability_fraction
            * (self.trpa1.E_rev - self.V_astro)
        )

    def glutamate_from_ca(
        self,
        Ca: NDArray[np.floating] | float,
    ) -> NDArray[np.floating] | float:
        """Hill-type Ca²⁺-triggered glutamate release [μM]."""
        ca = np.maximum(np.asarray(Ca, dtype=np.float64), 1e-9)
        x_n = ca ** self.glu_n_hill
        frac = x_n / (x_n + self.glu_Ca_half ** self.glu_n_hill)
        out = self.glu_max * frac
        return float(out) if out.ndim == 0 else out

    # ------------------------------------------------------------------
    # Time-course (forward) model
    # ------------------------------------------------------------------
    def relay_timecourse(
        self,
        tension_trace: NDArray[np.floating] | float,
        dt: float = 0.1,
        duration_ms: float | None = None,
    ) -> dict[str, NDArray[np.floating]]:
        """Integrate the astrocytic Ca²⁺ pool and glutamate output.

            d[Ca]/dt = flux_factor · I_Ca(T) − ([Ca] − Ca_rest)/tau_decay

        Parameters
        ----------
        tension_trace : ndarray or float
            Membrane-tension drive [mN/m]. Scalar → constant drive for
            ``duration_ms``; 1-D array → per-step drive (``duration_ms``
            ignored).
        dt : float
            Time step [ms].
        duration_ms : float or None
            Total time [ms] when ``tension_trace`` is scalar. Defaults to
            5·tau_decay so the slow pool reaches quasi-steady state.

        Returns
        -------
        dict
            ``'t'`` [ms], ``'Ca'`` [μM], ``'glutamate'`` [μM],
            ``'P_o'`` (TRPA1 open prob), ``'I_Ca'`` [μA/cm²].
        """
        if np.isscalar(tension_trace):
            if duration_ms is None:
                duration_ms = 5.0 * self.calcium.tau_decay
            n = max(int(round(duration_ms / dt)), 1)
            T_arr = np.full(n, float(tension_trace), dtype=np.float64)
        else:
            T_arr = np.asarray(tension_trace, dtype=np.float64)
            n = T_arr.size

        Ca = np.empty(n, dtype=np.float64)
        ca = float(self.calcium.Ca_rest)
        ff = self.calcium.flux_factor
        tau = self.calcium.tau_decay
        ca_rest = self.calcium.Ca_rest
        for i in range(n):
            Ca[i] = ca
            i_ca = float(self.trpa1_ca_current(T_arr[i]))
            dca = ff * i_ca - (ca - ca_rest) / tau
            ca = max(ca + dt * dca, 0.0)

        glutamate = self.glutamate_from_ca(Ca)
        p_o = np.asarray(self.trpa1.open_probability(T_arr), dtype=np.float64)
        i_ca_arr = np.asarray(self.trpa1_ca_current(T_arr), dtype=np.float64)
        t = np.arange(n) * dt
        return {
            "t": t,
            "Ca": Ca,
            "glutamate": np.asarray(glutamate, dtype=np.float64),
            "P_o": p_o,
            "I_Ca": i_ca_arr,
        }

    # ------------------------------------------------------------------
    # Cheap per-case scalar — the downstream interface contract
    # ------------------------------------------------------------------
    def excitation_drive(
        self,
        peak_tension_mNm: float,
        duty_cycle: float = 1.0,
        on_time_s: float = 1.0,
        nmda: NMDAReceptor | None = None,
        v_ref_mV: float = -50.0,
    ) -> float:
        """Scalar astrocytic-NMDA excitation drive for the inference pipeline.

        Parallels ``TREK1.open_probability(tension)`` /
        ``TRPV4.open_probability(37+dT)`` / ``KCaChannel.open_probability(Ca)``:
        a single cheap number, monotone in the astrocytic-NMDA excitation,
        computed without re-running the forward model.

        Because the relay is slow and Ca²⁺-accumulating, the scalar
        *depends on duty cycle × on-time*: the astrocytic Ca²⁺ pool fills
        toward a duty-weighted quasi-steady level over the sonication,

            Ca(t_on) = Ca_rest + (f·I_Ca·duty)·tau·(1 − exp(−t_on/tau)),

        so longer / higher-duty sonication → more Ca²⁺ → more glutamate →
        more NMDA drive (mirroring how the K_Ca pathway couples mechanical
        drive × duty).

        Parameters
        ----------
        peak_tension_mNm : float
            Peak membrane tension at the focal voxel [mN/m].
        duty_cycle : float
            Macroscopic ON fraction of the sonication (0–1).
        on_time_s : float
            Total sonication ON-time [s] (e.g. session_duration × duty, or
            the protocol's total on-window). Longer → more accumulation.
        nmda : NMDAReceptor or None
            Receptor used for the glutamate-binding × Mg²⁺-unblock readout.
            Default: library ``NMDAReceptor()``.
        v_ref_mV : float
            Reference membrane potential for the Mg²⁺ unblock term [mV].
            Default -50 mV (sub-threshold operating point where NMDA's
            voltage-dependent contribution becomes physiologically
            relevant).

        Returns
        -------
        float
            Dimensionless drive in [0, 1] — the NMDA glutamate-bound,
            Mg²⁺-unblocked open fraction. Monotone in the astrocytic-NMDA
            excitation; 0 under TRPA1 knockout (g_density → 0).
        """
        nmda = nmda or NMDAReceptor()
        i_ca_peak = float(self.trpa1_ca_current(peak_tension_mNm))  # μA/cm²
        j_avg = self.calcium.flux_factor * i_ca_peak * float(duty_cycle)  # μM/ms
        tau = self.calcium.tau_decay
        t_on_ms = float(on_time_s) * 1e3
        fill = 1.0 - np.exp(-t_on_ms / tau)
        ca = self.calcium.Ca_rest + j_avg * tau * fill
        glu = float(self.glutamate_from_ca(ca))
        drive = (
            float(nmda.glutamate_binding(glu))
            * float(nmda.mg_unblock(v_ref_mV))
        )
        return float(drive)


def simulate_astrocyte_relay(
    peak_tension_mNm: float | None = None,
    *,
    tension_trace: NDArray[np.floating] | None = None,
    relay: AstrocyteRelay | None = None,
    nmda: NMDAReceptor | None = None,
    neuron: MechanoHHNeuron | None = None,
    dt: float = 0.025,
    duration_ms: float = 300.0,
) -> dict[str, object]:
    """Full forward model: US tension → TRPA1 → astro Ca²⁺ → glutamate → NMDA → neuron.

    Parallels :func:`~ultraneuron.neuron.simulate_map`. The
    target neuron carries **no direct mechanosensitive channels**, so the
    returned firing / depolarisation is attributable solely to the
    astrocytic NMDA pathway. The slowly-varying NMDA current is injected
    through the neuron's external-stimulus port; its voltage-dependent
    Mg²⁺ block is evaluated at the sub-threshold resting potential
    (operator-split — appropriate for the slow relay, and conservative
    since it omits the regenerative unblock during a spike).

    Parameters
    ----------
    peak_tension_mNm : float or None
        Constant focal tension drive [mN/m] over ``duration_ms``.
        Provide this *or* ``tension_trace``.
    tension_trace : ndarray or None
        Per-step tension drive [mN/m]; overrides ``peak_tension_mNm`` and
        ``duration_ms``.
    relay, nmda, neuron : optional
        Components; sensible literature defaults are built if omitted.
    dt : float
        Time step [ms] (default 0.025, HH-stable for a point neuron).
    duration_ms : float
        Simulated time [ms] when driving from a scalar tension.

    Returns
    -------
    dict
        ``'t'`` [ms], ``'Ca_astro'`` [μM], ``'glutamate'`` [μM],
        ``'I_NMDA'`` [μA/cm²], ``'V_neuron'`` [mV], ``'spike_times'`` [ms],
        ``'n_spikes'`` (int), ``'V_mean'`` [mV], ``'depolarisation'`` [mV]
        (mean V minus resting V), and ``'excitation_drive'`` (the scalar).
    """
    relay = relay or AstrocyteRelay()
    nmda = nmda or NMDAReceptor()
    neuron = neuron or MechanoHHNeuron.cortical_pyramidal(mechano_channels=[])

    if tension_trace is not None:
        T_trace = np.asarray(tension_trace, dtype=np.float64)
        tc = relay.relay_timecourse(T_trace, dt=dt)
    else:
        if peak_tension_mNm is None:
            msg = "Provide peak_tension_mNm or tension_trace."
            raise ValueError(msg)
        tc = relay.relay_timecourse(
            float(peak_tension_mNm), dt=dt, duration_ms=duration_ms,
        )

    glu = tc["glutamate"]
    v_ref = neuron.V_rest
    i_nmda = np.asarray(nmda.current(v_ref, glu), dtype=np.float64)

    # Drive the neuron over exactly the relay's step count. The +0.5
    # guards against float-floor mismatch between len(i_nmda) and
    # simulate()'s n_steps = int(duration/dt).
    n_relay = i_nmda.shape[0]
    res = neuron.simulate(
        dt=dt,
        duration=(n_relay + 0.5) * dt,
        I_stim=i_nmda,
    )
    V = res["V"]
    spike_times = res["spike_times"]

    return {
        "t": tc["t"],
        "Ca_astro": tc["Ca"],
        "glutamate": glu,
        "I_NMDA": i_nmda,
        "V_neuron": V,
        "spike_times": spike_times,
        "n_spikes": int(len(spike_times)),
        "V_mean": float(np.mean(V)),
        "depolarisation": float(np.mean(V) - neuron.V_rest),
        "excitation_drive": relay.excitation_drive(
            float(peak_tension_mNm) if peak_tension_mNm is not None
            else float(np.max(tension_trace)),
            duty_cycle=1.0,
            on_time_s=float(tc["t"][-1]) * 1e-3,
            nmda=nmda,
        ),
    }
