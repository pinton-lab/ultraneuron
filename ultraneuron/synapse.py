"""Mechanosensitive presynaptic release and the synaptic-relay pathway.

The repo's other channels model *post-synaptic* mechanosensitivity on a
single membrane. This module adds the missing *presynaptic* and
*circuit-level* link: a minimal single pre → single post synapse whose
release probability is modulated by presynaptic membrane tension
(Tyler 2012, "The mechanobiology of brain function"; Tyler 2008).
Mechanical perturbation of the presynaptic terminal raises the
vesicle-release probability ``Pr``, scaling synaptic gain. The released
glutamate produces a cleft transient that drives post-synaptic AMPA and
(stretch-enhanced) NMDA conductances on the target neuron.

Pipeline::

    US tension ──▶ Pr(tension)           (presynaptic, Tyler 2012)
               ──▶ glutamate transient   (first-order cleft clearance)
               ──▶ AMPA + NMDA current   (post-synaptic; NMDA stretch = Maneshi 2017)
               ──▶ post-synaptic neuron  (+ mechanosensitive Nav/Cav shift, Kubanek 2016)

There is NO spatial network and NO plasticity ODE: LTP induction (the
stretch-enhanced-LTP motivation behind NMDA mechanosensitivity) is a
documented downstream consequence, not modelled here. Complexity tier
matches ``nice.py`` / ``calcium.py``.

Units (repo convention):
    - Voltage: mV
    - Conductance: mS/cm²
    - Current density: μA/cm² (positive = inward / depolarising)
    - Glutamate: μM
    - Membrane tension: mN/m
    - Time: ms
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from .neuron import K_MECHANO_NAV, MechanoHHNeuron
from .nmda import NMDAReceptor

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _sigmoid(x: NDArray[np.floating] | float) -> NDArray[np.floating]:
    """Numerically stable logistic 1 / (1 + exp(-x))."""
    x = np.asarray(x, dtype=np.float64)
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-np.minimum(x, 500.0))),
        np.exp(np.maximum(x, -500.0)) / (1.0 + np.exp(np.maximum(x, -500.0))),
    )


@dataclass
class MechanoSynapse:
    """Single pre → single post synapse with tension-modulated release.

    Presynaptic membrane tension raises the release probability ``Pr``
    along a saturating sigmoid normalised so ``Pr(0) = Pr_baseline`` and
    ``Pr(∞) = Pr_max`` (Tyler 2012). The cleft glutamate concentration
    is tracked as a first-order pool driven by the release rate while
    the ultrasound is on, and read out post-synaptically through a fast
    AMPA conductance plus a stretch-enhanced :class:`NMDAReceptor`.

    Parameters
    ----------
    Pr_baseline : float
        Resting vesicle release probability (dimensionless, in [0, 1]).
        Central synapses sit around 0.1–0.3 (Tyler 2008). Default 0.10.
    Pr_max : float
        Saturating release probability at high tension. Default 0.6.
    T_half_pre : float
        Half-modulation presynaptic tension [mN/m]. Default 4.0.
    k_pre : float
        Boltzmann slope of the Pr sigmoid [mN/m]. Default 1.5.
    glu_scale : float
        Sustained cleft glutamate concentration [μM] reached at full
        release (``Pr = 1``) under steady drive. Default 30.0 — calibrated
        so the resting (baseline-Pr, zero-tension) glutamate level is
        sub-threshold and ultrasound-driven release brings the post
        neuron to firing.
    tau_glu : float
        First-order cleft glutamate clearance time constant [ms].
        Default 5.0.
    g_ampa : float
        Maximum post-synaptic AMPA conductance density [mS/cm²].
        Default 0.08 (calibrated with ``glu_scale`` for a sub-threshold
        resting state).
    E_ampa : float
        AMPA reversal potential [mV]. Non-selective cation, ~0 mV.
    K_ampa : float
        AMPA glutamate half-saturation [μM]. AMPA is low-affinity
        (EC50 ~ hundreds of μM). Default 300.0.
    nmda : NMDAReceptor
        Post-synaptic NMDA receptor (carries its own stretch
        enhancement). Default a :class:`NMDAReceptor` with the synaptic
        density ``g_max = 0.45`` mS/cm².
    """

    Pr_baseline: float = 0.10
    Pr_max: float = 0.6
    T_half_pre: float = 4.0   # mN/m
    k_pre: float = 1.5        # mN/m
    glu_scale: float = 30.0   # μM at Pr = 1, steady state
    tau_glu: float = 5.0      # ms
    g_ampa: float = 0.08      # mS/cm²
    E_ampa: float = 0.0       # mV
    K_ampa: float = 300.0     # μM
    nmda: NMDAReceptor = field(default_factory=lambda: NMDAReceptor(g_max=0.45))

    # ------------------------------------------------------------------
    # Presynaptic release
    # ------------------------------------------------------------------
    def release_probability(
        self,
        tension: NDArray[np.floating] | float,
    ) -> NDArray[np.floating] | float:
        """Tension-dependent vesicle release probability ``Pr(T)``.

        Normalised so ``Pr(0) = Pr_baseline`` exactly and
        ``Pr(∞) = Pr_max``. Monotonic non-decreasing; negative tensions
        clamp to 0.
        """
        T = np.maximum(np.asarray(tension, dtype=np.float64), 0.0)
        s0 = _sigmoid(-self.T_half_pre / self.k_pre)
        s = _sigmoid((T - self.T_half_pre) / self.k_pre)
        frac = (s - s0) / (1.0 - s0)
        out = self.Pr_baseline + (self.Pr_max - self.Pr_baseline) * frac
        return float(out) if out.ndim == 0 else out

    # ------------------------------------------------------------------
    # Post-synaptic readout
    # ------------------------------------------------------------------
    def ampa_conductance(
        self,
        glutamate: NDArray[np.floating] | float,
    ) -> NDArray[np.floating] | float:
        """Saturating AMPA conductance density [mS/cm²]."""
        glu = np.maximum(np.asarray(glutamate, dtype=np.float64), 0.0)
        return self.g_ampa * glu / (glu + self.K_ampa)

    def synaptic_current(
        self,
        V_m: float,
        glutamate: float,
        tension: float = 0.0,
    ) -> float:
        """Total post-synaptic synaptic current density [μA/cm²].

        Sum of AMPA and (stretch-enhanced) NMDA currents. Positive =
        inward / depolarising. ``tension`` applies the post-synaptic
        NMDA stretch enhancement (Maneshi 2017).
        """
        I_ampa = float(self.ampa_conductance(glutamate)) * (self.E_ampa - V_m)
        I_nmda = float(self.nmda.current(V_m, glutamate, tension))
        return I_ampa + I_nmda

    def glutamate_trace(
        self,
        tension_trace: NDArray[np.floating],
        dt: float,
        us_mask: NDArray[np.bool_] | None = None,
    ) -> NDArray[np.floating]:
        """Cleft glutamate transient [μM] driven by presynaptic release.

        First-order pool integrated by forward Euler:

            dGlu/dt = (glu_scale / tau_glu) · Pr(T) · us − Glu / tau_glu

        so the steady glutamate during sustained ultrasound is
        ``glu_scale · Pr(T)``. ``us_mask`` (default: wherever tension
        > 0) gates release on the ultrasound window.

        Parameters
        ----------
        tension_trace : ndarray, shape (n_steps,)
            Presynaptic membrane tension [mN/m] per step.
        dt : float
            Time step [ms].
        us_mask : bool ndarray or None
            Ultrasound-on gate. Default: ``tension_trace > 0``.

        Returns
        -------
        ndarray, shape (n_steps,)
            Cleft glutamate concentration [μM] per step.
        """
        T = np.asarray(tension_trace, dtype=np.float64)
        n = T.shape[0]
        if us_mask is None:
            us_mask = T > 0.0
        Pr = np.asarray(self.release_probability(T), dtype=np.float64)
        drive = (self.glu_scale / self.tau_glu) * Pr * us_mask.astype(np.float64)
        glu = np.zeros(n)
        g = 0.0
        for i in range(n):
            glu[i] = g
            g += dt * (drive[i] - g / self.tau_glu)
            if g < 0.0:
                g = 0.0
        return glu


def synaptic_gain(
    peak_tension_mNm: float,
    Pr_baseline: float = 0.10,
    Pr_max: float = 0.6,
    T_half_pre: float = 4.0,
    k_pre: float = 1.5,
) -> float:
    """Presynaptic synaptic-gain factor from tension-modulated release.

    DOWNSTREAM SCALAR ENTRY POINT for the ``neuromod_parameters`` pipeline.

    Returns the relative scaling of synaptic gain caused by the
    presynaptic release-probability modulation (Tyler 2012):

        synaptic_gain(T) = Pr(T) / Pr(0)

    Parameters
    ----------
    peak_tension_mNm : float
        Peak presynaptic membrane tension [mN/m].
    Pr_baseline, Pr_max, T_half_pre, k_pre : float
        Release-sigmoid parameters (see :class:`MechanoSynapse`).

    Returns
    -------
    float
        ``gain ≥ 1``; ``1.0`` at zero tension, → ``Pr_max / Pr_baseline``
        at saturating tension. Multiply a baseline presynaptic
        excitation score by this factor.
    """
    syn = MechanoSynapse(
        Pr_baseline=Pr_baseline,
        Pr_max=Pr_max,
        T_half_pre=T_half_pre,
        k_pre=k_pre,
    )
    return float(syn.release_probability(float(peak_tension_mNm))) / syn.Pr_baseline


def simulate_synaptic_relay(
    peak_tension_mNm: float | None = None,
    *,
    tension_trace: NDArray[np.floating] | None = None,
    synapse: MechanoSynapse | None = None,
    post_neuron: MechanoHHNeuron | None = None,
    dt: float = 0.025,
    duration: float = 80.0,
    us_onset: float = 5.0,
    us_offset: float | None = None,
    temperature: float = 37.0,
) -> dict[str, NDArray | float]:
    """Drive a post-synaptic neuron through the mechanosensitive relay.

    Parallel to :func:`ultraneuron.simulate_map`, but instead
    of a 3-D field it takes the ultrasound *mechanical* input (a peak
    tension, or an explicit tension time series) and returns the
    neuronal excitation attributable to the synaptic + mechanosensitive
    Nav/Cav pathway. Both literature mechanisms are active:

    * presynaptic ``Pr(T)`` → glutamate → AMPA/NMDA (stretch-enhanced),
    * post-synaptic Na⁺/Ca²⁺ activation shift via the post neuron's
      ``k_mechano_nav``.

    Parameters
    ----------
    peak_tension_mNm : float or None
        Peak membrane tension [mN/m] applied as a square ultrasound
        window over ``[us_onset, us_offset]``. Mutually exclusive with
        ``tension_trace``.
    tension_trace : ndarray or None
        Explicit per-step tension [mN/m] of length ``int(duration/dt)``.
        Overrides ``peak_tension_mNm`` when given; the ultrasound window
        is taken as wherever the trace is > 0.
    synapse : MechanoSynapse or None
        Synapse model. Default a fresh :class:`MechanoSynapse`.
    post_neuron : MechanoHHNeuron or None
        Post-synaptic neuron. Default a cortical pyramidal neuron with
        the Kubanek Nav shift enabled (``k_mechano_nav = K_MECHANO_NAV``).
    dt, duration : float
        Integration step and total time [ms].
    us_onset, us_offset : float
        Ultrasound window [ms] when ``peak_tension_mNm`` is used.
        ``us_offset`` defaults to ``duration``.
    temperature : float
        Body temperature [°C] for the default post neuron.

    Returns
    -------
    dict
        ``'t'``           : time vector [ms].
        ``'V'``           : post-synaptic voltage [mV].
        ``'glutamate'``   : cleft glutamate transient [μM].
        ``'Pr'``          : presynaptic release probability per step.
        ``'tension'``     : tension trace [mN/m].
        ``'I_syn'``       : injected synaptic current [μA/cm²].
        ``'spike_times'`` : post-synaptic spike times [ms].
        ``'spike_count'`` : number of post-synaptic spikes (int).
        ``'peak_depolarisation'`` : max V − V_rest [mV].
        ``'excitation'``  : scalar excitation summary (= spike_count).
    """
    n_steps = int(duration / dt)

    if tension_trace is not None:
        T_arr = np.asarray(tension_trace, dtype=np.float64)
        if T_arr.shape != (n_steps,):
            msg = (
                f"tension_trace length {T_arr.shape} does not match "
                f"int(duration/dt) = {n_steps}"
            )
            raise ValueError(msg)
        us_mask = T_arr > 0.0
    else:
        if peak_tension_mNm is None:
            msg = "Provide either peak_tension_mNm or tension_trace."
            raise ValueError(msg)
        if us_offset is None:
            us_offset = duration
        t = np.arange(n_steps) * dt
        us_mask = (t >= us_onset) & (t < us_offset)
        T_arr = np.where(us_mask, float(peak_tension_mNm), 0.0)

    if synapse is None:
        synapse = MechanoSynapse()
    if post_neuron is None:
        post_neuron = MechanoHHNeuron.cortical_pyramidal(
            temperature=temperature,
            k_mechano_nav=K_MECHANO_NAV,
        )

    glu = synapse.glutamate_trace(T_arr, dt, us_mask=us_mask)
    Pr = np.asarray(synapse.release_probability(T_arr), dtype=np.float64)

    def _extra_current(V: float, i: int) -> float:
        return synapse.synaptic_current(V, float(glu[i]), float(T_arr[i]))

    result = post_neuron.simulate(
        dt=dt,
        duration=duration,
        I_stim=0.0,
        tension_trace=T_arr,
        temperature_trace=temperature,
        extra_current_fn=_extra_current,
    )

    spike_times = result["spike_times"]
    spike_count = int(len(spike_times))
    peak_depol = float(np.max(result["V"]) - post_neuron.V_rest)

    return {
        "t": result["t"],
        "V": result["V"],
        "glutamate": glu,
        "Pr": Pr,
        "tension": T_arr,
        "I_syn": result["I_syn"],
        "spike_times": spike_times,
        "spike_count": spike_count,
        "peak_depolarisation": peak_depol,
        "excitation": float(spike_count),
    }
