"""Modified Hodgkin-Huxley point neuron with mechanosensitive coupling.

Supports two model parameterizations:
  - ``"squid"``   : Original Hodgkin-Huxley (1952), baseline 6.3 °C.
  - ``"cortical"`` : Pospischil et al. (2008) mammalian cortical pyramidal
    cell, baseline 36 °C.  Correct for human body temperature simulations.

Units (biophysics convention):
    - Voltage: mV
    - Current density: uA/cm^2
    - Conductance density: mS/cm^2
    - Time: ms
    - Tension: mN/m

The integrator uses RK4, which is numerically *stable* (does not diverge)
at dt = 0.01 ms for both models. Spike-COUNT accuracy is a stricter
requirement: a per-voxel dt-convergence study on the Halle/dACC field
shows the firing-voxel and spike totals are bit-identical for dt <=
0.001 ms but collapse by orders of magnitude at coarser steps (e.g. 8
spikes at 0.01 ms vs 23,260 at 0.001 ms), because the fast Na+ upstroke
is unresolved at large dt and threshold crossings are missed. Run the
per-voxel firing maps at dt <= 0.001 ms; do not relax to the stability
bound.

References
----------
.. [1] Hodgkin & Huxley, "A quantitative description of membrane current
       and its application to conduction and excitation in nerve",
       J. Physiol., 1952.
.. [2] Pospischil et al., "Minimal Hodgkin-Huxley type models for different
       classes of cortical and thalamic neurons",
       Biological Cybernetics, 2008.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from .channels import MechanosensitiveChannel

# ---------------------------------------------------------------------------
# Numba-accelerated batch kernel (optional, ~50-100x faster for large N)
# ---------------------------------------------------------------------------
try:
    from numba import njit, prange

    @njit(parallel=True, cache=True, fastmath=True)
    def _rk4_batch_numba(  # noqa: PLR0912
        V_rest: float,
        g_Na: float,
        g_K: float,
        g_L: float,
        E_Na: float,
        E_K: float,
        E_L: float,
        C_m: float,
        is_cortical: bool,  # noqa: FBT001
        VT: float,
        tensions: np.ndarray,
        phi_arr: np.ndarray,
        dt: float,
        n_steps: int,
        I_stim: float,
        ch_g_dens: np.ndarray,
        ch_E_revs: np.ndarray,
        ch_w_inf: np.ndarray,
        ch_tau_inv: np.ndarray,
        m0: float,
        h0: float,
        n0: float,
    ) -> np.ndarray:
        """Numba-compiled parallel RK4 across N voxels."""
        N = tensions.shape[0]
        n_ch = ch_g_dens.shape[0]
        spike_counts = np.zeros(N, dtype=np.int32)

        for idx in prange(N):
            V = V_rest
            m = m0
            h = h0
            n = n0
            phi = phi_arr[idx]

            # Initialise mechano gates at steady state
            w = np.empty(n_ch)
            w_inf_i = np.empty(n_ch)
            tau_inv_i = np.empty(n_ch)
            for k in range(n_ch):
                w_inf_i[k] = ch_w_inf[k, idx]
                tau_inv_i[k] = ch_tau_inv[k, idx]
                w[k] = w_inf_i[k]

            V_prev = V

            for _step in range(n_steps):
                V_prev = V

                # --- k1 ---
                k1V, k1m, k1h, k1n = _derivs_scalar(
                    V,
                    m,
                    h,
                    n,
                    w,
                    phi,
                    is_cortical,
                    VT,
                    g_Na,
                    g_K,
                    g_L,
                    E_Na,
                    E_K,
                    E_L,
                    C_m,
                    I_stim,
                    n_ch,
                    ch_g_dens,
                    ch_E_revs,
                )
                k1w = np.empty(n_ch)
                for k in range(n_ch):
                    k1w[k] = (w_inf_i[k] - w[k]) * tau_inv_i[k]

                # --- k2 ---
                w2 = np.empty(n_ch)
                for k in range(n_ch):
                    w2[k] = w[k] + 0.5 * dt * k1w[k]
                k2V, k2m, k2h, k2n = _derivs_scalar(
                    V + 0.5 * dt * k1V,
                    m + 0.5 * dt * k1m,
                    h + 0.5 * dt * k1h,
                    n + 0.5 * dt * k1n,
                    w2,
                    phi,
                    is_cortical,
                    VT,
                    g_Na,
                    g_K,
                    g_L,
                    E_Na,
                    E_K,
                    E_L,
                    C_m,
                    I_stim,
                    n_ch,
                    ch_g_dens,
                    ch_E_revs,
                )
                k2w = np.empty(n_ch)
                for k in range(n_ch):
                    k2w[k] = (w_inf_i[k] - w2[k]) * tau_inv_i[k]

                # --- k3 ---
                w3 = np.empty(n_ch)
                for k in range(n_ch):
                    w3[k] = w[k] + 0.5 * dt * k2w[k]
                k3V, k3m, k3h, k3n = _derivs_scalar(
                    V + 0.5 * dt * k2V,
                    m + 0.5 * dt * k2m,
                    h + 0.5 * dt * k2h,
                    n + 0.5 * dt * k2n,
                    w3,
                    phi,
                    is_cortical,
                    VT,
                    g_Na,
                    g_K,
                    g_L,
                    E_Na,
                    E_K,
                    E_L,
                    C_m,
                    I_stim,
                    n_ch,
                    ch_g_dens,
                    ch_E_revs,
                )
                k3w = np.empty(n_ch)
                for k in range(n_ch):
                    k3w[k] = (w_inf_i[k] - w3[k]) * tau_inv_i[k]

                # --- k4 ---
                w4 = np.empty(n_ch)
                for k in range(n_ch):
                    w4[k] = w[k] + dt * k3w[k]
                k4V, k4m, k4h, k4n = _derivs_scalar(
                    V + dt * k3V,
                    m + dt * k3m,
                    h + dt * k3h,
                    n + dt * k3n,
                    w4,
                    phi,
                    is_cortical,
                    VT,
                    g_Na,
                    g_K,
                    g_L,
                    E_Na,
                    E_K,
                    E_L,
                    C_m,
                    I_stim,
                    n_ch,
                    ch_g_dens,
                    ch_E_revs,
                )
                k4w = np.empty(n_ch)
                for k in range(n_ch):
                    k4w[k] = (w_inf_i[k] - w4[k]) * tau_inv_i[k]

                # --- Update state ---
                s = dt / 6.0
                V = V + s * (k1V + 2.0 * k2V + 2.0 * k3V + k4V)
                m = m + s * (k1m + 2.0 * k2m + 2.0 * k3m + k4m)
                h = h + s * (k1h + 2.0 * k2h + 2.0 * k3h + k4h)
                n = n + s * (k1n + 2.0 * k2n + 2.0 * k3n + k4n)
                for k in range(n_ch):
                    dw_avg = (k1w[k] + 2.0 * k2w[k] + 2.0 * k3w[k] + k4w[k]) / 6.0
                    w[k] = w[k] + dt * dw_avg

                # Clamp gates
                m = min(1.0, max(0.0, m))
                h = min(1.0, max(0.0, h))
                n = min(1.0, max(0.0, n))
                for k in range(n_ch):
                    w[k] = min(1.0, max(0.0, w[k]))

                # Spike detection
                if V > 0.0 and V_prev <= 0.0:
                    spike_counts[idx] += 1

        return spike_counts

    @njit(cache=True, fastmath=True)
    def _derivs_scalar(
        V: float,
        m: float,
        h: float,
        n: float,
        w_gates: np.ndarray,
        phi: float,
        is_cortical: bool,  # noqa: FBT001
        VT: float,
        g_Na: float,
        g_K: float,
        g_L: float,
        E_Na: float,
        E_K: float,
        E_L: float,
        C_m: float,
        I_stim: float,
        n_ch: int,
        ch_g_dens: np.ndarray,
        ch_E_revs: np.ndarray,
    ) -> tuple:
        """Scalar HH derivatives for one voxel (called from numba kernel)."""
        # Rate functions
        if is_cortical:
            dVm = V - VT - 13.0
            am = -0.32 * dVm / (np.exp(-dVm / 4.0) - 1.0) if abs(dVm) > 1e-7 else 0.32 * 4.0
            dVm2 = V - VT - 40.0
            bm = 0.28 * dVm2 / (np.exp(dVm2 / 5.0) - 1.0) if abs(dVm2) > 1e-7 else 0.28 * 5.0
            ah = 0.128 * np.exp(-(V - VT - 17.0) / 18.0)
            bh = 4.0 / (1.0 + np.exp(-(V - VT - 40.0) / 5.0))
            dVn = V - VT - 15.0
            an = -0.032 * dVn / (np.exp(-dVn / 5.0) - 1.0) if abs(dVn) > 1e-7 else 0.032 * 5.0
            bn = 0.5 * np.exp(-(V - VT - 10.0) / 40.0)
        else:
            dVm = V + 40.0
            am = 0.1 * dVm / (1.0 - np.exp(-dVm / 10.0)) if abs(dVm) > 1e-7 else 1.0
            bm = 4.0 * np.exp(-(V + 65.0) / 18.0)
            ah = 0.07 * np.exp(-(V + 65.0) / 20.0)
            bh = 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))
            dVn = V + 55.0
            an = 0.01 * dVn / (1.0 - np.exp(-dVn / 10.0)) if abs(dVn) > 1e-7 else 0.1
            bn = 0.125 * np.exp(-(V + 65.0) / 80.0)

        am *= phi
        bm *= phi
        ah *= phi
        bh *= phi
        an *= phi
        bn *= phi

        I_Na = g_Na * (m * m * m) * h * (V - E_Na)
        I_K = g_K * (n * n * n * n) * (V - E_K)
        I_L = g_L * (V - E_L)

        I_mechano = 0.0
        for k in range(n_ch):
            I_mechano += ch_g_dens[k] * w_gates[k] * (ch_E_revs[k] - V)

        dV = (-I_Na - I_K - I_L + I_mechano + I_stim) / C_m
        dm = am * (1.0 - m) - bm * m
        dh = ah * (1.0 - h) - bh * h
        dn = an * (1.0 - n) - bn * n
        return dV, dm, dh, dn

    @njit(parallel=True, cache=True, fastmath=True)
    def _rk4_batch_numba_tv(  # noqa: PLR0912
        V_rest: float,
        g_Na: float,
        g_K: float,
        g_L: float,
        E_Na: float,
        E_K: float,
        E_L: float,
        C_m: float,
        is_cortical: bool,  # noqa: FBT001
        VT: float,
        tensions_2d: np.ndarray,
        phi_arr: np.ndarray,
        dt: float,
        n_steps: int,
        I_stim: float,
        ch_g_dens: np.ndarray,
        ch_E_revs: np.ndarray,
        ch_T_halfs: np.ndarray,
        ch_delta_As: np.ndarray,
        ch_tau_0s: np.ndarray,
        ch_kT: float,
        m0: float,
        h0: float,
        n0: float,
    ) -> np.ndarray:
        """Numba-compiled parallel RK4 with time-varying tension (n_steps, N)."""
        N = tensions_2d.shape[1]
        n_ch = ch_g_dens.shape[0]
        spike_counts = np.zeros(N, dtype=np.int32)

        for idx in prange(N):
            V = V_rest
            m = m0
            h = h0
            n = n0
            phi = phi_arr[idx]

            w = np.empty(n_ch)
            # Initialise gates at steady state for initial tension
            for k in range(n_ch):
                T0 = tensions_2d[0, idx]
                exponent = -ch_delta_As[k] * (T0 - ch_T_halfs[k]) / ch_kT
                if exponent >= 0.0:
                    e = np.exp(-exponent)
                    w[k] = e / (1.0 + e)
                else:
                    w[k] = 1.0 / (1.0 + np.exp(exponent))

            V_prev = V

            for step in range(n_steps):
                V_prev = V
                T_now = tensions_2d[step, idx]

                # Precompute channel steady-state and time constants
                w_inf_now = np.empty(n_ch)
                tau_inv_now = np.empty(n_ch)
                for k in range(n_ch):
                    exponent = -ch_delta_As[k] * (T_now - ch_T_halfs[k]) / ch_kT
                    if exponent >= 0.0:
                        e = np.exp(-exponent)
                        po = e / (1.0 + e)
                    else:
                        po = 1.0 / (1.0 + np.exp(exponent))
                    w_inf_now[k] = po
                    denom = 4.0 * po * (1.0 - po) + 0.01
                    tau_inv_now[k] = denom / ch_tau_0s[k]

                # --- k1 ---
                k1V, k1m, k1h, k1n = _derivs_scalar(
                    V,
                    m,
                    h,
                    n,
                    w,
                    phi,
                    is_cortical,
                    VT,
                    g_Na,
                    g_K,
                    g_L,
                    E_Na,
                    E_K,
                    E_L,
                    C_m,
                    I_stim,
                    n_ch,
                    ch_g_dens,
                    ch_E_revs,
                )
                k1w = np.empty(n_ch)
                for k in range(n_ch):
                    k1w[k] = (w_inf_now[k] - w[k]) * tau_inv_now[k]

                # --- k2 ---
                w2 = np.empty(n_ch)
                for k in range(n_ch):
                    w2[k] = w[k] + 0.5 * dt * k1w[k]
                k2V, k2m, k2h, k2n = _derivs_scalar(
                    V + 0.5 * dt * k1V,
                    m + 0.5 * dt * k1m,
                    h + 0.5 * dt * k1h,
                    n + 0.5 * dt * k1n,
                    w2,
                    phi,
                    is_cortical,
                    VT,
                    g_Na,
                    g_K,
                    g_L,
                    E_Na,
                    E_K,
                    E_L,
                    C_m,
                    I_stim,
                    n_ch,
                    ch_g_dens,
                    ch_E_revs,
                )
                k2w = np.empty(n_ch)
                for k in range(n_ch):
                    k2w[k] = (w_inf_now[k] - w2[k]) * tau_inv_now[k]

                # --- k3 ---
                w3 = np.empty(n_ch)
                for k in range(n_ch):
                    w3[k] = w[k] + 0.5 * dt * k2w[k]
                k3V, k3m, k3h, k3n = _derivs_scalar(
                    V + 0.5 * dt * k2V,
                    m + 0.5 * dt * k2m,
                    h + 0.5 * dt * k2h,
                    n + 0.5 * dt * k2n,
                    w3,
                    phi,
                    is_cortical,
                    VT,
                    g_Na,
                    g_K,
                    g_L,
                    E_Na,
                    E_K,
                    E_L,
                    C_m,
                    I_stim,
                    n_ch,
                    ch_g_dens,
                    ch_E_revs,
                )
                k3w = np.empty(n_ch)
                for k in range(n_ch):
                    k3w[k] = (w_inf_now[k] - w3[k]) * tau_inv_now[k]

                # --- k4 ---
                w4 = np.empty(n_ch)
                for k in range(n_ch):
                    w4[k] = w[k] + dt * k3w[k]
                k4V, k4m, k4h, k4n = _derivs_scalar(
                    V + dt * k3V,
                    m + dt * k3m,
                    h + dt * k3h,
                    n + dt * k3n,
                    w4,
                    phi,
                    is_cortical,
                    VT,
                    g_Na,
                    g_K,
                    g_L,
                    E_Na,
                    E_K,
                    E_L,
                    C_m,
                    I_stim,
                    n_ch,
                    ch_g_dens,
                    ch_E_revs,
                )
                k4w = np.empty(n_ch)
                for k in range(n_ch):
                    k4w[k] = (w_inf_now[k] - w4[k]) * tau_inv_now[k]

                # --- Update state ---
                s = dt / 6.0
                V = V + s * (k1V + 2.0 * k2V + 2.0 * k3V + k4V)
                m = m + s * (k1m + 2.0 * k2m + 2.0 * k3m + k4m)
                h = h + s * (k1h + 2.0 * k2h + 2.0 * k3h + k4h)
                n = n + s * (k1n + 2.0 * k2n + 2.0 * k3n + k4n)
                for k in range(n_ch):
                    dw_avg = (k1w[k] + 2.0 * k2w[k] + 2.0 * k3w[k] + k4w[k]) / 6.0
                    w[k] = w[k] + dt * dw_avg

                m = min(1.0, max(0.0, m))
                h = min(1.0, max(0.0, h))
                n = min(1.0, max(0.0, n))
                for k in range(n_ch):
                    w[k] = min(1.0, max(0.0, w[k]))

                if V > 0.0 and V_prev <= 0.0:
                    spike_counts[idx] += 1

        return spike_counts

    _HAS_NUMBA = True

except ImportError:
    _HAS_NUMBA = False


@dataclass
class MechanoHHNeuron:
    """Hodgkin-Huxley neuron with mechanosensitive channel currents.

    Parameters
    ----------
    g_Na : float
        Maximum Na conductance [mS/cm^2].
    g_K : float
        Maximum delayed-rectifier K conductance [mS/cm^2].
    g_L : float
        Leak conductance [mS/cm^2].
    E_Na : float
        Na reversal potential [mV].
    E_K : float
        K reversal potential [mV].
    E_L : float
        Leak reversal potential [mV].
    C_m : float
        Membrane capacitance [uF/cm^2]. Default 1.0.
    V_rest : float
        Resting potential for gate initialisation [mV].
    temperature : float
        Simulation temperature [°C]. Default 37 (body temp).
    mechano_channels : list[MechanosensitiveChannel]
        Mechanosensitive channels to include. Default empty.
    model : str
        ``"squid"`` for original HH at 6.3 °C, or ``"cortical"`` for
        Pospischil 2008 mammalian cortical neuron at 36 °C.
        Default ``"squid"`` (backward compatible).
    k_mechano_nav : float
        Mechanosensitive shift of the voltage-gated Na⁺ (and, by
        proxy, Ca²⁺) activation half-voltage [mV per (mN/m)], following
        Kubanek et al. 2016. Membrane tension hyperpolarises the
        activation curve,
        ``V_half(T) = V_half0 - k_mechano_nav · T`` (a leftward shift),
        which increases the Na⁺ current at fixed voltage. Default 0.0
        recovers the unmodified neuron exactly. The Kubanek-calibrated
        canonical value is :data:`K_MECHANO_NAV` — opt in by setting
        this field. Only the scalar :meth:`simulate` path honours this
        shift; the vectorised batch / firing-map kernels ignore it.

    """

    g_Na: float = 120.0
    g_K: float = 36.0
    g_L: float = 0.3
    E_Na: float = 50.0
    E_K: float = -77.0
    E_L: float = -54.387
    C_m: float = 1.0
    V_rest: float = -65.0
    temperature: float = 37.0
    mechano_channels: list[MechanosensitiveChannel] = field(default_factory=list)
    model: str = "squid"
    k_mechano_nav: float = 0.0  # mV per (mN/m); Kubanek 2016 Nav/Cav shift

    # Internal parameters (set by __post_init__)
    _VT: float = field(init=False, repr=False, default=0.0)
    _ref_temp: float = field(init=False, repr=False, default=6.3)
    _q10_base: float = field(init=False, repr=False, default=3.0)

    def __post_init__(self) -> None:
        """Set internal parameters based on model type."""
        if self.model == "cortical":
            self._VT = -56.2  # threshold shift [mV], Pospischil 2008
            self._ref_temp = 36.0  # parameterized at 36 °C
            self._q10_base = 2.3  # mammalian Q10
        elif self.model == "squid":
            self._VT = 0.0
            self._ref_temp = 6.3
            self._q10_base = 3.0
        else:
            msg = f"Unknown model '{self.model}'. Use 'squid' or 'cortical'."
            raise ValueError(msg)

    @classmethod
    def cortical_pyramidal(
        cls,
        temperature: float = 37.0,
        mechano_channels: list[MechanosensitiveChannel] | None = None,
        **kwargs: float,
    ) -> MechanoHHNeuron:
        """Create a mammalian cortical pyramidal neuron (Pospischil 2008).

        Uses the regular-spiking (RS) parameterization, suitable for
        human body temperature simulations with bioheat coupling.

        Parameters
        ----------
        temperature : float
            Body temperature [°C]. Default 37.
        mechano_channels : list or None
            Mechanosensitive channels. Default empty.
        **kwargs
            Override any parameter (g_Na, g_K, etc.).

        Returns
        -------
        MechanoHHNeuron
            Configured cortical neuron.

        """
        defaults = {
            "g_Na": 50.0,
            "g_K": 5.0,
            "g_L": 0.1,
            "E_Na": 50.0,
            "E_K": -100.0,
            "E_L": -70.0,
            "C_m": 1.0,
            "V_rest": -70.0,
        }
        defaults.update(kwargs)
        return cls(
            temperature=temperature,
            mechano_channels=mechano_channels or [],
            model="cortical",
            **defaults,
        )

    # ------------------------------------------------------------------
    # Rate functions — dispatch based on model
    # ------------------------------------------------------------------

    def _alpha_m(self, V: float | NDArray) -> float | NDArray:
        """Na activation forward rate."""
        if self.model == "cortical":
            dV = V - self._VT - 13.0
            return np.where(
                np.abs(dV) < 1e-7,
                0.32 * 4.0,
                -0.32 * dV / (np.exp(-dV / 4.0) - 1.0),
            )
        # squid
        dV = V + 40.0
        return np.where(np.abs(dV) < 1e-7, 1.0, 0.1 * dV / (1.0 - np.exp(-dV / 10.0)))

    def _beta_m(self, V: float | NDArray) -> float | NDArray:
        """Na activation backward rate."""
        if self.model == "cortical":
            dV = V - self._VT - 40.0
            return np.where(
                np.abs(dV) < 1e-7,
                0.28 * 5.0,
                0.28 * dV / (np.exp(dV / 5.0) - 1.0),
            )
        return 4.0 * np.exp(-(V + 65.0) / 18.0)

    def _alpha_h(self, V: float | NDArray) -> float | NDArray:
        """Na inactivation forward rate."""
        if self.model == "cortical":
            return 0.128 * np.exp(-(V - self._VT - 17.0) / 18.0)
        return 0.07 * np.exp(-(V + 65.0) / 20.0)

    def _beta_h(self, V: float | NDArray) -> float | NDArray:
        """Na inactivation backward rate."""
        if self.model == "cortical":
            return 4.0 / (1.0 + np.exp(-(V - self._VT - 40.0) / 5.0))
        return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))

    def _alpha_n(self, V: float | NDArray) -> float | NDArray:
        """K activation forward rate."""
        if self.model == "cortical":
            dV = V - self._VT - 15.0
            return np.where(
                np.abs(dV) < 1e-7,
                0.032 * 5.0,
                -0.032 * dV / (np.exp(-dV / 5.0) - 1.0),
            )
        dV = V + 55.0
        return np.where(np.abs(dV) < 1e-7, 0.1, 0.01 * dV / (1.0 - np.exp(-dV / 10.0)))

    def _beta_n(self, V: float | NDArray) -> float | NDArray:
        """K activation backward rate."""
        if self.model == "cortical":
            return 0.5 * np.exp(-(V - self._VT - 10.0) / 40.0)
        return 0.125 * np.exp(-(V + 65.0) / 80.0)

    # ------------------------------------------------------------------
    # Q10 temperature scaling
    # ------------------------------------------------------------------

    def _q10_scale(self) -> float:
        """Q10 temperature scaling factor relative to model baseline."""
        return self._q10_base ** ((self.temperature - self._ref_temp) / 10.0)

    def _q10_for_temperature(self, temp: float) -> float:
        """Q10 scaling at an arbitrary temperature [°C]."""
        return self._q10_base ** ((temp - self._ref_temp) / 10.0)

    # ------------------------------------------------------------------
    # Gate and current helpers
    # ------------------------------------------------------------------

    def _steady_state_gates(self, V: float) -> tuple[float, float, float]:
        """Compute steady-state gate values at voltage *V*."""
        am = float(self._alpha_m(np.float64(V)))
        bm = float(self._beta_m(np.float64(V)))
        ah = float(self._alpha_h(np.float64(V)))
        bh = float(self._beta_h(np.float64(V)))
        an = float(self._alpha_n(np.float64(V)))
        bn = float(self._beta_n(np.float64(V)))
        return am / (am + bm), ah / (ah + bh), an / (an + bn)

    def _mechano_current(
        self,
        V: float,
        mechano_gates: list[float],
        tension: float,
    ) -> tuple[float, list[float]]:
        """Total mechanosensitive current and updated gate derivatives.

        Uses dynamic first-order gating: dw/dt = (w_inf - w) / tau.

        Returns
        -------
        I_mechano : float
            Total mechano current [uA/cm^2].
        dw_list : list[float]
            Gate derivatives for each channel.

        """
        I_total = 0.0
        dw_list = []
        for ch, w in zip(self.mechano_channels, mechano_gates, strict=True):
            w_inf = ch.open_probability(tension)
            tau = ch.time_constant(tension)
            dw = (w_inf - w) / tau
            I_ch = ch.g_density * w * (ch.E_rev - V)
            I_total += I_ch
            dw_list.append(dw)
        return I_total, dw_list

    # ------------------------------------------------------------------
    # RK4 integrator
    # ------------------------------------------------------------------

    def _rk4_step(
        self,
        V: float,
        m: float,
        h: float,
        n: float,
        mechano_gates: list[float],
        I_stim: float,
        tension: float,
        dt: float,
        phi: float,
        extra_current_fn: Callable[[float, int], float] | None = None,
        step_index: int = 0,
    ) -> tuple[float, float, float, float, list[float]]:
        """Single RK4 integration step.

        ``extra_current_fn`` (optional) is called as
        ``extra_current_fn(V, step_index) -> float`` and returns an
        additional inward (depolarising-positive) current density
        [μA/cm²], used to inject voltage-dependent synaptic currents
        (see :func:`ultraneuron.synapse.simulate_synaptic_relay`).
        The mechanosensitive Na⁺ activation shift ``k_mechano_nav`` is
        applied here by evaluating the Na⁺ activation rates at the
        tension-shifted voltage ``V + k_mechano_nav · tension``.
        """
        # Kubanek 2016: tension hyperpolarises the Na⁺ activation V_half.
        # Evaluating the m-gate rates at (V + S) shifts m_inf(V) to the
        # left by S = k_mechano_nav · tension. Applied to activation only
        # (not inactivation) so the net effect is a current *increase*.
        v_shift = self.k_mechano_nav * tension

        def derivs(
            V: float,
            m: float,
            h: float,
            n: float,
            w_gates: list[float],
        ) -> tuple[float, float, float, float, list[float]]:
            V_act = V + v_shift
            am = float(self._alpha_m(np.float64(V_act))) * phi
            bm = float(self._beta_m(np.float64(V_act))) * phi
            ah = float(self._alpha_h(np.float64(V))) * phi
            bh = float(self._beta_h(np.float64(V))) * phi
            an = float(self._alpha_n(np.float64(V))) * phi
            bn = float(self._beta_n(np.float64(V))) * phi

            I_Na = self.g_Na * (m**3) * h * (V - self.E_Na)
            I_K = self.g_K * (n**4) * (V - self.E_K)
            I_L = self.g_L * (V - self.E_L)
            I_mechano, dw_list = self._mechano_current(V, w_gates, tension)
            I_extra = (
                float(extra_current_fn(V, step_index))
                if extra_current_fn is not None
                else 0.0
            )

            dV = (-I_Na - I_K - I_L + I_mechano + I_stim + I_extra) / self.C_m
            dm = am * (1.0 - m) - bm * m
            dh = ah * (1.0 - h) - bh * h
            dn = an * (1.0 - n) - bn * n
            return dV, dm, dh, dn, dw_list

        k1V, k1m, k1h, k1n, k1w = derivs(V, m, h, n, mechano_gates)
        w2 = [w + 0.5 * dt * dw for w, dw in zip(mechano_gates, k1w, strict=True)]
        k2V, k2m, k2h, k2n, k2w = derivs(
            V + 0.5 * dt * k1V,
            m + 0.5 * dt * k1m,
            h + 0.5 * dt * k1h,
            n + 0.5 * dt * k1n,
            w2,
        )
        w3 = [w + 0.5 * dt * dw for w, dw in zip(mechano_gates, k2w, strict=True)]
        k3V, k3m, k3h, k3n, k3w = derivs(
            V + 0.5 * dt * k2V,
            m + 0.5 * dt * k2m,
            h + 0.5 * dt * k2h,
            n + 0.5 * dt * k2n,
            w3,
        )
        w4 = [w + dt * dw for w, dw in zip(mechano_gates, k3w, strict=True)]
        k4V, k4m, k4h, k4n, k4w = derivs(
            V + dt * k3V,
            m + dt * k3m,
            h + dt * k3h,
            n + dt * k3n,
            w4,
        )

        V_new = V + (dt / 6.0) * (k1V + 2 * k2V + 2 * k3V + k4V)
        m_new = m + (dt / 6.0) * (k1m + 2 * k2m + 2 * k3m + k4m)
        h_new = h + (dt / 6.0) * (k1h + 2 * k2h + 2 * k3h + k4h)
        n_new = n + (dt / 6.0) * (k1n + 2 * k2n + 2 * k3n + k4n)

        w_new = []
        for i, _w in enumerate(mechano_gates):
            dw_avg = (k1w[i] + 2 * k2w[i] + 2 * k3w[i] + k4w[i]) / 6.0
            w_new.append(_w + dt * dw_avg)

        # Clamp gates to [0, 1]
        m_new = max(0.0, min(1.0, m_new))
        h_new = max(0.0, min(1.0, h_new))
        n_new = max(0.0, min(1.0, n_new))
        w_new = [max(0.0, min(1.0, w)) for w in w_new]

        return V_new, m_new, h_new, n_new, w_new

    @staticmethod
    def _detect_spikes(
        V_trace: NDArray[np.floating],
        threshold: float = 0.0,
    ) -> NDArray[np.integer]:
        """Detect spike times as upward threshold crossings.

        Parameters
        ----------
        V_trace : 1-D array
            Voltage trace [mV].
        threshold : float
            Crossing threshold [mV]. Default 0.

        Returns
        -------
        spike_indices : 1-D integer array
            Indices where spikes occur.

        """
        above = V_trace > threshold
        crossings = np.diff(above.astype(np.int8)) > 0
        return np.nonzero(crossings)[0] + 1

    def simulate(
        self,
        dt: float = 0.01,
        duration: float = 50.0,
        I_stim: NDArray[np.floating] | float = 0.0,
        tension_trace: NDArray[np.floating] | float = 0.0,
        temperature_trace: NDArray[np.floating] | float | None = None,
        extra_current_fn: Callable[[float, int], float] | None = None,
    ) -> dict[str, NDArray[np.floating]]:
        """Run a point-neuron simulation.

        Parameters
        ----------
        dt : float
            Time step [ms]. Default 0.01.
        duration : float
            Total simulation time [ms]. Default 50.
        I_stim : ndarray or float
            Injected stimulus current density [uA/cm^2].
            Scalar for constant, or 1-D array of length ``n_steps``.
        tension_trace : ndarray or float
            Membrane tension time series [mN/m].
            Scalar for constant, or 1-D array of length ``n_steps``.
            Also drives the ``k_mechano_nav`` activation shift.
        temperature_trace : ndarray, float, or None
            Temperature [°C] for Q10 scaling. When ``None`` (default),
            uses ``self.temperature``. When scalar, uses that constant.
            When 1-D array of length ``n_steps``, recomputes phi per
            timestep from the temperature trace.
        extra_current_fn : callable or None
            Optional ``f(V, step_index) -> float`` returning an extra
            inward (depolarising-positive) current density [μA/cm²]
            evaluated each RK4 sub-step. Used to inject voltage-dependent
            synaptic currents in the synaptic-relay pathway. Default
            ``None`` (no extra current — identical to the base model).

        Returns
        -------
        dict
            ``'t'`` : time vector [ms].
            ``'V'`` : voltage trace [mV].
            ``'m'``, ``'h'``, ``'n'`` : HH gate traces.
            ``'I_Na'``, ``'I_K'``, ``'I_L'`` : ionic currents [uA/cm^2].
            ``'I_mechano'`` : total mechanosensitive current [uA/cm^2].
            ``'I_syn'`` : injected ``extra_current_fn`` current [uA/cm^2]
            (zeros when no callback is supplied).
            ``'spike_times'`` : spike time values [ms].

        """
        n_steps = int(duration / dt)

        # Build per-step phi array from temperature input
        ref = self._ref_temp
        q10 = self._q10_base
        if temperature_trace is None:
            phi_arr = np.full(n_steps, self._q10_scale())
        elif np.isscalar(temperature_trace):
            phi_val = q10 ** ((float(temperature_trace) - ref) / 10.0)
            phi_arr = np.full(n_steps, phi_val)
        else:
            temp_arr = np.asarray(temperature_trace, dtype=np.float64)
            phi_arr = q10 ** ((temp_arr - ref) / 10.0)

        # Expand scalar inputs
        if np.isscalar(I_stim):
            I_stim_arr = np.full(n_steps, float(I_stim))
        else:
            I_stim_arr = np.asarray(I_stim, dtype=np.float64)

        if np.isscalar(tension_trace):
            tension_arr = np.full(n_steps, float(tension_trace))
        else:
            tension_arr = np.asarray(tension_trace, dtype=np.float64)

        # Initialise state
        V = self.V_rest
        m0, h0, n0 = self._steady_state_gates(V)
        m, h, n = m0, h0, n0

        # Mechano gate initialisation (steady state at initial tension)
        mechano_gates = []
        for ch in self.mechano_channels:
            w_init = float(ch.open_probability(tension_arr[0]))
            mechano_gates.append(w_init)

        # Storage
        V_out = np.zeros(n_steps)
        m_out = np.zeros(n_steps)
        h_out = np.zeros(n_steps)
        n_out = np.zeros(n_steps)
        I_Na_out = np.zeros(n_steps)
        I_K_out = np.zeros(n_steps)
        I_L_out = np.zeros(n_steps)
        I_mech_out = np.zeros(n_steps)
        I_syn_out = np.zeros(n_steps)

        for i in range(n_steps):
            # Record
            V_out[i] = V
            m_out[i] = m
            h_out[i] = h
            n_out[i] = n
            I_Na_out[i] = self.g_Na * (m**3) * h * (V - self.E_Na)
            I_K_out[i] = self.g_K * (n**4) * (V - self.E_K)
            I_L_out[i] = self.g_L * (V - self.E_L)
            I_mech, _ = self._mechano_current(V, mechano_gates, tension_arr[i])
            I_mech_out[i] = I_mech
            if extra_current_fn is not None:
                I_syn_out[i] = float(extra_current_fn(V, i))

            # Advance
            V, m, h, n, mechano_gates = self._rk4_step(
                V,
                m,
                h,
                n,
                mechano_gates,
                I_stim_arr[i],
                tension_arr[i],
                dt,
                phi_arr[i],
                extra_current_fn,
                i,
            )

        t = np.arange(n_steps) * dt
        spike_idx = self._detect_spikes(V_out)
        spike_times = t[spike_idx] if len(spike_idx) > 0 else np.array([])

        return {
            "t": t,
            "V": V_out,
            "m": m_out,
            "h": h_out,
            "n": n_out,
            "I_Na": I_Na_out,
            "I_K": I_K_out,
            "I_L": I_L_out,
            "I_mechano": I_mech_out,
            "I_syn": I_syn_out,
            "spike_times": spike_times,
        }

    def _simulate_batch(
        self,
        tensions: NDArray[np.floating],
        phi_arr: NDArray[np.floating],
        dt: float,
        n_steps: int,
        I_stim: float,
        tensions_2d: NDArray[np.floating] | None = None,
        backend: str = "numba",
    ) -> NDArray[np.integer]:
        """Vectorized RK4 integration over N voxels.

        Runs the full HH + mechano-channel model simultaneously across
        all voxels.  Each voxel has its own constant (or time-varying)
        tension and Q10 scaling factor, but shares the same neuron
        parameters.

        Parameters
        ----------
        tensions : ndarray, shape (N,)
            Constant membrane tension per voxel [mN/m].
            Used for the constant-tension kernel and for pre-computing
            channel steady-state when ``tensions_2d`` is None.
        phi_arr : ndarray, shape (N,)
            Q10 temperature scaling factor per voxel.
        dt : float
            Time step [ms].
        n_steps : int
            Number of integration steps.
        I_stim : float
            Background stimulus current [uA/cm^2].
        tensions_2d : ndarray, shape (n_steps, N), optional
            Time-varying tension per voxel per step [mN/m].
            When provided, each voxel sees a different tension at each
            time step. Overrides ``tensions`` for integration (though
            ``tensions`` is still used if ``tensions_2d`` is None).

        Returns
        -------
        spike_counts : ndarray, shape (N,), dtype int32
            Number of upward zero-crossings (spikes) per voxel.

        """
        n_ch = len(self.mechano_channels)
        m0, h0, n0 = self._steady_state_gates(self.V_rest)
        N = len(tensions)

        # CUDA backend selection — fall back to Numba if unavailable.
        from . import cuda_backend as _cuda  # noqa: PLC0415
        _use_cuda = False
        if backend == "cuda":
            if not _cuda.is_cuda_available():
                _cuda.maybe_warn_unavailable("cuda")
            elif n_ch > _cuda._MAX_CH:
                import warnings as _warn  # noqa: PLC0415
                _warn.warn(
                    f"CUDA backend supports up to {_cuda._MAX_CH} mechanosensitive "
                    f"channels; got n_ch={n_ch}. Falling back to Numba.",
                    RuntimeWarning, stacklevel=2,
                )
            else:
                _use_cuda = True

        ch_g_dens = np.array([ch.g_density for ch in self.mechano_channels], dtype=np.float64)
        ch_E_revs = np.array([ch.E_rev for ch in self.mechano_channels], dtype=np.float64)

        if tensions_2d is not None:
            # Time-varying path
            ch_T_halfs = np.array([ch.T_half for ch in self.mechano_channels], dtype=np.float64)
            ch_delta_As = np.array([ch.delta_A for ch in self.mechano_channels], dtype=np.float64)
            ch_tau_0s = np.array([ch.tau_0 for ch in self.mechano_channels], dtype=np.float64)
            kT = self.mechano_channels[0]._kT if n_ch > 0 else 4.114

            if _use_cuda:
                return _cuda.rk4_batch_cuda_tv(
                    self.V_rest, self.g_Na, self.g_K, self.g_L,
                    self.E_Na, self.E_K, self.E_L, self.C_m,
                    self.model == "cortical", self._VT,
                    tensions_2d.astype(np.float64),
                    phi_arr, dt, n_steps, I_stim,
                    ch_g_dens, ch_E_revs,
                    ch_T_halfs, ch_delta_As, ch_tau_0s, kT,
                    m0, h0, n0,
                )

            if _HAS_NUMBA:
                return _rk4_batch_numba_tv(
                    self.V_rest,
                    self.g_Na,
                    self.g_K,
                    self.g_L,
                    self.E_Na,
                    self.E_K,
                    self.E_L,
                    self.C_m,
                    self.model == "cortical",
                    self._VT,
                    tensions_2d.astype(np.float64),
                    phi_arr,
                    dt,
                    n_steps,
                    I_stim,
                    ch_g_dens,
                    ch_E_revs,
                    ch_T_halfs,
                    ch_delta_As,
                    ch_tau_0s,
                    kT,
                    m0,
                    h0,
                    n0,
                )

            return self._simulate_batch_numpy_tv(
                tensions_2d,
                phi_arr,
                dt,
                n_steps,
                I_stim,
                ch_g_dens,
                ch_E_revs,
                ch_T_halfs,
                ch_delta_As,
                ch_tau_0s,
                kT,
                m0,
                h0,
                n0,
            )

        # Constant-tension path: precompute channel steady-state
        ch_w_inf = np.empty((max(n_ch, 1), N))
        ch_tau_inv = np.empty((max(n_ch, 1), N))
        for k, ch in enumerate(self.mechano_channels):
            ch_w_inf[k] = ch.open_probability(tensions)
            ch_tau_inv[k] = 1.0 / ch.time_constant(tensions)

        if _use_cuda:
            return _cuda.rk4_batch_cuda(
                self.V_rest, self.g_Na, self.g_K, self.g_L,
                self.E_Na, self.E_K, self.E_L, self.C_m,
                self.model == "cortical", self._VT,
                tensions, phi_arr, dt, n_steps, I_stim,
                ch_g_dens if n_ch > 0 else np.empty(0),
                ch_E_revs if n_ch > 0 else np.empty(0),
                ch_w_inf[:n_ch], ch_tau_inv[:n_ch],
                m0, h0, n0,
            )

        if _HAS_NUMBA:
            return _rk4_batch_numba(
                self.V_rest,
                self.g_Na,
                self.g_K,
                self.g_L,
                self.E_Na,
                self.E_K,
                self.E_L,
                self.C_m,
                self.model == "cortical",
                self._VT,
                tensions,
                phi_arr,
                dt,
                n_steps,
                I_stim,
                ch_g_dens if n_ch > 0 else np.empty(0),
                ch_E_revs if n_ch > 0 else np.empty(0),
                ch_w_inf[:n_ch],
                ch_tau_inv[:n_ch],
                m0,
                h0,
                n0,
            )

        return self._simulate_batch_numpy(
            tensions,
            phi_arr,
            dt,
            n_steps,
            I_stim,
            ch_w_inf[:n_ch],
            ch_tau_inv[:n_ch],
            ch_g_dens if n_ch > 0 else np.empty(0),
            ch_E_revs if n_ch > 0 else np.empty(0),
            m0,
            h0,
            n0,
        )

    def _simulate_batch_numpy(
        self,
        tensions: NDArray[np.floating],
        phi_arr: NDArray[np.floating],
        dt: float,
        n_steps: int,
        I_stim: float,
        w_inf: NDArray[np.floating],
        tau_inv: NDArray[np.floating],
        g_dens: NDArray[np.floating],
        E_revs: NDArray[np.floating],
        m0: float,
        h0: float,
        n0: float,
    ) -> NDArray[np.integer]:
        """Numpy fallback for ``_simulate_batch`` (no numba)."""
        N = len(tensions)
        n_ch = len(g_dens)

        V = np.full(N, self.V_rest)
        m = np.full(N, m0)
        h = np.full(N, h0)
        n = np.full(N, n0)
        w = w_inf.copy()

        spike_counts = np.zeros(N, dtype=np.int32)
        V_prev = np.empty(N)

        def derivs(
            V: NDArray[np.floating],
            m: NDArray[np.floating],
            h: NDArray[np.floating],
            n: NDArray[np.floating],
            w_gates: NDArray[np.floating],
        ) -> tuple[
            NDArray[np.floating],
            NDArray[np.floating],
            NDArray[np.floating],
            NDArray[np.floating],
            NDArray[np.floating],
        ]:
            am = self._alpha_m(V) * phi_arr
            bm = self._beta_m(V) * phi_arr
            ah = self._alpha_h(V) * phi_arr
            bh = self._beta_h(V) * phi_arr
            an = self._alpha_n(V) * phi_arr
            bn = self._beta_n(V) * phi_arr

            I_Na = self.g_Na * (m**3) * h * (V - self.E_Na)
            I_K = self.g_K * (n**4) * (V - self.E_K)
            I_L = self.g_L * (V - self.E_L)

            I_mechano = np.zeros(N)
            dw = np.empty_like(w_gates)
            for k in range(n_ch):
                dw[k] = (w_inf[k] - w_gates[k]) * tau_inv[k]
                I_mechano += g_dens[k] * w_gates[k] * (E_revs[k] - V)

            dV = (-I_Na - I_K - I_L + I_mechano + I_stim) / self.C_m
            dm = am * (1.0 - m) - bm * m
            dh = ah * (1.0 - h) - bh * h
            dn = an * (1.0 - n) - bn * n
            return dV, dm, dh, dn, dw

        for _i in range(n_steps):
            V_prev[:] = V

            k1V, k1m, k1h, k1n, k1w = derivs(V, m, h, n, w)

            w2 = w + 0.5 * dt * k1w
            k2V, k2m, k2h, k2n, k2w = derivs(
                V + 0.5 * dt * k1V,
                m + 0.5 * dt * k1m,
                h + 0.5 * dt * k1h,
                n + 0.5 * dt * k1n,
                w2,
            )

            w3 = w + 0.5 * dt * k2w
            k3V, k3m, k3h, k3n, k3w = derivs(
                V + 0.5 * dt * k2V,
                m + 0.5 * dt * k2m,
                h + 0.5 * dt * k2h,
                n + 0.5 * dt * k2n,
                w3,
            )

            w4 = w + dt * k3w
            k4V, k4m, k4h, k4n, k4w = derivs(
                V + dt * k3V,
                m + dt * k3m,
                h + dt * k3h,
                n + dt * k3n,
                w4,
            )

            s = dt / 6.0
            V += s * (k1V + 2.0 * k2V + 2.0 * k3V + k4V)
            m += s * (k1m + 2.0 * k2m + 2.0 * k3m + k4m)
            h += s * (k1h + 2.0 * k2h + 2.0 * k3h + k4h)
            n += s * (k1n + 2.0 * k2n + 2.0 * k3n + k4n)
            w += dt * (k1w + 2.0 * k2w + 2.0 * k3w + k4w) / 6.0

            np.clip(m, 0.0, 1.0, out=m)
            np.clip(h, 0.0, 1.0, out=h)
            np.clip(n, 0.0, 1.0, out=n)
            np.clip(w, 0.0, 1.0, out=w)

            spike_counts += ((V > 0.0) & (V_prev <= 0.0)).astype(np.int32)

        return spike_counts

    def _simulate_batch_numpy_tv(
        self,
        tensions_2d: NDArray[np.floating],
        phi_arr: NDArray[np.floating],
        dt: float,
        n_steps: int,
        I_stim: float,
        g_dens: NDArray[np.floating],
        E_revs: NDArray[np.floating],
        T_halfs: NDArray[np.floating],
        delta_As: NDArray[np.floating],
        tau_0s: NDArray[np.floating],
        kT: float,
        m0: float,
        h0: float,
        n0: float,
    ) -> NDArray[np.integer]:
        """Numpy fallback for time-varying tension batch (no numba)."""
        from .channels import _safe_sigmoid

        N = tensions_2d.shape[1]
        n_ch = len(g_dens)

        V = np.full(N, self.V_rest)
        m = np.full(N, m0)
        h = np.full(N, h0)
        n = np.full(N, n0)

        # Init mechano gates at steady state for t=0 tension
        w = np.empty((n_ch, N))
        for k in range(n_ch):
            exponent = -delta_As[k] * (tensions_2d[0] - T_halfs[k]) / kT
            w[k] = _safe_sigmoid(exponent)

        spike_counts = np.zeros(N, dtype=np.int32)
        V_prev = np.empty(N)

        for i in range(n_steps):
            V_prev[:] = V
            T_now = tensions_2d[i]

            # Recompute w_inf and tau_inv for current tension
            w_inf = np.empty((n_ch, N))
            tau_inv = np.empty((n_ch, N))
            for k in range(n_ch):
                exponent = -delta_As[k] * (T_now - T_halfs[k]) / kT
                po = _safe_sigmoid(exponent)
                w_inf[k] = po
                denom = 4.0 * po * (1.0 - po) + 0.01
                tau_inv[k] = denom / tau_0s[k]

            def _derivs_tv(
                V_: NDArray,
                m_: NDArray,
                h_: NDArray,
                n_: NDArray,
                w_: NDArray,
                *,
                _w_inf: NDArray = w_inf,
                _tau_inv: NDArray = tau_inv,
            ) -> tuple:
                am = self._alpha_m(V_) * phi_arr
                bm = self._beta_m(V_) * phi_arr
                ah = self._alpha_h(V_) * phi_arr
                bh = self._beta_h(V_) * phi_arr
                an = self._alpha_n(V_) * phi_arr
                bn = self._beta_n(V_) * phi_arr

                I_Na = self.g_Na * (m_**3) * h_ * (V_ - self.E_Na)
                I_K = self.g_K * (n_**4) * (V_ - self.E_K)
                I_L = self.g_L * (V_ - self.E_L)

                I_mechano = np.zeros(N)
                dw = np.empty_like(w_)
                for k in range(n_ch):
                    dw[k] = (_w_inf[k] - w_[k]) * _tau_inv[k]
                    I_mechano += g_dens[k] * w_[k] * (E_revs[k] - V_)

                dV = (-I_Na - I_K - I_L + I_mechano + I_stim) / self.C_m
                dm = am * (1.0 - m_) - bm * m_
                dh = ah * (1.0 - h_) - bh * h_
                dn = an * (1.0 - n_) - bn * n_
                return dV, dm, dh, dn, dw

            k1V, k1m, k1h, k1n, k1w = _derivs_tv(V, m, h, n, w)
            k2V, k2m, k2h, k2n, k2w = _derivs_tv(
                V + 0.5 * dt * k1V,
                m + 0.5 * dt * k1m,
                h + 0.5 * dt * k1h,
                n + 0.5 * dt * k1n,
                w + 0.5 * dt * k1w,
            )
            k3V, k3m, k3h, k3n, k3w = _derivs_tv(
                V + 0.5 * dt * k2V,
                m + 0.5 * dt * k2m,
                h + 0.5 * dt * k2h,
                n + 0.5 * dt * k2n,
                w + 0.5 * dt * k2w,
            )
            k4V, k4m, k4h, k4n, k4w = _derivs_tv(
                V + dt * k3V,
                m + dt * k3m,
                h + dt * k3h,
                n + dt * k3n,
                w + dt * k3w,
            )

            s = dt / 6.0
            V += s * (k1V + 2.0 * k2V + 2.0 * k3V + k4V)
            m += s * (k1m + 2.0 * k2m + 2.0 * k3m + k4m)
            h += s * (k1h + 2.0 * k2h + 2.0 * k3h + k4h)
            n += s * (k1n + 2.0 * k2n + 2.0 * k3n + k4n)
            w += dt * (k1w + 2.0 * k2w + 2.0 * k3w + k4w) / 6.0

            np.clip(m, 0.0, 1.0, out=m)
            np.clip(h, 0.0, 1.0, out=h)
            np.clip(n, 0.0, 1.0, out=n)
            np.clip(w, 0.0, 1.0, out=w)

            spike_counts += ((V > 0.0) & (V_prev <= 0.0)).astype(np.int32)

        return spike_counts


def simulate_map(
    strain_3d: NDArray[np.floating],
    neuron: MechanoHHNeuron,
    K_A: float = 0.25,
    coupling_factor: float = 1.0,
    dt: float = 0.01,
    duration: float = 50.0,
    I_stim: float = 0.0,
    tension_threshold: float | None = None,
    temperature_3d: NDArray[np.floating] | None = None,
    backend: str = "numba",
) -> dict[str, NDArray]:
    """Run neuron simulations across a 3-D strain field.

    For each voxel, converts deviatoric strain to membrane tension and
    runs a point-neuron simulation. Voxels below ``tension_threshold``
    are skipped for efficiency.

    Parameters
    ----------
    strain_3d : ndarray, shape (nx, ny, nz)
        Deviatoric strain magnitude (from ``compute_strain_invariants``).
    neuron : MechanoHHNeuron
        Neuron model (with mechanosensitive channels configured).
    K_A : float
        Bilayer area expansion modulus [N/m]. Default 0.25.
    coupling_factor : float
        Tissue-to-membrane coupling factor. Default 1.0.
    dt : float
        Time step [ms]. Default 0.01.
    duration : float
        Simulation duration [ms]. Default 50.
    I_stim : float
        Background stimulus current [uA/cm^2]. Default 0.
    tension_threshold : float or None
        Minimum tension [mN/m] to simulate. If None, uses binary search
        to estimate the minimum tension that produces spikes.
    temperature_3d : ndarray (nx, ny, nz) or None
        Absolute temperature at each voxel [°C]. When ``None``,
        ``neuron.temperature`` is used everywhere (isothermal).

    Returns
    -------
    dict
        ``'firing_map'`` : bool array (nx, ny, nz) — True where spikes occur.
        ``'spike_count_map'`` : int array (nx, ny, nz) — number of spikes.
        ``'threshold_tension'`` : float — estimated minimum firing tension [mN/m].

    """
    from .channels import strain_to_membrane_tension

    nx, ny, nz = strain_3d.shape

    if temperature_3d is not None and temperature_3d.shape != (nx, ny, nz):
        msg = (
            f"temperature_3d shape {temperature_3d.shape} does not match "
            f"strain_3d shape {(nx, ny, nz)}"
        )
        raise ValueError(msg)

    tension_3d = strain_to_membrane_tension(strain_3d, K_A, coupling_factor)

    # Estimate threshold via binary search if not provided
    if tension_threshold is None:
        tension_threshold = _find_threshold_tension(
            neuron,
            dt=dt,
            duration=duration,
            I_stim=I_stim,
            T_min=0.0,
            T_max=float(np.max(tension_3d)) * 1.1,
        )

    firing_map = np.zeros((nx, ny, nz), dtype=bool)
    spike_count_map = np.zeros((nx, ny, nz), dtype=np.int32)

    # Gather voxels above threshold and run vectorized batch
    mask = tension_3d >= tension_threshold
    if np.any(mask):
        tensions = tension_3d[mask]
        n_steps = int(duration / dt)

        # Compute per-voxel Q10 scaling
        if temperature_3d is not None:
            phi_arr = neuron._q10_base ** ((temperature_3d[mask] - neuron._ref_temp) / 10.0)
        else:
            phi_arr = np.full(len(tensions), neuron._q10_scale())

        spike_counts = neuron._simulate_batch(
            tensions, phi_arr, dt, n_steps, I_stim, backend=backend,
        )

        spike_count_map[mask] = spike_counts
        firing_map[mask] = spike_counts > 0

    return {
        "firing_map": firing_map,
        "spike_count_map": spike_count_map,
        "threshold_tension": tension_threshold,
    }


def _find_threshold_tension(
    neuron: MechanoHHNeuron,
    dt: float,
    duration: float,
    I_stim: float,
    T_min: float,
    T_max: float,
    tol: float = 0.01,
    max_iter: int = 30,
) -> float:
    """Find minimum constant tension that produces a spike.

    Uses a vectorized sweep: evaluates 64 tension values in a single
    batch call to bracket the threshold, then refines with a second
    batch if needed. Much faster than the previous 30-iteration scalar
    binary search.
    """
    if T_max <= 0:
        return 0.0

    n_steps = int(duration / dt)
    n_probe = 64

    for _ in range(max_iter):
        tensions = np.linspace(T_min, T_max, n_probe)
        phi_arr = np.full(n_probe, neuron._q10_scale())
        spike_counts = neuron._simulate_batch(tensions, phi_arr, dt, n_steps, I_stim)

        fires = spike_counts > 0
        if not np.any(fires):
            return T_max

        first_fire = int(np.argmax(fires))
        if first_fire == 0:
            T_max = tensions[0]
        else:
            T_min = float(tensions[first_fire - 1])
            T_max = float(tensions[first_fire])

        if T_max - T_min < tol:
            break

    return T_min


def bioheat_to_absolute_temperature(
    delta_T: NDArray[np.floating] | float,
    baseline: float = 37.0,
) -> NDArray[np.floating] | float:
    """Convert bioheat temperature rise to absolute temperature.

    The Pennes bioheat solver outputs a temperature *rise* ΔT [K] above
    baseline. The HH neuron model expects absolute temperature [°C].
    This helper performs the conversion: ``T_abs = baseline + delta_T``.

    Parameters
    ----------
    delta_T : ndarray or float
        Temperature rise from bioheat solver [K].
    baseline : float
        Baseline body temperature [°C]. Default 37.0.

    Returns
    -------
    ndarray or float
        Absolute temperature [°C], same shape as *delta_T*.

    """
    return baseline + delta_T


# ---------------------------------------------------------------------------
# Mechanosensitive voltage-gated Na⁺/Ca²⁺ shift (Kubanek et al. 2016)
# ---------------------------------------------------------------------------
# Kubanek et al. 2016 (Sci Rep, "Ultrasound modulates ion channel currents")
# patch-clamped Nav/Cav/Kv channels under ultrasound and found that the drive
# shifts the activation/inactivation voltage dependence, increasing the
# Na⁺/Ca²⁺ current. We model this as a tension-proportional hyperpolarising
# shift of the activation half-voltage:
#
#     V_half(T) = V_half0 - K_MECHANO_NAV · T          [mV]
#
# The canonical magnitude below is CALIBRATED (not a directly tabulated
# Kubanek number): K_MECHANO_NAV is chosen so a few-mN/m membrane tension
# produces a ~mV-scale activation shift and a tens-of-percent Na⁺ current
# change, the order of effect Kubanek report. Users should refine it against
# their own exposure/preparation. See provenance ``"MechanoNav"``.
K_MECHANO_NAV: float = 0.3
"""Canonical Na⁺/Ca²⁺ activation shift [mV per (mN/m)] (Kubanek 2016, calibrated).

This is the value :func:`nav_current_gain` uses by default and the
recommended setting for :attr:`MechanoHHNeuron.k_mechano_nav` (whose own
default is 0.0, i.e. the pathway off, to preserve the baseline neuron).
"""

NAV_ACT_VHALF: float = -35.0   # mV — representative mammalian Nav activation midpoint
NAV_ACT_SLOPE: float = 7.0     # mV — Boltzmann slope of the activation curve
NAV_GAIN_EVAL_V: float = -55.0  # mV — near-threshold reference voltage for the scalar gain


def nav_current_gain(
    peak_tension_mNm: float,
    k_mechano: float = K_MECHANO_NAV,
    v_half0: float = NAV_ACT_VHALF,
    k_a: float = NAV_ACT_SLOPE,
    v_eval: float = NAV_GAIN_EVAL_V,
) -> float:
    """Relative Na⁺/Ca²⁺ current gain from the Kubanek 2016 activation shift.

    DOWNSTREAM SCALAR ENTRY POINT for the ``neuromod_parameters`` pipeline.

    Returns the fold-change in the voltage-gated Na⁺ activation open
    fraction at a near-threshold reference voltage caused by a
    hyperpolarising shift ``S = k_mechano · T`` of the activation
    half-voltage:

        gain(T) = m_inf(v_eval; V_half0 - S) / m_inf(v_eval; V_half0)

    where ``m_inf(V; V_half) = 1 / (1 + exp(-(V - V_half) / k_a))``.

    Parameters
    ----------
    peak_tension_mNm : float
        Peak membrane tension [mN/m]. Clamped at 0 below.
    k_mechano : float
        Activation-shift sensitivity [mV per (mN/m)]. Default
        :data:`K_MECHANO_NAV`.
    v_half0, k_a, v_eval : float
        Baseline activation midpoint, Boltzmann slope, and reference
        evaluation voltage [mV].

    Returns
    -------
    float
        ``gain ≥ 1``; ``1.0`` at zero tension, monotonically increasing
        with tension. Multiply a baseline Na⁺/Ca²⁺ excitation score by
        this factor.
    """
    T = max(float(peak_tension_mNm), 0.0)
    shift = k_mechano * T

    def _m_inf(v_half: float) -> float:
        x = (v_eval - v_half) / k_a
        # stable logistic
        return 1.0 / (1.0 + np.exp(-x)) if x >= 0 else np.exp(x) / (1.0 + np.exp(x))

    m_base = _m_inf(v_half0)
    m_shift = _m_inf(v_half0 - shift)
    return float(m_shift / m_base)
