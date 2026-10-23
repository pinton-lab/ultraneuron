"""Multi-compartment Hodgkin-Huxley neuron with mechanosensitive coupling.

Couples a dendrite-soma-AIS skeleton (see ``compartments.py``) by
axial conductances and integrates the full HH membrane equation in
each compartment with RK4.  Mechanosensitive channels are present in
all compartments at compartment-specific density (typically
dendrite-heavy for Piezo1).

The integrator is fully vectorised: state is laid out as
``(3_compartments, N_voxels)`` for V/gates and integrated in a single
numpy expression per RK4 stage.

Backward compatibility
----------------------
This module adds new public types — ``MulticompartmentMechanoHHNeuron``
and the helpers exported from ``compartments.py``.  The existing
``MechanoHHNeuron`` class and ``simulate_map`` function are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from .calcium import CalciumDynamics, KCaChannel
from .compartments import CompartmentalGeometry, cortical_pyramidal_3comp
from .neuron import MECHANO_INIT_MODES, MechanoHHNeuron, _check_mechano_init
from .nice import NICEModel
from .thermosensors import TRPThermosensorChannel

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .channels import MechanosensitiveChannel


# Stable sigmoid (vectorised). Mirrors channels._safe_sigmoid but
# specialised for batched (...) shapes used here.
def _safe_sigmoid(x: NDArray[np.floating]) -> NDArray[np.floating]:
    pos = x >= 0
    out = np.empty_like(x, dtype=np.float64)
    e_neg = np.exp(-np.minimum(x, 500.0))
    e_pos = np.exp(np.maximum(x, -500.0))
    out = np.where(pos, e_neg / (1.0 + e_neg), 1.0 / (1.0 + e_pos))
    return out


# ---------------------------------------------------------------------------
# Numba-accelerated 3-compartment batch kernel.
# Fast path for the canonical Yaakub configuration: Piezo1 + (optional TREK1)
# strain channels with HH gating, no NICE, no piezo inactivation, no Ca,
# no KCa / TRP, constant per-voxel tension. Falls back to the NumPy path
# below when any of those features are enabled or when numba is missing.
# ---------------------------------------------------------------------------
try:
    from numba import njit, prange

    @njit(cache=True, fastmath=True)
    def _hh_rates_cortical(V: float, VT: float):
        """Pospischil cortical HH rate functions, scalar form."""
        dVm = V - VT - 13.0
        if abs(dVm) > 1e-7:
            am = -0.32 * dVm / (np.exp(-dVm / 4.0) - 1.0)
        else:
            am = 0.32 * 4.0
        dVm2 = V - VT - 40.0
        if abs(dVm2) > 1e-7:
            bm = 0.28 * dVm2 / (np.exp(dVm2 / 5.0) - 1.0)
        else:
            bm = 0.28 * 5.0
        ah = 0.128 * np.exp(-(V - VT - 17.0) / 18.0)
        bh = 4.0 / (1.0 + np.exp(-(V - VT - 40.0) / 5.0))
        dVn = V - VT - 15.0
        if abs(dVn) > 1e-7:
            an = -0.032 * dVn / (np.exp(-dVn / 5.0) - 1.0)
        else:
            an = 0.032 * 5.0
        bn = 0.5 * np.exp(-(V - VT - 10.0) / 40.0)
        return am, bm, ah, bh, an, bn

    @njit(cache=True, fastmath=True)
    def _hh_rates_squid(V: float):
        """Squid HH rate functions, scalar form."""
        dVm = V + 40.0
        if abs(dVm) > 1e-7:
            am = 0.1 * dVm / (1.0 - np.exp(-dVm / 10.0))
        else:
            am = 1.0
        bm = 4.0 * np.exp(-(V + 65.0) / 18.0)
        ah = 0.07 * np.exp(-(V + 65.0) / 20.0)
        bh = 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))
        dVn = V + 55.0
        if abs(dVn) > 1e-7:
            an = 0.01 * dVn / (1.0 - np.exp(-dVn / 10.0))
        else:
            an = 0.1
        bn = 0.125 * np.exp(-(V + 65.0) / 80.0)
        return am, bm, ah, bh, an, bn

    @njit(cache=True, fastmath=True)
    def _derivs_3comp_scalar(
        V0: float, V1: float, V2: float,
        m0: float, m1: float, m2: float,
        h0: float, h1: float, h2: float,
        n0: float, n1: float, n2: float,
        w_flat: np.ndarray,         # (3 * n_ch,)
        h_mech_flat: np.ndarray,    # (3 * n_ch,) — 1.0 if no inactivation
        Ca0: float, Ca1: float, Ca2: float,
        n_ch: int,
        # Per-compartment scalars
        g_Na0: float, g_Na1: float, g_Na2: float,
        g_K0: float, g_K1: float, g_K2: float,
        g_L0: float, g_L1: float, g_L2: float,
        E_Na: float, E_K: float, E_L: float,
        C_m0: float, C_m1: float, C_m2: float,
        # Coupling
        g_ds: float, g_sa: float,
        AsAd: float, AsAa: float,
        # Mechano
        g_mech: np.ndarray,         # (3, n_ch)
        E_revs: np.ndarray,         # (n_ch,)
        Ca_frac: np.ndarray,        # (n_ch,) — Ca²⁺ permeability fraction
        # Calcium pool dynamics
        flux_factor: float,         # μM·cm²/(μA·ms); 0 disables pool
        tau_decay_inv: float,       # 1/ms; 0 disables decay
        Ca_rest: float,             # μM
        # KCa / SK channels
        n_kca: int,
        kca_g: np.ndarray,          # (n_kca,) mS/cm²
        kca_E_K: np.ndarray,        # (n_kca,) mV
        kca_half: np.ndarray,       # (n_kca,) μM
        kca_n: np.ndarray,          # (n_kca,) Hill exponent
        kca_mask: np.ndarray,       # (3,) compartment mask
        # TRP thermosensors — P_o pre-evaluated at body temperature.
        n_trp: int,
        trp_g_Po: np.ndarray,       # (n_trp,) g_density·P_o(T)
        trp_E_rev: np.ndarray,      # (n_trp,) mV
        trp_mask: np.ndarray,       # (3,) compartment mask
        # Stim
        I0: float, I1: float, I2: float,
        # Q10
        phi: float,
        is_cortical: bool,  # noqa: FBT001
        VT: float,
    ):
        """Scalar 3-compartment derivatives. Returns flattened tuple."""
        # HH rate functions per compartment
        if is_cortical:
            am0, bm0, ah0, bh0, an0, bn0 = _hh_rates_cortical(V0, VT)
            am1, bm1, ah1, bh1, an1, bn1 = _hh_rates_cortical(V1, VT)
            am2, bm2, ah2, bh2, an2, bn2 = _hh_rates_cortical(V2, VT)
        else:
            am0, bm0, ah0, bh0, an0, bn0 = _hh_rates_squid(V0)
            am1, bm1, ah1, bh1, an1, bn1 = _hh_rates_squid(V1)
            am2, bm2, ah2, bh2, an2, bn2 = _hh_rates_squid(V2)
        am0 *= phi; bm0 *= phi; ah0 *= phi; bh0 *= phi; an0 *= phi; bn0 *= phi
        am1 *= phi; bm1 *= phi; ah1 *= phi; bh1 *= phi; an1 *= phi; bn1 *= phi
        am2 *= phi; bm2 *= phi; ah2 *= phi; bh2 *= phi; an2 *= phi; bn2 *= phi

        # Ionic currents per compartment
        I_Na0 = g_Na0 * (m0 ** 3) * h0 * (V0 - E_Na)
        I_K0  = g_K0  * (n0 ** 4) * (V0 - E_K)
        I_L0  = g_L0  * (V0 - E_L)
        I_Na1 = g_Na1 * (m1 ** 3) * h1 * (V1 - E_Na)
        I_K1  = g_K1  * (n1 ** 4) * (V1 - E_K)
        I_L1  = g_L1  * (V1 - E_L)
        I_Na2 = g_Na2 * (m2 ** 3) * h2 * (V2 - E_Na)
        I_K2  = g_K2  * (n2 ** 4) * (V2 - E_K)
        I_L2  = g_L2  * (V2 - E_L)

        # Mechano currents per compartment (sum over channels).
        # I_k = g_k · w · h_mech · (E_rev - V); h_mech=1 disables inactivation.
        # Ca²⁺ portion of I_mech feeds the intracellular Ca pool.
        I_mech0 = 0.0
        I_mech1 = 0.0
        I_mech2 = 0.0
        I_Ca0 = 0.0
        I_Ca1 = 0.0
        I_Ca2 = 0.0
        for k in range(n_ch):
            E_rev = E_revs[k]
            j0 = 0 * n_ch + k
            j1 = 1 * n_ch + k
            j2 = 2 * n_ch + k
            i0_k = g_mech[0, k] * w_flat[j0] * h_mech_flat[j0] * (E_rev - V0)
            i1_k = g_mech[1, k] * w_flat[j1] * h_mech_flat[j1] * (E_rev - V1)
            i2_k = g_mech[2, k] * w_flat[j2] * h_mech_flat[j2] * (E_rev - V2)
            I_mech0 += i0_k
            I_mech1 += i1_k
            I_mech2 += i2_k
            I_Ca0 += i0_k * Ca_frac[k]
            I_Ca1 += i1_k * Ca_frac[k]
            I_Ca2 += i2_k * Ca_frac[k]

        # KCa / SK current (Hill-type Ca activation), masked by compartment.
        I_KCa0 = 0.0
        I_KCa1 = 0.0
        I_KCa2 = 0.0
        if n_kca > 0:
            Ca0_safe = Ca0 if Ca0 > 1e-9 else 1e-9
            Ca1_safe = Ca1 if Ca1 > 1e-9 else 1e-9
            Ca2_safe = Ca2 if Ca2 > 1e-9 else 1e-9
            for kk in range(n_kca):
                nh = kca_n[kk]
                half_n = kca_half[kk] ** nh
                Po0 = (Ca0_safe ** nh) / ((Ca0_safe ** nh) + half_n)
                Po1 = (Ca1_safe ** nh) / ((Ca1_safe ** nh) + half_n)
                Po2 = (Ca2_safe ** nh) / ((Ca2_safe ** nh) + half_n)
                gK = kca_g[kk]
                EK = kca_E_K[kk]
                I_KCa0 += kca_mask[0] * gK * Po0 * (EK - V0)
                I_KCa1 += kca_mask[1] * gK * Po1 * (EK - V1)
                I_KCa2 += kca_mask[2] * gK * Po2 * (EK - V2)

        # TRP thermosensor current. P_o is fixed at the neuron's temperature
        # (TRPV1 ~0.018 at 37 °C; TRPV4 ~tonic non-selective leak), so the
        # entire compartment-channel contribution collapses to a sum of
        # (g·P_o)·(E_rev - V) terms gated by trp_mask.
        I_TRP0 = 0.0
        I_TRP1 = 0.0
        I_TRP2 = 0.0
        if n_trp > 0:
            for kk in range(n_trp):
                gP = trp_g_Po[kk]
                Erev = trp_E_rev[kk]
                I_TRP0 += trp_mask[0] * gP * (Erev - V0)
                I_TRP1 += trp_mask[1] * gP * (Erev - V1)
                I_TRP2 += trp_mask[2] * gP * (Erev - V2)

        # Axial coupling (Pinsky-Rinzel form, dimensionalised by area ratios)
        I_ax0 = g_ds * AsAd * (V1 - V0)
        I_ax1 = g_ds * (V0 - V1) + g_sa * (V2 - V1)
        I_ax2 = g_sa * AsAa * (V1 - V2)

        # dV per compartment
        dV0 = (-I_Na0 - I_K0 - I_L0 + I_mech0 + I_KCa0 + I_TRP0 + I_ax0 + I0) / C_m0
        dV1 = (-I_Na1 - I_K1 - I_L1 + I_mech1 + I_KCa1 + I_TRP1 + I_ax1 + I1) / C_m1
        dV2 = (-I_Na2 - I_K2 - I_L2 + I_mech2 + I_KCa2 + I_TRP2 + I_ax2 + I2) / C_m2

        # Ca²⁺ pool dynamics: flux_factor·I_Ca - (Ca - Ca_rest)/τ_decay
        dCa0 = flux_factor * I_Ca0 - (Ca0 - Ca_rest) * tau_decay_inv
        dCa1 = flux_factor * I_Ca1 - (Ca1 - Ca_rest) * tau_decay_inv
        dCa2 = flux_factor * I_Ca2 - (Ca2 - Ca_rest) * tau_decay_inv

        # HH gate derivatives
        dm0 = am0 * (1.0 - m0) - bm0 * m0
        dm1 = am1 * (1.0 - m1) - bm1 * m1
        dm2 = am2 * (1.0 - m2) - bm2 * m2
        dh0 = ah0 * (1.0 - h0) - bh0 * h0
        dh1 = ah1 * (1.0 - h1) - bh1 * h1
        dh2 = ah2 * (1.0 - h2) - bh2 * h2
        dn0 = an0 * (1.0 - n0) - bn0 * n0
        dn1 = an1 * (1.0 - n1) - bn1 * n1
        dn2 = an2 * (1.0 - n2) - bn2 * n2

        return (dV0, dV1, dV2,
                dm0, dm1, dm2,
                dh0, dh1, dh2,
                dn0, dn1, dn2,
                dCa0, dCa1, dCa2)

    @njit(cache=True, fastmath=True)
    def _sigmoid_safe_scalar(x: float) -> float:
        """Scalar mirror of the vectorised ``_safe_sigmoid`` (= 1/(1+e^x))."""
        if x >= 0.0:
            e = np.exp(-min(x, 500.0))
            return e / (1.0 + e)
        return 1.0 / (1.0 + np.exp(max(x, -500.0)))

    @njit(cache=True, fastmath=True)
    def _mech_kinetics_scalar(
        T: float, k: int,
        ch_T_half: np.ndarray, ch_dA: np.ndarray,
        ch_tau_act_inv: np.ndarray, ch_tau0: np.ndarray,
        ch_inact_on: np.ndarray, ch_hT_half: np.ndarray,
        ch_hdA: np.ndarray, ch_htau_inv: np.ndarray,
        kT: float,
    ):
        """(w_inf, 1/tau_w, h_inf, 1/tau_h) of channel ``k`` at tension T.

        Mirrors ``MulticompartmentMechanoHHNeuron._channel_kinetics`` and
        ``_inactivation_kinetics`` (scalar form). ``ch_tau_act_inv[k] > 0``
        selects the fixed Phase-5 activation time constant; otherwise the
        bell-shaped ``tau_0 / (4 w (1-w) + 0.01)`` law. Non-inactivating
        channels (``ch_inact_on[k] == 0``) return h_inf = 1, rate 0.
        """
        w_inf = _sigmoid_safe_scalar(-ch_dA[k] * (T - ch_T_half[k]) / kT)
        if ch_tau_act_inv[k] > 0.0:
            tau_inv = ch_tau_act_inv[k]
        else:
            tau_inv = (4.0 * w_inf * (1.0 - w_inf) + 0.01) / ch_tau0[k]
        if ch_inact_on[k] != 0.0:
            h_inf = _sigmoid_safe_scalar(ch_hdA[k] * (T - ch_hT_half[k]) / kT)
            h_tau_inv = ch_htau_inv[k]
        else:
            h_inf = 1.0
            h_tau_inv = 0.0
        return w_inf, tau_inv, h_inf, h_tau_inv

    @njit(parallel=True, cache=True, fastmath=True)
    def _rk4_batch_numba_3comp(
        # Per-voxel mechano state (steady-state w from tension)
        w_inf_per_vox: np.ndarray,    # (n_ch, N)
        tau_inv_per_vox: np.ndarray,  # (n_ch, N)
        # Per-channel inactivation: h_inf, h_tau_inv per voxel.
        # When inactivation is disabled, callers pass h_inf=1, h_tau_inv=0
        # (so h_mech stays at its initial value of 1.0 — equivalent to no
        # inactivation, exactly matching the existing fast-path semantics).
        h_inf_per_vox: np.ndarray,    # (n_ch, N)
        h_tau_inv_per_vox: np.ndarray,  # (n_ch, N)
        # Per-compartment scalars
        g_Na: np.ndarray,             # (3,)
        g_K: np.ndarray,              # (3,)
        g_L: np.ndarray,              # (3,)
        E_Na: float, E_K: float, E_L: float,
        # Effective capacitance (3, N) — builds in the constant-envelope
        # NICE perturbation when active; otherwise broadcasts plain C_m.
        C_eff_per_vox: np.ndarray,    # (3, N)
        # Coupling
        g_ds: float, g_sa: float,
        AsAd: float, AsAa: float,
        # Mechano
        g_mech: np.ndarray,           # (3, n_ch)
        E_revs: np.ndarray,           # (n_ch,)
        Ca_frac: np.ndarray,          # (n_ch,)
        # Calcium pool (flux_factor=0 disables; Ca stays at 0)
        flux_factor: float,
        tau_decay_inv: float,
        Ca_rest: float,
        # KCa channels (n_kca=0 disables)
        kca_g: np.ndarray,            # (n_kca,)
        kca_E_K: np.ndarray,          # (n_kca,)
        kca_half: np.ndarray,         # (n_kca,)
        kca_n: np.ndarray,            # (n_kca,)
        kca_mask: np.ndarray,         # (3,)
        # TRP channels (n_trp=0 disables); P_o pre-evaluated at body temp.
        trp_g_Po: np.ndarray,         # (n_trp,) g·P_o
        trp_E_rev: np.ndarray,        # (n_trp,)
        trp_mask: np.ndarray,         # (3,)
        # HH kinetics flag and threshold
        is_cortical: bool,            # noqa: FBT001
        VT: float,
        # Q10
        phi: float,
        # Stim
        I_stim: np.ndarray,           # (3,)
        # Init
        V_rest: float,
        m0_init: float, h0_init: float, n0_init: float,
        # Time
        dt: float, n_steps: int,
        primary_compartment: int,
        # Optional spike-step recording. When max_spikes > 0, each
        # detected primary-compartment crossing's step index (1-based to
        # match the existing NumPy convention) is written into row idx
        # of spike_step_buf. Pass max_spikes=0 + a (1, 1) dummy buffer to
        # disable recording (existing behaviour).
        spike_step_buf: np.ndarray,    # (N, max_spikes) int32
        max_spikes: int,
        # ---- Time-varying tension drive (optional) --------------------
        # T(t) = T_peak[idx] · drive[step]; drive is shared by all voxels
        # and held constant within each RK4 step (as in ``simulate``).
        # len(drive) == 0 → constant tension (the precomputed per-voxel
        # w_inf/tau/h arrays above are used throughout). Steps with
        # drive == 1.0 always use the precomputed arrays, so the ON
        # segment is bit-identical to the constant-tension path.
        drive: np.ndarray,             # (n_drive,) float64, n_drive ∈ {0, n_steps}
        T_peak_per_vox: np.ndarray,    # (N,) [mN/m]
        ch_T_half: np.ndarray,         # (n_ch,)
        ch_dA: np.ndarray,             # (n_ch,)
        ch_tau_act_inv: np.ndarray,    # (n_ch,) >0 → fixed tau_act
        ch_tau0: np.ndarray,           # (n_ch,)
        ch_inact_on: np.ndarray,       # (n_ch,) 1.0 if inactivating
        ch_hT_half: np.ndarray,        # (n_ch,)
        ch_hdA: np.ndarray,            # (n_ch,)
        ch_htau_inv: np.ndarray,       # (n_ch,)
        kT: float,
        # ---- State carry-over (optional) ------------------------------
        # Layout per voxel (see ``state_layout``): V[3], m[3], h[3], n[3],
        # w[3*n_ch], h_mech[3*n_ch], Ca[3]. Shape (N, n_state) enables;
        # a (1, 1) dummy disables.
        init_state: np.ndarray,        # (N, n_state) or (1, 1)
        final_state: np.ndarray,       # (N, n_state) or (1, 1) — output
        # ---- Ca readouts (optional) -----------------------------------
        ca_peak: np.ndarray,           # (N, 3) or (1, 1) — output, running max
        ca_trace: np.ndarray,          # (N, 3, n_tr) float32 or (1, 1, 1)
        ca_decim: int,                 # record every ca_decim steps; 0 = off
        # ---- Mechano-gate initial condition ---------------------------
        # mech_init_rest != 0 → w, h_mech start at the zero-tension steady
        # state given per channel in w_init_ch / hm_init_ch (true rest;
        # the default). 0 → steady state for the step-0 tension (legacy
        # 'peak_steady'). An enabled init_state overrides both.
        mech_init_rest: int,
        w_init_ch: np.ndarray,         # (n_ch,)
        hm_init_ch: np.ndarray,        # (n_ch,)
    ) -> np.ndarray:
        N = w_inf_per_vox.shape[1]
        n_ch = w_inf_per_vox.shape[0]
        n_kca = kca_g.shape[0]
        n_trp = trp_g_Po.shape[0]
        n_state = 15 + 6 * n_ch
        n_drive = drive.shape[0]
        use_init = init_state.shape[0] == N and init_state.shape[1] == n_state
        write_final = final_state.shape[0] == N and final_state.shape[1] == n_state
        write_peak = ca_peak.shape[0] == N and ca_peak.shape[1] == 3
        n_tr = ca_trace.shape[2]
        write_trace = ca_decim > 0 and ca_trace.shape[0] == N
        spike_counts = np.zeros(N, dtype=np.int32)

        I0 = I_stim[0]
        I1 = I_stim[1]
        I2 = I_stim[2]

        for idx in prange(N):
            V0 = V_rest; V1 = V_rest; V2 = V_rest
            m0 = m0_init; m1 = m0_init; m2 = m0_init
            h0 = h0_init; h1 = h0_init; h2 = h0_init
            n0 = n0_init; n1 = n0_init; n2 = n0_init
            # Ca pool starts at Ca_rest (or 0 when pool disabled)
            Ca_a = Ca_rest; Ca_b = Ca_rest; Ca_c = Ca_rest
            # Per-voxel effective capacitance (constant over time in this path)
            Cm0 = C_eff_per_vox[0, idx]
            Cm1 = C_eff_per_vox[1, idx]
            Cm2 = C_eff_per_vox[2, idx]

            # Per-voxel constants for this idx (peak-tension kinetics)
            w_inf_local = np.empty(n_ch)
            tau_inv_local = np.empty(n_ch)
            h_inf_local = np.empty(n_ch)
            h_tau_inv_local = np.empty(n_ch)
            for k in range(n_ch):
                w_inf_local[k] = w_inf_per_vox[k, idx]
                tau_inv_local[k] = tau_inv_per_vox[k, idx]
                h_inf_local[k] = h_inf_per_vox[k, idx]
                h_tau_inv_local[k] = h_tau_inv_per_vox[k, idx]
            # Kinetics in force for the current step
            w_inf_now = np.empty(n_ch)
            tau_inv_now = np.empty(n_ch)
            h_inf_now = np.empty(n_ch)
            h_tau_inv_now = np.empty(n_ch)
            T_pk = T_peak_per_vox[idx] if n_drive > 0 else 0.0
            f_prev = 1.0
            if n_drive > 0:
                f_prev = drive[0]
            for k in range(n_ch):
                if f_prev == 1.0:
                    w_inf_now[k] = w_inf_local[k]
                    tau_inv_now[k] = tau_inv_local[k]
                    h_inf_now[k] = h_inf_local[k]
                    h_tau_inv_now[k] = h_tau_inv_local[k]
                else:
                    wi_, ti_, hi_, hti_ = _mech_kinetics_scalar(
                        T_pk * f_prev, k, ch_T_half, ch_dA, ch_tau_act_inv,
                        ch_tau0, ch_inact_on, ch_hT_half, ch_hdA,
                        ch_htau_inv, kT,
                    )
                    w_inf_now[k] = wi_; tau_inv_now[k] = ti_
                    h_inf_now[k] = hi_; h_tau_inv_now[k] = hti_

            # Mechano gates (shared across compartments at t=0): zero-tension
            # steady state when mech_init_rest (true rest, default), else
            # steady state for the step-0 tension (legacy 'peak_steady').
            w = np.empty(3 * n_ch)
            h_mech = np.empty(3 * n_ch)
            for c in range(3):
                for k in range(n_ch):
                    j = c * n_ch + k
                    if mech_init_rest != 0:
                        w[j] = w_init_ch[k]
                        h_mech[j] = hm_init_ch[k]
                    else:
                        w[j] = w_inf_now[k]
                        h_mech[j] = h_inf_now[k]

            if use_init:
                V0 = init_state[idx, 0]; V1 = init_state[idx, 1]; V2 = init_state[idx, 2]
                m0 = init_state[idx, 3]; m1 = init_state[idx, 4]; m2 = init_state[idx, 5]
                h0 = init_state[idx, 6]; h1 = init_state[idx, 7]; h2 = init_state[idx, 8]
                n0 = init_state[idx, 9]; n1 = init_state[idx, 10]; n2 = init_state[idx, 11]
                for j in range(3 * n_ch):
                    w[j] = init_state[idx, 12 + j]
                    h_mech[j] = init_state[idx, 12 + 3 * n_ch + j]
                Ca_a = init_state[idx, 12 + 6 * n_ch]
                Ca_b = init_state[idx, 13 + 6 * n_ch]
                Ca_c = init_state[idx, 14 + 6 * n_ch]

            pk_a = Ca_a; pk_b = Ca_b; pk_c = Ca_c

            # Scratch buffers for RK4 stages on w and h_mech
            k1w = np.empty(3 * n_ch)
            k2w = np.empty(3 * n_ch)
            k3w = np.empty(3 * n_ch)
            k4w = np.empty(3 * n_ch)
            w_tmp = np.empty(3 * n_ch)
            k1hm = np.empty(3 * n_ch)
            k2hm = np.empty(3 * n_ch)
            k3hm = np.empty(3 * n_ch)
            k4hm = np.empty(3 * n_ch)
            hm_tmp = np.empty(3 * n_ch)

            V1_prev = V1

            for _step in range(n_steps):
                if primary_compartment == 0:
                    V_prev = V0
                elif primary_compartment == 1:
                    V_prev = V1
                else:
                    V_prev = V2
                V1_prev = V_prev

                # Time-varying drive: refresh kinetics when the factor changes
                if n_drive > 0:
                    f_now = drive[_step]
                    if f_now != f_prev:
                        for k in range(n_ch):
                            if f_now == 1.0:
                                w_inf_now[k] = w_inf_local[k]
                                tau_inv_now[k] = tau_inv_local[k]
                                h_inf_now[k] = h_inf_local[k]
                                h_tau_inv_now[k] = h_tau_inv_local[k]
                            else:
                                wi_, ti_, hi_, hti_ = _mech_kinetics_scalar(
                                    T_pk * f_now, k, ch_T_half, ch_dA,
                                    ch_tau_act_inv, ch_tau0, ch_inact_on,
                                    ch_hT_half, ch_hdA, ch_htau_inv, kT,
                                )
                                w_inf_now[k] = wi_; tau_inv_now[k] = ti_
                                h_inf_now[k] = hi_; h_tau_inv_now[k] = hti_
                        f_prev = f_now

                # ---- k1 ----
                (k1V0, k1V1, k1V2,
                 k1m0, k1m1, k1m2,
                 k1h0, k1h1, k1h2,
                 k1n0, k1n1, k1n2,
                 k1Ca0, k1Ca1, k1Ca2) = _derivs_3comp_scalar(
                    V0, V1, V2, m0, m1, m2, h0, h1, h2, n0, n1, n2,
                    w, h_mech,
                    Ca_a, Ca_b, Ca_c,
                    n_ch,
                    g_Na[0], g_Na[1], g_Na[2],
                    g_K[0], g_K[1], g_K[2],
                    g_L[0], g_L[1], g_L[2],
                    E_Na, E_K, E_L,
                    Cm0, Cm1, Cm2,
                    g_ds, g_sa, AsAd, AsAa,
                    g_mech, E_revs, Ca_frac,
                    flux_factor, tau_decay_inv, Ca_rest,
                    n_kca, kca_g, kca_E_K, kca_half, kca_n, kca_mask,
                    n_trp, trp_g_Po, trp_E_rev, trp_mask,
                    I0, I1, I2,
                    phi, is_cortical, VT,
                )
                for j in range(3 * n_ch):
                    c_idx = j // n_ch
                    k_idx = j - c_idx * n_ch
                    k1w[j] = (w_inf_now[k_idx] - w[j]) * tau_inv_now[k_idx]
                    k1hm[j] = (h_inf_now[k_idx] - h_mech[j]) * h_tau_inv_now[k_idx]

                # ---- k2 ----
                for j in range(3 * n_ch):
                    w_tmp[j] = w[j] + 0.5 * dt * k1w[j]
                    hm_tmp[j] = h_mech[j] + 0.5 * dt * k1hm[j]
                (k2V0, k2V1, k2V2,
                 k2m0, k2m1, k2m2,
                 k2h0, k2h1, k2h2,
                 k2n0, k2n1, k2n2,
                 k2Ca0, k2Ca1, k2Ca2) = _derivs_3comp_scalar(
                    V0 + 0.5 * dt * k1V0,
                    V1 + 0.5 * dt * k1V1,
                    V2 + 0.5 * dt * k1V2,
                    m0 + 0.5 * dt * k1m0,
                    m1 + 0.5 * dt * k1m1,
                    m2 + 0.5 * dt * k1m2,
                    h0 + 0.5 * dt * k1h0,
                    h1 + 0.5 * dt * k1h1,
                    h2 + 0.5 * dt * k1h2,
                    n0 + 0.5 * dt * k1n0,
                    n1 + 0.5 * dt * k1n1,
                    n2 + 0.5 * dt * k1n2,
                    w_tmp, hm_tmp,
                    Ca_a + 0.5 * dt * k1Ca0,
                    Ca_b + 0.5 * dt * k1Ca1,
                    Ca_c + 0.5 * dt * k1Ca2,
                    n_ch,
                    g_Na[0], g_Na[1], g_Na[2],
                    g_K[0], g_K[1], g_K[2],
                    g_L[0], g_L[1], g_L[2],
                    E_Na, E_K, E_L,
                    Cm0, Cm1, Cm2,
                    g_ds, g_sa, AsAd, AsAa,
                    g_mech, E_revs, Ca_frac,
                    flux_factor, tau_decay_inv, Ca_rest,
                    n_kca, kca_g, kca_E_K, kca_half, kca_n, kca_mask,
                    n_trp, trp_g_Po, trp_E_rev, trp_mask,
                    I0, I1, I2,
                    phi, is_cortical, VT,
                )
                for j in range(3 * n_ch):
                    c_idx = j // n_ch
                    k_idx = j - c_idx * n_ch
                    k2w[j] = (w_inf_now[k_idx] - w_tmp[j]) * tau_inv_now[k_idx]
                    k2hm[j] = (h_inf_now[k_idx] - hm_tmp[j]) * h_tau_inv_now[k_idx]

                # ---- k3 ----
                for j in range(3 * n_ch):
                    w_tmp[j] = w[j] + 0.5 * dt * k2w[j]
                    hm_tmp[j] = h_mech[j] + 0.5 * dt * k2hm[j]
                (k3V0, k3V1, k3V2,
                 k3m0, k3m1, k3m2,
                 k3h0, k3h1, k3h2,
                 k3n0, k3n1, k3n2,
                 k3Ca0, k3Ca1, k3Ca2) = _derivs_3comp_scalar(
                    V0 + 0.5 * dt * k2V0,
                    V1 + 0.5 * dt * k2V1,
                    V2 + 0.5 * dt * k2V2,
                    m0 + 0.5 * dt * k2m0,
                    m1 + 0.5 * dt * k2m1,
                    m2 + 0.5 * dt * k2m2,
                    h0 + 0.5 * dt * k2h0,
                    h1 + 0.5 * dt * k2h1,
                    h2 + 0.5 * dt * k2h2,
                    n0 + 0.5 * dt * k2n0,
                    n1 + 0.5 * dt * k2n1,
                    n2 + 0.5 * dt * k2n2,
                    w_tmp, hm_tmp,
                    Ca_a + 0.5 * dt * k2Ca0,
                    Ca_b + 0.5 * dt * k2Ca1,
                    Ca_c + 0.5 * dt * k2Ca2,
                    n_ch,
                    g_Na[0], g_Na[1], g_Na[2],
                    g_K[0], g_K[1], g_K[2],
                    g_L[0], g_L[1], g_L[2],
                    E_Na, E_K, E_L,
                    Cm0, Cm1, Cm2,
                    g_ds, g_sa, AsAd, AsAa,
                    g_mech, E_revs, Ca_frac,
                    flux_factor, tau_decay_inv, Ca_rest,
                    n_kca, kca_g, kca_E_K, kca_half, kca_n, kca_mask,
                    n_trp, trp_g_Po, trp_E_rev, trp_mask,
                    I0, I1, I2,
                    phi, is_cortical, VT,
                )
                for j in range(3 * n_ch):
                    c_idx = j // n_ch
                    k_idx = j - c_idx * n_ch
                    k3w[j] = (w_inf_now[k_idx] - w_tmp[j]) * tau_inv_now[k_idx]
                    k3hm[j] = (h_inf_now[k_idx] - hm_tmp[j]) * h_tau_inv_now[k_idx]

                # ---- k4 ----
                for j in range(3 * n_ch):
                    w_tmp[j] = w[j] + dt * k3w[j]
                    hm_tmp[j] = h_mech[j] + dt * k3hm[j]
                (k4V0, k4V1, k4V2,
                 k4m0, k4m1, k4m2,
                 k4h0, k4h1, k4h2,
                 k4n0, k4n1, k4n2,
                 k4Ca0, k4Ca1, k4Ca2) = _derivs_3comp_scalar(
                    V0 + dt * k3V0,
                    V1 + dt * k3V1,
                    V2 + dt * k3V2,
                    m0 + dt * k3m0,
                    m1 + dt * k3m1,
                    m2 + dt * k3m2,
                    h0 + dt * k3h0,
                    h1 + dt * k3h1,
                    h2 + dt * k3h2,
                    n0 + dt * k3n0,
                    n1 + dt * k3n1,
                    n2 + dt * k3n2,
                    w_tmp, hm_tmp,
                    Ca_a + dt * k3Ca0,
                    Ca_b + dt * k3Ca1,
                    Ca_c + dt * k3Ca2,
                    n_ch,
                    g_Na[0], g_Na[1], g_Na[2],
                    g_K[0], g_K[1], g_K[2],
                    g_L[0], g_L[1], g_L[2],
                    E_Na, E_K, E_L,
                    Cm0, Cm1, Cm2,
                    g_ds, g_sa, AsAd, AsAa,
                    g_mech, E_revs, Ca_frac,
                    flux_factor, tau_decay_inv, Ca_rest,
                    n_kca, kca_g, kca_E_K, kca_half, kca_n, kca_mask,
                    n_trp, trp_g_Po, trp_E_rev, trp_mask,
                    I0, I1, I2,
                    phi, is_cortical, VT,
                )
                for j in range(3 * n_ch):
                    c_idx = j // n_ch
                    k_idx = j - c_idx * n_ch
                    k4w[j] = (w_inf_now[k_idx] - w_tmp[j]) * tau_inv_now[k_idx]
                    k4hm[j] = (h_inf_now[k_idx] - hm_tmp[j]) * h_tau_inv_now[k_idx]
                # ---- Update state ----
                s = dt / 6.0
                V0 = V0 + s * (k1V0 + 2.0 * k2V0 + 2.0 * k3V0 + k4V0)
                V1 = V1 + s * (k1V1 + 2.0 * k2V1 + 2.0 * k3V1 + k4V1)
                V2 = V2 + s * (k1V2 + 2.0 * k2V2 + 2.0 * k3V2 + k4V2)
                m0 = m0 + s * (k1m0 + 2.0 * k2m0 + 2.0 * k3m0 + k4m0)
                m1 = m1 + s * (k1m1 + 2.0 * k2m1 + 2.0 * k3m1 + k4m1)
                m2 = m2 + s * (k1m2 + 2.0 * k2m2 + 2.0 * k3m2 + k4m2)
                h0 = h0 + s * (k1h0 + 2.0 * k2h0 + 2.0 * k3h0 + k4h0)
                h1 = h1 + s * (k1h1 + 2.0 * k2h1 + 2.0 * k3h1 + k4h1)
                h2 = h2 + s * (k1h2 + 2.0 * k2h2 + 2.0 * k3h2 + k4h2)
                n0 = n0 + s * (k1n0 + 2.0 * k2n0 + 2.0 * k3n0 + k4n0)
                n1 = n1 + s * (k1n1 + 2.0 * k2n1 + 2.0 * k3n1 + k4n1)
                n2 = n2 + s * (k1n2 + 2.0 * k2n2 + 2.0 * k3n2 + k4n2)
                for j in range(3 * n_ch):
                    avg = (k1w[j] + 2.0 * k2w[j] + 2.0 * k3w[j] + k4w[j]) / 6.0
                    w[j] = w[j] + dt * avg
                    if w[j] < 0.0:
                        w[j] = 0.0
                    elif w[j] > 1.0:
                        w[j] = 1.0
                    avg_hm = (k1hm[j] + 2.0 * k2hm[j] + 2.0 * k3hm[j] + k4hm[j]) / 6.0
                    h_mech[j] = h_mech[j] + dt * avg_hm
                    if h_mech[j] < 0.0:
                        h_mech[j] = 0.0
                    elif h_mech[j] > 1.0:
                        h_mech[j] = 1.0

                # Ca²⁺ pool update with non-negative clamp
                Ca_a = Ca_a + s * (k1Ca0 + 2.0 * k2Ca0 + 2.0 * k3Ca0 + k4Ca0)
                Ca_b = Ca_b + s * (k1Ca1 + 2.0 * k2Ca1 + 2.0 * k3Ca1 + k4Ca1)
                Ca_c = Ca_c + s * (k1Ca2 + 2.0 * k2Ca2 + 2.0 * k3Ca2 + k4Ca2)
                if Ca_a < 0.0: Ca_a = 0.0
                if Ca_b < 0.0: Ca_b = 0.0
                if Ca_c < 0.0: Ca_c = 0.0

                # Clamp HH gates
                if m0 < 0.0: m0 = 0.0
                elif m0 > 1.0: m0 = 1.0
                if m1 < 0.0: m1 = 0.0
                elif m1 > 1.0: m1 = 1.0
                if m2 < 0.0: m2 = 0.0
                elif m2 > 1.0: m2 = 1.0
                if h0 < 0.0: h0 = 0.0
                elif h0 > 1.0: h0 = 1.0
                if h1 < 0.0: h1 = 0.0
                elif h1 > 1.0: h1 = 1.0
                if h2 < 0.0: h2 = 0.0
                elif h2 > 1.0: h2 = 1.0
                if n0 < 0.0: n0 = 0.0
                elif n0 > 1.0: n0 = 1.0
                if n1 < 0.0: n1 = 0.0
                elif n1 > 1.0: n1 = 1.0
                if n2 < 0.0: n2 = 0.0
                elif n2 > 1.0: n2 = 1.0

                # Spike detection at primary compartment
                if primary_compartment == 0:
                    V_now = V0
                elif primary_compartment == 1:
                    V_now = V1
                else:
                    V_now = V2
                if V_now > 0.0 and V1_prev <= 0.0:
                    prev_count = spike_counts[idx]
                    if max_spikes > 0 and prev_count < max_spikes:
                        # 1-based step index, matching the NumPy
                        # convention r["spike_times_per_compartment"][c]
                        # = (i+1)*dt.
                        spike_step_buf[idx, prev_count] = _step + 1
                    spike_counts[idx] = prev_count + 1

                # Ca readouts (passive: never feed back into V unless the
                # caller attached KCa channels)
                if write_peak:
                    if Ca_a > pk_a: pk_a = Ca_a
                    if Ca_b > pk_b: pk_b = Ca_b
                    if Ca_c > pk_c: pk_c = Ca_c
                if write_trace:
                    if (_step + 1) % ca_decim == 0:
                        i_tr = (_step + 1) // ca_decim - 1
                        if i_tr < n_tr:
                            ca_trace[idx, 0, i_tr] = Ca_a
                            ca_trace[idx, 1, i_tr] = Ca_b
                            ca_trace[idx, 2, i_tr] = Ca_c

            if write_peak:
                ca_peak[idx, 0] = pk_a
                ca_peak[idx, 1] = pk_b
                ca_peak[idx, 2] = pk_c
            if write_final:
                final_state[idx, 0] = V0; final_state[idx, 1] = V1; final_state[idx, 2] = V2
                final_state[idx, 3] = m0; final_state[idx, 4] = m1; final_state[idx, 5] = m2
                final_state[idx, 6] = h0; final_state[idx, 7] = h1; final_state[idx, 8] = h2
                final_state[idx, 9] = n0; final_state[idx, 10] = n1; final_state[idx, 11] = n2
                for j in range(3 * n_ch):
                    final_state[idx, 12 + j] = w[j]
                    final_state[idx, 12 + 3 * n_ch + j] = h_mech[j]
                final_state[idx, 12 + 6 * n_ch] = Ca_a
                final_state[idx, 13 + 6 * n_ch] = Ca_b
                final_state[idx, 14 + 6 * n_ch] = Ca_c

        return spike_counts

    @njit(cache=True, fastmath=True, inline="always")
    def _rl_exp_step(x, x_inf, rate, dt):
        """Exact update of dx/dt = (x_inf - x) * rate over dt."""
        return x_inf + (x - x_inf) * np.exp(-rate * dt)

    @njit(cache=True, fastmath=True, parallel=True)
    def _rl_batch_numba_3comp(
        w_inf_per_vox,    # (n_ch, N)
        tau_inv_per_vox,  # (n_ch, N)
        g_Na, g_K, g_L,   # (3,)
        E_Na, E_K, E_L,
        C_eff_per_vox,    # (3, N) — constant C_m (no NICE on the RL path)
        g_ds, g_sa, AsAd, AsAa,
        g_mech,           # (3, n_ch)
        E_revs,           # (n_ch,)
        is_cortical,      # noqa: FBT001
        VT, phi,
        I_stim,           # (3,)
        V_rest, m0_init, h0_init, n0_init,
        dt, n_steps, primary_compartment,
        spike_step_buf,   # (N, max_spikes) int32
        max_spikes,
        mech_init_rest,   # != 0 → w starts at w_init_ch (zero tension)
        w_init_ch,        # (n_ch,)
    ):
        """Generalised Rush-Larsen / exponential-Euler integrator for the
        S2 scenario (HH + Piezo1/TREK1 mechano gates; no Ca pool, KCa, TRP,
        inactivation, or NICE).

        Per step, with all rate coefficients frozen at the current state:
        (1) update HH gates m, h, n and mechano gates w by their exact
        exponential solution; (2) update each compartment voltage by the
        exact solution of the linearised membrane equation
        ``C dV/dt = -G_tot (V - V_inf)``, where ``G_tot`` collects every
        conductance multiplying V (ionic + mechano + axial) and ``V_inf =
        B/G_tot``.  Axial-neighbour voltages are frozen at their
        start-of-step values (Gauss-Jacobi); the stiff diagonal (the large
        AIS coupling + g_Na) is taken exactly, so the scheme is L-stable at
        any dt where the explicit RK4 kernel collapses.  Physics mirrors
        ``_rk4_batch_numba_3comp`` for the S2 feature set, and spike-step
        recording is identical, so this is a drop-in method swap.  ~3x
        faster than RK4 at equal dt; degrades smoothly (not catastrophically)
        at coarser dt.  See ``solver_investigation/FINDINGS.md``.
        """
        N = w_inf_per_vox.shape[1]
        n_ch = w_inf_per_vox.shape[0]
        spike_counts = np.zeros(N, dtype=np.int32)
        I0 = I_stim[0]; I1 = I_stim[1]; I2 = I_stim[2]

        for idx in prange(N):
            V0 = V_rest; V1 = V_rest; V2 = V_rest
            m0 = m0_init; m1 = m0_init; m2 = m0_init
            h0 = h0_init; h1 = h0_init; h2 = h0_init
            n0 = n0_init; n1 = n0_init; n2 = n0_init

            w_inf_l = np.empty(n_ch)
            tau_inv_l = np.empty(n_ch)
            for k in range(n_ch):
                w_inf_l[k] = w_inf_per_vox[k, idx]
                tau_inv_l[k] = tau_inv_per_vox[k, idx]

            w = np.empty(3 * n_ch)
            for c in range(3):
                for k in range(n_ch):
                    if mech_init_rest != 0:
                        w[c * n_ch + k] = w_init_ch[k]
                    else:
                        w[c * n_ch + k] = w_inf_l[k]

            Cm0 = C_eff_per_vox[0, idx]
            Cm1 = C_eff_per_vox[1, idx]
            Cm2 = C_eff_per_vox[2, idx]

            for _step in range(n_steps):
                if primary_compartment == 0:
                    V_prev = V0
                elif primary_compartment == 1:
                    V_prev = V1
                else:
                    V_prev = V2

                # Freeze neighbour voltages for axial coupling (Gauss-Jacobi)
                V0f = V0; V1f = V1; V2f = V2

                # ---- (1) Gate updates (frozen coefficients at current V) ----
                if is_cortical:
                    am0, bm0, ah0, bh0, an0, bn0 = _hh_rates_cortical(V0, VT)
                    am1, bm1, ah1, bh1, an1, bn1 = _hh_rates_cortical(V1, VT)
                    am2, bm2, ah2, bh2, an2, bn2 = _hh_rates_cortical(V2, VT)
                else:
                    am0, bm0, ah0, bh0, an0, bn0 = _hh_rates_squid(V0)
                    am1, bm1, ah1, bh1, an1, bn1 = _hh_rates_squid(V1)
                    am2, bm2, ah2, bh2, an2, bn2 = _hh_rates_squid(V2)
                am0*=phi; bm0*=phi; ah0*=phi; bh0*=phi; an0*=phi; bn0*=phi
                am1*=phi; bm1*=phi; ah1*=phi; bh1*=phi; an1*=phi; bn1*=phi
                am2*=phi; bm2*=phi; ah2*=phi; bh2*=phi; an2*=phi; bn2*=phi

                m0 = _rl_exp_step(m0, am0/(am0+bm0), am0+bm0, dt)
                m1 = _rl_exp_step(m1, am1/(am1+bm1), am1+bm1, dt)
                m2 = _rl_exp_step(m2, am2/(am2+bm2), am2+bm2, dt)
                h0 = _rl_exp_step(h0, ah0/(ah0+bh0), ah0+bh0, dt)
                h1 = _rl_exp_step(h1, ah1/(ah1+bh1), ah1+bh1, dt)
                h2 = _rl_exp_step(h2, ah2/(ah2+bh2), ah2+bh2, dt)
                n0 = _rl_exp_step(n0, an0/(an0+bn0), an0+bn0, dt)
                n1 = _rl_exp_step(n1, an1/(an1+bn1), an1+bn1, dt)
                n2 = _rl_exp_step(n2, an2/(an2+bn2), an2+bn2, dt)
                for j in range(3 * n_ch):
                    k = j % n_ch
                    w[j] = _rl_exp_step(w[j], w_inf_l[k], tau_inv_l[k], dt)

                # ---- (2) Voltage updates (exact, frozen conductances using
                #          the just-updated gates) ----
                gNa0 = g_Na[0]*(m0*m0*m0)*h0; gK0 = g_K[0]*(n0*n0*n0*n0)
                gNa1 = g_Na[1]*(m1*m1*m1)*h1; gK1 = g_K[1]*(n1*n1*n1*n1)
                gNa2 = g_Na[2]*(m2*m2*m2)*h2; gK2 = g_K[2]*(n2*n2*n2*n2)

                Gm0 = 0.0; Bm0 = 0.0
                Gm1 = 0.0; Bm1 = 0.0
                Gm2 = 0.0; Bm2 = 0.0
                for k in range(n_ch):
                    gk0 = g_mech[0, k]*w[0*n_ch+k]
                    gk1 = g_mech[1, k]*w[1*n_ch+k]
                    gk2 = g_mech[2, k]*w[2*n_ch+k]
                    Gm0 += gk0; Bm0 += gk0*E_revs[k]
                    Gm1 += gk1; Bm1 += gk1*E_revs[k]
                    Gm2 += gk2; Bm2 += gk2*E_revs[k]

                gax0 = g_ds*AsAd          # dendrite <- soma
                gax1d = g_ds; gax1a = g_sa  # soma <- dendrite, ais
                gax2 = g_sa*AsAa          # ais <- soma

                G0 = gNa0 + gK0 + g_L[0] + Gm0 + gax0
                B0 = gNa0*E_Na + gK0*E_K + g_L[0]*E_L + Bm0 + gax0*V1f + I0
                G1 = gNa1 + gK1 + g_L[1] + Gm1 + gax1d + gax1a
                B1 = gNa1*E_Na + gK1*E_K + g_L[1]*E_L + Bm1 + gax1d*V0f + gax1a*V2f + I1
                G2 = gNa2 + gK2 + g_L[2] + Gm2 + gax2
                B2 = gNa2*E_Na + gK2*E_K + g_L[2]*E_L + Bm2 + gax2*V1f + I2

                V0 = _rl_exp_step(V0, B0/G0, G0/Cm0, dt)
                V1 = _rl_exp_step(V1, B1/G1, G1/Cm1, dt)
                V2 = _rl_exp_step(V2, B2/G2, G2/Cm2, dt)

                # Spike detect + step recording (identical to RK4 kernel)
                if primary_compartment == 0:
                    V_now = V0
                elif primary_compartment == 1:
                    V_now = V1
                else:
                    V_now = V2
                if V_now > 0.0 and V_prev <= 0.0:
                    prev_count = spike_counts[idx]
                    if max_spikes > 0 and prev_count < max_spikes:
                        spike_step_buf[idx, prev_count] = _step + 1
                    spike_counts[idx] = prev_count + 1

        return spike_counts

    _HAS_NUMBA_3COMP = True
except Exception:
    _HAS_NUMBA_3COMP = False


@dataclass
class MulticompartmentMechanoHHNeuron:
    """Three-compartment HH neuron with mechanosensitive channels.

    Parameters
    ----------
    geometry : CompartmentalGeometry
        Dendrite + soma + AIS configuration.  Defaults to the literature
        cortical-pyramidal preset (see ``cortical_pyramidal_3comp``).
    mechano_channels : list[MechanosensitiveChannel]
        Channels active in every compartment, scaled per-compartment by
        ``Compartment.mechano_density_scale``.
    temperature : float
        Body temperature [°C] for Q10 scaling. Default 37.
    V_rest : float
        Initial membrane potential in every compartment [mV]. Default -70.
    model : str
        Channel-kinetics family for Na/K/Leak: ``"squid"`` or
        ``"cortical"``. Default ``"cortical"`` (Pospischil RS).

    Notes
    -----
    The three compartments share Na/K/Leak gating *kinetics* but have
    independent state and compartment-specific *densities*.  This
    matches the standard practice for lumped multi-compartment models
    of cortical pyramidal cells.
    """

    geometry: CompartmentalGeometry
    mechano_channels: list[MechanosensitiveChannel] = field(default_factory=list)
    temperature: float = 37.0
    V_rest: float = -70.0
    model: str = "cortical"
    nice_model: NICEModel | None = None
    nice_compartments: tuple[int, ...] = (1,)  # default: soma only
    # Phase-3 opt-ins. All default to "off" → exact Phase-2 behaviour.
    piezo_inactivation: bool = False
    calcium_dynamics: CalciumDynamics | None = None
    kca_channels: list[KCaChannel] = field(default_factory=list)
    kca_compartments: tuple[int, ...] = (1,)  # which compartments host KCa channels
    # Phase-4: TRP thermosensor channels (TRPV1 / TRPV4). Default empty.
    # Open probability evaluated at ``self.temperature``; current contributes
    # to the listed compartments only.
    trp_channels: list[TRPThermosensorChannel] = field(default_factory=list)
    trp_compartments: tuple[int, ...] = (1,)  # default soma

    _kinetics: MechanoHHNeuron = field(init=False, repr=False)
    _nice_P_grid: NDArray[np.floating] | None = field(init=False, repr=False, default=None)
    _nice_dC_grid: NDArray[np.floating] | None = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        # Build a kinetics-only MechanoHHNeuron (used purely for its
        # rate-function methods and Q10 helpers; its g_Na/g_K/g_L are
        # ignored — we use per-compartment values instead).
        self._kinetics = MechanoHHNeuron(
            temperature=self.temperature,
            mechano_channels=[],
            model=self.model,
            V_rest=self.V_rest,
        )

        # Validate per-compartment mechano density lists
        n_ch = len(self.mechano_channels)
        for c in (self.geometry.dendrite, self.geometry.soma, self.geometry.ais):
            scale = c.mechano_density_scale
            if scale and len(scale) != n_ch:
                msg = (
                    f"Compartment '{c.name}' has {len(scale)} mechano density "
                    f"scales but neuron has {n_ch} mechano channels."
                )
                raise ValueError(msg)
            if not scale:
                # Default to scale = 1 for every channel
                c.mechano_density_scale = [1.0] * n_ch

        # Validate NICE compartment indices and pre-build a lookup table
        if self.nice_model is not None:
            for idx in self.nice_compartments:
                if idx not in (0, 1, 2):
                    msg = f"nice_compartments index {idx} out of range [0,1,2]"
                    raise ValueError(msg)
            # Cache lookup over a wide pressure range; user can override
            # by replacing self._nice_P_grid / self._nice_dC_grid before sim.
            self._nice_P_grid, self._nice_dC_grid = self.nice_model.precompute_lookup(
                P_max=1e6,  # 1 MPa — full validity range with LJ + ST walls
                n_points=128,
            )

        # Validate Phase-3 opt-ins
        if self.kca_channels and self.calcium_dynamics is None:
            msg = "kca_channels requires calcium_dynamics to be set."
            raise ValueError(msg)
        for idx in self.kca_compartments:
            if idx not in (0, 1, 2):
                msg = f"kca_compartments index {idx} out of range [0,1,2]"
                raise ValueError(msg)
        # Validate Phase-4 TRP opt-in
        for idx in self.trp_compartments:
            if idx not in (0, 1, 2):
                msg = f"trp_compartments index {idx} out of range [0,1,2]"
                raise ValueError(msg)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def cortical_pyramidal_3comp(
        cls,
        mechano_channels: list[MechanosensitiveChannel] | None = None,
        temperature: float = 37.0,
        V_rest: float = -70.0,
        dendrite_piezo_scale: float = 5.0,
        ais_piezo_scale: float = 0.0,
        nice_model: NICEModel | None = None,
        nice_compartments: tuple[int, ...] = (1,),
        piezo_inactivation: bool = False,
        calcium_dynamics: CalciumDynamics | None = None,
        kca_channels: list[KCaChannel] | None = None,
        kca_compartments: tuple[int, ...] = (1,),
        trp_channels: list[TRPThermosensorChannel] | None = None,
        trp_compartments: tuple[int, ...] = (1,),
    ) -> MulticompartmentMechanoHHNeuron:
        """Literature-parameterised RS pyramidal 3-compartment neuron."""
        channels = mechano_channels or []
        geom = cortical_pyramidal_3comp(
            n_mechano_channels=len(channels),
            dendrite_piezo_scale=dendrite_piezo_scale,
            ais_piezo_scale=ais_piezo_scale,
        )
        return cls(
            geometry=geom,
            mechano_channels=channels,
            temperature=temperature,
            V_rest=V_rest,
            model="cortical",
            nice_model=nice_model,
            nice_compartments=nice_compartments,
            piezo_inactivation=piezo_inactivation,
            calcium_dynamics=calcium_dynamics,
            kca_channels=kca_channels or [],
            kca_compartments=kca_compartments,
            trp_channels=trp_channels or [],
            trp_compartments=trp_compartments,
        )

    # ------------------------------------------------------------------
    # NICE helpers: cycle-averaged ΔC_m as a function of pressure envelope
    # ------------------------------------------------------------------

    def _delta_Cm_uF_per_cm2(
        self, P_env: NDArray[np.floating] | float
    ) -> NDArray[np.floating] | float:
        """Cycle-averaged <ΔC_m> in μF/cm² (units used by HH solver).

        Returns 0 if no NICE model is configured.
        """
        if self.nice_model is None or self._nice_P_grid is None:
            return 0.0
        dC_SI = np.interp(np.asarray(P_env), self._nice_P_grid, self._nice_dC_grid)
        return dC_SI * 100.0  # F/m² → μF/cm²

    def _C_eff_per_compartment(
        self,
        params: dict[str, NDArray[np.floating]],
        P_env: float | NDArray[np.floating],
        N: int,
    ) -> NDArray[np.floating]:
        """Build effective capacitance array of shape (3, N).

        Adds <ΔC_m>(P_env) to compartments listed in ``nice_compartments``.
        For per-voxel P_env (array of length N), each voxel can see a
        different pressure.
        """
        C_eff = np.broadcast_to(params["C_m"], (3, N)).astype(np.float64).copy()
        if self.nice_model is None:
            return C_eff
        dC = self._delta_Cm_uF_per_cm2(P_env)
        if np.isscalar(dC):
            dC_arr = np.full(N, float(dC))
        else:
            dC_arr = np.asarray(dC).reshape(-1)
            if dC_arr.size == 1:
                dC_arr = np.full(N, float(dC_arr.item()))
        for c in self.nice_compartments:
            C_eff[c, :] += dC_arr
        return C_eff

    # ------------------------------------------------------------------
    # Internal: stacked per-compartment parameters as (3, 1) arrays
    # ------------------------------------------------------------------

    def _stacked_params(self) -> dict[str, NDArray[np.floating]]:
        comps = (self.geometry.dendrite, self.geometry.soma, self.geometry.ais)
        out: dict[str, NDArray[np.floating]] = {}
        for k in ("g_Na", "g_K", "g_L", "E_Na", "E_K", "E_L", "C_m", "area"):
            out[k] = np.array([[getattr(c, k)] for c in comps], dtype=np.float64)
        # Mechano density per (compartment, channel): shape (3, n_ch, 1)
        n_ch = len(self.mechano_channels)
        if n_ch:
            base = np.array([ch.g_density for ch in self.mechano_channels], dtype=np.float64)
            scales = np.array(
                [c.mechano_density_scale for c in comps],
                dtype=np.float64,
            )  # (3, n_ch)
            out["g_mech"] = (base[None, :] * scales)[..., None]  # (3, n_ch, 1)
        else:
            out["g_mech"] = np.zeros((3, 0, 1), dtype=np.float64)
        return out

    def _axial_currents(
        self,
        V: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Axial coupling current density per compartment [uA/cm^2].

        Parameters
        ----------
        V : ndarray, shape (3, N)
            Membrane potentials [dendrite, soma, ais].

        Returns
        -------
        I_axial : ndarray, shape (3, N)

        """
        g_ds = self.geometry.g_dend_soma  # mS/cm^2 of soma
        g_sa = self.geometry.g_soma_ais
        A_d = self.geometry.dendrite.area
        A_s = self.geometry.soma.area
        A_a = self.geometry.ais.area

        Vd, Vs, Va = V[0], V[1], V[2]
        I_d = g_ds * (A_s / A_d) * (Vs - Vd)
        I_s = g_ds * (Vd - Vs) + g_sa * (Va - Vs)
        I_a = g_sa * (A_s / A_a) * (Vs - Va)
        return np.stack([I_d, I_s, I_a], axis=0)

    def _channel_kinetics(
        self,
        T: NDArray[np.floating],  # (N,) tension
        n_ch: int,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
        """Compute (E_revs, w_inf, tau_inv, kT) helpers for activation gates.

        Returns
        -------
        E_revs : ndarray, shape (n_ch,)
        w_inf  : ndarray, shape (n_ch, N)
        tau_inv: ndarray, shape (n_ch, N)
        kT     : float
        """
        if n_ch == 0:
            return (
                np.empty(0),
                np.empty((0, T.shape[0])),
                np.empty((0, T.shape[0])),
                4.114,
            )
        E_revs = np.array([ch.E_rev for ch in self.mechano_channels], dtype=np.float64)
        T_halfs = np.array([ch.T_half for ch in self.mechano_channels], dtype=np.float64)
        delta_As = np.array([ch.delta_A for ch in self.mechano_channels], dtype=np.float64)
        tau_0s = np.array([ch.tau_0 for ch in self.mechano_channels], dtype=np.float64)
        kT = self.mechano_channels[0]._kT

        # Steady-state w from Boltzmann
        exponent = -delta_As[:, None] * (T[None, :] - T_halfs[:, None]) / kT
        w_inf = _safe_sigmoid(exponent)

        # tau: per-channel, fixed Phase-5 ``tau_act`` if set, else bell-shaped
        # ``tau_0 / (4·P_o(1-P_o) + 0.01)`` for backward compatibility.
        tau_inv = np.empty_like(w_inf)
        for k, ch in enumerate(self.mechano_channels):
            if ch.tau_act is not None:
                tau_inv[k] = 1.0 / float(ch.tau_act)
            else:
                denom_k = 4.0 * w_inf[k] * (1.0 - w_inf[k]) + 0.01
                tau_inv[k] = denom_k / tau_0s[k]
        return E_revs, w_inf, tau_inv, kT

    def _mechano_channel_params(self) -> dict:
        """Per-channel scalars for in-kernel tension → gate kinetics.

        Used by the fast-path kernels when the tension is time-varying
        (``drive``); mirrors ``_channel_kinetics`` and
        ``_inactivation_kinetics`` exactly.
        """
        chs = self.mechano_channels
        n_ch = len(chs)
        f64 = np.float64
        out = {
            "T_half": np.array([ch.T_half for ch in chs], dtype=f64),
            "dA": np.array([ch.delta_A for ch in chs], dtype=f64),
            "tau_act_inv": np.array(
                [1.0 / float(ch.tau_act) if ch.tau_act is not None else 0.0
                 for ch in chs], dtype=f64),
            "tau0": np.array([ch.tau_0 for ch in chs], dtype=f64),
            "inact_on": np.zeros(n_ch, dtype=f64),
            "hT_half": np.zeros(n_ch, dtype=f64),
            "hdA": np.zeros(n_ch, dtype=f64),
            "htau_inv": np.zeros(n_ch, dtype=f64),
            "kT": float(chs[0]._kT) if n_ch else 4.114,
        }
        if self.piezo_inactivation:
            for k, ch in enumerate(chs):
                if ch.tau_inact is None:
                    continue
                out["inact_on"][k] = 1.0
                out["hT_half"][k] = (ch.T_half if ch.inact_T_half is None
                                     else ch.inact_T_half)
                out["hdA"][k] = (ch.delta_A if ch.inact_delta_A is None
                                 else ch.inact_delta_A)
                out["htau_inv"][k] = 1.0 / ch.tau_inact
        return out

    def _rest_mechano_gates(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Zero-tension steady state of the mechano gates, per channel.

        Returns ``(w_rest, h_rest)``, each shape (n_ch,): activation
        ``w_inf(T=0)`` from :meth:`_channel_kinetics` and inactivation
        ``h_inf(T=0)`` from :meth:`_inactivation_kinetics` (1.0 when
        inactivation is off). These are the ``mechano_init="rest"``
        initial values used by every integrator path.
        """
        n_ch = len(self.mechano_channels)
        T0 = np.zeros(1, dtype=np.float64)
        _, w0, _, _ = self._channel_kinetics(T0, n_ch)
        h0, _, _ = self._inactivation_kinetics(T0, n_ch)
        return (np.ascontiguousarray(w0[:, 0], dtype=np.float64),
                np.ascontiguousarray(h0[:, 0], dtype=np.float64))

    def _inactivation_kinetics(
        self,
        T: NDArray[np.floating],  # (N,) tension
        n_ch: int,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.bool_]]:
        """Inactivation steady-state and time constants, per channel per voxel.

        Returns
        -------
        h_inf  : ndarray, shape (n_ch, N)  — i_∞(T); 1.0 for non-inactivating channels
        h_tau_inv : ndarray, shape (n_ch, N)  — 1/τ_inact; 0 for non-inactivating channels
        has_inact : ndarray, shape (n_ch,)   — bool mask of inactivating channels
        """
        N = T.shape[0]
        if n_ch == 0 or not self.piezo_inactivation:
            return (
                np.ones((n_ch, N), dtype=np.float64),
                np.zeros((n_ch, N), dtype=np.float64),
                np.zeros(n_ch, dtype=bool),
            )
        kT = self.mechano_channels[0]._kT
        has_inact = np.array(
            [ch.tau_inact is not None for ch in self.mechano_channels], dtype=bool
        )
        h_inf = np.ones((n_ch, N), dtype=np.float64)
        h_tau_inv = np.zeros((n_ch, N), dtype=np.float64)
        for k, ch in enumerate(self.mechano_channels):
            if ch.tau_inact is None:
                continue
            T_h = ch.T_half if ch.inact_T_half is None else ch.inact_T_half
            dA = ch.delta_A if ch.inact_delta_A is None else ch.inact_delta_A
            # Inactivation gate: closes (→0) as T rises; opposite sign to activation
            exponent = dA * (T - T_h) / kT
            h_inf[k] = _safe_sigmoid(exponent)
            h_tau_inv[k] = 1.0 / ch.tau_inact
        return h_inf, h_tau_inv, has_inact

    # ------------------------------------------------------------------
    # Derivatives
    # ------------------------------------------------------------------

    def _derivs(
        self,
        V: NDArray[np.floating],  # (3, N)
        m: NDArray[np.floating],  # (3, N)
        h: NDArray[np.floating],  # (3, N)
        n: NDArray[np.floating],  # (3, N)
        w: NDArray[np.floating],  # (3, n_ch, N)
        params: dict[str, NDArray[np.floating]],
        E_revs: NDArray[np.floating],  # (n_ch,)
        w_inf: NDArray[np.floating],  # (n_ch, N)
        tau_inv: NDArray[np.floating],  # (n_ch, N)
        I_stim: NDArray[np.floating],  # (3, N) or scalar
        phi: float,
        C_eff: NDArray[np.floating] | None = None,  # (3, N)
        dC_eff_dt: NDArray[np.floating] | None = None,  # (3, N)
        h_mech: NDArray[np.floating] | None = None,  # (3, n_ch, N), Phase-3 inactivation
        h_inf: NDArray[np.floating] | None = None,  # (n_ch, N)
        h_tau_inv: NDArray[np.floating] | None = None,  # (n_ch, N)
        Ca_i: NDArray[np.floating] | None = None,  # (3, N) intracellular [Ca²⁺]
    ) -> tuple:
        kin = self._kinetics
        am = kin._alpha_m(V) * phi
        bm = kin._beta_m(V) * phi
        ah = kin._alpha_h(V) * phi
        bh = kin._beta_h(V) * phi
        an = kin._alpha_n(V) * phi
        bn = kin._beta_n(V) * phi

        I_Na = params["g_Na"] * (m**3) * h * (V - params["E_Na"])
        I_K = params["g_K"] * (n**4) * (V - params["E_K"])
        I_L = params["g_L"] * (V - params["E_L"])

        # Mechano current per compartment per channel.
        # With Phase-3 inactivation: I_k = g_k · m_k · h_k · (E_rev - V).
        # Without: I_k = g_k · m_k · (E_rev - V) — h_mech is None / treated as 1.
        n_ch = w.shape[1]
        per_channel_curr: NDArray[np.floating] | None = None
        if n_ch > 0:
            E_rev_b = E_revs[None, :, None]  # (1, n_ch, 1)
            gating = w * h_mech if h_mech is not None else w
            per_channel_curr = params["g_mech"] * gating * (E_rev_b - V[:, None, :])
            I_mech = np.sum(per_channel_curr, axis=1)
        else:
            I_mech = np.zeros_like(V)

        I_axial = self._axial_currents(V)

        # SK / KCa current — only from compartments listed in kca_compartments
        if self.kca_channels and Ca_i is not None:
            I_KCa: NDArray[np.floating] | float = np.zeros_like(V)
            mask = np.zeros((3, 1), dtype=np.float64)
            for c in self.kca_compartments:
                mask[c, 0] = 1.0
            for ch in self.kca_channels:
                P_o = ch.open_probability(Ca_i)  # (3, N)
                I_KCa = I_KCa + mask * (ch.g_density * P_o * (ch.E_K - V))
        else:
            I_KCa = 0.0

        # Phase-4: TRP thermosensors. Open probability evaluated once at
        # self.temperature; current contributes to trp_compartments only.
        if self.trp_channels:
            I_TRP: NDArray[np.floating] | float = np.zeros_like(V)
            trp_mask = np.zeros((3, 1), dtype=np.float64)
            for c in self.trp_compartments:
                trp_mask[c, 0] = 1.0
            for ch in self.trp_channels:
                P_o_T = float(ch.open_probability(self.temperature))
                I_TRP = I_TRP + trp_mask * (ch.g_density * P_o_T * (ch.E_rev - V))
        else:
            I_TRP = 0.0

        # Effective capacitance: C_eff defaults to compartment C_m if NICE
        # is disabled. The displacement current term V·dC_eff/dt comes
        # from differentiating Q = C_eff·V at constant total charge flux.
        if C_eff is None:
            C_eff = params["C_m"]
        I_disp = (V * dC_eff_dt) if dC_eff_dt is not None else 0.0

        dV = (-I_Na - I_K - I_L + I_mech + I_KCa + I_TRP + I_axial + I_stim - I_disp) / C_eff
        dm = am * (1.0 - m) - bm * m
        dh = ah * (1.0 - h) - bh * h
        dn = an * (1.0 - n) - bn * n

        # Mechano activation gates
        if n_ch > 0:
            dw = (w_inf[None, :, :] - w) * tau_inv[None, :, :]
        else:
            dw = np.zeros_like(w)

        # Mechano inactivation gates (Phase 3)
        if h_mech is not None:
            dh_mech = (h_inf[None, :, :] - h_mech) * h_tau_inv[None, :, :]
        else:
            dh_mech = None

        # Ca²⁺ pool dynamics (Phase 3): one pool per compartment, fed by
        # the Ca²⁺ portion of mechano current (positive = Ca²⁺ entering).
        if Ca_i is not None and self.calcium_dynamics is not None:
            cd = self.calcium_dynamics
            if n_ch > 0 and per_channel_curr is not None:
                Ca_frac = np.array(
                    [ch.Ca_permeability_fraction for ch in self.mechano_channels],
                    dtype=np.float64,
                )
                I_Ca = np.sum(per_channel_curr * Ca_frac[None, :, None], axis=1)
            else:
                I_Ca = np.zeros_like(V)
            dCa = cd.flux_factor * I_Ca - (Ca_i - cd.Ca_rest) / cd.tau_decay
        else:
            dCa = None

        return dV, dm, dh, dn, dw, dh_mech, dCa

    # ------------------------------------------------------------------
    # simulate (single voxel, returns full traces per compartment)
    # ------------------------------------------------------------------

    def simulate(
        self,
        dt: float = 0.001,
        duration: float = 50.0,
        I_stim: NDArray[np.floating] | float | dict[str, float] = 0.0,
        tension_trace: NDArray[np.floating] | float = 0.0,
        pressure_envelope: NDArray[np.floating] | float = 0.0,
        mechano_init: str = "rest",
    ) -> dict[str, NDArray[np.floating]]:
        """Single-voxel multi-compartment trace simulation.

        Parameters
        ----------
        dt : float
            Time step [ms]. Default 0.001 ms — 10× finer than the
            single-compartment default because the high AIS Na density
            (~2000 mS/cm^2 by default) makes the RK4 stability margin
            tighter. Reduce further if g_Na on any compartment is
            increased above ~3000 mS/cm^2.
        duration : float
            Total simulation time [ms]. Default 50.
        I_stim : float, ndarray, or dict
            Stimulus current density [uA/cm^2].  Scalar applied to soma
            only; ndarray of length 3 distributed across compartments;
            dict with keys ``"dendrite", "soma", "ais"`` for named
            injection (missing keys default to 0).
        tension_trace : float or 1-D ndarray
            Membrane tension [mN/m].  Scalar for constant; 1-D array of
            length n_steps for time-varying.
        pressure_envelope : float or 1-D ndarray
            Slow (ms-scale) pressure envelope [Pa] for the NICE model.
            Ignored when ``nice_model is None``. Scalar for constant
            amplitude, or 1-D array of length n_steps for pulse trains.
        mechano_init : {"rest", "peak_steady"}
            Initial mechano gates (activation ``w`` and inactivation
            ``h_mech``). ``"rest"`` (default): steady state for zero
            tension — the ON window starts from true rest, as every
            theta burst does because the tension relaxes during the
            180 ms OFF time (burst-periodicity study,
            ``data/halle_dacc/calcium/burst_periodicity.json``).
            ``"peak_steady"``: steady state for ``tension_trace[0]`` (the
            legacy canonical initial condition). V, HH gates and [Ca]
            start at rest in both modes. See :data:`MECHANO_INIT_MODES`.

        Returns
        -------
        dict
            ``'t'``     : (n_steps,)            time [ms]
            ``'V'``     : (n_steps, 3)          voltage per compartment
            ``'m','h','n'`` : (n_steps, 3)      HH gates per compartment
            ``'I_mech'``: (n_steps, 3)          mechano current per compartment
            ``'C_eff'`` : (n_steps, 3)          effective capacitance [μF/cm²]
            ``'spike_times_per_compartment'`` : list of length 3
            ``'spike_count_per_compartment'`` : ndarray (3,)
            Compartment ordering is ``[dendrite, soma, ais]``.

        """
        n_steps = int(duration / dt)
        params = self._stacked_params()
        n_ch = len(self.mechano_channels)
        phi = self._kinetics._q10_scale()

        # Build I_stim shape (3, 1)
        if isinstance(I_stim, dict):
            I_stim_arr = np.array(
                [[I_stim.get(name, 0.0)] for name in ("dendrite", "soma", "ais")],
                dtype=np.float64,
            )
        elif np.isscalar(I_stim):
            I_stim_arr = np.array([[0.0], [float(I_stim)], [0.0]])  # default → soma
        else:
            arr = np.asarray(I_stim, dtype=np.float64).reshape(-1)
            if arr.size != 3:
                msg = "I_stim ndarray must have length 3 ([dend, soma, ais])."
                raise ValueError(msg)
            I_stim_arr = arr.reshape(3, 1)

        if np.isscalar(tension_trace):
            tension_arr = np.full(n_steps, float(tension_trace))
        else:
            tension_arr = np.asarray(tension_trace, dtype=np.float64)
            if tension_arr.size != n_steps:
                msg = f"tension_trace length must be {n_steps}, got {tension_arr.size}"
                raise ValueError(msg)

        # Pressure envelope → cycle-averaged C_eff trace (per timestep)
        if np.isscalar(pressure_envelope):
            P_env_arr = np.full(n_steps, float(pressure_envelope))
        else:
            P_env_arr = np.asarray(pressure_envelope, dtype=np.float64)
            if P_env_arr.size != n_steps:
                msg = f"pressure_envelope length must be {n_steps}, got {P_env_arr.size}"
                raise ValueError(msg)
        # Pre-compute (3, n_steps) effective capacitance and its time-derivative
        if self.nice_model is not None:
            dC_uF_per_cm2 = self._delta_Cm_uF_per_cm2(P_env_arr)  # (n_steps,)
            # Build per-compartment trace; only nice_compartments get the boost
            C_trace = np.broadcast_to(params["C_m"], (3, n_steps)).astype(np.float64).copy()
            for c in self.nice_compartments:
                C_trace[c, :] += dC_uF_per_cm2
            # dC/dt by central differences on the slow envelope (ms timescale)
            dC_dt_trace = np.gradient(C_trace, dt, axis=1)
        else:
            C_trace = np.broadcast_to(params["C_m"], (3, n_steps)).astype(np.float64)
            dC_dt_trace = np.zeros_like(C_trace)

        # State as (3, 1) — N=1 voxel, broadcasts cleanly with batch code.
        m0, h0, n0 = self._kinetics._steady_state_gates(self.V_rest)
        V = np.full((3, 1), self.V_rest)
        m = np.full((3, 1), m0)
        h = np.full((3, 1), h0)
        nn = np.full((3, 1), n0)
        # Initialise mechano gates at steady state for zero tension (rest,
        # default) or for tension[0] (legacy 'peak_steady')
        T_init = 0.0 if _check_mechano_init(mechano_init) else tension_arr[0]
        E_revs0, w_inf0, tau_inv0, _kT = self._channel_kinetics(
            np.array([T_init], dtype=np.float64), n_ch
        )
        w = np.broadcast_to(w_inf0[None, :, :], (3, n_ch, 1)).copy() if n_ch else np.zeros((3, 0, 1))
        # Phase-3: inactivation gate, initialised at h_inf for the same tension
        if self.piezo_inactivation:
            h_inf0, h_tau_inv0, _ = self._inactivation_kinetics(
                np.array([T_init], dtype=np.float64), n_ch)
            h_mech = np.broadcast_to(h_inf0[None, :, :], (3, n_ch, 1)).copy() if n_ch else None
        else:
            h_mech = None
        # Phase-3: Ca²⁺ pool initialised at rest concentration
        if self.calcium_dynamics is not None:
            Ca_i = np.full((3, 1), self.calcium_dynamics.Ca_rest)
        else:
            Ca_i = None

        V_out = np.empty((n_steps, 3))
        m_out = np.empty((n_steps, 3))
        h_out = np.empty((n_steps, 3))
        n_out = np.empty((n_steps, 3))
        Imech_out = np.empty((n_steps, 3))
        Ceff_out = np.empty((n_steps, 3))
        Ca_out = np.empty((n_steps, 3)) if Ca_i is not None else None
        h_mech_out = np.empty((n_steps, 3, n_ch)) if h_mech is not None else None
        spike_count = np.zeros(3, dtype=np.int32)
        spike_times: list[list[float]] = [[], [], []]

        def _step_derivs(Vs, ms, hs, ns, ws, hm, ca, T_step, C_eff_step, dC_eff_dt_step,
                         I_stim_step, w_inf_s, tau_inv_s, h_inf_s, h_tau_inv_s, E_revs_s):
            return self._derivs(
                Vs, ms, hs, ns, ws, params, E_revs_s, w_inf_s, tau_inv_s,
                I_stim_step, phi,
                C_eff=C_eff_step, dC_eff_dt=dC_eff_dt_step,
                h_mech=hm, h_inf=h_inf_s, h_tau_inv=h_tau_inv_s,
                Ca_i=ca,
            )

        V_prev = V.copy()
        for i in range(n_steps):
            T_now = np.array([tension_arr[i]])
            E_revs, w_inf, tau_inv, _ = self._channel_kinetics(T_now, n_ch)
            if self.piezo_inactivation:
                h_inf_now, h_tau_inv_now, _has = self._inactivation_kinetics(T_now, n_ch)
            else:
                h_inf_now = h_tau_inv_now = None

            C_eff_step = C_trace[:, i : i + 1]
            dC_eff_dt_step = dC_dt_trace[:, i : i + 1]

            # Record state *before* stepping
            V_out[i] = V[:, 0]
            m_out[i] = m[:, 0]
            h_out[i] = h[:, 0]
            n_out[i] = nn[:, 0]
            Ceff_out[i] = C_eff_step[:, 0]
            if Ca_out is not None:
                Ca_out[i] = Ca_i[:, 0]
            if h_mech_out is not None:
                h_mech_out[i] = h_mech[:, :, 0]
            if n_ch:
                E_rev_b = E_revs[None, :, None]
                gating = w * h_mech if h_mech is not None else w
                Imech_out[i] = np.sum(
                    params["g_mech"] * gating * (E_rev_b - V[:, None, :]), axis=1
                )[:, 0]
            else:
                Imech_out[i] = 0.0

            # RK4: hold C_eff/dC_eff_dt fixed across stages
            k1 = _step_derivs(V, m, h, nn, w, h_mech, Ca_i, T_now,
                              C_eff_step, dC_eff_dt_step, I_stim_arr,
                              w_inf, tau_inv, h_inf_now, h_tau_inv_now, E_revs)
            k1V, k1m, k1h, k1n, k1w, k1h_mech, k1Ca = k1

            def _add(state, k):
                return state + 0.5 * dt * k if state is not None else None

            k2 = _step_derivs(
                V + 0.5*dt*k1V, m + 0.5*dt*k1m, h + 0.5*dt*k1h,
                nn + 0.5*dt*k1n, w + 0.5*dt*k1w,
                _add(h_mech, k1h_mech), _add(Ca_i, k1Ca),
                T_now, C_eff_step, dC_eff_dt_step, I_stim_arr,
                w_inf, tau_inv, h_inf_now, h_tau_inv_now, E_revs,
            )
            k2V, k2m, k2h, k2n, k2w, k2h_mech, k2Ca = k2

            k3 = _step_derivs(
                V + 0.5*dt*k2V, m + 0.5*dt*k2m, h + 0.5*dt*k2h,
                nn + 0.5*dt*k2n, w + 0.5*dt*k2w,
                _add(h_mech, k2h_mech), _add(Ca_i, k2Ca),
                T_now, C_eff_step, dC_eff_dt_step, I_stim_arr,
                w_inf, tau_inv, h_inf_now, h_tau_inv_now, E_revs,
            )
            k3V, k3m, k3h, k3n, k3w, k3h_mech, k3Ca = k3

            def _full(state, k):
                return state + dt * k if state is not None else None

            k4 = _step_derivs(
                V + dt*k3V, m + dt*k3m, h + dt*k3h,
                nn + dt*k3n, w + dt*k3w,
                _full(h_mech, k3h_mech), _full(Ca_i, k3Ca),
                T_now, C_eff_step, dC_eff_dt_step, I_stim_arr,
                w_inf, tau_inv, h_inf_now, h_tau_inv_now, E_revs,
            )
            k4V, k4m, k4h, k4n, k4w, k4h_mech, k4Ca = k4

            s = dt / 6.0
            V_prev = V.copy()
            V = V + s * (k1V + 2*k2V + 2*k3V + k4V)
            m = m + s * (k1m + 2*k2m + 2*k3m + k4m)
            h = h + s * (k1h + 2*k2h + 2*k3h + k4h)
            nn = nn + s * (k1n + 2*k2n + 2*k3n + k4n)
            if n_ch:
                w = w + s * (k1w + 2*k2w + 2*k3w + k4w)
                np.clip(w, 0.0, 1.0, out=w)
            if h_mech is not None:
                h_mech = h_mech + s * (k1h_mech + 2*k2h_mech + 2*k3h_mech + k4h_mech)
                np.clip(h_mech, 0.0, 1.0, out=h_mech)
            if Ca_i is not None:
                Ca_i = Ca_i + s * (k1Ca + 2*k2Ca + 2*k3Ca + k4Ca)
                np.clip(Ca_i, 0.0, None, out=Ca_i)  # [Ca²⁺] non-negative
            np.clip(m, 0.0, 1.0, out=m)
            np.clip(h, 0.0, 1.0, out=h)
            np.clip(nn, 0.0, 1.0, out=nn)

            for c in range(3):
                if V_prev[c, 0] <= 0.0 < V[c, 0]:
                    spike_count[c] += 1
                    spike_times[c].append((i + 1) * dt)

        out = {
            "t": np.arange(n_steps) * dt,
            "V": V_out,
            "m": m_out,
            "h": h_out,
            "n": n_out,
            "I_mech": Imech_out,
            "C_eff": Ceff_out,
            "spike_count_per_compartment": spike_count,
            "spike_times_per_compartment": [np.asarray(s) for s in spike_times],
        }
        if Ca_out is not None:
            out["Ca_i"] = Ca_out
        if h_mech_out is not None:
            out["h_mech"] = h_mech_out
        return out

    # ------------------------------------------------------------------
    # Batch over voxels (constant tension per voxel)
    # ------------------------------------------------------------------

    def _simulate_batch(
        self,
        tensions: NDArray[np.floating],  # (N,)
        dt: float,
        n_steps: int,
        I_stim_per_compartment: NDArray[np.floating],  # (3,)
        primary_compartment: int = 1,  # soma
        pressure_envelope: NDArray[np.floating] | float = 0.0,  # (N,) or (n_steps, N) or scalar
        backend: str = "auto",
        integrator: str = "rk4",
        *,
        drive: NDArray[np.floating] | None = None,
        initial_state: NDArray[np.floating] | None = None,
        return_state: bool = False,
        ca_trace_decim: int = 0,
        mechano_init: str = "rest",
    ) -> NDArray[np.integer]:
        """Run N voxels in parallel; return spike count at primary compartment.

        ``mechano_init`` selects the initial mechano gates of every path
        (Numba/CUDA RK4, Rush-Larsen, NumPy fallback): ``"rest"`` (default)
        = zero-tension steady state (true rest); ``"peak_steady"`` = steady
        state for each voxel's step-0 tension (legacy canonical). An
        explicit ``initial_state`` overrides it. See
        :data:`ultraneuron.neuron.MECHANO_INIT_MODES`.

        Optional window-chaining / readout arguments (RK4 fast path only;
        see :func:`simulate_voxels_3comp` for the public wrapper):

        * ``drive`` — (n_steps,) tension multiplier shared by all voxels,
          T_i(t) = tensions[i] · drive[t] (see :func:`tension_drive`).
          ``None`` → constant tension (canonical).
        * ``initial_state`` — (N, n_state) start state (layout from
          :func:`state_layout`); ``None`` → V_rest, HH gates at steady
          state, [Ca] = Ca_rest and mechano gates per ``mechano_init``
          (:func:`rest_initial_state` / :func:`peak_steady_initial_state`).
        * ``return_state`` — store the end-of-window (N, n_state) state in
          ``self._last_final_state``.
        * ``ca_trace_decim`` — when > 0 and a Ca pool is attached, store
          [Ca] every ``ca_trace_decim`` steps in ``self._last_ca_trace``
          (N, 3, n_steps // ca_trace_decim) float32.

        With a Ca pool attached the per-voxel per-compartment peak [Ca]
        over the window is always stored in ``self._last_ca_peak`` (N, 3).

        ``pressure_envelope`` follows NICE conventions:
            * scalar — applied uniformly across voxels and time.
            * (N,)    — constant per voxel (one envelope value each).
            * (n_steps, N) — fully time- and voxel-resolved.

        ``backend`` selects the per-voxel integrator (``"auto"``, the default,
        uses CUDA whenever a device is visible and the ``AMN_BACKEND``
        environment variable does not say otherwise; see
        :func:`ultraneuron.cuda_backend.resolve_backend`):
            * ``"numba"`` — multi-threaded CPU Numba kernel (default).
            * ``"cuda"``  — single-GPU @cuda.jit kernel via
              :mod:`ultraneuron.cuda_backend`. Falls back to
              ``"numba"`` with a warning when no CUDA device is visible
              or when the configuration is outside the kernel's scope
              (e.g. fully time-varying NICE envelopes).
        """
        N = len(tensions)
        n_ch = len(self.mechano_channels)
        params = self._stacked_params()
        phi = self._kinetics._q10_scale()
        I_stim_arr = np.asarray(I_stim_per_compartment, dtype=np.float64).reshape(3, 1)
        _init_rest = _check_mechano_init(mechano_init)
        w_rest_ch, hm_rest_ch = self._rest_mechano_gates()

        # ---- Fast path: numba 3-comp kernel ------------------------------
        # Eligible for HH + mechanosensitive activation/inactivation +
        # (optional) Ca²⁺ pool + KCa + TRP thermosensors + NICE
        # intramembrane cavitation with a constant pressure envelope.
        # Covers manuscript scenarios S1–S5. Falls through to the NumPy
        # path only for fully time-varying NICE envelopes ((n_steps, N)).
        _P_arr = np.asarray(pressure_envelope, dtype=np.float64)
        _nice_envelope_constant = (self.nice_model is None
                                   or _P_arr.ndim == 0
                                   or _P_arr.shape == (N,))
        # Pick the per-voxel kernel. CUDA falls back to Numba when the
        # backend is unavailable or asks for a configuration outside the
        # CUDA kernel's scope.
        from . import cuda_backend as _cuda  # noqa: PLC0415
        backend = _cuda.resolve_backend(backend)
        if backend == "cuda":
            if not _cuda.is_cuda_available():
                _cuda.maybe_warn_unavailable("cuda")
                _use_cuda = False
            elif n_ch > _cuda._MAX_CH:
                import warnings as _warn  # noqa: PLC0415
                _warn.warn(
                    f"CUDA backend supports up to {_cuda._MAX_CH} mechanosensitive "
                    f"channels; got n_ch={n_ch}. Falling back to Numba.",
                    RuntimeWarning, stacklevel=2,
                )
                _use_cuda = False
            else:
                _use_cuda = True
        else:
            _use_cuda = False
        _fast_eligible = (_HAS_NUMBA_3COMP or _use_cuda) and _nice_envelope_constant
        if integrator not in ("rk4", "rl"):
            raise ValueError(
                f"integrator must be 'rk4' or 'rl', got {integrator!r}",
            )
        if integrator == "rl" and not (_HAS_NUMBA_3COMP and _nice_envelope_constant):
            raise NotImplementedError(
                "integrator='rl' requires the Numba 3-comp fast path with a "
                "constant pressure envelope. Use integrator='rk4'.",
            )
        # Window chaining / readouts are implemented in the RK4 fast-path
        # kernels (Numba CPU and CUDA) only.
        self._last_final_state = None
        self._last_ca_peak = None
        self._last_ca_trace = None
        _extended = (drive is not None or initial_state is not None
                     or bool(return_state) or int(ca_trace_decim) > 0)
        if _extended and (integrator != "rk4" or not _fast_eligible):
            raise NotImplementedError(
                "drive / initial_state / return_state / ca_trace_decim require "
                "integrator='rk4' on the Numba/CUDA 3-comp fast path with a "
                "constant pressure envelope.",
            )
        if _fast_eligible:
            tensions_fp = np.asarray(tensions, dtype=np.float64)
            E_revs_fp, w_inf_fp, tau_inv_fp, _ = self._channel_kinetics(
                tensions_fp, n_ch,
            )
            if self.piezo_inactivation:
                h_inf_fp, h_tau_inv_fp, _ = self._inactivation_kinetics(
                    tensions_fp, n_ch,
                )
            else:
                # h=1 with zero rate ⟹ h_mech stays at 1.0 (no inactivation).
                h_inf_fp = np.ones((n_ch, N), dtype=np.float64)
                h_tau_inv_fp = np.zeros((n_ch, N), dtype=np.float64)
            m0_init, h0_init, n0_init = self._kinetics._steady_state_gates(
                self.V_rest,
            )
            geom = self.geometry

            # Ca²⁺ pool inputs
            if self.calcium_dynamics is not None:
                cd = self.calcium_dynamics
                ca_flux_factor = float(cd.flux_factor)
                ca_tau_inv = 1.0 / float(cd.tau_decay)
                ca_rest = float(cd.Ca_rest)
                Ca_frac_fp = np.array(
                    [ch.Ca_permeability_fraction
                     for ch in self.mechano_channels],
                    dtype=np.float64,
                )
            else:
                ca_flux_factor = 0.0
                ca_tau_inv = 0.0
                ca_rest = 0.0
                Ca_frac_fp = np.zeros(n_ch, dtype=np.float64)

            # KCa channel inputs
            if self.kca_channels:
                kca_g_fp = np.array(
                    [ch.g_density for ch in self.kca_channels],
                    dtype=np.float64,
                )
                kca_E_K_fp = np.array(
                    [ch.E_K for ch in self.kca_channels],
                    dtype=np.float64,
                )
                kca_half_fp = np.array(
                    [ch.Ca_half for ch in self.kca_channels],
                    dtype=np.float64,
                )
                kca_n_fp = np.array(
                    [ch.n_hill for ch in self.kca_channels],
                    dtype=np.float64,
                )
                kca_mask_fp = np.zeros(3, dtype=np.float64)
                for c in self.kca_compartments:
                    kca_mask_fp[c] = 1.0
            else:
                kca_g_fp = np.zeros(0, dtype=np.float64)
                kca_E_K_fp = np.zeros(0, dtype=np.float64)
                kca_half_fp = np.zeros(0, dtype=np.float64)
                kca_n_fp = np.zeros(0, dtype=np.float64)
                kca_mask_fp = np.zeros(3, dtype=np.float64)

            # TRP channel inputs — Po(T) evaluated once at body temperature
            if self.trp_channels:
                trp_g_Po_fp = np.array(
                    [float(ch.g_density)
                     * float(ch.open_probability(self.temperature))
                     for ch in self.trp_channels],
                    dtype=np.float64,
                )
                trp_E_rev_fp = np.array(
                    [float(ch.E_rev) for ch in self.trp_channels],
                    dtype=np.float64,
                )
                trp_mask_fp = np.zeros(3, dtype=np.float64)
                for c in self.trp_compartments:
                    trp_mask_fp[c] = 1.0
            else:
                trp_g_Po_fp = np.zeros(0, dtype=np.float64)
                trp_E_rev_fp = np.zeros(0, dtype=np.float64)
                trp_mask_fp = np.zeros(3, dtype=np.float64)

            # Build C_eff (3, N): NICE folds in a per-voxel ΔC perturbation
            # on its compartments under a constant pressure envelope.
            C_m_per_comp_fp = np.broadcast_to(
                params["C_m"], (3, N),
            ).astype(np.float64).copy()
            if self.nice_model is not None:
                if _P_arr.ndim == 0:
                    dC_fp = float(self._delta_Cm_uF_per_cm2(float(_P_arr)))
                    for c in self.nice_compartments:
                        C_m_per_comp_fp[c, :] = (
                            float(params["C_m"][c, 0]) + dC_fp
                        )
                else:  # (N,) per-voxel constant
                    dC_fp = self._delta_Cm_uF_per_cm2(_P_arr)
                    for c in self.nice_compartments:
                        C_m_per_comp_fp[c, :] = (
                            float(params["C_m"][c, 0]) + dC_fp
                        )

            # Spike-step recording: present only if caller asked for it.
            _record_steps = bool(getattr(self, "_record_spike_steps", False))
            _max_spikes = int(getattr(self, "_max_spikes_per_voxel", 0))
            if _record_steps and _max_spikes > 0:
                spike_step_buf_fp = np.zeros(
                    (N, _max_spikes), dtype=np.int32,
                )
            else:
                spike_step_buf_fp = np.zeros((1, 1), dtype=np.int32)
                _max_spikes = 0
            self._last_spike_step_buf = spike_step_buf_fp  # caller can read

            # ---- Rush-Larsen (exponential-Euler) fast path ----------------
            # L-stable, ~3x faster than RK4 at equal dt; S2-only (HH +
            # Piezo1/TREK1, no Ca pool / KCa / TRP / inactivation / NICE).
            # Numba-only for now. See solver_investigation/FINDINGS.md.
            if integrator == "rl":
                _rl_blockers = []
                if self.calcium_dynamics is not None:
                    _rl_blockers.append("Ca pool")
                if self.kca_channels:
                    _rl_blockers.append("KCa")
                if self.trp_channels:
                    _rl_blockers.append("TRP")
                if self.piezo_inactivation:
                    _rl_blockers.append("Piezo1 inactivation")
                if self.nice_model is not None:
                    _rl_blockers.append("NICE")
                if _rl_blockers:
                    raise NotImplementedError(
                        "integrator='rl' (Rush-Larsen) supports the S2 scenario "
                        "only (HH + Piezo1/TREK1 mechano gates); this neuron "
                        f"enables {', '.join(_rl_blockers)}. Use integrator='rk4'."
                    )
                if not _HAS_NUMBA_3COMP:
                    raise NotImplementedError(
                        "integrator='rl' requires the Numba 3-comp kernel."
                    )
                if _use_cuda:
                    import warnings as _warn  # noqa: PLC0415
                    _warn.warn(
                        "RL integrator is Numba-only; running on Numba "
                        "(CUDA-RL not implemented).",
                        RuntimeWarning, stacklevel=2,
                    )
                return _rl_batch_numba_3comp(
                    np.ascontiguousarray(w_inf_fp, dtype=np.float64),
                    np.ascontiguousarray(tau_inv_fp, dtype=np.float64),
                    np.array([params["g_Na"][c, 0] for c in range(3)],
                             dtype=np.float64),
                    np.array([params["g_K"][c, 0] for c in range(3)],
                             dtype=np.float64),
                    np.array([params["g_L"][c, 0] for c in range(3)],
                             dtype=np.float64),
                    float(params["E_Na"][0, 0]),
                    float(params["E_K"][0, 0]),
                    float(params["E_L"][0, 0]),
                    np.ascontiguousarray(C_m_per_comp_fp, dtype=np.float64),
                    float(geom.g_dend_soma),
                    float(geom.g_soma_ais),
                    float(geom.soma.area / geom.dendrite.area),
                    float(geom.soma.area / geom.ais.area),
                    np.ascontiguousarray(params["g_mech"][:, :, 0],
                                         dtype=np.float64),
                    np.ascontiguousarray(E_revs_fp, dtype=np.float64),
                    bool(self.model == "cortical"),
                    float(self._kinetics._VT),
                    float(phi),
                    np.asarray(I_stim_per_compartment,
                               dtype=np.float64).reshape(3),
                    float(self.V_rest),
                    float(m0_init), float(h0_init), float(n0_init),
                    float(dt), int(n_steps),
                    int(primary_compartment),
                    spike_step_buf_fp,
                    int(_max_spikes),
                    int(_init_rest),
                    w_rest_ch,
                )

            # ---- Optional time-varying drive / state I/O / Ca readouts ----
            n_state = state_size(n_ch)
            if drive is not None:
                drive_fp = np.ascontiguousarray(drive, dtype=np.float64).reshape(-1)
                if drive_fp.size != n_steps:
                    raise ValueError(
                        f"drive must have n_steps={n_steps} entries, got "
                        f"{drive_fp.size}",
                    )
            else:
                drive_fp = np.zeros(0, dtype=np.float64)
            chp = self._mechano_channel_params()
            if initial_state is not None:
                init_fp = np.ascontiguousarray(initial_state, dtype=np.float64)
                if init_fp.shape != (N, n_state):
                    raise ValueError(
                        f"initial_state must have shape ({N}, {n_state}), got "
                        f"{init_fp.shape}",
                    )
            else:
                init_fp = np.zeros((1, 1), dtype=np.float64)
            final_fp = (np.zeros((N, n_state), dtype=np.float64) if return_state
                        else np.zeros((1, 1), dtype=np.float64))
            _has_ca = self.calcium_dynamics is not None
            ca_peak_fp = (np.zeros((N, 3), dtype=np.float64) if _has_ca
                          else np.zeros((1, 1), dtype=np.float64))
            _decim = int(ca_trace_decim) if _has_ca else 0
            if _decim > 0:
                ca_trace_fp = np.zeros((N, 3, n_steps // _decim), dtype=np.float32)
            else:
                ca_trace_fp = np.zeros((1, 1, 1), dtype=np.float32)
                _decim = 0

            _kernel_fn = (
                _cuda.rk4_batch_cuda_3comp if _use_cuda
                else _rk4_batch_numba_3comp
            )
            sc = _kernel_fn(
                np.ascontiguousarray(w_inf_fp, dtype=np.float64),
                np.ascontiguousarray(tau_inv_fp, dtype=np.float64),
                np.ascontiguousarray(h_inf_fp, dtype=np.float64),
                np.ascontiguousarray(h_tau_inv_fp, dtype=np.float64),
                np.array([params["g_Na"][c, 0] for c in range(3)],
                         dtype=np.float64),
                np.array([params["g_K"][c, 0] for c in range(3)],
                         dtype=np.float64),
                np.array([params["g_L"][c, 0] for c in range(3)],
                         dtype=np.float64),
                float(params["E_Na"][0, 0]),
                float(params["E_K"][0, 0]),
                float(params["E_L"][0, 0]),
                np.ascontiguousarray(C_m_per_comp_fp, dtype=np.float64),
                float(geom.g_dend_soma),
                float(geom.g_soma_ais),
                float(geom.soma.area / geom.dendrite.area),
                float(geom.soma.area / geom.ais.area),
                np.ascontiguousarray(params["g_mech"][:, :, 0],
                                     dtype=np.float64),  # (3, n_ch)
                np.ascontiguousarray(E_revs_fp, dtype=np.float64),
                Ca_frac_fp,
                ca_flux_factor, ca_tau_inv, ca_rest,
                kca_g_fp, kca_E_K_fp, kca_half_fp, kca_n_fp, kca_mask_fp,
                trp_g_Po_fp, trp_E_rev_fp, trp_mask_fp,
                bool(self.model == "cortical"),
                float(self._kinetics._VT),
                float(phi),
                np.asarray(I_stim_per_compartment, dtype=np.float64).reshape(3),
                float(self.V_rest),
                float(m0_init), float(h0_init), float(n0_init),
                float(dt), int(n_steps),
                int(primary_compartment),
                spike_step_buf_fp,
                int(_max_spikes),
                drive_fp, tensions_fp,
                chp["T_half"], chp["dA"], chp["tau_act_inv"], chp["tau0"],
                chp["inact_on"], chp["hT_half"], chp["hdA"], chp["htau_inv"],
                float(chp["kT"]),
                init_fp, final_fp,
                ca_peak_fp, ca_trace_fp, int(_decim),
                int(_init_rest), w_rest_ch, hm_rest_ch,
            )
            if return_state:
                self._last_final_state = final_fp
            if _has_ca:
                self._last_ca_peak = ca_peak_fp
            if _decim > 0:
                self._last_ca_trace = ca_trace_fp
            return sc

        # Pre-compute (3, n_steps, N) C_eff and dC/dt traces if NICE is enabled
        # — for the constant-envelope case the dC/dt is exactly zero.
        C_m_per_comp = np.broadcast_to(params["C_m"], (3, N)).astype(np.float64)
        if self.nice_model is not None:
            P_arr = np.asarray(pressure_envelope, dtype=np.float64)
            if P_arr.ndim == 0:
                # Scalar → constant in t and across voxels
                dC = float(self._delta_Cm_uF_per_cm2(float(P_arr)))
                C_traj_const = C_m_per_comp.copy()
                for c in self.nice_compartments:
                    C_traj_const[c, :] += dC
                # Constant envelope ⟹ dC/dt = 0; no per-step traces needed
                C_traj = None  # signal "use constant"
                dC_dt_traj = None
            elif P_arr.shape == (N,):
                # Per-voxel constant — same as scalar but voxel-varying
                dC = self._delta_Cm_uF_per_cm2(P_arr)  # (N,)
                C_traj_const = C_m_per_comp.copy()
                for c in self.nice_compartments:
                    C_traj_const[c, :] += dC
                C_traj = None
                dC_dt_traj = None
            elif P_arr.shape == (n_steps, N):
                # Full time-varying: build (3, n_steps, N) trace
                dC_t = self._delta_Cm_uF_per_cm2(P_arr)  # (n_steps, N)
                C_traj = np.broadcast_to(C_m_per_comp[:, None, :], (3, n_steps, N)).astype(
                    np.float64
                ).copy()
                for c in self.nice_compartments:
                    C_traj[c] += dC_t
                dC_dt_traj = np.gradient(C_traj, dt, axis=1)
                C_traj_const = None
            else:
                msg = (
                    f"pressure_envelope shape {P_arr.shape} not supported; "
                    f"expected scalar, ({N},), or ({n_steps},{N})."
                )
                raise ValueError(msg)
        else:
            C_traj_const = C_m_per_comp
            C_traj = None
            dC_dt_traj = None

        E_revs, w_inf, tau_inv, _ = self._channel_kinetics(tensions, n_ch)
        m0, h0, n0 = self._kinetics._steady_state_gates(self.V_rest)
        V = np.full((3, N), self.V_rest)
        m = np.full((3, N), m0)
        h = np.full((3, N), h0)
        nn = np.full((3, N), n0)
        w_start = (np.broadcast_to(w_rest_ch[:, None], (n_ch, N))
                   if _init_rest else w_inf)
        w = (
            np.broadcast_to(w_start[None, :, :], (3, n_ch, N)).copy()
            if n_ch
            else np.zeros((3, 0, N))
        )
        # Phase-3 state — initialised at steady state for zero tension
        # (rest) or for tensions[t=0] (legacy 'peak_steady')
        if self.piezo_inactivation:
            h_inf0, h_tau_inv0, _ = self._inactivation_kinetics(tensions, n_ch)
            h_start = (np.broadcast_to(hm_rest_ch[:, None], (n_ch, N))
                       if _init_rest else h_inf0)
            h_mech = (
                np.broadcast_to(h_start[None, :, :], (3, n_ch, N)).copy()
                if n_ch
                else None
            )
        else:
            h_mech = None
            h_inf0 = h_tau_inv0 = None
        if self.calcium_dynamics is not None:
            Ca_i = np.full((3, N), self.calcium_dynamics.Ca_rest)
        else:
            Ca_i = None

        spike_counts = np.zeros((3, N), dtype=np.int32)

        def _add(state, k, frac):
            return state + frac * k if state is not None else None

        for i in range(n_steps):
            V_prev = V.copy()

            if C_traj is not None:
                C_eff_step = C_traj[:, i, :]
                dC_eff_dt_step = dC_dt_traj[:, i, :]
            else:
                C_eff_step = C_traj_const
                dC_eff_dt_step = None

            common = dict(
                params=params, E_revs=E_revs, w_inf=w_inf, tau_inv=tau_inv,
                I_stim=I_stim_arr, phi=phi,
                C_eff=C_eff_step, dC_eff_dt=dC_eff_dt_step,
                h_inf=h_inf0, h_tau_inv=h_tau_inv0,
            )

            k1V, k1m, k1h, k1n, k1w, k1h_mech, k1Ca = self._derivs(
                V, m, h, nn, w, h_mech=h_mech, Ca_i=Ca_i, **common,
            )
            k2V, k2m, k2h, k2n, k2w, k2h_mech, k2Ca = self._derivs(
                V + 0.5*dt*k1V, m + 0.5*dt*k1m, h + 0.5*dt*k1h,
                nn + 0.5*dt*k1n, w + 0.5*dt*k1w,
                h_mech=_add(h_mech, k1h_mech, 0.5*dt),
                Ca_i=_add(Ca_i, k1Ca, 0.5*dt), **common,
            )
            k3V, k3m, k3h, k3n, k3w, k3h_mech, k3Ca = self._derivs(
                V + 0.5*dt*k2V, m + 0.5*dt*k2m, h + 0.5*dt*k2h,
                nn + 0.5*dt*k2n, w + 0.5*dt*k2w,
                h_mech=_add(h_mech, k2h_mech, 0.5*dt),
                Ca_i=_add(Ca_i, k2Ca, 0.5*dt), **common,
            )
            k4V, k4m, k4h, k4n, k4w, k4h_mech, k4Ca = self._derivs(
                V + dt*k3V, m + dt*k3m, h + dt*k3h,
                nn + dt*k3n, w + dt*k3w,
                h_mech=_add(h_mech, k3h_mech, dt),
                Ca_i=_add(Ca_i, k3Ca, dt), **common,
            )

            s = dt / 6.0
            V = V + s * (k1V + 2*k2V + 2*k3V + k4V)
            m = m + s * (k1m + 2*k2m + 2*k3m + k4m)
            h = h + s * (k1h + 2*k2h + 2*k3h + k4h)
            nn = nn + s * (k1n + 2*k2n + 2*k3n + k4n)
            if n_ch:
                w = w + s * (k1w + 2*k2w + 2*k3w + k4w)
                np.clip(w, 0.0, 1.0, out=w)
            if h_mech is not None:
                h_mech = h_mech + s * (k1h_mech + 2*k2h_mech + 2*k3h_mech + k4h_mech)
                np.clip(h_mech, 0.0, 1.0, out=h_mech)
            if Ca_i is not None:
                Ca_i = Ca_i + s * (k1Ca + 2*k2Ca + 2*k3Ca + k4Ca)
                np.clip(Ca_i, 0.0, None, out=Ca_i)
            np.clip(m, 0.0, 1.0, out=m)
            np.clip(h, 0.0, 1.0, out=h)
            np.clip(nn, 0.0, 1.0, out=nn)

            spike_counts += ((V > 0.0) & (V_prev <= 0.0)).astype(np.int32)

        return spike_counts[primary_compartment]


def _per_voxel_pressure(pressure_envelope, mask):
    """Reduce a scalar / 3-D pressure map to the (N,) array of masked voxels."""
    if np.isscalar(pressure_envelope):
        return float(pressure_envelope)
    P = np.asarray(pressure_envelope, dtype=np.float64)
    if P.shape != mask.shape:
        msg = f"pressure_envelope shape {P.shape} must match the strain grid {mask.shape}"
        raise ValueError(msg)
    return P[mask]


def simulate_map_3comp(
    strain_3d: NDArray[np.floating],
    neuron: MulticompartmentMechanoHHNeuron,
    K_A: float = 0.25,
    coupling_factor: float = 1.0,
    dt: float = 0.001,
    duration: float = 50.0,
    I_stim: NDArray[np.floating] | float = 0.0,
    tension_threshold: float = 0.0,
    primary_compartment: int = 1,  # soma
    backend: str = "auto",
    integrator: str = "rk4",
    pressure_envelope: NDArray[np.floating] | float = 0.0,
    mechano_init: str = "rest",
) -> dict[str, NDArray]:
    """Per-voxel firing map using the 3-compartment neuron.

    Mirrors the API of ``ultraneuron.simulate_map`` but uses
    a multi-compartment neuron with axial coupling.

    Parameters
    ----------
    strain_3d : ndarray, shape (nx, ny, nz)
        Deviatoric strain.
    neuron : MulticompartmentMechanoHHNeuron
    K_A, coupling_factor : float
        Strain → tension parameters (see ``strain_to_membrane_tension``).
    dt, duration : float
        Integration time step and total time [ms].
    I_stim : float or array of length 3
        Stimulus current density [uA/cm^2] (default applied to soma if scalar).
    tension_threshold : float
        Skip voxels with tension below this threshold [mN/m].
    primary_compartment : int
        Compartment index used to define a "spike" for the firing map
        (0=dendrite, 1=soma, 2=AIS). Default 1 (soma).
    pressure_envelope : float or ndarray (nx, ny, nz)
        Carrier pressure amplitude [Pa] driving the NICE intramembrane-
        cavitation module (ignored when the neuron has no ``nice_model``).
        A scalar applies to every voxel; a 3-D array supplies the per-voxel
        peak pressure of the acoustic field (constant over the window).
    mechano_init : {"rest", "peak_steady"}
        Initial mechano gates of every simulated window. ``"rest"``
        (default): steady state for zero tension — the ON window starts
        from true rest, as every burst of the theta-burst protocol does
        because the tension relaxes during the 180 ms OFF time (chained
        200 ms periods with full state carry-over,
        ``scripts/run_burst_periodicity.py``). ``"peak_steady"``: steady
        state for each voxel's peak tension (legacy canonical runs:
        halle_dacc S2 13381 spikes / 3590 voxels vs 12250 / 3531 from
        rest). See :data:`ultraneuron.neuron.MECHANO_INIT_MODES`.

    Returns
    -------
    dict
        ``'firing_map'``      : bool (nx, ny, nz)
        ``'spike_count_map'`` : int  (nx, ny, nz)
        ``'tension_3d'``      : float (nx, ny, nz)
    """
    from .channels import strain_to_membrane_tension

    nx, ny, nz = strain_3d.shape
    tension_3d = strain_to_membrane_tension(strain_3d, K_A, coupling_factor)
    firing = np.zeros((nx, ny, nz), dtype=bool)
    counts = np.zeros((nx, ny, nz), dtype=np.int32)

    mask = tension_3d >= tension_threshold
    if np.any(mask):
        tensions = np.asarray(tension_3d[mask], dtype=np.float64)
        n_steps = int(duration / dt)
        if np.isscalar(I_stim):
            I_arr = np.array([0.0, float(I_stim), 0.0])
        else:
            I_arr = np.asarray(I_stim, dtype=np.float64)
        P_env = _per_voxel_pressure(pressure_envelope, mask)
        sc = neuron._simulate_batch(
            tensions, dt, n_steps, I_arr,
            primary_compartment=primary_compartment,
            backend=backend,
            integrator=integrator,
            pressure_envelope=P_env,
            mechano_init=mechano_init,
        )
        counts[mask] = sc
        firing[mask] = sc > 0

    return {"firing_map": firing, "spike_count_map": counts, "tension_3d": tension_3d}


def simulate_map_3comp_with_spike_times(
    strain_3d: NDArray[np.floating],
    neuron: MulticompartmentMechanoHHNeuron,
    K_A: float = 0.25,
    coupling_factor: float = 1.0,
    dt: float = 0.001,
    duration: float = 50.0,
    I_stim: NDArray[np.floating] | float = 0.0,
    tension_threshold: float = 0.0,
    primary_compartment: int = 1,
    backend: str = "auto",
    integrator: str = "rk4",
    pressure_envelope: NDArray[np.floating] | float = 0.0,
    calcium_readout: bool | CalciumDynamics = False,
    ca_trace_dt_ms: float | None = None,
    mechano_init: str = "rest",
) -> dict[str, NDArray]:
    """Same as ``simulate_map_3comp`` but also returns per-firing-voxel
    spike times in CSR layout, suitable for raster / PSTH / latency
    rendering downstream.

    Calcium readout (optional)
    --------------------------
    ``calcium_readout=True`` (or a ``CalciumDynamics`` instance) attaches
    the single-pool Ca model as a *passive* readout when the neuron has no
    pool (e.g. scenario S2): with no KCa channels it has no path back to
    V, so all spike outputs are bit-identical to ``calcium_readout=False``.
    Whenever a pool is active (readout, or S4/S5 neurons) the result also
    carries

    * ``ca_peak_map``    float32 (3, nx, ny, nz) — per-compartment peak
      [Ca] [μM] over the window (dendrite, soma, AIS); NaN for voxels
      below ``tension_threshold`` (not simulated);
    * ``ca_peak_firing`` float32 (n_fire, 3), aligned with
      ``spike_voxel_idx``;
    * ``ca_passive``     bool — True when no KCa feedback is present;

    and with ``ca_trace_dt_ms`` set, decimated traces of the firing voxels
    (``ca_trace_firing`` float32 (n_fire, 3, n_tr), ``ca_trace_t_ms``),
    recomputed by a second kernel pass over the firing subset (per-voxel
    integration is independent, so the pass reproduces pass 1 exactly).
    RK4 Numba/CUDA fast path only.

    Two-pass strategy:
      1. Vectorised constant-tension batch over all above-threshold
         voxels for spike *counts* (the existing ``_simulate_batch``
         path — fast; one numba JIT compile, then linear in voxels).
      2. Per-voxel ``simulate()`` *only* on the firing subset to recover
         spike *times*. This subset is much smaller than the eligible
         set, so the loop overhead is bounded.

    The map outputs (``firing_map``, ``spike_count_map``, ``tension_3d``)
    match ``simulate_map_3comp`` bit-for-bit so downstream consumers
    that don't care about times keep working.

    Initial condition
    -----------------
    ``mechano_init`` (default ``"rest"``) is applied identically to the
    counts pass, the Ca-trace re-run and the per-voxel ``simulate()``
    fallback, so counts and spike times stay mutually consistent.
    ``"rest"`` (default) starts every voxel from true rest (zero-tension
    steady state of the mechano gates), as every theta burst does because
    the tension relaxes during the 180 ms OFF time (chained 200 ms periods
    with full state carry-over, ``scripts/run_burst_periodicity.py``);
    ``"peak_steady"`` reproduces the legacy canonical runs (gates at the
    peak-tension steady state; halle_dacc S2 13381 spikes / 3590 voxels
    vs 12250 / 3531 from rest). See
    :data:`ultraneuron.neuron.MECHANO_INIT_MODES`.
    """
    from .channels import strain_to_membrane_tension

    nx, ny, nz = strain_3d.shape
    tension_3d = strain_to_membrane_tension(strain_3d, K_A, coupling_factor)
    firing = np.zeros((nx, ny, nz), dtype=bool)
    counts = np.zeros((nx, ny, nz), dtype=np.int32)
    latency = np.full((nx, ny, nz), np.nan, dtype=np.float32)

    mask = tension_3d >= tension_threshold
    eligible_idx = np.argwhere(mask)
    eligible_T = np.asarray(tension_3d[mask], dtype=np.float64)

    # No-data short-circuit
    if eligible_T.size == 0:
        return {
            "firing_map": firing,
            "spike_count_map": counts,
            "latency_map": latency,
            "tension_3d": tension_3d,
            "spike_voxel_idx": np.empty((0, 3), dtype=np.int32),
            "spike_offsets": np.zeros(1, dtype=np.int64),
            "spike_times_ms": np.empty(0, dtype=np.float32),
            "T_pulse_ms": float(duration),
            "duration_ms": float(duration),
            "dt_ms": float(dt),
        }

    n_steps = int(duration / dt)
    if np.isscalar(I_stim):
        I_arr = np.array([0.0, float(I_stim), 0.0])
    else:
        I_arr = np.asarray(I_stim, dtype=np.float64)

    # ---- Pass 1: counts (and inline spike steps when on the fast path)
    # Set a per-voxel cap that comfortably covers the duration window at
    # the maximum biophysically plausible firing rate (~1 kHz).
    max_spikes_per_voxel = max(int(np.ceil(float(duration) * 1.0)) + 8, 32)
    # Fast-path eligibility: the JIT 3-comp kernel handles NICE with a
    # constant (scalar or per-voxel) pressure envelope — only fully
    # time-varying NICE envelopes fall out of scope.  The earlier blanket exclusion of
    # any neuron with a nice_model attached was unnecessarily conservative
    # and forced the slow per-voxel Pass-2 NumPy loop for NICE-enabled
    # scenarios.
    fast_eligible = _HAS_NUMBA_3COMP
    spike_steps_buf = None
    if fast_eligible:
        neuron._record_spike_steps = True
        neuron._max_spikes_per_voxel = max_spikes_per_voxel
    P_env = _per_voxel_pressure(pressure_envelope, mask)
    ca_ctx = _PassiveCalcium(neuron, calcium_readout)
    ca_ctx.__enter__()
    try:
        has_ca = neuron.calcium_dynamics is not None
        ca_passive = bool(has_ca and not neuron.kca_channels)
        try:
            sc = neuron._simulate_batch(
                eligible_T, dt, n_steps, I_arr,
                primary_compartment=primary_compartment,
                backend=backend,
                integrator=integrator,
                pressure_envelope=P_env,
                mechano_init=mechano_init,
            )
        finally:
            if fast_eligible:
                neuron._record_spike_steps = False
                spike_steps_buf = getattr(neuron, "_last_spike_step_buf", None)
                neuron._last_spike_step_buf = None
        ca_peak_elig = getattr(neuron, "_last_ca_peak", None) if has_ca else None
        neuron._last_ca_peak = None
        ca_trace_fire = None
        if has_ca and ca_trace_dt_ms is not None and np.any(sc > 0):
            decim = max(int(round(float(ca_trace_dt_ms) / dt)), 1)
            P_fire = P_env if np.isscalar(P_env) else np.asarray(P_env)[sc > 0]
            neuron._simulate_batch(
                eligible_T[sc > 0], dt, n_steps, I_arr,
                primary_compartment=primary_compartment,
                backend=backend, integrator="rk4",
                pressure_envelope=P_fire, ca_trace_decim=decim,
                mechano_init=mechano_init,
            )
            ca_trace_fire = neuron._last_ca_trace
            neuron._last_ca_trace = None
            neuron._last_ca_peak = None
    finally:
        ca_ctx.__exit__(None, None, None)
    counts[mask] = sc
    firing[mask] = sc > 0

    fire_sub = sc > 0
    fire_idx = eligible_idx[fire_sub]
    fire_T = eligible_T[fire_sub]
    spike_times_per: list[NDArray[np.floating]] = []

    if (spike_steps_buf is not None
            and isinstance(spike_steps_buf, np.ndarray)
            and spike_steps_buf.shape == (eligible_T.size,
                                          max_spikes_per_voxel)):
        # Fast path — Pass-2 collapses to an array unpack.
        eligible_fire_rows = np.where(fire_sub)[0]
        for k, row in enumerate(eligible_fire_rows):
            n_sp = int(sc[row])
            if n_sp <= 0:
                continue
            steps = spike_steps_buf[row, :n_sp]
            st_ms = (steps.astype(np.float64) * float(dt)).astype(np.float32)
            spike_times_per.append(st_ms)
            ix, iy, iz = (int(fire_idx[k, 0]), int(fire_idx[k, 1]),
                          int(fire_idx[k, 2]))
            latency[ix, iy, iz] = float(st_ms[0])
    else:
        # Fallback Pass 2: per-voxel re-run via NumPy single-voxel solver
        # for configurations the kernel can't handle (e.g. NICE
        # time-varying envelope).
        for k in range(len(fire_idx)):
            r = neuron.simulate(
                dt=dt, duration=duration,
                I_stim=I_stim,
                tension_trace=float(fire_T[k]),
                mechano_init=mechano_init,
            )
            st = np.asarray(
                r["spike_times_per_compartment"][primary_compartment],
                dtype=np.float32,
            )
            spike_times_per.append(st)
            ix, iy, iz = (int(fire_idx[k, 0]), int(fire_idx[k, 1]),
                          int(fire_idx[k, 2]))
            if len(st) > 0:
                latency[ix, iy, iz] = float(st[0])

    # CSR pack
    if spike_times_per:
        offsets = np.concatenate(
            [[0], np.cumsum([len(s) for s in spike_times_per])]
        ).astype(np.int64)
        spike_times_ms = np.concatenate(spike_times_per).astype(np.float32)
    else:
        offsets = np.zeros(1, dtype=np.int64)
        spike_times_ms = np.empty(0, dtype=np.float32)

    result = {
        "firing_map": firing,
        "spike_count_map": counts,
        "latency_map": latency,
        "tension_3d": tension_3d,
        "spike_voxel_idx": fire_idx.astype(np.int32),
        "spike_offsets": offsets,
        "spike_times_ms": spike_times_ms,
        "T_pulse_ms": float(duration),
        "duration_ms": float(duration),
        "dt_ms": float(dt),
    }
    if has_ca:
        if ca_peak_elig is None:
            raise NotImplementedError(
                "Ca readouts require the RK4 Numba/CUDA fast path.",
            )
        ca_map = np.full((3, nx, ny, nz), np.nan, dtype=np.float32)
        for c in range(3):
            ca_map[c][mask] = ca_peak_elig[:, c]
        result["ca_peak_map"] = ca_map
        result["ca_peak_firing"] = ca_peak_elig[fire_sub].astype(np.float32)
        result["ca_passive"] = ca_passive
        if ca_trace_fire is not None:
            result["ca_trace_firing"] = ca_trace_fire
            result["ca_trace_t_ms"] = (
                np.arange(1, ca_trace_fire.shape[2] + 1)
                * max(int(round(float(ca_trace_dt_ms) / dt)), 1) * float(dt)
            )
    return result


# ---------------------------------------------------------------------------
# Window chaining: state layout, tension drive profiles, voxel-level API
# ---------------------------------------------------------------------------

#: Visco-elastic relaxation time constant of the Halle/dACC canonical field
#: [ms]: single-exponential fit of |u_x| at the peak-ARF voxel over the
#: 180 ms OFF segment (``data/halle_dacc/validation_report/manifest.json``,
#: ``displacement_outputs.relax_tau_ms`` = 32.42 ms; halle_caudate 33.3 ms,
#: halle_dcn 31.5 ms).
TAU_RELAX_MS_HALLE_DACC = 32.42024559090537


def state_size(n_ch: int) -> int:
    """Number of per-voxel state variables for ``n_ch`` mechano channels."""
    return 15 + 6 * int(n_ch)


def state_layout(n_ch: int) -> dict[str, slice]:
    """Column slices of the (N, n_state) per-voxel state array.

    Order: ``V`` (dendrite, soma, AIS) [mV], HH gates ``m``, ``h``, ``n``
    (3 each), mechano activation ``w`` (3·n_ch, compartment-major:
    index ``c*n_ch + k``), mechano inactivation ``h_mech`` (3·n_ch, same
    order; 1.0 when inactivation is off) and ``Ca`` [μM] (3; 0 when no
    Ca pool is attached).
    """
    n_ch = int(n_ch)
    return {
        "V": slice(0, 3),
        "m": slice(3, 6),
        "h": slice(6, 9),
        "n": slice(9, 12),
        "w": slice(12, 12 + 3 * n_ch),
        "h_mech": slice(12 + 3 * n_ch, 12 + 6 * n_ch),
        "Ca": slice(12 + 6 * n_ch, 15 + 6 * n_ch),
    }


def _base_initial_state(neuron, N, calcium):
    """V = V_rest, HH gates at V_rest steady state, [Ca]; mechano cols unset."""
    n_ch = len(neuron.mechano_channels)
    lay = state_layout(n_ch)
    st = np.empty((N, state_size(n_ch)), dtype=np.float64)
    m0, h0, n0 = neuron._kinetics._steady_state_gates(neuron.V_rest)
    st[:, lay["V"]] = neuron.V_rest
    st[:, lay["m"]] = m0
    st[:, lay["h"]] = h0
    st[:, lay["n"]] = n0
    has_ca = (neuron.calcium_dynamics is not None) if calcium is None else bool(calcium)
    cd = neuron.calcium_dynamics or CalciumDynamics()
    st[:, lay["Ca"]] = cd.Ca_rest if has_ca else 0.0
    return st


def _fill_mechano(st, n_ch, w_inf, h_inf):
    """Write per-channel (n_ch, N) or (n_ch,) w/h_mech values, all compartments."""
    for c in range(3):
        for k in range(n_ch):
            st[:, 12 + c * n_ch + k] = w_inf[k]
            st[:, 12 + 3 * n_ch + c * n_ch + k] = h_inf[k]


def rest_initial_state(
    neuron: MulticompartmentMechanoHHNeuron,
    tensions: NDArray[np.floating] | int,
    calcium: bool | None = None,
) -> NDArray[np.float64]:
    """Explicit (N, n_state) form of the default true-rest initial state.

    V = V_rest, HH gates at their V_rest steady state, mechano gates
    (activation ``w`` and inactivation ``h_mech``) at their steady state
    for ZERO tension, [Ca] = Ca_rest. This is ``mechano_init="rest"``,
    the default of every simulate path: every burst of the theta-burst
    protocol starts from rest because the tension relaxes during the
    180 ms OFF time (``scripts/run_burst_periodicity.py``). Passing it as
    ``initial_state`` reproduces ``initial_state=None`` bit-for-bit.

    ``tensions`` only sets N (an int is accepted). ``calcium`` overrides
    whether a Ca pool is assumed (default: whether
    ``neuron.calcium_dynamics`` is set).

    Note: this analytic rest state differs from the state reached by
    integrating a voxel at zero tension for 2 s (``B_rest_state`` in
    ``burst_periodicity.npz``) by ~2 µV in V and ~1e-6 in the HH gates
    (the nominal V_rest is not the exact fixed point of the 3-comp
    neuron); mechano gates agree exactly.
    """
    N = int(tensions) if np.isscalar(tensions) else int(np.asarray(tensions).size)
    n_ch = len(neuron.mechano_channels)
    st = _base_initial_state(neuron, N, calcium)
    w_rest, h_rest = neuron._rest_mechano_gates()
    _fill_mechano(st, n_ch, w_rest, h_rest)
    return st


def peak_steady_initial_state(
    neuron: MulticompartmentMechanoHHNeuron,
    tensions: NDArray[np.floating],
    calcium: bool | None = None,
) -> NDArray[np.float64]:
    """Explicit (N, n_state) form of the legacy ``"peak_steady"`` init.

    V = V_rest, HH gates at their V_rest steady state, mechano gates at
    steady state for ``tensions`` (the original canonical runs started
    the ON window with w = w_inf(T_peak), i.e. channels already open),
    [Ca] = Ca_rest. Passing this as ``initial_state`` reproduces
    ``mechano_init="peak_steady"`` bit-for-bit (constant tension).
    ``calcium`` overrides whether a Ca pool is assumed (default: whether
    ``neuron.calcium_dynamics`` is set).
    """
    T = np.asarray(tensions, dtype=np.float64).reshape(-1)
    n_ch = len(neuron.mechano_channels)
    st = _base_initial_state(neuron, T.size, calcium)
    _, w_inf, _, _ = neuron._channel_kinetics(T, n_ch)
    h_inf, _, _ = neuron._inactivation_kinetics(T, n_ch)
    _fill_mechano(st, n_ch, w_inf, h_inf)
    return st


def canonical_initial_state(
    neuron: MulticompartmentMechanoHHNeuron,
    tensions: NDArray[np.floating],
    calcium: bool | None = None,
) -> NDArray[np.float64]:
    """Deprecated alias of :func:`peak_steady_initial_state`.

    The canonical initial condition is now true rest
    (:func:`rest_initial_state`); this name kept the peak-tension
    steady state and is retained only for backward compatibility.
    """
    import warnings as _warn  # noqa: PLC0415
    _warn.warn(
        "canonical_initial_state is deprecated; use peak_steady_initial_state "
        "(legacy, mechano gates at peak-tension steady state) or "
        "rest_initial_state (the default initial condition).",
        DeprecationWarning, stacklevel=2,
    )
    return peak_steady_initial_state(neuron, tensions, calcium=calcium)


def tension_drive(
    dt: float,
    t_on_ms: float = 20.0,
    t_off_ms: float = 180.0,
    n_periods: int = 1,
    off_mode: str = "exp",
    tau_relax_ms: float = TAU_RELAX_MS_HALLE_DACC,
) -> NDArray[np.float64]:
    """Per-step tension multiplier for a pulsed ON/OFF protocol.

    Step ``i`` (time ``t_i = i·dt``, held over the step as in
    ``MulticompartmentMechanoHHNeuron.simulate``) has multiplier

    * ON  (``t_mod < t_on``): 1.0 — constant per-voxel peak tension, as in
      the canonical single-window runs;
    * OFF, ``off_mode="exp"`` (default): ``exp(-(t_mod - t_on)/tau_relax)``
      — the tissue strain (hence membrane tension, which is linear in
      strain) relaxes with the visco-elastic time constant fitted to the
      shear-FDTD displacement on the OFF segment (32.4 ms for Halle/dACC);
    * OFF, ``off_mode="step"``: 0.0 — instantaneous release.

    ``t_mod = t mod (t_on + t_off)``. The ON segment is a flat top (the
    viscoelastic rise during ON is not modelled, matching the canonical
    constant-tension convention). Returns an array of
    ``n_periods · round((t_on + t_off)/dt)`` float64 values.
    """
    if off_mode not in ("exp", "step"):
        raise ValueError(f"off_mode must be 'exp' or 'step', got {off_mode!r}")
    n_on = int(round(t_on_ms / dt))
    n_per = int(round((t_on_ms + t_off_ms) / dt))
    i_mod = np.arange(n_per, dtype=np.int64)
    f = np.ones(n_per, dtype=np.float64)
    off = i_mod >= n_on
    if off_mode == "exp":
        f[off] = np.exp(-(i_mod[off] - n_on) * float(dt) / float(tau_relax_ms))
    else:
        f[off] = 0.0
    return np.tile(f, int(n_periods))


class _PassiveCalcium:
    """Context manager: temporarily attach a Ca pool as a passive readout.

    Attaches ``CalciumDynamics`` (default parameters, or the instance
    given) when the neuron has none. Without KCa channels the pool has no
    path back to V, so spike output is unchanged (bit-identical: the
    kernels take the pool parameters as runtime arguments only).
    """

    def __init__(self, neuron, calcium_readout):
        self.neuron = neuron
        self.req = calcium_readout
        self.attached = False

    def __enter__(self):
        if self.req and self.neuron.calcium_dynamics is None:
            cd = (self.req if isinstance(self.req, CalciumDynamics)
                  else CalciumDynamics())
            self.neuron.calcium_dynamics = cd
            self.attached = True
        return self

    def __exit__(self, *exc):
        if self.attached:
            self.neuron.calcium_dynamics = None
        return False


def simulate_voxels_3comp(
    tensions: NDArray[np.floating],
    neuron: MulticompartmentMechanoHHNeuron,
    dt: float = 0.001,
    duration: float = 20.0,
    I_stim: NDArray[np.floating] | float = 0.0,
    primary_compartment: int = 1,
    backend: str = "auto",
    drive: NDArray[np.floating] | None = None,
    initial_state: NDArray[np.floating] | None = None,
    return_state: bool = True,
    record_spike_times: bool = True,
    max_spikes_per_voxel: int | None = None,
    calcium_readout: bool | CalciumDynamics = False,
    ca_trace_dt_ms: float | None = None,
    pressure_envelope: NDArray[np.floating] | float = 0.0,
    mechano_init: str = "rest",
) -> dict:
    """Integrate one window for N voxels with optional state carry-over.

    Voxel-level companion of :func:`simulate_map_3comp_with_spike_times`
    for chaining successive windows (e.g. theta-burst ON/OFF periods):
    feed ``out["final_state"]`` of one call as ``initial_state`` of the
    next. Chaining is exact: two consecutive windows reproduce one long
    window bit-for-bit (spike step indices are window-relative).

    Parameters
    ----------
    tensions : (N,) peak membrane tension per voxel [mN/m].
    drive : (n_steps,) tension multiplier, see :func:`tension_drive`;
        ``None`` = constant tension. ``n_steps = int(duration/dt)``.
    initial_state : (N, n_state) start state (:func:`state_layout`);
        ``None`` = init per ``mechano_init``.
    mechano_init : {"rest", "peak_steady"}
        Used when ``initial_state`` is None. ``"rest"`` (default): true
        rest, mechano gates at the zero-tension steady state
        (:func:`rest_initial_state`) — every theta burst starts from rest
        because the tension relaxes in the OFF time. ``"peak_steady"``:
        mechano gates at steady state for the step-0 tension
        (:func:`peak_steady_initial_state`, legacy canonical).
    calcium_readout : attach a ``CalciumDynamics`` pool (default params,
        or the instance given) as a passive readout when the neuron has
        none. With no KCa channels it cannot affect V.
    ca_trace_dt_ms : sample interval of the [Ca] trace output (rounded to
        a whole number of steps); ``None`` = no traces.

    Returns
    -------
    dict with ``spike_counts`` (N,), ``spike_offsets`` (N+1,) and
    ``spike_times_ms`` (CSR, window-relative, float64) when
    ``record_spike_times``; ``final_state`` (N, n_state) when
    ``return_state``; ``ca_peak_uM`` (N, 3) when a Ca pool is active;
    ``ca_trace_uM`` (N, 3, n_tr) float32 + ``ca_trace_t_ms`` when traces
    are requested; plus ``n_steps``, ``dt_ms``, ``backend``.
    """
    from . import cuda_backend as _cuda  # noqa: PLC0415

    T = np.ascontiguousarray(tensions, dtype=np.float64).reshape(-1)
    N = T.size
    n_steps = int(duration / dt)
    if np.isscalar(I_stim):
        I_arr = np.array([0.0, float(I_stim), 0.0])
    else:
        I_arr = np.asarray(I_stim, dtype=np.float64).reshape(3)
    if max_spikes_per_voxel is None:
        max_spikes_per_voxel = max(int(np.ceil(float(duration) * 1.0)) + 8, 32)
    decim = 0
    if ca_trace_dt_ms is not None:
        decim = max(int(round(float(ca_trace_dt_ms) / dt)), 1)

    out: dict = {"n_steps": n_steps, "dt_ms": float(dt),
                 "backend": _cuda.resolve_backend(backend)}
    with _PassiveCalcium(neuron, calcium_readout):
        has_ca = neuron.calcium_dynamics is not None
        neuron._record_spike_steps = bool(record_spike_times)
        neuron._max_spikes_per_voxel = int(max_spikes_per_voxel)
        try:
            sc = neuron._simulate_batch(
                T, dt, n_steps, I_arr,
                primary_compartment=primary_compartment,
                pressure_envelope=pressure_envelope,
                backend=backend, integrator="rk4",
                drive=drive, initial_state=initial_state,
                return_state=return_state, ca_trace_decim=decim,
                mechano_init=mechano_init,
            )
        finally:
            neuron._record_spike_steps = False
            buf = getattr(neuron, "_last_spike_step_buf", None)
            neuron._last_spike_step_buf = None
        out["ca_passive"] = bool(has_ca and not neuron.kca_channels)
    sc = np.asarray(sc, dtype=np.int32)
    out["spike_counts"] = sc
    if record_spike_times and buf is not None and buf.shape == (N, max_spikes_per_voxel):
        if int(sc.max(initial=0)) > max_spikes_per_voxel:
            import warnings as _warn  # noqa: PLC0415
            _warn.warn(
                f"spike count exceeded max_spikes_per_voxel={max_spikes_per_voxel}; "
                "spike times truncated.", RuntimeWarning, stacklevel=2,
            )
        n_rec = np.minimum(sc, max_spikes_per_voxel)
        offsets = np.concatenate([[0], np.cumsum(n_rec)]).astype(np.int64)
        keep = np.arange(max_spikes_per_voxel)[None, :] < n_rec[:, None]
        out["spike_offsets"] = offsets
        out["spike_times_ms"] = buf[keep].astype(np.float64) * float(dt)
    if return_state:
        out["final_state"] = neuron._last_final_state
    if has_ca:
        out["ca_peak_uM"] = neuron._last_ca_peak
        if decim > 0:
            out["ca_trace_uM"] = neuron._last_ca_trace
            out["ca_trace_t_ms"] = (np.arange(1, n_steps // decim + 1)
                                    * decim * float(dt))
    neuron._last_final_state = None
    neuron._last_ca_peak = None
    neuron._last_ca_trace = None
    return out
