"""CUDA backend for the batch HH simulation kernels.

Drop-in replacements for the @njit(parallel=True) CPU kernels in
``neuron.py`` (single-compartment) and ``multicompartment.py`` (three
-compartment). One CUDA thread integrates one voxel through the full
RK4 timestep loop, mirroring the per-voxel ``prange`` of the CPU path.

Lazy-loaded: importing this module does not require a working CUDA
toolchain. Call :func:`is_cuda_available` before dispatching; the
public ``simulate_*`` APIs fall back to the Numba CPU path with a
warning when CUDA isn't usable.

Numerical match is within fastmath rounding (~1e-4 relative) of the
CPU kernels. The serial timestep loop and RK4 stage logic are
structurally identical.
"""
from __future__ import annotations

import math
import warnings

import numpy as np

# Maximum mechanosensitive channels per neuron supported by the CUDA
# kernel. Per-voxel scratch state lives in CUDA local arrays sized to
# these literals at compile time. Typical pipelines (manuscript
# 3-comp Piezo1+TREK1) use n_ch=2; raise if you wire in additional
# channels and the host-side eligibility check below will catch it.
_MAX_CH = 4
_MAX_3CH = 3 * _MAX_CH

_HAS_CUDA: bool | None = None
_KERNELS: dict[str, object] = {}


def is_cuda_available() -> bool:
    """Return True iff numba.cuda is importable and a device is visible."""
    global _HAS_CUDA
    if _HAS_CUDA is not None:
        return _HAS_CUDA
    try:
        from numba import cuda  # noqa: PLC0415
        _HAS_CUDA = bool(cuda.is_available())
    except Exception:  # noqa: BLE001
        _HAS_CUDA = False
    return _HAS_CUDA


def _ensure_kernels() -> dict[str, object]:
    """Compile (once) and return the CUDA kernel registry."""
    if _KERNELS:
        return _KERNELS
    if not is_cuda_available():
        msg = "CUDA backend requested but numba.cuda is not available."
        raise RuntimeError(msg)
    _build_kernels()
    return _KERNELS


def _build_kernels() -> None:
    """Define the @cuda.jit kernels. Module-scope import of numba.cuda."""
    from numba import cuda  # noqa: PLC0415
    from numba import float64  # noqa: PLC0415

    MAX_CH = _MAX_CH
    MAX_3CH = _MAX_3CH

    # -------------------------------------------------------------- #
    # Device helpers
    # -------------------------------------------------------------- #
    @cuda.jit(device=True, inline=True)
    def _hh_rates_cortical(V, VT):
        dVm = V - VT - 13.0
        if abs(dVm) > 1e-7:
            am = -0.32 * dVm / (math.exp(-dVm / 4.0) - 1.0)
        else:
            am = 0.32 * 4.0
        dVm2 = V - VT - 40.0
        if abs(dVm2) > 1e-7:
            bm = 0.28 * dVm2 / (math.exp(dVm2 / 5.0) - 1.0)
        else:
            bm = 0.28 * 5.0
        ah = 0.128 * math.exp(-(V - VT - 17.0) / 18.0)
        bh = 4.0 / (1.0 + math.exp(-(V - VT - 40.0) / 5.0))
        dVn = V - VT - 15.0
        if abs(dVn) > 1e-7:
            an = -0.032 * dVn / (math.exp(-dVn / 5.0) - 1.0)
        else:
            an = 0.032 * 5.0
        bn = 0.5 * math.exp(-(V - VT - 10.0) / 40.0)
        return am, bm, ah, bh, an, bn

    @cuda.jit(device=True, inline=True)
    def _hh_rates_squid(V):
        dVm = V + 40.0
        if abs(dVm) > 1e-7:
            am = 0.1 * dVm / (1.0 - math.exp(-dVm / 10.0))
        else:
            am = 1.0
        bm = 4.0 * math.exp(-(V + 65.0) / 18.0)
        ah = 0.07 * math.exp(-(V + 65.0) / 20.0)
        bh = 1.0 / (1.0 + math.exp(-(V + 35.0) / 10.0))
        dVn = V + 55.0
        if abs(dVn) > 1e-7:
            an = 0.01 * dVn / (1.0 - math.exp(-dVn / 10.0))
        else:
            an = 0.1
        bn = 0.125 * math.exp(-(V + 65.0) / 80.0)
        return am, bm, ah, bh, an, bn

    # -------------------------------------------------------------- #
    # 3-compartment kernel (mirrors _rk4_batch_numba_3comp)
    # -------------------------------------------------------------- #
    @cuda.jit
    def rk4_batch_3comp_kernel(
        # Per-voxel mechano steady states
        w_inf_per_vox, tau_inv_per_vox,
        h_inf_per_vox, h_tau_inv_per_vox,
        # Per-compartment scalars (size 3)
        g_Na, g_K, g_L,
        E_Na, E_K, E_L,
        # Per-voxel effective capacitance
        C_eff_per_vox,
        # Coupling
        g_ds, g_sa, AsAd, AsAa,
        # Mechano (size (3, n_ch) flattened row-major, n_ch)
        g_mech, n_ch_int, E_revs, Ca_frac,
        # Calcium
        flux_factor, tau_decay_inv, Ca_rest,
        # KCa
        kca_g, kca_E_K, kca_half, kca_n, kca_mask, n_kca_int,
        # TRP
        trp_g_Po, trp_E_rev, trp_mask, n_trp_int,
        # HH branching + Q10
        is_cortical_int, VT, phi,
        # Stim
        I0, I1, I2,
        # Init
        V_rest, m0_init, h0_init, n0_init,
        # Time
        dt, n_steps, primary_compartment_int,
        # Spike step recording
        spike_step_buf, max_spikes,
        # Outputs
        spike_counts,
    ):
        idx = cuda.grid(1)
        N = spike_counts.shape[0]
        if idx >= N:
            return

        n_ch = n_ch_int
        n_kca = n_kca_int
        n_trp = n_trp_int
        is_cortical = is_cortical_int != 0
        primary_compartment = primary_compartment_int

        # Per-voxel cached steady-state arrays
        w_inf_local = cuda.local.array(MAX_CH, dtype=float64)
        tau_inv_local = cuda.local.array(MAX_CH, dtype=float64)
        h_inf_local = cuda.local.array(MAX_CH, dtype=float64)
        h_tau_inv_local = cuda.local.array(MAX_CH, dtype=float64)
        for k in range(n_ch):
            w_inf_local[k] = w_inf_per_vox[k, idx]
            tau_inv_local[k] = tau_inv_per_vox[k, idx]
            h_inf_local[k] = h_inf_per_vox[k, idx]
            h_tau_inv_local[k] = h_tau_inv_per_vox[k, idx]

        # State (3 compartments × per-channel gates flattened)
        w = cuda.local.array(MAX_3CH, dtype=float64)
        h_mech = cuda.local.array(MAX_3CH, dtype=float64)
        k1w = cuda.local.array(MAX_3CH, dtype=float64)
        k2w = cuda.local.array(MAX_3CH, dtype=float64)
        k3w = cuda.local.array(MAX_3CH, dtype=float64)
        k4w = cuda.local.array(MAX_3CH, dtype=float64)
        w_tmp = cuda.local.array(MAX_3CH, dtype=float64)
        k1hm = cuda.local.array(MAX_3CH, dtype=float64)
        k2hm = cuda.local.array(MAX_3CH, dtype=float64)
        k3hm = cuda.local.array(MAX_3CH, dtype=float64)
        k4hm = cuda.local.array(MAX_3CH, dtype=float64)
        hm_tmp = cuda.local.array(MAX_3CH, dtype=float64)

        for c in range(3):
            for k in range(n_ch):
                j = c * n_ch + k
                w[j] = w_inf_local[k]
                h_mech[j] = h_inf_local[k]

        V0 = V_rest; V1 = V_rest; V2 = V_rest
        m0 = m0_init; m1 = m0_init; m2 = m0_init
        h0 = h0_init; h1 = h0_init; h2 = h0_init
        n0 = n0_init; n1 = n0_init; n2 = n0_init
        Ca_a = Ca_rest; Ca_b = Ca_rest; Ca_c = Ca_rest

        Cm0 = C_eff_per_vox[0, idx]
        Cm1 = C_eff_per_vox[1, idx]
        Cm2 = C_eff_per_vox[2, idx]

        gNa0 = g_Na[0]; gNa1 = g_Na[1]; gNa2 = g_Na[2]
        gK0 = g_K[0]; gK1 = g_K[1]; gK2 = g_K[2]
        gL0 = g_L[0]; gL1 = g_L[1]; gL2 = g_L[2]

        spike_count_local = 0

        for step in range(n_steps):
            # Spike-detection V_prev at primary compartment
            if primary_compartment == 0:
                V_prev = V0
            elif primary_compartment == 1:
                V_prev = V1
            else:
                V_prev = V2

            # ----- k1 -----
            (k1V0, k1V1, k1V2,
             k1m0, k1m1, k1m2,
             k1h0, k1h1, k1h2,
             k1n0, k1n1, k1n2,
             k1Ca0, k1Ca1, k1Ca2) = _derivs_3comp(
                V0, V1, V2, m0, m1, m2, h0, h1, h2, n0, n1, n2,
                w, h_mech, Ca_a, Ca_b, Ca_c, n_ch,
                gNa0, gNa1, gNa2, gK0, gK1, gK2, gL0, gL1, gL2,
                E_Na, E_K, E_L, Cm0, Cm1, Cm2,
                g_ds, g_sa, AsAd, AsAa,
                g_mech, E_revs, Ca_frac,
                flux_factor, tau_decay_inv, Ca_rest,
                n_kca, kca_g, kca_E_K, kca_half, kca_n, kca_mask,
                n_trp, trp_g_Po, trp_E_rev, trp_mask,
                I0, I1, I2, phi, is_cortical, VT,
            )
            for j in range(3 * n_ch):
                c_idx = j // n_ch
                k_idx = j - c_idx * n_ch
                k1w[j] = (w_inf_local[k_idx] - w[j]) * tau_inv_local[k_idx]
                k1hm[j] = (h_inf_local[k_idx] - h_mech[j]) * h_tau_inv_local[k_idx]

            # ----- k2 -----
            for j in range(3 * n_ch):
                w_tmp[j] = w[j] + 0.5 * dt * k1w[j]
                hm_tmp[j] = h_mech[j] + 0.5 * dt * k1hm[j]
            (k2V0, k2V1, k2V2,
             k2m0, k2m1, k2m2,
             k2h0, k2h1, k2h2,
             k2n0, k2n1, k2n2,
             k2Ca0, k2Ca1, k2Ca2) = _derivs_3comp(
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
                gNa0, gNa1, gNa2, gK0, gK1, gK2, gL0, gL1, gL2,
                E_Na, E_K, E_L, Cm0, Cm1, Cm2,
                g_ds, g_sa, AsAd, AsAa,
                g_mech, E_revs, Ca_frac,
                flux_factor, tau_decay_inv, Ca_rest,
                n_kca, kca_g, kca_E_K, kca_half, kca_n, kca_mask,
                n_trp, trp_g_Po, trp_E_rev, trp_mask,
                I0, I1, I2, phi, is_cortical, VT,
            )
            for j in range(3 * n_ch):
                c_idx = j // n_ch
                k_idx = j - c_idx * n_ch
                k2w[j] = (w_inf_local[k_idx] - w_tmp[j]) * tau_inv_local[k_idx]
                k2hm[j] = (h_inf_local[k_idx] - hm_tmp[j]) * h_tau_inv_local[k_idx]

            # ----- k3 -----
            for j in range(3 * n_ch):
                w_tmp[j] = w[j] + 0.5 * dt * k2w[j]
                hm_tmp[j] = h_mech[j] + 0.5 * dt * k2hm[j]
            (k3V0, k3V1, k3V2,
             k3m0, k3m1, k3m2,
             k3h0, k3h1, k3h2,
             k3n0, k3n1, k3n2,
             k3Ca0, k3Ca1, k3Ca2) = _derivs_3comp(
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
                gNa0, gNa1, gNa2, gK0, gK1, gK2, gL0, gL1, gL2,
                E_Na, E_K, E_L, Cm0, Cm1, Cm2,
                g_ds, g_sa, AsAd, AsAa,
                g_mech, E_revs, Ca_frac,
                flux_factor, tau_decay_inv, Ca_rest,
                n_kca, kca_g, kca_E_K, kca_half, kca_n, kca_mask,
                n_trp, trp_g_Po, trp_E_rev, trp_mask,
                I0, I1, I2, phi, is_cortical, VT,
            )
            for j in range(3 * n_ch):
                c_idx = j // n_ch
                k_idx = j - c_idx * n_ch
                k3w[j] = (w_inf_local[k_idx] - w_tmp[j]) * tau_inv_local[k_idx]
                k3hm[j] = (h_inf_local[k_idx] - hm_tmp[j]) * h_tau_inv_local[k_idx]

            # ----- k4 -----
            for j in range(3 * n_ch):
                w_tmp[j] = w[j] + dt * k3w[j]
                hm_tmp[j] = h_mech[j] + dt * k3hm[j]
            (k4V0, k4V1, k4V2,
             k4m0, k4m1, k4m2,
             k4h0, k4h1, k4h2,
             k4n0, k4n1, k4n2,
             k4Ca0, k4Ca1, k4Ca2) = _derivs_3comp(
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
                gNa0, gNa1, gNa2, gK0, gK1, gK2, gL0, gL1, gL2,
                E_Na, E_K, E_L, Cm0, Cm1, Cm2,
                g_ds, g_sa, AsAd, AsAa,
                g_mech, E_revs, Ca_frac,
                flux_factor, tau_decay_inv, Ca_rest,
                n_kca, kca_g, kca_E_K, kca_half, kca_n, kca_mask,
                n_trp, trp_g_Po, trp_E_rev, trp_mask,
                I0, I1, I2, phi, is_cortical, VT,
            )
            for j in range(3 * n_ch):
                c_idx = j // n_ch
                k_idx = j - c_idx * n_ch
                k4w[j] = (w_inf_local[k_idx] - w_tmp[j]) * tau_inv_local[k_idx]
                k4hm[j] = (h_inf_local[k_idx] - hm_tmp[j]) * h_tau_inv_local[k_idx]

            # ----- update -----
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
                wj = w[j] + dt * avg
                if wj < 0.0:
                    wj = 0.0
                elif wj > 1.0:
                    wj = 1.0
                w[j] = wj
                avg_hm = (k1hm[j] + 2.0 * k2hm[j] + 2.0 * k3hm[j] + k4hm[j]) / 6.0
                hmj = h_mech[j] + dt * avg_hm
                if hmj < 0.0:
                    hmj = 0.0
                elif hmj > 1.0:
                    hmj = 1.0
                h_mech[j] = hmj

            Ca_a = Ca_a + s * (k1Ca0 + 2.0 * k2Ca0 + 2.0 * k3Ca0 + k4Ca0)
            Ca_b = Ca_b + s * (k1Ca1 + 2.0 * k2Ca1 + 2.0 * k3Ca1 + k4Ca1)
            Ca_c = Ca_c + s * (k1Ca2 + 2.0 * k2Ca2 + 2.0 * k3Ca2 + k4Ca2)
            if Ca_a < 0.0: Ca_a = 0.0
            if Ca_b < 0.0: Ca_b = 0.0
            if Ca_c < 0.0: Ca_c = 0.0

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

            if primary_compartment == 0:
                V_now = V0
            elif primary_compartment == 1:
                V_now = V1
            else:
                V_now = V2
            if V_now > 0.0 and V_prev <= 0.0:
                prev_count = spike_count_local
                if max_spikes > 0 and prev_count < max_spikes:
                    spike_step_buf[idx, prev_count] = step + 1
                spike_count_local = prev_count + 1

        spike_counts[idx] = spike_count_local

    @cuda.jit(device=True, inline=True)
    def _derivs_3comp(
        V0, V1, V2, m0, m1, m2, h0, h1, h2, n0, n1, n2,
        w_flat, h_mech_flat, Ca0, Ca1, Ca2,
        n_ch,
        g_Na0, g_Na1, g_Na2,
        g_K0, g_K1, g_K2,
        g_L0, g_L1, g_L2,
        E_Na, E_K, E_L,
        C_m0, C_m1, C_m2,
        g_ds, g_sa, AsAd, AsAa,
        g_mech, E_revs, Ca_frac,
        flux_factor, tau_decay_inv, Ca_rest,
        n_kca, kca_g, kca_E_K, kca_half, kca_n, kca_mask,
        n_trp, trp_g_Po, trp_E_rev, trp_mask,
        I0, I1, I2,
        phi, is_cortical, VT,
    ):
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

        I_Na0 = g_Na0 * (m0 * m0 * m0) * h0 * (V0 - E_Na)
        I_K0 = g_K0 * (n0 * n0 * n0 * n0) * (V0 - E_K)
        I_L0 = g_L0 * (V0 - E_L)
        I_Na1 = g_Na1 * (m1 * m1 * m1) * h1 * (V1 - E_Na)
        I_K1 = g_K1 * (n1 * n1 * n1 * n1) * (V1 - E_K)
        I_L1 = g_L1 * (V1 - E_L)
        I_Na2 = g_Na2 * (m2 * m2 * m2) * h2 * (V2 - E_Na)
        I_K2 = g_K2 * (n2 * n2 * n2 * n2) * (V2 - E_K)
        I_L2 = g_L2 * (V2 - E_L)

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

        I_ax0 = g_ds * AsAd * (V1 - V0)
        I_ax1 = g_ds * (V0 - V1) + g_sa * (V2 - V1)
        I_ax2 = g_sa * AsAa * (V1 - V2)

        dV0 = (-I_Na0 - I_K0 - I_L0 + I_mech0 + I_KCa0 + I_TRP0 + I_ax0 + I0) / C_m0
        dV1 = (-I_Na1 - I_K1 - I_L1 + I_mech1 + I_KCa1 + I_TRP1 + I_ax1 + I1) / C_m1
        dV2 = (-I_Na2 - I_K2 - I_L2 + I_mech2 + I_KCa2 + I_TRP2 + I_ax2 + I2) / C_m2

        dCa0 = flux_factor * I_Ca0 - (Ca0 - Ca_rest) * tau_decay_inv
        dCa1 = flux_factor * I_Ca1 - (Ca1 - Ca_rest) * tau_decay_inv
        dCa2 = flux_factor * I_Ca2 - (Ca2 - Ca_rest) * tau_decay_inv

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

    # -------------------------------------------------------------- #
    # Single-compartment kernels (mirror _rk4_batch_numba / _tv)
    # -------------------------------------------------------------- #
    @cuda.jit(device=True, inline=True)
    def _derivs_single(
        V, m, h, n, w_gates, phi, is_cortical, VT,
        g_Na, g_K, g_L, E_Na, E_K, E_L, C_m,
        I_stim, n_ch, ch_g_dens, ch_E_revs,
    ):
        if is_cortical:
            am, bm, ah, bh, an, bn = _hh_rates_cortical(V, VT)
        else:
            am, bm, ah, bh, an, bn = _hh_rates_squid(V)
        am *= phi; bm *= phi; ah *= phi; bh *= phi; an *= phi; bn *= phi

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

    @cuda.jit
    def rk4_batch_kernel(
        V_rest, g_Na, g_K, g_L, E_Na, E_K, E_L, C_m,
        is_cortical_int, VT,
        tensions, phi_arr, dt, n_steps, I_stim,
        ch_g_dens, ch_E_revs, ch_w_inf, ch_tau_inv,
        m0_init, h0_init, n0_init,
        spike_counts,
    ):
        idx = cuda.grid(1)
        N = spike_counts.shape[0]
        if idx >= N:
            return
        is_cortical = is_cortical_int != 0
        n_ch = ch_g_dens.shape[0]

        V = V_rest
        m = m0_init
        h = h0_init
        n = n0_init
        phi = phi_arr[idx]

        w = cuda.local.array(MAX_CH, dtype=float64)
        w_inf_i = cuda.local.array(MAX_CH, dtype=float64)
        tau_inv_i = cuda.local.array(MAX_CH, dtype=float64)
        k1w = cuda.local.array(MAX_CH, dtype=float64)
        k2w = cuda.local.array(MAX_CH, dtype=float64)
        k3w = cuda.local.array(MAX_CH, dtype=float64)
        k4w = cuda.local.array(MAX_CH, dtype=float64)
        wtmp = cuda.local.array(MAX_CH, dtype=float64)

        for k in range(n_ch):
            w_inf_i[k] = ch_w_inf[k, idx]
            tau_inv_i[k] = ch_tau_inv[k, idx]
            w[k] = w_inf_i[k]

        spike_count = 0
        for step in range(n_steps):  # noqa: B007
            V_prev = V

            # k1
            k1V, k1m, k1h, k1n = _derivs_single(
                V, m, h, n, w, phi, is_cortical, VT,
                g_Na, g_K, g_L, E_Na, E_K, E_L, C_m,
                I_stim, n_ch, ch_g_dens, ch_E_revs,
            )
            for k in range(n_ch):
                k1w[k] = (w_inf_i[k] - w[k]) * tau_inv_i[k]

            # k2
            for k in range(n_ch):
                wtmp[k] = w[k] + 0.5 * dt * k1w[k]
            k2V, k2m, k2h, k2n = _derivs_single(
                V + 0.5 * dt * k1V, m + 0.5 * dt * k1m,
                h + 0.5 * dt * k1h, n + 0.5 * dt * k1n,
                wtmp, phi, is_cortical, VT,
                g_Na, g_K, g_L, E_Na, E_K, E_L, C_m,
                I_stim, n_ch, ch_g_dens, ch_E_revs,
            )
            for k in range(n_ch):
                k2w[k] = (w_inf_i[k] - wtmp[k]) * tau_inv_i[k]

            # k3
            for k in range(n_ch):
                wtmp[k] = w[k] + 0.5 * dt * k2w[k]
            k3V, k3m, k3h, k3n = _derivs_single(
                V + 0.5 * dt * k2V, m + 0.5 * dt * k2m,
                h + 0.5 * dt * k2h, n + 0.5 * dt * k2n,
                wtmp, phi, is_cortical, VT,
                g_Na, g_K, g_L, E_Na, E_K, E_L, C_m,
                I_stim, n_ch, ch_g_dens, ch_E_revs,
            )
            for k in range(n_ch):
                k3w[k] = (w_inf_i[k] - wtmp[k]) * tau_inv_i[k]

            # k4
            for k in range(n_ch):
                wtmp[k] = w[k] + dt * k3w[k]
            k4V, k4m, k4h, k4n = _derivs_single(
                V + dt * k3V, m + dt * k3m,
                h + dt * k3h, n + dt * k3n,
                wtmp, phi, is_cortical, VT,
                g_Na, g_K, g_L, E_Na, E_K, E_L, C_m,
                I_stim, n_ch, ch_g_dens, ch_E_revs,
            )
            for k in range(n_ch):
                k4w[k] = (w_inf_i[k] - wtmp[k]) * tau_inv_i[k]

            s = dt / 6.0
            V = V + s * (k1V + 2.0 * k2V + 2.0 * k3V + k4V)
            m = m + s * (k1m + 2.0 * k2m + 2.0 * k3m + k4m)
            h = h + s * (k1h + 2.0 * k2h + 2.0 * k3h + k4h)
            n = n + s * (k1n + 2.0 * k2n + 2.0 * k3n + k4n)
            for k in range(n_ch):
                dw_avg = (k1w[k] + 2.0 * k2w[k] + 2.0 * k3w[k] + k4w[k]) / 6.0
                w[k] = w[k] + dt * dw_avg

            if m < 0.0: m = 0.0
            elif m > 1.0: m = 1.0
            if h < 0.0: h = 0.0
            elif h > 1.0: h = 1.0
            if n < 0.0: n = 0.0
            elif n > 1.0: n = 1.0
            for k in range(n_ch):
                wk = w[k]
                if wk < 0.0: wk = 0.0
                elif wk > 1.0: wk = 1.0
                w[k] = wk

            if V > 0.0 and V_prev <= 0.0:
                spike_count += 1

        spike_counts[idx] = spike_count

    @cuda.jit
    def rk4_batch_tv_kernel(
        V_rest, g_Na, g_K, g_L, E_Na, E_K, E_L, C_m,
        is_cortical_int, VT,
        tensions_2d, phi_arr, dt, n_steps, I_stim,
        ch_g_dens, ch_E_revs, ch_T_halfs, ch_delta_As, ch_tau_0s, ch_kT,
        m0_init, h0_init, n0_init,
        spike_counts,
    ):
        idx = cuda.grid(1)
        N = spike_counts.shape[0]
        if idx >= N:
            return
        is_cortical = is_cortical_int != 0
        n_ch = ch_g_dens.shape[0]

        V = V_rest
        m = m0_init
        h = h0_init
        n = n0_init
        phi = phi_arr[idx]

        w = cuda.local.array(MAX_CH, dtype=float64)
        w_inf_now = cuda.local.array(MAX_CH, dtype=float64)
        tau_inv_now = cuda.local.array(MAX_CH, dtype=float64)
        k1w = cuda.local.array(MAX_CH, dtype=float64)
        k2w = cuda.local.array(MAX_CH, dtype=float64)
        k3w = cuda.local.array(MAX_CH, dtype=float64)
        k4w = cuda.local.array(MAX_CH, dtype=float64)
        wtmp = cuda.local.array(MAX_CH, dtype=float64)

        # Initialise gates at steady state for initial tension
        for k in range(n_ch):
            T0 = tensions_2d[0, idx]
            exponent = -ch_delta_As[k] * (T0 - ch_T_halfs[k]) / ch_kT
            if exponent >= 0.0:
                e = math.exp(-exponent)
                w[k] = e / (1.0 + e)
            else:
                w[k] = 1.0 / (1.0 + math.exp(exponent))

        spike_count = 0
        for step in range(n_steps):
            V_prev = V
            T_now = tensions_2d[step, idx]

            for k in range(n_ch):
                exponent = -ch_delta_As[k] * (T_now - ch_T_halfs[k]) / ch_kT
                if exponent >= 0.0:
                    e = math.exp(-exponent)
                    po = e / (1.0 + e)
                else:
                    po = 1.0 / (1.0 + math.exp(exponent))
                w_inf_now[k] = po
                denom = 4.0 * po * (1.0 - po) + 0.01
                tau_inv_now[k] = denom / ch_tau_0s[k]

            k1V, k1m, k1h, k1n = _derivs_single(
                V, m, h, n, w, phi, is_cortical, VT,
                g_Na, g_K, g_L, E_Na, E_K, E_L, C_m,
                I_stim, n_ch, ch_g_dens, ch_E_revs,
            )
            for k in range(n_ch):
                k1w[k] = (w_inf_now[k] - w[k]) * tau_inv_now[k]

            for k in range(n_ch):
                wtmp[k] = w[k] + 0.5 * dt * k1w[k]
            k2V, k2m, k2h, k2n = _derivs_single(
                V + 0.5 * dt * k1V, m + 0.5 * dt * k1m,
                h + 0.5 * dt * k1h, n + 0.5 * dt * k1n,
                wtmp, phi, is_cortical, VT,
                g_Na, g_K, g_L, E_Na, E_K, E_L, C_m,
                I_stim, n_ch, ch_g_dens, ch_E_revs,
            )
            for k in range(n_ch):
                k2w[k] = (w_inf_now[k] - wtmp[k]) * tau_inv_now[k]

            for k in range(n_ch):
                wtmp[k] = w[k] + 0.5 * dt * k2w[k]
            k3V, k3m, k3h, k3n = _derivs_single(
                V + 0.5 * dt * k2V, m + 0.5 * dt * k2m,
                h + 0.5 * dt * k2h, n + 0.5 * dt * k2n,
                wtmp, phi, is_cortical, VT,
                g_Na, g_K, g_L, E_Na, E_K, E_L, C_m,
                I_stim, n_ch, ch_g_dens, ch_E_revs,
            )
            for k in range(n_ch):
                k3w[k] = (w_inf_now[k] - wtmp[k]) * tau_inv_now[k]

            for k in range(n_ch):
                wtmp[k] = w[k] + dt * k3w[k]
            k4V, k4m, k4h, k4n = _derivs_single(
                V + dt * k3V, m + dt * k3m,
                h + dt * k3h, n + dt * k3n,
                wtmp, phi, is_cortical, VT,
                g_Na, g_K, g_L, E_Na, E_K, E_L, C_m,
                I_stim, n_ch, ch_g_dens, ch_E_revs,
            )
            for k in range(n_ch):
                k4w[k] = (w_inf_now[k] - wtmp[k]) * tau_inv_now[k]

            s = dt / 6.0
            V = V + s * (k1V + 2.0 * k2V + 2.0 * k3V + k4V)
            m = m + s * (k1m + 2.0 * k2m + 2.0 * k3m + k4m)
            h = h + s * (k1h + 2.0 * k2h + 2.0 * k3h + k4h)
            n = n + s * (k1n + 2.0 * k2n + 2.0 * k3n + k4n)
            for k in range(n_ch):
                dw_avg = (k1w[k] + 2.0 * k2w[k] + 2.0 * k3w[k] + k4w[k]) / 6.0
                w[k] = w[k] + dt * dw_avg

            if m < 0.0: m = 0.0
            elif m > 1.0: m = 1.0
            if h < 0.0: h = 0.0
            elif h > 1.0: h = 1.0
            if n < 0.0: n = 0.0
            elif n > 1.0: n = 1.0
            for k in range(n_ch):
                wk = w[k]
                if wk < 0.0: wk = 0.0
                elif wk > 1.0: wk = 1.0
                w[k] = wk

            if V > 0.0 and V_prev <= 0.0:
                spike_count += 1

        spike_counts[idx] = spike_count

    _KERNELS["3comp"] = rk4_batch_3comp_kernel
    _KERNELS["1comp"] = rk4_batch_kernel
    _KERNELS["1comp_tv"] = rk4_batch_tv_kernel


# ------------------------------------------------------------------ #
# Host-side dispatchers — same signatures as their CPU counterparts.
# ------------------------------------------------------------------ #

def _check_max_ch(n_ch: int) -> None:
    if n_ch > _MAX_CH:
        msg = (
            f"CUDA backend supports up to {_MAX_CH} mechanosensitive "
            f"channels; got n_ch={n_ch}. Raise _MAX_CH in "
            "cuda_backend.py and rebuild."
        )
        raise ValueError(msg)


def _launch_config(N: int) -> tuple[int, int]:
    # 128 threads/block: best throughput at N≥10k on Ampere; smaller
    # workloads benefit from 64 to spread across more SMs. Override via
    # ``ACOUSTO_NEURO_CUDA_THREADS``.
    import os  # noqa: PLC0415
    default = "64" if N <= 8192 else "128"
    threads = int(os.environ.get("ACOUSTO_NEURO_CUDA_THREADS", default))
    blocks = (N + threads - 1) // threads
    return blocks, threads


def rk4_batch_cuda_3comp(
    w_inf_per_vox,
    tau_inv_per_vox,
    h_inf_per_vox,
    h_tau_inv_per_vox,
    g_Na,
    g_K,
    g_L,
    E_Na, E_K, E_L,
    C_eff_per_vox,
    g_ds, g_sa, AsAd, AsAa,
    g_mech, E_revs, Ca_frac,
    flux_factor, tau_decay_inv, Ca_rest,
    kca_g, kca_E_K, kca_half, kca_n, kca_mask,
    trp_g_Po, trp_E_rev, trp_mask,
    is_cortical,
    VT,
    phi,
    I_stim,
    V_rest,
    m0_init, h0_init, n0_init,
    dt, n_steps,
    primary_compartment,
    spike_step_buf, max_spikes,
):
    """CUDA dispatch for :func:`_rk4_batch_numba_3comp`. Same signature."""
    from numba import cuda  # noqa: PLC0415

    kernels = _ensure_kernels()
    kernel = kernels["3comp"]

    N = int(w_inf_per_vox.shape[1])
    n_ch = int(w_inf_per_vox.shape[0])
    _check_max_ch(n_ch)

    # Host arrays → device. Numba copies on launch when given ndarrays,
    # but explicit transfers let us reuse the spike_step_buf for read-back.
    d_w_inf = cuda.to_device(np.ascontiguousarray(w_inf_per_vox, dtype=np.float64))
    d_tau_inv = cuda.to_device(np.ascontiguousarray(tau_inv_per_vox, dtype=np.float64))
    d_h_inf = cuda.to_device(np.ascontiguousarray(h_inf_per_vox, dtype=np.float64))
    d_h_tau_inv = cuda.to_device(np.ascontiguousarray(h_tau_inv_per_vox, dtype=np.float64))
    d_g_Na = cuda.to_device(np.ascontiguousarray(g_Na, dtype=np.float64))
    d_g_K = cuda.to_device(np.ascontiguousarray(g_K, dtype=np.float64))
    d_g_L = cuda.to_device(np.ascontiguousarray(g_L, dtype=np.float64))
    d_C_eff = cuda.to_device(np.ascontiguousarray(C_eff_per_vox, dtype=np.float64))
    d_g_mech = cuda.to_device(np.ascontiguousarray(g_mech, dtype=np.float64))
    d_E_revs = cuda.to_device(np.ascontiguousarray(E_revs, dtype=np.float64))
    d_Ca_frac = cuda.to_device(np.ascontiguousarray(Ca_frac, dtype=np.float64))
    d_kca_g = cuda.to_device(np.ascontiguousarray(kca_g, dtype=np.float64))
    d_kca_E_K = cuda.to_device(np.ascontiguousarray(kca_E_K, dtype=np.float64))
    d_kca_half = cuda.to_device(np.ascontiguousarray(kca_half, dtype=np.float64))
    d_kca_n = cuda.to_device(np.ascontiguousarray(kca_n, dtype=np.float64))
    d_kca_mask = cuda.to_device(np.ascontiguousarray(kca_mask, dtype=np.float64))
    d_trp_g_Po = cuda.to_device(np.ascontiguousarray(trp_g_Po, dtype=np.float64))
    d_trp_E_rev = cuda.to_device(np.ascontiguousarray(trp_E_rev, dtype=np.float64))
    d_trp_mask = cuda.to_device(np.ascontiguousarray(trp_mask, dtype=np.float64))
    d_I_stim = np.asarray(I_stim, dtype=np.float64).reshape(-1)
    I0 = float(d_I_stim[0])
    I1 = float(d_I_stim[1])
    I2 = float(d_I_stim[2])

    d_spike_step_buf = cuda.to_device(
        np.ascontiguousarray(spike_step_buf, dtype=np.int32)
    )
    d_spike_counts = cuda.device_array(N, dtype=np.int32)

    blocks, threads = _launch_config(N)
    n_kca = int(kca_g.shape[0])
    n_trp = int(trp_g_Po.shape[0])

    kernel[blocks, threads](
        d_w_inf, d_tau_inv, d_h_inf, d_h_tau_inv,
        d_g_Na, d_g_K, d_g_L,
        float(E_Na), float(E_K), float(E_L),
        d_C_eff,
        float(g_ds), float(g_sa), float(AsAd), float(AsAa),
        d_g_mech, int(n_ch), d_E_revs, d_Ca_frac,
        float(flux_factor), float(tau_decay_inv), float(Ca_rest),
        d_kca_g, d_kca_E_K, d_kca_half, d_kca_n, d_kca_mask, int(n_kca),
        d_trp_g_Po, d_trp_E_rev, d_trp_mask, int(n_trp),
        int(bool(is_cortical)), float(VT), float(phi),
        I0, I1, I2,
        float(V_rest), float(m0_init), float(h0_init), float(n0_init),
        float(dt), int(n_steps), int(primary_compartment),
        d_spike_step_buf, int(max_spikes),
        d_spike_counts,
    )
    spike_counts = d_spike_counts.copy_to_host()
    # Write back the spike-step buffer so callers can read it.
    if max_spikes > 0:
        spike_step_buf[...] = d_spike_step_buf.copy_to_host()
    return spike_counts


def rk4_batch_cuda(
    V_rest, g_Na, g_K, g_L, E_Na, E_K, E_L, C_m,
    is_cortical, VT,
    tensions, phi_arr, dt, n_steps, I_stim,
    ch_g_dens, ch_E_revs, ch_w_inf, ch_tau_inv,
    m0, h0, n0,
):
    """CUDA dispatch for :func:`_rk4_batch_numba`. Same signature."""
    from numba import cuda  # noqa: PLC0415
    kernels = _ensure_kernels()
    kernel = kernels["1comp"]

    N = int(tensions.shape[0])
    n_ch = int(ch_g_dens.shape[0]) if ch_g_dens.size > 0 else 0
    _check_max_ch(n_ch)

    d_tensions = cuda.to_device(np.ascontiguousarray(tensions, dtype=np.float64))
    d_phi = cuda.to_device(np.ascontiguousarray(phi_arr, dtype=np.float64))
    d_g = cuda.to_device(np.ascontiguousarray(ch_g_dens, dtype=np.float64))
    d_E = cuda.to_device(np.ascontiguousarray(ch_E_revs, dtype=np.float64))
    if n_ch == 0:
        d_w_inf = cuda.to_device(np.zeros((1, N), dtype=np.float64))
        d_tau_inv = cuda.to_device(np.zeros((1, N), dtype=np.float64))
    else:
        d_w_inf = cuda.to_device(np.ascontiguousarray(ch_w_inf, dtype=np.float64))
        d_tau_inv = cuda.to_device(np.ascontiguousarray(ch_tau_inv, dtype=np.float64))
    d_spike = cuda.device_array(N, dtype=np.int32)

    blocks, threads = _launch_config(N)
    kernel[blocks, threads](
        float(V_rest), float(g_Na), float(g_K), float(g_L),
        float(E_Na), float(E_K), float(E_L), float(C_m),
        int(bool(is_cortical)), float(VT),
        d_tensions, d_phi, float(dt), int(n_steps), float(I_stim),
        d_g, d_E, d_w_inf, d_tau_inv,
        float(m0), float(h0), float(n0),
        d_spike,
    )
    return d_spike.copy_to_host()


def rk4_batch_cuda_tv(
    V_rest, g_Na, g_K, g_L, E_Na, E_K, E_L, C_m,
    is_cortical, VT,
    tensions_2d, phi_arr, dt, n_steps, I_stim,
    ch_g_dens, ch_E_revs, ch_T_halfs, ch_delta_As, ch_tau_0s, ch_kT,
    m0, h0, n0,
):
    """CUDA dispatch for :func:`_rk4_batch_numba_tv`. Same signature."""
    from numba import cuda  # noqa: PLC0415
    kernels = _ensure_kernels()
    kernel = kernels["1comp_tv"]

    N = int(tensions_2d.shape[1])
    n_ch = int(ch_g_dens.shape[0]) if ch_g_dens.size > 0 else 0
    _check_max_ch(n_ch)

    d_tensions = cuda.to_device(np.ascontiguousarray(tensions_2d, dtype=np.float64))
    d_phi = cuda.to_device(np.ascontiguousarray(phi_arr, dtype=np.float64))
    d_g = cuda.to_device(np.ascontiguousarray(ch_g_dens, dtype=np.float64))
    d_E = cuda.to_device(np.ascontiguousarray(ch_E_revs, dtype=np.float64))
    d_T_halfs = cuda.to_device(np.ascontiguousarray(ch_T_halfs, dtype=np.float64))
    d_delta_As = cuda.to_device(np.ascontiguousarray(ch_delta_As, dtype=np.float64))
    d_tau_0s = cuda.to_device(np.ascontiguousarray(ch_tau_0s, dtype=np.float64))
    d_spike = cuda.device_array(N, dtype=np.int32)

    blocks, threads = _launch_config(N)
    kernel[blocks, threads](
        float(V_rest), float(g_Na), float(g_K), float(g_L),
        float(E_Na), float(E_K), float(E_L), float(C_m),
        int(bool(is_cortical)), float(VT),
        d_tensions, d_phi, float(dt), int(n_steps), float(I_stim),
        d_g, d_E, d_T_halfs, d_delta_As, d_tau_0s, float(ch_kT),
        float(m0), float(h0), float(n0),
        d_spike,
    )
    return d_spike.copy_to_host()


def maybe_warn_unavailable(requested: str) -> None:
    """Emit a warning when a CUDA backend was asked for but isn't usable."""
    if requested == "cuda" and not is_cuda_available():
        warnings.warn(
            "CUDA backend requested but unavailable; falling back to Numba CPU.",
            RuntimeWarning,
            stacklevel=2,
        )
