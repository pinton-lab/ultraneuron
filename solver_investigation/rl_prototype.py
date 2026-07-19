"""Generalized Rush-Larsen (exponential-Euler) prototype for the
3-compartment mechano-HH neuron, S2 scenario (Piezo1+TREK1, no Ca/KCa/
TRP/NICE/inactivation).

Integrator: per step, with all rate coefficients frozen at the current
state, (1) update HH gates m,h,n and mechano gates w by their exact
exponential solution, then (2) update each compartment voltage by the
exact exponential solution of the *linearised* (frozen-conductance)
membrane equation  C dV/dt = -G_tot (V - V_inf), where G_tot collects
every conductance multiplying V (ionic + mechano + axial) and V_inf is
the instantaneous steady state.  Axial coupling neighbour voltages are
frozen at their start-of-step values (Gauss-Jacobi); the stiff diagonal
(the huge AIS coupling + g_Na) is taken exactly, so the scheme is
stable at any dt.

This mirrors _derivs_3comp_scalar physics exactly so A/B vs RK4 is a
pure method swap.  Gates-then-voltage ordering captures the regenerative
upstroke within a step.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

# Cortical (Pospischil) HH rates, copied from multicompartment._hh_rates_cortical.
@njit(cache=True, fastmath=True, inline="always")
def _rates_cortical(V, VT):
    dVm = V - VT - 13.0
    am = -0.32 * dVm / (np.exp(-dVm / 4.0) - 1.0) if abs(dVm) > 1e-7 else 0.32 * 4.0
    dVm2 = V - VT - 40.0
    bm = 0.28 * dVm2 / (np.exp(dVm2 / 5.0) - 1.0) if abs(dVm2) > 1e-7 else 0.28 * 5.0
    ah = 0.128 * np.exp(-(V - VT - 17.0) / 18.0)
    bh = 4.0 / (1.0 + np.exp(-(V - VT - 40.0) / 5.0))
    dVn = V - VT - 15.0
    an = -0.032 * dVn / (np.exp(-dVn / 5.0) - 1.0) if abs(dVn) > 1e-7 else 0.032 * 5.0
    bn = 0.5 * np.exp(-(V - VT - 10.0) / 40.0)
    return am, bm, ah, bh, an, bn


@njit(cache=True, fastmath=True, inline="always")
def _exp_step(x, x_inf, rate, dt):
    """Exact update of dx/dt = (x_inf - x)*rate over dt."""
    return x_inf + (x - x_inf) * np.exp(-rate * dt)


@njit(parallel=True, cache=True, fastmath=True)
def rl_batch_3comp(
    w_inf_per_vox,    # (n_ch, N)
    tau_inv_per_vox,  # (n_ch, N)
    g_Na, g_K, g_L,   # (3,)
    E_Na, E_K, E_L,
    C_m,              # (3,)
    g_ds, g_sa, AsAd, AsAa,
    g_mech,           # (3, n_ch)
    E_revs,           # (n_ch,)
    VT, phi,
    I_stim,           # (3,)
    V_rest, m0_init, h0_init, n0_init,
    dt, n_steps, primary_compartment,
):
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
                w[c * n_ch + k] = w_inf_l[k]

        Cm0 = C_m[0]; Cm1 = C_m[1]; Cm2 = C_m[2]

        for _step in range(n_steps):
            if primary_compartment == 0:
                V_prev = V0
            elif primary_compartment == 1:
                V_prev = V1
            else:
                V_prev = V2

            # Freeze neighbour voltages for axial coupling (Gauss-Jacobi)
            V0f = V0; V1f = V1; V2f = V2

            # ---- (1) Gate updates (RL, frozen at current V) ----
            am0, bm0, ah0, bh0, an0, bn0 = _rates_cortical(V0, VT)
            am1, bm1, ah1, bh1, an1, bn1 = _rates_cortical(V1, VT)
            am2, bm2, ah2, bh2, an2, bn2 = _rates_cortical(V2, VT)
            am0*=phi; bm0*=phi; ah0*=phi; bh0*=phi; an0*=phi; bn0*=phi
            am1*=phi; bm1*=phi; ah1*=phi; bh1*=phi; an1*=phi; bn1*=phi
            am2*=phi; bm2*=phi; ah2*=phi; bh2*=phi; an2*=phi; bn2*=phi

            m0 = _exp_step(m0, am0/(am0+bm0), am0+bm0, dt)
            m1 = _exp_step(m1, am1/(am1+bm1), am1+bm1, dt)
            m2 = _exp_step(m2, am2/(am2+bm2), am2+bm2, dt)
            h0 = _exp_step(h0, ah0/(ah0+bh0), ah0+bh0, dt)
            h1 = _exp_step(h1, ah1/(ah1+bh1), ah1+bh1, dt)
            h2 = _exp_step(h2, ah2/(ah2+bh2), ah2+bh2, dt)
            n0 = _exp_step(n0, an0/(an0+bn0), an0+bn0, dt)
            n1 = _exp_step(n1, an1/(an1+bn1), an1+bn1, dt)
            n2 = _exp_step(n2, an2/(an2+bn2), an2+bn2, dt)
            for j in range(3 * n_ch):
                k = j % n_ch
                w[j] = _exp_step(w[j], w_inf_l[k], tau_inv_l[k], dt)

            # ---- (2) Voltage updates (exponential, frozen conductances
            #          using the JUST-updated gates) ----
            # Conductances multiplying V and the driving (E*g) sums.
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

            # Axial coupling conductances (into-diagonal G and neighbour B)
            # comp0 (dend): g_ds*AsAd*(V1-V0)
            gax0 = g_ds*AsAd
            # comp1 (soma): g_ds*(V0-V1) + g_sa*(V2-V1)
            gax1d = g_ds; gax1a = g_sa
            # comp2 (ais): g_sa*AsAa*(V1-V2)
            gax2 = g_sa*AsAa

            G0 = gNa0 + gK0 + g_L[0] + Gm0 + gax0
            B0 = gNa0*E_Na + gK0*E_K + g_L[0]*E_L + Bm0 + gax0*V1f + I0
            G1 = gNa1 + gK1 + g_L[1] + Gm1 + gax1d + gax1a
            B1 = gNa1*E_Na + gK1*E_K + g_L[1]*E_L + Bm1 + gax1d*V0f + gax1a*V2f + I1
            G2 = gNa2 + gK2 + g_L[2] + Gm2 + gax2
            B2 = gNa2*E_Na + gK2*E_K + g_L[2]*E_L + Bm2 + gax2*V1f + I2

            V0 = _exp_step(V0, B0/G0, G0/Cm0, dt)
            V1 = _exp_step(V1, B1/G1, G1/Cm1, dt)
            V2 = _exp_step(V2, B2/G2, G2/Cm2, dt)

            # Spike detect at primary compartment
            if primary_compartment == 0:
                V_now = V0
            elif primary_compartment == 1:
                V_now = V1
            else:
                V_now = V2
            if V_now > 0.0 and V_prev <= 0.0:
                spike_counts[idx] += 1

    return spike_counts


def run(neuron, T, dt, duration, primary_compartment=1):
    """Mirror neuron._simulate_batch front-end, then call the RL kernel."""
    n_ch = len(neuron.mechano_channels)
    params = neuron._stacked_params()
    phi = neuron._kinetics._q10_scale()
    n_steps = int(duration / dt)
    T = np.asarray(T, dtype=np.float64)
    E_revs, w_inf, tau_inv, _ = neuron._channel_kinetics(T, n_ch)
    m0i, h0i, n0i = neuron._kinetics._steady_state_gates(neuron.V_rest)
    geom = neuron.geometry
    g_mech = np.empty((3, n_ch))
    comps = (geom.dendrite, geom.soma, geom.ais)
    for c in range(3):
        for k in range(n_ch):
            g_mech[c, k] = neuron.mechano_channels[k].g_density * comps[c].mechano_density_scale[k]
    g_Na = np.array([params["g_Na"][c, 0] for c in range(3)])
    g_K = np.array([params["g_K"][c, 0] for c in range(3)])
    g_L = np.array([params["g_L"][c, 0] for c in range(3)])
    C_m = np.array([params["C_m"][c, 0] for c in range(3)])
    return rl_batch_3comp(
        w_inf, tau_inv, g_Na, g_K, g_L,
        float(params["E_Na"][1, 0]), float(params["E_K"][1, 0]), float(params["E_L"][1, 0]),
        C_m,
        float(geom.g_dend_soma), float(geom.g_soma_ais),
        float(geom.soma.area / geom.dendrite.area),
        float(geom.soma.area / geom.ais.area),
        g_mech, E_revs,
        float(neuron._kinetics._VT), float(phi),
        np.array([0.0, 0.0, 0.0]),
        float(neuron.V_rest), float(m0i), float(h0i), float(n0i),
        float(dt), int(n_steps), int(primary_compartment),
    )
