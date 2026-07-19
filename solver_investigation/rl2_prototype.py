"""Second-order (midpoint) generalized Rush-Larsen for the 3-comp
mechano-HH neuron, S2 scenario.

Same exponential-Euler building block as rl_prototype, but evaluated with
a midpoint predictor so the scheme is 2nd-order accurate.  Per step:
  1. Predictor: GRL1 half-step (dt/2) using coefficients at state_n ->
     state_mid (also unlags the axial coupling to 2nd order).
  2. Corrector: full step from state_n using all coefficients (gate
     rates/steady-states, membrane G/B, neighbour V for coupling)
     evaluated at state_mid.
Cost: 2 coefficient evaluations + exp() per step (vs RK4's 4 stages).
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

from rl_prototype import _exp_step, _rates_cortical


@njit(cache=True, fastmath=True, inline="always")
def _coeffs(V, m, h, n, w, n_ch, g_Na, g_K, g_L, E_Na, E_K, E_L,
           g_mech, E_revs, g_ds, g_sa, AsAd, AsAa, VT, phi, I0, I1, I2):
    """Return gate (inf,rate) for m,h,n per comp and membrane (G,B) per comp.

    V,m,h,n: length-3; w: length 3*n_ch.  Outputs packed in fixed arrays.
    """
    minf = np.empty(3); mrate = np.empty(3)
    hinf = np.empty(3); hrate = np.empty(3)
    ninf = np.empty(3); nrate = np.empty(3)
    G = np.empty(3); B = np.empty(3)
    gNa_arr = np.empty(3); gK_arr = np.empty(3)
    for c in range(3):
        am, bm, ah, bh, an, bn = _rates_cortical(V[c], VT)
        am*=phi; bm*=phi; ah*=phi; bh*=phi; an*=phi; bn*=phi
        minf[c] = am/(am+bm); mrate[c] = am+bm
        hinf[c] = ah/(ah+bh); hrate[c] = ah+bh
        ninf[c] = an/(an+bn); nrate[c] = an+bn
        gNa_arr[c] = g_Na[c]*(m[c]*m[c]*m[c])*h[c]
        gK_arr[c] = g_K[c]*(n[c]*n[c]*n[c]*n[c])
    Istim = (I0, I1, I2)
    for c in range(3):
        Gm = 0.0; Bm = 0.0
        for k in range(n_ch):
            gk = g_mech[c, k]*w[c*n_ch+k]
            Gm += gk; Bm += gk*E_revs[k]
        Gc = gNa_arr[c] + gK_arr[c] + g_L[c] + Gm
        Bc = gNa_arr[c]*E_Na + gK_arr[c]*E_K + g_L[c]*E_L + Bm + Istim[c]
        # axial
        if c == 0:
            gax = g_ds*AsAd; Gc += gax; Bc += gax*V[1]
        elif c == 1:
            Gc += g_ds + g_sa; Bc += g_ds*V[0] + g_sa*V[2]
        else:
            gax = g_sa*AsAa; Gc += gax; Bc += gax*V[1]
        G[c] = Gc; B[c] = Bc
    return minf, mrate, hinf, hrate, ninf, nrate, G, B


@njit(parallel=True, cache=True, fastmath=True)
def rl2_batch_3comp(
    w_inf_per_vox, tau_inv_per_vox,
    g_Na, g_K, g_L, E_Na, E_K, E_L, C_m,
    g_ds, g_sa, AsAd, AsAa,
    g_mech, E_revs, VT, phi, I_stim,
    V_rest, m0_init, h0_init, n0_init,
    dt, n_steps, primary_compartment,
):
    N = w_inf_per_vox.shape[1]
    n_ch = w_inf_per_vox.shape[0]
    spike_counts = np.zeros(N, dtype=np.int32)
    I0 = I_stim[0]; I1 = I_stim[1]; I2 = I_stim[2]
    hdt = 0.5 * dt

    for idx in prange(N):
        V = np.empty(3); m = np.empty(3); h = np.empty(3); nn = np.empty(3)
        for c in range(3):
            V[c] = V_rest; m[c] = m0_init; h[c] = h0_init; nn[c] = n0_init
        w_inf_l = np.empty(n_ch); tau_inv_l = np.empty(n_ch)
        for k in range(n_ch):
            w_inf_l[k] = w_inf_per_vox[k, idx]
            tau_inv_l[k] = tau_inv_per_vox[k, idx]
        w = np.empty(3 * n_ch)
        for c in range(3):
            for k in range(n_ch):
                w[c*n_ch+k] = w_inf_l[k]

        Vm = np.empty(3); mm = np.empty(3); hm = np.empty(3); nm = np.empty(3)
        wm = np.empty(3 * n_ch)

        for _step in range(n_steps):
            V_prev = V[primary_compartment]

            # ---- Predictor: half step using coeffs at state_n ----
            (minf, mrate, hinf, hrate, ninf, nrate, G, B) = _coeffs(
                V, m, h, nn, w, n_ch, g_Na, g_K, g_L, E_Na, E_K, E_L,
                g_mech, E_revs, g_ds, g_sa, AsAd, AsAa, VT, phi, I0, I1, I2)
            for c in range(3):
                mm[c] = _exp_step(m[c], minf[c], mrate[c], hdt)
                hm[c] = _exp_step(h[c], hinf[c], hrate[c], hdt)
                nm[c] = _exp_step(nn[c], ninf[c], nrate[c], hdt)
            for j in range(3 * n_ch):
                k = j % n_ch
                wm[j] = _exp_step(w[j], w_inf_l[k], tau_inv_l[k], hdt)
            for c in range(3):
                Vm[c] = _exp_step(V[c], B[c]/G[c], G[c]/C_m[c], hdt)

            # ---- Corrector: full step using coeffs at midpoint ----
            (minf, mrate, hinf, hrate, ninf, nrate, G, B) = _coeffs(
                Vm, mm, hm, nm, wm, n_ch, g_Na, g_K, g_L, E_Na, E_K, E_L,
                g_mech, E_revs, g_ds, g_sa, AsAd, AsAa, VT, phi, I0, I1, I2)
            for c in range(3):
                m[c] = _exp_step(m[c], minf[c], mrate[c], dt)
                h[c] = _exp_step(h[c], hinf[c], hrate[c], dt)
                nn[c] = _exp_step(nn[c], ninf[c], nrate[c], dt)
            for j in range(3 * n_ch):
                k = j % n_ch
                w[j] = _exp_step(w[j], w_inf_l[k], tau_inv_l[k], dt)
            for c in range(3):
                V[c] = _exp_step(V[c], B[c]/G[c], G[c]/C_m[c], dt)

            if V[primary_compartment] > 0.0 and V_prev <= 0.0:
                spike_counts[idx] += 1

    return spike_counts


def run(neuron, T, dt, duration, primary_compartment=1):
    n_ch = len(neuron.mechano_channels)
    params = neuron._stacked_params()
    phi = neuron._kinetics._q10_scale()
    n_steps = int(duration / dt)
    T = np.asarray(T, dtype=np.float64)
    E_revs, w_inf, tau_inv, _ = neuron._channel_kinetics(T, n_ch)
    m0i, h0i, n0i = neuron._kinetics._steady_state_gates(neuron.V_rest)
    geom = neuron.geometry
    comps = (geom.dendrite, geom.soma, geom.ais)
    g_mech = np.empty((3, n_ch))
    for c in range(3):
        for k in range(n_ch):
            g_mech[c, k] = neuron.mechano_channels[k].g_density * comps[c].mechano_density_scale[k]
    g_Na = np.array([params["g_Na"][c, 0] for c in range(3)])
    g_K = np.array([params["g_K"][c, 0] for c in range(3)])
    g_L = np.array([params["g_L"][c, 0] for c in range(3)])
    C_m = np.array([params["C_m"][c, 0] for c in range(3)])
    return rl2_batch_3comp(
        w_inf, tau_inv, g_Na, g_K, g_L,
        float(params["E_Na"][1, 0]), float(params["E_K"][1, 0]), float(params["E_L"][1, 0]),
        C_m, float(geom.g_dend_soma), float(geom.g_soma_ais),
        float(geom.soma.area/geom.dendrite.area), float(geom.soma.area/geom.ais.area),
        g_mech, E_revs, float(neuron._kinetics._VT), float(phi),
        np.array([0.0, 0.0, 0.0]),
        float(neuron.V_rest), float(m0i), float(h0i), float(n0i),
        float(dt), int(n_steps), int(primary_compartment),
    )
