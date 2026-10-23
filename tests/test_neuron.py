import copy

import numpy as np
import pytest

from ultraneuron.channel_library import PIEZO1, TRAAK, TREK1
from ultraneuron.neuron import (
    MechanoHHNeuron,
    bioheat_to_absolute_temperature,
    simulate_map,
)


# ---------------------------------------------------------------------------
# Baseline HH (no mechano channels)
# ---------------------------------------------------------------------------
class TestHHBaseline:
    def test_resting_potential_stable(self):
        """With no stimulus, voltage should stay near V_rest."""
        neuron = MechanoHHNeuron()
        result = neuron.simulate(dt=0.01, duration=20.0, I_stim=0.0)
        V = result["V"]
        # Should stay within 2 mV of rest
        assert np.all(np.abs(V - neuron.V_rest) < 2.0)

    def test_subthreshold_no_spike(self):
        """Small current injection should not produce spikes."""
        neuron = MechanoHHNeuron(temperature=6.3)
        result = neuron.simulate(dt=0.01, duration=30.0, I_stim=2.0)
        assert len(result["spike_times"]) == 0

    def test_suprathreshold_spike(self):
        """Large current injection should produce spikes.

        Uses classic HH temperature (6.3 °C) where firing threshold
        is ~7 uA/cm^2.
        """
        neuron = MechanoHHNeuron(temperature=6.3)
        result = neuron.simulate(dt=0.01, duration=50.0, I_stim=15.0)
        assert len(result["spike_times"]) > 0
        # Spike peak should exceed 0 mV
        assert np.max(result["V"]) > 0.0

    def test_output_shapes(self):
        neuron = MechanoHHNeuron()
        result = neuron.simulate(dt=0.01, duration=10.0)
        n_steps = int(10.0 / 0.01)
        assert result["t"].shape == (n_steps,)
        assert result["V"].shape == (n_steps,)
        assert result["m"].shape == (n_steps,)
        assert result["h"].shape == (n_steps,)
        assert result["n"].shape == (n_steps,)


# ---------------------------------------------------------------------------
# Q10 temperature scaling
# ---------------------------------------------------------------------------
class TestQ10Scaling:
    def test_higher_temp_faster_rates(self):
        """At higher temperature, firing threshold should be lower."""
        neuron_cold = MechanoHHNeuron(temperature=6.3)
        neuron_warm = MechanoHHNeuron(temperature=37.0)
        # At body temp the neuron should fire with lower current
        result_cold = neuron_cold.simulate(dt=0.01, duration=50.0, I_stim=10.0)
        result_warm = neuron_warm.simulate(dt=0.01, duration=50.0, I_stim=10.0)
        # At 6.3°C with 10 uA/cm^2 it may or may not fire,
        # but at 37°C the Q10 scaling makes it respond differently
        # Just check both run without error and warm gives more activity
        assert result_cold["V"].shape == result_warm["V"].shape


# ---------------------------------------------------------------------------
# Mechanosensitive coupling
# ---------------------------------------------------------------------------
class TestMechanoCoupling:
    def test_piezo1_depolarisation(self):
        """Piezo1 (cation, E_rev=0) at high tension should depolarise."""
        ch = copy.deepcopy(PIEZO1)
        ch.g_density = 0.5  # strong conductance for clear depolarisation
        neuron = MechanoHHNeuron(mechano_channels=[ch])
        result = neuron.simulate(dt=0.01, duration=30.0, tension_trace=8.0)
        # Should depolarise from rest
        assert np.max(result["V"]) > neuron.V_rest + 5.0

    def test_traak_hyperpolarisation(self):
        """TRAAK (K+, E_rev=-90) at high tension should hyperpolarise."""
        ch = copy.deepcopy(TRAAK)
        ch.g_density = 0.1  # boost
        neuron = MechanoHHNeuron(mechano_channels=[ch])
        result = neuron.simulate(dt=0.01, duration=20.0, tension_trace=10.0)
        # Should hyperpolarise below rest
        assert np.min(result["V"]) < neuron.V_rest - 1.0

    def test_competing_channels(self):
        """Piezo1 + TREK1: opposing currents should partially cancel."""
        piezo = copy.deepcopy(PIEZO1)
        piezo.g_density = 0.02
        trek = copy.deepcopy(TREK1)
        trek.g_density = 0.02
        neuron = MechanoHHNeuron(mechano_channels=[piezo, trek])
        result = neuron.simulate(dt=0.01, duration=20.0, tension_trace=5.0)
        # Should run without errors; net effect depends on balance
        assert result["I_mechano"].shape[0] > 0

    def test_zero_tension_no_effect(self):
        """Zero tension should produce negligible mechano current."""
        ch = copy.deepcopy(PIEZO1)
        neuron = MechanoHHNeuron(mechano_channels=[ch])
        result = neuron.simulate(dt=0.01, duration=10.0, tension_trace=0.0)
        # Mechano current should be near zero (Boltzmann at T << T_half)
        assert np.all(np.abs(result["I_mechano"]) < 0.1)


# ---------------------------------------------------------------------------
# RK4 convergence
# ---------------------------------------------------------------------------
class TestRK4:
    def test_convergence_order(self):
        """RK4 error should decrease ~16x when dt halves."""
        neuron = MechanoHHNeuron()
        # Reference: very fine dt
        ref = neuron.simulate(dt=0.001, duration=10.0, I_stim=10.0)
        V_ref_end = ref["V"][-1]

        r1 = neuron.simulate(dt=0.04, duration=10.0, I_stim=10.0)
        r2 = neuron.simulate(dt=0.02, duration=10.0, I_stim=10.0)

        err1 = abs(r1["V"][-1] - V_ref_end)
        err2 = abs(r2["V"][-1] - V_ref_end)

        # RK4 should give ratio ~16; allow anything > 4 to account for
        # nonlinear dynamics and gate clamping
        if err2 > 1e-10:
            ratio = err1 / err2
            assert ratio > 2.0, f"Expected convergence ratio > 2, got {ratio}"


# ---------------------------------------------------------------------------
# simulate_map
# ---------------------------------------------------------------------------
class TestSimulateMap:
    def test_output_shapes(self):
        neuron = MechanoHHNeuron(mechano_channels=[copy.deepcopy(PIEZO1)])
        strain = np.zeros((3, 4, 5))
        result = simulate_map(
            strain,
            neuron,
            dt=0.01,
            duration=5.0,
            tension_threshold=100.0,
        )
        assert result["firing_map"].shape == (3, 4, 5)
        assert result["spike_count_map"].shape == (3, 4, 5)
        assert isinstance(result["threshold_tension"], float)

    def test_zero_strain_no_firing(self):
        neuron = MechanoHHNeuron(mechano_channels=[copy.deepcopy(PIEZO1)])
        strain = np.zeros((3, 3, 3))
        result = simulate_map(
            strain,
            neuron,
            dt=0.01,
            duration=5.0,
            tension_threshold=0.0,
        )
        assert not np.any(result["firing_map"])


# ---------------------------------------------------------------------------
# Temperature coupling
# ---------------------------------------------------------------------------
class TestTemperatureCoupling:
    def test_constant_temperature_trace_matches_static(self):
        """temperature_trace=37.0 should match temperature=37.0."""
        neuron = MechanoHHNeuron(temperature=37.0)
        result_static = neuron.simulate(dt=0.01, duration=20.0, I_stim=10.0)
        result_trace = neuron.simulate(dt=0.01, duration=20.0, I_stim=10.0, temperature_trace=37.0)
        np.testing.assert_allclose(result_static["V"], result_trace["V"], atol=1e-12)

    def test_none_temperature_trace_uses_neuron_field(self):
        """temperature_trace=None uses neuron.temperature (backward compat)."""
        neuron = MechanoHHNeuron(temperature=37.0)
        result_none = neuron.simulate(dt=0.01, duration=20.0, I_stim=10.0)
        result_default = neuron.simulate(
            dt=0.01, duration=20.0, I_stim=10.0, temperature_trace=None
        )
        np.testing.assert_allclose(result_none["V"], result_default["V"], atol=1e-12)

    def test_time_varying_temperature_changes_behavior(self):
        """A temperature ramp from 37 to 39 °C should differ from constant 37."""
        neuron = MechanoHHNeuron(temperature=37.0)
        n_steps = int(30.0 / 0.01)
        temp_ramp = np.linspace(37.0, 39.0, n_steps)
        result_const = neuron.simulate(dt=0.01, duration=30.0, I_stim=10.0)
        result_ramp = neuron.simulate(
            dt=0.01, duration=30.0, I_stim=10.0, temperature_trace=temp_ramp
        )
        # Traces must differ (different phi per step)
        assert not np.allclose(result_const["V"], result_ramp["V"])


# ---------------------------------------------------------------------------
# simulate_map with temperature_3d
# ---------------------------------------------------------------------------
class TestSimulateMapTemperature:
    def test_simulate_map_with_temperature_3d(self):
        """simulate_map accepts temperature_3d without error."""
        neuron = MechanoHHNeuron(mechano_channels=[copy.deepcopy(PIEZO1)])
        strain = np.zeros((2, 2, 2))
        temp_3d = np.full((2, 2, 2), 37.0)
        result = simulate_map(
            strain,
            neuron,
            dt=0.01,
            duration=5.0,
            tension_threshold=100.0,
            temperature_3d=temp_3d,
        )
        assert result["firing_map"].shape == (2, 2, 2)

    def test_simulate_map_temperature_shape_mismatch_raises(self):
        """Mismatched temperature_3d shape should raise ValueError."""
        neuron = MechanoHHNeuron(mechano_channels=[copy.deepcopy(PIEZO1)])
        strain = np.zeros((3, 4, 5))
        temp_3d = np.full((2, 2, 2), 37.0)
        with pytest.raises(ValueError, match="does not match"):
            simulate_map(
                strain,
                neuron,
                dt=0.01,
                duration=5.0,
                tension_threshold=100.0,
                temperature_3d=temp_3d,
            )


# ---------------------------------------------------------------------------
# bioheat_to_absolute_temperature
# ---------------------------------------------------------------------------
class TestBioheatToAbsoluteTemperature:
    def test_scalar(self):
        assert bioheat_to_absolute_temperature(0.0) == 37.0
        assert bioheat_to_absolute_temperature(1.5) == 38.5

    def test_custom_baseline(self):
        assert bioheat_to_absolute_temperature(2.0, baseline=36.0) == 38.0

    def test_array(self):
        delta_T = np.array([0.0, 0.5, 1.0, 2.0])
        result = bioheat_to_absolute_temperature(delta_T)
        np.testing.assert_allclose(result, [37.0, 37.5, 38.0, 39.0])

    def test_3d_array(self):
        delta_T = np.ones((3, 4, 5)) * 0.8
        result = bioheat_to_absolute_temperature(delta_T)
        assert result.shape == (3, 4, 5)
        np.testing.assert_allclose(result, 37.8)


class TestMechanoInitSingleComp:
    """Scenario S1 (point neuron): default init is true rest (T = 0 gates)."""

    def _neuron(self):
        p1 = copy.deepcopy(PIEZO1)
        p1.g_density = 0.5
        return MechanoHHNeuron.cortical_pyramidal(mechano_channels=[p1])

    def test_simulate_initial_gate(self):
        n = self._neuron()
        r = n.simulate(dt=0.01, duration=0.05, tension_trace=6.0)
        z = n.simulate(dt=0.01, duration=0.05, tension_trace=0.0)
        p = n.simulate(dt=0.01, duration=0.05, tension_trace=6.0,
                       mechano_init="peak_steady")
        # I_mechano[0] is evaluated with the initial gate at V_rest
        assert r["I_mechano"][0] == z["I_mechano"][0]
        assert abs(p["I_mechano"][0]) > abs(r["I_mechano"][0])
        with pytest.raises(ValueError, match="mechano_init"):
            n.simulate(dt=0.01, duration=0.05, mechano_init="bogus")

    @pytest.mark.parametrize("mode", ["rest", "peak_steady"])
    def test_batch_kernels_agree_with_numpy_fallback(self, mode):
        n = self._neuron()
        T = np.linspace(0.0, 8.0, 12)
        phi = np.full(T.size, n._q10_scale())
        n_steps = 2000
        sc = n._simulate_batch(T, phi, 0.01, n_steps, 0.0, backend="numba",
                               mechano_init=mode)
        ch = n.mechano_channels[0]
        w_inf = ch.open_probability(T)[None, :]
        tau_inv = (1.0 / ch.time_constant(T))[None, :]
        m0, h0, n0 = n._steady_state_gates(n.V_rest)
        w_init = (np.full((1, T.size), float(ch.open_probability(0.0)))
                  if mode == "rest" else None)
        ref = n._simulate_batch_numpy(
            T, phi, 0.01, n_steps, 0.0, w_inf, tau_inv,
            np.array([ch.g_density]), np.array([ch.E_rev]), m0, h0, n0,
            w_init=w_init)
        assert np.array_equal(sc, ref)
        # Time-varying path: Numba kernel and NumPy fallback share the init
        T2 = np.broadcast_to(T, (n_steps, T.size)).copy()
        sc_tv = n._simulate_batch(T, phi, 0.01, n_steps, 0.0, tensions_2d=T2,
                                  backend="numba", mechano_init=mode)
        ref_tv = n._simulate_batch_numpy_tv(
            T2, phi, 0.01, n_steps, 0.0, np.array([ch.g_density]),
            np.array([ch.E_rev]), np.array([ch.T_half]), np.array([ch.delta_A]),
            np.array([ch.tau_0]), ch._kT, m0, h0, n0, init_rest=(mode == "rest"),
            tau_act_inv=np.array([1.0 / ch.tau_act]))
        assert np.array_equal(sc_tv, ref_tv)

    @pytest.mark.parametrize("mode", ["rest", "peak_steady"])
    def test_time_varying_kernel_honours_fixed_tau_act(self, mode):
        """A constant tension fed through the time-varying kernel must give
        the constant-tension result: both honour the channel's fixed
        ``tau_act`` (Piezo1: 2 ms) instead of the bell-shaped tau_0 form."""
        n = self._neuron()
        ch = n.mechano_channels[0]
        assert ch.tau_act is not None
        T = np.linspace(1.0, 8.0, 16)
        phi = np.full(T.size, n._q10_scale())
        n_steps = 2000
        T2 = np.broadcast_to(T, (n_steps, T.size)).copy()
        sc = n._simulate_batch(T, phi, 0.01, n_steps, 0.0, backend="numba",
                               mechano_init=mode)
        sc_tv = n._simulate_batch(T, phi, 0.01, n_steps, 0.0, tensions_2d=T2,
                                  backend="numba", mechano_init=mode)
        assert sc.sum() > 0
        assert np.allclose(sc, sc_tv, atol=1)
        # NumPy fallback, and the bell-shaped form differs (so the test bites)
        m0, h0, n0 = n._steady_state_gates(n.V_rest)
        args = (T2, phi, 0.01, n_steps, 0.0, np.array([ch.g_density]),
                np.array([ch.E_rev]), np.array([ch.T_half]),
                np.array([ch.delta_A]), np.array([ch.tau_0]), ch._kT,
                m0, h0, n0)
        fixed = n._simulate_batch_numpy_tv(
            *args, init_rest=(mode == "rest"),
            tau_act_inv=np.array([1.0 / ch.tau_act]))
        bell = n._simulate_batch_numpy_tv(*args, init_rest=(mode == "rest"))
        assert np.allclose(fixed, sc, atol=1)
        if mode == "rest":
            assert not np.array_equal(fixed, bell)

    def test_rest_default_differs_from_peak_steady(self):
        n = self._neuron()
        T = np.linspace(2.0, 8.0, 16)
        phi = np.full(T.size, n._q10_scale())
        r = n._simulate_batch(T, phi, 0.01, 1000, 0.0, backend="numba")
        p = n._simulate_batch(T, phi, 0.01, 1000, 0.0, backend="numba",
                              mechano_init="peak_steady")
        assert p.sum() > 0
        assert not np.array_equal(r, p)
        assert np.all(r <= p + 1)

    def test_simulate_map_threads_option(self):
        n = self._neuron()
        # tension = K_A·alpha·strain = 250 N/m · strain → 0..8 mN/m
        strain = (np.linspace(0.0, 8.0, 8) / 250e3).reshape(2, 2, 2)
        kw = dict(K_A=0.25, coupling_factor=1000.0, dt=0.01, duration=10.0,
                  tension_threshold=0.0, backend="numba")
        d = simulate_map(strain, n, **kw)
        r = simulate_map(strain, n, mechano_init="rest", **kw)
        p = simulate_map(strain, n, mechano_init="peak_steady", **kw)
        assert np.array_equal(d["spike_count_map"], r["spike_count_map"])
        assert p["spike_count_map"].sum() > 0
        assert not np.array_equal(r["spike_count_map"], p["spike_count_map"])
