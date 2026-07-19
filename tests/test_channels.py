import copy

import numpy as np
import pytest

from ultraneuron.channel_library import (
    PIEZO1,
    TRAAK,
    TREK1,
    TREK2,
    get_channel,
    get_neuron_profile,
)
from ultraneuron.channels import (
    MechanosensitiveChannel,
    strain_to_membrane_tension,
)


# ---------------------------------------------------------------------------
# strain_to_membrane_tension
# ---------------------------------------------------------------------------
class TestStrainToMembraneTension:
    def test_zero_strain(self):
        result = strain_to_membrane_tension(0.0)
        assert result == 0.0

    def test_scaling(self):
        """Output should be K_A * coupling * strain * 1e3 (mN/m)."""
        result = strain_to_membrane_tension(0.01, K_A=0.25, coupling_factor=1.0)
        expected = 0.25 * 1.0 * 0.01 * 1e3  # = 2.5 mN/m
        np.testing.assert_allclose(result, expected)

    def test_array_input(self):
        strains = np.array([0.0, 0.005, 0.01])
        result = strain_to_membrane_tension(strains, K_A=0.2)
        expected = 0.2 * strains * 1e3
        np.testing.assert_allclose(result, expected)


# ---------------------------------------------------------------------------
# MechanosensitiveChannel — Boltzmann gating
# ---------------------------------------------------------------------------
class TestMechanosensitiveChannel:
    def test_half_activation(self):
        """At T = T_half, P_o should be 0.5."""
        ch = MechanosensitiveChannel(
            name="test",
            T_half=3.0,
            delta_A=15.0,
            g_max=30.0,
            g_density=0.01,
            E_rev=0.0,
        )
        np.testing.assert_allclose(ch.open_probability(3.0), 0.5, atol=1e-10)

    def test_monotonic_cation(self):
        """Open probability increases with tension for positive delta_A."""
        ch = copy.deepcopy(PIEZO1)
        tensions = np.linspace(0, 10, 50)
        P = ch.open_probability(tensions)
        assert np.all(np.diff(P) >= 0), "P_o should be monotonically increasing"

    def test_bounds(self):
        """P_o should always be in [0, 1]."""
        ch = copy.deepcopy(PIEZO1)
        tensions = np.linspace(-100, 100, 200)
        P = ch.open_probability(tensions)
        assert np.all(P >= 0.0)
        assert np.all(P <= 1.0)

    def test_current_reversal(self):
        """Current should reverse sign when V_m crosses E_rev."""
        ch = MechanosensitiveChannel(
            name="test",
            T_half=3.0,
            delta_A=15.0,
            g_max=30.0,
            g_density=0.01,
            E_rev=0.0,
        )
        T = 5.0  # well above T_half -> P_o near 1
        I_hyper = ch.current(T, V_m=10.0)  # V_m > E_rev -> outward (negative)
        I_depol = ch.current(T, V_m=-65.0)  # V_m < E_rev -> inward (positive)
        assert I_hyper < 0
        assert I_depol > 0

    def test_potassium_hyperpolarisation(self):
        """K+ channel (E_rev = -90) should produce outward current at rest."""
        ch = copy.deepcopy(TRAAK)
        T = 8.0  # above T_half
        I = ch.current(T, V_m=-65.0)
        # E_rev (-90) < V_m (-65) -> current = g * P * (E_rev - V_m) < 0 -> hyperpolarising
        assert I < 0

    def test_time_constant_positive(self):
        ch = copy.deepcopy(PIEZO1)
        tensions = np.linspace(0, 10, 20)
        tau = ch.time_constant(tensions)
        assert np.all(tau > 0)


# ---------------------------------------------------------------------------
# Channel library
# ---------------------------------------------------------------------------
class TestChannelLibrary:
    def test_get_channel_case_insensitive(self):
        ch = get_channel("piezo1")
        assert ch.name == "Piezo1"
        ch2 = get_channel("PIEZO1")
        assert ch2.name == "Piezo1"

    def test_get_channel_copy_semantics(self):
        """Returned channel should be a copy, not the library original."""
        ch = get_channel("Piezo1")
        ch.T_half = 999.0
        ch_orig = get_channel("Piezo1")
        assert ch_orig.T_half != 999.0

    def test_get_channel_unknown(self):
        with pytest.raises(KeyError, match="Unknown channel"):
            get_channel("NONEXISTENT")

    def test_all_predefined_channels(self):
        """Smoke test: all pre-defined channels should be valid."""
        for ch in [PIEZO1, TRAAK, TREK1, TREK2]:
            assert ch.T_half > 0
            assert ch.delta_A > 0
            assert ch.g_density > 0

    def test_get_neuron_profile(self):
        channels = get_neuron_profile("cortical_pyramidal")
        assert len(channels) == 2
        names = {ch.name for ch in channels}
        assert "Piezo1" in names
        assert "TREK-1" in names

    def test_get_neuron_profile_unknown(self):
        with pytest.raises(KeyError, match="Unknown neuron profile"):
            get_neuron_profile("alien_brain")
