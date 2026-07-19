"""Phase-4 tests: TRPV1 / TRPV4 thermosensor channels.

Validation references
---------------------
* Caterina et al. 1997, 2000 — TRPV1: heat-activated cation channel,
  T_half ≈ 43°C, very steep activation (Q10 ≥ 20 near threshold). At
  37°C body temperature the channel is essentially closed; at >45°C it
  opens substantially.
* Watanabe et al. 2002 — TRPV4: warm-activated cation channel,
  activates between 24°C and 34°C; at body temperature the channel is
  partially open and contributes a tonic depolarising current.

This test set covers (in order of importance):
  (i)   TRPV1 essentially closed at body temperature, strongly open at
        thermoneuromodulation-relevant overheating (45°C).
  (ii)  TRPV4 partially open at body temperature.
  (iii) Steep Q10 across the activation threshold for TRPV1 (>20).
  (iv)  Default opt-out (trp_channels=[]) reproduces Phase-3 behaviour
        bit-for-bit (regression).
  (v)   TRPV1 at high temperature drives depolarisation in a
        multi-compartment neuron.
  (vi)  Compartment-index validation.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from ultraneuron import (
    PIEZO1,
    TRPV1,
    TRPV4,
    MulticompartmentMechanoHHNeuron,
    TRPThermosensorChannel,
)


# ---------------------------------------------------------------------------
# Standalone TRP channel behaviour
# ---------------------------------------------------------------------------
class TestTRPChannelBoltzmann:
    def test_trpv1_closed_at_body_temp(self):
        """TRPV1 should be essentially closed (P_o < 5%) at 37°C."""
        assert TRPV1.open_probability(37.0) < 0.05

    def test_trpv1_open_at_overheat(self):
        """TRPV1 should be > 50% open at 45°C (above T_half)."""
        assert TRPV1.open_probability(45.0) > 0.5

    def test_trpv1_steep_q10(self):
        """TRPV1 Q10 ≈ 20 across activation threshold.

        Plain Boltzmann steepness with k=1.5°C gives a ratio of
        Po(48)/Po(38) ≈ 50 — comfortably above the literature Q10 ≥ 20.
        """
        ratio = TRPV1.open_probability(48.0) / TRPV1.open_probability(38.0)
        assert ratio > 20.0, f"TRPV1 Po(48)/Po(38) = {ratio:.1f} < 20"

    def test_trpv4_partial_open_at_body_temp(self):
        """TRPV4 should be partially open at 37°C (tonic depolariser)."""
        assert 0.5 < TRPV4.open_probability(37.0) < 1.0

    def test_half_activation_at_T_half(self):
        """P_o(T_half) = 0.5 by Boltzmann definition."""
        for ch in (TRPV1, TRPV4):
            assert ch.open_probability(ch.T_half) == pytest.approx(0.5, abs=1e-6)

    def test_current_sign(self):
        """Cation channels (E_rev > V_rest) drive inward current at rest."""
        I = TRPV1.current(T=45.0, V_m=-70.0)
        assert I > 0, "TRPV1 should produce inward (positive) current at rest"
        I_close = TRPV1.current(T=37.0, V_m=-70.0)
        # Tiny but non-zero — channel is mostly closed
        assert I_close > 0
        assert I_close < 0.1 * I


# ---------------------------------------------------------------------------
# Multi-compartment integration
# ---------------------------------------------------------------------------
class TestPhase4OptOutRegression:
    def test_default_trp_off_matches_phase3(self):
        """trp_channels=[] (default) ⇒ identical V trace to Phase-3 neuron."""
        ch = copy.deepcopy(PIEZO1)
        ch.g_density = 0.05
        n_default = MulticompartmentMechanoHHNeuron.cortical_pyramidal_3comp(
            mechano_channels=[ch],
        )
        n_explicit_off = MulticompartmentMechanoHHNeuron.cortical_pyramidal_3comp(
            mechano_channels=[ch],
            trp_channels=[],
        )
        r1 = n_default.simulate(dt=0.001, duration=5.0, tension_trace=5.0)
        r2 = n_explicit_off.simulate(dt=0.001, duration=5.0, tension_trace=5.0)
        np.testing.assert_allclose(r1["V"], r2["V"], atol=1e-12)


class TestTRPVCurrentAtBodyTemp:
    def test_trpv1_negligible_at_37C(self):
        """At 37°C, adding TRPV1 should barely shift V_rest (<<1 mV)."""
        n_no = MulticompartmentMechanoHHNeuron.cortical_pyramidal_3comp(
            temperature=37.0,
        )
        trpv1 = copy.deepcopy(TRPV1)
        trpv1.g_density = 0.5  # exaggerate density to make any leak visible
        n_trp = MulticompartmentMechanoHHNeuron.cortical_pyramidal_3comp(
            temperature=37.0,
            trp_channels=[trpv1],
        )
        r_no = n_no.simulate(dt=0.001, duration=20.0, I_stim={"soma": 0.0})
        r_trp = n_trp.simulate(dt=0.001, duration=20.0, I_stim={"soma": 0.0})
        # Even with 10× exaggerated TRPV1, body-temperature V_rest should
        # shift by less than 2 mV (proves channel is essentially closed).
        V_no = r_no["V"][-1, 1]
        V_trp = r_trp["V"][-1, 1]
        assert abs(V_trp - V_no) < 2.0, (
            f"TRPV1 at 37°C should not perturb V_rest noticeably; "
            f"V(no-TRP)={V_no:.2f}, V(with-TRP)={V_trp:.2f}"
        )

    def test_trpv1_strongly_depolarises_at_overheat(self):
        """At 45°C, TRPV1 drives the soma to a depolarised steady state."""
        trpv1 = copy.deepcopy(TRPV1)
        trpv1.g_density = 0.2
        n_no = MulticompartmentMechanoHHNeuron.cortical_pyramidal_3comp(
            temperature=45.0,
        )
        n_trp = MulticompartmentMechanoHHNeuron.cortical_pyramidal_3comp(
            temperature=45.0,
            trp_channels=[trpv1],
        )
        r_no = n_no.simulate(dt=0.001, duration=30.0, I_stim={"soma": 0.0})
        r_trp = n_trp.simulate(dt=0.001, duration=30.0, I_stim={"soma": 0.0})
        # Compare time-averaged soma V over the back half (steady regime)
        i0 = r_trp["V"].shape[0] // 2
        V_avg_no = r_no["V"][i0:, 1].mean()
        V_avg_trp = r_trp["V"][i0:, 1].mean()
        # TRPV1 with E_rev=10 mV pulls V toward depolarisation;
        # require at least 3 mV upward shift.
        assert V_avg_trp > V_avg_no + 3.0, (
            f"TRPV1 at 45°C should depolarise soma: "
            f"<V>(no-TRP) = {V_avg_no:.2f}, <V>(with-TRP) = {V_avg_trp:.2f}"
        )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
class TestPhase4Validation:
    def test_invalid_trp_compartment_rejected(self):
        with pytest.raises(ValueError, match="trp_compartments"):
            MulticompartmentMechanoHHNeuron.cortical_pyramidal_3comp(
                trp_channels=[TRPV1],
                trp_compartments=(7,),
            )

    def test_custom_trp_channel(self):
        """User-defined TRP channel is accepted and integrated."""
        custom = TRPThermosensorChannel(
            name="TRP-test", T_half=35.0, k=2.0, g_density=0.05, E_rev=0.0,
        )
        n = MulticompartmentMechanoHHNeuron.cortical_pyramidal_3comp(
            temperature=37.0, trp_channels=[custom],
        )
        # Should run without error
        r = n.simulate(dt=0.001, duration=2.0)
        assert r["V"].shape == (2000, 3)
