"""Tests for the mechanosensitive synaptic / voltage-gated transmission layer.

Covers the three literature-grounded mechanisms added on top of the
post-synaptic mechanosensitive channels:

  1. Mechanosensitive voltage-gated Na⁺/Ca²⁺ shift (Kubanek 2016) —
     ``MechanoHHNeuron.k_mechano_nav`` + ``nav_current_gain``.
  2. NMDA-receptor stretch enhancement (Maneshi 2017) —
     ``NMDAReceptor.stretch_factor`` + ``nmdar_stretch_factor``.
  3. Presynaptic release modulation + relay (Tyler 2018) —
     ``MechanoSynapse`` + ``synaptic_gain`` + ``simulate_synaptic_relay``.
"""

from __future__ import annotations

import numpy as np
import pytest

from ultraneuron import (
    ParamSource,
    Provenance,
    provenance_for,
    registered_names,
)
from ultraneuron.neuron import (
    K_MECHANO_NAV,
    MechanoHHNeuron,
    nav_current_gain,
)
from ultraneuron.nmda import NMDAReceptor, nmdar_stretch_factor
from ultraneuron.synapse import (
    MechanoSynapse,
    simulate_synaptic_relay,
    synaptic_gain,
)


# ---------------------------------------------------------------------------
# 1. Mechanosensitive voltage-gated Na⁺/Ca²⁺ shift (Kubanek 2016)
# ---------------------------------------------------------------------------
class TestMechanoNav:
    def test_zero_k_recovers_baseline(self):
        """k_mechano_nav=0 (default) makes tension irrelevant to the membrane."""
        neuron = MechanoHHNeuron(temperature=6.3)  # default k_mechano_nav=0
        r_lo = neuron.simulate(dt=0.01, duration=20.0, I_stim=10.0, tension_trace=0.0)
        r_hi = neuron.simulate(dt=0.01, duration=20.0, I_stim=10.0, tension_trace=50.0)
        np.testing.assert_allclose(r_lo["V"], r_hi["V"], atol=1e-12)

    def test_positive_tension_increases_peak_na_current(self):
        """Hyperpolarising activation shift raises peak Na⁺ current (deliverable 1)."""
        neuron = MechanoHHNeuron(temperature=6.3, k_mechano_nav=2.0)
        r0 = neuron.simulate(dt=0.01, duration=40.0, I_stim=10.0, tension_trace=0.0)
        rT = neuron.simulate(dt=0.01, duration=40.0, I_stim=10.0, tension_trace=10.0)
        peak0 = np.max(np.abs(r0["I_Na"]))
        peakT = np.max(np.abs(rT["I_Na"]))
        assert peakT > peak0

    def test_nav_current_gain_unity_at_zero(self):
        assert nav_current_gain(0.0) == pytest.approx(1.0)

    def test_nav_current_gain_monotonic_increasing(self):
        tensions = [0.0, 1.0, 2.0, 4.0, 8.0, 12.0]
        gains = [nav_current_gain(T) for T in tensions]
        assert all(g >= 1.0 for g in gains)
        assert all(b > a for a, b in zip(gains, gains[1:], strict=False))

    def test_canonical_k_mechano_is_positive(self):
        """The Kubanek-calibrated default shift is positive (hyperpolarising)."""
        assert K_MECHANO_NAV > 0.0


# ---------------------------------------------------------------------------
# 2. NMDA-receptor stretch enhancement (Maneshi 2017)
# ---------------------------------------------------------------------------
class TestNMDAStretch:
    def test_stretch_factor_unity_at_zero(self):
        """At zero tension the factor is exactly 1 (baseline NMDA)."""
        r = NMDAReceptor()
        np.testing.assert_allclose(r.stretch_factor(0.0), 1.0, atol=1e-12)

    def test_stretch_factor_saturates_to_one_plus_alpha(self):
        r = NMDAReceptor(alpha_NMDA=1.0)
        np.testing.assert_allclose(r.stretch_factor(1e3), 2.0, rtol=1e-6)

    def test_stretch_factor_monotonic(self):
        r = NMDAReceptor()
        T = np.linspace(0, 20, 50)
        f = r.stretch_factor(T)
        assert np.all(np.diff(f) >= -1e-12)

    def test_conductance_unaffected_at_zero_tension(self):
        """Backward compat: omitting tension == passing tension=0."""
        r = NMDAReceptor()
        g_default = r.conductance(-40.0, 10.0)
        g_zero = r.conductance(-40.0, 10.0, tension=0.0)
        np.testing.assert_allclose(g_default, g_zero, atol=1e-12)

    def test_conductance_increases_with_tension(self):
        r = NMDAReceptor()
        g0 = r.conductance(-40.0, 10.0, tension=0.0)
        g8 = r.conductance(-40.0, 10.0, tension=8.0)
        assert g8 > g0

    def test_disable_with_zero_alpha(self):
        r = NMDAReceptor(alpha_NMDA=0.0)
        for T in (0.0, 5.0, 50.0):
            np.testing.assert_allclose(r.stretch_factor(T), 1.0, atol=1e-12)

    def test_nmdar_stretch_factor_scalar(self):
        assert nmdar_stretch_factor(0.0) == pytest.approx(1.0)
        assert nmdar_stretch_factor(20.0) > nmdar_stretch_factor(2.0) > 1.0


# ---------------------------------------------------------------------------
# 3. Presynaptic release modulation (Tyler 2018)
# ---------------------------------------------------------------------------
class TestMechanoSynapse:
    def test_release_probability_baseline_at_zero(self):
        syn = MechanoSynapse()
        np.testing.assert_allclose(
            syn.release_probability(0.0), syn.Pr_baseline, atol=1e-12
        )

    def test_release_probability_monotonic_and_bounded(self):
        syn = MechanoSynapse()
        T = np.linspace(0, 20, 50)
        Pr = syn.release_probability(T)
        assert np.all(np.diff(Pr) >= -1e-12)
        assert Pr[-1] <= syn.Pr_max + 1e-9
        assert Pr[-1] > syn.Pr_baseline

    def test_ampa_conductance_monotonic_and_zero_at_zero(self):
        syn = MechanoSynapse()
        assert syn.ampa_conductance(0.0) == pytest.approx(0.0)
        assert syn.ampa_conductance(100.0) > syn.ampa_conductance(10.0)

    def test_synaptic_current_inward_at_rest(self):
        """With glutamate present, the synapse depolarises (inward > 0) at rest."""
        syn = MechanoSynapse()
        i_syn = syn.synaptic_current(V_m=-70.0, glutamate=15.0, tension=4.0)
        assert i_syn > 0.0

    def test_glutamate_trace_silent_without_us(self):
        syn = MechanoSynapse()
        T = np.zeros(2000)
        glu = syn.glutamate_trace(T, dt=0.025, us_mask=np.zeros(2000, dtype=bool))
        assert np.all(glu == 0.0)

    def test_glutamate_trace_rises_under_us(self):
        syn = MechanoSynapse()
        T = np.full(4000, 8.0)
        glu = syn.glutamate_trace(T, dt=0.025)
        # Steady state should approach glu_scale * Pr(8)
        expected_ss = syn.glu_scale * float(syn.release_probability(8.0))
        np.testing.assert_allclose(glu[-1], expected_ss, rtol=0.05)

    def test_synaptic_gain_scalar(self):
        assert synaptic_gain(0.0) == pytest.approx(1.0)
        assert synaptic_gain(12.0) > synaptic_gain(2.0) > 1.0


# ---------------------------------------------------------------------------
# 4. Integration: simulate_synaptic_relay
# ---------------------------------------------------------------------------
class TestSimulateSynapticRelay:
    def test_returns_expected_keys(self):
        r = simulate_synaptic_relay(peak_tension_mNm=8.0, duration=40.0)
        for key in (
            "t", "V", "glutamate", "Pr", "tension", "I_syn",
            "spike_times", "spike_count", "peak_depolarisation", "excitation",
        ):
            assert key in r

    def test_zero_tension_is_silent(self):
        r = simulate_synaptic_relay(peak_tension_mNm=0.0, duration=80.0)
        assert r["spike_count"] == 0

    def test_high_tension_drives_firing(self):
        r = simulate_synaptic_relay(peak_tension_mNm=8.0, duration=80.0)
        assert r["spike_count"] > 0

    def test_excitation_increases_with_tension(self):
        e0 = simulate_synaptic_relay(peak_tension_mNm=0.0, duration=80.0)["excitation"]
        e_lo = simulate_synaptic_relay(peak_tension_mNm=2.0, duration=80.0)["excitation"]
        e_hi = simulate_synaptic_relay(peak_tension_mNm=8.0, duration=80.0)["excitation"]
        assert e_hi > e_lo >= e0

    def test_tension_trace_input(self):
        dt, duration = 0.025, 60.0
        n = int(duration / dt)
        trace = np.zeros(n)
        trace[int(10.0 / dt):int(50.0 / dt)] = 8.0
        r = simulate_synaptic_relay(tension_trace=trace, dt=dt, duration=duration)
        assert r["V"].shape == (n,)
        np.testing.assert_array_equal(r["tension"], trace)

    def test_tension_trace_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="does not match"):
            simulate_synaptic_relay(
                tension_trace=np.zeros(10), dt=0.025, duration=80.0
            )

    def test_requires_some_input(self):
        with pytest.raises(ValueError, match="peak_tension_mNm or tension_trace"):
            simulate_synaptic_relay()


# ---------------------------------------------------------------------------
# 5. Provenance coverage for the new parameters
# ---------------------------------------------------------------------------
class TestSynapseProvenance:
    def test_mechano_nav_registered(self):
        assert "MechanoNav" in registered_names()
        pmap = provenance_for("MechanoNav")
        assert "k_mechano" in pmap
        assert isinstance(pmap["k_mechano"], Provenance)

    def test_mechano_synapse_registered(self):
        assert "MechanoSynapse" in registered_names()

    def test_nmda_stretch_params_have_provenance(self):
        pmap = provenance_for("NMDAReceptor")
        for key in ("alpha_NMDA", "T_half_stretch", "k_stretch"):
            assert key in pmap, f"NMDAReceptor.{key} missing provenance"

    @staticmethod
    def _public_numeric_fields(obj) -> set[str]:
        skip = {"name", "nmda"}
        result = set()
        for f_name in obj.__dataclass_fields__:
            if f_name.startswith("_") or f_name in skip:
                continue
            v = getattr(obj, f_name)
            if v is None or isinstance(v, (int, float)):
                result.add(f_name)
        return result

    def test_mechano_synapse_field_coverage(self):
        fields = self._public_numeric_fields(MechanoSynapse())
        prov_keys = set(provenance_for("MechanoSynapse").keys())
        missing = fields - prov_keys
        assert not missing, f"MechanoSynapse fields without provenance: {missing}"

    def test_nmda_field_coverage(self):
        fields = self._public_numeric_fields(NMDAReceptor())
        prov_keys = set(provenance_for("NMDAReceptor").keys())
        missing = fields - prov_keys
        assert not missing, f"NMDAReceptor fields without provenance: {missing}"

    def test_new_entries_typed_and_referenced(self):
        for name in ("MechanoNav", "MechanoSynapse"):
            for param, prov in provenance_for(name).items():
                assert isinstance(prov.source, ParamSource)
                if prov.source == ParamSource.LITERATURE:
                    assert prov.reference, f"{name}.{param} LITERATURE w/o reference"
                if prov.source == ParamSource.CALIBRATED:
                    assert prov.notes, f"{name}.{param} CALIBRATED w/o notes"
