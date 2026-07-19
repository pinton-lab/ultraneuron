"""Tests for the astrocytic TRPA1 → Ca²⁺ → glutamate → NMDA relay.

Mechanism and calibration anchors from Oh et al. 2019 (Curr Biol 29:3386,
"Ultrasonic Neuromodulation via Astrocytic TRPA1"):

  (i)   TRPA1 is the necessary sensor — knockout / antagonist (g_density
        → 0) abolishes the relay output.
  (ii)  AP5 condition — zeroing NMDA g_max abolishes neuronal excitation
        while astrocytic Ca²⁺ / glutamate persist.
  (iii) The relay is SLOW — astrocytic Ca²⁺ rises over ~0.1–1 s, not per
        carrier cycle.
  (iv)  The cheap scalar entry point excitation_drive(peak_tension, duty,
        on_time) is monotone in the drive and grows with duty × on-time.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from ultraneuron import (
    TRPA1,
    AstrocyteRelay,
    NMDAReceptor,
    ParamSource,
    provenance_for,
    registered_names,
    simulate_astrocyte_relay,
)


# ---------------------------------------------------------------------------
# TRPA1 channel preset
# ---------------------------------------------------------------------------
class TestTRPA1Channel:
    def test_is_mechanosensitive_cation(self):
        assert TRPA1.channel_type == "cation"
        assert TRPA1.E_rev == pytest.approx(0.0, abs=1e-9)

    def test_ca_permeable(self):
        # Non-selective but high Ca²⁺ permeability (P_Ca/P_Na ≈ 5–8).
        assert 0.10 < TRPA1.Ca_permeability_fraction < 0.30

    def test_boltzmann_tension_gate(self):
        # Half-activation at T_half; rises with tension.
        assert TRPA1.open_probability(TRPA1.T_half) == pytest.approx(0.5, abs=1e-6)
        assert TRPA1.open_probability(TRPA1.T_half + 5.0) > 0.5
        assert TRPA1.open_probability(0.0) < 0.5


# ---------------------------------------------------------------------------
# NMDA receptor
# ---------------------------------------------------------------------------
class TestNMDAReceptor:
    def test_mg_block_voltage_dependence(self):
        nmda = NMDAReceptor()
        # Unblock grows monotonically with depolarisation.
        assert nmda.mg_unblock(-80.0) < nmda.mg_unblock(-40.0) < nmda.mg_unblock(0.0)
        assert 0.0 < nmda.mg_unblock(-80.0) < 0.2
        assert nmda.mg_unblock(40.0) > 0.8

    def test_glutamate_saturation(self):
        nmda = NMDAReceptor(K_glu=2.0)
        assert nmda.glutamate_binding(0.0) == pytest.approx(0.0)
        assert nmda.glutamate_binding(2.0) == pytest.approx(0.5, abs=1e-9)
        assert nmda.glutamate_binding(1e6) == pytest.approx(1.0, abs=1e-3)

    def test_ap5_zeroes_current(self):
        ap5 = NMDAReceptor(g_max=0.0)
        assert ap5.current(-50.0, 10.0) == pytest.approx(0.0)

    def test_inward_current_when_glutamate_present(self):
        nmda = NMDAReceptor()
        # Depolarising (positive / inward) at sub-threshold V with glutamate.
        assert nmda.current(-50.0, 10.0) > 0.0


# ---------------------------------------------------------------------------
# AstrocyteRelay — Ca²⁺ pool + Hill glutamate
# ---------------------------------------------------------------------------
class TestAstrocyteRelay:
    def test_ca_rises_with_tension(self):
        relay = AstrocyteRelay()
        tc_low = relay.relay_timecourse(0.0, dt=1.0, duration_ms=1500.0)
        tc_high = relay.relay_timecourse(10.0, dt=1.0, duration_ms=1500.0)
        assert tc_high["Ca"].max() > tc_low["Ca"].max()
        # Physiological astrocytic peak ~ a few hundred nM to ~1 μM.
        assert 0.2 < tc_high["Ca"].max() < 3.0

    def test_glutamate_hill_in_calcium(self):
        relay = AstrocyteRelay()
        g_lo = relay.glutamate_from_ca(0.05)
        g_hi = relay.glutamate_from_ca(2.0)
        assert g_hi > g_lo
        assert g_hi <= relay.glu_max + 1e-9
        # Half-release near glu_Ca_half.
        assert relay.glutamate_from_ca(relay.glu_Ca_half) == pytest.approx(
            0.5 * relay.glu_max, rel=1e-6
        )

    def test_trpa1_knockout_abolishes_relay(self):
        """TRPA1 KO / antagonist (g_density → 0) abolishes the relay output."""
        ko = AstrocyteRelay()
        ko.trpa1 = copy.deepcopy(TRPA1)
        ko.trpa1.g_density = 0.0
        tc = ko.relay_timecourse(10.0, dt=1.0, duration_ms=1500.0)
        # Ca stays at rest; glutamate ≈ 0.
        assert tc["Ca"].max() == pytest.approx(ko.calcium.Ca_rest, abs=1e-9)
        intact = AstrocyteRelay().relay_timecourse(10.0, dt=1.0, duration_ms=1500.0)
        assert tc["glutamate"].max() < 0.05 * intact["glutamate"].max()

    def test_relay_is_slow(self):
        """Astrocytic Ca²⁺ rises over 0.1–1 s, not per carrier cycle (μs)."""
        relay = AstrocyteRelay()
        dt = 1.0
        tc = relay.relay_timecourse(10.0, dt=dt, duration_ms=2000.0)
        Ca = tc["Ca"]
        peak = Ca.max()
        half_level = relay.calcium.Ca_rest + 0.5 * (peak - relay.calcium.Ca_rest)
        idx = int(np.argmax(Ca >= half_level))
        t_half_ms = tc["t"][idx]
        # Slow: tens-of-ms to seconds, nowhere near a 500 kHz carrier cycle (2 μs).
        assert 50.0 < t_half_ms < 2000.0


# ---------------------------------------------------------------------------
# Cheap scalar entry point — the downstream interface contract
# ---------------------------------------------------------------------------
class TestExcitationDrive:
    def test_monotone_in_peak_tension(self):
        relay = AstrocyteRelay()
        d_lo = relay.excitation_drive(2.0, duty_cycle=0.15, on_time_s=1.0)
        d_hi = relay.excitation_drive(10.0, duty_cycle=0.15, on_time_s=1.0)
        assert d_hi > d_lo

    def test_grows_with_on_time(self):
        """Slow Ca²⁺ accumulation → longer sonication gives a larger drive."""
        relay = AstrocyteRelay()
        d_short = relay.excitation_drive(8.0, duty_cycle=0.15, on_time_s=0.05)
        d_long = relay.excitation_drive(8.0, duty_cycle=0.15, on_time_s=2.0)
        assert d_long > d_short

    def test_grows_with_duty(self):
        relay = AstrocyteRelay()
        d_lo = relay.excitation_drive(8.0, duty_cycle=0.05, on_time_s=1.0)
        d_hi = relay.excitation_drive(8.0, duty_cycle=0.50, on_time_s=1.0)
        assert d_hi > d_lo

    def test_in_unit_interval(self):
        relay = AstrocyteRelay()
        d = relay.excitation_drive(8.0, duty_cycle=0.15, on_time_s=80.0)
        assert 0.0 <= d <= 1.0

    def test_knockout_drive_near_zero(self):
        intact = AstrocyteRelay().excitation_drive(8.0, duty_cycle=0.15, on_time_s=1.0)
        ko = AstrocyteRelay()
        ko.trpa1 = copy.deepcopy(TRPA1)
        ko.trpa1.g_density = 0.0
        d_ko = ko.excitation_drive(8.0, duty_cycle=0.15, on_time_s=1.0)
        assert d_ko < 0.1 * intact


# ---------------------------------------------------------------------------
# Full forward model
# ---------------------------------------------------------------------------
class TestSimulateRelay:
    def test_nmda_drives_depolarisation(self):
        res = simulate_astrocyte_relay(
            peak_tension_mNm=8.0, dt=0.025, duration_ms=300.0,
        )
        assert res["glutamate"].max() > 1.0  # relay produced glutamate
        assert res["depolarisation"] > 2.0   # neuron depolarised via NMDA

    def test_ap5_abolishes_excitation_but_relay_persists(self):
        """AP5 (NMDA g_max → 0): no neuronal excitation, Ca²⁺/glutamate persist."""
        ap5 = NMDAReceptor(g_max=0.0)
        res = simulate_astrocyte_relay(
            peak_tension_mNm=8.0, nmda=ap5, dt=0.025, duration_ms=300.0,
        )
        # Gliotransmission upstream is unaffected.
        assert res["Ca_astro"].max() > 0.2
        assert res["glutamate"].max() > 1.0
        # But the neuron is not excited.
        assert abs(res["depolarisation"]) < 0.5
        assert res["n_spikes"] == 0

    def test_ap5_vs_intact_depolarisation(self):
        intact = simulate_astrocyte_relay(
            peak_tension_mNm=8.0, dt=0.025, duration_ms=300.0,
        )
        ap5 = simulate_astrocyte_relay(
            peak_tension_mNm=8.0, nmda=NMDAReceptor(g_max=0.0),
            dt=0.025, duration_ms=300.0,
        )
        assert intact["depolarisation"] > ap5["depolarisation"] + 1.0


# ---------------------------------------------------------------------------
# Provenance coverage
# ---------------------------------------------------------------------------
class TestAstrocyteProvenance:
    def test_components_registered(self):
        for key in ("TRPA1", "AstrocyteRelay", "NMDAReceptor"):
            assert key in registered_names(), f"missing provenance for {key!r}"

    def test_estimate_source_used_for_guesses(self):
        """Guessed TRPA1 mechanogating params are flagged ESTIMATE."""
        prov = provenance_for("TRPA1")
        assert prov["T_half"].source == ParamSource.ESTIMATE
        assert prov["delta_A"].source == ParamSource.ESTIMATE

    def test_every_entry_typed(self):
        for key in ("TRPA1", "AstrocyteRelay", "NMDAReceptor"):
            for pname, prov in provenance_for(key).items():
                assert prov.source in ParamSource, f"{key}.{pname} bad source"
                if prov.source == ParamSource.LITERATURE:
                    assert prov.reference, f"{key}.{pname} LITERATURE w/o reference"
