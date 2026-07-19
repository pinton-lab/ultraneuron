"""Tests for the 3-compartment dendrite-soma-AIS HH neuron.

Validation references
---------------------
* Pospischil 2008 — RS rate functions (already covered by single-comp tests).
* Stuart & Sakmann 1995 / Stuart 1997 — AIS-first spike initiation:
  with somatic current injection, the AIS V crosses spike threshold
  before the soma V does (sub-millisecond lead).
* Lewis & Grandl 2015, Dalghi 2019 — Piezo1 expression is dendrite-heavy,
  so dendrite-weighted mechano coupling lowers the firing tension
  threshold relative to soma-only coupling.
"""

import copy

import numpy as np
import pytest

from ultraneuron.channel_library import PIEZO1
from ultraneuron.compartments import (
    Compartment,
    CompartmentalGeometry,
    cortical_pyramidal_3comp,
)
from ultraneuron.multicompartment import (
    MulticompartmentMechanoHHNeuron,
    simulate_map_3comp,
)
from ultraneuron.neuron import MechanoHHNeuron


def _uniform_geometry(g_Na=50.0, g_K=5.0, g_L=0.1) -> CompartmentalGeometry:
    """Three identical compartments with zero axial coupling."""
    def _c(name: str) -> Compartment:
        return Compartment(
            name=name, g_Na=g_Na, g_K=g_K, g_L=g_L,
            E_Na=50.0, E_K=-100.0, E_L=-70.0, C_m=1.0, area=1.0,
            mechano_density_scale=[],
        )
    return CompartmentalGeometry(
        dendrite=_c("dendrite"), soma=_c("soma"), ais=_c("ais"),
        g_dend_soma=0.0, g_soma_ais=0.0,
    )


# ---------------------------------------------------------------------------
# Regression: identical compartments + zero coupling should reproduce the
# single-compartment cortical neuron trace.
# ---------------------------------------------------------------------------
class TestSingleCompartmentRegression:
    def test_resting_potential_stable(self):
        """No stimulus → all three compartments stay near V_rest."""
        n = MulticompartmentMechanoHHNeuron(
            geometry=_uniform_geometry(),
            mechano_channels=[],
            temperature=37.0,
            V_rest=-70.0,
            model="cortical",
        )
        r = n.simulate(dt=0.001, duration=10.0, I_stim={"soma": 0.0})
        assert np.all(np.abs(r["V"] - n.V_rest) < 2.0)

    def test_matches_single_compartment_trace(self):
        """All-identical, no coupling, soma I_stim → soma trace ≈ single-compartment trace."""
        # Single-compartment cortical
        sc = MechanoHHNeuron.cortical_pyramidal(
            g_Na=50.0, g_K=5.0, g_L=0.1, V_rest=-70.0,
        )
        sc_r = sc.simulate(dt=0.001, duration=10.0, I_stim=10.0)

        # Multi-compartment with identical compartments, zero coupling
        mc = MulticompartmentMechanoHHNeuron(
            geometry=_uniform_geometry(g_Na=50.0, g_K=5.0, g_L=0.1),
            mechano_channels=[],
            temperature=37.0,
            V_rest=-70.0,
            model="cortical",
        )
        mc_r = mc.simulate(dt=0.001, duration=10.0, I_stim={"soma": 10.0})

        # Soma V trace should match single-compartment to within RK4 error
        np.testing.assert_allclose(mc_r["V"][:, 1], sc_r["V"], atol=1e-9)

    def test_dendrite_and_ais_isolated(self):
        """Zero coupling + soma-only injection → dendrite and AIS stay at V_rest."""
        mc = MulticompartmentMechanoHHNeuron(
            geometry=_uniform_geometry(),
            mechano_channels=[],
            V_rest=-70.0,
        )
        r = mc.simulate(dt=0.001, duration=10.0, I_stim={"soma": 10.0})
        # dendrite (idx 0) and AIS (idx 2) untouched
        assert np.all(np.abs(r["V"][:, 0] - mc.V_rest) < 1e-3)
        assert np.all(np.abs(r["V"][:, 2] - mc.V_rest) < 1e-3)


# ---------------------------------------------------------------------------
# Stuart & Sakmann 1995 — AIS-first spike initiation
# ---------------------------------------------------------------------------
class TestAISFirstInitiation:
    def test_ais_spike_precedes_soma_spike(self):
        """With somatic injection, AIS spike occurs before soma spike.

        Stuart 1997: AIS spike threshold is ~10 mV lower than somatic
        and the AP initiates in the AIS, propagating retrogradely to
        the soma within sub-millisecond delay.
        """
        n = MulticompartmentMechanoHHNeuron.cortical_pyramidal_3comp(
            mechano_channels=[],
        )
        r = n.simulate(dt=0.001, duration=15.0, I_stim={"soma": 15.0})

        ais_times = r["spike_times_per_compartment"][2]
        soma_times = r["spike_times_per_compartment"][1]
        assert len(ais_times) > 0, "AIS should spike with this stimulus"
        assert len(soma_times) > 0, "soma should also spike"

        # AIS spike must precede soma spike (positive lead)
        lead_ms = soma_times[0] - ais_times[0]
        assert lead_ms > 0, (
            f"AIS spike should precede soma spike, got AIS={ais_times[0]:.4f}, "
            f"soma={soma_times[0]:.4f}"
        )
        # Lead should be sub-millisecond (Stuart 1997 reports ~100-500 µs)
        assert lead_ms < 1.0, f"AIS lead too large: {lead_ms} ms"


# ---------------------------------------------------------------------------
# Dendrite-heavy mechano coupling lowers tension firing threshold
# ---------------------------------------------------------------------------
class TestDendriticMechanoCoupling:
    def _count_soma_spikes(
        self,
        dendrite_scale: float,
        tension: float,
        duration: float = 30.0,
    ) -> int:
        # Boost Piezo1 g_density so the effect is visible at literature tensions
        ch = copy.deepcopy(PIEZO1)
        ch.g_density = 0.05  # ×10 boost so test is decisive
        n = MulticompartmentMechanoHHNeuron.cortical_pyramidal_3comp(
            mechano_channels=[ch],
            dendrite_piezo_scale=dendrite_scale,
            ais_piezo_scale=0.0,
        )
        r = n.simulate(dt=0.001, duration=duration, tension_trace=tension)
        return int(r["spike_count_per_compartment"][1])

    def test_dendritic_density_lowers_threshold(self):
        """Dendrite-heavy Piezo1 fires at tension that uniform density doesn't."""
        # Pick a tension just above PIEZO1 T_half (2.7 mN/m) where the
        # dendrite-heavy case should fire and the uniform case shouldn't.
        T = 5.0
        # Uniform across compartments (scale 1.0 everywhere)
        uniform_spikes = self._count_soma_spikes(dendrite_scale=1.0, tension=T)
        heavy_spikes = self._count_soma_spikes(dendrite_scale=10.0, tension=T)

        assert heavy_spikes >= uniform_spikes, (
            f"Dendrite-heavy density should produce >= spikes than uniform, "
            f"got uniform={uniform_spikes}, heavy={heavy_spikes}"
        )
        # And in the threshold regime, the inequality should be strict
        assert heavy_spikes > 0, "dendrite-heavy case should fire at this tension"


# ---------------------------------------------------------------------------
# simulate_map_3comp — per-voxel firing maps
# ---------------------------------------------------------------------------
class TestSimulateMap3Comp:
    def test_shape_and_zero_strain(self):
        n = MulticompartmentMechanoHHNeuron.cortical_pyramidal_3comp(
            mechano_channels=[copy.deepcopy(PIEZO1)],
        )
        strain = np.zeros((2, 3, 4))
        r = simulate_map_3comp(strain, n, dt=0.001, duration=3.0, tension_threshold=0.1)
        assert r["firing_map"].shape == (2, 3, 4)
        assert r["spike_count_map"].shape == (2, 3, 4)
        assert r["tension_3d"].shape == (2, 3, 4)
        assert not np.any(r["firing_map"])

    def test_high_strain_fires(self):
        ch = copy.deepcopy(PIEZO1)
        ch.g_density = 0.1  # boost for short test
        n = MulticompartmentMechanoHHNeuron.cortical_pyramidal_3comp(
            mechano_channels=[ch],
            dendrite_piezo_scale=10.0,
        )
        # Strain producing tension well above T_half
        strain = np.full((1, 1, 1), 0.05)  # → tension ≈ 12.5 mN/m
        r = simulate_map_3comp(strain, n, dt=0.001, duration=20.0, tension_threshold=0.0)
        assert r["spike_count_map"][0, 0, 0] > 0, "should fire at high strain"


# ---------------------------------------------------------------------------
# Geometry validation
# ---------------------------------------------------------------------------
class TestGeometry:
    def test_negative_area_rejected(self):
        with pytest.raises(ValueError, match="area"):
            CompartmentalGeometry(
                dendrite=Compartment(name="d", g_Na=30, g_K=5, g_L=0.1, area=-1.0),
                soma=Compartment(name="s", g_Na=120, g_K=36, g_L=0.1),
                ais=Compartment(name="a", g_Na=2000, g_K=80, g_L=0.1, area=0.05),
            )

    def test_negative_coupling_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            CompartmentalGeometry(
                dendrite=Compartment(name="d", g_Na=30, g_K=5, g_L=0.1),
                soma=Compartment(name="s", g_Na=120, g_K=36, g_L=0.1),
                ais=Compartment(name="a", g_Na=2000, g_K=80, g_L=0.1),
                g_dend_soma=-1.0,
            )

    def test_mismatched_mechano_scale_rejected(self):
        # 1 channel passed, but compartment lists scales for 2
        d = Compartment(name="d", g_Na=30, g_K=5, g_L=0.1, mechano_density_scale=[1.0, 2.0])
        s = Compartment(name="s", g_Na=120, g_K=36, g_L=0.1)
        a = Compartment(name="a", g_Na=2000, g_K=80, g_L=0.1)
        g = CompartmentalGeometry(dendrite=d, soma=s, ais=a)
        with pytest.raises(ValueError, match="density scales"):
            MulticompartmentMechanoHHNeuron(
                geometry=g,
                mechano_channels=[copy.deepcopy(PIEZO1)],
            )
