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


# ---------------------------------------------------------------------------
# Passive Ca readout, state carry-over and time-varying tension drive
# ---------------------------------------------------------------------------
def _canonical_like_neuron(scenario: str = "S2") -> MulticompartmentMechanoHHNeuron:
    """Manuscript S2 (Piezo1 0.10 + TREK-1 0.04) or S4 (+inact, Ca, SK)."""
    from ultraneuron.calcium import CalciumDynamics, KCaChannel
    from ultraneuron.channel_library import TREK1

    p1 = copy.deepcopy(PIEZO1); p1.g_density = 0.10
    t1 = copy.deepcopy(TREK1); t1.g_density = 0.04
    kw = {}
    if scenario == "S4":
        p1.tau_inact = 12.0
        kw = dict(piezo_inactivation=True, calcium_dynamics=CalciumDynamics(),
                  kca_channels=[KCaChannel(g_density=0.3)], kca_compartments=(1,))
    return MulticompartmentMechanoHHNeuron.cortical_pyramidal_3comp(
        mechano_channels=[p1, t1], **kw)


_T_TEST = np.array([0.8, 2.0, 2.4, 3.0, 4.0, 5.5])


class TestPassiveCalciumReadout:
    def test_spikes_bit_identical_with_and_without_readout(self):
        from ultraneuron.multicompartment import simulate_voxels_3comp
        n = _canonical_like_neuron("S2")
        a = simulate_voxels_3comp(_T_TEST, n, duration=10.0, backend="numba")
        b = simulate_voxels_3comp(_T_TEST, n, duration=10.0, backend="numba",
                                  calcium_readout=True, ca_trace_dt_ms=0.5)
        assert a["spike_counts"].sum() > 0
        assert np.array_equal(a["spike_counts"], b["spike_counts"])
        assert np.array_equal(a["spike_offsets"], b["spike_offsets"])
        assert np.array_equal(a["spike_times_ms"], b["spike_times_ms"])
        # Every non-Ca state variable is bit-identical too.
        assert np.array_equal(a["final_state"][:, :-3], b["final_state"][:, :-3])
        assert n.calcium_dynamics is None, "readout must not stick to the neuron"
        assert b["ca_passive"] and "ca_peak_uM" not in a
        # Ca enters through Piezo1 (dendrite-heavy) and rises above rest.
        assert np.all(b["ca_peak_uM"][:, 0] > 0.05)
        assert np.all(b["ca_peak_uM"][:, 0] >= b["ca_peak_uM"][:, 1])
        assert b["ca_trace_uM"].shape == (6, 3, 20)
        assert np.allclose(b["ca_trace_uM"][:, :, -1], b["final_state"][:, -3:],
                           rtol=1e-6)
        # Peak is the running max of the trace (sampled subset ≤ peak).
        assert np.all(b["ca_trace_uM"].max(axis=2) <= b["ca_peak_uM"] * (1 + 1e-6))

    def test_map_with_spike_times_new_keys_do_not_change_old(self):
        from ultraneuron.multicompartment import (
            simulate_map_3comp_with_spike_times,
        )
        n = _canonical_like_neuron("S2")
        strain = (np.linspace(0.0, 0.022, 8) * 1.0).reshape(2, 2, 2)
        kw = dict(K_A=0.25, coupling_factor=1.0, dt=0.001, duration=8.0,
                  tension_threshold=0.5, backend="numba")
        r0 = simulate_map_3comp_with_spike_times(strain, n, **kw)
        r1 = simulate_map_3comp_with_spike_times(strain, n, calcium_readout=True,
                                                 ca_trace_dt_ms=1.0, **kw)
        for k in r0:
            assert np.array_equal(np.asarray(r0[k]), np.asarray(r1[k]),
                                  equal_nan=True), k
        assert r0["firing_map"].sum() > 0
        assert r1["ca_peak_map"].shape == (3, 2, 2, 2)
        elig = r0["tension_3d"] >= 0.5
        assert np.all(np.isfinite(r1["ca_peak_map"][:, elig]))
        assert np.all(np.isnan(r1["ca_peak_map"][:, ~elig]))
        n_fire = len(r0["spike_voxel_idx"])
        assert r1["ca_peak_firing"].shape == (n_fire, 3)
        assert r1["ca_trace_firing"].shape == (n_fire, 3, 8)
        ix = tuple(r0["spike_voxel_idx"].T)
        assert np.array_equal(r1["ca_peak_firing"][:, 0], r1["ca_peak_map"][0][ix])


class TestStateCarryOver:
    @pytest.mark.parametrize("scenario", ["S2", "S4"])
    @pytest.mark.parametrize("use_drive", [False, True])
    def test_two_windows_equal_one_long_window(self, scenario, use_drive):
        from ultraneuron.multicompartment import (
            simulate_voxels_3comp,
            tension_drive,
        )
        n = _canonical_like_neuron(scenario)
        dt = 0.001
        drv = tension_drive(dt, t_on_ms=8.0, t_off_ms=7.0) if use_drive else None
        full = simulate_voxels_3comp(_T_TEST, n, dt=dt, duration=15.0,
                                     backend="numba", drive=drv,
                                     calcium_readout=True)
        cut = 9500  # mid-OFF and mid-spike-train boundary (step index)
        w1 = simulate_voxels_3comp(_T_TEST, n, dt=dt, duration=cut * dt,
                                   backend="numba", calcium_readout=True,
                                   drive=None if drv is None else drv[:cut])
        w2 = simulate_voxels_3comp(_T_TEST, n, dt=dt, duration=15.0 - cut * dt,
                                   backend="numba", calcium_readout=True,
                                   drive=None if drv is None else drv[cut:],
                                   initial_state=w1["final_state"])
        assert w1["n_steps"] + w2["n_steps"] == full["n_steps"]
        assert full["spike_counts"].sum() > 0
        assert np.array_equal(full["final_state"], w2["final_state"])
        assert np.array_equal(full["spike_counts"],
                              w1["spike_counts"] + w2["spike_counts"])
        for i in range(len(_T_TEST)):
            t_full = full["spike_times_ms"][full["spike_offsets"][i]:full["spike_offsets"][i + 1]]
            t1 = w1["spike_times_ms"][w1["spike_offsets"][i]:w1["spike_offsets"][i + 1]]
            t2 = w2["spike_times_ms"][w2["spike_offsets"][i]:w2["spike_offsets"][i + 1]]
            steps_full = np.rint(t_full / dt).astype(int)
            steps_chain = np.concatenate([np.rint(t1 / dt), np.rint(t2 / dt) + cut]).astype(int)
            assert np.array_equal(steps_full, steps_chain)
        assert np.array_equal(np.maximum(w1["ca_peak_uM"], w2["ca_peak_uM"]),
                              full["ca_peak_uM"])

    def test_explicit_rest_init_matches_default(self):
        """Default init (None) is the true-rest state, bit-for-bit."""
        from ultraneuron.multicompartment import (
            rest_initial_state,
            simulate_voxels_3comp,
        )
        for scen in ("S2", "S4"):
            n = _canonical_like_neuron(scen)
            a = simulate_voxels_3comp(_T_TEST, n, duration=5.0, backend="numba")
            b = simulate_voxels_3comp(_T_TEST, n, duration=5.0, backend="numba",
                                      initial_state=rest_initial_state(n, _T_TEST))
            assert np.array_equal(a["final_state"], b["final_state"])
            assert np.array_equal(a["spike_times_ms"], b["spike_times_ms"])

    def test_explicit_peak_steady_init_matches_option(self):
        """peak_steady_initial_state == mechano_init='peak_steady' (legacy)."""
        from ultraneuron.multicompartment import (
            peak_steady_initial_state,
            simulate_voxels_3comp,
        )
        for scen in ("S2", "S4"):
            n = _canonical_like_neuron(scen)
            a = simulate_voxels_3comp(_T_TEST, n, duration=5.0, backend="numba",
                                      mechano_init="peak_steady")
            b = simulate_voxels_3comp(_T_TEST, n, duration=5.0, backend="numba",
                                      initial_state=peak_steady_initial_state(n, _T_TEST))
            assert np.array_equal(a["final_state"], b["final_state"])
            assert np.array_equal(a["spike_times_ms"], b["spike_times_ms"])

    def test_canonical_initial_state_is_deprecated_alias(self):
        from ultraneuron.multicompartment import (
            canonical_initial_state,
            peak_steady_initial_state,
        )
        n = _canonical_like_neuron("S4")
        with pytest.warns(DeprecationWarning, match="peak_steady_initial_state"):
            a = canonical_initial_state(n, _T_TEST)
        assert np.array_equal(a, peak_steady_initial_state(n, _T_TEST))

    def test_constant_drive_equals_no_drive_and_legacy(self):
        from ultraneuron.multicompartment import simulate_voxels_3comp
        n = _canonical_like_neuron("S2")
        a = simulate_voxels_3comp(_T_TEST, n, duration=6.0, backend="numba")
        b = simulate_voxels_3comp(_T_TEST, n, duration=6.0, backend="numba",
                                  drive=np.ones(6000))
        legacy = n._simulate_batch(_T_TEST, 0.001, 6000, np.zeros(3), backend="numba")
        assert np.array_equal(a["final_state"], b["final_state"])
        assert np.array_equal(a["spike_counts"], legacy)

    def test_off_drive_relaxes_towards_rest(self):
        from ultraneuron.multicompartment import (
            simulate_voxels_3comp,
            state_layout,
            tension_drive,
        )
        n = _canonical_like_neuron("S2")  # S4: SK keeps V hyperpolarised longer
        dt = 0.001
        lay = state_layout(2)
        for mode in ("exp", "step"):
            drv = tension_drive(dt, 5.0, 45.0, off_mode=mode, tau_relax_ms=5.0)
            r = simulate_voxels_3comp(np.array([5.0]), n, dt=dt, duration=50.0,
                                      backend="numba", drive=drv)
            st = r["final_state"][0]
            assert np.all(st[lay["w"]] < 1e-3), mode  # mechano gates closed
            assert abs(st[lay["V"]][1] + 70.0) < 1.0, mode

    def test_tension_drive_profile(self):
        from ultraneuron.multicompartment import (
            TAU_RELAX_MS_HALLE_DACC,
            tension_drive,
        )
        d = tension_drive(0.001, 20.0, 180.0, n_periods=2)
        assert d.shape == (400000,)
        assert np.all(d[:20001] == 1.0)
        assert np.isclose(d[20000 + 32420], np.exp(-32.420 / TAU_RELAX_MS_HALLE_DACC))
        assert np.array_equal(d[:200000], d[200000:])
        s = tension_drive(0.001, 20.0, 180.0, off_mode="step")
        assert np.all(s[20000:] == 0.0) and np.all(s[:20000] == 1.0)
        with pytest.raises(ValueError):
            tension_drive(0.001, off_mode="linear")

    def test_bad_initial_state_shape_rejected(self):
        from ultraneuron.multicompartment import simulate_voxels_3comp
        n = _canonical_like_neuron("S2")
        with pytest.raises(ValueError, match="initial_state"):
            simulate_voxels_3comp(_T_TEST, n, duration=1.0, backend="numba",
                                  initial_state=np.zeros((len(_T_TEST), 5)))


class TestMechanoInit:
    """Default ON-window initial condition is true rest (PI decision).

    Every theta burst starts from rest because the tension relaxes during
    the 180 ms OFF time (``scripts/run_burst_periodicity.py``); the legacy
    peak-tension steady state stays reachable via
    ``mechano_init="peak_steady"``.
    """

    def test_rest_state_values(self):
        from ultraneuron.multicompartment import (
            rest_initial_state,
            state_layout,
        )
        n = _canonical_like_neuron("S4")
        st = rest_initial_state(n, 4)
        lay = state_layout(2)
        _, w0, _, _ = n._channel_kinetics(np.zeros(1), 2)
        h0, _, _ = n._inactivation_kinetics(np.zeros(1), 2)
        assert st.shape == (4, 27)
        assert np.all(st[:, lay["V"]] == n.V_rest)
        m0, hh0, n0 = n._kinetics._steady_state_gates(n.V_rest)
        assert np.all(st[:, lay["m"]] == m0) and np.all(st[:, lay["n"]] == n0)
        assert np.all(st[:, lay["h"]] == hh0)
        assert np.array_equal(st[0, lay["w"]], np.tile(w0[:, 0], 3))
        assert np.array_equal(st[0, lay["h_mech"]], np.tile(h0[:, 0], 3))
        assert np.all(st[:, lay["Ca"]] == n.calcium_dynamics.Ca_rest)
        # Gates closed at rest (w_inf(0) << 1), inactivation fully available
        assert np.all(st[:, lay["w"]] < 1e-3)
        assert np.all(st[:, lay["h_mech"]] > 0.99)

    def test_invalid_mode_rejected(self):
        from ultraneuron.multicompartment import simulate_voxels_3comp
        n = _canonical_like_neuron("S2")
        with pytest.raises(ValueError, match="mechano_init"):
            n.simulate(dt=0.001, duration=0.01, tension_trace=2.0, mechano_init="peak")
        with pytest.raises(ValueError, match="mechano_init"):
            simulate_voxels_3comp(_T_TEST, n, duration=0.01, backend="numba",
                                  mechano_init="canonical")

    @pytest.mark.parametrize("scenario", ["S2", "S4"])
    def test_rest_vs_peak_steady_differ_and_rest_is_later(self, scenario):
        from ultraneuron.multicompartment import simulate_voxels_3comp
        n = _canonical_like_neuron(scenario)
        T = np.array([3.0, 4.0, 5.5])
        r = simulate_voxels_3comp(T, n, duration=10.0, backend="numba")
        p = simulate_voxels_3comp(T, n, duration=10.0, backend="numba",
                                  mechano_init="peak_steady")
        assert r["spike_counts"].sum() > 0
        assert not np.array_equal(r["final_state"], p["final_state"])
        # Channels must first open from rest: first spike no earlier
        for i in range(len(T)):
            tr = r["spike_times_ms"][r["spike_offsets"][i]:r["spike_offsets"][i + 1]]
            tp = p["spike_times_ms"][p["spike_offsets"][i]:p["spike_offsets"][i + 1]]
            if len(tr) and len(tp):
                assert tr[0] >= tp[0]

    def test_zero_tension_unaffected(self):
        """At T = 0 both modes coincide (rest == steady state)."""
        from ultraneuron.multicompartment import simulate_voxels_3comp
        n = _canonical_like_neuron("S4")
        T = np.zeros(2)
        a = simulate_voxels_3comp(T, n, duration=2.0, backend="numba")
        b = simulate_voxels_3comp(T, n, duration=2.0, backend="numba",
                                  mechano_init="peak_steady")
        assert np.array_equal(a["final_state"], b["final_state"])

    @pytest.mark.parametrize("mode", ["rest", "peak_steady"])
    def test_per_voxel_simulate_matches_batch(self, mode):
        """simulate() spike times agree with the batch kernel in both modes."""
        from ultraneuron.multicompartment import simulate_voxels_3comp
        n = _canonical_like_neuron("S2")
        T = np.array([2.4, 4.0, 5.5])
        dt = 0.001
        b = simulate_voxels_3comp(T, n, dt=dt, duration=6.0, backend="numba",
                                  mechano_init=mode)
        assert b["spike_counts"].sum() > 0
        for i in range(len(T)):
            r = n.simulate(dt=dt, duration=6.0, tension_trace=float(T[i]),
                           mechano_init=mode)
            a = np.rint(r["spike_times_per_compartment"][1] / dt).astype(int)
            bb = np.rint(b["spike_times_ms"][b["spike_offsets"][i]:
                                            b["spike_offsets"][i + 1]] / dt).astype(int)
            assert np.array_equal(a, bb), (mode, T[i], a, bb)

    def test_per_voxel_simulate_initial_mechano_current(self):
        """At t=0 the rest init has the T=0 mechano current, not the peak one."""
        n = _canonical_like_neuron("S2")
        r_rest = n.simulate(dt=0.001, duration=0.01, tension_trace=5.0)
        r_zero = n.simulate(dt=0.001, duration=0.01, tension_trace=0.0)
        r_peak = n.simulate(dt=0.001, duration=0.01, tension_trace=5.0,
                            mechano_init="peak_steady")
        assert np.array_equal(r_rest["I_mech"][0], r_zero["I_mech"][0])
        assert not np.allclose(r_rest["I_mech"][0], r_peak["I_mech"][0])

    def test_map_functions_thread_option(self):
        from ultraneuron.multicompartment import (
            simulate_map_3comp,
            simulate_map_3comp_with_spike_times,
            simulate_voxels_3comp,
        )
        n = _canonical_like_neuron("S2")
        # tension = K_A·alpha·strain = 250 N/m · strain → 0..5.5 mN/m
        strain = (np.array([0.0, 0.8, 2.0, 2.4, 3.0, 4.0, 5.0, 5.5]) / 250e3).reshape(2, 2, 2)
        kw = dict(K_A=0.25, coupling_factor=1000.0, dt=0.001, duration=6.0,
                  tension_threshold=0.5, backend="numba")
        for mode in ("rest", "peak_steady"):
            m = simulate_map_3comp(strain, n, mechano_init=mode, **kw)
            s = simulate_map_3comp_with_spike_times(strain, n, mechano_init=mode, **kw)
            assert np.array_equal(m["spike_count_map"], s["spike_count_map"])
            T = s["tension_3d"][s["tension_3d"] >= 0.5]
            v = simulate_voxels_3comp(T, n, dt=0.001, duration=6.0,
                                      backend="numba", mechano_init=mode)
            assert np.array_equal(s["spike_count_map"][s["tension_3d"] >= 0.5],
                                  v["spike_counts"])
            assert np.array_equal(
                np.rint(s["spike_times_ms"].astype(np.float64) / 0.001),
                np.rint(v["spike_times_ms"] / 0.001))
        d = simulate_map_3comp(strain, n, **kw)
        r = simulate_map_3comp(strain, n, mechano_init="rest", **kw)
        p = simulate_map_3comp(strain, n, mechano_init="peak_steady", **kw)
        assert np.array_equal(d["spike_count_map"], r["spike_count_map"])
        assert not np.array_equal(r["spike_count_map"], p["spike_count_map"])

    def test_rush_larsen_honours_option(self):
        n = _canonical_like_neuron("S2")
        T = np.array([2.4, 3.0, 4.0, 5.5])
        kw = dict(dt=0.001, n_steps=6000, I_stim_per_compartment=np.zeros(3),
                  backend="numba")
        rl_rest = n._simulate_batch(T, integrator="rl", **kw)
        rl_peak = n._simulate_batch(T, integrator="rl", mechano_init="peak_steady", **kw)
        rk_rest = n._simulate_batch(T, integrator="rk4", **kw)
        rk_peak = n._simulate_batch(T, integrator="rk4", mechano_init="peak_steady", **kw)
        assert not np.array_equal(rl_rest, rl_peak)
        # RL tracks RK4 within one spike per voxel in each mode
        assert np.all(np.abs(rl_rest - rk_rest) <= 1)
        assert np.all(np.abs(rl_peak - rk_peak) <= 1)

    def test_numpy_fallback_honours_option(self):
        """Time-varying NICE envelope → NumPy batch path; init must match."""
        from ultraneuron.nice import NICEModel
        p1 = copy.deepcopy(PIEZO1); p1.g_density = 0.10
        n = MulticompartmentMechanoHHNeuron.cortical_pyramidal_3comp(
            mechano_channels=[p1], nice_model=NICEModel())
        T = np.array([3.0, 5.5])
        n_steps = 3000
        env = np.zeros((n_steps, T.size))  # zero pressure: ΔC = 0
        kw = dict(dt=0.001, n_steps=n_steps, I_stim_per_compartment=np.zeros(3),
                  backend="numba")
        for mode in ("rest", "peak_steady"):
            np_path = n._simulate_batch(T, pressure_envelope=env, mechano_init=mode, **kw)
            fast = n._simulate_batch(T, pressure_envelope=0.0, mechano_init=mode, **kw)
            assert np.array_equal(np_path, fast), mode
