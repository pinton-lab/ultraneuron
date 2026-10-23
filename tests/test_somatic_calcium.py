"""Tests for the slow spike-driven somatic Ca²⁺ pool and the session read-out.

Model: d[Ca]/dt = -([Ca] - Ca_rest)/τ + ΔCa Σ_k δ(t - t_k), optional clamp.
Checks: single-spike response, superposition, clamp, CSR handling,
periodic analytic (geometric series) vs direct event-driven session,
exact summary maps, and provenance coverage.
"""

from __future__ import annotations

import numpy as np
import pytest

from ultraneuron import (
    CalciumDynamics,
    ParamSource,
    PulsedProtocol,
    SomaticCalcium,
    provenance_for,
    session_somatic_calcium,
    session_spike_train,
)
from ultraneuron.calcium import periodic_excess, spike_lists_to_csr

REST, TAU, DCA = 0.05, 700.0, 0.08


def _brute_force(times, t, pool):
    """Unclamped superposition of single-spike kernels."""
    t = np.asarray(t, float)
    e = np.zeros_like(t)
    for tk in times:
        e += np.where(t >= tk, pool.dCa_per_spike * np.exp(-(t - tk) / pool.tau_ms), 0.0)
    return pool.Ca_rest + e


class TestDefaults:
    def test_defaults(self):
        p = SomaticCalcium()
        assert p.Ca_rest == CalciumDynamics().Ca_rest == REST
        assert p.tau_ms == TAU
        assert p.dCa_per_spike == DCA
        assert p.Ca_max == 10.0

    def test_invalid(self):
        with pytest.raises(ValueError):
            SomaticCalcium(tau_ms=0.0)
        with pytest.raises(ValueError):
            SomaticCalcium(Ca_max=0.01)


class TestSingleSpike:
    def test_peak_and_decay(self):
        p = SomaticCalcium()
        t = np.array([-1.0, 0.0, 10.0, TAU, 2 * TAU, 5000.0])
        ca = p.from_spike_times(np.array([0.0]), t)
        assert ca.shape == t.shape
        assert ca[0] == pytest.approx(REST)
        assert ca[1] == pytest.approx(REST + DCA)  # right-continuous: jump at t_k
        np.testing.assert_allclose(ca[2:] - REST, DCA * np.exp(-t[2:] / TAU), rtol=1e-12)
        assert (ca[3] - REST) / DCA == pytest.approx(np.exp(-1.0))

    def test_off_grid_spike(self):
        """Spikes need not lie on the output grid."""
        p = SomaticCalcium()
        t = np.arange(0.0, 50.0, 1.0)
        ca = p.from_spike_times(np.array([3.37]), t)
        np.testing.assert_allclose(ca, _brute_force([3.37], t, p), rtol=1e-12)

    def test_initial_condition(self):
        p = SomaticCalcium()
        t = np.array([0.0, 100.0, 700.0])
        ca = p.from_spike_times(np.empty(0), t, Ca0=0.5, t0_ms=0.0)
        np.testing.assert_allclose(ca - REST, 0.45 * np.exp(-t / TAU), rtol=1e-12)


class TestSuperposition:
    def test_linear_superposition(self):
        p = SomaticCalcium(Ca_max=None)
        rng = np.random.default_rng(1)
        spikes = np.sort(rng.uniform(0, 2000, 25))
        t = np.linspace(-100, 5000, 777)
        np.testing.assert_allclose(p.from_spike_times(spikes, t), _brute_force(spikes, t, p), rtol=1e-11)

    def test_sum_of_parts(self):
        p = SomaticCalcium(Ca_max=None)
        a, b = np.array([1.0, 20.0, 300.0]), np.array([5.0, 900.0])
        t = np.linspace(0, 3000, 301)
        both = p.from_spike_times(np.concatenate([a, b]), t) - REST
        parts = (p.from_spike_times(a, t) - REST) + (p.from_spike_times(b, t) - REST)
        np.testing.assert_allclose(both, parts, rtol=1e-12, atol=1e-15)

    def test_unsorted_input(self):
        p = SomaticCalcium()
        t = np.linspace(0, 500, 51)
        np.testing.assert_allclose(
            p.from_spike_times(np.array([30.0, 10.0, 20.0]), t),
            p.from_spike_times(np.array([10.0, 20.0, 30.0]), t),
        )


class TestClamp:
    def test_clamp_caps_value(self):
        p = SomaticCalcium(dCa_per_spike=1.0, Ca_max=2.0)
        spikes = np.arange(10) * 1.0  # 10 spikes in 10 ms -> would reach ~10 uM
        t = np.arange(0.0, 20.0, 0.25)
        ca = p.from_spike_times(spikes, t)
        assert ca.max() == pytest.approx(2.0)
        assert ca.max() <= 2.0 + 1e-12
        unclamped = SomaticCalcium(dCa_per_spike=1.0, Ca_max=None).from_spike_times(spikes, t)
        assert unclamped.max() > 9.0
        # after the last spike the clamped pool decays from Ca_max
        np.testing.assert_allclose(ca[t >= 9] - REST, (2.0 - REST) * np.exp(-(t[t >= 9] - 9) / TAU), rtol=1e-12)

    def test_clamp_inactive_below_cap(self):
        spikes = np.array([0.0, 50.0, 100.0])
        t = np.linspace(0, 1000, 101)
        np.testing.assert_allclose(
            SomaticCalcium().from_spike_times(spikes, t),
            SomaticCalcium(Ca_max=None).from_spike_times(spikes, t),
        )


class TestCSR:
    def test_csr_shapes_and_rows(self):
        lists = [np.array([1.0, 5.0]), np.empty(0), np.array([2.0]), np.array([0.5, 1.5, 3.0])]
        off, st = spike_lists_to_csr(lists)
        assert off.tolist() == [0, 2, 2, 3, 6]
        p = SomaticCalcium()
        t = np.linspace(0, 100, 41)
        ca = p.from_csr(off, st, t)
        assert ca.shape == (4, 41)
        for i, s in enumerate(lists):
            np.testing.assert_allclose(ca[i], _brute_force(s, t, p), rtol=1e-12)
        np.testing.assert_allclose(ca[1], REST)
        # sequence input returns (n_vox, n_t) too
        assert p.from_spike_times(lists, t).shape == (4, 41)

    def test_simulator_dict_layout(self):
        """Matches the simulate_map_3comp_with_spike_times return keys."""
        res = {
            "spike_voxel_idx": np.array([[1, 2, 3], [4, 5, 6]], np.int32),
            "spike_offsets": np.array([0, 1, 3], np.int64),
            "spike_times_ms": np.array([2.5, 1.0, 7.0], np.float32),
        }
        t = np.array([0.0, 3.0, 10.0])
        ca = SomaticCalcium().from_spike_results(res, t)
        assert ca.shape == (2, 3)
        assert ca[0, 0] == pytest.approx(REST)
        assert ca[0, 1] == pytest.approx(REST + DCA * np.exp(-0.5 / TAU), rel=1e-6)

    def test_empty_layouts(self):
        p = SomaticCalcium()
        t = np.linspace(0, 10, 5)
        assert p.from_csr(np.zeros(1, np.int64), np.empty(0), t).shape == (0, 5)
        np.testing.assert_allclose(p.from_csr(np.array([0, 0, 0]), np.empty(0), t), REST)

    def test_bad_layouts(self):
        p = SomaticCalcium()
        t = np.linspace(0, 10, 5)
        with pytest.raises(ValueError):
            p.from_csr(np.array([0, 2]), np.array([1.0]), t)
        with pytest.raises(ValueError):
            p.from_csr(np.array([1, 2]), np.array([1.0]), t)
        with pytest.raises(ValueError):
            p.from_csr(np.array([0, 2, 1]), np.array([1.0, 2.0]), t)


# ---------------------------------------------------------------------------
# Protocol / session
# ---------------------------------------------------------------------------
BURST = [np.array([2.8, 7.1, 12.0, 17.5]), np.array([3.0]), np.empty(0), np.array([0.0, 19.9])]


class TestProtocol:
    def test_yaakub_defaults(self):
        pr = PulsedProtocol()
        assert (pr.burst_ms, pr.period_ms, pr.n_bursts) == (20.0, 200.0, 400)
        assert pr.session_ms == 80000.0
        assert pr.prf_hz == pytest.approx(5.0)
        assert pr.duty_cycle == pytest.approx(0.1)

    def test_session_train_periodic(self):
        pr = PulsedProtocol(n_bursts=3)
        off1, st1 = spike_lists_to_csr(BURST)
        off, st = session_spike_train(off1, st1, pr)
        assert np.diff(off).tolist() == [12, 3, 0, 6]
        np.testing.assert_allclose(st[off[0]:off[1]], np.concatenate([BURST[0] + 200 * b for b in range(3)]))
        np.testing.assert_allclose(st[off[3]:off[4]], [0.0, 19.9, 200.0, 219.9, 400.0, 419.9])

    def test_per_burst_hook(self):
        pr = PulsedProtocol(n_bursts=2)
        b0 = spike_lists_to_csr([np.array([1.0]), np.array([2.0, 3.0])])
        b1 = spike_lists_to_csr([np.empty(0), np.array([4.0])])
        off, st = session_spike_train(None, None, pr, per_burst=[b0, b1])
        assert np.diff(off).tolist() == [1, 3]
        np.testing.assert_allclose(st, [1.0, 2.0, 3.0, 204.0])
        off2, st2 = session_spike_train(None, None, pr, per_burst=lambda b: (b0, b1)[b])
        np.testing.assert_array_equal(off, off2)
        np.testing.assert_allclose(st, st2)

    def test_spikes_outside_period_rejected(self):
        with pytest.raises(ValueError):
            session_spike_train(np.array([0, 1]), np.array([250.0]), PulsedProtocol())


class TestPeriodicAnalyticVsDirect:
    @pytest.mark.parametrize("n_bursts", [1, 7, 60])
    def test_trace_and_summaries(self, n_bursts):
        pr = PulsedProtocol(n_bursts=n_bursts)
        off1, st1 = spike_lists_to_csr(BURST)
        t = np.arange(-500.0, pr.session_ms + 6000.0, 3.7)
        ana = session_somatic_calcium(off1, st1, pr, t_ms=t, method="analytic")
        dirc = session_somatic_calcium(off1, st1, pr, t_ms=t, method="direct")
        assert ana.method == "analytic" and dirc.method == "direct"
        assert ana.ca.shape == dirc.ca.shape == (4, t.size)
        np.testing.assert_allclose(ana.ca, dirc.ca, rtol=2e-6, atol=1e-7)
        for f in ("peak", "plateau", "ca_end", "t_return_ms"):
            np.testing.assert_allclose(getattr(ana, f), getattr(dirc, f), rtol=1e-9, atol=1e-9, err_msg=f)

    def test_against_brute_force(self):
        pr = PulsedProtocol(n_bursts=12)
        pool = SomaticCalcium()
        session = np.concatenate([BURST[0] + b * pr.period_ms for b in range(pr.n_bursts)])
        t = np.linspace(0, 5000, 1001)
        e = periodic_excess(pool, *spike_lists_to_csr([BURST[0]]), pr, t)[0]
        np.testing.assert_allclose(e + REST, _brute_force(session, t, pool), rtol=1e-11)

    def test_steady_state_plateau(self):
        """Mean over a period in steady state = Ca_rest + n ΔCa τ / P exactly."""
        pr = PulsedProtocol()
        off1, st1 = spike_lists_to_csr(BURST)
        r = session_somatic_calcium(off1, st1, pr, t_ms=np.array([0.0]))
        n = np.array([4, 1, 0, 2])
        np.testing.assert_allclose(r.plateau, REST + n * DCA * TAU / pr.period_ms, rtol=1e-9)
        assert r.plateau[1] == pytest.approx(0.33)
        assert r.t_return_ms[2] == 0.0

    def test_exact_plateau_vs_grid_mean(self):
        pr = PulsedProtocol(n_bursts=20)
        off1, st1 = spike_lists_to_csr(BURST)
        t_end = pr.session_ms
        t = np.linspace(t_end - pr.period_ms, t_end, 200001)
        r = session_somatic_calcium(off1, st1, pr, t_ms=t, method="direct")
        np.testing.assert_allclose(r.plateau, np.trapezoid(r.ca.astype(float), t, axis=1) / pr.period_ms, rtol=1e-5)

    def test_return_time(self):
        pr = PulsedProtocol(n_bursts=1)
        r = session_somatic_calcium(np.array([0, 1]), np.array([0.0]), pr, t_ms=np.array([0.0]))
        expect = TAU * np.log(DCA / (0.1 * REST)) - pr.period_ms
        assert r.t_return_ms[0] == pytest.approx(expect)

    def test_auto_falls_back_to_direct_when_clamped(self):
        pr = PulsedProtocol(n_bursts=50)
        pool = SomaticCalcium(dCa_per_spike=1.0, Ca_max=5.0)
        off1, st1 = spike_lists_to_csr(BURST)
        r = session_somatic_calcium(off1, st1, pr, pool, t_ms=np.array([0.0, 1000.0]))
        assert r.method == "direct"
        assert r.peak.max() == pytest.approx(5.0)
        r2 = session_somatic_calcium(off1, st1, pr, t_ms=np.array([0.0]))
        assert r2.method == "analytic"

    def test_per_burst_forces_direct(self):
        pr = PulsedProtocol(n_bursts=2)
        b = spike_lists_to_csr([np.array([1.0])])
        r = session_somatic_calcium(None, None, pr, per_burst=[b, b], t_ms=np.array([0.0]))
        assert r.method == "direct"
        with pytest.raises(ValueError):
            session_somatic_calcium(None, None, pr, per_burst=[b, b], method="analytic")


class TestSomaticProvenance:
    def test_registered_and_assumption(self):
        pmap = provenance_for("SomaticCalcium")
        assert set(pmap) == {"tau_ms", "dCa_per_spike", "Ca_max"}
        for k, prov in pmap.items():
            assert prov.source == ParamSource.ASSUMPTION, k
            assert "not fitted" in prov.notes, k

    def test_field_coverage(self):
        """Every numeric field is covered; Ca_rest is inherited from CalciumDynamics."""
        p = SomaticCalcium()
        fields = {f for f in p.__dataclass_fields__ if not f.startswith("_")}
        assert fields - set(provenance_for("SomaticCalcium")) == {"Ca_rest"}
        assert "Ca_rest" in provenance_for("CalciumDynamics")
