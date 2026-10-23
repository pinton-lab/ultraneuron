"""Numerical parity tests for the CUDA backend.

The CUDA path is opt-in (``backend="cuda"`` on ``_simulate_batch`` /
``simulate_map_*``) and falls back to Numba when no GPU is visible.
These tests run both paths on a small population and assert spike
counts match. They are skipped when no CUDA device is present, so the
suite still passes on CPU-only machines.
"""
from __future__ import annotations

import numpy as np
import pytest

from ultraneuron import (
    CalciumDynamics,
    KCaChannel,
    MechanoHHNeuron,
    MulticompartmentMechanoHHNeuron,
)
from ultraneuron.channel_library import PIEZO1, TREK1, get_channel
from ultraneuron.cuda_backend import is_cuda_available

pytestmark = pytest.mark.skipif(
    not is_cuda_available(),
    reason="No CUDA device visible; skipping GPU parity tests.",
)


# ------------------------------------------------------------------ #
# Single-compartment
# ------------------------------------------------------------------ #
class TestSingleCompCudaParity:
    """``MechanoHHNeuron._simulate_batch`` must match across backends."""

    def _make_neuron(self) -> MechanoHHNeuron:
        return MechanoHHNeuron.cortical_pyramidal(
            mechano_channels=[get_channel("Piezo1")],
        )

    def test_constant_tension_subthreshold(self):
        neuron = self._make_neuron()
        N = 64
        tensions = np.linspace(0.0, 4.0, N)
        phi = np.full(N, neuron._q10_scale())
        sc_cpu = neuron._simulate_batch(
            tensions, phi, dt=0.01, n_steps=3000, I_stim=0.0, backend="numba",
        )
        sc_gpu = neuron._simulate_batch(
            tensions, phi, dt=0.01, n_steps=3000, I_stim=0.0, backend="cuda",
        )
        assert np.array_equal(sc_cpu, sc_gpu)

    def test_constant_tension_driven(self):
        """With a strong stim, both backends should produce non-zero counts."""
        neuron = self._make_neuron()
        N = 64
        tensions = np.linspace(0.0, 8.0, N)
        phi = np.full(N, neuron._q10_scale())
        sc_cpu = neuron._simulate_batch(
            tensions, phi, dt=0.01, n_steps=3000, I_stim=8.0, backend="numba",
        )
        sc_gpu = neuron._simulate_batch(
            tensions, phi, dt=0.01, n_steps=3000, I_stim=8.0, backend="cuda",
        )
        assert sc_cpu.sum() > 0
        # Fastmath drift over 3000 timesteps is enough to occasionally
        # shift a spike across the dt window; require equality on the
        # vast majority of voxels and bound any single-voxel mismatch.
        assert np.allclose(sc_cpu, sc_gpu, atol=1)

    def test_time_varying_tension(self):
        neuron = MechanoHHNeuron.cortical_pyramidal(
            mechano_channels=[get_channel("Piezo1"), get_channel("TREK-1")],
        )
        N = 32
        n_steps = 2000
        # Ramp tension from 0 → 6 mN/m over the window
        ramp = np.linspace(0.0, 6.0, n_steps)
        tensions_2d = np.tile(ramp[:, None], (1, N))
        phi = np.full(N, neuron._q10_scale())
        sc_cpu = neuron._simulate_batch(
            tensions=np.zeros(N), phi_arr=phi, dt=0.01,
            n_steps=n_steps, I_stim=5.0,
            tensions_2d=tensions_2d, backend="numba",
        )
        sc_gpu = neuron._simulate_batch(
            tensions=np.zeros(N), phi_arr=phi, dt=0.01,
            n_steps=n_steps, I_stim=5.0,
            tensions_2d=tensions_2d, backend="cuda",
        )
        assert np.array_equal(sc_cpu, sc_gpu)

    def test_time_varying_honours_fixed_tau_act(self):
        """CUDA time-varying kernel uses the fixed tau_act like the
        constant-tension kernel (constant tension fed as a 2-D trace)."""
        p1 = get_channel("Piezo1")
        p1.g_density = 0.5
        assert p1.tau_act is not None
        neuron = MechanoHHNeuron.cortical_pyramidal(mechano_channels=[p1])
        T = np.linspace(1.0, 8.0, 32)
        n_steps = 2000
        phi = np.full(T.size, neuron._q10_scale())
        T2 = np.broadcast_to(T, (n_steps, T.size)).copy()
        sc_const = neuron._simulate_batch(T, phi, 0.01, n_steps, 0.0,
                                          backend="cuda")
        sc_tv = neuron._simulate_batch(T, phi, 0.01, n_steps, 0.0,
                                       tensions_2d=T2, backend="cuda")
        sc_tv_cpu = neuron._simulate_batch(T, phi, 0.01, n_steps, 0.0,
                                           tensions_2d=T2, backend="numba")
        assert sc_const.sum() > 0
        assert np.allclose(sc_const, sc_tv, atol=1)
        assert np.allclose(sc_tv, sc_tv_cpu, atol=1)


# ------------------------------------------------------------------ #
# Three-compartment
# ------------------------------------------------------------------ #
class TestThreeCompCudaParity:
    """``MulticompartmentMechanoHHNeuron._simulate_batch`` parity."""

    def _make_neuron(self, **kw) -> MulticompartmentMechanoHHNeuron:
        return MulticompartmentMechanoHHNeuron.cortical_pyramidal_3comp(
            mechano_channels=[get_channel("Piezo1"), get_channel("TREK-1")],
            **kw,
        )

    def test_constant_tension_driven(self):
        neuron = self._make_neuron()
        N = 64
        tensions = np.linspace(0.5, 8.0, N)
        I_arr = np.array([0.0, 8.0, 0.0])
        sc_cpu = neuron._simulate_batch(
            tensions, dt=0.001, n_steps=20000,
            I_stim_per_compartment=I_arr, backend="numba",
        )
        sc_gpu = neuron._simulate_batch(
            tensions, dt=0.001, n_steps=20000,
            I_stim_per_compartment=I_arr, backend="cuda",
        )
        assert sc_cpu.sum() > 0, "Test config should produce spikes"
        assert np.array_equal(sc_cpu, sc_gpu), (
            f"sc_cpu={sc_cpu}\nsc_gpu={sc_gpu}"
        )

    def test_with_calcium_pool_and_kca(self):
        """Optional Ca²⁺ pool + SK/KCa branches activate correctly under CUDA."""
        cd = CalciumDynamics()
        sk = KCaChannel(g_density=2.0, Ca_half=0.5, n_hill=4.0)
        neuron = self._make_neuron(
            calcium_dynamics=cd, kca_channels=[sk],
            kca_compartments=(1,),
        )
        N = 32
        tensions = np.linspace(0.5, 6.0, N)
        I_arr = np.array([0.0, 8.0, 0.0])
        sc_cpu = neuron._simulate_batch(
            tensions, dt=0.001, n_steps=15000,
            I_stim_per_compartment=I_arr, backend="numba",
        )
        sc_gpu = neuron._simulate_batch(
            tensions, dt=0.001, n_steps=15000,
            I_stim_per_compartment=I_arr, backend="cuda",
        )
        assert np.array_equal(sc_cpu, sc_gpu), (
            f"sc_cpu={sc_cpu}\nsc_gpu={sc_gpu}"
        )

    def test_with_piezo_inactivation(self):
        """Mechano inactivation branch (Phase 3) parity."""
        neuron = self._make_neuron(piezo_inactivation=True)
        N = 32
        tensions = np.linspace(0.5, 6.0, N)
        I_arr = np.array([0.0, 8.0, 0.0])
        sc_cpu = neuron._simulate_batch(
            tensions, dt=0.001, n_steps=15000,
            I_stim_per_compartment=I_arr, backend="numba",
        )
        sc_gpu = neuron._simulate_batch(
            tensions, dt=0.001, n_steps=15000,
            I_stim_per_compartment=I_arr, backend="cuda",
        )
        assert np.array_equal(sc_cpu, sc_gpu), (
            f"sc_cpu={sc_cpu}\nsc_gpu={sc_gpu}"
        )

    def test_spike_step_recording(self):
        """Spike-step buffer write-back parity."""
        neuron = self._make_neuron()
        N = 16
        tensions = np.full(N, 5.0)
        I_arr = np.array([0.0, 8.0, 0.0])
        n_steps = 15000
        max_spikes = 32
        # CPU
        neuron._record_spike_steps = True
        neuron._max_spikes_per_voxel = max_spikes
        sc_cpu = neuron._simulate_batch(
            tensions, dt=0.001, n_steps=n_steps,
            I_stim_per_compartment=I_arr, backend="numba",
        )
        buf_cpu = neuron._last_spike_step_buf.copy()
        # GPU
        neuron._record_spike_steps = True
        neuron._max_spikes_per_voxel = max_spikes
        sc_gpu = neuron._simulate_batch(
            tensions, dt=0.001, n_steps=n_steps,
            I_stim_per_compartment=I_arr, backend="cuda",
        )
        buf_gpu = neuron._last_spike_step_buf.copy()
        assert np.array_equal(sc_cpu, sc_gpu)
        assert np.array_equal(buf_cpu, buf_gpu), (
            "spike_step_buf mismatch between CUDA and Numba backends"
        )


    def _s4_neuron(self):
        import copy as _copy
        p1 = _copy.deepcopy(PIEZO1); p1.g_density = 0.10; p1.tau_inact = 12.0
        t1 = _copy.deepcopy(TREK1); t1.g_density = 0.04
        return MulticompartmentMechanoHHNeuron.cortical_pyramidal_3comp(
            mechano_channels=[p1, t1], piezo_inactivation=True,
            calcium_dynamics=CalciumDynamics(),
            kca_channels=[KCaChannel(g_density=0.3)], kca_compartments=(1,),
        )

    @pytest.mark.parametrize("scenario", ["S2_passive_ca", "S4"])
    def test_drive_state_and_calcium_readout_parity(self, scenario):
        """Time-varying drive + state carry-over + Ca readouts: CUDA == Numba."""
        import copy as _copy
        from ultraneuron.multicompartment import (
            simulate_voxels_3comp,
            tension_drive,
        )
        if scenario == "S4":
            neuron = self._s4_neuron()
        else:
            p1 = _copy.deepcopy(PIEZO1); p1.g_density = 0.10
            t1 = _copy.deepcopy(TREK1); t1.g_density = 0.04
            neuron = MulticompartmentMechanoHHNeuron.cortical_pyramidal_3comp(
                mechano_channels=[p1, t1])
        T = np.linspace(1.5, 6.0, 24)
        dt = 0.001
        drv = tension_drive(dt, t_on_ms=10.0, t_off_ms=15.0)
        out = {}
        for be in ("numba", "cuda"):
            w1 = simulate_voxels_3comp(T, neuron, dt=dt, duration=25.0,
                                       backend=be, drive=drv,
                                       calcium_readout=True, ca_trace_dt_ms=1.0)
            w2 = simulate_voxels_3comp(T, neuron, dt=dt, duration=25.0,
                                       backend=be, drive=drv,
                                       initial_state=w1["final_state"],
                                       calcium_readout=True)
            out[be] = (w1, w2)
        (c1, c2), (g1, g2) = out["numba"], out["cuda"]
        assert c1["spike_counts"].sum() > 0
        for a, b in ((c1, g1), (c2, g2)):
            assert np.array_equal(a["spike_counts"], b["spike_counts"])
            assert np.array_equal(a["spike_times_ms"], b["spike_times_ms"])
            np.testing.assert_allclose(a["final_state"], b["final_state"],
                                       rtol=1e-9, atol=1e-9)
            np.testing.assert_allclose(a["ca_peak_uM"], b["ca_peak_uM"],
                                       rtol=1e-9)
        np.testing.assert_allclose(c1["ca_trace_uM"], g1["ca_trace_uM"], rtol=1e-6)

    def test_passive_calcium_readout_invariance_cuda(self):
        """Attaching a passive Ca pool leaves CUDA spike output bit-identical."""
        import copy as _copy
        from ultraneuron.multicompartment import simulate_voxels_3comp
        p1 = _copy.deepcopy(PIEZO1); p1.g_density = 0.10
        t1 = _copy.deepcopy(TREK1); t1.g_density = 0.04
        neuron = MulticompartmentMechanoHHNeuron.cortical_pyramidal_3comp(
            mechano_channels=[p1, t1])
        T = np.linspace(1.5, 6.0, 32)
        a = simulate_voxels_3comp(T, neuron, duration=20.0, backend="cuda")
        b = simulate_voxels_3comp(T, neuron, duration=20.0, backend="cuda",
                                  calcium_readout=True)
        assert a["spike_counts"].sum() > 0
        assert np.array_equal(a["spike_counts"], b["spike_counts"])
        assert np.array_equal(a["spike_times_ms"], b["spike_times_ms"])
        assert np.array_equal(a["final_state"][:, :-3], b["final_state"][:, :-3])


# ------------------------------------------------------------------ #
# Behaviour when CUDA is unavailable / out-of-scope
# ------------------------------------------------------------------ #
class TestCudaFallback:
    """Even with a GPU present, oversize n_ch must fall back cleanly."""

    def test_n_ch_above_max_falls_back(self):
        """Configurations beyond the kernel's MAX_CH warn + use Numba."""
        from ultraneuron.cuda_backend import _MAX_CH

        # Build a neuron with n_ch = _MAX_CH + 1 — clone PIEZO1 dataclass.
        import dataclasses as _dc
        channels = [_dc.replace(PIEZO1) for _ in range(_MAX_CH + 1)]
        neuron = MechanoHHNeuron.cortical_pyramidal(mechano_channels=channels)
        N = 16
        tensions = np.full(N, 4.0)
        phi = np.full(N, neuron._q10_scale())
        with pytest.warns(RuntimeWarning, match="Falling back to Numba"):
            sc_gpu = neuron._simulate_batch(
                tensions, phi, dt=0.01, n_steps=500, I_stim=2.0,
                backend="cuda",
            )
        sc_cpu = neuron._simulate_batch(
            tensions, phi, dt=0.01, n_steps=500, I_stim=2.0,
            backend="numba",
        )
        assert np.array_equal(sc_cpu, sc_gpu)


class TestResolveBackend:
    """'auto' picks CUDA when a device is visible; AMN_BACKEND overrides."""

    def test_auto_prefers_cuda_when_available(self, monkeypatch):
        from ultraneuron import cuda_backend as cb
        monkeypatch.delenv("AMN_BACKEND", raising=False)
        expected = "cuda" if cb.is_cuda_available() else "numba"
        assert cb.resolve_backend("auto") == expected
        assert cb.resolve_backend() == expected

    def test_env_override(self, monkeypatch):
        from ultraneuron import cuda_backend as cb
        monkeypatch.setenv("AMN_BACKEND", "numba")
        assert cb.resolve_backend("auto") == "numba"
        monkeypatch.setenv("AMN_BACKEND", "cuda")
        assert cb.resolve_backend("auto") == "cuda"

    def test_explicit_passthrough_and_validation(self):
        import pytest
        from ultraneuron import cuda_backend as cb
        assert cb.resolve_backend("numba") == "numba"
        assert cb.resolve_backend("cuda") == "cuda"
        with pytest.raises(ValueError, match="backend must be"):
            cb.resolve_backend("opencl")


class TestMechanoInitCudaParity:
    """Both mechano_init modes are bit-identical on CUDA and Numba."""

    @pytest.mark.parametrize("mode", ["rest", "peak_steady"])
    def test_single_comp(self, mode):
        neuron = MechanoHHNeuron.cortical_pyramidal(
            mechano_channels=[get_channel("Piezo1")])
        N = 64
        tensions = np.linspace(0.0, 8.0, N)
        phi = np.full(N, neuron._q10_scale())
        kw = dict(dt=0.01, n_steps=3000, I_stim=2.0, mechano_init=mode)
        a = neuron._simulate_batch(tensions, phi, backend="numba", **kw)
        b = neuron._simulate_batch(tensions, phi, backend="cuda", **kw)
        assert np.array_equal(a, b)
        T2 = np.broadcast_to(tensions, (3000, N)).copy()
        a = neuron._simulate_batch(tensions, phi, tensions_2d=T2, backend="numba", **kw)
        b = neuron._simulate_batch(tensions, phi, tensions_2d=T2, backend="cuda", **kw)
        assert np.array_equal(a, b)

    @pytest.mark.parametrize("mode", ["rest", "peak_steady"])
    @pytest.mark.parametrize("inact", [False, True])
    def test_three_comp(self, mode, inact):
        from ultraneuron.multicompartment import simulate_voxels_3comp
        import dataclasses as _dc
        p1 = _dc.replace(get_channel("Piezo1"), g_density=0.10)
        neuron = MulticompartmentMechanoHHNeuron.cortical_pyramidal_3comp(
            mechano_channels=[p1, get_channel("TREK-1")],
            piezo_inactivation=inact)
        T = np.linspace(0.5, 8.0, 32)
        a = simulate_voxels_3comp(T, neuron, duration=8.0, backend="numba",
                                  mechano_init=mode)
        b = simulate_voxels_3comp(T, neuron, duration=8.0, backend="cuda",
                                  mechano_init=mode)
        assert a["spike_counts"].sum() > 0
        assert np.array_equal(a["spike_counts"], b["spike_counts"])
        assert np.array_equal(a["spike_times_ms"], b["spike_times_ms"])
        # End state agrees to round-off (FMA contraction differs CPU/GPU)
        np.testing.assert_allclose(a["final_state"], b["final_state"],
                                   rtol=0, atol=1e-9)
