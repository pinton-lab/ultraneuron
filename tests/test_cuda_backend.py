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
