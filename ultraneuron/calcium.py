"""Intracellular Ca²⁺ pool and Ca²⁺-gated K⁺ channels.

Provides the second inhibitory pathway distinct from K2P (TRAAK/TREK):
Piezo1 admits a Ca²⁺ flux which charges a single-pool intracellular
[Ca²⁺]_i, and a small-conductance Ca²⁺-activated K⁺ channel (SK) opens
in proportion to [Ca²⁺]_i, producing a hyperpolarising counter-current
that can convert sustained Piezo1 activation into net inhibition (the
mechanism candidate M5 for Legon-style high-pressure → inhibition
outcomes).

Single-pool single-time-constant model:

    d[Ca²⁺]_i/dt = -k_flux · I_Ca - ([Ca²⁺]_i - [Ca²⁺]_rest) / τ_Ca

where I_Ca is the inward Ca²⁺ current density [μA/cm²] (positive into
the cell, but we follow the channel-current sign convention where
inward = positive in our HH equations) and k_flux converts current
density to concentration rate [μM·cm²/(μA·ms)] by lumping membrane
surface-to-volume ratio, valence (z = 2), Faraday constant, and
fraction not buffered:

    k_flux = 1 / (z · F · d_shell)
           ≈ 5.18e-3 · (1 - β) / d_shell[μm]   [μM·cm²/(μA·ms)]

with d_shell ≈ 0.1 μm (sub-membrane Ca²⁺ shell) and buffering fraction
β ≈ 0.99 (only ~1% of incoming Ca²⁺ is free, the rest binds to
buffers). Default ``flux_factor`` = 0.5 μM·cm²/(μA·ms) — within the
range of values used in published cortical models (Mainen-Sejnowski
1996, Borg-Graham 1999).

The Ca²⁺ pool is modelled per-compartment (one [Ca²⁺]_i value per
compartment per voxel).

Slow somatic Ca²⁺ pool (read-out, not fed back)
-----------------------------------------------
:class:`SomaticCalcium` is a second, spike-driven pool on a seconds
time scale, the bulk somatic [Ca²⁺] a calcium indicator or reporter
would see over a stimulation session:

    d[Ca]/dt = -([Ca] - Ca_rest)/τ + ΔCa · Σ_k δ(t - t_k)

with Ca_rest = 0.05 μM (the :class:`CalciumDynamics` default),
τ = 700 ms, ΔCa = 0.08 μM per action potential and an optional
saturation clamp at 10 μM. It is evaluated exactly (event-driven
exponential update between spikes, no time-step error) from the CSR
spike-time layout returned by
:func:`~ultraneuron.multicompartment.simulate_map_3comp_with_spike_times`.
It does not act back on the membrane (the SK pathway uses the fast
sub-membrane pool above). :class:`PulsedProtocol` and
:func:`session_somatic_calcium` extend one simulated burst to a whole
pulsed session (e.g. the 400-burst theta-burst protocol of Yaakub et
al. 2023).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class CalciumDynamics:
    """Single-pool intracellular Ca²⁺ tracker.

    Parameters
    ----------
    Ca_rest : float
        Resting intracellular Ca²⁺ concentration [μM]. Default 0.05 μM
        (50 nM, typical for cortical pyramidal cells).
    tau_decay : float
        First-order decay time constant back to ``Ca_rest`` [ms].
        Default 50 ms (lumps SERCA pump, plasma-membrane extrusion,
        and slow diffusion to bulk cytoplasm).
    flux_factor : float
        Conversion from inward Ca²⁺ current [μA/cm²] to concentration
        rate [μM/ms]. Default 0.5 — a typical lumped value for a
        ~100 nm sub-membrane shell with ~99% buffering.
    """

    Ca_rest: float = 0.05  # μM (50 nM)
    tau_decay: float = 50.0  # ms
    flux_factor: float = 0.5  # μM·cm²/(μA·ms)


@dataclass
class KCaChannel:
    """Small-conductance Ca²⁺-activated K⁺ channel (SK-like).

    Hill-type activation by intracellular Ca²⁺:

        P_o([Ca]) = [Ca]^n / ([Ca]^n + Ca_half^n)

    Current density (per compartment):

        I_KCa = g_density · P_o · (E_K - V)

    Parameters
    ----------
    name : str
        Identifier (e.g. ``"SK"``, ``"BK"``).
    Ca_half : float
        Half-activation [Ca²⁺]_i [μM]. SK channels: ~0.5 μM
        (Köhler 1996); BK channels: voltage-dependent and not modelled
        here.
    n_hill : float
        Hill coefficient — activation curve steepness. SK: ~4
        (Köhler 1996, Hirschberg 1998).
    g_density : float
        Maximum conductance density [mS/cm²]. SK in cortical pyramidal
        cells: 0.1–1.0 mS/cm² depending on layer (Sah 1996, Stocker 2004).
    E_K : float
        K⁺ reversal potential [mV]. Default -100 (matches Pospischil RS).
    description : str
        Optional provenance note.
    """

    name: str = "SK"
    Ca_half: float = 0.5  # μM (Köhler 1996)
    n_hill: float = 4.0
    g_density: float = 0.3  # mS/cm² — mid-range cortical SK density
    E_K: float = -100.0  # mV
    description: str = "Small-conductance Ca²⁺-activated K⁺ channel (SK)"
    # Per-compartment density scale (parallels MechanosensitiveChannel API)
    # Set at neuron-construction time when the channel is attached to a
    # specific 3-compartment geometry; left unfilled here.
    compartment_scale: list[float] = field(default_factory=list)

    def open_probability(
        self,
        Ca_i: NDArray[np.floating] | float,
    ) -> NDArray[np.floating] | float:
        """Hill-type Ca²⁺ activation."""
        Ca = np.asarray(Ca_i, dtype=np.float64)
        # Avoid 0**negative for very low [Ca]; clamp to tiny positive
        Ca_safe = np.maximum(Ca, 1e-9)
        x_n = Ca_safe**self.n_hill
        return x_n / (x_n + self.Ca_half**self.n_hill)


# ---------------------------------------------------------------------------
# Slow somatic Ca²⁺ pool driven by spike times
# ---------------------------------------------------------------------------
# CSR spike layout (as returned by simulate_map_3comp_with_spike_times):
#   spike_offsets : (n_vox + 1,) int, spike_offsets[0] = 0
#   spike_times_ms: (n_spikes,) float, voxel i owns
#                   spike_times_ms[spike_offsets[i]:spike_offsets[i + 1]]


def _as_csr(
    spike_offsets: NDArray[np.integer],
    spike_times_ms: NDArray[np.floating],
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Validate a CSR spike layout and sort the times within each voxel."""
    off = np.asarray(spike_offsets, dtype=np.int64).ravel()
    st = np.asarray(spike_times_ms, dtype=np.float64).ravel()
    if off.size < 1 or off[0] != 0:
        msg = "spike_offsets must be a 1-D array starting at 0"
        raise ValueError(msg)
    if np.any(np.diff(off) < 0):
        msg = "spike_offsets must be non-decreasing"
        raise ValueError(msg)
    if off[-1] != st.size:
        msg = (
            f"spike_offsets[-1] = {off[-1]} does not match "
            f"len(spike_times_ms) = {st.size}"
        )
        raise ValueError(msg)
    if not np.all(np.isfinite(st)):
        msg = "spike_times_ms must be finite"
        raise ValueError(msg)
    row = np.repeat(np.arange(off.size - 1), np.diff(off))
    order = np.lexsort((st, row))
    return off, st[order]


def spike_lists_to_csr(
    spike_times_ms: Sequence[NDArray[np.floating]] | NDArray[np.floating],
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Convert per-voxel spike-time lists (or one 1-D array) to CSR.

    A single 1-D array is treated as one voxel.
    """
    if isinstance(spike_times_ms, np.ndarray) and spike_times_ms.ndim <= 1:
        lists = [np.asarray(spike_times_ms, dtype=np.float64).ravel()]
    else:
        lists = [np.asarray(s, dtype=np.float64).ravel() for s in spike_times_ms]
    counts = np.array([s.size for s in lists], dtype=np.int64)
    off = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    st = np.concatenate(lists) if lists else np.empty(0)
    return off, st


@dataclass
class SomaticCalcium:
    """Slow, spike-driven somatic Ca²⁺ pool (one exponential per spike).

    Model::

        d[Ca]/dt = -([Ca] - Ca_rest)/tau_ms + dCa_per_spike · Σ_k δ(t - t_k)
        [Ca] ← min([Ca], Ca_max)          after every spike (if Ca_max set)

    Between spikes the excess ``[Ca] - Ca_rest`` decays exactly as
    ``exp(-Δt/tau_ms)``; at a spike it jumps by ``dCa_per_spike``. The
    solution is right-continuous: at ``t = t_k`` the jump is included.
    Without the clamp the response is the linear superposition of
    single-spike kernels ``dCa_per_spike · exp(-(t - t_k)/tau_ms)``.

    This is a read-out pool: it is not coupled back into the membrane
    equations (the SK pathway uses the fast sub-membrane
    :class:`CalciumDynamics` pool, τ = 50 ms, far too fast for a
    seconds-scale session read-out). Ported from the sister project's
    ``gvmp.neuro.ultraneuron.SomaticCalcium``, which integrates the same
    equation on a 1 ms grid from a spike rate; here the update is
    event-driven and exact on any output grid.

    Parameters
    ----------
    Ca_rest : float
        Resting [Ca²⁺] [μM]. Default is the :class:`CalciumDynamics`
        default, 0.05 μM (50 nM).
    tau_ms : float
        Decay time constant of the bulk somatic pool [ms]. Default
        700 ms. ASSUMPTION (PI's choice for a somatic bulk pool in the
        0.5-1 s range), not fitted.
    dCa_per_spike : float
        Free [Ca²⁺] increment per action potential [μM]. Default
        0.08 μM. ASSUMPTION, in the range quoted for cortical somatic
        single-spike transients, not fitted.
    Ca_max : float or None
        Saturation clamp [μM] applied after each spike; ``None`` leaves
        the pool unbounded. Default 10 μM (ASSUMPTION).
    """

    Ca_rest: float = CalciumDynamics.Ca_rest  # μM
    tau_ms: float = 700.0  # ms
    dCa_per_spike: float = 0.08  # μM per spike
    Ca_max: float | None = 10.0  # μM; None = unbounded

    def __post_init__(self) -> None:
        if self.tau_ms <= 0:
            msg = f"tau_ms must be positive, got {self.tau_ms}"
            raise ValueError(msg)
        if self.Ca_max is not None and self.Ca_max <= self.Ca_rest:
            msg = f"Ca_max ({self.Ca_max}) must exceed Ca_rest ({self.Ca_rest})"
            raise ValueError(msg)

    # ----------------------------------------------------------- internals
    @property
    def _e_max(self) -> float:
        return np.inf if self.Ca_max is None else self.Ca_max - self.Ca_rest

    def _initial(
        self, n: int, Ca0: NDArray[np.floating] | float | None
    ) -> NDArray[np.float64]:
        if Ca0 is None:
            return np.zeros(n)
        return np.broadcast_to(np.asarray(Ca0, dtype=np.float64), (n,)) - self.Ca_rest

    def event_states(
        self,
        spike_offsets: NDArray[np.integer],
        spike_times_ms: NDArray[np.floating],
        *,
        Ca0: NDArray[np.floating] | float | None = None,
        t0_ms: float = -np.inf,
    ) -> dict[str, NDArray]:
        """Excess [Ca] just after each spike (exact event-driven recursion).

        Vectorised over voxels: the loop runs over the spike rank within
        a voxel (at most ``max spikes per voxel`` iterations), each
        iteration updating all voxels that have that many spikes.

        Parameters
        ----------
        spike_offsets, spike_times_ms
            CSR spike layout (times in ms; sorted within each voxel on
            return).
        Ca0 : float or (n_vox,) array, optional
            [Ca] at ``t0_ms`` [μM]. Default: rest. Spikes before
            ``t0_ms`` are ignored when ``Ca0`` is given.
        t0_ms : float
            Time at which ``Ca0`` holds. Default ``-inf`` (at rest
            forever before the first spike).

        Returns
        -------
        dict with ``offsets``, ``times_ms`` (sorted CSR), ``e_after``
        (excess [μM] right after each spike, post-clamp), ``jump``
        (actual increment at each spike, = ``dCa_per_spike`` unless the
        clamp is active), ``e0`` (initial excess per voxel) and
        ``t0_ms``.
        """
        off, st = _as_csr(spike_offsets, spike_times_ms)
        n = off.size - 1
        e0 = self._initial(n, Ca0)
        if Ca0 is not None and np.isfinite(t0_ms):
            keep = st >= t0_ms
            if not np.all(keep):
                row = np.repeat(np.arange(n), np.diff(off))[keep]
                off = np.concatenate(
                    [[0], np.cumsum(np.bincount(row, minlength=n))]
                ).astype(np.int64)
                st = st[keep]
        counts = np.diff(off)
        e_after = np.empty(st.size)
        jump = np.empty(st.size)
        e_prev = e0.copy()
        t_prev = np.full(n, float(t0_ms))
        e_max = self._e_max
        # voxels sorted by decreasing count -> rows with > j spikes are a prefix
        by_count = np.argsort(-counts, kind="stable")
        sorted_counts = counts[by_count]
        for j in range(int(counts.max()) if n else 0):
            rows = by_count[: np.searchsorted(-sorted_counts, -j, side="left")]
            k = off[rows] + j
            t = st[k]
            with np.errstate(over="ignore", invalid="ignore"):
                e_b = e_prev[rows] * np.exp(-(t - t_prev[rows]) / self.tau_ms)
            e_b = np.where(np.isfinite(e_b), e_b, 0.0)
            e_a = np.minimum(e_b + self.dCa_per_spike, e_max)
            e_after[k] = e_a
            jump[k] = e_a - e_b
            e_prev[rows] = e_a
            t_prev[rows] = t
        return {
            "offsets": off,
            "times_ms": st,
            "e_after": e_after,
            "jump": jump,
            "e0": e0,
            "t0_ms": float(t0_ms),
        }

    def excess_at(
        self,
        states: dict[str, NDArray],
        t_ms: NDArray[np.floating],
        *,
        chunk_voxels: int = 512,
        dtype: type = np.float64,
    ) -> NDArray:
        """Excess [Ca] - Ca_rest [μM] at the times ``t_ms``, (n_vox, n_t).

        ``states`` is the output of :meth:`event_states`. ``t_ms`` is any
        1-D array of output times (need not be uniform or sorted).
        """
        off = states["offsets"]
        st = states["times_ms"]
        e_after = states["e_after"]
        e0 = states["e0"]
        t0 = states["t0_ms"]
        tq = np.asarray(t_ms, dtype=np.float64).ravel()
        n = off.size - 1
        out = np.empty((n, tq.size), dtype=dtype)
        if n == 0 or tq.size == 0:
            return out
        # Flattened search: key = row * span + (t - base) is globally sorted.
        lo = min(tq.min(), st.min()) if st.size else tq.min()
        hi = max(tq.max(), st.max()) if st.size else tq.max()
        span = (hi - lo) + 1.0
        row = np.repeat(np.arange(n), np.diff(off))
        key = row * span + (st - lo)
        # initial-condition term (zero when Ca0 was not given)
        with np.errstate(over="ignore", invalid="ignore"):
            g0 = np.exp(-(tq - t0) / self.tau_ms) if np.isfinite(t0) else np.zeros_like(tq)
        g0 = np.where(np.isfinite(g0), g0, 0.0)
        for r0 in range(0, n, chunk_voxels):
            r1 = min(n, r0 + chunk_voxels)
            rows = np.arange(r0, r1)
            base = e0[rows, None] * g0[None, :]
            if st.size:
                qkey = rows[:, None] * span + (tq - lo)[None, :]
                idx = np.searchsorted(key, qkey, side="right") - 1
                has = idx >= off[rows, None]
                idc = np.where(has, idx, 0)
                ev = e_after[idc] * np.exp(-np.maximum(tq[None, :] - st[idc], 0.0) / self.tau_ms)
                out[r0:r1] = np.where(has, ev, base)
            else:
                out[r0:r1] = base
        return out

    # ----------------------------------------------------------- public API
    def from_csr(
        self,
        spike_offsets: NDArray[np.integer],
        spike_times_ms: NDArray[np.floating],
        t_ms: NDArray[np.floating],
        *,
        Ca0: NDArray[np.floating] | float | None = None,
        t0_ms: float = -np.inf,
        chunk_voxels: int = 512,
        dtype: type = np.float64,
    ) -> NDArray:
        """[Ca](t) [μM] for every voxel of a CSR spike layout, (n_vox, n_t).

        Exact on any output grid ``t_ms`` [ms]; the spikes need not lie on
        the grid. See :meth:`event_states` for ``Ca0`` / ``t0_ms``.
        """
        states = self.event_states(spike_offsets, spike_times_ms, Ca0=Ca0, t0_ms=t0_ms)
        ca = self.excess_at(states, t_ms, chunk_voxels=chunk_voxels, dtype=dtype)
        ca += self.Ca_rest
        return ca

    def from_spike_times(
        self,
        spike_times_ms: Sequence[NDArray[np.floating]] | NDArray[np.floating],
        t_ms: NDArray[np.floating],
        *,
        Ca0: NDArray[np.floating] | float | None = None,
        t0_ms: float = -np.inf,
    ) -> NDArray[np.float64]:
        """[Ca](t) [μM] from spike times [ms] on an arbitrary output grid.

        ``spike_times_ms`` is one 1-D array (one voxel; returns
        ``(n_t,)``) or a sequence of per-voxel arrays (returns
        ``(n_vox, n_t)``).
        """
        single = isinstance(spike_times_ms, np.ndarray) and spike_times_ms.ndim <= 1
        off, st = spike_lists_to_csr(spike_times_ms)
        ca = self.from_csr(off, st, t_ms, Ca0=Ca0, t0_ms=t0_ms)
        return ca[0] if single else ca

    def from_spike_results(
        self,
        results: dict[str, NDArray],
        t_ms: NDArray[np.floating],
        **kwargs,
    ) -> NDArray:
        """[Ca](t) [μM] per firing voxel from a
        ``simulate_map_3comp_with_spike_times`` result dict (or a loaded
        ``neural_results.npz``); row ``i`` matches
        ``results["spike_voxel_idx"][i]``."""
        return self.from_csr(results["spike_offsets"], results["spike_times_ms"], t_ms, **kwargs)


# ---------------------------------------------------------------------------
# Session summaries from event states (exact)
# ---------------------------------------------------------------------------


def _peak_excess(states: dict[str, NDArray]) -> NDArray[np.float64]:
    off, e_after, e0 = states["offsets"], states["e_after"], states["e0"]
    n = off.size - 1
    peak = np.maximum(e0, 0.0).copy()
    nz = np.flatnonzero(np.diff(off) > 0)
    if nz.size:
        peak[nz] = np.maximum(peak[nz], np.maximum.reduceat(e_after, off[nz]))
    return peak[:n]


def _window_mean_excess(
    pool: SomaticCalcium, states: dict[str, NDArray], a_ms: float, b_ms: float
) -> NDArray[np.float64]:
    """Exact time-mean of the excess over [a, b].

    Integrating d e/dt = -e/τ + Σ jumps over [a, b] gives
    ∫ e dt = τ (e(a) - e(b) + Σ_{a < t_k ≤ b} jump_k) with e
    right-continuous.
    """
    off, st, jump = states["offsets"], states["times_ms"], states["jump"]
    n = off.size - 1
    e_ab = pool.excess_at(states, np.array([a_ms, b_ms]))
    row = np.repeat(np.arange(n), np.diff(off))
    inwin = (st > a_ms) & (st <= b_ms)
    J = np.bincount(row[inwin], weights=jump[inwin], minlength=n)
    return pool.tau_ms * (e_ab[:, 0] - e_ab[:, 1] + J) / (b_ms - a_ms)


def _return_time(
    pool: SomaticCalcium,
    states: dict[str, NDArray],
    t_end_ms: float,
    tol: float,
) -> NDArray[np.float64]:
    """Time after ``t_end_ms`` [ms] at which the excess falls to
    ``tol · Ca_rest`` for good (0 if already there). Exact: after the
    last spike the excess decays as a single exponential."""
    off, st, e_after, e0 = states["offsets"], states["times_ms"], states["e_after"], states["e0"]
    n = off.size - 1
    thr = tol * pool.Ca_rest
    counts = np.diff(off)
    e_last = e0.copy()
    t_last = np.full(n, states["t0_ms"])
    has = counts > 0
    e_last[has] = e_after[off[1:][has] - 1]
    t_last[has] = st[off[1:][has] - 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        t_cross = t_last + pool.tau_ms * np.log(np.where(e_last > thr, e_last / thr, 1.0))
    t_cross = np.where(e_last > thr, t_cross, -np.inf)
    return np.maximum(t_cross - t_end_ms, 0.0)


# ---------------------------------------------------------------------------
# Pulsed protocol and session read-out
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PulsedProtocol:
    """A pulsed (burst-mode) ultrasound session.

    Burst ``b`` (0-based) starts at ``b · period_ms``; the session ends
    at ``n_bursts · period_ms`` (end of the last OFF interval).

    Defaults are the theta-burst protocol of Yaakub et al. 2023 (Nat
    Commun): 20 ms bursts every 200 ms (5 Hz, 10 % duty), 400 bursts =
    80 s.

    Parameters
    ----------
    burst_ms : float
        Burst (ON) duration [ms].
    period_ms : float
        Pulse-repetition period, burst onset to burst onset [ms].
    n_bursts : int
        Number of bursts in the session.
    """

    burst_ms: float = 20.0
    period_ms: float = 200.0
    n_bursts: int = 400

    def __post_init__(self) -> None:
        if not (0 < self.burst_ms <= self.period_ms):
            msg = "need 0 < burst_ms <= period_ms"
            raise ValueError(msg)
        if self.n_bursts < 1:
            msg = "n_bursts must be >= 1"
            raise ValueError(msg)

    @property
    def session_ms(self) -> float:
        return self.n_bursts * self.period_ms

    @property
    def prf_hz(self) -> float:
        return 1e3 / self.period_ms

    @property
    def duty_cycle(self) -> float:
        return self.burst_ms / self.period_ms

    def burst_onsets_ms(self) -> NDArray[np.float64]:
        return np.arange(self.n_bursts) * self.period_ms


BurstSpikes = "tuple[NDArray[np.integer], NDArray[np.floating]]"  # (offsets, times_ms) CSR pair


def session_spike_train(
    spike_offsets: NDArray[np.integer] | None,
    spike_times_ms: NDArray[np.floating] | None,
    protocol: PulsedProtocol = PulsedProtocol(),
    *,
    per_burst: Sequence[BurstSpikes] | Callable[[int], BurstSpikes] | None = None,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Build the session spike train (CSR, times in ms from session start).

    Default (periodic-steady-state assumption): the per-voxel spike
    times of one simulated burst (times relative to burst onset) are
    repeated at every burst onset. This assumes the neuron returns to
    the same state in the OFF interval, so every burst fires the same
    spikes; it ignores slow carry-over (channel inactivation, the fast
    Ca²⁺/SK pool, temperature) between bursts.

    Hook: pass ``per_burst`` to supply each burst's own spikes instead,
    either a sequence of ``n_bursts`` ``(offsets, times_ms)`` CSR pairs
    or a callable ``b -> (offsets, times_ms)``; all must share the same
    voxel rows. ``spike_offsets``/``spike_times_ms`` are then ignored
    (may be ``None``).

    Raises ``ValueError`` if a burst's spikes fall outside
    ``[0, period_ms)`` (they would overlap the next burst).
    """
    P = protocol.period_ms
    if per_burst is None:
        if spike_offsets is None or spike_times_ms is None:
            msg = "give spike_offsets/spike_times_ms or per_burst"
            raise ValueError(msg)
        off1, st1 = _as_csr(spike_offsets, spike_times_ms)
        bursts: Sequence[BurstSpikes] | Callable[[int], BurstSpikes] = None  # type: ignore[assignment]
    else:
        bursts = per_burst
    if per_burst is not None and not callable(per_burst) and len(per_burst) != protocol.n_bursts:
        msg = f"per_burst has {len(per_burst)} entries, protocol has {protocol.n_bursts} bursts"
        raise ValueError(msg)

    if per_burst is None:
        if st1.size and (st1.min() < 0 or st1.max() >= P):
            msg = f"burst spike times must lie in [0, {P}) ms"
            raise ValueError(msg)
        n = off1.size - 1
        counts1 = np.diff(off1)
        B = protocol.n_bursts
        off = np.concatenate([[0], np.cumsum(counts1 * B)]).astype(np.int64)
        # per voxel: burst-major then spike order -> already time-sorted
        row1 = np.repeat(np.arange(n), counts1)
        rank = np.arange(st1.size) - off1[row1]
        b = np.arange(B)
        pos = off[row1][:, None] + b[None, :] * counts1[row1][:, None] + rank[:, None]
        st = np.empty(off[-1])
        st[pos.ravel()] = (st1[:, None] + b[None, :] * P).ravel()
        return off, st

    lists_off, lists_st = [], []
    n = None
    for b in range(protocol.n_bursts):
        ob, sb = bursts(b) if callable(bursts) else bursts[b]
        ob, sb = _as_csr(ob, sb)
        if n is None:
            n = ob.size - 1
        elif ob.size - 1 != n:
            msg = f"burst {b} has {ob.size - 1} voxels, expected {n}"
            raise ValueError(msg)
        if sb.size and (sb.min() < 0 or sb.max() >= P):
            msg = f"burst {b}: spike times must lie in [0, {P}) ms"
            raise ValueError(msg)
        lists_off.append(ob)
        lists_st.append(sb + b * P)
    counts = np.sum([np.diff(o) for o in lists_off], axis=0)
    off = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    rows = np.concatenate([np.repeat(np.arange(n), np.diff(o)) for o in lists_off])
    st_all = np.concatenate(lists_st)
    order = np.lexsort((st_all, rows))
    return off, st_all[order]


@dataclass
class SomaticSessionResult:
    """Session read-out of the slow somatic pool.

    Attributes
    ----------
    t_ms : (n_t,) output times [ms], 0 = session start.
    ca : (n_vox, n_t) float32, [Ca](x, t) [μM]; row i = CSR voxel i.
    peak : (n_vox,) exact peak [Ca] over the session [μM] (just after a
        spike, not grid-sampled).
    plateau : (n_vox,) exact time-mean [Ca] over the last burst period
        ``[session_ms - period_ms, session_ms]`` [μM].
    ca_end : (n_vox,) [Ca] at the session end [μM].
    t_return_ms : (n_vox,) time after the session end at which [Ca]
        returns to within ``return_tol`` (fraction) of Ca_rest [ms].
    method : "analytic" (geometric series) or "direct" (event-driven on
        the expanded session train).
    """

    t_ms: NDArray[np.float64]
    ca: NDArray[np.float32]
    peak: NDArray[np.float64]
    plateau: NDArray[np.float64]
    ca_end: NDArray[np.float64]
    t_return_ms: NDArray[np.float64]
    method: str
    protocol: PulsedProtocol
    pool: SomaticCalcium
    return_tol: float


def periodic_excess(
    pool: SomaticCalcium,
    spike_offsets: NDArray[np.integer],
    spike_times_ms: NDArray[np.floating],
    protocol: PulsedProtocol,
    t_ms: NDArray[np.floating],
    *,
    chunk_t: int = 256,
) -> NDArray[np.float64]:
    """Analytic (unclamped) excess [Ca] - Ca_rest for a periodic train.

    One burst's spikes ``s_i`` repeated at ``m·P``, ``m = 0..N-1``. For
    spike ``i`` the bursts already fired at time t are
    ``m = 0..M_i``, ``M_i = min(N-1, floor((t - s_i)/P))``, and the
    geometric series gives::

        e(t) = ΔCa Σ_i exp(-(t - s_i - M_i P)/τ) · (1 - q^(M_i+1)) / (1 - q),
        q = exp(-P/τ).

    Cost is O(n_vox · spikes per burst · n_t), independent of N. The
    clamp is ignored (check the result against ``pool.Ca_max``).
    """
    off, st = _as_csr(spike_offsets, spike_times_ms)
    n = off.size - 1
    tq = np.asarray(t_ms, dtype=np.float64).ravel()
    out = np.zeros((n, tq.size))
    if st.size == 0:
        return out
    P, N, tau = protocol.period_ms, protocol.n_bursts, pool.tau_ms
    q = np.exp(-P / tau)
    nz = np.flatnonzero(np.diff(off) > 0)
    for c0 in range(0, tq.size, chunk_t):
        t = tq[c0 : c0 + chunk_t]
        d = t[None, :] - st[:, None]  # (n_spk, n_t)
        valid = d >= 0
        M = np.minimum(N - 1, np.floor(np.where(valid, d, 0.0) / P))
        term = np.exp(-(d - M * P) / tau) * (1.0 - q ** (M + 1)) / (1.0 - q)
        term = np.where(valid, term, 0.0)
        out[nz, c0 : c0 + chunk_t] = np.add.reduceat(term, off[nz], axis=0)
    return pool.dCa_per_spike * out


def _periodic_summaries(
    pool: SomaticCalcium,
    off: NDArray[np.int64],
    st: NDArray[np.float64],
    protocol: PulsedProtocol,
    tol: float,
) -> tuple[NDArray[np.float64], ...]:
    """Peak, plateau, end value and return time for the periodic train
    (analytic, unclamped)."""
    n = off.size - 1
    P, N = protocol.period_ms, protocol.n_bursts
    t_end = protocol.session_ms
    counts = np.diff(off)
    row = np.repeat(np.arange(n), counts)
    # e just after each spike of the last burst; without clamp the excess
    # at a given burst phase grows with the burst index, so the peak lies
    # in the last burst.
    # vectorised over spikes: excess of voxel row[k] at (N-1)P + st[k]
    tk = (N - 1) * P + st
    e_spk = np.zeros(st.size)
    tau, q = pool.tau_ms, np.exp(-P / pool.tau_ms)
    for j in range(int(counts.max()) if st.size else 0):
        rows = np.flatnonzero(counts > j)
        src = off[rows][:, None] + np.arange(counts.max())[None, :]
        ok = np.arange(counts.max())[None, :] < counts[rows][:, None]
        src = np.where(ok, src, 0)
        d = tk[off[rows] + j][:, None] - st[src]
        valid = ok & (d >= 0)
        M = np.minimum(N - 1, np.floor(np.where(valid, d, 0.0) / P))
        term = np.where(valid, np.exp(-(d - M * P) / tau) * (1 - q ** (M + 1)) / (1 - q), 0.0)
        e_spk[off[rows] + j] = pool.dCa_per_spike * term.sum(1)
    peak = np.zeros(n)
    nz = np.flatnonzero(counts > 0)
    if nz.size:
        peak[nz] = np.maximum.reduceat(e_spk, off[nz])
    e_ab = periodic_excess(pool, off, st, protocol, np.array([t_end - P, t_end]))
    J = pool.dCa_per_spike * np.bincount(row[st > 0], minlength=n)
    plateau = pool.tau_ms * (e_ab[:, 0] - e_ab[:, 1] + J) / P
    thr = tol * pool.Ca_rest
    e_last = np.zeros(n)
    t_last = np.full(n, -np.inf)
    e_last[nz] = e_spk[off[1:][nz] - 1]
    t_last[nz] = tk[off[1:][nz] - 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        t_cross = t_last + pool.tau_ms * np.log(np.where(e_last > thr, e_last / thr, 1.0))
    t_ret = np.maximum(np.where(e_last > thr, t_cross, -np.inf) - t_end, 0.0)
    return peak, plateau, e_ab[:, 1], t_ret


def session_somatic_calcium(
    spike_offsets: NDArray[np.integer] | None,
    spike_times_ms: NDArray[np.floating] | None,
    protocol: PulsedProtocol = PulsedProtocol(),
    pool: SomaticCalcium | None = None,
    *,
    t_ms: NDArray[np.floating] | None = None,
    dt_out_ms: float = 10.0,
    t_pre_ms: float = 1000.0,
    t_post_ms: float = 10000.0,
    per_burst: Sequence[BurstSpikes] | Callable[[int], BurstSpikes] | None = None,
    method: str = "auto",
    return_tol: float = 0.1,
    chunk_voxels: int = 512,
) -> SomaticSessionResult:
    """Session somatic [Ca](x, t) for a pulsed protocol, plus summary maps.

    Parameters
    ----------
    spike_offsets, spike_times_ms
        CSR spike layout of ONE burst (times from burst onset, ms), as
        returned by ``simulate_map_3comp_with_spike_times``.
    protocol : PulsedProtocol
        Session protocol (default: Yaakub theta burst, 20 ms / 200 ms /
        400 bursts).
    pool : SomaticCalcium
        Pool parameters (default ``SomaticCalcium()``).
    t_ms : array, optional
        Output grid [ms, from session start]. Default: decimated grid
        ``arange(-t_pre_ms, session_ms + t_post_ms, dt_out_ms)``. The
        summary maps are exact and do not depend on this grid.
    per_burst
        Optional per-burst spike lists (see :func:`session_spike_train`);
        forces ``method="direct"``.
    method : {"auto", "analytic", "direct"}
        ``"analytic"``: geometric series over bursts (periodic train, no
        clamp). ``"direct"``: event-driven recursion over the expanded
        session train (handles the clamp and ``per_burst``). ``"auto"``:
        analytic when periodic and the analytic peak stays below
        ``pool.Ca_max``, else direct.
    return_tol : float
        Return-to-rest criterion: [Ca] - Ca_rest <= return_tol · Ca_rest.

    Returns
    -------
    SomaticSessionResult
    """
    pool = SomaticCalcium() if pool is None else pool
    if method not in ("auto", "analytic", "direct"):
        msg = f"unknown method {method!r}"
        raise ValueError(msg)
    if t_ms is None:
        t_ms = np.arange(-t_pre_ms, protocol.session_ms + t_post_ms + 0.5 * dt_out_ms, dt_out_ms)
    tq = np.asarray(t_ms, dtype=np.float64).ravel()
    t_end = protocol.session_ms

    use = method
    if per_burst is not None:
        if method == "analytic":
            msg = "per_burst spike trains are not periodic; use method='direct'"
            raise ValueError(msg)
        use = "direct"
    if use in ("auto", "analytic"):
        off1, st1 = _as_csr(spike_offsets, spike_times_ms)
        if st1.size and (st1.min() < 0 or st1.max() >= protocol.period_ms):
            msg = f"burst spike times must lie in [0, {protocol.period_ms}) ms"
            raise ValueError(msg)
        peak_e, plat_e, end_e, t_ret = _periodic_summaries(pool, off1, st1, protocol, return_tol)
        clamp_hit = pool.Ca_max is not None and np.any(peak_e > pool._e_max)
        if use == "analytic" or not clamp_hit:
            e = periodic_excess(pool, off1, st1, protocol, tq)
            return SomaticSessionResult(
                t_ms=tq,
                ca=(e + pool.Ca_rest).astype(np.float32),
                peak=peak_e + pool.Ca_rest,
                plateau=plat_e + pool.Ca_rest,
                ca_end=end_e + pool.Ca_rest,
                t_return_ms=t_ret,
                method="analytic",
                protocol=protocol,
                pool=pool,
                return_tol=return_tol,
            )
    off, st = session_spike_train(spike_offsets, spike_times_ms, protocol, per_burst=per_burst)
    states = pool.event_states(off, st)
    e = pool.excess_at(states, tq, chunk_voxels=chunk_voxels, dtype=np.float32)
    e += np.float32(pool.Ca_rest)
    return SomaticSessionResult(
        t_ms=tq,
        ca=e,
        peak=_peak_excess(states) + pool.Ca_rest,
        plateau=_window_mean_excess(pool, states, t_end - protocol.period_ms, t_end) + pool.Ca_rest,
        ca_end=pool.excess_at(states, np.array([t_end]))[:, 0] + pool.Ca_rest,
        t_return_ms=_return_time(pool, states, t_end, return_tol),
        method="direct",
        protocol=protocol,
        pool=pool,
        return_tol=return_tol,
    )
