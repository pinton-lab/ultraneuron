"""Plaksin-Krasovitski 2014 NICE (intramembrane cavitation) model.

The NICE (Neuronal Intramembrane Cavitation Excitation) model represents
the lipid bilayer as two leaflets that can separate under ultrasound
carrier pressure, producing a time-varying membrane capacitance ΔC_m(t).
This module implements the carrier-frequency leaflet ODE with a
Lennard-Jones-like molecular restoring pressure (Plaksin & Krasovitski
2014, Eq. 5), and provides a cycle-averaged ΔC_m as a function of
pressure amplitude — the SONIC simplification of Lemaire et al. 2019.

For small displacements the ODE reduces to a damped linear oscillator;
the LJ stiffening on Z<0 prevents leaflet collapse at high carrier
amplitudes, giving the cycle-averaged ΔC_m saturation that the linear
form misses above ~300 kPa.

The averaged ΔC_m is wired into the multi-compartment HH membrane
equation as an effective capacitance perturbation:

    [C_m + ΔC_m(P_env(t))] · dV/dt + V · dΔC_m/dt = -ΣI - I_stim

The displacement-current term V · dΔC_m/dt is what drives the
duty-cycle-dependent excitation/inhibition flip predicted by Plaksin
2014 Fig. 6 (excitation at low duty cycle, inhibition at high duty
cycle for the same source amplitude).

Units
-----
Internal NICE state uses SI: pressure [Pa], displacement [m], time [s],
inertia [kg/m^2], capacitance [F/m^2]. Conversion to/from biophysics
units (μF/cm^2, ms) happens at the boundary with the HH solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class NICEModel:
    """Reduced-order NICE leaflet model with two-sided non-linear walls.

    The leaflet displacement Z (perpendicular to the membrane plane,
    positive = leaflets pull apart) obeys

        m_eff · Z'' + β · m_eff · Z' + m_eff · ω_0² · Z
            = P_ac(t) + P_LJ(Z) + P_S(Z)

        P_LJ(Z) = K_LJ · [(δ_0/(δ_0+Z))^5 - (δ_0/(δ_0+Z))^3]   for Z < 0
                = 0                                              for Z ≥ 0

        P_S(Z)  = -2σ_eff · [1/(δ_0+Z) - 1/δ_0 + Z/δ_0²]        for Z > 0
                = 0                                              for Z ≤ 0

        K_LJ    = lj_strength · m_eff · ω_0² · δ_0
        σ_eff   = st_strength · m_eff · ω_0² · δ_0² / 2
        β       = 2 ζ ω_0

    The damped linear oscillator handles the small-amplitude regime.
    Two non-linear walls take over at large |Z|:

    * ``P_LJ`` on Z<0 (Plaksin–Krasovitski 5/3 LJ form): repulsive,
      diverges as Z → -δ_0, prevents leaflet collapse at high P_amp.
    * ``P_S`` on Z>0 (surface-tension correction with the linear part
      subtracted out so it does not change small-Z calibration):
      restoring, asymptotically linear at large Z, caps the expansion
      swing.

    With both walls active the quantitative validity range extends to
    ~1 MPa (matching Plaksin & Krasovitski 2014 Fig. 2 qualitatively).
    Either wall can be disabled by setting its strength prefactor to 0.

    The membrane capacitance is the geometric series

        C_m(Z) = C_m0 · δ_0 / (δ_0 + Z)

    so that <ΔC_m> = <C_m(Z)> - C_m0 is positive (because the geometry
    is concave in Z) and approximately quadratic in pressure amplitude
    at small drives.

    Parameters
    ----------
    f_carrier : float
        Ultrasound carrier frequency [Hz]. Default 500 kHz (typical TUS).
    f_resonant : float
        Leaflet natural resonance [Hz]. Default 5 MHz, above the carrier
        as in Plaksin 2014 — places the carrier in the quasi-static
        response regime.
    damping_ratio : float
        Dimensionless damping ζ. Default 0.1 (Q ≈ 5).
    m_eff : float
        Effective leaflet inertia per unit area [kg/m²]. Default 0.16
        — calibrated so Z amplitude at 100 kPa carrier pressure is
        ~0.6 nm, giving <ΔC_m>/C_m0 ≈ 0.05 (5% capacitance increase),
        within the SONIC-quoted operating range.
    delta_0 : float
        Half-bilayer thickness [m]. Default 2 nm.
    C_m0 : float
        Resting membrane capacitance [F/m²]. Default 0.01 F/m² = 1 μF/cm².
    """

    f_carrier: float = 500e3
    f_resonant: float = 5e6
    damping_ratio: float = 0.1
    m_eff: float = 0.16
    delta_0: float = 2e-9
    C_m0: float = 0.01  # 1 μF/cm² in SI
    lj_strength: float = 0.01
    """Dimensionless prefactor for the one-sided LJ wall on Z<0.

    K_LJ = lj_strength · m_eff · ω_0² · δ_0. The default 0.01 is small
    enough that the LJ term contributes <5% of the linear restoring
    force for |Z| < 0.1·δ_0 (preserves the existing low-pressure
    calibration), while diverging fast enough to dominate by >100× as
    Z → -δ_0. Set to 0 to recover the pure linear model on Z<0.
    """
    st_strength: float = 1.0
    """Dimensionless prefactor for the one-sided surface-tension wall on Z>0.

    σ_eff = st_strength · m_eff · ω_0² · δ_0² / 2. The form has its
    linear part subtracted out so it contributes 0 at small Z (preserves
    calibration); at large Z>>δ_0 it asymptotes to a linear restoring
    force with stiffness equal to ``st_strength × m·ω_0²``. The default
    1.0 doubles the asymptotic restoring on the expansion side, which
    extends the quantitative validity range to ~1 MPa. Set to 0 to
    recover the pure linear model on Z>0.
    """

    # ------------------------------------------------------------------
    # Leaflet oscillator response
    # ------------------------------------------------------------------

    def _omega0(self) -> float:
        return 2.0 * np.pi * self.f_resonant

    def _omega_carrier(self) -> float:
        return 2.0 * np.pi * self.f_carrier

    def _leaflet_trace(
        self,
        P_amp: float,
        n_cycles: int = 8,
        n_steps_per_cycle: int = 256,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Integrate the leaflet ODE over ``n_cycles`` of carrier.

        Returns the time vector and Z trace from t=0 to t=n_cycles/f_carrier.
        Uses RK4 at the carrier sub-cycle dt; the LJ molecular pressure
        is evaluated with Z clipped to ``-0.95·δ_0`` so the (δ_0+Z)^-5
        term stays bounded if a transient pushes the leaflet against its
        compressive limit.

        The defaults (8 cycles × 256 steps/cycle) give ~50 e-folds of
        transient decay and a CFL margin of ω_0·dt ≈ 0.12, comfortably
        stable for the LJ stiffening regime up to ~600 kPa.
        """
        dt = 1.0 / (self.f_carrier * n_steps_per_cycle)
        n_steps = n_cycles * n_steps_per_cycle
        omega_c = self._omega_carrier()
        omega0 = self._omega0()
        omega0_sq = omega0 * omega0
        beta = 2.0 * self.damping_ratio * omega0
        m = self.m_eff
        delta = self.delta_0
        K_LJ = self.lj_strength * m * omega0_sq * delta
        sigma_eff = self.st_strength * m * omega0_sq * delta * delta / 2.0
        Z_floor = -0.95 * delta
        inv_delta = 1.0 / delta
        inv_delta_sq = inv_delta * inv_delta

        def deriv(t: float, Z: float, V: float) -> tuple[float, float]:
            P_ac = P_amp * np.sin(omega_c * t)
            P_lin = -m * omega0_sq * Z
            if Z < 0.0:
                Z_eff = Z if Z > Z_floor else Z_floor
                ratio = delta / (delta + Z_eff)
                r3 = ratio * ratio * ratio
                r5 = r3 * ratio * ratio
                P_extra = K_LJ * (r5 - r3)
            elif Z > 0.0:
                # Surface-tension correction with the linear part subtracted
                # out: 1/(δ+Z) - 1/δ + Z/δ² ≈ Z²/δ³ at small Z (zero linear
                # contribution), and → Z/δ² at large Z (linear restoring).
                P_extra = -2.0 * sigma_eff * (
                    1.0 / (delta + Z) - inv_delta + Z * inv_delta_sq
                )
            else:
                P_extra = 0.0
            return V, (P_ac + P_lin + P_extra - beta * m * V) / m

        t = np.zeros(n_steps + 1)
        Z = np.zeros(n_steps + 1)
        Vd = np.zeros(n_steps + 1)
        for i in range(n_steps):
            ti, Zi, Vi = t[i], Z[i], Vd[i]
            k1Z, k1V = deriv(ti, Zi, Vi)
            k2Z, k2V = deriv(ti + 0.5 * dt, Zi + 0.5 * dt * k1Z, Vi + 0.5 * dt * k1V)
            k3Z, k3V = deriv(ti + 0.5 * dt, Zi + 0.5 * dt * k2Z, Vi + 0.5 * dt * k2V)
            k4Z, k4V = deriv(ti + dt, Zi + dt * k3Z, Vi + dt * k3V)
            t[i + 1] = ti + dt
            Z[i + 1] = Zi + dt * (k1Z + 2 * k2Z + 2 * k3Z + k4Z) / 6.0
            Vd[i + 1] = Vi + dt * (k1V + 2 * k2V + 2 * k3V + k4V) / 6.0
        return t, Z

    def _C_m_of_Z(self, Z: NDArray[np.floating]) -> NDArray[np.floating]:
        """Capacitance C_m(Z) [F/m²] for a leaflet displacement trace."""
        # Clip to avoid division by zero if Z somehow approaches -delta_0
        return self.C_m0 * self.delta_0 / np.clip(self.delta_0 + Z, 1e-12, None)

    # ------------------------------------------------------------------
    # Public API: cycle-averaged ΔC_m as a function of pressure amplitude
    # ------------------------------------------------------------------

    def delta_Cm_avg(
        self,
        P_amp: float,
        n_cycles: int = 8,
        n_steps_per_cycle: int = 256,
    ) -> float:
        """Cycle-averaged <ΔC_m> at carrier pressure amplitude *P_amp* [Pa].

        Returns
        -------
        delta_Cm : float
            <C_m(Z(t))> - C_m0 averaged over the last carrier cycle [F/m²].
            Positive for any non-zero P_amp (geometric concavity).
        """
        if P_amp <= 0.0:
            return 0.0
        _t, Z = self._leaflet_trace(P_amp, n_cycles, n_steps_per_cycle)
        # Average over the FINAL cycle to skip transient
        last = Z[-n_steps_per_cycle - 1 :]
        C = self._C_m_of_Z(last)
        # Geometry is concave in Z so the true cycle mean is non-negative.
        # Floor at zero to absorb finite-precision noise at very small P_amp
        # (Z amplitudes near 1e-12 m exhaust float64's significant digits
        # against δ_0 = 2e-9 m).
        return float(max(0.0, np.mean(C) - self.C_m0))

    PRESSURE_VALIDITY_LIMIT: float = 1.0e6
    """Soft upper bound [Pa] on the leaflet model.

    Below this drive amplitude both the LJ wall (compressive side) and
    the surface-tension wall (expansion side) keep |Z| within ~3·δ_0,
    matching Plaksin & Krasovitski 2014 Fig. 2 qualitatively. Above
    1 MPa the integrator remains stable up to ~1.5 MPa but the
    quantitative agreement degrades (Z_max grows past 3·δ_0 and
    gas-pocket effects we do not model become first-order).
    ``precompute_lookup`` warns when ``P_max`` exceeds this.
    """

    def precompute_lookup(
        self,
        P_max: float,
        n_points: int = 64,
        n_cycles: int = 8,
        n_steps_per_cycle: int = 256,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Precompute (P_amp_grid, ΔC_m_grid) for fast lookup.

        Returns arrays of shape (n_points,) suitable for
        ``np.interp`` evaluation in the inner HH loop.

        Raises a ``UserWarning`` when ``P_max`` exceeds
        ``PRESSURE_VALIDITY_LIMIT`` (1 MPa) — the integrator remains
        numerically stable, but quantitative agreement with Plaksin
        Fig. 2 is no longer guaranteed.
        """
        if P_max > self.PRESSURE_VALIDITY_LIMIT:
            import warnings

            warnings.warn(
                f"NICE precompute_lookup called with P_max={P_max:.2e} Pa > "
                f"validity limit {self.PRESSURE_VALIDITY_LIMIT:.2e} Pa; "
                "ΔC_m values above the limit are extrapolated and may not "
                "match Plaksin & Krasovitski 2014 quantitatively.",
                stacklevel=2,
            )
        P_grid = np.linspace(0.0, P_max, n_points)
        dC_grid = np.array(
            [self.delta_Cm_avg(float(P), n_cycles, n_steps_per_cycle) for P in P_grid]
        )
        return P_grid, dC_grid

    @staticmethod
    def lookup(
        P_env: NDArray[np.floating] | float,
        P_grid: NDArray[np.floating],
        dC_grid: NDArray[np.floating],
    ) -> NDArray[np.floating] | float:
        """Evaluate <ΔC_m>(P_env) by linear interpolation on a precomputed grid.

        Out-of-range pressures are clipped to the nearest endpoint.
        """
        return np.interp(P_env, P_grid, dC_grid)

    # ------------------------------------------------------------------
    # Unit conversion to biophysics units used by the HH solver
    # ------------------------------------------------------------------

    @staticmethod
    def to_uF_per_cm2(C_SI: float | NDArray[np.floating]) -> float | NDArray[np.floating]:
        """Convert F/m² to μF/cm². 1 F/m² = 100 μF/cm²."""
        return C_SI * 100.0
