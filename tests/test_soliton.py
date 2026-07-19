"""Tests for the Heimburg-Jackson soliton (thermodynamic membrane) model.

Validation references
---------------------
* Heimburg & Jackson 2005, PNAS 102:9790-9795 — the nerve impulse as an
  electromechanical density soliton in the lipid membrane near its
  melting transition T_m.
* Heimburg 2007, *Thermal Biophysics of Membranes* — the
  compressibility ↔ excess-heat-capacity proportionality (κ_A^T peaks /
  diverges at T_m), and the report that biological membranes melt
  ~10-15 °C below body temperature.

The model returns a scalar excitability, wrapped by the downstream
entry point ``soliton_excitation(P_focal_pa, peak_dT_K)`` that scores a
case as ``SolitonModel().excitability(P_focal_pa, 37.0 + peak_dT_K)``.

Coverage:
  (a) compressibility peaks at T_m (= 1) and decays away on both sides;
  (b) excitability rises monotonically with pressure amplitude;
  (c) the distinctive falsifiable sign — with T_m < 37 °C, *raising* the
      temperature above body temperature DECREASES excitability (heating
      moves the membrane away from the transition), while cooling toward
      T_m increases it.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from ultraneuron import (
    SOLITON_PROVENANCE,
    SolitonModel,
    provenance_for,
    soliton_excitation,
)


class TestCompressibility:
    def test_peaks_at_Tm_normalised_to_one(self):
        m = SolitonModel()
        assert m.compressibility(m.T_m) == pytest.approx(1.0)

    def test_decays_away_from_Tm_both_sides(self):
        m = SolitonModel()
        peak = m.compressibility(m.T_m)
        for delta in (1.0, 3.0, 7.0, 15.0):
            assert m.compressibility(m.T_m + delta) < peak
            assert m.compressibility(m.T_m - delta) < peak

    def test_monotone_decay_moving_away(self):
        m = SolitonModel()
        deltas = [0.0, 1.0, 2.0, 4.0, 8.0]
        vals = [m.compressibility(m.T_m + d) for d in deltas]
        for i in range(1, len(vals)):
            assert vals[i] < vals[i - 1]

    def test_symmetric_about_Tm(self):
        m = SolitonModel()
        for delta in (2.0, 5.0):
            assert m.compressibility(m.T_m + delta) == pytest.approx(
                m.compressibility(m.T_m - delta)
            )

    def test_far_field_floor(self):
        """Far from T_m the relative compressibility → 1/(1 + c_p_max)."""
        m = SolitonModel()
        far = m.compressibility(m.T_m + 1000.0 * m.sigma_T)
        assert far == pytest.approx(1.0 / (1.0 + m.c_p_max), abs=1e-9)

    def test_lorentzian_shape_also_peaks_at_Tm(self):
        m = SolitonModel(peak_shape="lorentzian")
        assert m.compressibility(m.T_m) == pytest.approx(1.0)
        assert m.compressibility(m.T_m + 4.0) < 1.0

    def test_array_input(self):
        m = SolitonModel()
        T = np.array([m.T_m - 5.0, m.T_m, m.T_m + 5.0])
        k = m.compressibility(T)
        assert k.shape == (3,)
        assert k[1] == pytest.approx(1.0)
        assert k[0] < k[1] and k[2] < k[1]

    def test_unknown_shape_raises(self):
        m = SolitonModel(peak_shape="triangular")
        with pytest.raises(ValueError, match="peak_shape"):
            m.compressibility(m.T_m)


class TestAcousticStrain:
    def test_linear_in_pressure(self):
        m = SolitonModel()
        s1 = m.acoustic_strain(1e5)
        s2 = m.acoustic_strain(2e5)
        assert s2 == pytest.approx(2.0 * s1)

    def test_value_matches_bulk_modulus(self):
        m = SolitonModel()
        P = 5e5
        assert m.acoustic_strain(P) == pytest.approx(P / (m.rho * m.c * m.c))

    def test_zero_pressure(self):
        m = SolitonModel()
        assert m.acoustic_strain(0.0) == 0.0

    def test_array_input(self):
        m = SolitonModel()
        P = np.array([1e5, 2e5, 4e5])
        eps = m.acoustic_strain(P)
        assert eps.shape == (3,)
        np.testing.assert_allclose(eps, P / (m.rho * m.c * m.c))


class TestExcitability:
    def test_rises_with_pressure(self):
        m = SolitonModel()
        amps = [1e4, 5e4, 1e5, 2e5, 5e5, 1e6]
        vals = [m.excitability(P, 37.0) for P in amps]
        for i in range(1, len(vals)):
            assert vals[i] > vals[i - 1]

    def test_zero_pressure_zero_excitability(self):
        m = SolitonModel()
        assert m.excitability(0.0, 37.0) == 0.0

    def test_proportional_to_compressibility_and_strain(self):
        m = SolitonModel()
        P, T = 4e5, 35.0
        assert m.excitability(P, T) == pytest.approx(
            m.compressibility(T) * abs(m.acoustic_strain(P))
        )

    def test_nonnegative(self):
        m = SolitonModel()
        for T in (20.0, 30.0, 37.0, 45.0):
            assert m.excitability(3e5, T) >= 0.0


class TestFalsifiableSign:
    """The soliton model's sharpest discriminator: with T_m < 37 °C,
    heating reduces excitability (opposite to thermal-excitation models)."""

    def test_heating_above_body_temp_decreases_excitability(self):
        m = SolitonModel()
        assert m.T_m < 37.0
        P = 5e5
        base = m.excitability(P, 37.0)
        for dT in (0.35, 1.0, 3.0, 6.0):
            assert m.excitability(P, 37.0 + dT) < base, (
                f"heating by {dT} K should reduce soliton excitability"
            )

    def test_monotone_decrease_with_heating(self):
        m = SolitonModel()
        P = 5e5
        prev = m.excitability(P, 37.0)
        for dT in (1.0, 2.0, 4.0, 8.0):
            cur = m.excitability(P, 37.0 + dT)
            assert cur < prev
            prev = cur

    def test_cooling_toward_Tm_increases_excitability(self):
        m = SolitonModel()
        P = 5e5
        base = m.excitability(P, 37.0)
        # 31 °C is closer to T_m = 30 °C than 37 °C is
        assert m.excitability(P, 31.0) > base


class TestOptionalDispersion:
    def test_lateral_sound_speed_minimum_at_Tm(self):
        m = SolitonModel()
        c_at_Tm = m.relative_lateral_sound_speed(m.T_m)
        assert c_at_Tm == pytest.approx(1.0)
        # Away from the transition the membrane stiffens → faster lateral sound
        assert m.relative_lateral_sound_speed(m.T_m + 6.0) > c_at_Tm
        assert m.relative_lateral_sound_speed(m.T_m - 6.0) > c_at_Tm

    def test_far_field_sound_speed_value(self):
        """Far from T_m, c_lat/c_lat(T_m) → √(1 + c_p_max)."""
        m = SolitonModel()
        far = m.relative_lateral_sound_speed(m.T_m + 1000.0 * m.sigma_T)
        assert far == pytest.approx(np.sqrt(1.0 + m.c_p_max), rel=1e-6)


class TestScalarEntryPoint:
    """The downstream per-case scalar contract ``soliton_excitation``."""

    def test_matches_model_excitability(self):
        m = SolitonModel()
        assert soliton_excitation(5e5, 1.0, baseline_T_C=37.0, model=m) == (
            pytest.approx(m.excitability(5e5, 38.0))
        )

    def test_isothermal_default_is_body_temperature(self):
        m = SolitonModel()
        assert soliton_excitation(5e5, 0.0) == pytest.approx(m.excitability(5e5, 37.0))

    def test_heating_reduces_score(self):
        """The falsifiable sign survives the wrapper: +ΔT lowers the score."""
        assert soliton_excitation(5e5, 2.0) < soliton_excitation(5e5, 0.0)

    def test_accepts_injected_model(self):
        warm = SolitonModel(T_m=20.0)  # transition further below body temp
        assert soliton_excitation(5e5, 0.0, model=warm) < soliton_excitation(5e5, 0.0)


class TestPressureValidityLimit:
    """Parity with NICE: the documented limit is actually enforced."""

    def test_warns_above_limit(self):
        m = SolitonModel()
        with pytest.warns(UserWarning, match="validity limit"):
            m.excitability(2.0 * m.PRESSURE_VALIDITY_LIMIT, 37.0)

    def test_no_warning_at_or_below_limit(self):
        m = SolitonModel()
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("error")
            m.excitability(m.PRESSURE_VALIDITY_LIMIT, 37.0)  # exactly at limit: OK
            m.excitability(0.5 * m.PRESSURE_VALIDITY_LIMIT, 37.0)


class TestTransitionTemperatureGuard:
    """T_m ≥ body temperature flips the sign and must warn (audit #7)."""

    def test_warns_when_Tm_at_or_above_body_temperature(self):
        with pytest.warns(UserWarning, match="body temperature"):
            SolitonModel(T_m=37.5)

    def test_default_does_not_warn(self):
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("error")
            SolitonModel()  # default T_m = 30 °C < 37 °C


class TestProvenance:
    def test_registered(self):
        prov = provenance_for("SolitonModel")
        assert prov is SOLITON_PROVENANCE
        for key in ("T_m", "sigma_T", "c_p_max", "rho", "c", "peak_shape"):
            assert key in prov

    def test_provenance_keys_are_real_fields(self):
        """Drift guard: every classified key is an actual model field."""
        fields = {f.name for f in dataclasses.fields(SolitonModel)}
        assert set(SOLITON_PROVENANCE).issubset(fields)

    def test_default_Tm_below_body_temperature(self):
        """Default must keep T_m < 37 °C so the distinctive sign holds."""
        assert SolitonModel().T_m < 37.0
