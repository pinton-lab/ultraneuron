"""Phase-5 tests: parameter-provenance audit + tau_act override.

Verifies that every numerical parameter exposed by the framework has a
provenance entry classifying it as LITERATURE, CALIBRATED, ASSUMPTION,
or DERIVED, with a human-readable reference / notes string. Also
verifies that the Phase-5 ``tau_act`` override on
``MechanosensitiveChannel`` produces a constant time constant when
set, and that the legacy bell-shaped τ is preserved when ``tau_act``
is left at its default ``None``.

This is the cleanup that makes absolute-rate predictions traceable
between mechanism candidates and lets us state per-parameter
uncertainty bounds in downstream papers.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from ultraneuron import (
    PIEZO1,
    TRAAK,
    TREK1,
    TREK2,
    TRPV1,
    TRPV4,
    ParamSource,
    Provenance,
    format_provenance_report,
    provenance_for,
    registered_names,
)
from ultraneuron.channels import MechanosensitiveChannel


# ---------------------------------------------------------------------------
# Provenance registry coverage
# ---------------------------------------------------------------------------
class TestProvenanceRegistry:
    def test_all_library_channels_have_entries(self):
        """Every literature-preset channel name has a provenance map."""
        for ch in (PIEZO1, TRAAK, TREK1, TREK2, TRPV1, TRPV4):
            assert ch.name in registered_names(), (
                f"missing provenance for library channel {ch.name!r}"
            )

    def test_pathway_components_registered(self):
        """Strain conversion + NICE + Ca/SK + multi-comp geometry are registered."""
        for key in (
            "strain_to_tension",
            "NICEModel",
            "CalciumDynamics",
            "KCaChannel",
            "cortical_pyramidal_3comp",
        ):
            assert key in registered_names(), f"missing provenance for {key!r}"

    def test_unknown_name_raises(self):
        with pytest.raises(KeyError, match="No provenance"):
            provenance_for("definitely-not-a-real-channel")

    def test_every_entry_has_typed_source(self):
        for name in registered_names():
            pmap = provenance_for(name)
            assert pmap, f"{name} has empty provenance map"
            for param_name, prov in pmap.items():
                assert isinstance(prov, Provenance), (
                    f"{name}.{param_name} is not a Provenance instance"
                )
                assert isinstance(prov.source, ParamSource), (
                    f"{name}.{param_name} has source of wrong type"
                )

    def test_literature_entries_have_references(self):
        """Anything classified LITERATURE should cite a paper."""
        for name in registered_names():
            for param_name, prov in provenance_for(name).items():
                if prov.source == ParamSource.LITERATURE:
                    assert prov.reference, (
                        f"{name}.{param_name} marked LITERATURE but has no reference"
                    )

    def test_calibrated_entries_have_notes(self):
        """CALIBRATED entries should explain the calibration target."""
        for name in registered_names():
            for param_name, prov in provenance_for(name).items():
                if prov.source == ParamSource.CALIBRATED:
                    assert prov.notes, (
                        f"{name}.{param_name} marked CALIBRATED but has no notes"
                    )


# ---------------------------------------------------------------------------
# Per-channel field coverage: every public field should be classified
# ---------------------------------------------------------------------------
class TestFieldCoverage:
    """Detect when a channel field is added without a provenance entry."""

    @staticmethod
    def _public_numeric_fields(ch_obj) -> set[str]:
        """Return set of public, numeric-or-fraction fields on a dataclass instance."""
        skip = {
            "name", "channel_type", "description",
            "compartment_scale",  # KCaChannel, populated at construction
        }
        result = set()
        for f_name in ch_obj.__dataclass_fields__:
            if f_name.startswith("_") or f_name in skip:
                continue
            v = getattr(ch_obj, f_name)
            # accept ints, floats, None (optional fields), or floats inside lists
            if v is None or isinstance(v, (int, float)):
                result.add(f_name)
        return result

    def test_piezo1_field_coverage(self):
        fields = self._public_numeric_fields(PIEZO1)
        prov_keys = set(provenance_for("Piezo1").keys())
        missing = fields - prov_keys
        # tau_0 is a legacy parameter — superseded by tau_act when set —
        # but we still document it in provenance.
        assert not missing, (
            f"Piezo1 fields without provenance: {missing}. "
            f"Update provenance.PIEZO1_PROVENANCE."
        )

    def test_trpv1_field_coverage(self):
        fields = self._public_numeric_fields(TRPV1)
        prov_keys = set(provenance_for("TRPV1").keys())
        missing = fields - prov_keys
        assert not missing, f"TRPV1 fields without provenance: {missing}"

    @pytest.mark.parametrize("ch_name,channel", [
        ("TRAAK", TRAAK), ("TREK-1", TREK1), ("TREK-2", TREK2),
    ])
    def test_k2p_field_coverage(self, ch_name, channel):
        """Every K2P public field has a provenance entry (Phase-5 audit)."""
        fields = self._public_numeric_fields(channel)
        prov_keys = set(provenance_for(ch_name).keys())
        missing = fields - prov_keys
        assert not missing, (
            f"{ch_name} fields without provenance: {missing}. "
            f"Update provenance.{ch_name.replace('-', '')}_PROVENANCE."
        )


# ---------------------------------------------------------------------------
# K2P tau_act literature pass — Phase-5 follow-up
# ---------------------------------------------------------------------------
class TestK2PTauAct:
    """All four library mechanosensitive channels ship with a literature-fit
    constant tau_act, replacing the bell-shaped τ_w(T) Phase-1 default."""

    @pytest.mark.parametrize("channel,expected", [
        (PIEZO1, 2.0),
        (TRAAK, 3.0),
        (TREK1, 5.0),
        (TREK2, 4.0),
    ])
    def test_default_tau_act_set(self, channel, expected):
        assert channel.tau_act == expected

    @pytest.mark.parametrize("ch_name", ["Piezo1", "TRAAK", "TREK-1", "TREK-2"])
    def test_tau_act_provenance_is_literature(self, ch_name):
        prov = provenance_for(ch_name)["tau_act"]
        assert prov.source == ParamSource.LITERATURE
        assert prov.reference, (
            f"{ch_name}.tau_act marked LITERATURE but missing reference"
        )


# ---------------------------------------------------------------------------
# format_provenance_report: structural sanity
# ---------------------------------------------------------------------------
class TestProvenanceReport:
    def test_report_groups_by_source(self):
        text = format_provenance_report("Piezo1")
        assert "Piezo1" in text
        # At least one LITERATURE field on Piezo1 (T_half, etc.)
        assert "[LITERATURE]" in text
        # And at least one ASSUMPTION (tau_0, inact_T_half)
        assert "[ASSUMPTION]" in text

    def test_report_includes_param_names(self):
        text = format_provenance_report("Piezo1")
        for param in ("T_half", "delta_A", "tau_inact"):
            assert param in text, f"{param} missing from report"


# ---------------------------------------------------------------------------
# tau_act override: replaces bell-shaped τ when set
# ---------------------------------------------------------------------------
class TestTauActOverride:
    def test_default_tau_act_none_keeps_bell(self):
        """A channel with tau_act=None uses the bell-shaped τ."""
        ch = MechanosensitiveChannel(
            name="bell", T_half=3.0, delta_A=15.0, g_max=10, g_density=0.01,
            E_rev=0.0, tau_0=2.0,
        )
        assert ch.tau_act is None
        # At T_half (P_o = 0.5), bell formula gives τ = τ_0 / (1.0 + 0.01) ≈ 1.98 ms
        tau_at_half = ch.time_constant(3.0)
        assert 1.9 < tau_at_half < 2.0
        # Far from T_half, τ = τ_0 / 0.01 ≈ 200 ms
        tau_far = ch.time_constant(0.0)
        assert 100 < tau_far < 300

    def test_tau_act_set_overrides_bell(self):
        """A channel with tau_act=X returns X regardless of tension."""
        ch = MechanosensitiveChannel(
            name="fixed", T_half=3.0, delta_A=15.0, g_max=10, g_density=0.01,
            E_rev=0.0, tau_0=2.0, tau_act=1.5,
        )
        for T in (0.0, 1.5, 3.0, 5.0, 100.0):
            np.testing.assert_allclose(ch.time_constant(T), 1.5)

    def test_tau_act_array_input(self):
        ch = copy.deepcopy(PIEZO1)
        # Piezo1 has tau_act=2 ms (Phase-5 default); array input returns array
        T = np.array([0.0, 2.7, 10.0])
        tau = ch.time_constant(T)
        assert isinstance(tau, np.ndarray)
        np.testing.assert_allclose(tau, np.full(3, 2.0))

    def test_piezo1_default_uses_tau_act(self):
        """The library PIEZO1 ships with tau_act=2 ms (Coste 2010 fit)."""
        assert PIEZO1.tau_act == 2.0
