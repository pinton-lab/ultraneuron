"""Tests for the I/O standards module: units, schema, plot conventions."""

from __future__ import annotations

import json

import pytest

from ultraneuron.io import (
    FIGSIZE,
    PALETTE,
    SCHEMA_VERSION,
    VALID_PHASES,
    apply_publication_style,
    colorbar_label,
    from_SI,
    savefig_both,
    to_SI,
    validate_key,
    validate_plot_filename,
    validate_summary,
    write_summary_json,
)


# ---------------------------------------------------------------------
# Unit conversions
# ---------------------------------------------------------------------
class TestUnits:
    def test_voltage_round_trip(self):
        # 1 mV → 0.001 V → 1 mV
        assert to_SI(1.0, "mV") == pytest.approx(1e-3)
        assert from_SI(1e-3, "mV") == pytest.approx(1.0)

    def test_capacitance_round_trip(self):
        # 1 μF/cm² → 0.01 F/m² (the C_m0 used in NICE)
        assert to_SI(1.0, "uF_per_cm2") == pytest.approx(0.01)
        assert from_SI(0.01, "uF_per_cm2") == pytest.approx(1.0)

    def test_conductance_round_trip(self):
        # 1 mS/cm² → 10 S/m²
        assert to_SI(1.0, "mS_per_cm2") == pytest.approx(10.0)

    def test_tension_round_trip(self):
        # 2.7 mN/m → 2.7e-3 N/m (Piezo1 T_half)
        assert to_SI(2.7, "mN_per_m") == pytest.approx(2.7e-3)
        assert from_SI(2.7e-3, "mN_per_m") == pytest.approx(2.7)

    def test_time_round_trip(self):
        assert to_SI(1.0, "ms") == pytest.approx(1e-3)
        assert to_SI(1.0, "us") == pytest.approx(1e-6)

    def test_length_round_trip(self):
        assert to_SI(1.0, "nm") == pytest.approx(1e-9)
        assert to_SI(2.0, "nm") == pytest.approx(2e-9)  # δ_0

    def test_pressure_round_trip(self):
        assert to_SI(100.0, "kPa") == pytest.approx(1e5)
        assert to_SI(1.0, "MPa") == pytest.approx(1e6)

    def test_unknown_unit_raises(self):
        with pytest.raises(KeyError, match="Unknown biophysics unit"):
            to_SI(1.0, "furlongs_per_fortnight")
        with pytest.raises(KeyError, match="Unknown target unit"):
            from_SI(1.0, "furlongs_per_fortnight")


# ---------------------------------------------------------------------
# JSON-key validator
# ---------------------------------------------------------------------
class TestKeyValidator:
    @pytest.mark.parametrize("key", [
        "peak_dT_K",
        "Q_peak_W_per_m3",
        "tension_peak_N_per_m",
        "peak_pressure_Pa",
        "duty_cycle_fraction",
        "firing_count",
        "f0_Hz",
        "alpha_dB_per_cm_per_MHz",
    ])
    def test_valid_keys_accepted(self, key):
        validate_key(key)  # no raise

    @pytest.mark.parametrize("key", [
        "peakDT",          # no underscore
        "peak_dT",         # no unit suffix
        "my-key_K",        # hyphen
        "peak dT_K",       # space
        "",                # empty
    ])
    def test_invalid_keys_rejected(self, key):
        with pytest.raises(ValueError):
            validate_key(key)


# ---------------------------------------------------------------------
# Plot filename validator
# ---------------------------------------------------------------------
class TestPlotFilenameValidator:
    @pytest.mark.parametrize("name", [
        "thermal_ortho.png",
        "thermal_ortho.pdf",
        "firing_static_mip_alpha5000.png",
        "firing_static_mip_alpha5000.pdf",
        "firing_temporal_psth.png",
        "firing_temporal_raster.png",
        "transcranial_summary.pdf",
        "elastic_slice_z42.png",
        "neuroview_dose.png",
        "pulse_trace.png",
        "setup_summary.png",
    ])
    def test_valid_filenames_accepted(self, name):
        validate_plot_filename(f"/tmp/figs/{name}")  # no raise

    @pytest.mark.parametrize("name", [
        "cumulative_spike_evolution.png",  # free-form, doesn't follow rule
        "firing_alpha5000.png",            # missing static/temporal qualifier
        "thermal.png",                      # no view
        "rogue_phase_ortho.png",            # bad phase token
        "thermal_unknown_view.png",         # bad view token
        "thermal_ortho.svg",                # bad extension
    ])
    def test_invalid_filenames_rejected(self, name):
        with pytest.raises(ValueError, match="does not match the standard"):
            validate_plot_filename(f"/tmp/figs/{name}")


# ---------------------------------------------------------------------
# Schema write + validate roundtrip
# ---------------------------------------------------------------------
class TestSchemaRoundtrip:
    def _minimal_payload(self):
        return dict(
            phase="thermal",
            case={"name": "Legon_2014_S1", "paper": "Legon 2014",
                  "doi": "10.1038/nn.3620"},
            inputs={"f0_Hz": 500e3, "alpha_brain_dB_per_cm_per_MHz": 0.5},
            metrics={"peak_dT_K": 0.115, "fda_1K_compliant_fraction": 1.0},
        )

    def test_write_and_validate_roundtrip(self, tmp_path):
        path = tmp_path / "summary.json"
        write_summary_json(path, **self._minimal_payload())
        with open(path) as fh:
            payload = json.load(fh)
        validate_summary(payload)
        # Top-level keys present
        for k in ("schema_version", "case", "phase",
                  "generated_utc", "inputs", "metrics"):
            assert k in payload
        assert payload["schema_version"] == SCHEMA_VERSION
        assert payload["phase"] == "thermal"
        assert payload["case"]["name"] == "Legon_2014_S1"

    def test_unknown_phase_rejected(self, tmp_path):
        bad = self._minimal_payload()
        bad["phase"] = "spaghetti"
        with pytest.raises(ValueError, match="Unknown phase"):
            write_summary_json(tmp_path / "x.json", **bad)

    def test_missing_case_field_rejected(self, tmp_path):
        bad = self._minimal_payload()
        bad["case"] = {"name": "X"}  # missing 'paper'
        with pytest.raises(ValueError, match="case must be a dict"):
            write_summary_json(tmp_path / "x.json", **bad)

    def test_invalid_input_key_rejected(self, tmp_path):
        bad = self._minimal_payload()
        bad["inputs"] = {"badKey": 1.0}  # no <name>_<unit> form
        with pytest.raises(ValueError, match="does not match"):
            write_summary_json(tmp_path / "x.json", **bad)

    def test_biophysics_namespace_exempt(self, tmp_path):
        """Keys inside metrics.biophysics need not follow <name>_<unit>."""
        payload = self._minimal_payload()
        payload["metrics"]["biophysics"] = {
            "Vrest_mV": -70.0,
            "spike_count": 4,
            # Free-form key allowed inside biophysics:
            "anyOldThing": 1.23,
        }
        path = tmp_path / "summary.json"
        write_summary_json(path, **payload)  # no raise
        with open(path) as fh:
            validate_summary(json.load(fh))

    def test_git_sha_block_optional(self, tmp_path):
        payload = self._minimal_payload()
        path = tmp_path / "summary.json"
        write_summary_json(path, git_sha={"ultraneuron": "abc1234"},
                           **payload)
        with open(path) as fh:
            data = json.load(fh)
        assert data["git_sha"]["ultraneuron"] == "abc1234"

    def test_validate_rejects_missing_top_level(self):
        with pytest.raises(ValueError, match="missing required top-level"):
            validate_summary({"phase": "thermal"})


# ---------------------------------------------------------------------
# Plotstyle constants and helpers
# ---------------------------------------------------------------------
class TestPlotstyle:
    def test_palette_keys(self):
        for k in ("acoustic", "thermal", "mechanical", "neural",
                  "highlight", "neutral"):
            assert k in PALETTE
            assert PALETTE[k].startswith("#")

    def test_figsize_presets(self):
        for k in ("single_column", "1.5_column", "two_column",
                  "two_column_tall", "square"):
            w, h = FIGSIZE[k]
            assert w > 0 and h > 0

    def test_apply_publication_style_sets_dpi(self):
        import matplotlib.pyplot as plt
        apply_publication_style()
        assert plt.rcParams["figure.dpi"] == 300
        assert plt.rcParams["font.family"] == ["serif"]

    def test_savefig_both_writes_pdf_and_png(self, tmp_path):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=FIGSIZE["single_column"])
        ax.plot([0, 1], [0, 1])
        base = tmp_path / "thermal_ortho"
        pdf, png = savefig_both(fig, str(base))
        plt.close(fig)
        assert pdf.endswith(".pdf")
        assert png.endswith(".png")
        import os
        assert os.path.exists(pdf) and os.path.getsize(pdf) > 0
        assert os.path.exists(png) and os.path.getsize(png) > 0

    def test_colorbar_label(self):
        assert colorbar_label("Pressure", "Pa") == "Pressure (Pa)"
        # Dimensionless: empty unit suppresses parens
        assert colorbar_label("Open probability", "") == "Open probability"

    def test_valid_phases_constant(self):
        assert "thermal" in VALID_PHASES
        assert "firing_temporal" in VALID_PHASES
