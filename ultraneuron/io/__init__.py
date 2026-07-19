"""Data and plot/image standards for simulation outputs.

This sub-package provides shared I/O utilities used by both
``acoustoelastic_ultrasound_neurons`` (this framework) and the sister
case-driver repository ``neuromod_parameters``:

* :mod:`ultraneuron.io.units` --- SI / biophysics unit
  conversions and a ``<name>_<unit>`` JSON-key validator.
* :mod:`ultraneuron.io.plotstyle` --- shared matplotlib
  rcParams, categorical palette, figure-size presets, paired
  PDF + PNG saving, and a filename-convention validator.
* :mod:`ultraneuron.io.schema` --- the per-phase
  ``summary.json`` base schema and a write helper that injects
  the required top-level metadata.
"""

from .plotstyle import (
    FIGSIZE,
    PALETTE,
    apply_publication_style,
    apply_screen_style,
    colorbar_label,
    savefig_both,
    validate_plot_filename,
)
from .schema import (
    SCHEMA_VERSION,
    VALID_PHASES,
    validate_summary,
    write_summary_json,
)
from .units import from_SI, to_SI, validate_key

__all__ = [
    "FIGSIZE",
    "PALETTE",
    "SCHEMA_VERSION",
    "VALID_PHASES",
    "apply_publication_style",
    "apply_screen_style",
    "colorbar_label",
    "from_SI",
    "savefig_both",
    "to_SI",
    "validate_key",
    "validate_plot_filename",
    "validate_summary",
    "write_summary_json",
]
