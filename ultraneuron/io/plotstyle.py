"""Shared plot styling, palette, figure sizes, and filename convention.

Both ``acoustoelastic_ultrasound_neurons`` (figure-generation in
``docs/generate_figures.py``) and the sister case-driver repo
``neuromod_parameters`` import this module so a single change to the
publication style propagates everywhere.

Usage::

    from ultraneuron.io import (
        apply_publication_style, PALETTE, FIGSIZE, savefig_both,
    )
    apply_publication_style()
    fig, ax = plt.subplots(figsize=FIGSIZE["two_column"])
    ax.plot(x, y, color=PALETTE["thermal"])
    savefig_both(fig, "/path/to/figdir/thermal_ortho")
"""

from __future__ import annotations

import re
from typing import Final


# ---------------------------------------------------------------------
# Categorical palette --- one colour per physical pathway
# ---------------------------------------------------------------------

PALETTE: Final[dict[str, str]] = {
    "acoustic":   "#1f77b4",  # pressure / radiation force --- blue
    "thermal":    "#d62728",  # temperature / bioheat       --- red
    "mechanical": "#2ca02c",  # strain / tension            --- green
    "neural":     "#9467bd",  # voltage / firing            --- purple
    "highlight":  "#ff7f0e",  # call-out / annotations      --- orange
    "neutral":    "#7f7f7f",  # baselines / reference lines --- grey
}


# ---------------------------------------------------------------------
# Figure sizes (width, height) in inches
# ---------------------------------------------------------------------

FIGSIZE: Final[dict[str, tuple[float, float]]] = {
    "single_column":   (3.5, 2.6),
    "1.5_column":      (5.0, 2.8),
    "two_column":      (7.0, 2.6),
    "two_column_tall": (7.0, 4.0),
    "square":          (3.5, 3.5),
}


# ---------------------------------------------------------------------
# rcParams presets
# ---------------------------------------------------------------------

_PUBLICATION_RC: Final[dict[str, object]] = {
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "lines.linewidth": 1.2,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
}

_SCREEN_RC: Final[dict[str, object]] = {
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    "lines.linewidth": 1.6,
    "axes.linewidth": 0.8,
}


def apply_publication_style() -> None:
    """Set matplotlib rcParams for publication PDFs (8--9 pt serif, 300 dpi)."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(_PUBLICATION_RC)


def apply_screen_style() -> None:
    """Set matplotlib rcParams for slide / screen PNGs (10--11 pt sans, 150 dpi)."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(_SCREEN_RC)


# ---------------------------------------------------------------------
# Save helper --- always writes both PDF (vector) and PNG (raster)
# ---------------------------------------------------------------------


def savefig_both(fig, path_no_ext: str, dpi: int = 300) -> tuple[str, str]:
    """Save *fig* as both ``<path_no_ext>.pdf`` and ``<path_no_ext>.png``.

    Returns the two written paths so callers can log them.
    """
    pdf_path = f"{path_no_ext}.pdf"
    png_path = f"{path_no_ext}.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.05, dpi=dpi)
    return pdf_path, png_path


# ---------------------------------------------------------------------
# Colorbar / axis label helper
# ---------------------------------------------------------------------


def colorbar_label(quantity: str, unit: str) -> str:
    """Return a standardised ``"Quantity (unit)"`` label string.

    Use ``""`` for ``unit`` to indicate dimensionless quantities; the
    parenthetical is then suppressed.
    """
    return f"{quantity} ({unit})" if unit else f"{quantity}"


# ---------------------------------------------------------------------
# Plot filename convention
# ---------------------------------------------------------------------

#: Allowed pipeline-phase tokens that may begin a plot filename.
_VALID_PHASES = (
    "transcranial",
    "thermal",
    "elastic",
    "firing_static",
    "firing_temporal",
    "neuroview",
    "pulse",
    "setup",
)

#: Allowed view tokens.
_VALID_VIEWS = (
    "mip",
    "ortho",
    "trace",
    "psth",
    "raster",
    "dose",
    "summary",
)

# Build a regex once. Examples that should match:
#   thermal_ortho.png
#   firing_static_mip_alpha5000.png
#   firing_temporal_psth.png
#   elastic_slice_z42.png
_PHASE_TOKEN = "(?:" + "|".join(_VALID_PHASES) + ")"
_VIEW_TOKEN = "(?:" + "|".join(_VALID_VIEWS) + r"|slice_[xyz][0-9]+)"
_PARAM_TOKEN = r"(?:_[A-Za-z][A-Za-z0-9p]*)*"  # zero or more "_keyvalue" suffixes
_FILENAME_RE = re.compile(rf"^{_PHASE_TOKEN}_{_VIEW_TOKEN}{_PARAM_TOKEN}\.(?:pdf|png)$")


def validate_plot_filename(path: str) -> None:
    """Raise ``ValueError`` if *path*'s basename does not match the convention.

    The convention is ``<phase>_<view>[_<param>...]\\.(pdf|png)``, where:

    * **phase** is one of ``transcranial``, ``thermal``, ``elastic``,
      ``firing_static``, ``firing_temporal``, ``neuroview``, ``pulse``,
      ``setup``;
    * **view** is one of ``mip``, ``ortho``, ``trace``, ``psth``,
      ``raster``, ``dose``, ``summary``, or ``slice_<x|y|z><N>``;
    * **param** suffixes are zero or more ``_<key><value>`` tokens with
      letters, digits, and ``p`` (used for ``.``).
    """
    import os

    name = os.path.basename(path)
    if not _FILENAME_RE.match(name):
        msg = (
            f"Plot filename '{name}' does not match the standard. "
            "Expected '<phase>_<view>[_<param>...].(pdf|png)' with phase in "
            f"{_VALID_PHASES} and view in {_VALID_VIEWS} (or slice_<axis><N>). "
            "Examples: 'thermal_ortho.png', "
            "'firing_static_mip_alpha5000.pdf', 'firing_temporal_psth.png'."
        )
        raise ValueError(msg)
