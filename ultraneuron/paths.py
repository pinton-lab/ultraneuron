"""Resolve optional external resources without hard-coding absolute paths.

The transcranial demonstration scripts and the validation-report builders need
a few resources that live outside this repository: sibling checkouts (the
angular-spectrum and sparse-transcranial helpers), a cached micro-CT skull
volume from Zenodo, and a local Fullwave build.  None of these are dependencies
of the core neural-simulation package; the resolvers below keep their locations
out of the source so the code runs on any machine.

Every resolver honours an environment-variable override first, then falls back
to a conventional location derived from the user's home directory or from this
repository's parent directory (the sibling-checkout layout used in
development).  See :mod:`ultraneuron.registration` for the
neuromod_parameters placement code, which follows the same convention.
"""
from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "repo_root",
    "sibling_repo",
    "halle_skull_nrrd",
    "sparse_connector_mat",
    "angular_spectrum_dir",
    "fullwave_root",
    "redact_path",
]


def redact_path(path) -> str:
    """Replace the user's home directory in ``path`` with ``~``.

    Used when writing provenance (manifests, report tables) so those records do
    not leak absolute paths or the account name.  Paths outside ``$HOME`` are
    returned unchanged.
    """
    text = str(path)
    home = str(Path.home())
    if text == home:
        return "~"
    if text.startswith(home + os.sep):
        return "~" + text[len(home):]
    return text


def repo_root() -> Path:
    """Root of this repository (…/acoustoelastic_ultrasound_neurons)."""
    return Path(__file__).resolve().parents[1]


def _env_or(env_var: str, default: Path) -> Path:
    value = os.environ.get(env_var)
    return Path(value).expanduser() if value else default


def sibling_repo(dirname: str, env_var: str) -> Path:
    """A sibling repository conventionally checked out next to this one.

    ``$<env_var>`` overrides; otherwise ``<repo parent>/<dirname>``.  The path
    is returned whether or not it exists — callers decide how to react.
    """
    return _env_or(env_var, repo_root().parent / dirname)


def halle_skull_nrrd() -> Path:
    """Micro-CT Halle skull NRRD (Zenodo cache).  ``$HALLE_SKULL_NRRD`` overrides."""
    return _env_or(
        "HALLE_SKULL_NRRD",
        Path.home() / ".cache" / "fullwave25" / "skull_microCT_zenodo" / "halle_skull.nrrd",
    )


def sparse_connector_mat() -> Path:
    """``Sparse_TransConnector.mat`` from the fullwave2_sparse_transcranial sibling.

    ``$SPARSE_CONNECTOR_MAT`` overrides the full file path; otherwise it is
    ``Sparse_TransConnector.mat`` inside the fullwave2_sparse_transcranial
    sibling repo (which ``$FULLWAVE_SPARSE_DIR`` may relocate).
    """
    value = os.environ.get("SPARSE_CONNECTOR_MAT")
    if value:
        return Path(value).expanduser()
    sibling = sibling_repo("fullwave2_sparse_transcranial", "FULLWAVE_SPARSE_DIR")
    return sibling / "Sparse_TransConnector.mat"


def angular_spectrum_dir() -> Path:
    """The angularspectrum_heterogeneous sibling repo.  ``$ANGULARSPECTRUM_DIR`` overrides."""
    return sibling_repo("angularspectrum_heterogeneous", "ANGULARSPECTRUM_DIR")


def fullwave_root() -> Path:
    """Local Fullwave build tree.  ``$FULLWAVE_ROOT`` overrides; else ``~/fullwave25-speckle``."""
    return _env_or("FULLWAVE_ROOT", Path.home() / "fullwave25-speckle")
