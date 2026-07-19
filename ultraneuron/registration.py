"""Locate the external ``neuromod_parameters`` case-driver repository.

The transcranial *placement* / *registration* steps — mapping an MNI target
into the Halle micro-CT frame and projecting the transducer bowl onto the
outer-skull mesh — live in a separate case-driver repository,
``neuromod_parameters``, that is **not** bundled with (and not a dependency of)
``ultraneuron``.  The core neural-simulation and mechanism modules
do not need it; only the placement scripts and the placement figures in the
validation report do.

This module is the single place that knows how to find that optional external
repository, so no absolute path is hard-coded anywhere else.  Resolution order:

1. ``$NEUROMOD_PARAMETERS_DIR`` — an explicit checkout path.
2. A sibling directory named ``neuromod_parameters`` next to this repository
   (the source / editable-install layout).

If neither is found, the placement helpers raise :class:`RegistrationUnavailable`
with an actionable message instead of a bare ``ModuleNotFoundError``.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

__all__ = [
    "ENV_VAR",
    "RegistrationUnavailable",
    "neuromod_parameters_root",
    "registration_dir",
    "ensure_registration_importable",
]

#: Environment variable that overrides where ``neuromod_parameters`` lives.
ENV_VAR = "NEUROMOD_PARAMETERS_DIR"


class RegistrationUnavailable(RuntimeError):
    """The external neuromod_parameters registration code is needed but missing.

    Subclasses :class:`RuntimeError` (not :class:`SystemExit`) so that report
    builders can catch it and fall back to a placeholder figure, while CLI
    scripts still surface the message and exit non-zero.
    """


_MESSAGE = (
    "This step needs the external 'neuromod_parameters' case-driver repository "
    "(skull registration and transducer placement), which is not bundled with "
    "ultraneuron.\n"
    f"Set the {ENV_VAR} environment variable to your checkout, or place it as a "
    "sibling directory named 'neuromod_parameters' next to this repository."
)


def _candidate_roots():
    env = os.environ.get(ENV_VAR)
    if env:
        yield Path(env).expanduser()
    # Sibling of the repo root: <repo_parent>/neuromod_parameters.
    # __file__ = <repo>/ultraneuron/registration.py
    yield Path(__file__).resolve().parents[2] / "neuromod_parameters"


def neuromod_parameters_root() -> Path | None:
    """Return the ``neuromod_parameters`` checkout root, or ``None`` if not found."""
    for root in _candidate_roots():
        if root.is_dir():
            return root
    return None


def registration_dir(required: bool = True) -> Path | None:
    """Return the ``registration/`` subdirectory of ``neuromod_parameters``.

    Returns ``None`` when unavailable and ``required`` is ``False``; raises
    :class:`RegistrationUnavailable` when unavailable and ``required`` is ``True``.
    """
    root = neuromod_parameters_root()
    reg = root / "registration" if root is not None else None
    if reg is not None and reg.is_dir():
        return reg
    if required:
        raise RegistrationUnavailable(_MESSAGE)
    return None


def ensure_registration_importable() -> Path:
    """Put ``neuromod_parameters/registration`` on ``sys.path`` so that
    ``halle_placement`` and ``visualize_placement_3d`` import, and return that
    directory.

    Raises :class:`RegistrationUnavailable` with an actionable message if the
    external repository is not available.
    """
    reg = registration_dir(required=True)
    reg_str = str(reg)
    if reg_str not in sys.path:
        sys.path.insert(0, reg_str)
    return reg
