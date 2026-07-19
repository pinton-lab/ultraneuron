"""Per-phase ``summary.json`` base schema and write helper.

Every JSON summary written by either repo carries a fixed top-level
metadata block so consumers can reason about provenance without
inspecting per-phase keys::

    {
      "schema_version": "1.0",
      "case": {"name": ..., "paper": ..., "doi": ...},
      "phase": "thermal",
      "generated_utc": "2026-04-30T15:13:42Z",
      "git_sha": {"acoustoelastic_ultrasound_neurons": "abc1234", ...},
      "inputs": {...},   # phase-specific input parameters
      "metrics": {...},  # phase-specific scalar metrics
    }

:func:`write_summary_json` injects ``schema_version``, ``phase``, and
``generated_utc`` automatically; the caller supplies ``case``,
``inputs``, ``metrics``, and (optionally) ``git_sha``.

:func:`validate_summary` checks the structure of an already-loaded
dict and raises ``ValueError`` on the first violation, with messages
that name the offending key.
"""

from __future__ import annotations

import datetime
import json
import os
from typing import Any, Final

from .units import validate_key

SCHEMA_VERSION: Final[str] = "1.0"

#: Phase tokens accepted in the top-level ``"phase"`` field. Mirrors the
#: pipeline subdirectory names produced by the case drivers.
VALID_PHASES: Final[tuple[str, ...]] = (
    "transcranial_acoustic",
    "thermal",
    "elastic",
    "firing_static",
    "firing_temporal",
    "neuro_views",
    "freefield",
)


def _now_utc_iso() -> str:
    """Return the current UTC time in ISO-8601 with second precision."""
    return datetime.datetime.now(tz=datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def write_summary_json(
    path: str | os.PathLike,
    *,
    phase: str,
    case: dict[str, str],
    inputs: dict[str, Any],
    metrics: dict[str, Any],
    git_sha: dict[str, str] | None = None,
    indent: int = 2,
) -> None:
    """Serialise a phase summary to *path* with the standard top-level block.

    Parameters
    ----------
    path : str | PathLike
        Destination JSON path (overwritten).
    phase : str
        Pipeline phase. Must be in :data:`VALID_PHASES`.
    case : dict
        Case metadata --- at minimum ``{"name": ..., "paper": ...}``.
        ``"doi"`` is recommended but optional.
    inputs : dict
        Phase-specific input parameters. Top-level keys must follow the
        ``<name>_<unit>`` convention (see :func:`validate_key`); nested
        dicts are not validated recursively.
    metrics : dict
        Phase-specific scalar metrics. Same key convention as ``inputs``.
        Nested ``"biophysics"`` sub-dict is allowed for biophysics-unit
        keys (mV, uA_per_cm2, mN_per_m, ms, ...) and is not validated.
    git_sha : dict, optional
        Mapping of repo-name to SHA. If ``None``, the field is omitted.
    indent : int
        ``json.dump`` indent. Default 2.

    Raises
    ------
    ValueError
        If ``phase`` is unknown, ``case`` is missing required keys, or
        any flat key in ``inputs`` / ``metrics`` violates the ``<name>_<unit>``
        convention.
    """
    if phase not in VALID_PHASES:
        msg = (
            f"Unknown phase '{phase}'. Allowed: {VALID_PHASES}."
        )
        raise ValueError(msg)

    if not isinstance(case, dict) or "name" not in case or "paper" not in case:
        msg = (
            "case must be a dict with at least 'name' and 'paper' keys; "
            f"got {case!r}."
        )
        raise ValueError(msg)

    for k in inputs:
        validate_key(k)
    for k, v in metrics.items():
        if k == "biophysics" and isinstance(v, dict):
            continue  # biophysics sub-namespace exempt
        validate_key(k)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "case": dict(case),
        "phase": phase,
        "generated_utc": _now_utc_iso(),
        "inputs": inputs,
        "metrics": metrics,
    }
    if git_sha is not None:
        payload["git_sha"] = dict(git_sha)

    # Insert git_sha after generated_utc for readability when present
    if git_sha is not None:
        ordered = {
            "schema_version": payload["schema_version"],
            "case": payload["case"],
            "phase": payload["phase"],
            "generated_utc": payload["generated_utc"],
            "git_sha": payload["git_sha"],
            "inputs": payload["inputs"],
            "metrics": payload["metrics"],
        }
        payload = ordered

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=indent, sort_keys=False)
        fh.write("\n")


def validate_summary(payload: dict[str, Any]) -> None:
    """Raise ``ValueError`` if *payload* does not satisfy the base schema.

    Used by CI conformance tests that walk a ``validation_results/``
    tree and validate every JSON in place. Phase-specific input/metric
    keys are checked against the ``<name>_<unit>`` convention; their
    values are not constrained.
    """
    required = ("schema_version", "case", "phase", "generated_utc",
                "inputs", "metrics")
    for k in required:
        if k not in payload:
            msg = f"Summary missing required top-level key '{k}'."
            raise ValueError(msg)

    if payload["schema_version"] != SCHEMA_VERSION:
        msg = (
            f"schema_version mismatch: got {payload['schema_version']!r}, "
            f"expected {SCHEMA_VERSION!r}."
        )
        raise ValueError(msg)

    case = payload["case"]
    if not isinstance(case, dict) or "name" not in case or "paper" not in case:
        msg = (
            "case block must be a dict with at least 'name' and 'paper' keys."
        )
        raise ValueError(msg)

    if payload["phase"] not in VALID_PHASES:
        msg = (
            f"phase '{payload['phase']}' not in {VALID_PHASES}."
        )
        raise ValueError(msg)

    inputs = payload["inputs"]
    metrics = payload["metrics"]
    if not isinstance(inputs, dict) or not isinstance(metrics, dict):
        msg = "inputs and metrics must both be dicts."
        raise ValueError(msg)
    for k in inputs:
        validate_key(k)
    for k, v in metrics.items():
        if k == "biophysics" and isinstance(v, dict):
            continue
        validate_key(k)
