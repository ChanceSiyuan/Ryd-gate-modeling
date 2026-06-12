"""Validation primitives and the plain-dict serialization contract.

Validation: one vocabulary for "what is wrong" — severity, stable
machine-readable code, human-readable message, and a path locating the
offending field. Validation methods accumulate :class:`ValidationIssue`
objects without raising; :func:`raise_for_errors` is the explicit raise
boundary.

Serialization: every product object serializes to a JSON-compatible dict
tagged with ``"schema": "ryd-gate/<kind>/v1"`` and reconstructs via
``from_dict``. These helpers keep the tag format and the numpy-conversion
rule in one place; each class still owns its own ``to_dict``/``from_dict``.

Stage 6 freezes the ``v1`` payloads as JSON Schema files shipped in
``ryd_gate/schemas/``; :func:`validate_json_schema` checks a payload against
its frozen schema (optional ``jsonschema`` dependency — install the
``schema`` extra).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np

# ── Validation primitives ────────────────────────────────────────────────────

ValidationSeverity = Literal["error", "warning"]

_VALID_SEVERITIES = ("error", "warning")


@dataclass(frozen=True)
class ValidationIssue:
    """One validation problem.

    Attributes
    ----------
    severity : {"error", "warning"}
        Errors block execution at ``raise_for_errors``; warnings never raise.
    code : str
        Stable machine-readable code (e.g. ``"register.min_distance"``).
        Codes are API: tests and downstream callers branch on them.
    message : str
        Human-readable description.
    path : tuple[str, ...]
        Location of the invalid field, e.g. ``("register", "coords")``.
    """

    severity: ValidationSeverity
    code: str
    message: str
    path: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.severity not in _VALID_SEVERITIES:
            raise ValueError(
                f"severity must be 'error' or 'warning', got {self.severity!r}."
            )
        if not isinstance(self.code, str) or not self.code:
            raise ValueError("code must be a non-empty string.")
        if not isinstance(self.message, str):
            raise ValueError("message must be a string.")
        if not isinstance(self.path, tuple) or not all(
            isinstance(p, str) for p in self.path
        ):
            raise ValueError("path must be a tuple of strings.")


def raise_for_errors(issues: list[ValidationIssue]) -> None:
    """Raise ``ValueError`` if any issue has severity ``"error"``.

    Returns ``None`` when there are no errors (warnings never raise). The
    raised message lists each error's code and message on its own line.
    """
    errors = [issue for issue in issues if issue.severity == "error"]
    if not errors:
        return None
    lines = [f"{issue.code}: {issue.message}" for issue in errors]
    raise ValueError("validation failed:\n" + "\n".join(lines))


# ── Schema-tag helpers ───────────────────────────────────────────────────────

SCHEMA_PREFIX = "ryd-gate"
SCHEMA_VERSION = "v1"


def schema_tag(kind: str) -> str:
    """Return the schema tag ``'ryd-gate/<kind>/v1'`` for a payload kind."""
    return f"{SCHEMA_PREFIX}/{kind}/{SCHEMA_VERSION}"


def check_schema(data: Any, kind: str) -> None:
    """Raise ``ValueError`` unless *data* is a mapping tagged for *kind*."""
    if not isinstance(data, Mapping):
        raise ValueError(
            f"expected a mapping for {kind!r}, got {type(data).__name__}."
        )
    expected = schema_tag(kind)
    tag = data.get("schema")
    if tag != expected:
        raise ValueError(f"schema mismatch: expected {expected!r}, got {tag!r}.")


def json_ready(value: Any, path: str = "metadata") -> Any:
    """Convert *value* to JSON-compatible types or raise ``ValueError``.

    Numpy scalars become Python scalars, numpy arrays become nested lists,
    tuples become lists. Anything that is not ``str``/``bool``/``int``/
    ``float``/``None``/list/dict after conversion raises, naming *path*.
    """
    if isinstance(value, (str, bool)) or value is None:
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [json_ready(v, f"{path}[{i}]") for i, v in enumerate(value)]
    if isinstance(value, Mapping):
        out = {}
        for key, val in value.items():
            if not isinstance(key, str):
                raise ValueError(
                    f"{path} keys must be strings, got {type(key).__name__}."
                )
            out[key] = json_ready(val, f"{path}.{key}")
        return out
    raise ValueError(
        f"{path} contains a non-JSON-compatible value of type {type(value).__name__}."
    )


# ── Frozen v1 JSON Schemas (Stage 6) ─────────────────────────────────────────


def schema_path(kind: str) -> Path:
    """Filesystem path of the frozen ``<kind>.v1.schema.json`` file."""
    from importlib import resources

    resource = resources.files("ryd_gate") / "schemas" / f"{kind}.v1.schema.json"
    with resources.as_file(resource) as path:
        return Path(path)


def load_json_schema(kind: str) -> dict:
    """Load the frozen JSON Schema document for *kind* via importlib.resources."""
    from importlib import resources

    resource = resources.files("ryd_gate") / "schemas" / f"{kind}.v1.schema.json"
    try:
        text = resource.read_text()
    except FileNotFoundError:
        raise ValueError(f"no frozen schema for kind {kind!r}.") from None
    return json.loads(text)


def validate_json_schema(data: Mapping[str, Any], kind: str) -> list[ValidationIssue]:
    """Validate *data* against the frozen schema for *kind*; never raises.

    Returns one ``serialization.jsonschema_missing`` error when the optional
    ``jsonschema`` dependency is not installed (``pip install ryd-gate[schema]``).
    """
    try:
        import jsonschema
    except ImportError:
        return [ValidationIssue(
            "error", "serialization.jsonschema_missing",
            "schema validation needs the optional 'jsonschema' dependency "
            "(install the 'schema' extra).",
            ("schema", kind),
        )]
    schema = load_json_schema(kind)
    validator = jsonschema.Draft202012Validator(schema)
    return [
        ValidationIssue(
            "error", "serialization.schema", error.message,
            ("schema", kind, *(str(part) for part in error.absolute_path)),
        )
        for error in validator.iter_errors(data)
    ]
