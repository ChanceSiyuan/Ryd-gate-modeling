"""Schema-tag helpers for the plain-dict serialization contract.

Every product object serializes to a JSON-compatible dict tagged with
``"schema": "ryd-gate/<kind>/v1"`` and reconstructs via ``from_dict``. These
helpers keep the tag format and the numpy-conversion rule in one place; each
class still owns its own ``to_dict``/``from_dict``.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

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
