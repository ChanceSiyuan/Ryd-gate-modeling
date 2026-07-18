"""Named parameter packing shared by the qoc solvers (ADR-0016).

The public candidate representation is an ordinary mapping from names to real
scalars or real arrays. Solvers privately pack those values into one flat
float64 vector of packed coordinates; packing and unpacking preserve the
public names and shapes.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Union

import numpy as np

ParameterValue = Union[float, np.ndarray]
ParameterMapping = Mapping[str, ParameterValue]


def _as_real_block(name: str, value: object, *, what: str = "parameter") -> np.ndarray:
    """Return ``value`` as a finite real float64 array (0-d for scalars)."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{what} {name!r} must be a real scalar or real array; got a boolean.")
    arr = np.asarray(value)
    if arr.dtype == object or not np.issubdtype(arr.dtype, np.number):
        raise TypeError(f"{what} {name!r} must be a real scalar or real array; got dtype {arr.dtype!r}.")
    if np.iscomplexobj(arr):
        raise TypeError(f"{what} {name!r} must be real; got complex values.")
    out = np.asarray(arr, dtype=float)
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{what} {name!r} must contain only finite values.")
    return out


class ParameterPacker:
    """Pack and unpack one named parameter mapping to and from flat coordinates.

    The name order, scalar-versus-array kind, and array shapes are fixed by the
    initial candidate ``x0``; every later mapping must use exactly the same
    names and shapes.
    """

    def __init__(self, x0: ParameterMapping) -> None:
        if not isinstance(x0, Mapping):
            raise TypeError(f"x0 must be a mapping from names to real scalars or real arrays; got {type(x0).__name__}.")
        if len(x0) == 0:
            raise ValueError("x0 must contain at least one named parameter.")
        self._names: tuple[str, ...] = ()
        self._shapes: dict[str, tuple[int, ...]] = {}
        self._is_scalar: dict[str, bool] = {}
        names: list[str] = []
        for name, value in x0.items():
            if not isinstance(name, str):
                raise TypeError(f"parameter names must be strings; got {type(name).__name__}.")
            block = _as_real_block(name, value)
            names.append(name)
            self._shapes[name] = block.shape
            self._is_scalar[name] = block.ndim == 0
        self._names = tuple(names)
        self._sizes = {name: int(np.prod(self._shapes[name], dtype=int)) for name in self._names}
        self.size = int(sum(self._sizes.values()))

    @property
    def names(self) -> tuple[str, ...]:
        return self._names

    def pack(self, params: ParameterMapping, *, what: str = "parameter") -> np.ndarray:
        """Flatten one named mapping into packed coordinates."""
        if not isinstance(params, Mapping):
            raise TypeError(f"expected a named {what} mapping; got {type(params).__name__}.")
        if set(params.keys()) != set(self._names):
            missing = sorted(set(self._names) - set(params.keys()))
            extra = sorted(set(params.keys()) - set(self._names))
            raise ValueError(f"named {what} mapping does not match x0: missing {missing}, unexpected {extra}.")
        parts = []
        for name in self._names:
            block = _as_real_block(name, params[name], what=what)
            if block.shape != self._shapes[name]:
                raise ValueError(
                    f"{what} {name!r} must have shape {self._shapes[name]}; got shape {block.shape}."
                )
            parts.append(np.ravel(block))
        return np.concatenate(parts)

    def unpack(self, x: np.ndarray) -> dict[str, float | np.ndarray]:
        """Rebuild the named mapping (original names and shapes) from flat coordinates."""
        flat = np.asarray(x, dtype=float)
        if flat.shape != (self.size,):
            raise ValueError(f"packed coordinates must have shape ({self.size},); got {flat.shape}.")
        out: dict[str, float | np.ndarray] = {}
        pos = 0
        for name in self._names:
            n = self._sizes[name]
            block = flat[pos : pos + n]
            pos += n
            out[name] = float(block[0]) if self._is_scalar[name] else block.reshape(self._shapes[name]).copy()
        return out

    def _broadcast(self, name: str, value: object, *, what: str) -> np.ndarray:
        """Broadcast one scalar-or-array entry across parameter ``name``; return flat float64."""
        arr = np.asarray(value, dtype=float)
        shape = self._shapes[name]
        if arr.ndim == 0:
            return np.full(self._sizes[name], float(arr))
        if arr.shape != shape:
            raise ValueError(f"{what} for {name!r} must be a scalar or match shape {shape}; got shape {arr.shape}.")
        return np.ravel(np.asarray(arr, dtype=float))

    def pack_bounds(self, bounds: Mapping[str, tuple] | None) -> list[tuple[float, float]] | None:
        """Turn a named bounds mapping into per-coordinate ``(lower, upper)`` pairs.

        A parameter omitted from the mapping is unbounded; scalar limits
        broadcast across an array parameter (ADR-0016).
        """
        if bounds is None:
            return None
        if not isinstance(bounds, Mapping):
            raise TypeError(f"bounds must be a mapping from names to (lower, upper) pairs; got {type(bounds).__name__}.")
        unknown = sorted(set(bounds.keys()) - set(self._names))
        if unknown:
            raise ValueError(f"bounds given for unknown parameters {unknown}.")
        lower = {name: np.full(self._sizes[name], -np.inf) for name in self._names}
        upper = {name: np.full(self._sizes[name], np.inf) for name in self._names}
        for name, pair in bounds.items():
            try:
                lo, hi = pair
            except (TypeError, ValueError):
                raise TypeError(f"bounds[{name!r}] must be a (lower, upper) pair; got {pair!r}.") from None
            lower[name] = self._broadcast(name, lo, what="lower bound")
            upper[name] = self._broadcast(name, hi, what="upper bound")
            if np.any(np.isnan(lower[name])) or np.any(np.isnan(upper[name])):
                raise ValueError(f"bounds[{name!r}] must not contain NaN.")
            if np.any(lower[name] > upper[name]):
                raise ValueError(f"bounds[{name!r}] must satisfy lower <= upper.")
        flat_lower = np.concatenate([lower[name] for name in self._names])
        flat_upper = np.concatenate([upper[name] for name in self._names])
        return list(zip(flat_lower.tolist(), flat_upper.tolist()))

    def pack_scales(self, scales: Mapping[str, object] | None) -> np.ndarray:
        """Turn a named scales mapping into flat positive per-coordinate scales.

        Missing scales equal one and are never inferred (ADR-0016).
        """
        flat = np.ones(self.size)
        if scales is None:
            return flat
        if not isinstance(scales, Mapping):
            raise TypeError(f"scales must be a mapping from names to positive scales; got {type(scales).__name__}.")
        unknown = sorted(set(scales.keys()) - set(self._names))
        if unknown:
            raise ValueError(f"scales given for unknown parameters {unknown}.")
        pos = 0
        for name in self._names:
            n = self._sizes[name]
            if name in scales:
                block = self._broadcast(name, scales[name], what="scale")
                if not np.all(np.isfinite(block)) or np.any(block <= 0.0):
                    raise ValueError(f"scales[{name!r}] must be positive and finite.")
                flat[pos : pos + n] = block
            pos += n
        return flat
