"""Nested rational axes and the canonical point-key factory.

An axis is three anchor segments; a node is the piecewise-linear interpolation of
the anchors at position p = num/den in "segment units" in [0, 3], with (num, den)
in lowest terms and den a power of two (<= 8 through level 3).  Reduced (num, den)
pairs are the canonical resume keys: they never depend on string-rounded floats,
and a node inserted at level L+1 that coincides with a level-L node reduces to the
identical pair.

``make_pointkey_type`` builds a per-script ``PointKey`` dataclass whose first
field name and id prefix are supplied by the caller, so each sweep script keeps
its serialized field name (``delta_idx`` / ``n_idx``) and id prefix.
"""
from __future__ import annotations

import dataclasses
import math
from fractions import Fraction
from typing import Sequence

LEVEL_DENS = (1, 2, 4, 8)          # nesting level 0..3 -> axis of 3*den+1 nodes
LEVEL_SIZES = (4, 7, 13, 25)       # nodes per axis at each level
LEVEL_FROM_SIZE = {4: 0, 7: 1, 13: 2, 25: 3}


def canon_coord(num: int, den: int) -> tuple[int, int]:
    """Reduce a fractional axis coordinate ``num/den`` to lowest terms."""
    if den <= 0 or num < 0 or num > 3 * den:
        raise ValueError(f"coordinate {num}/{den} outside the axis range [0, 3]")
    g = math.gcd(num, den)
    return num // g, den // g


def coord_value_mhz(anchors: Sequence[Fraction], num: int, den: int) -> Fraction:
    """Exact axis value (MHz) of the reduced coordinate ``num/den``."""
    seg, rem = divmod(num, den)
    if rem == 0:
        return anchors[seg]
    return anchors[seg] + (anchors[seg + 1] - anchors[seg]) * Fraction(rem, den)


def axis_coords(level: int) -> list[tuple[int, int]]:
    """Canonical coordinates of every axis node at nesting ``level`` (0..3)."""
    den = LEVEL_DENS[level]
    return [canon_coord(k, den) for k in range(3 * den + 1)]


def axis_values_mhz(anchors: Sequence[Fraction], level: int) -> list[Fraction]:
    return [coord_value_mhz(anchors, n, d) for n, d in axis_coords(level)]


def make_pointkey_type(
    panel_field: str,
    id_prefix: str,
    omega_anchors: Sequence[Fraction],
    dsweep_anchors: Sequence[Fraction],
    panel_len: int,
    n_t: int,
):
    """Build ``(PointKey, make_key, panel_keys, all_panels, all_keys)`` for a script.

    The returned dataclass's first field is NAMED ``panel_field`` (serialized in the
    live store), and ``.id()`` prefixes the panel index with ``id_prefix``; the
    remaining fields (``t_idx``, ``om_num``, ``om_den``, ``dw_num``, ``dw_den``) and
    the ``.panel``/``.omega_mhz()``/``.dsweep_mhz()``/``.level()`` semantics are shared.
    """

    def _id(self) -> str:
        return (f"{id_prefix}{getattr(self, panel_field)}_t{self.t_idx}"
                f"_om{self.om_num}-{self.om_den}_dw{self.dw_num}-{self.dw_den}")

    def _panel(self) -> tuple[int, int]:
        return (getattr(self, panel_field), self.t_idx)

    def _omega_mhz(self) -> Fraction:
        return coord_value_mhz(omega_anchors, self.om_num, self.om_den)

    def _dsweep_mhz(self) -> Fraction:
        return coord_value_mhz(dsweep_anchors, self.dw_num, self.dw_den)

    def _level(self) -> int:
        """Finest nesting level this node first appears at."""
        den = max(self.om_den, self.dw_den)
        return LEVEL_DENS.index(den)

    PointKey = dataclasses.make_dataclass(
        "PointKey",
        [(panel_field, int), ("t_idx", int),
         ("om_num", int), ("om_den", int), ("dw_num", int), ("dw_den", int)],
        namespace={
            "__doc__": "Canonical identity of one scan node: panel indices + reduced axis coords.",
            "id": _id,
            "panel": property(_panel),
            "omega_mhz": _omega_mhz,
            "dsweep_mhz": _dsweep_mhz,
            "level": _level,
        },
        frozen=True,
        order=True,
    )

    def make_key(panel_idx: int, t_idx: int,
                 om: tuple[int, int], dw: tuple[int, int]):
        om = canon_coord(*om)
        dw = canon_coord(*dw)
        return PointKey(panel_idx, t_idx, om[0], om[1], dw[0], dw[1])

    def panel_keys(panel_idx: int, t_idx: int, level: int) -> list:
        """All nodes of one panel at nesting ``level`` (row-major: omega outer, dw inner)."""
        coords = axis_coords(level)
        return [make_key(panel_idx, t_idx, om, dw) for om in coords for dw in coords]

    def all_panels() -> list[tuple[int, int]]:
        return [(pi, ti) for pi in range(panel_len) for ti in range(n_t)]

    def all_keys(level: int) -> list:
        return [k for pi, ti in all_panels() for k in panel_keys(pi, ti, level)]

    return PointKey, make_key, panel_keys, all_panels, all_keys
