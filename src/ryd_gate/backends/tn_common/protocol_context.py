"""Shared protocol-lowering helpers for tensor-network backends.

All TN backends resolve a :class:`~ryd_gate.protocols.base.Protocol` against a
:class:`~ryd_gate.backends.tn_common.lattice_spec.TNLatticeSpec` rather than a full
``RydbergSystem``. :class:`TNProtocolContext` is the minimal system-like adapter that
``protocol._resolve(context)`` needs (``N``, ``basis.n_sites``,
``level_structure``, ``interaction_pairs``), and
``pin_deltas_from_params``/``merge_pin_deltas`` turn the resolved context into
per-site local-detuning profiles. Keeping them here avoids each backend (TeNPy MPS,
YASTN PEPS) carrying its own copy.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from ryd_gate.backends.tn_common.lattice_spec import TNLatticeSpec


class TNProtocolContext:
    """Minimal ``RydbergSystem``-like adapter over a :class:`TNLatticeSpec`.

    A ``Protocol`` resolves against a spec instead of a full system.  This
    exposes the system attributes ``_resolve`` may consult — ``N`` /
    ``basis.n_sites`` (to size per-site profiles), ``interaction_pairs``, and
    ``level_structure`` (for analog_3 specs with ``local_blocks``, a view
    carrying the blocks' ``rabi_eff`` / ``time_scale`` and a *unit* 420 ratio,
    since the TN model itself carries the full g-e Rabi).
    """

    def __init__(self, spec: TNLatticeSpec) -> None:
        self._spec = spec
        self.basis = SimpleNamespace(n_sites=spec.N)

    @property
    def N(self) -> int:
        return self._spec.N

    @property
    def interaction_pairs(self) -> tuple:
        return tuple(
            (int(i), int(j), float(self._spec.V_nn) * float(v_rel))
            for i, j, v_rel in self._spec.vdw_pairs
        )

    @property
    def level_structure(self):
        blocks = self._spec.local_blocks
        if blocks is not None:
            # analog_3: the TN model carries the full g-e Rabi in local_blocks,
            # so the protocol emits the *unitless* envelope on E[e,g] (unit ratio).
            return SimpleNamespace(
                name=self._spec.level_structure,
                rabi_eff=blocks.rabi_eff,
                time_scale=blocks.time_scale,
                laser_channel_ratios={"420": {"E[e,g]": 1.0}},
            )
        return self._spec.level_spec


def require_tn_dt(dt) -> float:
    """Validate the mandatory TN step option ``backend_options={"dt": ...}``.

    ``dt`` is the ONLY step-control option (the ``n_steps`` option and the
    silent 200-step default were removed): a TN evolution without an explicit,
    finite, positive ``dt`` is a preflight error.
    """
    if dt is None:
        raise ValueError(
            "TN evolution requires an explicit time step: pass "
            "backend_options={'dt': ...} (the 'n_steps' option and the "
            "200-step default were removed)."
        )
    dt = float(dt)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(
            f"backend_options['dt'] must be a finite positive time step; got {dt!r}."
        )
    return dt


@dataclass(frozen=True)
class TNSegment:
    """One anchor-to-anchor evolution stretch: ``n_sub`` equal steps of ``dt_sub``.

    ``record`` marks segments that end on a *requested* measurement anchor
    (``t1``); the trailing internal ``t_gate`` segment of an endpoint-free
    request evolves without recording.
    """

    t0: float
    t1: float
    n_sub: int
    dt_sub: float
    record: bool


def plan_tn_segments(
    t_gate: float, record_times: np.ndarray, dt: float
) -> tuple[bool, list[TNSegment]]:
    """Plan the piecewise-constant TN stepping through exact sampling anchors.

    ``record_times`` are the validated measurement times (the requested
    ``t_eval``, or ``[t_gate]`` for an endpoint-only request).  Sampling
    anchors are those times plus the internal ``t_gate`` endpoint; between
    consecutive anchors ``[a, b]`` the evolution takes
    ``n_sub = ceil((b - a) / dt)`` equal steps of ``(b - a) / n_sub``, so every
    anchor is hit exactly (segment endpoints are the anchor values themselves —
    no accumulated ``dt`` arithmetic, no rounding of time labels).

    Returns ``(record_at_start, segments)``; ``record_at_start`` is True when
    ``t=0.0`` was itself requested (measured before the first step).
    """
    t_gate = float(t_gate)
    dt = float(dt)
    times = np.asarray(record_times, dtype=float)
    record_at_start = bool(times.size and times[0] == 0.0)
    anchors: list[tuple[float, bool]] = [(float(t), True) for t in times if t > 0.0]
    if not anchors or not np.isclose(anchors[-1][0], t_gate, rtol=1e-12, atol=0.0):
        anchors.append((t_gate, False))
    segments: list[TNSegment] = []
    prev = 0.0
    for t_anchor, record in anchors:
        span = t_anchor - prev
        # The (1 - 1e-12) shave keeps float fuzz in span/dt from adding a step
        # when the ratio is an exact integer.
        n_sub = max(1, int(np.ceil(span / dt * (1.0 - 1e-12))))
        segments.append(TNSegment(prev, t_anchor, n_sub, span / n_sub, record))
        prev = t_anchor
    return record_at_start, segments


def analog3_dt_guard(spec, dt) -> None:
    """Raise if an analog_3 time step is catastrophically coarse (wrong units).

    analog_3 runs in physical seconds with off-diagonal couplings ~2pi*491 MHz, so
    the natural-unit default ``dt`` (0.05/0.2) gives ~1 step over a ~0.5 us gate.
    """
    if getattr(spec, "level_structure", None) != "analog_3" or spec.local_blocks is None:
        return
    time_scale = spec.local_blocks.time_scale
    if float(dt) >= time_scale:
        raise ValueError(
            f"analog_3 needs a small TN step: dt={float(dt):.3g}s >= the effective Rabi "
            f"period time_scale={time_scale:.3g}s. Pass dt ~ time_scale/20-40 "
            f"(about {time_scale / 30:.2g}s); the default natural-unit dt is wrong for analog_3."
        )


def pin_deltas_from_params(params: dict, n_sites: int) -> np.ndarray | None:
    """Return a length-``n_sites`` local-detuning profile, or ``None`` if unset."""
    pin_map = params.get("pin_deltas") or {}
    if not pin_map:
        return None
    pin = np.zeros(n_sites)
    for idx, value in pin_map.items():
        if int(idx) < n_sites:
            pin[int(idx)] = float(value)
    return pin


def merge_pin_deltas(*profiles: np.ndarray | None, n_sites: int) -> np.ndarray | None:
    """Sum any number of optional per-site profiles, or ``None`` if all absent."""
    merged = np.zeros(n_sites)
    any_profile = False
    for profile in profiles:
        if profile is None:
            continue
        merged += profile
        any_profile = True
    return merged if any_profile else None
