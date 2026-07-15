"""Shared tensor-network lowering from a :class:`RydbergSystem` (the new seam).

Both TN engines (MPS / PEPS) consume a ``RydbergSystem`` exactly like the exact
backend: the private static terms (atomic diagonal + Rydberg pair interaction)
plus the canonical lowered drive channels
(:func:`ryd_gate.core.lowering.lower_drives`).  There is no ``TNProtocolContext``
and no ``TNLatticeSpec`` — the system is the single object the backends see.

Only the effective ``1r`` / ``01r`` presets are TN-capable (O09); their level
labels are drawn from ``("0", "1", "r")`` so TeNPy operator names ``E_<ket>_<bra>``
are valid identifiers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ryd_gate.core.lowering import lower_drives
from ryd_gate.core.operators import (
    RydbergPairInteractionSpec,
    SumProjectorSpec,
    parse_E,
)

_TN_PRESETS = frozenset({"1r", "01r"})


@dataclass(frozen=True)
class _Channel:
    """One lowered drive channel with its per-site matrix-element callable."""

    ket: str
    bra: str
    is_diag: bool
    coefficient: object  # callable t -> complex scalar or (N,) complex array


@dataclass(frozen=True)
class TNTerms:
    """The Hamiltonian ingredients a TN engine needs, all from the system."""

    t_gate: float
    n_sites: int
    levels: tuple[str, ...]
    rydberg_levels: tuple[str, ...]
    onsite_static: tuple[tuple[str, float], ...]      # (level, real energy)
    pairs: tuple[tuple[int, int, float], ...]         # (i, j, V) with i < j
    channels: tuple[_Channel, ...]

    @property
    def local_dim(self) -> int:
        return len(self.levels)

    def _index(self, level: str) -> int:
        return self.levels.index(level)

    def rydberg_projector(self) -> np.ndarray:
        """Diagonal ``(d, d)`` projector onto the Rydberg levels (``n_R``)."""
        d = self.local_dim
        m = np.zeros((d, d), dtype=complex)
        for level in self.rydberg_levels:
            m[self._index(level), self._index(level)] = 1.0
        return m

    def local_hamiltonians(self, t: float) -> np.ndarray:
        """Per-site local Hamiltonian matrices ``(N, d, d)`` at time ``t``.

        Hermitian by construction: diagonal atomic energies and diagonal drive
        channels add to the diagonal; each off-diagonal channel adds the matrix
        element and its conjugate transpose (P10 — no extra ``1/2``).
        """
        n, d = self.n_sites, self.local_dim
        h = np.zeros((n, d, d), dtype=complex)
        for level, energy in self.onsite_static:
            k = self._index(level)
            h[:, k, k] += float(energy)
        for ch in self.channels:
            values = _broadcast(ch.coefficient(t), n)
            if ch.is_diag:
                k = self._index(ch.ket)
                h[:, k, k] += values.real
            else:
                up, lo = self._index(ch.ket), self._index(ch.bra)
                h[:, up, lo] += values
                h[:, lo, up] += np.conj(values)
        return h


def compile_tn_terms(system, *, realization=None) -> TNTerms:
    """Lower a protocol-bound system into :class:`TNTerms` (shared by MPS/PEPS)."""
    name = system.level_structure.name
    if name not in _TN_PRESETS:
        raise ValueError(
            f"tensor-network backends support only the {sorted(_TN_PRESETS)} "
            f"presets; level structure {name!r} is not TN-capable (use "
            "backend='exact_ode')."
        )
    t_gate, lowered = lower_drives(system, realization=realization)

    onsite_static: list[tuple[str, float]] = []
    pairs: list[tuple[int, int, float]] = []
    rydberg_levels: tuple[str, ...] = ()
    for term in system._static_terms:
        op = term.operator
        if isinstance(op, SumProjectorSpec):
            onsite_static.append((op.level, float(complex(term.coefficient).real)))
        elif isinstance(op, RydbergPairInteractionSpec):
            rydberg_levels = op.rydberg_levels
            scale = float(complex(term.coefficient).real)
            for i, j, V in op.pairs:
                a, b = (int(i), int(j)) if int(i) < int(j) else (int(j), int(i))
                pairs.append((a, b, float(V) * scale))
        else:  # pragma: no cover - 1r/01r only produce the two specs above
            raise ValueError(
                f"tensor-network backends cannot lower static operator "
                f"{type(op).__name__}; only 1r/01r systems are TN-capable."
            )

    channels = []
    for lc in lowered:
        ket, bra = parse_E(lc.channel)
        channels.append(_Channel(ket, bra, ket == bra, lc.coefficient))

    levels = tuple(system.level_structure.levels)
    return TNTerms(
        t_gate=float(t_gate),
        n_sites=int(system.N),
        levels=levels,
        rydberg_levels=tuple(rydberg_levels),
        onsite_static=tuple(onsite_static),
        pairs=tuple(pairs),
        channels=tuple(channels),
    )


def _broadcast(value, n: int) -> np.ndarray:
    """Coerce a scalar or length-``n`` channel coefficient to a ``(n,)`` array."""
    arr = np.asarray(value, dtype=complex)
    if arr.ndim == 0:
        return np.full(n, complex(arr))
    if arr.shape != (n,):
        raise ValueError(
            f"channel coefficient must be a scalar or length-{n} array; got shape {arr.shape}."
        )
    return arr


# ── anchor-exact stepping plan (E06/E23/E25) ────────────────────────────────


@dataclass(frozen=True)
class TNSegment:
    """One anchor-to-anchor stretch: ``n_sub`` equal steps of ``dt_sub``.

    ``record`` marks segments ending on a *requested* measurement anchor; the
    trailing internal ``t_gate`` segment of an endpoint-free request evolves
    without recording.
    """

    t0: float
    t1: float
    n_sub: int
    dt_sub: float
    record: bool


def plan_segments(t_gate: float, record_times: np.ndarray, time_step_s: float) -> tuple[bool, list[TNSegment]]:
    """Plan piecewise-constant TN stepping that hits every anchor exactly.

    ``record_times`` are the validated measurement times (the requested
    ``t_eval``, or ``[t_gate]``).  Between consecutive anchors ``[a, b]`` the
    evolution takes ``ceil((b - a) / time_step_s)`` equal steps, so every anchor is
    a true step boundary (E06 — never round-and-lie).  Returns
    ``(record_at_start, segments)``; ``record_at_start`` is True when ``t=0`` was
    itself requested.
    """
    t_gate = float(t_gate)
    dt = float(time_step_s)
    times = np.asarray(record_times, dtype=float)
    record_at_start = bool(times.size and times[0] == 0.0)
    anchors: list[tuple[float, bool]] = [(float(t), True) for t in times if t > 0.0]
    if not anchors or not np.isclose(anchors[-1][0], t_gate, rtol=1e-12, atol=0.0):
        anchors.append((t_gate, False))
    segments: list[TNSegment] = []
    prev = 0.0
    for t_anchor, record in anchors:
        span = t_anchor - prev
        n_sub = max(1, int(np.ceil(span / dt * (1.0 - 1e-12))))
        segments.append(TNSegment(prev, t_anchor, n_sub, span / n_sub, record))
        prev = t_anchor
    return record_at_start, segments
