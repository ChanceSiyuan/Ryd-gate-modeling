"""RydbergSystem: level structure + register + a mandatory bound protocol.

Constructed explicitly and immutably (S01/S16). The public surface is frozen to
the S15 members; the basis, operators, static terms and resolved interaction are
private compiler data consumed through the single canonical lowering seam
(``ryd_gate.core.lowering.lower_drives``). All Hamiltonians are Hermitian (E08).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ryd_gate.core.level_structures import LevelStructure
from ryd_gate.core.model import BasisSpec
from ryd_gate.core.observables import ObservableFactory
from ryd_gate.core.operators import (
    BasisOperatorFactory,
    ChannelResolvedPairSpec,
    RydbergPairInteractionSpec,
    StaticHamiltonianTerm,
)
from ryd_gate.lattice import Register

if TYPE_CHECKING:
    from ryd_gate.protocols.base import Protocol

_CUTOFF_REL_TOL = 1e-9


class RydbergSystem:
    """Immutable Rydberg system: level structure + 2D register + bound protocol."""

    __slots__ = (
        "_level_structure", "_register", "_protocol", "_interaction_cutoff_um",
        "_basis", "_operators", "_observables", "_static_terms", "_realization",
    )

    def __init__(
        self,
        *,
        level_structure: LevelStructure,
        register: Register,
        protocol: "Protocol",
        interaction_cutoff_um: float | None = None,
    ) -> None:
        if not isinstance(level_structure, LevelStructure):
            raise TypeError(
                "level_structure must come from ryd_gate.level_structure(name, ...); "
                f"got {type(level_structure).__name__}."
            )
        if not isinstance(register, Register):
            raise TypeError(f"register must be a Register; got {type(register).__name__}.")
        if protocol is None:
            raise TypeError("protocol is mandatory (RydbergSystem must be protocol-bound).")
        if interaction_cutoff_um is not None:
            c = float(interaction_cutoff_um)
            if not np.isfinite(c) or c < 0.0:
                raise ValueError(f"interaction_cutoff_um must be finite and >= 0, or None; got {interaction_cutoff_um!r}.")
            interaction_cutoff_um = c
        _check_protocol_compatible(protocol, level_structure)

        ls = level_structure
        n = register.N
        d = len(ls.levels)
        self._level_structure = ls
        self._register = register
        self._protocol = protocol
        self._interaction_cutoff_um = interaction_cutoff_um

        self._basis = BasisSpec(
            site_labels=tuple(str(i) for i in range(n)),
            local_levels=ls.levels,
            local_dim=d,
            total_dim=d**n,
        )
        self._operators = BasisOperatorFactory(self._basis)
        self._observables = ObservableFactory(self._basis)
        self._realization = None
        self._static_terms = _build_static_terms(ls, register, self._operators, interaction_cutoff_um)

    # -- public S15 surface --------------------------------------------------

    @property
    def level_structure(self) -> LevelStructure:
        return self._level_structure

    @property
    def register(self) -> Register:
        return self._register

    @property
    def protocol(self) -> "Protocol":
        return self._protocol

    @property
    def interaction_cutoff_um(self) -> float | None:
        return self._interaction_cutoff_um

    @property
    def N(self) -> int:
        return self._basis.n_sites

    @property
    def t_gate(self) -> float:
        """Real gate duration (s) resolved from the bound protocol (read-only)."""
        return float(self._protocol._resolve(self).t_gate)

    @property
    def observables(self) -> ObservableFactory:
        """Read-only ``E()`` / ``n()`` observable factory bound to this basis."""
        return self._observables

    def with_protocol(self, protocol: "Protocol") -> "RydbergSystem":
        """Return a new system binding another compatible protocol (original unchanged)."""
        if protocol is None:
            raise TypeError("protocol is mandatory.")
        _check_protocol_compatible(protocol, self._level_structure)
        return RydbergSystem(
            level_structure=self._level_structure,
            register=self._register,
            protocol=protocol,
            interaction_cutoff_um=self._interaction_cutoff_um,
        )

    def ground_state(self, *, at, method, initial_state, observables=None, method_options):
        """Many-body ground state of a fully constructed ``1r`` system (E12-E21)."""
        from ryd_gate.backends.ground_state import solve_ground_state

        return solve_ground_state(
            self, at=at, method=method, initial_state=initial_state,
            observables=observables, method_options=method_options,
        )

    # -- private noise seam (used by simulate_ensemble via lower_drives) ------

    def _with_realization(self, realization) -> "RydbergSystem":
        """Return a new system carrying a shot's noise realization (private, N08/N15).

        Laser amplitude/frequency noise is applied at lowering time; position
        noise rebuilds only the interaction geometry (nominal pair topology
        stays fixed). The public register remains the nominal 2D geometry (S09).
        """
        new = RydbergSystem(
            level_structure=self._level_structure,
            register=self._register,
            protocol=self._protocol,
            interaction_cutoff_um=self._interaction_cutoff_um,
        )
        new._realization = realization
        offsets = realization.get("position_offsets_um") if realization else None
        if offsets is not None:
            new._static_terms = _build_static_terms(
                self._level_structure, self._register, new._operators,
                self._interaction_cutoff_um, position_offsets=offsets,
            )
        return new


def _check_protocol_compatible(protocol, ls: LevelStructure) -> None:
    compatible = getattr(protocol, "_compatible_presets", frozenset())
    if ls.name not in compatible:
        raise ValueError(
            f"protocol {type(protocol).__name__} is not compatible with level "
            f"structure {ls.name!r}; compatible presets: {sorted(compatible)}."
        )


def _build_static_terms(ls, register, operators, cutoff_um, position_offsets=None) -> list[StaticHamiltonianTerm]:
    terms: list[StaticHamiltonianTerm] = []
    for level, energy in ls._static_diag.items():
        if energy != 0.0:
            name = f"E[{level},{level}]"
            terms.append(StaticHamiltonianTerm(name, operators.sum(name), complex(energy)))
    interaction = _resolve_interaction(ls, register, cutoff_um, position_offsets)
    if interaction is not None:
        terms.append(StaticHamiltonianTerm("H_pair", interaction, 1.0))
    return terms


def _resolve_interaction(ls, register, cutoff_um, position_offsets=None):
    """Resolve the private Rydberg pair-interaction operator spec (I04-I08).

    Nominal pair topology (which pairs) is fixed by the 2D register + cutoff;
    ``position_offsets`` (shape ``(N, 3)``, um) jitters only the pair distances
    and directions for a noise realization (N15).
    """
    ryd_levels = ls._rydberg_levels
    if not ryd_levels:
        return None

    if ls._pair_c6_channels is not None:  # channel-resolved P-state (I06)
        axis = ls.quantization_axis
        channel_terms: list[tuple[int, int, str, str, float]] = []
        for i, j, r, theta, phi in _pair_geometry(register.coords, axis, cutoff_um, position_offsets):
            r6 = r**6
            for (level_a, level_b), c6_fn in ls._pair_c6_channels:
                V = float(c6_fn(round(theta, 9), round(phi, 9))) / r6
                channel_terms.append((i, j, level_a, level_b, V))
                if level_a != level_b:
                    channel_terms.append((i, j, level_b, level_a, V))
        return ChannelResolvedPairSpec(tuple(channel_terms))

    C6 = ls._pair_c6_isotropic  # isotropic S-state (I05)
    pairs = [
        (i, j, C6 / r**6)
        for i, j, r, _theta, _phi in _pair_geometry(register.coords, None, cutoff_um, position_offsets)
    ]
    return RydbergPairInteractionSpec(tuple(pairs), ryd_levels)


def _pair_geometry(coords, axis, cutoff_um, position_offsets=None):
    """Yield ``(i, j, r, theta, phi)`` for each in-range pair.

    Topology (the cutoff test) always uses the nominal 2D coordinates; the pair
    distance ``r`` and orientation ``(theta, phi)`` use the jittered 3D positions
    when ``position_offsets`` is given (nominal 2D embedded at ``z = 0`` plus the
    offset). When ``axis`` is ``None`` the angles are returned as ``0.0``.
    """
    nominal = np.asarray(coords, dtype=float)
    n = nominal.shape[0]
    pos3d = np.column_stack([nominal, np.zeros(n)])
    if position_offsets is not None:
        pos3d = pos3d + np.asarray(position_offsets, dtype=float)

    frame = None
    if axis is not None:
        a = np.asarray(axis, dtype=float)
        a = a / np.linalg.norm(a)
        seed = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        e1 = seed - a * (seed @ a)
        e1 = e1 / np.linalg.norm(e1)
        e2 = np.cross(a, e1)
        frame = (a, e1, e2)

    for i in range(n):
        for j in range(i + 1, n):
            r_nominal = float(np.hypot(*(nominal[j] - nominal[i])))
            if cutoff_um is not None and r_nominal > cutoff_um * (1 + _CUTOFF_REL_TOL):
                continue
            d = pos3d[j] - pos3d[i]
            r = float(np.linalg.norm(d))
            if frame is None:
                yield i, j, r, 0.0, 0.0
            else:
                a, e1, e2 = frame
                theta = float(np.arccos(np.clip((d @ a) / r, -1.0, 1.0)))
                phi = float(np.arctan2(d @ e2, d @ e1))
                yield i, j, r, theta, phi
