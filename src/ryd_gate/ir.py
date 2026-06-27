"""Unified Hamiltonian and evolution data representations.

Algorithm-agnostic Hamiltonian intermediate representation
(:class:`HamiltonianTerm` / :class:`HamiltonianIR` /
:func:`compile_hamiltonian_ir`) plus the :class:`EvolutionResult`
container returned by all simulation algorithm packages.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass
class EvolutionResult:
    """Unified result object returned by simulation algorithm packages.

    ``psi_final`` is the final state — a dense state vector for the exact
    backend, or a backend-native state handle for tensor-network backends.
    ``times`` / ``states`` hold the trajectory when ``t_eval`` was requested.

    Convenience accessors (:attr:`final_state`, :meth:`expectation`,
    :meth:`probabilities`, :meth:`sample`) are wired up by
    :func:`ryd_gate.simulate`, which attaches the measuring ``system`` and any
    values requested via ``simulate(..., observables=[...])`` so results can be
    read without re-threading the state back through the system.
    """

    psi_final: Any
    times: np.ndarray | None = None
    states: Any | None = None
    metadata: dict = field(default_factory=dict)
    expectations: dict | None = None
    system: Any = field(default=None, repr=False, compare=False)

    @property
    def final_state(self) -> Any:
        """Alias for :attr:`psi_final` (peer-library naming)."""
        return self.psi_final

    def expectation(self, name: str) -> Any:
        """Expectation value of a named observable in the final state.

        Returns the precomputed value when ``simulate(..., observables=[...])``
        requested it; otherwise measures it on the dense final state through the
        bound system. Raises ``RuntimeError`` if no measurement context is
        attached (e.g. a bare result built outside :func:`ryd_gate.simulate`).
        """
        if self.expectations is not None and name in self.expectations:
            return self.expectations[name]
        if self.system is None:
            raise RuntimeError(
                "no measurement context on this result; request it via "
                "simulate(..., observables=[name]), or measure on the system "
                "directly: system.expectation(name, result.final_state)."
            )
        return self.system.expectation(name, self._dense_final())

    def probabilities(self) -> np.ndarray:
        """Computational-basis probabilities ``|psi|**2`` of the final state."""
        psi = np.asarray(self._dense_final())
        return np.abs(psi) ** 2

    def sample(self, n_shots: int, seed: int | None = None) -> Counter:
        """Multinomially sample measurement outcomes from the final state.

        Returns a :class:`collections.Counter` keyed by per-site level-label
        strings in basis site order (e.g. ``"rr"``, ``"1r"`` for a 2-level
        chain; ``"|"``-joined when level labels are multi-character). Requires a
        dense final state and the bound system's basis.
        """
        basis = getattr(self.system, "basis", None)
        if basis is None:
            raise RuntimeError("sample() needs the bound system's basis; use ryd_gate.simulate(...).")
        probs = self.probabilities()
        total = probs.sum()
        if not np.isclose(total, 1.0):
            probs = probs / total
        rng = np.random.default_rng(seed)
        draws = rng.choice(probs.size, size=int(n_shots), p=probs)
        d, n_sites, levels = basis.local_dim, basis.n_sites, basis.local_levels
        joiner = "" if all(len(str(s)) == 1 for s in levels) else "|"
        counts: Counter = Counter()
        for idx in draws:
            site_levels = (levels[(int(idx) // d ** (n_sites - 1 - i)) % d] for i in range(n_sites))
            counts[joiner.join(site_levels)] += 1
        return counts

    def _dense_final(self) -> np.ndarray:
        psi = self.psi_final
        if not isinstance(psi, np.ndarray):
            raise TypeError(
                "a dense final state vector is not available (the backend returned "
                f"a {type(psi).__name__} state handle); request named observables via "
                "simulate(..., observables=[...]) instead."
            )
        return psi


@dataclass
class HamiltonianTerm:
    """A single symbolic or materialized Hamiltonian term."""

    name: str
    operator: Any
    coefficient: Callable[[float], complex] | complex = 1.0
    add_hermitian_conjugate: bool = False
    channel: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class HamiltonianIR:
    """Unified Hamiltonian representation emitted by ``ryd_gate``.

    ``operator`` payloads remain algorithm-agnostic: they can be symbolic core
    operator specs for large systems, or concrete matrix blocks for small
    preset systems. Algorithm packages lower this IR into their own numerical
    input format.
    """

    static_terms: list[HamiltonianTerm]
    drive_terms: list[HamiltonianTerm]
    dim: int
    is_sparse: bool = False
    metadata: dict = field(default_factory=dict)
    basis: Any | None = None
    geometry: Any | None = None
    level_spec: Any | None = None
    protocol: Any | None = None
    params: dict | None = None


_STATIC_BLOCKS = ("H_const", "H_vdw", "H_1013", "H_1013_conj")


def compile_hamiltonian_ir(system, params: dict) -> HamiltonianIR:
    """Compile a protocol-bound system into the unified Hamiltonian IR."""
    from ryd_gate.core.level_structures import (
        LevelStructureSpec,
        block_name_for_drive_channel,
        channel_needs_hermitian_conjugate,
    )
    from ryd_gate.core.operators import LocalProjectorSpec

    protocol = system._require_protocol()
    level_spec = system.meta("level_spec", None)
    if not isinstance(level_spec, LevelStructureSpec):
        level_spec = None

    drive_channel_set = protocol.drive_channels(system)
    static_terms: list[HamiltonianTerm] = []
    for name in _STATIC_BLOCKS:
        if name in drive_channel_set:
            continue
        if system.blocks.has(name):
            static_terms.append(HamiltonianTerm(name, system.blocks.get(name), 1.0))

    static_terms.extend(_static_overlay_terms(system, params, LocalProjectorSpec))

    amplitude_scale = system.amplitude_scale
    drive_terms: list[HamiltonianTerm] = []
    unmapped_channels: list[str] = []
    for channel in sorted(drive_channel_set):
        block_name = block_name_for_drive_channel(system, channel)
        if block_name is None:
            unmapped_channels.append(channel)
            continue

        def coeff_fn(t, channel=channel):
            coeffs = protocol.get_drive_coefficients(t, params)
            coeff = coeffs.get(channel, 0.0)
            return amplitude_scale * coeff

        drive_terms.append(
            HamiltonianTerm(
                channel,
                system.blocks.get(block_name),
                coeff_fn,
                add_hermitian_conjugate=(
                    channel_needs_hermitian_conjugate(channel, level_spec)
                    and _explicit_dagger_channel(channel) not in drive_channel_set
                ),
                channel=channel,
                metadata={"block": block_name},
            )
        )
    if unmapped_channels:
        level_name = getattr(level_spec, "name", system.meta("level_structure", None))
        raise ValueError(
            "Protocol/system channel mismatch while compiling HamiltonianIR. "
            f"Protocol {type(protocol).__name__} drives channels that are not "
            f"available on level structure {level_name!r}: {unmapped_channels}. "
            "Choose a compatible level structure/protocol pair or add matching "
            "transitions/detuning levels to the system."
        )

    metadata = {
        "compiler": "ryd_gate",
        "t_gate": params["t_gate"],
        "param_set": system.param_set,
        "n_sites": system.basis.n_sites,
        "local_dim": system.basis.local_dim,
        "level_structure": getattr(level_spec, "name", system.meta("level_structure", None)),
        "interaction_pairs": tuple(system.meta("interaction_pairs", ())),
        "Omega": system.meta("Omega", None),
    }
    return HamiltonianIR(
        static_terms=static_terms,
        drive_terms=drive_terms,
        dim=system.basis.total_dim,
        is_sparse=system.is_sparse,
        metadata=metadata,
        basis=system.basis,
        geometry=system.geometry,
        level_spec=level_spec,
        protocol=protocol,
        params=params,
    )


def _explicit_dagger_channel(channel: str) -> str:
    if channel.endswith("_dag"):
        return channel[:-4]
    return f"{channel}_dag"


def _static_overlay_terms(system, params: dict, local_projector_cls) -> list[HamiltonianTerm]:
    terms: list[HamiltonianTerm] = []
    n_sites = system.basis.n_sites
    ground_label = system.basis.local_levels[0]
    ryd_label = system.basis.local_levels[-1]

    for idx, delta in params.get("pin_deltas", {}).items():
        if abs(delta) <= 1e-15 or idx >= n_sites:
            continue
        block_name = f"n_{ryd_label}_{idx}"
        operator = (
            system.blocks.get(block_name) if system.blocks.has(block_name) else local_projector_cls(ryd_label, idx)
        )
        terms.append(
            HamiltonianTerm(
                "pinning",
                operator,
                -delta,
                metadata={"site": idx, "level": ryd_label},
            )
        )

    for idx, rate in params.get("scatter_rates", {}).items():
        if rate <= 0 or idx >= n_sites:
            continue
        block_name = f"n_{ground_label}_{idx}"
        operator = (
            system.blocks.get(block_name) if system.blocks.has(block_name) else local_projector_cls(ground_label, idx)
        )
        terms.append(
            HamiltonianTerm(
                "scatter_loss",
                operator,
                -1j * rate / 2,
                metadata={"site": idx, "level": ground_label},
            )
        )

    for block_name, coeff in params.get("static_overlays", []):
        if system.blocks.has(block_name):
            terms.append(HamiltonianTerm(block_name, system.blocks.get(block_name), coeff))

    return terms
