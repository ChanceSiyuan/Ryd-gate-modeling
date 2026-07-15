"""Unified Hamiltonian and evolution data representations.

Algorithm-agnostic Hamiltonian intermediate representation
(:class:`HamiltonianTerm` / :class:`HamiltonianIR` /
:func:`compile_hamiltonian_ir`), the canonical :class:`EvolutionResult`
container returned by all simulation engines, and the shared strict
``t_eval`` / observables preflight (:func:`validate_evolution_request`)
enforced by every backend.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass
class EvolutionResult:
    """Canonical result of one evolution.

    Fields
    ------
    final_state
        The state at the true evolution endpoint ``t_gate`` — a dense vector
        for the exact backend, a backend-native handle (MPS/PEPS) for
        tensor-network backends.  ``t_eval`` never changes it.
    times
        Exactly the requested measurement times: the ``t_eval`` array as
        passed in, or ``[t_gate]`` (shape ``(1,)``) when ``t_eval=None``.
    expectations
        ``{label: complex ndarray}`` — one raw ``<psi|O|psi>`` value per entry
        of ``times`` (time-major, ``shape == times.shape``).  Raw means no
        norm division: with decay the identity expectation is the survival
        norm, preserving the ``Gamma * integral(n_R) dt`` bookkeeping.  Empty
        dict when no observables were requested.
    metadata
        Engine diagnostics (solver, tolerances, formats, ...).
    basis
        Immutable basis info (site count, level labels) used by
        :meth:`sample` to decode outcome labels.
    """

    final_state: Any
    times: np.ndarray
    expectations: dict[str, np.ndarray]
    metadata: dict
    basis: Any

    def expectation(self, name: str) -> np.ndarray:
        """The stored expectation array for ``name`` (shape ``(n_times,)``).

        Only values requested via ``simulate(..., observables={name: expr})``
        exist; anything else raises ``KeyError``.  There is no on-demand
        measurement fallback.
        """
        try:
            return self.expectations[name]
        except KeyError:
            raise KeyError(
                f"no expectation {name!r} on this result; it must be requested "
                "at simulate time via observables={"
                f"{name!r}: <ObservableExpr>}}. Stored: {sorted(self.expectations)}."
            ) from None

    def sample(self, shots: int, seed: int) -> Counter:
        """Sample computational-basis outcomes from the dense final state.

        Semantics: **conditional on survival**.  Outcome weights are
        ``|psi_i|**2 / <psi|psi>`` — with decay (non-unitary evolution) the
        lost norm is renormalized away, i.e. outcomes are conditioned on the
        atoms having survived; there is no loss label.  The surviving norm
        itself is available as an ``identity()`` observable expectation.

        Fully lazy: weights are computed from ``final_state`` on every call
        (no cached probability vector).  ``seed`` is a required integer — the
        same seed always reproduces the same draws.  Outcomes are keyed by
        per-site level-label strings in basis site order (``"1r"``;
        ``"|"``-joined when labels are multi-character).

        Only dense (exact-backend) final states can be sampled; tensor-network
        state handles raise ``TypeError`` (no dense conversion is attempted).
        """
        psi = self.final_state
        if not isinstance(psi, np.ndarray):
            raise TypeError(
                "sample() requires a dense final state; this result holds a "
                f"{type(psi).__name__} tensor-network handle, which is never "
                "densified. Request observables at simulate time instead."
            )
        if not isinstance(shots, (int, np.integer)) or isinstance(shots, bool) or shots < 1:
            raise ValueError(f"shots must be a positive integer; got {shots!r}.")
        if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool):
            raise TypeError(f"seed must be an integer; got {seed!r}.")
        weights = np.abs(np.asarray(psi)) ** 2
        total = weights.sum()
        if not total > 0.0:
            raise ValueError("cannot sample from a zero-norm final state.")
        probs = weights / total
        rng = np.random.default_rng(int(seed))
        draws = rng.choice(probs.size, size=int(shots), p=probs)
        basis = self.basis
        d, n_sites, levels = basis.local_dim, basis.n_sites, basis.local_levels
        joiner = "" if all(len(str(s)) == 1 for s in levels) else "|"
        counts: Counter = Counter()
        for idx in draws:
            site_levels = (levels[(int(idx) // d ** (n_sites - 1 - i)) % d] for i in range(n_sites))
            counts[joiner.join(site_levels)] += 1
        return counts


def validate_evolution_request(
    t_gate: float,
    t_eval,
    observables,
    *,
    t_start: float = 0.0,
) -> tuple[np.ndarray | None, dict]:
    """Shared strict preflight for measurement requests (all backends).

    Rules (design: ``t_eval`` decides measurement times only, never the
    evolution endpoint):

    - ``t_eval=None`` -> measure only at ``t_gate`` (result ``times == [t_gate]``).
    - explicit ``t_eval``: 1-D, non-empty, finite, strictly increasing, within
      ``[t_start, t_gate]``; returned to the caller exactly (no auto-appended
      endpoint in public times).  Boolean ``t_eval`` is a ``TypeError``.
    - explicit ``t_eval`` with no observables is a ``ValueError`` (nothing to
      record).
    - ``observables`` must be ``None`` or a ``dict[str, ObservableExpr]``
      (string-name lists were removed).

    Returns ``(validated t_eval copy or None, observables dict or {})``.
    ``t_start`` is the private continuation-seam lower bound (0.0 publicly).
    """
    from ryd_gate.core.observables import ObservableExpr

    t_gate = float(t_gate)
    if not t_gate > 0.0:
        raise ValueError(f"t_gate must be positive; got {t_gate!r}.")

    if observables is None:
        obs: dict = {}
    elif isinstance(observables, dict):
        for label, expr in observables.items():
            if not isinstance(label, str):
                raise TypeError(
                    f"observable labels must be strings; got {type(label).__name__}."
                )
            if not isinstance(expr, ObservableExpr):
                raise TypeError(
                    f"observables[{label!r}] must be an ObservableExpr built from "
                    f"system.observables; got {type(expr).__name__} (string-name "
                    "observables were removed)."
                )
        obs = dict(observables)
    else:
        raise TypeError(
            "observables must be a dict[str, ObservableExpr] or None; got "
            f"{type(observables).__name__} (string-name lists were removed)."
        )

    if t_eval is None:
        return None, obs
    if isinstance(t_eval, (bool, np.bool_)):
        raise TypeError(
            "t_eval must be an array of times or None (boolean t_eval has been removed)."
        )
    times = np.array(t_eval, dtype=float, copy=True)
    if times.ndim != 1:
        raise ValueError("t_eval must be a one-dimensional array of times.")
    if times.size == 0:
        raise ValueError(
            "explicit t_eval must be non-empty; use t_eval=None for an "
            "endpoint-only run."
        )
    if not np.all(np.isfinite(times)):
        raise ValueError("t_eval must contain only finite times.")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("t_eval must be strictly increasing (no duplicates).")
    if times[0] < t_start:
        raise ValueError(
            f"t_eval must lie within [{t_start}, t_gate]; got t_eval[0]={times[0]}."
        )
    if times[-1] > t_gate and not np.isclose(times[-1], t_gate, rtol=1e-12, atol=0.0):
        raise ValueError(
            f"t_eval must lie within [{t_start}, t_gate={t_gate}]; got "
            f"t_eval[-1]={times[-1]}."
        )
    if not obs:
        raise ValueError(
            "explicit t_eval requires observables (nothing to record otherwise); "
            "pass observables={label: expr} or drop t_eval."
        )
    return times, obs


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


def compile_hamiltonian_ir(system) -> HamiltonianIR:
    """Compile a protocol-bound system into the unified Hamiltonian IR.

    The bound protocol is fully specified; its normalized quantities are
    resolved against the system here (``protocol._resolve(system)``), once per
    compile.  Static terms come from ``system.static_hamiltonian_terms`` (a
    static term is dropped when a protocol drives a channel of the same name).
    Drive terms come from ``system.hamiltonian_channels`` keyed by primitive
    ``E[ket,bra]`` names (per-site ``E[...]_<i>`` resolved via the operator
    factory); the compiler auto-adds ``conj(coeff)·E[bra,ket]`` for every
    non-diagonal channel.
    """
    from ryd_gate.core.level_structures import split_site_channel
    from ryd_gate.core.operators import BasisOperatorFactory

    protocol = system._require_protocol()
    params = protocol._resolve(system)
    level_spec = system.level_structure

    drive_channel_set = protocol.drive_channels(system)

    static_terms: list[HamiltonianTerm] = []
    for term in system.static_hamiltonian_terms:
        if term.name in drive_channel_set:
            continue  # a driven channel supersedes its static term
        static_terms.append(
            HamiltonianTerm(
                term.name,
                term.operator,
                term.coefficient,
                add_hermitian_conjugate=term.add_hermitian_conjugate,
            )
        )
    static_terms.extend(_static_overlay_terms(system, params))

    amplitude_scale = system.amplitude_scale
    drive_terms: list[HamiltonianTerm] = []
    unmapped_channels: list[str] = []
    for channel in sorted(drive_channel_set):
        base, site = split_site_channel(channel)
        if base not in system.hamiltonian_channels:
            unmapped_channels.append(channel)
            continue
        operator = (
            system.operators.local(base, site)
            if site is not None
            else system.hamiltonian_channels[base]
        )

        def coeff_fn(t, channel=channel):
            coeffs = protocol.get_drive_coefficients(t, params)
            return amplitude_scale * coeffs.get(channel, 0.0)

        drive_terms.append(
            HamiltonianTerm(
                channel,
                operator,
                coeff_fn,
                add_hermitian_conjugate=not BasisOperatorFactory.is_hermitian(base),
                channel=channel,
                metadata={"base": base},
            )
        )
    if unmapped_channels:
        level_name = level_spec.name
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
        "model_tag": level_spec.name,
        "n_sites": system.basis.n_sites,
        "local_dim": system.basis.local_dim,
        "level_structure": level_spec.name,
        "interaction_pairs": tuple(system.interaction_pairs),
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


def _static_overlay_terms(system, params: dict) -> list[HamiltonianTerm]:
    """Protocol-supplied static terms (Rydberg pinning, ground-state scatter
    loss, arbitrary overlays), built from the system's primitive operators."""
    from ryd_gate.core.operators import is_operator_spec

    terms: list[HamiltonianTerm] = []
    n_sites = system.basis.n_sites
    ground_label = system.basis.local_levels[0]
    ryd_label = system.basis.local_levels[-1]

    for idx, delta in params.get("pin_deltas", {}).items():
        if abs(delta) <= 1e-15 or idx >= n_sites:
            continue
        terms.append(
            HamiltonianTerm(
                "pinning",
                system.operators.local(f"E[{ryd_label},{ryd_label}]", idx),
                -delta,
                metadata={"site": idx, "level": ryd_label},
            )
        )

    for idx, rate in params.get("scatter_rates", {}).items():
        if rate <= 0 or idx >= n_sites:
            continue
        terms.append(
            HamiltonianTerm(
                "scatter_loss",
                system.operators.local(f"E[{ground_label},{ground_label}]", idx),
                -1j * rate / 2,
                metadata={"site": idx, "level": ground_label},
            )
        )

    # Arbitrary overlays: each entry is (operator_spec | "E[ket,bra]" name, coeff).
    for op, coeff in params.get("static_overlays", []):
        operator = op if is_operator_spec(op) else system.operators.sum(op)
        terms.append(HamiltonianTerm("static_overlay", operator, coeff))

    return terms
