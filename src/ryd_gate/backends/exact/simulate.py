"""Exact state-vector simulation entry point (ODE-only)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ryd_gate.ir import EvolutionResult

if TYPE_CHECKING:
    from ryd_gate.backends.exact.compiler import SolverBackend

# The exact backend options; anything else is an explicit error.
_ALLOWED_OPTIONS = ("hamiltonian_format", "rtol", "atol")


def validated_exact_options(backend_options) -> dict:
    """Normalize/validate exact ``backend_options`` (unknown keys error)."""
    opts = dict(backend_options or {})
    unknown = set(opts) - set(_ALLOWED_OPTIONS)
    if unknown:
        raise ValueError(
            f"unknown exact backend_options: {sorted(unknown)}; "
            f"allowed: {list(_ALLOWED_OPTIONS)}."
        )
    return opts


def _make_backend(opts: dict) -> "SolverBackend":
    from ryd_gate.backends.exact.ode import ExactODEBackend

    return ExactODEBackend(**opts)


def simulate(
    system,
    psi0="ground",
    t_eval: np.ndarray | None = None,
    observables: dict | None = None,
    backend: "SolverBackend | None" = None,
    compiler=None,
    backend_options: dict | None = None,
) -> EvolutionResult:
    """Compile a protocol-bound Rydberg system and evolve exactly (internal seam).

    ``system`` must have a fully specified protocol bound; the protocol's
    normalized quantities are resolved against the system at compile time.
    The solver is the adaptive DOP853 ODE integrator
    (:class:`~ryd_gate.backends.exact.ode.ExactODEBackend`); the Hamiltonian
    storage/matvec strategy and tolerances come from ``backend_options``
    (``{"hamiltonian_format": "auto"|"dense"|"sparse", "rtol": ..., "atol": ...}``).
    ``observables`` is a ``dict[str, ObservableExpr]`` evaluated at the
    validated ``t_eval`` times (or at ``t_gate`` when ``t_eval=None``).
    ``psi0`` may be a dense vector, ``"plus"``, or a per-site level-label list —
    dense-vector ``psi0`` is the private continuation seam research scripts use
    to hand in superpositions or chain evolutions from a previous
    ``final_state``; the public :func:`ryd_gate.simulate` accepts only
    label-based inputs.
    """
    from ryd_gate.backends.exact.compiler import ExactCompiler
    from ryd_gate.core.system import RydbergSystem

    if not isinstance(system, RydbergSystem):
        raise TypeError(
            "simulate() requires RydbergSystem instances with a bound protocol "
            "(construct with protocol=... or call .with_protocol(...))."
        )

    opts = validated_exact_options(backend_options)
    compiler = compiler or ExactCompiler()
    ir = compiler.compile(system)
    solver = backend if backend is not None else _make_backend(opts)
    return solver.evolve(
        ir, _exact_initial_state(system, psi0), ir.metadata["t_gate"], t_eval, observables
    )


def simulate_states(
    system,
    states,
    t_eval=None,
    observables: dict | None = None,
    backend_options=None,
    backend: "SolverBackend | None" = None,
):
    """Evolve several initial states under the same bound protocol, sharing work.

    The Hamiltonian compilation and RHS construction are shared across the
    batch; each initial state runs its own independent adaptive solve (per-state
    error control).  Returns a list of
    :class:`~ryd_gate.ir.EvolutionResult`, one per entry of *states*.
    """
    from ryd_gate.backends.exact.compiler import ExactCompiler
    from ryd_gate.core.system import RydbergSystem

    if not isinstance(system, RydbergSystem):
        raise TypeError("simulate_states() requires a RydbergSystem with a bound protocol.")

    opts = validated_exact_options(backend_options)
    ir = ExactCompiler().compile(system)
    solver = backend if backend is not None else _make_backend(opts)
    psis = [_exact_initial_state(system, s) for s in states]
    t_gate = ir.metadata["t_gate"]
    if hasattr(solver, "evolve_many"):
        return solver.evolve_many(ir, psis, t_gate, t_eval, observables)
    return [solver.evolve(ir, p, t_gate, t_eval, observables) for p in psis]


def _simulate_from(
    system,
    psi0: np.ndarray,
    t_start: float,
    *,
    t_eval=None,
    observables: dict | None = None,
    backend_options=None,
) -> EvolutionResult:
    """Private exact continuation seam: resume from a dense state at ``t_start``.

    Integrates the bound protocol's Hamiltonian over ``[t_start, t_gate]``
    starting from ``psi0`` (a dense vector, e.g. a previous run's
    ``final_state``).  ``t_eval`` must lie within ``[t_start, t_gate]``.
    Not public API — used by scripts that chain evolutions.
    """
    from ryd_gate.backends.exact.compiler import ExactCompiler
    from ryd_gate.backends.exact.ode import ExactODEBackend
    from ryd_gate.core.system import RydbergSystem

    if not isinstance(system, RydbergSystem):
        raise TypeError("_simulate_from() requires a RydbergSystem with a bound protocol.")
    if not isinstance(psi0, np.ndarray):
        raise TypeError("_simulate_from() resumes from a dense state vector.")

    opts = validated_exact_options(backend_options)
    ir = ExactCompiler().compile(system)
    solver = ExactODEBackend(**opts)
    return solver.evolve_many(
        ir, [psi0], ir.metadata["t_gate"], t_eval, observables, t_start=float(t_start)
    )[0]


def _exact_initial_state(system, psi0):
    """Lower an initial-state request to a dense vector (internal).

    Accepts a dense vector (the research-script superposition seam), the
    literal ``"ground"`` (per-site preset initial level), ``"plus"``, or a
    per-site level-label list.
    """
    if isinstance(psi0, np.ndarray):
        return psi0
    if isinstance(psi0, str):
        if psi0 == "ground":
            from ryd_gate.core.states import preset_initial_label

            label = preset_initial_label(system.basis.local_levels)
            return system.product_state([label] * system.N)
        if psi0 == "plus":
            from ryd_gate.core.states import plus_local_amplitudes, product_superposition_state

            return product_superposition_state(
                plus_local_amplitudes(system.basis.local_levels), system.N
            )
        raise ValueError(
            f"unknown initial-state string {psi0!r}; use 'ground', 'plus', or a "
            "per-site level-label list."
        )
    if isinstance(psi0, (list, tuple)):
        return system.product_state(list(psi0))
    return psi0
