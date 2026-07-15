"""Unified simulation entry point (E01).

``ryd_gate.simulate(system, initial_state=None, ...)`` dispatches over the exact
state-vector engine (``backend="exact_ode"``) and the tensor-network engines
(``backend`` in ``{"mps", "peps"}``). One initial state returns one
``EvolutionResult``; a nested batch returns a ``tuple`` of them (E10).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ryd_gate.core.observables import ObservableExpr
    from ryd_gate.results import EvolutionResult

_EXACT = "exact_ode"
_TN = frozenset({"mps", "peps", "graph_peps"})


def simulate(
    system,
    initial_state=None,
    *,
    backend: str = "exact_ode",
    t_eval=None,
    observables: "dict[str, ObservableExpr] | None" = None,
    backend_options: dict | None = None,
) -> "EvolutionResult | tuple[EvolutionResult, ...]":
    """Evolve a protocol-bound Rydberg system.

    ``initial_state``: ``None`` starts every site in ``|1>`` (E27); ``"plus"``
    starts every site in ``(|0>+|1>)/sqrt(2)``; a flat label list is one product
    state; a nested list of label lists is a batch (returns a tuple). ``backend``
    is ``"exact_ode"`` (default), ``"mps"``, ``"peps"`` (Cartesian grids) or
    ``"graph_peps"`` (arbitrary 2D geometry). ``t_eval`` are
    measurement times only (``None`` -> record at ``t_gate``). ``observables`` is
    a ``dict[str, ObservableExpr]`` from ``system.observables``.
    """
    from ryd_gate.core.states import normalize_initial_state

    key = backend.lower()
    kind, state = normalize_initial_state(initial_state, system.N)

    if key == _EXACT:
        from ryd_gate.backends.exact import simulate as simulate_exact
        from ryd_gate.backends.exact import simulate_states as simulate_exact_states

        if kind == "batch":
            return simulate_exact_states(
                system, state, t_eval=t_eval, observables=observables,
                backend_options=backend_options,
            )
        return simulate_exact(
            system, state, t_eval=t_eval, observables=observables,
            backend_options=backend_options,
        )

    if key in _TN:
        from ryd_gate.backends.tn_common.simulate import simulate_tn

        if kind == "batch":
            return tuple(
                simulate_tn(
                    system, s, backend=key, t_eval=t_eval,
                    observables=observables, backend_options=backend_options,
                )
                for s in state
            )
        return simulate_tn(
            system, state, backend=key, t_eval=t_eval,
            observables=observables, backend_options=backend_options,
        )

    raise ValueError(
        f"unknown backend {backend!r}; use 'exact_ode', 'mps', 'peps' or 'graph_peps'."
    )
