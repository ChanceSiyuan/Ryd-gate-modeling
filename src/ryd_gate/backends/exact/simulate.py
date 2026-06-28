"""Exact state-vector simulation entry point and typed options."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ryd_gate.backends._options import as_backend_options
from ryd_gate.ir import EvolutionResult

if TYPE_CHECKING:
    from ryd_gate.backends.exact.compiler import SolverBackend


@dataclass(frozen=True)
class ExactOptions:
    """Options for :func:`ryd_gate.backends.exact.simulate`.

    ``None`` means "use the backend default".
    """

    n_steps: int | None = None


# For small Hilbert spaces a dense ``expm`` per step beats the sparse ``expm_multiply``
# backend (whose per-call setup overhead dominates ~``n_steps`` calls, and whose cost
# also grows with ``||dt*H||``), so the auto selector routes them to
# :class:`DenseExpmBackend`. Two caps: any small system below ``_DENSE_EXPM_SMALL_DIM``
# (dense is just faster), and the physical-ladder models (``rb87_7``/``analog_3``, whose
# ~GHz intermediate manifold makes ``expm_multiply`` *pathologically* slow) up to a
# larger cap. Dense ``expm`` is O(dim^3)/step, so larger systems stay on the sparse path.
_DENSE_LADDER_MODELS = frozenset({"rb87_7", "analog_3"})
_DENSE_EXPM_SMALL_DIM = 256
_DENSE_EXPM_MAX_DIM = 1024


def _prefer_dense_expm(system) -> bool:
    dim = system.dim
    if dim <= _DENSE_EXPM_SMALL_DIM:
        return True
    return (
        system.meta("level_structure") in _DENSE_LADDER_MODELS
        and dim <= _DENSE_EXPM_MAX_DIM
    )


def _select_backend(system, ir, n_steps):
    """Auto-select the exact solver backend for a compiled IR (see ``simulate``)."""
    from ryd_gate.backends.exact.dense_ode import DenseODEBackend

    if not ir.is_sparse:
        return DenseODEBackend()
    if _prefer_dense_expm(system):
        return make_forced_expm_backend("dense", n_steps=n_steps)
    return make_forced_expm_backend("sparse", n_steps=n_steps)


def _resolve_backend(system, ir, opts, backend, force_kind):
    """Pick the solver: explicit ``backend`` > forced ``force_kind`` > auto-select."""
    if backend is not None:
        return backend
    n_steps = resolve_n_steps(system, opts)
    if force_kind is not None:
        return make_forced_expm_backend(force_kind, n_steps=n_steps)
    return _select_backend(system, ir, n_steps)


def make_forced_expm_backend(kind: str, *, n_steps: int) -> "SolverBackend":
    """Build a piecewise-exponential backend, bypassing auto-selection.

    Parameters
    ----------
    kind
        ``"dense"`` or ``"sparse"``. Public routing-key aliases (``exact_dense``,
        ``dense_expm``, …) are normalized to these by :func:`ryd_gate.simulate`.
    n_steps
        Number of piecewise-constant intervals over the gate.
    """
    from ryd_gate.backends.exact.dense_expm import DenseExpmBackend
    from ryd_gate.backends.exact.sparse_expm import SparseExpmBackend

    if kind == "dense":
        return DenseExpmBackend(n_steps=n_steps)
    if kind == "sparse":
        return SparseExpmBackend(n_steps=n_steps)
    raise ValueError(
        f"Unknown exact expm backend {kind!r}; expected 'dense' or 'sparse'."
    )


def resolve_n_steps(system, opts) -> int:
    """Number of piecewise-constant intervals: explicit option, else protocol, else 200."""
    return opts.get("n_steps", getattr(system.protocol, "n_steps", 200))


def _build_cz_builder(system, x):
    """If a CZ *builder* (TO/AR, exposing ``build``) is bound, materialize a concrete
    pulse from ``x`` and rebind it, returning ``(system, ())``.  Otherwise pass through.
    Keeps the public ``simulate(system, x, ...)`` surface while leaving the runtime
    protocol free of the ``x`` vector."""
    proto = getattr(system, "protocol", None)
    if proto is not None and hasattr(proto, "build"):
        return system.with_protocol(proto.build(list(x), system)), ()
    return system, x


def simulate(
    system,
    x=(),
    psi0="all_ground",
    t_eval: np.ndarray | bool | None = None,
    backend: "SolverBackend | None" = None,
    force_kind: str | None = None,
    compiler=None,
    backend_options: "dict | ExactOptions | None" = None,
) -> EvolutionResult:
    """Compile a protocol-bound Rydberg system and evolve exactly.

    ``system`` must have a protocol bound. With ``backend=None`` the solver is
    selected automatically: sparse piecewise-exponential evolution for a sparse
    Hamiltonian IR, a dense piecewise matrix-exponential for the small physical-ladder
    models (``rb87_7``/``analog_3``, whose ~GHz intermediate manifold makes
    ``expm_multiply`` pathologically slow), otherwise a dense ODE integrator. Pass
    ``force_kind="dense"``/``"sparse"`` to force a piecewise-``expm`` solver (this is
    how :func:`ryd_gate.simulate`'s ``exact_dense``/``exact_sparse`` keys route here),
    or a concrete :class:`~ryd_gate.backends.exact.compiler.SolverBackend` as
    ``backend`` to override entirely. Backend dispatch by name lives in
    :func:`ryd_gate.simulate`; tensor-network and external algorithms live under
    ``ryd_gate.backends``. ``backend_options`` accepts a dict or an
    :class:`ExactOptions`.
    """
    from ryd_gate.backends.exact.compiler import ExactSparseCompiler
    from ryd_gate.core.system import RydbergSystem

    if not isinstance(system, RydbergSystem):
        raise TypeError(
            "simulate() requires RydbergSystem instances. "
            "Build one with RydbergSystem.set_atom_level(...).set_atom_geom(...).set_protocol(...) "
            "or call .with_protocol(...) before simulation."
        )

    system, x = _build_cz_builder(system, x)
    params = system.unpack_params(x)
    opts = as_backend_options(backend_options)

    compiler = compiler or ExactSparseCompiler()
    ir = compiler.compile(system, params)

    backend = _resolve_backend(system, ir, opts, backend, force_kind)

    return backend.evolve(ir, _exact_initial_state(system, psi0), params["t_gate"], t_eval)


def simulate_states(
    system,
    states,
    x=(),
    t_eval=None,
    backend_options=None,
    backend: "SolverBackend | None" = None,
    force_kind: str | None = None,
):
    """Evolve several initial states under the same bound protocol, sharing work.

    The dense ladder backend (``rb87_7``/``analog_3``) computes each step's propagator
    once and applies it to all states, so evolving N basis states costs ~the same as
    one -- far faster than calling :func:`simulate` per state. Other backends fall back
    to a per-state loop over the single compiled IR. ``backend``/``force_kind`` select
    the solver exactly as in :func:`simulate`. Returns a list of
    :class:`~ryd_gate.ir.EvolutionResult`, one per entry of *states*.
    """
    from ryd_gate.backends.exact.compiler import ExactSparseCompiler
    from ryd_gate.core.system import RydbergSystem

    if not isinstance(system, RydbergSystem):
        raise TypeError("simulate_states() requires a RydbergSystem with a bound protocol.")

    system, x = _build_cz_builder(system, x)
    params = system.unpack_params(x)
    opts = as_backend_options(backend_options)
    ir = ExactSparseCompiler().compile(system, params)
    backend = _resolve_backend(system, ir, opts, backend, force_kind)
    psis = [_exact_initial_state(system, s) for s in states]
    t_gate = params["t_gate"]
    if hasattr(backend, "evolve_many"):
        return backend.evolve_many(ir, psis, t_gate, t_eval)
    return [backend.evolve(ir, p, t_gate, t_eval) for p in psis]


def _exact_initial_state(system, psi0):
    if isinstance(psi0, np.ndarray):
        return psi0
    if isinstance(psi0, str):
        if psi0 == "all_ground":
            label = "1" if "1" in system.basis.local_levels else system.basis.local_levels[0]
            return system.product_state([label] * system.N)
        if psi0 == "all_1":
            return system.product_state(["1"] * system.N)
        if psi0 in {"all_0", "all_zero"}:
            return system.product_state(["0"] * system.N)
        if psi0 == "all_r":
            return system.product_state(["r"] * system.N)
        if psi0 == "plus":
            from ryd_gate.core.states import plus_local_amplitudes, product_superposition_state

            return product_superposition_state(
                plus_local_amplitudes(system.basis.local_levels), system.N
            )
    if isinstance(psi0, (list, tuple)):
        return system.product_state(list(psi0))
    return psi0
