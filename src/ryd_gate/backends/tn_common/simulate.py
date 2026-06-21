"""Small public dispatcher for tensor-network simulations.

Two TN backends are exposed:

- ``backend="mps"``: TeNPy CPU MPS reference path.
- ``backend="peps"``: YASTN finite-PEPS path for 2D rectangular lattices.

``simulate_tn`` lowers ``(spec, protocol, x)`` into a single :class:`TNEvolutionIR`
and hands it to :func:`simulate_tn_ir`, which is the one place that constructs a
backend object and calls its ``evolve_ir``. Every backend therefore consumes the
same IR through the same entry point.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ryd_gate.backends._options import as_backend_options
from ryd_gate.backends.tn_common.protocol_context import (
    TNProtocolContext,
    pin_deltas_from_params,
)
from ryd_gate.ir import EvolutionResult

if TYPE_CHECKING:
    from ryd_gate.backends.tenpy_mps.backends import TenpyOptions
    from ryd_gate.protocols.base import Protocol

    from .compiler import TNEvolutionIR

_TDVP_METHODS = frozenset({"tdvp", "mps_tdvp"})


def simulate_tn(
    spec,
    protocol: Protocol,
    x: list[float],
    initial_state: str | np.ndarray | object = "all_ground",
    method: str = "tdvp",
    backend: str = "mps",
    t_eval: np.ndarray | None = None,
    observables: list[str] | None = None,
    backend_options: "dict | TenpyOptions | None" = None,
) -> EvolutionResult:
    """Lower a TN lattice spec to a :class:`TNEvolutionIR` and evolve it."""
    from .compiler import TNEvolutionIR
    from .lattice_spec import TNLatticeSpec

    if not isinstance(spec, TNLatticeSpec):
        raise TypeError("simulate_tn() requires a TNLatticeSpec.")

    opts = as_backend_options(backend_options)
    backend = _normalize_backend(backend)

    if method == "dmrg" and backend == "mps":
        return _simulate_dmrg(spec, protocol, x, initial_state, opts)
    if method not in _TDVP_METHODS:
        extra = " or 'dmrg'" if backend == "mps" else ""
        raise ValueError(
            f"backend={backend!r} supports method='tdvp'/'mps_tdvp'{extra}; got {method!r}."
        )

    params = protocol.unpack_params(x, TNProtocolContext(spec))
    ir = TNEvolutionIR(
        spec=spec,
        protocol=protocol,
        params=params,
        metadata=_metadata(spec, backend),
    )
    return simulate_tn_ir(
        ir, initial_state, backend=backend,
        t_eval=t_eval, observables=observables, backend_options=opts,
    )


def simulate_tn_ir(
    ir: "TNEvolutionIR",
    initial_state: str | np.ndarray | object = "all_ground",
    backend: str = "mps",
    t_eval: np.ndarray | None = None,
    observables: list[str] | None = None,
    backend_options: "dict | TenpyOptions | None" = None,
) -> EvolutionResult:
    """Run a compiled TN IR with one of the public TN backends.

    The single place that selects a backend class and calls ``evolve_ir``.
    """
    opts = as_backend_options(backend_options)
    backend = _normalize_backend(backend)

    if backend == "mps":
        from ryd_gate.backends.tenpy_mps.backends import TenpyTDVPBackend

        return TenpyTDVPBackend(**opts).evolve_ir(
            ir, initial_state, t_eval=t_eval, observables=observables,
        )

    if backend == "peps":
        from ryd_gate.backends.peps2d import YASTNPEPSBackend

        return YASTNPEPSBackend(**opts).evolve_ir(
            ir, initial_state=initial_state, t_eval=t_eval, observables=observables,
        )

    raise AssertionError(f"Unhandled normalized TN backend {backend!r}.")


def _simulate_dmrg(spec, protocol, x, initial_state, opts: dict) -> EvolutionResult:
    """Ground-state DMRG path (mps only): structurally distinct from time evolution."""
    from ryd_gate.backends.tenpy_mps.backends import TenpyDMRGBackend

    backend = TenpyDMRGBackend(**opts)
    params = protocol.unpack_params(x, TNProtocolContext(spec))
    pin_deltas = pin_deltas_from_params(params, spec.N)
    delta = params.get("delta_end", params.get("Delta"))
    if delta is None:
        delta = x[1] if len(x) > 1 else x[0] if len(x) == 1 else None
    if delta is None:
        raise ValueError("DMRG requires a protocol exposing 'delta_end' or 'Delta'.")
    return backend.find_ground_state(
        spec, delta, pin_deltas=pin_deltas, initial_state=initial_state,
    )


def _normalize_backend(backend: str) -> str:
    key = str(backend).lower()
    if key in {"mps", "peps"}:
        return key
    raise ValueError(f"Unknown TN backend: {backend!r}. Use 'mps' or 'peps'.")


def _metadata(spec, backend: str) -> dict:
    return {
        "compiler": "tn",
        "tn_spec": spec,
        "backend": backend,
        "n_sites": spec.N,
        "local_dim": spec.level_spec.local_dim,
    }
