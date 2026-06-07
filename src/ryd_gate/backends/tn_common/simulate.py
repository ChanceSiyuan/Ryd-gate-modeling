"""Small public dispatcher for tensor-network simulations.

Only three TN backends are exposed:

- ``backend="mps"``: TeNPy CPU MPS reference path.
- ``backend="peps"``: YASTN finite-PEPS path for 2D lattices.
- ``backend="gputn"``: experimental CuPy/cuQuantum GPU validation path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ryd_gate.backends._options import as_backend_options
from ryd_gate.ir.evolution import EvolutionResult

if TYPE_CHECKING:
    from ryd_gate.backends.gputn.options import GPUTNOptions
    from ryd_gate.backends.tenpy_mps.options import TenpyOptions
    from ryd_gate.protocols.base import Protocol

    from .compiler import TNEvolutionIR


def simulate_tn(
    spec,
    protocol: Protocol,
    x: list[float],
    initial_state: str | np.ndarray | object = "all_ground",
    method: str = "tdvp",
    backend: str = "mps",
    t_eval: np.ndarray | None = None,
    observables: list[str] | None = None,
    backend_options: "dict | TenpyOptions | GPUTNOptions | None" = None,
) -> EvolutionResult:
    """Evolve a TN lattice spec with one of the public TN backends."""
    from .lattice_spec import TNLatticeSpec

    if not isinstance(spec, TNLatticeSpec):
        raise TypeError("simulate_tn() requires a TNLatticeSpec.")

    opts = as_backend_options(backend_options)
    backend = _normalize_backend(backend)

    if backend == "mps":
        return _simulate_mps(
            spec, protocol, x, initial_state, method,
            t_eval=t_eval, observables=observables, opts=opts,
        )
    if backend == "peps":
        return _simulate_peps(
            spec, protocol, x, initial_state, method,
            t_eval=t_eval, observables=observables, opts=opts,
        )
    if backend == "gputn":
        return _simulate_gputn(
            spec, protocol, x, initial_state, method,
            t_eval=t_eval, observables=observables, opts=opts,
        )
    if backend == "pepskit":
        return _simulate_pepskit(
            spec, protocol, x, initial_state, method,
            t_eval=t_eval, observables=observables, opts=opts,
        )
    raise AssertionError(f"Unhandled normalized TN backend {backend!r}.")


def simulate_tn_ir(
    ir: "TNEvolutionIR",
    initial_state: str | np.ndarray | object = "all_ground",
    backend: str = "mps",
    t_eval: np.ndarray | None = None,
    observables: list[str] | None = None,
    backend_options: "dict | TenpyOptions | GPUTNOptions | None" = None,
) -> EvolutionResult:
    """Run a compiled TN IR with one of the public TN backends."""
    opts = as_backend_options(backend_options)
    backend = _normalize_backend(backend)

    if backend == "mps":
        from ryd_gate.backends.tenpy_mps.backends import TenpyTDVPBackend
        from ryd_gate.backends.tenpy_mps.state import product_state_mps

        backend_obj = TenpyTDVPBackend(**opts)
        psi0 = initial_state if hasattr(initial_state, "expectation_value") else product_state_mps(ir.spec, initial_state)
        return backend_obj.evolve_ir(ir, psi0, t_eval=t_eval, observables=observables)

    if backend == "peps":
        from ryd_gate.backends.peps2d.yastn_backend import YASTNPEPSBackend

        return YASTNPEPSBackend(**_peps_options(opts)).evolve_ir(
            ir,
            initial_state=initial_state,
            t_eval=t_eval,
            observables=observables,
        )

    if backend == "gputn":
        from ryd_gate.backends.gputn.backend import GPUTNTDVPBackend

        return GPUTNTDVPBackend(**opts).evolve_ir(
            ir,
            initial_state,
            t_eval=t_eval,
            observables=observables,
        )

    if backend == "pepskit":
        from ryd_gate.backends.pepskit.backend import PEPSKitIPEPSBackend

        return PEPSKitIPEPSBackend(**opts).evolve_ir(
            ir,
            initial_state=initial_state,
            t_eval=t_eval,
            observables=observables,
        )

    raise AssertionError(f"Unhandled normalized TN backend {backend!r}.")


def _simulate_mps(
    spec,
    protocol: Protocol,
    x: list[float],
    initial_state: str | np.ndarray | object,
    method: str,
    *,
    t_eval: np.ndarray | None,
    observables: list[str] | None,
    opts: dict,
) -> EvolutionResult:
    if method == "dmrg":
        from ryd_gate.backends.tenpy_mps.backends import (
            TenpyDMRGBackend,
            _pin_deltas_from_params,
        )

        backend = TenpyDMRGBackend(**opts)
        params = protocol.unpack_params(x, _protocol_context(spec))
        pin_deltas = _pin_deltas_from_params(params, spec.N)
        delta = params.get("delta_end", params.get("Delta"))
        if delta is None:
            delta = x[1] if len(x) > 1 else x[0] if len(x) == 1 else None
        if delta is None:
            raise ValueError("DMRG requires a protocol exposing 'delta_end' or 'Delta'.")
        return backend.find_ground_state(
            spec,
            delta,
            pin_deltas=pin_deltas,
            initial_state=initial_state,
        )

    if method not in {"tdvp", "mps_tdvp"}:
        raise ValueError("backend='mps' supports method='tdvp', 'mps_tdvp', or 'dmrg'.")

    from ryd_gate.backends.tenpy_mps.backends import TenpyTDVPBackend
    from ryd_gate.backends.tenpy_mps.state import product_state_mps

    backend = TenpyTDVPBackend(**opts)
    psi0 = initial_state if hasattr(initial_state, "expectation_value") else product_state_mps(spec, initial_state)
    return backend.evolve(
        spec,
        protocol,
        x,
        psi0,
        t_eval=t_eval,
        observables=observables,
    )


def _simulate_peps(
    spec,
    protocol: Protocol,
    x: list[float],
    initial_state: str | np.ndarray | object,
    method: str,
    *,
    t_eval: np.ndarray | None,
    observables: list[str] | None,
    opts: dict,
) -> EvolutionResult:
    if method not in {"tdvp", "mps_tdvp"}:
        raise ValueError("backend='peps' supports method='tdvp'/'mps_tdvp' only.")

    from ryd_gate.backends.peps2d.yastn_backend import YASTNPEPSBackend

    from .compiler import TNEvolutionIR

    params = protocol.unpack_params(x, _protocol_context(spec))
    ir = TNEvolutionIR(
        spec=spec,
        protocol=protocol,
        params=params,
        method="peps_yastn",
        metadata=_metadata(spec, "peps"),
    )
    return YASTNPEPSBackend(**_peps_options(opts)).evolve_ir(
        ir,
        initial_state=initial_state,
        t_eval=t_eval,
        observables=observables,
    )


def _simulate_gputn(
    spec,
    protocol: Protocol,
    x: list[float],
    initial_state: str | np.ndarray | object,
    method: str,
    *,
    t_eval: np.ndarray | None,
    observables: list[str] | None,
    opts: dict,
) -> EvolutionResult:
    if method not in {"tdvp", "mps_tdvp"}:
        raise ValueError("backend='gputn' supports method='tdvp'/'mps_tdvp' only.")

    from ryd_gate.backends.gputn.backend import GPUTNTDVPBackend

    backend = GPUTNTDVPBackend(**opts)
    return backend.evolve(
        spec,
        protocol,
        x,
        initial_state,
        t_eval=t_eval,
        observables=observables,
    )


def _simulate_pepskit(
    spec,
    protocol: Protocol,
    x: list[float],
    initial_state: str | np.ndarray | object,
    method: str,
    *,
    t_eval: np.ndarray | None,
    observables: list[str] | None,
    opts: dict,
) -> EvolutionResult:
    if method not in {"tdvp", "mps_tdvp"}:
        raise ValueError("backend='pepskit' supports method='tdvp'/'mps_tdvp' only.")

    from ryd_gate.backends.pepskit.backend import PEPSKitIPEPSBackend

    from .compiler import TNEvolutionIR

    params = protocol.unpack_params(x, _protocol_context(spec))
    ir = TNEvolutionIR(
        spec=spec,
        protocol=protocol,
        params=params,
        method="pepskit_ipeps_su",
        metadata=_metadata(spec, "pepskit"),
    )
    return PEPSKitIPEPSBackend(**opts).evolve_ir(
        ir,
        initial_state=initial_state,
        t_eval=t_eval,
        observables=observables,
    )


def _normalize_backend(backend: str) -> str:
    key = str(backend).lower()
    if key in {"mps", "peps", "gputn"}:
        return key
    if key in {"pepskit", "ipeps", "pepskit_su", "pepskit_ipeps"}:
        return "pepskit"
    raise ValueError(f"Unknown TN backend: {backend!r}. Use 'mps', 'peps', 'gputn', or 'pepskit'.")


def _peps_options(opts: dict) -> dict:
    options = dict(opts)
    engine = options.pop("engine", None)
    if engine is not None:
        raise ValueError("backend='peps' uses the built-in YASTN PEPS backend; custom engines are not public.")
    engine_package = options.pop("engine_package", None)
    if engine_package is not None and str(engine_package).strip().lower() != "yastn":
        raise ValueError("backend='peps' uses engine_package='yastn' only.")
    return options


def _metadata(spec, backend: str) -> dict:
    return {
        "compiler": "tn",
        "tn_spec": spec,
        "backend": backend,
        "n_sites": spec.N,
        "local_dim": spec.level_spec.local_dim,
    }


def _protocol_context(spec):
    from ryd_gate.backends.tenpy_mps.backends import _TNProtocolContext

    return _TNProtocolContext(spec)
