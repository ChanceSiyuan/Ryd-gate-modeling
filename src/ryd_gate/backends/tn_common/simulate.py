"""High-level TN simulation entry point."""

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
    backend: str = "tenpy",
    t_eval: np.ndarray | None = None,
    observables: list[str] | None = None,
    backend_options: "dict | TenpyOptions | GPUTNOptions | None" = None,
) -> EvolutionResult:
    """High-level TN simulation entry point.

    Parameters
    ----------
    spec : TNLatticeSpec
        Tensor-network lattice specification.
    protocol : Protocol
        Protocol describing drive coefficients and local addressing.
        TDVP supports ``SweepProtocol`` plus ``DigitalAnalogProtocol`` on
        either the effective ``1r`` TN subspace or explicit ``01r`` TN
        local levels.
    x : list
        Protocol parameters. ``SweepProtocol`` and ``DigitalAnalogProtocol``
        store their schedules internally and expect ``[]``.
    initial_state : str, ndarray, or tenpy MPS
        Initial state. Strings: ``"all_ground"``, ``"af1"``,
        ``"af2"``. Arrays: per-site occupation (0/1).
    method : str
        ``"tdvp"``/``"mps_tdvp"`` for MPS time evolution, ``"dmrg"``
        for ground state, or a delegated method such as ``"itensors_tebd"``,
        ``"gputtn_tdvp"``, ``"ttn_tdvp"``, ``"2dtn_bp"``, or ``"nqs_tvmc"``.
    backend : str
        Tensor-network engine backend. ``"tenpy"``/``"mps"`` selects the
        CPU MPS path, ``"itensors"`` selects the Julia ITensors adapter,
        ``"gputn"`` selects the CUDA MPS/TN adapter, ``"gputtn"`` selects the
        Julia ITensorNetworks TTN-TDVP adapter, ``"2dtn"`` selects the Julia
        TensorNetworkQuantumSimulator.jl adapter by default, and ``"ttn"`` or
        ``"nqs"`` select external adapter paths.
    t_eval : ndarray or None
        Times at which to record observables (only for TDVP).
    observables : list of str or None
        Observable names to record (only for TDVP).
    backend_options : dict or None
        Options passed to the backend constructor
        (e.g., ``{"chi_max": 512, "dt": 0.1}``). For external backends,
        ``engine_package`` selects a supported Python package such as
        ``"yastn"``, ``"pytreenet"``, or ``"netket"``; ``engine`` can still be
        supplied for a custom adapter exposing ``evolve(payload, ...)``. For
        ``backend="2dtn"``, omitting ``engine``/``engine_package`` uses the
        Julia TensorNetworkQuantumSimulator.jl bridge.

    Returns
    -------
    EvolutionResult
    """
    from .lattice_spec import TNLatticeSpec

    if not isinstance(spec, TNLatticeSpec):
        raise TypeError("simulate_tn() requires a TNLatticeSpec.")

    opts = as_backend_options(backend_options)
    backend = backend.lower()

    backend = _normalize_backend(backend)

    if backend == "gputn":
        if method not in {"tdvp", "mps_tdvp"}:
            raise ValueError("backend='gputn' currently supports method='tdvp'/'mps_tdvp' only.")
        from ryd_gate.backends.gputn.backend import GPUTNTDVPBackend

        gpu_backend = GPUTNTDVPBackend(**opts)
        return gpu_backend.evolve(
            spec, protocol, x, initial_state,
            t_eval=t_eval,
            observables=observables,
        )

    if backend == "itensors":
        from ryd_gate.backends.itensor.backend import ITensorsJuliaBackend

        from .compiler import TNEvolutionIR

        params = protocol.unpack_params(x, _protocol_context(spec))
        ir = TNEvolutionIR(
            spec=spec,
            protocol=protocol,
            params=params,
            method=_default_method_for_backend(backend, method),
            metadata={
                "compiler": "tn",
                "tn_spec": spec,
                "backend": backend,
                "n_sites": spec.N,
                "local_dim": spec.level_spec.local_dim,
            },
        )
        backend_obj = ITensorsJuliaBackend(**opts)
        return backend_obj.evolve_ir(
            ir,
            initial_state=initial_state,
            t_eval=t_eval,
            observables=observables,
        )

    if backend == "gputtn":
        from ryd_gate.backends.itensor.gputtn_backend import GPUITensorNetworksTTNBackend

        from .compiler import TNEvolutionIR

        params = protocol.unpack_params(x, _protocol_context(spec))
        ir = TNEvolutionIR(
            spec=spec,
            protocol=protocol,
            params=params,
            method="gputtn_tdvp",
            metadata={
                "compiler": "tn",
                "tn_spec": spec,
                "backend": backend,
                "n_sites": spec.N,
                "local_dim": spec.level_spec.local_dim,
            },
        )
        return GPUITensorNetworksTTNBackend(**opts).evolve_ir(
            ir,
            initial_state=initial_state,
            t_eval=t_eval,
            observables=observables,
        )

    if backend == "yastn_mps":
        from ryd_gate.backends.yastn_mps import YASTNMPSBackend

        from .compiler import TNEvolutionIR

        params = protocol.unpack_params(x, _protocol_context(spec))
        ir = TNEvolutionIR(
            spec=spec,
            protocol=protocol,
            params=params,
            method="mps_tdvp",
            metadata={
                "compiler": "tn",
                "tn_spec": spec,
                "backend": backend,
                "n_sites": spec.N,
                "local_dim": spec.level_spec.local_dim,
            },
        )
        return YASTNMPSBackend(**opts).evolve_ir(
            ir,
            initial_state=initial_state,
            t_eval=t_eval,
            observables=observables,
        )

    if backend in {"ttn", "2dtn", "nqs"}:
        from .compiler import TNEvolutionIR

        params = protocol.unpack_params(x, _protocol_context(spec))
        ir = TNEvolutionIR(
            spec=spec,
            protocol=protocol,
            params=params,
            method=_default_method_for_backend(backend, method),
            metadata={
                "compiler": "tn",
                "tn_spec": spec,
                "backend": backend,
                "n_sites": spec.N,
                "local_dim": spec.level_spec.local_dim,
            },
        )
        if backend == "2dtn" and _use_tnqs_2dtn_backend(opts):
            from ryd_gate.backends.itensor.tnqs_backend import TNQSJulia2DTNBackend

            backend_obj = TNQSJulia2DTNBackend(**_tnqs_backend_options(opts))
            return backend_obj.evolve_ir(
                ir,
                initial_state=initial_state,
                t_eval=t_eval,
                observables=observables,
            )
        return _external_backend(backend, opts).evolve_ir(
            ir,
            initial_state=initial_state,
            t_eval=t_eval,
            observables=observables,
        )

    if method == "dmrg":
        from ryd_gate.backends.tenpy_mps.backends import (
            TenpyDMRGBackend,
            _pin_deltas_from_params,
        )

        backend = TenpyDMRGBackend(**opts)
        params = protocol.unpack_params(x, _protocol_context(spec))
        pin_deltas = _pin_deltas_from_params(params, spec.N)
        # For DMRG, use the final/constant detuning from the protocol.
        Delta = params.get("delta_end", params.get("Delta"))
        if Delta is None:
            Delta = x[1] if len(x) > 1 else x[0] if len(x) == 1 else None
        if Delta is None:
            raise ValueError("DMRG requires a protocol exposing 'delta_end' or 'Delta'.")
        return backend.find_ground_state(
            spec, Delta, pin_deltas=pin_deltas,
            initial_state=initial_state,
        )

    elif method in {"tdvp", "mps_tdvp"}:
        from ryd_gate.backends.tenpy_mps.backends import TenpyTDVPBackend
        from ryd_gate.backends.tenpy_mps.state import product_state_mps

        backend = TenpyTDVPBackend(**opts)
        # Build initial MPS if not already one
        if not hasattr(initial_state, "expectation_value"):
            psi0 = product_state_mps(spec, initial_state)
        else:
            psi0 = initial_state

        return backend.evolve(
            spec, protocol, x, psi0,
            t_eval=t_eval,
            observables=observables,
        )

    else:
        raise ValueError(
        f"Unknown method: {method!r}. Use 'dmrg', 'tdvp', 'mps_tdvp', "
            "'itensors_tebd', 'gputtn_tdvp', 'ttn_tdvp', '2dtn_bp', or 'nqs_tvmc'."
        )


def simulate_tn_ir(
    ir: "TNEvolutionIR",
    initial_state: str | np.ndarray | object = "all_ground",
    backend: str = "tenpy",
    t_eval: np.ndarray | None = None,
    observables: list[str] | None = None,
    backend_options: "dict | TenpyOptions | GPUTNOptions | None" = None,
) -> EvolutionResult:
    """Run a compiled TN IR with the requested TN backend."""
    opts = as_backend_options(backend_options)
    backend = backend.lower()

    backend = _normalize_backend(backend)

    if backend == "tenpy":
        from ryd_gate.backends.tenpy_mps.backends import TenpyTDVPBackend
        from ryd_gate.backends.tenpy_mps.state import product_state_mps

        backend_obj = TenpyTDVPBackend(**opts)
        if not hasattr(initial_state, "expectation_value"):
            psi0 = product_state_mps(ir.spec, initial_state)
        else:
            psi0 = initial_state
        return backend_obj.evolve_ir(
            ir,
            psi0,
            t_eval=t_eval,
            observables=observables,
        )

    if backend == "gputn":
        from ryd_gate.backends.gputn.backend import GPUTNTDVPBackend

        backend_obj = GPUTNTDVPBackend(**opts)
        return backend_obj.evolve_ir(
            ir,
            initial_state,
            t_eval=t_eval,
            observables=observables,
        )

    if backend == "yastn_mps":
        from ryd_gate.backends.yastn_mps import YASTNMPSBackend

        backend_obj = YASTNMPSBackend(**opts)
        return backend_obj.evolve_ir(
            ir,
            initial_state=initial_state,
            t_eval=t_eval,
            observables=observables,
        )

    if backend == "itensors":
        from ryd_gate.backends.itensor.backend import ITensorsJuliaBackend

        backend_obj = ITensorsJuliaBackend(**opts)
        return backend_obj.evolve_ir(
            ir,
            initial_state=initial_state,
            t_eval=t_eval,
            observables=observables,
        )

    if backend == "gputtn":
        from ryd_gate.backends.itensor.gputtn_backend import GPUITensorNetworksTTNBackend

        backend_obj = GPUITensorNetworksTTNBackend(**opts)
        return backend_obj.evolve_ir(
            ir,
            initial_state=initial_state,
            t_eval=t_eval,
            observables=observables,
        )

    if backend in {"ttn", "2dtn", "nqs"}:
        if backend == "2dtn" and _use_tnqs_2dtn_backend(opts):
            from ryd_gate.backends.itensor.tnqs_backend import TNQSJulia2DTNBackend

            backend_obj = TNQSJulia2DTNBackend(**_tnqs_backend_options(opts))
            return backend_obj.evolve_ir(
                ir,
                initial_state=initial_state,
                t_eval=t_eval,
                observables=observables,
            )
        return _external_backend(backend, opts).evolve_ir(
            ir,
            initial_state=initial_state,
            t_eval=t_eval,
            observables=observables,
        )

    raise ValueError(
        f"Unknown TN backend: {backend!r}. Use 'tenpy', 'mps', 'mps_gpu', 'gputn', "
        "'itensors', 'gputtn', 'ttn', '2dtn', or 'nqs'."
    )


def _normalize_backend(backend: str) -> str:
    key = backend.lower()
    if key in {"tenpy", "tn", "mps", "mps_tdvp"}:
        return "tenpy"
    if key in {"gputn", "gpu"}:
        return "gputn"
    if key in {"yastn_mps", "mps_yastn", "mps_gpu", "gpu_mps"}:
        return "yastn_mps"
    if key in {"itensors", "itensors_tebd", "itensor", "itensor_mps"}:
        return "itensors"
    if key in {"gputtn", "gpu_ttn", "itensornetworks_ttn", "itensornetworks"}:
        return "gputtn"
    if key in {"ttn", "ttn_tdvp"}:
        return "ttn"
    if key in {"2dtn", "2dtn_bp", "peps", "peps_bp"}:
        return "2dtn"
    if key in {"nqs", "nqs_tvmc", "tvmc"}:
        return "nqs"
    raise ValueError(
        f"Unknown TN backend: {backend!r}. Use 'tenpy', 'mps', 'mps_gpu', 'gputn', "
        "'itensors', 'gputtn', 'ttn', '2dtn', or 'nqs'."
    )


def _default_method_for_backend(backend: str, method: str) -> str:
    if method not in {"tdvp", "mps_tdvp"}:
        return method
    return {
        "itensors": "itensors_tebd",
        "gputtn": "gputtn_tdvp",
        "ttn": "ttn_tdvp",
        "2dtn": "2dtn_bp",
        "nqs": "nqs_tvmc",
    }[backend]


def _external_backend(backend: str, opts: dict):
    if backend == "ttn":
        from .external_backends import ExternalTTNTDVPBackend

        return ExternalTTNTDVPBackend(**opts)
    if backend == "2dtn":
        from .external_backends import External2DTNBPBackend

        return External2DTNBPBackend(**opts)
    if backend == "nqs":
        from .external_backends import ExternalNQSTVMCBackend

        return ExternalNQSTVMCBackend(**opts)
    raise ValueError(f"No external adapter for backend {backend!r}.")


_TNQS_ENGINE_PACKAGES = {
    "tnqs",
    "tnqsim",
    "tensornetworkquantumsimulator",
    "tensornetworkquantumsimulator.jl",
    "tensor_network_quantum_simulator",
    "julia",
    "itensors",
}


def _use_tnqs_2dtn_backend(opts: dict) -> bool:
    if opts.get("engine") is not None:
        return False
    engine_package = opts.get("engine_package")
    if engine_package is None:
        return True
    return str(engine_package).strip().lower() in _TNQS_ENGINE_PACKAGES


def _tnqs_backend_options(opts: dict) -> dict:
    options = dict(opts)
    engine_package = options.pop("engine_package", None)
    if engine_package is not None and str(engine_package).strip().lower() not in _TNQS_ENGINE_PACKAGES:
        options["engine_package"] = engine_package
    return options


def _protocol_context(spec):
    from ryd_gate.backends.tenpy_mps.backends import _TNProtocolContext

    return _TNProtocolContext(spec)
