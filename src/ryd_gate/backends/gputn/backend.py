"""GPU tensor-network backend entry points.

This module intentionally does not import CUDA libraries at module import time.
The CPU TeNPy path should remain usable on machines without a GPU, while
``backend="gputn"`` fails early with a clear dependency/device error.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING

from ryd_gate.ir.evolution import EvolutionResult

if TYPE_CHECKING:
    import numpy as np

    from ryd_gate.backends.tn_common.lattice_spec import TNLatticeSpec
    from ryd_gate.protocols.base import Protocol


class GPUTNDependencyError(ImportError):
    """Raised when the GPU TN backend dependencies or CUDA device are unavailable."""


def _missing_dependency_message(missing: list[str]) -> str:
    missing_text = ", ".join(missing)
    return (
        f"GPUTN backend requires CUDA tensor-network dependencies: {missing_text}. "
        "For CUDA 12, install the optional extra with "
        "`pip install -e '.[gputn-cu12]'` or install NVIDIA cuQuantum Python "
        "manually for your CUDA version. The CPU TeNPy backend remains available "
        "with `backend='mps'`."
    )


def _require_gputn_dependencies(
    *,
    require_gpu: bool = True,
    device_id: int | None = 0,
) -> SimpleNamespace:
    """Return GPU dependency modules after checking importability and device access."""
    missing = [
        module_name
        for module_name in ("cupy", "cuquantum")
        if importlib.util.find_spec(module_name) is None
    ]
    if missing:
        raise GPUTNDependencyError(_missing_dependency_message(missing))

    import cupy
    import cuquantum

    n_devices: int | None = None
    if require_gpu:
        try:
            n_devices = int(cupy.cuda.runtime.getDeviceCount())
        except Exception as exc:  # pragma: no cover - depends on local CUDA runtime
            raise GPUTNDependencyError(
                "GPUTN backend found CuPy/cuQuantum but could not query CUDA devices. "
                "Check the NVIDIA driver, CUDA runtime libraries, and LD_LIBRARY_PATH."
            ) from exc
        if n_devices <= 0:
            raise GPUTNDependencyError("GPUTN backend requires at least one CUDA device.")
        if device_id is not None and not 0 <= int(device_id) < n_devices:
            raise GPUTNDependencyError(
                f"GPUTN backend requested CUDA device {device_id}, "
                f"but only {n_devices} device(s) are visible."
            )

    return SimpleNamespace(cupy=cupy, cuquantum=cuquantum, n_devices=n_devices)


def gputn_available(*, require_gpu: bool = True, device_id: int | None = 0) -> bool:
    """Return whether the optional GPU TN dependencies and device are available."""
    try:
        _require_gputn_dependencies(require_gpu=require_gpu, device_id=device_id)
    except GPUTNDependencyError:
        return False
    return True


@dataclass
class GPUTNTDVPBackend:
    """GPU tensor-network lattice-evolution backend adapter.

    The public adapter is wired into :func:`tn_common.simulate_tn` so callers
    can switch from ``backend="mps"`` to ``backend="gputn"`` without changing
    protocol/notebook code.  When ``engine`` is omitted, the built-in
    :mod:`gputn` package runs a cuTensorNet MPS Trotter kernel when available,
    with a small-system CuPy state-vector fallback.  Custom engines can still be
    injected by exposing an ``evolve(...)`` method with the signature below.
    """

    chi_max: int = 256
    dt: float = 0.2
    svd_min: float = 1e-10
    device_id: int | None = 0
    require_gpu: bool = True
    engine: object | None = None
    kernel: str = "auto"
    trotter_order: int = 2
    statevector_max_sites: int | None = 24
    return_state_vector: bool = False

    def evolve_ir(
        self,
        ir,
        initial_state: str | np.ndarray | object = "all_ground",
        t_eval: np.ndarray | None = None,
        observables: list[str] | None = None,
    ) -> EvolutionResult:
        """Evolve a compiled TN IR with the configured GPU TN engine."""
        return self.evolve_compiled(
            ir.spec,
            ir.protocol,
            ir.params,
            initial_state,
            t_eval=t_eval,
            observables=observables,
        )

    def evolve_compiled(
        self,
        spec: TNLatticeSpec,
        protocol: Protocol,
        params: dict,
        psi0: object,
        t_eval: np.ndarray | None = None,
        observables: list[str] | None = None,
    ) -> EvolutionResult:
        """Evolve with already-unpacked protocol parameters."""
        deps = _require_gputn_dependencies(
            require_gpu=self.require_gpu,
            device_id=self.device_id,
        )
        engine = self.engine
        if engine is None:
            from ryd_gate.backends.gputn import CuTensorNetRydbergEngine

            engine = CuTensorNetRydbergEngine(
                kernel=self.kernel,
                trotter_order=self.trotter_order,
                statevector_max_sites=self.statevector_max_sites,
                return_state_vector=self.return_state_vector,
            )

        result = engine.evolve(
            spec,
            protocol,
            params,
            psi0,
            t_eval=t_eval,
            observables=observables,
            chi_max=self.chi_max,
            dt=self.dt,
            svd_min=self.svd_min,
            device_id=self.device_id,
            xp=deps.cupy,
            cuquantum=deps.cuquantum,
        )
        result.metadata.setdefault("backend", "gputn")
        result.metadata.setdefault("accelerator", "cuda")
        if self.engine is None:
            result.metadata.setdefault("engine_package", "gputn")
        return result
