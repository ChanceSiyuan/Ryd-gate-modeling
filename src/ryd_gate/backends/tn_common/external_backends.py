"""External tensor-network / variational solver adapter boundaries.

The large-lattice algorithms discussed in ``main.tex`` are delegated to
specialized optional packages instead of becoming hard dependencies here:

- 2D-TN/BP or PEPS: ``yastn`` first, ``quimb`` as a Python TN alternative
- TTN-TDVP: ``pytreenet``
- NQS-tVMC: ``netket`` first, ``jVMC`` as a JAX NQS alternative

This module keeps the core Python API stable by building a serializable payload
from ``TNEvolutionIR``.  Users can still inject a custom engine, but a backend
without an engine now resolves to an explicit Python package target and fails
early with actionable dependency/adapter messages.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any

import numpy as np

from ryd_gate.ir.evolution import EvolutionResult


class ExternalSolverDependencyError(ImportError):
    """Raised when an optional external solver package is unavailable."""


@dataclass(frozen=True)
class SolverPackageSpec:
    """Metadata for an optional external solver package."""

    package: str
    import_names: tuple[str, ...]
    extra: str
    role: str
    docs_url: str
    gpu_note: str


_PACKAGE_SPECS: dict[str, SolverPackageSpec] = {
    "yastn": SolverPackageSpec(
        package="yastn",
        import_names=("yastn",),
        extra="tn-2d",
        role="2D PEPS/fPEPS with BP-style environments",
        docs_url="https://yastn.github.io/yastn/",
        gpu_note=(
            "install from GitHub; GPU runs use YASTN's PyTorch backend with "
            "default_device='cuda'"
        ),
    ),
    "quimb": SolverPackageSpec(
        package="quimb",
        import_names=("quimb",),
        extra="tn-2d",
        role="general Python tensor networks, 2D TN, and PEPS/TEBD tooling",
        docs_url="https://quimb.readthedocs.io/",
        gpu_note="CuPy/JAX-style acceleration through array backends; this adapter supports NumPy/CuPy",
    ),
    "pytreenet": SolverPackageSpec(
        package="pytreenet",
        import_names=("ryd_gate._vendor.pytreenet", "pytreenet"),
        extra="tn-ttn",
        role="tree tensor networks and TTN time evolution",
        docs_url="https://pytreenet.readthedocs.io/",
        gpu_note="vendored PyTreeNet 1.0.0 is NumPy/SciPy CPU-only",
    ),
    "netket": SolverPackageSpec(
        package="netket",
        import_names=("netket",),
        extra="nqs",
        role="JAX neural quantum states, VMC, and TDVP/tVMC",
        docs_url="https://netket.readthedocs.io/",
        gpu_note="JAX accelerator support; install a CUDA-enabled JAX build",
    ),
    "jvmc": SolverPackageSpec(
        package="jVMC",
        import_names=("jVMC", "jvmc"),
        extra="nqs",
        role="JAX neural quantum states and time-dependent VMC",
        docs_url="https://jvmc.readthedocs.io/",
        gpu_note="JAX accelerator support; install a CUDA-enabled JAX build",
    ),
}

_PACKAGE_ALIASES = {
    "ptn": "pytreenet",
    "pytreenet": "pytreenet",
    "netket": "netket",
    "jvmc": "jvmc",
    "jvmc.py": "jvmc",
    "jvmc_python": "jvmc",
    "jvmc-python": "jvmc",
    "jvmcjax": "jvmc",
    "jvmc-jax": "jvmc",
    "jVMC": "jvmc",
    "yastn": "yastn",
    "quimb": "quimb",
}

_BACKEND_PACKAGES = {
    "2dtn": ("yastn", "quimb"),
    "ttn": ("pytreenet",),
    "nqs": ("netket", "jvmc"),
}

_DEFAULT_BACKEND_PACKAGE = {
    "2dtn": "yastn",
    "ttn": "pytreenet",
    "nqs": "netket",
}


def build_external_solver_payload(
    spec,
    protocol,
    params: dict,
    *,
    t_eval: np.ndarray | None,
    observables: list[str] | None,
) -> dict[str, Any]:
    """Create a backend-agnostic payload for external lattice solvers."""
    return {
        "lattice": {
            "Lx": spec.Lx,
            "Ly": spec.Ly,
            "N": spec.N,
            "coords": np.asarray(spec.coords, dtype=float),
            "sublattice": np.asarray(spec.sublattice, dtype=float),
            "vdw_pairs": tuple(
                (int(i), int(j), float(spec.V_nn) * float(v_rel))
                for i, j, v_rel in spec.vdw_pairs
            ),
            "level_structure": spec.level_structure,
            "ordering": getattr(spec, "ordering", "snake"),
        },
        "protocol": {
            "class": protocol.__class__.__name__,
            "params": params,
            "required_channels": sorted(protocol.required_channels),
        },
        "t_eval": None if t_eval is None else np.asarray(t_eval, dtype=float),
        "observables": list(observables or []),
    }


def available_external_solver_packages(
    backend: str | None = None,
    *,
    require_installed: bool = False,
) -> dict[str, SolverPackageSpec]:
    """Return supported optional solver packages.

    Parameters
    ----------
    backend
        ``"2dtn"``, ``"ttn"``, ``"nqs"``, or None for all packages.
    require_installed
        If True, only include packages whose import target is available.
    """
    if backend is None:
        names = tuple(_PACKAGE_SPECS)
    else:
        key = _canonical_backend_name(backend)
        names = _BACKEND_PACKAGES[key]

    packages = {
        name: _PACKAGE_SPECS[name]
        for name in names
        if not require_installed or _package_available(_PACKAGE_SPECS[name])
    }
    return packages


@dataclass
class ExternalSolverBackend:
    """Base adapter for external many-body solvers.

    ``engine`` is expected to expose ``evolve(payload, initial_state, **kwargs)``
    and return :class:`EvolutionResult`.  Keeping this narrow interface avoids
    importing optional PEPS/TTN/NQS stacks inside the core package.
    """

    solver_name: str
    method: str
    engine: object | None = None
    engine_package: str | None = None
    options: dict | None = None

    def evolve_ir(
        self,
        ir,
        initial_state: str | np.ndarray | object = "all_ground",
        t_eval: np.ndarray | None = None,
        observables: list[str] | None = None,
    ) -> EvolutionResult:
        payload = build_external_solver_payload(
            ir.spec,
            ir.protocol,
            ir.params,
            t_eval=t_eval,
            observables=observables,
        )
        payload["method"] = self.method
        payload["metadata"] = ir.metadata or {}

        engine = self.engine or _package_engine_for_backend(
            self.solver_name,
            self.engine_package,
        )

        result = engine.evolve(
            payload,
            initial_state,
            **(self.options or {}),
        )
        if not isinstance(result, EvolutionResult):
            raise TypeError(
                f"{self.solver_name} engine must return EvolutionResult, "
                f"got {type(result).__name__}."
            )
        result.metadata.setdefault("backend", self.solver_name)
        result.metadata.setdefault("method", self.method)
        return result


@dataclass(frozen=True)
class PythonPackageSolverEngine:
    """Package-selected external solver boundary.

    This class performs package resolution and dependency checks.  The
    package-specific numerical implementation remains intentionally separate:
    it can be added without changing the public ``simulate_tn`` API.
    """

    solver_name: str
    method: str
    package_name: str
    package_spec: SolverPackageSpec

    def evolve(self, payload, initial_state, **kwargs) -> EvolutionResult:
        self.require_dependencies()
        raise NotImplementedError(
            f"backend='{self.solver_name}' selected Python package "
            f"{self.package_spec.package!r} for method {self.method!r}, and the "
            "package dependency is available, but the concrete ryd_gate adapter "
            "for this numerical kernel is not implemented yet. Either pass a "
            "custom `backend_options={'engine': ...}` object exposing "
            "`evolve(payload, initial_state, **kwargs)`, or implement the "
            f"{self.package_spec.package} kernel against this payload. "
            f"Payload keys: {sorted(payload.keys())}; docs: "
            f"{self.package_spec.docs_url}"
        )

    def require_dependencies(self) -> None:
        if _package_available(self.package_spec):
            return
        imports = " or ".join(self.package_spec.import_names)
        raise ExternalSolverDependencyError(
            f"backend='{self.solver_name}' with engine_package={self.package_name!r} "
            f"requires optional package {self.package_spec.package!r} "
            f"(import {imports}). Install it with "
            f"`pip install -e '.[{self.package_spec.extra}]'` or install the "
            "package manually. GPU support is not controlled by ryd_gate: "
            f"{self.package_spec.gpu_note}."
        )


class ExternalTTNTDVPBackend(ExternalSolverBackend):
    def __init__(
        self,
        engine: object | None = None,
        engine_package: str | None = None,
        **options,
    ) -> None:
        super().__init__(
            solver_name="ttn",
            method="ttn_tdvp",
            engine=engine,
            engine_package=engine_package,
            options=options,
        )

    def evolve_ir(
        self,
        ir,
        initial_state: str | np.ndarray | object = "all_ground",
        t_eval: np.ndarray | None = None,
        observables: list[str] | None = None,
    ) -> EvolutionResult:
        if self.engine is None:
            package_name = _canonical_package_name(
                self.engine_package or _DEFAULT_BACKEND_PACKAGE["ttn"]
            )
            if package_name == "pytreenet":
                from ryd_gate.backends.ttn.backend import PyTreeNetTTNTDVPBackend

                return PyTreeNetTTNTDVPBackend(**(self.options or {})).evolve_ir(
                    ir,
                    initial_state=initial_state,
                    t_eval=t_eval,
                    observables=observables,
                )
        return super().evolve_ir(
            ir,
            initial_state=initial_state,
            t_eval=t_eval,
            observables=observables,
        )


class External2DTNBPBackend(ExternalSolverBackend):
    def __init__(
        self,
        engine: object | None = None,
        engine_package: str | None = None,
        **options,
    ) -> None:
        super().__init__(
            solver_name="2dtn",
            method="2dtn_bp",
            engine=engine,
            engine_package=engine_package,
            options=options,
        )

    def evolve_ir(
        self,
        ir,
        initial_state: str | np.ndarray | object = "all_ground",
        t_eval: np.ndarray | None = None,
        observables: list[str] | None = None,
    ) -> EvolutionResult:
        if self.engine is None:
            package_name = _canonical_package_name(
                self.engine_package or _DEFAULT_BACKEND_PACKAGE["2dtn"]
            )
            if package_name == "yastn":
                from ryd_gate.backends.peps2d.yastn_backend import YASTN2DTNBackend

                return YASTN2DTNBackend(**(self.options or {})).evolve_ir(
                    ir,
                    initial_state=initial_state,
                    t_eval=t_eval,
                    observables=observables,
                )
            if package_name == "quimb":
                from ryd_gate.backends.peps2d.quimb_backend import Quimb2DTNBackend

                return Quimb2DTNBackend(**(self.options or {})).evolve_ir(
                    ir,
                    initial_state=initial_state,
                    t_eval=t_eval,
                    observables=observables,
                )
        return super().evolve_ir(
            ir,
            initial_state=initial_state,
            t_eval=t_eval,
            observables=observables,
        )


class ExternalNQSTVMCBackend(ExternalSolverBackend):
    def __init__(
        self,
        engine: object | None = None,
        engine_package: str | None = None,
        **options,
    ) -> None:
        super().__init__(
            solver_name="nqs",
            method="nqs_tvmc",
            engine=engine,
            engine_package=engine_package,
            options=options,
        )

    def evolve_ir(
        self,
        ir,
        initial_state: str | np.ndarray | object = "all_ground",
        t_eval: np.ndarray | None = None,
        observables: list[str] | None = None,
    ) -> EvolutionResult:
        if self.engine is None:
            package_name = _canonical_package_name(
                self.engine_package or _DEFAULT_BACKEND_PACKAGE["nqs"]
            )
            if package_name == "netket":
                from ryd_gate.backends.nqs.netket_backend import NetKetNQSTVMCBackend

                return NetKetNQSTVMCBackend(**(self.options or {})).evolve_ir(
                    ir,
                    initial_state=initial_state,
                    t_eval=t_eval,
                    observables=observables,
                )
        return super().evolve_ir(
            ir,
            initial_state=initial_state,
            t_eval=t_eval,
            observables=observables,
        )


def _package_engine_for_backend(
    solver_name: str,
    engine_package: str | None,
) -> PythonPackageSolverEngine:
    backend = _canonical_backend_name(solver_name)
    package_name = _canonical_package_name(
        engine_package or _DEFAULT_BACKEND_PACKAGE[backend]
    )
    allowed = _BACKEND_PACKAGES[backend]
    if package_name not in allowed:
        choices = ", ".join(allowed)
        raise ValueError(
            f"backend={backend!r} does not support engine_package={engine_package!r}. "
            f"Use one of: {choices}."
        )
    return PythonPackageSolverEngine(
        solver_name=backend,
        method={
            "ttn": "ttn_tdvp",
            "2dtn": "2dtn_bp",
            "nqs": "nqs_tvmc",
        }[backend],
        package_name=package_name,
        package_spec=_PACKAGE_SPECS[package_name],
    )


def _canonical_backend_name(backend: str) -> str:
    key = backend.lower()
    if key in {"2dtn", "2dtn_bp", "peps", "peps_bp"}:
        return "2dtn"
    if key in {"ttn", "ttn_tdvp"}:
        return "ttn"
    if key in {"nqs", "nqs_tvmc", "tvmc"}:
        return "nqs"
    raise ValueError("backend must be '2dtn', 'ttn', or 'nqs'.")


def _canonical_package_name(package_name: str) -> str:
    key = str(package_name).strip()
    if key in _PACKAGE_ALIASES:
        return _PACKAGE_ALIASES[key]
    key = key.lower()
    if key in _PACKAGE_ALIASES:
        return _PACKAGE_ALIASES[key]
    raise ValueError(
        f"Unknown engine_package={package_name!r}. Supported packages: "
        f"{', '.join(sorted(_PACKAGE_SPECS))}."
    )


def _package_available(spec: SolverPackageSpec) -> bool:
    return any(importlib.util.find_spec(name) is not None for name in spec.import_names)
