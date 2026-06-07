"""Lower unified Hamiltonian IR into TN evolution inputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ryd_gate.core.level_structures import LevelStructureSpec, level_structure
from ryd_gate.core.system import RydbergSystem
from ryd_gate.ir import HamiltonianIR, compile_hamiltonian_ir

from .lattice_spec import TNLatticeSpec, snake_order_mapping
from .sites import resolve_level_structure

SUPPORTED_TN_METHODS = frozenset({
    "tdvp",
    "mps_tdvp",
    "peps_yastn",
    "gputn_tebd",
    "pepskit_ipeps_su",
})


@dataclass(frozen=True)
class TNEvolutionIR:
    """Tensor-network input derived from unified Hamiltonian IR."""

    spec: TNLatticeSpec
    protocol: object
    params: dict
    method: str = "tdvp"
    metadata: dict | None = None
    hamiltonian: HamiltonianIR | None = None


@dataclass(frozen=True)
class TNCompiler:
    """Lower unified Hamiltonian IR into TN metadata."""

    method: str = "tdvp"

    def compile(
        self,
        system_or_ir: RydbergSystem | HamiltonianIR,
        params: dict | None = None,
    ) -> TNEvolutionIR:
        if self.method not in SUPPORTED_TN_METHODS:
            supported = ", ".join(sorted(SUPPORTED_TN_METHODS))
            raise ValueError(f"Unknown TN method {self.method!r}. Supported: {supported}.")

        hamiltonian = (
            system_or_ir
            if isinstance(system_or_ir, HamiltonianIR)
            else compile_hamiltonian_ir(_require_rydberg_system(system_or_ir), _require_params(params))
        )
        if hamiltonian.protocol is None or hamiltonian.params is None:
            raise ValueError("TN lowering requires HamiltonianIR.protocol and HamiltonianIR.params.")

        spec = tn_lattice_spec_from_hamiltonian_ir(hamiltonian)
        metadata = dict(hamiltonian.metadata)
        metadata.update(
            {
                "compiler": "tn",
                "source_compiler": hamiltonian.metadata.get("compiler", "unknown"),
                "tn_spec": spec,
                "param_set": hamiltonian.metadata.get("param_set"),
                "level_structure": spec.level_structure,
                "n_sites": spec.N,
                "local_dim": spec.level_spec.local_dim,
            }
        )
        return TNEvolutionIR(
            spec=spec,
            protocol=hamiltonian.protocol,
            params=hamiltonian.params,
            method=self.method,
            metadata=metadata,
            hamiltonian=hamiltonian,
        )


def tn_lattice_spec_from_system(system: RydbergSystem) -> TNLatticeSpec:
    """Create a TN lattice spec from a core system without compiling matrices."""
    geometry = system.geometry
    if geometry is None:
        raise ValueError("TN lowering requires system.geometry.")

    level_spec = _system_level_spec(system)
    return _tn_lattice_spec_from_geometry(
        geometry,
        level_spec,
        tuple(system.meta("interaction_pairs", ())),
        system.meta("Omega", 1.0),
    )


def tn_lattice_spec_from_hamiltonian_ir(ir: HamiltonianIR) -> TNLatticeSpec:
    """Create a TN lattice spec from the unified Hamiltonian IR."""
    if ir.geometry is None:
        raise ValueError("TN lowering requires HamiltonianIR.geometry.")
    level_spec = ir.level_spec
    if not isinstance(level_spec, LevelStructureSpec):
        level_spec = resolve_level_structure(level_structure(ir.metadata.get("level_structure", "1r")))
    interaction_pairs = tuple(ir.metadata.get("interaction_pairs", ()))
    omega = ir.metadata.get("Omega", 1.0)
    if omega is None:
        omega = 1.0
    return _tn_lattice_spec_from_geometry(
        ir.geometry,
        resolve_level_structure(level_spec),
        interaction_pairs,
        omega,
    )


def _tn_lattice_spec_from_geometry(
    geometry,
    level_spec: LevelStructureSpec,
    interaction_pairs: tuple,
    omega: float,
) -> TNLatticeSpec:
    Lx, Ly = _infer_square_lattice_shape(np.asarray(geometry.coords, dtype=float), geometry.N)
    snake_to_2d, inv_snake = snake_order_mapping(Lx, Ly)
    return TNLatticeSpec(
        Lx=Lx,
        Ly=Ly,
        N=geometry.N,
        coords=np.asarray(geometry.coords, dtype=float),
        sublattice=np.asarray(geometry.sublattice),
        vdw_pairs=interaction_pairs,
        V_nn=1.0,
        Omega=omega,
        level_spec=level_spec,
        snake_to_2d=snake_to_2d,
        inv_snake=inv_snake,
        bc="open",
        interaction_mode="system",
    )


def _system_level_spec(system: RydbergSystem) -> LevelStructureSpec:
    spec = system.meta("level_spec", None)
    if isinstance(spec, LevelStructureSpec):
        return resolve_level_structure(spec)
    return resolve_level_structure(level_structure(system.meta("level_structure", "1r")))


def _require_rydberg_system(system) -> RydbergSystem:
    if not isinstance(system, RydbergSystem):
        raise TypeError("TNCompiler.compile() requires RydbergSystem or HamiltonianIR.")
    if system.geometry is None:
        raise ValueError("TNCompiler requires lattice geometry.")
    return system


def _require_params(params: dict | None) -> dict:
    if params is None:
        raise TypeError("TNCompiler.compile() requires params when given a RydbergSystem.")
    return params


def _infer_square_lattice_shape(coords: np.ndarray, n_sites: int) -> tuple[int, int]:
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError("TN lowering currently requires 2D lattice coordinates.")
    x_vals = _unique_axis_values(coords[:, 0])
    y_vals = _unique_axis_values(coords[:, 1])
    Lx, Ly = len(x_vals), len(y_vals)
    if Lx * Ly != n_sites:
        raise ValueError(
            "TN lowering currently supports rectangular square-lattice geometries; "
            f"could not infer Lx*Ly={n_sites} from coordinates."
        )
    return Lx, Ly


def _unique_axis_values(values: np.ndarray) -> np.ndarray:
    scale = max(1.0, float(np.max(np.abs(values))) if values.size else 1.0)
    rounded = np.round(values / (scale * 1e-12)).astype(np.int64)
    _, idx = np.unique(rounded, return_index=True)
    return np.sort(values[np.sort(idx)])
