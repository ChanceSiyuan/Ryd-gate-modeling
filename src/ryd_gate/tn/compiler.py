"""Compiler from unified :class:`RydbergSystem` objects to TN evolution IR."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ryd_gate.core.rydberg_system import LevelStructureSpec, RydbergSystem, level_structure

from .lattice_spec import TNLatticeSpec, snake_order_mapping
from .sites import resolve_level_structure

SUPPORTED_TN_METHODS = frozenset({
    "tdvp",
    "mps_tdvp",
    "itensors_tebd",
    "ttn_tdvp",
    "2dtn_bp",
    "peps_bp",
    "nqs_tvmc",
})


@dataclass(frozen=True)
class TNEvolutionIR:
    """Tensor-network evolution IR produced from a unified RydbergSystem."""

    spec: TNLatticeSpec
    protocol: object
    params: dict
    method: str = "tdvp"
    metadata: dict | None = None


@dataclass(frozen=True)
class TNCompiler:
    """Lower a protocol-bound :class:`RydbergSystem` into TN metadata.

    The compiler reuses the same central ``LevelStructureSpec`` and symbolic
    interaction pairs used by the exact sparse compiler; only the final
    representation differs.
    """

    method: str = "tdvp"

    def compile(self, system: RydbergSystem, params: dict) -> TNEvolutionIR:
        if not isinstance(system, RydbergSystem):
            raise TypeError("TNCompiler.compile() requires a RydbergSystem.")
        if system.geometry is None:
            raise ValueError("TNCompiler requires a lattice RydbergSystem with geometry.")
        if self.method not in SUPPORTED_TN_METHODS:
            supported = ", ".join(sorted(SUPPORTED_TN_METHODS))
            raise ValueError(f"Unknown TN method {self.method!r}. Supported: {supported}.")

        spec = tn_lattice_spec_from_system(system)
        return TNEvolutionIR(
            spec=spec,
            protocol=system._require_protocol(),
            params=params,
            method=self.method,
            metadata={
                "compiler": "tn",
                "tn_spec": spec,
                "param_set": system.param_set,
                "level_structure": spec.level_structure,
                "n_sites": spec.N,
                "local_dim": spec.level_spec.local_dim,
            },
        )


def tn_lattice_spec_from_system(system: RydbergSystem) -> TNLatticeSpec:
    """Create a TN lattice spec from the unified system geometry and interactions."""
    geometry = system.geometry
    if geometry is None:
        raise ValueError("TN lowering requires system.geometry.")

    Lx, Ly = _infer_square_lattice_shape(np.asarray(geometry.coords, dtype=float), geometry.N)
    snake_to_2d, inv_snake = snake_order_mapping(Lx, Ly)
    level_spec = _system_level_spec(system)

    # Use exact-side couplings directly. Setting V_nn=1 makes vdw_pairs carry
    # the physical pair strength instead of a relative NN-normalized value.
    interaction_pairs = tuple(system.meta("interaction_pairs", ()))

    return TNLatticeSpec(
        Lx=Lx,
        Ly=Ly,
        N=geometry.N,
        coords=np.asarray(geometry.coords, dtype=float),
        sublattice=np.asarray(geometry.sublattice),
        vdw_pairs=interaction_pairs,
        V_nn=1.0,
        Omega=system.meta("Omega", 1.0),
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
