"""Script-side system builders using the current from_lattice paradigm."""

from __future__ import annotations

import numpy as np

from ryd_gate import RydbergSystem
from ryd_gate.lattice import make_chain, make_geometry_from_coords


def make_analog_3_system(
    *,
    protocol=None,
    n_atoms: int = 2,
    distance_um: float = 3.0,
    positions=None,
    **physical_params,
) -> RydbergSystem:
    geometry = (
        make_geometry_from_coords(np.asarray(positions, dtype=float))
        if positions is not None
        else make_chain(n_atoms, spacing_um=distance_um)
    )
    return RydbergSystem.from_lattice(
        geometry,
        "ger",
        param_set="analog_3",
        protocol=protocol,
        **physical_params,
    )


def make_our_system(
    *,
    protocol=None,
    n_atoms: int = 2,
    distance_um: float = 3.0,
    **physical_params,
) -> RydbergSystem:
    return RydbergSystem.from_lattice(
        make_chain(n_atoms, spacing_um=distance_um),
        "rb87_7",
        param_set="our",
        protocol=protocol,
        **physical_params,
    )


def make_lukin_system(
    *,
    protocol=None,
    n_atoms: int = 2,
    distance_um: float = 3.0,
    **physical_params,
) -> RydbergSystem:
    return RydbergSystem.from_lattice(
        make_chain(n_atoms, spacing_um=distance_um),
        "rb87_7",
        param_set="lukin",
        protocol=protocol,
        **physical_params,
    )
