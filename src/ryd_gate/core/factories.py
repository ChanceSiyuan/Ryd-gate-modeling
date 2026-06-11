"""Construction of :class:`RydbergSystem` instances from a lattice.

Holds the ``build_from_lattice`` factory (registers symbolic blocks and
observables for a geometry + level structure) plus the interaction-pair and
physical-model resolution helpers it relies on.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ryd_gate.core.basis import BasisSpec
from ryd_gate.core.blocks import BlockRegistry
from ryd_gate.core.interactions import vdw_couplings
from ryd_gate.core.level_structures import (
    DEFAULT_C6,
    InteractionSpec,
    LevelStructureSpec,
)
from ryd_gate.core.level_structures import level_structure as level_structure_preset
from ryd_gate.core.local_blocks import _apply_analog_3_lattice_blocks, _apply_rb87_7_lattice_blocks
from ryd_gate.core.observables import ObservableRegistry
from ryd_gate.core.operator_spec import (
    LocalProjectorSpec,
    RydbergPairInteractionSpec,
    SumProjectorSpec,
    TransitionOperatorSpec,
    WeightedProjectorSumSpec,
)
from ryd_gate.core.rb87_params import _rb87_default_c6
from ryd_gate.lattice.geometry import Register

if TYPE_CHECKING:
    from ryd_gate.protocols.base import Protocol


def build_from_lattice(
    cls,
    geometry: Register,
    level_structure: str | LevelStructureSpec = "1r",
    interaction: InteractionSpec | None = None,
    *,
    protocol: "Protocol | None" = None,
    param_set: str | None = None,
    Omega: float = 1.0,
    **physical_params,
):
    spec = (
        level_structure
        if isinstance(level_structure, LevelStructureSpec)
        else level_structure_preset(level_structure)
    )
    physical_model = _physical_model_for(spec.name, param_set)
    interaction = interaction or _default_interaction_for_physical_model(physical_model)
    d = spec.local_dim
    N = geometry.N
    dim = d**N
    basis = BasisSpec(
        site_labels=tuple(str(i) for i in range(N)),
        local_levels=spec.levels,
        local_dim=d,
        total_dim=dim,
    )

    blocks = BlockRegistry()
    observables = ObservableRegistry()

    for level in spec.levels:
        sum_spec = SumProjectorSpec(level)
        blocks.register(f"sum_n_{level}", sum_spec, description=f"total |{level}> occupation")
        for i in range(N):
            site_spec = LocalProjectorSpec(level, i)
            blocks.register(f"n_{level}_{i}", site_spec, description=f"|{level}><{level}| on site {i}")
            observables.register(
                f"n_{level}_{i}",
                site_spec,
                description=f"|{level}> population on site {i}",
                per_site=True,
            )
        observables.register(f"sum_n_{level}", sum_spec, description=f"total |{level}> population")

    pairs = _interaction_pairs(geometry, interaction)
    if spec.rydberg_levels:
        blocks.register(
            "H_vdw",
            RydbergPairInteractionSpec(pairs, spec.rydberg_levels),
            description="Rydberg VdW interaction",
        )

    for transition in spec.transitions:
        blocks.register(
            transition.channel,
            TransitionOperatorSpec(transition.lower, transition.upper),
            description=transition.name,
            hermitian=False,
        )
        for i in range(N):
            blocks.register(
                f"{transition.channel}_{i}",
                TransitionOperatorSpec(transition.lower, transition.upper, site=i),
                description=f"{transition.name} on site {i}",
                hermitian=False,
            )

    for channel, level in spec.detuning_levels.items():
        blocks.register(channel, SumProjectorSpec(level), description=f"detuning projector for |{level}>")

    if "r" in spec.levels:
        sum_r = SumProjectorSpec("r")
        blocks.register("sum_nr", sum_r, description="total Rydberg occupation")
        observables.register("sum_nr", sum_r, description="total Rydberg population")
    if spec.name == "1r":
        blocks.register("global_n", SumProjectorSpec("r"), description="2-level Rydberg occupation")

    if geometry.sublattice is not None and np.any(geometry.sublattice):
        if "r" in spec.levels:
            observables.register(
                "staggered_rydberg",
                WeightedProjectorSumSpec("r", tuple(float(x) for x in geometry.sublattice)),
                description="staggered Rydberg occupation",
            )

    metadata = {
        "level_structure": spec.name,
        "level_spec": spec,
        "interaction_pairs": pairs,
        "Omega": Omega,
        "local_dim": d,
        "n_sites": N,
    }
    model = cls(
        param_set=param_set or f"lattice_{spec.name}",
        basis=basis,
        blocks=blocks,
        observables=observables,
        protocol=protocol,
        metadata=metadata,
        geometry=geometry,
        is_sparse=True,
    )
    if physical_model == "analog_3":
        _apply_analog_3_lattice_blocks(model, **physical_params)
    elif physical_model in {"our", "lukin"}:
        _apply_rb87_7_lattice_blocks(model, physical_model, **physical_params)
    return model


def _interaction_pairs(geometry: Register, interaction: InteractionSpec) -> tuple:
    if interaction.mode == "all":
        return vdw_couplings(geometry.coords, interaction.C6, interaction.max_range_um)

    coords = np.asarray(geometry.coords, dtype=float)
    spacing = geometry.spacing_um or min(
        (d for _, _, d in geometry.distance_pairs()), default=0.0
    )
    max_dist = spacing * (1.01 if interaction.mode == "nn" else np.sqrt(2) * 1.01)
    max_range = interaction.max_range_um if interaction.max_range_um is not None else max_dist
    return vdw_couplings(coords, interaction.C6, max_range)


def _physical_model_for(level_name: str, param_set: str | None) -> str | None:
    # Preset *names* carry Hamiltonian semantics (stageplans/README D11):
    # bare `ger` is always symbolic; physical analog-3 construction is the
    # `analog_3` preset only.
    if level_name == "analog_3":
        return "analog_3"
    if level_name == "rb87_7" and param_set in {"our", "lukin"}:
        return param_set
    return None


def _default_interaction_for_physical_model(physical_model: str | None) -> InteractionSpec:
    if physical_model == "lukin":
        return InteractionSpec(C6=_rb87_default_c6("lukin"))
    return InteractionSpec(C6=DEFAULT_C6)
