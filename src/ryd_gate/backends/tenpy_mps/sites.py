"""Local TeNPy site builders for TN simulations."""

from __future__ import annotations

import numpy as np

from ryd_gate.backends.tn_common.sites import require_transition, resolve_level_structure
from ryd_gate.core.level_structures import LevelStructureSpec


def _require_tenpy():
    try:
        import tenpy

        return tenpy
    except ImportError as exc:
        raise ImportError(
            "TeNPy is required for tensor network simulations. "
            "Install via: pip install physics-tenpy  "
            "or: pip install ryd-gate[tn]"
        ) from exc


def build_tenpy_site(level_structure: str | LevelStructureSpec) -> object:
    """Build the TeNPy local site for a shared :class:`LevelStructureSpec`."""
    _require_tenpy()
    spec = resolve_level_structure(level_structure)
    if spec.name == "1r":
        from tenpy.networks.site import SpinHalfSite

        return SpinHalfSite(conserve=None)
    return _build_explicit_site(spec)


def transition_x_op_name(spec: LevelStructureSpec, lower: str, upper: str) -> str:
    """Return the Hermitian transition operator name for a central transition."""
    transition = require_transition(spec, lower, upper)
    return f"X_{transition.name}"


def _build_explicit_site(spec: LevelStructureSpec) -> object:
    from tenpy.linalg import np_conserved as npc
    from tenpy.networks.site import Site

    dim = spec.local_dim
    leg = npc.LegCharge.from_trivial(dim)
    ops = {}

    for idx, level in enumerate(spec.levels):
        projector = np.zeros((dim, dim), dtype=float)
        projector[idx, idx] = 1.0
        ops[f"n_{level}"] = projector

    for transition in spec.transitions:
        lower = spec.index(transition.lower)
        upper = spec.index(transition.upper)
        x_op = np.zeros((dim, dim), dtype=complex)
        x_op[upper, lower] = 1.0
        x_op[lower, upper] = 1.0
        ops[f"X_{transition.name}"] = x_op

    return Site(leg, state_labels=list(spec.levels), sort_charge=False, **ops)
