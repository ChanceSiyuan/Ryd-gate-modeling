"""TeNPy local site + Rydberg lattice model built from :class:`TNTerms`.

The site is generic over the ``1r`` / ``01r`` level order: it carries a projector
``n_<level>`` per level, the Rydberg projector ``n_R``, and every transition
``E_<ket>_<bra> = |ket><bra|``.  The register site order IS the MPS chain order
(site ``i`` in the register is MPS position ``i``); the Rydberg pair interaction
enters as (generally long-range) ``V_ij n_R_i n_R_j`` coupling terms, so no snake
permutation is needed for correctness.  A time-dependent Hamiltonian is realized
by rebuilding this piecewise-constant model at each TDVP step's midpoint.
"""

from __future__ import annotations

import numpy as np


def _require_tenpy():
    try:
        import tenpy
        return tenpy
    except ImportError as exc:  # pragma: no cover - env guard
        raise ImportError(
            "TeNPy is required for backend='mps'. Install via `pip install ryd-gate[tn]`."
        ) from exc


def op_name(ket: str, bra: str) -> str:
    """Valid TeNPy operator name for ``|ket><bra|`` (labels are 0/1/r)."""
    return f"E_{ket}_{bra}"


def build_tn_site(terms):
    """Build the generic TeNPy :class:`Site` for a TN level structure."""
    _require_tenpy()
    from tenpy.linalg import np_conserved as npc
    from tenpy.networks.site import Site

    levels = terms.levels
    d = terms.local_dim
    ops: dict[str, np.ndarray] = {}
    for ket in levels:
        for bra in levels:
            m = np.zeros((d, d), dtype=complex)
            m[levels.index(ket), levels.index(bra)] = 1.0
            ops[op_name(ket, bra)] = m
    for level in levels:
        p = np.zeros((d, d), dtype=complex)
        p[levels.index(level), levels.index(level)] = 1.0
        ops[f"n_{level}"] = p
    ops["n_R"] = terms.rydberg_projector()
    leg = npc.LegCharge.from_trivial(d)
    return Site(leg, state_labels=list(levels), sort_charge=False, **ops)


def build_mps_model(terms, site, t_mid: float):
    """Piecewise-constant TeNPy model for one TDVP step (coefficients at ``t_mid``)."""
    _require_tenpy()
    from tenpy.models.lattice import Chain
    from tenpy.models.model import CouplingMPOModel

    n = terms.n_sites
    levels = terms.levels
    h_local = terms.local_hamiltonians(float(t_mid))  # (N, d, d), Hermitian
    pairs = terms.pairs

    class RydbergTNModel(CouplingMPOModel):
        def init_lattice(self, model_params):
            return Chain(n, site, bc="open", bc_MPS="finite")

        def init_terms(self, model_params):
            for i in range(n):
                for a, ket in enumerate(levels):
                    for b, bra in enumerate(levels):
                        c = h_local[i, a, b]
                        if c != 0.0:
                            self.add_onsite_term(complex(c), i, op_name(ket, bra))
            for i, j, V in pairs:
                if V != 0.0:
                    self.add_coupling_term(float(V), i, j, "n_R", "n_R")

    return RydbergTNModel({})
