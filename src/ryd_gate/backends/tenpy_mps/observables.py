"""Lower ``ObservableExpr`` onto a TeNPy MPS and measure at sampling anchors.

Register site indices are MPS positions directly (identity ordering).  Local
factor matrices are already in the site's level order, so no permutation is
needed.  Single-site factors sharing one distinct matrix are measured in a
single bulk ``expectation_value`` sweep; multi-site products go through
``expectation_value_multi_sites`` (identity-padded between factor sites).  Values
are raw ``<psi|O|psi>`` (canonical expectation times ``psi.norm**2``).
"""

from __future__ import annotations

import numpy as np


class MPSObservablePlan:
    """A lowered measurement plan; :meth:`measure` evaluates it on a state."""

    def __init__(self, lowered: dict, groups: dict) -> None:
        self._lowered = lowered
        self._groups = groups  # matrix key -> (npc op, sorted positions)

    def measure(self, psi) -> dict[str, complex]:
        scale = complex(psi.norm) ** 2
        site_values: dict[bytes, dict[int, complex]] = {}
        for key, (op, positions) in self._groups.items():
            values = np.asarray(psi.expectation_value([op], sites=positions))
            site_values[key] = {pos: complex(v) for pos, v in zip(positions, values)}
        out: dict[str, complex] = {}
        for label, (identity_coeff, singles, multis) in self._lowered.items():
            value = identity_coeff
            for coeff, key, pos in singles:
                value += coeff * site_values[key][pos]
            for coeff, i0, ops_list in multis:
                value += coeff * complex(psi.expectation_value_multi_sites(ops_list, i0))
            out[label] = scale * value
        return out


def lower_observables(exprs: dict, terms) -> MPSObservablePlan:
    """Lower validated observable expressions onto the MPS geometry."""
    from tenpy.linalg import np_conserved as npc

    npc_cache: dict[bytes, object] = {}

    def to_npc(matrix: np.ndarray) -> tuple[bytes, object]:
        arr = np.asarray(matrix, dtype=complex)
        key = arr.tobytes()
        op = npc_cache.get(key)
        if op is None:
            op = npc.Array.from_ndarray_trivial(arr, dtype=complex, labels=["p", "p*"])
            npc_cache[key] = op
        return key, op

    group_sites: dict[bytes, set[int]] = {}
    lowered: dict = {}
    for label, expr in exprs.items():
        identity_coeff = 0.0 + 0.0j
        singles: list[tuple[complex, bytes, int]] = []
        multis: list[tuple[complex, int, list]] = []
        for term in expr._terms:
            if not term.factors:
                identity_coeff += complex(term.coefficient)
            elif len(term.factors) == 1:
                site_i, matrix = term.factors[0]
                key, _ = to_npc(matrix)
                group_sites.setdefault(key, set()).add(int(site_i))
                singles.append((complex(term.coefficient), key, int(site_i)))
            else:
                positioned = sorted(
                    ((int(site_i), matrix) for site_i, matrix in term.factors),
                    key=lambda pair: pair[0],
                )
                i0 = positioned[0][0]
                ops_list: list = []
                prev = i0 - 1
                for pos, matrix in positioned:
                    ops_list.extend(["Id"] * (pos - prev - 1))
                    ops_list.append(to_npc(matrix)[1])
                    prev = pos
                multis.append((complex(term.coefficient), i0, ops_list))
        lowered[label] = (identity_coeff, singles, multis)
    groups = {key: (npc_cache[key], sorted(positions)) for key, positions in group_sites.items()}
    return MPSObservablePlan(lowered, groups)
