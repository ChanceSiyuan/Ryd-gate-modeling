"""MPS measurement utilities for tensor network simulations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ryd_gate.backends.tn_common.lattice_spec import TNLatticeSpec
    from ryd_gate.core.observables import ObservableExpr


class MPSExpressionPlan:
    """Lowered ``ObservableExpr`` measurement plan for a TeNPy MPS.

    Built once per evolution by :func:`lower_expressions`; :meth:`measure`
    evaluates every labelled expression on the current state per sampling
    anchor: single-site factors are grouped by *distinct* local matrix — one
    bulk ``psi.expectation_value`` sweep per matrix covers all its sites (so a
    per-site population profile is a single call) — and multi-site products go
    through ``psi.expectation_value_multi_sites`` (identity-padded between
    factor sites).  Note TeNPy indexes a heterogeneous ops list by *site*
    (``ops[i % len(ops)]``), so mixed operators must never share one call.
    Values are raw complex ``<psi|O|psi>`` — the canonical-form TeNPy
    expectations times ``psi.norm**2``, never divided by the norm.
    """

    def __init__(self, lowered: dict, groups: dict) -> None:
        self._lowered = lowered
        self._groups = groups  # matrix-bytes key -> (npc op, sorted site positions)

    def measure(self, psi: object) -> dict[str, complex]:
        scale = complex(psi.norm) ** 2
        site_values: dict[bytes, dict[int, complex]] = {}
        for key, (op, positions) in self._groups.items():
            values = np.asarray(psi.expectation_value([op], sites=positions))
            site_values[key] = {
                pos: complex(value) for pos, value in zip(positions, values)
            }
        out: dict[str, complex] = {}
        for label, (identity_coeff, singles, multis) in self._lowered.items():
            value = identity_coeff
            for coeff, key, pos in singles:
                value += coeff * site_values[key][pos]
            for coeff, i0, ops_list in multis:
                value += coeff * complex(psi.expectation_value_multi_sites(ops_list, i0))
            out[label] = scale * value
        return out


def lower_expressions(
    exprs: "dict[str, ObservableExpr]", spec: TNLatticeSpec
) -> MPSExpressionPlan:
    """Lower validated observable expressions onto the TeNPy MPS geometry.

    Register site indices map to 1D MPS positions through ``spec.inv_snake``
    (the same permutation the Hamiltonian builder uses).  Local matrices are
    permuted from the shared level order (``spec.level_spec.levels``) into the
    TeNPy site basis — for the ``1r`` ``SpinHalfSite`` that is the
    ``|up>=|r>, |down>=|1>`` order, so arbitrary 2x2 matrices work without
    touching the site's named operators — and converted to npc arrays once.
    """
    from tenpy.linalg import np_conserved as npc

    from .model import build_tenpy_site
    from .state import _tenpy_label

    site = build_tenpy_site(spec.level_spec)
    perm = [
        site.state_index(_tenpy_label(spec, level)) for level in spec.level_spec.levels
    ]
    inv_snake = np.asarray(spec.inv_snake, dtype=int)
    d = int(spec.level_spec.local_dim)

    npc_cache: dict[bytes, object] = {}

    def to_npc(matrix: np.ndarray) -> tuple[bytes, object]:
        permuted = np.zeros((d, d), dtype=complex)
        permuted[np.ix_(perm, perm)] = np.asarray(matrix, dtype=complex)
        key = permuted.tobytes()
        op = npc_cache.get(key)
        if op is None:
            op = npc.Array.from_ndarray_trivial(permuted, dtype=complex, labels=["p", "p*"])
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
                site_2d, matrix = term.factors[0]
                key, _ = to_npc(matrix)
                pos = int(inv_snake[site_2d])
                group_sites.setdefault(key, set()).add(pos)
                singles.append((complex(term.coefficient), key, pos))
            else:
                positioned = sorted(
                    ((int(inv_snake[site_2d]), matrix) for site_2d, matrix in term.factors),
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
    groups = {
        key: (npc_cache[key], sorted(positions))
        for key, positions in group_sites.items()
    }
    return MPSExpressionPlan(lowered, groups)


def measure_site_occupations(
    psi: object,
    spec: TNLatticeSpec,
) -> np.ndarray:
    """Measure Rydberg occupation ``<n_r_i>`` in 2D site order.

    Parameters
    ----------
    psi : tenpy.networks.mps.MPS
        MPS state (in snake order).
    spec : TNLatticeSpec

    Returns
    -------
    occ : ndarray, shape (N,)
        Per-site Rydberg occupation in 2D site order.
    """
    if spec.level_structure in {"01r", "analog_3"}:
        occ_snake = np.array(psi.expectation_value("n_r"))
    else:
        sz_vals = psi.expectation_value("Sz")
        occ_snake = 0.5 + np.array(sz_vals)
    # Reorder from snake to 2D
    occ_2d = np.empty(spec.N)
    occ_2d[spec.snake_to_2d] = occ_snake
    return occ_2d


def measure_level_occupations(
    psi: object,
    spec: TNLatticeSpec,
    level: str,
) -> np.ndarray:
    """Measure ``<n_level_i>`` in 2D site order."""
    if level == "r":
        return measure_site_occupations(psi, spec)

    if spec.level_structure == "1r":
        if level == "1":
            return 1.0 - measure_site_occupations(psi, spec)
        if level == "0":
            return np.zeros(spec.N)
        raise ValueError(f"Unknown level for 1r TN lattice: {level!r}")

    if spec.level_structure == "analog_3":
        if level not in {"g", "e"}:
            raise ValueError(f"Unknown level for analog_3 TN lattice: {level!r}")
    elif level not in {"0", "1"}:
        raise ValueError(f"Unknown level for 01r TN lattice: {level!r}")
    occ_snake = np.array(psi.expectation_value(f"n_{level}"))
    occ_2d = np.empty(spec.N)
    occ_2d[spec.snake_to_2d] = occ_snake
    return occ_2d


def measure_staggered_magnetization(
    psi: object,
    spec: TNLatticeSpec,
) -> float:
    """Compute m_s = (1/N) sum_i s_i (2<n_i> - 1).

    Parameters
    ----------
    psi : tenpy.networks.mps.MPS
    spec : TNLatticeSpec

    Returns
    -------
    m_s : float
    """
    occ = measure_site_occupations(psi, spec)
    return float(np.sum(spec.sublattice * (2 * occ - 1)) / spec.N)


def measure_mean_rydberg(
    psi: object,
    spec: TNLatticeSpec,
) -> float:
    """Compute mean Rydberg fraction <n> = (1/N) sum_i <n_i>.

    Parameters
    ----------
    psi : tenpy.networks.mps.MPS
    spec : TNLatticeSpec

    Returns
    -------
    n_mean : float
    """
    occ = measure_site_occupations(psi, spec)
    return float(occ.mean())


def measure_sigma_z(
    psi: object,
    spec: TNLatticeSpec,
) -> np.ndarray:
    """Measure per-site ``<sigma_z_i> = 2<n_r_i> - 1`` in 2D site order."""
    return 2.0 * measure_site_occupations(psi, spec) - 1.0


def measure_correlation(
    psi: object,
    spec: TNLatticeSpec,
    i_2d: int,
    j_2d: int,
) -> float:
    """Compute connected correlator <n_i n_j> - <n_i><n_j>.

    Parameters
    ----------
    psi : tenpy.networks.mps.MPS
    spec : TNLatticeSpec
    i_2d, j_2d : int
        Site indices in 2D site order.

    Returns
    -------
    corr : float
    """
    i_1d = int(spec.inv_snake[i_2d])
    j_1d = int(spec.inv_snake[j_2d])
    if i_1d > j_1d:
        i_1d, j_1d = j_1d, i_1d

    # n_i = 0.5 + Sz_i  =>  n_i n_j = (0.5+Szi)(0.5+Szj)
    if spec.level_structure in {"01r", "analog_3"}:
        n_vals = psi.expectation_value("n_r")
        n_i = float(n_vals[i_1d])
        n_j = float(n_vals[j_1d])
        nn = float(psi.expectation_value_term([("n_r", i_1d), ("n_r", j_1d)]))
    else:
        sz_i = float(psi.expectation_value("Sz")[i_1d])
        sz_j = float(psi.expectation_value("Sz")[j_1d])
        n_i = 0.5 + sz_i
        n_j = 0.5 + sz_j
        szsz = psi.expectation_value_term([("Sz", i_1d), ("Sz", j_1d)])
        nn = float(szsz) + 0.5 * (sz_i + sz_j) + 0.25
    return float(nn - n_i * n_j)


def measure_connected_zz(
    psi: object,
    spec: TNLatticeSpec,
    i_2d: int,
    j_2d: int,
) -> float:
    """Compute connected ``<sigma_z_i sigma_z_j> - <sigma_z_i><sigma_z_j>``."""
    return 4.0 * measure_correlation(psi, spec, i_2d, j_2d)


def measure_centerline_connected_zz(
    psi: object,
    spec: TNLatticeSpec,
    *,
    axis: str = "horizontal",
) -> np.ndarray:
    """Measure connected ``C_zz`` from a center site along a center line."""
    from ryd_gate.backends.tn_common._geometry import line_pairs_from_reference

    return np.array([
        measure_connected_zz(psi, spec, i, j)
        for i, j in line_pairs_from_reference(spec.Lx, spec.Ly, axis=axis)
    ])
