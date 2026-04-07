"""Utility operators for two-atom Rydberg gate models.

Provides functions to build occupation operators, state maps,
van der Waals operators, and atom projectors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


# ======================================================================
# UTILITY FUNCTIONS
# ======================================================================


def build_occ_operator(index: int, n_levels: int = 7) -> "NDArray[np.complexfloating]":
    """Build occupation number operator for level `index`.

    Creates |i><i| x I + I x |i><i| for measuring total population
    in level i across both atoms.

    Parameters
    ----------
    index : int
        Single-atom level index.
    n_levels : int
        Number of single-atom levels (default 7).
    """
    oper_sq = np.zeros((n_levels, n_levels), dtype=np.complex128)
    oper_sq[index, index] = 1
    oper_tq = np.kron(np.eye(n_levels), oper_sq) + np.kron(oper_sq, np.eye(n_levels))
    return oper_tq


def build_sss_state_map(n_levels: int = 7) -> "dict[str, NDArray[np.complexfloating]]":
    """Build mapping from qubit state labels to two-atom state vectors.

    For the 7-level system, qubit states use levels 0 (|g⟩) and 1 (|1⟩),
    plus SSS (symmetric subspace) superpositions used for gate characterization.

    Parameters
    ----------
    n_levels : int
        Number of single-atom levels. Must be >= 7.

    Raises
    ------
    ValueError
        If n_levels < 7. For 3-level systems, use
        :func:`build_product_state_map` instead.
    """
    if n_levels < 7:
        raise ValueError(
            f"build_sss_state_map requires n_levels >= 7, got {n_levels}. "
            "For 3-level systems, use build_product_state_map(n_levels=3)."
        )
    s0 = np.zeros(n_levels, dtype=complex)
    s0[0] = 1.0
    s1 = np.zeros(n_levels, dtype=complex)
    s1[1] = 1.0
    state_00 = np.kron(s0, s0)
    state_01 = np.kron(s0, s1)
    state_10 = np.kron(s1, s0)
    state_11 = np.kron(s1, s1)
    return {
        "00": state_00,
        "01": state_01,
        "10": state_10,
        "11": state_11,
        "SSS-0": 0.5 * state_00 + 0.5 * state_01 + 0.5 * state_10 + 0.5 * state_11,
        "SSS-1": 0.5 * state_00 - 0.5 * state_01 - 0.5 * state_10 + 0.5 * state_11,
        "SSS-2": 0.5 * state_00 + 0.5j * state_01 + 0.5j * state_10 - 0.5 * state_11,
        "SSS-3": 0.5 * state_00 - 0.5j * state_01 - 0.5j * state_10 - 0.5 * state_11,
        "SSS-4": state_00,
        "SSS-5": state_11,
        "SSS-6": 0.5 * state_00 + 0.5 * state_01 + 0.5 * state_10 - 0.5 * state_11,
        "SSS-7": 0.5 * state_00 - 0.5 * state_01 - 0.5 * state_10 - 0.5 * state_11,
        "SSS-8": 0.5 * state_00 + 0.5j * state_01 + 0.5j * state_10 + 0.5 * state_11,
        "SSS-9": 0.5 * state_00 - 0.5j * state_01 - 0.5j * state_10 + 0.5 * state_11,
        "SSS-10": state_00 / np.sqrt(2) + 1j * state_11 / np.sqrt(2),
        "SSS-11": state_00 / np.sqrt(2) - 1j * state_11 / np.sqrt(2),
    }


def build_product_state_map(
    n_levels: int = 3,
    level_labels: tuple[str, ...] | None = None,
) -> "dict[str, NDArray[np.complexfloating]]":
    """Build mapping from product-state labels to two-atom state vectors.

    Generates all ``n_levels^2`` product states ``|i,j⟩`` for a
    two-atom system and labels them using the provided level names.

    Parameters
    ----------
    n_levels : int
        Number of single-atom levels (default 3).
    level_labels : tuple of str or None
        Human-readable names for each level.  If None, defaults to
        ``("g", "e", "r")`` for n_levels=3 or ``("0", "1", ...)``
        otherwise.

    Returns
    -------
    dict mapping "AB" labels to state vectors of dimension n_levels^2.

    Examples
    --------
    >>> m = build_product_state_map(n_levels=3)
    >>> m["gg"]   # |g,g⟩
    >>> m["gr"]   # |g,r⟩
    >>> m["rr"]   # |r,r⟩
    """
    if level_labels is None:
        if n_levels == 3:
            level_labels = ("g", "e", "r")
        else:
            level_labels = tuple(str(i) for i in range(n_levels))

    states: dict[str, NDArray[np.complexfloating]] = {}
    for i in range(n_levels):
        si = np.zeros(n_levels, dtype=complex)
        si[i] = 1.0
        for j in range(n_levels):
            sj = np.zeros(n_levels, dtype=complex)
            sj[j] = 1.0
            label = level_labels[i] + level_labels[j]
            states[label] = np.kron(si, sj)
    return states


def build_vdw_unit_operator(
    rydberg_indices: tuple[int, ...] = (5, 6), n_levels: int = 7,
) -> "NDArray[np.complexfloating]":
    """Build the unit van der Waals interaction operator."""
    ryd_proj = np.zeros((n_levels, n_levels), dtype=np.complex128)
    for i in rydberg_indices:
        ryd_proj[i, i] = 1.0
    return np.kron(ryd_proj, ryd_proj)


def build_atom_projector(
    atom_idx: int, level: int, n_atoms: int, n_levels: int,
) -> "NDArray[np.complexfloating]":
    """Build |level><level| projector on atom ``atom_idx`` in an N-atom system.

    Returns an ``n_levels^n_atoms × n_levels^n_atoms`` matrix that is
    the identity on all atoms except ``atom_idx``, where it projects
    onto level ``level``.

    Parameters
    ----------
    atom_idx : int
        Which atom (0-indexed) the projector acts on.
    level : int
        Single-atom level index to project onto.
    n_atoms : int
        Total number of atoms.
    n_levels : int
        Number of single-atom levels.
    """
    sq = np.zeros((n_levels, n_levels), dtype=np.complex128)
    sq[level, level] = 1.0
    op = np.eye(1, dtype=np.complex128)
    for k in range(n_atoms):
        op = np.kron(op, sq if k == atom_idx else np.eye(n_levels, dtype=np.complex128))
    return op


def build_atom_a_projector(index: int, n_levels: int = 7) -> "NDArray[np.complexfloating]":
    """Build |i><i| x I -- projects Atom A (left) onto level index."""
    sq = np.zeros((n_levels, n_levels), dtype=np.complex128)
    sq[index, index] = 1
    return np.kron(sq, np.eye(n_levels, dtype=np.complex128))


def build_atom_b_projector(index: int, n_levels: int = 7) -> "NDArray[np.complexfloating]":
    """Build I x |i><i| -- projects Atom B (right) onto level index."""
    sq = np.zeros((n_levels, n_levels), dtype=np.complex128)
    sq[index, index] = 1
    return np.kron(np.eye(n_levels, dtype=np.complex128), sq)


def get_nominal_distance(param_set: str) -> float:
    """Get the nominal interatomic distance in um."""
    return 3.0
