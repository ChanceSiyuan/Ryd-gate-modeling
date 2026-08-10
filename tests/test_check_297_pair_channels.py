"""Focused tests for scripts/check_297_pair_channels.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix, diags

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "check_297_pair_channels", ROOT / "scripts" / "check_297_pair_channels.py"
)
assert SPEC is not None and SPEC.loader is not None
pair = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pair
SPEC.loader.exec_module(pair)


def test_assemble_pair_hamiltonian_applies_arc_distance_powers():
    """Catch treating ARC's metre-scaled matR entries as already evaluated."""
    spacing_um = 2.0
    distance_m = spacing_um * 1e-6
    calc = SimpleNamespace(
        matDiagonal=diags([0.1, 0.2], format="csr"),
        matR=[
            csr_matrix(
                np.array([[0.0, 0.05], [0.05, 0.0]]) * distance_m**3
            )
        ],
    )

    actual = pair.assemble_pair_hamiltonian(calc, spacing_um).toarray()

    np.testing.assert_allclose(actual, [[0.1, 0.05], [0.05, 0.2]])


def test_find_basis_state_index_requires_one_exact_pair_state():
    """Catch silently using index zero when the requested ARC state is absent."""
    states = [[53, 1, 1.5, -1.5, 53, 1, 1.5, -1.5]]
    target = (53, 1, 1.5, -1.5, 53, 1, 1.5, -1.5)

    assert pair.find_basis_state_index(states, target) == 0
    with pytest.raises(ValueError, match="exactly one"):
        pair.find_basis_state_index(states, (*target[:-1], -0.5))


def test_extract_local_eigenpairs_expands_until_weak_window_is_bracketed():
    """Catch stopping after a fixed k that leaves part of the weak window out."""
    values = np.array(
        [-0.20, -0.12, -0.06, -0.01, 0.02, 0.07, 0.11, 0.15, 0.25, 0.40]
    )

    eigenvalues, eigenvectors, meta = pair.extract_local_eigenpairs(
        diags(values, format="csr"),
        reference_ghz=0.005,
        bare_index=3,
        weak_threshold_mhz=80.0,
        initial_k=4,
        max_k=8,
        capture_target=0.99,
    )

    shifts_mhz = (eigenvalues - 0.005) * 1e3
    assert shifts_mhz.min() < -80.0
    assert shifts_mhz.max() > 80.0
    assert meta["window_bracketed"] is True
    assert meta["eigenpairs"] == 8
    assert eigenvectors.shape == (10, 8)


def test_extract_local_eigenpairs_rejects_unbracketed_window():
    """Catch reporting a partial weak weight as though the energy window were complete."""
    values = np.linspace(-0.2, 0.2, 10)

    with pytest.raises(RuntimeError, match="did not bracket"):
        pair.extract_local_eigenpairs(
            diags(values, format="csr"),
            reference_ghz=0.0,
            bare_index=4,
            weak_threshold_mhz=150.0,
            initial_k=2,
            max_k=4,
            capture_target=0.99,
        )


def test_summarize_eigenpairs_uses_channel_reference_and_overlap_weights():
    """Catch using absolute ARC energies or amplitudes instead of squared overlaps."""
    eigenvalues = np.array([-0.05, 0.02, 0.12])
    eigenvectors = np.array(
        [
            [0.5, np.sqrt(0.75), 0.0],
            [np.sqrt(0.75), -0.5, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    basis_states = [
        [53, 1, 1.5, -1.5, 53, 1, 1.5, -1.5],
        [53, 1, 1.5, -0.5, 53, 1, 1.5, -0.5],
        [53, 0, 0.5, -0.5, 54, 0, 0.5, 0.5],
    ]

    summary = pair.summarize_eigenpairs(
        eigenvalues,
        eigenvectors,
        reference_ghz=0.0,
        bare_index=0,
        basis_states=basis_states,
        target_manifold_indices=[0, 1],
        weak_threshold_mhz=80.0,
        report_overlap_cutoff=0.01,
    )

    assert summary["weak_shift_weight"] == pytest.approx(1.0)
    assert summary["captured_overlap"] == pytest.approx(1.0)
    assert summary["states"][0]["overlap"] == pytest.approx(0.75)
    assert summary["states"][0]["shift_mhz"] == pytest.approx(20.0)
