"""The generated capability matrix is current and derived from code."""

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
GENERATOR = REPO / "docs" / "_scripts" / "build_capability_matrix.py"
MATRIX = REPO / "docs" / "capability_matrix.md"


@pytest.fixture(scope="module")
def generator():
    spec = importlib.util.spec_from_file_location("build_capability_matrix", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def matrix_text(generator):
    return generator.build_matrix()


class TestMatrixFreshness:
    def test_checked_in_matrix_is_not_stale(self, matrix_text):
        assert MATRIX.read_text() == matrix_text, (
            "docs/capability_matrix.md is stale; regenerate with "
            "`uv run python docs/_scripts/build_capability_matrix.py`."
        )


class TestMatrixCoverage:
    def test_all_level_structures_and_backends_present(self, generator, matrix_text):
        from ryd_gate import level_structure

        for name in generator.PRESETS:
            level_structure(name)  # every documented preset must exist
            assert f"`{name}`" in matrix_text
        for backend in generator.BACKENDS:
            assert backend in matrix_text


class TestMatrixDerivation:
    def test_noise_rows_come_from_validate_for(self, generator, monkeypatch):
        """Forcing validate_for to reject everything must flip the noise rows."""
        from ryd_gate import NoiseModel
        from ryd_gate.core.serialization import ValidationIssue

        def reject(self, *, backend, level_structure=None, n_atoms=None):
            return [ValidationIssue("error", "noise.backend_unsupported", "forced", ())]

        monkeypatch.setattr(NoiseModel, "validate_for", reject)
        lines = generator._noise_table()
        body = [line for line in lines[2:]]
        assert body, "noise table must have rows"
        for row in body:
            assert generator.YES not in row
