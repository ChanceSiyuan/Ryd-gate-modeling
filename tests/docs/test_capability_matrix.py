"""The generated capability matrix is current and derived from code."""

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
GENERATOR = REPO / "docs" / "_scripts" / "build_capability_matrix.py"
MATRIX = REPO / "docs" / "capability_matrix.qmd"


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
            "docs/capability_matrix.qmd is stale; regenerate with "
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

    def test_removed_capabilities_are_not_documented(self, generator, matrix_text):
        """The '01' preset and the removed backend names are gone."""
        assert "01" not in generator.PRESETS
        assert "stabilizer" not in generator.BACKENDS
        assert "`01`" not in matrix_text
        assert "stabilizer" not in matrix_text
        assert "exact_dense" not in matrix_text
        assert "exact_sparse" not in matrix_text

    def test_backend_columns_are_simulate_names(self, generator):
        """Columns carry the exact simulate(..., backend=...) strings."""
        assert generator.BACKENDS == ("exact_ode", "mps", "peps")

    def test_presets_come_from_the_registry(self, generator):
        from ryd_gate.core.level_structures import _STRUCTURAL_PRESETS

        assert generator.PRESETS == tuple(_STRUCTURAL_PRESETS)


class TestMatrixDerivation:
    def test_noise_rows_are_exact_only(self, generator):
        """Noise execution is exact-only: the TN columns must be empty."""
        lines = generator._noise_table()
        body = [line for line in lines[2:]]
        assert body, "noise table must have rows"
        for row in body:
            cells = [c.strip() for c in row.strip("|").split("|")]
            assert cells[1] == generator.YES          # exact_ode column
            assert all(c == generator.NO for c in cells[2:])  # mps/peps

    def test_observable_floors_come_from_code(self, generator):
        """The observable section derives from the engines' term-site caps."""
        from ryd_gate.backends.tn_common._expressions import (
            MPS_MAX_TERM_SITES,
            PEPS_MAX_TERM_SITES,
        )

        floors = generator._term_site_floors()
        assert floors == {
            "exact_ode": None,
            "mps": MPS_MAX_TERM_SITES,
            "peps": PEPS_MAX_TERM_SITES,
        }

        rows = {line.strip("|").split("|")[0].strip(): line
                for line in generator._observable_table()[2:]}
        assert "any finite product" in rows["exact_ode"]
        assert "any finite product" in rows["mps"]         # cap is None
        assert f"at most {PEPS_MAX_TERM_SITES} distinct sites" in rows["peps"]
