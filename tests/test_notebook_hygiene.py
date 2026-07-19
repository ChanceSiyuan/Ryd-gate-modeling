"""Keep checked-in research notebooks portable.

Notebooks are source-only (no committed execution state) with two deliberate
exceptions: notebooks 01 and 06 commit their rendered figures and summary
tables. Per the user's 2026-07-17 decisions, the two-atom ``01r`` optimization
notebook is executed in fast, deterministic cached-replay mode, and the CZ TO
calibration notebook's executed outputs are themselves the calibration record;
both are exempt from the execution-state check. The portability checks (no
machine-local or retired paths) apply to every notebook, in cell source *and*
in any committed text output.
"""

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NOTEBOOKS = tuple(sorted((REPO / "scripts" / "notebooks").glob("*.ipynb")))
MACHINE_HOME = re.compile(
    r"(?:/(?:home|Users)/[^/\s]+/|[A-Za-z]:\\Users\\[^\\\s]+\\)"
)
RETIRED_PATHS = ("main.tex", "scripts/results/")
# Exempt from the source-only execution-state check (see module docstring).
RENDERED_OUTPUT_NOTEBOOKS = frozenset(
    {"01_cz_gate.ipynb", "06_01r_adiabatic_optimization.ipynb"}
)


def _cell_source(cell):
    source = cell.get("source", [])
    return "".join(source) if isinstance(source, list) else source


def _cell_text_outputs(cell):
    """Committed text outputs only (stream text and ``text/*`` data), never binary
    image payloads, so base64 figures cannot cause false path matches."""
    chunks = []
    for output in cell.get("outputs", []):
        text = output.get("text", "")
        chunks.append("".join(text) if isinstance(text, list) else text)
        for mime, value in output.get("data", {}).items():
            if mime.startswith("text/"):
                chunks.append("".join(value) if isinstance(value, list) else value)
    return "\n".join(chunks)


def test_notebooks_do_not_commit_execution_state():
    assert NOTEBOOKS, "expected research notebooks under scripts/notebooks"

    for path in NOTEBOOKS:
        if path.name in RENDERED_OUTPUT_NOTEBOOKS:
            continue
        notebook = json.loads(path.read_text())
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            assert cell.get("execution_count") is None, f"{path}: cell {index} has an execution count"
            assert cell.get("outputs", []) == [], f"{path}: cell {index} has committed outputs"
            assert "execution" not in cell.get("metadata", {}), (
                f"{path}: cell {index} has execution metadata"
            )


def test_notebooks_do_not_reference_machine_or_retired_paths():
    for path in NOTEBOOKS:
        notebook = json.loads(path.read_text())
        source = "\n".join(
            "\n".join((_cell_source(cell), _cell_text_outputs(cell)))
            for cell in notebook["cells"]
        )
        assert MACHINE_HOME.search(source) is None, (
            f"{path}: contains a machine-local absolute path"
        )
        for retired_path in RETIRED_PATHS:
            # Word-bounded so e.g. "domain.tex" does not count as "main.tex".
            pattern = re.compile(rf"(?<!\w){re.escape(retired_path)}")
            assert pattern.search(source) is None, (
                f"{path}: references retired path {retired_path}"
            )
