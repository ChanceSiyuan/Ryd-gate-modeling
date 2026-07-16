"""Keep checked-in research notebooks portable and source-only."""

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NOTEBOOKS = tuple(sorted((REPO / "scripts" / "notebooks").glob("*.ipynb")))
MACHINE_HOME = re.compile(
    r"(?:/(?:home|Users)/[^/\s]+/|[A-Za-z]:\\Users\\[^\\\s]+\\)"
)
RETIRED_PATHS = ("main.tex", "scripts/results/")


def test_notebooks_do_not_commit_execution_state():
    assert NOTEBOOKS, "expected research notebooks under scripts/notebooks"

    for path in NOTEBOOKS:
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
            "".join(cell.get("source", []))
            if isinstance(cell.get("source", []), list)
            else cell.get("source", "")
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
