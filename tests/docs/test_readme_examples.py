"""README quickstart snippets execute against the real installed package."""

import re
from pathlib import Path

README = Path(__file__).resolve().parents[2] / "README.md"


def _python_blocks():
    text = README.read_text()
    return re.findall(r"```python\n(.*?)```", text, flags=re.S)


def test_readme_has_two_quickstart_blocks():
    blocks = _python_blocks()
    assert len(blocks) >= 2
    assert "TFIMQuenchProtocol" in blocks[0]
    assert "cz_gate_report" in blocks[1]


def test_quench_quickstart_executes():
    blocks = _python_blocks()
    namespace: dict = {}
    exec(compile(blocks[0], str(README), "exec"), namespace)  # asserts inside the snippet


def test_gate_report_quickstart_executes():
    # Bounded runtime: three exact 49-dim solves (~10-15 s single-threaded).
    blocks = _python_blocks()
    namespace: dict = {}
    exec(compile(blocks[1], str(README), "exec"), namespace)
    report = namespace["report"]
    assert report.fidelity > 0.9999
