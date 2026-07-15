"""README quickstart snippets execute against the real installed package."""

import re
from pathlib import Path

import pytest

README = Path(__file__).resolve().parents[2] / "README.md"


def _python_blocks():
    text = README.read_text()
    return re.findall(r"```python\n(.*?)```", text, flags=re.S)


def test_readme_has_two_quickstart_blocks():
    blocks = _python_blocks()
    assert len(blocks) >= 2
    assert "TFIMQuenchProtocol" in blocks[0]
    assert "TOProtocol" in blocks[1]


def test_quench_quickstart_executes():
    blocks = _python_blocks()
    namespace: dict = {}
    exec(compile(blocks[0], str(README), "exec"), namespace)  # asserts inside the snippet


@pytest.mark.slow
def test_gate_fidelity_quickstart_executes():
    # Three adaptive-ODE 49-dim solves on the GHz 7-level ladder (~1 min each
    # single-threaded), so this runs in the slow suite.
    blocks = _python_blocks()
    namespace: dict = {}
    exec(compile(blocks[1], str(README), "exec"), namespace)
    assert namespace["fidelity"] > 0.9999
