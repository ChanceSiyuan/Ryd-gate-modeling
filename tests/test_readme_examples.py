"""The single README quick-start executes against the installed package."""

import re
from pathlib import Path

README = Path(__file__).resolve().parents[1] / "README.md"


def _python_blocks():
    text = README.read_text()
    return re.findall(r"```python\n(.*?)```", text, flags=re.S)


def test_readme_has_one_focused_quickstart():
    blocks = _python_blocks()
    assert len(blocks) == 1
    assert "SweepProtocol" in blocks[0]
    assert "simulate(" in blocks[0]


def test_quickstart_executes():
    namespace: dict = {}
    exec(compile(_python_blocks()[0], str(README), "exec"), namespace)
