"""Guard the intentionally small, build-free Markdown documentation surface."""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DOCS = REPO / "docs"
EXPECTED_DOCS = {"model.md", "simulation.md", "gates.md"}


def test_docs_contains_only_the_three_user_guides():
    files = {path.name for path in DOCS.iterdir() if path.is_file()}
    directories = {path.name for path in DOCS.iterdir() if path.is_dir()}
    assert files == EXPECTED_DOCS
    assert directories == set()


def test_local_markdown_links_exist():
    sources = [REPO / "README.md", *(DOCS / name for name in EXPECTED_DOCS)]
    pattern = re.compile(r"\[[^]]*\]\(([^)]+)\)")

    for source in sources:
        for target in pattern.findall(source.read_text()):
            if target.startswith(("http://", "https://", "#")):
                continue
            path_text = target.split("#", 1)[0]
            target_path = (source.parent / path_text).resolve()
            assert target_path.exists(), f"{source}: broken local link {target!r}"


def test_python_blocks_are_syntactically_valid():
    sources = [REPO / "README.md", *(DOCS / name for name in EXPECTED_DOCS)]
    pattern = re.compile(r"```python\n(.*?)```", flags=re.S)

    for source in sources:
        for index, block in enumerate(pattern.findall(source.read_text())):
            compile(block, f"{source}::python-block-{index}", "exec")


def test_quarto_surface_is_gone():
    assert not list(DOCS.rglob("*.qmd"))
    assert not (DOCS / "_quarto.yml").exists()
    assert "quartodoc" not in (REPO / "pyproject.toml").read_text()
