"""Guard the intentionally small, build-free Markdown documentation surface."""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"
EXPECTED_DOCS = {"model.md", "simulation.md", "gates.md"}
EXPECTED_DOCS_DIRECTORIES = {"adr", "agents", "contexts"}


def test_docs_contains_only_the_three_user_guides_and_adrs():
    files = {path.name for path in DOCS.iterdir() if path.is_file()}
    directories = {path.name for path in DOCS.iterdir() if path.is_dir()}
    assert files == EXPECTED_DOCS
    assert directories == EXPECTED_DOCS_DIRECTORIES


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


def test_docs_guide_python_blocks_are_syntactically_valid():
    # The README block is already compiled *and executed* by test_readme_examples;
    # here we only guard the three docs guides, which are not executed.
    sources = [DOCS / name for name in EXPECTED_DOCS]
    pattern = re.compile(r"```python\n(.*?)```", flags=re.S)

    for source in sources:
        for index, block in enumerate(pattern.findall(source.read_text())):
            compile(block, f"{source}::python-block-{index}", "exec")
