"""Guard the intentionally small, build-free documentation surface."""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"
USER_GUIDES = {"model.md", "simulation.md", "gates.md"}
EXPECTED_DOCS = {"README.md", *USER_GUIDES}
EXPECTED_DOCS_DIRECTORIES = {"adr", "agents", "contexts", "designs"}


def test_docs_has_only_documented_top_level_categories():
    files = {path.name for path in DOCS.iterdir() if path.is_file()}
    directories = {path.name for path in DOCS.iterdir() if path.is_dir()}
    assert files == EXPECTED_DOCS
    assert directories == EXPECTED_DOCS_DIRECTORIES


def test_design_records_have_lifecycle_metadata():
    for path in (DOCS / "designs").glob("*.md"):
        text = path.read_text()
        assert text.startswith("---\n"), f"{path}: missing YAML frontmatter"
        frontmatter = text.split("---\n", 2)[1]
        assert re.search(r"^status: \S+", frontmatter, flags=re.M), f"{path}: missing status"
        assert "document-role: design-record" in frontmatter, f"{path}: wrong document role"


def test_local_markdown_links_exist():
    sources = [REPO / "README.md", *DOCS.rglob("*.md")]
    pattern = re.compile(r"\[[^]]*\]\(([^)]+)\)")

    for source in sources:
        text = source.read_text()
        text = re.sub(r"```.*?```", "", text, flags=re.S)
        text = re.sub(r"`[^`\n]*`", "", text)
        for target in pattern.findall(text):
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            path_text = target.split("#", 1)[0]
            target_path = (source.parent / path_text).resolve()
            assert target_path.exists(), f"{source}: broken local link {target!r}"


def test_docs_guide_python_blocks_are_syntactically_valid():
    # The README block is already compiled *and executed* by test_readme_examples;
    # here we only guard the three docs guides, which are not executed.
    sources = [DOCS / name for name in USER_GUIDES]
    pattern = re.compile(r"```python\n(.*?)```", flags=re.S)

    for source in sources:
        for index, block in enumerate(pattern.findall(source.read_text())):
            compile(block, f"{source}::python-block-{index}", "exec")
