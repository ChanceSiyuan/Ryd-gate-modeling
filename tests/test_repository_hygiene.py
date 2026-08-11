"""Prevent machine-local data and agent state from returning to Git."""

import subprocess
from pathlib import Path, PurePosixPath

REPO = Path(__file__).resolve().parents[1]
LOCAL_ONLY_SUFFIXES = {
    ".csv",
    ".db",
    ".h5",
    ".hdf5",
    ".log",
    ".npy",
    ".npz",
    ".parquet",
    ".pdf",
    ".pickle",
    ".pkl",
    ".png",
    ".sqlite",
    ".sqlite3",
}
LOCAL_ONLY_PREFIXES = {
    (".agents", "work"),
    (".knowledge", ".raw"),
    (".claude",),
    (".codex",),
    (".superpowers",),
}


def _tracked_worktree_paths() -> list[PurePosixPath]:
    output = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    paths = [PurePosixPath(item) for item in output.split("\0") if item]
    return [path for path in paths if (REPO / path).exists()]


def test_only_reports_are_tracked_under_results():
    offenders = [
        str(path)
        for path in _tracked_worktree_paths()
        if path.parts[0] == "results" and path.name != "README.md"
    ]
    assert not offenders, f"results payloads must stay local-only: {offenders}"


def test_machine_local_data_and_agent_state_are_not_tracked():
    offenders = []
    for path in _tracked_worktree_paths():
        if "data" in path.parts or path.suffix.lower() in LOCAL_ONLY_SUFFIXES:
            offenders.append(str(path))
            continue
        if any(path.parts[: len(prefix)] == prefix for prefix in LOCAL_ONLY_PREFIXES):
            offenders.append(str(path))

    assert not offenders, f"machine-local files must not be tracked: {offenders}"


def test_agent_instructions_are_codex_native():
    agents = REPO / "AGENTS.md"
    assert agents.is_file()
    assert not agents.is_symlink()
    assert not (REPO / "CLAUDE.md").exists()
