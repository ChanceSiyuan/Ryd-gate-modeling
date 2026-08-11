#!/usr/bin/env python3
"""Check that results/ reports are complete and that everything they point at exists.

Usage:
    python .agents/skills/results-report/validate.py            # all reports
    python .agents/skills/results-report/validate.py results/cz_gate/README.md
    python .agents/skills/results-report/validate.py \
        --allow-missing-local-artifacts                         # clean CI checkout

Checks per report:
  1. the title and required provenance/reproduction sections are present
  2. every ![](path) image reference resolves on disk
  3. every repo path named in a fenced command block exists

Exit status is non-zero if any check fails.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Structure, not a fixed outline: reports are written in whatever shape the study
# needs (a pulse study and a spectrum study do not have the same sections), but a
# reader must always be able to find the provenance and the way to re-run it.
# Each entry is (description, accepted heading substrings).
REQUIRED = [
    ("provenance / data inventory", ("数据", "溯源", "Data", "provenance")),
    ("reproduce", ("复现", "Reproduce", "reproduce")),
]

IMAGE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
FENCE = re.compile(r"```[a-z]*\n(.*?)```", re.S)
# Repo-relative paths worth existence-checking when they appear in a command.
CMD_PATH = re.compile(r"\b((?:scripts|src|tests|docs|results)/[\w./{}*,-]+)")


def repo_root() -> Path:
    for p in [Path.cwd(), *Path.cwd().parents]:
        if (p / "pyproject.toml").exists():
            return p
    raise SystemExit("not inside the repository")


def expand_braces(token: str) -> list[str]:
    """`a{1,2}/b` -> [`a1/b`, `a2/b`]; leaves other tokens alone."""
    m = re.search(r"\{([^{}]*)\}", token)
    if not m:
        return [token]
    out = []
    for alt in m.group(1).split(","):
        out.extend(expand_braces(token[: m.start()] + alt + token[m.end():]))
    return out


def check(
    report: Path,
    root: Path,
    *,
    allow_missing_local_artifacts: bool = False,
) -> list[str]:
    text = report.read_text()
    where = report.parent
    problems = []
    is_index = report.parent == root / "results"

    if is_index:
        # The index is a map, not a study report: it needs a row per directory
        # instead of the study-report structure, so it cannot silently go stale.
        for d in sorted((root / "results").iterdir()):
            if not d.is_dir():
                continue
            if not (d / "README.md").exists():
                problems.append(f"{d.name}/ has no README.md")
            if f"({d.name}/)" not in text and f"`{d.name}`" not in text:
                problems.append(f"{d.name}/ is missing from the index table")
    else:
        headings = [ln.lstrip("#").strip()
                    for ln in text.splitlines() if ln.startswith("#")]
        if not text.lstrip().startswith("# "):
            problems.append("no H1 title on the first line")
        elif " — " not in headings[0] and " - " not in headings[0]:
            problems.append("H1 title states no topic: expected '# <dir> — <topic>'")
        for what, accepted in REQUIRED:
            if not any(any(a in h for a in accepted) for h in headings):
                problems.append(
                    f"no {what} section (heading containing one of {'/'.join(accepted)})")
        if not IMAGE.search(text) and "无图" not in text and "no image files" not in text:
            problems.append("no figure referenced and no statement that there are none")

    for ref in IMAGE.findall(text):
        if ref.startswith(("http://", "https://")):
            continue
        if not allow_missing_local_artifacts and not (where / ref).exists():
            problems.append(f"image not found: {ref}")

    for block in FENCE.findall(text):
        for token in CMD_PATH.findall(block):
            token = token.rstrip(".,;")
            candidates = expand_braces(token)
            if allow_missing_local_artifacts and token.startswith("results/"):
                continue
            if not any((root / c).exists() or list(root.glob(c)) for c in candidates):
                problems.append(f"path in command does not exist: {token}")

    return problems


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-missing-local-artifacts",
        action="store_true",
        help="allow missing result images and result payloads in a clean checkout",
    )
    parser.add_argument("reports", nargs="*")
    args = parser.parse_args(argv[1:])

    root = repo_root()
    targets = [Path(path).resolve() for path in args.reports] or sorted(
        [root / "results" / "README.md"]
        + [p / "README.md" for p in sorted((root / "results").iterdir()) if p.is_dir()])

    failed = 0
    for report in targets:
        if not report.exists():
            print(f"MISSING  {report.relative_to(root)}")
            failed += 1
            continue
        problems = check(
            report,
            root,
            allow_missing_local_artifacts=args.allow_missing_local_artifacts,
        )
        rel = report.relative_to(root)
        if problems:
            failed += 1
            print(f"FAIL     {rel}")
            for p in problems:
                print(f"           {p}")
        else:
            print(f"ok       {rel}")

    print(f"\n{len(targets) - failed}/{len(targets)} reports pass")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
