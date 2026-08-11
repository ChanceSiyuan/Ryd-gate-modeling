# Documentation map

Tracked `docs/` content is durable repository knowledge. Material that is useful
only while one task is running belongs in an issue or ignored agent workspace,
not here.

| Content | Location | Lifecycle |
|---|---|---|
| Public model and API guides | `model.md`, `simulation.md`, `gates.md` | Update with the code |
| Architecture decisions | `adr/` | Keep; mark superseded decisions rather than erasing history |
| Domain boundaries and vocabulary | `contexts/` plus root `CONTEXT-MAP.md` | Update when ownership or concepts change |
| Stable agent workflows | `agents/` | Keep tool-neutral where possible |
| Approved design records | `designs/` | Keep while they explain choices; mark superseded records |

Result provenance is different documentation: every result directory owns its
`results/<run>/README.md` and must follow the `results-report` skill. Numerical
payloads, figures, logs, and exports under `results/` remain local-only; only
the README reports are tracked.

## Placement rules

1. Search for an existing canonical document and update it before creating a
   second account of the same subject.
2. Put user-facing behavior in one of the three public guides, a lasting
   architecture choice in `adr/`, cross-module language in `contexts/`, and an
   approved implementation-neutral design in `designs/`.
3. Put implementation plans, task briefs, progress notes, agent reports,
   review diffs, and conversation logs in ignored `.agents/work/<task>/` or in
   the relevant GitHub issue. Never add them to tracked `docs/`.
4. Never create `docs/work/`. Git history preserves retired plans; active work
   belongs in ignored `.agents/work/` or the relevant GitHub issue.
5. Give new design records `YYYY-MM-DD-slug.md` names and YAML frontmatter with
   `status` and `document-role`. ADRs retain their numbered names.
6. Do not add another top-level docs category without updating this map and
   `tests/test_documentation_structure.py` in the same change.

## Theory and literature

External papers and downloaded TeX are reference inputs, not project docs.
Store their citable metadata in `.knowledge/references.bib`, their index and
project notes in `.knowledge/INDEX.md` and `.knowledge/NOTES.md`, and ignored
source material under `.knowledge/.raw/`. Repository-owned derivations should
be linked from the code and indexed in `theory/README.md`; project manuscripts
belong under a dedicated `manuscripts/` tree rather than at repository root.

## Agent workflow

Before creating, moving, or deleting tracked documentation, use
[`curate-repo-docs`](../.agents/skills/curate-repo-docs/SKILL.md). Validate a
documentation change with:

```bash
uv run pytest -q tests/test_documentation_structure.py
git diff --check
```
