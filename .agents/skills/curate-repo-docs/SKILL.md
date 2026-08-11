---
name: curate-repo-docs
description: Curate this repository's durable documentation and keep transient agent artifacts out of tracked docs. Use when creating, revising, moving, deleting, auditing, or reorganizing files under docs/; when writing an ADR, context document, design record, implementation plan, or agent policy; or when deciding where a theoretical reference or research note belongs. Do not use for per-run results README reports, which require the results-report skill.
---

# Curate Repository Documentation

## Establish the documentation class

Read `docs/README.md` completely before changing documentation. Search the
repository for the subject and update its canonical document instead of
creating a parallel account.

Classify the artifact before choosing a path:

- Put public model, simulator, and gate behavior in the existing root guides
  under `docs/`.
- Put durable architecture choices in `docs/adr/` and cross-module vocabulary
  or ownership in `docs/contexts/`.
- Put approved, implementation-neutral designs in `docs/designs/`.
- Put stable agent operating policy in `docs/agents/`.
- Put temporary plans, task briefs, progress notes, reviews, diffs, and
  conversation logs in ignored `.agents/work/<task>/` or the GitHub issue.
- Put literature metadata and notes in `.knowledge/`; keep downloaded full text
  and external TeX under ignored `.knowledge/.raw/`.
- Use `results-report` for every `results/<run>/README.md` change.

Do not create `docs/work/`. Git history already preserves retired plans; active
work belongs in ignored `.agents/work/` or the relevant GitHub issue.

## Make the smallest durable change

1. Inspect `git status --short` and preserve unrelated or pre-existing edits.
2. Use `rg` to find inbound links, code references, and overlapping documents.
3. Edit only the canonical document and the links or navigation that the move
   makes stale.
4. Give design records YAML frontmatter containing `status` and
   `document-role`. Mark superseded records; do not silently erase decisions.
5. Update `docs/README.md` and `tests/test_documentation_structure.py` together
   if a genuinely new top-level category is necessary.
6. Remove a completed transient plan after its branch or issue closes. Do not
   keep a tracked plan archive; Git history already preserves deleted plans.

Never create a tool-named tracked directory such as `docs/superpowers/` or
`docs/claude/`. Never put raw paper sources, agent execution logs, or generated
review artifacts in `docs/`. Do not invent citation or result provenance.

## Validate the result

Run:

```bash
uv run pytest -q tests/test_documentation_structure.py
git diff --check
```

After moving or deleting a document, use `rg` on its old path and basename to
verify that no live references remain. Inspect `git status --short` again and
report any transitional or archived files deliberately left behind.
