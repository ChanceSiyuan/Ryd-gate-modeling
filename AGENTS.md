# AGENTS.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with
project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial
tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:

- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them; don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No flexibility or configurability that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes,
simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:

- Don't improve adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it; don't delete it.

When your changes create orphans:

- Remove imports, variables, and functions that your changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:

- "Add validation" → write tests for invalid inputs, then make them pass.
- "Fix the bug" → write a test that reproduces it, then make it pass.
- "Refactor X" → ensure tests pass before and after.

For multi-step tasks, state a brief plan:

```text
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria such as
"make it work" require clarification.

## Agent skills

Use only skills exposed by the current agent environment. A skill name in an
old design or Git history does not prove that the skill is installed.

### Documentation curation

Before creating, moving, or deleting tracked documentation, invoke the
`curate-repo-docs` skill in `.agents/skills/curate-repo-docs/`. The canonical
map and placement policy are in `docs/README.md`. Temporary plans, progress
notes, review artifacts, and conversation logs belong in ignored
`.agents/work/`, not tracked `docs/`.

### Issue tracker

Issues are tracked on GitHub (`ChanceSiyuan/Ryd-gate-modeling`). Prefer the
`gh` CLI when it is installed and authenticated; otherwise use an available
GitHub API/connector or report that the operation cannot be performed. See
`docs/agents/issue-tracker.md`.

### Triage labels

Default label vocabulary: `needs-triage`, `needs-info`, `ready-for-agent`,
`ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Multi-context: root `CONTEXT-MAP.md` → per-src-block context docs centralized
in `docs/contexts/`. See `docs/agents/domain.md`.

### Results reports (mandatory)

Any run that creates a new directory under `results/` or writes new data into
one is not complete until that directory's `README.md` is written or updated
before delivery. Invoke the `results-report` skill in
`.agents/skills/results-report/`; it contains the report rules and validator.

All numerical payloads under `results/` are local-only and ignored by Git.
Track only each directory's `README.md`; never use `git add -f` to commit result
data, figures, logs, checkpoints, or exports.

### Research skills (sci-brain)

When the current environment exposes `sci-brain`, route by intent:

| Intent | Use |
|---|---|
| Pick a research direction or decide which physics to pursue | `sci-brain:brainstorm-ideas` |
| Find, read, or cite papers; survey a field | `sci-brain:survey` pipeline |
| Draft or review a scientific manuscript | `sci-brain:paper-writer` or `sci-brain:paper-reviewer` |
| Critique an existing scientific figure | `sci-brain:figure-taste` |
| Attack one hard goal after a direct approach stalls | `sci-brain:flow` |

Code changes, API design, routine debugging, plotting implementation, and
library/tooling facts follow the normal repository workflow and whatever
relevant engineering or documentation skills are actually exposed. Do not
route them to research brainstorming merely because the subject is physics.

The `autoresearch-*` skills are not used here without asking first: each
attempt runs in its own Git worktree, while this repository normally works
directly on the checkout.

The knowledge base is `.knowledge/`. Commit `references.bib`, `INDEX.md`, and
`NOTES.md`; keep rendered paper full texts and `.raw/` PDFs ignored. Point
manuscript bibliographies at `.knowledge/references.bib`.
