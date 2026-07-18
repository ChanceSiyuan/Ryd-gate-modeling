# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase.

## Before exploring, read these

- **`CONTEXT-MAP.md`** at the repo root — it points at one context doc per `src` block, all collected under `docs/contexts/`. Read each one relevant to the topic, and respect its status annotation (a **pinned down** context's vocabulary is settled; an **under active design** context is still open to grilling and revision).
- **`docs/adr/`** — all ADRs (system-wide and context-scoped) live in this single flat folder. Read those that touch the area you're about to work in.

If any of these files don't exist, **proceed silently**. Don't flag their absence; don't suggest creating them upfront. The `/domain-modeling` skill (reached via `/grill-with-docs` and `/improve-codebase-architecture`) creates them lazily when terms or decisions actually get resolved.

## File structure

Multi-context, with every context doc centralized in one folder. Do **not** create `src/<block>/CONTEXT.md` or `src/<block>/docs/adr/` — context docs and ADRs stay out of `src/`.

```
/
├── CONTEXT-MAP.md
├── docs/
│   ├── contexts/
│   │   ├── rydberg-simulation.md          ← src/ryd_gate (pinned down)
│   │   └── quantum-optimal-control.md     ← src/qoc (under active design)
│   └── adr/                               ← all ADRs, one flat folder
└── src/
```

When a new `src` block gains its own vocabulary, add `docs/contexts/<block>.md` and a new entry in `CONTEXT-MAP.md`.

## Use the glossary's vocabulary

When your output names a domain concept (in an issue title, a refactor proposal, a hypothesis, a test name), use the term as defined in the context docs. Don't drift to synonyms the glossary explicitly avoids.

If the concept you need isn't in the glossary yet, that's a signal — either you're inventing language the project doesn't use (reconsider) or there's a real gap (note it for `/domain-modeling`).

## Flag ADR conflicts

If your output contradicts an existing ADR, surface it explicitly rather than silently overriding:

> _Contradicts ADR-0007 (reserve method names for their required structure) — but worth reopening because…_
