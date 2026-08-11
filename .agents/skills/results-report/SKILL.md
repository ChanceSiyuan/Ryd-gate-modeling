---
name: results-report
description: Use after any script or notebook run that creates a directory under results/ or writes data into one, and whenever creating, revising, or simplifying a results directory README.md. Make the report explain the final physical model and calculation, compare the physical outcome with and without the studied effect, preserve relevant figures, document provenance and reproduction, update the results index, and run the validator before delivery.
---

# Writing a results report

A run that leaves data behind is not finished until `results/<dir>/README.md` says what
that data is. Someone opening the directory in six months — often the person who made it —
must not have to reverse-engineer a `.npz` to find out.

The file is named `README.md`, not `ANALYSIS.md`, because GitHub renders it inline when
browsing the directory. That is the whole point: walk into the folder, know immediately.

## Shape

Reports are written in Chinese, and in whatever shape the study needs — a pulse-synthesis
study and a spectrum study do not have the same sections. `results/zxz_direct_qoc/README.md`
and `results/297_laser_noise/README.md` are the reference examples; read one before writing
a new report.

Use a flat structure like `results/zxz_direct_qoc/README.md`: one H1 followed by a short
sequence of H2 sections. Prefer bold lead-ins inside a section to numbered H3/H4 headings
unless the hierarchy is necessary.

What every report must contain, in roughly this order:

1. **`# <dir> — <一句话说明>`** — the H1 says what the study is, not just its name.
2. **问题与模型** — the actual physics. Write the Hamiltonian, the objective, the
   acceptance criteria, the parameterisation, with units and real parameter values. A
   reader must not have to open the script to learn what was simulated. Use LaTeX
   (`$…$`, `$$…$$`).
3. **核心结果** — tables, not prose. Compare against something: the paper being
   reproduced, a baseline method, the other manifold, the other resolution. A number
   with nothing to compare it to teaches nothing. Then say in one bold sentence what the
   table means.
4. **Figures** — `![说明](relative/path.png)` inline where they belong, each with a line
   saying what it shows. If the directory has no images, say so and point at the notebook
   whose committed outputs hold them.
5. **告示 sections** — whatever a reader would otherwise trip over: an obsolete cost
   model, pre-rewrite data, a headerless CSV, a concurrently-running job, a schema
   difference. Give these their own heading; do not bury them.
6. **数据与溯源 / 复现** — file table, the main artifact's schema, generating script with
   its last commit, and copy-pasteable commands (cheapest first: replay before recompute).
   Name the design/plan docs and any ADR.

The section names are yours to choose; the validator only checks that provenance and
reproduce sections exist and that everything they point at is real.

## Writing and result-selection rules

**Write only the final implementation.** Explain the model, observable, derivation, and
algorithm that produced the reported result. Omit discarded formulas, failed approaches,
debugging history, and discovery narrative. Keep history only when an old artifact,
schema, or cost definition remains operationally relevant; put that warning in a notice.

**Keep the shortest complete theory chain.** State the physical process, start from the
Hamiltonian or governing equation, define the target observable, and show the direct steps
needed to reach the computed quantity. Do not replace the physics with an unexplained
table, but omit algebra that does not help the reader understand what was calculated.

**Make the core result a physical comparison.** Name the process or parameter point, the
assumptions, the baseline without the studied effect, the result with it, and the
difference. For a noise study, prefer columns such as `F_without_noise`, `F_with_noise`,
and `Delta F` (or the corresponding loss quantities). Intermediate diagnostics such as
measured-band fractions, cutoff sweeps, and spectrum-crossing points do not belong in the
main report unless they are the study's explicit target.

**Define assumptions at first use.** State what every named model changes and what it
holds fixed. For example, `flat` can mean holding the PSD at its measurement-edge value
outside the measured band, while `power` can mean continuing a fitted power-law tail.
Use the definitions actually implemented by the study; never infer them from the labels.

**Simplify numerical detail without removing evidence.** Keep units, physical parameter
ranges, validity regimes, acceptance criteria, convergence evidence, and independent
validation. Remove solver plumbing, sampling post-mortems, array-index reminders, and
secondary sensitivity tables unless interpretation or reproduction depends on them.

**Separate direct results from estimates.** Distinguish simulation or measurement from
reweighting, interpolation, extrapolation, and scaling. Never describe a rescaled value
as a directly simulated point.

**Preserve relevant figures.** Keep existing measurement inputs and result figures when
they remain accurate, place each beside the result it supports, and give it a short
caption. Remove a figure only when it is invalid, irrelevant to the final implementation,
or the user explicitly asks to remove it.

**Select results under the physical constraints first.** If the requested scientific
comparison differs from existing diagnostic tables, recompute it from stored artifacts.
Apply hardware and validity constraints before selecting a best point, and state the size
and scope of the compared set.

**Keep prose compact.** Use tables for repeated comparisons, state each conclusion once,
and prefer short paragraphs. Put display math between standalone `$$` delimiters with
blank lines around the block so Cursor and common Markdown renderers recognize it.

## Hard rules

**Every number traces to a field.** Name the file and the key: `status.json:
completed_points_this_run`, not "about 30k points". A number you cannot point at does not
go in.

**Never invent provenance.** If no script or notebook references the directory, search
harder — `git grep <dirname> -- scripts src docs` catches producers that the literal path
string misses, but beware the reverse: a hit on a *module* or *notebook filename* that
merely shares the directory's name is not provenance. If nothing real turns up, write
"provenance unconfirmed" and list only what the data itself shows. Never guess a filename.

**Measure before you claim, and say when you measured.** If a report asserts that two
quantities agree, or that an artifact is stale, that assertion is a measurement with a
date — write both. When something looks broken, check whether it actually breaks anything
before recommending a fix; consumers often already have fallbacks.

**Do not tidy the data to make the report neat.** Artifacts stay byte-for-byte as the
study wrote them. A schema difference across files is something to *document*, not to
paper over by rewriting old artifacts — provenance is worth more than uniformity.

**Report the gaps.** Script committed after the data it supposedly produced, a
subdirectory nothing references, results predating an architecture rewrite, a
concurrently-running job — these are the things a reader most needs and least expects.
State them plainly in `## Data and provenance` without dramatising.

**Figures are not added to git.** `.gitignore` excludes `*.png`/`*.pdf`. Reference them by
relative path, note they are untracked, give the regeneration command. Do not `git add -f`
a generated figure. All numerical payloads and figures under `results/` are local-only;
track only each result directory's `README.md`.

## Keep the index alive

A new directory under `results/` needs a row in `results/README.md` — directory, what it
answers, produced by, state. Without it the index rots into a lie. The validator enforces
this.

## Before delivering

```bash
python .agents/skills/results-report/validate.py
```

It checks the required report structure, that every `![](…)` path resolves on disk, that
every repo path named in a command block exists, and that the index has a row and a README
for every directory. Fix what it reports, then paste the output as your evidence — "the
report is written" is not a claim you get to make without it.

Clean CI checkouts intentionally lack local-only figures and numerical payloads. CI may
run the structural subset with `--allow-missing-local-artifacts`; delivery from a
data-bearing working checkout must still run the strict command above.

Also confirm by hand what the validator cannot:

- every quoted number was read from the artifact, not remembered;
- the cheapest reproduce command was actually run, or is stated as unverified;
- anything unconfirmed is labelled unconfirmed.
