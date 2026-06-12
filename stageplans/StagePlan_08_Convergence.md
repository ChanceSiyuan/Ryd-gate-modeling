# Stage 8 Plan: Surface Convergence (Sequence ⊇ more of Protocol)

## Purpose

Close the perceived "two simulation systems" gap. The kernel is already
unified (every `Sequence` lowers to a `SequenceProtocol`, which *is* a kernel
`Protocol`); this stage widens the Sequence surface and adds an explicit
lossy bridge in the other direction:

1. **Sequence coverage**: pulse phase (+ Pulser-style virtual-Z
   `post_phase_shift`), local channels with `Sequence.target(...)`, and the
   `gputn`/`peps` backends on the sequence path.
2. **`sequence_from_protocol`**: an explicit, lossy, opt-in discretization of
   a bound continuous protocol onto an integer-ns `Sequence` (export /
   hardware use; the continuous path stays authoritative for gate-grade
   fidelity).
3. **Docs**: state the `SequenceProtocol` convergence point explicitly.

## No-Wrapper / No-Fake Rules

1. No new solver or compiler paths. Phase and local addressing lower onto
   the *existing* kernel channel conventions: complex amplitude coefficients
   (h.c. added per `channel_needs_hermitian_conjugate`) and per-site channel
   names `f"{base}_{site}"` resolved by the kernel channel-lowering helpers
   (`split_site_channel` / `block_name_for_drive_channel`; at implementation
   time in `core/channel_lowering.py`, since consolidated into
   `ir/hamiltonian.py` + `core/level_structures.py`).
2. No silent lossiness. The TN profile helpers take `np.real(...)` of
   coefficients, so nonzero pulse phase on a TN backend raises
   `sequence.phase_backend_unsupported` instead of dropping the imaginary
   part. The discretization bridge stamps its loss parameters into
   `Pulse.metadata` and refuses constructs it cannot represent.
3. `sequence_from_protocol` inverts the documented lowering exactly
   (amp = Ω/2, det = −Δ, rad/s ↔ rad/µs); it must not re-implement protocol
   evaluation — it samples `protocol.get_drive_coefficients`.

## Binding decisions

- **Phase convention**: amplitude-channel coefficient is
  `(Ω(t)/2)·e^{−i·φ_eff}` with `φ_eff = pulse.phase_rad + Σ post_phase_shift`
  of earlier pulses on the same channel (Pulser virtual-Z semantics).
  Exact backend only; TN backends raise `sequence.phase_backend_unsupported`.
- **Local channels**: `declare_channel` accepts `addressing="local"` Rydberg
  channels; `Sequence.target(atom_ids, channel)` records a replayable
  `TargetOp` (consumes `retarget_time_ns`, respects `max_targets`); pulses on
  a local channel without targets raise `sequence.local_targets_missing`;
  `target()` on a global channel raises `sequence.target_global`.
  `virtual_rb87()`'s `rydberg_local` gains the `1r` channel maps
  (`global_X`/`global_n`) so local sequences run on the TN-capable model.
  Lowering emits per-site keys; works on exact and mps (real coefficients).
- **Backends**: `simulate_sequence` allows `exact`/`mps`/`gputn`/`peps`;
  non-native states get `UnsupportedStateHandle` (Stage 3 machinery, already
  in place). Error code `simulate_sequence.backend_not_stage3` becomes
  `simulate_sequence.backend_unsupported`.
- **Schema**: `sequence/v1` operations enum gains the additive `"target"` op
  (pre-release additive extension; old payloads still validate).
- **Bridge**: `sequence_from_protocol(system, x, *, device=None,
  channel_id="rydberg_global", dt_ns=1)` in `src/ryd_gate/discretize.py`.
  Coefficients must map onto exactly one device channel's (amp, det) pair for
  the system's level structure — anything else (`drive_420` sets, per-site
  keys, complex/time-varying phase) raises
  `discretize.channel_not_representable` / `discretize.phase_not_representable`.

## Allowed File Operations

Create: `src/ryd_gate/discretize.py`, `tests/sequence/test_phase_and_local.py`,
`tests/sequence/test_discretize.py`, this plan.

Modify: `src/ryd_gate/sequence.py`, `src/ryd_gate/protocols/sequence_protocol.py`,
`src/ryd_gate/simulate.py`, `src/ryd_gate/devices.py`, `src/ryd_gate/__init__.py`,
`src/ryd_gate/schemas/sequence.v1.schema.json`, affected tests
(`test_sequence.py`, `test_compile_exact.py`, `test_result_exact.py`,
`test_init.py`), docs (`fundamentals.md`, `how_to_sequences.md`,
`capability_matrix.md` regenerated), `stageplans/README.md`, `CHANGELOG.md`.

Do not modify: `src/ryd_gate/backends/*`, `src/ryd_gate/ir/*`,
the kernel channel-lowering helpers (then `core/channel_lowering.py`),
`core/factories.py` (since consolidated),
notebooks.

## Tests / Acceptance

- Ramsey physics through the exact solver: two π/2 pulses, relative phase 0 →
  full transfer; phase π → return to |1⟩; `post_phase_shift` reproduces the
  same as explicit phase on the next pulse.
- Local addressing physics: 2 atoms (C6=0), local π pulse on one target →
  populations `[~1, ~0]`; same on mps within TN tolerance; serialization
  replay of `TargetOp` round-trips.
- gputn/peps dispatch: monkeypatched kernel `simulate` → result carries
  `UnsupportedStateHandle` with the existing reason codes; no
  `NotImplementedError`.
- Bridge: discretized `SweepProtocol` matches the direct protocol simulation
  (fine `dt_ns`) on exact; `TOProtocol` refuses with the typed code; loss
  metadata stamped.
- `OMP_NUM_THREADS=1 uv run pytest -m "not slow" -q` green; mypy green; ruff
  green; capability matrix regenerated and fresh.
