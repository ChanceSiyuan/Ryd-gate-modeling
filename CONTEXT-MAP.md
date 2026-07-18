# Context Map

## Contexts

- [Rydberg Simulation](./docs/contexts/rydberg-simulation.md) — defines physical systems, fully specified protocols, and numerical evolution. **Status: pinned down.**
- [Quantum Optimal Control](./docs/contexts/quantum-optimal-control.md) — formulates and solves research pulse-optimization problems. **Status: under active design.**

## Relationships

- **Research study → Rydberg Simulation**: the caller constructs protocols, selects states and observables, runs `simulate`, and reduces physical results to a scalar loss.
- **Research study → Quantum Optimal Control**: the caller gives that closed scalar loss and its finite parameter space to `qoc` — or, for GRAPE, the bilinear control model exported from `ryd_gate` together with a control map and terminal objective.
- **Quantum Optimal Control ↛ Rydberg Simulation**: `qoc` does not import `ryd_gate` or interpret physical systems, protocols, states, results, or objectives.
- **Rydberg Simulation ↛ Quantum Optimal Control**: `ryd_gate` does not import `qoc` or carry optimizer state.
