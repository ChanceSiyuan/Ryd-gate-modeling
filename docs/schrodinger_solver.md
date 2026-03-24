# Schrodinger Solver for CZ Gate Simulation

This document describes the Schrodinger equation solvers in `ryd_gate.solvers.schrodinger` and the backward-compatible `CZGateSimulator` facade in `ryd_gate.ideal_cz`.

## Overview

### When to Use Each Solver

| Use Case | Function | Module |
|----------|----------|--------|
| CZ gate fidelity (TO/AR) | `solve_gate()` | `ryd_gate.solvers.schrodinger` |
| Generic time-dependent H(t) | `evolve()` | `ryd_gate.solvers.schrodinger` |
| Backward-compatible API | `CZGateSimulator` | `ryd_gate.ideal_cz` |
| Local addressing sweep | `evolve()` + `SweepAddressingProtocol` | `ryd_gate.solvers` + `ryd_gate.protocols` |

## Modular Architecture

The solver has been decomposed from the original monolithic `CZGateSimulator` into focused modules:

```
core/atomic_system.py      -- AtomicSystem dataclass + factory functions
protocols/base.py           -- Protocol ABC + SweepProtocol ABC
protocols/gate_cz_to.py     -- TOProtocol (cos phase, 6 params)
protocols/gate_cz_ar.py     -- ARProtocol (dual-sine phase, 8 params)
protocols/local_sweep.py    -- SweepAddressingProtocol
solvers/schrodinger.py      -- solve_gate() + evolve()
solvers/monte_carlo.py      -- MonteCarloEngine + AddressingMCEngine
analysis/gate_metrics.py    -- Fidelity, error budget, diagnostics
analysis/addressing_metrics.py -- Pinning error, crosstalk, leakage
ideal_cz.py                 -- CZGateSimulator facade (backward-compatible)
```

### CZ Gate Solver: `solve_gate()`

For CZ gate protocols (TO/AR), the Hamiltonian has a fixed structure where time-dependence comes only from 420nm laser phase modulation:

```python
from ryd_gate.solvers.schrodinger import solve_gate

final_state = solve_gate(system, protocol, x, initial_state)
```

### Generic Solver: `evolve()`

For arbitrary time-dependent Hamiltonians (e.g., sweep protocols):

```python
from ryd_gate.solvers.schrodinger import evolve

final_state = evolve(hamiltonian_fn, t_gate, initial_state)
```

## Constructor Parameters (CZGateSimulator Facade)

```python
CZGateSimulator(
    param_set: Literal['our', 'lukin'] = 'our',
    strategy: Literal['TO', 'AR'] = 'AR',
    blackmanflag: bool = True,
    detuning_sign: Literal[1, -1] = 1,
    *,
    enable_rydberg_decay: bool = False,
    enable_intermediate_decay: bool = False,
    enable_0_scattering: bool = True,
    enable_rydberg_dephasing: bool = False,
    enable_position_error: bool = False,
    enable_polarization_leakage: bool = False,
    sigma_detuning: float | None = None,
    sigma_pos_xyz: tuple[float, float, float] | None = None,
    n_mc_shots: int = 100,
    mc_seed: int | None = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `param_set` | `str` | `'our'` | Physical parameter configuration |
| `strategy` | `str` | `'AR'` | Pulse optimization strategy |
| `blackmanflag` | `bool` | `True` | Apply Blackman envelope for smooth pulses |
| `detuning_sign` | `int` | `1` | Sign of intermediate detuning (+1 bright, -1 dark) |
| `enable_rydberg_decay` | `bool` | `False` | Include Rydberg state decay as imaginary energy shifts |
| `enable_intermediate_decay` | `bool` | `False` | Include intermediate state decay as imaginary energy shifts |
| `enable_0_scattering` | `bool` | `True` | Include |0> AC Stark shift and scattering from intermediates |
| `enable_rydberg_dephasing` | `bool` | `False` | Gate T2* dephasing noise in Monte Carlo |
| `enable_position_error` | `bool` | `False` | Gate position fluctuation noise in Monte Carlo |
| `enable_polarization_leakage` | `bool` | `False` | Include coupling to unwanted Rydberg state |r'> |
| `sigma_detuning` | `float\|None` | `None` | Dephasing noise std dev (Hz). Required when dephasing enabled |
| `sigma_pos_xyz` | `tuple\|None` | `None` | Position noise (sigma_x, sigma_y, sigma_z) in meters |
| `n_mc_shots` | `int` | `100` | Number of Monte Carlo shots |
| `mc_seed` | `int\|None` | `None` | RNG seed for reproducibility |

### Public Methods

| Method | Description |
|--------|-------------|
| `setup_protocol(x)` | Store pulse parameters for repeated use |
| `optimize(x_initial, fid_type)` | Run pulse parameter optimization |
| `gate_fidelity(x, fid_type)` | Calculate average gate infidelity |
| `error_budget(x, initial_states)` | XYZ/AL/LG error decomposition |
| `state_infidelity(initial_state, x)` | Per-state infidelity |
| `run_monte_carlo_simulation(x, ...)` | Full MC simulation |
| `diagnose_plot(x, initial_state)` | Plot population evolution |
| `diagnose_run(x, initial_state)` | Return population time series |
| `plot_bloch(x, save=True)` | Generate Bloch sphere plots (TO only) |

## Level Structure

Each atom has 7 levels forming a 49-dimensional two-atom Hilbert space:

```
Index   Label   Description                     Quantum Numbers
-----   -----   -----------                     ---------------
  0     |0>     Ground state (dark)             5S1/2, F=1
  1     |1>     Ground state (qubit)            5S1/2, F=2
  2     |e1>    Intermediate                    6P3/2, F'=1
  3     |e2>    Intermediate                    6P3/2, F'=2
  4     |e3>    Intermediate                    6P3/2, F'=3
  5     |r>     Target Rydberg                  nS1/2, mJ=-1/2
  6     |r'>    Unwanted Rydberg                nS1/2, mJ=+1/2
```

The two-atom basis states are tensor products: |ij> = |i> x |j>, giving indices 0-48.

## Physical Parameters

### 'our' Configuration (n=70 Rydberg)

| Parameter | Value | Description |
|-----------|-------|-------------|
| Rydberg level | n=70 | 70S1/2 state |
| Intermediate detuning (Delta) | 9.1 GHz | From 6P3/2 resonance |
| Effective Rabi (Omega_eff) | 7 MHz | Two-photon coupling |
| Rydberg interaction (V) | 874 GHz / d^6 | Van der Waals coefficient |
| Zeeman shift | 56 MHz | For |r'> state |
| 6P3/2 lifetime | 110.7 ns | Intermediate state |
| 70S lifetime | 151.55 us | Rydberg state |

### 'lukin' Configuration (n=53 Rydberg)

| Parameter | Value | Description |
|-----------|-------|-------------|
| Rydberg level | n=53 | 53S1/2 state |
| Intermediate detuning (Delta) | 7.8 GHz | From 6P3/2 resonance |
| 420nm Rabi | 237 MHz | Blue laser coupling |
| 1013nm Rabi | 303 MHz | IR laser coupling |
| Rydberg interaction (V) | 450 MHz | At experimental distance |
| Zeeman shift | 2.4 GHz | For |r'> state |
| 53S lifetime | 88 us | Rydberg state |

## Optimization Strategies

### Time-Optimal (TO) Strategy

Phase modulation with single cosine:

```
phi(t) = A*cos(wt + phi_0) + delta*t
```

**Parameter vector:** `x = [A, w/Omega_eff, phi_0, delta/Omega_eff, theta, T/T_scale]`

### Amplitude-Robust (AR) Strategy

Phase modulation with dual sine for first-order amplitude robustness:

```
phi(t) = A1*sin(wt + phi_1) + A2*sin(2wt + phi_2) + delta*t
```

**Parameter vector:** `x = [w/Omega_eff, A1, phi_1, A2, phi_2, delta/Omega_eff, T/T_scale, theta]`

## Hamiltonian Structure

The total CZ gate Hamiltonian is:

```
H(t) = H_static + amplitude(t) * [e^{-i*phi(t)} H_420 + e^{i*phi(t)} H_420_dag]
       + amplitude(t)^2 * H_lightshift
```

where:
- `H_static = H_const + H_1013 + H_1013_dag`: time-independent part
- `H_420`: 420nm laser coupling (ground -> intermediate)
- `H_1013`: 1013nm laser coupling (intermediate -> Rydberg)
- `amplitude(t)`: Blackman envelope (if enabled)
- `phi(t)`: Phase modulation (TO or AR strategy)
- `H_lightshift`: AC Stark shift from |0> -> intermediate off-resonant coupling

## Notes

1. **Imaginary energy approximation**: Decay is modeled as imaginary energy shifts in the diagonal Hamiltonian. Suitable for estimating decay contributions to infidelity.

2. **Optimization algorithm**: Uses SciPy's Nelder-Mead optimizer with `fatol=1e-9`.

3. **Time discretization**: Diagnostic methods use 1000 time points uniformly distributed over the gate duration.

4. **Numerical precision**: ODE integration uses DOP853 (8th order Runge-Kutta) with `rtol=1e-8`, `atol=1e-12`.

5. **Unified fidelity**: The `average_gate_infidelity()` function in `analysis/gate_metrics.py` replaces the previously separate `_fidelity_avg` (TO) and `_avg_fidelity_AR` methods. Both strategies now support all fidelity types (average, sss, bell).

## See Also

- `docs/error_budget_methodology.md` - Error budget decomposition
- `docs/validation.md` - Test suite documentation
