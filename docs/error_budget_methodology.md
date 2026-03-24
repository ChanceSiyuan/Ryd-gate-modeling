# Error Budget Methodology

This document explains the CZ gate error budget decomposition and how to reproduce the SI tables using the public API.

## Error Classification

### Error Types (XYZ / AL / LG / Phase)

Every gate infidelity contribution is decomposed into four channels:

- **XYZ (Pauli errors)** -- Decay to computational states |0> or |1> via radiative decay from Rydberg or intermediate states. Correctable with quantum error correction.
- **AL (Atom Loss)** -- BBR ionization from the Rydberg state. Unrecoverable; requires atom reloading.
- **LG (Leakage)** -- Decay to non-computational hyperfine mF states. Requires leakage reduction protocols.
- **Phase** -- Coherent phase errors from quasi-static noise (dephasing, position fluctuations). Only appears in Monte Carlo simulations as the residual after XYZ + AL + LG accounting.

### Error Sources

**Deterministic (always present):**

| Source | Description |
|---|---|
| Rydberg decay | Radiative decay (RD) + BBR ionization during Rydberg excitation |
| Scattering |0> | Off-resonant scattering from intermediate F=1,2,3 states, originating from |0> |
| Scattering |1> | Off-resonant scattering from intermediate states, originating from |1> |
| Polarization leakage | Population transferred to "garbage" Rydberg state (wrong mF) |

**Stochastic (quasi-static noise, evaluated via Monte Carlo):**

| Source | Description |
|---|---|
| Rydberg dephasing | Shot-to-shot laser detuning fluctuations (sigma = 130 kHz) |
| Position error | 3D atomic position fluctuations (sigma_x,y = 70 nm, sigma_z = 130 nm) |

## Detuning Sign Convention

| Label | `detuning_sign` | Parameters |
|---|---|---|
| Bright | +1 | `X_TO_OUR_BRIGHT` |
| Dark | -1 | `X_TO_OUR_DARK` |

## Public API Usage

### Basic Setup

```python
from ryd_gate import CZGateSimulator, MonteCarloResult

X_TO_OUR_DARK = [
   -0.9509172186259588, 1.105272315809505, 0.383911389220584,
   1.2848721417313045, 1.3035218398648376, 1.246566016566724
]
X_TO_OUR_BRIGHT = [
   -1.7370398295694707, 0.7988774460188806, 2.3116588890406224,
   0.5186261498956248, 0.900066116155231, 1.2415235064066774
]
```

### Gate Fidelity

```python
sim = CZGateSimulator(
    param_set="our", strategy="TO",
    blackmanflag=True, detuning_sign=-1,  # dark
    enable_rydberg_decay=True,
)
infidelity = sim.gate_fidelity(X_TO_OUR_DARK)
```

### Error Budget (XYZ/AL/LG Decomposition)

```python
sim = CZGateSimulator(
    param_set="our", strategy="TO",
    blackmanflag=True, detuning_sign=-1,
    enable_rydberg_decay=True,
    enable_intermediate_decay=True,
    enable_polarization_leakage=True,
)

budget = sim.error_budget(X_TO_OUR_DARK)

# Returns:
# {
#   "rydberg_decay":        {"total": ..., "XYZ": ..., "AL": ..., "LG": ...},
#   "intermediate_decay":   {"total": ..., "XYZ": ..., "AL": ..., "LG": ...},
#   "polarization_leakage": {"total": ..., "XYZ": ..., "AL": ..., "LG": ...},
# }
```

### Direct Module Usage

```python
from ryd_gate.core.atomic_system import create_atomic_system
from ryd_gate.protocols.gate_cz_to import TOProtocol
from ryd_gate.analysis.gate_metrics import error_budget

system = create_atomic_system(
    param_set="our", detuning_sign=-1,
    enable_rydberg_decay=True,
    enable_intermediate_decay=True,
    enable_polarization_leakage=True,
)
protocol = TOProtocol()
budget = error_budget(system, protocol, X_TO_OUR_DARK)
```

### Scattering Decomposition (|0> vs |1>)

Separate |0> and |1> scattering contributions via budget differencing:

```python
sim_mid = CZGateSimulator(
    param_set="our", strategy="TO",
    blackmanflag=True, detuning_sign=-1,
    enable_intermediate_decay=True,
)
budget_mid = sim_mid.error_budget(X_TO_OUR_DARK)

sim_no0 = CZGateSimulator(
    param_set="our", strategy="TO",
    blackmanflag=True, detuning_sign=-1,
    enable_intermediate_decay=True,
    enable_0_scattering=False,
)
budget_no0 = sim_no0.error_budget(X_TO_OUR_DARK)

# |0> scattering = difference
xyz_scat0 = budget_mid["intermediate_decay"]["XYZ"] - budget_no0["intermediate_decay"]["XYZ"]

# |1> scattering = no-|0> case
xyz_scat1 = budget_no0["intermediate_decay"]["XYZ"]
```

### Monte Carlo Simulation (Stochastic Errors)

```python
sim_deph = CZGateSimulator(
    param_set="our", strategy="TO",
    blackmanflag=True, detuning_sign=-1,
    enable_rydberg_dephasing=True,
    sigma_detuning=130e3,
)

result = sim_deph.run_monte_carlo_simulation(
    X_TO_OUR_DARK,
    n_shots=1000,
    sigma_detuning=130e3,
    seed=42,
    compute_branching=True,
)

print(f"Infidelity: {result.mean_infidelity:.6e} +/- {result.std_infidelity:.6e}")
print(f"XYZ:   {result.mean_branch_XYZ:.6e}")
print(f"AL:    {result.mean_branch_AL:.6e}")
print(f"LG:    {result.mean_branch_LG:.6e}")
print(f"Phase: {result.mean_branch_phase:.6e}")

# Save / load for reproducibility
result.save_to_file("data/mc_dark_dephasing.txt")
loaded = MonteCarloResult.load_from_file("data/mc_dark_dephasing.txt")
```

### Position Error

```python
sigma_pos = (70e-9, 70e-9, 130e-9)  # (sigma_x, sigma_y, sigma_z) in meters

sim_pos = CZGateSimulator(
    param_set="our", strategy="TO",
    blackmanflag=True, detuning_sign=-1,
    enable_position_error=True,
    sigma_pos_xyz=sigma_pos,
)

result_pos = sim_pos.run_monte_carlo_simulation(
    X_TO_OUR_DARK,
    n_shots=1000,
    sigma_pos_xyz=sigma_pos,
    seed=42,
    compute_branching=True,
)
```

## Reproducing the SI Tables

### Step 1: Generate MC data (run once)

```bash
uv run python scripts/generate_mc_data.py --n-shots 1000
```

Creates 6 text files in `data/`:
- `mc_bright_dephasing.txt`, `mc_bright_position.txt`, `mc_bright_all.txt`
- `mc_dark_dephasing.txt`, `mc_dark_position.txt`, `mc_dark_all.txt`

### Step 2: Generate PDF tables

```bash
uv run python scripts/generate_si_tables.py
```

### Step 3: Verify

- XYZ + AL + LG + Phase ~ Total infidelity (within numerical precision)
- For deterministic errors: XYZ + AL + LG = total (no phase component)
- For MC errors: Phase component should be non-zero

## Implementation Notes

**Budget Differencing:** The `enable_0_scattering=False` flag disables only |0> scattering from intermediate states. The difference between the full and no-|0> budgets isolates the |0> contribution.

**Modular implementation:** The error budget is computed by `ryd_gate.analysis.gate_metrics.error_budget()`, which uses `population_evolution()` and `decay_integrate()` internally. The `CZGateSimulator.error_budget()` method delegates to this function.
