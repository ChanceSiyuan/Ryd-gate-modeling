# Model

This page defines the physical object that `ryd-gate` simulates. Solver choices
and result semantics are documented in [Simulation](simulation.md); gate metrics
are documented in [Gates](gates.md).

## The four objects

Every calculation starts from four explicit pieces:

```python
from ryd_gate import Register, RydbergSystem, level_structure

system = RydbergSystem(
    level_structure=level_structure("1r", ryd_level=70),
    register=Register.chain(4, spacing_um=5.0),
    protocol=protocol,
    interaction_cutoff_um=None,
)
```

- `Register` says where the atoms are.
- `level_structure(...)` selects the atomic levels and their fixed physics.
- A protocol specifies every time-dependent drive in physical units.
- `RydbergSystem` binds those inputs and resolves the interacting Hamiltonian.

The system is immutable. `system.with_protocol(other)` returns a new system
with the same level structure, register, and interaction cutoff. The protocol
sets the duration; read it from the read-only `system.t_gate`.

## Units and labels

The package uses one convention throughout:

| quantity | unit |
|---|---|
| time | s |
| Hamiltonian coefficient and Rabi frequency | rad/s |
| position and distance | µm |
| decay rate | s⁻¹ |

Local states use physical string labels such as `"0"`, `"1"`, `"r"`, and
`"r_garb"`. Sites use integer indices `0, ..., system.N - 1`. The same site
order is used by initial-state labels, observables, amplitudes, and sampled
outcomes.

## Geometry: `Register`

`Register` contains only an `(N, 2)` coordinate array in µm. The exposed array
has its NumPy write flag disabled; treat it as immutable and copy it before
editing:

```python
import numpy as np
from ryd_gate import Register

custom = Register(np.array([[0.0, 0.0], [4.0, 1.0]]))
chain = Register.chain(6, spacing_um=4.0)
rectangle = Register.rectangle(3, 5, spacing_um=5.0)
square = Register.square(4, spacing_um=5.0)
triangular = Register.triangular(3, 5, spacing_um=5.0)

print(custom.coords)
print(custom.N)
```

Site `i` is exactly row `i` of `register.coords`. A factory's `spacing_um` is
only an input used to generate coordinates; it is not stored as a property.
Register has no atom IDs, sublattice field, interaction query, or plotting API.
Those meanings are analysis-specific and belong in the calling script.

The nominal register is strictly two-dimensional. Out-of-plane thermal motion
is represented by position noise during an ensemble calculation, not by adding
a third public coordinate column.

## Level-structure presets

Users obtain a level structure only through `level_structure(name, **kwargs)`.
There is no public custom-level DSL and no public C6 override.

| preset | levels | accepted physical arguments | default Rydberg state |
|---|---|---|---|
| `1r` | `1, r` | `ryd_level` | 70S |
| `01r` | `0, 1, r` | `ryd_level` | 70S |
| `rb87_7_mp` | `0, 1, e1, e2, e3, r, r_garb` | `ryd_level`, `magnetic_field_G` | 70S |
| `rb87_7_pm` | `0, 1, e1, e2, e3, r, r_garb` | `ryd_level`, `magnetic_field_G` | 53S |
| `rb87_297_clock_4` | `0, 1, r, r_garb` | `ryd_level`, `magnetic_field_G`, `quantization_axis` | 53P3/2 |

The resolved object exposes only physical characterization:

```python
ls.name
ls.levels
ls.ryd_level
ls.magnetic_field_G
ls.quantization_axis
ls.decay_rates_per_s
ls.branching_ratios
```

Decay data does not add non-Hermitian terms to the Hamiltonian. It is used for
first-order post-processing from explicitly requested populations; see
[Gates](gates.md#first-order-decay-budget).

The seven-level presets provide radiative branching data for `r`, `e1`, `e2`,
and `e3`, but not for `r_garb`. The direct-297 preset currently provides no
branching table. Missing source-level data is not inferred automatically.

## Pair interactions

The register supplies geometry and the preset supplies Rydberg-state physics.
For a pair at separation `R_ij`, the resolved interaction has the form

$$
V_{ij} = \frac{C_{6,ij}}{R_{ij}^6}.
$$

ARC is the only source of C6 coefficients. Changing `ryd_level` therefore
recomputes the interaction for that atomic state instead of retaining a
hard-coded value.

- `1r`, `01r`, `rb87_7_mp`, and `rb87_7_pm` use the current isotropic S-state
  model. In the seven-level model, `r` and `r_garb` are magnetic sublevels of
  the same nS state and share the pair coefficient.
- `rb87_297_clock_4` uses nP3/2 states. Its `rr`, `r-r_garb`, and
  `r_garb-r_garb` channels are computed separately and depend on the pair
  direction relative to `quantization_axis`.

`RydbergSystem(..., interaction_cutoff_um=...)` selects physical pairs:

```python
interaction_cutoff_um=None   # all pairs
interaction_cutoff_um=5.0    # nominal distance <= 5 µm
interaction_cutoff_um=0.0    # no pair interaction
```

There is no public `InteractionSpec`, `interaction_pairs()`, or
`interaction_strength()` API.

## Hamiltonian convention

All Hamiltonians are Hermitian and use `hbar = 1`:

$$
H(t) = H_{\mathrm{atom}} + H_{\mathrm{drive}}(t)
     + H_{\mathrm{pair}}.
$$

An off-diagonal protocol coefficient is the matrix element itself; the
compiler adds its Hermitian conjugate. It does not add another factor of
one-half. A diagonal coefficient is a real energy.

### `1r`: driven Rydberg lattice

`SweepProtocol` supplies `omega_half_rad_s(t) = Omega(t)/2`, the global Rydberg
detuning, and an optional per-site detuning:

$$
H(t) = \frac{\Omega(t)}{2}\sum_i
       \left(|r\rangle\langle1|_i + \mathrm{h.c.}\right)
       - \sum_i [\Delta(t)+\delta_i(t)]n_i^r
       + \sum_{i<j}V_{ij}n_i^r n_j^r.
$$

This is also the Rydberg representation of the transverse-field Ising model,
with `Omega/2 = h_x`. TFIM parameter conversion, sublattice weights, annealing
schedules, and critical-point analysis are explicit research-script logic, not
separate protocol classes.

### `01r`: effective three-level lattice

`SweepProtocol` drives only `1 <-> r`; `0` is a spectator. For full qutrit
control, `DigitalAnalogProtocol` provides five independent, possibly
site-dependent controls:

$$
\begin{aligned}
H_{\mathrm{drive}}(t)=\sum_i [&c_{10,i}|1\rangle\langle0|_i
 +c_{r0,i}|r\rangle\langle0|_i
 +c_{r1,i}|r\rangle\langle1|_i + \mathrm{h.c.}\\
 &+\epsilon_{1,i}n_i^1+\epsilon_{r,i}n_i^r].
\end{aligned}
$$

The corresponding constructor arguments are `coupling_10_rad_s`,
`coupling_r0_rad_s`, `coupling_r1_rad_s`, `energy_1_rad_s`, and
`energy_r_rad_s`. Each callable returns either one uniform value or a length-N
site profile. `|0>` is the energy gauge zero.

### Seven-level 420/1013-nm model

`rb87_7_mp` and `rb87_7_pm` contain the clock states, the 6P3/2 hyperfine
manifold, and the target/garbage nS Rydberg states. The preset fixes atomic
splittings, Zeeman shifts, dipole ratios, pair interactions, and decay data.
The bound `CZProtocol`, `TOProtocol`, or `ARProtocol` supplies the signed
intermediate detuning and the complete 420/1013-nm laser waveforms.

For a laser group `L`, the drive convention is

$$
\Omega_L(t)=\Omega_L^{\max}A_L(t)e^{-i\phi_L(t)}.
$$

The preset privately expands that physical laser onto all allowed transitions.
See [Gates](gates.md) for the gate-specific pulse families.

### Direct 297-nm model

`rb87_297_clock_4` uses a single 297-nm laser to drive `|1>` to the target
`|r>` state and, with a fixed dipole ratio, to `|r_garb>`. `|0>` is dark. The
same preset carries the channel-resolved nP3/2 pair interactions. Its pulse
families are described in [Gates](gates.md).

## Protocol rules

Protocols are imported from `ryd_gate.protocols` and are fully specified at
construction. No optimizer vector or runtime `x` is passed to `simulate()` or
`plot()`.

| protocol | compatible presets |
|---|---|
| `SweepProtocol` | `1r`, `01r` |
| `DigitalAnalogProtocol` | `01r` |
| `CZProtocol`, `TOProtocol`, `ARProtocol` | `rb87_7_mp`, `rb87_7_pm` |
| `Direct297PiProtocol`, `Direct297CZProtocol`, `Direct297TOProtocol` | `rb87_297_clock_4` |

Deterministic multi-stage controls should be written as one continuous
piecewise protocol. Backends assume matching interior values and permit
derivative kinks; protocol constructors do not sample a callable to prove this,
so avoiding true interior jumps is the caller's responsibility.

Every protocol provides the same input-only visualization:

```python
figure = system.protocol.plot(system, n_points=400)
axes = figure.axes
```

It does not run a simulation, call `show()`, or save a file.

Two public pulse-construction helpers are available:

```python
from ryd_gate.protocols import blackman_pulse, phase_from_chirp

envelope = lambda t: blackman_pulse(t, rise_time_s=20e-9, t_gate_s=300e-9)
phase = phase_from_chirp(chirp_rad_s, t_gate_s=300e-9, n_samples=1001)
```

Both use physical time. `phase_from_chirp` returns the optical phase
`phi(t) = integral_0^t chirp(t') dt'`.

## Physics helpers

Experimental and atomic inputs can be converted into Hamiltonian parameters
through the expert `ryd_gate.physics` module:

```python
from ryd_gate.physics import (
    arc_pair_c6_rad_s_um6,
    rb87_297_clock_rabi_frequencies,
    rb87_7_mp_rabi_frequencies,
    single_photon_rabi,
    zeeman_shift_rad_s,
)
```

These functions calculate forward physical quantities. They do not consume a
simulation result or perform gate/error analysis. Inspect their public
signatures with an IDE or `help(...)`; the functions themselves are the
authority for input validation and error messages.
