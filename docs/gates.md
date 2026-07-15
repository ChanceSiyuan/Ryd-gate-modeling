# Gates

`ryd-gate` supplies microscopic systems, fully specified pulses, and raw quantum
evolution. It deliberately does not supply a gate-report or optimizer API. A
gate workflow therefore remains visible in the calling script:

1. choose a physical model and pulse protocol;
2. inspect the pulse;
3. evolve the computational basis states;
4. calculate fidelity, phase, leakage, and error budgets explicitly.

General model conventions are in [Model](model.md); solver and result semantics
are in [Simulation](simulation.md).

## Choose a gate protocol

| physical model | protocols | purpose |
|---|---|---|
| `rb87_7_mp`, `rb87_7_pm` | `CZProtocol`, `TOProtocol`, `ARProtocol` | microscopic seven-level 420/1013-nm gates |
| `rb87_297_clock_4` | `Direct297PiProtocol`, `Direct297CZProtocol`, `Direct297TOProtocol` | direct single-photon 297-nm excitation and gates |

Effective gate models can instead be written on `01r` with
`DigitalAnalogProtocol`. Backend compatibility is documented in
[Simulation](simulation.md#backends).

Import protocols from their dedicated module:

```python
from ryd_gate.protocols import (
    ARProtocol,
    CZProtocol,
    Direct297CZProtocol,
    Direct297PiProtocol,
    Direct297TOProtocol,
    TOProtocol,
    blackman_pulse,
    phase_from_chirp,
)
```

- `CZProtocol` represents an arbitrary 420/1013-nm waveform.
- `TOProtocol` is the fixed time-optimal cosine-phase family.
- `ARProtocol` is the fixed amplitude-robust dual-sine family.
- An adiabatic rapid-passage waveform is `CZProtocol + phase_from_chirp()`;
  there is no separate ARP class.
- `Direct297PiProtocol` automatically calibrates the target-branch pi area.
- `Direct297CZProtocol` represents an arbitrary fixed-duration 297-nm pulse.
- `Direct297TOProtocol` is the fixed 297-nm cosine-phase family.

Inspect constructor signatures with an IDE or `help(...)`; constructors are the
authority for validation and error messages.

## Pulse conventions

Use the repository-wide [unit conventions](model.md#units-and-labels). For the
seven-level gate family,

$$
\Omega_{420}(t)=\Omega_{420}^{\max}A_{420}(t)e^{-i\phi_{420}(t)},
\qquad
\Omega_{1013}(t)=\Omega_{1013}^{\max}A_{1013}(t)e^{-i\phi_{1013}(t)}.
$$

The protocol owns both peak Rabi frequencies, the signed intermediate
detuning, its duration, and every waveform. The level-structure preset owns
atomic splittings and fixed transition-strength ratios.

For TO and AR, define

$$
\Omega_{\mathrm{eff}}
=\frac{\Omega_{420}^{\max}\Omega_{1013}^{\max}}{2|\Delta_e|}.
$$

Their dimensionless family coordinates resolve as

$$
t_{\mathrm{gate}}=\mathrm{duration\_ratio}\frac{2\pi}{\Omega_{\mathrm{eff}}},
\quad
\omega_{\mathrm{mod}}=\mathrm{modulation\_frequency\_ratio}\,\Omega_{\mathrm{eff}},
\quad
\delta=\mathrm{frequency\_offset\_ratio}\,\Omega_{\mathrm{eff}}.
$$

The 420-nm phase is

$$
\phi_{\mathrm{TO}}(t)=A_\phi\cos(\omega_{\mathrm{mod}}t+\phi_0)+\delta t
$$

or

$$
\phi_{\mathrm{AR}}(t)=
A_1\sin(\omega_{\mathrm{mod}}t+\phi_1)
+A_2\sin(2\omega_{\mathrm{mod}}t+\phi_2)+\delta t.
$$

Both families always use a Blackman rise/fall on the 420-nm leg and a constant
1013-nm leg. Generic `CZProtocol` callables instead receive physical time over
`[0, t_gate_s]`; omitted 1013 envelope/phase mean one/zero.

For Direct-297, `omega_297_max_rad_s` always means the target `1 <-> r`
branch. The garbage-branch ratio remains fixed by the preset. The Pi protocol
uses

$$
t_{\mathrm{gate}}=
\frac{\pi}{\Omega_{297}^{\max}\int_0^1 A(s)\,ds},
$$

while Direct297TO uses `omega_297_max_rad_s` as the scale for its duration and
phase ratios.

## Build and inspect a two-photon gate

The rounded values below illustrate the constructor shape; they are not an
optimized gate or a fidelity claim:

```python
import numpy as np

from ryd_gate import Register, RydbergSystem, level_structure, simulate
from ryd_gate.protocols import TOProtocol

protocol = TOProtocol(
    intermediate_detuning_rad_s=2 * np.pi * 9.1e9,
    omega_420_max_rad_s=2 * np.pi * 491e6,
    omega_1013_max_rad_s=2 * np.pi * 185e6,
    rise_time_s=20e-9,
    phase_amplitude_rad=-0.7,
    modulation_frequency_ratio=1.0,
    phase_offset_rad=0.3,
    frequency_offset_ratio=1.5,
    duration_ratio=1.4,
)

system = RydbergSystem(
    level_structure=level_structure(
        "rb87_7_mp",
        ryd_level=70,
        magnetic_field_G=20.0,
    ),
    register=Register.chain(2, spacing_um=3.0),
    protocol=protocol,
)

figure = protocol.plot(system, n_points=400)
```

`plot()` returns one Matplotlib `Figure`. It only resolves and draws the input
controls; it does not evolve a state, display the figure, or save a file.

### Arbitrary chirped waveform

If `chi(t)` is an instantaneous chirp, its optical phase is

$$
\phi(t)=\int_0^t\chi(t')\,dt'.
$$

```python
from ryd_gate.protocols import CZProtocol, blackman_pulse, phase_from_chirp

t_gate = 0.54e-6
rise = 0.05e-6
d_chirp = 2 * np.pi * 20e6

def chirp(t):
    return -d_chirp * np.cos(2 * np.pi * t / t_gate)

phase_420 = phase_from_chirp(chirp, t_gate_s=t_gate, n_samples=4001)

chirped = CZProtocol(
    t_gate_s=t_gate,
    intermediate_detuning_rad_s=2 * np.pi * 9.1e9,
    omega_420_max_rad_s=2 * np.pi * 491e6,
    omega_1013_max_rad_s=2 * np.pi * 185e6,
    envelope_420=lambda t: blackman_pulse(t, rise, t_gate),
    phase_420_rad=phase_420,
)

chirped_system = system.with_protocol(chirped)
```

`n_samples` controls construction of the phase callable, not ODE integration.

### Direct 297-nm Pi pulse

Convert experimental power to target/garbage Rabi frequencies explicitly, then
pass only the target branch to the protocol:

```python
from ryd_gate.physics import rb87_297_clock_rabi_frequencies
from ryd_gate.protocols import Direct297PiProtocol

omega_target, omega_garbage = rb87_297_clock_rabi_frequencies(
    power_297_w=1e-3,
    beam_area_um2=100.0,
    ryd_level=53,
)

system_297 = RydbergSystem(
    level_structure=level_structure(
        "rb87_297_clock_4",
        ryd_level=53,
        magnetic_field_G=20.0,
    ),
    register=Register.chain(1),
    protocol=Direct297PiProtocol(
        omega_297_max_rad_s=omega_target,
        rise_fraction=0.15,
    ),
)
```

See `scripts/notebooks/single_photon.ipynb` for a complete 297-nm comparison.

## Evolve the logical basis

A general two-qubit audit evolves all four inputs:

```python
logical = {
    "00": ["0", "0"],
    "01": ["0", "1"],
    "10": ["1", "0"],
    "11": ["1", "1"],
}
order = ("00", "01", "10", "11")

batch = simulate(system, [logical[key] for key in order])
results = dict(zip(order, batch))

K = np.empty((4, 4), dtype=complex)
for column, input_key in enumerate(order):
    for row, output_key in enumerate(order):
        K[row, column] = results[input_key].amplitude(logical[output_key])
```

Here `K[b, a] = <b|U(t_gate)|a>` is the logical block, including its complex
phases and any loss of weight to non-logical levels.

## Nielsen average fidelity

In the ordered basis `(00, 01, 10, 11)`, let

$$
U_{\mathrm{CZ}}=\operatorname{diag}(1,1,1,-1).
$$

For local output-Z phase corrections `theta_a` and `theta_b`, define

$$
C_Z=\operatorname{diag}
\left(1,e^{-i\theta_b},e^{-i\theta_a},
e^{-i(\theta_a+\theta_b)}\right),
\qquad M=U_{\mathrm{CZ}}^\dagger C_ZK.
$$

The computational-subspace average fidelity, including coherent leakage, is

$$
F_{\mathrm{avg}}=
\frac{\operatorname{Tr}(M^\dagger M)+|\operatorname{Tr}M|^2}{4(4+1)}.
$$

One deterministic diagnostic takes the single-excitation return phases
relative to `00`:

```python
diagonal = np.diag(K)
theta_b = np.angle(diagonal[1]) - np.angle(diagonal[0])
theta_a = np.angle(diagonal[2]) - np.angle(diagonal[0])

local_z = np.diag([
    1.0,
    np.exp(-1j * theta_b),
    np.exp(-1j * theta_a),
    np.exp(-1j * (theta_a + theta_b)),
])
ideal_cz = np.diag([1.0, 1.0, 1.0, -1.0])
M = ideal_cz.conj().T @ local_z @ K

fidelity = (np.vdot(M, M).real + abs(np.trace(M)) ** 2) / 20.0
```

For an experimental gate, use the correction phases that will actually be
applied. During optimization they may instead be explicit scoring parameters.
They do not belong in the protocol.

If exchange symmetry has been independently verified so that the `10` and `01`
responses are equal, the three-state `00/01/11` formula is a valid shortcut. It
is not the general definition and should not silently replace the four-state
audit.

## Conditional phase and leakage

The raw conditional phase is independent of local-Z gauge:

$$
\Phi_{ZZ}=\arg K_{11,11}-\arg K_{10,10}
-\arg K_{01,01}+\arg K_{00,00}.
$$

For a CZ target, report the wrapped error `wrap(Phi_ZZ - pi)`:

```python
def wrap_phase(angle):
    return float(np.angle(np.exp(1j * angle)))

phase = np.angle(np.diag(K))
phase_error = wrap_phase(phase[3] - phase[2] - phase[1] + phase[0] - np.pi)
```

For logical input `a`, total coherent leakage from the computational subspace is

$$
L_a=1-\sum_{b\in\{00,01,10,11\}}|K_{b,a}|^2.
$$

```python
leakage = {
    key: 1.0 - float(np.vdot(K[:, column], K[:, column]).real)
    for column, key in enumerate(order)
}
```

For larger systems, explicitly request a logical-subspace projector expectation
instead of enumerating exponentially many amplitudes.

## First-order decay budget

All library Hamiltonians are Hermitian. There is no non-Hermitian decay overlay
and no norm-loss interpretation. In the weak-decay approximation, integrate the
coherent level populations:

$$
p_k^{(1)}=\Gamma_k\int_0^{t_{\mathrm{gate}}}
\langle n_k(t)\rangle_{\Gamma=0}\,dt.
$$

```python
t_eval = np.linspace(0.0, system.t_gate, 401)
obs = system.observables
n_r = sum(obs.n("r", site) for site in range(system.N))

trajectory = simulate(
    system,
    ["1", "1"],
    t_eval=t_eval,
    observables={"n_r": n_r},
)

population_time = np.trapz(
    trajectory.expectation("n_r"),
    trajectory.times,
)
rates = system.level_structure.decay_rates_per_s["r"]
p_radiative = rates["radiative"] * population_time
p_blackbody = rates["blackbody"] * population_time
```

Apply `branching_ratios` only when the source level is present in the preset's
[branching-data schema](model.md#level-structure-presets), and only to
radiative decay. Missing `r_garb` or direct-297 data must not be silently
invented; reusing another level's fractions is a script-level approximation
that should be named as such. Repeat the population integral for every relevant
level. Residual endpoint population is a separate coherent error. If the
integrated probability is not small, this first-order treatment is insufficient
and a genuine open-system model is needed.

## Noisy gates and optimization

`simulate_ensemble()` applies the same realization to every logical input in a
shot. Reconstruct `K`, apply a fixed nominal local-Z calibration, calculate one
gate metric per shot, and then compute the desired statistics in the script.
The library intentionally does not aggregate a gate report.

An optimizer similarly owns its vector layout, bounds, local-Z scoring phases,
objective, restarts, checkpointing, and interpretation. Each evaluation should:

1. translate its vector into named protocol arguments;
2. construct one fully specified protocol;
3. bind it with `system.with_protocol(protocol)`;
4. evolve the required inputs;
5. calculate the chosen metric explicitly.

There is no runtime `x`, protocol optimizer metadata, `cz_gate_report()`, or
`error_budget()` helper.

## Effective-theory comparisons

The expert module `ryd_gate.core.effective_theory` contains research tools for
comparing a concrete seven-level pulse with an effective `01r` Hamiltonian. It
is not part of the six-name top-level API. Effective populations and especially
the interacting ZZ phase must be checked against the full model in the intended
drive/blockade regime; a formal elimination is not a guarantee that a strong
drive remains quantitatively perturbative.

The working comparison is in `scripts/notebooks/find_phase.ipynb`. For routine
gate work, prefer the full microscopic model when seven-level accuracy is
required, or specify `01r + DigitalAnalogProtocol` directly when the effective
Hamiltonian is itself the intended model.

## Research entry points

- `examples/demo_cz_gate.py`: complete TO gate with inline metrics.
- `examples/demo_noise_model.py`: noisy gate ensemble.
- `scripts/notebooks/01_cz_gate.ipynb`: populations and decay budgets.
- `scripts/notebooks/single_photon.ipynb`: Direct-297 comparison.
- `scripts/optimize_ar_cz.py`: AR optimization.
- `scripts/diagnose_ar_target.py`: symmetry and target diagnostics.
- `scripts/gen_error_budget_g20.py`: error-budget maps.
- `scripts/max_leakage_ode_sweep.py`: specialized large leakage scan.

Specific optimum values and benchmark outcomes belong to those scripts and
their result files, not to this stable guide.
