"""``exact_ode`` backend: alias-free intermediate-state population.

The adaptive ``exact_ode`` (scipy DOP853) integrator resolves the fast ~GHz
``|0>->e`` optical coherence internally with error control, so the spectator
``|0>`` only picks up the true off-resonant 6P admixture (peak ``n_e`` ~ 1e-4 at
this operating point). The deleted piecewise-``expm`` solvers froze the drive
over fixed steps and, when the optical phase was commensurate with the step
rate, aliased it into spurious resonant pumping of ``|0>`` into the 6P manifold
(peak ``n_e`` jumped ~40x, could exceed 1).

This pins the clean band the ODE backend guarantees, on a short single-atom CZ
pulse whose intermediate detuning puts the ``|0>->e`` gap at ~20 GHz — the
operating point where the old fixed-step solvers aliased worst. Needs ARC and
resolves ~1000 optical cycles adaptively on a 201-point trajectory, so it is the
slowest test in ``tests/backends`` (~40 s) — the GHz aliasing it guards against
has no cheaper surrogate. (dense == sparse matvec equivalence is pinned cheaply,
away from ARC, by ``test_hamiltonian_formats_agree``.)
"""

import numpy as np

import ryd_gate as rg
from ryd_gate import Register, RydbergSystem, level_structure
from ryd_gate.protocols import CZProtocol, blackman_pulse

# Delta_e (13.165 GHz) + clock hyperfine (6.835 GHz) puts the |0>->e gap at
# ~20 GHz, so the |0>->e optical cycle count over the gate is
# (20 GHz)(0.05 us) = 1000 — the point where the old fixed-step solvers aliased.
_DELTA_E_HZ = 13.165e9
_T_GATE = 0.05e-6
_RISE = 0.15 * _T_GATE
_OMEGA_420 = 2 * np.pi * 300e6
_OMEGA_1013 = 2 * np.pi * 500e6


def _cz_system():
    """Single-atom ``rb87_7_mp`` CZ pulse with a GHz intermediate detuning."""
    env = lambda t: blackman_pulse(t, _RISE, _T_GATE)
    return RydbergSystem(
        level_structure=level_structure("rb87_7_mp", ryd_level=70, magnetic_field_G=20.0),
        register=Register.chain(1),
        protocol=CZProtocol(
            t_gate_s=_T_GATE,
            intermediate_detuning_rad_s=2 * np.pi * _DELTA_E_HZ,
            omega_420_max_rad_s=_OMEGA_420,
            omega_1013_max_rad_s=_OMEGA_1013,
            envelope_420=env,
            phase_420_rad=lambda t: 0.0,
            envelope_1013=env,
            phase_1013_rad=lambda t: 0.0,
        ),
    )


def test_exact_ode_is_alias_free():
    """For the spectator ``|0>`` input the peak 6P population stays in the true
    off-resonant admixture band (< 1e-2), where the deleted expm solvers spiked
    above 0.1."""
    system = _cz_system()
    obs = system.observables
    n_e = obs.n("e1", 0) + obs.n("e2", 0) + obs.n("e3", 0)
    t_eval = np.linspace(0.0, _T_GATE, 201)

    dense = rg.simulate(
        system, ["0"], t_eval=t_eval, observables={"n_e": n_e},
        backend_options={"hamiltonian_format": "dense"},
    )

    assert dense.expectation("n_e").max() < 0.01
