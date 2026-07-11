"""exact_ode backend: alias-free intermediate-state population.

The piecewise-constant ``expm`` backends freeze the drive over ``dt = t_gate/n_steps``
and undersample the fast ~GHz ``|0>->e`` / ``|1>->e`` optical coherence. When that
coherence is commensurate with the step rate --- ``(Delta_e + eps0) * t_gate / n_steps``
near an integer --- the aliased phase adds coherently and spuriously pumps the
spectator ``|0>`` into the ``6P`` manifold (peak ``n_e`` jumps ~40x, can exceed 1).
The adaptive ``exact_ode`` (scipy DOP853) integrator resolves that phase internally, so
it is immune. These tests pin the bug in the expm backend and the immunity in the ODE
backend on a short single-atom pulse (kept small so the ODE run is a few seconds).
"""

import numpy as np

import ryd_gate as rg
from ryd_gate.gates import CZProtocol, phase_from_chirp
from ryd_gate.lattice import Register
from ryd_gate.physics import our_laser_rabis

# Delta_e + eps0 = 13.165 + 6.835 = 20.0 GHz exactly, so the |0>->e optical cycle
# count over the gate is (20 GHz)(0.05 us) = 1000. n_steps that divide 1000 are
# commensurate (aliasing spikes); others are clean.
_DELTA_E_HZ = 13.165e9
_T_GATE = 0.05e-6
_D_SWEEP_HZ = 20e6
_OPTICS_LOSS = 0.9
_RYD_LEVEL = 70
_BEAM_AREA_UM2 = 7 * 20 * 3.0
# single-atom projector onto e1,e2,e3 in the [0,1,e1,e2,e3,r,r_garb] basis
_OCC_E = np.diag([0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0]).astype(complex)


def _smooth_env(t, t_gate, ramp=0.15):
    s = float(np.clip(t / t_gate, 0.0, 1.0))
    q = lambda u: (lambda v: 10 * v**3 - 15 * v**4 + 6 * v**5)(np.clip(u, 0, 1))
    if s < ramp:
        return float(q(s / ramp))
    if s > 1 - ramp:
        return float(q((1 - s) / ramp))
    return 1.0


def _cz_system(n_steps):
    """Single-atom rb87_7 CZ pulse (quintic envelope + Stark-compensated chirp)."""
    T = _T_GATE
    Delta_e = 2 * np.pi * _DELTA_E_HZ
    D_sweep = 2 * np.pi * _D_SWEEP_HZ
    omega_420, omega_1013 = our_laser_rabis(
        6.41 * (1 - _OPTICS_LOSS), 100.0 * (1 - _OPTICS_LOSS),
        beam_area=_BEAM_AREA_UM2, ryd_level=_RYD_LEVEL,
    )
    env = lambda t: _smooth_env(t, T)
    base_chirp = lambda t: -D_sweep * np.cos(2.0 * np.pi * t / T)
    d1 = -(4.0 / 3.0) * omega_420**2 / (4.0 * Delta_e)
    dr = -(omega_1013**2) / (4.0 * Delta_e)

    def chirp(t):
        a = np.sqrt(env(t))
        return base_chirp(t) + dr * a * a - d1 * a * a

    phi = phase_from_chirp(chirp, t_gate=T, n_samples=4 * n_steps + 1)
    proto = CZProtocol(
        t_gate=T,
        A_420=lambda s: np.sqrt(env(float(np.clip(s, 0.0, 1.0)) * T)),
        phi_420=lambda s: phi(float(np.clip(s, 0.0, 1.0)) * T),
        A_1013=lambda s: np.sqrt(env(float(np.clip(s, 0.0, 1.0)) * T)),
        phi_1013=lambda s: 0.0,
        omega_420_max=omega_420, omega_1013_max=omega_1013, n_steps=n_steps,
    )
    return (
        rg.RydbergSystem
        .set_atom_level("rb87_7_mp", detuning_sign=1, Delta_Hz=_DELTA_E_HZ, magnetic_field_G=20.0)
        .set_atom_geom(Register.chain(1, spacing_um=3.0))
        .with_protocol(proto)
    )


def _peak_ne_zero(backend, n_steps):
    """Peak intermediate-state population for the spectator |0> input."""
    system = _cz_system(n_steps)
    t_eval = np.linspace(0.0, _T_GATE, 201)
    opts = {"n_steps": n_steps} if backend != "exact_ode" else None
    res = rg.simulate(system, psi0=[["0"]], backend=backend, t_eval=t_eval, backend_options=opts)
    r = res[0] if isinstance(res, list) else res
    ne = np.array([np.real(np.vdot(p, _OCC_E @ p)) for p in r.states])
    return float(ne.max())


def test_expm_aliasing_spike_at_commensurate_n_steps():
    """The bug: commensurate n_steps (100, m=10) spikes; non-commensurate (137) is clean."""
    ne_commensurate = _peak_ne_zero("exact_dense", 100)
    ne_clean = _peak_ne_zero("exact_dense", 137)
    assert ne_commensurate > 0.1      # spurious resonant pumping of |0> into 6P
    assert ne_clean < 0.01
    assert ne_commensurate > 20 * ne_clean


def test_exact_ode_is_alias_free():
    """The fix: exact_ode stays small at an n_steps where expm spikes, and matches
    the non-commensurate expm reference (the true off-resonant admixture)."""
    ne_ode = _peak_ne_zero("exact_ode", 100)          # would spike on expm
    ne_expm_clean = _peak_ne_zero("exact_dense", 137)  # true value
    assert ne_ode < 0.01
    assert abs(ne_ode - ne_expm_clean) < 3e-3


def test_exact_ode_matches_converged_expm_final_state():
    """Correctness: as the expm step count grows the piecewise-constant solver converges
    to the ODE result. At a well-resolved, non-commensurate n_steps the driven |1> final
    state agrees to ~1e-7 (fidelity 0.99896 at n_steps=137 -> 0.99999994 at 1373)."""
    system = _cz_system(1373)
    r_ode = rg.simulate(system, psi0=[["1"]], backend="exact_ode")
    r_expm = rg.simulate(system, psi0=[["1"]], backend="exact_dense",
                         backend_options={"n_steps": 1373})
    a = (r_ode[0] if isinstance(r_ode, list) else r_ode).psi_final
    b = (r_expm[0] if isinstance(r_expm, list) else r_expm).psi_final
    fidelity = abs(np.vdot(a, b)) ** 2 / (np.vdot(a, a).real * np.vdot(b, b).real)
    assert fidelity > 0.9999
