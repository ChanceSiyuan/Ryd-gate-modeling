"""exact_ode backend: alias-free intermediate-state population.

The adaptive ``exact_ode`` (scipy DOP853) integrator resolves the fast ~GHz
``|0>->e`` / ``|1>->e`` optical coherence internally with error control, so the
spectator ``|0>`` only picks up the true off-resonant 6P admixture (peak
``n_e`` ~ 1e-3 at this operating point).  The deleted piecewise-``expm``
solvers froze the drive over fixed steps and, when the optical phase was
commensurate with the step rate, aliased it into spurious resonant pumping of
``|0>`` into the 6P manifold (peak ``n_e`` jumped ~40x, could exceed 1).  This
test pins the clean band the ODE backend guarantees, on a short single-atom
pulse (kept small so the run is a few seconds).
"""

import numpy as np

import ryd_gate as rg
from ryd_gate.lattice import Register
from ryd_gate.physics import our_laser_rabis
from ryd_gate.protocols import CZProtocol
from ryd_gate.protocols.gate_cz import cz_effective_rabi, phase_from_chirp

# Delta_e + eps0 = 13.165 + 6.835 = 20.0 GHz exactly, so the |0>->e optical
# cycle count over the gate is (20 GHz)(0.05 us) = 1000 — the operating point
# where the old fixed-step solvers aliased worst.
_DELTA_E_HZ = 13.165e9
_T_GATE = 0.05e-6
_D_SWEEP_HZ = 20e6
_OPTICS_LOSS = 0.9
_RYD_LEVEL = 70
_BEAM_AREA_UM2 = 7 * 20 * 3.0
# Fine fixed sampling for the chirp-integral 420 phase (interpolation accuracy
# only; the ODE solver itself is adaptive).
_N_CHIRP_SAMPLES = 4 * 1373 + 1


def _smooth_env(t, t_gate, ramp=0.15):
    s = float(np.clip(t / t_gate, 0.0, 1.0))
    q = lambda u: (lambda v: 10 * v**3 - 15 * v**4 + 6 * v**5)(np.clip(u, 0, 1))
    if s < ramp:
        return float(q(s / ramp))
    if s > 1 - ramp:
        return float(q((1 - s) / ramp))
    return 1.0


def _cz_system():
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

    phi = phase_from_chirp(chirp, t_gate=T, n_samples=_N_CHIRP_SAMPLES)
    system = rg.RydbergSystem(
        level_structure=rg.level_structure(
            "rb87_7_mp", detuning_sign=1, Delta_Hz=_DELTA_E_HZ, magnetic_field_G=20.0
        ),
        register=Register.chain(1, spacing_um=3.0),
    )
    _, time_scale = cz_effective_rabi(system, omega_420, omega_1013)
    proto = CZProtocol(
        duration_ratio=T / time_scale,
        A_420=lambda s: np.sqrt(env(float(np.clip(s, 0.0, 1.0)) * T)),
        phi_420=lambda s: phi(float(np.clip(s, 0.0, 1.0)) * T),
        A_1013=lambda s: np.sqrt(env(float(np.clip(s, 0.0, 1.0)) * T)),
        phi_1013=lambda s: 0.0,
        omega_420_max=omega_420, omega_1013_max=omega_1013,
    )
    return system.with_protocol(proto)


def _peak_ne_zero():
    """Peak intermediate-state population for the spectator |0> input."""
    system = _cz_system()
    obs = system.observables
    n_e = obs.level_sum("e1") + obs.level_sum("e2") + obs.level_sum("e3")
    t_eval = np.linspace(0.0, _T_GATE, 201)
    res = rg.simulate(system, [["0"]], t_eval=t_eval, observables={"n_e": n_e})
    r = res[0] if isinstance(res, list) else res
    ne = np.real(r.expectation("n_e"))
    return float(ne.max())


def test_exact_ode_is_alias_free():
    """The spectator |0> stays in the true off-resonant admixture band (< 1e-2)
    at the operating point where the deleted expm solvers spiked above 0.1."""
    ne_ode = _peak_ne_zero()
    assert ne_ode < 0.01
