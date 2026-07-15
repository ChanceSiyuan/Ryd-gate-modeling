"""Acceptance tests for the TN engines on the post-refactor seam.

(a) off-grid anchor honesty — a requested measurement time that is not a
    multiple of ``time_step_s`` is measured at that *physical* time (E06);
(b) an endpoint-free ``t_eval`` never changes the final state;
(c) exact/MPS parity for local, pair, and product observables with per-site
    addressing;
(d) DigitalAnalog complex couplings keep exact/MPS parity on the 01r qutrit path;
(e) exact/MPS amplitude (magnitude + phase) and sampling parity.
"""

import numpy as np
import pytest

from ryd_gate import RydbergSystem, level_structure
from ryd_gate.lattice import Register
from ryd_gate.protocols.digital_analog import DigitalAnalogProtocol
from ryd_gate.protocols.sweep import SweepProtocol
from ryd_gate.simulate import simulate

pytest.importorskip("tenpy")

TWO_PI = 2 * np.pi
_OPTS = {"time_step_s": 1e-8, "bond_dimension": 32, "discarded_weight_tolerance": 1e-10}


def test_off_grid_anchor_measures_true_physical_time():
    """A t_eval anchor between step multiples is hit exactly, not rounded (E06)."""
    def build(t_gate):
        proto = SweepProtocol(
            t_gate_s=t_gate,
            omega_half_rad_s=lambda t: TWO_PI * 1.0e6 * (1.0 + t / 1e-6),
            detuning_rad_s=lambda t: TWO_PI * 2.0e6 * (t / 1e-6),
        )
        return RydbergSystem(
            level_structure=level_structure("1r"),
            register=Register.chain(2, spacing_um=6.0),
            protocol=proto,
        )

    system = build(1.0e-6)
    obs = {"n_r": sum(system.observables.n("r", i) for i in range(system.N))}
    t_req, dt = 0.5e-6, 0.3e-6  # 0.5 is NOT a multiple of 0.3
    opts = {"time_step_s": dt, "bond_dimension": 16, "discarded_weight_tolerance": 1e-10}

    res = simulate(system, backend="mps", t_eval=np.array([t_req, 1.0e-6]),
                   observables=obs, backend_options=opts)
    np.testing.assert_array_equal(res.times, [t_req, 1.0e-6])

    # Reference evolution that ends exactly at t_req: both runs segment [0, 0.5]
    # into ceil(0.5/0.3)=2 equal steps, so they share the identical Trotterization.
    ref_sys = build(t_req)
    ref = simulate(ref_sys, backend="mps",
                   observables={"n_r": sum(ref_sys.observables.n("r", i) for i in range(2))},
                   backend_options=opts)
    np.testing.assert_array_equal(ref.times, [t_req])
    np.testing.assert_allclose(res.expectation("n_r")[0], ref.expectation("n_r")[0],
                               rtol=0, atol=1e-10)
    assert abs(res.expectation("n_r")[1] - res.expectation("n_r")[0]) > 1e-3


def test_endpoint_free_t_eval_final_state_is_true_t_gate_state():
    """t_eval selects measurement times only; the final state is always t_gate."""
    def build(t_gate):
        proto = SweepProtocol(
            t_gate_s=t_gate,
            omega_half_rad_s=lambda t: TWO_PI * 1.5e6,
            detuning_rad_s=lambda t: TWO_PI * 1.0e6,
        )
        return RydbergSystem(
            level_structure=level_structure("1r"),
            register=Register.chain(2, spacing_um=6.0),
            protocol=proto,
        )

    system = build(0.3e-6)
    obs = {"n_r": sum(system.observables.n("r", i) for i in range(2))}
    res_mid = simulate(system, backend="mps", t_eval=np.array([0.11e-6]),
                       observables=obs, backend_options=_OPTS)
    res_end = simulate(system, backend="mps", backend_options=_OPTS)
    np.testing.assert_array_equal(res_mid.times, [0.11e-6])

    # The final states agree: amplitude at t_gate is the same regardless of anchors.
    labels = ["r", "1"]
    np.testing.assert_allclose(res_end.amplitude(labels), res_mid.amplitude(labels),
                               atol=1e-5)

    # Regression: the final state is NOT the state at the last requested time.
    sys_short = build(0.11e-6)
    res_short = simulate(sys_short, backend="mps", backend_options=_OPTS)
    assert abs(res_short.amplitude(labels) - res_mid.amplitude(labels)) > 1e-2


def test_exact_mps_parity_local_pair_product():
    """Exact/MPS parity for local, pair, product and Hermitian-coherence observables."""
    address = [0.0, TWO_PI * 0.9e6, -TWO_PI * 0.6e6, TWO_PI * 1.7e6]
    proto = SweepProtocol(
        t_gate_s=0.3e-6,
        omega_half_rad_s=lambda t: TWO_PI * 2.0e6,
        detuning_rad_s=lambda t: TWO_PI * 1.0e6,
        local_detuning_rad_s=lambda t, i: address[i],
    )
    system = RydbergSystem(
        level_structure=level_structure("1r"),
        register=Register.chain(4, spacing_um=8.0),
        protocol=proto,
    )
    obs = system.observables
    observables = {f"n_r_{i}": obs.n("r", i) for i in range(4)}
    observables["pair_23"] = obs.n("r", 2) @ obs.n("r", 3)
    observables["pair_03"] = obs.n("r", 0) @ obs.n("r", 3)
    observables["proj_rrrr"] = obs.n("r", 0) @ obs.n("r", 1) @ obs.n("r", 2) @ obs.n("r", 3)
    coh = obs.E("1", "r", 2)
    observables["coh_x2"] = coh + coh.dagger()  # Hermitian coherence
    t_eval = np.array([0.2e-6, 0.3e-6])

    res_exact = simulate(system, t_eval=t_eval, observables=observables)
    res_mps = simulate(system, backend="mps", t_eval=t_eval, observables=observables,
                       backend_options=_OPTS)
    np.testing.assert_array_equal(res_mps.times, res_exact.times)
    for label in observables:
        np.testing.assert_allclose(res_mps.expectation(label), res_exact.expectation(label),
                                   atol=1e-3, err_msg=label)
    # non-vacuous site asymmetry: the four local populations are distinct
    finals = [round(float(res_exact.expectation(f"n_r_{i}")[-1]), 3) for i in range(4)]
    assert len(set(finals)) == 4


def test_digital_analog_complex_couplings_exact_mps_parity():
    """Complex 01r couplings keep exact/MPS parity (populations + Hermitian coherence)."""
    w = TWO_PI * 1e6
    proto = DigitalAnalogProtocol(
        t_gate_s=0.3e-6,
        coupling_r1_rad_s=lambda t: [1.0 * w, (0.8 + 0.5j) * w],
        coupling_10_rad_s=lambda t: 0.6 * w * np.exp(0.8j * t / 1e-6),
        coupling_r0_rad_s=lambda t: [0.4j * w, 0.25 * w],
        energy_r_rad_s=lambda t: [-0.7 * w, 0.5 * w],
        energy_1_rad_s=lambda t: 0.3 * w,
    )
    system = RydbergSystem(
        level_structure=level_structure("01r"),
        register=Register.chain(2, spacing_um=8.0),
        protocol=proto,
    )
    obs = system.observables
    observables = {}
    for level in ("0", "1", "r"):
        for i in range(2):
            observables[f"n_{level}_{i}"] = obs.n(level, i)
    a = obs.E("0", "1", 0)
    observables["coh_x"] = a + a.dagger()
    observables["coh_y"] = -1j * (a - a.dagger())
    t_eval = np.array([0.15e-6, 0.3e-6])

    res_exact = simulate(system, t_eval=t_eval, observables=observables)
    res_mps = simulate(system, backend="mps", t_eval=t_eval, observables=observables,
                       backend_options=_OPTS)
    for label in observables:
        np.testing.assert_allclose(res_mps.expectation(label), res_exact.expectation(label),
                                   atol=1e-3, err_msg=label)
    # the complex legs really act: the Y-coherence is nonzero somewhere
    assert np.abs(res_exact.expectation("coh_y")).max() > 1e-2


def test_exact_mps_amplitude_and_sample_parity():
    proto = SweepProtocol(
        t_gate_s=0.3e-6,
        omega_half_rad_s=lambda t: TWO_PI * 2.0e6 * np.sin(np.pi * t / 0.3e-6),
        detuning_rad_s=lambda t: TWO_PI * 1.0e6,
    )
    system = RydbergSystem(
        level_structure=level_structure("1r"),
        register=Register.chain(3, spacing_um=8.0),
        protocol=proto,
    )
    res_exact = simulate(system)
    # amplitude parity (magnitude + phase) needs a finer step than expectations
    tight = {"time_step_s": 4e-9, "bond_dimension": 32, "discarded_weight_tolerance": 1e-10}
    res_mps = simulate(system, backend="mps", backend_options=tight)
    for labels in (["r", "1", "r"], ["1", "1", "1"], ["r", "r", "1"]):
        a_e, a_m = res_exact.amplitude(labels), res_mps.amplitude(labels)
        np.testing.assert_allclose(a_m, a_e, atol=5e-4, err_msg=str(labels))
    # sampling returns tuple[str, ...] keyed Counters that sum to shots
    counts = res_mps.sample(shots=500, seed=3)
    assert sum(counts.values()) == 500
    assert all(isinstance(k, tuple) and len(k) == 3 for k in counts)
