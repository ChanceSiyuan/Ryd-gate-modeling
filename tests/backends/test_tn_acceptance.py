"""Patch 5 H7 acceptance tests for the TN engines.

(a) off-grid anchor honesty — a requested measurement time that is not a
    multiple of ``dt`` is measured at that *physical* time (a separate
    reference evolution ending exactly there agrees), not at a rounded step;
(b) an endpoint-free ``t_eval`` never changes the final state;
(c) exact/MPS parity for local, pair, and product observables on a 2x2
    lattice with site-asymmetric addressing (site-ordering / snake-permutation
    parity);
(d) DigitalAnalog complex ``K10``/``Kr0``/``Kr1`` couplings keep exact/MPS
    parity on the explicit 01r qutrit path.
"""

import numpy as np
import pytest

from ryd_gate import RydbergSystem, level_structure
from ryd_gate.backends.tn_common._expressions import tn_basis
from ryd_gate.backends.tn_common.lattice_spec import create_tn_lattice_spec
from ryd_gate.backends.tn_common.simulate import simulate_tn
from ryd_gate.core.level_structures import InteractionSpec
from ryd_gate.core.observables import ObservableFactory
from ryd_gate.lattice import Register
from ryd_gate.protocols.digital_analog import DigitalAnalogProtocol
from ryd_gate.protocols.sweep import SweepProtocol
from ryd_gate.simulate import simulate

tenpy = pytest.importorskip("tenpy")


def test_off_grid_anchor_measures_true_physical_time():
    """A t_eval anchor between step multiples is hit exactly, not rounded."""
    spec = create_tn_lattice_spec(Lx=1, Ly=2, V_nn=4.0, interaction_mode="nn")
    obs = ObservableFactory(tn_basis(spec))
    observables = {"n_r": obs.level_sum("r")}

    def proto(t_gate):
        # time-dependent drive so the state at 0.5 and 0.6 differ measurably
        return SweepProtocol(
            t_gate=t_gate,
            omega_half_fn=lambda t: 0.5 * (1.0 + t),
            delta_fn=lambda t: 2.0 * t,
        )

    t_req, dt = 0.5, 0.3  # 0.5 is NOT a multiple of 0.3 (old code rounded to 0.6)
    opts = {"chi_max": 16, "dt": dt}
    res = simulate_tn(
        spec, proto(1.0), t_eval=np.array([t_req, 1.0]),
        observables=observables, backend_options=opts,
    )
    np.testing.assert_array_equal(res.times, [t_req, 1.0])

    # Reference evolution that ends exactly at the requested physical time.
    # Both runs segment [0, 0.5] into ceil(0.5/0.3)=2 equal steps, so they
    # share the identical Trotterization and must agree at solver precision.
    ref = simulate_tn(spec, proto(t_req), observables=observables, backend_options=opts)
    np.testing.assert_array_equal(ref.times, [t_req])
    np.testing.assert_allclose(
        res.expectation("n_r")[0], ref.expectation("n_r")[0], rtol=0, atol=1e-10
    )
    # non-vacuous: the observable really moves between the two anchors
    assert abs(res.expectation("n_r")[1] - res.expectation("n_r")[0]) > 1e-3


def test_endpoint_free_t_eval_final_state_is_true_t_gate_state():
    """t_eval selects measurement times only; final_state is always t_gate."""
    from ryd_gate.backends.tenpy_mps.state import mps_fidelity

    spec = create_tn_lattice_spec(Lx=1, Ly=2, V_nn=4.0, interaction_mode="nn")
    obs = ObservableFactory(tn_basis(spec))
    observables = {"n_r": obs.level_sum("r")}
    proto = SweepProtocol(
        t_gate=1.0, omega_half_fn=lambda t: 0.5, delta_fn=lambda t: 1.0
    )
    opts = {"chi_max": 16, "dt": 0.02, "svd_min": 1e-12}

    res_mid = simulate_tn(
        spec, proto, t_eval=np.array([0.37]), observables=observables,
        backend_options=opts,
    )
    res_end = simulate_tn(spec, proto, backend_options=opts)
    np.testing.assert_array_equal(res_mid.times, [0.37])

    # equal within truncation/Trotter tolerance (the [0, 0.37, 1.0] anchor
    # grid re-segments the stepping, so this is not bitwise)
    assert mps_fidelity(res_end.final_state, res_mid.final_state) > 1.0 - 1e-6

    # regression against the old bug class: the final state is NOT the state
    # at the last requested time
    proto_037 = SweepProtocol(
        t_gate=0.37, omega_half_fn=lambda t: 0.5, delta_fn=lambda t: 1.0
    )
    res_037 = simulate_tn(spec, proto_037, backend_options=opts)
    assert mps_fidelity(res_037.final_state, res_mid.final_state) < 0.99


def test_exact_mps_parity_local_pair_product_2x2():
    """Exact/MPS parity for local, pair, and product expressions.

    The per-site addressing detunings are all different, so any site
    transposition (e.g. a dropped snake permutation: register sites 2 and 3
    swap in MPS order on a 2x2 lattice) breaks the per-site comparison.
    """
    address = [0.0, 0.9, -0.6, 1.7]
    proto = SweepProtocol(
        t_gate=1.0,
        omega_half_fn=lambda t: 0.5,
        delta_fn=lambda t: 0.5,
        address_fn=lambda t, i: address[i],
    )
    system = RydbergSystem(
        level_structure=level_structure("1r"),
        register=Register.rectangle(2, 2, spacing_um=1.0),
        interaction=InteractionSpec(C6=4.0, mode="nn"),
        protocol=proto,
    )
    obs = system.observables
    observables = {f"n_r_{i}": obs.n("r", i) for i in range(4)}
    observables["pair_23"] = obs.n("r", 2) @ obs.n("r", 3)  # snake-transposed pair
    observables["pair_03"] = obs.n("r", 0) @ obs.n("r", 3)
    observables["proj_1111"] = obs.product_projector(["1", "1", "1", "1"])
    observables["coh_2"] = obs.E("1", "r", 2)  # complex, non-Hermitian
    t_eval = np.array([0.6, 1.0])

    res_exact = simulate(system, t_eval=t_eval, observables=observables)
    res_mps = simulate(
        system, backend="mps", t_eval=t_eval, observables=observables,
        backend_options={"chi_max": 32, "dt": 0.01, "svd_min": 1e-12},
    )

    np.testing.assert_array_equal(res_mps.times, res_exact.times)
    for label in observables:
        np.testing.assert_allclose(
            res_mps.expectation(label), res_exact.expectation(label),
            atol=5e-4, err_msg=label,
        )
    # non-vacuous site asymmetry: all four local populations are distinct
    finals = [res_exact.expectation(f"n_r_{i}")[-1].real for i in range(4)]
    assert len({round(v, 3) for v in finals}) == 4


def test_digital_analog_complex_couplings_exact_mps_parity():
    """Complex K10/Kr0/Kr1 matrix elements keep exact/MPS parity (01r)."""
    proto = DigitalAnalogProtocol(
        t_gate=1.0,
        coupling_r1=lambda t: [0.4, 0.3 + 0.2j],
        coupling_10=lambda t: 0.25 * np.exp(0.8j * t),
        coupling_r0=lambda t: [0.15j, 0.1],
        energy_r=lambda t: [-0.3, 0.2],
        energy_1=lambda t: 0.1,
    )
    system = RydbergSystem(
        level_structure=level_structure("01r"),
        register=Register.chain(2, spacing_um=1.0),
        interaction=InteractionSpec(C6=2.0, mode="nn"),
        protocol=proto,
    )
    obs = system.observables
    observables = {}
    for level in ("0", "1", "r"):
        observables.update(obs.site_populations(level))
    observables["coh_01_0"] = obs.E("0", "1", 0)  # complex coherence
    t_eval = np.array([0.5, 1.0])

    res_exact = simulate(system, t_eval=t_eval, observables=observables)
    res_mps = simulate(
        system, backend="mps", t_eval=t_eval, observables=observables,
        backend_options={"chi_max": 16, "dt": 0.01, "svd_min": 1e-12},
    )

    for label in observables:
        np.testing.assert_allclose(
            res_mps.expectation(label), res_exact.expectation(label),
            atol=5e-4, err_msg=label,
        )
    # the complex legs actually act: coherence has a nonzero imaginary part
    assert np.abs(res_exact.expectation("coh_01_0").imag).max() > 1e-3
