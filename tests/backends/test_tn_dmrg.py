"""DMRG ground-state solver tests on the new system-based seam (E12-E21).

``system.ground_state(at, method="dmrg", initial_state, observables, method_options)``
-> GroundStateResult with expectation("energy") + observables + amplitude + sample.
"""

import numpy as np
import pytest
from scipy.sparse.linalg import eigsh

from ryd_gate import RydbergSystem, level_structure
from ryd_gate.backends.exact.compiler import compile_exact
from ryd_gate.lattice import Register
from ryd_gate.protocols.sweep import SweepProtocol

pytest.importorskip("tenpy")

TWO_PI = 2 * np.pi
_AT = 0.15e-6
_DMRG_OPTS = {
    "bond_dimension": 32, "discarded_weight_tolerance": 1e-10,
    "relative_energy_tolerance": 1e-9, "entropy_tolerance": 1e-7, "max_sweeps": 30,
}


def _system(n=6, spacing=8.0):
    proto = SweepProtocol(
        t_gate_s=0.3e-6,
        omega_half_rad_s=lambda t: TWO_PI * 2.0e6,
        detuning_rad_s=lambda t: TWO_PI * 1.5e6,
    )
    return RydbergSystem(level_structure=level_structure("1r"),
                         register=Register.chain(n, spacing_um=spacing),
                         protocol=proto, interaction_cutoff_um=9.0)


def _exact_ground_energy(system, at):
    ham, _ = compile_exact(system, hamiltonian_format="dense")
    dim = system._basis.total_dim
    H = np.zeros((dim, dim), dtype=complex)
    for k in range(dim):
        e = np.zeros(dim, complex)
        e[k] = 1.0
        H[:, k] = ham.apply(at, e)
    return float(eigsh(H, k=1, which="SA", return_eigenvectors=False)[0])


def test_dmrg_energy_matches_exact():
    system = _system(n=6)
    res = system.ground_state(
        at=_AT, method="dmrg",
        initial_state=["1", "r", "1", "r", "1", "r"],
        observables={"n_r": sum(system.observables.n("r", i) for i in range(6))},
        method_options=_DMRG_OPTS,
    )
    e_exact = _exact_ground_energy(system, _AT)
    np.testing.assert_allclose(res.expectation("energy"), e_exact, rtol=1e-7)
    assert isinstance(res.expectation("n_r"), float)


def test_dmrg_amplitude_phase_reference_is_real_positive():
    system = _system(n=4)
    res = system.ground_state(
        at=_AT, method="dmrg", initial_state=["1", "r", "1", "r"],
        method_options=_DMRG_OPTS,
    )
    ref = ["1", "1", "1", "1"]
    a_ref = res.amplitude(ref, phase_reference=ref)
    assert abs(a_ref.imag) < 1e-9 and a_ref.real >= 0.0
    # sampling returns tuple-keyed counts
    counts = res.sample(shots=200, seed=0)
    assert sum(counts.values()) == 200
    assert all(isinstance(k, tuple) and len(k) == 4 for k in counts)


def test_dmrg_reserved_energy_observable_rejected():
    system = _system(n=4)
    with pytest.raises(ValueError, match="reserved"):
        system.ground_state(
            at=_AT, method="dmrg", initial_state=["1", "1", "1", "1"],
            observables={"energy": system.observables.n("r", 0)},
            method_options=_DMRG_OPTS,
        )


def test_dmrg_option_schema_enforced():
    system = _system(n=4)
    with pytest.raises(ValueError, match="unknown"):
        system.ground_state(at=_AT, method="dmrg", initial_state=["1"] * 4,
                            method_options={**_DMRG_OPTS, "svd_min": 1e-12})
    with pytest.raises(ValueError, match="missing"):
        system.ground_state(at=_AT, method="dmrg", initial_state=["1"] * 4,
                            method_options={"bond_dimension": 8})


def test_dmrg_requires_1r_and_explicit_initial_state():
    system = _system(n=4)
    with pytest.raises(TypeError, match="explicit level-label"):
        system.ground_state(at=_AT, method="dmrg", initial_state=None,
                            method_options=_DMRG_OPTS)
    with pytest.raises(ValueError, match=r"\[0, t_gate"):
        system.ground_state(at=1.0, method="dmrg", initial_state=["1"] * 4,
                            method_options=_DMRG_OPTS)

    # 01r is not a ground-state-capable preset (E12)
    sys01r = RydbergSystem(
        level_structure=level_structure("01r"),
        register=Register.chain(3, spacing_um=8.0),
        protocol=SweepProtocol(t_gate_s=0.3e-6, omega_half_rad_s=lambda t: 1.0,
                               detuning_rad_s=lambda t: 1.0),
    )
    with pytest.raises(ValueError, match="'1r'"):
        sys01r.ground_state(at=0.1e-6, method="dmrg", initial_state=["1", "1", "1"],
                            method_options=_DMRG_OPTS)


def test_dmrg_unmet_convergence_raises():
    # bond_dimension=1 cannot represent the entangled ground state, so the
    # discarded-weight criterion is not met within max_sweeps.
    system = _system(n=6, spacing=5.0)  # strong blockade
    with pytest.raises(RuntimeError, match="convergence"):
        system.ground_state(
            at=_AT, method="dmrg", initial_state=["1", "r", "1", "r", "1", "r"],
            method_options={"bond_dimension": 1, "discarded_weight_tolerance": 1e-10,
                            "relative_energy_tolerance": 1e-9, "entropy_tolerance": 1e-9,
                            "max_sweeps": 4},
        )
