"""``backend="exact_ode"`` contract: the only exact backend is the adaptive
DOP853 integrator.

Pins for the rewritten API: ``hamiltonian_format`` selects storage/matvec (dense
vs sparse-matvec) and the two paths agree numerically; ``rtol``/``atol`` are
validated (finite, strictly positive); unknown ``backend_options`` keys and the
removed ``exact`` / ``exact_dense`` / ``exact_sparse`` backend names error
loudly; a non-interacting single atom reproduces the analytic Rabi flop; a
nested initial-state list batches into a tuple matching the separate solves; and
the public ``EvolutionResult`` exposes no state trajectory.
"""

import numpy as np
import pytest

import ryd_gate as rg
from ryd_gate import Register, RydbergSystem, level_structure
from ryd_gate.backends.exact.ode import validated_exact_options
from ryd_gate.protocols import SweepProtocol

MHZ = 2 * np.pi * 1e6


def _interacting_1r(n_side=2, spacing_um=9.0, t_gate=0.4e-6, hx=MHZ):
    """Small interacting ``1r`` transverse-field system (TFIM as a SweepProtocol)."""
    return RydbergSystem(
        level_structure=level_structure("1r"),
        register=Register.square(n_side, spacing_um=spacing_um),
        protocol=SweepProtocol(
            t_gate_s=t_gate,
            omega_half_rad_s=lambda t: hx,
            detuning_rad_s=lambda t: 0.0,
        ),
    )


# ── Hamiltonian storage formats ──────────────────────────────────────────────


def test_hamiltonian_formats_agree():
    """auto / dense / sparse are numerically identical (expectations + amplitude)."""
    system = _interacting_1r()
    n_r = sum(system.observables.n("r", i) for i in range(system.N))
    obs = {"n_r": n_r}

    auto = rg.simulate(system, observables=obs)
    dense = rg.simulate(system, observables=obs, backend_options={"hamiltonian_format": "dense"})
    sparse = rg.simulate(system, observables=obs, backend_options={"hamiltonian_format": "sparse"})

    # dense and sparse are the two storage strategies for the same operator:
    # they must agree to near machine precision.
    np.testing.assert_allclose(dense.expectation("n_r"), sparse.expectation("n_r"), atol=1e-12)
    np.testing.assert_allclose(auto.expectation("n_r"), dense.expectation("n_r"), atol=1e-12)

    labels = ["1"] * system.N
    np.testing.assert_allclose(dense.amplitude(labels), sparse.amplitude(labels), atol=1e-10)
    np.testing.assert_allclose(auto.amplitude(labels), dense.amplitude(labels), atol=1e-10)


# ── option validation ────────────────────────────────────────────────────────


def test_valid_options_resolve():
    opts = validated_exact_options({"hamiltonian_format": "sparse", "rtol": 1e-9, "atol": 1e-13})
    assert opts == {"hamiltonian_format": "sparse", "rtol": 1e-9, "atol": 1e-13}
    assert validated_exact_options(None) == {"hamiltonian_format": "auto", "rtol": 1e-8, "atol": 1e-12}


def test_unknown_backend_option_errors():
    with pytest.raises(ValueError, match="unknown exact backend option"):
        validated_exact_options({"n_steps": 100})


def test_bad_hamiltonian_format_errors():
    with pytest.raises(ValueError, match="hamiltonian_format"):
        validated_exact_options({"hamiltonian_format": "dense_expm"})


@pytest.mark.parametrize("key", ["rtol", "atol"])
@pytest.mark.parametrize("bad", [0.0, -1e-6, np.inf, np.nan])
def test_rtol_atol_must_be_finite_positive(key, bad):
    with pytest.raises(ValueError, match="strictly positive"):
        validated_exact_options({key: bad})


def test_simulate_surfaces_bad_options():
    """Option validation runs at ``simulate`` time, before any evolution."""
    system = _interacting_1r()
    with pytest.raises(ValueError, match="unknown exact backend option"):
        rg.simulate(system, backend_options={"n_steps": 100})
    with pytest.raises(ValueError, match="hamiltonian_format"):
        rg.simulate(system, backend_options={"hamiltonian_format": "dense_expm"})
    with pytest.raises(ValueError, match="strictly positive"):
        rg.simulate(system, backend_options={"rtol": -1.0})


def test_removed_exact_backends_error():
    """The old expm backend names are gone: they route to the unknown-backend error."""
    system = _interacting_1r()
    for name in ("exact", "exact_dense", "exact_sparse"):
        with pytest.raises(ValueError, match="unknown backend"):
            rg.simulate(system, backend=name)


# ── dynamics ─────────────────────────────────────────────────────────────────


def test_analytic_rabi_single_atom():
    """A single non-interacting ``1r`` atom under a resonant drive flops as
    ``n_r(t) = sin^2(Omega t / 2)`` (omega_half = Omega/2, detuning 0)."""
    omega = MHZ
    t_gate = 1e-6
    system = RydbergSystem(
        level_structure=level_structure("1r"),
        register=Register.chain(1),
        protocol=SweepProtocol(
            t_gate_s=t_gate,
            omega_half_rad_s=lambda t: 0.5 * omega,
            detuning_rad_s=lambda t: 0.0,
        ),
    )
    t_eval = np.linspace(0.0, t_gate, 41)
    res = rg.simulate(system, t_eval=t_eval, observables={"n_r": system.observables.n("r", 0)})
    arr = res.expectation("n_r")
    assert arr.shape == (41,) and arr.dtype == np.float64
    np.testing.assert_allclose(arr, np.sin(0.5 * omega * t_eval) ** 2, atol=1e-6)


def test_batch_returns_tuple_matching_single_solves():
    """A nested initial-state list batches under one compilation into a tuple that
    reproduces the separate single-state solves (E10)."""
    system = _interacting_1r()
    n_r = sum(system.observables.n("r", i) for i in range(system.N))
    obs = {"n_r": n_r}

    batch = rg.simulate(system, [["1", "1", "1", "1"], ["r", "1", "1", "1"]], observables=obs)
    assert isinstance(batch, tuple) and len(batch) == 2

    single_a = rg.simulate(system, ["1", "1", "1", "1"], observables=obs)
    single_b = rg.simulate(system, ["r", "1", "1", "1"], observables=obs)
    np.testing.assert_allclose(batch[0].expectation("n_r"), single_a.expectation("n_r"), atol=1e-12)
    np.testing.assert_allclose(batch[1].expectation("n_r"), single_b.expectation("n_r"), atol=1e-12)


# ── measurement request + result surface ─────────────────────────────────────


def test_endpoint_only_times():
    """``t_eval=None`` records only at ``t_gate``; ``times == [t_gate]`` (shape (1,))."""
    system = _interacting_1r()
    res = rg.simulate(system)
    np.testing.assert_array_equal(res.times, [system.t_gate])
    assert res.times.shape == (1,)


def test_explicit_t_eval_requires_observables():
    system = _interacting_1r()
    with pytest.raises(ValueError, match="requires observables"):
        rg.simulate(system, t_eval=np.linspace(0.0, system.t_gate, 5))


def test_result_has_no_state_trajectory():
    """The trajectory is discarded after expectations: no public state attributes."""
    system = _interacting_1r()
    res = rg.simulate(system)
    for attr in ("final_state", "psi_final", "states", "metadata", "expectations"):
        assert not hasattr(res, attr)
    with pytest.raises(KeyError):
        res.expectation("n_r")  # never requested
