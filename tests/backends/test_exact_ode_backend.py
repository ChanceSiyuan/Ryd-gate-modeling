"""``backend="exact_ode"`` contract: the only exact backend is the adaptive
DOP853 integrator.

Pins for the rewritten API: ``hamiltonian_format`` selects storage/matvec (dense
vs sparse-matvec) and the two paths agree numerically; ``rtol``/``atol`` are
validated (finite, strictly positive); unknown ``backend_options`` keys and the
removed expm-era backend names error loudly with a message that names the
surviving ``exact_ode`` backend; a non-interacting single atom reproduces the
analytic Rabi flop; and a nested initial-state list batches into a tuple matching
the separate solves. Also pins the compiled Hamiltonian's Hermiticity (E08) and
the raw (un-renormalized) physical norm the backend exposes.
"""

import numpy as np
import pytest

import ryd_gate as rg
from ryd_gate import Register, RydbergSystem, level_structure
from ryd_gate.backends.exact.compiler import compile_exact
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


@pytest.mark.parametrize("name", ["exact", "exact_dense", "exact_sparse", "exact_expm", "dense_expm"])
def test_removed_exact_backends_error(name):
    """The removed expm-era backend names route to the unknown-backend error, whose
    message names the surviving ``exact_ode`` backend. This is the canonical pin for the
    removed-alias contract (the tests/core copies are deleted)."""
    system = _interacting_1r()
    with pytest.raises(ValueError, match="unknown backend") as exc:
        rg.simulate(system, backend=name)
    assert "exact_ode" in str(exc.value)


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


# ── Hermiticity (E08) + physical norm ────────────────────────────────────────


def test_compiled_hamiltonian_is_hermitian_with_complex_channels():
    """The compiled ``H(t)`` is Hermitian at every time, including off-diagonal complex
    drive channels (each adds a matrix element and its conjugate transpose; diagonal
    channels take the real part) — E08."""
    from ryd_gate.protocols import DigitalAnalogProtocol

    w = 2 * np.pi * 1e6
    proto = DigitalAnalogProtocol(
        t_gate_s=0.2e-6,
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
    ham, t_gate = compile_exact(system)
    dim = system._basis.total_dim
    for t in (0.0, 0.05e-6, 0.123e-6, t_gate):
        H = np.empty((dim, dim), dtype=complex)
        for k in range(dim):
            e = np.zeros(dim, dtype=complex)
            e[k] = 1.0
            H[:, k] = ham.apply(t, e)
        np.testing.assert_allclose(H, H.conj().T, atol=1e-9, err_msg=f"t={t}")


def test_real_expectation_rejects_non_real():
    """The real-expectation guard passes tiny roundoff imaginary parts but rejects a
    genuinely non-real value (defense in depth; observables are pre-checked Hermitian)."""
    from ryd_gate.backends.exact.simulate import _real_expectation

    out = _real_expectation(np.array([1.0 + 1e-12j, 0.5 + 0j]), "ok")
    np.testing.assert_allclose(out, [1.0, 0.5])
    assert out.dtype == np.float64
    with pytest.raises(ValueError, match="non-real"):
        _real_expectation(np.array([1.0 + 1.0j]), "bad")


def test_exact_ode_exposes_raw_physical_norm():
    """The exact_ode backend integrates the Schrodinger equation and does NOT renormalize
    the state: the reader exposes the true integrated amplitudes, whose summed populations
    are the physical ``||psi||^2``. For the coherent (Hermitian) Rydberg Hamiltonian this
    is unitary, so the norm stays ~1 — i.e. no hidden renormalization inflates or truncates
    it. (No decay term is currently folded into H, so there is no norm loss to observe;
    ``1 - ||psi||^2`` would be the no-jump decay probability once decay is wired in.)"""
    import itertools

    system = RydbergSystem(
        level_structure=level_structure("1r"),
        register=Register.chain(2, spacing_um=9.0),
        protocol=SweepProtocol(
            t_gate_s=0.3e-6,
            omega_half_rad_s=lambda t: 2 * np.pi * 2.0e6 * np.sin(np.pi * t / 0.3e-6),
            detuning_rad_s=lambda t: 2 * np.pi * 1.0e6,
        ),
    )
    res = rg.simulate(system)
    levels = system.level_structure.levels
    norm2 = sum(abs(res.amplitude(list(lab))) ** 2 for lab in itertools.product(levels, repeat=system.N))
    assert norm2 == pytest.approx(1.0, abs=1e-6)
