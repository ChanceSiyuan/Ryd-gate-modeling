"""Stage A: rydtn core (array backend, operators, product PEPS)."""

import numpy as np
import pytest

from ryd_gate.backends.rydtn.operators import PEPSOps
from ryd_gate.backends.rydtn.peps import product_peps
from ryd_gate.backends.rydtn.tensors import RydTNError, resolve_backend


def test_operators_known_values_1r():
    ops = PEPSOps(("1", "r"))
    np.testing.assert_allclose(ops.n_r, np.diag([0.0, 1.0]))
    np.testing.assert_allclose(ops.Z, np.diag([-1.0, 1.0]))
    np.testing.assert_allclose(ops.X_1r, [[0, 1], [1, 0]])
    np.testing.assert_allclose(ops.I, np.eye(2))
    # X_01 / n_0 / n_1 vanish when those levels are absent (mirror _YASTNPEPSOps).
    np.testing.assert_allclose(ops.X_01, np.zeros((2, 2)))
    np.testing.assert_allclose(ops.n_0, np.zeros((2, 2)))
    # In 1r, level "1" is index 0 so n_1 = diag(1, 0); omega_hf/X_01 vanish.
    H = ops.local_hamiltonian(omega_R=2.0, omega_hf=3.0, delta_R=0.5, delta_hf=0.7)
    expected = ops.X_1r - 0.5 * np.diag([0.0, 1.0]) - 0.7 * np.diag([1.0, 0.0])
    np.testing.assert_allclose(H, expected)


def test_operators_known_values_01r():
    ops = PEPSOps(("0", "1", "r"))
    np.testing.assert_allclose(ops.n_0, np.diag([1.0, 0.0, 0.0]))
    np.testing.assert_allclose(ops.n_1, np.diag([0.0, 1.0, 0.0]))
    np.testing.assert_allclose(ops.n_r, np.diag([0.0, 0.0, 1.0]))
    np.testing.assert_allclose(ops.Z, np.diag([-1.0, -1.0, 1.0]))
    x01 = np.zeros((3, 3)); x01[0, 1] = x01[1, 0] = 1.0
    x1r = np.zeros((3, 3)); x1r[1, 2] = x1r[2, 1] = 1.0
    np.testing.assert_allclose(ops.X_01, x01)
    np.testing.assert_allclose(ops.X_1r, x1r)
    # nn Hamiltonian n_r ⊗ n_r, leg order (s0o, s0i, s1o, s1i): only [2,2,2,2] is 1.
    Hnn = ops.nn_hamiltonian()
    assert Hnn.shape == (3, 3, 3, 3)
    expected = np.zeros((3, 3, 3, 3)); expected[2, 2, 2, 2] = 1.0
    np.testing.assert_allclose(Hnn, expected)


def _toy_payload(Lx, Ly, labels):
    from ryd_gate.backends.tn_common.lattice_spec import snake_order_mapping

    snake_to_2d, inv_snake = snake_order_mapping(Lx, Ly)
    return {
        "lattice": {"Lx": Lx, "Ly": Ly, "snake_to_2d": snake_to_2d, "inv_snake": inv_snake},
        "initial_labels_1d": labels,
        "initial_superposition": None,
    }


def test_product_peps_reconstructs_state():
    ops = PEPSOps(("1", "r"))
    backend = resolve_backend(dtype="complex128")
    payload = _toy_payload(1, 2, ["1", "r"])  # snake order == 2D order for a single row
    psi = product_peps(payload, ops, backend)
    assert set(psi.tensors) == {(0, 0), (0, 1)}
    for site in psi.sites():
        assert psi[site].shape == (1, 1, 1, 1, 2)
    np.testing.assert_allclose(backend.to_numpy(psi[(0, 0)]).reshape(2), [1.0, 0.0])  # |1>
    np.testing.assert_allclose(backend.to_numpy(psi[(0, 1)]).reshape(2), [0.0, 1.0])  # |r>
    # bond geometry
    assert psi.nn_bond_dirn((0, 0), (0, 1)) == "lr"
    assert psi.bonds() == [((0, 0), (0, 1))]


@pytest.mark.parametrize("kind,dtype", [("numpy", "complex128"), ("numpy", "complex64")])
def test_backend_numpy(kind, dtype):
    ab = resolve_backend(backend=kind, dtype=dtype)
    a = ab.asarray(np.eye(3))
    assert ab.to_numpy(a).shape == (3, 3)
    # einsum + svd_truncate sanity
    mat = ab.asarray(np.diag([3.0, 2.0, 1e-14]))
    U, s, Vh, disc = ab.svd_truncate(mat, chi_max=10, tol=1e-10)
    assert s.shape[0] == 2  # the 1e-14 singular value is dropped by tol


def test_backend_torch_cpu_complex64_and_128():
    pytest.importorskip("torch")
    for dtype in ("complex128", "complex64"):
        ab = resolve_backend(backend="torch", device="cpu", dtype=dtype)
        a = ab.asarray(np.eye(2))
        assert ab.to_numpy(a).shape == (2, 2)
        U, s, Vh, _ = ab.svd_truncate(ab.asarray(np.diag([2.0, 1.0])), chi_max=1, tol=0.0)
        assert s.shape[0] == 1  # capped at chi_max


def test_numpy_backend_rejects_cuda_device():
    with pytest.raises(RydTNError):
        resolve_backend(backend="numpy", device="cuda")
