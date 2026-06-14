"""Stage 5: boundary-MPS measurement vs exact dense contraction."""

import numpy as np
import pytest

from ryd_gate.backends.rydtn.boundary import BoundaryMPS, contract_peps_dense
from ryd_gate.backends.rydtn.operators import PEPSOps
from ryd_gate.backends.rydtn.peps import FinitePEPS, product_peps
from ryd_gate.backends.rydtn.tensors import resolve_backend

rng = np.random.default_rng(1)


def _rand(shape):
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex128)


def _random_peps(Lx, Ly, D, d, ab):
    psi = FinitePEPS(Lx, Ly, ab)
    for x in range(Lx):
        for y in range(Ly):
            Dt = D if x > 0 else 1
            Db = D if x < Lx - 1 else 1
            Dl = D if y > 0 else 1
            Dr = D if y < Ly - 1 else 1
            psi[(x, y)] = ab.asarray(_rand((Dt, Dl, Db, Dr, d)))
    return psi


def _exact_1site(vec, axis, O, N, d):
    v = vec.reshape((d,) * N)
    Ov = np.moveaxis(np.tensordot(O, v, axes=([1], [axis])), 0, axis)
    return np.vdot(v, Ov) / np.vdot(v, v)


def _exact_2site(vec, ax_i, ax_j, O, P, N, d):
    v = vec.reshape((d,) * N)
    w = np.moveaxis(np.tensordot(O, v, axes=([1], [ax_i])), 0, ax_i)
    w = np.moveaxis(np.tensordot(P, w, axes=([1], [ax_j])), 0, ax_j)
    return np.vdot(v, w) / np.vdot(v, v)


@pytest.mark.parametrize("Lx,Ly,D", [(1, 2, 2), (2, 2, 2), (2, 3, 2), (3, 3, 2)])
def test_boundary_1site_matches_exact(Lx, Ly, D):
    ab = resolve_backend(dtype="complex128")
    ops = PEPSOps(("1", "r"))
    psi = _random_peps(Lx, Ly, D, ops.dim, ab)
    vec = contract_peps_dense(psi, ab)
    N = Lx * Ly

    env = BoundaryMPS(psi, ab, chi=None)
    # norm
    np.testing.assert_allclose(env.norm(), np.vdot(vec, vec), rtol=1e-9, atol=1e-9)
    # <n_r> and <Z> at every site
    for O in (ops.n_r, ops.Z):
        meas = env.measure_1site(O)
        for x in range(Lx):
            for y in range(Ly):
                exact = _exact_1site(vec, x * Ly + y, O, N, ops.dim)
                np.testing.assert_allclose(meas[(x, y)], exact, rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("Lx,Ly", [(2, 2), (2, 3), (3, 3)])
def test_boundary_2site_matches_exact(Lx, Ly):
    ab = resolve_backend(dtype="complex128")
    ops = PEPSOps(("1", "r"))
    psi = _random_peps(Lx, Ly, 2, ops.dim, ab)
    vec = contract_peps_dense(psi, ab)
    N = Lx * Ly
    env = BoundaryMPS(psi, ab, chi=None)
    # a horizontal pair, a vertical pair, and a diagonal pair
    pairs = [((0, 0), (0, 1)), ((0, 0), (1, 0)), ((0, 0), (1, 1))]
    for si, sj in pairs:
        if sj[0] >= Lx or sj[1] >= Ly:
            continue
        got = env.measure_2site(ops.Z, ops.n_r, si, sj)
        exact = _exact_2site(vec, si[0] * Ly + si[1], sj[0] * Ly + sj[1], ops.Z, ops.n_r, N, ops.dim)
        np.testing.assert_allclose(got, exact, rtol=1e-8, atol=1e-8)


def test_boundary_product_state_exact():
    ab = resolve_backend(dtype="complex128")
    ops = PEPSOps(("0", "1", "r"))
    from ryd_gate.backends.tn_common.lattice_spec import snake_order_mapping

    Lx, Ly = 2, 2
    s2d, inv = snake_order_mapping(Lx, Ly)
    payload = {
        "lattice": {"Lx": Lx, "Ly": Ly, "snake_to_2d": s2d, "inv_snake": inv},
        "initial_labels_1d": ["1"] * (Lx * Ly),
        "initial_superposition": None,
    }
    psi = product_peps(payload, ops, ab)
    env = BoundaryMPS(psi, ab, chi=None)
    n1 = env.measure_1site(ops.n_1)
    nr = env.measure_1site(ops.n_r)
    for s in psi.sites():
        np.testing.assert_allclose(n1[s], 1.0, atol=1e-12)
        np.testing.assert_allclose(nr[s], 0.0, atol=1e-12)


def test_boundary_chi_convergence():
    """Compressed boundary MPS converges to the exact contraction as chi grows."""
    ab = resolve_backend(dtype="complex128")
    ops = PEPSOps(("1", "r"))
    psi = _random_peps(3, 3, 3, ops.dim, ab)
    exact = BoundaryMPS(psi, ab, chi=None).measure_1site(ops.n_r)
    approx = BoundaryMPS(psi, ab, chi=81).measure_1site(ops.n_r)  # chi >= (D^2)^... exact here
    for s in psi.sites():
        np.testing.assert_allclose(approx[s], exact[s], rtol=1e-8, atol=1e-8)
