"""Stage B: rydtn gate construction (vs scipy expm and vs YASTN gates)."""

import numpy as np
import pytest
from scipy.linalg import expm

from ryd_gate.backends.rydtn.gates import gate_local_exp, gate_nn_exp
from ryd_gate.backends.rydtn.operators import PEPSOps


@pytest.mark.parametrize("levels", [("1", "r"), ("0", "1", "r")])
@pytest.mark.parametrize("coeff", [0.5j * 0.1, 0.5 * 0.07])  # real-time half, imag half
def test_gate_local_matches_expm(levels, coeff):
    ops = PEPSOps(levels)
    H = ops.local_hamiltonian(omega_R=1.3, omega_hf=0.4, delta_R=0.6, delta_hf=-0.2)
    G = gate_local_exp(coeff, H)
    np.testing.assert_allclose(G, expm(-coeff * H), atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("levels", [("1", "r"), ("0", "1", "r")])
@pytest.mark.parametrize("coeff", [1j * 0.1, 1.0 * 0.05])
def test_gate_nn_matches_expm(levels, coeff):
    ops = PEPSOps(levels)
    d = ops.dim
    Hnn4 = 4.0 * ops.nn_hamiltonian()  # strength * n_r⊗n_r
    G = gate_nn_exp(coeff, Hnn4)
    assert G.shape == (d, d, d, d)
    # fuse my gate to a matrix and compare with expm of the fused Hamiltonian
    M = Hnn4.transpose(0, 2, 1, 3).reshape(d * d, d * d)
    Gmat = G.transpose(0, 2, 1, 3).reshape(d * d, d * d)
    np.testing.assert_allclose(Gmat, expm(-coeff * M), atol=1e-12, rtol=1e-12)


def _yastn_ops(cfg, yastn, levels):
    def mat(M):
        t = yastn.Tensor(config=cfg, s=(1, -1))
        t.set_block(ts=(), Ds=M.shape, val=np.asarray(M, dtype=complex))
        return t
    return mat


def test_gates_match_yastn():
    yastn = pytest.importorskip("yastn")
    import yastn.tn.fpeps.gates as ygates

    cfg = yastn.make_config(backend="np", sym="none", default_dtype="complex128")
    levels = ("0", "1", "r")
    ops = PEPSOps(levels)
    mat = _yastn_ops(cfg, yastn, levels)
    I = mat(ops.I)
    H = ops.local_hamiltonian(1.3, 0.4, 0.6, -0.2)
    coeff = 0.5j * 0.1

    # local
    gl = ygates.gate_local_exp(coeff, I, mat(H))
    np.testing.assert_allclose(gate_local_exp(coeff, H), gl.G[0].to_numpy(), atol=1e-12)

    # nn: reconstruct yastn's split gate G0[s0o,s0i,a], G1[s1o,s1i,a]
    nr = mat(ops.n_r)
    Hnn = ygates.fkron(nr, nr)
    gnn = ygates.gate_nn_exp(1j * 0.1, I, 4.0 * Hnn)
    G0 = np.asarray(gnn.G[0].to_numpy())
    G1 = np.asarray(gnn.G[1].to_numpy())
    G_yastn = np.einsum("ija,kla->ijkl", G0, G1)
    np.testing.assert_allclose(gate_nn_exp(1j * 0.1, 4.0 * ops.nn_hamiltonian()), G_yastn, atol=1e-12)
