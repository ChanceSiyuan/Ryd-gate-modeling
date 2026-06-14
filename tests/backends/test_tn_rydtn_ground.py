"""Stage G: imaginary-time ground state energy vs exact diagonalization."""

import numpy as np
import pytest

from ryd_gate.backends.peps2d import build_yastn_peps_payload
from ryd_gate.backends.rydtn.backend import RydTNPEPSBackend
from ryd_gate.backends.rydtn.operators import PEPSOps
from ryd_gate.backends.tn_common.compiler import TNEvolutionIR
from ryd_gate.backends.tn_common.lattice_spec import create_tn_lattice_spec
from ryd_gate.backends.tn_common.protocol_context import TNProtocolContext
from ryd_gate.protocols.lattice_dynamics import TFIMQuenchProtocol


def _embed1(O, i, N, d):
    mats = [np.eye(d, dtype=complex)] * N
    mats[i] = O
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def _embed2(O, P, i, j, N, d):
    mats = [np.eye(d, dtype=complex)] * N
    mats[i] = O
    mats[j] = P
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def _build_H(payload, ops):
    lat = payload["lattice"]
    N, Ly, d = int(lat["N"]), int(lat["Ly"]), ops.dim
    inv = np.asarray(lat["inv_snake"], dtype=int)
    s2d = np.asarray(lat["snake_to_2d"], dtype=int)
    sd = payload["schedule"][0]
    oR, ohf = np.asarray(sd["omega_R_1d"]), np.asarray(sd["omega_hf_1d"])
    dR, dhf = np.asarray(sd["delta_R_1d"]), np.asarray(sd["delta_hf_1d"])
    H = np.zeros((d**N, d**N), dtype=complex)
    for site_2d in range(N):
        pos = int(inv[site_2d])
        H = H + _embed1(ops.local_hamiltonian(oR[pos], ohf[pos], dR[pos], dhf[pos]), site_2d, N, d)
    for i_pos, j_pos, strength in lat["vdw_pairs_1d"]:
        i2, j2 = int(s2d[int(i_pos) - 1]), int(s2d[int(j_pos) - 1])
        if i2 == j2 or abs(float(strength)) == 0:
            continue
        H = H + float(strength) * _embed2(ops.n_r, ops.n_r, i2, j2, N, d)
    return H


@pytest.mark.slow
@pytest.mark.parametrize("Lx,Ly", [(2, 2), (2, 3)])
def test_ground_energy_matches_exact(Lx, Ly):
    spec = create_tn_lattice_spec(Lx, Ly, V_nn=2.0, interaction_mode="nn")  # 1r
    proto = TFIMQuenchProtocol(hx=1.0, hz=0.1, t_gate=1.0)
    params = proto.unpack_params([], TNProtocolContext(spec))
    ir = TNEvolutionIR(spec=spec, protocol=proto, params=params)

    payload = build_yastn_peps_payload(
        ir, initial_state="af1", t_eval=None, observables=["n_mean"],
        dt=0.05, chi_max=8, svd_min=1e-12, use_cuda=False,
    )
    ops = PEPSOps(("1", "r"))
    E0 = float(np.linalg.eigvalsh(_build_H(payload, ops))[0])

    res = RydTNPEPSBackend(chi_max=8, dt=0.05).find_ground_state(
        ir, dtau_schedule=((0.2, 30), (0.05, 30), (0.02, 40)),
        initial_state="af1", energy_tol=1e-7,
    )
    E = res.metadata["energy"]
    assert res.metadata["method"] == "peps_rydtn_imag"
    # D=8 is lossless for these small systems -> imaginary time reaches the exact GS.
    assert E >= E0 - 1e-6  # variational lower bound
    assert abs(E - E0) <= 5e-3 * max(1.0, abs(E0))
