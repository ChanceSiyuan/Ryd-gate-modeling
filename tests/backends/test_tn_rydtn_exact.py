"""Stage E: rydtn quench vs an independent full-statevector reference.

At a bond dimension large enough to be lossless, rydtn must reproduce the *same*
2nd-order Trotter evolution applied to the full statevector (same gates, same
order) to machine precision.  This validates gate construction/application, the
NTU bond update (lossless limit), and boundary-MPS measurement end-to-end.
"""

import numpy as np
import pytest

from ryd_gate.backends.rydtn.backend import RydTNPEPSBackend
from ryd_gate.backends.rydtn.gates import gate_local_exp, gate_nn_exp
from ryd_gate.backends.rydtn.operators import PEPSOps
from ryd_gate.backends.peps2d import build_yastn_peps_payload
from ryd_gate.backends.tn_common.compiler import TNEvolutionIR
from ryd_gate.backends.tn_common.lattice_spec import create_tn_lattice_spec
from ryd_gate.backends.tn_common.protocol_context import TNProtocolContext
from ryd_gate.protocols.digital_analog import DigitalAnalogProtocol
from ryd_gate.protocols.lattice_dynamics import TFIMQuenchProtocol


def _embed1(O, i, N, d):
    mats = [np.eye(d, dtype=complex)] * N
    mats[i] = O
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def _apply_2site(vec, G4, i, j, N, d):
    Mt = G4.transpose(0, 2, 1, 3)  # [s0o, s1o, s0i, s1i]
    v = np.tensordot(Mt, vec.reshape((d,) * N), axes=([2, 3], [i, j]))  # [s0o, s1o, *rest]
    rest = [ax for ax in range(N) if ax not in (i, j)]
    perm = [0] * N
    perm[i], perm[j] = 0, 1
    for k, ax in enumerate(rest):
        perm[ax] = 2 + k
    return np.transpose(v, perm).reshape(-1)


def _sv_reference(payload, ops, initial_idx):
    """Full-statevector 2nd-order Trotter with the same per-step gates/order as rydtn."""
    lat = payload["lattice"]
    N, Ly, d = int(lat["N"]), int(lat["Ly"]), ops.dim
    inv = np.asarray(lat["inv_snake"], dtype=int)
    s2d = np.asarray(lat["snake_to_2d"], dtype=int)
    dt = float(payload["runtime"]["dt"])
    Hnn = ops.nn_hamiltonian()

    bond_sites = []
    for i_pos, j_pos, strength in lat["vdw_pairs_1d"]:
        i2, j2 = int(s2d[int(i_pos) - 1]), int(s2d[int(j_pos) - 1])
        if i2 == j2 or abs(float(strength)) == 0:
            continue
        bond_sites.append((i2, j2, float(strength)))

    # initial product state
    psi = np.zeros(1, dtype=complex); psi[0] = 1.0
    e = np.zeros(d, dtype=complex); e[initial_idx] = 1.0
    for _ in range(N):
        psi = np.kron(psi, e)

    for sd in payload["schedule"]:
        oR, ohf = np.asarray(sd["omega_R_1d"]), np.asarray(sd["omega_hf_1d"])
        dR, dhf = np.asarray(sd["delta_R_1d"]), np.asarray(sd["delta_hf_1d"])
        Uloc = []
        for site_2d in range(N):
            pos = int(inv[site_2d])
            g = gate_local_exp(0.5j * dt, ops.local_hamiltonian(oR[pos], ohf[pos], dR[pos], dhf[pos]))
            Uloc.append(_embed1(g, site_2d, N, d))

        def local_layer(p):
            for U in Uloc:
                p = U @ p
            return p

        psi = local_layer(psi)
        for i2, j2, strength in bond_sites:
            psi = _apply_2site(psi, gate_nn_exp(1j * dt, strength * Hnn), i2, j2, N, d)
        psi = local_layer(psi)

    nr = ops.n_r
    return np.array([float(np.real(np.vdot(psi, _embed1(nr, s, N, d) @ psi) / np.vdot(psi, psi)))
                     for s in range(N)])


@pytest.mark.parametrize("Lx,Ly,levels,idx", [
    (1, 2, ("1", "r"), 0),
    (2, 2, ("1", "r"), 0),
    (1, 2, ("0", "1", "r"), 1),
    (2, 2, ("0", "1", "r"), 1),
])
def test_rydtn_matches_statevector_trotter(Lx, Ly, levels, idx):
    level_structure = "1r" if len(levels) == 2 else "01r"
    spec = create_tn_lattice_spec(Lx, Ly, V_nn=2.0, interaction_mode="nn", level_structure=level_structure)
    if level_structure == "01r":
        proto = DigitalAnalogProtocol(
            t_gate=0.3,
            omega_R_fn=lambda t: 0.9, omega_hf_fn=lambda t: 0.5,
            delta_R_fn=lambda t: 0.4, delta_hf_fn=lambda t: -0.2,
        )
    else:  # 1r: TFIMQuench emits only the R channels declared by '1r'
        proto = TFIMQuenchProtocol(hx=0.9, hz=0.3, t_gate=0.3)
    params = proto.unpack_params([], TNProtocolContext(spec))
    ir = TNEvolutionIR(spec=spec, protocol=proto, params=params)
    dt = 0.05

    result = RydTNPEPSBackend(chi_max=16, dt=dt, svd_min=1e-14).evolve_ir(
        ir, initial_state="all_1", t_eval=np.array([0.3]), observables=["n_r"],
    )
    nr_peps = result.metadata["obs"]["n_r"][-1]

    payload = build_yastn_peps_payload(
        ir, initial_state="all_1", t_eval=np.array([0.3]), observables=["n_r"],
        dt=dt, chi_max=16, svd_min=1e-14, use_cuda=False,
    )
    ops = PEPSOps(levels)
    nr_sv = _sv_reference(payload, ops, idx)

    np.testing.assert_allclose(nr_peps, nr_sv, atol=1e-9)
