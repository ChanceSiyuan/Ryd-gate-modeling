"""Stage F: rydtn vs the YASTN engine on identical payloads (via the dispatch).

Also exercises ``backend="peps"`` engine_package routing.  We compare on 1D
*chains* (no loops) at a lossless bond dimension: there both engines perform the
*same* exact 2nd-order Trotter, and both belief-propagation (YASTN) and
boundary-MPS (rydtn) measurements are exact, so the observables agree to
numerical precision.  Genuinely-2D correctness is covered by the exact-statevector
test (``test_tn_rydtn_exact.py``); YASTN's own 2D measurement here would be an
approximate (BP) or unconverged (CTM) reference, not ground truth.
"""

import numpy as np
import pytest

from ryd_gate.backends.tn_common.lattice_spec import create_tn_lattice_spec
from ryd_gate.backends.tn_common.simulate import simulate_tn
from ryd_gate.protocols.lattice_dynamics import TFIMQuenchProtocol


def _run(spec, proto, engine, chi, observables):
    return simulate_tn(
        spec, proto, [], backend="peps",
        t_eval=np.array([0.5]), observables=observables,
        backend_options={
            "engine_package": engine, "chi_max": chi, "dt": 0.05,
            "measurement_environment": "bp",  # exact on a chain for both engines
        },
    )


@pytest.mark.slow
@pytest.mark.parametrize("Lx,Ly", [(1, 2), (1, 3), (3, 1), (1, 4)])
def test_rydtn_vs_yastn_chain_quench(Lx, Ly):
    """Lossless chain: rydtn and YASTN agree to numerical precision."""
    pytest.importorskip("yastn")
    spec = create_tn_lattice_spec(Lx, Ly, V_nn=3.0, interaction_mode="nn")
    proto = TFIMQuenchProtocol(hx=1.2, hz=0.2, t_gate=0.5)
    r = _run(spec, proto, "rydtn", 16, ["n_r", "sigma_z"])
    y = _run(spec, proto, "yastn", 16, ["n_r", "sigma_z"])
    assert r.metadata["engine_package"] == "rydtn"
    assert y.metadata["engine_package"] == "yastn"
    np.testing.assert_allclose(r.metadata["obs"]["n_r"][-1], y.metadata["obs"]["n_r"][-1], atol=1e-6)
    np.testing.assert_allclose(r.metadata["obs"]["sigma_z"][-1], y.metadata["obs"]["sigma_z"][-1], atol=1e-6)
