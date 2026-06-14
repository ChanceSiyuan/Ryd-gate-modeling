"""Stage H/I: bond-dimension convergence and GPU (torch/cuda) parity."""

import numpy as np
import pytest

from ryd_gate.backends.rydtn.backend import RydTNPEPSBackend
from ryd_gate.backends.tn_common.compiler import TNEvolutionIR
from ryd_gate.backends.tn_common.lattice_spec import create_tn_lattice_spec
from ryd_gate.backends.tn_common.protocol_context import TNProtocolContext
from ryd_gate.protocols.lattice_dynamics import TFIMQuenchProtocol


def _ir(Lx, Ly, V_nn=3.0):
    spec = create_tn_lattice_spec(Lx, Ly, V_nn=V_nn, interaction_mode="nn")
    proto = TFIMQuenchProtocol(hx=1.5, hz=0.2, t_gate=0.3)
    params = proto.unpack_params([], TNProtocolContext(spec))
    return TNEvolutionIR(spec=spec, protocol=proto, params=params)


def _nr(ir, chi, **kw):
    res = RydTNPEPSBackend(chi_max=chi, dt=0.05, **kw).evolve_ir(
        ir, initial_state="all_1", t_eval=np.array([0.3]), observables=["n_r"],
    )
    return res.metadata["obs"]["n_r"][-1], res.metadata


@pytest.mark.slow
def test_chi_convergence_3x3():
    """Truncation error vs the near-lossless reference decreases as chi grows."""
    ir = _ir(3, 3)
    ref, _ = _nr(ir, 8)
    errs = [float(np.max(np.abs(_nr(ir, c)[0] - ref))) for c in (2, 4, 6)]
    assert errs[0] >= errs[1] - 1e-9          # monotone improvement with chi
    assert errs[1] >= errs[2] - 1e-9
    assert errs[2] < 5e-3                       # near-converged by chi=6


@pytest.mark.slow
def test_gpu_matches_cpu():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    ir = _ir(2, 2, V_nn=2.0)
    cpu, _ = _nr(ir, 6)
    gpu, meta = _nr(ir, 6, backend_name="torch", device="cuda")
    assert meta["accelerator"] == "cuda" and meta["gpu"] is True
    np.testing.assert_allclose(gpu, cpu, atol=1e-6)
    # single precision (complex64) on GPU — broken in YASTN, works here
    gpu32, meta32 = _nr(ir, 6, backend_name="torch", device="cuda", dtype="complex64")
    np.testing.assert_allclose(gpu32, cpu, atol=1e-3)
