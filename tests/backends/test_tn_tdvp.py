"""Tests for TN TDVP backend (requires tenpy)."""

import numpy as np
import pytest

from ryd_gate import RydbergSystem
from ryd_gate.backends.tenpy_mps.backends import TenpyTDVPBackend
from ryd_gate.backends.tenpy_mps.state import product_state_mps
from ryd_gate.backends.tn_common.compiler import TNCompiler
from ryd_gate.backends.tn_common.lattice_spec import create_tn_lattice_spec
from ryd_gate.backends.tn_common.simulate import simulate_tn, simulate_tn_ir
from ryd_gate.core.level_structures import InteractionSpec
from ryd_gate.lattice import make_square_lattice
from ryd_gate.protocols.sweep import SweepProtocol

tenpy = pytest.importorskip("tenpy")


@pytest.fixture
def spec_2x2():
    return create_tn_lattice_spec(Lx=2, Ly=2, V_nn=24.0, Omega=1.0)


def _sweep(t_gate=1.0, omega=1.0, delta=2.0):
    return SweepProtocol(
        t_gate=t_gate,
        omega_half_fn=lambda t: 0.5 * omega,
        delta_fn=lambda t: delta,
    )


class TestTDVPBackend:
    @pytest.mark.slow
    def test_short_evolution_norm_preserved(self, spec_2x2):
        """MPS norm is approximately preserved after short TDVP evolution."""
        psi0 = product_state_mps(spec_2x2, "all_ground")
        proto = _sweep(t_gate=2.0)
        backend = TenpyTDVPBackend(chi_max=16, dt=0.5)

        result = backend.evolve(
            spec_2x2, proto,
            x=[],
            psi0=psi0,
        )
        # MPS norm should be ~1
        norm = result.psi_final.norm
        np.testing.assert_allclose(norm, 1.0, atol=0.01)

    @pytest.mark.slow
    def test_observable_streaming(self, spec_2x2):
        """TDVP records observables at requested times."""
        psi0 = product_state_mps(spec_2x2, "all_ground")
        proto = _sweep(t_gate=2.0)
        t_eval = np.linspace(0, 2.0, 5)

        backend = TenpyTDVPBackend(chi_max=16, dt=0.5)
        result = backend.evolve(
            spec_2x2, proto,
            x=[],
            psi0=psi0,
            t_eval=t_eval,
            observables=["m_s", "n_mean"],
        )

        obs = result.metadata["obs"]
        assert "m_s" in obs
        assert "n_mean" in obs
        assert len(obs["m_s"]) > 0
        assert result.times is not None


class TestSimulateTN:
    @pytest.mark.slow
    def test_dmrg_method(self, spec_2x2):
        """simulate_tn with method='dmrg' returns ground state."""
        proto = _sweep(t_gate=1.0)
        result = simulate_tn(
            spec_2x2, proto, [],
            initial_state="all_ground",
            method="dmrg",
            backend_options={"chi_max": 32},
        )
        assert result.metadata["method"] == "dmrg"
        assert "energy" in result.metadata

    @pytest.mark.slow
    def test_tdvp_method(self, spec_2x2):
        """simulate_tn with method='tdvp' returns evolved state."""
        proto = _sweep(t_gate=1.0)
        result = simulate_tn(
            spec_2x2, proto, [],
            initial_state="all_ground",
            method="tdvp",
            t_eval=np.array([0.5, 1.0]),
            backend_options={"chi_max": 16, "dt": 0.5},
        )
        assert result.metadata["method"] == "tdvp"
        assert result.psi_final is not None

    def test_invalid_method(self, spec_2x2):
        """Unknown method raises ValueError."""
        proto = _sweep()
        with pytest.raises(ValueError, match="Unknown method"):
            simulate_tn(spec_2x2, proto, [], method="invalid")

    @pytest.mark.slow
    def test_dmrg_accepts_tn_spec(self, spec_2x2):
        """simulate_tn accepts the explicit TN lattice spec."""
        proto = _sweep(t_gate=1.0)
        result = simulate_tn(
            spec_2x2, proto, [],
            initial_state="all_ground",
            method="dmrg",
            backend_options={"chi_max": 16},
        )
        assert result.metadata["method"] == "dmrg"

    @pytest.mark.slow
    def test_unified_simulate_tenpy_backend(self):
        proto = _sweep(t_gate=1.0, omega=1.0)
        system = RydbergSystem.from_lattice(
            make_square_lattice(2, 2, spacing_um=1.0),
            level_structure="1r",
            interaction=InteractionSpec(C6=24.0, mode="nnn"),
            protocol=proto,
            Omega=1.0,
        )

        params = system.unpack_params([])
        ir = TNCompiler(method="tdvp").compile(system, params)
        result = simulate_tn_ir(
            ir,
            "all_1",
            backend="tenpy",
            backend_options={"chi_max": 16, "dt": 0.5},
            t_eval=np.array([0.5, 1.0]),
            observables=["n_mean"],
        )

        assert result.metadata["compiler"] == "tn"
        assert result.metadata["backend"] == "tenpy"
        assert result.psi_final is not None
        assert "n_mean" in result.metadata["obs"]
