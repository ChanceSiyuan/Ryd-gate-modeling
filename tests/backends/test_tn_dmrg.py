"""Tests for TN DMRG backend (requires tenpy)."""

import numpy as np
import pytest

from ryd_gate.backends.tenpy_mps.backends import TenpyDMRGBackend
from ryd_gate.backends.tenpy_mps.model import build_tenpy_model
from ryd_gate.backends.tenpy_mps.observables import (
    measure_mean_rydberg,
    measure_site_occupations,
    measure_staggered_magnetization,
)
from ryd_gate.backends.tenpy_mps.state import mps_fidelity, product_state_mps, product_superposition_mps
from ryd_gate.backends.tn_common.lattice_spec import create_tn_lattice_spec

pytest.importorskip("tenpy")


@pytest.fixture
def spec_2x2():
    return create_tn_lattice_spec(Lx=2, Ly=2, V_nn=24.0, Omega=1.0)


@pytest.fixture
def spec_3x3():
    return create_tn_lattice_spec(Lx=3, Ly=3, V_nn=24.0, Omega=1.0)


class TestBuildTenpyModel:
    def test_model_builds(self, spec_2x2):
        """Model builds without error."""
        model = build_tenpy_model(spec_2x2, Delta=2.0)
        assert model is not None

    def test_model_with_pinning(self, spec_2x2):
        """Model builds with local detunings."""
        pin = np.array([-4.0, 0.0, 0.0, -4.0])
        model = build_tenpy_model(spec_2x2, Delta=2.0, pin_deltas=pin)
        assert model is not None


class TestProductStateMPS:
    def test_all_ground(self, spec_2x2):
        psi = product_state_mps(spec_2x2, "all_ground")
        occ = measure_site_occupations(psi, spec_2x2)
        np.testing.assert_allclose(occ, 0.0, atol=1e-12)

    def test_af1(self, spec_2x2):
        psi = product_state_mps(spec_2x2, "af1")
        occ = measure_site_occupations(psi, spec_2x2)
        expected = (spec_2x2.sublattice > 0).astype(float)
        np.testing.assert_allclose(occ, expected, atol=1e-12)

    def test_af2(self, spec_2x2):
        psi = product_state_mps(spec_2x2, "af2")
        occ = measure_site_occupations(psi, spec_2x2)
        expected = (spec_2x2.sublattice < 0).astype(float)
        np.testing.assert_allclose(occ, expected, atol=1e-12)

    def test_three_level_complex_superposition_preserves_phase(self):
        spec = create_tn_lattice_spec(Lx=2, Ly=2, level_structure="01r")
        psi = product_superposition_mps(
            spec,
            zero_amp=-1j / np.sqrt(2),
            ground_amp=1 / np.sqrt(2),
            rydberg_amp=0.0,
        )
        np.testing.assert_allclose(mps_fidelity(psi, psi), 1.0, atol=1e-12)


class TestDMRG:
    @pytest.mark.slow
    def test_2x2_energy_vs_exact(self, spec_2x2):
        """DMRG energy matches exact diagonalization for 2x2."""
        from ryd_gate import RydbergSystem, compile_hamiltonian_ir
        from ryd_gate.backends.exact import compile_expm_ir
        from ryd_gate.core.level_structures import InteractionSpec
        from ryd_gate.lattice import make_square_lattice
        from ryd_gate.protocols.sweep import SweepProtocol

        Delta = 2.0
        proto = SweepProtocol(
            t_gate=1.0,
            omega_half_fn=lambda t: 0.5,
            delta_fn=lambda t: Delta,
        )
        system = RydbergSystem.from_lattice(
            make_square_lattice(2, 2, spacing_um=1.0),
            "1r",
            interaction=InteractionSpec(C6=24.0, mode="nnn"),
            protocol=proto,
            Omega=1.0,
        )
        params = system.unpack_params([])
        ham = compile_hamiltonian_ir(system, params)
        ir = compile_expm_ir(ham)

        # Build full Hamiltonian at t=0.5 (midpoint, Omega fully on)
        t_mid = 0.5
        H_full = None
        for term in ir.static_terms:
            c = term.coefficient(t_mid) if callable(term.coefficient) else term.coefficient
            contrib = c * term.operator
            H_full = contrib if H_full is None else H_full + contrib
        for term in ir.drive_terms:
            c = term.coefficient(t_mid) if callable(term.coefficient) else term.coefficient
            H_full = H_full + c * term.operator
            if term.add_hermitian_conjugate:
                H_full = H_full + np.conjugate(c) * term.operator.conj().T

        from scipy.sparse.linalg import eigsh

        E_exact = eigsh(H_full.tocsc(), k=1, which="SA", return_eigenvectors=False)[0]

        backend = TenpyDMRGBackend(chi_max=32, n_sweeps=20)
        result = backend.find_ground_state(spec_2x2, Delta)
        E_dmrg = result.metadata["energy"]

        np.testing.assert_allclose(E_dmrg, E_exact, atol=1e-6, err_msg=f"DMRG E={E_dmrg}, exact E={E_exact}")

    @pytest.mark.slow
    def test_3x3_dmrg_converges(self, spec_3x3):
        """DMRG converges for 3x3 lattice."""
        backend = TenpyDMRGBackend(chi_max=64, n_sweeps=20)
        result = backend.find_ground_state(spec_3x3, Delta=2.0)
        assert result.metadata["energy"] < 0  # should be negative for Delta>0
        assert result.metadata["chi"] > 0


class TestMPSObservables:
    def test_all_ground_ms(self, spec_2x2):
        psi = product_state_mps(spec_2x2, "all_ground")
        ms = measure_staggered_magnetization(psi, spec_2x2)
        # All ground: n_i = 0, so (2*0 - 1) = -1 for all sites
        # ms = (1/N) sum s_i * (-1) = 0 for checkerboard sublattice
        np.testing.assert_allclose(ms, 0.0, atol=1e-12)

    def test_af1_ms_positive(self, spec_2x2):
        psi = product_state_mps(spec_2x2, "af1")
        ms = measure_staggered_magnetization(psi, spec_2x2)
        assert ms > 0.5

    def test_mean_rydberg_af1(self, spec_2x2):
        psi = product_state_mps(spec_2x2, "af1")
        n_mean = measure_mean_rydberg(psi, spec_2x2)
        np.testing.assert_allclose(n_mean, 0.5, atol=1e-12)
