"""Tests for TN TDVP backend (requires tenpy)."""

import numpy as np
import pytest

from ryd_gate import RydbergSystem, level_structure
from ryd_gate.backends.tn_common._expressions import tn_basis
from ryd_gate.backends.tn_common.compiler import TNCompiler
from ryd_gate.backends.tn_common.lattice_spec import create_tn_lattice_spec
from ryd_gate.backends.tn_common.simulate import simulate_tn, simulate_tn_ir
from ryd_gate.core.level_structures import InteractionSpec
from ryd_gate.core.observables import ObservableFactory
from ryd_gate.lattice import Register
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


def _observables(spec):
    """{"n_mean", "m_s"} as expressions on the spec's register basis."""
    obs = ObservableFactory(tn_basis(spec))
    sublattice = np.asarray(spec.sublattice, dtype=float)
    n_mean = (1.0 / spec.N) * obs.level_sum("r")
    m_s = (2.0 / spec.N) * obs.weighted_level_sum("r", sublattice) - (
        float(np.sum(sublattice)) / spec.N
    ) * obs.identity()
    return {"n_mean": n_mean, "m_s": m_s}


class TestTDVPBackend:
    @pytest.mark.slow
    def test_short_evolution_norm_preserved(self, spec_2x2):
        """MPS norm is approximately preserved after short TDVP evolution."""
        proto = _sweep(t_gate=2.0)

        result = simulate_tn(
            spec_2x2, proto,
            initial_state="all_ground",
            method="tdvp",
            backend_options={"chi_max": 16, "dt": 0.5},
        )
        # MPS norm should be ~1
        norm = result.final_state.norm
        np.testing.assert_allclose(norm, 1.0, atol=0.01)
        # endpoint-only contract: times == [t_gate], no expectations requested
        np.testing.assert_array_equal(result.times, [2.0])
        assert result.expectations == {}

    @pytest.mark.slow
    def test_observable_streaming(self, spec_2x2):
        """TDVP records expression observables at exactly the requested times."""
        proto = _sweep(t_gate=2.0)
        t_eval = np.linspace(0, 2.0, 5)

        result = simulate_tn(
            spec_2x2, proto,
            initial_state="all_ground",
            method="tdvp",
            t_eval=t_eval,
            observables=_observables(spec_2x2),
            backend_options={"chi_max": 16, "dt": 0.5},
        )

        np.testing.assert_array_equal(result.times, t_eval)
        for name in ("m_s", "n_mean"):
            values = result.expectation(name)
            assert values.shape == t_eval.shape
            assert values.dtype == complex
        # all-ground start: zero Rydberg population at t=0
        np.testing.assert_allclose(result.expectation("n_mean")[0], 0.0, atol=1e-12)
        with pytest.raises(KeyError, match="n_r_total"):
            result.expectation("n_r_total")

    def test_missing_dt_is_preflight_error(self, spec_2x2):
        proto = _sweep(t_gate=1.0)
        with pytest.raises(ValueError, match=r"backend_options=\{'dt'"):
            simulate_tn(spec_2x2, proto, backend_options={"chi_max": 8})

    def test_n_steps_option_removed(self, spec_2x2):
        proto = _sweep(t_gate=1.0)
        with pytest.raises(TypeError, match="n_steps"):
            simulate_tn(spec_2x2, proto, backend_options={"n_steps": 100})

    def test_wrong_shape_expression_is_preflight_error(self, spec_2x2):
        proto = _sweep(t_gate=1.0)
        other = ObservableFactory(tn_basis(create_tn_lattice_spec(Lx=1, Ly=2)))
        with pytest.raises(ValueError, match="n_sites, local_dim"):
            simulate_tn(
                spec_2x2, proto,
                t_eval=np.array([1.0]),
                observables={"bad": other.level_sum("r")},
                backend_options={"chi_max": 8, "dt": 0.5},
            )

    def test_t_eval_without_observables_is_preflight_error(self, spec_2x2):
        proto = _sweep(t_gate=1.0)
        with pytest.raises(ValueError, match="observables"):
            simulate_tn(
                spec_2x2, proto,
                t_eval=np.array([0.5, 1.0]),
                backend_options={"chi_max": 8, "dt": 0.5},
            )


class TestSimulateTN:
    @pytest.mark.slow
    def test_dmrg_method(self, spec_2x2):
        """simulate_tn with method='dmrg' returns ground state."""
        proto = _sweep(t_gate=1.0)
        result = simulate_tn(
            spec_2x2, proto,
            initial_state="all_ground",
            method="dmrg",
            backend_options={"chi_max": 32},
        )
        assert result.metadata["method"] == "dmrg"
        assert "energy" in result.metadata
        # ground-state search has no evolution clock: minimal time axis
        np.testing.assert_array_equal(result.times, [0.0])
        assert result.expectations == {}
        assert result.final_state is not None

    def test_dmrg_rejects_t_eval_and_observables(self, spec_2x2):
        proto = _sweep(t_gate=1.0)
        with pytest.raises(ValueError, match="dmrg"):
            simulate_tn(
                spec_2x2, proto, method="dmrg",
                observables=_observables(spec_2x2),
                backend_options={"chi_max": 16},
            )

    @pytest.mark.slow
    def test_tdvp_method(self, spec_2x2):
        """simulate_tn with method='tdvp' returns evolved state."""
        proto = _sweep(t_gate=1.0)
        result = simulate_tn(
            spec_2x2, proto,
            initial_state="all_ground",
            method="tdvp",
            t_eval=np.array([0.5, 1.0]),
            observables=_observables(spec_2x2),
            backend_options={"chi_max": 16, "dt": 0.5},
        )
        assert result.metadata["method"] == "mps_tdvp"
        assert result.final_state is not None
        np.testing.assert_array_equal(result.times, [0.5, 1.0])

    def test_invalid_method(self, spec_2x2):
        """Unknown method raises ValueError."""
        proto = _sweep()
        with pytest.raises(ValueError, match="supports method"):
            simulate_tn(spec_2x2, proto, method="invalid")

    @pytest.mark.slow
    def test_dmrg_accepts_tn_spec(self, spec_2x2):
        """simulate_tn accepts the explicit TN lattice spec."""
        proto = _sweep(t_gate=1.0)
        result = simulate_tn(
            spec_2x2, proto,
            initial_state="all_ground",
            method="dmrg",
            backend_options={"chi_max": 16},
        )
        assert result.metadata["method"] == "dmrg"

    @pytest.mark.slow
    def test_unified_simulate_mps_backend(self):
        proto = _sweep(t_gate=1.0, omega=1.0)
        system = RydbergSystem(
            level_structure=level_structure("1r"),
            register=Register.rectangle(2, 2, spacing_um=1.0),
            interaction=InteractionSpec(C6=24.0, mode="nnn"),
            protocol=proto,
        )

        ir = TNCompiler(method="tdvp").compile(system)
        t_eval = np.array([0.5, 1.0])
        result = simulate_tn_ir(
            ir,
            "all_1",
            backend="mps",
            backend_options={"chi_max": 16, "dt": 0.5},
            t_eval=t_eval,
            observables={"n_mean": (1.0 / system.N) * system.observables.level_sum("r")},
        )

        assert result.metadata["compiler"] == "tn"
        assert result.metadata["backend"] == "mps"
        assert result.final_state is not None
        np.testing.assert_array_equal(result.times, t_eval)
        assert result.expectation("n_mean").shape == (2,)
