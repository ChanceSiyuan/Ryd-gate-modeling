"""quimb graph-PEPS backend: arbitrary-geometry real-time + ground-state parity.

The distinguishing capability of ``backend='graph_peps'`` /
``method='graph_peps_imaginary_time'`` is that it accepts ANY register — a
triangular lattice or a direct-coordinate register — that the Cartesian YASTN
``peps`` backend rejects, and contracts on the arbitrary interaction graph.

The option-schema and graph/readout unit tests are dependency-free (no quimb). All
parity tests ``importorskip('quimb')`` and check against dense diagonalization / the
exact state-vector backend on a genuinely loopy triangular geometry.  Evolutions are
deliberately short (few simple-update sweeps / a shrunk anneal) with proportionately
loose tolerances; module-scoped fixtures evolve each configuration once.
"""

import itertools

import numpy as np
import pytest

from ryd_gate import RydbergSystem, level_structure
from ryd_gate.lattice import Register
from ryd_gate.protocols.sweep import SweepProtocol
from ryd_gate.simulate import simulate

TWO_PI = 2 * np.pi
_T_GATE = 0.2e-6

# Moderate blockade regime (V/2pi ~ 3.3 MHz, Omega/2pi ~ 2 MHz) where simple-update
# converges on the loopy triangular graph (strong blockade does not).
_RT_OPTS = {
    "time_step_s": _T_GATE / 40,  # 40 symmetric Strang sweeps
    "bond_dimension": 8,
    "svd_cutoff": 1e-12,
    "measurement_method": "exact",
    "cluster_max_distance": 3,
    "device": "cpu",
}
_RT_OPTS_TINY = {**_RT_OPTS, "time_step_s": _T_GATE / 8}  # ~8 sweeps, finiteness-only checks
_GROUND_OPTS = {
    "bond_dimension": 8,
    "svd_cutoff": 1e-12,
    "imaginary_time_schedule": ((1e-8, 40), (2e-9, 40)),
    "measurement_method": "exact",
    "cluster_max_distance": 3,
    "device": "cpu",
}


def _sweep():
    return SweepProtocol(
        t_gate_s=_T_GATE,
        omega_half_rad_s=lambda t: TWO_PI * 2.0e6 * np.sin(np.pi * t / _T_GATE),
        detuning_rad_s=lambda t: TWO_PI * 1.0e6,
    )


def _triangular_system(rows=2, per_row=3, spacing=8.0, cutoff=9.0, proto=None):
    return RydbergSystem(
        level_structure=level_structure("1r"),
        register=Register.triangular(rows, per_row, spacing_um=spacing),
        protocol=proto if proto is not None else _sweep(),
        interaction_cutoff_um=cutoff,
    )


def _direct_system():
    # A direct-coordinate fused-triangle register (a loop) with no factory provenance.
    coords = [(0.0, 0.0), (8.0, 0.0), (4.0, 6.93), (12.0, 6.93)]
    return RydbergSystem(
        level_structure=level_structure("1r"),
        register=Register(coords),
        protocol=_sweep(),
        interaction_cutoff_um=9.0,
    )


# ── dependency-free contract tests (no quimb) ────────────────────────────────


class TestOptionSchema:
    def test_real_time_rejects_missing_and_unknown(self):
        from ryd_gate.backends.graph_tn import GraphTNError, validate_real_time_options

        good = dict(_RT_OPTS)
        assert validate_real_time_options(good).bond_dimension == 8
        with pytest.raises(GraphTNError):
            validate_real_time_options({k: v for k, v in good.items() if k != "device"})
        with pytest.raises(GraphTNError):
            validate_real_time_options({**good, "unexpected": 1})
        with pytest.raises(GraphTNError):
            validate_real_time_options({**good, "measurement_method": "ctm"})
        with pytest.raises(GraphTNError):
            validate_real_time_options({**good, "svd_cutoff": 2.0})

    def test_ground_schedule_and_device(self):
        from ryd_gate.backends.graph_tn import GraphTNError, validate_ground_options

        assert validate_ground_options(_GROUND_OPTS).device == "cpu"
        with pytest.raises(GraphTNError):  # non-decreasing schedule
            validate_ground_options({**_GROUND_OPTS, "imaginary_time_schedule": ((1e-9, 10), (3e-9, 10))})
        with pytest.raises(GraphTNError):  # list schedule
            validate_ground_options({**_GROUND_OPTS, "imaginary_time_schedule": [(1e-9, 10)]})
        with pytest.raises(GraphTNError):  # bad device
            validate_ground_options({**_GROUND_OPTS, "device": "gpu"})


class _FakeTerms:
    """Minimal stand-in for compiled TNTerms (only what _graph reads)."""

    def __init__(self, n_sites, pairs):
        self.n_sites = n_sites
        self.pairs = tuple(pairs)


class TestBuildGraph:
    """``build_graph`` maps compiled ``terms.pairs`` onto the interaction graph and
    rejects malformed pairs — all dependency-free (no quimb)."""

    def test_needs_at_least_two_atoms(self):
        from ryd_gate.backends.graph_tn import GraphTNError, build_graph

        with pytest.raises(GraphTNError, match="at least 2"):
            build_graph(_FakeTerms(1, []))

    def test_invalid_pair_index_rejected(self):
        from ryd_gate.backends.graph_tn import GraphTNError, build_graph

        with pytest.raises(GraphTNError, match="invalid Rydberg pair index"):
            build_graph(_FakeTerms(3, [(0, 5, 1.0)]))
        with pytest.raises(GraphTNError, match="invalid Rydberg pair index"):
            build_graph(_FakeTerms(3, [(1, 1, 1.0)]))  # self-pair

    def test_non_finite_coupling_rejected(self):
        from ryd_gate.backends.graph_tn import GraphTNError, build_graph

        with pytest.raises(GraphTNError, match="non-finite"):
            build_graph(_FakeTerms(2, [(0, 1, float("inf"))]))

    def test_zero_coupling_dropped_and_accumulated(self):
        from ryd_gate.backends.graph_tn import build_graph

        # exact-zero pairs are dropped; repeated (i, j) pairs accumulate.
        g = build_graph(_FakeTerms(2, [(0, 1, 0.0), (0, 1, 1.0), (0, 1, 0.5)]))
        assert g.edges == ((0, 1),)
        assert g.couplings[(0, 1)] == pytest.approx(1.5)

    def test_isolated_atom_is_capability_error(self):
        from ryd_gate.backends.graph_tn import GraphTNError, build_graph

        # atom 2 shares no edge: the simple update needs every atom covered.
        with pytest.raises(GraphTNError, match="no interaction edge"):
            build_graph(_FakeTerms(3, [(0, 1, 1.0)]))


class TestEngineReadoutUnits:
    """Pure engine/readout helpers, unit-tested without quimb."""

    def test_real_scalar_rejects_non_real(self):
        from ryd_gate.backends.graph_tn import GraphTNError
        from ryd_gate.backends.graph_tn._engine import _real_scalar

        assert _real_scalar(2.0 + 0j, "x") == 2.0
        with pytest.raises(GraphTNError, match="non-real"):
            _real_scalar(1.0 + 1.0j, "x")

    def test_sampling_dense_dim_cap(self):
        from ryd_gate.backends.graph_tn import GraphTNError
        from ryd_gate.backends.graph_tn._readout import _RealTimeReader

        class _T:
            n_sites, local_dim, levels = 21, 2, ("1", "r")  # 2**21 > 1<<20

        class _EmptyPsi:
            sites = []

        with pytest.raises(GraphTNError, match="exceeds"):
            _RealTimeReader(_EmptyPsi(), _T()).sample(shots=1, seed=0)

    def test_sampling_zero_norm_state_rejected(self):
        from ryd_gate.backends.graph_tn import GraphTNError
        from ryd_gate.backends.graph_tn._readout import _RealTimeReader

        class _T:
            n_sites, local_dim, levels = 1, 2, ("1", "r")

        class _ZeroPsi:
            sites = [0]
            site_ind_id = "k{}"

            def to_dense(self, inds):
                return np.zeros((2, 1))

        with pytest.raises(GraphTNError, match="non-finite/zero norm"):
            _RealTimeReader(_ZeroPsi(), _T()).sample(shots=1, seed=0)


# ── quimb-backed parity tests ────────────────────────────────────────────────

pytest.importorskip("quimb")


def _dense_ground(system, at):
    from ryd_gate.backends.tn_common.compiler import compile_tn_terms

    terms = compile_tn_terms(system)
    hloc = terms.local_hamiltonians(at)
    n, d = terms.n_sites, terms.local_dim

    def op(mat, i):
        mats = [np.eye(d, dtype=complex)] * n
        mats[i] = mat
        out = mats[0]
        for m in mats[1:]:
            out = np.kron(out, m)
        return out

    H = np.zeros((d ** n, d ** n), dtype=complex)
    for i in range(n):
        H += op(hloc[i], i)
    nr = terms.rydberg_projector()
    for i, j, V in terms.pairs:
        if V != 0.0:
            H += V * (op(nr, i) @ op(nr, j))
    w, v = np.linalg.eigh(H)
    return terms, w[0], v[:, 0], op


class TestGroundState:
    @pytest.fixture(scope="module")
    def ground(self):
        system = _triangular_system(rows=2, per_row=2, spacing=8.0, cutoff=9.0)  # N=4, loopy
        at = _T_GATE / 2
        terms, e0, v0, op = _dense_ground(system, at)
        gs = system.ground_state(
            at=at, method="graph_peps_imaginary_time", initial_state=["1"] * system.N,
            method_options=_GROUND_OPTS,
            observables={f"n_r{i}": system.observables.n("r", i) for i in range(system.N)},
        )
        return system, terms, e0, v0, op, gs

    def test_energy_population_and_gauge_fix(self, ground):
        system, terms, e0, v0, op, gs = ground
        n = terms.n_sites
        nr = terms.rydberg_projector()

        # (a) energy sanity vs dense eigh — the shrunk anneal is only partly converged,
        # so the band is proportionately loose.
        assert gs.expectation("energy") == pytest.approx(e0, rel=5e-2)

        # (b) per-site Rydberg population wiring matches the dense eigenvector.
        for i in range(n):
            exact = float(np.real(v0.conj() @ op(nr, i) @ v0))
            assert abs(gs.expectation(f"n_r{i}") - exact) <= 3e-2, f"site {i}"

        # (c) gauge-fixed amplitude needs no convergence: the reference is real-positive.
        ref = ["1"] * n
        self_amp = gs.amplitude(ref, phase_reference=ref)
        assert abs(self_amp.imag) < 1e-6 * max(1.0, abs(self_amp))
        assert self_amp.real > 0.0
        assert isinstance(gs.amplitude(["r"] + ["1"] * (n - 1), phase_reference=ref), complex)

    def test_phase_reference_zero_amplitude_rejected(self):
        # Zero drive -> diagonal H -> imaginary time keeps a product basis state (|1,1,1,1>
        # since +detuning + repulsion penalise |r>). A single-|r> reference then has exactly
        # zero amplitude and cannot fix the global phase.
        from ryd_gate.backends.graph_tn import GraphTNError

        system = _triangular_system(
            rows=2, per_row=2, spacing=8.0, cutoff=9.0,
            proto=SweepProtocol(t_gate_s=_T_GATE, omega_half_rad_s=lambda t: 0.0,
                                detuning_rad_s=lambda t: TWO_PI * 1.0e6),
        )
        gs = system.ground_state(at=_T_GATE / 2, method="graph_peps_imaginary_time",
                                 initial_state=["1"] * system.N, method_options=_GROUND_OPTS)
        with pytest.raises(GraphTNError, match="phase_reference amplitude is numerically zero"):
            gs.amplitude(["1"] * system.N, phase_reference=["r", "1", "1", "1"])


class TestRealTime:
    @pytest.fixture(scope="module")
    def rt(self):
        system = _triangular_system(rows=2, per_row=2, spacing=8.0, cutoff=9.0)  # N=4, loopy
        obs = {f"n_r{i}": system.observables.n("r", i) for i in range(system.N)}
        res = simulate(system, backend="graph_peps", observables=obs, backend_options=_RT_OPTS)
        return system, obs, res

    def test_rydberg_population_matches_exact_backend(self, rt):
        system, obs, gtn = rt
        exact = simulate(system, observables=obs)
        for name in obs:
            np.testing.assert_allclose(gtn.expectation(name), exact.expectation(name), atol=1e-2, err_msg=name)

    def test_complex_amplitude_matches_exact(self, rt):
        system, _obs, gtn = rt
        exact = simulate(system)
        for labels in (["1", "1", "1", "1"], ["r", "1", "1", "1"], ["r", "1", "1", "r"]):
            ea, ga = exact.amplitude(labels), gtn.amplitude(labels)
            assert abs(ea - ga) < 1e-2, labels
        assert isinstance(gtn.amplitude(["1", "1", "1", "1"]), complex)

    def test_sampling_matches_born(self, rt):
        system, _obs, res = rt
        shots = 2000
        counts = res.sample(shots=shots, seed=0)
        assert sum(counts.values()) == shots
        max_sigma = 0.0
        for labels in itertools.product(("1", "r"), repeat=system.N):
            p = abs(res.amplitude(list(labels))) ** 2
            emp = counts.get(tuple(labels), 0) / shots
            sigma = (p * (1 - p) / shots) ** 0.5 or 1e-9
            max_sigma = max(max_sigma, abs(emp - p) / sigma)
        assert max_sigma < 5.0, f"empirical vs Born deviation {max_sigma:.1f} sigma"

    @pytest.mark.parametrize("method", ["cluster", "bp"])
    def test_cluster_and_bp_measurement_methods_execute(self, method):
        # The cluster / belief-propagation contraction environments are otherwise dead
        # in the suite; run each once on a short evolution and check finiteness.
        system = _triangular_system(rows=2, per_row=2, spacing=8.0, cutoff=9.0)
        obs = {f"n_r{i}": system.observables.n("r", i) for i in range(system.N)}
        res = simulate(system, backend="graph_peps", observables=obs,
                       backend_options={**_RT_OPTS_TINY, "measurement_method": method})
        for name in obs:
            assert np.all(np.isfinite(res.expectation(name)))


class TestArbitraryGeometryAcceptance:
    def test_graph_peps_accepts_geometry_that_peps_rejects(self):
        from ryd_gate.backends.peps._numerics import PEPSError

        for system in (_triangular_system(), _direct_system()):
            obs = {"n_r0": system.observables.n("r", 0)}
            # YASTN peps rejects non-factory / triangular provenance outright (dependency-free
            # preflight, no evolution).
            with pytest.raises((PEPSError, ValueError)):
                simulate(system, backend="peps", observables=obs, backend_options={
                    "time_step_s": 1e-9, "bond_dimension": 8, "svd_tolerance": 1e-12,
                    "ntu_max_iterations": 100, "ntu_iteration_tolerance": 1e-12,
                    "measurement_method": "belief_propagation", "environment_bond_dimension": 16,
                    "environment_tolerance": 1e-10, "environment_max_iterations": 100, "device": "cpu",
                })
            # graph_peps accepts it (a few simple-update sweeps) and returns a finite population.
            gtn = simulate(system, backend="graph_peps", observables=obs, backend_options=_RT_OPTS_TINY)
            assert np.all(np.isfinite(gtn.expectation("n_r0")))

    def test_cuda(self):
        import importlib.util

        from ryd_gate.backends.graph_tn import GraphTNError

        system = _triangular_system(rows=2, per_row=2, spacing=8.0, cutoff=9.0)  # N=4, loopy
        have_cuda = importlib.util.find_spec("torch") is not None
        if have_cuda:
            import torch

            have_cuda = torch.cuda.is_available()
        if not have_cuda:
            # no PyTorch/CUDA: device='cuda' rejects cleanly (no silent CPU fallback)
            with pytest.raises(GraphTNError, match="cuda|PyTorch"):
                simulate(system, backend="graph_peps", backend_options={**_RT_OPTS, "device": "cuda"})
            return
        # CUDA available: a short torch-backed real-time solve + short anneal match exact.
        obs = {f"n_r{i}": system.observables.n("r", i) for i in range(system.N)}
        exact = simulate(system, observables=obs)
        cuda = simulate(system, backend="graph_peps", observables=obs,
                        backend_options={**_RT_OPTS, "device": "cuda"})
        for name in obs:
            np.testing.assert_allclose(cuda.expectation(name), exact.expectation(name), atol=1e-2, err_msg=name)
        assert isinstance(cuda.amplitude(["1", "1", "1", "1"]), complex)
        _terms, e0, _v0, _op = _dense_ground(system, _T_GATE / 2)
        gs = system.ground_state(at=_T_GATE / 2, method="graph_peps_imaginary_time",
                                 initial_state=["1"] * system.N,
                                 method_options={**_GROUND_OPTS, "device": "cuda"})
        assert gs.expectation("energy") == pytest.approx(e0, rel=5e-2)
