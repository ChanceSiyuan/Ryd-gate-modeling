"""quimb graph-PEPS backend: arbitrary-geometry real-time + ground-state parity.

The distinguishing capability of ``backend='graph_peps'`` /
``method='graph_peps_imaginary_time'`` is that it accepts ANY register — a
triangular lattice or a direct-coordinate register — that the Cartesian YASTN
``peps`` backend rejects, and contracts on the arbitrary interaction graph.

The option-schema test is dependency-free (no quimb). All other tests
``importorskip('quimb')`` and check parity against dense diagonalization / the
exact state-vector backend on a genuinely loopy triangular geometry.
"""

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
    "time_step_s": _T_GATE / 400,
    "bond_dimension": 8,
    "svd_cutoff": 1e-12,
    "measurement_method": "exact",
    "cluster_max_distance": 3,
    "device": "cpu",
}
_GROUND_OPTS = {
    "bond_dimension": 8,
    "svd_cutoff": 1e-12,
    "imaginary_time_schedule": ((3e-8, 300), (1e-8, 300), (3e-9, 500), (1e-9, 800), (3e-10, 800)),
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


def _triangular_system(rows=2, per_row=3, spacing=8.0, cutoff=9.0):
    return RydbergSystem(
        level_structure=level_structure("1r"),
        register=Register.triangular(rows, per_row, spacing_um=spacing),
        protocol=_sweep(),
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


# ── dependency-free option-schema test (no quimb) ────────────────────────────


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
    return terms, w[0], v[:, 0]


class TestGroundState:
    def test_energy_matches_diagonalization(self):
        system = _triangular_system()
        at = _T_GATE / 2
        _terms, e0, _v0 = _dense_ground(system, at)
        gs = system.ground_state(at=at, method="graph_peps_imaginary_time",
                                 initial_state=["1"] * system.N, method_options=_GROUND_OPTS)
        assert abs(gs.expectation("energy") - e0) <= 3e-3 * abs(e0)

    def test_rydberg_population_matches_eigenvector(self):
        system = _triangular_system()
        at = _T_GATE / 2
        terms, _e0, v0 = _dense_ground(system, at)
        n, d = terms.n_sites, terms.local_dim
        nr = terms.rydberg_projector()

        def op(mat, i):
            mats = [np.eye(d, dtype=complex)] * n
            mats[i] = mat
            out = mats[0]
            for m in mats[1:]:
                out = np.kron(out, m)
            return out

        gs = system.ground_state(at=at, method="graph_peps_imaginary_time",
                                 initial_state=["1"] * n,
                                 method_options={**_GROUND_OPTS, "measurement_method": "exact"},
                                 observables={f"n_r{i}": system.observables.n("r", i) for i in range(n)})
        for i in range(n):
            exact = float(np.real(v0.conj() @ op(nr, i) @ v0))
            assert abs(gs.expectation(f"n_r{i}") - exact) <= 5e-3, f"site {i}"

    def test_phase_referenced_amplitude(self):
        system = _triangular_system()
        gs = system.ground_state(at=_T_GATE / 2, method="graph_peps_imaginary_time",
                                 initial_state=["1"] * system.N, method_options=_GROUND_OPTS)
        ref = ["1"] * system.N
        self_amp = gs.amplitude(ref, phase_reference=ref)
        assert abs(self_amp.imag) < 1e-6 * max(1.0, abs(self_amp))
        assert self_amp.real > 0.0
        assert isinstance(gs.amplitude(["r"] + ["1"] * (system.N - 1), phase_reference=ref), complex)


class TestRealTime:
    def test_rydberg_population_matches_exact_backend(self):
        system = _triangular_system()
        obs = {f"n_r{i}": system.observables.n("r", i) for i in range(system.N)}
        exact = simulate(system, observables=obs)
        gtn = simulate(system, backend="graph_peps", observables=obs, backend_options=_RT_OPTS)
        for name in obs:
            np.testing.assert_allclose(gtn.expectation(name), exact.expectation(name), atol=3e-3, err_msg=name)

    def test_complex_amplitude_matches_exact(self):
        system = _triangular_system(rows=2, per_row=2, spacing=8.0, cutoff=9.0)  # N=4, loopy
        exact = simulate(system)
        gtn = simulate(system, backend="graph_peps", backend_options=_RT_OPTS)
        for labels in (["1", "1", "1", "1"], ["r", "1", "1", "1"], ["r", "1", "1", "r"]):
            ea, ga = exact.amplitude(labels), gtn.amplitude(labels)
            assert abs(ea - ga) < 3e-3, labels
        assert isinstance(gtn.amplitude(["1", "1", "1", "1"]), complex)

    def test_sampling_matches_born(self):
        import itertools

        system = _triangular_system(rows=2, per_row=2, spacing=8.0, cutoff=9.0)
        res = simulate(system, backend="graph_peps", backend_options=_RT_OPTS)
        shots = 4000
        counts = res.sample(shots=shots, seed=0)
        assert sum(counts.values()) == shots
        max_sigma = 0.0
        for labels in itertools.product(("1", "r"), repeat=system.N):
            p = abs(res.amplitude(list(labels))) ** 2
            emp = counts.get(tuple(labels), 0) / shots
            sigma = (p * (1 - p) / shots) ** 0.5 or 1e-9
            max_sigma = max(max_sigma, abs(emp - p) / sigma)
        assert max_sigma < 5.0, f"empirical vs Born deviation {max_sigma:.1f} sigma"


class TestArbitraryGeometryAcceptance:
    def test_graph_peps_accepts_geometry_that_peps_rejects(self):
        from ryd_gate.backends.peps._numerics import PEPSError

        for system in (_triangular_system(), _direct_system()):
            obs = {"n_r0": system.observables.n("r", 0)}
            # YASTN peps rejects non-factory / triangular provenance outright.
            with pytest.raises((PEPSError, ValueError)):
                simulate(system, backend="peps", observables=obs, backend_options={
                    "time_step_s": 1e-9, "bond_dimension": 8, "svd_tolerance": 1e-12,
                    "ntu_max_iterations": 100, "ntu_iteration_tolerance": 1e-12,
                    "measurement_method": "belief_propagation", "environment_bond_dimension": 16,
                    "environment_tolerance": 1e-10, "environment_max_iterations": 100, "device": "cpu",
                })
            # graph_peps accepts it and returns a finite population.
            gtn = simulate(system, backend="graph_peps", observables=obs, backend_options=_RT_OPTS)
            assert np.all(np.isfinite(gtn.expectation("n_r0")))

    def test_cuda_rejected(self):
        from ryd_gate.backends.graph_tn import GraphTNError

        system = _triangular_system(rows=2, per_row=2, spacing=8.0, cutoff=9.0)
        with pytest.raises(GraphTNError, match="cuda"):
            simulate(system, backend="graph_peps", backend_options={**_RT_OPTS, "device": "cuda"})
