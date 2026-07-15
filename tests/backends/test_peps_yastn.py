"""YASTN finite-PEPS backend: real-time + ground-state parity against exact (E24/E25/E16/E20).

Requires the ``tn-2d`` extra (yastn). Covers factory geometries and orientation
branches, exact/PEPS expectation + complex-amplitude parity, report-only evidence,
and the normalized imaginary-time ground state vs dense diagonalization.
"""

import numpy as np
import pytest

from ryd_gate import RydbergSystem, level_structure
from ryd_gate.lattice import Register
from ryd_gate.protocols.sweep import SweepProtocol
from ryd_gate.simulate import simulate

pytest.importorskip("yastn")

TWO_PI = 2 * np.pi
_T_GATE = 0.2e-6

_RT_OPTS = {
    "time_step_s": 1e-9,
    "bond_dimension": 8,
    "svd_tolerance": 1e-12,
    "ntu_max_iterations": 100,
    "ntu_iteration_tolerance": 1e-12,
    "measurement_method": "belief_propagation",
    "environment_bond_dimension": 16,
    "environment_tolerance": 1e-10,
    "environment_max_iterations": 100,
    "device": "cpu",
}
_GROUND_OPTS = {
    "bond_dimension": 8,
    "svd_tolerance": 1e-12,
    "ntu_max_iterations": 100,
    "ntu_iteration_tolerance": 1e-12,
    "environment_bond_dimension": 16,
    "environment_tolerance": 1e-10,
    "environment_max_iterations": 100,
    "imaginary_time_schedule": ((0.2, 200), (0.05, 400), (0.01, 800)),
    "device": "cpu",
}


def _sweep():
    return SweepProtocol(
        t_gate_s=_T_GATE,
        omega_half_rad_s=lambda t: TWO_PI * 2.0e6 * np.sin(np.pi * t / _T_GATE),
        detuning_rad_s=lambda t: TWO_PI * 1.0e6,
    )


def _system(register):
    # cutoff 7.0 um keeps only the 6 um Cartesian nearest neighbours (excludes the
    # 8.49 um square diagonal and 12 um next-nearest), so exact and PEPS share the
    # same NN-only pair topology that the PEPS grid supports.
    return RydbergSystem(
        level_structure=level_structure("1r"), register=register, protocol=_sweep(),
        interaction_cutoff_um=7.0,
    )


def _n_r(system, i):
    return system.observables.n("r", i)


class TestRealTimeParity:
    # Tree geometries (chain / 1xN / Nx1) have no interaction loops: NTU and BP are
    # near-exact there (tight atol). The 2x2 square is a 4-site plaquette *loop*, where
    # the nearest-neighbour NTU update and the tree-based BP environment are genuine
    # approximations (looser atol) — the boundary-contraction math itself is exact
    # (validated separately) and ground energy matches diagonalization to ~1e-9.
    @pytest.mark.parametrize("register, atol", [
        (Register.chain(2, spacing_um=6.0), 2e-3),
        (Register.rectangle(1, 3, spacing_um=6.0), 2e-3),
        (Register.rectangle(3, 1, spacing_um=6.0), 2e-3),
        (Register.square(2, spacing_um=6.0), 3e-2),
    ])
    def test_expectation_matches_exact(self, register, atol):
        system = _system(register)
        obs = {f"n_r{i}": _n_r(system, i) for i in range(system.N)}
        exact = simulate(system, observables=obs)
        peps = simulate(system, backend="peps", observables=obs, backend_options=_RT_OPTS)
        for name in obs:
            np.testing.assert_allclose(peps.expectation(name), exact.expectation(name), atol=atol)

    def test_complex_amplitude_phase_matches_exact(self):
        system = _system(Register.chain(2, spacing_um=6.0))
        exact = simulate(system)
        peps = simulate(system, backend="peps", backend_options=_RT_OPTS)
        for labels in (["1", "1"], ["1", "r"], ["r", "1"], ["r", "r"]):
            ea, pa = exact.amplitude(labels), peps.amplitude(labels)
            assert abs(ea - pa) < 3e-3  # magnitude AND phase
        assert isinstance(peps.amplitude(["1", "1"]), complex)

    def test_01r_full_local_hamiltonian_matches_exact(self):
        # 01r qutrit: PEPS must carry the complete 3x3 DigitalAnalog local Hamiltonian
        # (|0> is a spectator level). chain(2) is a tree so PEPS is near-exact here.
        from ryd_gate.protocols.digital_analog import DigitalAnalogProtocol

        w = TWO_PI * 1e6
        proto = DigitalAnalogProtocol(
            t_gate_s=0.2e-6,
            coupling_r1_rad_s=lambda t: [1.0 * w, (0.8 + 0.5j) * w],
            coupling_10_rad_s=lambda t: 0.6 * w * np.exp(0.8j * t / 1e-6),
            coupling_r0_rad_s=lambda t: [0.4j * w, 0.25 * w],
            energy_r_rad_s=lambda t: [-0.7 * w, 0.5 * w],
            energy_1_rad_s=lambda t: 0.3 * w,
        )
        system = RydbergSystem(
            level_structure=level_structure("01r"),
            register=Register.chain(2, spacing_um=8.0),
            protocol=proto,
            interaction_cutoff_um=9.0,
        )
        obs = {f"n_{lvl}_{i}": system.observables.n(lvl, i) for lvl in ("0", "1", "r") for i in range(2)}
        exact = simulate(system, observables=obs)
        peps = simulate(system, backend="peps", observables=obs, backend_options=_RT_OPTS)
        for label in obs:
            np.testing.assert_allclose(peps.expectation(label), exact.expectation(label), atol=3e-3, err_msg=label)

    def test_cuda_real_time_smoke(self):
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("no CUDA device available")
        system = _system(Register.square(2, spacing_um=6.0))
        opts = {**_RT_OPTS, "device": "cuda"}
        res = simulate(system, backend="peps", observables={"n_r0": _n_r(system, 0)}, backend_options=opts)
        assert np.all(np.isfinite(res.expectation("n_r0")))
        assert isinstance(res.amplitude(["1", "1", "1", "1"]), complex)
        assert res.peps_evidence.parameters["device"] == "cuda"

    def test_ctm_two_site_correlator(self):
        system = _system(Register.chain(2, spacing_um=6.0))
        obs = {"nrnr": _n_r(system, 0) @ _n_r(system, 1)}
        opts = {**_RT_OPTS, "measurement_method": "ctm"}
        exact = simulate(system, observables=obs)
        peps = simulate(system, backend="peps", observables=obs, backend_options=opts)
        np.testing.assert_allclose(peps.expectation("nrnr"), exact.expectation("nrnr"), atol=2e-3)

    def test_bp_rejects_two_site_observable(self):
        system = _system(Register.chain(2, spacing_um=6.0))
        obs = {"nrnr": _n_r(system, 0) @ _n_r(system, 1)}
        with pytest.raises((ValueError, RuntimeError)):
            simulate(system, backend="peps", observables=obs, backend_options=_RT_OPTS)


class TestRealTimeEvidence:
    def test_evidence_populated_and_json(self):
        system = _system(Register.chain(2, spacing_um=6.0))
        obs = {"n_r0": _n_r(system, 0)}
        peps = simulate(system, backend="peps", observables=obs, backend_options=_RT_OPTS)
        ev = peps.peps_evidence
        assert set(ev.parameters) == set(_RT_OPTS)
        assert ev.hamiltonian_scale_rad_s is None
        assert ev.max_ntu_truncation_error >= 0.0
        assert ev.cumulative_ntu_truncation_error >= 0.0
        assert ev.environment_iterations is not None
        # amplitude records accrue lazily; JSON round-trips
        peps.amplitude(["1", "1"])
        import json

        payload = json.dumps(peps.peps_evidence.to_dict())
        assert json.loads(payload)["norm_contraction_error"] is not None
        assert len(peps.peps_evidence.amplitudes) == 1

    def test_no_observables_builds_no_environment(self):
        system = _system(Register.chain(2, spacing_um=6.0))
        peps = simulate(system, backend="peps", backend_options=_RT_OPTS)
        ev = peps.peps_evidence
        assert ev.environment_residual is None
        assert ev.environment_iterations is None


class TestSampling:
    """Sampling must reproduce the Born distribution that amplitude computes (verified separately)."""

    def _born_check(self, result, sampler, n_sites, ground_ref=None):
        import itertools

        # sampler and amplitude derive from the same PEPS state, so the deviation is
        # pure multinomial sampling noise — a few thousand shots give a robust 5-sigma check.
        shots = 2000
        counts = sampler(shots=shots, seed=0)
        assert sum(counts.values()) == shots
        assert all(len(k) == n_sites and all(x in ("1", "r") for x in k) for k in counts)
        max_sigma = 0.0
        for labels in itertools.product(("1", "r"), repeat=n_sites):
            if ground_ref is None:
                amp = result.amplitude(list(labels))
            else:
                amp = result.amplitude(list(labels), phase_reference=ground_ref)
            p = abs(amp) ** 2
            emp = counts.get(tuple(labels), 0) / shots
            sigma = (p * (1 - p) / shots) ** 0.5 or 1e-9
            max_sigma = max(max_sigma, abs(emp - p) / sigma)
        assert max_sigma < 5.0, f"empirical vs Born deviation {max_sigma:.1f} sigma"

    def test_real_time_bp_sampling_matches_born(self):
        system = _system(Register.chain(2, spacing_um=6.0))
        res = simulate(system, backend="peps", backend_options=_RT_OPTS)
        self._born_check(res, res.sample, n_sites=2)

    def test_real_time_ctm_sampling_matches_born(self):
        # CTM sampling uses the boundary-MPS/EnvWindow path; exercise it on a genuine
        # 2D grid (a 2x2 plaquette) rather than a 1-wide chain.
        system = _system(Register.square(2, spacing_um=6.0))
        res = simulate(system, backend="peps", backend_options={**_RT_OPTS, "measurement_method": "ctm"})
        self._born_check(res, res.sample, n_sites=4)

    def test_negative_seed_rejected_at_boundary(self):
        system = _system(Register.chain(2, spacing_um=6.0))
        res = simulate(system, backend="peps", backend_options=_RT_OPTS)
        with pytest.raises(ValueError, match="non-negative"):
            res.sample(shots=10, seed=-1)

    def test_ctm_sampling_on_chain_raises_cleanly(self):
        # A 1-wide chain makes the CTM boundary-MPS contraction degenerate; it must
        # raise a clear capability error, never hang.
        system = _system(Register.chain(2, spacing_um=6.0))
        res = simulate(system, backend="peps", backend_options={**_RT_OPTS, "measurement_method": "ctm"})
        with pytest.raises(RuntimeError, match="2D grid"):
            res.sample(shots=10, seed=0)

    def test_seed_reproducible_and_global_rng_untouched(self):
        system = _system(Register.chain(2, spacing_um=6.0))
        res = simulate(system, backend="peps", backend_options=_RT_OPTS)
        state = np.random.get_state()
        a = res.sample(shots=200, seed=7)
        b = res.sample(shots=200, seed=7)
        c = res.sample(shots=200, seed=8)
        assert a == b
        assert a != c
        # the module must not touch the global NumPy RNG
        s0, s1 = state, np.random.get_state()
        assert s0[0] == s1[0] and np.array_equal(s0[1], s1[1]) and s0[2:] == s1[2:]

    def test_ground_state_sampling_matches_born(self):
        # The Born check only needs sampler == amplitude on the *same* PEPS state, so a
        # short (unconverged) imaginary-time schedule keeps this 4-site case fast.
        system = _system(Register.square(2, spacing_um=6.0))
        fast = {**_GROUND_OPTS, "imaginary_time_schedule": ((0.1, 30),)}
        gs = system.ground_state(at=_T_GATE / 2, method="peps_imaginary_time",
                                 initial_state=["r", "1", "1", "r"], method_options=fast)
        self._born_check(gs, gs.sample, n_sites=4, ground_ref=["r", "1", "1", "r"])


class TestGroundState:
    def _dense_ground(self, system, at):
        from ryd_gate.backends.tn_common.compiler import compile_tn_terms

        terms = compile_tn_terms(system)
        hloc = terms.local_hamiltonians(at)
        d = terms.local_dim
        Id = np.eye(d)
        H = np.kron(hloc[0], Id) + np.kron(Id, hloc[1])
        nr = terms.rydberg_projector()
        for i, j, V in terms.pairs:
            if V != 0.0:
                H = H + V * np.kron(nr, nr)
        w, v = np.linalg.eigh(H)
        return terms, w[0], v[:, 0]

    def test_energy_matches_diagonalization(self):
        system = _system(Register.chain(2, spacing_um=6.0))
        at = _T_GATE / 2
        _terms, e0, _v0 = self._dense_ground(system, at)
        gs = system.ground_state(at=at, method="peps_imaginary_time",
                                 initial_state=["r", "1"], method_options=_GROUND_OPTS)
        assert abs(gs.expectation("energy") - e0) <= 1e-5 * abs(e0)
        ev = gs.peps_evidence
        assert ev.hamiltonian_scale_rad_s > 0.0
        assert ev.cumulative_ntu_truncation_error is None

    def test_phase_referenced_amplitude_is_gauge_fixed(self):
        system = _system(Register.chain(2, spacing_um=6.0))
        gs = system.ground_state(at=_T_GATE / 2, method="peps_imaginary_time",
                                 initial_state=["r", "1"], method_options=_GROUND_OPTS)
        ref = ["r", "1"]
        # reference amplitude is real-positive in its own gauge
        self_amp = gs.amplitude(ref, phase_reference=ref)
        assert abs(self_amp.imag) < 1e-6 * max(1.0, abs(self_amp))
        assert self_amp.real > 0.0
        assert isinstance(gs.amplitude(["1", "1"], phase_reference=ref), complex)
