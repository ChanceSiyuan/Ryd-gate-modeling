"""YASTN finite-PEPS backend: real-time + ground-state parity against exact (E24/E25/E16/E20).

Requires the ``tn-2d`` extra (yastn). Covers exact/PEPS expectation + complex-amplitude
parity, report-only evidence, the two ported sampler kinds (belief propagation + CTM)
against the Born distribution their own amplitude defines, and the normalized
imaginary-time ground state vs dense diagonalization.

The evolutions are deliberately short: on a tree (chain / 1xN) the NTU update and BP
environment are near-exact, so a ~40-step second-order Strang solve already reproduces
exact to a few 1e-3.  Sampling Born checks only need ``sampler == amplitude`` on the
*same* PEPS state, so they run on a near-product (few-step) state whose small bond
dimension keeps the CTM boundary-MPS sampler cheap.  Module-scoped fixtures evolve each
configuration once and share it across the tests that read it.
"""

import itertools

import numpy as np
import pytest

from ryd_gate import RydbergSystem, level_structure
from ryd_gate.lattice import Register
from ryd_gate.protocols.sweep import SweepProtocol
from ryd_gate.simulate import simulate

pytest.importorskip("yastn")

TWO_PI = 2 * np.pi
_T_GATE = 0.1e-6

# 40 second-order Strang steps over the gate: enough for tree parity to a few 1e-3.
_RT_OPTS = {
    "time_step_s": 2.5e-9,
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
    # A short unconverged schedule: gauge-fixing holds for any state and the energy
    # is compared to dense eigh with a proportionately loose band.
    "imaginary_time_schedule": ((0.2, 30),),
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


def _born_check(result, sampler, n_sites, shots, ground_ref=None):
    """sampler and amplitude derive from the SAME PEPS state, so the deviation is pure
    multinomial sampling noise; the fixed seed makes this a deterministic Born check."""
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


# ── module-scoped short evolutions (each config evolved once) ─────────────────


@pytest.fixture(scope="module")
def rt_chain_bp():
    """chain(2) 1r, belief-propagation, both populations measured; shared by the
    amplitude-parity, evidence, sampling and seed-reproducibility tests."""
    system = _system(Register.chain(2, spacing_um=6.0))
    obs = {"n_r0": _n_r(system, 0), "n_r1": _n_r(system, 1)}
    res = simulate(system, backend="peps", observables=obs, backend_options=_RT_OPTS)
    return system, res


@pytest.fixture(scope="module")
def rt_chain_ctm():
    """chain(2) 1r with the CTM measurement path + a two-site correlator observable."""
    system = _system(Register.chain(2, spacing_um=6.0))
    obs = {"nrnr": _n_r(system, 0) @ _n_r(system, 1)}
    opts = {**_RT_OPTS, "measurement_method": "ctm"}
    res = simulate(system, backend="peps", observables=obs, backend_options=opts)
    return system, res


class TestRealTimeParity:
    # Tree geometries (chain / 1xN / Nx1) have no interaction loops: NTU and BP are
    # near-exact there, so a short Strang solve tracks exact to a few 1e-3.
    def test_complex_amplitude_phase_matches_exact(self, rt_chain_bp):
        system, peps = rt_chain_bp
        exact = simulate(system)
        for labels in (["1", "1"], ["1", "r"], ["r", "1"], ["r", "r"]):
            ea, pa = exact.amplitude(labels), peps.amplitude(labels)
            assert abs(ea - pa) < 3e-3  # magnitude AND phase
        assert isinstance(peps.amplitude(["1", "1"]), complex)

    def test_populations_match_exact(self, rt_chain_bp):
        system, peps = rt_chain_bp
        exact = simulate(system, observables={"n_r0": _n_r(system, 0), "n_r1": _n_r(system, 1)})
        for name in ("n_r0", "n_r1"):
            np.testing.assert_allclose(peps.expectation(name), exact.expectation(name), atol=2e-3)

    def test_01r_full_local_hamiltonian_matches_exact(self):
        # 01r qutrit: PEPS must carry the complete 3x3 DigitalAnalog local Hamiltonian
        # (|0> is a spectator level). chain(2) is a tree so PEPS is near-exact here.
        from ryd_gate.protocols.digital_analog import DigitalAnalogProtocol

        w = TWO_PI * 1e6
        proto = DigitalAnalogProtocol(
            t_gate_s=0.1e-6,
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
        # A few steps only: this pins OUR device wiring, not convergence.
        system = _system(Register.square(2, spacing_um=6.0))
        opts = {**_RT_OPTS, "time_step_s": 2e-8, "device": "cuda"}
        res = simulate(system, backend="peps", observables={"n_r0": _n_r(system, 0)}, backend_options=opts)
        assert np.all(np.isfinite(res.expectation("n_r0")))
        assert isinstance(res.amplitude(["1", "1", "1", "1"]), complex)
        assert res.peps_evidence.parameters["device"] == "cuda"

    def test_ctm_two_site_correlator(self, rt_chain_ctm):
        system, peps = rt_chain_ctm
        obs = {"nrnr": _n_r(system, 0) @ _n_r(system, 1)}
        exact = simulate(system, observables=obs)
        np.testing.assert_allclose(peps.expectation("nrnr"), exact.expectation("nrnr"), atol=2e-3)

    def test_bp_rejects_two_site_observable(self):
        system = _system(Register.chain(2, spacing_um=6.0))
        obs = {"nrnr": _n_r(system, 0) @ _n_r(system, 1)}
        with pytest.raises((ValueError, RuntimeError)):
            simulate(system, backend="peps", observables=obs, backend_options=_RT_OPTS)


class TestRealTimeEvidence:
    def test_evidence_populated_and_json(self, rt_chain_bp):
        _system_, peps = rt_chain_bp
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
        assert len(peps.peps_evidence.amplitudes) >= 1

    def test_no_observables_builds_no_environment(self):
        system = _system(Register.chain(2, spacing_um=6.0))
        peps = simulate(system, backend="peps", backend_options=_RT_OPTS)
        ev = peps.peps_evidence
        assert ev.environment_residual is None
        assert ev.environment_iterations is None

    def test_bond_starved_run_reports_truncation_without_raising(self):
        # Report-only NTU truncation (PEPS02, deliberate opposite of MPS E23): a
        # bond-dimension-1 ansatz on a strong-blockade 2x2 loop cannot represent the
        # entangled state, so the reported truncation error is large but the run does
        # NOT raise (unlike the MPS discarded-weight guard).
        system = _system(Register.square(2, spacing_um=6.0))
        opts = {**_RT_OPTS, "time_step_s": 5e-9, "bond_dimension": 1,
                "measurement_method": "ctm"}
        res = simulate(system, backend="peps", observables={"n_r0": _n_r(system, 0)},
                       backend_options=opts)
        ev = res.peps_evidence
        assert ev.max_ntu_truncation_error > 1e-6
        assert ev.cumulative_ntu_truncation_error >= ev.max_ntu_truncation_error


class TestSampling:
    """The two ported samplers must reproduce the Born distribution their own amplitude
    defines (the amplitude/norm boundary contraction is validated separately)."""

    def test_real_time_bp_sampling_matches_born(self, rt_chain_bp):
        _system_, res = rt_chain_bp
        _born_check(res, res.sample, n_sites=2, shots=800)

    def test_real_time_ctm_sampling_matches_born(self):
        # CTM sampling uses the boundary-MPS/EnvWindow path; exercise it on a genuine
        # 2D grid (a 2x2 plaquette). This is a sampler-wiring check — sampler and
        # amplitude read the SAME (truncated) PEPS state — so bond_dimension=2 is
        # deliberate: the per-shot boundary contraction cost scales steeply with D
        # (D=8 costs ~0.5 s/shot), while Born consistency holds at any D.
        system = _system(Register.square(2, spacing_um=6.0))
        opts = {**_RT_OPTS, "time_step_s": 1.25e-8, "bond_dimension": 2,
                "environment_bond_dimension": 8, "measurement_method": "ctm"}
        res = simulate(system, backend="peps", backend_options=opts)
        # Per-shot cost is dominated by the EnvWindow boundary sweeps, so shots are the
        # budget knob; 200 keeps the 5-sigma Born check meaningful.
        _born_check(res, res.sample, n_sites=4, shots=200)

    def test_ctm_sampling_on_chain_raises_cleanly(self, rt_chain_ctm):
        # A 1-wide chain makes the CTM boundary-MPS contraction degenerate; it must
        # raise a clear capability error, never hang.
        _system_, res = rt_chain_ctm
        with pytest.raises(RuntimeError, match="2D grid"):
            res.sample(shots=10, seed=0)

    def test_seed_reproducible_and_global_rng_untouched(self, rt_chain_bp):
        _system_, res = rt_chain_bp
        state = np.random.get_state()
        a = res.sample(shots=60, seed=7)
        b = res.sample(shots=60, seed=7)
        c = res.sample(shots=60, seed=8)
        assert a == b
        assert a != c
        # the module must not touch the global NumPy RNG
        s0, s1 = state, np.random.get_state()
        assert s0[0] == s1[0] and np.array_equal(s0[1], s1[1]) and s0[2:] == s1[2:]


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

    @pytest.fixture(scope="module")
    def ground(self):
        system = _system(Register.chain(2, spacing_um=6.0))
        at = _T_GATE / 2
        # A slightly longer (but still ~140-step, 2-site) schedule than _GROUND_OPTS:
        # 30 steps leave the energy ~18% off, this gets it inside a 10% sanity band.
        opts = {**_GROUND_OPTS, "imaginary_time_schedule": ((0.2, 60), (0.05, 80))}
        gs = system.ground_state(at=at, method="peps_imaginary_time",
                                 initial_state=["r", "1"], method_options=opts)
        return system, at, gs

    def test_energy_and_gauge_fix(self, ground):
        system, at, gs = ground
        _terms, e0, _v0 = self._dense_ground(system, at)
        # The short anneal is only partly converged: this is a wiring sanity band
        # (sign/scale/composition), not a convergence benchmark.
        assert gs.expectation("energy") == pytest.approx(e0, rel=1e-1)
        ev = gs.peps_evidence
        assert ev.hamiltonian_scale_rad_s > 0.0
        assert ev.cumulative_ntu_truncation_error is None

        # Gauge-fixing holds for ANY state (needs no convergence): the reference amplitude
        # is real-positive in its own gauge.
        ref = ["r", "1"]
        self_amp = gs.amplitude(ref, phase_reference=ref)
        assert abs(self_amp.imag) < 1e-6 * max(1.0, abs(self_amp))
        assert self_amp.real > 0.0
        assert isinstance(gs.amplitude(["1", "1"], phase_reference=ref), complex)

    def test_phase_reference_zero_amplitude_rejected(self):
        # With zero drive the Hamiltonian is diagonal, so imaginary time keeps the state
        # an exact product basis state (|1,1> here, since +detuning + repulsion penalise
        # |r>). A reference basis state that differs then has *exactly* zero amplitude and
        # cannot fix the global phase (E16).
        from ryd_gate.backends.peps._numerics import PEPSError

        system = RydbergSystem(
            level_structure=level_structure("1r"),
            register=Register.chain(2, spacing_um=6.0),
            protocol=SweepProtocol(
                t_gate_s=_T_GATE,
                omega_half_rad_s=lambda t: 0.0,
                detuning_rad_s=lambda t: TWO_PI * 1.0e6,
            ),
            interaction_cutoff_um=7.0,
        )
        gs = system.ground_state(at=_T_GATE / 2, method="peps_imaginary_time",
                                 initial_state=["1", "1"], method_options=_GROUND_OPTS)
        with pytest.raises(PEPSError, match="phase_reference amplitude is numerically zero"):
            gs.amplitude(["1", "1"], phase_reference=["r", "r"])
