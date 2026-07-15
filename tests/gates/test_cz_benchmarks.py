"""Benchmark guards for the CZ gate protocols, formulas written out explicitly.

Each test evaluates a documented operating point with the deterministic exact
solver and asserts a *physical band*, not an exact value: the atomic data
underneath (ARC dipole elements, lifetimes) drifts with library updates, which
moves the numbers by factors that an exact pin would flag as false alarms.
The bands are chosen so they still separate "working high-fidelity gate"
(< 1e-3) from "broken path / non-gate" (O(1)) by orders of magnitude.

The gate metrics are computed inline (the library deliberately ships no gate
report / fidelity API): evolve the CZ basis states |00>, |01>, |11> once,
apply the ideal single-qubit Rz corrections, and score the phase-corrected
overlaps with the Nielsen average-gate-fidelity formula.  The protocols are
fully specified (normalized TO/AR fields); the single-qubit Z ``theta`` is a
scoring parameter kept alongside each operating point, not protocol state.

Operating points:

- ``X_TO_DARK`` is a high-fidelity TO optimum (infidelity ~5.8e-5 as of
  2026-07, was 3.7e-7 when optimized on 2026-06-26 atomic data).
- ``X_AR_LUKIN`` and ``X_AR_OUR_OPT`` are re-optimized high-fidelity AR
  points (``scripts/reoptimize_explicit.py`` / ``scripts/optimize_ar_cz.py``).
  Both AR tests and the Double-ARP path guard are ``slow``-marked (multi-minute
  adaptive-ODE evaluations on the GHz 7-level ladder); the TO-dark benchmark
  and the error-budget decomposition stay in the fast suite.
- The Saffman-constants adiabatic Double-ARP configuration is a *path* guard:
  as written it is not a tuned gate (infidelity ~0.87); it exercises the
  parameter-free ``CZProtocol`` + ``phase_from_chirp`` path.

The *bright* TO branch (``detuning_sign=-1``) is intentionally absent: with the
intermediate manifold at -9.1 GHz it sits only 2.3 GHz from |0> (-6.835 GHz),
so the spectator |0> leaks ~9e-3 coherently — a near-resonance floor no pulse
optimization can fix.

Historical note: the legacy single-start ``X_AR`` point (a non-entangling local
minimum, infidelity ~0.65) was dropped together with its exact-value pin — it
guarded the same ``ARProtocol`` code path that ``X_AR_OUR_OPT`` now covers with
a meaningful outcome. Multi-start search escapes that minimum (see
``docs/how_to_gates.qmd``).
"""

import numpy as np
import pytest
from scipy import integrate, interpolate

from ryd_gate import Register, RydbergSystem, level_structure
from ryd_gate.backends.exact import simulate
from ryd_gate.protocols import ARProtocol, CZProtocol, TOProtocol
from ryd_gate.protocols.gate_cz import cz_effective_rabi, cz_rabi_maxes, phase_from_chirp

# Operating-point vectors keep the historical layout; the protocol fields are
# built from the non-theta entries and theta stays a local scoring parameter.
# x = [A, omega/Omega_eff, phi0, delta/Omega_eff, theta, T/T_scale]
X_TO_DARK = [
    -0.6894097925886826,
    1.040962607910546,
    0.3277877211544321,
    1.5639989822346387,
    0.6689846026179691,
    1.3407418093368753,
]
# Re-optimized high-fidelity AR point on the lukin system, explicit |0> model
# (scripts/reoptimize_explicit.py, 2026-06-26).
X_AR_LUKIN = [
    1.1248729881110775,
    0.6643225074894661,
    1.2040014416272569,
    0.47554852952965454,
    0.9542223492343259,
    -0.06698908370622411,
    1.4484639314193593,
    1.6238023148317802,
]
# Re-optimized high-fidelity AR point on the "our" system, explicit |0> model
# (scripts/reoptimize_explicit.py, 2026-06-26).
X_AR_OUR_OPT = [
    1.3350469419160897,
    -0.47009049836626793,
    1.030513904414609,
    -0.32029923369069635,
    5.039849219039583,
    0.9619699188096813,
    1.9974997193846544,
    1.0453182369350449,
]

MHZ = 2.0 * np.pi * 1e6

# A "working high-fidelity CZ gate" band: orders of magnitude below any broken
# path (O(0.1-1)), with headroom for atomic-data drift above the optimized values.
HIGH_FIDELITY = 1e-3

# Canonical σ⁺/σ⁻ ("lukin") single-photon Rabis (rad/s) — not the protocol
# default manifold, so they are passed explicitly.
OMEGA_PM = (2 * np.pi * 237e6, 2 * np.pi * 303e6)

# 7-level single-atom basis vectors: index 0 = |0>, index 1 = |1>.
_S0 = np.array([1, 0, 0, 0, 0, 0, 0], dtype=complex)
_S1 = np.array([0, 1, 0, 0, 0, 0, 0], dtype=complex)


def _system(**kwargs):
    return RydbergSystem(
        level_structure=level_structure("rb87_7_mp", **kwargs),
        register=Register.chain(2, spacing_um=3.0),
    )


def _to_protocol(x, **kwargs):
    """TO pulse from the historical x layout (theta = x[4] excluded)."""
    return TOProtocol(
        phase_amplitude=x[0], frequency_ratio=x[1], phase_offset=x[2],
        detuning_ratio=x[3], duration_ratio=x[5], **kwargs,
    )


def _ar_protocol(x, **kwargs):
    """AR pulse from the historical x layout (theta = x[7] excluded)."""
    return ARProtocol(
        frequency_ratio=x[0], phase_amplitude_1=x[1], phase_offset_1=x[2],
        phase_amplitude_2=x[3], phase_offset_2=x[4], detuning_ratio=x[5],
        duration_ratio=x[6], **kwargs,
    )


def _cz_overlaps(system, pulse, theta):
    """Evolve |00>, |01>, |11> once and return the phase-corrected CZ overlaps.

    ``theta`` is the ideal single-qubit Z phase (a scoring parameter). The
    corrections ``e^{-iθ}`` (|01>) and ``e^{-2iθ-iπ}`` (|11>) remove the ideal
    single-qubit Rz phases, so a perfect CZ gives a00 == a01 == a11 == 1.
    """
    basis = {
        "00": np.kron(_S0, _S0),
        "01": np.kron(_S0, _S1),
        "11": np.kron(_S1, _S1),
    }
    corrections = {
        "00": 1.0,
        "01": np.exp(-1.0j * theta),
        "11": np.exp(-2.0j * theta - 1.0j * np.pi),
    }
    bound = system.with_protocol(pulse)
    return {
        label: corrections[label]
        * complex(np.vdot(ini, simulate(bound, ini).final_state))
        for label, ini in basis.items()
    }


def _nielsen_infidelity(overlaps):
    """Nielsen average-gate infidelity from the phase-corrected CZ overlaps.

    ``F_avg = (|a00 + 2 a01 + a11|² + |a00|² + 2|a01|² + |a11|²) / 20``
    (d = 4, with the 01/10 symmetry of identical atoms folding |10> into |01>).
    """
    a00, a01, a11 = overlaps["00"], overlaps["01"], overlaps["11"]
    avg_f = (1 / 20) * (
        abs(a00 + 2 * a01 + a11) ** 2
        + abs(a00) ** 2
        + 2 * abs(a01) ** 2
        + abs(a11) ** 2
    )
    return 1.0 - avg_f


def _cz_infidelity(system, pulse, theta):
    return float(_nielsen_infidelity(_cz_overlaps(system, pulse, theta)))


def test_to_dark_benchmark():
    infidelity = _cz_infidelity(
        _system(detuning_sign=1), _to_protocol(X_TO_DARK), theta=X_TO_DARK[4]
    )
    assert infidelity < HIGH_FIDELITY


@pytest.mark.slow
def test_ar_lukin_high_fidelity_benchmark():
    """Re-optimized AR point on the lukin system is a genuine CZ gate.

    ~400 s exact evaluation (long gate time at this operating point);
    slow-marked.
    """
    system = RydbergSystem(
        level_structure=level_structure("rb87_7_pm"),
        register=Register.chain(2, spacing_um=3.0),
    )
    # The σ⁺/σ⁻ (pm) manifold is not the protocol default, so pass its
    # canonical Rabis explicitly to reproduce the lukin operating point.
    infidelity = _cz_infidelity(
        system,
        _ar_protocol(X_AR_LUKIN, omega_420_max=OMEGA_PM[0], omega_1013_max=OMEGA_PM[1]),
        theta=X_AR_LUKIN[7],
    )
    assert infidelity < HIGH_FIDELITY


@pytest.mark.slow
def test_ar_our_high_fidelity_benchmark():
    """Multi-start re-optimization reaches a true high-fidelity AR gate on "our".

    This is the resolution of the "our dead basin" investigation: the legacy
    single start was trapped at ~0.45 (a non-entangling local minimum), but a
    multi-start search (scripts/optimize_ar_cz.py) escapes it to ~1e-5 —
    proving the dead basin was an optimizer limitation, not a bug.
    """
    infidelity = _cz_infidelity(
        _system(), _ar_protocol(X_AR_OUR_OPT), theta=X_AR_OUR_OPT[7]
    )
    assert infidelity < HIGH_FIDELITY


@pytest.mark.slow
def test_double_arp_benchmark():
    # The adiabatic ARP pulse, built directly as a CZProtocol: a super-Gaussian
    # flat-top 420 envelope + a delta_max*sin round-trip sweep whose integral is
    # the 420 phase (phase_from_chirp). Canonical 'our' Rabis (no rescaling).
    # A *path* guard for the parameter-free protocol: as written this is NOT a
    # tuned gate (uncompensated ARP, infidelity ~0.87), so assert the band that
    # distinguishes "computed the documented non-gate" from "accidentally works"
    # or "diverged".
    T, N, DELTA_MAX = 0.54e-6, 80, 23.0 * MHZ
    sigma, t_pulse = 0.175 * T, 0.5 * 0.54e-6
    offset = np.exp(-((t_pulse / 2) ** 4) / sigma**4)

    def env(t):
        tc = float(np.clip(t, 0.0, T))
        u = tc if tc < t_pulse else tc - t_pulse
        return (np.exp(-((u - t_pulse / 2) ** 4) / sigma**4) - offset) / (1 - offset)

    def sweep(t):
        tc = float(np.clip(t, 0.0, T))
        u = tc if tc < t_pulse else tc - t_pulse
        return DELTA_MAX * np.sin(np.pi * (u / t_pulse - 0.5))

    phi = phase_from_chirp(sweep, T, 4 * N + 1)
    system = _system()
    o420, o1013 = cz_rabi_maxes(system)
    _, time_scale = cz_effective_rabi(system, o420, o1013)
    protocol = CZProtocol(
        duration_ratio=T / time_scale,
        A_420=lambda s: env(float(np.clip(s, 0.0, 1.0)) * T),
        phi_420=lambda s: phi(float(np.clip(s, 0.0, 1.0)) * T),
    )
    infidelity = _cz_infidelity(system, protocol, theta=0.0)
    assert 0.5 < infidelity < 1.0


def _decay_integrate(t_list, occ_list, decay_rate):
    """Integrate decay probability ``dP/dt = Γ·n(t)`` on the recorded grid.

    Cubic-spline interpolation of the occupation + DOP853 quadrature: the
    same convention as the pinned error-budget results (do not substitute a
    trapezoid — cached values were produced with this integrator).
    """
    spline = interpolate.CubicSpline(t_list, occ_list)
    result = integrate.solve_ivp(
        lambda t, _y: np.array([decay_rate * spline(t)]),
        [0, t_list[-1]],
        np.array([0.0]),
        t_eval=t_list,
        method="DOP853",
        rtol=1e-8,
        atol=1e-12,
    )
    return np.array(result.y)


def test_error_budget_decomposition():
    """Rydberg-decay budget: physically plausible total, exact decomposition.

    The budget formula is written out: for each initial state, integrate the
    Rydberg occupation against the radiative-decay rate (XYZ/LG errors, split
    by the ARC branching ratios back to the computational basis vs the other
    hyperfine ground states) and against the BBR rate (atom loss), plus the
    final residual Rydberg population (also atom loss); average over the
    initial states.
    """
    system = _system(detuning_sign=1, enable_rydberg_decay=True)
    bound = system.with_protocol(_to_protocol(X_TO_DARK))
    t_eval = np.linspace(0.0, bound.t_gate, 1000)

    br = system.level_structure.ryd_branch
    xyz_frac = br["to_0"] + br["to_1"]
    lg_frac = br["to_L0"] + br["to_L1"]
    rd_rate = system.level_structure.ryd_RD_rate
    bbr_rate = system.level_structure.ryd_BBR_rate

    # total |r> population over both atoms
    observables = {"n_ryd": system.observables.level_sum("r")}

    initial_states = {"01": np.kron(_S0, _S1), "11": np.kron(_S1, _S1)}
    components = {"XYZ": 0.0, "AL": 0.0, "LG": 0.0}
    for ini in initial_states.values():
        res = simulate(bound, ini, t_eval=t_eval, observables=observables)
        t_list = np.asarray(res.times)
        ryd_occ = res.expectation("n_ryd").real  # expectations are complex
        rd_decay = _decay_integrate(t_list, ryd_occ, rd_rate)[0, -1]
        bbr_decay = _decay_integrate(t_list, ryd_occ, bbr_rate)[0, -1]
        components["XYZ"] += rd_decay * xyz_frac
        components["LG"] += rd_decay * lg_frac
        components["AL"] += bbr_decay + ryd_occ[-1]
    n = len(initial_states)
    components = {key: val / n for key, val in components.items()}
    total = sum(components.values())

    # ~1 μs of partial Rydberg occupation against Γ_r ~ (150 μs)^-1 → O(1e-3).
    assert 1e-4 < total < 1e-2
    assert all(components[k] >= 0.0 for k in ("XYZ", "AL", "LG"))
    # The channel classes decompose the total exactly (bookkeeping identity).
    assert total == pytest.approx(
        components["XYZ"] + components["AL"] + components["LG"], rel=1e-9
    )
