"""Benchmark guards for the CZ gate protocols, formulas written out explicitly.

Each test evaluates a documented operating point with the deterministic exact
solver and asserts a *physical band*, not an exact value: the atomic data
underneath (ARC dipole elements, lifetimes, ARC-computed C6) drifts with library
updates, which moves the numbers. The bands separate "working high-fidelity gate"
(< 1e-3) from "broken path / non-gate" (O(1)) by orders of magnitude.

Gate metrics are computed inline (the library ships no fidelity API, A02/W02):
evolve the CZ basis states |00>, |01>, |11> once, read the diagonal return
amplitudes with ``result.amplitude(labels)`` (W01), apply the ideal Rz
corrections, and score the phase-corrected overlaps with the Nielsen
average-gate-fidelity formula. Protocols are fully specified (P20/P23): the two
peak Rabis and the signed intermediate detuning are explicit; the single-qubit Z
``theta`` is a scoring parameter kept alongside each operating point.

The intermediate detuning is now a protocol input (P19). C6 comes from ARC at
the preset's Rydberg level (I04): the ``rb87_7_mp`` 70S value barely moves, but
``rb87_7_pm`` 53S drops ~10x versus the old hardcoded coefficient, so the AR
"lukin" point is no longer a high-fidelity gate without re-optimization (xfail).
"""

import numpy as np
import pytest
from scipy import integrate, interpolate

from ryd_gate import Register, RydbergSystem, level_structure, simulate
from ryd_gate.protocols import ARProtocol, CZProtocol, TOProtocol, phase_from_chirp

# Canonical peak single-photon Rabis (rad/s) and intermediate detunings.
OMEGA_MP = (2 * np.pi * 491e6, 2 * np.pi * 185e6)   # sigma-/sigma+ ("our")
OMEGA_PM = (2 * np.pi * 237e6, 2 * np.pi * 303e6)   # sigma+/sigma- ("lukin")
DELTA_MP = 2 * np.pi * 9.1e9
DELTA_PM = 2 * np.pi * 7.8e9
RISE = 20e-9
HIGH_FIDELITY = 1e-3
MHZ = 2.0 * np.pi * 1e6

# x layouts (historical): TO = [A, omega/Omega_eff, phi0, delta/Omega_eff, theta, T/T_scale]
X_TO_DARK = [
    -0.6894097925886826, 1.040962607910546, 0.3277877211544321,
    1.5639989822346387, 0.6689846026179691, 1.3407418093368753,
]
# AR = [freq_ratio, A1, phi1, A2, phi2, delta_ratio, T/T_scale, theta]
X_AR_LUKIN = [
    1.1248729881110775, 0.6643225074894661, 1.2040014416272569, 0.47554852952965454,
    0.9542223492343259, -0.06698908370622411, 1.4484639314193593, 1.6238023148317802,
]
X_AR_OUR_OPT = [
    1.3350469419160897, -0.47009049836626793, 1.030513904414609, -0.32029923369069635,
    5.039849219039583, 0.9619699188096813, 1.9974997193846544, 1.0453182369350449,
]


def _to_protocol(x, *, omega, delta):
    return TOProtocol(
        intermediate_detuning_rad_s=delta,
        omega_420_max_rad_s=omega[0], omega_1013_max_rad_s=omega[1], rise_time_s=RISE,
        phase_amplitude_rad=x[0], modulation_frequency_ratio=x[1], phase_offset_rad=x[2],
        frequency_offset_ratio=x[3], duration_ratio=x[5],
    )


def _ar_protocol(x, *, omega, delta):
    return ARProtocol(
        intermediate_detuning_rad_s=delta,
        omega_420_max_rad_s=omega[0], omega_1013_max_rad_s=omega[1], rise_time_s=RISE,
        modulation_frequency_ratio=x[0], phase_amplitude_1_rad=x[1], phase_offset_1_rad=x[2],
        phase_amplitude_2_rad=x[3], phase_offset_2_rad=x[4], frequency_offset_ratio=x[5],
        duration_ratio=x[6],
    )


def _bound(ls_name, pulse, **ls_kwargs):
    return RydbergSystem(
        level_structure=level_structure(ls_name, **ls_kwargs),
        register=Register.chain(2, spacing_um=3.0),
        protocol=pulse,
    )


def _cz_overlaps(system, theta):
    """Evolve |00>, |01>, |11> once; phase-corrected diagonal return amplitudes."""
    labels = {"00": ["0", "0"], "01": ["0", "1"], "11": ["1", "1"]}
    corr = {"00": 1.0, "01": np.exp(-1.0j * theta), "11": np.exp(-2.0j * theta - 1.0j * np.pi)}
    return {k: corr[k] * simulate(system, labels[k]).amplitude(labels[k]) for k in labels}


def _nielsen_infidelity(overlaps):
    a00, a01, a11 = overlaps["00"], overlaps["01"], overlaps["11"]
    avg_f = (1 / 20) * (
        abs(a00 + 2 * a01 + a11) ** 2 + abs(a00) ** 2 + 2 * abs(a01) ** 2 + abs(a11) ** 2
    )
    return 1.0 - avg_f


def _cz_infidelity(system, theta):
    return float(_nielsen_infidelity(_cz_overlaps(system, theta)))


def test_to_dark_benchmark():
    system = _bound("rb87_7_mp", _to_protocol(X_TO_DARK, omega=OMEGA_MP, delta=DELTA_MP))
    assert _cz_infidelity(system, theta=X_TO_DARK[4]) < HIGH_FIDELITY


@pytest.mark.slow
@pytest.mark.xfail(
    reason="rb87_7_pm 53S ARC C6 is ~10x weaker than the old hardcoded value (I04); "
    "the lukin AR point needs re-optimization for the new blockade.",
    strict=False,
)
def test_ar_lukin_high_fidelity_benchmark():
    system = _bound("rb87_7_pm", _ar_protocol(X_AR_LUKIN, omega=OMEGA_PM, delta=DELTA_PM))
    assert _cz_infidelity(system, theta=X_AR_LUKIN[7]) < HIGH_FIDELITY


@pytest.mark.slow
def test_ar_our_high_fidelity_benchmark():
    """Multi-start re-optimized AR gate on the "our" (70S) manifold; C6 barely moved."""
    system = _bound("rb87_7_mp", _ar_protocol(X_AR_OUR_OPT, omega=OMEGA_MP, delta=DELTA_MP))
    assert _cz_infidelity(system, theta=X_AR_OUR_OPT[7]) < HIGH_FIDELITY


@pytest.mark.slow
def test_double_arp_benchmark():
    # Adiabatic ARP as a parameter-free CZProtocol (super-Gaussian flat-top + sin
    # round-trip sweep integrated to the 420 phase). A *path* guard, not a tuned
    # gate (infidelity ~0.87): asserts the band that separates the documented
    # non-gate from "accidentally works"/"diverged".
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

    phi = phase_from_chirp(sweep, T, n_samples=4 * N + 1)
    protocol = CZProtocol(
        t_gate_s=T, intermediate_detuning_rad_s=DELTA_MP,
        omega_420_max_rad_s=OMEGA_MP[0], omega_1013_max_rad_s=OMEGA_MP[1],
        envelope_420=env, phase_420_rad=phi,
    )
    system = _bound("rb87_7_mp", protocol)
    assert 0.5 < _cz_infidelity(system, theta=0.0) < 1.0


def _decay_integrate(t_list, occ_list, decay_rate):
    """Integrate ``dP/dt = Γ·n(t)`` on the recorded grid (cubic spline + DOP853)."""
    spline = interpolate.CubicSpline(t_list, occ_list)
    result = integrate.solve_ivp(
        lambda t, _y: np.array([decay_rate * spline(t)]),
        [0, t_list[-1]], np.array([0.0]), t_eval=t_list, method="DOP853",
        rtol=1e-8, atol=1e-12,
    )
    return np.array(result.y)


def test_error_budget_decomposition():
    """First-order Rydberg-decay budget (E09): coherent evolution + Γ_k ∫<n_r> dt.

    Decay is characterization data on the level structure, not evolution (E08);
    integrate the Rydberg occupation against the radiative rate (split by the ARC
    branching ratios into computational vs other ground states) and the BBR rate.
    """
    ls = level_structure("rb87_7_mp")
    system = _bound("rb87_7_mp", _to_protocol(X_TO_DARK, omega=OMEGA_MP, delta=DELTA_MP))
    t_eval = np.linspace(0.0, system.t_gate, 1000)

    br = ls.branching_ratios["r"]
    xyz_frac = br["to_0"] + br["to_1"]
    lg_frac = br["to_L0"] + br["to_L1"]
    rd_rate = ls.decay_rates_per_s["r"]["radiative"]
    bbr_rate = ls.decay_rates_per_s["r"]["blackbody"]

    n_ryd = sum(system.observables.n("r", i) for i in range(system.N))
    components = {"XYZ": 0.0, "AL": 0.0, "LG": 0.0}
    initial_states = (["0", "1"], ["1", "1"])
    for labels in initial_states:
        res = simulate(system, labels, t_eval=t_eval, observables={"n_ryd": n_ryd})
        t_list = np.asarray(res.times)
        ryd_occ = res.expectation("n_ryd")
        rd_decay = _decay_integrate(t_list, ryd_occ, rd_rate)[0, -1]
        bbr_decay = _decay_integrate(t_list, ryd_occ, bbr_rate)[0, -1]
        components["XYZ"] += rd_decay * xyz_frac
        components["LG"] += rd_decay * lg_frac
        components["AL"] += bbr_decay + ryd_occ[-1]
    n = len(initial_states)
    components = {key: val / n for key, val in components.items()}
    total = sum(components.values())

    assert 1e-4 < total < 1e-2
    assert all(components[k] >= 0.0 for k in ("XYZ", "AL", "LG"))
    assert total == pytest.approx(components["XYZ"] + components["AL"] + components["LG"], rel=1e-9)
