"""phase_from_chirp + direct adiabatic CZProtocol construction.

The adiabatic ARP pulse is no longer a dedicated protocol class — it is just a
CZProtocol whose 420 phase is the chirp integral phase_from_chirp(detuning).
"""

from types import SimpleNamespace

import numpy as np
import pytest

from ryd_gate.protocols import CZProtocol
from ryd_gate.protocols.gate_cz import phase_from_chirp


def test_phase_from_chirp_constant_is_linear():
    """phi(t) = ∫_0^t c dt' = c*t for a constant chirp, clamped outside [0, t_gate]."""
    c = 1.234
    phi = phase_from_chirp(lambda t: c, t_gate=2.0, n_samples=201)
    assert np.isclose(phi(0.0), 0.0)
    assert np.isclose(phi(1.0), c * 1.0)
    assert np.isclose(phi(2.0), c * 2.0)
    assert np.isclose(phi(3.0), c * 2.0)   # clamps at t_gate
    assert np.isclose(phi(-1.0), 0.0)      # clamps at 0


def test_phase_from_chirp_cosine_sweep():
    """A round-trip cosine sweep integrates to the known sine; net phase is 0 at t_gate."""
    T, A = 1.0, 5.0
    phi = phase_from_chirp(lambda t: -A * np.cos(2 * np.pi * t / T), t_gate=T, n_samples=4001)
    for t in (0.1, 0.37, 0.5, 0.83):
        exact = -(A * T / (2 * np.pi)) * np.sin(2 * np.pi * t / T)
        assert np.isclose(phi(t), exact, atol=1e-4)
    assert np.isclose(phi(T), 0.0, atol=1e-6)


def test_phase_from_chirp_validation():
    with pytest.raises(ValueError):
        phase_from_chirp(lambda t: 0.0, t_gate=0.0)
    with pytest.raises(ValueError):
        phase_from_chirp(lambda t: 0.0, t_gate=1.0, n_samples=1)


class _FakeSystem:
    """Duck-typed system: a single 420 leg, no 1013 group (analog-like).

    Carries ``time_scale = 1.0`` so a CZProtocol ``duration_ratio`` resolves
    1:1 to seconds (``t_gate == duration_ratio``).
    """

    N = 2
    level_structure = SimpleNamespace(
        name="fake_analog",
        laser_channel_ratios={"420": {"E[e,g]": 1.0}},
        time_scale=1.0,
    )


_OM420 = r"$\Omega_{420}$"
_OM1013 = r"$\Omega_{1013}$"
_DPHI420 = r"$\dot\phi_{420}$"
_DPHI1013 = r"$\dot\phi_{1013}$"


def test_pulse_traces_includes_amplitudes_and_chirps():
    proto = CZProtocol(
        duration_ratio=2.0, A_420=lambda s: 1.0, phi_420=lambda s: 0.7,
        omega_420_max=2.0, omega_1013_max=3.0,
    )
    tr = proto.pulse_traces(0.9, _FakeSystem())
    assert set(tr) == {_OM420, _OM1013, _DPHI420, _DPHI1013}


def test_pulse_traces_constant_phase_zero_chirp():
    """A constant optical phase has zero chirp; amplitudes are Omega_max*A."""
    proto = CZProtocol(
        duration_ratio=2.0, A_420=lambda s: 0.5, phi_420=lambda s: 0.7,
        omega_420_max=2.0, omega_1013_max=3.0,   # phi_1013 defaults to 0
    )
    tr = proto.pulse_traces(0.9, _FakeSystem())
    assert np.isclose(tr[_DPHI420], 0.0, atol=1e-6)
    assert np.isclose(tr[_DPHI1013], 0.0, atol=1e-6)
    assert np.isclose(tr[_OM420], 2.0 * 0.5)
    assert np.isclose(tr[_OM1013], 3.0 * 1.0)


def test_pulse_traces_linear_phase_constant_chirp():
    """phi(s) = delta * s * t_gate  ->  dot_phi = delta (rad/s), incl. boundaries."""
    T, delta = 2.0, 1.5e7
    proto = CZProtocol(
        duration_ratio=T, A_420=lambda s: 1.0, phi_420=lambda s: delta * s * T,
        omega_420_max=1.0, omega_1013_max=1.0,
    )
    for t in (0.0, 0.5, 1.3, T):   # endpoints exercise the one-sided difference
        tr = proto.pulse_traces(t, _FakeSystem())
        assert np.isclose(tr[_DPHI420], delta, rtol=1e-4)


def test_pulse_traces_chirp_matches_phase_from_chirp():
    """A phase built from phase_from_chirp recovers its input chirp away from edges."""
    T, A = 1.0, 2 * np.pi * 20e6
    chirp = lambda t: -A * np.cos(2 * np.pi * t / T)
    phi = phase_from_chirp(chirp, T, 4001)
    proto = CZProtocol(
        duration_ratio=T, A_420=lambda s: 1.0,
        phi_420=lambda s: phi(float(np.clip(s, 0.0, 1.0)) * T),
        omega_420_max=1.0, omega_1013_max=1.0,
    )
    for t in (0.13, 0.37, 0.5, 0.71, 0.88):   # away from the t=0 / t=T boundaries
        tr = proto.pulse_traces(t, _FakeSystem())
        assert np.isclose(tr[_DPHI420], chirp(t), atol=A * 5e-3)


def test_plot_stacks_trace_subplots():
    """CZProtocol.plot(system) stacks the 4 traces (amplitudes + chirps) in
    their own panels with a shared time axis and returns (fig, [axes])."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    proto = CZProtocol(
        duration_ratio=1e-6, A_420=lambda s: 1.0, phi_420=lambda s: 3.0e7 * s * 1e-6,
        omega_420_max=1.0, omega_1013_max=1.0,
    )
    fig, axs = proto.plot(_FakeSystem(), n_points=64)
    assert isinstance(axs, list) and len(axs) == 4       # one panel per trace
    assert axs[0].get_xlabel() == "" and axs[-1].get_xlabel() != ""   # shared x: only bottom labeled
    plt.close(fig)


def test_direct_adiabatic_cz_protocol():
    """An adiabatic pulse is a CZProtocol whose 420 phase is the chirp integral."""
    T, D_AMP = 2.0, 3.0
    phi = phase_from_chirp(lambda t: D_AMP, t_gate=T, n_samples=257)  # constant chirp
    proto = CZProtocol(
        duration_ratio=T,                                 # time_scale 1.0 -> t_gate = T
        A_420=lambda s: 1.0,                              # flat envelope
        phi_420=lambda s: phi(float(np.clip(s, 0.0, 1.0)) * T),
        omega_420_max=2.0, omega_1013_max=3.0,
    )
    fake = _FakeSystem()
    assert proto.resolve_t_gate(fake) == T
    ctx = proto._resolve(fake)
    coeffs = proto.get_drive_coefficients(0.5 * T, ctx)
    # only the 420 group on this duck system; forward primitive channels only (the
    # compiler adds each leg's h.c.).
    assert set(coeffs) == {"E[e,g]"}
    # phase at t = T/2 is D_AMP * (T/2) = 3.0  ->  arg(c420) = -3.0
    wrap = lambda a: float(np.angle(np.exp(1j * a)))
    assert np.isclose(wrap(np.angle(coeffs["E[e,g]"])), wrap(-D_AMP * T / 2))
