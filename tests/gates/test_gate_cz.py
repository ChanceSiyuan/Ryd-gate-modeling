"""Fast pins for the CZ-gate protocols (``ryd_gate.protocols.gate_cz``).

Covers the three public protocols — generic :class:`CZProtocol`, Time-Optimal
:class:`TOProtocol`, Amplitude-Robust :class:`ARProtocol` — at the construction /
resolve seam (P17/P19-P24), plus one short seven-level integration sentinel.

Everything here is deterministic and coefficient-level except the final sentinel,
which runs a genuine two-atom ``rb87_7_mp`` ODE. That sentinel is deliberately
short: the |0> clock-hyperfine (6.8 GHz) diagonal sets an ODE-cost floor whenever
|0> is populated (as it is for |01>/|10>), so the gate duration is kept tiny.
ARC C6 for ``rb87_7_mp`` is lru-cached per process, so it is paid once.
"""

import numpy as np
import pytest

from ryd_gate import Register, RydbergSystem, level_structure, simulate
from ryd_gate.protocols import ARProtocol, CZProtocol, TOProtocol, blackman_pulse
from ryd_gate.protocols._resolved import _ChannelDrive, _LaserDrive

MHZ = 2 * np.pi * 1e6


@pytest.fixture(scope="module")
def sys7():
    """A two-atom ``rb87_7_mp`` system (only ``level_structure`` is used at resolve time)."""
    return RydbergSystem(
        level_structure=level_structure("rb87_7_mp"),
        register=Register.chain(2, spacing_um=4.0),
        protocol=CZProtocol(
            t_gate_s=0.05e-6, intermediate_detuning_rad_s=50 * MHZ,
            omega_420_max_rad_s=10 * MHZ, omega_1013_max_rad_s=10 * MHZ,
            envelope_420=lambda t: 1.0, phase_420_rad=lambda t: 0.0,
        ),
    )


def _omega_eff(o420, o1013, delta):
    return o420 * o1013 / (2 * abs(delta))


def _cz_kwargs(**overrides):
    kw = dict(
        t_gate_s=0.05e-6, intermediate_detuning_rad_s=50 * MHZ,
        omega_420_max_rad_s=10 * MHZ, omega_1013_max_rad_s=10 * MHZ,
        envelope_420=lambda t: 1.0, phase_420_rad=lambda t: 0.0,
    )
    kw.update(overrides)
    return kw


def _to_kwargs(**overrides):
    kw = dict(
        intermediate_detuning_rad_s=50 * MHZ, omega_420_max_rad_s=10 * MHZ,
        omega_1013_max_rad_s=10 * MHZ, rise_time_s=5e-9, phase_amplitude_rad=0.5,
        modulation_frequency_ratio=1.2, phase_offset_rad=0.3,
        frequency_offset_ratio=-0.1, duration_ratio=1.4,
    )
    kw.update(overrides)
    return kw


def _ar_kwargs(**overrides):
    kw = dict(
        intermediate_detuning_rad_s=50 * MHZ, omega_420_max_rad_s=10 * MHZ,
        omega_1013_max_rad_s=10 * MHZ, rise_time_s=5e-9, modulation_frequency_ratio=1.1,
        phase_amplitude_1_rad=0.6, phase_offset_1_rad=0.2, phase_amplitude_2_rad=0.4,
        phase_offset_2_rad=0.9, frequency_offset_ratio=-0.07, duration_ratio=1.5,
    )
    kw.update(overrides)
    return kw


def _laser(resolved, group):
    return next(d for d in resolved.drives if isinstance(d, _LaserDrive) and d.group == group)


# ── construction validation (_require_positive / _require_finite / _require_callable) ─


def test_cz_constructor_validation():
    with pytest.raises(ValueError, match="t_gate_s"):
        CZProtocol(**_cz_kwargs(t_gate_s=0.0))
    with pytest.raises(ValueError, match="t_gate_s"):
        CZProtocol(**_cz_kwargs(t_gate_s=-1.0))
    with pytest.raises(ValueError, match="t_gate_s"):
        CZProtocol(**_cz_kwargs(t_gate_s=True))  # bool rejected by _require_positive
    with pytest.raises(ValueError, match="omega_420_max_rad_s"):
        CZProtocol(**_cz_kwargs(omega_420_max_rad_s=0.0))
    with pytest.raises(ValueError, match="omega_1013_max_rad_s"):
        CZProtocol(**_cz_kwargs(omega_1013_max_rad_s=-3.0))
    # CZ allows zero detuning, but the value must be finite (_require_finite).
    with pytest.raises(ValueError, match="intermediate_detuning_rad_s"):
        CZProtocol(**_cz_kwargs(intermediate_detuning_rad_s=np.inf))
    with pytest.raises(ValueError, match="intermediate_detuning_rad_s"):
        CZProtocol(**_cz_kwargs(intermediate_detuning_rad_s=np.nan))
    # _require_callable
    with pytest.raises(TypeError, match="envelope_420"):
        CZProtocol(**_cz_kwargs(envelope_420=1.0))
    with pytest.raises(TypeError, match="phase_420_rad"):
        CZProtocol(**_cz_kwargs(phase_420_rad=2.0))
    with pytest.raises(TypeError, match="envelope_1013"):
        CZProtocol(**_cz_kwargs(envelope_1013=5.0))
    with pytest.raises(TypeError, match="phase_1013_rad"):
        CZProtocol(**_cz_kwargs(phase_1013_rad=5.0))


def test_to_constructor_validation():
    with pytest.raises(ValueError, match="nonzero intermediate_detuning"):
        TOProtocol(**_to_kwargs(intermediate_detuning_rad_s=0.0))
    with pytest.raises(ValueError, match="omega_420_max_rad_s"):
        TOProtocol(**_to_kwargs(omega_420_max_rad_s=0.0))
    with pytest.raises(ValueError, match="rise_time_s"):
        TOProtocol(**_to_kwargs(rise_time_s=0.0))
    with pytest.raises(ValueError, match="duration_ratio"):
        TOProtocol(**_to_kwargs(duration_ratio=-1.0))
    with pytest.raises(ValueError, match="phase_amplitude_rad"):
        TOProtocol(**_to_kwargs(phase_amplitude_rad=np.inf))
    with pytest.raises(ValueError, match="modulation_frequency_ratio"):
        TOProtocol(**_to_kwargs(modulation_frequency_ratio=np.nan))


def test_ar_constructor_validation():
    with pytest.raises(ValueError, match="nonzero intermediate_detuning"):
        ARProtocol(**_ar_kwargs(intermediate_detuning_rad_s=0.0))
    with pytest.raises(ValueError, match="omega_1013_max_rad_s"):
        ARProtocol(**_ar_kwargs(omega_1013_max_rad_s=0.0))
    with pytest.raises(ValueError, match="rise_time_s"):
        ARProtocol(**_ar_kwargs(rise_time_s=-1e-9))
    with pytest.raises(ValueError, match="duration_ratio"):
        ARProtocol(**_ar_kwargs(duration_ratio=0.0))
    with pytest.raises(ValueError, match="phase_amplitude_1_rad"):
        ARProtocol(**_ar_kwargs(phase_amplitude_1_rad=np.inf))


# ── effective Rabi and gate duration (via system.t_gate, no simulation) ──────


def test_to_t_gate_is_duration_ratio_over_effective_rabi(sys7):
    o420, o1013, delta, dur = 10 * MHZ, 7 * MHZ, 50 * MHZ, 1.3
    p = TOProtocol(**_to_kwargs(
        omega_420_max_rad_s=o420, omega_1013_max_rad_s=o1013,
        intermediate_detuning_rad_s=delta, duration_ratio=dur,
    ))
    expected = dur * 2 * np.pi / _omega_eff(o420, o1013, delta)
    assert sys7.with_protocol(p).t_gate == pytest.approx(expected)


def test_ar_t_gate_uses_absolute_detuning(sys7):
    o420, o1013, dur = 12 * MHZ, 8 * MHZ, 1.7
    expected = dur * 2 * np.pi / _omega_eff(o420, o1013, 40 * MHZ)
    for delta in (40 * MHZ, -40 * MHZ):  # Omega_eff uses |Delta|: sign leaves t_gate unchanged
        p = ARProtocol(**_ar_kwargs(
            omega_420_max_rad_s=o420, omega_1013_max_rad_s=o1013,
            intermediate_detuning_rad_s=delta, duration_ratio=dur,
        ))
        assert sys7.with_protocol(p).t_gate == pytest.approx(expected)


def test_to_resolve_requires_two_rise_within_gate(sys7):
    # Tiny duration_ratio -> t_gate << 2*rise_time_s -> ValueError at resolve time.
    p = TOProtocol(**_to_kwargs(duration_ratio=0.001, rise_time_s=1e-6))
    with pytest.raises(ValueError, match=r"2\*rise_time_s <= t_gate"):
        p._resolve(sys7)


def test_ar_resolve_requires_two_rise_within_gate(sys7):
    p = ARProtocol(**_ar_kwargs(duration_ratio=0.001, rise_time_s=1e-6))
    with pytest.raises(ValueError, match=r"2\*rise_time_s <= t_gate"):
        p._resolve(sys7)


# ── resolve structure: intermediate-detuning diagonal drives ─────────────────


@pytest.mark.parametrize("delta", [50 * MHZ, -30 * MHZ])
def test_cz_resolve_adds_one_diagonal_drive_per_intermediate_level(sys7, delta):
    resolved = CZProtocol(**_cz_kwargs(intermediate_detuning_rad_s=delta))._resolve(sys7)
    lasers = [d for d in resolved.drives if isinstance(d, _LaserDrive)]
    channels = [d for d in resolved.drives if isinstance(d, _ChannelDrive)]
    assert {d.group for d in lasers} == {"420", "1013"}
    inter = sys7.level_structure._intermediate_levels
    assert {d.channel for d in channels} == {f"E[{lev},{lev}]" for lev in inter}
    for d in channels:  # constant in time, signed detuning written verbatim (P19)
        assert d.coefficient(0.0) == delta
        assert d.coefficient(123e-9) == delta


def test_cz_resolve_zero_detuning_adds_no_channel_drives(sys7):
    resolved = CZProtocol(**_cz_kwargs(intermediate_detuning_rad_s=0.0))._resolve(sys7)
    assert all(isinstance(d, _LaserDrive) for d in resolved.drives)
    assert {d.group for d in resolved.drives} == {"420", "1013"}


def test_cz_default_1013_is_flat_unit_envelope_zero_phase(sys7):
    o1013 = 7 * MHZ
    resolved = CZProtocol(**_cz_kwargs(omega_1013_max_rad_s=o1013))._resolve(sys7)
    d1013 = _laser(resolved, "1013")
    for t in (0.0, 13e-9, 41e-9):  # envelope_1013 == 1, phase_1013 == 0
        assert d1013.coefficient(t) == pytest.approx(o1013 + 0j)


# ── phase families: coefficient-level waveform checks ────────────────────────


def test_to_resolved_420_drive_matches_blackman_cosine_phase_family(sys7):
    o420, o1013, delta = 10 * MHZ, 10 * MHZ, 50 * MHZ
    rise, amp, mod_ratio, off, freq_off, dur = 5e-9, 0.5, 1.2, 0.3, -0.1, 1.4
    p = TOProtocol(
        intermediate_detuning_rad_s=delta, omega_420_max_rad_s=o420,
        omega_1013_max_rad_s=o1013, rise_time_s=rise, phase_amplitude_rad=amp,
        modulation_frequency_ratio=mod_ratio, phase_offset_rad=off,
        frequency_offset_ratio=freq_off, duration_ratio=dur,
    )
    resolved = p._resolve(sys7)
    t_gate = resolved.t_gate
    d420 = _laser(resolved, "420")
    omega_eff = _omega_eff(o420, o1013, delta)
    omega_mod = mod_ratio * omega_eff
    delta_phase = freq_off * omega_eff
    for s in (0.35, 0.65):  # two sample times
        t = s * t_gate
        env = blackman_pulse(t, rise, t_gate)
        phase = amp * np.cos(omega_mod * t + off) + delta_phase * t
        assert d420.coefficient(t) == pytest.approx(o420 * env * np.exp(-1j * phase))
    assert _laser(resolved, "1013").coefficient(0.4 * t_gate) == pytest.approx(o1013 + 0j)


def test_ar_resolved_420_drive_matches_dual_sine_phase_family(sys7):
    o420, o1013, delta = 10 * MHZ, 10 * MHZ, 50 * MHZ
    rise, mod_ratio = 5e-9, 1.1
    a1, off1, a2, off2, freq_off, dur = 0.6, 0.2, 0.4, 0.9, -0.07, 1.5
    p = ARProtocol(
        intermediate_detuning_rad_s=delta, omega_420_max_rad_s=o420,
        omega_1013_max_rad_s=o1013, rise_time_s=rise, modulation_frequency_ratio=mod_ratio,
        phase_amplitude_1_rad=a1, phase_offset_1_rad=off1, phase_amplitude_2_rad=a2,
        phase_offset_2_rad=off2, frequency_offset_ratio=freq_off, duration_ratio=dur,
    )
    resolved = p._resolve(sys7)
    t_gate = resolved.t_gate
    d420 = _laser(resolved, "420")
    omega_eff = _omega_eff(o420, o1013, delta)
    omega_mod = mod_ratio * omega_eff
    delta_phase = freq_off * omega_eff
    for s in (0.3, 0.55, 0.8):  # several sample times
        t = s * t_gate
        env = blackman_pulse(t, rise, t_gate)
        phase = (
            a1 * np.sin(omega_mod * t + off1)
            + a2 * np.sin(2 * omega_mod * t + off2)
            + delta_phase * t
        )
        assert d420.coefficient(t) == pytest.approx(o420 * env * np.exp(-1j * phase))


# ── non-finite envelope/phase guard (evaluated lazily at coefficient time) ───


def test_nonfinite_envelope_or_phase_raises_at_coefficient_time(sys7):
    bad_env = _laser(CZProtocol(**_cz_kwargs(envelope_420=lambda t: np.inf))._resolve(sys7), "420")
    with pytest.raises(ValueError, match="non-finite"):
        bad_env.coefficient(1e-9)
    bad_phase = _laser(CZProtocol(**_cz_kwargs(phase_420_rad=lambda t: np.nan))._resolve(sys7), "420")
    with pytest.raises(ValueError, match="non-finite"):
        bad_phase.coefficient(1e-9)


# ── one short seven-level ODE sentinel ───────────────────────────────────────


def test_two_atom_cz_ode_returns_to_subspace_and_is_symmetric(sys7):
    """Short two-atom ``rb87_7_mp`` CZ ODE at MHz-scale detuning.

    Physically meaningful sentinel: the population stays in / returns to the
    computational subspace within a loose band, and |01> and |10> give identical
    return amplitudes (register permutation symmetry). ODE tolerances are the
    project defaults (loose tolerances corrupt benchmarks).
    """
    proto = CZProtocol(
        t_gate_s=10e-9, intermediate_detuning_rad_s=50 * MHZ,
        omega_420_max_rad_s=10 * MHZ, omega_1013_max_rad_s=10 * MHZ,
        envelope_420=lambda t: 1.0, phase_420_rad=lambda t: 0.0,
    )
    system = sys7.with_protocol(proto)  # reuses the lru-cached ARC C6
    comp = sum(system.observables.n(lev, i) for lev in ("0", "1") for i in range(system.N))
    r01, r10 = simulate(system, [["0", "1"], ["1", "0"]], observables={"comp": comp})

    for r in (r01, r10):  # returns to the two-atom computational subspace (loose band)
        pop = float(r.expectation("comp")[-1])
        assert 1.9 < pop <= 2.0 + 1e-9
    assert float(r01.expectation("comp")[-1]) < 1.999  # the drive actually excites out of it

    a01 = r01.amplitude(["0", "1"])
    a10 = r10.amplitude(["1", "0"])
    np.testing.assert_allclose(a01, a10, rtol=1e-6, atol=1e-9)  # |01>/|10> symmetry
