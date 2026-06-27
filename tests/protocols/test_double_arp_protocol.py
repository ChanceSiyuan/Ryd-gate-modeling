import numpy as np

from ryd_gate.protocols.gate_cz import DoubleARPProtocol


class _FakeSystem:
    N = 2

    def meta(self, name, default=None):
        if name == "rabi_eff":
            return 2.0
        return default


class _FakeBlockRegistry:
    def __init__(self):
        h_const = np.zeros((7, 7), dtype=complex)
        h_const[2, 2] = 10.0
        h_const[3, 3] = 20.0
        h_const[4, 4] = 30.0

        h420 = np.zeros((7, 7), dtype=complex)
        h420[2, 1] = 1.0

        h1013 = np.zeros((7, 7), dtype=complex)
        h1013[5, 2] = 1.0
        h1013[6, 2] = 0.5

        self._blocks = {
            "H_const": type("_Block", (), {"matrix": h_const})(),
            "drive_420": type("_Block", (), {"matrix": h420})(),
            "H_1013": type("_Block", (), {"matrix": h1013})(),
        }

    def get(self, name):
        return self._blocks[name]


class _FakeStarkSystem(_FakeSystem):
    blocks = _FakeBlockRegistry()


def test_double_arp_protocol_channels_and_envelope():
    protocol = DoubleARPProtocol(
        omega_max=4.0,
        delta_max=3.0,
        t_gate=2.0,
        sigma=0.35,
        n_steps=10,
    )
    params = protocol.unpack_params([], _FakeSystem())

    assert protocol.required_channels == frozenset(
        {"drive_420", "drive_420_dag"}
    )
    assert np.isclose(protocol.envelope(0.0), 0.0)
    assert np.isclose(protocol.envelope(0.5), 1.0)
    assert np.isclose(protocol.envelope(1.0), 0.0)
    assert np.isclose(protocol.envelope(1.5), 1.0)
    assert np.isclose(protocol.envelope(2.0), 0.0)

    coeffs = protocol.get_drive_coefficients(0.5, params)
    assert set(coeffs) == {"drive_420", "drive_420_dag"}
    assert np.isclose(coeffs["drive_420_dag"], np.conjugate(coeffs["drive_420"]))


def test_double_arp_detuning_resets_each_half():
    protocol = DoubleARPProtocol(delta_max=3.0, t_gate=2.0, sigma=0.35)
    params = protocol.unpack_params([], _FakeSystem())

    assert np.isclose(protocol.detuning(0.0, params), -3.0)
    assert np.isclose(protocol.detuning(0.5, params), 0.0)
    assert np.isclose(protocol.detuning(1.0 - 1e-12, params), 3.0)
    assert np.isclose(protocol.detuning(1.0, params), -3.0)
    assert np.isclose(protocol.detuning(1.5, params), 0.0)
    assert np.isclose(protocol.detuning(2.0, params), 3.0)

    assert np.isclose(protocol.chirp_detuning(0.25, params), protocol.detuning(0.25, params))


def test_double_arp_stark_compensation_targets_effective_detuning():
    protocol = DoubleARPProtocol(
        omega_max=4.0,
        delta_max=3.0,
        t_gate=2.0,
        sigma=0.35,
        compensate_stark=True,
        stark_compensation_sign=-1.0,
    )
    params = protocol.unpack_params([], _FakeStarkSystem())

    t_peak = 0.5
    actual_detuning = protocol.chirp_detuning(t_peak, params) + protocol.stark_shift(
        t_peak, params
    )

    assert np.isclose(actual_detuning, protocol.detuning(t_peak, params))
