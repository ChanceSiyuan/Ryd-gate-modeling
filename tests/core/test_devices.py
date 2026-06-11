"""Tests for DeviceSpec validation at the product boundary (devices.py)."""

import numpy as np
import pytest

from ryd_gate.core.level_structures import LevelStructureSpec
from ryd_gate.devices import DeviceSpec
from ryd_gate.lattice import Register
from ryd_gate.protocols.channels import ChannelSpec
from ryd_gate.pulse import Pulse


class TestVirtualRb87:
    def test_channels_present(self):
        device = DeviceSpec.virtual_rb87()
        assert set(device.channels) == {"rydberg_global", "rydberg_local", "hyperfine_global"}
        ryd = device.channels["rydberg_global"]
        assert ryd.kind == "rydberg"
        assert ryd.transition == "1_r"
        assert ryd.addressing == "global"
        assert ryd.amplitude_channels == {"1r": "global_X", "01r": "drive_R"}
        assert ryd.detuning_channels == {"1r": "global_n", "01r": "delta_R"}
        assert device.channels["rydberg_local"].addressing == "local"
        assert device.channels["hyperfine_global"].kind == "microwave"

    def test_fixed_values(self):
        device = DeviceSpec.virtual_rb87()
        assert device.name == "virtual_rb87"
        assert device.dimensions == 2
        assert device.atom_species == "Rb87"
        assert device.allowed_level_structures == ("01", "1r", "01r", "ger", "analog_3", "rb87_7")
        assert device.default_level_structure == "01r"
        assert device.min_atom_distance_um == 2.0
        assert device.max_atom_num is None
        assert "C6_rad_s_um6" in device.interaction_coeffs


class TestValidateRegister:
    def test_valid_register_passes(self):
        device = DeviceSpec.virtual_rb87()
        assert device.validate_register(Register.chain(2, 4.0)) == []

    def test_min_distance_violation(self):
        device = DeviceSpec.virtual_rb87()
        issues = device.validate_register(Register.chain(2, 1.0))
        assert [i.code for i in issues] == ["register.min_distance"]

    def test_dimensions_violation(self):
        device = DeviceSpec.virtual_rb87()
        reg3d = Register(N=2, coords=[[0, 0, 0], [4, 0, 0]], sublattice=[0, 0], spacing_um=4.0)
        codes = {i.code for i in device.validate_register(reg3d)}
        assert "register.dimensions" in codes

    def test_max_atom_and_radius_when_configured(self):
        device = DeviceSpec(
            name="tiny",
            dimensions=2,
            atom_species="Rb87",
            allowed_level_structures=("1r",),
            default_level_structure="1r",
            min_atom_distance_um=2.0,
            max_atom_num=2,
            max_radial_distance_um=5.0,
        )
        codes = {i.code for i in device.validate_register(Register.chain(3, 4.0))}
        assert "register.max_atom_num" in codes
        assert "register.max_radial_distance" in codes


class TestValidateLevelStructure:
    def test_allowed_names_pass(self):
        device = DeviceSpec.virtual_rb87()
        for name in device.allowed_level_structures:
            assert device.validate_level_structure(name) == []

    def test_unsupported_name(self):
        device = DeviceSpec.virtual_rb87()
        issues = device.validate_level_structure("xy_model")
        assert [i.code for i in issues] == ["level_structure.unsupported"]

    def test_species_mismatch(self):
        device = DeviceSpec.virtual_rb87()
        spec = LevelStructureSpec(
            name="1r", levels=("1", "r"), rydberg_levels=("r",), species="Cs133"
        )
        issues = device.validate_level_structure(spec)
        assert [i.code for i in issues] == ["level_structure.species"]


class TestValidatePulse:
    def test_unknown_channel(self):
        device = DeviceSpec.virtual_rb87()
        issues = device.validate_pulse(Pulse.constant(1000, 1.0, 0.0), "nope")
        assert [i.code for i in issues] == ["channel.unknown"]

    def test_unconstrained_channel_passes(self):
        device = DeviceSpec.virtual_rb87()
        assert device.validate_pulse(Pulse.constant(1000, 1.0, 0.0), "rydberg_global") == []

    def test_limits_fire(self):
        limited = ChannelSpec(
            channel_id="ryd",
            kind="rydberg",
            transition="1_r",
            addressing="global",
            min_duration_ns=16,
            max_duration_ns=10_000,
            clock_period_ns=4,
            max_abs_amplitude_rad_per_us=2 * np.pi * 2.0,
            max_abs_detuning_rad_per_us=2 * np.pi * 20.0,
        )
        device = DeviceSpec(
            name="limited",
            dimensions=2,
            atom_species="Rb87",
            allowed_level_structures=("1r",),
            default_level_structure="1r",
            min_atom_distance_um=2.0,
            channels={"ryd": limited},
        )
        assert {i.code for i in device.validate_pulse(Pulse.constant(1002, 1.0, 0.0), "ryd")} == {
            "pulse.clock_period"
        }
        assert {i.code for i in device.validate_pulse(Pulse.constant(1000, 100.0, 0.0), "ryd")} == {
            "pulse.amplitude_limit"
        }

    def test_shared_rule_set_with_pulse_validate(self):
        """Device path and Pulse.validate must report identical codes."""
        channel = ChannelSpec(
            channel_id="ryd",
            kind="rydberg",
            transition="1_r",
            addressing="global",
            clock_period_ns=4,
            max_abs_amplitude_rad_per_us=1.0,
        )
        device = DeviceSpec(
            name="d",
            dimensions=2,
            atom_species="Rb87",
            allowed_level_structures=("1r",),
            default_level_structure="1r",
            min_atom_distance_um=2.0,
            channels={"ryd": channel},
        )
        pulse = Pulse.constant(1002, 5.0, 0.0)
        via_device = [i.code for i in device.validate_pulse(pulse, "ryd")]
        via_pulse = [i.code for i in pulse.validate(channel)]
        assert via_device == via_pulse == ["pulse.clock_period", "pulse.amplitude_limit"]


class TestPhysicsHelpers:
    def test_blockade_radius_formula(self):
        device = DeviceSpec.virtual_rb87()
        rabi = 2 * np.pi
        expected = ((device.interaction_coeffs["C6_rad_s_um6"] * 1e-6) / rabi) ** (1 / 6)
        assert device.rydberg_blockade_radius_um(rabi) == pytest.approx(expected)
        # sanity: order of magnitude for Rb87 at Omega = 2*pi rad/us
        assert 5.0 < device.rydberg_blockade_radius_um(rabi) < 20.0

    def test_blockade_radius_errors(self):
        device = DeviceSpec.virtual_rb87()
        with pytest.raises(ValueError):
            device.rydberg_blockade_radius_um(0.0)
        no_c6 = DeviceSpec(
            name="bare",
            dimensions=2,
            atom_species="Rb87",
            allowed_level_structures=("1r",),
            default_level_structure="1r",
            min_atom_distance_um=2.0,
        )
        with pytest.raises(ValueError, match="C6"):
            no_c6.rydberg_blockade_radius_um(1.0)

    def test_describe_contains_name_and_channels(self):
        device = DeviceSpec.virtual_rb87()
        text = device.describe()
        assert "virtual_rb87" in text
        for channel_id in device.channels:
            assert channel_id in text
