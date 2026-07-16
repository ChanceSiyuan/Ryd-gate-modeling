"""Pin the four frozen public namespaces of ryd_gate.

After the breaking rewrite the public API is exactly four frozen surfaces, each
declared by its module ``__all__``:

  * ``ryd_gate``            — 6 names
  * ``ryd_gate.protocols``  — 10 names
  * ``ryd_gate.physics``    — 5 names
  * ``ryd_gate.results``    — 6 names (3 result types + 3 PEPS evidence records)

For each namespace this file asserts ``__all__`` equals the exact list below (in
API order) and every listed name is importable.
"""

import importlib

import pytest

TOP_LEVEL_API = [
    "Register",
    "RydbergSystem",
    "NoiseModel",
    "level_structure",
    "simulate",
    "simulate_ensemble",
]

PROTOCOLS_API = [
    "CZProtocol",
    "TOProtocol",
    "ARProtocol",
    "SweepProtocol",
    "DigitalAnalogProtocol",
    "Direct297PiProtocol",
    "Direct297CZProtocol",
    "Direct297TOProtocol",
    "blackman_pulse",
    "phase_from_chirp",
]

PHYSICS_API = [
    "single_photon_rabi",
    "rb87_7_mp_rabi_frequencies",
    "rb87_297_clock_rabi_frequencies",
    "zeeman_shift_rad_s",
    "arc_pair_c6_rad_s_um6",
]

RESULTS_API = [
    "EvolutionResult",
    "GroundStateResult",
    "EnsembleResult",
    "PEPSAmplitudeEvidence",
    "PEPSSampleEvidence",
    "PEPSEvidence",
]

FROZEN_NAMESPACES = [
    ("ryd_gate", TOP_LEVEL_API),
    ("ryd_gate.protocols", PROTOCOLS_API),
    ("ryd_gate.physics", PHYSICS_API),
    ("ryd_gate.results", RESULTS_API),
]


class TestFrozenNamespaces:
    @pytest.mark.parametrize("module_name, expected", FROZEN_NAMESPACES)
    def test_all_is_exactly_frozen_list(self, module_name, expected):
        """``__all__`` equals the frozen list, in API order."""
        module = importlib.import_module(module_name)
        assert module.__all__ == expected

    @pytest.mark.parametrize("module_name, expected", FROZEN_NAMESPACES)
    def test_every_all_name_is_importable(self, module_name, expected):
        """Every name in ``__all__`` is a resolvable attribute of the module."""
        module = importlib.import_module(module_name)
        for name in expected:
            assert hasattr(module, name), f"{module_name}.{name} is missing"
            assert getattr(module, name) is not None

    def test_top_level_import_star_binds_exactly_the_six(self):
        """``from ryd_gate import *`` binds exactly the six frozen names."""
        namespace: dict = {}
        exec("from ryd_gate import *", namespace)
        bound = {k for k in namespace if not k.startswith("__")}
        assert bound == set(TOP_LEVEL_API)
