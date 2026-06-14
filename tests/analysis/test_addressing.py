"""Regression tests for the analysis public surface (addressing + observables).

Pins two reframe fixes:
- ``default_sweep_x`` must read physical parameters from system metadata
  (it previously used ``system.rabi_eff`` / ``system.time_scale`` attributes,
  which RydbergSystem does not define -> AttributeError);
- the documented ``analysis.observables`` helpers must be importable from the
  ``ryd_gate.analysis`` package (they were public + documented but missing from
  ``__all__``).
"""

import numpy as np

from ryd_gate import Register, RydbergSystem


def test_default_sweep_x_uses_metadata():
    from ryd_gate.analysis.addressing import default_sweep_x

    system = RydbergSystem.from_lattice(
        Register.chain(2, spacing_um=3.0), "analog_3", detuning_sign=1,
    )
    x = default_sweep_x(system)
    assert len(x) == 3
    assert all(np.isfinite(v) for v in x)
    # symmetric detuning sweep endpoints; positive normalized gate time
    assert np.isclose(x[0], -x[1])
    assert x[2] > 0.0


def test_analysis_observables_exports():
    from ryd_gate.analysis import (
        measure_observables,
        measure_trajectory,
        norm_squared,
        state_overlap,
    )

    assert all(
        callable(f)
        for f in (measure_observables, measure_trajectory, norm_squared, state_overlap)
    )
