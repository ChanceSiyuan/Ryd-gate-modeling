"""phase_from_chirp (ryd_gate.protocols) — P33 pulse-construction helper.

Protocol-facing usage (CZProtocol phase built from a chirp integral) is covered
in the protocol test suite; here we pin only the helper contract.
"""

import numpy as np
import pytest

from ryd_gate.protocols import phase_from_chirp


def test_constant_chirp_is_linear():
    """phi(t) = ∫_0^t c dt' = c*t for a constant chirp."""
    c = 1.234
    phi = phase_from_chirp(lambda t: c, 2.0, n_samples=201)
    assert np.isclose(phi(0.0), 0.0)
    assert np.isclose(phi(1.0), c * 1.0)
    assert np.isclose(phi(2.0), c * 2.0)


def test_cosine_sweep_matches_analytic_integral():
    """A round-trip cosine sweep integrates to the known sine; net phase 0 at t_gate."""
    T, A = 1.0, 5.0
    phi = phase_from_chirp(lambda t: -A * np.cos(2 * np.pi * t / T), T, n_samples=4001)
    for t in (0.1, 0.37, 0.5, 0.83):
        exact = -(A * T / (2 * np.pi)) * np.sin(2 * np.pi * t / T)
        assert np.isclose(phi(t), exact, atol=1e-4)
    assert np.isclose(phi(T), 0.0, atol=1e-6)


def test_out_of_domain_raises_not_clips():
    """The returned callable is defined strictly on [0, t_gate]; outside raises."""
    phi = phase_from_chirp(lambda t: 1.0, 2.0, n_samples=101)
    with pytest.raises(ValueError):
        phi(3.0)
    with pytest.raises(ValueError):
        phi(-1.0)


def test_validation():
    with pytest.raises(ValueError):
        phase_from_chirp(lambda t: 0.0, 0.0)
    with pytest.raises(ValueError):
        phase_from_chirp(lambda t: 0.0, 1.0, n_samples=1)
    with pytest.raises(ValueError):
        phase_from_chirp(lambda t: 0.0, 1.0, n_samples=True)  # bool is not an int here
    with pytest.raises(TypeError):
        phase_from_chirp(3.0, 1.0)  # not callable


def test_n_samples_is_keyword_only():
    with pytest.raises(TypeError):
        phase_from_chirp(lambda t: 0.0, 1.0, 101)  # positional n_samples rejected
