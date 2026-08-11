"""Regression tests for the scripts-local shared ``01r`` pulse basis."""

import numpy as np
from scripts.one_r_control import BASIS, ControlBasis, power_envelope


def test_power_envelope_is_symmetric_and_flat():
    points = np.array([0.0, 0.05, 0.15, 0.5, 0.85, 0.95, 1.0])
    values = power_envelope(points)
    np.testing.assert_allclose(values, values[::-1], rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(values[[0, 2, 3, 4, 6]], [0, 1, 1, 1, 0])
    assert power_envelope(0.5) == 1.0


def test_control_basis_shapes_seeds_and_bounds():
    basis = ControlBasis(n_coeffs=8)
    theta = basis.seed(13.0, 15.0)
    assert theta.shape == (17,)
    assert basis.matrix(np.linspace(0.0, 1.0, 11)).shape == (11, 8)
    assert len(basis.bounds()) == basis.n_parameters == 17
    assert BASIS.n_coeffs == 8 and BASIS.degree == 3


def test_control_jacobian_matches_central_differences():
    basis = ControlBasis(n_coeffs=8)
    theta = basis.seed(13.0, 15.0)
    theta[:8] += np.linspace(-0.1, 0.05, 8)
    theta[8:16] = np.linspace(-0.7, 0.8, 8)
    sample = np.linspace(0.03, 0.97, 9)
    _, _, d_amplitude, d_chirp = basis.controls(
        theta, sample, jacobian=True)

    epsilon = 1e-5
    for index in range(basis.n_parameters):
        step = np.zeros_like(theta)
        step[index] = epsilon
        amp_plus, chirp_plus = basis.controls(theta + step, sample)
        amp_minus, chirp_minus = basis.controls(theta - step, sample)
        np.testing.assert_allclose(
            (amp_plus - amp_minus) / (2 * epsilon), d_amplitude[:, index],
            rtol=1e-6, atol=1e-1)
        np.testing.assert_allclose(
            (chirp_plus - chirp_minus) / (2 * epsilon), d_chirp[:, index],
            rtol=1e-6, atol=1e-1)
