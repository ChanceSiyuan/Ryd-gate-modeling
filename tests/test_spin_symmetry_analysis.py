import numpy as np

from ryd_gate.analysis import (
    center_line_sites,
    connected_zz_from_connected_nn,
    d4_permutations,
    d4_symmetry_error,
    epsilon_z,
    epsilon_zz,
    first_unconverged_time,
    sigma_z_from_rydberg_occ,
)


def test_spin_observable_conversions():
    occ = np.array([0.0, 0.5, 1.0])

    np.testing.assert_allclose(sigma_z_from_rydberg_occ(occ), [-1.0, 0.0, 1.0])
    np.testing.assert_allclose(connected_zz_from_connected_nn([0.25, -0.5]), [1.0, -2.0])


def test_center_line_and_benchmark_errors():
    np.testing.assert_array_equal(center_line_sites(3, 3), [3, 4, 5])
    assert np.isclose(epsilon_z([1.0, 0.0], [0.0, 0.0]), 1.0)
    assert np.isclose(epsilon_zz([2.0, 2.0], [1.0, 3.0]), 0.5)


def test_d4_permutations_are_valid_permutations():
    perms = d4_permutations(3, 3)

    assert len(perms) == 8
    for perm in perms:
        assert set(perm) == set(range(9))


def test_d4_symmetry_error_zero_for_symmetric_observable():
    obs = np.ones(9)

    err = d4_symmetry_error(obs, 3, 3)

    assert err.max_abs == 0.0
    assert err.max_rel == 0.0


def test_d4_symmetry_error_detects_asymmetry_and_threshold_time():
    obs = np.ones((3, 9))
    obs[1, 0] = 2.0
    obs[2, 0] = 3.0

    err = d4_symmetry_error(obs, 3, 3)
    times = np.array([0.0, 0.5, 1.0])

    assert err.max_abs > 0.0
    assert first_unconverged_time(times, np.max(err.per_site_rel, axis=1), threshold=0.2) == 0.5
