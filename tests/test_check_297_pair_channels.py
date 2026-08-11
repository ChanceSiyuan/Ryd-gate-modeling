"""Focused tests for scripts/check_297_pair_channels.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix, diags

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "check_297_pair_channels", ROOT / "scripts" / "check_297_pair_channels.py"
)
assert SPEC is not None and SPEC.loader is not None
pair = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pair
SPEC.loader.exec_module(pair)


def test_assemble_pair_hamiltonian_applies_arc_distance_powers():
    """Catch treating ARC's metre-scaled matR entries as already evaluated."""
    spacing_um = 2.0
    distance_m = spacing_um * 1e-6
    calc = SimpleNamespace(
        matDiagonal=diags([0.1, 0.2], format="csr"),
        matR=[
            csr_matrix(
                np.array([[0.0, 0.05], [0.05, 0.0]]) * distance_m**3
            )
        ],
    )

    actual = pair.assemble_pair_hamiltonian(calc, spacing_um).toarray()

    np.testing.assert_allclose(actual, [[0.1, 0.05], [0.05, 0.2]])


def test_find_basis_state_index_requires_one_exact_pair_state():
    """Catch silently using index zero when the requested ARC state is absent."""
    states = [[53, 1, 1.5, -1.5, 53, 1, 1.5, -1.5]]
    target = (53, 1, 1.5, -1.5, 53, 1, 1.5, -1.5)

    assert pair.find_basis_state_index(states, target) == 0
    with pytest.raises(ValueError, match="exactly one"):
        pair.find_basis_state_index(states, (*target[:-1], -0.5))


def test_extract_local_eigenpairs_expands_until_weak_window_is_bracketed():
    """Catch stopping after a fixed k that leaves part of the weak window out."""
    values = np.array(
        [-0.20, -0.12, -0.06, -0.01, 0.02, 0.07, 0.11, 0.15, 0.25, 0.40]
    )

    eigenvalues, eigenvectors, meta = pair.extract_local_eigenpairs(
        diags(values, format="csr"),
        reference_ghz=0.005,
        bare_index=3,
        weak_threshold_mhz=80.0,
        initial_k=4,
        max_k=8,
        capture_target=0.99,
    )

    shifts_mhz = (eigenvalues - 0.005) * 1e3
    assert shifts_mhz.min() < -80.0
    assert shifts_mhz.max() > 80.0
    assert meta["window_bracketed"] is True
    assert meta["eigenpairs"] == 8
    assert eigenvectors.shape == (10, 8)


def test_extract_local_eigenpairs_rejects_unbracketed_window():
    """Catch reporting a partial weak weight as though the energy window were complete."""
    values = np.linspace(-0.2, 0.2, 10)

    with pytest.raises(RuntimeError, match="did not bracket"):
        pair.extract_local_eigenpairs(
            diags(values, format="csr"),
            reference_ghz=0.0,
            bare_index=4,
            weak_threshold_mhz=150.0,
            initial_k=2,
            max_k=4,
            capture_target=0.99,
        )


def test_summarize_eigenpairs_uses_channel_reference_and_overlap_weights():
    """Catch using absolute ARC energies or amplitudes instead of squared overlaps."""
    eigenvalues = np.array([-0.05, 0.02, 0.12])
    eigenvectors = np.array(
        [
            [0.5, np.sqrt(0.75), 0.0],
            [np.sqrt(0.75), -0.5, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    basis_states = [
        [53, 1, 1.5, -1.5, 53, 1, 1.5, -1.5],
        [53, 1, 1.5, -0.5, 53, 1, 1.5, -0.5],
        [53, 0, 0.5, -0.5, 54, 0, 0.5, 0.5],
    ]

    summary = pair.summarize_eigenpairs(
        eigenvalues,
        eigenvectors,
        reference_ghz=0.0,
        bare_index=0,
        basis_states=basis_states,
        target_manifold_indices=[0, 1],
        weak_threshold_mhz=80.0,
        report_overlap_cutoff=0.01,
    )

    assert summary["weak_shift_weight"] == pytest.approx(1.0)
    assert summary["captured_overlap"] == pytest.approx(1.0)
    assert summary["states"][0]["overlap"] == pytest.approx(0.75)
    assert summary["states"][0]["shift_mhz"] == pytest.approx(20.0)


def test_build_output_separates_authoritative_and_comparison_models(monkeypatch):
    """Catch putting hand-Zeeman C6 spectra back in the authoritative slot."""
    monkeypatch.setattr(
        pair,
        "calculate_full_pair_field",
        lambda atom, b: {"b_gauss": b},
        raising=False,
    )
    monkeypatch.setattr(
        pair,
        "calculate_effective_c6_comparison",
        lambda atom: {"model": "effective"},
        raising=False,
    )
    monkeypatch.setattr(
        pair,
        "radial_defect_ranking",
        lambda atom: [],
        raising=False,
    )

    output = pair.build_output(object())

    assert set(output) == {
        "schema_version",
        "params",
        "full_pair",
        "effective_c6_comparison",
        "radial_defect_ranking",
    }
    assert output["full_pair"]["authoritative"] is True
    assert set(output["full_pair"]["fields"]) == {"20.0", "160.0"}
    assert output["effective_c6_comparison"]["authoritative"] is False


def test_pair_potential_config_covers_all_53p_zeeman_doorways():
    """Catch omitting a Zeeman doorway from the field/angle production scan."""
    expected = {
        "53P3_2": {"n": 53, "l": 1, "j": 1.5, "mj": -1.5},
        "53P3_2_mj_m1_2": {"n": 53, "l": 1, "j": 1.5, "mj": -0.5},
        "53P3_2_mj_p1_2": {"n": 53, "l": 1, "j": 1.5, "mj": 0.5},
        "53P3_2_mj_p3_2": {"n": 53, "l": 1, "j": 1.5, "mj": 1.5},
        "70S1_2": {"n": 70, "l": 0, "j": 0.5, "mj": -0.5},
    }

    assert pair._pair_potential_params()["manifold_definitions"] == expected
    assert len(expected) * 3 * 7 == 105


def _two_point_curve_case():
    overlaps = [0.1, 1.0]
    return {
        "curves": {
            "distance_um": [2.5, 8.0],
            "spectrum_shift_mhz": [[-5.0], [6.0]],
            "spectrum_rr_overlap": [[overlaps[0]], [overlaps[1]]],
            "branches": [
                {"shift_mhz": [-5.0, 6.0], "rr_overlap": overlaps}
            ],
            "branch_count": 1,
        }
    }


def test_curve_panel_uses_gray_points_with_one_overlap_area_scale():
    """Catch a color-only spectrum or inconsistent sizes across plot layers."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    pair._plot_curve_panel(
        ax, _two_point_curve_case(), 100.0, show_spectrum=True
    )
    fig.canvas.draw()

    np.testing.assert_allclose(
        pair._overlap_marker_area([0.0, 0.1, 0.5, 1.0]),
        [2.0, 7.8, 31.0, 60.0],
    )
    np.testing.assert_allclose(ax.collections[0].get_sizes(), [7.8, 60.0])
    np.testing.assert_allclose(ax.collections[1].get_sizes(), [7.8])
    background_rgb = ax.collections[0].get_facecolors()[:, :3]
    np.testing.assert_allclose(
        background_rgb, np.full_like(background_rgb, 0.55)
    )
    plt.close(fig)


def _limit_state(max_shift_mhz):
    case = _two_point_curve_case()
    case["curves"]["spectrum_shift_mhz"] = [[max_shift_mhz], [0.0]]
    case["curves"]["branches"] = []
    return {"fields": {"20.0": {"angles": {"0.0": case}}}}


def test_pair_potential_y_limits_are_shared_across_53p_zeeman_levels():
    """Catch per-mj autoscaling that makes Zeeman figures incomparable."""
    zeeman_keys = (
        "53P3_2",
        "53P3_2_mj_m1_2",
        "53P3_2_mj_p1_2",
        "53P3_2_mj_p3_2",
    )
    result = {
        "manifolds": {
            zeeman_keys[0]: _limit_state(10.0),
            zeeman_keys[1]: _limit_state(20.0),
            zeeman_keys[2]: _limit_state(30.0),
            zeeman_keys[3]: _limit_state(40.0),
            "70S1_2": _limit_state(100.0),
        }
    }

    limits = pair._pair_potential_y_limits(result)

    np.testing.assert_allclose([limits[key] for key in zeeman_keys], 41.6)
    assert limits["70S1_2"] == pytest.approx(104.0)


def _legend_figure_result():
    angles = {
        str(theta): _two_point_curve_case()
        for theta in pair.POTENTIAL_THETA_DEG
    }
    return {
        "manifolds": {
            "53P3_2": {
                "label": r"$53P_{3/2},\,m_j=-3/2$",
                "fields": {"20.0": {"angles": angles}},
            }
        }
    }


def test_state_figure_replaces_colorbar_with_overlap_size_legend(
    tmp_path, monkeypatch
):
    """Catch reintroducing an orphan colorbar or omitting the size key."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.legend import Legend

    monkeypatch.setattr(plt, "close", lambda figure: None)
    path = pair._render_state_field_potential(
        _legend_figure_result(), tmp_path, "53P3_2", 20.0, 100.0
    )
    fig = plt.gcf()
    legend_text = {
        text.get_text()
        for legend in fig.findobj(match=Legend)
        for text in legend.get_texts()
    }

    assert path.exists()
    assert len(fig.axes) == 8
    assert {r"$p_k=0.1$", r"$p_k=0.5$", r"$p_k=1.0$"} <= legend_text
    plt.close(fig)


def _complete_synthetic_pair_potential_study():
    distances = pair.potential_distance_grid().tolist()
    point_count = len(distances)

    def case(b_gauss, theta_deg):
        return {
            "b_gauss": b_gauss,
            "theta_deg": theta_deg,
            "phi_rad": pair.POTENTIAL_PHI_RAD,
            "curves": {
                "distance_um": distances,
                "eigenpairs": [1] * point_count,
                "captured_rr_overlap": [1.0] * point_count,
                "unresolved_rr_overlap": [0.0] * point_count,
                "max_eigensystem_residual_mhz": [0.0] * point_count,
                "weak_shift_weight": [1.0] * point_count,
                "spectrum_shift_mhz": [[0.0] for _ in distances],
                "spectrum_rr_overlap": [[1.0] for _ in distances],
                "requested_branch_count": pair.POTENTIAL_BRANCH_COUNT,
                "branch_count": 1,
                "branches": [
                    {
                        "anchor_rr_overlap": 1.0,
                        "min_adjacent_match_overlap": 1.0,
                        "shift_mhz": [0.0] * point_count,
                        "rr_overlap": [1.0] * point_count,
                        "adjacent_match_overlap": [1.0] * point_count,
                    }
                ],
            },
        }

    expected_states = {
        "53P3_2": (53, 1, 1.5, -1.5),
        "53P3_2_mj_m1_2": (53, 1, 1.5, -0.5),
        "53P3_2_mj_p1_2": (53, 1, 1.5, 0.5),
        "53P3_2_mj_p3_2": (53, 1, 1.5, 1.5),
        "70S1_2": (70, 0, 0.5, -0.5),
    }
    result = pair._new_pair_potential_study()
    result["status"] = "complete"
    for state_key, (n, l, j, mj) in expected_states.items():
        result["manifolds"][state_key] = {
            "n": n,
            "l": l,
            "j": j,
            "mj": mj,
            "label": state_key,
            "fields": {
                str(b_gauss): {
                    "angles": {
                        str(theta_deg): case(b_gauss, theta_deg)
                        for theta_deg in pair.POTENTIAL_THETA_DEG
                    }
                }
                for b_gauss in pair.POTENTIAL_FIELDS_G
            },
        }
    return result, tuple(expected_states)


def test_renderer_outputs_only_fifteen_scheme_one_figures(
    tmp_path, monkeypatch
):
    """Catch regenerating the removed field-summary figures."""
    result, expected_states = _complete_synthetic_pair_potential_study()

    def state_path(result, output_dir, state_key, b_gauss, y_limit):
        return output_dir / f"pair_potential_{state_key}_B{b_gauss:g}G.png"

    monkeypatch.setattr(pair, "_render_state_field_potential", state_path)
    monkeypatch.setattr(
        pair,
        "_render_field_summary",
        lambda result, output_dir, b_gauss, y_limits: output_dir
        / f"pair_potential_summary_B{b_gauss:g}G.png",
        raising=False,
    )
    paths = pair.render_pair_potential_figures(result, tmp_path)
    expected_names = {
        f"pair_potential_{state_key}_B{b_gauss:g}G.png"
        for state_key in expected_states
        for b_gauss in (20.0, 40.0, 60.0)
    }

    assert {path.name for path in paths} == expected_names


@pytest.mark.slow
def test_arc_bz_changes_intermediate_pair_state_references():
    """Characterize the ARC Bz behavior the authoritative path depends on."""
    from arc import PairStateInteractions, Rubidium87

    calculations = []
    for b_tesla in (0.0, 20e-4):
        calc = PairStateInteractions(
            Rubidium87(),
            53,
            1,
            1.5,
            53,
            1,
            1.5,
            -1.5,
            -1.5,
            interactionsUpTo=1,
        )
        calc.defineBasis(np.pi / 2, 0.0, 1, 2, 1e9, Bz=b_tesla)
        calculations.append(calc)

    zero, field = calculations
    assert zero.basisStates == field.basisStates
    rr = pair.find_basis_state_index(
        zero.basisStates, (53, 1, 1.5, -1.5, 53, 1, 1.5, -1.5)
    )
    intermediate = [
        i
        for i, state in enumerate(zero.basisStates)
        if not (
            state[0:3] == [53, 1, 1.5]
            and state[4:7] == [53, 1, 1.5]
        )
    ]
    relative_change_mhz = (
        field.matDiagonal.diagonal()
        - zero.matDiagonal.diagonal()
        - (field.matDiagonal[rr, rr] - zero.matDiagonal[rr, rr])
    ) * 1e3

    assert intermediate
    assert np.max(np.abs(relative_change_mhz[intermediate])) > 1.0
