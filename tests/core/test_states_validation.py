"""Error paths of ``core.states.normalize_initial_state`` (S05/S11/E27).

Pins the input guards for the public ``initial_state`` argument: the unknown
string, the empty sequence, the per-site count mismatch, non-string labels, and
the non-sequence type rejection. The accepted forms (``None`` -> ``["1"]*N``,
``"plus"``, flat and nested label lists) are covered end-to-end elsewhere.
"""

import pytest

from ryd_gate.core.states import normalize_initial_state


def test_unknown_string_rejected():
    with pytest.raises(ValueError, match="unknown initial_state string"):
        normalize_initial_state("ground", 2)


def test_empty_sequence_rejected():
    with pytest.raises(ValueError, match="empty sequence"):
        normalize_initial_state([], 2)


def test_wrong_label_count_rejected():
    with pytest.raises(ValueError, match="per-site labels"):
        normalize_initial_state(["1"], 2)


def test_non_string_labels_rejected():
    with pytest.raises(TypeError, match="level-label strings"):
        normalize_initial_state([1, 2], 2)


def test_non_sequence_rejected():
    with pytest.raises(TypeError, match="initial_state must be None"):
        normalize_initial_state(42, 2)
