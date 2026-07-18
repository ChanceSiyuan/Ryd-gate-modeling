"""Named parameter mapping packing/unpacking (ADR-0016).

qoc's independence certificate: no test under ``tests/qoc`` imports ``ryd_gate``.
"""

import numpy as np
import pytest

from qoc._parameters import ParameterPacker

X0 = {"a": 1.5, "w": np.array([0.1, -0.2, 0.3])}


def test_pack_unpack_roundtrip_preserves_names_shapes_and_kinds():
    packer = ParameterPacker(X0)
    flat = packer.pack(X0)
    assert flat.shape == (4,)
    out = packer.unpack(flat)
    assert set(out) == {"a", "w"}
    assert isinstance(out["a"], float) and out["a"] == 1.5
    assert isinstance(out["w"], np.ndarray) and out["w"].shape == (3,)
    np.testing.assert_allclose(out["w"], X0["w"])


def test_pack_rejects_name_and_shape_mismatches():
    packer = ParameterPacker(X0)
    with pytest.raises(ValueError, match="missing"):
        packer.pack({"a": 1.0})
    with pytest.raises(ValueError, match="shape"):
        packer.pack({"a": 1.0, "w": np.zeros(4)})


def test_invalid_values_fail_fast():
    with pytest.raises(TypeError, match="complex"):
        ParameterPacker({"a": 1.0 + 2.0j})
    with pytest.raises(TypeError, match="boolean"):
        ParameterPacker({"a": True})
    with pytest.raises(ValueError, match="finite"):
        ParameterPacker({"a": np.nan})
    with pytest.raises(ValueError, match="at least one"):
        ParameterPacker({})


def test_bounds_broadcast_and_validate():
    packer = ParameterPacker(X0)
    packed = packer.pack_bounds({"w": (0.0, 2.0)})
    assert packed is not None and len(packed) == 4
    assert packed[0] == (-np.inf, np.inf)  # "a" omitted -> unbounded
    assert packed[1:] == [(0.0, 2.0)] * 3  # scalar limits broadcast across the array
    with pytest.raises(ValueError, match="unknown"):
        packer.pack_bounds({"nope": (0.0, 1.0)})
    with pytest.raises(ValueError, match="lower <= upper"):
        packer.pack_bounds({"a": (2.0, 1.0)})


def test_scales_default_to_one_and_must_be_positive():
    packer = ParameterPacker(X0)
    np.testing.assert_allclose(packer.pack_scales(None), np.ones(4))
    scale = packer.pack_scales({"w": 10.0})
    np.testing.assert_allclose(scale, [1.0, 10.0, 10.0, 10.0])
    with pytest.raises(ValueError, match="positive"):
        packer.pack_scales({"a": 0.0})
    with pytest.raises(ValueError, match="unknown"):
        packer.pack_scales({"nope": 1.0})
