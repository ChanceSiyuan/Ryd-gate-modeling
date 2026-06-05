"""MPS state initializers for tensor network simulations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ryd_gate.backends.tn_common.lattice_spec import TNLatticeSpec


def _require_tenpy():
    try:
        import tenpy
        return tenpy
    except ImportError as exc:
        raise ImportError(
            "TeNPy is required for tensor network simulations. "
            "Install via: pip install physics-tenpy  "
            "or: pip install ryd-gate[tn]"
        ) from exc


def product_state_mps(
    spec: TNLatticeSpec,
    config: np.ndarray | Sequence[str] | str,
) -> object:
    """Create a product-state MPS from a per-site config or named pattern.

    Parameters
    ----------
    spec : TNLatticeSpec
    config : ndarray of 0/1, sequence of labels, or str
        Supported strings: ``"all_ground"``, ``"af1"``, ``"af2"``.
        For ndarray: 0 = non-Rydberg reference state, 1 = |r>.  In
        ``01r`` this non-Rydberg reference is ``|1>``.

    Returns
    -------
    psi : tenpy.networks.mps.MPS
        Product-state MPS in snake order.
    """
    _require_tenpy()
    from tenpy.networks.mps import MPS

    from .sites import build_tenpy_site

    labels_2d = _state_labels_2d(spec, config)
    labels_1d = [labels_2d[i] for i in spec.snake_to_2d]

    site = build_tenpy_site(spec.level_spec)
    sites = [site] * spec.N
    return MPS.from_product_state(sites, labels_1d, bc="finite")


def _state_labels_2d(spec: TNLatticeSpec, config: np.ndarray | Sequence[str] | str) -> list[str]:
    if isinstance(config, str):
        return _named_state_labels_2d(spec, config)

    arr = np.asarray(config)
    if arr.shape != (spec.N,):
        raise ValueError(f"config must have shape ({spec.N},), got {arr.shape}.")

    if arr.dtype.kind in {"U", "S", "O"}:
        labels = [str(x) for x in arr]
        _validate_level_labels(spec, labels)
        return [_tenpy_label(spec, label) for label in labels]

    occ = arr.astype(int)
    non_rydberg = "1"
    labels = ["r" if c == 1 else non_rydberg for c in occ]
    return [_tenpy_label(spec, label) for label in labels]


def _named_state_labels_2d(spec: TNLatticeSpec, name: str) -> list[str]:
    if name in {"all_ground", "all_1"}:
        labels = ["1"] * spec.N
    elif name in {"all_0", "all_zero"}:
        if "0" not in spec.level_spec.levels:
            raise ValueError("'all_0' requires a TN lattice spec with a |0> level.")
        labels = ["0"] * spec.N
    elif name == "all_r":
        labels = ["r"] * spec.N
    elif name == "af1":
        labels = ["r" if s > 0 else "1" for s in spec.sublattice]
    elif name == "af2":
        labels = ["r" if s < 0 else "1" for s in spec.sublattice]
    else:
        raise ValueError(f"Unknown config string: {name!r}")
    return [_tenpy_label(spec, label) for label in labels]


def _validate_level_labels(spec: TNLatticeSpec, labels: Sequence[str]) -> None:
    allowed = set(spec.level_spec.levels)
    unknown = sorted(set(labels) - allowed)
    if unknown:
        raise ValueError(f"Unknown level label(s) for {spec.level_structure}: {unknown}.")


def _tenpy_label(spec: TNLatticeSpec, label: str) -> str:
    if spec.level_structure == "1r":
        return "up" if label == "r" else "down"
    return label


def product_superposition_mps(
    spec: TNLatticeSpec,
    ground_amp: complex = 1 / np.sqrt(2),
    rydberg_amp: complex = -1j / np.sqrt(2),
    zero_amp: complex = 0.0,
) -> object:
    """Create a uniform product superposition MPS.

    In ``1r`` the local state is ``ground_amp * |1> + rydberg_amp * |r>``.
    In ``01r`` it is
    ``zero_amp * |0> + ground_amp * |1> + rydberg_amp * |r>``.
    """
    _require_tenpy()
    from tenpy.networks.mps import MPS

    from .sites import build_tenpy_site

    site = build_tenpy_site(spec.level_spec)
    sites = [site] * spec.N
    if spec.level_structure == "1r":
        if zero_amp != 0:
            raise ValueError("zero_amp requires a 01r TN lattice spec.")
        norm = np.sqrt(abs(ground_amp) ** 2 + abs(rydberg_amp) ** 2)
        if norm == 0:
            raise ValueError("At least one local amplitude must be nonzero.")
        # SpinHalfSite basis order is |up>, |down>; our |r> is up and |1> is down.
        local_state = np.array([rydberg_amp, ground_amp], dtype=complex) / norm
    else:
        local_amps = {
            "0": zero_amp,
            "1": ground_amp,
            "r": rydberg_amp,
        }
        norm = np.sqrt(sum(abs(local_amps.get(level, 0.0)) ** 2 for level in spec.level_spec.levels))
        if norm == 0:
            raise ValueError("At least one local amplitude must be nonzero.")
        local_state = np.array(
            [local_amps.get(level, 0.0) for level in spec.level_spec.levels],
            dtype=complex,
        ) / norm
    return MPS.from_product_state(sites, [local_state] * spec.N, bc="finite", dtype=complex)


def mps_fidelity(psi_target: object, psi: object) -> float:
    """Return ``|<psi_target|psi>|^2`` for TeNPy MPS states."""
    overlap = psi_target.overlap(psi)
    return float(abs(overlap) ** 2)


def domain_state_mps(
    spec: TNLatticeSpec,
    domain_center: tuple[float, float],
    domain_radius: float,
) -> object:
    """Create an AF1-bulk + AF2-domain product state MPS.

    Parameters
    ----------
    spec : TNLatticeSpec
    domain_center : (cx, cy)
    domain_radius : float
        Chebyshev (square) domain radius.

    Returns
    -------
    psi : tenpy.networks.mps.MPS
    """
    from ryd_gate.core.states import domain_config

    config = domain_config(spec.coords, spec.sublattice,
                           domain_center, domain_radius)
    return product_state_mps(spec, config)
