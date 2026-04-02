"""MPS state initializers for tensor network simulations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .lattice_spec import TNLatticeSpec


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
    config: np.ndarray | str,
) -> object:
    """Create a product-state MPS from a per-site config or named pattern.

    Parameters
    ----------
    spec : TNLatticeSpec
    config : ndarray of 0/1 (in 2D site order) or str
        Supported strings: ``"all_ground"``, ``"af1"``, ``"af2"``.
        For ndarray: 0 = |g> (down), 1 = |r> (up).

    Returns
    -------
    psi : tenpy.networks.mps.MPS
        Product-state MPS in snake order.
    """
    tenpy = _require_tenpy()
    from tenpy.networks.mps import MPS
    from tenpy.networks.site import SpinHalfSite

    if isinstance(config, str):
        if config == "all_ground":
            config_2d = np.zeros(spec.N, dtype=int)
        elif config == "af1":
            config_2d = (spec.sublattice > 0).astype(int)
        elif config == "af2":
            config_2d = (spec.sublattice < 0).astype(int)
        else:
            raise ValueError(f"Unknown config string: {config!r}")
    else:
        config_2d = np.asarray(config, dtype=int)

    # Reorder to snake order for MPS
    config_1d = config_2d[spec.snake_to_2d]

    site = SpinHalfSite(conserve=None)
    sites = [site] * spec.N
    # TeNPy SpinHalfSite: index 0 = |up>, index 1 = |down>
    # Our convention: 0 = |g> = |down>, 1 = |r> = |up>
    state_labels = ["up" if c == 1 else "down" for c in config_1d]
    return MPS.from_product_state(sites, state_labels, bc="finite")


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
    from ryd_gate.lattice.states import domain_config

    config = domain_config(spec.coords, spec.sublattice,
                           domain_center, domain_radius)
    return product_state_mps(spec, config)
