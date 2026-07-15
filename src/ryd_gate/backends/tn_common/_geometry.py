"""Private center-line pair geometry for TN correlation measurements.

Relocated from the deleted ``ryd_gate.analysis.observables`` module; consumed
by the MPS and PEPS measurement kernels (center-line connected-``C_zz``
profiles).  Not public API.
"""

from __future__ import annotations

import numpy as np


def center_line_sites(Lx: int, Ly: int, *, axis: str = "horizontal") -> np.ndarray:
    """Return site indices on a center-most line of a row-major square grid."""
    if axis not in {"horizontal", "vertical"}:
        raise ValueError("axis must be 'horizontal' or 'vertical'.")
    if axis == "horizontal":
        ix = Lx // 2
        return np.array([ix * Ly + iy for iy in range(Ly)], dtype=int)
    iy = Ly // 2
    return np.array([ix * Ly + iy for ix in range(Lx)], dtype=int)


def center_reference_site(Lx: int, Ly: int) -> int:
    """Return one center-most site for an open square/rectangular grid."""
    return (Lx // 2) * Ly + (Ly // 2)


def line_pairs_from_reference(
    Lx: int,
    Ly: int,
    *,
    reference: int | None = None,
    axis: str = "horizontal",
) -> list[tuple[int, int]]:
    """Pair a reference site with all other sites on the chosen center line."""
    ref = center_reference_site(Lx, Ly) if reference is None else int(reference)
    return [(ref, int(site)) for site in center_line_sites(Lx, Ly, axis=axis) if int(site) != ref]
