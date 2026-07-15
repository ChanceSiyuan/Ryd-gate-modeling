"""Deterministic boundary-MPS contraction: double-layer norm + product-bra coefficient (PEPS30/32).

No module-scope YASTN import; the YASTN/MPS/fpeps handles arrive at call time from
the engine. The identity transfer-boundary constructor is ported from YASTN commit
``30b1d8bb4dc691a25bf6394b061c564128ede8e0`` (Apache-2.0) using only public tensor/
MPS operations, so the adapter does not depend on YASTN private symbols.

Orientation is fixed by grid shape (PEPS33): vertical transfers when ``Nx <= Ny``,
horizontal otherwise, always low-to-high. Numerator (single-layer, projected onto
physical labels) and norm (double-layer) use the identical orientation, order,
per-site positive scales, and ``normalize=False`` controls, so the unknown scale
product cancels while the original complex global phase survives.
"""

from __future__ import annotations

from dataclasses import dataclass

from ryd_gate.backends.peps._numerics import (
    PEPSError,
    finite_complex,
    finite_nonneg,
    positive_norm,
)


@dataclass(frozen=True, slots=True)
class _ContractionControls:
    env_bond_dimension: int
    env_tolerance: float
    env_max_iterations: int


# ── ported identity transfer boundary (YASTN 30b1d8b, Apache-2.0) ────────────


def _identity_boundary(yastn, config, leg):
    if leg.is_fused():
        return yastn.eye(config, legs=leg.unfuse_leg(), isdiag=False).fuse_legs(axes=[(0, 1)])
    return yastn.ones(config, legs=[leg])


def _identity_tm_boundary(yastn, mps, tmpo):
    """Identity boundary MPS matching a transfer MPO's outer bra leg (public-op port)."""
    phi = mps.Mps(N=tmpo.N)
    config = tmpo.config
    for n in phi.sweep(to="last"):
        legf = tmpo[n].get_legs(axes=3).conj()
        tmp = _identity_boundary(yastn, config, legf)
        phi[n] = tmp.add_leg(0, s=-1).add_leg(2, s=1)
    return phi


def _orientation(shape: tuple[int, int]) -> tuple[str, int]:
    nx, ny = shape
    return ("v", ny) if nx <= ny else ("h", nx)


# ── core contraction ─────────────────────────────────────────────────────────


def _contract_network(h, peps, controls: _ContractionControls) -> tuple[complex, float]:
    """Boundary-MPS contraction of a finite PEPS network → (scalar, heuristic error)."""
    yastn, mps = h.yastn, h.mps
    dirn, nlayers = _orientation((peps.Nx, peps.Ny))
    opts_svd = {"D_total": controls.env_bond_dimension, "tol": controls.env_tolerance}
    try:
        transfers = [peps.transfer_mpo(n=n, dirn=dirn) for n in range(nlayers)]
        boundary = _identity_tm_boundary(yastn, mps, transfers[0])
        high = _identity_tm_boundary(yastn, mps, transfers[-1].T)
        error = 0.0
        for transfer in transfers[:-1]:
            nxt, discarded = mps.zipper(
                transfer, boundary, opts_svd, normalize=False, return_discarded=True
            )
            out = mps.compression_(
                nxt,
                (transfer, boundary),
                method="1site",
                overlap_tol=controls.env_tolerance,
                max_sweeps=controls.env_max_iterations,
                opts_svd=opts_svd,
                normalize=False,
            )
            d = float(discarded)
            if d == d and d not in (float("inf"), float("-inf")) and d >= 0.0:
                error = max(error, d)
            error = max(error, abs(complex(out.doverlap)))
            boundary = nxt
        value = complex(mps.vdot(high.conj(), transfers[-1], boundary))
    except yastn.YastnError as exc:  # pinned YASTN tensor error; do not broaden
        raise PEPSError(f"YASTN boundary contraction failed: {exc}") from exc
    return value, finite_nonneg(error, "boundary contraction error")


def _scaled_peps(h, psi, scales):
    """Temporary PEPS with each rank-5 tensor divided by its positive site scale."""
    tensors = {coord: psi[coord] / scales[coord] for coord in scales}
    return h.fpeps.Peps(psi.geometry, tensors=tensors)


def double_layer_norm(h, psi, scales, controls: _ContractionControls) -> tuple[float, float]:
    """One double-layer norm contraction of the scaled PEPS (PEPS32)."""
    scaled = _scaled_peps(h, psi, scales)
    value, error = _contract_network(h, scaled, controls)
    return positive_norm(value, "PEPS double-layer norm"), error


def product_bra_coefficient(
    h, psi, scales, ops, site_to_coord, labels, controls: _ContractionControls
) -> tuple[complex, float]:
    """One single-layer product-bra coefficient contraction of the scaled PEPS (PEPS32)."""
    tensors = {}
    for i, coord in enumerate(site_to_coord):
        bra = ops.bra_leg(ops.levels.index(labels[i]))
        tensors[coord] = h.yastn.tensordot(psi[coord] / scales[coord], bra, axes=(4, 0))
    projected = h.fpeps.Peps(psi.geometry, tensors=tensors)
    value, error = _contract_network(h, projected, controls)
    return finite_complex(value, "PEPS product-bra coefficient"), error
