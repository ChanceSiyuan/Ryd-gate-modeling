"""TeNPy model builder for the 2-level Rydberg lattice Hamiltonian.

H = (Omega/2) sum_i X_i - sum_i (Delta + delta_i) n_i + sum_{i<j} V_ij n_i n_j
"""

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


def build_tenpy_model(
    spec: TNLatticeSpec,
    Delta: float,
    Omega: float | None = None,
    pin_deltas: np.ndarray | None = None,
) -> object:
    """Build a TeNPy CouplingMPOModel for the 2-level Rydberg lattice.

    The Hamiltonian is expressed in the spin-1/2 basis where
    |g> = |down> and |r> = |up>, so n_i = (1 + Sz_i)/2 and
    X_i = Sx_i (mapped from sigma^x).

    Sites are ordered in snake order for optimal MPS geometry.

    Parameters
    ----------
    spec : TNLatticeSpec
        Lattice specification with geometry and snake ordering.
    Delta : float
        Global detuning.
    Omega : float or None
        Rabi frequency. If None, uses ``spec.Omega``.
    pin_deltas : ndarray, shape (N,) or None
        Per-site local detunings (in 2D site order).

    Returns
    -------
    model : tenpy.models.model.CouplingMPOModel
        TeNPy model ready for DMRG or TDVP.
    """
    tenpy = _require_tenpy()
    from tenpy.models.lattice import Chain
    from tenpy.models.model import CouplingMPOModel
    from tenpy.networks.site import SpinHalfSite

    if Omega is None:
        Omega = spec.Omega
    if pin_deltas is None:
        pin_deltas = np.zeros(spec.N)

    site = SpinHalfSite(conserve=None)

    bc_MPS = "finite" if spec.bc == "open" else "infinite"
    bc = "open" if spec.bc == "open" else "periodic"

    class RydbergLatticeModel(CouplingMPOModel):
        def init_lattice(self, model_params):
            return Chain(spec.N, site, bc=bc, bc_MPS=bc_MPS)

        def init_terms(self, model_params):
            # Per-site onsite terms using add_onsite_term(strength, i, op)
            # where i is the MPS site index.
            # Convention: n_i = 0.5 + Sz_i  (TeNPy Sz = +-0.5)
            for i_2d in range(spec.N):
                i_1d = int(spec.inv_snake[i_2d])
                total_delta = Delta + pin_deltas[i_2d]
                # -(Delta+delta_i) * n_i = -(Delta+delta_i) * (0.5 + Sz)
                # = -(Delta+delta_i) * Sz - (Delta+delta_i)/2
                self.add_onsite_term(-total_delta, i_1d, "Sz")
                self.add_onsite_term(-total_delta / 2.0, i_1d, "Id")
                # (Omega/2) * sigma_x = (Omega/2) * 2*Sx = Omega*Sx
                self.add_onsite_term(Omega, i_1d, "Sx")

            # Interaction: V_ij * n_i * n_j
            # = V_ij * (0.5+Szi)(0.5+Szj)
            # = V_ij * Szi*Szj + V_ij/2*(Szi+Szj) + V_ij/4
            for i_2d, j_2d, v_rel in spec.vdw_pairs:
                V_ij = spec.V_nn * v_rel
                i_1d = int(spec.inv_snake[i_2d])
                j_1d = int(spec.inv_snake[j_2d])
                if i_1d > j_1d:
                    i_1d, j_1d = j_1d, i_1d
                self.add_coupling_term(
                    V_ij, i_1d, j_1d, "Sz", "Sz")
                self.add_onsite_term(V_ij / 2.0, i_1d, "Sz")
                self.add_onsite_term(V_ij / 2.0, j_1d, "Sz")
                # Constant term V_ij/4 split across two sites
                self.add_onsite_term(V_ij / 8.0, i_1d, "Id")
                self.add_onsite_term(V_ij / 8.0, j_1d, "Id")

    return RydbergLatticeModel({})
