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
    Omega: float | np.ndarray | None = None,
    pin_deltas: np.ndarray | None = None,
    omega_R: float | np.ndarray | None = None,
    omega_hf: float | np.ndarray | None = None,
    delta_R: float | np.ndarray | None = None,
    delta_hf: float | np.ndarray | None = None,
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
    Omega : float, ndarray, or None
        Rabi frequency. If None, uses ``spec.Omega``. An ndarray gives a
        per-site profile in 2D site order.
    pin_deltas : ndarray, shape (N,) or None
        Per-site local detunings (in 2D site order).
    omega_R, omega_hf, delta_R, delta_hf : float, ndarray, or None
        Explicit three-level 01r profiles in 2D site order. When
        ``spec.level_structure == "01r"``, these implement
        ``(Omega_R/2) X_R + (Omega_hf/2) X_hf - delta_R n_r
        - delta_hf n_1``.

    Returns
    -------
    model : tenpy.models.model.CouplingMPOModel
        TeNPy model ready for DMRG or TDVP.
    """
    _require_tenpy()
    from tenpy.models.lattice import Chain
    from tenpy.models.model import CouplingMPOModel

    from .sites import build_tenpy_site, transition_x_op_name

    def profile(value, default=0.0) -> np.ndarray:
        if value is None:
            value = default
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            arr = np.full(spec.N, float(arr))
        if arr.shape != (spec.N,):
            raise ValueError(
                f"Profile must be a scalar or length-{spec.N}; got shape {arr.shape}."
            )
        return arr

    if Omega is None:
        Omega = spec.Omega
    omega_profile = profile(Omega)
    if pin_deltas is None:
        pin_deltas = np.zeros(spec.N)
    pin_deltas = profile(pin_deltas)

    if spec.level_structure == "01r":
        omega_R_profile = profile(omega_R, default=Omega)
        omega_hf_profile = profile(omega_hf)
        delta_R_profile = profile(delta_R, default=Delta + pin_deltas)
        delta_hf_profile = profile(delta_hf)
    elif spec.level_structure != "1r":
        raise ValueError("TN lattice level_structure must be '1r' or '01r'.")

    site = build_tenpy_site(spec.level_spec)
    x_r_op = None
    x_hf_op = None
    if spec.level_structure == "01r":
        x_r_op = transition_x_op_name(spec.level_spec, "1", "r")
        x_hf_op = transition_x_op_name(spec.level_spec, "0", "1")

    bc_MPS = "finite" if spec.bc == "open" else "infinite"
    bc = "open" if spec.bc == "open" else "periodic"

    class RydbergLatticeModel(CouplingMPOModel):
        def init_lattice(self, model_params):
            return Chain(spec.N, site, bc=bc, bc_MPS=bc_MPS)

        def init_terms(self, model_params):
            if spec.level_structure == "1r":
                self._init_two_level_terms()
            else:
                self._init_three_level_terms()

        def _init_two_level_terms(self):
            for i_2d in range(spec.N):
                i_1d = int(spec.inv_snake[i_2d])
                total_delta = Delta + pin_deltas[i_2d]
                self.add_onsite_term(-total_delta, i_1d, "Sz")
                self.add_onsite_term(-total_delta / 2.0, i_1d, "Id")
                self.add_onsite_term(float(omega_profile[i_2d]), i_1d, "Sx")

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

        def _init_three_level_terms(self):
            for i_2d in range(spec.N):
                i_1d = int(spec.inv_snake[i_2d])
                self.add_onsite_term(0.5 * float(omega_R_profile[i_2d]), i_1d, x_r_op)
                self.add_onsite_term(0.5 * float(omega_hf_profile[i_2d]), i_1d, x_hf_op)
                self.add_onsite_term(-float(delta_R_profile[i_2d]), i_1d, "n_r")
                self.add_onsite_term(-float(delta_hf_profile[i_2d]), i_1d, "n_1")

            for i_2d, j_2d, v_rel in spec.vdw_pairs:
                V_ij = spec.V_nn * v_rel
                i_1d = int(spec.inv_snake[i_2d])
                j_1d = int(spec.inv_snake[j_2d])
                if i_1d > j_1d:
                    i_1d, j_1d = j_1d, i_1d
                self.add_coupling_term(V_ij, i_1d, j_1d, "n_r", "n_r")

    return RydbergLatticeModel({})
