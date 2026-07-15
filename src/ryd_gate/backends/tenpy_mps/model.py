"""TeNPy local site builders and the Rydberg lattice model builder.

1r: H = (Omega/2) sum_i X_i - sum_i (Delta + delta_i) n_i + sum_{i<j} V_ij n_i n_j.
01r uses the direct matrix-element convention: each (complex) coupling ``c`` on a
transition adds ``c |upper><lower| + conj(c) |lower><upper|`` and the real
diagonal energies add ``energy_1 n_1 + energy_r n_r``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ryd_gate.backends.tn_common.sites import resolve_level_structure
from ryd_gate.core.level_structures import LevelStructure

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


def build_tenpy_model(
    spec: TNLatticeSpec,
    Delta: float,
    Omega: float | np.ndarray | None = None,
    pin_deltas: np.ndarray | None = None,
    coupling_10: complex | np.ndarray | None = None,
    coupling_r0: complex | np.ndarray | None = None,
    coupling_r1: complex | np.ndarray | None = None,
    energy_1: float | np.ndarray | None = None,
    energy_r: float | np.ndarray | None = None,
    drive_420_coeff: complex | None = None,
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
    coupling_10, coupling_r0, coupling_r1 : complex, ndarray, or None
        Explicit three-level 01r coupling matrix elements in 2D site order
        (possibly complex). When ``spec.level_structure == "01r"``, each
        coupling ``c`` on transition ``|lower> <-> |upper>`` implements
        ``c |upper><lower| + conj(c) |lower><upper|``.
    energy_1, energy_r : float, ndarray, or None
        Real 01r diagonal energies in 2D site order, implementing
        ``energy_1 n_1 + energy_r n_r`` (the ``|0>`` energy is gauge zero).

    Returns
    -------
    model : tenpy.models.model.CouplingMPOModel
        TeNPy model ready for DMRG or TDVP.
    """
    _require_tenpy()
    from tenpy.models.lattice import Chain
    from tenpy.models.model import CouplingMPOModel

    def profile(value, default=0.0, dtype=float) -> np.ndarray:
        if value is None:
            value = default
        arr = np.asarray(value, dtype=dtype)
        if arr.ndim == 0:
            arr = np.full(spec.N, arr[()])
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
        coupling_profiles = {
            ("1", "r"): profile(coupling_r1, default=0.5 * omega_profile, dtype=complex),
            ("0", "1"): profile(coupling_10, dtype=complex),
            ("0", "r"): profile(coupling_r0, dtype=complex),
        }
        energy_1_profile = profile(energy_1)
        energy_r_profile = profile(energy_r, default=-(Delta + pin_deltas))
    elif spec.level_structure == "analog_3":
        lb = spec.local_blocks
        if lb is None:
            raise ValueError("analog_3 TeNPy model requires spec.local_blocks.")
        coeff_420 = complex(0.0 if drive_420_coeff is None else drive_420_coeff)
    elif spec.level_structure != "1r":
        raise ValueError("TN lattice level_structure must be '1r', '01r', or 'analog_3'.")

    site = build_tenpy_site(spec.level_spec)
    couplings_01r: list[tuple[np.ndarray, str, str]] = []
    if spec.level_structure == "01r":
        for (lower, upper), c_profile in coupling_profiles.items():
            transition = _find_transition(spec.level_spec, lower, upper)
            if transition is None:
                if np.any(np.abs(c_profile) > 0):
                    raise ValueError(
                        f"Level structure {spec.level_spec.name!r} declares no "
                        f"|{lower}> <-> |{upper}> transition for a nonzero coupling."
                    )
                continue
            couplings_01r.append((c_profile, f"Sp_{transition.name}", f"Sm_{transition.name}"))

    bc_MPS = "finite" if spec.bc == "open" else "infinite"
    bc = "open" if spec.bc == "open" else "periodic"

    class RydbergLatticeModel(CouplingMPOModel):
        def init_lattice(self, model_params):
            return Chain(spec.N, site, bc=bc, bc_MPS=bc_MPS)

        def init_terms(self, model_params):
            if spec.level_structure == "1r":
                self._init_two_level_terms()
            elif spec.level_structure == "01r":
                self._init_three_level_terms()
            else:
                self._init_analog3_terms()

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
                for c_profile, sp_op, sm_op in couplings_01r:
                    c = complex(c_profile[i_2d])
                    self.add_onsite_term(c, i_1d, sp_op)
                    self.add_onsite_term(np.conj(c), i_1d, sm_op)
                self.add_onsite_term(float(energy_1_profile[i_2d]), i_1d, "n_1")
                self.add_onsite_term(float(energy_r_profile[i_2d]), i_1d, "n_r")

            for i_2d, j_2d, v_rel in spec.vdw_pairs:
                V_ij = spec.V_nn * v_rel
                i_1d = int(spec.inv_snake[i_2d])
                j_1d = int(spec.inv_snake[j_2d])
                if i_1d > j_1d:
                    i_1d, j_1d = j_1d, i_1d
                self.add_coupling_term(V_ij, i_1d, j_1d, "n_r", "n_r")

        def _init_analog3_terms(self):
            # Physical g/e/r ladder: large static intermediate detuning + static
            # e-r coupling + (complex) time-dependent g-e drive, all from
            # spec.local_blocks; the protocol only modulates drive_420 (coeff_420).
            for i_2d in range(spec.N):
                i_1d = int(spec.inv_snake[i_2d])
                self.add_onsite_term(float(lb.Delta), i_1d, "n_e")
                self.add_onsite_term(0.5 * float(lb.rabi_1013), i_1d, "X_1013")
                self.add_onsite_term(0.5 * lb.rabi_420 * coeff_420, i_1d, "Sp_420")
                self.add_onsite_term(0.5 * lb.rabi_420 * np.conj(coeff_420), i_1d, "Sm_420")

            for i_2d, j_2d, v_rel in spec.vdw_pairs:
                V_ij = spec.V_nn * v_rel
                i_1d = int(spec.inv_snake[i_2d])
                j_1d = int(spec.inv_snake[j_2d])
                if i_1d > j_1d:
                    i_1d, j_1d = j_1d, i_1d
                self.add_coupling_term(V_ij, i_1d, j_1d, "n_r", "n_r")

    return RydbergLatticeModel({})


def build_tenpy_site(level_structure: str | LevelStructure) -> object:
    """Build the TeNPy local site for a shared :class:`LevelStructure`."""
    _require_tenpy()
    spec = resolve_level_structure(level_structure)
    if spec.name == "1r":
        from tenpy.networks.site import SpinHalfSite

        return SpinHalfSite(conserve=None)
    return _build_explicit_site(spec)


def _find_transition(spec: LevelStructure, lower: str, upper: str):
    """Return the declared ``|lower> <-> |upper>`` transition spec, or ``None``."""
    for transition in spec.transitions:
        if transition.lower == lower and transition.upper == upper:
            return transition
    return None


def _build_explicit_site(spec: LevelStructure) -> object:
    from tenpy.linalg import np_conserved as npc
    from tenpy.networks.site import Site

    dim = spec.local_dim
    leg = npc.LegCharge.from_trivial(dim)
    ops = {}

    for idx, level in enumerate(spec.levels):
        projector = np.zeros((dim, dim), dtype=float)
        projector[idx, idx] = 1.0
        ops[f"n_{level}"] = projector

    for transition in spec.transitions:
        lower = spec.index(transition.lower)
        upper = spec.index(transition.upper)
        x_op = np.zeros((dim, dim), dtype=complex)
        x_op[upper, lower] = 1.0
        x_op[lower, upper] = 1.0
        ops[f"X_{transition.name}"] = x_op
        # Raising/lowering operators for complex (phase-carrying) drives, e.g.
        # the analog_3 420nm drive: |upper><lower| and its conjugate.
        sp_op = np.zeros((dim, dim), dtype=complex)
        sp_op[upper, lower] = 1.0
        ops[f"Sp_{transition.name}"] = sp_op
        ops[f"Sm_{transition.name}"] = sp_op.conj().T

    return Site(leg, state_labels=list(spec.levels), sort_charge=False, **ops)
