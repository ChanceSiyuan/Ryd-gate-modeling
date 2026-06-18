"""Physical-model specialization: interactions, Rb87 parameters, local blocks.

Three pieces of atomic/interaction physics in one module:

- :func:`vdw_couplings` — the standard isotropic VdW pair sum
  V_ij = C6 / R_ij^6 with optional range truncation. Lives in ``core/``
  (not ``lattice/``) because computing interaction strengths is physics;
  the lattice package is reserved for pure geometry.
- Rb87 seven-level physical parameter sets for the ``our`` and ``lukin``
  configurations (level energies, Rabi frequencies, decay/branching rates)
  and the helper that flattens them into a system metadata dict.
- Local single-atom Hamiltonian blocks: the per-site complex matrices
  (energies, drives, light shifts) for the ``analog_3`` and Rb87
  seven-level physical models, registered on a system's block registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ryd_gate.core.level_structures import DEFAULT_C6, level_structure
from ryd_gate.core.model import BlockRegistry
from ryd_gate.core.operators import LocalMatrixSumSpec

if TYPE_CHECKING:
    from ryd_gate.core.system import RydbergSystem


# ── Rydberg-Rydberg interactions ─────────────────────────────────────────────


def vdw_couplings(
    coords_um: np.ndarray,
    C6: float,
    max_range_um: float | None = None,
) -> tuple:
    """Compute all-pairs van der Waals couplings ``V_ij = C6 / R_ij^6``.

    Parameters
    ----------
    coords_um : ndarray, shape (N, 2) or (N, 3)
        Atom positions in microns.
    C6 : float
        Isotropic VdW coefficient in rad/s · μm^6.
    max_range_um : float or None
        If given, omit pairs with separation > max_range_um.

    Returns
    -------
    tuple of (i, j, V_ij)
        Upper-triangular list of pairs with V_ij in rad/s.
    """
    coords_um = np.asarray(coords_um, dtype=float)
    N = len(coords_um)
    pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            r = float(np.linalg.norm(coords_um[i] - coords_um[j]))
            if max_range_um is not None and r > max_range_um:
                continue
            pairs.append((i, j, C6 / r ** 6))
    return tuple(pairs)


# ── Rb87 seven-level physical parameter sets ─────────────────────────────────


@dataclass(frozen=True)
class _RB87PhysicalParams:
    param_set: str
    ryd_level: int
    Delta: float
    rabi_420: float
    rabi_1013: float
    rabi_eff: float
    time_scale: float
    rabi_420_garbage: float
    rabi_1013_garbage: float
    d_mid_ratio: float
    d_ryd_ratio: float
    v_ryd: float
    v_ryd_garb: float
    ryd_zeeman_shift: float
    detuning_sign: int
    mid_state_decay_rate: float
    mid_garb_decay_rate: float
    ryd_state_decay_rate: float
    ryd_RD_rate: float
    ryd_BBR_rate: float
    ryd_garb_decay_rate: float
    ryd_branch: dict
    mid_branch: dict
    t_rise: float
    blackmanflag: bool
    enable_rydberg_decay: bool
    enable_intermediate_decay: bool
    enable_0_scattering: bool
    enable_polarization_leakage: bool
    n_levels: int = 7
    rydberg_indices: tuple[int, ...] = (5, 6)
    n_atoms: int = 2


def _rb87_default_c6(param_set: str) -> float:
    if param_set == "lukin":
        return 2 * np.pi * 450e6 * 3.0**6
    return DEFAULT_C6


def _rb87_physical_params(
    param_set: str,
    *,
    detuning_sign: int,
    blackmanflag: bool,
    enable_rydberg_decay: bool,
    enable_intermediate_decay: bool,
    enable_0_scattering: bool,
    enable_polarization_leakage: bool,
) -> _RB87PhysicalParams:
    from arc import Rubidium87

    from ryd_gate.physics import _mid_branching_ratios, _rydberg_branching_ratios

    atom = Rubidium87()
    if param_set == "our":
        ryd_level = 70
        Delta = detuning_sign * 2 * np.pi * 9.1e9
        rabi_420 = 2 * np.pi * 491e6
        rabi_1013 = 2 * np.pi * 185e6
        d_mid_ratio = atom.getDipoleMatrixElement(5, 0, 0.5, 0.5, 6, 1, 1.5, -0.5, -1) / atom.getDipoleMatrixElement(
            5, 0, 0.5, -0.5, 6, 1, 1.5, -1.5, -1
        )
        d_ryd_ratio = atom.getDipoleMatrixElement(
            6, 1, 1.5, -0.5, ryd_level, 0, 0.5, 0.5, 1
        ) / atom.getDipoleMatrixElement(6, 1, 1.5, -1.5, ryd_level, 0, 0.5, -0.5, 1)
        v_ryd = 2 * np.pi * 874e9 / 3**6
        v_ryd_garb = v_ryd
        ryd_zeeman_shift = 2 * np.pi * 56e6 if enable_polarization_leakage else 2 * np.pi * 56e9
        mid_state_decay_rate = 1 / 110.7e-9
        ryd_state_decay_rate = 1 / 151.55e-6
        ryd_RD_rate = 1 / 410.41e-6
        ryd_branch = _rydberg_branching_ratios(atom, ryd_level, "our")
        mid_branch = {F: _mid_branching_ratios(atom, F, mF=-1) for F in (1, 2, 3)}
    elif param_set == "lukin":
        ryd_level = 53
        Delta = detuning_sign * 2 * np.pi * 7.8e9
        rabi_420 = 2 * np.pi * 237e6
        rabi_1013 = 2 * np.pi * 303e6
        d_mid_ratio = atom.getDipoleMatrixElement(5, 0, 0.5, -0.5, 6, 1, 1.5, 0.5, 1) / atom.getDipoleMatrixElement(
            5, 0, 0.5, 0.5, 6, 1, 1.5, 1.5, 1
        )
        d_ryd_ratio = atom.getDipoleMatrixElement(
            6, 1, 1.5, 0.5, ryd_level, 0, 0.5, -0.5, -1
        ) / atom.getDipoleMatrixElement(6, 1, 1.5, 1.5, ryd_level, 0, 0.5, 0.5, -1)
        v_ryd = 2 * np.pi * 450e6
        v_ryd_garb = v_ryd
        ryd_zeeman_shift = 2 * np.pi * 2.4e9 if enable_polarization_leakage else 2 * np.pi * 2.4e12
        mid_state_decay_rate = 1 / 110e-9
        ryd_state_decay_rate = 1 / 88e-6
        ryd_RD_rate = 1 / 147.64e-6
        ryd_branch = _rydberg_branching_ratios(atom, ryd_level, "lukin")
        mid_branch = {F: _mid_branching_ratios(atom, F, mF=1) for F in (1, 2, 3)}
    else:
        raise ValueError(f"Unknown rb87_7 parameter set '{param_set}'.")

    rabi_420_garbage = rabi_420 * d_mid_ratio
    rabi_1013_garbage = rabi_1013 * d_ryd_ratio
    rabi_eff = rabi_420 * rabi_1013 / (2 * abs(Delta))
    time_scale = 2 * np.pi / rabi_eff
    ryd_BBR_rate = ryd_state_decay_rate - ryd_RD_rate

    return _RB87PhysicalParams(
        param_set=param_set,
        ryd_level=ryd_level,
        Delta=Delta,
        rabi_420=rabi_420,
        rabi_1013=rabi_1013,
        rabi_eff=rabi_eff,
        time_scale=time_scale,
        rabi_420_garbage=rabi_420_garbage,
        rabi_1013_garbage=rabi_1013_garbage,
        d_mid_ratio=d_mid_ratio,
        d_ryd_ratio=d_ryd_ratio,
        v_ryd=v_ryd,
        v_ryd_garb=v_ryd_garb,
        ryd_zeeman_shift=ryd_zeeman_shift,
        detuning_sign=detuning_sign,
        mid_state_decay_rate=mid_state_decay_rate,
        mid_garb_decay_rate=mid_state_decay_rate,
        ryd_state_decay_rate=ryd_state_decay_rate,
        ryd_RD_rate=ryd_RD_rate,
        ryd_BBR_rate=ryd_BBR_rate,
        ryd_garb_decay_rate=ryd_state_decay_rate,
        ryd_branch=ryd_branch,
        mid_branch=mid_branch,
        t_rise=20e-9,
        blackmanflag=blackmanflag,
        enable_rydberg_decay=enable_rydberg_decay,
        enable_intermediate_decay=enable_intermediate_decay,
        enable_0_scattering=enable_0_scattering,
        enable_polarization_leakage=enable_polarization_leakage,
    )


def _metadata_from_rb87_params(system: _RB87PhysicalParams) -> dict[str, Any]:
    return {
        "rabi_eff": system.rabi_eff,
        "time_scale": system.time_scale,
        "t_rise": system.t_rise,
        "blackmanflag": system.blackmanflag,
        "n_atoms": system.n_atoms,
        "n_levels": system.n_levels,
        "rabi_420": system.rabi_420,
        "rabi_1013": system.rabi_1013,
        "rabi_420_garbage": system.rabi_420_garbage,
        "rabi_1013_garbage": system.rabi_1013_garbage,
        "Delta": system.Delta,
        "v_ryd": system.v_ryd,
        "v_ryd_garb": system.v_ryd_garb,
        "ryd_state_decay_rate": system.ryd_state_decay_rate,
        "ryd_RD_rate": system.ryd_RD_rate,
        "ryd_BBR_rate": system.ryd_BBR_rate,
        "mid_state_decay_rate": system.mid_state_decay_rate,
        "ryd_branch": system.ryd_branch,
        "mid_branch": system.mid_branch,
        "rydberg_indices": system.rydberg_indices,
        "enable_rydberg_decay": system.enable_rydberg_decay,
        "enable_intermediate_decay": system.enable_intermediate_decay,
        "enable_0_scattering": system.enable_0_scattering,
        "enable_polarization_leakage": system.enable_polarization_leakage,
    }


# ── Local single-atom Hamiltonian blocks ─────────────────────────────────────


def _register_local_matrix_block(
    blocks: BlockRegistry,
    name: str,
    matrix: np.ndarray,
    *,
    hermitian: bool = True,
    description: str = "",
) -> None:
    blocks.register(
        name,
        LocalMatrixSumSpec(np.asarray(matrix, dtype=np.complex128)),
        description=description,
        hermitian=hermitian,
    )


@dataclass(frozen=True, eq=False)
class Analog3Blocks:
    """analog_3 single-atom 3x3 blocks and scalars (shared by exact + TN paths).

    ``h_const``/``h_1013``/``drive_420`` are the registered exact-backend blocks;
    ``static`` is their time-independent sum used by the TN backends, while
    ``drive_420`` is the base operator modulated each step by the protocol's
    (generally complex) ``drive_420`` coefficient.
    """

    h_const: np.ndarray
    h_1013: np.ndarray
    drive_420: np.ndarray
    rydberg_index: int
    hermitian: bool
    Delta: float
    rabi_420: float
    rabi_1013: float
    rabi_eff: float
    time_scale: float

    @property
    def static(self) -> np.ndarray:
        """``H_const + H_1013 + H_1013^dag`` — the time-independent local Hamiltonian."""
        return self.h_const + self.h_1013 + self.h_1013.conj().T

    @property
    def drive_420_dag(self) -> np.ndarray:
        return self.drive_420.conj().T


_ANALOG3_MID_DECAY_RATE = 1 / 110.7e-9
_ANALOG3_RYD_DECAY_RATE = 1 / 151.55e-6


def _analog3_blocks(Delta, rabi_420, rabi_1013, mid_decay, ryd_decay, rabi_eff, time_scale) -> Analog3Blocks:
    h_const = np.zeros((3, 3), dtype=np.complex128)
    h_const[1, 1] = Delta - 1j * mid_decay / 2
    h_const[2, 2] = -1j * ryd_decay / 2
    h_1013 = np.zeros((3, 3), dtype=np.complex128)
    h_1013[2, 1] = rabi_1013 / 2
    drive_420 = np.zeros((3, 3), dtype=np.complex128)
    drive_420[1, 0] = rabi_420 / 2
    return Analog3Blocks(
        h_const=h_const, h_1013=h_1013, drive_420=drive_420, rydberg_index=2,
        hermitian=(mid_decay == 0.0 and ryd_decay == 0.0),
        Delta=float(Delta), rabi_420=float(rabi_420), rabi_1013=float(rabi_1013),
        rabi_eff=float(rabi_eff), time_scale=float(time_scale),
    )


def analog_3_local_blocks(
    *,
    Delta_Hz: float | None = None,
    rabi_420_Hz: float | None = None,
    rabi_1013_Hz: float | None = None,
    detuning_sign: int = 1,
    enable_rydberg_decay: bool = False,
    enable_intermediate_decay: bool = False,
) -> Analog3Blocks:
    """Build the analog_3 single-atom blocks from physical (Hz) knobs.

    Single source of truth for the analog_3 local Hamiltonian: the exact path
    (``_apply_analog_3_lattice_blocks``) and the TN lattice-spec builders both go
    through this, so the matrices stay bit-identical across backends.
    """
    Delta = detuning_sign * 2 * np.pi * (Delta_Hz if Delta_Hz is not None else 9.1e9)
    rabi_420 = 2 * np.pi * (rabi_420_Hz if rabi_420_Hz is not None else 491e6)
    rabi_1013 = 2 * np.pi * (rabi_1013_Hz if rabi_1013_Hz is not None else 491e6)
    rabi_eff = rabi_420 * rabi_1013 / (2 * abs(Delta))
    time_scale = 2 * np.pi / rabi_eff
    mid_decay = _ANALOG3_MID_DECAY_RATE if enable_intermediate_decay else 0.0
    ryd_decay = _ANALOG3_RYD_DECAY_RATE if enable_rydberg_decay else 0.0
    return _analog3_blocks(Delta, rabi_420, rabi_1013, mid_decay, ryd_decay, rabi_eff, time_scale)


def analog_3_local_blocks_from_metadata(metadata: dict | None) -> Analog3Blocks:
    """Reconstruct analog_3 blocks from a system/IR metadata dict (angular rad/s scalars).

    Falls back to the default analog_3 constants when the scalars are absent.
    """
    if not metadata or "Delta" not in metadata:
        return analog_3_local_blocks()
    Delta = float(metadata["Delta"])
    rabi_420 = float(metadata["rabi_420"])
    rabi_1013 = float(metadata["rabi_1013"])
    rabi_eff = float(metadata.get("rabi_eff") or rabi_420 * rabi_1013 / (2 * abs(Delta)))
    time_scale = float(metadata.get("time_scale") or 2 * np.pi / rabi_eff)
    mid_decay = float(metadata.get("mid_state_decay_rate", 0.0)) if metadata.get("enable_intermediate_decay") else 0.0
    ryd_decay = float(metadata.get("ryd_state_decay_rate", 0.0)) if metadata.get("enable_rydberg_decay") else 0.0
    return _analog3_blocks(Delta, rabi_420, rabi_1013, mid_decay, ryd_decay, rabi_eff, time_scale)


def _apply_analog_3_lattice_blocks(
    model: "RydbergSystem",
    *,
    detuning_sign: int = 1,
    blackmanflag: bool = True,
    enable_rydberg_decay: bool = False,
    enable_intermediate_decay: bool = False,
    Delta_Hz: float | None = None,
    rabi_420_Hz: float | None = None,
    rabi_1013_Hz: float | None = None,
    **unused,
) -> None:
    _reject_unused(unused)
    ryd_level = 70
    blk = analog_3_local_blocks(
        Delta_Hz=Delta_Hz,
        rabi_420_Hz=rabi_420_Hz,
        rabi_1013_Hz=rabi_1013_Hz,
        detuning_sign=detuning_sign,
        enable_rydberg_decay=enable_rydberg_decay,
        enable_intermediate_decay=enable_intermediate_decay,
    )
    ryd_RD_rate = 1 / 410.41e-6
    ryd_BBR_rate = _ANALOG3_RYD_DECAY_RATE - ryd_RD_rate
    lightshift_zero = np.zeros((3, 3), dtype=np.complex128)

    _register_local_matrix_block(model.blocks, "H_const", blk.h_const, description="single-atom ger energies")
    _register_local_matrix_block(model.blocks, "H_1013", blk.h_1013, hermitian=False, description="static e-r coupling")
    _register_local_matrix_block(model.blocks, "H_1013_conj", blk.h_1013.conj().T, hermitian=False)
    _register_local_matrix_block(model.blocks, "drive_420", blk.drive_420, hermitian=False, description="g-e drive")
    _register_local_matrix_block(model.blocks, "drive_420_dag", blk.drive_420.conj().T, hermitian=False)
    _register_local_matrix_block(model.blocks, "lightshift_zero", lightshift_zero)

    model.metadata.update(
        {
            "physical_model": "analog_3",
            "rabi_eff": blk.rabi_eff,
            "time_scale": blk.time_scale,
            "t_rise": 20e-9,
            "blackmanflag": blackmanflag,
            "n_atoms": model.N,
            "n_levels": 3,
            "rabi_420": blk.rabi_420,
            "rabi_1013": blk.rabi_1013,
            "rabi_420_garbage": 0.0,
            "rabi_1013_garbage": 0.0,
            "Delta": blk.Delta,
            "v_ryd": _nearest_pair_strength(model.metadata.get("interaction_pairs", ())),
            "v_ryd_garb": 0.0,
            "ryd_level": ryd_level,
            "ryd_state_decay_rate": _ANALOG3_RYD_DECAY_RATE,
            "ryd_RD_rate": ryd_RD_rate,
            "ryd_BBR_rate": ryd_BBR_rate,
            "mid_state_decay_rate": _ANALOG3_MID_DECAY_RATE,
            "ryd_branch": {},
            "mid_branch": {},
            "rydberg_indices": (2,),
            "enable_rydberg_decay": enable_rydberg_decay,
            "enable_intermediate_decay": enable_intermediate_decay,
            "enable_0_scattering": False,
            "enable_polarization_leakage": False,
        }
    )


def _apply_rb87_7_lattice_blocks(
    model: "RydbergSystem",
    param_set: str,
    *,
    detuning_sign: int = 1,
    blackmanflag: bool = True,
    enable_rydberg_decay: bool = False,
    enable_intermediate_decay: bool = False,
    enable_0_scattering: bool = True,
    enable_polarization_leakage: bool = False,
    **unused,
) -> None:
    _reject_unused(unused)
    physical = _rb87_physical_params(
        param_set,
        detuning_sign=detuning_sign,
        blackmanflag=blackmanflag,
        enable_rydberg_decay=enable_rydberg_decay,
        enable_intermediate_decay=enable_intermediate_decay,
        enable_0_scattering=enable_0_scattering,
        enable_polarization_leakage=enable_polarization_leakage,
    )

    h_const = _rb87_local_h_const(
        physical.Delta,
        physical.ryd_zeeman_shift,
        physical.mid_state_decay_rate if enable_intermediate_decay else 0.0,
        physical.ryd_state_decay_rate if enable_rydberg_decay else 0.0,
    )
    h420 = _rb87_local_h420(param_set, physical.rabi_420, physical.rabi_420_garbage)
    h1013 = _rb87_local_h1013(param_set, physical.rabi_1013, physical.rabi_1013_garbage)
    lightshift_zero = _rb87_local_zero_lightshift(
        param_set,
        physical.Delta,
        physical.rabi_420,
        physical.rabi_420_garbage,
        physical.mid_state_decay_rate,
        enable_intermediate_decay,
        enable_0_scattering,
    )

    _register_local_matrix_block(model.blocks, "H_const", h_const, description="single-atom rb87_7 energies")
    _register_local_matrix_block(model.blocks, "H_1013", h1013, hermitian=False, description="static 1013nm coupling")
    _register_local_matrix_block(model.blocks, "H_1013_conj", h1013.conj().T, hermitian=False)
    _register_local_matrix_block(model.blocks, "drive_420", h420, hermitian=False, description="420nm drive")
    _register_local_matrix_block(model.blocks, "drive_420_dag", h420.conj().T, hermitian=False)
    _register_local_matrix_block(model.blocks, "lightshift_zero", lightshift_zero)

    model.metadata.update(_metadata_from_rb87_params(physical))
    model.metadata.update(
        {
            "physical_model": param_set,
            "n_atoms": model.N,
            "n_sites": model.N,
            "level_structure": "rb87_7",
            "level_spec": level_structure("rb87_7"),
            "v_ryd": _nearest_pair_strength(model.metadata.get("interaction_pairs", ())),
            "ryd_level": physical.ryd_level,
        }
    )


def _rb87_local_h_const(
    Delta: float,
    ryd_zeeman_shift: float,
    middecay: float,
    ryddecay: float,
) -> np.ndarray:
    h = np.zeros((7, 7), dtype=np.complex128)
    h[2, 2] = Delta - 2 * np.pi * 51e6 - 1j * middecay / 2
    h[3, 3] = Delta - 1j * middecay / 2
    h[4, 4] = Delta + 2 * np.pi * 87e6 - 1j * middecay / 2
    h[5, 5] = -1j * ryddecay / 2
    h[6, 6] = ryd_zeeman_shift - 1j * ryddecay / 2
    return h


def _rb87_local_h420(param_set: str, rabi_420: float, rabi_420_garbage: float) -> np.ndarray:
    from arc.wigner import CG

    h = np.zeros((7, 7), dtype=np.complex128)
    if param_set == "our":
        for row, F in zip((2, 3, 4), (1, 2, 3)):
            h[row, 1] = (
                rabi_420 * CG(3 / 2, -3 / 2, 3 / 2, 1 / 2, F, -1)
                + rabi_420_garbage * CG(3 / 2, -1 / 2, 3 / 2, -1 / 2, F, -1)
            ) / 2
    else:
        for row, F in zip((2, 3, 4), (1, 2, 3)):
            h[row, 1] = (
                rabi_420 * CG(3 / 2, 3 / 2, 3 / 2, -1 / 2, F, 1)
                + rabi_420_garbage * CG(3 / 2, 1 / 2, 3 / 2, 1 / 2, F, 1)
            ) / 2
    return h


def _rb87_local_h1013(param_set: str, rabi_1013: float, rabi_1013_garbage: float) -> np.ndarray:
    from arc.wigner import CG

    h = np.zeros((7, 7), dtype=np.complex128)
    if param_set == "our":
        for col, F in zip((2, 3, 4), (1, 2, 3)):
            h[5, col] = (rabi_1013 / 2) * CG(3 / 2, -3 / 2, 3 / 2, 1 / 2, F, -1)
            h[6, col] = (rabi_1013_garbage / 2) * CG(3 / 2, -1 / 2, 3 / 2, -1 / 2, F, -1)
    else:
        for col, F in zip((2, 3, 4), (1, 2, 3)):
            h[5, col] = (rabi_1013 / 2) * CG(3 / 2, 3 / 2, 3 / 2, -1 / 2, F, 1)
            h[6, col] = (rabi_1013_garbage / 2) * CG(3 / 2, 1 / 2, 3 / 2, 1 / 2, F, 1)
    return h


def _rb87_local_zero_lightshift(
    param_set: str,
    Delta: float,
    rabi_420: float,
    rabi_420_garbage: float,
    mid_state_decay_rate: float,
    enable_intermediate_decay: bool,
    enable_0_scattering: bool,
) -> np.ndarray:
    from arc.wigner import CG

    E_0 = -2 * np.pi * 6.835e9
    mid_energies = np.array(
        [
            Delta - 2 * np.pi * 51e6,
            Delta,
            Delta + 2 * np.pi * 87e6,
        ],
        dtype=np.float64,
    )
    if param_set == "our":
        cg_ratio_main = CG(1 / 2, -1 / 2, 3 / 2, 1 / 2, 1, 0) / CG(1 / 2, -1 / 2, 3 / 2, 1 / 2, 2, 0)
        cg_ratio_garb = CG(1 / 2, 1 / 2, 3 / 2, -1 / 2, 1, 0) / CG(1 / 2, 1 / 2, 3 / 2, -1 / 2, 2, 0)
        couplings = [
            (
                cg_ratio_main * rabi_420 * CG(3 / 2, -3 / 2, 3 / 2, 1 / 2, F, -1)
                + cg_ratio_garb * rabi_420_garbage * CG(3 / 2, -1 / 2, 3 / 2, -1 / 2, F, -1)
            )
            / 2
            for F in (1, 2, 3)
        ]
    else:
        cg_ratio_main = CG(1 / 2, 1 / 2, 3 / 2, -1 / 2, 1, 0) / CG(1 / 2, 1 / 2, 3 / 2, -1 / 2, 2, 0)
        cg_ratio_garb = CG(1 / 2, -1 / 2, 3 / 2, 1 / 2, 1, 0) / CG(1 / 2, -1 / 2, 3 / 2, 1 / 2, 2, 0)
        couplings = [
            (
                cg_ratio_main * rabi_420 * CG(3 / 2, 3 / 2, 3 / 2, -1 / 2, F, 1)
                + cg_ratio_garb * rabi_420_garbage * CG(3 / 2, 1 / 2, 3 / 2, 1 / 2, F, 1)
            )
            / 2
            for F in (1, 2, 3)
        ]

    local = np.zeros((7, 7), dtype=np.complex128)
    total_shift = 0.0
    scatter_rate = 0.0
    gamma = mid_state_decay_rate if enable_intermediate_decay else 0.0
    for idx, (g_i, E_e) in enumerate(zip(couplings, mid_energies), start=2):
        detuning = E_e - E_0
        shift = (np.abs(g_i) ** 2) / detuning
        local[idx, idx] = shift
        total_shift += shift
        scatter_rate += (np.abs(g_i) ** 2) * gamma / (detuning**2)
    local[0, 0] = -total_shift - 1j * scatter_rate / 2 if enable_0_scattering else -total_shift
    return local


def _nearest_pair_strength(pairs: tuple) -> float:
    if not pairs:
        return 0.0
    return float(max(abs(strength) for _, _, strength in pairs))


def _reject_unused(unused: dict) -> None:
    if unused:
        names = ", ".join(sorted(unused))
        raise TypeError(f"Unused physical parameter(s): {names}")
