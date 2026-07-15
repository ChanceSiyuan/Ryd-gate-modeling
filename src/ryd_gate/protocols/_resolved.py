"""Private resolved-drive IR — the single protocol→backend lowering seam (P17).

``protocol._resolve(system) -> _ResolvedProtocol`` is the *only* call any
backend (exact / MPS / PEPS) or the noise layer makes into a protocol. These
types are private implementation, never user API and never exported.

A drive is exactly one of:

- :class:`_LaserDrive` — one physical laser group. Its ``coefficient(t)`` is the
  laser's complex drive amplitude in rad/s *before* per-transition expansion (it
  already contains the envelope, the peak Rabi and ``exp(-i*phase)``). The
  system preset's private ``_LaserLeg`` table expands it onto primitive
  ``E[ket,bra]`` channels (carrying the CG/dipole factor, which already includes
  the ``1/2``). Laser identity is preserved so amplitude/frequency noise can be
  injected per group (N02/N14).
- :class:`_ChannelDrive` — one effective-Hamiltonian primitive channel. Its
  ``coefficient(t)`` *is* the Hamiltonian matrix element ``H[ket,bra](t)`` (P10:
  the compiler adds only the Hermitian conjugate for off-diagonal channels,
  never an extra ``1/2``). It returns a scalar (uniform, broadcast over sites)
  or a length-``N`` array (per-site profile).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Union

import numpy as np


@dataclass(frozen=True)
class _LaserDrive:
    """One physical laser group (``"420"`` / ``"1013"`` / ``"297"``)."""

    group: str
    coefficient: Callable[[float], complex]


@dataclass(frozen=True)
class _ChannelDrive:
    """One effective-Hamiltonian primitive channel (``E[ket,bra]``)."""

    channel: str
    coefficient: Callable[[float], "complex | np.ndarray"]


_Drive = Union[_LaserDrive, _ChannelDrive]


@dataclass(frozen=True)
class _ResolvedProtocol:
    """Immutable resolved protocol: real gate duration + typed drive list."""

    t_gate: float
    drives: tuple[_Drive, ...]
