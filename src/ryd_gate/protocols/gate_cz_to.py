"""Time-Optimal (TO) CZ gate pulse protocol.

Phase function: φ(t) = A·cos(ωt + φ₀) + δ·t
Parameters x = [A, ω/Ω_eff, φ₀, δ/Ω_eff, θ, T/T_scale]
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ryd_gate.protocols.base import Protocol

if TYPE_CHECKING:
    from ryd_gate.core.atomic_system import AtomicSystem


class TOProtocol(Protocol):
    """Time-Optimal pulse protocol with cosine phase modulation."""

    @property
    def n_params(self) -> int:
        return 6

    @property
    def theta_index(self) -> int:
        return 4

    @property
    def t_gate_index(self) -> int:
        return 5

    def validate_params(self, x: list[float]) -> None:
        if len(x) != 6:
            raise ValueError(
                f"TO parameters must be a list of 6 elements. Got {len(x)} elements."
            )

    def unpack_params(self, x: list[float], system: AtomicSystem) -> dict:
        return {
            "phase_amp": x[0],
            "omega": x[1] * system.rabi_eff,
            "phase_init": x[2],
            "delta": x[3] * system.rabi_eff,
            "theta": x[4],
            "t_gate": x[5] * system.time_scale,
        }

    def phase_420(self, t: float, params: dict) -> complex:
        return np.exp(
            -1j * (params["phase_amp"] * np.cos(params["omega"] * t + params["phase_init"])
                   + params["delta"] * t)
        )

    def get_optimization_bounds(self) -> tuple:
        return (
            (-np.pi, np.pi),
            (-10, 10),
            (-np.pi, np.pi),
            (-2, 2),
            (-np.inf, np.inf),
            (-np.pi, np.pi),
        )
