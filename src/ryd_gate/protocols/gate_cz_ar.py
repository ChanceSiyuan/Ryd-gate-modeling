"""Amplitude-Robust (AR) CZ gate pulse protocol.

Phase function: φ(t) = A₁·sin(ωt + φ₁) + A₂·sin(2ωt + φ₂) + δ·t
Parameters x = [ω/Ω_eff, A₁, φ₁, A₂, φ₂, δ/Ω_eff, T/T_scale, θ]
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ryd_gate.protocols.base import Protocol

if TYPE_CHECKING:
    from ryd_gate.core.atomic_system import AtomicSystem


class ARProtocol(Protocol):
    """Amplitude-Robust pulse protocol with dual-sine phase modulation."""

    @property
    def n_params(self) -> int:
        return 8

    @property
    def theta_index(self) -> int:
        return 7

    @property
    def t_gate_index(self) -> int:
        return 6

    def validate_params(self, x: list[float]) -> None:
        if len(x) != 8:
            raise ValueError(
                f"AR parameters must be a list of 8 elements. Got {len(x)} elements."
            )

    def unpack_params(self, x: list[float], system: AtomicSystem) -> dict:
        return {
            "omega": x[0] * system.rabi_eff,
            "phase_amp1": x[1],
            "phase_init1": x[2],
            "phase_amp2": x[3],
            "phase_init2": x[4],
            "delta": x[5] * system.rabi_eff,
            "t_gate": x[6] * system.time_scale,
            "theta": x[7],
            "t_rise": system.t_rise,
            "blackmanflag": system.blackmanflag,
        }

    def phase_420(self, t: float, params: dict) -> complex:
        return np.exp(
            -1j * (
                params["phase_amp1"] * np.sin(params["omega"] * t + params["phase_init1"])
                + params["phase_amp2"] * np.sin(2 * params["omega"] * t + params["phase_init2"])
                + params["delta"] * t
            )
        )

    def get_drive_coefficients(self, t: float, params: dict) -> dict[str, complex]:
        """Return drive coefficients including Blackman amplitude envelope."""
        from ryd_gate.blackman import blackman_pulse

        phase = self.phase_420(t, params)
        amplitude = (
            blackman_pulse(t, params["t_rise"], params["t_gate"])
            if params.get("blackmanflag", True)
            else 1.0
        )
        return {
            "drive_420": amplitude * phase,
            "drive_420_dag": amplitude * np.conjugate(phase),
            "lightshift_zero": amplitude * amplitude,
        }

    def get_optimization_bounds(self) -> tuple:
        return (
            (-10, 10),
            (-np.pi, np.pi),
            (-np.pi, np.pi),
            (-np.pi, np.pi),
            (-np.pi, np.pi),
            (-2, 2),
            (-np.inf, np.inf),
            (-np.pi, np.pi),
        )
