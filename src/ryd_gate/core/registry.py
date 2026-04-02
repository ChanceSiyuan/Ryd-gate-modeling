"""Protocol-system compatibility registry.

Maps param_set names to the Protocol subclass names they support.
Uses strings (not class references) to avoid circular imports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ryd_gate.core.atomic_system import AtomicSystem

# ======================================================================
# PROTOCOL-SYSTEM COMPATIBILITY REGISTRY
# ======================================================================

# Maps param_set names to the Protocol subclass names they support.
# Uses strings (not class references) to avoid circular imports.
PROTOCOL_REGISTRY: dict[str, list[str]] = {
    "our":     ["TOProtocol", "ARProtocol"],
    "lukin":   ["TOProtocol", "ARProtocol"],
    "analog":  ["SweepProtocol"],
    "lattice": ["SweepProtocol"],
}


def compatible_protocols(param_set: str) -> list[str]:
    """Return protocol names compatible with the given system.

    >>> compatible_protocols("our")
    ['TOProtocol', 'ARProtocol']
    >>> compatible_protocols("analog")
    ['SweepProtocol']
    """
    if param_set not in PROTOCOL_REGISTRY:
        raise ValueError(
            f"Unknown system '{param_set}'. "
            f"Available: {list(PROTOCOL_REGISTRY.keys())}"
        )
    return PROTOCOL_REGISTRY[param_set]


def check_protocol_compatibility(
    system: "AtomicSystem", protocol: object,
) -> None:
    """Raise ValueError if *protocol* is incompatible with *system*."""
    allowed = PROTOCOL_REGISTRY.get(system.param_set, [])
    proto_name = type(protocol).__name__
    if proto_name not in allowed:
        raise ValueError(
            f"Protocol '{proto_name}' is not compatible with system "
            f"'{system.param_set}'. Compatible protocols: {allowed}"
        )
