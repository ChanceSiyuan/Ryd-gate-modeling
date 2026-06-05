"""Protocol compatibility checks for the historical exact CZ solver."""

from __future__ import annotations

from typing import Any

_PROTOCOL_REGISTRY: dict[str, list[str]] = {
    "our": ["TOProtocol", "ARProtocol"],
    "lukin": ["TOProtocol", "ARProtocol"],
    "analog": ["SweepProtocol"],
    "lattice": ["SweepProtocol"],
}


def check_protocol_compatibility(system: Any, protocol: object) -> None:
    """Raise ValueError if *protocol* is incompatible with a legacy system."""
    allowed = _PROTOCOL_REGISTRY.get(system.param_set, [])
    proto_name = type(protocol).__name__
    if proto_name not in allowed:
        raise ValueError(
            f"Protocol '{proto_name}' is not compatible with system "
            f"'{system.param_set}'. Compatible protocols: {allowed}"
        )
