"""Physics calculations for Rb87: AC Stark shifts and ARC decay branching."""

def __getattr__(name: str):
    if name == "compute_shift_scatter":
        from ryd_gate.physics.ac_stark import compute_shift_scatter

        return compute_shift_scatter
    raise AttributeError(f"module 'ryd_gate.physics' has no attribute {name!r}")

__all__ = ["compute_shift_scatter"]
