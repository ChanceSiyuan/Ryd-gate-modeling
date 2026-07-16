"""Array-backend conversion for the optional CUDA path. No quimb import.

CPU keeps NumPy arrays; ``device='cuda'`` moves operators/gates onto Torch-CUDA
arrays (quimb dispatches its tensor ops to that backend through autoray). Scalars
and statevectors are pulled back to the host for the public result values.
"""

from __future__ import annotations

import numpy as np


def to_array(mat, device: str):
    """A local operator on the target array backend (NumPy for cpu, Torch-CUDA for cuda)."""
    a = np.asarray(mat, dtype=complex)
    if device == "cuda":
        import torch

        return torch.as_tensor(a, dtype=torch.complex128, device="cuda")
    return a


def to_host_complex(value) -> complex:
    """Bring a possibly-device (NumPy or Torch) scalar to a host Python ``complex``."""
    return complex(value.item()) if hasattr(value, "item") else complex(value)


def to_host_array(value) -> np.ndarray:
    """Bring a possibly-device (NumPy or Torch-CUDA) array to a host NumPy array."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)
