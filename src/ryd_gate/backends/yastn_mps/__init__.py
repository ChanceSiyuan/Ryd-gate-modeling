"""YASTN MPS-TDVP backend.

This package provides a GPU-capable MPS TDVP implementation through YASTN.
YASTN can run on NumPy for CPU smoke tests or on its PyTorch backend with
``default_device="cuda"`` when PyTorch/CUDA is installed.
"""

from .backend import YASTNBackendError, YASTNMPSBackend

__all__ = ["YASTNMPSBackend", "YASTNBackendError"]
