"""Array-backend abstraction for the self-written finite-PEPS engine (``rydtn``).

Supports NumPy (CPU, the default for tests) and PyTorch (CPU/CUDA).  Every PEPS
tensor is a single dense complex array: YASTN runs here with ``sym="none"`` and a
bosonic config, so there is no block/symmetry structure to track and all network
contractions reduce to plain ``einsum``/``reshape``.

Spectral decompositions (``eigh``/``svd``) are computed in ``complex128``
internally even when the working dtype is ``complex64``; the small bond/metric
matrices they act on are cheap to upcast and this keeps single precision stable.
"""

from __future__ import annotations

import importlib.util

import numpy as np


class RydTNError(RuntimeError):
    """Raised when the rydtn PEPS engine cannot run."""


_NP_CDTYPE = {"complex128": np.complex128, "complex64": np.complex64}


class ArrayBackend:
    """Thin dispatch over NumPy / PyTorch for dense complex tensor algebra."""

    def __init__(self, kind: str, device: str, dtype: str) -> None:
        if dtype not in _NP_CDTYPE:
            raise RydTNError(f"dtype must be 'complex128' or 'complex64'; got {dtype!r}.")
        self.kind = kind
        self.device = device
        self.dtype = dtype
        if kind == "numpy":
            self.xp = np
            self.cdtype = _NP_CDTYPE[dtype]
        elif kind == "torch":
            import torch

            self.torch = torch
            self.xp = torch
            self.cdtype = torch.complex128 if dtype == "complex128" else torch.complex64
        else:
            raise RydTNError(f"backend kind must be 'numpy' or 'torch'; got {kind!r}.")

    # ---- construction ----
    def asarray(self, x, dtype=None):
        """Move ``x`` (numpy array / scalar / backend tensor) onto the backend."""
        dt = self.cdtype if dtype is None else dtype
        if self.kind == "numpy":
            return np.asarray(x, dtype=dt)
        t = self.torch
        if isinstance(x, t.Tensor):
            return x.to(dtype=dt, device=self.device)
        return t.as_tensor(np.asarray(x), device=self.device).to(dt)

    def zeros(self, shape, dtype=None):
        dt = self.cdtype if dtype is None else dtype
        if self.kind == "numpy":
            return np.zeros(shape, dtype=dt)
        return self.torch.zeros(shape, dtype=dt, device=self.device)

    def eye(self, n, dtype=None):
        dt = self.cdtype if dtype is None else dtype
        if self.kind == "numpy":
            return np.eye(n, dtype=dt)
        return self.torch.eye(n, dtype=dt, device=self.device)

    # ---- contraction / reshaping ----
    def einsum(self, spec, *ops):
        if self.kind == "numpy":
            return np.einsum(spec, *ops, optimize=True)
        return self.torch.einsum(spec, *ops)

    def transpose(self, a, axes):
        if self.kind == "numpy":
            return np.transpose(a, axes)
        return a.permute(*axes)

    def conj(self, a):
        return a.conj()

    def norm(self, a) -> float:
        if self.kind == "numpy":
            return float(np.linalg.norm(a))
        return float(self.torch.linalg.norm(a).item())

    def to_numpy(self, a) -> np.ndarray:
        if self.kind == "numpy":
            return np.asarray(a)
        return a.detach().cpu().numpy()

    def empty_cache(self) -> None:
        """Return cached CUDA blocks to the driver (no-op on CPU/NumPy)."""
        if self.kind == "torch" and str(self.device).startswith("cuda"):
            self.torch.cuda.empty_cache()

    # ---- linear algebra (computed in complex128 for stability) ----
    def eigh(self, a):
        """Hermitian eigendecomposition; ascending real eigenvalues, complex128."""
        if self.kind == "numpy":
            return np.linalg.eigh(np.asarray(a, dtype=np.complex128))
        t = self.torch
        return t.linalg.eigh(a.to(t.complex128))

    def svd(self, a):
        """Reduced SVD ``U, s, Vh`` with descending real ``s``; complex128."""
        if self.kind == "numpy":
            return np.linalg.svd(np.asarray(a, dtype=np.complex128), full_matrices=False)
        t = self.torch
        return t.linalg.svd(a.to(t.complex128), full_matrices=False)

    def qr(self, a):
        """Reduced QR in the working dtype (backward stable; ``a`` may be large)."""
        if self.kind == "numpy":
            return np.linalg.qr(a)
        return self.torch.linalg.qr(a)

    def svd_truncate(self, mat, chi_max, tol):
        """SVD of ``mat`` truncated to ``<=chi_max`` and singulars ``> tol*s0``.

        Returns ``(U, s, Vh, disc)`` where ``disc`` is the discarded weight
        ``sqrt(sum s_drop^2 / sum s^2)``.  ``U/Vh`` are cast back to the working
        dtype; ``s`` stays real.
        """
        U, s, Vh = self.svd(mat)
        k = self._num_keep(s, chi_max, tol)
        disc = self._discarded_weight(s, k)
        U, s, Vh = U[:, :k], s[:k], Vh[:k, :]
        return self.asarray(U), s, self.asarray(Vh), disc

    def _num_keep(self, s, chi_max, tol) -> int:
        if self.kind == "numpy":
            n_pos = int(np.count_nonzero(s > tol * s[0])) if s.size else 0
        else:
            n_pos = int((s > tol * s[0]).sum().item()) if s.numel() else 0
        return max(1, min(int(chi_max), n_pos))

    def _discarded_weight(self, s, k) -> float:
        if self.kind == "numpy":
            total = float(np.sum(s**2))
            drop = float(np.sum(s[k:] ** 2))
        else:
            total = float((s**2).sum().item())
            drop = float((s[k:] ** 2).sum().item())
        if total <= 0:
            return 0.0
        return float(np.sqrt(max(0.0, drop) / total))


def resolve_backend(
    use_cuda: bool = False,
    backend: str | None = None,
    device: str | None = None,
    dtype: str = "complex128",
    require_gpu: bool = False,
) -> ArrayBackend:
    """Pick a backend, mirroring ``YASTNPEPSBackend._load_yastn`` device logic."""
    kind = backend or ("torch" if use_cuda else "numpy")
    kind = {"np": "numpy", "numpy": "numpy", "torch": "torch", "pytorch": "torch"}.get(kind, kind)
    dev = device or ("cuda" if use_cuda else "cpu")
    if require_gpu or use_cuda:
        if kind != "torch":
            raise RydTNError("CUDA runs require backend='torch'.")
        if importlib.util.find_spec("torch") is None:
            raise RydTNError("CUDA runs require PyTorch with CUDA support.")
        import torch

        if not torch.cuda.is_available():
            raise RydTNError("use_cuda=True but torch.cuda.is_available() is False.")
    if kind == "numpy" and dev not in ("cpu",):
        raise RydTNError("numpy backend only supports device='cpu'.")
    return ArrayBackend(kind, dev, dtype)
