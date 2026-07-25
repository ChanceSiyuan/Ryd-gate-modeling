"""The on-disk scan store: manifest + append-only chunk/scatter series.

``Store`` owns the atomic NPZ writes, the three-hash provenance gates, the
chunk/trajectory/scatter series and the derived record loaders — everything that
is byte-identical between the two sweep scripts.  Each script keeps its own
serialized field name (``delta_idx`` / ``n_idx``) and physical descriptor
columns; the Store takes them as ``key_type`` / ``key_fields`` and a
``ProvenanceColumns`` bundle so the serialized formats stay frozen.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
import uuid
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np

# Repo root of the sweeplib checkout: scripts/sweeplib/store.py -> up three dirs.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TIER_RANK = {"production": 0, "audit": 1}
_NO_STATES = np.empty((4, 0), dtype=np.complex128)  # states-skipped sentinel


def _atomic_savez(path: str, **arrays) -> None:
    for name, value in arrays.items():
        if np.asarray(value).dtype == object:
            raise TypeError(
                f"array {name!r} has dtype=object; chunks must load with "
                "allow_pickle=False")
    tmp = f"{path}.tmp-{uuid.uuid4().hex[:8]}"
    try:
        with open(tmp, "wb") as fh:
            np.savez(fh, **arrays)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    dir_fd = os.open(os.path.dirname(path) or ".", os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def _git_state(repo_root: str) -> dict:
    def _run(*args):
        try:
            return subprocess.run(
                ["git", *args], cwd=repo_root, capture_output=True, text=True,
                timeout=10).stdout.strip()
        except Exception:
            return ""
    return {
        "commit": _run("rev-parse", "HEAD"),
        "dirty": bool(_run("status", "--porcelain")),
    }


@dataclass(frozen=True)
class ProvenanceColumns:
    """Per-script serialization specifics for the frozen chunk/scatter format.

    ``descriptor`` returns the base physical-descriptor columns of one batch
    (panel axis / t_gate / omega / dsweep); ``result_extra`` returns the extended
    coherent-chunk columns (rad/s conversions and any fixed model constants such
    as ODE's ``omega_1013_rad_s``, read from the manifest).  ``scatter_channels``
    is the ordered channel table (its ``gamma_<ch[2:]>`` columns are derived);
    ``default_dim`` is the Hilbert dimension used for failed/empty rows.
    """

    scatter_channels: tuple[str, ...]
    default_dim: int
    descriptor: Callable[[object, Sequence], dict]
    result_extra: Callable[[object, Sequence, dict], dict]
    schema_version: int = 1


@dataclass
class PointRecord:
    """One per-point result row loaded back from a chunk."""

    key: object
    tier: str
    rtol: float
    atol: float
    status: str                  # 'ok' | 'failed' | 'timeout'
    max_leakage: float
    leakage: np.ndarray          # (4,)
    worst_input: str
    return_prob: np.ndarray      # (4,)
    norm_err: np.ndarray         # (4,)
    psi_final: np.ndarray        # (4, dim)
    nfev: int
    runtime_s: float
    batch_id: str
    batch_size: int
    retry_count: int
    priority_score: float
    message: str
    chunk_file: str
    used_swap: bool


class Store:
    """The on-disk scan store: manifest + chunk read/write + derived indices."""

    def __init__(self, output_dir: str, *, key_type, key_fields: Sequence[str],
                 provenance_columns: ProvenanceColumns):
        self.root = output_dir
        self.chunks_dir = os.path.join(output_dir, "chunks")
        self.traj_dir = os.path.join(output_dir, "trajectories")
        self.scatter_dir = os.path.join(output_dir, "scatter")
        self.logs_dir = os.path.join(output_dir, "logs")
        self.reports_dir = os.path.join(output_dir, "reports")
        self.exports_dir = os.path.join(output_dir, "exports")
        self.plots_dir = os.path.join(output_dir, "plots")
        self.manifest_path = os.path.join(output_dir, "manifest.json")
        self.key_type = key_type
        self.key_fields = tuple(key_fields)
        self.provenance = provenance_columns

    @property
    def scatter_channels(self) -> tuple[str, ...]:
        return self.provenance.scatter_channels

    def ensure_dirs(self) -> None:
        for d in (self.root, self.chunks_dir, self.traj_dir, self.scatter_dir,
                  self.logs_dir, self.reports_dir, self.exports_dir,
                  self.plots_dir):
            os.makedirs(d, exist_ok=True)

    # -- key <-> array serialization --------------------------------------

    def keys_to_arrays(self, keys: Sequence) -> dict[str, np.ndarray]:
        f = self.key_fields
        arrays = {
            f[0]: np.asarray([getattr(k, f[0]) for k in keys], dtype=np.int16),
            f[1]: np.asarray([getattr(k, f[1]) for k in keys], dtype=np.int16),
        }
        for name in f[2:]:
            arrays[name] = np.asarray([getattr(k, name) for k in keys], dtype=np.int32)
        return arrays

    def arrays_to_keys(self, d) -> list:
        cols = [d[name] for name in self.key_fields]
        return [self.key_type(*(int(v) for v in row)) for row in zip(*cols)]

    # -- manifest ---------------------------------------------------------

    def load_manifest(self) -> dict | None:
        if not os.path.exists(self.manifest_path):
            return None
        with open(self.manifest_path) as fh:
            return json.load(fh)

    def init_or_validate_manifest(
        self,
        cfg,
        model_hash: str,
        code_hash: str,
        run_meta: dict,
        *,
        pulse_hash: str,
        axes: dict,
        extra_fields: dict | None = None,
        extra_guard: Callable[[dict], None] | None = None,
    ) -> dict:
        """Create the manifest on first run; on resume, refuse hash mismatches.

        The three provenance hashes are always guarded; ``extra_guard`` covers any
        script-specific resume check (ODE's fixed 1013 Rabi).  ``axes`` and
        ``extra_fields`` carry the script-specific manifest payload (panel axis
        anchors; ODE's ``omega_1013`` block).
        """
        existing = self.load_manifest()
        if existing is not None:
            for name, val in (("physics_hash", cfg.physics_hash()),
                              ("model_hash", model_hash),
                              ("pulse_hash", pulse_hash)):
                if existing.get(name) != val:
                    raise RuntimeError(
                        f"{name} mismatch: manifest has {existing.get(name)!r}, current "
                        f"code/model gives {val!r}.  Refusing to mix data produced by "
                        "different physics/model/pulse code — use a fresh --output "
                        "directory."
                    )
            if extra_guard is not None:
                extra_guard(existing)
            return existing
        self.ensure_dirs()
        manifest = {
            "schema_version": self.provenance.schema_version,
            "scan_uuid": uuid.uuid4().hex,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "git": _git_state(_REPO_ROOT),
            "physics": cfg.physics_payload(),
            "physics_hash": cfg.physics_hash(),
            "model_hash": model_hash,
            "pulse_hash": pulse_hash,
            "code_hash": code_hash,
            **(extra_fields or {}),
            "tolerances": {
                "production": {"rtol": cfg.rtol_production, "atol": cfg.atol_production},
                "audit": {"rtol": cfg.rtol_audit, "atol": cfg.atol_audit},
            },
            "axes": axes,
            "policies": {
                "interp_space": cfg.interp_space,
                "credibility_floor": "vmin = max(1e-12, 10 * P95(|L_prod - L_audit|))",
                "refine_residual_dex": 0.25,
                "refine_contour_residual_dex": 0.10,
                "decision_contours": [1e-3, 1e-2, 1e-4],
            },
            "run_meta": run_meta,
        }
        tmp = self.manifest_path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(manifest, fh, indent=2, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.manifest_path)
        return manifest

    # -- chunk writing ------------------------------------------------------

    def next_seq(self) -> int:
        seqs = [0]
        if os.path.isdir(self.chunks_dir):
            for name in os.listdir(self.chunks_dir):
                if name.startswith("chunk_") and name.endswith(".npz"):
                    try:
                        seqs.append(int(name[len("chunk_"):-len(".npz")]))
                    except ValueError:
                        pass
        if os.path.isdir(self.traj_dir):
            for name in os.listdir(self.traj_dir):
                if name.startswith("traj_") and name.endswith(".npz"):
                    try:
                        seqs.append(int(name[len("traj_"):-len(".npz")]))
                    except ValueError:
                        pass
        return max(seqs) + 1

    def write_result_chunk(
        self,
        seq: int,
        manifest: dict,
        keys: Sequence,
        cfg,
        tier: str,
        rtol: float,
        atol: float,
        batch_id: str,
        result,
        runtime_s: float,
        statuses: Sequence[str] | None = None,
        message: str = "",
        retry_count: int = 0,
        priority_scores: Sequence[float] | None = None,
    ) -> str:
        """Persist one completed (or failed) batch as an append-only chunk."""
        n = len(keys)
        dim = self.provenance.default_dim
        if result is not None:
            psi = result.psi_final
            dim = psi.shape[2]
            leak, max_leak = result.leakage, result.max_leakage
            worst, ret = result.worst_input, result.return_prob
            nerr = result.norm_err
            nfev = np.full(n, result.nfev // max(n, 1), dtype=np.int64)
            used_swap = result.used_swap
        else:
            psi = np.full((n, 4, dim), np.nan, dtype=np.complex128)
            leak = np.full((n, 4), np.nan)
            max_leak = np.full(n, np.nan)
            worst = [""] * n
            ret = np.full((n, 4), np.nan)
            nerr = np.full((n, 4), np.nan)
            nfev = np.zeros(n, dtype=np.int64)
            used_swap = False
        statuses = list(statuses) if statuses is not None else ["ok"] * n
        scores = (np.asarray(priority_scores, dtype=float)
                  if priority_scores is not None else np.zeros(n))

        payload = dict(
            schema_version=np.int64(self.provenance.schema_version),
            scan_uuid=str(manifest["scan_uuid"]),
            physics_hash=str(manifest["physics_hash"]),
            model_hash=str(manifest["model_hash"]),
            pulse_hash=str(manifest["pulse_hash"]),
            **self.keys_to_arrays(keys),
            **self.provenance.descriptor(cfg, keys),
            **self.provenance.result_extra(cfg, keys, manifest),
            psi_final=psi,
            leakage=leak,
            max_leakage=max_leak,
            worst_input=np.asarray(worst, dtype="U2"),
            return_prob=ret,
            norm_err=nerr,
            all_finite=np.asarray([bool(np.all(np.isfinite(psi[i].view(float))))
                                   for i in range(n)]),
            solver=np.asarray(["DOP853"] * n, dtype="U8"),
            tier=np.asarray([tier] * n, dtype="U10"),
            rtol=np.full(n, rtol),
            atol=np.full(n, atol),
            used_swap=np.asarray([used_swap] * n),
            nfev=nfev,
            runtime_s=np.full(n, runtime_s / max(n, 1)),
            batch_id=np.asarray([batch_id] * n, dtype="U40"),
            batch_size=np.full(n, n, dtype=np.int32),
            status=np.asarray(statuses, dtype="U16"),
            message=np.asarray([message] * n, dtype="U240"),
            retry_count=np.full(n, retry_count, dtype=np.int32),
            priority_score=scores,
        )
        path = os.path.join(self.chunks_dir, f"chunk_{seq:06d}.npz")
        _atomic_savez(path, **payload)
        return path

    def write_trajectory_chunk(
        self,
        seq: int,
        manifest: dict,
        key,
        tier: str,
        times: np.ndarray,
        states: np.ndarray,
    ) -> str:
        payload = dict(
            schema_version=np.int64(self.provenance.schema_version),
            scan_uuid=str(manifest["scan_uuid"]),
            **self.keys_to_arrays([key]),
            tier=np.asarray([tier], dtype="U10"),
            times=np.asarray(times, dtype=float),
            states=np.asarray(states, dtype=np.complex128),
        )
        path = os.path.join(self.traj_dir, f"traj_{seq:06d}.npz")
        _atomic_savez(path, **payload)
        return path

    # -- scattering supplement (separate append-only series) ---------------

    def next_scatter_seq(self) -> int:
        seqs = [0]
        if os.path.isdir(self.scatter_dir):
            for name in os.listdir(self.scatter_dir):
                if name.startswith("scatter_") and name.endswith(".npz"):
                    try:
                        seqs.append(int(name[len("scatter_"):-len(".npz")]))
                    except ValueError:
                        pass
        return max(seqs) + 1

    def write_scatter_chunk(
        self,
        seq: int,
        manifest: dict,
        keys: Sequence,
        cfg,
        gammas: dict,
        rtol: float,
        atol: float,
        batch_id: str,
        scatter: dict[str, np.ndarray],
        max_leakage: np.ndarray,
        runtime_s: float,
        statuses: Sequence[str] | None = None,
        message: str = "",
    ) -> str:
        """Persist one scattering-supplement batch; never touches other series.

        ``gammas`` is keyed by panel row index (all keys of a batch share one
        panel); the ``gamma_<ch[2:]>`` columns record that panel's decay rates.
        """
        n = len(keys)
        statuses = list(statuses) if statuses is not None else ["ok"] * n
        channels = self.provenance.scatter_channels
        panel_gammas = gammas[getattr(keys[0], self.key_fields[0])]
        payload = dict(
            schema_version=np.int64(self.provenance.schema_version),
            scan_uuid=str(manifest["scan_uuid"]),
            physics_hash=str(manifest["physics_hash"]),
            model_hash=str(manifest["model_hash"]),
            pulse_hash=str(manifest["pulse_hash"]),
            **self.keys_to_arrays(keys),
            **self.provenance.descriptor(cfg, keys),
            **{name: np.asarray(scatter[name]).reshape(n, 4) for name in channels},
            max_leakage_check=np.asarray(max_leakage),
            **{f"gamma_{name[2:]}": np.full(n, panel_gammas[name]) for name in channels},
            n_eval=np.full(n, cfg.n_eval_trajectory, dtype=np.int32),
            rtol=np.full(n, rtol),
            atol=np.full(n, atol),
            batch_id=np.asarray([batch_id] * n, dtype="U40"),
            batch_size=np.full(n, n, dtype=np.int32),
            status=np.asarray(statuses, dtype="U16"),
            message=np.asarray([message] * n, dtype="U240"),
            runtime_s=np.full(n, runtime_s / max(n, 1)),
        )
        path = os.path.join(self.scatter_dir, f"scatter_{seq:06d}.npz")
        _atomic_savez(path, **payload)
        return path

    def load_scatter_records(self, manifest: dict | None = None) -> list[dict]:
        """Per-point scattering rows from every scatter chunk (hash-validated)."""
        if manifest is None:
            manifest = self.load_manifest()
        channels = self.provenance.scatter_channels
        rows: list[dict] = []
        if not os.path.isdir(self.scatter_dir):
            return rows
        for name in sorted(os.listdir(self.scatter_dir)):
            if not (name.startswith("scatter_") and name.endswith(".npz")):
                continue
            with np.load(os.path.join(self.scatter_dir, name),
                         allow_pickle=False) as npz:
                d = {f: npz[f] for f in npz.files}
            if manifest is not None:
                for fieldname in ("physics_hash", "model_hash", "pulse_hash"):
                    if str(d[fieldname]) != manifest[fieldname]:
                        raise RuntimeError(
                            f"scatter chunk {name} has a different {fieldname}; "
                            "refusing to merge data from different model/pulse code")
            for i, key in enumerate(self.arrays_to_keys(d)):
                rows.append({
                    "key": key,
                    "status": str(d["status"][i]),
                    "rtol": float(d["rtol"][i]),
                    **{ch: np.array(d[ch][i]) for ch in channels},
                    "max_leakage_check": float(d["max_leakage_check"][i]),
                    "runtime_s": float(d["runtime_s"][i]),
                })
        return rows

    # -- loading / derived indices -----------------------------------------

    def load_records(self, manifest: dict | None = None,
                     include_states: bool = True) -> list[PointRecord]:
        """All per-point records from every chunk (hash-validated).

        ``include_states=False`` skips the (4, dim) final-state payload — use it
        for scheduling/status/plotting indices over large stores.
        """
        if manifest is None:
            manifest = self.load_manifest()
        records: list[PointRecord] = []
        if not os.path.isdir(self.chunks_dir):
            return records
        for name in sorted(os.listdir(self.chunks_dir)):
            if not (name.startswith("chunk_") and name.endswith(".npz")):
                continue
            path = os.path.join(self.chunks_dir, name)
            with np.load(path, allow_pickle=False) as npz:
                # NpzFile re-decompresses a member on every __getitem__; read once.
                d = {f: npz[f] for f in npz.files
                     if include_states or f != "psi_final"}
                if manifest is not None:
                    for fieldname in ("physics_hash", "model_hash", "pulse_hash"):
                        if str(d[fieldname]) != manifest[fieldname]:
                            raise RuntimeError(
                                f"chunk {name} has a different {fieldname}; refusing to "
                                "merge data from different model/pulse code")
                keys = self.arrays_to_keys(d)
                for i, key in enumerate(keys):
                    records.append(PointRecord(
                        key=key,
                        tier=str(d["tier"][i]),
                        rtol=float(d["rtol"][i]),
                        atol=float(d["atol"][i]),
                        status=str(d["status"][i]),
                        max_leakage=float(d["max_leakage"][i]),
                        leakage=np.array(d["leakage"][i]),
                        worst_input=str(d["worst_input"][i]),
                        return_prob=np.array(d["return_prob"][i]),
                        norm_err=np.array(d["norm_err"][i]),
                        psi_final=(np.array(d["psi_final"][i]) if include_states
                                   else _NO_STATES),
                        nfev=int(d["nfev"][i]),
                        runtime_s=float(d["runtime_s"][i]),
                        batch_id=str(d["batch_id"][i]),
                        batch_size=int(d["batch_size"][i]),
                        retry_count=int(d["retry_count"][i]),
                        priority_score=float(d["priority_score"][i]),
                        message=str(d["message"][i]),
                        chunk_file=name,
                        used_swap=bool(d["used_swap"][i]),
                    ))
        return records


def best_records(records: Iterable[PointRecord]) -> dict:
    """Tightest successful record per point: lowest rtol wins; the audit tier only
    breaks ties, so a loosened ad-hoc audit can never displace a tighter
    production record in exports."""
    best: dict = {}
    for r in records:
        if r.status != "ok":
            continue
        cur = best.get(r.key)
        if cur is None:
            best[r.key] = r
            continue
        a = (-r.rtol, TIER_RANK.get(r.tier, 0))
        b = (-cur.rtol, TIER_RANK.get(cur.tier, 0))
        if a > b:
            best[r.key] = r
    return best


def completed_keys(records: Iterable[PointRecord], tier: str | None = None) -> set:
    return {r.key for r in records
            if r.status == "ok" and (tier is None or r.tier == tier)}


def audit_pairs(records: Iterable[PointRecord]) -> list[tuple]:
    """(key, L_production, L_audit) for every point holding both successful tiers."""
    prod: dict = {}
    aud: dict = {}
    for r in records:
        if r.status != "ok":
            continue
        d = prod if r.tier == "production" else aud
        if r.key not in d:
            d[r.key] = r.max_leakage
    return [(k, prod[k], aud[k]) for k in sorted(prod.keys() & aud.keys())]
