"""Batching, the fork-pool worker entry, the cost model and the budget runner.

The worker context (``set_worker_context`` / ``_WORKER_CTX``) is populated in the
parent before the fork pool is created, so workers inherit the immutable model
context copy-on-write.  It carries the script's injected ``solve`` (the two-drive
vs single-drive integration wrapper, closing over any fixed model constant such as
ODE's 1013 Rabi) and its ``scattering_integrals`` so the shared worker entry stays
model-agnostic.  Gammas are always keyed by panel row index.
"""
from __future__ import annotations

import json
import math
import os
import signal
import time
import uuid
from collections import deque
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Sequence

import numpy as np

TAU = 2.0 * math.pi


# ── Worker process entry point ───────────────────────────────────────────────
#
# The parent warms ARC, compiles/aggregates every panel row, and stores the
# immutable context in module globals BEFORE creating the fork pool, so workers
# share it copy-on-write and never touch ARC's SQLite database.

_WORKER_CTX: dict = {}


def _worker_process_init() -> None:
    """Fork-pool initializer: workers ignore SIGINT (the parent coordinates a
    graceful drain on Ctrl-C) but keep the default SIGTERM so a hard abort can
    still terminate them immediately."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)


def set_worker_context(cfg, panel_ops: dict, use_swap: bool, gammas: dict | None,
                       *, key_type, solve, scattering_integrals) -> None:
    """Populate the fork-inherited worker context (parent process, pre-fork).

    ``solve(ops, t_gate, omega, d_sweep, *, rtol, atol, ramp, use_swap, t_eval)``
    and ``scattering_integrals(times, states, gammas)`` are the script's injected
    model layer; ``gammas`` is ``dict[panel_row -> dict[channel -> rate]]``.
    """
    _WORKER_CTX.clear()
    _WORKER_CTX.update(
        cfg=cfg, panel_ops=panel_ops, use_swap=use_swap, gammas=gammas,
        key_type=key_type, solve=solve, scattering_integrals=scattering_integrals)


def _worker_run_batch(spec: dict) -> dict:
    """Integrate one batch in a worker.  Never raises: failures are reported."""
    start = time.time()

    def _on_alarm(signum, frame):
        raise TimeoutError("batch exceeded its wall-clock timeout")

    prev = signal.signal(signal.SIGALRM, _on_alarm)
    signal.alarm(int(spec.get("timeout_s") or 0))
    try:
        ctx = _WORKER_CTX
        cfg = ctx["cfg"]
        ops = ctx["panel_ops"][spec["panel_idx"]]
        keys = [ctx["key_type"](*k) for k in spec["keys"]]
        t_gate = cfg.t_gate_us[spec["t_idx"]] * 1e-6
        omega = np.asarray([float(k.omega_mhz()) for k in keys]) * 1e6 * TAU
        d_sweep = np.asarray([float(k.dsweep_mhz()) for k in keys]) * 1e6 * TAU
        # ``scatter`` -> scatter/ series only; ``both`` -> the merged single-pass
        # run (coherent chunk AND scatter records from the one solve).  Either one
        # needs the trajectory sampled at ``n_eval_trajectory`` points.
        want_scatter = bool(spec.get("scatter")) or bool(spec.get("both"))
        t_eval = (np.linspace(0.0, t_gate, cfg.n_eval_trajectory)
                  if (spec.get("save_traj") or want_scatter) else None)
        result = ctx["solve"](
            ops, t_gate, omega, d_sweep,
            rtol=spec["rtol"], atol=spec["atol"], ramp=cfg.ramp_frac,
            use_swap=ctx["use_swap"] and ops.swap_symmetric, t_eval=t_eval)
        if not np.all(np.isfinite(result.psi_final.view(float))):
            raise FloatingPointError("non-finite terminal state")
        out = {"ok": True, "result": result, "runtime_s": time.time() - start}
        if want_scatter:
            # Gamma depends on the panel row; every key in a batch shares one
            # panel, so index the per-panel gammas by this batch's row.
            out["scatter"] = ctx["scattering_integrals"](
                result.times, result.states, ctx["gammas"][spec["panel_idx"]])
            if not spec.get("save_traj"):
                result.times = None      # keep the return payload small: the
                result.states = None     # trajectory is consumed, not persisted
        return out
    except TimeoutError:
        return {"ok": False, "reason": "timeout",
                "message": f"timeout after {time.time() - start:.0f}s",
                "runtime_s": time.time() - start}
    except Exception as exc:  # noqa: BLE001 — worker must never crash the pool
        import traceback
        return {"ok": False, "reason": "failed",
                "message": f"{type(exc).__name__}: {exc}"[:240],
                "traceback": traceback.format_exc()[-2000:],
                "runtime_s": time.time() - start}
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev)


# ── Cost model, batches, and the budget-aware runner ─────────────────────────


@dataclass
class Batch:
    keys: list
    tier: str = "production"
    save_traj: bool = False
    scatter: bool = False       # scattering-supplement solve -> scatter/ series
    retry_count: int = 0        # failure-split escalations (this batch's points)
    pool_retries: int = 0       # innocent requeues after pool crashes (separate budget)
    priority_scores: list | None = None
    batch_id: str = ""

    def __post_init__(self):
        if not self.batch_id:
            self.batch_id = uuid.uuid4().hex[:12]
        panels = {k.panel for k in self.keys}
        if len(panels) != 1:
            raise ValueError("a batch must stay within one panel")
        (self.panel_idx, self.t_idx), = panels


class CostModel:
    """Measured per-point runtimes -> deterministic panel-level ETA prediction."""

    DEFAULT_POINT_S = 150.0

    def __init__(self, cfg):
        self.cfg = cfg
        self.samples: dict[tuple[int, int], list[float]] = {}
        self.ratios: list[float] = []       # actual / predicted per completed batch

    def observe(self, panel: tuple[int, int], per_point_s: float) -> None:
        if np.isfinite(per_point_s) and per_point_s > 0:
            self.samples.setdefault(panel, []).append(per_point_s)

    def observe_batch(self, predicted_s: float, actual_s: float) -> None:
        if predicted_s > 0 and actual_s > 0:
            self.ratios.append(actual_s / predicted_s)

    def predict_point(self, panel: tuple[int, int]) -> float:
        if panel in self.samples:
            return float(np.median(self.samples[panel]))
        t_here = self.cfg.t_gate_us[panel[1]]
        same_row = [(np.median(v), self.cfg.t_gate_us[p[1]])
                    for p, v in self.samples.items() if p[0] == panel[0]]
        if same_row:
            med, t_ref = min(same_row, key=lambda mt: abs(mt[1] - t_here))
            return float(med * t_here / t_ref)
        anywhere = [(np.median(v), self.cfg.t_gate_us[p[1]])
                    for p, v in self.samples.items()]
        if anywhere:
            med, t_ref = min(anywhere, key=lambda mt: abs(mt[1] - t_here))
            return float(med * t_here / t_ref)
        return self.DEFAULT_POINT_S * t_here

    def inflation_p90(self) -> float:
        if len(self.ratios) < 5:
            return 1.5
        return float(max(1.0, np.percentile(self.ratios, 90)))

    def predict_batch(self, batch: Batch) -> float:
        return self.predict_point((batch.panel_idx, batch.t_idx)) * len(batch.keys)

    def eta_seconds(self, keys: Iterable, n_workers: int) -> float:
        total = sum(self.predict_point(k.panel) for k in keys)
        return total * self.inflation_p90() / max(1, n_workers)


def group_batches(keys: Sequence, batch_size: int, tier: str = "production",
                  save_traj: bool = False,
                  scores: dict | None = None) -> list[Batch]:
    """Group keys into within-panel batches of adjacent points (axis row-major)."""
    by_panel: dict[tuple[int, int], list] = {}
    for k in keys:
        by_panel.setdefault(k.panel, []).append(k)
    batches = []
    for panel in sorted(by_panel):
        pts = sorted(by_panel[panel],
                     key=lambda k: (Fraction(k.om_num, k.om_den), Fraction(k.dw_num, k.dw_den)))
        for i in range(0, len(pts), max(1, batch_size)):
            chunk = pts[i:i + max(1, batch_size)]
            batches.append(Batch(
                keys=chunk, tier=tier, save_traj=save_traj,
                priority_scores=[scores.get(k, 0.0) for k in chunk] if scores else None))
    return batches


class Runner:
    """Submits batches to a fork pool, handles retries/splits, writes all chunks."""

    def __init__(self, store, manifest: dict, cfg, args, cost: CostModel):
        self.store = store
        self.manifest = manifest
        self.cfg = cfg
        self.args = args
        self.cost = cost
        self._acquire_store_lock()
        self.seq = store.next_seq()
        self.scatter_seq = store.next_scatter_seq()
        self.gammas: dict | None = None   # dict[panel_row -> dict] set for scatter runs
        self.write_both_series = False    # merged single-pass: coherent + scatter in one step
        self.scatter_done: set = set()    # keys already in the scatter series (write-time dedup)
        self.stop_requested = False
        self.start_time = time.time()
        self.dispatch_deadline = (
            self.start_time
            + (getattr(args, "budget_hours", 24.0)
               - getattr(args, "reserve_hours", 2.0)) * 3600.0)
        self.point_timeout_s = float(args.point_timeout)
        self.failures: list[dict] = []
        self.deferred: int = 0
        self.completed_points = 0
        self.aborted = False
        self._signal_time = 0.0
        self._pool_restarts = 0
        self._executor = self._make_executor()
        self._install_signals()

    def _make_executor(self):
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        return ProcessPoolExecutor(
            max_workers=self.args.workers, mp_context=mp.get_context("fork"),
            initializer=_worker_process_init)

    def _acquire_store_lock(self) -> None:
        """Single-coordinator lock: chunk sequence numbers must not race.

        Opened append-mode so a *rejected* second launch cannot truncate the live
        coordinator's PID record; the PID/run id is written only after the flock
        succeeds.
        """
        import fcntl

        self.store.ensure_dirs()
        self._lock_fh = open(os.path.join(self.store.logs_dir, "store.lock"), "a+")
        try:
            fcntl.flock(self._lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._lock_fh.close()
            raise SystemExit(
                f"another pilot/run/audit already holds {self.store.root}; "
                "wait for it or use a different --output")
        self._lock_fh.seek(0)
        self._lock_fh.truncate()
        self._lock_fh.write(
            f"pid {os.getpid()} scan {self.manifest['scan_uuid']} "
            f"started {time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n")
        self._lock_fh.flush()

    def _install_signals(self):
        def handler(signum, frame):
            now = time.monotonic()
            # Debounce duplicate deliveries: `timeout`/terminals signal the whole
            # process group AND `uv run` forwards to its child, so one Ctrl-C can
            # arrive several times within milliseconds.  A deliberate second
            # Ctrl-C (>2 s later) escalates to a hard abort.
            if self.stop_requested and now - self._signal_time > 2.0:
                self.aborted = True
                raise KeyboardInterrupt
            if not self.stop_requested:
                self.stop_requested = True
                self._signal_time = now
                print(f"\n[signal {signum}] stopping dispatch; in-flight batches "
                      "will finish and checkpoint (repeat to abort hard)", flush=True)
        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

    def shutdown(self):
        # On a hard abort the in-flight results are already lost — don't block on
        # the workers (they ignore SIGINT), SIGTERM them so interpreter shutdown
        # doesn't join them for up to a full solve; on a clean stop everything is
        # drained, so wait is instant.
        if self.aborted:
            procs = list(getattr(self._executor, "_processes", {}).values())
            self._executor.shutdown(wait=False, cancel_futures=True)
            for p in procs:
                try:
                    p.terminate()
                except Exception:
                    pass
        else:
            self._executor.shutdown(wait=True, cancel_futures=True)
        self._lock_fh.close()

    def _submit(self, batch: Batch):
        """Submit a batch spec; a broken pool (worker died) is recreated once per
        incident so a single crash cannot burn the remaining wall budget."""
        from concurrent.futures.process import BrokenProcessPool

        spec = self._spec(batch)
        try:
            return self._executor.submit(_worker_run_batch, spec)
        except BrokenProcessPool:
            self._pool_restarts += 1
            print(f"[pool] worker pool broke (restart #{self._pool_restarts}); "
                  "recreating", flush=True)
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = self._make_executor()
            return self._executor.submit(_worker_run_batch, spec)

    def _tier_tols(self, tier: str) -> tuple[float, float]:
        if tier == "audit":
            return self.cfg.rtol_audit, self.cfg.atol_audit
        return self.cfg.rtol_production, self.cfg.atol_production

    def _spec(self, batch: Batch) -> dict:
        rtol, atol = self._tier_tols(batch.tier)
        return dict(
            keys=[tuple(getattr(k, f) for f in self.store.key_fields) for k in batch.keys],
            panel_idx=batch.panel_idx, t_idx=batch.t_idx,
            tier=batch.tier, rtol=rtol, atol=atol,
            save_traj=batch.save_traj,
            scatter=batch.scatter,
            both=(self.write_both_series and batch.tier == "production"
                  and not batch.scatter),
            timeout_s=self.point_timeout_s * len(batch.keys),
        )

    def _write_success(self, batch: Batch, out: dict) -> None:
        rtol, atol = self._tier_tols(batch.tier)
        result = out["result"]
        if batch.scatter:
            # Supplemental data: its own append-only series; the coherent-leakage
            # chunks and trajectories are never touched by a scatter batch.
            self.store.write_scatter_chunk(
                self.scatter_seq, self.manifest, batch.keys, self.cfg,
                self.gammas, rtol, atol, batch.batch_id, out["scatter"],
                result.max_leakage, out["runtime_s"])
            self.scatter_seq += 1
            self.cost.observe((batch.panel_idx, batch.t_idx),
                              out["runtime_s"] / len(batch.keys))
            self.completed_points += len(batch.keys)
            return
        self.store.write_result_chunk(
            self.seq, self.manifest, batch.keys, self.cfg,
            batch.tier, rtol, atol, batch.batch_id, result, out["runtime_s"],
            retry_count=batch.retry_count,
            priority_scores=batch.priority_scores,
        )
        self.seq += 1
        if batch.save_traj and result.states is not None:
            for i, key in enumerate(batch.keys):
                self.store.write_trajectory_chunk(
                    self.seq, self.manifest, key, batch.tier,
                    result.times, result.states[:, i])
                self.seq += 1
        if "scatter" in out:
            # Merged single-pass: the same solve carried the scattering integrals;
            # append them to the scatter/ series alongside the coherent chunk.
            self._write_merged_scatter(batch, out, rtol, atol)
        per_point = out["runtime_s"] / len(batch.keys)
        self.cost.observe((batch.panel_idx, batch.t_idx), per_point)
        self.completed_points += len(batch.keys)

    def _write_merged_scatter(self, batch: Batch, out: dict, rtol: float,
                              atol: float) -> None:
        """Append this batch's scatter integrals, skipping keys already in the
        scatter series (write-time resume dedup)."""
        new_idx = [i for i, k in enumerate(batch.keys) if k not in self.scatter_done]
        if not new_idx:
            return
        sub_keys = [batch.keys[i] for i in new_idx]
        idx = np.asarray(new_idx)
        scatter = {ch: np.asarray(out["scatter"][ch])[idx]
                   for ch in self.store.scatter_channels}
        max_leakage = np.asarray(out["result"].max_leakage)[idx]
        self.store.write_scatter_chunk(
            self.scatter_seq, self.manifest, sub_keys, self.cfg,
            self.gammas, rtol, atol, batch.batch_id, scatter, max_leakage,
            out["runtime_s"])
        self.scatter_seq += 1
        self.scatter_done.update(sub_keys)

    def _write_failure(self, batch: Batch, out: dict) -> None:
        rtol, atol = self._tier_tols(batch.tier)
        if batch.scatter:
            n = len(batch.keys)
            self.store.write_scatter_chunk(
                self.scatter_seq, self.manifest, batch.keys, self.cfg,
                self.gammas, rtol, atol, batch.batch_id,
                {ch: np.full((n, 4), np.nan) for ch in self.store.scatter_channels},
                np.full(n, np.nan), out.get("runtime_s", 0.0),
                statuses=[out.get("reason", "failed")] * n,
                message=out.get("message", ""))
            self.scatter_seq += 1
        else:
            self.store.write_result_chunk(
                self.seq, self.manifest, batch.keys, self.cfg,
                batch.tier, rtol, atol, batch.batch_id, None,
                out.get("runtime_s", 0.0),
                statuses=[out.get("reason", "failed")] * len(batch.keys),
                message=out.get("message", ""), retry_count=batch.retry_count,
                priority_scores=batch.priority_scores,
            )
            self.seq += 1
        self.failures.append({
            "batch_id": batch.batch_id, "keys": [k.id() for k in batch.keys],
            "tier": batch.tier, "reason": out.get("reason", "failed"),
            "message": out.get("message", ""), "retry_count": batch.retry_count,
        })

    def _split(self, batch: Batch) -> list[Batch]:
        half = len(batch.keys) // 2
        parts = []
        for sl in (slice(0, half), slice(half, None)):
            keys = batch.keys[sl]
            if keys:
                parts.append(Batch(
                    keys=keys, tier=batch.tier, save_traj=batch.save_traj,
                    scatter=batch.scatter,
                    retry_count=batch.retry_count + 1,
                    priority_scores=(batch.priority_scores[sl]
                                     if batch.priority_scores else None)))
        return parts

    def run_batches(self, batches: list[Batch], phase: str,
                    enforce_deadline: bool = True,
                    preserve_order: bool = False) -> None:
        """Run batches (longest-predicted first unless ``preserve_order``); split
        failures recursively; write every chunk from this (parent) process."""
        from concurrent.futures import FIRST_COMPLETED, wait

        if preserve_order:
            pending = deque(batches)
        else:
            pending = deque(sorted(batches, key=self.cost.predict_batch, reverse=True))
        inflight: dict = {}
        # Keep the executor's internal queue shallow: the deadline check runs at
        # submit time, so a deep queue could carry a whole extra wave of batches
        # past the reserve boundary.
        max_outstanding = self.args.workers + max(2, self.args.workers // 8)
        n_total = sum(len(b.keys) for b in batches)
        n_done = 0
        t0 = time.time()
        print(f"[{phase}] {len(batches)} batches / {n_total} points", flush=True)

        while pending or inflight:
            while pending and len(inflight) < max_outstanding and not self.stop_requested:
                batch = pending.popleft()
                # A batch occupies one core; its predicted wall time is the P90-
                # inflated single-core estimate.  Anything that cannot finish
                # before the dispatch deadline is deferred, keeping the reserve.
                predicted = self.cost.predict_batch(batch) * self.cost.inflation_p90()
                if enforce_deadline and time.time() + predicted > self.dispatch_deadline:
                    self.deferred += len(batch.keys)
                    continue
                fut = self._submit(batch)
                inflight[fut] = (batch, self.cost.predict_batch(batch))
            if not inflight:
                break
            done, _ = wait(list(inflight), return_when=FIRST_COMPLETED)
            for fut in done:
                batch, predicted = inflight.pop(fut)
                try:
                    out = fut.result()
                except Exception as exc:
                    # In-flight victims of a pool crash are innocent: requeue them
                    # on their own bounded budget (separate from failure splits) so
                    # repeated worker deaths cannot mark never-computed points as
                    # failed, while a deterministically pool-killing point is still
                    # isolated by splitting once its own requeues are exhausted.
                    if batch.pool_retries < 5:
                        requeued = Batch(
                            keys=batch.keys, tier=batch.tier,
                            save_traj=batch.save_traj, scatter=batch.scatter,
                            retry_count=batch.retry_count,
                            pool_retries=batch.pool_retries + 1,
                            priority_scores=batch.priority_scores)
                        pending.appendleft(requeued)
                        print(f"[{phase}] batch {batch.batch_id} lost to a pool "
                              f"error ({exc}); requeued "
                              f"({requeued.pool_retries}/5)", flush=True)
                        continue
                    out = {"ok": False, "reason": "failed",
                           "message": f"pool error: {exc}"[:240], "runtime_s": 0.0}
                if out["ok"]:
                    self._write_success(batch, out)
                    self.cost.observe_batch(predicted, out["runtime_s"])
                    n_done += len(batch.keys)
                elif len(batch.keys) > 1:
                    print(f"[{phase}] batch {batch.batch_id} "
                          f"{out.get('reason')}: splitting {len(batch.keys)} points",
                          flush=True)
                    for part in self._split(batch):
                        pending.appendleft(part)
                else:
                    self._write_failure(batch, out)
                    n_done += 1
                    print(f"[{phase}] point {batch.keys[0].id()} "
                          f"{out.get('reason')}: {out.get('message', '')}", flush=True)
            elapsed = time.time() - t0
            rate = n_done / elapsed if elapsed > 0 and n_done else 0.0
            eta = (n_total - n_done) / rate if rate > 0 else float("nan")
            print(f"[{phase}] {n_done}/{n_total} points "
                  f"({elapsed / 60:.1f} min elapsed, ~{eta / 60:.0f} min left)",
                  flush=True)
            self.write_status(phase)
            # when stop was requested, the submit loop stays closed and the outer
            # loop simply drains the in-flight futures before returning
        if self.stop_requested:
            print(f"[{phase}] stopped on request; "
                  f"{sum(len(b.keys) for b in pending)} points left unscheduled",
                  flush=True)

    def write_status(self, phase: str) -> None:
        status = {
            "phase": phase,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "elapsed_s": time.time() - self.start_time,
            "dispatch_deadline_in_s": self.dispatch_deadline - time.time(),
            "completed_points_this_run": self.completed_points,
            "deferred_points": self.deferred,
            "failures": len(self.failures),
            "inflation_p90": self.cost.inflation_p90(),
            "workers": self.args.workers,
        }
        path = os.path.join(self.store.reports_dir, "status.json")
        with open(path + ".tmp", "w") as fh:
            json.dump(status, fh, indent=2)
        os.replace(path + ".tmp", path)

    def write_failure_report(self) -> None:
        path = os.path.join(self.store.reports_dir, "failures.json")
        with open(path + ".tmp", "w") as fh:
            json.dump(self.failures, fh, indent=2)
        os.replace(path + ".tmp", path)
