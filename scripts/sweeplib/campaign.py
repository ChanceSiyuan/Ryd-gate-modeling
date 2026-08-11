"""Shared orchestration and reporting for the two max-leakage campaigns."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np

from .plotting import credibility_floor
from .runner import Batch, CostModel, Runner, group_batches
from .solver import BatchResult
from .store import (
    PointRecord,
    _atomic_savez,
    audit_pairs,
    best_records,
    completed_keys,
)


@dataclass(frozen=True)
class CampaignHooks:
    """The small model-specific surface used by shared campaign commands."""

    setup_run: Callable
    parse_panels: Callable
    all_keys: Callable[[int], list]
    all_panels: Callable[[], list]
    pilot_keys: Callable[[], list]
    stage_pilot: Callable
    ensure_scatter_gate: Callable
    gammas: Callable[[object, dict], dict]
    export_store: Callable
    write_summary_reports: Callable
    level_sizes: Sequence[int]
    level_from_size: dict[int, int]
    row_description: Callable[[object], str]


def parse_panels(args, n_rows: int, n_times: int) -> set[tuple[int, int]] | None:
    """Parse an optional ``--panels "row,time;row,time"`` restriction."""
    spec = getattr(args, "panels", None)
    if not spec:
        return None
    panels = set()
    for part in spec.split(";"):
        row, time_idx = (int(v) for v in part.split(","))
        if not (0 <= row < n_rows and 0 <= time_idx < n_times):
            raise SystemExit(f"--panels: ({row},{time_idx}) out of range")
        panels.add((row, time_idx))
    return panels


def filter_panels(keys: Iterable, panels: set[tuple[int, int]] | None) -> list:
    return [key for key in keys if panels is None or key.panel in panels]


def feed_cost_model(cost: CostModel, records: Sequence[PointRecord]) -> None:
    for record in records:
        if record.status == "ok" and record.tier == "production":
            cost.observe(record.key.panel, record.runtime_s)


def effective_batch_size(store, args) -> int:
    """Return the requested size only after the packing gate has passed."""
    if args.batch_size <= 1:
        return 1
    path = os.path.join(store.reports_dir, "pilot.json")
    if os.path.exists(path):
        try:
            with open(path) as fh:
                gate = json.load(fh).get("packing_gate", {})
            if gate.get("enabled"):
                return args.batch_size
        except (OSError, json.JSONDecodeError):
            pass
    print("[run] packing gate not passed/recorded; using one point per solve",
          flush=True)
    return 1


def run_level(
    runner: Runner,
    level: int,
    done: set,
    batch_size: int,
    rerun_failures: bool,
    failed: set,
    all_keys: Callable[[int], list],
    level_sizes: Sequence[int],
    panels: set[tuple[int, int]] | None = None,
) -> None:
    missing = [
        key for key in filter_panels(all_keys(level), panels)
        if key not in done and (rerun_failures or key not in failed)
    ]
    if not missing:
        print(f"[level {level_sizes[level]}] complete", flush=True)
        return
    runner.run_batches(
        group_batches(missing, batch_size), f"level-{level_sizes[level]}")


def run_packing_gate(
    runner: Runner,
    done: set,
    *,
    make_key: Callable,
    panel: tuple[int, int],
    coords: Sequence,
    state_tol: float,
    leakage_tol: float,
) -> dict:
    """Compare packed and isolated solves at production and audit tolerances."""
    keys = [make_key(*panel, omega, sweep) for omega, sweep in coords]
    isolated: dict[tuple[object, str], BatchResult] = {}

    try:
        futures = {}
        for tier in ("production", "audit"):
            for key in keys:
                batch = Batch(keys=[key], tier=tier)
                futures[runner._submit(batch)] = batch
        packed = {
            tier: runner._submit(Batch(keys=keys, tier=tier))
            for tier in ("production", "audit")
        }

        for future, batch in futures.items():
            out = future.result()
            if not out["ok"]:
                reason = out.get("message", out.get("reason"))
                return {"enabled": False,
                        "reason": f"isolated gate run failed: {reason}"}
            key = batch.keys[0]
            if not (key in done and batch.tier == "production"):
                runner._write_success(batch, out)
            isolated[(key, batch.tier)] = out["result"]
        packed_out = {tier: future.result() for tier, future in packed.items()}
    except Exception as exc:
        return {"enabled": False, "reason": f"gate execution error: {exc}"[:240]}

    deviations = {}
    for tier, out in packed_out.items():
        if not out["ok"]:
            return {"enabled": False,
                    "reason": f"packed {tier} gate run failed: {out.get('message')}"}
        result: BatchResult = out["result"]
        deviations[tier] = {
            "max_state_dev": max(
                float(np.max(np.abs(
                    result.psi_final[i] - isolated[(key, tier)].psi_final[0])))
                for i, key in enumerate(keys)),
            "max_leakage_dev": max(
                float(np.max(np.abs(
                    result.leakage[i] - isolated[(key, tier)].leakage[0])))
                for i, key in enumerate(keys)),
        }
    enabled = all(
        values["max_state_dev"] < state_tol
        and values["max_leakage_dev"] < leakage_tol
        for values in deviations.values()
    )
    return {
        "enabled": bool(enabled),
        "panel": panel,
        "n_points": len(keys),
        "tiers": deviations,
        "max_state_dev": max(v["max_state_dev"] for v in deviations.values()),
        "max_leakage_dev": max(v["max_leakage_dev"] for v in deviations.values()),
        "state_tol": state_tol,
        "leak_tol": leakage_tol,
    }


def stage_pilot(
    runner: Runner,
    panels: set[tuple[int, int]] | None,
    *,
    pilot_keys: Callable[[], list],
    packing_gate: Callable[[Runner, set], dict],
    packing_gate_panel: tuple[int, int],
    all_keys: Callable[[int], list],
    level_sizes: Sequence[int],
    report_extra: Callable[[Runner], dict] | None = None,
) -> dict:
    """Run reusable pilot nodes, initial audits and the packing gate."""
    records = runner.store.load_records(runner.manifest, include_states=False)
    done = completed_keys(records)
    audit_done = completed_keys(records, "audit")
    keys = filter_panels(pilot_keys(), panels)
    missing = [key for key in keys if key not in done]
    if missing:
        runner.run_batches(
            [Batch(keys=[key], save_traj=True) for key in missing], "pilot")
        records = runner.store.load_records(runner.manifest, include_states=False)
        done = completed_keys(records)

    centers = [key for key in keys if (key.om_num, key.om_den) == (3, 2)]
    audit_keys = [key for key in centers[::10]
                  if key not in audit_done and key in done]
    if audit_keys:
        runner.run_batches(
            [Batch(keys=[key], tier="audit", save_traj=True)
             for key in audit_keys],
            "pilot-audit")

    gate = {"enabled": False, "reason": "batching disabled (--batch-size 1)"}
    pilot_path = os.path.join(runner.store.reports_dir, "pilot.json")
    prior_gate = None
    if os.path.exists(pilot_path):
        try:
            with open(pilot_path) as fh:
                prior_gate = json.load(fh).get("packing_gate")
        except (OSError, json.JSONDecodeError):
            pass
    if runner.args.batch_size > 1:
        verification_path = os.path.join(
            runner.store.reports_dir, "verification.json")
        with open(verification_path) as fh:
            norm_ok = json.load(fh)["error_norm_verified"]
        if not norm_ok:
            gate = {"enabled": False,
                    "reason": "SciPy error-norm seam unverified; one point per solve"}
        elif prior_gate and "max_state_dev" in prior_gate:
            gate = prior_gate
        elif panels is not None and packing_gate_panel not in panels:
            gate = {"enabled": False,
                    "reason": f"--panels excludes gate panel {packing_gate_panel}"}
        elif not runner.stop_requested:
            gate = packing_gate(runner, done)

    records = runner.store.load_records(runner.manifest, include_states=False)
    per_panel = {
        f"{panel[0]},{panel[1]}": float(np.median(values))
        for panel, values in sorted(runner.cost.samples.items())
    }
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "n_pilot_points": len(keys),
        "packing_gate": gate,
        "per_panel_median_point_s": per_panel,
        "inflation_p90": runner.cost.inflation_p90(),
        "audit_pairs": len(audit_pairs(records)),
        "eta_hours_full_levels": {
            str(size): runner.cost.eta_seconds(all_keys(level), runner.args.workers)
            / 3600.0
            for level, size in enumerate(level_sizes)
        },
        **(report_extra(runner) if report_extra is not None else {}),
    }
    with open(pilot_path + ".tmp", "w") as fh:
        json.dump(report, fh, indent=2)
    os.replace(pilot_path + ".tmp", pilot_path)
    print(f"[pilot] packing gate: {gate}", flush=True)
    print(f"[pilot] full-level ETA estimates (h): "
          f"{report['eta_hours_full_levels']}", flush=True)
    return report


def audit_targets(args, records: Sequence[PointRecord]) -> list:
    """Choose explicit, candidate-ranked or random production audit points."""
    production = sorted(completed_keys(records, "production"))
    audited = completed_keys(records, "audit")
    if args.audit_point:
        by_id = {key.id(): key for key in production}
        if args.audit_point not in by_id:
            raise SystemExit(
                f"--audit-point {args.audit_point}: no successful production "
                "record with that id")
        return [by_id[args.audit_point]]
    if args.candidates:
        best = best_records(records)
        ranked = sorted(
            (record.max_leakage, key) for key, record in best.items()
            if record.tier == "production")
        return [key for _, key in ranked[:args.candidates] if key not in audited]

    pool = [key for key in production if key not in audited]
    n_points = min(args.n_points, len(pool))
    if not n_points:
        return []
    rng = np.random.default_rng(args.seed)
    return [pool[i] for i in sorted(
        rng.choice(len(pool), size=n_points, replace=False))]


def _setup_parts(args, hooks: CampaignHooks):
    setup = hooks.setup_run(args)
    return setup[0], setup[1], setup[2], setup[-1]


def pilot_command(args, hooks: CampaignHooks) -> None:
    store, manifest, cfg, _ = _setup_parts(args, hooks)
    cost = CostModel(cfg)
    feed_cost_model(cost, store.load_records(manifest, include_states=False))
    runner = Runner(store, manifest, cfg, args, cost)
    try:
        hooks.stage_pilot(runner, hooks.parse_panels(args))
    except KeyboardInterrupt:
        print("[pilot] hard abort", flush=True)
    finally:
        runner.write_failure_report()
        runner.write_status("pilot-aborted" if runner.aborted else "pilot-done")
        runner.shutdown()
    if not runner.aborted:
        hooks.export_store(store)


def run_command(args, hooks: CampaignHooks) -> None:
    store, manifest, cfg, checks = _setup_parts(args, hooks)
    panels = hooks.parse_panels(args)
    records = store.load_records(manifest, include_states=False)
    done = completed_keys(records)
    failed = {record.key for record in records if record.status != "ok"} - done
    cost = CostModel(cfg)
    feed_cost_model(cost, records)
    max_level = hooks.level_from_size[int(args.target_level)]

    if args.dry_run:
        n_panels = len(panels) if panels is not None else len(hooks.all_panels())
        print(f"panels: {n_panels}  ({hooks.row_description(cfg)})")
        for level, size in enumerate(hooks.level_sizes):
            keys = filter_panels(hooks.all_keys(level), panels)
            missing = [key for key in keys if key not in done]
            eta_hours = cost.eta_seconds(missing, args.workers) / 3600.0
            eta_text = f"{eta_hours:8.2f} h" if cost.samples else "unmeasured"
            print(f"level {size:>2}x{size:<2}: "
                  f"{len(keys) - len(missing):>6}/{len(keys):>6} done, "
                  f"{len(missing):>6} missing, predicted ETA {eta_text} "
                  f"@ {args.workers} workers")
        keys = filter_panels(hooks.pilot_keys(), panels)
        print(f"pilot: {sum(1 for key in keys if key in done)}/{len(keys)} done")
        print(f"failed points on record: {len(failed)}" +
              (" (will retry: --rerun-failures)" if args.rerun_failures else ""))
        return

    runner = Runner(store, manifest, cfg, args, cost)
    runner.gammas = hooks.gammas(cfg, checks)
    runner.scatter_done = {
        record["key"] for record in store.load_scatter_records(manifest)
        if record["status"] == "ok"
    }
    try:
        hooks.stage_pilot(runner, panels)
        gate = hooks.ensure_scatter_gate(runner, store)
        if gate.get("ok"):
            runner.write_both_series = True
            print("[run] single-pass scatter enabled (equivalence gate ok)",
                  flush=True)
        else:
            print(f"[run] scatter-equivalence gate not ok "
                  f"({gate.get('reason')}); writing the coherent series only",
                  flush=True)
        batch_size = effective_batch_size(store, args)
        print(f"[run] effective batch size: {batch_size}", flush=True)

        for level in range(len(hooks.level_sizes)):
            if runner.stop_requested or level > max_level:
                break
            records = store.load_records(manifest, include_states=False)
            done = completed_keys(records)
            run_level(runner, level, done, batch_size, args.rerun_failures,
                      failed, hooks.all_keys, hooks.level_sizes, panels)
    except KeyboardInterrupt:
        print("[run] hard abort; in-flight batches were discarded (their points "
              "resume on the next run)", flush=True)
    finally:
        runner.write_failure_report()
        runner.write_status("run-aborted" if runner.aborted else "run-done")
        runner.shutdown()

    if not runner.aborted:
        hooks.export_store(store)
        hooks.write_summary_reports(store)
        print("[run] done; exports refreshed", flush=True)


def audit_command(args, hooks: CampaignHooks) -> None:
    store, manifest, cfg, _ = _setup_parts(args, hooks)
    records = store.load_records(manifest, include_states=False)
    targets = audit_targets(args, records)
    if not targets:
        print("[audit] nothing to audit")
        return
    cost = CostModel(cfg)
    feed_cost_model(cost, records)
    runner = Runner(store, manifest, cfg, args, cost)
    try:
        runner.run_batches(
            [Batch(keys=[key], tier="audit", save_traj=True) for key in targets],
            "audit")
    except KeyboardInterrupt:
        print("[audit] hard abort", flush=True)
    finally:
        runner.write_failure_report()
        runner.shutdown()
    if not runner.aborted:
        hooks.write_summary_reports(store)


def scatter_command(args, hooks: CampaignHooks) -> None:
    store, manifest, cfg, checks = _setup_parts(args, hooks)
    level = hooks.level_from_size[int(args.level)]
    panels = hooks.parse_panels(args)
    done = {record["key"] for record in store.load_scatter_records(manifest)
            if record["status"] == "ok"}
    missing = [key for key in filter_panels(hooks.all_keys(level), panels)
               if key not in done]
    print(f"[scatter] level {args.level}: {len(missing)} points to compute "
          f"({len(done)} already stored)", flush=True)
    if not missing:
        return

    cost = CostModel(cfg)
    feed_cost_model(cost, store.load_records(manifest, include_states=False))
    runner = Runner(store, manifest, cfg, args, cost)
    runner.gammas = hooks.gammas(cfg, checks)
    gate_failed = False
    try:
        gate = hooks.ensure_scatter_gate(runner, store)
        print(f"[scatter] trajectory-equivalence gate: {gate}", flush=True)
        if not gate.get("ok"):
            gate_failed = True
            raise SystemExit("[scatter] equivalence gate failed; not running")
        batches = group_batches(missing, effective_batch_size(store, args))
        for batch in batches:
            batch.mode = "scatter"
        runner.run_batches(batches, f"scatter-{args.level}")
    except KeyboardInterrupt:
        print("[scatter] hard abort", flush=True)
    finally:
        if gate_failed:
            runner.write_status(f"scatter-{args.level}-gate-failed")
        else:
            runner.write_failure_report()
            runner.write_status(
                f"scatter-{args.level}-aborted" if runner.aborted
                else f"scatter-{args.level}-done")
        runner.shutdown()


def ensure_scatter_gate(runner: Runner, store, gate_fn: Callable) -> dict:
    """Run and persist a store's scatter-equivalence gate once."""
    path = os.path.join(store.reports_dir, "scatter_gate.json")
    if os.path.exists(path):
        try:
            with open(path) as fh:
                previous = json.load(fh)
            if previous.get("ok"):
                return previous
        except (OSError, json.JSONDecodeError):
            pass
    gate = gate_fn(runner, store)
    with open(path + ".tmp", "w") as fh:
        json.dump(gate, fh, indent=2)
    os.replace(path + ".tmp", path)
    return gate


def export_store(store, config_type, records: list[PointRecord] | None = None):
    """Regenerate the schema-compatible merged NPZ and points CSV exports."""
    manifest = store.load_manifest()
    if manifest is None:
        raise RuntimeError(f"no manifest under {store.root}")
    if records is None:
        records = store.load_records(manifest)
    best = best_records(records)
    keys = sorted(best)
    rows = [best[key] for key in keys]
    cfg = config_type(**{
        key: tuple(value) if isinstance(value, list) else value
        for key, value in manifest["physics"].items()
        if key != "schema_version"
    })
    descriptors = store.provenance.descriptor(cfg, keys)

    store.ensure_dirs()
    merged_path = os.path.join(store.exports_dir, "latest_merged.npz")
    n_rows = len(rows)
    payload = {
        "schema_version": np.int64(store.provenance.schema_version),
        "scan_uuid": str(manifest["scan_uuid"]),
        **store.keys_to_arrays(keys),
        **descriptors,
        "max_leakage": np.asarray([row.max_leakage for row in rows]),
        "leakage": np.asarray([row.leakage for row in rows]).reshape(n_rows, 4),
        "worst_input": np.asarray([row.worst_input for row in rows], dtype="U2"),
        "return_prob": np.asarray([row.return_prob for row in rows]).reshape(n_rows, 4),
        "norm_err_max": np.asarray([
            float(np.max(row.norm_err)) for row in rows]),
        "tier": np.asarray([row.tier for row in rows], dtype="U10"),
        "rtol": np.asarray([row.rtol for row in rows]),
        "atol": np.asarray([row.atol for row in rows]),
        "psi_final": (
            np.asarray([row.psi_final for row in rows]).reshape(n_rows, 4, -1)
            if n_rows else np.zeros(
                (0, 4, store.provenance.default_dim), dtype=np.complex128)
        ),
    }
    _atomic_savez(merged_path, **payload)

    csv_path = os.path.join(store.exports_dir, "points.csv")
    descriptor_names = list(descriptors)
    columns = [
        "point_id", *descriptor_names,
        "max_leakage", "leak_00", "leak_01", "leak_10", "leak_11",
        "worst_input", "min_return_prob", "norm_err_max", "tier", "rtol",
        "atol", "nfev", "runtime_s", "batch_id",
    ]
    tmp = csv_path + ".tmp"
    with open(tmp, "w") as fh:
        fh.write(",".join(columns) + "\n")
        for index, (key, row) in enumerate(zip(keys, rows)):
            descriptor_values = [descriptors[name][index]
                                 for name in descriptor_names]
            values = (
                key.id(), *descriptor_values, repr(row.max_leakage),
                *(repr(value) for value in row.leakage), row.worst_input,
                repr(float(np.min(row.return_prob))),
                repr(float(np.max(row.norm_err))), row.tier, row.rtol, row.atol,
                row.nfev, f"{row.runtime_s:.3f}", row.batch_id,
            )
            fh.write(",".join(str(value) for value in values) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, csv_path)
    return merged_path, csv_path


def write_summary_reports(store, omega_field: str) -> None:
    """Regenerate audit-summary and per-panel candidate reports."""
    manifest = store.load_manifest()
    records = store.load_records(manifest, include_states=False)
    pairs = audit_pairs(records)
    _, floor_info = credibility_floor(records)
    diffs = np.abs([production - audit for _, production, audit in pairs]) \
        if pairs else np.zeros(0)
    audit_summary = {
        "n_pairs": len(pairs),
        "max_abs_leakage_diff": float(diffs.max()) if diffs.size else None,
        "p95_abs_leakage_diff": (
            float(np.percentile(diffs, 95)) if diffs.size else None),
        "credibility_floor": floor_info,
        "worst_pairs": [
            {"point_id": key.id(), "L_production": production,
             "L_audit": audit, "abs_diff": abs(production - audit)}
            for key, production, audit in sorted(
                pairs, key=lambda item: -abs(item[1] - item[2]))[:10]
        ],
    }
    path = os.path.join(store.reports_dir, "audit_summary.json")
    with open(path + ".tmp", "w") as fh:
        json.dump(audit_summary, fh, indent=2)
    os.replace(path + ".tmp", path)

    candidates: dict[str, dict] = {}
    for key, record in best_records(records).items():
        panel = f"{key.panel[0]},{key.panel[1]}"
        if (panel not in candidates
                or record.max_leakage < candidates[panel]["max_leakage"]):
            candidates[panel] = {
                "point_id": key.id(),
                "max_leakage": record.max_leakage,
                omega_field: float(key.omega_mhz()),
                "dsweep_mhz": float(key.dsweep_mhz()),
                "worst_input": record.worst_input,
                "tier": record.tier,
            }
    path = os.path.join(store.reports_dir, "candidates.json")
    with open(path + ".tmp", "w") as fh:
        json.dump({"note": "per-panel minima over exact ODE nodes only",
                   "panels": candidates}, fh, indent=2, sort_keys=True)
    os.replace(path + ".tmp", path)


def print_status(
    store,
    *,
    all_keys: Callable[[int], list],
    pilot_keys: Callable[[], list],
    level_sizes: Sequence[int],
    header_extra: Callable[[dict], str] | None = None,
) -> None:
    """Print the common manifest, completion, runtime and last-run status."""
    manifest = store.load_manifest()
    if manifest is None:
        print(f"no manifest under {store.root} (scan not initialized)")
        return
    records = store.load_records(manifest, include_states=False)
    done = completed_keys(records)
    failed = {record.key for record in records if record.status != "ok"} - done
    print(f"scan {manifest['scan_uuid'][:12]}  created {manifest['created_at']}")
    prefix = header_extra(manifest) if header_extra is not None else ""
    print(f"{prefix}git {manifest['git']['commit'][:10]}"
          f"{' (dirty)' if manifest['git']['dirty'] else ''}")
    for level, size in enumerate(level_sizes):
        keys = all_keys(level)
        n_done = sum(1 for key in keys if key in done)
        print(f"level {size:>2}x{size:<2}: {n_done:>6}/{len(keys):>6} nodes complete")
    keys = pilot_keys()
    print(f"pilot: {sum(1 for key in keys if key in done)}/{len(keys)} done")
    n_audit = len(completed_keys(records, "audit"))
    pairs = audit_pairs(records)
    print(f"records: {len(records)} rows, {len(done)} unique ok points, "
          f"{n_audit} audit points, {len(pairs)} audit pairs, {len(failed)} failed")
    production = [record for record in records
                  if record.status == "ok" and record.tier == "production"]
    if production:
        runtimes = np.asarray([record.runtime_s for record in production])
        print(f"per-point runtime (production): median {np.median(runtimes):.1f} s, "
              f"P90 {np.percentile(runtimes, 90):.1f} s, "
              f"total {runtimes.sum() / 3600:.2f} core-h")
    status_path = os.path.join(store.reports_dir, "status.json")
    if os.path.exists(status_path):
        with open(status_path) as fh:
            print(f"last run status: {json.load(fh)}")
