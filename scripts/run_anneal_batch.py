"""Batch runner for the Anneal I/II protocols in main.tex."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ryd_gate.protocols.lattice_dynamics import TFIMAnnealProtocol
from tn_common import create_tn_lattice_spec, simulate_tn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run main.tex Anneal I/II benchmarks.")
    parser.add_argument("--L", type=int, default=10, help="square lattice size")
    parser.add_argument("--Lx", type=int, default=None, help="override lattice rows")
    parser.add_argument("--Ly", type=int, default=None, help="override lattice columns")
    parser.add_argument(
        "--schedules",
        default="both",
        choices=("I", "II", "both"),
        help="which main.tex anneal schedule to run",
    )
    parser.add_argument(
        "--backend",
        default="2dtn",
        choices=("tenpy", "mps", "itensors", "gputn", "2dtn", "ttn", "nqs"),
        help="simulation backend",
    )
    parser.add_argument(
        "--interaction-mode",
        default="nn",
        choices=("nn", "nnn"),
        help="use nn for the main.tex TFIM, nnn for Rydberg-tail tests",
    )
    parser.add_argument("--V-nn", type=float, default=4.0, help="Rydberg V_nn; V_nn=4 gives TFIM J=1")
    parser.add_argument("--dt", type=float, default=0.01, help="time step in units of 1/J")
    parser.add_argument("--t-eval-interval", type=float, default=0.1, help="observable sampling interval")
    parser.add_argument("--observables", default="sigma_z,czz_centerline", help="comma-separated observables")
    parser.add_argument("--chi-max", type=int, default=256, help="generic MPS/TN bond-dimension cap")
    parser.add_argument("--chi-2d", type=int, default=40, help="2D-TN evolution bond dimension")
    parser.add_argument("--chi-2d-prime", type=int, default=24, help="2D-TN measurement pre-truncation dimension")
    parser.add_argument("--svd-min", type=float, default=1e-10, help="SVD cutoff")
    parser.add_argument(
        "--measurement-alg",
        default="boundarymps",
        choices=("bp", "boundarymps", "exact"),
        help="2D-TN observable contraction algorithm",
    )
    parser.add_argument("--measurement-bond-dim", type=int, default=32, help="BoundaryMPS bond dimension")
    parser.add_argument("--normalize-tensors", action="store_true", help="normalize tensors inside TNQS updates")
    parser.add_argument("--use-cuda", action="store_true", help="request CUDA in compatible Julia backends")
    parser.add_argument("--julia-cmd", default="julia", help="Julia executable for backend=itensors/2dtn")
    parser.add_argument("--julia-project", default=None, help="Julia project dir for backend=itensors/2dtn")
    parser.add_argument("--julia-script", default=None, help="Julia kernel script for backend=itensors/2dtn")
    parser.add_argument("--julia-timeout", type=float, default=None, help="Julia subprocess timeout")
    parser.add_argument("--julia-no-bashrc", action="store_true", help="do not fallback to source ~/.bashrc")
    parser.add_argument("--julia-threads", type=int, default=None, help="JULIA_NUM_THREADS")
    parser.add_argument("--anneal-i-hx-peak", type=float, default=3.5, help="Anneal I peak h_x/J")
    parser.add_argument("--anneal-i-hz", type=float, default=0.0, help="Anneal I constant h_z/J")
    parser.add_argument("--anneal-i-t-rise", type=float, default=1.5, help="Anneal I t_rise J")
    parser.add_argument("--anneal-i-t-sweep", type=float, default=1.5, help="Anneal I t_sweep J")
    parser.add_argument(
        "--anneal-i-t-fall-values",
        default="3.0",
        help="comma-separated Anneal I t_fall J values",
    )
    parser.add_argument("--anneal-ii-hx-peak", type=float, default=0.5, help="Anneal II peak h_x/J")
    parser.add_argument("--anneal-ii-hz-initial", type=float, default=-8.0, help="Anneal II initial h_z/J")
    parser.add_argument("--anneal-ii-hz-final", type=float, default=0.0, help="Anneal II final h_z/J")
    parser.add_argument("--anneal-ii-t-rise", type=float, default=1.5, help="Anneal II t_rise J")
    parser.add_argument("--anneal-ii-t-fall", type=float, default=1.5, help="Anneal II t_fall J")
    parser.add_argument(
        "--anneal-ii-t-sweep-values",
        default="3.0",
        help="comma-separated Anneal II t_sweep J values",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("scripts/results/anneal_batch"),
        help="directory for .npz benchmark outputs",
    )
    parser.add_argument("--dry-run", action="store_true", help="print run configs without invoking backends")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    Lx = args.L if args.Lx is None else args.Lx
    Ly = args.L if args.Ly is None else args.Ly
    observables = _parse_csv(args.observables)
    specs = _anneal_specs(args)
    backend_options = _backend_options(args)

    runs = []
    for spec_data in specs:
        protocol = spec_data["protocol"]
        t_eval = _t_eval(protocol.t_gate, args.t_eval_interval)
        output = args.output_dir / _output_name(
            schedule=spec_data["schedule"],
            Lx=Lx,
            Ly=Ly,
            backend=args.backend,
            sweep_value=spec_data["sweep_value"],
        )
        run_config = {
            "schedule": spec_data["schedule"],
            "Lx": Lx,
            "Ly": Ly,
            "backend": args.backend,
            "interaction_mode": args.interaction_mode,
            "protocol": spec_data["metadata"],
            "t_eval_steps": int(len(t_eval)),
            "observables": observables,
            "backend_options": _jsonable_options(backend_options),
            "output": str(output),
        }
        runs.append(run_config)
        if args.dry_run:
            continue

        tn_spec = create_tn_lattice_spec(
            Lx,
            Ly,
            V_nn=args.V_nn,
            Omega=2.0 * spec_data["metadata"]["hx_peak"],
            level_structure="1r",
            interaction_mode=args.interaction_mode,
            ordering="snake",
        )
        result = simulate_tn(
            tn_spec,
            protocol,
            [],
            initial_state="all_ground",
            backend=args.backend,
            t_eval=t_eval,
            observables=observables,
            backend_options=backend_options,
        )
        _save_result(output, result, run_config)

    print(json.dumps({"runs": runs}, indent=2, sort_keys=True))


def _anneal_specs(args: argparse.Namespace) -> list[dict]:
    specs = []
    if args.schedules in {"I", "both"}:
        for t_fall in _parse_float_csv(args.anneal_i_t_fall_values):
            protocol = TFIMAnnealProtocol(
                hx_peak=args.anneal_i_hx_peak,
                hx_initial=0.0,
                hx_final=0.0,
                hz_initial=args.anneal_i_hz,
                hz_final=args.anneal_i_hz,
                t_rise=args.anneal_i_t_rise,
                t_sweep=args.anneal_i_t_sweep,
                t_fall=t_fall,
            )
            specs.append(
                {
                    "schedule": "I",
                    "protocol": protocol,
                    "sweep_value": t_fall,
                    "metadata": {
                        "hx_peak": args.anneal_i_hx_peak,
                        "hz_initial": args.anneal_i_hz,
                        "hz_final": args.anneal_i_hz,
                        "t_rise": args.anneal_i_t_rise,
                        "t_sweep": args.anneal_i_t_sweep,
                        "t_fall": t_fall,
                        "t_gate": protocol.t_gate,
                    },
                }
            )
    if args.schedules in {"II", "both"}:
        for t_sweep in _parse_float_csv(args.anneal_ii_t_sweep_values):
            protocol = TFIMAnnealProtocol(
                hx_peak=args.anneal_ii_hx_peak,
                hx_initial=0.0,
                hx_final=0.0,
                hz_initial=args.anneal_ii_hz_initial,
                hz_final=args.anneal_ii_hz_final,
                t_rise=args.anneal_ii_t_rise,
                t_sweep=t_sweep,
                t_fall=args.anneal_ii_t_fall,
            )
            specs.append(
                {
                    "schedule": "II",
                    "protocol": protocol,
                    "sweep_value": t_sweep,
                    "metadata": {
                        "hx_peak": args.anneal_ii_hx_peak,
                        "hz_initial": args.anneal_ii_hz_initial,
                        "hz_final": args.anneal_ii_hz_final,
                        "t_rise": args.anneal_ii_t_rise,
                        "t_sweep": t_sweep,
                        "t_fall": args.anneal_ii_t_fall,
                        "t_gate": protocol.t_gate,
                    },
                }
            )
    return specs


def _backend_options(args: argparse.Namespace) -> dict:
    options: dict[str, object] = {
        "chi_max": args.chi_max,
        "dt": args.dt,
        "svd_min": args.svd_min,
    }
    if args.backend in {"itensors", "2dtn"}:
        options.update(
            {
                "julia_cmd": args.julia_cmd,
                "source_bashrc": not args.julia_no_bashrc,
                "use_cuda": bool(args.use_cuda),
            }
        )
        if args.julia_project:
            options["project_dir"] = args.julia_project
        if args.julia_script:
            options["script_path"] = args.julia_script
        if args.julia_timeout is not None:
            options["timeout"] = args.julia_timeout
        if args.julia_threads is not None:
            options["threads"] = args.julia_threads
    if args.backend == "2dtn":
        options.update(
            {
                "chi_2d": args.chi_2d,
                "chi_2d_prime": args.chi_2d_prime,
                "measurement_alg": args.measurement_alg,
                "measurement_bond_dim": args.measurement_bond_dim,
                "normalize_tensors": bool(args.normalize_tensors),
            }
        )
    return options


def _t_eval(t_gate: float, interval: float) -> np.ndarray:
    if interval <= 0:
        raise ValueError("t_eval interval must be positive.")
    n_steps = int(np.floor(t_gate / interval))
    values = np.arange(n_steps + 1, dtype=float) * interval
    if not np.isclose(values[-1], t_gate):
        values = np.append(values, float(t_gate))
    return values


def _save_result(path: Path, result, run_config: dict) -> None:
    arrays = {}
    if result.times is not None:
        arrays["times"] = np.asarray(result.times)
    for name, value in (result.metadata.get("obs") or {}).items():
        arrays[f"obs_{name}"] = np.asarray(value)
    if "final_sigma_z" in result.metadata:
        arrays["final_sigma_z"] = np.asarray(result.metadata["final_sigma_z"])
    if "truncation_error" in result.metadata:
        arrays["truncation_error"] = np.asarray(result.metadata["truncation_error"])
    metadata = {
        "run_config": run_config,
        "result_metadata": _metadata_for_json(result.metadata),
    }
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def _output_name(*, schedule: str, Lx: int, Ly: int, backend: str, sweep_value: float) -> str:
    tag = "tfall" if schedule == "I" else "tsweep"
    value = str(float(sweep_value)).replace(".", "p")
    return f"anneal_{schedule}_L{Lx}x{Ly}_{tag}{value}_{backend}.npz"


def _parse_csv(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def _parse_float_csv(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one numeric value.")
    return values


def _jsonable_options(options: dict) -> dict:
    return {
        key: value
        for key, value in options.items()
        if isinstance(value, (str, int, float, bool)) or value is None
    }


def _metadata_for_json(metadata: dict) -> dict:
    jsonable = {}
    for key, value in metadata.items():
        if key == "obs":
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            jsonable[key] = value
    return jsonable


if __name__ == "__main__":
    main()
