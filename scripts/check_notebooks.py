"""Execute the CPU-gated research notebooks without modifying checked-in files.

Each selected notebook is executed through ``jupyter nbconvert`` in a temporary
output directory. Long-running benchmark and optional-GPU notebooks remain
listed with explicit reasons instead of being silently skipped.

Usage:
    OMP_NUM_THREADS=1 uv run python scripts/check_notebooks.py [name.ipynb ...]
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

NOTEBOOK_DIR = Path(__file__).resolve().parent / "notebooks"
TIMEOUT_S = 1800

EXECUTE = [
    "02_ac_stark_addressing.ipynb",
    "03_lattice_dynamics_annealing.ipynb",
]

NON_GATED = {
    "01_cz_gate.ipynb": "long-running 7-level exact_ode CZ scans",
    "04_quench_and_state_prep.ipynb": "long-running exact/MPS/PEPS benchmarks",
    "05_tn_and_error_budget.ipynb": "long-running PEPS/DMRG/error-budget scans",
    "error_buget.ipynb": "long-running parallel exact_ode CZ error-budget maps",
    "find_phase.ipynb": "long-running exact_ode effective-theory phase study",
    "single_photon.ipynb": "long-running 297-nm versus two-photon comparison",
}


def run_notebook(path: Path) -> bool:
    print(f"== executing {path.name} (timeout {TIMEOUT_S}s)")
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            f"--ExecutePreprocessor.timeout={TIMEOUT_S}",
            "--output-dir",
            tmp,
            str(path),
        ]
        proc = subprocess.run(cmd, cwd=NOTEBOOK_DIR.parents[1])
    ok = proc.returncode == 0
    print(f"   {'OK' if ok else 'FAILED'}: {path.name}")
    return ok


def main() -> int:
    targets = sys.argv[1:] or EXECUTE
    failures = []
    for name in targets:
        path = NOTEBOOK_DIR / name
        if not path.exists():
            print(f"== missing {name}", file=sys.stderr)
            failures.append(name)
            continue
        if not run_notebook(path):
            failures.append(name)
    for name in sorted(set(NON_GATED) - set(targets)):
        print(f"-- not execution-gated: {name} ({NON_GATED[name]})")
    if failures:
        print(f"FAILED: {failures}", file=sys.stderr)
        return 1
    print("all gated notebooks executed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
