"""Execute the CPU-gated tutorial notebooks (Stage 7 acceptance runner).

Runs each notebook in EXECUTE with ``jupyter nbconvert --to notebook
--execute`` under a per-notebook timeout (in a temp copy; the checked-in
files are not modified). Notebooks in SKIP are import-migrated but not
execution-gated, each for a named reason (optional GPU/yastn backends or
long-running benchmark sweeps).

Usage:
    OMP_NUM_THREADS=1 uv run python docs/_scripts/run_notebooks.py [name.ipynb ...]
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

NOTEBOOK_DIR = Path(__file__).resolve().parents[2] / "scripts" / "notebooks"
TIMEOUT_S = 1800

EXECUTE = [
    "cz_gate_validation_and_errors.ipynb",
    "02_ac_stark_local_addressing.ipynb",
]

SKIP = {
    "01r_saffman_double_arp_exact.ipynb": "long-running exact ARP scans (research benchmark)",
    "01r_lattice_dynamics.ipynb": "long-running MPS/TDVP sweeps (research benchmark)",
    "03_lattice_dynamics_and_annealing.ipynb": "long-running MPS/TDVP sweeps (research benchmark)",
    "plus_state_preparation.ipynb": "long-running MPS state-prep sweeps (research benchmark)",
    "run_quench_benchmark.ipynb": "long-running quench benchmark",
    "01r_plus_quench_benchmark.ipynb": "long-running quench benchmark",
    "01r_tfim_critical_field.ipynb": "long-running DMRG critical-field scan",
    "01r_yastn_peps_convergence.ipynb": "needs the yastn / tn-2d extra",
}


def run_notebook(path: Path) -> bool:
    print(f"== executing {path.name} (timeout {TIMEOUT_S}s)")
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [
            sys.executable, "-m", "jupyter", "nbconvert",
            "--to", "notebook", "--execute",
            f"--ExecutePreprocessor.timeout={TIMEOUT_S}",
            "--output-dir", tmp,
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
        if name in SKIP and name not in sys.argv[1:]:
            print(f"== skipping {name}: {SKIP[name]}")
            continue
        path = NOTEBOOK_DIR / name
        if not path.exists():
            print(f"== missing {name}", file=sys.stderr)
            failures.append(name)
            continue
        if not run_notebook(path):
            failures.append(name)
    for name in sorted(set(SKIP) - set(targets)):
        print(f"-- not execution-gated: {name} ({SKIP[name]})")
    if failures:
        print(f"FAILED: {failures}", file=sys.stderr)
        return 1
    print("all gated notebooks executed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
