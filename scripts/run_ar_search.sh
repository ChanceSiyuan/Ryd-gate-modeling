#!/bin/bash
# Parallel multi-start AR-CZ search on "our" (one process per start, OMP pinned
# to 1 thread so independent numpy solves do not oversubscribe the cores).
# Usage: bash scripts/run_ar_search.sh [N_random_workers] [maxiter]
set -u
cd "$(dirname "$0")/.."
N=${1:-30}
MAXITER=${2:-250}
PY=.venv/bin/python
[ -x "$PY" ] || PY=$(command -v python3)

rm -f data/ar_opt_our_w*.json data/ar_opt_our_search_best.json /tmp/ar_w_*.log

# ARC builds its sqlite database on first use; 30 cold starts at once corrupt it
# (concurrent DDL race). Warm the DB once in a single process first -- after that
# ARC is parallel-safe (verified: 4 concurrent post-warmup workers, no race).
echo "warming ARC database (single process)..."
OMP_NUM_THREADS=1 "$PY" -c "
from ryd_gate import Register, RydbergSystem
from ryd_gate.analysis.gate_metrics import average_gate_infidelity
from ryd_gate.protocols.gate_cz import ARProtocol
s = RydbergSystem.from_lattice(Register.chain(2, spacing_um=3.0), 'rb87_7', param_set='our')
average_gate_infidelity(s, ARProtocol(), [0.86,0.39,0.99,0.19,-1.17,-0.008,1.67,0.29])
print('ARC warm')
"

# Curated worker: legacy + lukin-opt cross-seed + resume the 0.451 baseline.
OMP_NUM_THREADS=1 nice -n 10 "$PY" scripts/optimize_ar_cz.py our "$MAXITER" --resume \
    > /tmp/ar_w_curated.log 2>&1 &

# Random-restart workers: one full Nelder-Mead from a distinct random seed each.
# Stagger launches so the ARC-accessing system-construction phase (~25 s) of two
# workers rarely overlaps -- post-warmup ARC is safe at low concurrency but still
# races when ~28 processes construct the rb87_7 system at once.
STAGGER=${STAGGER:-15}
for i in $(seq 0 $((N-1))); do
    OMP_NUM_THREADS=1 nice -n 10 "$PY" scripts/optimize_ar_cz.py our "$MAXITER" \
        --no-curated --restarts 1 --seed "$i" --tag "w$i" \
        > "/tmp/ar_w_$i.log" 2>&1 &
    sleep "$STAGGER"
done

wait

# Merge all checkpoint files into the global best.
"$PY" - <<'EOF'
import json, glob, pathlib
best = None
for f in sorted(glob.glob("data/ar_opt_our*.json")):
    if f.endswith("search_best.json"):
        continue
    try:
        d = json.load(open(f))
    except Exception:
        continue
    bi = d.get("best_infidelity")
    if bi is not None and (best is None or bi < best["best_infidelity"]):
        best = {"best_infidelity": bi, "best_x": d.get("best_x"),
                "source": f, "best_seed": d.get("best_seed")}
pathlib.Path("data/ar_opt_our_search_best.json").write_text(json.dumps(best, indent=1))
print("GLOBAL BEST infidelity:", best["best_infidelity"])
print("source:", best["source"], "seed:", best["best_seed"])
print("best_x:", best["best_x"])
EOF
echo "AR_SEARCH_DONE"
