#!/usr/bin/env bash
# Build a KKT corpus from pounce's Pyomo-generated large-scale NLP suite.
#
# Replaces the AMPL path. `scripts/harvest-mittelmann-kkt.sh` needs the
# Mittelmann .mod sources translated by an AMPL Community Edition binary,
# which needs a licence UUID. pounce's `benchmarks/large_scale/generate_nl.py`
# builds the same *kind* of problems in Pyomo and writes .nl directly, so
# the whole corpus can be regenerated with no AMPL and no network.
#
# Six families, covering the shapes the ordering work cares about:
#
#   bratu       1-D PDE            (the #67 bratu3d family)
#   poisson     2-D elliptic control  (the #73 cont5_1_l family)
#   optcontrol  discrete-time control (the #73 dtoc2 family)
#   laptime     nonconvex Radau collocation (the issue #203 family)
#   sparseqp    convex QP
#   rosenbrock  unconstrained, tridiagonal
#
# Two things that will waste an hour if you do not know them:
#
#   1. pounce options are `key=value`, not `--key value`. `--max-iter 12`
#      silently prints the help text and solves with defaults.
#   2. `solver_selection=auto` sends convex QPs to `pounce-convex`, which
#      does not dump KKT systems. optcontrol, poisson and sparseqp produce
#      *zero* dumps unless you force `solver_selection=nlp`.
#
# Usage:
#   scripts/build-pounce-corpus.sh                  # default sizes
#   POISSON_K=450 scripts/build-pounce-corpus.sh poisson
#
# Env:
#   POUNCE_DIR  (default ../pounce)
#   OUT_DIR     (default data/matrices/kkt-pounce)
#   MAX_ITER    (default 30)
#   WORK        scratch for the .nl and dump trees (default $TMPDIR/pounce-corpus)

set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
POUNCE_DIR="${POUNCE_DIR:-${ROOT}/../pounce}"
OUT_DIR="${OUT_DIR:-${ROOT}/data/matrices/kkt-pounce}"
MAX_ITER="${MAX_ITER:-30}"
WORK="${WORK:-${TMPDIR:-/tmp}/pounce-corpus}"
GEN="${POUNCE_DIR}/benchmarks/large_scale/generate_nl.py"
BIN="${POUNCE_DIR}/target/release/pounce"

PROBLEMS=("$@")
if [ ${#PROBLEMS[@]} -eq 0 ]; then
    PROBLEMS=(rosenbrock bratu optcontrol poisson sparseqp laptime)
fi

for req in "$GEN" "$BIN"; do
    [ -e "$req" ] || { echo "missing: $req" >&2; exit 1; }
done
python3 -c "import pyomo" 2>/dev/null || { echo "pyomo not installed" >&2; exit 1; }

mkdir -p "$WORK/nl" "$OUT_DIR"
echo "generating .nl into $WORK/nl"
python3 "$GEN" "${PROBLEMS[@]}" --out-dir "$WORK/nl" || exit 1

for p in "${PROBLEMS[@]}"; do
    nl="$WORK/nl/$p.nl"
    [ -f "$nl" ] || { echo "SKIP $p: no $nl" >&2; continue; }
    echo "=== $p ==="
    rm -rf "$WORK/dump/$p"
    "$BIN" "$nl" "max_iter=$MAX_ITER" solver_selection=nlp \
        --dump kkt:all --dump-dir "$WORK/dump/$p" >/dev/null 2>&1
    n=$(find "$WORK/dump/$p" -name 'kkt_solve_*.jsonl' 2>/dev/null | wc -l | tr -d ' ')
    if [ "$n" = "0" ]; then
        echo "  no dumps — check solver_selection and the option syntax" >&2
        continue
    fi
    python3 "$ROOT/scripts/harvest-pounce-kkt.py" \
        --dump-dir "$WORK/dump/$p" --name "$p" --out "$OUT_DIR"
done

echo
echo "corpus at $OUT_DIR"
du -sh "$OUT_DIR"
echo "run the bench with:  FERAL_KKT_ROOTS=kkt-pounce cargo run --bin bench --release"
