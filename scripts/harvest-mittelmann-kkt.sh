#!/usr/bin/env bash
# Harvest KKT matrices from ripopt runs on the Mittelmann ampl-nlp benchmark.
#
# Background. The Mittelmann set (47 medium-to-large NLPs, n = 500 to 261k)
# is a class ripopt does not yet do well on. KKT systems from the *failing*
# iterations of those runs are exactly the matrices we want in the FERAL
# corpus: large, ill-conditioned, and from a workload where the linear
# solver matters. Each iteration dumps a {.mtx, .json} pair regardless of
# whether the outer IPM converges.
#
# Prerequisites:
#   1. Mittelmann .nl files cached under
#      ../ripopt/benchmarks/mittelmann/nl/*.nl
#      (run `make translate` over there once if not).
#   2. A ripopt build whose `ripopt_ampl` driver accepts
#      `kkt_dump_dir=` and `kkt_dump_name=` as CLI key=value options.
#      That requires the small patch shown in
#      dev/notes/mittelmann-harvest.md (two arms in apply_option).
#
# Output layout (matches dev/scripts/collect_kkt convention):
#   data/matrices/kkt-mittelmann/<problem>/<problem>_<iter:04>.mtx
#   data/matrices/kkt-mittelmann/<problem>/<problem>_<iter:04>.json
#
# Usage:
#   scripts/harvest-mittelmann-kkt.sh                     # all problems
#   scripts/harvest-mittelmann-kkt.sh nql180 marine_1600  # subset
#
# Env vars:
#   MITT_NL_DIR     — source dir for .nl files
#                     (default: ../ripopt/benchmarks/mittelmann/nl)
#   MITT_OUT_DIR    — output root
#                     (default: data/matrices/kkt-mittelmann)
#   RIPOPT_BIN      — ripopt binary path
#                     (default: ../ripopt/target/release/ripopt)
#   PER_PROBLEM_TIMEOUT — seconds per problem (default: 300)
#   PER_PROBLEM_MAX_ITER — IPM iteration cap (default: 200)
#
# A 200-iter cap with a 5-min wall-time cap usually yields 10-200 KKT
# matrices per problem. Hard problems that would otherwise loop forever
# get bounded; easy problems just stop at convergence.

set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MITT_NL_DIR="${MITT_NL_DIR:-${ROOT}/../ripopt/benchmarks/mittelmann/nl}"
MITT_OUT_DIR="${MITT_OUT_DIR:-${ROOT}/data/matrices/kkt-mittelmann}"
RIPOPT_BIN="${RIPOPT_BIN:-${ROOT}/../ripopt/target/release/ripopt}"
PER_PROBLEM_TIMEOUT="${PER_PROBLEM_TIMEOUT:-300}"
PER_PROBLEM_MAX_ITER="${PER_PROBLEM_MAX_ITER:-200}"

if [ ! -x "$RIPOPT_BIN" ]; then
  echo "ripopt binary not found or not executable: $RIPOPT_BIN" >&2
  echo "Build it with: (cd ../ripopt && cargo build --release --bin ripopt)" >&2
  exit 1
fi

if [ ! -d "$MITT_NL_DIR" ]; then
  echo "Mittelmann .nl cache not found: $MITT_NL_DIR" >&2
  echo "Populate it with: (cd ../ripopt/benchmarks/mittelmann && make translate)" >&2
  exit 1
fi

mkdir -p "$MITT_OUT_DIR"

if [ "$#" -gt 0 ]; then
  PROBLEMS=("$@")
else
  PROBLEMS=()
  while IFS= read -r -d '' nl; do
    PROBLEMS+=("$(basename "$nl" .nl)")
  done < <(find "$MITT_NL_DIR" -maxdepth 1 -name '*.nl' -print0 | sort -z)
fi

total=${#PROBLEMS[@]}
if [ "$total" -eq 0 ]; then
  echo "No .nl files found under $MITT_NL_DIR" >&2
  exit 1
fi

echo "Harvesting KKT matrices from $total Mittelmann problem(s)"
echo "  source: $MITT_NL_DIR"
echo "  output: $MITT_OUT_DIR"
echo "  ripopt: $RIPOPT_BIN"
echo "  timeout: ${PER_PROBLEM_TIMEOUT}s, max_iter: ${PER_PROBLEM_MAX_ITER}"
echo

i=0
n_done=0
n_skipped=0
for problem in "${PROBLEMS[@]}"; do
  i=$((i+1))
  nl="${MITT_NL_DIR}/${problem}.nl"
  out_dir="${MITT_OUT_DIR}/${problem}"

  if [ ! -f "$nl" ]; then
    printf "[%3d/%d] %-25s SKIP (no .nl file)\n" "$i" "$total" "$problem"
    n_skipped=$((n_skipped+1))
    continue
  fi

  if [ -d "$out_dir" ] && [ -n "$(ls -A "$out_dir"/*.mtx 2>/dev/null)" ]; then
    n=$(ls "$out_dir"/*.mtx 2>/dev/null | wc -l | tr -d ' ')
    printf "[%3d/%d] %-25s SKIP (%s mtx files already present)\n" "$i" "$total" "$problem" "$n"
    n_skipped=$((n_skipped+1))
    continue
  fi

  mkdir -p "$out_dir"
  printf "[%3d/%d] %-25s ... " "$i" "$total" "$problem"

  start=$(python3 -c 'import time; print(time.time())')
  # `print_level=0` because we don't care about the IPM stdout — only the
  # KKT files matter. `tol=1e-8` so the iteration cap is what bounds
  # us, not premature acceptable-solution termination on hard problems.
  timeout "$PER_PROBLEM_TIMEOUT" \
    "$RIPOPT_BIN" "$nl" -AMPL \
      "kkt_dump_dir=${out_dir}" \
      "kkt_dump_name=${problem}" \
      "max_iter=${PER_PROBLEM_MAX_ITER}" \
      "max_wall_time=${PER_PROBLEM_TIMEOUT}" \
      "print_level=0" \
      > "${out_dir}/${problem}.solver.log" 2>&1
  rc=$?
  end=$(python3 -c 'import time; print(time.time())')
  elapsed=$(python3 -c "print(f'{$end - $start:.1f}')")

  n_mtx=$(ls "$out_dir"/*.mtx 2>/dev/null | wc -l | tr -d ' ')

  case "$rc" in
    0)   status="OK" ;;
    124) status="TIMEOUT" ;;
    *)   status="EXIT=$rc" ;;
  esac

  printf "%-10s %4d mtx (%ss)\n" "$status" "$n_mtx" "$elapsed"
  n_done=$((n_done+1))
done

echo
echo "Done: $n_done processed, $n_skipped skipped"
echo "Total mtx files written: $(find "$MITT_OUT_DIR" -name '*.mtx' | wc -l | tr -d ' ')"
