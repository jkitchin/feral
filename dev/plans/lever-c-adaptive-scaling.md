# Plan — Lever C: Adaptive Scaling Policies (1, 2, 3)

**Date:** 2026-04-19
**Source note:** `dev/research/lever-c-adaptive-scaling.md`
**Touches:** `src/scaling/mod.rs`, `src/bin/bench.rs`
**Out of scope (this plan):** Policy 4 (try-MC64-fallback-to-InfNorm).
The note's §5 defers Policy 4 until 1+2+3 are inconclusive.

## Goal

Land the smallest, most reversible code path that lets the corpus bench
measure the four scaling policies side by side, without changing the
production default. The current default (`ScalingStrategy::InfNorm`)
remains untouched so existing tests and benchmarks behave identically.

## Design

1. **New variant** `ScalingStrategy::Auto` in `src/scaling/mod.rs`.
   Resolved at the entry to `compute_scaling` by a private
   `pick_scaling_strategy(matrix: &CscMatrix) -> ScalingStrategy`
   helper. Returns `Mc64Symmetric` when `diag_only_rows / n >= 0.3`,
   else `InfNorm`. The threshold mirrors the note's §4 candidate.

2. **`pick_scaling_strategy` is one O(nnz) pass.** Counts columns
   whose only stored row is the diagonal (the "constraint slack"
   shape that `vesuvio_diag` already prints). No allocations beyond
   one usize counter.

3. **`SupernodeParams::default` is not changed.** Production keeps
   `InfNorm`. Policy selection is bench-local via env var.

4. **Bench env var `FERAL_SCALING={infnorm,mc64,adaptive}`.**
   Mirrors the existing `FERAL_ORDERING` pattern in
   `src/bin/bench.rs::ordering_method_from_env`. Unset → current
   default (`InfNorm`, baseline). The bench overrides
   `snode_params.scaling_strategy` once for the entire corpus run.

   - `infnorm`  → `ScalingStrategy::InfNorm`     (Policy 1, baseline)
   - `mc64`     → `ScalingStrategy::Mc64Symmetric` (Policy 2)
   - `adaptive` → `ScalingStrategy::Auto`         (Policy 3)

   Echoed in the bench banner alongside the ordering line.

## Tests

In `src/scaling/mod.rs`:

- `pick_scaling_strategy_picks_mc64_for_arrow_kkt`: constructs a small
  CSC with `diag_only / n = 0.5` (10 of 20 columns), asserts
  `Mc64Symmetric`.
- `pick_scaling_strategy_picks_infnorm_for_dense`: constructs a small
  CSC with no diag-only columns, asserts `InfNorm`.
- `pick_scaling_strategy_threshold_boundary`: constructs CSCs at
  exactly 0.29 and 0.30 ratios, asserts `InfNorm` then `Mc64Symmetric`.
- `compute_scaling_auto_routes_correctly`: end-to-end —
  `compute_scaling(matrix, &Auto)` returns a vector consistent with
  what `compute_scaling(matrix, &Mc64Symmetric)` returns on a matrix
  the heuristic routes to MC64.

No production-code regression test needed — the production default
hasn't changed. Existing 101 lib tests must still pass with no edits.

## Bench harness changes

1. Add `scaling_strategy_from_env() -> Option<ScalingStrategy>` next
   to `ordering_method_from_env`. Returns `None` when the env var is
   unset (so the existing "default" behavior is preserved exactly).
2. In `main`, after the ordering banner, print the scaling banner.
3. At the `let snode_params = SupernodeParams::default()` site,
   override `scaling_strategy` if the env var is set.

That's the entire diff. No new bench binary; the same harness emits
the same tables under each policy by varying one env var.

## Validation procedure

Three back-to-back runs:

```
cargo run --bin bench --release                        > bench-baseline.txt 2>&1
FERAL_SCALING=mc64     cargo run --bin bench --release > bench-mc64.txt    2>&1
FERAL_SCALING=adaptive cargo run --bin bench --release > bench-adaptive.txt 2>&1
```

Capture from each:

- factor/MUMPS geomean, p50, p90, p99, max
- sparse residual-pass count (currently 154 241 of 154 588)
- sparse inertia-pass count
- top-10 worst factor-ratio entries
- failures listed under "factor_fail" / "solve_fail"

Decision rule (reiterating the note's §5):
- Highest residual-pass + inertia-pass count wins; geomean is the
  tie-breaker.
- Tolerances are NOT loosened. CLAUDE.md hard rule.
- If Policy 3 ≥ Policy 2 on all four metrics, Policy 3 is the
  recommendation.
- If Policy 2 dominates Policy 3, the recommendation is Policy 2 (one
  line in `SupernodeParams::default`).
- If neither dominates Policy 1 (baseline) on residual count, no
  default change; lever C documented as opt-in.

A measurement note `dev/research/lever-c-corpus-bench-2026-04-NN.md`
records the headline tables and the decision. The implementation plan
to flip the default (or wire `Auto`) is separate and follows the
measurement.

## Risks & mitigations

- **Risk:** the `diag_only / n >= 0.3` threshold is overfit to the
  seven measured matrices. **Mitigation:** the corpus bench validates
  it against all 154 588; if Policy 3 underperforms Policy 1 on any
  family, the threshold is wrong and the next plan re-calibrates.
- **Risk:** MC64 is 2–4× slower than InfNorm at symbolic time. On
  ~150k tiny matrices this could swamp the factor win.
  **Mitigation:** the bench reports total wall time; if Policy 2's
  factor-geomean win is wiped out by symbolic overhead, that's data
  in favor of Policy 3 (selective MC64 only on the matrices that need
  it).
- **Risk:** `Auto` semantics could surprise external users.
  **Mitigation:** `Auto` is an opt-in variant, never the default of
  `ScalingStrategy::default`. Documented as such.

## Estimated cost

- Plan: this file. Done.
- Code (Step 1–3 above): one session, ~150 LOC + tests.
- Three bench runs + measurement note: same session, ~30 min wall.
- Implementation plan to flip the default: separate session if
  warranted by data.
