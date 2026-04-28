# Plan: MUMPS missing-diagonal MC64 skip

## Background

`profile_hot` (session 2026-04-28-01) showed `mc64::compute_matching`
burning **26%** of wall time on the 7-matrix mix. MC64 fires through
two independent gates today:

1. **Symbolic** — `OrderingPreprocess::Auto` resolves to
   `LdltCompress` when `n >= 128` and `low_degree/n >= 0.30`
   (`src/symbolic/mod.rs:299-321`). Runs MC64 once for compression
   and stashes the cache.
2. **Numeric scaling** — `ScalingStrategy::Auto` resolves to
   `Mc64Symmetric` when `diag_only/n >= 0.30`
   (`src/scaling/mod.rs:371-392`). When the symbolic cache is
   present this is O(n); otherwise it reruns MC64.

The numeric scaling gate today has only the *positive* arrow-KKT
detector. There is no *negative* test that says "diagonal is already
strong, MC64 has nothing to recover, skip it."

MUMPS adds exactly that test in its analysis phase. From
`mumps/src/dana_aux.F:1388-1416` (the SYM=2 ordering preprocess),
when

```
(missing_diag + zero_diag) < max(1, N/10)
```

MC64 (KEEP(52)=4) is skipped and SYM_PERM/Identity scaling is used
instead — falling through to cheap symmetric Ruiz equilibration
(SIMSCA / KEEP(52)=7) at numeric scaling time. The rationale is
purely numerical: when the matrix's structural-and-numerical
diagonal is dense, the matching algorithm cannot improve diagonal
dominance — it only adds cost.

## Goal

Replicate the MUMPS skip rule in `pick_scaling_strategy`. When the
diagonal is mostly populated, route to InfNorm (feral's analog of
SIMSCA) instead of MC64 — even if the arrow check would otherwise
say MC64.

Caveat: the lever-C empirical work (2026-04-19) shows MC64 wins on
the VESUVIO/CRESC arrow-KKT corpus. Those matrices are KKT
matrices with structurally absent dual-block diagonals; their
`missing_diag/n` ratio is far above 1/10, so the MUMPS skip
naturally does not fire on them. The arrow-KKT and missing-diagonal
tests are complementary, not contradictory:

| matrix class                | missing+zero diag / n | arrow ratio | rule       |
|-----------------------------|----------------------:|------------:|------------|
| H-only block (NLP H+Σ)      |                  ~0  |       0–0.2 | InfNorm    |
| arrow-KKT (constraint slack + dual block) | >> 1/10 |       >0.30 | MC64       |
| dense indefinite            |                  ~0  |       0–0.1 | InfNorm    |

The MUMPS skip should win on cases like HS118-class small NLP H
blocks where `pick_scaling_strategy` today returns InfNorm anyway —
in that case the rule is a no-op. But when:

- the symbolic phase did NOT run LdltCompress (so no cache exists), and
- the matrix has the arrow signature AND a fully populated diagonal

we currently rerun MC64 from scratch. The skip cuts that.

## Rule

In `pick_scaling_strategy`, add the skip BEFORE the arrow check:

```rust
let n = matrix.n;
if n == 0 { return ScalingStrategy::InfNorm; }

let missing_or_zero = count_missing_or_zero_diag(matrix);
let skip_threshold = (n / 10).max(1);
if missing_or_zero < skip_threshold {
    return ScalingStrategy::InfNorm;  // MUMPS rule
}

// existing arrow-KKT check
let mut diag_only = 0;
for j in 0..n { ... }
if diag_only as f64 / n as f64 >= 0.30 {
    ScalingStrategy::Mc64Symmetric
} else {
    ScalingStrategy::InfNorm
}
```

`count_missing_or_zero_diag(matrix)` walks each column and checks
whether `A[j,j]` is structurally absent OR present-and-numerically-zero.
O(nnz), no allocations.

## What this does NOT change

- The Policy 4 fallback (Auto → MC64 path that detects MC64
  catastrophe and falls back to InfNorm) is untouched.
- The Mc64Cache reuse path is untouched. When LdltCompress runs in
  symbolic and produces a cache, the scaling phase still consumes
  it cheaply if Auto picks MC64.
- The MSS1_0009 fallback test still passes (MSS1 has many zero/missing
  dual diagonals → skip does NOT fire → existing arrow check picks
  MC64 → Policy 4 falls back to InfNorm).
- The VESUVIA_0000 arrow-KKT test still passes (same reason — many
  zero dual diagonals → skip does NOT fire → MC64 runs).

## Tests (TDD order)

1. `pick_scaling_strategy_skips_mc64_on_dense_diagonal`: small matrix
   with all diagonals present and nonzero, plus arrow signature →
   returns InfNorm (skip overrides arrow).
2. `pick_scaling_strategy_keeps_mc64_when_dual_block_zero_diag`:
   half the diagonals zero/missing → arrow check applies.
3. `pick_scaling_strategy_threshold_n_over_10`: exactly N/10 missing
   → keep MC64 if arrow; just under → skip.
4. `pick_scaling_strategy_treats_value_zero_as_missing`: structural
   present but numeric value 0.0 counts toward the missing budget.
5. Existing tests must still pass:
   - `pick_scaling_strategy_picks_mc64_for_arrow_kkt` — `shape_csc`
     uses degree-1 columns where the diagonal IS present. Need to
     update those to also have missing diagonals on the non-diag-only
     columns, OR explicitly count missing_diag and ensure the skip
     fires/doesn't fire as documented.
   - `auto_keeps_mc64_on_vesuvia_0000` — verify VESUVIA_0000 has
     enough missing/zero diags.
   - `auto_falls_back_to_infnorm_on_mss1_0009` — MSS1 already has
     enough missing diags.

## Validation against the perf bench

After implementation:

1. `cargo test --release` — must stay 205/205.
2. `cargo bench --bin bench_solver_corpus` — record before/after
   aggregate speedup, geomean, p50, p90.
3. `cargo run --release --bin bench` — record before/after of the
   per-matrix bench. Expect a bigger movement here since the
   per-matrix bench pays scaling on every matrix (no Solver).

## Estimated wall savings

5–15% on `bench_solver_corpus` (most reuse already amortizes MC64
via Mc64Cache); larger on the per-matrix `bench` where MC64 reruns
on every call when LdltCompress doesn't fire.

## Reference

- `mumps/src/dana_aux.F:1388-1416` — SYM=2 missing-diagonal skip
- `dev/research/mc64-scaling.md` — MC64 design context
- `dev/research/lever-c-adaptive-scaling.md` — empirical arrow-KKT
  routing rationale
- `dev/research/policy-4-scaling-fallback.md` — orthogonal fallback
- `dev/sessions/2026-04-28-01.md` — origin profile
- `dev/tried-and-rejected.md:977-1021` — `MIN_N_FOR_COMPRESSION`
  rejection (NOT the same lesson — that was gating on `n` alone for
  symbolic compression; this is gating on the *structural diagonal*
  signal for numeric scaling).
