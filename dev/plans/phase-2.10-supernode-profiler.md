# Phase 2.10 — Per-supernode profiler

**Date:** 2026-04-25
**Research note:** `dev/research/reference-solver-comparison.md` (item 2 of
"Proper next investigation," lines 144–161).
**Spec sections:** §5.1 (feature lifecycle), §13.3 (research-first rule).

## Motivation

Phases 2.9 and 2.9.2 each tried to close the "5–8× MUMPS gap on tiny IPM
KKT" by profiling individual leaf calls and were both null. The
post-mortem (`reference-solver-comparison.md`) identified the problem:
neither phase decomposed feral's `factor_us` *by front size*, so we
never knew whether time was concentrated in 10,000 tiny fronts or 100
big fronts. Without that decomposition every "fix amalgamation" or
"fix per-front kernel" plan is a guess.

The 2026-04-25 4-agent investigation confirmed two independent
mechanisms for MUMPS's edge:

1. Aggressive amalgamation (NEMIN=5, tiny-front merge rule) collapses
   the front count.
2. NASS<24 single-shot kernel path (no inner blocking, one trailing
   GEMM) makes each tiny front cheap.

Both fixes are *plausible*. To know which one matters more on feral
(and on which matrices) we need the histogram first. The same agent
investigation also widened the slow set: LAKES (1,254 matrices), SWOPF
(649), NELSON (500) are *not* in the four "known" tail families — they
constitute ~20% of the >2× slowdown set and have never been profiled.
This phase produces the instrument that makes them tractable.

## Scope

This phase ships **a diagnostic, not a perf change.** No factorization
output changes. Acceptance is "the histogram is trustworthy and shows
something we did not already know."

In scope:
- A `Profiler` struct + optional field on `NumericParams`.
- Per-supernode timing in `factorize_multifrontal_supernodal_with_workspace`.
- Prologue/epilogue timing.
- A bucketing/report module.
- A `src/bin/profile_supernode_distribution.rs` binary.

Out of scope:
- The parallel driver path (`factorize_multifrontal_supernodal_parallel`,
  src/numeric/factorize.rs:1027). Separate phase if needed.
- Acting on the histogram (Phase 2.11+ is "respond to what the data
  says": amalgamation tuning vs small-front fast-path vs both).

## Files to create / modify

- `src/numeric/factorize.rs` — add `Profiler`, `SupernodeTiming`,
  `ProfileReport`, `BucketStats`; add `profiler: Option<Arc<Mutex<Profiler>>>`
  field to `NumericParams`; instrument the sequential driver.
- `src/bin/profile_supernode_distribution.rs` — new binary.
- `tests/profiler_smoke.rs` — new test (acceptance invariants).

No public API breakage: `NumericParams` already derives `Default`, so
existing call sites that pass a constructed `NumericParams` continue
to compile (the new field defaults to `None`).

## Implementation steps

### Step A — Types and field (no instrumentation yet)

Add to `src/numeric/factorize.rs`:

```rust
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone)]
pub struct SupernodeTiming {
    pub snode_idx: usize,
    pub nrow: usize,
    pub ncol: usize,
    pub us: u64,
}

#[derive(Debug, Clone, Default)]
pub struct Profiler {
    timings: Vec<SupernodeTiming>,
    prologue_us: u64,
    epilogue_us: u64,
    total_us: u64,
}
```

Add field:

```rust
pub struct NumericParams {
    pub bk: BunchKaufmanParams,
    pub scaling: ScalingStrategy,
    pub small_leaf: SmallLeafBatch,
    /// Optional per-invocation profiler; zero overhead when None.
    pub profiler: Option<Arc<Mutex<Profiler>>>,
}
```

`Default` and `with_bk` updated to initialize `profiler: None`.

**Gate:** `cargo check && cargo clippy -- -D warnings` clean.

### Step B — Instrument the sequential driver

In `factorize_multifrontal_supernodal_with_workspace`
(src/numeric/factorize.rs:439):

- Take a `total_us` start `Instant` at the top (line 444).
- Take a `prologue_us` start `Instant` at the top, end it just before
  the supernode loop at line 540.
- Inside both loop bodies (small-leaf path lines 549–569 and generic
  path lines 574–584), wrap the `factor_one_*` call with
  `Instant::now()` and on completion lock the profiler and push a
  `SupernodeTiming { snode_idx, nrow: snode.nrow, ncol: snode.ncol, us }`.
- Take an `epilogue_us` start `Instant` at line 595 and stop it at the
  end of the function before `Ok(...)`.

All Mutex locks via `if let Ok(mut prof) = arc.lock()` — no `unwrap`,
no `expect`. A poisoned mutex is recorded as zero (best-effort
diagnostic, not a correctness path).

**Critical invariant:** when `params.profiler.is_none()`, *no*
`Instant::now()` calls fire and *no* timing arithmetic runs. Use
`if let Some(ref arc) = params.profiler` to gate every checkpoint.
This guarantees zero overhead in production.

**Gate:** `cargo test` green (existing parity tests must still pass —
the profiler adds nothing observable when `profiler: None`).

### Step C — Bucketing / report

Add to `src/numeric/factorize.rs`:

```rust
#[derive(Debug, Clone, serde::Serialize)]
pub struct BucketStats {
    pub range: &'static str,
    pub count: usize,
    pub sum_us: u64,
    pub pct_of_total: f64,
    pub avg_us: f64,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ProfileReport {
    pub n_supernodes: usize,
    pub prologue_us: u64,
    pub epilogue_us: u64,
    pub loop_us: u64,           // sum of bucket sum_us
    pub total_us: u64,           // wallclock
    pub overhead_pct: f64,       // (prologue + epilogue) / total
    pub buckets: Vec<BucketStats>,
}

impl Profiler {
    pub fn report(&self) -> ProfileReport { ... }
}
```

Bucket ranges (per the research note line 150):
`≤8`, `9–16`, `17–32`, `33–64`, `65–128`, `>128`. Bucket by `nrow`.

**Acceptance invariants enforced in `report()`:**
- `sum(bucket.count) == n_supernodes`
- `sum(bucket.sum_us) == loop_us` (exact, since both are sums of the
  same u64s).
- `loop_us + prologue_us + epilogue_us <= total_us` (allows for
  Mutex-lock and timer overhead absorbed into total_us).

If any invariant fails, `report()` returns a `ProfileReport` with
counts/sums as observed plus a `validation_warnings: Vec<String>` —
do not panic in a diagnostic.

**Gate:** Unit test in `tests/profiler_smoke.rs` constructs a tiny
matrix, runs with profiler attached, calls `.report()`, asserts all
invariants hold.

### Step D — Binary

`src/bin/profile_supernode_distribution.rs`:

- Hardcoded matrix list: `ACOPR30_0067`, `CRESC100_0000`,
  `LAKES_0000`, plus one NELSON and one SWOPF representative.
- For each: load CSC, run `symbolic_factorize`, then
  `factorize_multifrontal_with_workspace` 5× with profiler attached;
  take the run with median `total_us`.
- Skip-if-missing pattern (matches the corpus-skip CI fix).
- Output: pretty-printed JSON to stdout, one object per matrix.

### Step E — Verification

1. `cargo test --release` green (corpus present locally).
2. `cargo run --release --bin profile_supernode_distribution` prints
   non-degenerate histograms (not all in one bucket; bucket sums add
   to loop_us).
3. Quick read of the histograms — does ACOPR30 actually have a long
   tail of small fronts? If yes, this validates the amalgamation
   hypothesis. If the time is concentrated in a few large fronts,
   we redirect the next phase. Either result is informative.
4. `cargo clippy -- -D warnings` clean.
5. `cargo fmt` clean.

## Tests (write first)

`tests/profiler_smoke.rs`:

1. `profiler_none_is_zero_overhead_smoke` — factor a small matrix
   with `profiler: None`, ensure factorization succeeds. (Doesn't
   measure overhead — just verifies the None path compiles and runs.)
2. `profiler_records_one_per_supernode` — factor a matrix where
   `n_supernodes` is known, attach profiler, call `.report()`, assert
   `sum(bucket.count) == n_supernodes`.
3. `profiler_loop_us_sums_match` — same setup, assert
   `sum(bucket.sum_us) == loop_us` exactly.
4. `profiler_buckets_partition_by_nrow` — assert each timing falls
   in exactly one bucket.

These tests use the block-diag SPD matrix from `small_leaf_parity.rs`
so they don't depend on the gitignored corpus.

## Success criteria

- All four smoke tests pass.
- `cargo run --release --bin profile_supernode_distribution` produces
  JSON output for every matrix present locally (skips missing
  cleanly).
- The output for `ACOPR30_0067` shows a non-trivial bucket
  distribution (no single bucket >95% of `loop_us`) — this is what
  makes the data actionable.
- `prologue_us + epilogue_us + loop_us` is within 1% of `total_us`
  (the residual is timer/mutex overhead, which we want bounded).
- No `unwrap`, no `unsafe`, no test tolerance changes, no production
  perf regression on Off path (`tests/parallel_parity.rs` and
  `tests/small_leaf_parity.rs` still bit-equal).

## Rejection criteria

- If wiring the profiler causes any parity test to fail (i.e., the
  Off path is no longer bit-equal), the design is wrong — back out
  and redesign before pushing.
- If the histogram on ACOPR30 shows >95% of time in one bucket, the
  "long tail of tiny fronts" hypothesis is refuted on this matrix
  and Phase 2.11 needs rethinking. Record in
  `dev/tried-and-rejected.md` and do not proceed to amalgamation
  tuning blind.

## What this enables

After Phase 2.10 lands, Phase 2.11 picks one of:

- **2.11a — amalgamation tuning** (if the histogram shows >50% of
  loop_us in `≤16` and `17–32` buckets with high count): match
  MUMPS's NEMIN=5 + tiny-front merge rule.
- **2.11b — small-front fast path** (if loop_us is dominated by
  per-front fixed cost regardless of bucket): NASS<24 single-shot
  dense kernel with no inner blocking.
- **2.11c — LAKES profile pass** (if LAKES_0000 looks structurally
  different from ACOPR30): treat LAKES as a separate diagnostic
  problem.

We pick after seeing the histogram, not before.
