# Implementation Plan: opt-in instrumentation accessors (issue #52)

**Date:** 2026-05-25
**Issue:** https://github.com/jkitchin/feral/issues/52
**Branch:** `feature/issue-52-opt-in-stats`
**Research note:** none yet (see "Research gaps" below — may not need one)
**Spec sections:** §2.12.1 (POUNCE integration interface)

## Motivating goal

Make pounce-side debugging easier without paying for it on every IPM
iteration. Concretely: when a solve is slow, fragile, or hits unexpected
regularization, the IPM driver should be able to read back attribution
data via `Solver` accessors instead of re-running with `cargo flamegraph`
or instrumenting forks.

**Hard constraint (from the user, this session):** no noticeable
performance loss when not debugging. The default code path must be
bit-for-bit the same as today.

## Files to Create/Modify

```
src/numeric/solver.rs        (edit)  — accessors, optional Profiler ownership
src/numeric/factorize.rs     (edit)  — only if Phase B needs thread-local refactor
tests/issue52_stats.rs       (new)   — accessor surface, zero-overhead invariants
benches/issue52_overhead.rs  (new)   — measure default-off vs profiling-on
dev/plans/issue-52-opt-in-stats.md   (this file)
dev/journal/2026-05-25-NN.org        (per-session log)
```

No new dependencies. Specifically: **no `tracing` crate** in this issue
— that goes in a follow-up if the accessor pattern alone is insufficient.

## Phasing

### Phase A — cheap wrappers, zero new instrumentation

Surface fields that `Solver` already collects but doesn't expose, plus
trivial computations from already-exposed state. **No runtime cost
anywhere.**

New type:

```rust
/// Snapshot of per-`factor()` diagnostic state. All fields are read
/// off state `Solver` already maintains; producing this snapshot is
/// O(1) work and allocates nothing beyond the struct itself.
#[derive(Debug, Clone)]
pub struct FactorStats {
    pub nnz_a: usize,
    pub nnz_l: usize,
    pub fill_ratio: f64,            // nnz_l / nnz_a
    pub inertia: Inertia,
    pub min_abs_pivot: f64,
    pub max_abs_pivot: f64,
    pub pattern_reused: bool,       // symbolic cache hit this factor()?
    pub scaling_info: ScalingInfo,  // already on Solver
}
```

New accessor:

```rust
impl Solver {
    /// Snapshot of the most recent successful `factor()` call. `None`
    /// until the first successful `factor()`.
    pub fn last_factor_stats(&self) -> Option<FactorStats>;
}
```

What's net-new state on `Solver`:
- `last_nnz_a: Option<usize>` (single `usize` write per `factor()`)
- `last_pattern_reused: Option<bool>` (single `bool` write per `factor()`)

Everything else (`nnz_l`, `inertia`, `min/max_abs_pivot`, `scaling_info`)
is already there. Per-factor cost: two integer writes. Zero allocation,
zero locks, zero added syscalls. This satisfies the "no noticeable
performance loss when not debugging" constraint trivially because the
writes happen unconditionally and cost less than a single L1 hit.

**Rationale for unconditional collection in Phase A:** the two new
fields cost less than the gating check would. Keeping it
unconditional avoids any `Solver::with_stats_enabled` flag at all and
keeps the API tiny.

### Phase B — opt-in profile reports, default off

Wire the existing `Profiler` and `SymbolicProfiler` through `Solver`
behind a runtime toggle:

```rust
impl Solver {
    /// Enable per-factor profile collection. Default off. When off,
    /// `factor()` is byte-identical to current behavior; when on,
    /// every supernode pays two `Instant::now()` calls and one
    /// `Mutex` lock acquisition (sequential driver) or one
    /// thread-local push (parallel driver, see B2 below).
    pub fn with_profiling(mut self, on: bool) -> Self;

    /// Profile report from the most recent factor. `None` if
    /// profiling is off or no factor has run.
    pub fn profile_report(&self) -> Option<ProfileReport>;

    pub fn symbolic_profile_report(&self) -> Option<SymbolicProfileReport>;
}
```

Two sub-decisions to validate empirically on the branch before
shipping:

**B1: runtime flag vs Cargo feature.** Start with runtime flag because:
- The collection sites already gate on `params.profiler.as_ref()`
  being `None`, which is a single null-check. When `with_profiling(false)`
  the `Solver` simply never constructs the `Arc<Mutex<Profiler>>`, so
  the gate stays cold.
- A Cargo feature would require every downstream crate (pounce,
  pounce-feral, feral-ipopt-shim) to enable it, complicating
  packaging.
- If the bench in Phase B shows the *off* path is not free, escalate
  to a Cargo feature in a follow-up.

**B2: thread-local accumulation for parallel driver.** The current
`Arc<Mutex<Profiler>>` is fine sequentially (uncontended lock) but
under the rayon driver it adds a shared-mutex hit per supernode
completion — exactly the contention class that issue #19 was about.
Two paths:

- **B2a (cheap):** ship Phase B with documented "diagnostic mode,
  expect 1–5% wall-time overhead under parallel driver, do not leave
  on across an IPM run." This is honest and matches what the issue
  actually motivates (post-mortem debugging, not always-on
  observation).
- **B2b (more work):** refactor `Profiler` to per-worker thread-local
  `Vec<SupernodeTiming>` merged at epilogue. Eliminates contention,
  changes the profiler internal API.

Default to B2a on this branch. Only do B2b if benchmark numbers say
the IPM warm-cache path regresses meaningfully (> 1%).

### Phase C — deferred

Not in this issue:
- `n_inertia_corrections` aggregate counter (needs reason-for-refactor
  enum that doesn't exist; `quality_level()` partly covers it).
- `n_refinement_iters` / `SolveStats` (would need plumbing into
  `solve_sparse_refined`; do as separate issue if pounce needs it).
- `tracing` integration (separate issue, weigh dep cost then).
- Cached `condition_estimate()` accessor (the existing explicit
  `Solver::estimate_condition_1norm(matrix)` is the right API).

## Implementation Steps

1. Write `tests/issue52_stats.rs` with the accessor surface tests
   (failing) **before** any production code.
2. Phase A: add `FactorStats`, `last_factor_stats()`, and the two new
   `Solver` fields. Wire writes inside the existing `factor()` body
   adjacent to where `last_inertia` / `last_pattern_fingerprint` are
   already updated.
3. Add the Phase A zero-overhead bench (compare wall time of a tight
   `factor()` loop with and without merging Phase A — should be
   indistinguishable; this is a sanity check on the implementation,
   not a release gate).
4. Phase B: add `with_profiling(on)` and the two report accessors.
   Internally, when `on`, lazily construct `Arc<Mutex<Profiler>>` and
   inject into `numeric_params.profiler` for the duration of each
   `factor()` call. Same for `SymbolicProfiler` on cache-miss factors.
5. Add the Phase B overhead bench: same workload, `with_profiling(false)`
   vs `with_profiling(true)`, sequential and parallel drivers. Record
   results in the session checkpoint. The default-off case must be
   within bench noise of the pre-issue-52 baseline.
6. If parallel-driver overhead regresses > 1%, do B2b (thread-local
   refactor). Otherwise document the overhead in the rustdoc on
   `with_profiling`.
7. Update CHANGELOG.md (Unreleased) with the new accessors.

## Tests (write first)

`tests/issue52_stats.rs`:
- `last_factor_stats_returns_none_before_factor` — guards the
  `Option` contract.
- `last_factor_stats_after_success` — checks every field is populated
  on a small known matrix where `nnz_l`, `fill_ratio`, and pivots are
  hand-computable.
- `pattern_reused_false_first_factor_true_second` — exercises the
  symbolic cache hit signal on bit-identical replays.
- `pattern_reused_false_after_pattern_change` — exercises the
  cache-miss case.
- `profile_report_none_when_disabled` — Phase B opt-in gate.
- `profile_report_some_when_enabled` — Phase B happy path, checks
  `n_supernodes > 0` and `total_us > 0` on a non-trivial matrix.
- `symbolic_profile_report_some_only_on_symbolic_factor` — confirms
  the symbolic report is `None` on cache hits and `Some` on the
  first factor of a pattern.

`benches/issue52_overhead.rs` (criterion):
- `factor_default` (Phase A merged, profiling off)
- `factor_with_profiling` (Phase B on, sequential)
- `factor_with_profiling_parallel` (Phase B on, parallel)

On a small set of representative matrices (one IPM warm-cache pattern
from the corpus, one mid-size from the bench suite).

## Success Criteria

1. **Functional.** `cargo test --test issue52_stats` passes; pounce can
   read every field the issue listed in its `FactorStats` design
   (modulo the deferred Phase C fields, which the user agreed to omit).
2. **Default-off performance.** `factor_default` on the new branch is
   within criterion noise of the same workload on `main`. No
   measurable regression on any matrix in the bench corpus.
3. **Profiling-on overhead.** `factor_with_profiling` documented in
   the `with_profiling` rustdoc with the actual measured overhead on
   the bench corpus. Sequential should be < 1%; parallel may be
   higher and is acceptable if documented.
4. **API ergonomics.** A single line in pounce —
   `solver.with_profiling(true)` — flips on every diagnostic surface
   the issue lists. Reading is `solver.last_factor_stats()` and
   `solver.profile_report()`. No `Arc<Mutex<...>>` plumbing leaks
   into the pounce caller.

## Research gaps

None blocking. The internal types (`Profiler`, `ProfileReport`,
`SymbolicProfiler`, `BucketStats`, `FrontalProfile`) already have
research notes (`dev/research/phase-2.13b-symbolic-profiler.md`,
`dev/research/feral-kernel-profile-chainwoo.md`); this issue exposes
them rather than designing new instrumentation. A research note is
warranted only if Phase B benchmarks force B2b (thread-local
refactor) — at that point document the merge protocol and ordering
guarantees.

## Out of scope (explicit, repeated)

- No control feedback: stats never influence pivoting / regularization /
  refactor decisions.
- No I/O inside FERAL: no file dumping, no stdout/stderr, no log
  subscribers. Caller decides persistence.
- No public API for `Profiler` mechanics beyond the frozen
  `ProfileReport` snapshot.
