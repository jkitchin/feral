# Research Note: Work-aware parallel-assembly gate (issue #19)

**Status:** Pre-implementation
**Date:** 2026-05-15
**Related spec sections:** Phase 2.5.2 Step D parallel dispatch
**Key references:**
- `src/numeric/factorize.rs:2087-2103` — current `should_parallelize_assembly` (n_snodes ≥ 32 + ≥1 multi-child)
- `src/numeric/solver.rs:127-130` — Solver `use_parallel` default-on
- feral GitHub issue #19 — `robot_1600` claimed 12× regression, `henon120` 2.8× win
- Phase 2.5.2 Step E "reassess after corpus bench shows where per-task overhead breaks even" — never closed

## Overview

The current gate is purely structural: parallel fires when (a) the
assembly tree has at least 32 supernodes and (b) at least one supernode
has ≥ 2 children. It does not estimate per-supernode *work*. As a
result the parallel driver fires on assembly trees with many tiny
supernodes — exactly the IPM-KKT-control-problem profile where rayon
spawn / cv-wait costs dominate the factor cost.

On `robot_1600` (n=24k, KKT from control-NLP) the issue reports a
12× per-iter wall regression with `use_parallel = true`. On `henon120`
(n=32k, KKT with larger frontal blocks) parallel wins 2.8×. The same
default cannot be right for both: the binary structural gate is
under-determined.

## Reproduction

On Apple M4 Pro (14 physical cores), feral `597a90a` + pounce-feral
`792f412`, 200 iterations both runs, refine-on:

| problem | parallel=on | parallel=off | wall ratio | sys/wall (par) |
|---|---|---|---|---|
| robot_1600 | 25.3 s | 34.4 s | 0.74× | 53% |
| henon120 | 101 s | 294 s | 0.34× | 21% |

The issue's claim of 12× regression on `robot_1600` does not reproduce
at this magnitude on M4 Pro. Parallel is *faster* in wall (1.36×) but
burns 27 s of sys time for a 25.3 s wall — rayon-overhead-bound CPU
utilization, but not yet wall-dominant on this CPU.

Likely the issue's machine has a different core/spawn-cost profile
(higher core count, slower per-spawn cv-wait wakeup, OS scheduler
differences). The fundamental defect is the same — `should_parallelize_
assembly` is firing on a problem too small to amortize overhead — but
the magnitude varies with hardware.

This research note targets the fix that is directionally correct on
any CPU: gate parallel dispatch on a *flop-cost estimate* derived from
the symbolic factorization, not on supernode count alone.

## Algorithm

### Per-supernode flop estimate

For supernode `s` with `ncol = s.ncol` eliminated columns and
`nrow = s.nrow` total frontal rows:

```
nrow_below = nrow - ncol      (rows in the contribution block)
panel_flops = ncol^3 / 3      (LDL^T on the ncol×ncol diagonal panel)
trsm_flops  = ncol^2 * nrow_below
schur_flops = ncol * nrow_below^2
total       = panel_flops + trsm_flops + schur_flops
```

Simplification: for typical KKT supernodes `nrow ≫ ncol` so
`schur_flops` dominates. The cheaper proxy `ncol * nrow^2`
(overestimates slightly but O(1) per supernode and never zero on
non-trivial nodes) is sufficient for a threshold gate.

Total tree flops: `sum_s ncol_s * nrow_s^2`. Computed in O(n_snodes) at
dispatch time — cheap. Stored on the SymbolicFactorization for reuse
across refactorizations.

### Threshold

Calibration target: total flops at which the rayon overhead amortizes.
Order-of-magnitude argument:

- Rayon `spawn + join` cost on macOS (Apple M-series): ~1–10 μs per
  task in steady state, plus a per-pool cv-wait cost dominated by
  `__psynch_cvwait` (~100–1000 μs for cold wake of N workers).
- Sequential factor throughput: ~10 GFLOP/s for the Schur kernel on
  M4 Pro (dense path).
- Break-even: parallel only helps when sequential time per call
  exceeds the per-call rayon overhead. For ~14 workers and ~100 μs
  cv-wait, sequential time should be ≥ ~1–10 ms to amortize.
- 10 ms at 10 GFLOP/s = 10^8 flops.

Therefore: gate parallel dispatch on `total_flops ≥ 10^8` as a first-
cut threshold. The value lives behind a const `PAR_MIN_FLOPS` so it
can be tuned per-CPU in follow-up calibration.

### Composition with existing gates

The new flop gate composes AND with the existing structural gate:

```rust
pub fn should_parallelize_assembly(sym: &SymbolicFactorization) -> bool {
    if sym.supernodes.len() < N_PAR_MIN { return false; }
    if !sym.supernodes.iter().any(|s| s.children.len() >= 2) { return false; }
    estimate_assembly_flops(sym) >= PAR_MIN_FLOPS
}
```

The structural gate stays — it rejects pure-chain trees (which the
parallel driver can't accelerate regardless of flops). The flop gate
adds the work test.

`estimate_assembly_flops` is a free function (no SymbolicFactorization
struct change) — caching is an optional follow-up.

## Design Decisions

**Why not cache per-thread pool.** Issue suggests reusing the rayon
pool across calls. That's a meaningful win on overhead but doesn't
address the underlying mismatch (firing parallel on too-small trees).
The work-aware gate is the more fundamental fix; pool reuse is
complementary and orthogonal.

**Why a hard threshold rather than a per-CPU calibration.** A const
threshold is observable, testable, and easy to tune by editing one
line. A runtime calibration (microbenchmark rayon overhead at startup)
would be more accurate but adds startup cost and complexity. Start
with the const; expose tuning via a `NumericParams::min_parallel_flops`
field in a follow-up if the const proves insufficient.

**Why `ncol * nrow^2` instead of the exact flop formula.** The exact
formula sums three terms and requires computing `nrow_below`. The
proxy is O(1) per supernode and never underestimates (the omitted
panel+TRSM terms are smaller than the Schur term whenever `nrow > 2*
ncol`, which holds for all but trivial supernodes). Order-of-magnitude
gate; precision doesn't matter.

**Default threshold value.** 10^8 from the M4 Pro break-even argument
above. The issue's machine may have a different cross-over but this
is at least a defensible starting point. Calibration on a larger
corpus is recorded as a follow-up.

## Test Strategy

Three unit tests in `numeric::factorize::tests`:

1. **Tiny problem stays sequential** — synthesize a SymbolicFactorization
   with 100 supernodes each of `ncol=2, nrow=4` (4 flops × 100 = 400
   total, way below threshold). Assert `should_parallelize_assembly`
   returns `false` even though `n_snodes ≥ N_PAR_MIN` and the
   structural gate would pass.
2. **Large problem stays parallel** — synthesize 100 supernodes each
   of `ncol=50, nrow=200` (50 × 40000 = 2 × 10^6 per supernode × 100
   = 2 × 10^8 total, above threshold). Assert `true`.
3. **Pure chain stays sequential** — same as test #2 but with a chain
   tree (no multi-child supernode). Assert `false` because the
   structural gate vetoes regardless of flops.

End-to-end measurement on `robot_1600` and `henon120` confirming the
new gate produces the expected dispatch (sequential for robot_1600,
parallel for henon120) — recorded in the session checkpoint.

## Calibration follow-up (not in this commit)

- Run the new heuristic across the dense-IPM-KKT corpus (when corpus
  is available) and record which problems flip parallel↔sequential.
- Re-measure robot_1600 / henon120 / NARX_CFy / 3-4 other Mittelmann
  problems on at least one other CPU (Intel x86-v3 if available) to
  see how the 10^8 const holds up.
- If calibration shows the const is wrong by ≥ 5× on some CPU, expose
  `NumericParams::min_parallel_flops`.

## Out of scope for this fix

- Rayon `ThreadPool` reuse (issue's fix #2). Complementary; tracked
  separately.
- Workspace amortization across refactorizations.
- Removing the per-call cv-wait via custom worker management.
