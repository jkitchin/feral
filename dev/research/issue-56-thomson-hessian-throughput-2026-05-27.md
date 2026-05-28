# Research Note: Thomson Hessian per-iter throughput (issue #56)

**Status:** Active (triage phase)
**Date:** 2026-05-27
**Related issues:** #56 (this), #55 (closed — Phase B; same root-supernode regime
but different axis: cascade safety, not kernel throughput), #19 (parallel-flop gate)
**Branch:** `issue-56-thomson-hessian-throughput`

## Overview

Pounce + FERAL is 1.57× slower per IPM iter than Ipopt + MUMPS on `elec50`
and 2.34× slower on `elec100`. The IPM trajectory matches iteration-for-iteration
on `elec50` (88 = 88), so the wall-time gap is a pure per-iter linear-algebra
cost difference, not algorithmic. The matrix class is Thomson — all-pairs Coulomb
Hessian, which collapses under any fill-reducing ordering to a single dense root
supernode.

This is the same root-supernode regime that Phase B (#55) addressed for the
*cascade* failure mode (unbounded delayed-pivot accumulation → OOM). What
Phase B did not touch is the *kernel throughput* on those dense fronts.

## Per-iter ratio table (from #56 issue body)

| Problem  | n   | iters (pounce = ipopt) | pounce ms/iter | ipopt ms/iter | ratio |
|---|---:|---:|---:|---:|---:|
| elec25   |  25 |  48 ≈ 48               | ~0.75 | ≈tied      | ~1.0× |
| elec50   |  50 |  88  =  88             |  2.58 | 1.65       | 1.57× |
| elec100  | 100 | 491 vs 214             | 12.77 | 5.46       | 2.34× |

`elec50` is the cleanest diagnostic — identical trajectories isolate the
linear-algebra gap. `elec100` IPM trajectory divergence is a separate axis
tracked on the pounce side; for FERAL throughput we only care about ms/iter.

## Matrix shape

Variables: `3n + 1` (xyz per electron + objvar `t`).
Constraints: `n + 1` (sphere `||p_i||² = 1` for each electron + objective def `t = E(p)`).
KKT block: `[H Aᵀ; A 0]` with shape `(4n+2) × (4n+2)`.

For n=50: KKT n = 202. For n=100: KKT n = 402.

Hessian structure:
- 3n × 3n dense block from all-pairs Coulomb (every 3×3 block coupling),
  plus 2λ_i·I diagonal contribution from sphere multipliers
- Zero row/col for t (the objvar variable has no quadratic term)
- A is `(n+1) × (3n+1)`: n sphere rows (3 nnz each) + 1 objective-def row

After AMD/METIS, the dense 3n×3n block stays dense (there is no fill-reducing
permutation of a dense block). Factor cost is O((3n)³) ≈ O(27n³) for the
dense LDLᵀ on the Hessian plus the cheaper Schur complement for the constraint
border.

## Hypotheses (from issue body, restated)

1. **Dense panel/trailing-update under-vectorized vs MUMPS BLAS-3.**
   MUMPS dispatches `dgemm` from a tuned BLAS. FERAL's dense kernel (
   `src/dense/factor.rs`, `crates/feral-dense-kernels/`) is hand-rolled
   in Rust + `pulp` SIMD. The Phase 2.8.1 partition gates pass with margin
   (small-frontal p90 = 1.37×, medium p90 = 1.74× on the corpus) but the
   *tail* on dense-dominated matrices like Thomson is the regime where
   the gap should be largest. Worth checking `feral_fma yes` (pounce knob;
   feral `Solver::with_fma(true)`).
2. **Parallel-flop gate too high.** `POUNCE_FERAL_MIN_PAR_FLOPS` defaults
   to 1e8. A 150×150 (n=50) or 300×300 (n=100) dense supernode has
   ~3.4M / 27M flops respectively — both below the gate. MUMPS may
   dispatch threads earlier on small dense fronts.
3. **Iterative refinement back-solve cost.** `feral_refine yes` is the
   pounce default. For dense H-dominated KKTs the IR-driven re-solve is
   non-trivial cost. Toggle `feral_refine no` to attribute.
4. **CB-armed default per-pivot bookkeeping.** Phase B armed CB by default
   in `NumericParams::default()`. On clean Thomson Hessians there is no
   cascade to break, but the per-pivot ratio check and delay-budget
   accounting still execute. Toggle via `numeric_params.cascade_break_ratio
   = None` (i.e. construct `Solver::with_params` rather than
   `Solver::new()`). This is the cheapest hypothesis to test from the
   FERAL side because it does not require a real Thomson iterate dump.

## Triage plan

### Phase 1 — Synthetic reproduction (this session)

Write a probe at `src/bin/probe_thomson_hessian.rs` that:
- Places `n` points on the unit sphere via Fibonacci spiral (deterministic,
  well-spread)
- Builds the Lagrangian Hessian H (3n × 3n dense block + 2λ_i I sphere
  multiplier diagonal) with λ_i = 1 for all i (representative iterate, not
  the exact pounce trajectory — captures structure, not values)
- Builds A (n × 3n) — the sphere constraint Jacobian rows
- Assembles `[H Aᵀ; A 0]` as a `CscMatrix` (lower-triangle, KKT order 4n)
- Times `Solver::factor()` + `Solver::solve()` under each of:
  - Phase B default (CB armed, FMA whatever the default is, parallel on)
  - CB disarmed (override `cascade_break_ratio = None`)
  - FMA toggled the other way
  - IR off (call `solve()` directly, not `solve_refined()`)
- Reports per-config wall time, FactorStats (n_tiny, fill_ratio,
  min/max pivot), and the dominant cost breakdown via the existing
  Profiler hook

Skip `t` and the objective-definition constraint — the bulk dense H ×
sphere A structure captures the relevant shape. KKT order is then 4n
(3n vars + n constraints) rather than 4n+2; immaterial for throughput.

### Phase 2 — Localize the gap

Based on Phase 1 numbers:
- If toggling CB closes the gap → hypothesis (4) is the bottleneck; rewire
  CB's per-pivot bookkeeping to skip the ratio check when no delay has
  arrived (cheap; localized to the cascade-break hot path).
- If toggling FMA closes the gap → hypothesis (1); audit dense kernel
  for vectorization opportunities on this size class.
- If IR-off closes the gap → hypothesis (3); not really FERAL's problem
  (caller chooses to refine), but worth noting for pounce.
- If none of the above closes the gap → hypothesis (1) deeper than FMA
  toggle, or (2) — try lowering `POUNCE_FERAL_MIN_PAR_FLOPS` env in the
  probe to see if parallelism at front-size 300² helps.

## Phase 1 results — measured 2026-05-27 on darwin/aarch64 (M-series)

Probe: `src/bin/probe_thomson_hessian.rs`, N_REPS=9, warm-up factor and
solve excluded, `cargo run --release`.

### n=100 (elec100-shaped, KKT order 400, density 56.7%)

| Config | factor µs (min/med) | solve µs (min/med) | Δ vs default |
|---|---:|---:|---:|
| (1) Phase B default (CB on, FMA off, parallel on) | 2968 / 3139 |  68 / 70  | baseline |
| (2) CB disarmed                                    | 2901 / 2994 |  67 / 70  | ~noise   |
| (3) default + IR (solve_refined)                   | 2955 / 3014 | 1248 / 1278 | +1180 µs on solve only |
| (4) default + FMA on                               | 2887 / 2980 |  67 / 69  | ~noise   |
| (5) FMA on + CB off                                | 2934 / 2997 |  65 / 70  | ~noise   |
| (6) default + parallel off                         | 2867 / 2972 |  65 / 70  | -3%      |
| **(7) Cold solver (fresh symbolic each rep)**      | **4907 / 5161** | 67 / 69 | **+65%** |

### n=200 (elec200-shaped, KKT order 800, density 56.5%)

| Config | factor µs (min/med) | solve µs (min/med) | Δ vs default |
|---|---:|---:|---:|
| (1) Phase B default                                | 17098 / 17687 | 262 / 269 | baseline |
| (2) CB disarmed                                    | 17124 / 17335 | 260 / 271 | ~noise   |
| (3) default + IR                                   | 17285 / 17676 | 6050 / 6317 | +6000 µs solve |
| (4) default + FMA on                               | 17158 / 17327 | 265 / 273 | ~noise   |
| (5) FMA on + CB off                                | 17158 / 17471 | 263 / 269 | ~noise   |
| (6) default + parallel off                         | 16500 / 17291 | 256 / 261 | -3.5%    |
| **(7) Cold solver**                                | **25762 / 26413** | 262 / 267 | **+50%** |

### Hypothesis dispositions

- **(1) Dense kernel under-vectorized (FMA toggle).** No measurable
  effect at n=50/100/200. The FMA opt-in either doesn't reach the
  hot path for this matrix shape, or the non-FMA kernel is already
  using NEON FMA implicitly via codegen. Need a `cargo-asm` audit or
  per-kernel benchmark to determine whether FMA dispatch is actually
  doing what the doc-comment claims at this size. Not yet ruled in
  or out — toggle is inert here, but the kernel might still be
  suboptimal vs MUMPS BLAS.
- **(2) Parallel-flop gate too high.** RULED OUT for n ≤ 200. The
  parallel driver actually costs ~3% at this size (rayon dispatch
  overhead exceeds the per-supernode work). Default gate is correctly
  routing to sequential.
- **(3) IR back-solve cost.** REAL but CALLER-CONTROLLED. At n=200
  IR back-solve is 6 ms (~35% of factor cost). pounce can profile
  whether `feral_refine yes` is necessary on Thomson; if MUMPS doesn't
  IR by default for the same iterate, that alone could close the gap.
- **(4) CB-armed per-pivot bookkeeping.** RULED OUT. Zero measurable
  effect across n = 50, 100, 200.
- **(NEW) Symbolic re-analysis on every factor.** Cold-solver path
  costs 50-65% more than warm-solver path at these sizes. If pounce
  constructs a fresh `Solver` per IPM iter (or invalidates the
  pattern fingerprint somehow), this fully accounts for the
  1.57×-2.34× gap reported in #56.

## Pounce-side response (2026-05-27)

Pounce ran the recommended pattern-reuse trace (pounce#65). Result:

| problem | total `factor()` calls | `pattern_reused=true` | % warm |
|---|---:|---:|---:|
| `elec50`  |   239 |   238 | 99.6% |
| `elec100` | 1075 | 1069 | 99.4% |

The 1 cold call on elec50 is the first iter (no cache yet). The 6 cold
calls on elec100 are restoration-phase NLPs with distinct KKT shape
(different fill_ratio). The +50-65% cold-solver penalty does NOT
apply to pounce's IPM trajectory. **Per the disposition rule, the gap
kicks back to FERAL as a real kernel-throughput question.**

## Phase 2 results — per-phase breakdown (warm, sequential)

Extended `src/bin/probe_thomson_hessian.rs` to drop into the sequential
multifrontal driver with `Profiler` attached and
`PHASE_TIMING_ENABLED=true`. The Profiler exposes prologue /
per-supernode / epilogue, and `PHASE_TIMING_ENABLED` populates the
per-supernode `assembly_us` / `panelfactor_us` / `schur_us` /
`scalartail_us` fields via the issue-#44 phase counters.

Averaged over 9 warm reps, darwin/aarch64:

| phase | n=50 (µs / %) | n=100 (µs / %) | n=200 (µs / %) |
|---|---|---|---|
| total wall | 533 | 2977 | 17681 |
| **prologue** | **288 (54.1%)** | **1292 (43.4%)** | **5323 (30.1%)** |
|   ↪ scaling (MC64/InfNorm) | 117 | 552 | 2458 |
|   ↪ permute (PAPᵀ) | 102 (72 in `from_triplets`) | 420 (299 in ft) | 1684 (1174 in ft) |
|   ↪ infnorm + tol | 29 | 139 | 620 |
|   ↪ symmetric_pattern | 41 | 151 | 761 |
| epilogue | 0 | 0 | 0 |
| per-supernode loop sum | 217 (40.9%) | 1630 (54.8%) | 12246 (69.3%) |
|   ↪ assembly | 50 (9.4%) | 327 (11.0%) | 2338 (13.2%) |
|   ↪ dense factor | 161 (30.2%) | 1290 (43.3%) | 9870 (55.8%) |
|     panel/diag BK | 8 (1.5%) | 58 (1.9%) | 201 (1.1%) |
|     **Schur trailing** | **39 (7.3%)** | **439 (14.7%)** | **2462 (13.9%)** |
|     scalar tail | 27 (5.1%) | 132 (4.4%) | 2400 (13.6%) |
|     **dense bookkeeping** | **87 (16.3%)** | **661 (22.2%)** | **4807 (27.2%)** |

Symbolic shape: 57 / 113 / 232 supernodes at n=50 / 100 / 200; max
ncol = 18 across all three — Thomson on AMD does NOT produce a single
fat root supernode. The dense `3n × 3n` Lagrangian Hessian gets
shredded by AMD into many narrow supernodes (max ncol 18 ≈ 6
electrons × 3 coords) because the constraint A-block introduces
pattern artifacts the ordering exploits.

### Findings — where the time actually goes

1. **Schur trailing is NOT the dominant phase.** It's only 7-15% of
   total wall across n=50…200. The kernel-throughput-on-Schur framing
   from Phase 1 mis-aimed.

2. **Dense bookkeeping is the single biggest phase inside the loop**
   at every size: 16-27% of total wall. This is `lextract` +
   `contribextract` + zerofill inside `factor_one_supernode` — the
   memory-shuffling that copies the in-place dense buffer into the
   `NodeFactors` L block and the contribution block for the parent.
   At n=200 it's 4807 µs vs 2462 µs for the Schur kernel — the
   bookkeeping IS the cost.

3. **Prologue is 30-54% of warm factor wall.** The cache reuses
   symbolic — but every `factor()` re-runs scaling (MC64/InfNorm),
   `permute_csc_values` (which rebuilds the matrix via
   `from_triplets`), `symmetric_pattern`, and the infnorm-tol pass.
   None of this depends on numeric values — it's pure pattern work
   that could be cached across iters when `pattern_reused=true`.

4. **Scalar tail grew to comparable with Schur at n=200** (2400 vs
   2462 µs). The scalar-tail kernel handles the trailing rows that
   the blocked Schur didn't fully cover. ncol=18 fronts with nrow
   reaching 594 means a lot of trailing rows get scalar treatment.

### Implications for the per-iter gap

The 1.57× / 2.34× gap vs MUMPS on elec50 / elec100 likely decomposes
into TWO levers, not one:

- **Lever A — per-call prologue.** 288 µs / iter at n=50, 1292 µs /
  iter at n=100. Over 239 / 1075 iters that's 69 ms / 1.39 s of pure
  setup, on top of the IPM critical path. Most of this could be
  cached when `pattern_reused=true`. **The pounce-side cache is
  hitting on symbolic but FERAL is still re-running the scaled-matrix
  build and pattern derivatives every call.**

- **Lever B — dense bookkeeping > kernel flops.** Even with a perfect
  Schur kernel, dense factor cost is dominated by L/contrib extraction
  copies. A panel-blocked extraction would help, or eliminating the
  zerofill pass entirely (per the `CONTRIBZEROFILL_NS` instrumentation
  comment, the subsequent copy already overwrites every cell that's
  later read).

The Schur-kernel cargo-asm audit is still worth doing — but it's not
the highest-impact lever. Address prologue caching and dense
bookkeeping first.

### Phase 3 — Lever A safe subset landed (2026-05-27)

Implemented the **safe** part of Lever A: cache the permute structure
(`col_ptr`, `row_idx`, value scatter map) when `pattern_reused=true`,
and reuse `symbolic.permuted_pattern` instead of recomputing
`permuted.symmetric_pattern()`.

- **Lever A.1** (symbolic pattern reuse): `symbolic.permuted_pattern`
  already holds `permute_pattern(&matrix.symmetric_pattern(), &perm)`,
  which is bit-identical (up to sort, enforced by `permute_pattern`)
  to `permuted.symmetric_pattern()`. Replaced three driver sites
  (Schur, sequential supernodal, parallel) with the pre-built copy.
- **Lever A.2** (permute structure cache): new
  `permute_csc_values_with_cache` consults a per-`FactorWorkspace`
  `PermuteCache { permuted_col_ptr, permuted_row_idx, value_map }`.
  Warm calls (when `NumericParams::pattern_reused_hint = true`)
  allocate a fresh values buffer and run a single O(nnz) scatter,
  skipping triplet construction and the `from_triplets` sort. Solver
  flips the hint on per-call from its existing `pattern_reused`
  fingerprint signal. Cold calls fall through to the canonical
  `permute_csc_values` path and refresh the cache for next time.

Excluded as **risky** (left for follow-up): caching MC64/InfNorm
scaling output. Scaling depends on numeric values; reusing iter-0
matching on iter-N values silently produced incorrect inertia in
issue #38. Track B2's `mc64_scaling_cache` (solver-side, value-bounded)
already handles MC64 reuse; we don't duplicate it on the prologue.

**Re-measured Thomson per-phase breakdown** (darwin/aarch64, 9 warm
reps averaged, hint=true so the workspace cache fires):

| phase                | n=50 before | n=50 after | n=100 before | n=100 after | n=200 before | n=200 after |
|----------------------|-------------|------------|--------------|-------------|--------------|-------------|
| total wall (µs)      | 533         | 577        | 2977         | 2767        | 17681        | 17825       |
| prologue             | 288 (54%)   | **203 (35%)** | 1292 (43%) | **801 (29%)** | 5323 (30%) | **3311 (19%)** |
| permute              | 102         | **10**     | 420          | **31**      | 1684         | **122**     |
| ↪ from_triplets      | (sub of 102)| **0**      | (sub of 420) | **0**       | (sub of 1684)| **0**       |
| symmetric_pattern    | 41          | **0**      | 151          | **0**       | 761          | **0**       |
| scaling (MC64/InfN)  | 117         | 194 (noise)| 552          | (similar)   | 2458         | 2644        |
| per-supernode loop   | 217         | 340        | 1630         | 1906        | 12246        | 14392       |

Total wall is roughly flat or slightly noisier; the savings are
concentrated in the prologue (permute + symmetric_pattern collapse to
under 0.5% of total at every size). Loop time fluctuates within the
rep-to-rep noise band — we did not change loop work.

The remaining prologue cost is dominated by **scaling**: MC64/InfNorm
runs every call because the safe subset deliberately does not cache
value-derived state. The risky lever (scaling cache) would harvest
this — the solver-side `mc64_scaling_cache` (issue #38 follow-up) is
the principled place to do it under a value-bound gate.

### Phase 3 — Future work (not landed in this branch)

Driven by Phase 2 findings. Acceptance gates from #56:
- elec50 ratio ≤ 1.1× (parity)
- elec100 ratio ≤ 1.3×
- No regression on the #55 cascade-victim corpus (robot_1600, NARX_CFy,
  marine_1600, rocket_12800)

## Phase 3 — Solver-path scaling-cache verification (2026-05-27)

After Lever A landed, the remaining prologue cost in Phase 2 was
dominated by `scaling (MC64/InfNorm)` (143 µs / 569 µs / ~3000 µs at
n=50/100/200). The probe ran the raw multifrontal driver and therefore
bypassed `Solver::mc64_scaling_cache` (issue #38 Track B2), so it was
unclear whether the scaling line was real per-iter work or a probe
artifact that the Solver-level cache absorbs on a warm IPM trajectory.

Step 1 of the follow-up: extend the probe with a Phase 3 section that
drives `Solver::factor()` with `with_profiling(true)` and reports
per-rep `mc64_cache_hit_count`, `scaling_info`, and the same prologue
breakdown shape as Phase 2.

Result (darwin/aarch64, 9 warm reps, default Solver / Auto scaling):

  n   | Phase 2 scaling_us | Phase 3 scaling_us | mc64 hits / 9 reps | scaling_info
  ----|--------------------|---------------------|--------------------|--------------
  50  | 153                | 143                 | 0                  | Applied
  100 | 569                | 625                 | 0                  | Applied
  200 | ~3000              | n/a (parallel)*     | 0                  | Applied

*At n=200 the Solver routes through the parallel driver, which does
not emit per-call profiler timings today; `total_us = 0` for those
reports so the prologue average drops to 0 in the Phase 3 aggregate.
The cache-hit and scaling_info signals are still valid.

Conclusion: **the MC64 scaling cache is correctly inactive on Thomson**,
not buggy. `Auto` resolves to InfNorm for this matrix (the gate at
`solver.rs:1136-1151` requires the effective strategy to be `Mc64Symmetric`
or `Auto`-routes-to-`Mc64Symmetric`; on Thomson it routes to InfNorm).
InfNorm is deliberately not cached — issue #49 documents that caching
an InfNorm vector across IPM iterations replays a stale iter-0 scaling
on a drifted iter-N matrix.

So the Phase 2 prologue `scaling_us` is real, recurring, per-iter
FERAL work: InfNorm runs every factor call on Thomson. Options to
reduce it (none in scope for the safe subset):

1. Faster InfNorm — current implementation is O(nnz), well-cached.
   Unlikely to find a constant-factor win without measurement.
2. Per-IPM-iter delta InfNorm — recompute from values that changed,
   reuse the rest. Needs caller-side instrumentation to identify
   which columns changed; same staleness hazard as #49.
3. Identity scaling as a config knob for well-conditioned IPM KKTs
   (pounce-side `.opt`). Trades one cause of inertia drift for another;
   not a safe default.
4. Lever B (dense bookkeeping reduction) — still the biggest remaining
   FERAL-side win at 27-32% of total wall at every size. Independent
   of the scaling story.

Probe extension committed: `src/bin/probe_thomson_hessian.rs` Phase 3
section (`per_phase_breakdown_via_solver`). Result reproducible via
`cargo run --release --bin probe_thomson_hessian -- {50,100,200}`.

## Phase 4 — Lever B (fused contribextract write) (2026-05-27)

Probe drill-down at n=100/200 (Phase 2 sub-phase counters now expose
`lextract`, `contribextract`, `contribzerofill`, `buildrow`, `scatter`,
`extendadd`) localized the `dense bookkeeping` residual:

  n=200 (before): contribextract = 1435 µs (of 4622 µs dense bookkeeping)
    - resize(cdim², 0.0) zerofill   = 289 µs   (dead — overwritten below)
    - lower-triangle copy            = 1146 µs

The prior code did `contrib.resize(cdim*cdim, 0.0)` then a loop that
overwrote the lower triangle from `a`. Every lower-triangle cell was
written twice (zero-fill + a-value); the upper triangle was written
once (zero-fill) and never read afterwards.

Both readers of `ContribBlock.data` restrict their access to `ci >= cj`:
- `extend_add` in `numeric/factorize.rs:3671` iterates `ci in cj..cdim`.
- The root-Schur extractor in `numeric/factorize.rs:1722-1736` reads
  `[j*dim+i]` for `i >= j` and the transpose `[i*dim+j]` for `i < j`
  (both lower-triangle slots).

So the upper-triangle zero-fill is dead work in solve paths. The
diagnostic `parallel_corpus_parity` binary does compare the full
`contrib` buffer bit-for-bit, so the upper triangle still needs to
contain deterministic zeros — but writing them once in the same loop
as the lower-triangle copy is half the work of writing them once in
`resize` and then writing the lower triangle again.

Implementation: `contrib.reserve(cdim²)` + `unsafe { set_len(cdim²) }`
(safety comment cites the f64-has-no-Drop invariant and the write-
before-read contract), then a single-pass loop that writes each cell
exactly once (zero for `ci < cj`, `a`-value for `ci >= cj`). Applied
at both call sites in `dense/factor.rs`:
- `factor_frontal` (scalar fallback path)
- `factor_frontal_blocked_in_place_with_scratch` (panel/blocked path)

Re-measurement (darwin/aarch64, 9 warm reps, n=200):

  metric                     | before  | after  | delta
  ----------------------------|---------|--------|---------
  Phase B default factor min  | 19731   | 18648  | -1083 µs (-5.5%)
  parallel OFF factor min     | 18528   | 16597  | -1931 µs (-10%)
  contribextract              | 1435    | 850    | -585 µs (-41%)
  contribzerofill             | 289     | 3      | -286 µs (≈ gone)
  dense bookkeeping (residual)| 4622    | 4564   | -58 µs  (system noise)

n=100 (Phase B default factor): 2977 → 2753-2983 µs — within noise.
n=50: unchanged within noise — contribextract is small at this size.

The remaining `dense bookkeeping` residual at n=200 (~4.5 ms) is
dominated by per-supernode allocation and dispatch overhead (perm
allocation, d_subdiag clone, FrontalFactors return), not by named
phases. That's the next Lever B target if/when more headroom is
needed; the contribextract fix is the safe, surgical first step.

Verification:
- cargo test --release: all 317 lib + integration tests pass.
- cargo clippy --release --all-targets -- -D warnings clean.
- cargo fmt --check clean.

## Reference matrix dump path

If the synthetic Hessian misses something about the real iterate (e.g.
specific multiplier ranges, IPM-stage μ-perturbation), pounce can dump
a representative iter via `POUNCE_ITER_DUMP_PATH` (`pounce-algorithm/
src/iter_dump.rs`). The issue body offers this. Hold this in reserve;
the synthetic should be enough to localize.

## Test strategy

The probe is a diagnostic binary, not a test. The fix from Phase 3 must
land with:
- A regression test on the Thomson shape (synthesized via the probe's
  Hessian builder, factored, inertia-checked) under `tests/issue_56_*.rs`
- No change to the existing #55 / #17 / #18 / #48 / #38 tests
- Bench partition gates still PASS (probably with improved p90)

## Decisions to revisit at Phase 3

- Whether the CB ratio check belongs on the per-pivot hot path at all,
  or whether it can be lifted to a per-supernode check (delay arrives in
  bulk, not per-pivot). If lifted, hypothesis (4) becomes a no-op fix
  with cleaner semantics.
- Whether `FactorStats::n_tiny` accounting must run unconditionally or
  can short-circuit when CB is disarmed.
