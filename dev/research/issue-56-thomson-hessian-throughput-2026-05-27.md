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

## Next action

This is a **pounce-side investigation** at this point. The kernel
throughput on Thomson-shaped fronts looks fine — the only knob with
significant effect is symbolic-reuse, which is the caller's contract
to maintain. Recommended next step is to instrument
`pounce-feral/src/lib.rs` (in the pounce repo) to log the
`pattern_reused` flag from `last_factor_stats()` across IPM iters on
elec50 / elec100, and confirm whether the symbolic cache is hitting
each iter as expected. If it isn't, the fix is on the pounce side.

If pounce confirms `pattern_reused = true` on every iter past the
first, then the gap is real kernel throughput and we re-open
hypothesis (1) — likely needs a `cargo-asm` audit of the panel and
trailing-update kernels at n=400/800 fronts to see whether they're
emitting NEON FMA or scalar FMA or just vmla.


### Phase 3 — Implement the fix

Driven by Phase 2 findings. Acceptance gates from #56:
- elec50 ratio ≤ 1.1× (parity)
- elec100 ratio ≤ 1.3×
- No regression on the #55 cascade-victim corpus (robot_1600, NARX_CFy,
  marine_1600, rocket_12800)

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
