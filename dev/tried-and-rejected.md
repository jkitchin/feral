# Tried and Rejected Log

Append-only. Do not modify existing entries.

---

## 2026-04-12 — Trace-based 2×2 inertia counting fix (deferred)

**What.** Replace the buggy `count_2x2_inertia` in `src/dense/factor.rs:929`
which uses `a00` to decide the sign of the non-zero eigenvalue in the
near-singular branch with `trace = a00 + a11`. The function comment said
"the other has sign of trace" but the code used `a00` alone.

**Why it's a real bug.** For 2×2 blocks where `a00 = 0` (KKT variable rows
have zero Hessian diagonal — common in ACOPP30, FBRAIN3LS, and similar
problem families), the `if a00 > 0.0` check is false and the inertia
falls into the negative branch regardless of what `a11` says. The
mathematically correct sign comes from the trace.

**Why it was deferred.** The fix was attempted during the ACOPP30
triage in this session. Two findings:

1. **It does not fix ACOPP30.** The blocking matrix
   (ACOPP30_0000 D[58]) has both diagonals zero, so trace is also
   zero. The trace-based fix would output `(0, 0, 2)` for the block
   instead of the buggy `(0, 1, 1)`, but neither matches the rmumps
   sidecar's `(72, 137, 0)`. ACOPP30 is fundamentally a different
   pivot strategy issue (delayed pivoting / Phase 2).

2. **It causes a 16-matrix dense regression on the 153k corpus.**
   With the trace-based fix, dense residual pass count drops from
   152717 to 152701. Sparse drops by 1 (152788 → 152787). The
   buggy code happens to be calibrated to rmumps's behavior on the
   regressed matrices, even though it's mathematically wrong. The
   trace fix is more correct in absolute terms but moves feral
   away from the current oracle.

**Decision.** Revert and re-attempt after canonical Fortran MUMPS becomes
available as a second oracle (per `dev/plans/phase-1b-consensus-exit.md`).
At that point we can verify whether canonical MUMPS uses trace-based or
a00-based inertia counting on the 16 regressed matrices and reapply the
fix in the direction that the canonical solver agrees with.

**Code state.** A `KNOWN BUG` comment is left in
`src/dense/factor.rs::count_2x2_inertia` documenting the issue and
linking back here. The function signature remains unchanged so we don't
need `#[allow(clippy::too_many_arguments)]` for code that we know will
need to change again.

**Symptoms.** Inertia error pattern `(p+1, n+1, 0) → (p, n, +1)` on
matrices with zero-diagonal Hessian rows. The "lost positive" appears
as a "gained zero" in feral's output. Most visible on the ACOPP30
family (68 matrices, all with the same `(72,137,0) → (71,137,1)`
mismatch).

---

## 2026-04-13 — Dense ACOPP30: reducible-column floor + Duff-Reid u backstop (rejected)

**What.** Two-part fix for the 67 ACOPP30 variants where dense produces
inertia `(72, 137, 0)` and residual 2.8e-2 while sparse (and MUMPS/SSIDS)
produce `(71, 137, 1)` and residual 1e-14:

- **(a) Duff-Reid u backstop.** In the 2×2 growth check in
  `factor()` (step 7, around line 301), replace
  `let u = params.pivot_threshold;` with
  `let u = params.pivot_threshold.max(f64::EPSILON.sqrt());` so the
  growth bound `(|a22|*rmax + |a10|*tmax)*u ≤ |det|` is not trivially
  satisfied at u=0 for 2×2 blocks with `|det|` near eps².
- **(b) Reducible-column floor.** At the top of the BK main loop
  (step 2), extend `if gamma0 == 0.0` to `if gamma0 ≤ sqrt(eps)` and
  also floor the diagonal: if `|a[k,k]| ≤ sqrt(eps)`, force-zero
  the diagonal and count as zero pivot.

**Why we tried it.** Traced the ACOPP30_0026 factorization to k=58
where the 2×2 block
  `[[ 0, -4.16e-15 ], [ -4.16e-15, -6.08e-9 ]]`
has `|det| = 1.7e-29`, passing `count_2x2_inertia`'s eps² floor by 350×.
At u=0 the Duff-Reid growth check becomes `0 ≤ |det|` (trivially true),
L21 = A21·inv(D) scales by ≈10²⁹, and the trailing submatrix is
destroyed. Fix (a) rejects this 2×2; fix (b) catches the next column
which has a[59,59]=-6e-9 (below sqrt(eps) ~ 1.49e-8).

**Why it was rejected.**

1. **Fix (a) alone makes ACOPP30 catastrophically worse.** When the
   2×2 is rejected by the backstop, the factor fallback calls
   `do_1x1_pivot(&mut a, n, k, gamma0, ...)` with the *same* a[k,k]=0
   diagonal. do_1x1_pivot then divides by 0 (or near-0), producing
   residuals 1e2..1e5 instead of the original 2.7e-2. There is no
   safe 1×1 fallback when both the 2×2 det is tiny and a[k,k] itself
   is zero.

2. **Fix (a) + fix (b) fixes ACOPP30 but causes a 6998-matrix
   regression on the 153k corpus.** After applying both, the
   ACOPP30 triage produces clean residuals (2.8e-2 → 1e-13) and
   matches sparse inertia `(71, 137, 1)`. But the full bench shows
   dense inertia match drops from 152979/154481 (99.0%) to
   146037/154481 (94.5%), dense residual pass drops from
   154141 (99.8%) to 149390 (96.7%), and the worst dense residual
   becomes 1.85e0 on MISTAKE_0101. Total dense failures jump
   from 1838 to 8836.

   Root cause of the regression: the sqrt(eps) absolute floor in
   fix (b) assumes an equilibrated matrix where ||A||∞ ≈ 1. The
   bench corpus is not equilibrated. For matrices where ||A||∞ is
   much larger than 1, legitimate columns with off-diagonal
   couplings ~1e-10 to 1e-8 get flagged as "reducible" and their
   diagonals force-zeroed, destroying otherwise-healthy pivots.
   MISTAKE_0101 output shows expected `(9, 13, 0)` → actual
   `(8, 13, 1)`, i.e. a positive pivot became a zero pivot.

**Decision.** Revert both fixes. The problem is real and specific to
dense single-frontal factorization (the sparse path avoids it via
delayed pivoting into the parent front — `try_reject_1x1_frontal` in
`src/dense/factor.rs:907`). A correct fix must either:

- Implement delayed pivoting for the dense path too (swap the bad
  column k with a downstream row that has a usable diagonal, instead
  of force-accepting an unstable 2×2), or
- Apply a scaled reducible-column floor using the running max
  diagonal magnitude or the matrix inf-norm, not an absolute
  sqrt(eps) threshold.

The triage harness (`examples/triage_dense_acopp30.rs` — committed
as 555b579) and bench cross-comparison metrics (committed as c55bacf)
remain valid infrastructure for the next attempt.

**Evidence.**
- `cargo run --release --example triage_dense_acopp30` after both
  fixes: ACOPP30_{0026,0018,0000} all produce residuals 1e-13..1e-14.
- `cargo run --release --bin bench` after both fixes:
  - Dense inertia match 146037/154481 (94.5%)
  - Dense residual pass 149390/154481 (96.7%)
  - Dense worst residual 1.85e0 on MISTAKE_0101 (expected `(9,13,0)`,
    got `(8,13,1)`)
  - 8836 total dense failures vs baseline 1838
- `cargo run --release --example triage_dense_acopp30` after revert:
  back to baseline 2.8e-2 with inertia `(72,137,0)`.

**Code state.** `src/dense/factor.rs` fully reverted to HEAD
(555b579). The attempted fix is not present in the tree.

---

## 2026-04-14 — Phase 2.4.1a contribution-block deferral (scalar)

**What.** Defer the rank-1/rank-2 updates on the contribution block
`a[ncol..nrow, ncol..nrow]` in `factor_frontal` and apply them as a single
rank-`nelim` triangular update at the end of the routine, keeping the
cross-strip `a[ncol..nrow, k+1..ncol)` updates eager so they remain
available for the next pivot's γ₀ search. Scalar kernel only — no SIMD,
no BLAS — the expected win was cache locality (load the contribution
block once instead of `nelim` times).

**Why it was tried.** MUMPS-style contribution-block deferral seemed
like the minimum-risk split of Phase 2.4.1 after mumps-expert and
spral-expert consultation. Targeted the sparse p90 (3.18 → ≤ 3.0)
for the multifrontal path; useless by construction for the dense path
since `factor_single_front` has `ncol = nrow` and the deferred update
becomes a no-op.

**Implementation.** `src/dense/factor.rs`: added `update_limit` parameter
to `do_1x1_update`/`do_2x2_update` so their rank-1/rank-2 outer loops
stopped at `ncol` instead of `nrow`; added a new
`apply_deferred_contribution_update` helper that built a
`DL[m,j] = (D·L^T)[m, ncol+j]` scratch buffer and then outer-product
updated the lower triangle of the contribution block. Called once
just before L/D/contrib extraction. Correctness preserved (build clean,
80/80 lib tests pass).

**Bench result vs Phase 2.1.8 baseline.** Sparse factor p90 vs MUMPS
regressed **3.18 → 3.53 (+11%)**, sparse p99 **11.40 → 12.03 (+5%)**.
Dense factor p90 moved 2.27 → 2.14 but that is run-to-run noise — the
dense path hits the early-return `ncol >= nrow` branch.

**Why it failed.** The deferred triangular update has *identical*
arithmetic cost to eager per-pivot rank-1 updates on the contribution
block (both are `nelim · cr · (cr+1)/2` flops). Without a SIMD
micro-kernel or a real BLAS-3 GEMM, loop reordering is a no-op on
throughput. The deferred path pays extra for:

1. `Vec::new` allocation of the `DL` scratch per frontal
2. Strided access `a[m*nrow + row]` in the inner `m`-accumulator loop
3. A second pass over the contribution-block memory

For the typical small-front majority in the sparse KKT corpus the
allocation overhead dominates.

**Independent confirmation.** After seeing the bench regression I
consulted faer-expert on the architecture of faer's blocked
Bunch-Kaufman. Verdict: the entire blocked-BK speedup in faer lives
in a pulp-dispatched register-blocked SIMD GEMM micro-kernel
(`matmul_simd` → `Ukr<MR, NR, T>`, x86-v3/v4 feature-gated,
masked-tail loads) that is called from the deferred
`triangular::matmul` at `factor.rs:684`. The panel routine
(`lblt_blocked_step`) itself is plain scalar Rust. This confirms
that copying the deferral loop structure without a vectorized
kernel gives zero speedup — exactly what the bench measured.

**What gets rescued.** Nothing — the `update_limit` parameter, the
`apply_deferred_contribution_update` helper, and the caller
rewiring were all reverted. The original `do_1x1_update`
/`do_2x2_update` signatures are restored. Phase 2.4.1b
(faer-style fully blocked kernel for `factor_single_front`) is
also mooted by the same logic: without a SIMD trailing-update
kernel, the outer panel structure is pure overhead.

**What replaces it.** Phase 2.4.2 (SIMD micro-kernel for the
Schur update) becomes the only remaining lever for the
`dense factor p90 ≤ 2.0` target. Open question pending user
direction: write `#[target_feature]` + `core::arch` intrinsics
(x86_64 AVX2/FMA + aarch64 NEON behind `cfg` gates) vs. accept
`pulp` as a dependency.

**Evidence.** Full bench output: session 2026-04-14-01 at
15:00 in `dev/journal/2026-04-14-01.org`. Reverted in same
session. Head commit: `ce09aa6` (pre-2.4.1a) remained the
measured baseline.

## 2026-04-14 — Phase 2.4.2 unroll4 FMA Schur-kernel wired into do_1x1_update/do_2x2_update (reverted)

**What.** Wire `schur_kernel::axpy_minus_unroll4` and
`axpy2_minus_unroll4` — the pulp-dispatched 4-way-unrolled NEON
kernels with independent FMA accumulators — into the rank-1 and
rank-2 Schur-update inner loops at `src/dense/factor.rs`
`do_1x1_update` and `do_2x2_update`, replacing the scalar loops that
had been autovectorized by rustc.

**Why it seemed to work.** The full KKT bench hit both Phase 2.8
exit targets simultaneously on the first run with unroll4 wired in:

| metric                  | baseline 2.1.8 | unroll4 | target |
|-------------------------|---------------:|--------:|-------:|
| dense factor/MUMPS p90  |           2.27 |    1.87 |  ≤ 2.0 |
| sparse factor/MUMPS p90 |           3.18 |    2.82 |  ≤ 3.0 |

Dense inertia match was byte-identical (152911/154481). Gains of
−18% on dense p90 and −11% on sparse p90 from hand-unrolling with 4
independent FMA accumulators, exposing extra ILP to the M-series
dual FMA pipes that the single-accumulator autovectorized scalar
loop could not.

**Why it was reverted.** Sparse inertia match dropped from
153009/154588 to 153005/154588 (−4 matrices) and sparse residual
pass dropped from 154329 to 154303 (−26 matrices). Per the
`USE_SIMD_SCHUR_KERNEL` runtime-flag triage (example
`triage_sparse_inertia_diff.rs`, deleted on revert), all four
inertia regressions are FMA rounding boundary flips:

| matrix          | expected      | scalar match  | unroll4        |
|-----------------|---------------|---------------|----------------|
| ACOPP14_0001    | (38, 68, 0)   | (38, 68, 0) ✓ | (37, 69, 0)    |
| ACOPP30_0004    | (72, 137, 0)  | (72,137, 0) ✓ | (71,138, 0)    |
| FBRAIN3LS_0848  | ( 6,  0, 0)   | ( 6,  0, 0) ✓ | ( 5,  0, 1)    |
| FBRAIN3LS_0851  | ( 6,  0, 0)   | ( 6,  0, 0) ✓ | ( 5,  0, 1)    |

The ACOPP cases are single-pivot sign flips (positive pivot crossed
zero to negative). The FBRAIN cases are single-pivot magnitude
drops below `zero_tol` (positive pivot classified as zero). Both
patterns are the classic FMA-vs-scalar 1-ULP rounding difference:

- scalar path: `d[i] -= α·s[i]` → `round(d − round(α·s))`, two roundings
- FMA  path: `mul_add_f64s(−α, s, d)` → `round(−α·s + d)`, one rounding

When a final pivot lands within 1 ULP of 0 or `zero_tol`, the
one-vs-two rounding delta flips the Bunch-Kaufman classification
from accept → zero-rank, or positive → negative. **Zero SIMD
improvements** (simd-only-match count was 0), confirming the math is
the same in both paths — only the rounding differs. The 0-delta
`both_match = 153005` + 4 regressions adds up to the exact scalar
baseline of 153009, confirming no genuine correctness changes.

**Assessment.** −30 matrices total regression on a 154588-matrix
corpus (0.019%) is statistical noise at the population level, but
CLAUDE.md's hard rule is *"Correctness before performance, always"*.
The regressions are per-matrix deterministic, not flaky — a user
running FERAL on one of those four ACOPP/FBRAIN KKT systems would
get the wrong inertia on every call. The Phase 2.8 exit criterion
wins do not justify shipping a known inertia regression without
mitigation.

**What gets rescued.** The `factor.rs` scalar Schur-update loops
were reverted to the exact HEAD form (`git diff src/dense/factor.rs`
is empty after revert). The `USE_SIMD_SCHUR_KERNEL` runtime flag
and the `examples/triage_sparse_inertia_diff.rs` triage binary were
deleted. What remains in-tree for a possible Phase 2.4.3 retry:

- `src/dense/schur_kernel.rs` — the pulp kernels and 11 ULP4 unit tests
- `benches/schur_kernel.rs` — the scalar/pulp/direct_neon/unroll4 microbench
- `pulp = 0.22.2` dev-dependency in `Cargo.toml`

**What replaces it (open question).** Two candidate mitigations for
Phase 2.4.3:

1. **Non-FMA unroll.** Replace `mul_add_f64s` with separate
   `mul_f64s` + `sub_f64s` in `axpy_minus_unroll4`/`axpy2_minus_unroll4`.
   Reproduces scalar rounding exactly → byte-identical inertia. Costs
   the single-cycle FMA → 2-op latency; unclear how much of the ILP
   gain from 4 independent accumulators survives.
2. **Pivot-boundary scalar fallback.** Detect pivots where the
   Schur-updated diagonal would be within `k·eps` of 0 or `zero_tol`
   and run the tail of the update loop scalar. Complex to implement
   correctly and may not catch all flips.

Option 1 is the cheaper experiment. Requires a second full KKT bench
to confirm zero inertia regressions and to measure how much of the
2.27→1.87 and 3.18→2.82 p90 gains are preserved without FMA.

**Evidence.** Full bench output with unroll4 wired in:
`/tmp/feral_bench_unroll4.txt` (session 2026-04-14-01). Triage
output: `/tmp/feral_inertia_triage2.txt`. Triage binary source:
preserved in git log of deleted `examples/triage_sparse_inertia_diff.rs`.
Session: `dev/sessions/2026-04-14-02.md` (to be written at session end).

---

## 2026-04-14 — Single-run p90 readings as primary signal during Phase 2.5.1′ iteration

**Approach.** During session 04 I judged optimization patches by
comparing before/after single-run `cargo run --release --bin bench`
p90 readings. The etree-renumbering fix in `src/symbolic/mod.rs` was
applied, measured as 2.08 vs a prior single-run 2.02, flagged as a
regression, and reverted. A subsequent 3-run sanity check showed
the actual without-fix baseline was 2.12/2.12/2.14 and with-fix was
2.03/2.06/2.08 — a real ~3% improvement. The fix was re-applied.

**Why it failed.** The sparse small-frontal p90 on the full 153455
matrix bucket has ~3–5% run-to-run noise, larger than the typical
single-fix improvement at this stage of optimization (~1–3%).
Single-run deltas inside that window are indistinguishable from
noise. Treating a single reading as ground truth led to reverting
a correct optimization and wasted a full iteration.

**Why the underlying fix was still correct.** Postorder is a
topological relabeling of the elimination tree, so
`etree(P·A·Pᵀ) = post-renumbering of etree(A)` when P is a
postorder of `etree(A)`. The second `from_pattern` call was
genuinely redundant; the measured improvement is real, it just had
to be measured as a 3-run median.

**Lesson.** Any decision based on a sparse-bench p90 delta smaller
than ~5% must be confirmed with at least a 3-run median. Single-run
readings are fine for sanity-checking that a change didn't cause a
catastrophic regression (e.g., 2.00 → 5.00), but not for judging
sub-noise-level optimizations. Specific to the sparse-bucket p90 on
the small-frontal partition; other metrics (max, geomean) have
different noise floors.

**Evidence.** Session journal `dev/journal/2026-04-14-04.org` entry
14:55 (etree renumbering); session checkpoint
`dev/sessions/2026-04-14-04.md` "Abandoned Approaches" section.

---

## 2026-04-17 — feral-metis FM neighbour-update sign bug (test gate)

**What.** `crates/feral-metis/src/fm_refine.rs:115/117` updates
neighbour gains with the wrong sign for the `gain = ed − id`
convention used by `compute_gains` and the cut update
`cur_cut -= gain[v]`. Discovered while implementing feral-scotch
halo FM (S3) — the corrected signs landed in
`feral-scotch/src/halo_fm.rs` and `band_fm.rs`.

**Why it slipped through.** All four FM tests in fm_refine.rs miss
the bug for structural reasons:

1. `refine_bisection_does_not_increase_cut`: `initial_bisect_ggp(grid(8,8))`
   already returns the optimal cut of 8, so FM is a trivial no-op
   (`initial=8 returned=8 actual=8`). The `final ≤ initial`
   assertion holds vacuously.
2. `refine_bisection_bad_init_improves`: `max_imbalance = 0.20` on
   n = 9 makes every candidate move violate the balance guard, so
   `best_prefix` stays 0 and the function returns the input cut.
3. `refine_bisection_balance_respected`: checks weights only.
4. `nd_order_*`: validate permutation bijectivity, not cut quality.

None of the tests assert the bookkeeping invariant
`returned_cut == cut_size(graph, labels)`. Adversarial input P_10
with alternating ABAB labels (cut = 9) produces `returned = -1143`
with labels unchanged — impossible negative cut, but the test would
pass `after < before` (because -1143 < 9).

**Lesson.** Two structural test-design failures combined: (a) using
oracle inputs (optimal-cut graphs and balance-blocked configurations)
that don't exercise the code under test, and (b) checking the
return value against itself rather than against an independent
re-derivation (`cut_size(labels)`). Any solver that maintains
incremental state must assert that incremental state matches a
from-scratch recomputation, at least once per test.

**Status.** Bug + comprehensive test plan (invariants I1–I7,
adversarial cases A1–A10) documented in
`dev/research/metis-fm-sign-bug.md`. Fix and test hardening are
listed there as actions 1–4. Not done in this session because the
session goal was feral-scotch S2–S5; deferring to keep the metis
fix and its regression tests as a single self-contained commit
backed by the documented plan.

**Evidence.** `dev/research/metis-fm-sign-bug.md` §1 (sign
derivation), §2 (adversarial output `before=9 after=-1143`), §3
(per-test analysis of why each existing gate misses).

---

## 2026-04-18 — feral-kahip K1 Rules 2-4: catastrophic fill regressions on bakeoff corpus

**Symptom.** Wiring K1 data reduction (all four Ost-Schulz-Strash
rules enabled) into `kahip_nd_order` caused fill to explode on
several matrices in the bakeoff corpus, even after fixing a Rule-2
expansion bug (see below). Concrete regressions (KaHIP fill vs AMD):

|           | before K1 | after K1 (all rules) | after (Rule 1 only) |
|-----------|----------:|---------------------:|--------------------:|
| vesuvia   | 1.002×    | 25.86×               | 1.000×              |
| vesuvio   | 1.003×    | 51.62×               | 1.000×              |
| vesuviou  | 1.002×    | 41.89×               | 1.000×              |
| cresc132  | 0.609×    | 95.31×               | —                   |
| c-big     | 3.29×     | 3.92×                | 2.59×               |

Geomean KaHIP/AMD fill across 41 matrices: **1.032** (no K1) →
**2.094** (all rules) → **1.023** (Rule 1 only).

**Rule-2 expansion bug (fixed).** The original expansion anchored
Rule-2 path interiors to endpoint `u` only. Fill-preservation
requires the path to be eliminated before BOTH endpoints — when
`pos(w) < pos(u)` in the reduced perm, `w`'s elimination happens
while the path still exists, producing extra fill. Fixed by
anchoring the path to whichever of the two endpoints' ultimate
(path-compressed) anchors has the lower reduced-perm position. This
reduced geomean fill from 2.094 → 1.876 but did not recover
vesuvio/vesuviou/cresc132.

**Rules 2-4 disabled (unresolved).** Isolating rules by toggling
`ReduceOptions` showed that disabling Rules 2-4 entirely (Rule 1 only)
recovers all regressions and actually makes KaHIP the best-on-average
ordering (geomean 1.023 vs AMD 1.000, METIS 1.024, SCOTCH 1.038).
The exact mechanism by which Rules 2-4 produce 40-50× fill on
vesuvio/vesuviou is not understood — the Ost-Schulz-Strash rules are
claimed fill-preserving in the paper. Candidate explanations:

1. **Open-twin interaction with partitioner.** Open twins merge
   vertices with shared open neighborhood into a single rep. The
   partitioner then sees a rep of weight 1 but with the merged
   neighborhood influence. If multiple twins merged to the same rep
   end up in different partitions in the expanded graph, fill
   propagation differs from the reduced-graph analysis.

2. **Subset cascade on dense subgraphs.** Rule 4 can eliminate large
   numbers of vertices in dense subgraphs. The surviving core
   becomes a denser clique-like structure that partitions poorly.

3. **Rule-2 fill-edge accumulation even in simplicial case.** After
   many simplicial compressions, `u` accumulates neighbors that
   weren't structurally connected. Even without adding a fill edge,
   the reduced graph's connectivity changes the partitioner's
   decisions.

**Current status.** Driver uses `ReduceOptions::conservative()`
(Rule 1 only). Rules 2-4 remain implemented, unit-tested (via
`ReduceOptions::full()`), and internal to the crate. Re-enabling
them requires first understanding the fill-blow-up mechanism — most
likely by comparing `symbolic_factor(original, expanded_perm)`
versus `symbolic_factor(reduced, reduced_perm)` on vesuvio to
localize where the fill diverges.

**Lesson.** Claims of "fill-preserving reduction" in papers require
the elimination order on the reduced graph to respect implicit
structural invariants that are not obvious from the rule statements
alone. Validate reductions via a symbolic-factor equivalence test,
not just permutation-bijection tests, before wiring them into a
production ordering driver.

**Evidence.** Session 2026-04-18-06 bakeoff runs; commits subsequent
to `023913c symbolic: wire OrderingMethod::KahipND into bakeoff`.

---

## 2026-04-18 — `OrderingMethod::Auto` as the default for `symbolic_factorize`

**What.** Flip `symbolic_factorize`'s default from `Amd` to `Auto`,
where `Auto` picks per-matrix from cheap features (n, nnz/n) — small
& sparse → KaHIP, large & sparse → SCOTCH, otherwise AMD. Motivated
by a 41-matrix shape bakeoff in which Auto won 41/41 on min-fill and
posted the best geomean (0.988× AMD).

**Why it failed.** The full 154,588-matrix IPM KKT bench showed Auto
*regresses* end-to-end:

| metric                      | AMD baseline | Auto |
|-----------------------------|-------------:|-----:|
| sparse factor/MUMPS geomean |        0.44  | 0.58 |
| sparse factor/SSIDS geomean |        0.02  | 0.03 |
| solve/MUMPS geomean         |        0.46  | 0.46 |

The shape bakeoff had one matrix per family, mostly n > 200. The IPM
corpus has thousands of small (n=5, n=8, n=157, …) iteration dumps
per family. Auto's `n < 10_000 && nnz/n < 15` rule routes all of them
to KaHIP, where the K1 + multilevel setup costs 2-3× per call vs AMD.
Per-call symbolic cost from the shape bakeoff already showed the
warning sign: at n~700 KaHIP took 520-760μs vs AMD's 250-450μs;
total time only netted out because cresc132 (n=5314, KaHIP 0.607×)
dominates the small corpus.

The fill geomean (0.988) is real but does not translate to numeric
factor speedup when the workload is dominated by tiny matrices —
`factorize_multifrontal` time on n=5 is dominated by symbolic-phase
overhead that AMD's O(n) inner loop wins by an order of magnitude.

**Resolution.** `Auto` stays in the public API as opt-in via
`symbolic_factorize_with_method`. `symbolic_factorize` keeps the
`Amd` default. The doc comment on `OrderingMethod::Auto` warns
callers about the per-call overhead and points here.

**What would change the calculus.** A heuristic that requires
n ≥ 5000 (or detects K1-fireable structure cheaply) before routing
to KaHIP could recover the cresc132-class wins without paying the
per-call tax on small matrices. Not pursued now — the IPM workload
profile makes the upside small.

**Evidence.** `/tmp/bench_amd.log` and `/tmp/bench_auto.log` from
2026-04-18 session continuation; commit `bc6ec82`.

---

## 2026-04-18-08 — Routing VESUVIO to MetisND

**Hypothesis.** VESUVIO's 84× factor ratio vs MUMPS could be a
bordered-KKT pathology like CRESC132, where AMD orders the
constraints into a near-dense root frontal that MetisND breaks up.

**Test.** `src/bin/vesuvio_diag.rs` ran symbolic + numeric
factorization under both AMD and MetisND on 5 VESUVIO samples
(VESUVIOU_0000, VESUVIOU_0005, VESUVIO_0000, VESUVIO_0021,
VESUVIA_0000), with CRESC132 as the positive-control reference.

**Result.** MetisND helps marginally on two samples (-5%, -8%)
and is slower on the other three. Both orderings produce the
same 67%-of-n root frontal because VESUVIO has a single dense
linking column (max_col_nnz=1026, diag_only=1025); any reasonable
ordering pushes it to the root. The factor cost is dense-kernel
limited, not ordering-limited. CRESC132 by contrast drops 96%→50%
under MetisND because it has thousands of dense constraint
columns that AMD bunches into one mega-supernode.

**Verdict.** No new dispatcher rule for VESUVIO. The remaining
factor-tail gap is `src/dense/factor.rs` work (blocked BK + SIMD).

**Evidence.** `dev/journal/2026-04-18-08.org` second entry, commit
`86cf1e8`.

## 2026-04-18-08 — Adding a narrow KaHIP rule to `pick_default_method`

**Hypothesis.** A narrow rule (e.g. by stored_nnz/n class or
specific arrow-pattern detector) could route some IPM family
where K1 + multilevel ND beats AMD or METIS.

**Test.** Re-ran `bench_orderings` (41 matrices, parity ∪ large)
including KahipND for the first time at corpus scale. Compared
fill counts and per-call symbolic time vs AMD/MetisND/ScotchND.

**Result.** KaHIP-with-K1 ties METIS on fill (geomean 1.023 vs
1.024 relative to AMD) at 1.2× METIS's per-call cost. Strict-fill
wins of KaHIP over AMD on only 4/41 matrices, and in every case
KaHIP merely ties the best other ordering rather than beating it.
On the IPM corpus the existing `n>=5000 && nnz/n<6 → MetisND`
rule already captures the fill wins KaHIP could provide; the
extra per-call setup cost is unrecouped.

**Verdict.** Status quo. KaHIP remains opt-in via
`symbolic_factorize_with_method` and `OrderingMethod::Auto`.
Pinned by `pick_default_method_never_returns_kahip` test.

**Evidence.** `dev/research/ordering-kahip-driver-integration.md`,
commit `b5c67cb`.

## 2026-04-19 — 2-condition Policy 4 rule (no raw_diag_range guard)

**Hypothesis.** Catching the MSS1_0009 fallback case needs
only `mc_off > 1e6 ∧ mc_off / in_off > 1e5`. The
`policy4_diag` 14-matrix panel showed 2.5 orders of
magnitude separation between MSS1 (mc/in ratio 3.9e6) and
the nearest "keep MC64" matrix (VESUVIOU at 1.05e4); a
1e5 threshold sits comfortably in the gap.

**Test.** Implemented in `compute_scaling_auto`, ran
`cargo test --release`.

**Result.** False-positives on MEYER3NE_{0220, 0259, 0253}
parity tests. MEYER3NE_0220 has mc_off = 8.56e13, in_off =
9.40e6, ratio 9.1e6 — well above the 1e5 threshold. But
unlike MSS1_0009, MEYER3NE has raw_diag_range = 4.77e19
(ill-conditioned raw matrix where MC64 is the only scaling
that produces a usable factor). Falling back to InfNorm on
MEYER3NE drove residuals to 4.77e15.

**Verdict.** Replaced with 3-condition rule adding
`raw_diag_range < 1e6` as a first-line guard. The shape-only
diagnostic works ONLY when combined with a measure of the
raw matrix's conditioning. See `dev/decisions.md`
(2026-04-19 Policy 4 entry).

**Evidence.** `dev/research/policy-4-scaling-fallback.md`
§5.1, commit `af9315d`.

## 2026-04-19 — `nemin` tuning to fix AVION2/BATCH families

**Hypothesis.** AVION2 (geomean 1.61, 2682 matrices) and
BATCH (1.85, 2054 matrices) lose to MUMPS on average.
Possibly the default `nemin=32` (matching SSIDS) is too
aggressive for these small-n matrices; MUMPS uses `nemin=5`.
Smaller nemin → smaller, more focused supernodes → less
zero-padding in frontal matrices.

**Test.** Added `FERAL_NEMIN` env-var override to
`profile_sparse`. Ran on AVION2_{0000, 0500, 1500} and
BATCH_{0000, 0500, 1500} at `nemin ∈ {1, 5, 32}`.

**Result.** `nemin=32` (current default) is at the optimum.
`nemin=5` is roughly tied or slightly worse;
`nemin=1` (no amalgamation) regresses by 30-40%:

| matrix       | n   | fac µs nemin=32 | nemin=5 | nemin=1 |
|--------------|----:|----------------:|--------:|--------:|
| AVION2_0000  |  94 |              35 |      33 |      48 |
| BATCH_0000   | 121 |              80 |      82 |      92 |

**Verdict.** The AVION2/BATCH gap is structural multifrontal
scaffolding overhead at small n, not amalgamation policy.
Lever D.1 (FactorWorkspace arena) is the right next attempt.

**Evidence.** `dev/research/sparse-tail-perf-2026-04-19.md`
§5b, commit `8e68482`.

## 2026-04-20: Phase 2.5.2 parallel driver root-cause — per-thread workspace theory

**Context.** Multi-thread rayon driver has ~1-2 % inertia mismatch
vs sequential on the KKT corpus; single-thread rayon gives 0 /
38 878 mismatches. First hypothesis: per-thread FactorWorkspace is
handing off dirty state (e.g. row_map invariant not restored) across
tasks scheduled on the same worker.

**Tried.** Replaced `Vec<Mutex<FactorWorkspace>>` (one per worker +
one for caller) with a single global `Mutex<FactorWorkspace>` so
every `factor_one_supernode` call serialises. Also tried
`FORCE_SCALAR_FRONTAL=true` to bypass pulp SIMD dispatch. Both
experiments still reproduced the race:

- Single global workspace: 5 / 364 matrices (~1.4 %).
- Scalar dense kernel: 5 / 279 matrices (~1.8 %).

**Why rejected.** Neither rule-out fixed the race, so the root cause
is neither workspace lifecycle nor SIMD nondeterminism. The race
must live in a non-obvious part of the parallel orchestration
(atomic ordering, rayon::scope happens-before subtleties, or shared
data read outside mutex protection that I haven't spotted).

**Evidence.** `src/bin/diag_acopr.rs`; `dev/journal/2026-04-20-11.org`
entries 00:20 and 00:30.

---

## 2026-04-23 — Flipping `SupernodeParams::default().preprocess` to `LdltCompress`

**Context.** Phase 2.4.4 dense-tail diagnostic (commit 332f23a) showed
`OrderingPreprocess::LdltCompress` produces 2–5× factor-time wins on
the worst matrices (HAHN1, CRESC100, GAUSS2, MUONSINE, VESUVIO).
`diag_compression_bench` across a 321-matrix stratified sample
reported factor geomean cmp/base = 0.758 — apparently clearing the
Phase 2.6.5 plan's ≤ 0.95 flip-default threshold.

**Tried.** Changed `SupernodeParams::default()` to use
`OrderingPreprocess::LdltCompress`. Ran the full 154,481-matrix
bench.

**Result (bench, sparse factor/MUMPS):**

| metric  | pre-flip | post-flip | delta   |
|---------|---------:|----------:|--------:|
| geomean | 0.36     | 0.49      | +36%    |
| p90     | 1.61     | 1.91      | +19%    |
| max     | 9.40     | 12.93     | +38%    |

All three metrics moved the wrong direction. Regression across the
board.

**Why rejected.** The `diag_compression_bench` 0.758 number was
misleading. It times symbolic and numeric *separately* and reports
only the numeric ratio. The real bench harness (`bench.rs` line
1281–1284) combines symbolic + numeric in `factor_us` to match
MUMPS's oracle JSON (single `factor_us` covers analysis + numeric).

Compression roughly doubles symbolic time (diag evidence: HAHN1_0153
sym 616→798 μs, GAUSS2_0035 211→285 μs, KIRBY2_0007 added ~17×
symbolic per compression_bench). On tail matrices (ms-range numeric)
this is noise. On bulk matrices (sub-ms numeric) the ~100-400 μs
symbolic overhead is the whole thing, and geomean over 154k matrices
propagates the penalty.

**Evidence.** `dev/journal/2026-04-23-02.org` entries 21:40 (initial
claim) and 22:05 (bench refutation). Commit was never made — the
flip existed only in working tree.

**Corrected path forward (not pursued yet).**

1. Make `ldlt_compress` symbolic work faster (the MC64 matching
   piece is already cached inside `compute_symmetric` scaling and
   could be plumbed through to avoid the double-Hungarian).
2. Auto-dispatch: enable compression only when heuristics predict a
   tail matrix (large-n + nontrivial MC64 compRat ≤ 0.7), same
   pattern as `ScalingStrategy::Auto`.
3. Separately: the "Dense" bench column uses `factor_single_front`
   (whole-matrix dense LDLT, no symbolic at all), so
   `SupernodeParams` changes have zero effect on it — the 53×
   "Dense max" is measuring feral's dense kernel vs MUMPS's sparse
   multifrontal and is not apples-to-apples.

## 2026-04-23-02: flip `LdltCompress` default after MC64 cache refactor — still rejected, geomean regresses

**What was tried.** Implemented the "speed up `ldlt_compress`
symbolic" path flagged in the 2026-04-23 entry above:
`SymbolicFactorization::cached_mc64` holds the full MC64 matching;
numeric's `compute_scaling_with_cache` reuses it instead of rerunning
Hungarian. Then flipped `SupernodeParams::default().preprocess` to
`LdltCompress` and re-ran the 154,588-matrix bench.

**Result.** Compared against the prior no-cache flip:

    metric   pre (None)   flip no-cache   flip with cache
    geomean        0.36           0.49             0.48
    p90            1.61           1.91             1.75
    max            9.40          12.93            10.42

Cache recovered ~71% of the `max` gap and ~55% of `p90` but only ~8%
of `geomean`.

**Why still rejected.** The cache only helps matrices where
`ScalingStrategy::Auto` resolves to `Mc64Symmetric`. On the
arrow-KKT families (VESUVIO/VESUVIOU/CRESC132/MUONSINE) it does,
and max + p90 improve. On the ACOPR30 family — 9 of the top-10
worst post-flip at ~9.5× — `diag_only/n < 0.3` routes Auto to
`InfNorm` and the compression MC64 has no sharing partner. The
structural compression overhead (supermap + compress_pattern +
ordering-on-compressed-graph) is still unamortized on small and
medium matrices, and the bulk of the corpus lives there.

**Disposition.**

- The cache refactor itself is *kept* — it's correct, it's a
  legitimate speedup on opt-in compression + MC64 scaling, and it
  has no downside. Committed as eea9f19.
- The default flip is *reverted* in the same commit — geomean 0.36
  → 0.48 is not acceptable as a blanket default.

**Proper next step.** Shape-based auto-dispatch for compression,
parallel to `pick_scaling_strategy`. Only run compression when
predicted to pay off: large-n + `Auto` picks `Mc64Symmetric` +
cheap heuristic says `ncmp < 0.9*n`. This isolates the tail wins
from the small-matrix geomean penalty. Flagged for a future
session, not this one.

**Evidence.** `dev/journal/2026-04-23-02.org` entry 22:50. Full
bench output kept locally at `/tmp/feral-bench-cache-flip.txt`.
143 tests pass after the revert.


## 2026-04-24 — Phase 2.9 SmallLeafSubtree batching (naive specialisation)

**Approach.** Precompute true-leaf supernode row-indices at symbolic
time; at numeric time, dispatch grouped leaves to a specialised
`factor_one_small_leaf` that skips `build_row_indices` and the
empty children loop. Corresponds to Steps A–E of
`dev/plans/phase-2.9-small-leaf-subtree.md` (minus the arena
allocator in the original research sketch).

**Outcome.** Correct (bit-exact parity on ACOPR30, CRESC100, HAIFAM,
VESUVIO plus block-diagonal fixtures) but **essentially no speedup**.
Geomean across 9 archetype matrices: ~1.00×. Worst case
VESUVIO_0000 at 0.95× (below noise). Step F bar was 3×; we are at 1×.

**Why rejected.** The per-front overhead on tiny fronts is *not*
in `build_row_indices` or the children-loop dispatch. It is in:
the `frontal_buf.resize(n*n, 0.0)` memset (separate call per
member, not amortised across a group); the `factor_frontal_blocked`
blocked kernel itself on ncol ≤ 8; and per-front BK bookkeeping.
The naive "precompute rows + skip no-op loop" specialisation touches
none of these. See `dev/journal/2026-04-24-01.org`.

**Disposition.**

- The gate and the specialised numeric path are *kept* in the source
  tree, gated `Off` by default. They carry zero runtime cost when
  disabled and will serve as scaffolding for Phase 2.9.2 (true
  stack arena / shared allocation across group members).
- The default flip (Step F of the plan) is *not performed*. Flipping
  it now would cost nothing but gain nothing; keep the simpler
  scalar path as the default.

**Proper next step.** Phase 2.9.2: implement a shared-arena
allocation strategy where all members of a leaf group write into
one contiguous slice that is memset once per group. This requires
`factor_frontal_blocked` to accept a backing `&mut [f64]`
instead of owning a `SymmetricMatrix`. Non-trivial kernel
refactor.

**Evidence.** `src/bin/diag_small_leaf` output in journal
`2026-04-24-01.org`; `tests/small_leaf_parity.rs` (7/7 pass).

## 2026-04-24 — Phase 2.9.2: `factor_frontal` arena refactor (REJECTED at Step A gate)

**What we tried.** Step A of `dev/plans/phase-2.9.2-factor-frontal-arena.md`:
instrument `factor_frontal` with a `FrontalProfile` sink (added as
`factor_frontal_with_profile(..., Option<&mut FrontalProfile>)` in
`src/dense/factor.rs`; existing `factor_frontal` is a None-passing
wrapper) to measure the removable fraction before committing to
the arena refactor.

**Result (1832 leaves × 50 repeats across ACOPR30/CRESC100/HAIFAM/VESUVIO).**

| sub-phase       | %bk_total | %inner |
|-----------------|-----------|--------|
| alloc+copy      |      9.7% |  17.7% |
| setup           |      7.8% |  14.2% |
| pivot_loop      |     17.6% |  32.0% |
| extract         |     19.9% |  36.1% |
| meas overhead   |     ~45%  |   —    |

Removable by the plan (alloc+copy + setup) = 17.6% of bk_total,
below the 25% gate. Best-case per-leaf speedup from eliminating
all of it: 1.22× × 52% bk-share = ~1.12× overall. Target was 1.5×.

**Why rejected.** The arena refactor targets only the caller-supplied
scratch (`a`, `perm`, `subdiag`, `d_panel`). It does not address
the `extract` phase (5 owned Vecs in `FrontalFactors`: `l`, `d_diag`,
`d_subdiag`, `contrib`, `perm_inv`) which is the largest allocation
phase at 19.9% of bk_total. It does not address `pivot_loop` (32%
of inner) which is actual arithmetic. The 10× gap vs MUMPS on
ACOPR30 is not sitting inside `factor_frontal`.

**Disposition.**

- The diagnostic hook `factor_frontal_with_profile` + `FrontalProfile`
  struct is *kept* in the source tree. Zero runtime overhead when
  unused (production `factor_frontal` passes None), valuable for
  future kernel triage.
- No `FrontalScratch` / `factor_frontal_into` are added. The plan
  in `dev/plans/phase-2.9.2-factor-frontal-arena.md` is closed.
- The Phase 2.9 small-leaf gate remains Off.

**Proper next direction.** The per-front gap is not in the dense
kernel. Investigate:
1. Scatter indirection / outer multifrontal driver bookkeeping
   (per-child loop, build_seen, etc.).
2. Supernodal amalgamation budget — MUMPS/SSIDS amalgamate more
   aggressively to produce fewer, larger fronts that shift cost
   from the long-tail leaf population into the BLAS-friendly bulk.
3. Nested-dissection vs AMD ordering choice on these matrices.

**Evidence.** `dev/journal/2026-04-24-01.org` entry 16:45,
`src/bin/diag_leaf_profile` output with sub-phase section.

## 2026-04-25 — Phase 2.11 Option B: SmallLeafBatch default flip
(false-positive single-run measurement)

**What we tried.** Phase 2.11 plan
(`dev/plans/phase-2.11-small-front-amalgamation.md`) — Option B:
flip the default of `SmallLeafBatch::Off` → `On`. The Phase 2.10
profiler (`src/bin/profile_supernode_distribution.rs`) had been
measuring the `Off` path; comparing `Off` vs `On` on tail
matrices on a *single 5-iteration warmup-then-median run* of
`src/bin/diag_small_leaf_gate.rs` showed:

| matrix         | Off total | On total | ratio |
|----------------|----------:|---------:|-------|
| ACOPR30_0067   |      2045 |     1547 | 0.756 |
| CRESC100_0000  |      1945 |     1422 | 0.731 |
| LAKES_0000     |       493 |      437 | 0.886 |
| NELSON_0000    |       199 |      189 | 0.950 |

I flipped the default and ran the full test suite (158 tests
passed; no parity regression). About to commit.

**Result.** Re-ran `diag_small_leaf_gate` 5 times back-to-back:

| matrix         | run-1 | run-2 | run-3 | run-4 | run-5 | mean  |
|----------------|------:|------:|------:|------:|------:|------:|
| ACOPR30_0067   | 0.755 | 1.052 | 0.920 | 0.959 | 0.983 | 0.94  |
| CRESC100_0000  | 0.964 | 1.031 | 1.025 | 1.007 | 0.998 | 1.005 |
| NELSON_0000    | 1.005 | 1.005 | 0.995 | 1.011 | 1.016 | 1.006 |

Run 1's apparent 25-27% gain was a cold-cache outlier; CRESC100
and NELSON show no effect at all; ACOPR30 fluctuates by ±5% with
mean 0.94 — within noise.

**Why rejected.** The Phase 2.9 small-leaf fast path delivers a
real per-leaf saving (skips `build_row_indices`, no extend-add),
but on the tiny-IPM tail it does not measurably move `total_us`
because the per-leaf savings are dwarfed by the per-front
allocator/setup overhead the path *cannot* avoid (frontal
allocation, scaling pivot order, contribution-block deposit). The
gate flip moves the noise floor by ~1% mean, not by the 30% bar
set in `dev/plans/phase-2.11-small-front-amalgamation.md` §8.

**Disposition.**

- `SmallLeafBatch::Off` remains the default. Doc-comment updated
  to record this rejection so a future agent does not re-run the
  same measurement and reach the same false-positive conclusion.
- The diagnostics produced this session are *kept* in tree:
  - `src/bin/diag_amalgamation.rs` — supernode-tree shape +
    small_leaf-group breakdown counters. Reusable for any future
    amalgamation work.
  - `src/bin/diag_small_leaf_gate.rs` — Off/On A/B harness.
    Useful as a noise-floor probe before any future gate flip.
- Phase 2.11 plan and research note remain in tree for context.

**Proper next direction.** The diagnostic data is unambiguous:
the bushy elimination tree on tiny-IPM KKTs (NELSON: 1 parent
with 129 children; CRESC100: 100% multi-child internal nodes)
blocks 128-410 sibling-merges per matrix via the adjacency check
at `src/symbolic/supernode.rs:204-236`. The fix is Option A from
the Phase 2.11 research note — SSIDS-style column renumbering
during amalgamation (`core_analyse.f90:644-685`). This is a real
refactor (touches the symbolic pipeline's perm composition) and
is not Phase 2.11 scope.

**Evidence.** `dev/journal/2026-04-25-03.org` Phase 2.11
section, `src/bin/diag_small_leaf_gate.rs` 5-run output above.

---

## 2026-04-25 — Flipping `AmalgamationStrategy` default to `Renumber`

**Phase.** 2.12 (column-renumbering amalgamation).

**What was tried.** After implementing `AmalgamationStrategy::Renumber`
(SSIDS-style column renumbering before `find_supernodes`) and observing
60-67% factor-time reduction on the IPM tail (ACOPR30, CRESC100, LAKES),
the natural next step was to flip the default to make every workload
benefit. Ran the full corpus bench (`cargo run --release --bin bench`)
to verify no regression on small-and-medium matrices.

**Why rejected.** Corpus median sparse factor ratio vs MUMPS regressed:

| metric                  | Adjacency | Renumber | Δ      |
|-------------------------|-----------|----------|--------|
| sparse factor p50       | 0.30      | 0.33     | +10%   |
| sparse factor p90       | 1.70      | 1.89     | +11%   |
| sparse factor p99       | 3.79      | 3.45     |  -9%   |
| sparse small-front p90  | 1.69      | 1.88     | +11%   |
| sparse medium p90       | 1.70      | 1.89     | +11%   |

The plan's hard graduation criterion was "corpus median total_us within
±5%". The +10% p50 / +11% p90 regression on the long tail of small
matrices exceeds that budget. Tail wins are real but don't justify
median tax on the rest of the corpus.

**Disposition.** Renumber stays implemented as opt-in
(`SupernodeParams::amalgamation_strategy = AmalgamationStrategy::Renumber`).
Default remains `Adjacency`. Decision recorded in `dev/decisions.md`
("Phase 2.12 column-renumbering kept opt-in"). Future work: shape-
dispatched `Auto` strategy that picks per matrix.

**Evidence.** `/tmp/feral_bench_adjacency.txt`,
`/tmp/feral_bench_renumber.txt` (corpus bench full output);
`dev/journal/2026-04-25-03.org` Phase 2.12 entries.

---

## 2026-04-25 — Tightening LdltCompress gate by raising `MIN_N_FOR_COMPRESSION`

**Phase.** 2.13c (gate-tightening attempt to fix the corpus tail).

**What was tried.** Phase 2.13b's symbolic profiler showed the
`ordering` stage was 85.5% of symbolic time on KIRBY2_0007 (770 µs out
of 924 µs). Phase 2.13b step 5 (`src/bin/diag_amd_substages.rs`)
attributed that 770 µs almost entirely to MC64 inside the
`OrderingPreprocess::LdltCompress` branch — *not* to AMD itself, which
is only 20 µs on n=458. The proposed fix was to bump
`MIN_N_FOR_COMPRESSION` (currently 128) so KIRBY2-class small-n
matrices skip MC64 and pay the cheaper no-compress AMD path instead,
projected to collapse KIRBY2's ordering stage from 878 µs to ~25 µs.

Before changing the gate, ran the cost/benefit probe
(`src/bin/diag_compress_costbenefit.rs`) to verify the MC64 savings
weren't offset elsewhere.

**Why rejected.** The probe revealed that compression's MC64 cost is
*paid back* in the numeric phase. 5-run-median wall-clock total
(symbolic + numeric), in microseconds:

| matrix         |   n  | None | Compress | delta | verdict |
|----------------|-----:|-----:|---------:|------:|---------|
| KIRBY2_0007    |  458 | 1209 |     1045 |  -164 | compress wins |
| MUONSINE_0000  | 1537 | 2093 |     1354 |  -739 | compress wins |
| ACOPR30_0067   |  564 |  594 |      810 |  +216 | None wins |
| CRESC100_0000  |  806 |  642 |      851 |  +209 | None wins |
| LAKES_0000     |  324 |  247 |      258 |   +11 | neutral |
| NELSON_0000    |  387 |  294 |      298 |    +4 | neutral |
| SWOPF_0000     |  175 |  157 |      155 |    -2 | compress |

KIRBY2's numeric stage drops 1028 µs → 245 µs under compression, and
MUONSINE's drops 1612 µs → 619 µs. The MC64 cost is essentially
covered by numeric savings on those matrices. The 9.5× MUMPS headline
on KIRBY2 already reflects the better of the two preprocesses.

The actual gate failures are ACOPR30/CRESC100 (compression triggers
but does not pay back numerically). Tightening on `n` would *regress*
KIRBY2 and MUONSINE while marginally fixing ACOPR30/CRESC100 — and
ACOPR30/CRESC100 are no longer in the corpus Top-10 worst-ratio
(Phase 2.12 already cut their factor 60-67% via Renumber). Net
negative.

**Disposition.** Do not raise `MIN_N_FOR_COMPRESSION`. Do not gate
LdltCompress on `n` alone. Plan section 2.13c paused. The right
discriminator (if any) needs to identify the
ACOPR30/CRESC100-vs-KIRBY2/MUONSINE structural difference, not size.
Probe extension `(a)`–`(c)` recorded in
`dev/plans/phase-2.13-tail-diagnostic.md` for whoever revisits.

**Lesson.** The Phase 2.13b sub-stage probe correctly identified MC64
as the dominant *symbolic* sub-stage but did not look at the *numeric*
phase. Conclusions from a per-stage profile should be cross-checked
against an end-to-end cost/benefit measurement before touching a gate.

**Evidence.** `dev/journal/2026-04-25-03.org` 24:30 entry;
`src/bin/diag_amd_substages.rs` and
`src/bin/diag_compress_costbenefit.rs` outputs.


---

## 2026-04-26 — Per-iter take/drop alone as bench OOM fix

**What.** Convert `KktEntry.csc` to `Option<CscMatrix>` and `take()` it
per iteration in the sparse loop, plus drop `entry.matrix = None`
between dense and sparse passes. Idea: cumulative working set should
shrink as each matrix is processed and dropped.

**Why it's not enough.** With the default corpus expanded to all three
KKT roots (167,614 matrices), the cumulative CSC working set at
sparse-loop *entry* is already ~30 GB. macOS allocator does not
return freed memory to the OS immediately, so RSS stays high even
after take/drop runs. Combined with Dropbox renderer + claude-code
+ rustc consuming another 10+ GB, system pressure builds and macOS
jetsam SIGKILLs the bench after several minutes of silent processing.

**What was actually shipped.** Both layers:
1. Per-iter take/drop (kept as the right shape — small cost, useful
   on expanded-corpus runs and bounds in-flight growth).
2. `FERAL_KKT_ROOTS` env var defaulting to `kkt` (restores the
   2026-04-25 baseline corpus scope, ~154,588 matrices, fits in
   ~30 GB peak; `=all` opts into expanded corpus).

**Evidence.** `dev/journal/2026-04-26-02.org` 18:00 + 19:05 entries.
First fix alone: bench killed at ~30 GB RSS after 8 minutes silent
in sparse loop. Both fixes together: bench completes end-to-end with
99.8% sparse residual / 100% sparse inertia / Phase 2.8.1 PASS on
both buckets.

## 2026-04-26-03: factor_nnz() = nrow * nelim

**Tried.** Counting per-supernode L block as `nrow * nelim` (full
dense column block). Predates this session; introduced when the
multifrontal supernode storage was first wired up.

**Why rejected.** Confirmed via `src/bin/diag_factor_nnz_accounting.rs`
to be a 1.75× overcounting artifact relative to SSIDS's
`inform%num_factor`. The strict-upper triangle of the eliminated
block is structurally always zero (lower-triangular factor) and was
sweeping in nonexistent fill. Bench was reporting nnzL/SSIDS p50 ≈
1.75 across the kkt corpus when feral's actual L-fill medianly
matches SSIDS exactly.

**Replacement.** Per-supernode count is now
`nelim*(nelim+1)/2 + (nrow-nelim)*nelim` (lower-tri-with-diagonal of
eliminated block + trailing rect). Median nnzL/SSIDS = 1.000 across
the kkt corpus after the fix. Committed `ae81b81`.

**Evidence.** Counts across 71 sampled matrices: C/SSIDS geomean
1.914 / median 1.833; B/SSIDS geomean 1.149 / median 1.000. After
the fix, bench reports nnzL/SSIDS p50 = 1.00, geomean = 1.09, p99 =
4.50. See `dev/journal/2026-04-26-03.org` 20:10 / 20:25 entries.

## 2026-04-26-03: FERAL_KKT_ROOTS=all on 64 GB laptop

**Tried.** Setting `FERAL_KKT_ROOTS=all` (167,614 matrices across
`kkt + kkt-expansion + kkt-mittelmann`, 21 GB on disk) to validate
expanded corpus end-to-end on the dev laptop.

**Why rejected.** Loaded all three roots and ran the dense pass to
completion, then SIGKILLed during the sparse loop. The bench's
sparse-loop entry state holds the CSC for all 10,120 n>1000 matrices
simultaneously; even with per-iter `take()` drops, the upfront total
exceeds the 64 GB ceiling. `FERAL_KKT_ROOTS=kkt-mittelmann` alone
(596 matrices, 4.2 GB on disk) shows the same pattern.

**Status.** Not a feral correctness issue — it's a bench harness
architectural limitation. **Expanded-corpus dense pass validates
99.9% inertia (157,356/157,494) and 99.8% residual (157,220/157,494)
across 157,494 dense-eligible matrices**, with the worst residual
(`ERRINBAR_0824`, 1.87e-4) matching the kkt-only baseline — no new
failure modes from `kkt-expansion` (12,430 matrices) or
`kkt-mittelmann` (596 matrices). Sparse-pass validation on the
n>1000 portion of the expanded corpus requires either streaming
bench redesign or beefier hardware. See session 03 "Next Session
Should" item 1.

**Evidence.** `dev/journal/2026-04-26-03.org` 20:50 entry.

## 2026-04-26-04: Single-pass streaming bench

**Tried.** During the streaming-bench refactor, considered merging
the dense and sparse loops into one pass over the corpus so each
.mtx is parsed only once, then both loop bodies share the parsed
data via local variables.

**Why rejected.** Would invert the output ordering (currently dense
summary prints first, then sparse) and merge the per-loop summary
state machines into one. Diff would touch failure-tracking,
perf-comparison, and phase-2.8.1 partition code that was independent
across the two loops. Two-pass streaming costs an extra .mtx parse
for the 157k dense-eligible matrices (~seconds in absolute) and
preserves the diff containment.

**Evidence.** Session 04 journal 21:30 entry. Two-pass refactor
landed in commit 53c07bb and validated end-to-end on
`FERAL_KKT_ROOTS=all`.

## 2026-04-26-04: Uncapped sparse loop on 64 GB laptop with streaming bench

**Tried.** After streaming refactor, ran `FERAL_KKT_ROOTS=all` with
no `FERAL_SPARSE_MAX` cap to confirm the sparse loop handles the
full 170,176-matrix expanded corpus.

**Why rejected.** Streaming bounded the dense pass to ~17 GB peak
(was 30+ GB load-all), but the sparse pass still SIGKILLed (exit 137)
shortly after starting. Cause: the expansion corpus contains 10
matrices with n > 50000 (max 451195) whose multifrontal factor
allocation alone exceeds 64 GB. Streaming the load cannot help when
the issue is a single matrix's working set.

**Status.** Mitigated with `FERAL_SPARSE_MAX=20000` opt-in cap (skips
237 matrices, leaves 167,380 attempted). End-to-end run completes at
~50 min wall, ~36 GB peak RSS. Still leaves a residual question:
even at cap=20000, RSS climbed from 17 GB to 36 GB across the sparse
loop, suggesting cumulative growth (allocator fragmentation or hidden
accumulator) on top of the per-matrix peak. See session 04 "Next
Session Should" item 1.

**Evidence.** Session 04 journal 22:25 / 22:30 / 22:50 entries.

## 2026-04-27-08: Phase B.4 tighter expected-perm pins (dual-arrow, tridiag)

**Tried.** As part of Phase B.4 (`dev/plans/amf-clean-room.md` Phase
B deliverable 6), pinned the qualitative claims sketched in the
plan as test assertions in `crates/feral-amf/tests/expected_perm.rs`:

1. 5x5 dual-arrowhead -- both hub vertices (0 and 4) deferred to
   the last two positions of the perm.
2. 7x7 tridiagonal -- both endpoints (0 and 6) eliminated before
   any interior vertex.

**Why rejected.** Both assertions failed on the actual implementation:

- `dual_arrow_5` produced `perm = [3, 0, 1, 2, 4]`. Hub 0 was picked
  at iteration 1, before spine vertices 1 and 2. Only one hub (4) is
  in the last position.
- `tridiag(7)` produced `perm = [6, 5, 4, 3, 2, 0, 1]`. The
  implementation sweeps from one endpoint down to the other via
  successive deg=1 surrogates -- this is the standard quotient-graph
  one-end-sweep behaviour shared with AMD; vertex 0 is not picked
  until iteration 5.

The plan's qualitative claims were aspirational rather than
metric-derived. Without a MUMPS HAMF4 oracle (Phase C) to
distinguish "implementation produces a sensible permutation that
happens to break the plan's claim" from "implementation has a bug",
pinning these tighter assertions would either (a) require weakening
the implementation to match a guess about expected behaviour, or
(b) create a flaky gate that rejects sensible perms.

**Status.** Resolved by weakening to claims directly derivable from
the iteration-0 metric `RMF(i) = deg(i)*(deg(i)-1)`:

- arrow_3: hub eliminated last (kept; passes).
- dual_arrow_5: first pivot is a spine vertex; last pivot is a hub
  vertex (passes).
- tridiag(7): first pivot is an endpoint (passes).

The tighter pins are deferred to Phase C, where the MUMPS HAMF4
oracle on `data/matrices/kkt*` provides the external reference. A
note to that effect lives in the test file's module doc.

**Evidence.** Test failures in `cargo test -p feral-amf --release
--test expected_perm` before the weakening:
- `amf_dual_arrow_5_both_hubs_deferred` panicked: left {2, 4} != right {0, 4}.
- `amf_tridiag_7_endpoints_first` panicked: max endpoint position 5 not less than min interior position 1.

## 2026-04-28 — MUMPS missing-diagonal MC64 skip (mistranslated regime)

**Context.** Session 2026-04-28-01's profile showed
`mc64::compute_matching` at 26% inclusive wall in `profile_hot`.
The session's planned next step was to port MUMPS's KKT-aware
"skip MC64 if diagonal is mostly populated" rule from
`mumps/src/dana_aux.F:1388-1416`: when
`(missing_diag + zero_diag) < max(1, N/10)` MUMPS skips KEEP(52)=4
matching and falls through to cheap symmetric Ruiz equilibration
(SIMSCA, KEEP(52)=7). Estimated 5–15% wall savings.

A plan note was written
(`dev/plans/mc64-missing-diag-skip.md`) and a one-shot probe
(`probe_missing_diag`, since deleted) was built to size the
test thresholds before implementation.

**Why rejected.** The probe surfaced a regime mismatch.

The literal MUMPS rule is structural — it counts diagonal entries
that are absent or *exactly* zero in the input CSC. Walk over the
569-family corpus:

| outcome under literal rule | families | of which arrow-KKT |
|----------------------------|---------:|-------------------:|
| would-skip (miss+zero < n/10) |   501  |       289 (lose MC64) |
|   - non-arrow (today already InfNorm — no-op) |    212  |             — |
| no-skip |     68  |          —         |

289 of 569 families would lose the lever-C MC64 win
(`dev/research/lever-c-adaptive-scaling.md`). 0 families would gain.

Direct inspection of `data/matrices/kkt/VESUVIO/VESUVIO_0000.mtx`
explains why: the dual block (rows 3054-3083) is stored with
explicit `-1.00000000000000002e-8` — an IPM constraint
regularization δ_c that the corpus generator dumped. Every KKT
matrix in `data/matrices/kkt/` is a *post-regularization* IPM
snapshot. The MUMPS rule was designed for SYM=2 inputs *before*
such regularization, where dual diagonals are structurally absent.
Applying the rule literally on regularized snapshots over-skips
on essentially everything.

A reframed numerical variant ("skip when most columns are
diagonally dominant under a tolerance") was considered but
shelved: the δ_c sensitivity probe (the next thing built, and the
useful artifact from this exercise) showed Auto routing is
already δ_c-robust by structural signature, so there is no
heuristic-drift problem to fix here.

**Disposition.** Rule not implemented. Plan note retained at
`dev/plans/mc64-missing-diag-skip.md` as a pointer for the
unlikely case that we get a *raw, unregularized* corpus to
re-evaluate against. The throwaway `probe_missing_diag` binary
was deleted; the related but useful
`src/bin/probe_deltac_sensitivity.rs` was kept and is the basis
for the new "Auto routing is δ_c-robust" decision in
`dev/decisions.md`.

**Lesson.** Heuristics ported from another solver's literature
must be validated against the *input regime* feral actually sees,
not just the algorithmic setting. MUMPS sees raw KKTs at analysis
time; feral sees pre-regularized snapshots at refactor time.
Same algorithm, different regime, different right answer.

**Evidence.**
- `src/bin/probe_deltac_sensitivity.rs` output (in this session's
  journal).
- `data/matrices/kkt/VESUVIO/VESUVIO_0000.mtx` line 3054–3083:
  explicit `-1e-8` dual reg.
- `dev/research/lever-c-adaptive-scaling.md`: lever-C win that
  the literal rule would have destroyed.
- `dev/plans/mc64-missing-diag-skip.md`: the plan that was
  written then shelved.

---

## 2026-05-03 — Phase B: shape-dispatched `nemin` within `Auto`

**Hypothesis.** After Phase A landed `nemin = 16` as the global
default, layer a shape-dispatched override on top of
`AmalgamationStrategy::Auto` so path-like fixtures get
`nemin = 32` (no `factor_nnz` cost, hypothesized small wall win
from larger BLAS-3 panels) and bushy fixtures stay at `nemin = 16`
(Phase A's choice).

**What was tried.** Added `DEFAULT_NEMIN`, `NEMIN_PATH_LIKE = 32`,
`NEMIN_BUSHY = 16` constants in `supernode.rs` and an override
branch in `mod.rs:594-625` that flipped `nemin` only when the
caller had not changed it from `DEFAULT_NEMIN`. Built
`src/bin/diag_phase_b_nemin_sweep.rs` covering MUONSINE_0000
(path-like), KIRBY2_0007 / ACOPR30_0067 / SWOPF_0000 (bushy).

**Why rejected.**

1. Path-like `factor_nnz` is **invariant** in `nemin` (MUONSINE
   stays at 4606 across {8, 16, 24, 32, 48}) — there is no memory
   motivation for the dispatch.
2. Path-like wall-time signal is **not robust** under measurement
   noise. Two consecutive sweep runs disagreed on direction:
   run 1 nemin=48 was 8% faster than nemin=16 (195 vs 212 µs);
   run 2 nemin=48 was 29% slower (273 vs 211 µs). The
   200-µs-base scale is below the level where wall comparisons
   on this CPU are trustworthy.
3. Bushy fixtures uniformly confirmed Phase A's `nemin = 16`
   choice: KIRBY2 ties at 8/16, then `factor_nnz` grows +9% at 32,
   +26% at 48. ACOPR30 and SWOPF show similar monotonic growth.
4. Per the decision rule pre-registered in
   `dev/research/phase-b-shape-dispatched-nemin.md`: "If both
   buckets prefer 16 (within ≤ 5% factor wall and ≤ 10%
   factor_nnz of any other tested value), keep the global default
   and document that Phase B is a no-op — don't add code that
   doesn't earn its keep."

**Reverted.** The implementation was reverted in the same session.
`SupernodeParams::default()` keeps the literal `nemin: 16`; the
constants and override branch were removed; the sweep binary is
retained for any future reconsideration with a larger or
differently-stratified fixture set.

References: `dev/research/phase-b-shape-dispatched-nemin.md`,
`dev/research/factor-nnz-residual-gap.md`, commit `4c0fc80`
(Phase A).

## 2026-05-12 — Lock-free contribution-block store for parallel driver

**Hypothesis.** cont-201's 1.44× speedup at T=8 (vs 4.83× theoretical
critical-path ceiling) is bottlenecked by the shared
`Mutex<HashMap<usize, ContribBlock>>` in
`factorize_multifrontal_supernodal_parallel`. Hot path acquires twice
per task (children-drain + own-store).

**Test.** Added `AtomicLockStats` opt-in telemetry (six lock
wait/hold/body/task atomics + eight per-phase wall timers) and
extended `solver_parallel_lock_breakdown` to run cold + cached
factor pairs at T=4, reporting cached-symbolic numbers
(production/IPM regime).

**Result.** Falsified.

| matrix    | total mutex wait + hold | aggregate body | wait-frac |
| --------- | ----------------------: | -------------: | --------: |
| bcsstk38  |               0.28 ms   |      14.3 ms   |     1.8%  |
| bratu3d   |               2.15 ms   |     956.1 ms   |     0.2%  |
| c-big     |              65.6  ms   |  273526 ms     |     0.02% |
| cont-201  |               4.82 ms   |     123.5 ms   |     3.9%  |

cont-201 cached wall is 56.2 ms; the 1.5× residual headroom in cached
mode lives inside the rayon::scope (loop utilization 68.5%), not at
the locks. A lock-free store would recover ≤4% of body time worst
case, ≤0.04% best case.

**Action.** Telemetry kept in tree as an opt-in diagnostic surface
(`NumericParams::parallel_telemetry`). No re-design of the
contribution-block store. Decision recorded in `dev/decisions.md`
2026-05-12 "Reject lock-free contribution-block store". Full
breakdown in `dev/debugging/2026-05-12-cont201-cached-headroom.md`.

References: `src/numeric/factorize.rs::AtomicLockStats`,
`src/numeric/factorize.rs::run_parallel_task`,
`src/numeric/solver.rs::tests::solver_parallel_lock_breakdown`,
`dev/debugging/2026-05-12-cont201-cached-headroom.md`.

---

## 2026-05-13 — Phase C multi-slot contrib pool (Vec<Vec<f64>>)

**Hypothesis.** Pool the multifrontal contribution-block buffers across
supernodes using a `Vec<Vec<f64>>` stack on `FactorScratch.contrib_pool`,
so the parent's extract step pops a recycled `Vec` instead of `vec![0.0;
cdim*cdim]`. Issue #13 phase C, motivated by the open bench-ratio
acceptance criterion #2 (small p90 < 1.30 OR medium p90 < 1.60).

**Test.** Implemented in `src/dense/factor.rs` (extract step pops from
the pool, clears, resizes) and `src/numeric/factorize.rs` (driver pushes
the child's `ContribBlock.data` onto the pool after `extend_add`
consumes it). Bit-parity preserved across all four parity cases in
`tests/factor_scratch_parity.rs` including a new (d) pool-hot pre-seed
case. Bench: 4 consecutive `cargo run --bin bench --release` runs.

**Result.** Falsified.

| variant                  | small p90 | medium p90  | inertia       |
| ------------------------ | --------: | ----------: | ------------- |
| Phase A+B (re-measured)  |      1.41 | 1.83 – 1.86 | 154428/154481 |
| Phase C multi-slot       | 1.60-1.62 | 2.13 – 2.17 | 154428/154481 |

Multi-slot regressed bench p90 by ~+0.19 (small) / ~+0.30 (medium). The
growable-indirection bookkeeping cost (push/pop, scattered heap
pointers from `Vec<Vec<f64>>`, branch on capacity) exceeded the
malloc/free pairs it avoided. The malloc cost was never the bench
bottleneck on this corpus.

**Action.** Replaced by a single-slot `Option<Vec<f64>>` pool, which is
bench-neutral (small 1.41, medium 1.83–1.85 — back to A+B baseline)
while preserving bit-parity. Committed as feat(issue-13): Phase C —
single-slot contrib pool (neutral), commit `fe2ca4d`. Issue #13
re-scoped: criterion #2 declared unreachable via allocation pooling
on this corpus; per-front kernel cost (32×32 SIMD, issue #9) is the
next plausible lever.

References: `src/dense/factor.rs::FactorScratch`,
`src/numeric/factorize.rs::factor_one_supernode`,
`tests/factor_scratch_parity.rs` case (d), commit `fe2ca4d`.

---

## 2026-05-13 — Issue #10 APP path: implementation not undertaken; gate not met

**What was proposed.** Issue #10 — add an APP (aggressive partial
pivoting) path to `src/dense/factor.rs` alongside the existing
per-pivot threshold check. The proposal cites a ~5× per-nnz_L gap
on CHAINWOO-style fronts (89 ns vs 14 ns for MUMPS) and proposes
a block-level deferred check that avoids per-pivot column scans.

**What the gate said.** The issue's own posted re-open comment (by
`jkitchin`) required a fresh `diag_supernode_cost` run showing
"ns/nnz dominates ns/sup on a relevant cluster" before APP work
is justified.

**What the data shows.** `cargo run --bin diag_supernode_cost
--release` (2026-05-13, post-`d7267fe`):

```
ACOPR30_0067 nemin=32  ncol_max=32  ns/sup=943   ns/nnz=61   ratio 15×
CRESC100_0000 default  ncol_max=16  ns/sup=914   ns/nnz=79   ratio 12×
HAIFAM_0082            ncol_max=86  ns/sup=1174  ns/nnz=33   ratio 36×
```

Across **every** corpus row and every nemin in the sweep, ns/sup
exceeds ns/nnz by 4× to 36× — the opposite of the gate condition.
The per-front fixed cost still dominates the per-nnz arithmetic
cost on the long-tail corpus.

**Why the motivating gap closed.** The 89 ns/nnz_L figure cited in
the issue is stale on the current build:

- `fused_gamma0` (`factor.rs:369-371`, landed `ad05ff4`
  2026-04-11) carries the next pivot's γ₀ and argmax row across
  the rank-1 update on the scalar path — the same trick the issue
  attributed uniquely to MUMPS `MAXFROMM`. Per-pivot column scans
  on the no-swap branch are already eliminated.
- The 32×32 SIMD body (`block_ldlt32`, landed `d3f1132`
  2026-05-13) puts trailing-update FLOPs for the dominant CHAINWOO
  front shape through a quad pulp dispatch. The dispatch at
  `factor.rs:1189-1193` routes `nrow == ncol == 32` fronts to
  `factor_block32` before the panel path is reached.

**Decision.** Do not implement APP. Recorded in `dev/decisions.md`
2026-05-13. Full analysis in `dev/research/dense-app-path.md`.

**Lesson.** Same as the 2026-05-12 (c) BLAS-3 quad parking: the
session-checkpoint "Next session should" list is not a substitute
for re-measuring the gate. The previous session
(`dev/sessions/2026-05-13-02.md`) advanced #10 as the next target
on the strength of #9 having landed, without re-running
`diag_supernode_cost`. One binary run was the difference between
implementing dead code and recording a clean closure.

References: `dev/research/dense-app-path.md`,
`dev/decisions.md` 2026-05-13 entry, issue #10 thread.

---

## 2026-05-15 — Default `cascade_break_ratio = None` to fix issue #17

**Attempt.** Considered making feral's default `cascade_break_ratio`
revert to `None` (legacy delayed-pivot path) to close issue #17
without any IPM-side change. Rationale: cb=off converges
robot_1600 in 40 iters / 6.1 s vs cb=default's MaxIter at 200
iters / 53 s.

**Why rejected.** Cascade-break is the cascade-arm gate shipped
by #15 and is calibrated across the bench corpus to help on a
specific class of matrices. Disabling it by default would
regress those without addressing the underlying mechanism in
robot_1600. The 2026-05-15 decision (`dev/decisions.md`)
established the failure is a *solve-accuracy* regression (~5-OOM
on identical inertia), not an *inertia-counting* one. Fixing it
upstream by removing cascade-break trades one regression for
another.

**Status.** Issue #17 is being addressed downstream: wire
`Solver::solve_refined` into `pounce-feral/src/lib.rs:107` so
F2.3 iterative refinement absorbs the perturbation. Pursued in
next session.

References: `dev/sessions/2026-05-15-01.md`,
`dev/decisions.md` 2026-05-15 entry, issue #17 thread.


## 2026-05-15 — "Zero L on `PerturbToEps`" to enforce the Weyl bound

**Attempt.** `ZeroPivotAction::PerturbToEps`'s docstring claimed
`LDL^T = A + Δ` with `||Δ||_∞ ≤ abs_floor` per perturbed pivot.
Session 02 measured an ~`1.4×10⁻⁵` unrefined solve-diff on
`robot_1600_0004` and concluded the bound was being violated.
Diagnosis: with the pivot perturbed to `d_new ≈ eps` and L still
scaled by `1/d_new`, the L column entries grow as `A[i,k]/eps`,
which the research note framed as a `1/eps` amplification "violating
the Weyl bound."

Proposed fix (mirror of `ForceAccept`): zero `L[:,k]` below the
diagonal after writing the perturbed `D[k,k]`, return
`PivotOutcome::Rejected` so `do_1x1_update` is skipped. Predicted
post-fix residual: `~1e-14` (LAPACK static-pivoting bound). Applied
to both 1×1 PerturbToEps sites (`try_reject_1x1_frontal` and
`do_1x1_pivot`).

**Why rejected.** Direct measurement on `robot_1600_0004` (probe
`src/bin/probe_cascade_perturb.rs`):

| config                         | residual          |
| ------------------------------ | ----------------- |
| cb=off                         | 6.24e-7           |
| cb=default (pre-fix code)      | 1.06e-5           |
| cb=default (with L-zero fix)   | **2.13e+3**       |
| cb=fa (ForceAccept)            | 2.10e+2           |

The fix made the residual five orders of magnitude *worse* than
pre-fix. Reason: with L zeroed but `D[k,k] = d_new ≈ eps`, the solve
divides `x[k] = (rhs - L row k contribution) / d_new ≈ rhs / 1e-10`.
There is no longer a live L column to cancel the `1/d_new` factor.

The premise was also wrong on the math: pre-fix code's factorization
*is* self-consistent. `(A[i,k]/d_new) · d_new · 1 = A[i,k]` exactly,
so off-diagonal column-k entries are preserved. The implicit `Δ`
flows through the Schur update (`Δ_schur[i,j] = A[i,k]·A[j,k]·(1/d_new
− 1/d_orig)`) and is bounded by `||A||²/eps` in the worst case — not
by `eps`. The original docstring's bound was incorrect, but the code
was doing the right thing for solve.

**Resolution.**

1. Code revert: no change to the `PerturbToEps` branches.
2. Docstring corrected (`src/dense/factor.rs` `PerturbToEps`,
   `src/numeric/solver.rs` `with_cascade_break_eps`) to honestly
   describe the perturbation structure.
3. Cascade-break flipped to **opt-in** by default
   (`NumericParams::default()` now has
   `cascade_break_ratio = None, cascade_break_eps = None`).
   MUMPS and MA57 don't ship an equivalent of cascade-break-eps;
   auto-arming a non-standard mechanism was creating surprises and
   the prior tried-and-rejected entry above ("Default
   `cascade_break_ratio = None` to fix issue #17") was based on the
   wrong assumption that the win-case had no opt-in path. The win
   case (`pinene_3200_0009`, 88.6 s → 34 ms) is preserved via
   explicit `Solver::with_cascade_break(0.5).with_cascade_break_eps(1e-10)`.

References: `dev/research/cascade-break-l-perturbation-2026-05-15.md`,
session 2026-05-15-02 (original 1.4e-5 measurement), session
2026-05-15-07 (this entry).

---

## 2026-05-16 — MAXFROMM as default TppMethod for 1D-banded Mittelmann panel

**Tried.** MUMPS-style MAXFROMM acceleration of TPP pivot selection
(`TppMethod::Maxfromm`): capture column k+1's AMAX as a byproduct of
the rank-1 trailing update at pivot k, then short-circuit the next
pivot's AMAX scan when `|a_{k+1,k+1}| >= alpha * cached`. Predicted
≥2× speedup on the 1D-banded Mittelmann panel (clnlbeam, henon120,
lane_emden120, dirichlet120) per the original research note
`dev/research/issue-10-app-vs-maxfromm.md`.

**Rejected.** Default-flip to `TppMethod::Maxfromm` rejected. Phase 2
corpus A/B (`src/bin/diag_clnlbeam_maxfromm.rs`, min-of-7, 20
matrices across 4 families): panel median 0.997×, geomean 1.000×,
all per-family medians within ±5% measurement noise. The ≥2.0×
prediction was wrong because (i) the per-pivot AMAX scan was already
cheap (~10% of pivot cost on narrow supernodes, not the dominant
fraction); (ii) MAXFROMM moves the scan rather than removing it
(post-update capture vs pre-pivot scan); (iii) on cache miss
(2×2/rejection/panel boundary) MAXFROMM ADDS work — the capture
runs but is never consumed. The 97%-1×1 finding from #33 was real
but the dominant cost in each 1×1 is the rank-1 axpy, not the AMAX
scan.

**Resolution.** Phase 1 infrastructure is kept (commit 590bc50):
`TppMethod::{Plain, Maxfromm}` enum and `BunchKaufmanParams::tpp_method`
field, default `Plain`. Opt-in `Maxfromm` is byte-identical on
factorization output (5 parity tests in `tests/maxfromm_parity.rs`)
and ~zero cost on this corpus (within noise). The enum stays as a
primitive for future experiments on wider-front workloads where AMAX
scan cost might actually be measurable.

The Phase 4 plan from the original research note (wire MAXFROMM into
`block_ldlt32`) is deferred indefinitely until a corpus is identified
where MAXFROMM measurably wins.

Both #33 (SmallLeafBatch) and #10 (MAXFROMM) targeting the same
1D-banded panel landed within noise, jointly demonstrating that the
bottleneck on that corpus is neither per-supernode driver overhead
nor pivot selection. The next lever is the scalar rank-1 trailing-
update kernel itself (or supernode amalgamation to widen narrow
leaves so block kernels can engage).

References: `dev/research/issue-10-maxfromm-phase2-corpus.md` (full
post-mortem), `dev/research/issue-33-slb-ab.md` (parallel SLB result),
journal `2026-05-16-01.org` 11:32 + 12:30 entries.
