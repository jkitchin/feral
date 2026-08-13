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

## 2026-05-16-02 — Manual 4-way unrolled scalar axpy as #10/#33 unblocker

`src/bin/bench_axpy_small.rs` (50M iters/measure, min-of-5) compared
`pulp` (current `axpy_minus_unroll4_nofma` SIMD dispatch), `scalar`
(`for (d,s) in dst.iter_mut().zip(src) { *d -= alpha * *s; }`), and
`unroll4` (manual 4-way unroll without pulp dispatch) at lengths
[3, 4, 5, 6, 8, 10, 16, 32, 64, 128].

Result: pulp ties with plain scalar within 1ns/call quantization at
all lengths 3..128; manual unroll4 is *slower* (0.25-1.00x). The
compiler auto-vectorizes the scalar form as well as the explicit SIMD
dispatch. Kernel-call overhead is *not* the bottleneck.

Implication: the #10 Phase 2 corpus post-mortem's hypothesis that
"the next lever is the scalar rank-1 trailing-update kernel" was
also wrong. Three architectural levers tried against the 1D-banded
Mittelmann panel (#33 SmallLeafBatch driver overhead, #10 MAXFROMM
pivot selection, axpy kernel tightening) all come up within noise.

Remaining levers for the corpus: (a) `Solver::with_ordering(ScotchND)`
to widen the supernode shape (untested, just landed via #33 §3);
(b) supernode amalgamation (symbolic-side restructure); (c) accept
a hardware floor for sequential factorization on this shape.

References: `src/bin/bench_axpy_small.rs`, journal
`2026-05-16-02.org` 11:30 entry.

## 2026-05-16 — Rank-1 perturbation for `mc64_resistant` synthetic

First attempt at the `mc64_resistant_n` generator (issue #27 stress
suite) built the matrix as `A = I + α · u u^T` with `u = 𝟙/√n` and
`α = -2`, on the theory that a rank-1 update of a flat diagonal would
defeat MC64's symmetric scaling.

Wrong: the eigenvalues of this rank-1 update are `1` (multiplicity
n-1) and `1 + α = -1`. So `cond(A) = 1` and there is nothing for MC64
scaling to fail at — the matrix is already perfectly equilibrated.

Direct verification on n=200, seed=601:
- `np.linalg.cond(A) = 1.48` before any scaling
- after a symmetric row-max scaling (proxy for MC64-style scaling):
  `cond = 1.48` (unchanged, as expected)

The "rank-1 perturbation of a diagonally dominant skeleton" framing
suggested in the issue was misleading: a low-rank update of an O(1)
diagonal redistributes O(1) mass; it does not produce the dispersed
ill-conditioning that defeats diagonal scaling.

Replaced with `A = Q D Q^T` construction where Q is a random dense
orthonormal basis and D has one eigenvalue at `1e-8` with the rest
O(1). Now `cond(A) = 2e8` before *and* after symmetric scaling — the
small eigenvalue is in a basis direction that diagonal scaling
cannot reach.

Documented in `dev/research/synthetic-generators-m4.md` §4. The
current generator uses the Q D Q^T construction.

---

## 2026-05-16 — Forced supernode amalgamation (`nemin > 16`) on the 1D-banded Mittelmann panel

**What.** Sweep `SupernodeParams::nemin ∈ {16, 32, 64, 128}` on the
4-family × 20-matrix Mittelmann panel (`clnlbeam`, `henon120`,
`lane_emden120`, `dirichlet120`). Hypothesis: widening the
amalgamation threshold above the Phase 2.13a default of 16 will
fuse bottom-of-tree chain-link supernodes into rectangular fronts
wide enough to re-engage MAXFROMM / APP-class kernels.

**Result. The shape lever engages but the time profile does not
respond.**

| family        | factor_us/nemin=16 (geomean, paired) |       |       |
|---------------+-------+-------+-------|
|               | n=32  | n=64  | n=128 |
| clnlbeam      | 1.032 | 1.356 | 1.989 |
| henon120      | 0.970 | 0.960 | 1.029 |
| lane_emden120 | 0.953 | 0.903 | 0.909 |
| dirichlet120  | 0.951 | 0.943 | 0.958 |

`ncol_mean` doubles at nemin=64 across three of four families, but
`factor_nnz` inflates 1.23–1.33× and factor time stays flat or
regresses. `clnlbeam` regresses 36% at nemin=64 because chain-link
merges blow trailing-fill faster than the wider panel can amortize.

**Why it was rejected.** Closes the fifth and final architectural
lever for issue #10. All five (SLB driver removal, MAXFROMM AMAX
cache, manual axpy SIMD, ordering swap, this nemin sweep) come up
negative on the 1D-banded panel. The rank-1 axpy kernel on
`ncol=1..16` fronts is bandwidth-bound; pulp saturates the vector
ALU; AMD's elimination tree is already shape-optimal under the
nnz_L bound. A pilot run at `nemin ∈ {256, MAX}` hung on
`clnlbeam_0000` — a single near-dense front of order >n/2 collapsed
the dense LDL into a non-returning state. Sweep capped at 128.

Issue #10 closes as "hardware floor reached on the 1D-banded panel."
The opt-in knobs (`Solver::with_ordering`, `SupernodeParams::nemin`)
stay shipped for workloads where the elimination tree genuinely
has fusion opportunities — they just don't help here.

Documented in `dev/research/issue-10-amalgamation-floor.md`. A/B
binary: `src/bin/diag_nemin_amalgamation_panel.rs`. Commit 61002f8.

## 2026-05-16 — M3 stress corpus: matrix-market reader unsupported formats (#26)

**What was tried.** Adding to the stress manifest the following
SuiteSparse matrices that match the M3 inclusion criteria (indefinite,
n ≤ 100k, in `GHS_indef` / `Boeing` groups):

- `Boeing/nasa2910` (2910×174k, "no" posdef, indef stiffness)
- `Boeing/nasa4704` (4704×104k, "no" posdef, indef stiffness)
- `GHS_indef/aug2d`  (29k×76k, saddle)
- `GHS_indef/aug2dc` (30k×80k, scaled saddle)
- `GHS_indef/aug3d`  (24k×69k, 3D saddle)

**Symptom.** `bench_one_matrix` factor failed at the I/O stage with:

```
status=fail  fail_reason=read_mtx IoError("...
  unsupported header '%%MatrixMarket matrix coordinate pattern symmetric'
  (expected: %%MatrixMarket matrix coordinate real symmetric)")
```

for the two NASA matrices, and the analogous `coordinate integer
symmetric` rejection for the three augmented-saddle matrices. The
NASA tarballs ship **pattern** matrices (no numeric values) and the
augmented matrices ship **integer** matrices; feral's MM reader at
`src/io/mtx.rs` accepts only `coordinate real symmetric`.

**Why it was rejected (for this issue).** Extending the MM reader to
synthesize values for pattern matrices (typical convention: all-ones)
and to parse integer values as `f64` is straightforward but out of
scope for a corpus-expansion ticket — it requires a separate test
harness and a small spec decision (what value does a pattern entry
take? plain 1.0, or random?). All five matrices were dropped from
the M3 manifest after numeric factorization confirmed the I/O
rejection. If a future session wants to re-add them, the gate is:
extend `mtx.rs` to handle `pattern` and `integer`, add round-trip
tests, then `cd external_benchmarks/stress && fetch.py` should
already have the .mtx files cached.

Final manifest count after drop: 104 SuiteSparse rows
(target ≥ 80 met with 30% headroom).

## 2026-05-16 — M3 stress corpus: Schenk_AFE skipped on size (#26)

**What was tried.** Including the Schenk_AFE group in the stress
manifest as called for by the M3 ticket.

**Symptom.** All 16 Schenk_AFE matrices have n ∈ [504855, 1508065].
The M3 issue specifies "n ≤ 100k" for the GHS_indef tier;
extrapolating that size cap to the AFE group leaves zero candidates.
Of the 16, 10 are SPD (`af_*_k101` family + half of `af_shell*`),
which the issue explicitly excludes ("skip the SPD ones"). The 6
indefinite shells (`af_shell1/2/5/6/9/10`) range from 504k to 1.5M
rows; sample timing on an existing 50k×500k matrix is ~40 ms factor,
so a 500k×17M shell would land in the 5–20 s range each. Six of
them would be ~1–2 minutes of suite time — fine for budget — but
they are coarse mesh slices of the same finite-element problem
(automotive shell, sequential time steps) and add little diversity
beyond a single representative.

**Why it was rejected (for this issue).** Bringing in 6 near-duplicate
mesh slices burns row-count headroom that's better spent on
structurally diverse matrices in the smaller-n tier. A future ticket
that wants to stress the dense-supernode path on million-row
matrices should add 1–2 representative `af_shell*` rows with a fresh
research note on whether they exhibit fill patterns distinct from
what `sparsine` / `copter2` already cover.

---

## 2026-05-16 — Issue #11 `SmallLeafBatch::On` default flip, post-SIMD+APP re-eval

**Phase.** Issue #11 — re-evaluate the Phase 2.11 default flip
after #9 (SIMD trailing-update kernel) and #10 (APP pivoting)
landed and closed. Hypothesis: with per-front kernel cost
reduced by SIMD + APP, the driver-overhead savings from
small-leaf batching should finally clear the +10% median /
−5% worst-case bar that the Phase 2.11 5-run repeat failed.

**Protocol.** 5 runs per variant on the full
`external_benchmarks/comparison/sample.tsv` (66 matrices that
have synthetic RHS on disk, spanning tiny/small/medium/large/
xlarge buckets). Variants interleaved per-run to mitigate
thermal / scheduler bias. Driver was
`target/release/bench_one_matrix` with a one-off
`FERAL_SMALL_LEAF=on|off` env-var override applied in
`src/bin/bench_one_matrix.rs` to A/B without recompiling. The
library default of `SmallLeafBatch::Off` was unchanged for the
run.

Per-matrix median across 5 runs, then per-bucket median speedup
and worst per-matrix slowdown. Driver: `solve_one` measures
`factor_us` via `Instant::now()` around
`factorize_multifrontal_parallel_with_workspace`. Wall-time per
run was ~5–6 min (10 runs total ≈ 55 min). Decision criterion
(must meet BOTH): median speedup ≥ +10% on tiny + small, worst
per-matrix slowdown ≤ +5%.

**Result.** Per-bucket medians (positive = `On` faster):

| bucket | n_mats | med_speedup_% | worst_slowdown_% | best_speedup_% |
|--------|-------:|--------------:|-----------------:|---------------:|
| tiny   |     20 |          0.00 |           +50.00 |         +40.00 |
| small  |     19 |         +1.90 |            +9.09 |          +8.97 |
| medium |      8 |         +0.30 |            +2.01 |          +6.37 |
| large  |     10 |         +7.70 |           +12.53 |         +21.46 |
| xlarge |      7 |         −4.35 |            +6.55 |          +8.57 |

Decision (tiny + small combined, n=39):
- median speedup +1.44% (criterion ≥ +10%) — **FAIL**
- worst per-matrix slowdown +50.00% (criterion ≤ +5%) — **FAIL**

**Worst offenders, tiny + small bucket (Off → On µs, median of 5):**

| matrix                | n   | off | on  | slow_% |
|-----------------------|----:|----:|----:|-------:|
| OSBORNE1_0041         |   5 |   2 |   3 | +50.00 |
| LANCZOS1_0029         |   6 |   2 |   3 | +50.00 |
| heart6_iter_c         |  12 |   4 |   5 | +25.00 |
| METHANB8LS_0004       |  31 |  19 |  21 | +10.53 |
| QPCBLEND_0210         | 157 |  88 |  96 |  +9.09 |
| HIMMELBJ_0023         |  57 |  50 |  53 |  +6.00 |

The +25–50% slowdowns on n≤12 matrices are 1 µs `Instant`
resolution noise (2→3, 4→5 µs); the QPCBLEND_0210 and
METHANB8LS_0004 entries (n=157, n=31) are real but ≤+10%.

**Best wins, tiny + small (Off → On µs, median of 5):**

| matrix                | n   | off | on  | speedup |
|-----------------------|----:|----:|----:|--------:|
| heart6_iter_b         |  12 |   5 |   3 |   1.667 |
| OSBORNEB_0008         |  11 |   4 |   3 |   1.333 |
| VESUVIA_0040          |   8 |   5 |   4 |   1.250 |
| BT2_0006              |   4 |   6 |   5 |   1.200 |
| HAIFAM_0370           | 249 | 145 | 132 |   1.098 |

The wins on n≤12 matrices are likewise within timer-resolution
noise; the only real bucket-relevant win is HAIFAM_0370 (+9.8%)
and a handful of small-bucket entries in the +5–8% range.

**Why rejected.** The post-SIMD + post-APP measurement
replicates the Phase 2.11 conclusion: small-leaf batching does
not move the median on tiny + small matrices clear of noise.
The hypothesis (kernel-cost amortization would expose
driver-overhead savings) is not supported by the data:
- tiny bucket median speedup is exactly 0.0% (the 1 µs timer
  resolution dominates).
- small bucket median speedup is +1.9% (within noise; the
  Phase 2.11 5-run repeat reported ±5% per-matrix noise on the
  IPM tail).
- The +50% / +25% / +10% slowdowns on individual matrices put
  the worst-case far above the +5% bar regardless of whether
  the median moved.

The `large` bucket shows a +7.7% median speedup with a +12.5%
worst slowdown, which is interesting but out of scope for
issue #11 (the criterion is tiny + small per the SSIDS-style
small-front amortization target).

**Disposition.**
- `SmallLeafBatch::Off` remains the compiled-in default. The
  doc-comment at `src/numeric/factorize.rs:203-209` already
  records the Phase 2.11 rejection; this re-eval extends that
  rejection to the post-SIMD + post-APP kernel regime.
- The diagnostics from this session are NOT kept in tree per
  the issue's failure-path workflow (single commit appending
  this entry). The harness (`issue_11_ab.py` + the
  `FERAL_SMALL_LEAF` env-var hook in `bench_one_matrix.rs`)
  is reproducible from this entry plus the journal at
  `dev/journal/2026-05-16-11.org` should a future agent want
  to re-run after another kernel improvement.
- Per the matching reasoning in the original Phase 2.11
  rejection: the tail gap is structural (bushy elimination
  tree, addressed by the Phase 2.12 column-renumbering
  amalgamation), not a driver-overhead problem the small-leaf
  fast path can solve. Closing #11.

**Evidence.** Full run log: `/tmp/issue11_full.log` (10 runs,
55 min wall). Per-matrix sidecars: `out_ab/{off,on}/run{0..4}/`.
Summary JSON: `out_ab/issue_11_summary.json`. Both deleted at
the end of session per workflow; reproduce from
`dev/journal/2026-05-16-11.org`.


## 2026-05-16: `OrderingMethod::AutoRace` as `Solver::new()` default (proposed for #37)

**What was tried.** Switching `Solver::new()`'s default ordering
from `OrderingMethod::Auto` to `OrderingMethod::AutoRace` to close
issue #37 (pinene_3200 CB=off regression) without requiring the
per-problem `Solver::with_cascade_break(0.5)` workaround. The
hypothesis rested on the c92cafe commit message claiming
"on pinene_3200_0009 the [`pick_default_method`] heuristic picks
`MetisND` (88 s numeric factor), but `Amd` factors in 19.5 s on
the same matrix — a 4.5× win that the cheap predicate misses."

**How it failed.** Diag runs of `diag_pinene_amd` on the actual
default-config Solver (CB=off since 585d739) show AMD on
pinene_3200 is **catastrophically worse** than MetisND, not better:

| variant            | iter   | factor    | delay_in   | n_2x2 |
|--------------------|--------|-----------|------------|-------|
| AMD     CB=off     | _0009  |  917.477s | 13,572,596 | 21,001|
| AMD     CB=off     | _0008  | 1055.306s | 13,425,707 | 20,790|
| MetisND CB=off     | _0009  |   ~88s    | (per c92cafe) |   —   |
| AMD     CB=on (0.5)| _0009  |   19.5s   | (per c92cafe) |   —   |

The 4.5× AMD win in the c92cafe writeup was measured under
cascade-break ARMED (CB ratio in the 0.94–0.95 sweet spot per
`dev/journal/2026-05-13-03.org`). After 585d739 made CB opt-in,
the AMD-without-CB path produces 13.5M delayed pivots that
cascade through the elimination tree, dominating the factor cost.
MetisND is "less bad" only because its larger root supernode
absorbs more delays before the cascade compounds.

Robot_1600_0003 (the #17 regression guard) was clean under either
ordering (both produce neg=9601 matching MUMPS, AMD 1.4× faster).
So the regression guard would not have caught this — the failure
is specific to pinene's elimination tree shape, not to AMD itself.

**Disposition.**
- `Solver::new()` retains `OrderingMethod::Auto` as the default.
- The c92cafe claim about AMD speedup on pinene is now historical
  (CB=on regime only) and should not be cited as motivation for
  default-ordering changes.
- Closing #37 requires fixing the underlying CB-mechanism gap, not
  the ordering choice. The follow-up investigation (see
  `dev/sessions/2026-05-16-30.md` and `dev/journal/2026-05-16-30.org`)
  redirects to issue #38 and surfaces a separate silent-correctness
  bug — stale MC64 cache producing wrong inertia on warm IPM
  re-factors — that is the more likely root cause for both #37 and #38.

**Evidence.** Diag outputs at
`/private/tmp/claude-501/-Users-jkitchin-projects-feral/<session>/tasks/{br9xq3zub,bh8zlfol7,b0ozjbi4h}.output`
(retained for the session; reproducible via
`cargo run --release --bin diag_pinene_amd -- pinene_3200_0009`,
`pinene_3200_0008`, and `robot_1600_0003` from a checkout at HEAD).

---

## 2026-05-16 — Behavioural integration test for #38 cache staleness on 4×4 matrix

**What.** Initial regression test for the issue #38 MC64-cache-staleness fix
(`tests/issue_38_mc64_cache_warm_vs_fresh.rs`) factored two value-perturbed
matrices on the same block-anti-diagonal 4×4 pattern through one warm
`Solver` and asserted `warm.inertia() == fresh.inertia()`. Mirrors the
rocket_12800 reproducer's warm-vs-fresh comparison in miniature.

**Why it was rejected.** The test passed with the fix removed. Sylvester's
law of inertia preserves inertia exactly under any symmetric non-singular
scaling, so on small well-conditioned matrices applying iter-0 MC64
scaling to iter-N values produces the correct inertia regardless. The bug
only manifests as wrong inertia on large arrow-KKTs where the mis-scaling
destabilises Bunch-Kaufman pivoting enough to trigger a delayed-pivot
cascade — and that cascade is not reproducible on a 4×4 matrix.

**Disposition.** Replaced with an in-module unit test
(`numeric::solver::tests::mc64_cache_invalidated_after_factor_issue_38`)
that inspects `last_symbolic.cached_mc64` directly and asserts it is
`None` after one `factor()` call. The pub(crate) field is only accessible
from `super::*` so the test had to move from `tests/` to the in-module
`#[cfg(test)]` block. Verified the unit test fails when the fix is
removed (panics on the assertion) and passes when restored.

**Lesson.** Behavioural tests for scaling-related bugs need either (a) a
matrix large enough to expose BK pivot-threshold sensitivity, which means
shipping corpus data, or (b) a direct-field assertion on the cache state.
For one-shot caches that should be cleared per call, (b) is cheaper and
more targeted than (a).

**Evidence.** See `dev/research/mc64-cache-staleness-2026-05-16.md` and
the diagnostic tables in `dev/journal/2026-05-16-30.org`. Verification
procedure (toggle fix, run `cargo test --release --lib
numeric::solver::tests::mc64_cache_invalidated_after_factor_issue_38`,
observe pass/fail) reproducible from HEAD.


## 2026-05-17 — cibuildwheel for Python wheels with a `path = ".."` workspace dep

**Approach.** Build Python wheels for `feral-solver` with cibuildwheel
2.21.3 from `.github/workflows/python-wheels.yml`. Tried three layered
configurations:

1. `CIBW_BUILD_FRONTEND: "build[uv]"` with `CIBW_BEFORE_BUILD_LINUX:
   "pip install maturin && rustup toolchain install stable"`.
2. After (1) failed on macOS/Windows: kept cibuildwheel, switched to
   plain `"build"` frontend (`uv` is not on the runner), bootstrapped
   rustup in the manylinux container via the `sh.rustup.rs` installer.
3. Both of the above with the matrix unchanged.

**Symptom of (1).** macOS-14 and windows-latest runners: `uv: command
not found` during cibuildwheel setup. Linux: `rustup: command not
found` inside the manylinux container. (commit 07d385e fixed both.)

**Symptom of (2) — the deeper failure.** Even after rustup +
`pip install maturin` ran cleanly inside the manylinux container,
`python -m build` errored with:

    error: failed to load manifest for dependency `feral`
      Caused by: failed to read `/Cargo.toml`
      Caused by: No such file or directory (os error 2)

cibuildwheel copies only the package dir (`python/`) into its build
sandbox at `/project`. `python/Cargo.toml` declares
`feral = { path = ".." }`, which resolves to `/Cargo.toml` inside the
sandbox — and that file does not exist because cibuildwheel never
shipped the parent crate. No `CIBW_BEFORE_BUILD` hook can fix this:
the source isn't even in the container.

**Disposition.** Replaced the wheel matrix with
`PyO3/maturin-action@v1` (commit 2442d1f). That action mounts the
whole `$GITHUB_WORKSPACE` into the manylinux container (and runs
natively on macOS/Windows), so the workspace path dependency just
works. Verified end-to-end via release-event run 26003542088:
4 wheel jobs + sdist + smoke-test + PyPI publish all green.

**Trade-off.** cibuildwheel offered `CIBW_TEST_COMMAND` to pytest
each wheel inside its build sandbox. `maturin-action` does not have a
direct equivalent. The `test` job (linux × py3.10/3.12/3.13) and the
`smoke-test` job (linux wheel + `uv pip install` + quickstart.py)
still gate the release, so coverage for the platforms that matter
most for the release gate is intact. Per-platform wheel-pytest could
be added back as a separate job that downloads the wheel artifact
and runs pytest against it, but for v0.4.0 it was not worth the
churn.

**Lesson.** cibuildwheel is the wrong tool when the Python crate has
a sibling-path Rust dependency. The package-dir-only copy semantics
fundamentally cannot see the parent. Reach for `maturin-action`
first when the layout is a Rust workspace with a Python crate
inside it.

**Evidence.** Run 26002981755 (initial failure, 4/4 wheels red).
Run 26003051776 (after fix 1, 3/4 still red — log of job
76429732912 contains the cargo manifest error verbatim). Run
26003260115 (after switching to maturin-action, 4/4 wheels green).
Run 26003542088 (re-cut v0.4.0 release event, full pipeline green
through PyPI publish).

---

## 2026-05-20 — #46 "activation-predicate" diagnosis (wrong; overturned by probes)

**What.** A three-agent research phase (one reading feral's pivoting
machinery, one MUMPS 5.8.2, one SPRAL SSIDS) diagnosed the issue-#46
delayed-pivot cascade on zero-(2,2)-block saddle KKTs as an
*analysis-phase ordering failure*. The proposed fix: broaden
`pick_ordering_preprocess` (`src/symbolic/mod.rs`) with a
zero/absent-diagonal-fraction predicate (MUMPS `dana_aux.F:1887`-style,
threshold ≈ 0.10·n) so `OrderingPreprocess::LdltCompress` would turn on
for these matrices. Captured in
`dev/research/kkt-zero-2x2-block-cascade-2026-05-20.md` (now corrected)
and `dev/plans/kkt-zero-2x2-cascade-fix.md` (now corrected) as the
load-bearing "Phase 1".

**Why rejected.** Ground-truth probes on the real CHO `parmest` KKT
refuted every load-bearing claim before any activation code was written:

- `probe_issue46_preprocess` — feral stores only the lower triangle, so
  KKT constraint columns are stored-degree 0/1, **not** high-degree. The
  existing `low_degree(≤2)` predicate already fires (frac 0.7505);
  `pick_ordering_preprocess` **already** returns `LdltCompress` for the
  CHO KKT and `build_supermap` already forms 21 660 pairs. The proposed
  activation predicate is a no-op.
- `probe_issue46_supernode` — with `LdltCompress` active: symbolic
  `factor_nnz_estimate = 1.22M`, numeric `factor_nnz = 28.05M` (23×
  blowup), max supernode `ncol = 133` (no giant root supernode),
  **20 918 / 21 660 pairs co-located in the same supernode, 20 794 at
  adjacent columns (96.6 %)**. The ordering is fine and the pairs are
  co-located; the cascade is purely a numeric delayed-pivot blowup.
- The MUMPS/SSIDS conclusion "matching-based ordering *is* the fix" was
  incomplete: MUMPS/MA57 also rely on MC64 *scaling* (makes matched
  entries magnitude ≈ 1 so BK's argmax hits the partner). feral's MC64
  scaling is degenerate on saddles (#45) and rejected — feral cannot use
  that mechanism. Worse, on CHO `preprocess=None` actually produced
  *less* fill (21.9M) than `LdltCompress` (28M).

**The actual bug** was the numeric kernel: `scalar_pivot_step`'s 2×2
partner search only considered the magnitude-argmax row `r` and, when
`r` was out-of-front, delayed instead of using the co-located partner
at `k+1`. Fixed there (see `decisions.md` 2026-05-20 #46 entry).

**Lesson.** A convergent multi-agent diagnosis is not evidence — three
agents reading reference solvers agreed on a story that a single probe
on the real matrix overturned in minutes. Probe the actual failing
matrix *before* writing a research note's "recommended fix", not after.

## 2026-05-21 — B2 value-bounded MC64 scaling cache: gate metric confounded by IPM δ

**Approach.** Track B2 of the per-factor cost-cluster plan: eliminate
the per-IPM-iteration MC64 Hungarian cost by caching the iter-0 MC64
scaling vector `D₀` at `Solver` scope and reusing it on warm
`factor()` replays, gated by an O(nnz) "value-bound" check
(`mc64_value_bound_passes`). The check accepts reuse while the
*diagonal dominance* of `D₀·A_N·D₀` stays within a growth budget of
its iter-0 baseline — the premise (`mc64-value-bounded-cache-2026-05-17.md`)
being that MC64 scaling is pattern-dominated and `D₀` stays "good"
across iterations as long as `D₀·A_N·D₀` remains diagonally dominant
enough that Bunch-Kaufman picks the same pivots.

**Symptoms / why rejected (as a payoff lever — the code itself is
correct and ships as latent infrastructure).**

- The value-bound gate rejects **every** warm iteration on
  pinene_3200. DBG instrumentation on condition 1 (ratio growth):
  iter 2 max_ratio 1.906e8 (budget 1.162e8); iter 3 7.770e8
  (5.180e8); iter 4 2.486e10 (1.657e10); iter 5 5.562e10 (3.708e10).
  All FAIL. `mc64 scaling-cache hits: 0`.

- The metric is **confounded**. The baseline `r0 ≈ 5.8e7` shows the
  MC64-scaled KKT is nowhere near diagonally dominant in the first
  place. The KKT (2,2)-block rows carry a tiny δ-regularized diagonal
  (≈1e-8) against ≈1 off-diagonals, so their off/diag ratio is ≈1/δ.
  As the interior-point method drives the regularization δ→0, the
  ratio explodes 1e8→1e10 — the gate is measuring the IPM's barrier
  trajectory, not whether `D₀` is still a usable scaling. There is no
  `GROWTH_FACTOR` value that separates "δ shrank" (safe to reuse)
  from "matching changed" (unsafe — corrupts inertia, #38). The
  premise that MC64-scaled indefinite KKT is diagonally dominant is
  false; "diagonal dominance of `D·A·D`" is the wrong proxy.

- Even with a perfect gate, B2 targets <2 % of the cost. pinene_3200's
  10 iters total 493.9 s; iters 6-9 are 64.8/77.8/135.7/208.2 s (the
  cost-cluster blowup, 98 %). The MC64 Hungarian is ≤6 s total.

- The named target rocket_12800 cannot even exhibit a hit: its 2-iter
  dump changes pattern between iters (332793→435190 nnz).

**What was kept.** The cache wiring (`Solver::with_mc64_cache`),
`src/scaling/value_bound.rs`, and — separately — the `External`
scaling correctness fix B2 surfaced (see `decisions.md` 2026-05-21).
All correct and tested; the *approach* of a cheap value-proxy gate
for cross-iteration MC64 reuse is what is rejected.

**Lesson.** Validate the cost model before building the optimization.
B2 assumed "MC64 Hungarian reruns every IPM iter and dominates" — true
for rocket_12800's iter-0 profile, false for pinene's actual 10-iter
trajectory where the delayed-pivot blowup dwarfs everything. A
per-factor profile of the *named target's full iteration sequence*,
not a single iteration, should precede the plan.

## 2026-05-22 — #44 NARX_CFy: amalgamation lever and contrib zero-fill removal

Two optimization ideas for the `NARX_CFy` numeric loop, both rejected
after measurement.

**Amalgamating tiny fronts — refuted by the front-size distribution.**
Hypothesis: NARX's loop is dominated by per-front fixed overhead, so
merging small supernodes amortizes it. `probe_narx_phases`
size-distribution: 35 430 fronts with `ncol ≤ 4` cost 0.2% of the loop
*combined*; ~950 medium fronts (`ncol` 5–64) carry 93%. The tiny
fronts are already free; there is nothing to amortize. Amalgamation is
not the lever.

**Removing the contrib-block zero-fill — not free, not pursued.**
The 16:00 journal claim "the `resize(cdim*cdim, 0.0)` is 100%
removable, provably safe" was wrong: it checked only `extend_add` (a
lower-triangle-only reader). Grepping every reader of `.contrib` found
**three consumers that bit-compare the full contrib Vec including the
upper triangle**: the `block_ldlt32` unit test (`to_bits()` per
element), `parallel_corpus_parity.rs:70`, and `diag_par_firstdiff.rs`.
The zero-fill is what makes the upper triangle deterministically
`0.0`; deleting it naively regresses the test and breaks parity.
Removing only its cost requires `unsafe Vec::set_len` — safe Rust
cannot length a `Vec` without N initializing writes, and `src/` has no
`unsafe` in the core numeric data path. The genuinely-wasted portion
is ~2% (the lower-triangle zeros the copy overwrites anyway); the
other half is load-bearing. Decision (jrk): not worth the first
core-path `unsafe` for ~2% on an already-correct solver. Issue #44
closed.

**Lesson.** "Provably dead" requires grepping *all* consumers of the
buffer, not the one obvious algorithmic reader. Diagnostic and test
binaries that bit-compare whole buffers make "never read" false.

## 2026-05-30 — Multi-RHS BLAS-3 GEMM loop reorder (c-block outer) — no effect, reverted (#57)

While implementing the issue #57 fix #2 BLAS-3 panel kernel, the first
draft of `gemm_panel_minus` tiled with the **row tile (MR) outer,
column block (NR) inner**. The n=1024 grid regressed (batched slower
than looping). Hypothesis: the m-outer order re-streams the large
`B` panel (`k_dim × nrhs`) `m_dim/MR` times, and swapping to
**c-block outer / m-tile inner** would keep a small `B`-block
L1-resident and cut the dominant re-streaming by the factor `NR/MR = 2`.

Tried the swap. **Measured: no improvement.** n=1024 stayed at ratio
~1.0–1.2 (still a regression), and n=484/2025 were within noise of the
m-outer order. The loop order was not the bottleneck at these sizes.

Reverted to the simpler m-outer kernel (the comment claiming the swap
fixed n=1024 would have been false). The actual n=1024 cause was the
**stride-`n` gather/scatter** reading the column-major `y` — power-of-
two `n` aliased RHS columns into the same cache sets. Flipping the
internal `y` buffer to row-major (contiguous memcpy gather/scatter)
fixed it (ratio 1.2 → 0.33) and ~halved wide-solve time everywhere.
See `dev/research/issue-57-blas3-panel.md` Results and the
`dev/decisions.md` 2026-05-30 entry.

**Lesson.** Diagnose the bottleneck before micro-optimizing the kernel:
the transpose in the gather/scatter dominated, not the GEMM's operand
re-streaming. A loop-order change to the GEMM was wasted motion until
the layout (row-major `y`) was fixed.

## 2026-06-03 — FERAL-side fixes for issue #63 (near-singular KKT IPM stall)

Context: scrs8-2c-8 stalls at "Acceptable" (constr.viol 2.30e-8) under
amd/amf/Auto but reaches "Optimal" (1.00e-8) under metis/scotch. Issue #63
hypothesised an ordering-dependent linear-solve backward error (~2e-8 AMD vs
~1e-8 metis) flooring IPM feasibility, and suggested pivot stabilization /
growth bounds / more refinement.

REJECTED (disproven by measurement, not shipped):

1. The premise. FERAL solves the iter-26 regularized stepped system (full rank)
   to ~1e-22 backward error under EVERY ordering AND both scaling modes (Auto,
   Identity), even at max_piv ~6e16. The claimed 2e-8/1e-8 gap and
   max_abs_pivot ~9.68e10 are not observed (auto-scaled max_piv ~30-60). The
   linear solve is not the bottleneck — so pivot stabilization / growth bounds /
   more refinement target a backward error that is already ~1e-22.

2. MA57-style static pivot (with_static_pivot_threshold(1e-8)). On the SINGULAR
   pre-regularization matrix the floor is 1e-8*||A||_inf ≈ 1.0 (μ→0 (1,1)-block
   blowup), perturbing ~510-542 of 727 pivots → backward error 1.0 (solve
   destroyed), inertia scrambled. Strictly worse. Force-accept-and-report-zeros
   is the useful behavior: it signals singularity so pounce escalates δ_w.

3. Any principled "better inertia" change. The ordering that wins (metis)
   reports a MORE pessimistic, LESS correct inertia (neg 255 ≠ 252 expected) on
   the singular matrix; that makes pounce regularize earlier and escape a frozen
   2.30e-8 fixed point. There is no known-correct inertia change that fixes
   scrs8 — "correct" inertia (amf) is what under-regularizes into the stall.

4. Ordering-class heuristic (route this KKT class to metis/scotch). Not pursued:
   the issue itself calls it "papering over the symptom," and it risks the
   cascade-break don't-regress set (robot_1600, NARX_CFy, marine_1600,
   rocket_12800, pinene_3200).

Conclusion: the durable fix is the δ_w / inertia-acceptance interaction
(pounce-side or joint), not FERAL factorization accuracy. Full analysis:
dev/research/issue-63-nearsingular-ordering-diagnosis.md;
dev/journal/2026-06-03-02.org; probe src/bin/probe_issue63_nearsingular.rs.
Future sessions: do NOT re-attempt a FERAL-only fix for scrs8 without first
re-checking these four dead ends.

## 2026-06-03 — Fill-guarded AMF reroute above 100k (issue #73)

The plan for extending the #67 AMF reroute past `AMF_BAND_MAX` (100k) was a
**fill-guarded race**: above 100k, route a would-be-MetisND `Auto` matrix to
AMF only when AMF's predicted fill `factor_nnz_estimate ≤ MetisND's`,
otherwise keep MetisND. The appeal: the symbolic probe (`probe_issue73_symbolic`)
showed nnz_L / flop_proxy already separated the AMF-wins families from the lone
predicted-MetisND-win (nql180) at ~zero cost relative to the numeric factor, so
a fill guard looked like it would capture the wins without an nql180 regression.
This design is recorded in the #73 research note's "Recommendation" (option b)
and was the originally-requested next step.

Rejected by the real factor+solve A/B (`probe_issue67_thin --reps 1`). The
guard's predicate is **anti-correlated with real speed on nql180**: MetisND has
2% *smaller* fill (fill_r 0.98) yet AMF is **2.05× faster** on the actual
factor+solve (fac_amf 1903 ms vs fac_met 3949 ms). The fill guard would have
read "MetisND fill is smaller → keep MetisND" and **forfeited a 2× speedup**.
nnz_L and the Σ ncol·nrow² flop_proxy do not predict factor+solve wall-time at
this scale (the numeric phase's cache/critical-path behavior dominates), so any
routing guard keyed on symbolic fill makes the wrong call exactly where it
matters — and adds a per-solve symbolic-race cost to do it.

Symptoms / evidence of the failure: on real factor+solve AMF wins ALL 5
measured n>100k families (dtoc2 2.49×, pinene 1.18×, cont5_1_l 2.75×, nql180
2.05×, YATP1NE 2.13×), including the matrix the guard would have demoted.
Superseded by the **unconditional** AMF extension (drop `AMF_BAND_MAX`; route
every would-be-MetisND decision to AMF), recorded in `dev/decisions.md`
(2026-06-03, issue #73). Future sessions: do NOT reintroduce a fill / nnz_L /
flop_proxy guard on the n>100k AMF reroute — nql180 is the standing
counterexample. Data: dev/research/issue-73-n100k-thin-regime.md (Finding 3),
dev/journal/2026-06-03-06.org (:issue-73:ab:factor-solve:surprise:).

---

## 2026-06-06 (issue #80): "Fix slow AMD / implement bucketed min-degree" — WRONG TARGET

What was tried: issue #80 reports a ~55s "ordering" stage on the pf22
powerflow KKT (n=2.8M) and asks for a faster AMD / bucketed min-degree. First
investigation profiled `src/ordering/amd.rs::amd_order`, found it is genuinely
O(n²) (linear min-degree selection scan; synthetic path graph 50k/100k/200k/
400k → 0.99/4.13/16.9/64.7s, clean quadratic), and was about to implement
degree buckets there.

Why rejected:
1. `src/ordering/amd.rs::amd_order` is **dead code**. Production dispatches
   `OrderingMethod::Amd` to `feral_amd::amd_order` (`symbolic/mod.rs:569`,
   `schur.rs:200`). Only `permute_pattern` from that file is still used. The
   real `feral_amd` is already a bucketed quotient-graph AMD and orders pf22 in
   **0.276s**. Implementing bucketed min-degree there would have fixed a
   function nobody calls.
2. The real ~53s is the **`LdltCompress` preprocessor's MC64 matching**
   (`mc64::compute_matching`, ~O(n^1.9)), which the per-stage profiler folded
   into the `ordering` stage timer. `preprocess=None` drops total symbolic
   from 54.5s to 1.23s.

Symptoms that revealed the false start: on real pf22 values
`feral_amd::amd_order` = 0.276s while the full symbolic = 54.5s with `ordering`
stage 53.6s; forcing `preprocess=None` collapsed it to 1.23s. With `vals=1.0`
(MC64 trivial) symbolic was only 1.5s — the value-dependence is the tell that
the cost is in MC64, not the structure-only ordering.

Future sessions: do NOT "optimize" `src/ordering/amd.rs` for performance — it
is not in the factorization path. The production AMD is `feral_amd`. For
issue #80 the lever is MC64 (gate it on large arrow-signature KKTs), not AMD.
Data: dev/research/issue-80-mc64-preprocessor-cost.md,
dev/journal/2026-06-06-01.org.

---

## 2026-06-06-03 — MC64 dense-column column-lower-bound inner-scan skip

What was tried: a behavior-preserving fast path for the MC64 Hungarian
dense-column cost (audit §8.1 option (a)). On rocket_12800 the matching is
O(searches × dense_deg): the degree-38401 coupling column is matched in the
main loop and rescanned every time its matched row is popped. The idea:
because `u` is monotone non-increasing over the main loop (the only update,
`u[i] += d[i]-csp` on rows popped with `d[i] < csp`, decreases it),
`lb[j] = min_k(cost[k] - u_init[row(k)])` is a permanent lower bound on a
column's minimum reduced cost. In the inner scan, `vj + lb[j2] >= csp` would
imply every edge has `dnew >= csp`, so the whole column scan is a provable
no-op and can be skipped bit-identically. Implemented (O(nnz) lb pass + a
one-line guard + `inner_scan_skips`/`edges_saved` counters).

Why rejected:
1. **Measured zero firing.** On rocket_12800_0000: `inner_scan_skips=0`,
   `edges_saved=0`, `edge_scans=3.71e8` unchanged, wall 3958 ms unchanged. The
   `u_init` bound is the loosest valid bound and never beats `csp`.
2. **Impossibility proof (any bound, any tightness).** For the matched column
   `j2` of a popped row `q0`, complementary slackness gives
   `v[j2] = cost[jperm[j2]] - u[q0]` and dual feasibility makes that edge the
   column's reduced-cost minimum, so the tightest bound is
   `lb_tight = cost[jperm[j2]] - u[q0]`. With `vj = dq0 - cost[jperm[j2]] +
   u[q0]`, `vj + lb_tight = dq0`. The skip fires iff `dq0 >= csp`, but `q0`
   was popped only because `dq0 < csp`. So a column-level reduced-cost bound
   can NEVER prune. The matched edge always sits exactly at `dq0`; the scan
   exists to find improving edges to *other* rows, about which a column
   aggregate carries no information. Maintaining `v` live would not help.
3. **SPRAL confirms the cost is inherent.** `ref/spral/src/scaling.f90::
   hungarian_match` (938-1171) walks the full matched column every settle with
   only per-entry filters (no range cut, no dense-column special case),
   computes `dualv` once at the end, and its only dense-aware logic is the
   greedy-init claim guard (line 857) — which feral already mirrors. SPRAL has
   the identical O(searches × dense_deg) cost; feral's port is faithful.

Symptoms: the lb-skip compiled, kept all 7 Hungarian unit tests and the full
317-test lib suite green (behavior preserved), but the counters showed it was
inert. Reverted (dead weight: never fires, adds an O(nnz) pass + a branch per
pop).

Future sessions: do NOT attempt to prune the MC64 inner column scan with any
per-column reduced-cost bound — it is provably impossible. The dense-column
MC64 cost is inherent to the sparse shortest-augmenting-path algorithm and
matches the SPRAL reference. The only remaining lever is to AVOID MC64 scaling
on single-dense-column KKTs (a scaling-policy change that alters the scaling
vector → needs a corpus inertia/residual study + human approval per the
constraints). Data: dev/research/mc64-dense-column-2026-06-06.md,
dev/journal/2026-06-06-03.org.

---

## 2026-06-06-04 — Scaling-aware `LdltCompress` skip (skip speculative MC64 when scaling won't reuse the cache)

Proposed "safe win" (dense-column follow-up, audit §8.1 option (b)): in the
symbolic `LdltCompress` branch, skip the MC64 matching when the resolved
numeric scaling will not reuse the cache (Identity/InfNorm/External, i.e. not
`Mc64Symmetric`). Rationale assumed the MC64 is "purely speculative" in that
case.

**Rejected — refuted empirically. The premise is false on two counts:**

1. **The MC64 is load-bearing for compression, not speculative.**
   `LdltCompress` is MUMPS `ICNTL(12)=2` (Duff-Pralet): `build_supermap`
   (ldlt_compress.rs:39-77) walks the MC64 matching permutation's cycle
   structure to form the super-variables. The matching *is* the compression
   input. "Skip the MC64" = "skip compression, fall through to uncompressed
   ordering" — an ordering change.

2. **The scaling-reuse signal does not predict compression cost/benefit.**
   `probe_compress_costbenefit_argv` (symbolic+numeric, None vs LdltCompress,
   5-run median, release) on the large dense-column + InfNorm (won't-reuse)
   bucket:
   - **ROSEPETAL** (n=3000, deg=2001): compression pays 0.68 s MC64 but the
     compressed ordering gives an 8x numeric speedup (5.72 s → 0.77 s), net
     **-75.7% total** (reproducible -75.0%). Skipping it = ~4x regression.
   - **ORTHREGF** (n=6405, deg=1601): compression adds ~5.6 ms MC64 with zero
     numeric benefit, **+91.8% total loss** (reproducible +89.4%).
   Both large, both a near-dense column, both InfNorm — opposite verdicts. The
   value of compression is its numeric fill reduction, which is independent of
   the scaling choice and unpredicted by max_col_deg / MC64 cost / n
   (ROSEPETAL's MC64 is 68x ORTHREGF's, yet ROSEPETAL is the win).

A gate keyed on "scaling won't reuse MC64" would regress the fill-reduction
wins (ROSEPETAL, ex8_2_2) to save milliseconds on the overhead-only losses
(ORTHREGF, SINQUAD2, sub-ms small matrices). Not a safe win.

Bucket size for the record (`probe_compress_scaling_bucket`, 3 roots, 1006
families): 376 LdltCompress, of which 118 reuse MC64 (keep) and 258 do not
(the target bucket — heterogeneous, contains both ROSEPETAL-type wins and
ORTHREGF-type losses).

Future sessions: do NOT gate `LdltCompress` on the scaling strategy. The real
(separate, harder) lever is an orthogonal **compression cost/benefit gate**
that estimates fill reduction vs MC64+ordering cost; the current cheap proxy is
`pick_ordering_preprocess`'s low-degree fraction, and no cheap structural
feature yet separates ROSEPETAL (win) from ORTHREGF (loss). Data:
dev/research/mc64-symbolic-skip-2026-06-06.md, dev/journal/2026-06-06-04.org.
This closes the dense-column follow-up: both option (a) (inner-loop fast path)
and option (b) (scaling-aware skip) are now closed with negative results.

---

## 2026-06-09 — D3: zero-bucket rook-rescued sub-floor 1×1 pivots (finding D3, repo-review-2026-06-09.md)

**Finding (D3):** the rook-rescue 1×1 accept branch in
`try_reject_1x1_with_rook_rescue` (`src/dense/factor.rs`) sign-counts the
rescued pivot with no `zero_tol` / `null_pivot_tol` floor. Rook's 1×1 gate
(`|a_rr| >= u·gamma_r`) is purely *relative*, so when the whole column is
noise it can "rescue" a strict-zero pivot (`|d| <= zero_tol`) and count it by
sign with no `needs_refinement` and no zero bucket — contradicting the
issue-#54 SSIDS strict-zero rule and the F-01 band rule that
`try_reject_1x1_frontal` implements. The finding's implied fix: make the
rook-rescued sub-floor pivot follow the same floor convention (zero bucket /
`needs_refinement`) the rook-free path uses.

**Reproduced as a divergence, rejected as a fix — the corpus shows the
current behavior is the *correct* one.**

A synthetic self-consistency test (rook vs no-rook on the same 2×2 by
Sylvester's law) did reproduce a ±1 inertia divergence. The 2×2 used was
`A = [[1e-3, 1e-4], [1e-4, 1.0]]` with a wide floor `zero_tol = 1e-2`:
- rook path (sign-count): `(2, 0, 0)`
- rook-free reference (floors the `1e-3` pivot to zero): `(1, 0, 1)`

But `A` is symmetric positive definite (`det = 1e-3 - 1e-8 > 0`, leading
minor `1e-3 > 0`), so the **true** inertia is `(2, 0, 0)`. Rook's sign-count
is correct; the "reference" `(1, 0, 1)` is a floor-induced artifact. The
premise that the two paths must agree by Sylvester is flawed: Sylvester's law
governs *exact* inertia, and `zero_tol` is a deliberate numerical floor that
the two paths apply at different points. They legitimately differ — and the
rook path is the more accurate of the two here.

**The implied fix violates the hard inertia constraint on the real corpus.**
Two fix attempts were tried, both regressing `parity_acopp30_0001`
(ACOPP30, a non-singular KKT where MUMPS=`(72,137,0)` and SSIDS=`(71,138,0)`,
both `zero=0`):

1. *Inline zero-bucketing* (count a strict-zero rook rescue in the zero bucket,
   return `Rejected`): feral → `(71, 137, 1)` — a **spurious zero**,
   disagreeing with *both* oracles. Violates "inertia must be exactly correct
   on non-singular matrices; on disagreement, agree with at least one oracle."

2. *Decline-and-fall-through* ("option B": peek the candidate diagonal before
   the swaps; if `|d| <= zero_tol` decline the rescue and route to the standard
   delay / force-accept path, so non-root fronts delay to the parent and only
   the root force-accepts as zero): feral → `(71, 137, 1)` again — same
   spurious zero. The pivot rook was sign-counting on ACOPP30 has `|d| <= EPS`
   (a near-exact zero in feral's elimination order), yet the matrix is
   non-singular: the oracles resolve that DOF to a definite sign via their own
   pivoting/delays, and feral's rook happens to recover the matching sign. Any
   path that does *not* sign-count it (zero-bucket at root, or delay-then-zero)
   produces the spurious singular result.

**Conclusion.** On the only corpus matrix where this branch fires (ACOPP30),
the current floor-less rook sign-count produces the inertia that *matches the
oracles*, and every attempt to impose the floor convention regresses it to a
spurious zero that matches *neither* oracle. The divergence the synthetic test
exposes is real but is by design (conservative `zero_tol` floor vs. rook
recovering the true sign of a borderline-but-nonsingular pivot). The fix
direction is wrong: it trades a correct-but-unconventional result for a
conventional-but-wrong one. No change made; `src/dense/factor.rs` rook branch
left as-is.

Evidence: parity 21/0 with original code (`acopp30_0001` green); the two fix
attempts each produced `acopp30_0001 feral=(71,137,1)` vs `mumps=(72,137,0)`
`ssids=(71,138,0)`. The synthetic SPD `(2,0,0)` vs floored `(1,0,1)` divergence
is a floor artifact, not a bug. Journal: dev/journal/2026-06-09-01.org.

*Possible future work (separate, not D3):* the rook 1×1 branch could set
`needs_refinement = true` when the rescued pivot lands in the band
`(zero_tol, null_pivot_tol]` — a refinement *flag* only, no inertia change, so
it cannot regress ACOPP30. That is an additive diagnostic improvement, not the
inertia-accounting change D3 asks for, and was not pursued here to keep the
rejection clean. Anyone picking it up must verify it does not perturb the
`needs_refinement` expectations of the existing `rook_rescue` tests.

---

## 2026-06-09 — D4 facet (a): naive-det-cancellation as an independent solve bug (finding D4, repo-review-2026-06-09.md)

Finding D4 lists two facets of the solve-time 2×2 gate (`d_block_solve`,
`src/dense/solve.rs`): (a) the gate uses the naive `a*c - b*b`, so a
*genuinely nonsingular* block whose naive determinant rounds to exactly
`0.0` is silently skipped; (b) the gate's floor is absolute
(`zero_tol_2x2 ≈ EPS²`) where the factor side accepts via the SSIDS
scale-invariant floor, so a well-conditioned block at small absolute scale
is skipped.

Facet (b) reproduced and was fixed (see the D4 commit; both sides now share
`ssids_det_floor_fail`). **Facet (a) could not be reproduced as a
factor-accepted-then-skipped bug and is recorded here.**

A naive `a*c - b*b` rounds to exactly `0.0` only when `fl(a*c) = fl(b*b)`,
which requires `|det| = |a·c - b·b| ≲ ULP(a*c) ≈ a·c·2⁻⁵²` — i.e. block
condition `≳ 2⁵²`. But the SSIDS scale-invariant floor the factor side uses
for *acceptance* rejects exactly those blocks: it tests
`|detpiv| = |det|/maxpiv` against `½·max(|detpiv0|, |detpiv1|)`, which a
condition-`2⁵²` block fails by a wide margin. Concretely
`D = [[2⁵³+1, 2⁵³], [2⁵³, 2⁵³]]` has true `det = 2⁵³ > 0` (nonsingular) yet
`detpiv = detpiv0 - detpiv1 = 2⁵³ - (2⁵³-1) = 1` rescaled, far below
`cancel_floor = 2⁵² ≈ 4.5e15` ⇒ `ssids_det_floor_fail = true`. Verified
numerically (`/tmp/check_a.py`): case (a) `detpiv = 0.0` (rejected); the
genuinely-reachable small-scale case (b) `detpiv = 9.9e-17 > 5e-17`
(accepted).

So any block whose naive determinant cancels to `0.0` is ill-conditioned
enough that the factor side never stores it as an invertible 2×2 (it delays
or falls back to 1×1). The solve never sees such a block, so the naive
cancellation cannot, on its own, cause a *validly-accepted* block to be
wrongly skipped. There is no inertia/solution divergence to reproduce on
that axis distinct from facet (b).

The fix routes the solve gate through the *same* `ssids_det_floor_fail`
predicate the factor uses, which makes solve/factor agree on this axis by
construction (a rejected block is skipped on both sides). A hand-built
`Factors` *can* exhibit naive-det-cancellation (the test
`d4_rejected_block_is_skipped_like_factor` builds exactly the `2⁵³` block
above), but that state is unreachable through the real factorization, so it
is pinned as a *consistency* guard (the block must be skipped, matching the
factor), not as a "should-have-been-solved" bug.

Note: the cancellation-free `det_sym2x2` (fma-based) remains necessary at
*factor* time for the inertia *sign* of borderline blocks
(`count_2x2_inertia`), where getting `sign(det)` right on a block near the
floor matters even though the block is rejected for inversion. That is a
separate concern from the solve gate and is unchanged.

Evidence: tests/d4_solve_2x2_gate.rs (facet b reproduced+fixed; facet a
pinned as consistency guard), /tmp/check_a.py, dev/journal/2026-06-09-01.org.

---

## 2026-06-10 — D5: exactly-singular 2×2 → 1/0 → NaN in legacy `factor()` (finding D5, repo-review-2026-06-09.md)

Finding D5 (`src/dense/factor.rs`, `do_2x2_pivot`): under `factor()` +
`ForceAccept`, an *exactly singular* 2×2 pivot block makes
`t = 1.0 / (d00*d11 - 1.0)` divide by zero → `±inf`, the rank-2 weights
`w0/w1` become `inf`/`NaN`, and the NaN is subtracted into the entire
trailing block. The frontal path is guarded (`do_2x2_update` early-returns
on `det == 0`); the legacy `do_2x2_pivot` is not. Confidence in the review
was "certain (path) / **likely** (triggering)".

**The code path exists but is unreachable through `factor()`'s
Bunch-Kaufman pivot selection — the bug cannot be triggered.** Recorded
here per the loop's "anything that can't be reproduced" rule.

### Proof of unreachability

A 2×2 is selected only at step 7, after steps 3/5/6 all fail
(`factor.rs:977-1045`). Let `γ0 = max|A[i,k]|` (col `k` off-diagonal,
attained at row `r`), `γr = max|A[i,r]|` (row `r` off-diagonal). The
selected block is `[[akk, d21], [d21, arr]]` with `|d21| = γ0` (since `r`
is the argmax row of column `k`), `akk = |A[k,k]|`, `arr = |A[r,r]|`.

The three rejection conditions that *force* step 7:
- step 3 fail: `akk < α·γ0`
- step 5 fail: `arr < α·γr`
- step 6 fail: `akk·γr < α·γ0²`

For the block to be exactly singular, `det = akk·arr − γ0² = 0`, i.e.
`akk·arr = γ0²` (taking `arr ≠ 0`; if `arr = 0` then `det = −γ0² < 0`,
not singular, since `γ0 ≠ 0` is guaranteed by the `gamma0 == 0` branch at
`factor.rs:956`).

From `det = 0`: `akk = γ0² / arr`. Substituting into step-6-fail:
`(γ0²/arr)·γr < α·γ0²  ⟹  γr/arr < α  ⟹  arr > γr/α`.
But step-5-fail says `arr < α·γr`. Together:
`γr/α < arr < α·γr  ⟹  1/α < α  ⟹  α² > 1`.
This is false: `α = (1+√17)/8 ≈ 0.6404 < 1`. **Contradiction.** No exactly
singular 2×2 block can satisfy the step-5 and step-6 rejection
conditions simultaneously, so BK never feeds one to `do_2x2_pivot`.

Stronger bound (near-singular is also safe): from step-6-fail
`akk < α·γ0²/γr` and step-5-fail `arr < α·γr`, the product
`akk·arr < α²·γ0²`, so
`det = akk·arr − γ0² < (α² − 1)·γ0² ≈ −0.59·γ0² < 0`.
Every BK-selected 2×2 block is comfortably indefinite — its determinant is
bounded *away* from zero by `(1−α²)·γ0² ≈ 0.59·γ0²`. Hence the normalized
`d00·d11 − 1 = det/γ0² ≤ −0.59`, and `t = 1/(d00·d11−1) ∈ [−1.7, 0)` — no
division by zero, no overflow, no NaN. The static-pivot perturbation
(`perturb_2x2_to_floor`) only *lifts* eigenvalues away from zero, so it
cannot create singularity either; with the default `static_pivot_floor = 0`
it is a no-op.

### Empirical corroboration

A deterministic sweep (LCG-seeded, no `rand`/`Date`) of 20,000 `factor()`
calls under `ForceAccept` — sizes n ∈ {3,4,5,6,8}, small-diagonal bias to
provoke 2×2 pivots, and 6,667 adversarial near-singular `[[a, g],[g,
g²/a]]` embeddings with large third-column coupling to push `γr` up —
selected **31,092 actual 2×2 blocks** and produced **zero NaN/inf** in any
output (`d_diag`, `d_subdiag`, `l`). Matches the proof.

### Disposition

No fix and no reproducing test (the path is dead code in practice). A
defensive `if (d00*d11 - 1.0) == 0.0 { /* degenerate */ }` guard mirroring
`do_2x2_update` is *possible* and harmless, but it would be untestable
(unreachable) speculative hardening; deferred rather than added blind. If a
future change to the pivot-selection α-test, the rook rescue, or the
threshold logic ever makes a singular 2×2 selectable, this entry is the
flag to add that guard at the same time.

Evidence: proof above; sweep `runs=20000 total_2x2_blocks=31092 any_nan=0`;
`do_2x2_pivot` at `src/dense/factor.rs` (`t = 1.0/(d00*d11-1.0)`), selection
steps at `factor.rs:977-1047`. Journal: dev/journal/2026-06-10-01.org.

## 2026-06-10 — X3: bench dense-KKT loop uses sparse pivot params (finding X3, repo-review-2026-06-09.md)

The finding (severity medium, confidence "certain (mismatch) / likely (bug rather
than undocumented change)") is that the dense KKT validation loop
(`src/bin/bench.rs:1569`, plus the resample at `:1622`) calls
`factor_single_front(&matrix, &params_kkt_sparse)` with `pivot_threshold = 0.01`,
while the rationale block (`:1356-1375`) mandates `pivot_threshold = 0.0` for the
dense path — "a non-zero threshold here sends rejected pivots through ForceAccept
and zeros out structural pivots on e.g. HYDCAR20, METHANL8, DEGENLPA, HS118."
`params_kkt_dense` (threshold 0.0) is built but only the synthetic micro-benchmarks
use it. The claim: either dense pass rates are quietly depressed, or the comment is
stale; both corrupt dense-vs-sparse triage.

### The mismatch is real; the claimed consequence is NOT reproducible

The mismatch itself is real by inspection (the dense loop reads `params_kkt_sparse`,
contradicting its own rationale). But the *harmful consequence* — that threshold
0.01 corrupts inertia / depresses pass rates on the dense single-front path — could
not be reproduced on any available matrix, including the two the rationale names by
name.

Reproduction attempt 1 (synthetic): four small symmetric indefinite matrices with
deliberately small structural pivots (`3x3_small_diag`, `3x3_tiny`, `4x4_kkt`,
`2x2_indef`) factored under both param sets. Result: identical inertia, identical
`max|L| = 1.0`, identical `needs_refinement = false` on all four. Equilibration
(`factor_single_front` applies `equilibrate_scaling` first) normalizes magnitude,
and BK selects 2×2 pivots for small-diagonal cases, so `pivot_threshold` never bites.

Reproduction attempt 2 (named real matrices): HYDCAR20_0000 (n=198, SSIDS oracle
inertia (99,99,0), `num_delay=60` — a genuine delayed-pivot matrix) and DEGENLPA_0065
(n=35, oracle (20,15,0)) from `tests/data/parity/`. Both param sets give the EXACT
oracle inertia (99,99,0) and (20,15,0) respectively, with identical `max|L|`
(3.59 / 10.6) and identical `needs_refinement = false`. No corruption, no depressed
pass rate.

Reproduction attempt 3 (full parity sweep): all 50 parity matrices with SSIDS
oracles and `n <= 600` factored under both `params_kkt_dense` (0.0) and
`params_kkt_sparse` (0.01). Result: `TOTAL=50 DIVERGE=0`. Zero matrices produce
different inertia between the two thresholds on the dense single-front path.

### Root cause of the non-divergence

`pivot_threshold` is immaterial to *inertia* on the dense single-front path because
the structural-pivot-zeroing branch in `do_1x1_pivot` (`factor.rs:4521-4545`) keys on
`|d| <= zero_tol` (strict zero), NOT on `pivot_threshold * col_max`. The band
`zero_tol < |d| <= pivot_threshold·col_max` routes to "small but real — count by
sign" (`:4585-4592`), which produces the same inertia as the threshold-0.0 "accept by
sign" branch (`:4594-4600`). The threshold only flips `needs_refinement` and pivot
*selection* swaps; on the equilibrated corpus neither changed the committed inertia
on any of the 50 matrices. The rationale's "zeros out structural pivots on HYDCAR20"
describes behavior that the current code (post-equilibration, post-issue-#54
inertia-bucketing) no longer exhibits — i.e. the "comment is stale" branch of the
finding's own disjunction is the reality.

### Disposition

No fix and no reproducing test. Per the /loop protocol ("anything that can't be
reproduced goes to tried-and-rejected citing the finding ID"), X3 is recorded here
rather than fixed: the behavioral consequence the finding asserts does not occur on
any available matrix, so there is no failing test to drive a fix, and changing the
harness wiring blind — with no observable difference to validate against — would be
speculative. The one-line wiring change (`params_kkt_sparse` → `params_kkt_dense` at
`:1569`/`:1622`) is harmless and would align code with comment, but it is a no-op on
every matrix tested, so it is deferred rather than applied without a discriminating
test. The stale rationale comment is the real defect; left for a documentation pass.
If a future BK pivot-selection change ever makes the two thresholds diverge on the
dense path, this entry is the flag to revisit both the wiring and the comment.

Evidence: synthetic sweep (4 matrices, all identical); named-matrix check
(HYDCAR20 (99,99,0), DEGENLPA (20,15,0) — both match oracle under both params); full
parity sweep `TOTAL=50 DIVERGE=0`; `do_1x1_pivot` band logic at
`src/dense/factor.rs:4513-4600`; bench mismatch at `src/bin/bench.rs:1569,1622` vs
rationale `:1356-1375`. Journal: dev/journal/2026-06-10-01.org.

## 2026-06-10 — D9 facets (b)/(c): legacy `factor()` 2×2 pivot-gate parity gaps (finding D9, repo-review-2026-06-09.md)

Finding D9 is a drift catalog across the four duplicated 1×1/2×2 zero-pivot
implementations in `src/dense/factor.rs`. Three facets were **fixed** in the
companion commit (a reproducing unit test for the code defect, doc corrections
for the two stale comments):

- **(a)** `do_1x1_pivot`'s `ZeroPivotAction::PerturbToEps` arm called
  `perturb_to_floor` without incrementing `n_tiny`, violating the documented
  "bump at each `perturb_to_floor` call site" contract honored by its three
  siblings. Fixed + pinned by
  `zero_pivot_n_tiny_tests::do_1x1_pivot_perturb_to_eps_counts_n_tiny`.
- **(d)** Stale F-01 overview comment in `try_reject_1x1_frontal` said case
  (a') "Count as zero in inertia"; the code sign-counts (2026-05-17
  sign-fallback) and the inline comment at the code site is correct. Comment
  corrected.
- **(e)** Both `needs_refinement` field docs said "when ForceAccept fired";
  the flag is also set by `PerturbToEps`, F-01 band pivots, static-pivot
  flooring, and growth flagging. Docs corrected.

Two facets are **recorded here rather than fixed**, because they cannot be
reproduced into a failing test without an external reference-solver oracle, and
"fixing" them blind would change the inertia committed by the legacy scalar
`factor()` path — which the hard constraint "inertia must be exactly correct"
forbids without an external oracle to validate against (CLAUDE.md: never write
both impl and oracle in the same session).

### (b) `factor()` evaluates the Duff-Reid 2×2 bound on the *unperturbed* block, while `scalar_pivot_step` perturbs before the gates

The legacy `factor()` 2×2 path runs the BK/Duff-Reid acceptance test on the raw
block, whereas the frontal `scalar_pivot_step` applies the static-pivot floor
*before* the acceptance gates. When `static_pivot_floor > 0.0` the two paths can
therefore make different 2×2-vs-1×1 pivot decisions on the same block. This is a
*pivot-selection* divergence; whether it changes committed *inertia* on any real
matrix is exactly what cannot be asserted without an oracle. Like D5/X3 before
it, the divergence is on the legacy `factor()` path (other findings — D1, D5 —
flag this path for eventual removal); the frontal path is the production path.

### (c) `factor()` lacks the SSIDS det floor and the issue-#46 partner fallback

The frontal 2×2 logic carries an SSIDS-style determinant floor and the issue-#46
partner-column fallback; the legacy `factor()` 2×2 path (`do_2x2_pivot`) does
not. This is a real feature gap, but adding either to `factor()` changes which
blocks it accepts and how it counts their inertia — again a change that needs a
MUMPS/SSIDS reference inertia to validate, and again on the legacy path.

### Disposition

No code change and no reproducing test for (b)/(c). A test that merely *exhibits*
a path divergence is not enough — the loop requires a test whose *failure* is the
bug, and the bug here is "wrong inertia", which needs an external oracle this
iteration cannot produce. Deferred to a dedicated legacy-`factor()` parity effort
(or its removal), validated against the SSIDS/MUMPS corpus. This entry is the flag
to revisit (b)/(c) when that effort happens.

Evidence: `do_1x1_pivot` / `try_reject_1x1_frontal` / `count_1x1_inertia` /
`scalar_pivot_step` / `do_2x2_pivot` in `src/dense/factor.rs`; sibling n_tiny
contract at `factor.rs` (count_1x1_inertia `n_tiny` doc). Companion commit fixes
(a)/(d)/(e). Journal: dev/journal/2026-06-10-01.org.

## 2026-06-10 — D11: `equilibrate_scaling` O(10·n²) branchy/stride-n access (finding D11, repo-review-2026-06-09.md)

Finding D11 reports that `equilibrate_scaling` (`src/dense/equilibrate.rs:8-45`)
runs up to 10 sweeps of an `n×n` double loop calling `matrix.get(i, j)` per
element, with a branch inside `get` and stride-n access for one half of each
row, and that it runs on every `factor()` / `factor_single_front` call
(`factor.rs:832, 1207, 2332`). Severity: low/certain (**perf**).

### Why this is recorded here rather than fixed

This is a pure performance observation, not a correctness defect, so it cannot
be turned into a reproducing test whose *failure is the bug* — the loop's gating
methodology. Two things were checked before concluding that:

1. **No stale-data correctness bug.** `get(i, j)` for `i < j` returns
   `data[i*n + j]`, which is the *lower-triangle* storage of the symmetric
   entry `(j, i)` (row `j` ≥ col `i`), not the strict upper triangle that
   `SymmetricMatrix::from_pooled_buf` leaves stale (cf. D10). So every read is
   valid lower-triangle data; equilibration produces correct scalings even on
   pooled-buffer fronts. There is no wrong-output behavior to reproduce.

2. **No admissible RED gate exists.** A timing assertion is flaky and is not an
   acceptable test gate. The only other "test-first" option would be a
   characterization test that pins the current scaling output and then refactors
   the loop to be faster while keeping it green — but that makes the *current
   implementation its own oracle*, which the hard rule forbids (CLAUDE.md:
   "NEVER write both the implementation and the test oracle in the same session
   without the oracle coming from an external source"). The refactor would also
   touch the hot path of every `factor()` call with no failing test to catch a
   regression, contradicting "correctness before performance, always."

### Disposition

No code change this iteration. The optimization is real and safe in principle —
the max-reduction over `j` is order-independent (associative/commutative max),
so splitting the inner loop into the contiguous `j ∈ (i, n)` half (stride-1 in
column `i`) and the strided `j ∈ [0, i]` half, or hoisting `d[i]` out of the
inner loop, would preserve the result bit-for-bit. It is deferred to a dedicated
dense-factor performance pass that can (a) bring an external/hand oracle for the
scaling vector and (b) benchmark the hot path before/after. This entry is the
flag to revisit when that pass happens.

Evidence: `src/dense/equilibrate.rs:8-45`; `SymmetricMatrix::get`
(`src/dense/matrix.rs:85-91`) reads lower-triangle storage for both branches;
callers at `src/dense/factor.rs:832, 1207, 2332`. Journal:
dev/journal/2026-06-10-01.org.

## 2026-06-10 — D12: `flag_growth_for_refinement` extra O(nrow·nelim) pass over L (finding D12, repo-review-2026-06-09.md)

Finding D12 reports that `flag_growth_for_refinement` (`src/dense/factor.rs:409-419`)
walks the entire extracted `L` slice (`O(nrow·nelim)`) at every dense-factor exit
(`factor.rs:1160, 1772, 2267, 2501`) checking `|v| > L_GROWTH_THRESHOLD`, and
that this scan "could be fused into the L-extract loop." Severity: low/certain
(**perf**).

### Why this is recorded here rather than fixed

Pure performance observation — there is no incorrect behavior to reproduce.
The function is correct (it early-returns once `needs_refinement` is already set
and short-circuits on the first over-threshold entry) and is already pinned by
existing characterization tests (`factor.rs:5061-5112`). The finding is purely
"this is a redundant second pass that could be fused."

A test whose *failure is the bug* cannot be written: there is no wrong output,
and a timing assertion is flaky and not an acceptable gate. The only "fix" —
fusing the threshold check into the L-extract loop — is a behavior-preserving
refactor with no RED state, and its only available oracle would be the current
implementation's output, which the hard rule forbids producing in the same
session (CLAUDE.md: no impl + oracle together). It also touches the hot path of
every dense-factor exit (four production call sites) with no failing test to
guard a regression, against the "correctness before performance, always"
constraint.

### Disposition

No code change this iteration. The fusion is safe in principle — the growth
flag is a monotonic OR over `|L_ij| > threshold`, so evaluating it as each L
entry is written during extraction yields the identical flag while removing the
separate pass. Deferred to a dedicated dense-factor performance pass that can
benchmark before/after and re-verify the flag against the existing
characterization tests. This entry is the flag to revisit then.

Evidence: `src/dense/factor.rs:409-419` (impl), call sites `:1160, 1772, 2267,
2501`, existing tests `:5061-5112`. Journal: dev/journal/2026-06-10-01.org.

## 2026-06-10 — D13: `update_2x2_block32` det==0 no-op leaves raw A in L (finding D13, repo-review-2026-06-09.md)

Finding D13 observes that `update_2x2_block32` (`src/dense/block_ldlt32.rs:247-279`)
returns early when `det.abs() == 0.0` (`:252-254`), leaving the trailing L
columns `(p+2)..n` holding their raw A values instead of normalized factors;
"unreachable through gated frontal paths, only via the D5 legacy path."
Severity: low/certain.

### Why this is recorded here rather than fixed

Three independent reasons, all pointing to "not a reproducible, safely-fixable
bug this iteration":

1. **The det==0 no-op is an intentional, characterized contract — not a bug.**
   An existing test, `update_2x2_block32_singular_is_noop`
   (`block_ldlt32.rs:544-555`), deliberately drives `(d11,d21,d22)=(1,1,1)`
   (det=0) through *both* `do_2x2_update` and `update_2x2_block32` and asserts
   they are byte-identical no-ops ("Singular 2×2 (det == 0) is a no-op in both
   paths."). The leftover raw A values the finding describes are the *defined*
   consequence of that no-op: there is no valid 2×2 inverse to normalize by, so
   the columns are intentionally left untouched. Adding a `debug_assert!` or a
   zero-fill — the only "fixes" — would directly contradict and break this
   tested contract. Loosening/replacing an existing test to chase the finding is
   not permitted without human approval.

2. **The impact is gated entirely behind the legacy D5 path.** The production
   frontal path applies a determinant floor + issue-#46 partner fallback before
   any 2×2 update, so det==0 never reaches `update_2x2_block32` there. The only
   route that delivers a singular 2×2 to this kernel is the legacy scalar
   `factor()` path — the same path as finding D5 (exactly-singular 2×2 → 1/0 →
   NaN), already recorded here and flagged for removal (D1, D5).

3. **No admissible reproducing test.** A test whose *failure is the bug* would
   have to drive the legacy `factor()` API to a state where raw-A-in-L corrupts
   a committed result — i.e., demonstrate a *wrong inertia / wrong solve* on the
   legacy path. That is exactly D5's deferred scenario and needs a MUMPS/SSIDS
   reference oracle to validate, which this iteration cannot produce (CLAUDE.md:
   inertia must be exactly correct; no impl + oracle in one session). A unit
   test that merely re-confirms the no-op is not a failing-on-the-bug test —
   `update_2x2_block32_singular_is_noop` already pins that behavior.

### Disposition

No code change and no new test. D13 is a latent symptom of the legacy scalar
`factor()` path (the det==0 no-op is correct for the production block path,
which never reaches it). It is subsumed by the D1/D5 decision to remove or
oracle-validate the legacy path; revisit D13 when that happens — at which point
the question becomes whether the legacy path should reject/delay a singular 2×2
upstream (as the frontal path does) rather than what the block kernel does on a
precondition it should never be handed.

Evidence: `src/dense/block_ldlt32.rs:247-279` (impl + det==0 early return),
existing characterization test `:544-555`, production caller chain via
`do_2x2_update` (`factor.rs:4220-4222`) which only forwards n==32. Related: D5
(legacy factor() singular 2×2), D1 (legacy path removal). Journal:
dev/journal/2026-06-10-01.org.

## 2026-06-10 — N8: empty-supernode early return does not drain child contribs (finding N8, repo-review-2026-06-09.md)

**Finding (verbatim).** "Empty-supernode early return (`factorize.rs:2138-2175`,
twin at `:2506-2543`) does not drain `contrib_blocks[child]`; if the symbolic
layer ever emits an empty supernode with contributing children, their Schur
data and delayed-pivot inertia are silently dropped. Unreachable today; add a
drain or `debug_assert!(children have no contribs)`. low/possible."

### Why this is recorded here rather than fixed

1. **The triggering state is unreachable through any real input.** The early
   return at `factor_one_supernode` (`factorize.rs:2138`) fires only when
   `nrow == 0 || own_ncol == 0`. The symbolic layer never emits such a
   supernode: every supernode carries ≥1 eliminated column (`ncol ≥ 1`) and a
   frontal with `nrow ≥ ncol ≥ 1` (`Supernode`/`find_supernodes`,
   `symbolic/supernode.rs`). So the branch is dead defensive code; no matrix the
   solver can be handed drives it. The finding itself classifies this
   "Unreachable today" / severity low.

2. **The twin site is already guarded.** `factor_one_small_leaf`
   (`factorize.rs:2496`) carries `debug_assert!(snode.children.is_empty(), …)`
   at `:2497-2501` *before* its identical early return at `:2506-2543`. A leaf
   has no children, hence no `contrib_blocks[child]` to drain — the twin's N8
   concern cannot arise. N8 reduces to the single `factor_one_supernode` site.

3. **No admissible reproducing test.** A test "whose failure is the bug" would
   have to exhibit dropped Schur data / wrong delayed-pivot inertia from a
   committed factorization — but no input reaches the branch. The only way to
   enter it is to hand-construct a `SymbolicFactorization` (20+ fields, incl.
   `EliminationTree`, `CscPattern`, `col_counts`, postordered `supernodes`)
   holding an empty supernode whose `children` reference a slot with a
   `Some(ContribBlock { n_delayed > 0, … })`, plus a matching `FactorWorkspace`,
   then call the private `factor_one_supernode` directly. That fabricates an
   internal state the symbolic invariant forbids; a `debug_assert!` firing on it
   tests the guard, not a real defect, and the construction would be the impl's
   own oracle (forbidden by CLAUDE.md: no impl + oracle in one session). The
   numeric correctness this protects (inertia exactness) has no external
   reference here because there is no admissible input to validate against.

### Disposition

No code change and no new test this iteration. N8 is dead-branch defensive
hardening, not a live defect — the recommended `debug_assert!(children have no
contribs)` guards a state the symbolic layer guarantees never occurs, and the
twin path is already guarded. Revisit if/when the symbolic layer is ever changed
to emit zero-column supernodes (e.g. a future Schur/elimination-skip feature):
at that point the drain (not merely the assert) becomes a real numeric
requirement and a reproducing test would have a real input to exercise it.

Evidence: `src/numeric/factorize.rs:2138-2175` (live site, no drain),
`:2496-2543` (twin, already guarded by `debug_assert!(snode.children.is_empty())`
at :2497-2501), `Supernode`/`ncol()` (`symbolic/supernode.rs:126-162`,
`nrow ≥ ncol ≥ 1`). Journal: dev/journal/2026-06-10-01.org.

## 2026-06-10 — N9: per-supernode phase deltas read process-global atomics (finding N9, repo-review-2026-06-09.md)

**Finding (verbatim).** "Sequential profiler per-supernode phase deltas read
process-global atomics (`factorize.rs:2020-2047`); two solvers factoring
concurrently in one process cross-contaminate the deltas. Diagnostics only.
low/certain."

### Why this is recorded here rather than fixed

1. **The contamination is real but confined to a default-off diagnostic flag.**
   The per-supernode `SupernodeTiming` deltas (`factorize.rs:2021`/`:2036`)
   snapshot `phase_timing::snapshot()` before/after one `factor_one_supernode`
   call. Those counters (`ASSEMBLY_NS`, `DENSEFACTOR_NS`, … in
   `dense/factor.rs:188`) are process-global `AtomicU64`s, written only when
   `PHASE_TIMING_ENABLED` (default `false`, `dense/factor.rs:178`) is set — a
   diagnostic binary's mode, never production solve/inertia. Two *separate*
   solvers factoring concurrently in one process would interleave increments,
   inflating each other's deltas.

2. **The probe site is single-threaded by design and already documents it.**
   The comment at `factorize.rs:2017-2020` states: "this driver loop is
   single-threaded so before/after differencing is exact." The contamination
   needs a *second* solver on another thread, which the probe's own mode does
   not create.

3. **The process-global design is load-bearing — a thread-local "fix" breaks
   the parallel path.** The same counters are written from inside
   `factor_one_supernode`'s assembly/dense helpers
   (`factorize.rs:2217-2406`), and `factor_one_supernode` is spawned on
   *rayon worker threads* by the parallel driver (`scope.spawn`,
   `factorize.rs:1217`, `:1943`). The diagnostic binary reads totals via
   `snapshot()` from the main thread after the run. Making the counters
   thread-local would isolate concurrent solvers but silently drop every
   counter increment performed on rayon workers — turning a rare diagnostic
   inaccuracy into a systematic undercount on the parallel path. The only
   correctness-preserving fix is to thread a per-solver counter set through
   the entire dense-factor call chain (assembly → panel → Schur → scalar
   tail), an invasive refactor of diagnostics-only plumbing.

4. **No admissible reproduce-first fix this iteration.** A deterministic
   barrier-synchronized two-thread test could demonstrate the shared-global
   contamination, but the only fix it would gate (per-solver counters) is the
   invasive refactor in (3) — impl plus its own timing oracle in one session,
   with no external reference, on a default-off diagnostic path. Severity is
   low/certain and "diagnostics only" by the finding's own classification.

### Disposition

No code change and no new test. N9 is an intentional consequence of the
process-global phase-probe design: the counters are deliberately global so the
diagnostic binary can read whole-run totals across rayon workers. The
cross-solver contamination only arises when two solvers factor concurrently in
one process with the default-off `PHASE_TIMING_ENABLED` flag set, and never
affects numeric results. Revisit only if a per-solver diagnostic profile is
ever required, at which point per-solver counters threaded through the dense
factor chain (preserving parallel-worker aggregation) become the real task.

Evidence: `src/numeric/factorize.rs:2017-2050` (sequential per-supernode
delta + single-threaded comment), `:2217-2406` (counter writes inside
`factor_one_supernode`), `:1217`/`:1943` (rayon `scope.spawn` of
`factor_one_supernode` on workers), `src/dense/factor.rs:178-250`
(`PHASE_TIMING_ENABLED` default false; process-global counters + `snapshot()`).
Journal: dev/journal/2026-06-10-01.org.

## 2026-06-10 — N10: doc/code drift bundle (finding N10, repo-review-2026-06-09.md)

**Finding (verbatim).** "Doc/code drift: `condition.rs:7,27` "3-5 solves"
(actually up to ~11); `solve.rs:389-391` claims an nrhs==1 thin-wrapper dispatch
that doesn't exist (bit-identical anyway); `BucketStats.pct_of_total`
(`factorize.rs:441-447`) is percent of loop_us, not total_us; the Schur inner
driver (`factorize.rs:1645`) uses `compute_scaling`, not
`compute_scaling_with_cache` — MC64 cache reuse silently doesn't apply on the
Schur path. low/certain."

### Why this is recorded here rather than fixed

N10 bundles four drifts. In every one the *runtime behavior is already correct*;
the drift is stale prose, a misleading field name, or a latent (not live)
consistency gap. None admits a test "whose failure is the bug," so per the loop
discipline (reproduce-first, else defer) all four are recorded here with the
code evidence that confirms each.

1. **`condition.rs:7,27` "3-5 solves" → actually up to 11 (doc only).** The
   estimator loops `for _iter in 0..MAX_ITER` (`condition.rs:88`) with
   `MAX_ITER = 5` (`:27`) and performs `2·MAX_ITER + 1 = 11` internal solves
   (two per iteration at `:90`/`:106` plus the final at `:154`); the code's own
   comment at `:68-69` and `:214` already says "up to 2·MAX_ITER + 1 … ~11×".
   The module-doc "3-5 solves" (`:7`, `:166`) is stale prose. The code is
   correct; there is no failing-on-the-bug test (a constant-arithmetic assert
   `2*MAX_ITER+1 == 11` is trivially green, not a RED, and counting solves would
   require instrumenting production for a comment fix).

2. **`solve.rs:389-391` phantom nrhs==1 dispatch (doc only).** The comment
   describes an `nrhs == 1` thin-wrapper dispatch that does not exist; the
   finding itself notes the result is "bit-identical anyway." Behavior is
   correct — only the comment describes a code path that was never written.
   A test asserting nrhs==1 equals the many-RHS path passes today (it is
   bit-identical), so it is GREEN, not a reproduction of any defect.

3. **`BucketStats.pct_of_total` is % of loop_us (correct; name misleads).**
   `factorize.rs:443` computes `b.pct_of_total = sum_us·100/loop_us`. This is
   the *correct* denominator: the front-size buckets partition the supernode
   loop, so percentages sum to 100% of `loop_us`; using `total_us` would
   exclude prologue/epilogue and the buckets would not sum to 100%. The field
   name (`pct_of_total`, `:476`) is the only thing that misleads. A RED test
   would have to assert "% of total_us" — i.e. assert *wrong* behavior — so
   there is no honest reproduction; the runtime value is right.

4. **`factorize.rs:1645` Schur driver uses `compute_scaling` — latent, not
   live.** The main drivers (`:1839`, `:2836`) call
   `compute_scaling_with_cache(matrix, &params.scaling,
   symbolic.cached_mc64.as_ref())`; the Schur inner driver calls plain
   `compute_scaling`. But `symbolic_factorize_with_schur` sets `cached_mc64:
   None` (`symbolic/mod.rs:1186`) — the Schur path *never populates* an MC64
   cache. So `compute_scaling_with_cache(.., None)` would be byte-identical to
   `compute_scaling(..)` today: there is no cache to reuse and no output change
   to observe or test. The drift is latent — it would only bite if the Schur
   symbolic path were later changed to populate `cached_mc64`, at which point
   the numeric Schur driver would silently ignore it. Aligning the call is safe
   future-proofing, but it is behavior-preserving with no RED gate today; and
   changing it to reuse a cache (were one ever present) trades a recomputed
   numeric-time matching for a symbolic-time one — a semantic choice that would
   need a MUMPS/SSIDS oracle, not a same-session self-comparison.

### Disposition

No code change and no new test. All four sub-items are doc/naming/latent-
consistency drift over already-correct runtime behavior; none is reproducible as
a failing test. Revisit as a dedicated documentation-accuracy pass (correct the
`condition.rs` "3-5 solves" prose, drop the `solve.rs` phantom-dispatch comment,
document `pct_of_total` as "percent of `loop_us`", and align the Schur
`compute_scaling_with_cache` call for consistency) if/when a docs sweep is
scheduled — explicitly outside the reproduce-first loop.

Evidence: `src/numeric/condition.rs:7,27,68-69,88,90,106,154,166,214`;
`src/numeric/solve.rs:389-391`; `src/numeric/factorize.rs:441-448,476,1645,
1839,2836`; `src/symbolic/mod.rs:1041,1186` (Schur `cached_mc64: None`). Journal:
dev/journal/2026-06-10-01.org.

## 2026-06-10 — N11: solve-time null-pivot semantics (finding N11, repo-review-2026-06-09.md)

**Finding (verbatim).** "Solve-time null-pivot semantics: a force-accepted zero
pivot is skipped, leaving the forward-substituted RHS value in `w[k]`
(`solve.rs:266-271`, `:770-775`) rather than zeroing the null-space component
(MUMPS ICNTL(25)=0 convention). Documented as deliberate
(`dev/plans/threshold-mismatch-fix.md`); flagged for awareness — the unrefined
`Solver::solve()` exposes it directly. low/certain (behavior) / possible (that
it's wrong)."

### Why this is recorded here rather than fixed

1. **It is documented, deliberate behavior — not a defect.** The D-block solve
   leaves `w[k]` untouched when the pivot was force-accepted as zero
   (`solve.rs:300-303` single-RHS: `if d_diag[k].abs() > zero_tol { w[k] /=
   d_diag[k]; } // else: leave w[k] alone`; twin at `:818-821` multi-RHS). The
   comment at `:271-272` and the design doc `dev/plans/threshold-mismatch-fix.md`
   (`:71`, `:85`, `:94`) record this as the intended outcome of the
   "store `zero_tol` and skip in solve" fix. The finding itself classifies the
   *wrongness* as only "possible," and flags it "for awareness," not for repair.

2. **No admissible RED.** A reproducing test would have to assert the *alternative*
   convention — zero the null-space component (MUMPS ICNTL(25)=0) — and show the
   current output is wrong. But which convention is correct on a singular system
   is exactly the open question, and `Solver::solve()` on a force-accepted-zero
   (genuinely singular) system has no unique right answer without choosing a
   convention. Validating the alternative requires a MUMPS/SSIDS reference oracle
   for the specific singular case, which this iteration cannot produce. Writing a
   test that pins the *current* behavior would make the implementation its own
   oracle and would not be failing-on-the-bug.

3. **Changing it needs human approval.** This alters documented, deliberate solve
   semantics on singular systems and would touch both the single- and multi-RHS
   D-block solves. The project's correctness rule requires inertia/solve
   behavior on such matrices to be validated against the canonical Fortran
   solvers; flipping the convention without that oracle and without sign-off is
   exactly the kind of change the hard rules forbid doing unilaterally.

### Disposition

No code change and no new test. N11 is a deliberate, documented design choice
(skip force-accepted-zero pivots, leaving the forward-substituted value in
`w[k]`), flagged for awareness. Revisit only as a human-approved decision to
adopt the MUMPS ICNTL(25)=0 null-space convention, gated on a MUMPS/SSIDS
reference oracle for the singular-system output — at which point both the
single-RHS (`solve.rs:300-303`) and multi-RHS (`:818-821`) D-block solves, plus
the design doc, would be updated together.

Evidence: `src/numeric/solve.rs:268-303` (single-RHS D-block, force-accepted-zero
skip at :300-303), `:815-821` (multi-RHS twin),
`dev/plans/threshold-mismatch-fix.md:71,85,94` (documented deliberate behavior).
Journal: dev/journal/2026-06-10-01.org.

## 2026-06-10 — S2: placeholder Supernode.nrow undercounts amalgamated frontals (finding S2, repo-review-2026-06-09.md)

**Finding (verbatim).** "Placeholder `Supernode.nrow =
col_counts[first_col].max(ncol)` (`symbolic/supernode.rs:399-405`) undercounts
amalgamated frontals (size-based merges need not nest). Downstream bias:
`contrib_sizes`/`peak_contrib_bytes` (`mod.rs:963-965`) underestimate pool
needs; `factor_nnz_estimate` (`mod.rs:935`) excludes amalgamation-induced zeros
and AutoRace (`mod.rs:640`) ranks on the biased metric; `find_small_leaf_groups`
(`small_leaf.rs:138-140`) gates on placeholder nrow ≤ 16 while sizing the arena
on actual rows.len(). Numeric correctness unaffected (`build_row_indices`
recomputes). medium/likely."

### Why this is recorded here rather than fixed

1. **No incorrect output — confirmed by the finding and by the code.** The
   placeholder `nrow = col_counts[first_col].max(ncol)` (`supernode.rs:399`) is
   never reassigned in the symbolic phase; the *true* frontal row set is
   recomputed at numeric time by `build_row_indices` (`factorize.rs:3656`),
   which is what factorization/solve/inertia actually use. So no numeric or
   inertia result is wrong — the finding states this explicitly ("Numeric
   correctness unaffected") and it holds. With no incorrect output, there is no
   test "whose failure is the bug": a pool-size or nrow-estimate assertion would
   either be flaky or a characterization test that pins internal state, making
   the implementation its own oracle (forbidden). This is the same disposition
   class as D11/D12 (perf/estimate-only, no RED gate).

2. **The blast radius is perf-estimate only — and narrower than the finding
   states.** The placeholder `nrow` feeds exactly two consumers via
   `contrib_size() = (nrow-ncol)²` (`supernode.rs:172-175`):
   - `contrib_sizes` / `peak_contrib_bytes` (`mod.rs:963-965`) — the
     contribution-pool pre-allocation estimate. An undercount means the pool may
     grow/reallocate at numeric time: a one-time perf cost, not a wrong answer.
   - `find_small_leaf_groups` (`small_leaf.rs:138-140`) — gates the batched
     small-leaf path on `nrow ≤ 16`. Both batched and per-supernode paths are
     numerically equivalent (cf. finding S3), so this only shifts which path
     runs: again perf, not correctness.

   The finding's claim that "AutoRace (`mod.rs:640`) ranks on the biased metric"
   is **inaccurate for the placeholder nrow**: AutoRace ranks candidates by
   `factor_nnz_estimate` (`mod.rs:640`), which is computed from `col_counts`
   (`mod.rs:935`, `total_factor_nnz(&col_counts)`) — independent of
   `Supernode.nrow`. Fixing the placeholder nrow does **not** change AutoRace's
   ranking, the selected ordering, or therefore any inertia result. (The
   separate `factor_nnz_estimate` "excludes amalgamation-induced zeros" bias the
   finding also mentions lives in `col_counts`, not in the nrow placeholder, and
   is a different issue.)

3. **The proper fix is a feature-sized symbolic pass, benchmark-gated.** Setting
   the true amalgamated `nrow` requires a new symbolic row-union pass over
   `permuted_pattern` after `find_supernodes` (the full pattern is not available
   inside `find_supernodes`, which receives only `etree` + `col_counts`,
   `mod.rs:939`) — essentially the symbolic analogue of `build_row_indices`.
   That shifts the benchmark-sensitive `small_leaf` gate and the pool-size
   estimate, so it must be validated against the full benchmark per the
   session protocol. It is a scoped feature, not a surgical reproduce-first bug
   fix, and it changes no output.

### Disposition

No code change and no new test. S2 produces no incorrect output (numeric and
inertia results are unaffected — `build_row_indices` recomputes the true rows);
its only consequences are an internal pool-size underestimate and a
small_leaf-gate shift, both perf-only and untestable as a failing-on-the-bug
test. Recommended as a dedicated, benchmarked work item: add a symbolic
row-union pass after `find_supernodes` to set the true amalgamated `nrow`, then
validate the small_leaf-gate / pool-size shift against the bench. Inertia gate
is not at risk (AutoRace ranks on `factor_nnz_estimate`, not nrow).

Evidence: `src/symbolic/supernode.rs:399-405` (placeholder), `:166-175`
(`contrib_size` uses nrow), `src/symbolic/mod.rs:935` (`factor_nnz_estimate`
from col_counts), `:939` (find_supernodes args), `:963-965`
(contrib_sizes/peak), `:640` (AutoRace ranks on factor_nnz_estimate, not nrow),
`src/symbolic/small_leaf.rs:138-140` (gate on nrow≤16),
`src/numeric/factorize.rs:3656` (build_row_indices recomputes true rows).
Journal: dev/journal/2026-06-10-01.org.

---

## 2026-06-10 — S3: compute_leaf_rows lacks defensive `r < own_last` filter (finding S3, repo-review-2026-06-09.md)

### Finding (verbatim)

> `compute_leaf_rows` (small_leaf.rs:208-217) lacks the defensive `r < own_last`
> filter its numeric twin (`build_row_indices`, factorize.rs:3632-3645) applies;
> if the leaf invariant ever cracks, the batched small-leaf path diverges
> silently from the per-supernode path. One-line guard. medium-low/possible.

### Why this is recorded here rather than fixed

1. **No RED for any valid input.** `compute_leaf_rows` (small_leaf.rs:193-229)
   scans the permuted pattern of a leaf supernode's own columns and collects
   every row index it sees. `build_row_indices` (factorize.rs:3656) does the same
   but skips rows `r < own_last` (`own_last = first_col + own_ncol`,
   factorize.rs:3716) — those are entries above the supernode's own column block,
   i.e. couplings to *earlier-eliminated* columns `r < first_col`.

2. **The elimination-tree property makes the filter a no-op for true leaves.** A
   leaf supernode has no descendants in the etree. A column `j` of the leaf can
   only couple to an earlier-eliminated column `r < first_col` if some descendant
   of `j` was eliminated first — but a leaf has none. Therefore for every *valid*
   leaf, no scanned `r` satisfies `r < first_col`, the `r < own_last` filter
   removes nothing, and `compute_leaf_rows` and `build_row_indices` produce
   identical row sets. The divergence the finding describes requires the leaf
   invariant to be violated upstream — a pipeline-forbidden state.

3. **A test would have to fabricate an invariant-violating leaf.** To make the two
   paths diverge you must hand a "leaf" with a sub-diagonal coupling to an
   earlier column — a structure `find_supernodes` never emits. Asserting on that
   fabricated input tests the guard against a state the pipeline cannot produce;
   it does not reproduce a real defect. This is exactly N8's category (a defensive
   guard for an unreachable state) and the same impl-as-own-oracle problem as
   D11/D12.

### Disposition

No code change and no new test. For all valid leaves the missing filter changes
nothing (proved above via the etree leaf property). Recommended as harmless
future hardening: add the one-line `if r < own_last { continue; }` guard to
`compute_leaf_rows` so the batched small-leaf path is textually identical to
`build_row_indices` and stays robust if an upstream invariant ever cracks — but
this is defensive alignment, not a bug fix, and carries no failing test.

Evidence: `src/symbolic/small_leaf.rs:193-229` (`compute_leaf_rows`, no filter),
`:138-140` (small_leaf gate), `src/numeric/factorize.rs:3656` (`build_row_indices`),
`:3716` (`if r < own_last { continue; }`). Etree leaf property: a leaf has no
descendants ⇒ no own column couples to `r < first_col` ⇒ filter is a no-op.
Journal: dev/journal/2026-06-10-01.org.

---

## 2026-06-10 — S4: run_amd skips the perm-length / bijectivity check (finding S4, repo-review-2026-06-09.md)

### Finding (verbatim)

> `schur.rs::run_amd` (`ordering/schur.rs:186-214`) skips the perm-length check
> `run_external_ordering` (`symbolic/mod.rs:591-597`) performs; only a
> debug_assert stands before `permute_pattern` corruption in release. Neither
> path checks bijectivity. medium-low/certain (drift) / possible (impact).

### Why this is recorded here rather than fixed

1. **The drift is real but the missing check is a no-op for valid input.**
   `run_external_ordering` (mod.rs:591-597) rejects a backend permutation whose
   length ≠ `pattern.n`. `run_amd` (schur.rs:186-214) omits that length check; it
   keeps only the per-element range check `u >= pattern.n` (schur.rs:206). The two
   paths therefore diverge *only* when the ordering backend returns a permutation
   of the wrong length or with duplicate entries.

2. **The backend cannot be forced to misbehave from any reachable input.**
   `feral_amd::amd_order` always returns a full-length permutation of `0..n` for a
   valid CSC pattern (every vertex, connected or not, is eliminated exactly once).
   To make `run_amd` emit a wrong-length or non-bijective perm I would have to mock
   or corrupt `amd_order`'s output, which no public/private call path does.

3. **The downstream consumer is already protected against the range failure.**
   `compute_schur_aware_perm` lifts the sub-perm via `non_schur_indices[sub_idx]`
   (schur.rs:110); `sub_idx < n_f` is guaranteed by run_amd's `u >= pattern.n`
   check, so there is no index-out-of-bounds panic. The remaining gap is a
   wrong-*length* sub_perm, caught only by `debug_assert_eq!(perm.len(), n)`
   (schur.rs:116) — absent in release — and a duplicate-entry sub_perm
   (bijectivity), which *neither* run_amd nor run_external_ordering checks. Both
   gaps require a malfunctioning backend.

4. **A test would assert the guard against an unproducible state.** Same class as
   S3/N8: a unit test would have to fabricate a malfunctioning AMD (impossible
   without mocking the crate) or feed a hand-built invalid perm to an internal
   helper — testing the guard against a pipeline-forbidden state
   (impl-as-own-oracle, forbidden).

### Disposition

No code change and no new test. For every valid input `run_amd` and
`run_external_ordering` behave identically. Recommended as harmless future
hardening (a drift fix, not a bug fix, carrying no failing test):
add `if perm_i32.len() != pattern.n { return Err(InvalidInput(...)); }` to
`run_amd` to mirror mod.rs:591-597, and — to close the bijectivity gap the
finding correctly notes is shared — add a duplicate-index check to *both*
paths (a `seen` bitset over `0..n`).

Evidence: `src/ordering/schur.rs:186-214` (run_amd, no length check), `:206`
(per-element range check), `:104,110,116` (consumer lift + debug_assert),
`src/symbolic/mod.rs:591-597` (the length check run_amd lacks). Bijectivity
unchecked in both. Journal: dev/journal/2026-06-10-01.org.

---

## 2026-06-10 — S5: reference column_counts is O(n³)-class but documented O(n²) (finding S5, repo-review-2026-06-09.md)

### Finding (verbatim)

> Reference `column_counts` (`symbolic/column_counts.rs:56-65`) is O(n³)-class
> (`contains` + re-sort per eliminated column) but documented "O(n²) elimination
> simulation" (`mod.rs:855-857`) and publicly re-exported. Production uses GNP
> everywhere; burdens tests. low/certain.

### Why this is recorded here rather than fixed

1. **No incorrect output.** `column_counts` is the bit-exact reference oracle for
   the production Gilbert-Ng-Peyton path; equivalence is verified on 169585 KKT
   matrices (mod.rs:857, `dev/validation/phase-2.5.1-*`). The finding is about its
   *cost* and a *false complexity claim*, not its result.

2. **The complexity claim is genuinely wrong.** The propagation step
   (column_counts.rs:55-65) does a linear `col_rows[min_row].contains(&row)` per
   propagated row plus a `sort_unstable` + `dedup` of the whole inherited list per
   eliminated column. On dense-ish patterns that is O(n³)-class, not the
   "O(n²) elimination simulation" mod.rs:855 advertises.

3. **Neither consequence is reproducible as a failing test.** A complexity claim
   in a doc comment cannot be made RED — asserting asymptotic class needs timing
   (flaky) or a counter the impl doesn't expose (impl-as-own-oracle). "Burdens
   tests" is wall-clock, not a correctness assertion. This is exactly N10's
   doc-drift category and S2's perf-only category.

### Disposition

No code change and no new test. Recommended (a doc-truth + hygiene fix, not a bug
fix, carrying no failing test):
(a) correct mod.rs:855 to call the reference "O(n³)-class elimination simulation"
    (or similar) so it no longer claims O(n²);
(b) if the public re-export is unnecessary, demote `column_counts` to
    `pub(crate)` so it is a test-only oracle and stops appearing in the public
    surface — production already uses `column_counts_gnp` everywhere (mod.rs:860).
Both are non-functional and out of scope for a reproduce-first loop fix.

Evidence: `src/symbolic/column_counts.rs:55-65` (contains + sort/dedup per
column), `:20` (`pub fn`), `src/symbolic/mod.rs:855-857` (the "O(n²)" claim),
`:860` (production uses GNP). No incorrect output. Journal:
dev/journal/2026-06-10-01.org.

---

## 2026-06-10 — S6: unchecked documented invariants (validate/subtree_sizes/schur postorder) (finding S6, repo-review-2026-06-09.md)

### Finding (verbatim)

> Unchecked documented invariants: no `CscPattern::validate()` (etree wants
> upper-triangle entries, GNP wants lower — both silently require full-symmetric;
> a lower-only pattern yields an edgeless forest with counts of 1 and no error);
> `subtree_sizes` assumes `parent[j] > j` (comment only, fields pub);
> `schur_constrained_postorder` correctness leans on parent>child within the
> Schur set, guarded only by a debug_assert tail check (`symbolic/mod.rs:1092-1098`).
> low/certain.

### Why this is recorded here rather than fixed

All three sub-items are defensive gaps against states the production pipeline
never produces; none yields incorrect output under correct usage.

1. **Lower-only pattern is unreachable in production.** The solver always builds
   the etree on a *symmetrized* pattern: mod.rs:703 calls
   `matrix.symmetric_pattern()`, and the etree (mod.rs:795
   `EliminationTree::from_pattern`) / GNP (mod.rs:860) only ever see that full
   pattern. The "edgeless forest, counts of 1, no error" behavior requires calling
   `from_pattern`/`column_counts` directly on a lower-only pattern — a violation
   of their *documented* precondition ("Input `pattern` should be the full
   symmetric pattern", column_counts.rs:16), not a defect in correct operation.

2. **`parent[j] > j` is a structural guarantee, not just a comment.** The real
   elimination-tree builder always emits `parent[j] > j` (an elimination tree
   orders every node before its parent). `subtree_sizes`
   (elimination_tree.rs:82-85) processing `0..n` in order is therefore correct for
   every etree the pipeline builds; the invariant only "cracks" for a hand-built
   etree the pipeline never emits.

3. **`schur_constrained_postorder` parent>child is likewise guaranteed**, and is
   additionally backed by the debug_assert tail check (mod.rs:1092-1098). Same
   unreachable-state category.

4. **A RED test would assert behavior on a precondition-violating input** — a
   lower-only pattern, or a fabricated etree with `parent[j] <= j` — i.e. it would
   test garbage-in handling, not reproduce a defect under correct usage. Same
   class as S3/S4/N8.

### Disposition

No code change and no new test. Recommended as harmless future hardening (a
robustness improvement, not a bug fix, carrying no failing test): add a
`CscPattern::is_symmetric()` (or `validate()`) debug-only assertion at the etree
/ GNP entry points, and promote the `parent[j] > j` comment in
`subtree_sizes` to a `debug_assert!`. These make the documented preconditions
self-checking in debug builds without changing release behavior or output.

Evidence: `src/symbolic/mod.rs:703` (symmetric_pattern), `:795`/`:860` (etree/GNP
see only the full pattern), `:1092-1098` (schur postorder debug_assert);
`src/symbolic/column_counts.rs:16` (documented full-symmetric precondition);
`src/ordering/elimination_tree.rs:82-85` (subtree_sizes parent>j comment). No
incorrect output for correct usage. Journal: dev/journal/2026-06-10-01.org.

---

## 2026-06-10 — S8: allocation-in-loop cluster (finding S8, repo-review-2026-06-09.md)

### Finding (verbatim)

> Allocation-in-loop cluster: per-column Vec in `sort_and_sum_duplicates`
> (`csc.rs:121`); provably redundant final per-column sort in `symmetric_pattern`
> (`csc.rs:242-246`); `children()` builds Vec<Vec> per postorder variant and in
> GNP which only needs a leaf flag (`elimination_tree.rs:66-74`);
> `compress_pattern` builds ~n two-element Vecs (`ldlt_compress.rs:161-166`);
> HashSet for a contiguous range test (`mod.rs:1331`). low/certain.

### Why this is recorded here rather than fixed

1. **Every item is perf-only; none produces incorrect output.** Verified the two
   anchor sites: `sort_and_sum_duplicates` allocates a fresh `pairs: Vec<(usize,
   f64)>` per column (csc.rs:160) but the summed/sorted matrix it produces is
   correct; `symmetric_pattern`'s per-column `sort_unstable` (csc.rs:301) is
   redundant only because the rows are *already* sorted, so removing it changes no
   output. The `children()` Vec<Vec>, the `compress_pattern` two-element Vecs, and
   the HashSet-for-range-test are likewise allocation-strategy choices with
   correct results.

2. **No reproducing test is possible.** A perf/allocation pattern has no failing
   output to assert. Pinning allocation counts would test the impl against itself
   (impl-as-own-oracle, forbidden); asserting wall-clock is flaky. This is exactly
   S2's perf-only category and the same reason N7's perf fix had to be reproduced
   via *observable cache state* — these S8 items expose no analogous observable
   state.

### Disposition

No code change and no new test. Recommended as benchmarked perf work items (not
reproduce-first loop fixes):
- reuse a single scratch `Vec` across columns in `sort_and_sum_duplicates`
  (csc.rs:142-160);
- delete the provably-redundant per-column sort in `symmetric_pattern`
  (csc.rs:301) — the merge preserves the already-sorted order;
- replace `children()`'s `Vec<Vec>` with a leaf-flag bitset where GNP/postorder
  only need leaf detection (elimination_tree.rs:66-74);
- preallocate / inline the `compress_pattern` two-element Vecs
  (ldlt_compress.rs:161-166);
- replace the HashSet contiguous-range test with a bounds comparison (mod.rs:1331).
Each should be validated against the corpus bench since they touch hot symbolic
paths.

Evidence: `src/sparse/csc.rs:142-160` (per-column pairs Vec), `:301` (redundant
sort), `src/ordering/elimination_tree.rs:66-74` (children Vec<Vec>),
`src/symbolic/ldlt_compress.rs:161-166` (two-element Vecs),
`src/symbolic/mod.rs:1331` (HashSet range test). All perf-only, correct output.
Journal: dev/journal/2026-06-10-01.org.

---

## 2026-06-10 — S9: symv validates nothing (finding S9, repo-review-2026-06-09.md)

### Finding (verbatim)

> `symv` validates nothing (`csc.rs:258`): panics on short x/y, silently
> zero-fills only the first n of an oversized y — inconsistent with the
> scrupulous from_triplets/validate on the same type. low/certain.

### Why this is recorded here rather than fixed

1. **For correct-length inputs symv is already correct.** `CscMatrix::symv`
   (csc.rs:314) zeros `y[..n]` and computes `y = A·x` over `0..n`. When
   `x.len() == y.len() == n` — the only contract the doc "y = A * x" implies —
   the result is correct. The two failure modes the finding cites are both
   precondition violations: a too-short `x`/`y` panics on slice indexing, and an
   oversized `y` leaves `y[n..]` at stale values (only `y[..n]` is written).

2. **Production never violates the precondition.** Every internal caller passes
   exactly-length-`n` buffers: `solve.rs:1092/1160/1374`, `dense/solve.rs:88/124`,
   `scaling/mod.rs:1493`, and the multi-RHS sites slice precisely
   (`solve.rs:1274` `&x[c*n..(c+1)*n]`, `&mut r_act[k*n..(k+1)*n]`). The
   panic/stale-tail paths are unreachable from the crate's own usage; only an
   external misuse of the `pub` method could hit them. Same class as S6 (unchecked
   documented invariants).

3. **No non-breaking fix is reproducible.** Making `symv` return
   `Result<(), FeralError>` is a breaking API change unjustified by a "low/certain"
   hygiene item and produces no different output for correct usage. The
   non-breaking alternative — a `debug_assert!(x.len() == self.n && y.len() ==
   self.n)` plus a documented precondition — does not change release behavior, so
   it has no RED test. A `#[should_panic]` test would merely pin the current panic,
   not reproduce a defect in correct operation.

### Disposition

No code change and no new test. Recommended as harmless future hardening
(API-consistency, not a bug fix, carrying no failing test):
document the `x.len() == y.len() == n` precondition on `symv` and add a
`debug_assert!` guard so misuse fails loudly in debug builds, matching the
type's otherwise-scrupulous validation. If a fallible variant is ever wanted,
add a separate `try_symv -> Result` rather than breaking `symv`.

Evidence: `src/sparse/csc.rs:314-328` (symv, no validation, `.take(self.n)`
zeroing); exact-length callers `src/numeric/solve.rs:1092,1160,1274,1304,1318,1374`,
`src/dense/solve.rs:88,124`, `src/scaling/mod.rs:1493`. No incorrect output for
correct usage. Journal: dev/journal/2026-06-10-01.org.

---

## 2026-06-10 — S10: predict_merges vs find_supernodes drift (finding S10, repo-review-2026-06-09.md)

### Finding (verbatim)

> `predict_merges` vs `find_supernodes` drift (`supernode.rs:628-671`): prediction
> ignores the Phase-B4 root cap and uses original parent ncols vs the real pass's
> cumulative ones. Harmless heuristic bias, but the root-cap omission is
> undocumented and the MUONSINE over-merge analysis suggests it's load-bearing.
> low/certain.

### Why this is recorded here rather than fixed

1. **predict_merges is a heuristic that never determines the final structure.**
   Its own doc (supernode.rs:612-627) is explicit: it "does not enforce
   adjacency — the caller uses the merge predictions to drive a merge-biased
   postorder that *makes* the merges adjacent." It returns a `Vec<bool>` bias
   (supernode.rs:634, :664-666); the real, adjacency-checked amalgamation is done
   afterward by `find_supernodes` on the re-postordered etree. The factorization
   is therefore correct for *any* prediction — a drift only shifts which subtrees
   get biased late, i.e. amalgamation quality, not output.

2. **The drift is perf-only and possibly load-bearing.** Using original
   fundamental-supernode ncols (supernode.rs:648-649,658) instead of the real
   pass's cumulative ncols, and omitting the Phase-B4 root cap, change *which*
   merges are predicted — a heuristic bias. The finding itself notes the MUONSINE
   over-merge analysis suggests the root-cap omission is *load-bearing*: aligning
   predict_merges to find_supernodes could re-introduce the MUONSINE over-merge
   regression (5.5× → 1.4× MUMPS, per the AmalgamationStrategy::Auto rationale).
   So this is not a safe surgical change.

3. **Not reproducible as a failing test.** There is no incorrect output to assert.
   Pinning the predicted bias vector tests the impl against itself
   (impl-as-own-oracle, forbidden); a merge-quality/fill assertion is a benchmark,
   not a unit test. Same class as S2/S8 (perf/heuristic, no incorrect output).

### Disposition

No code change and no new test. Recommended (documentation + a benchmarked study,
not a reproduce-first loop fix):
(a) document in `predict_merges` that it deliberately omits the Phase-B4 root cap
    and uses original (not cumulative) ncols, citing the MUONSINE over-merge
    analysis so the divergence is a recorded design choice rather than silent
    drift;
(b) if alignment is ever attempted, gate it behind a full corpus bench with
    MUONSINE in the watch set, since the omission may be load-bearing.

Evidence: `src/symbolic/supernode.rs:612-627` (doc: no adjacency enforced),
`:628-671` (predict_merges, original ncols at :648-649, size rule :658, bias
:664-666); the real merge is `find_supernodes` on the biased postorder. No
incorrect output. Journal: dev/journal/2026-06-10-01.org.

---

## 2026-06-10 — L7: stale column scaling on entering columns (finding L7, repo-review-2026-06-09.md)

### Finding (verbatim)

> Stale column scaling on entering columns (`dense_update.rs:55-57`,
> `sparse_update.rs:174`): entering column scaled by `d_col[leaving_slot]`
> computed for the old column. Algebraically consistent but equilibration quality
> decays arbitrarily over an update chain, inflating bump multipliers (interacts
> with L5). low/certain (mechanism) / possible (severity).

### Why this is recorded here rather than fixed

1. **The output is algebraically correct.** Both update paths scale the entering
   column by the leaving slot's column factor — dense_update.rs:55-57
   (`* self.scale.d_col[leaving_slot]`) and sparse_update.rs:195
   (`let dcol = self.scale.d_col[leaving_slot]`). The factorization maintains
   `P B̃ Q = L U` in the *scaled* frame, and the solve un-scales with the same
   `d_col`, so the replacement is consistent and the computed solution is correct
   to working precision. The finding itself classifies the mechanism as
   "algebraically consistent."

2. **The defect is numerical conditioning, not a wrong answer.** Applying the old
   column's scale to a new column with a different natural magnitude gives poor
   equilibration, which can inflate bump multipliers and the growth factor *over a
   chain of updates*. Severity is "possible": within the growth budget
   (`max_growth`, which triggers `NeedsRefactor`) the solve stays accurate; the
   decay only costs earlier refactors.

3. **Not reproducible as a clean RED test.** Demonstrating "equilibration quality
   decays" is a conditioning study — measuring growth-factor / residual trend
   across a long, matrix-specific update chain against an arbitrary threshold — not
   a deterministic unit assertion. There is no single input that yields a *wrong*
   output to assert against.

4. **The fix is a scoped, L5-coupled algorithmic change.** Correcting it means
   re-equilibrating the entering column with its *own* `d_col` (and reconciling the
   solve's un-scaling), which changes the scaling frame mid-update and, per the
   finding, "interacts with L5." That is research-plus-benchmark work, not a
   surgical reproduce-first fix.

### Disposition

No code change and no new test. Recommended as a tracked numerical study jointly
with L5: (a) instrument the bump-multiplier / growth-factor trend over synthetic
update chains to quantify the decay; (b) design an entering-column
re-equilibration (own `d_col`) with a matching solve un-scale; (c) validate
refactor frequency and accuracy against the corpus bench. Until then the behavior
is correct-but-suboptimally-conditioned, bounded by the `max_growth` refactor
trigger.

Evidence: `src/lu/dense_update.rs:50-58` (spike scaled by d_col[leaving_slot]),
`src/lu/sparse_update.rs:183` (doc), `:195` (`d_col[leaving_slot]`). Output
algebraically correct; growth bounded by `max_growth`/`NeedsRefactor`. Journal:
dev/journal/2026-06-10-01.org.

---

## 2026-06-10 — L11: DenseLu::perm_inv is dead state (finding L11, repo-review-2026-06-09.md)

### Finding (verbatim)

> **L11** `DenseLu::perm_inv` is dead state — built and maintained
> (`dense_factor.rs:31,59-61,94-96`), never read. low/certain.

### Why this is recorded here rather than fixed

1. **The finding is accurate — confirmed by static analysis.** `DenseLu::perm_inv`
   (`src/lu/dense_factor.rs:31`, the `perm_inv[orig_row] = pivot_position`
   inverse) is *written* in the constructor (`:75-84`) and *re-maintained* on every
   `refactor()` (`:115-117`), but is never *read*: a crate-wide grep for
   `perm_inv` as an rvalue finds zero readers in `src/lu/` outside the writes in
   `dense_factor.rs`. `dense_update.rs` and `dense_solve.rs` contain no reference
   to it at all. (The sibling `qcol_inv` is read for the Q-scatter; the *sparse*
   `SparseLu::perm_inv` is read for sparse-spike seeding — only the *dense* row
   `perm_inv` is dead.) The line numbers in the finding (`59-61`, `94-96`) are
   stale — the file has shifted since the review — but the substance holds at the
   current `:75-84` / `:115-117`.

2. **Dead state has no runtime behavior to reproduce with a RED test.** Its
   presence does not affect any output of any valid call; removing it would change
   no result, only eliminate the maintenance writes. There is no input that yields
   a *wrong* answer to assert against, so the reproduce-first lifecycle the /loop
   mandates does not apply. This places L11 in the same "no test can fail on the
   bug" bucket as the other non-reproducible findings in this log.

3. **Removing maintained state I did not author warrants human approval, not a
   silent unilateral delete.** `perm_inv` is deliberately re-maintained inside
   `refactor()` (`:116`), which reads as a field a prior author intends to consume
   (e.g. a planned dense FT-update or row-permutation-aware path). Per the project
   guidance to "look at the target before deleting — if you didn't create it,
   surface that rather than proceed," excising another developer's intentionally-
   maintained field is a judgment call about whether that intended use is still
   coming, which belongs to a human reviewer.

### Disposition

No code change and no new test. Recommended as a trivial, safe cleanup *pending a
human decision*: if no dense path will consume the row-permutation inverse, delete
the field at `dense_factor.rs:31`, the constructor build at `:75-84`, and the
`refactor()` maintenance at `:115-117` (the compiler's dead-code / unused-field
analysis is the oracle — removal must still compile and pass the full suite). If a
future dense update path is planned, leave it and annotate the intent. Until that
decision, the field is harmless dead state (a few `usize` writes per factor /
refactor, never read).

Evidence: `src/lu/dense_factor.rs:31` (field), `:75-84` (constructor build),
`:115-117` (`refactor()` maintenance); zero rvalue readers across `src/lu/`
(grep). Journal: dev/journal/2026-06-10-01.org.

---

## 2026-06-10 — L12: allocation cluster in factor / bump-elim / dense rollback (finding L12, repo-review-2026-06-09.md)

### Finding (verbatim)

> **L12** Allocation cluster: `u_entries` Vec per column in the factor loop
> (`sparse_factor.rs:215`); two Vecs per column in `remap_and_sort_l`
> (`:490-491`); `pivot_data` clone + per-op row allocation in bump elimination
> (`sparse_update.rs:299,309`); full L/U/qcol clones for rollback on every dense
> update (`dense_update.rs:66-68`) — an undo log of touched entries would be far
> cheaper on the success path. low/certain.

### Why this is recorded here rather than fixed

1. **The finding is accurate — confirmed at the current (drifted) line numbers.**
   The cited lines have shifted since the review, but every site is real today:
   - factor loop: `let mut u_entries: Vec<(usize, f64)> = Vec::with_capacity(reach.len())`
     allocated *per column* (`src/lu/sparse_factor.rs:236`, consumed via
     `into_iter()` at `:318`); plus the per-row `u_rows`/`row` Vecs at `:350-352`;
   - `remap_and_sort_l`: `let mut order: Vec<usize> = Vec::new()`
     (`src/lu/sparse_factor.rs:537`);
   - bump elimination: `let pivot_data = self.u_rows[k].clone()`
     (`src/lu/sparse_update.rs:340`) and the per-op `row_sub` allocation
     (`:346` → `Vec::with_capacity` at `:429`);
   - dense update rollback: `self.u.clone()` / `self.l.clone()` /
     `self.qcol.clone()` on *every* update (`src/lu/dense_update.rs:75-77`).

2. **Correct output — this is allocation efficiency, not a wrong answer.** None of
   these sites changes any computed result; they only allocate more than a tuned
   implementation would. So there is no input that yields a *wrong* output for a
   RED test to assert against. The reproduction vehicle here is an allocation
   *count* (the L3 thread-local-counter pattern in `sparse_solve.rs`, which pins
   "zero per-call allocations"), not a correctness assertion — i.e. L12 is
   reproducible-in-principle, unlike the timing-flaky findings, but only as a
   non-functional allocation-count contract.

3. **The flagship fix is a correctness-sensitive, bench-gated rewrite.** Replacing
   the dense rollback clones with an undo log means recording every touched
   `(index, old_value)` during the in-place update and replaying it on the
   `NeedsRefactor` path, instead of cloning `u`/`l`/`qcol` up front and committing
   on success. That reworks the update's commit/rollback mechanism on the
   basis-update critical path — exactly where a subtle rollback bug would corrupt
   the factorization silently. The project mandate is "correctness before
   performance, always," and the feature lifecycle requires perf changes to go
   through research → plan → tests → implement → **benchmark**. That does not fit
   the surgical reproduce-first-then-commit cadence of this pass.

4. **It is a cluster, best addressed coherently.** Four distinct sites with one
   shared theme (reuse scratch / avoid clones). Fixing one in isolation (e.g.
   hoisting `u_entries` to a reused buffer cleared each column) would leave the
   headline item — the dense rollback clones — open, and would add factor-path
   allocation instrumentation for a fractional gain. These belong in one tuned
   optimization pass with allocation-count regression tests, not piecemeal.

### Disposition

No code change and no new test. Recommended as a tracked optimization task: (a)
add allocation-count instrumentation on the factor and update paths (extend the
L3 thread-local counter); (b) hoist the per-column/per-row scratch Vecs
(`u_entries`, `remap_and_sort_l` `order`, bump-elim `row_sub` scratch) to reused
buffers cleared per iteration; (c) replace the dense rollback clones with a
touched-entry undo log replayed on `NeedsRefactor`; (d) pin each with an
allocation-count test and validate net speedup on the LU bench before committing.
Until then the behavior is correct but allocates more than necessary on the
factor and update hot paths.

Evidence: `src/lu/sparse_factor.rs:236,318,350-352,537`;
`src/lu/sparse_update.rs:340,346,429`; `src/lu/dense_update.rs:75-77`. Output
unaffected; reproduction vehicle is allocation count (L3 pattern), not a
correctness assertion. Journal: dev/journal/2026-06-10-01.org.

## X8 — C-ABI status-code "mirror" comment (capi.rs:13-14): non-reproducible behavioral risk; doc corrected (2026-06-10)

**Finding (repo-review-2026-06-09.md §X8, low/likely):** the comment at
`src/capi.rs:13-14` claims the `FERAL_*` status codes "mirror Ipopt's
`ESymSolverStatus` enum," but `FERAL_FATAL = 3` collides with Ipopt's
`SYMSOLVER_CALL_AGAIN` (also enum value 3); Ipopt's fatal code is
`SYMSOLVER_FATAL_ERROR = 4`. The review's stated risk: "a numerically
pass-through shim would turn fatal errors into call-again loops."

### Why the behavioral risk is non-reproducible

The shim does **not** pass the integer through. `feral-ipopt-shim/src/
IpFeralSolverInterface.cpp:11-15` translates explicitly:

    case FERAL_SUCCESS:        return SYMSOLVER_SUCCESS;
    case FERAL_SINGULAR:       return SYMSOLVER_SINGULAR;
    case FERAL_WRONG_INERTIA:  return SYMSOLVER_WRONG_INERTIA;
    case FERAL_FATAL:
    default:                   return SYMSOLVER_FATAL_ERROR;

`FERAL_FATAL` maps to `SYMSOLVER_FATAL_ERROR` (the correct enum value 4),
not to value 3. The shim header even documents the intent
(`include/IpFeralSolverInterface.hpp:11`: "no SYMSOLVER_CALL_AGAIN").
So the "call-again loop" the review hypothesizes ("*would* turn …")
cannot occur in the real integration — it is conditioned on a
pass-through shim that does not exist.

There is therefore no input that produces a wrong status at any FERAL or
shim boundary for a RED test to assert against:
- The FERAL C ABI is a self-consistent four-value contract
  (`FERAL_SUCCESS=0, FERAL_SINGULAR=1, FERAL_WRONG_INERTIA=2,
  FERAL_FATAL=3`). Asserting those values in a Rust test would pin the
  implementation against itself (characterization), not reproduce a bug.
- The only place the values meet Ipopt's enum is the C++ shim, which is
  correct (translates, not casts) and outside the Rust test harness.

### Disposition

The actionable core of X8 is the inaccurate comment, not a code defect.
Corrected `src/capi.rs:13-14` in the same change to state precisely:
codes 0-2 share Ipopt's `SUCCESS/SINGULAR/WRONG_INERTIA` values; FERAL
has no `CALL_AGAIN` analog (Ipopt value 3), so `FERAL_FATAL` reuses value
3 and the shim must **translate** it to `SYMSOLVER_FATAL_ERROR` (value 4)
— it is not a numeric pass-through. No failing test exists or is added,
per the non-reproducibility above. Not user-visible behavior (internal
doc comment) → no CHANGELOG entry.

Evidence: `src/capi.rs:13-14,22-25`;
`feral-ipopt-shim/src/IpFeralSolverInterface.cpp:11-15`;
`feral-ipopt-shim/include/IpFeralSolverInterface.hpp:11`;
`ref/Ipopt/src/Algorithm/LinearSolvers/IpSymLinearSolver.hpp:19-33`.
Journal: dev/journal/2026-06-10-01.org.

## 2026-06-10 — X12: build_cost_graph per-column sort is dead work / micro-allocations (finding X12, repo-review-2026-06-09.md)

### Finding (verbatim)

> **X12** `build_cost_graph` per-column sort is dead work with O(n)
> per-run allocations (`mc64.rs:342-351`): the two-pass expansion
> already emits rows ascending. MC64 is documented as the dominant
> symbolic cost. low/likely (ordering argument) / certain
> (allocations).

(The cited lines correspond to `src/scaling/mc64.rs`; in the current
tree the per-column sort is at `mc64.rs:357-370` and the two-pass
expansion at `:307-355`.)

### Why this is not reproducible as a failing test

X12 is a performance / dead-work observation, not a correctness defect.
There is no input for which the present code produces a wrong result, so
no RED test can be written against it.

1. **The sort never changes the factorization result.** `build_cost_graph`
   feeds the Hungarian matching kernel and the column-max normalization.
   Column-max normalization (`mc64.rs:372-394`) computes a per-column
   maximum and subtracts it — order-independent. The Hungarian kernel
   consumes the column as a set of (row, cost) pairs; its matching and the
   resulting scaling do not depend on the within-column row order. The
   in-code comment says as much: "Hungarian kernel does not strictly
   require this … a predictable order makes the greedy initialization
   deterministic and matches SPRAL's behaviour after `half_to_full`." So
   the sort is at most a determinism/parity nicety, never a correctness
   input.

2. **On the maintained invariant the sort is provably a no-op (the
   "ordering argument").** feral's CSC columns are row-sorted ascending
   (the documented `CscPattern` invariant — see also finding O3). Under
   that invariant the expansion already emits each column ascending:
   - Transpose entries `(j, i)` for `j < i` are appended to column `i` as
     the outer loop runs `j = 0,1,…`, so they land in ascending `j`, and
     every such row is `< i`.
   - Own-column entries `(i', i)` with `i' >= i` are appended when the
     outer loop reaches `j = i`, in the ascending CSC row order, every row
     `>= i`.
   The concatenation `[rows < i ascending] ++ [rows >= i ascending]` is
   globally ascending, so `pairs.sort_by_key` reorders nothing. A test
   asserting "columns come out ascending" therefore PASSES on the current
   code — it confirms the sort is dead, it does not reproduce a bug.

3. **The O(n) per-run allocations are micro-overhead, not a defect.** The
   per-column `Vec<(usize, f64)>` in the sort loop and the `offsets`
   clone are extra allocations, but they change neither the output nor the
   asymptotic cost of the surrounding MC64 work. Removing them is a pure
   optimization with no observable behavioral change to assert.

### Disposition

Routed here per the /loop rule ("anything that can't be reproduced goes
to dev/tried-and-rejected.md citing the finding ID"). X12 describes
redundant work, not incorrect behavior: the sort is correctness-neutral
(the kernel and column-max normalization are order-invariant) and, on the
maintained row-sorted-CSC invariant, provably a no-op; the allocations
are micro-overhead. No failing test exists or is added. Removing the sort
and the per-run allocations would be a performance change with no output
delta — deliberately left out of this bug-fix loop, which is anchored on
reproducing defects. Not user-visible → no CHANGELOG entry.

Evidence: `src/scaling/mc64.rs:294-394` (build_cost_graph: expansion
`:307-355`, sort `:357-370`, column-max normalization `:372-394`).
Journal: dev/journal/2026-06-10-01.org.

## 2026-06-10 — X13: value_bound defensive-fingerprint comment is false (finding X13, repo-review-2026-06-09.md)

### Finding (verbatim)

> **X13** `value_bound.rs:180-189,222-226`: the defensive-fingerprint
> comment ("makes the subsequent check reject") is false — with
> mean_diag_0 = 0, condition 3 is vacuous; impact nil today only
> because the length gate rejects first. low/certain.

### Why this is not reproducible as a failing test

X13 is a false *comment*, not a behavioral defect. The cited code path
(`src/scaling/value_bound.rs:180-189`, the `else` arm of
`precompute_mc64_validity`) builds an all-zero `DominanceStats` when
`scaling.len() != matrix.n`, yielding the fingerprint `r0 = 1`,
`n_off_dominant_0 = 0`, `mean_diag_0 = 0`. The old comment claimed this
"makes the subsequent check reject (r0 = 1, mean = 0)". That is wrong on
two counts:

1. **The length gate rejects first.** This fingerprint is produced *only*
   when `scaling.len() != matrix.n`. Its sole consumer,
   `mc64_value_bound_passes` (`:212-227`), opens with
   `if scaling.len() != n || validity.qualifying.len() != n { return false; }`
   (`:218-220`) and returns `false` before evaluating any of conditions
   1-3. So the fingerprint values never drive the decision on the very
   inputs that produce them.

2. **`mean_diag_0 = 0` makes condition 3 vacuous, not rejecting.** Were the
   conditions evaluated, condition 3 is
   `min_diag >= EPS_DIAG * mean_diag_0 = EPS_DIAG * 0 = 0`, satisfied for
   any non-negative scaled diagonal — it *passes*, the opposite of the
   comment's "reject". And `r0 = 1` does not force condition 1 to fail.
   Only condition 2 (`n_off_dominant <= GROWTH_COUNT * 0`) would reject,
   and only when off-dominant rows exist.

Furthermore the `else` arm is unreachable in practice: callers pass
`SparseFactors::scaling`, whose length is `matrix.n` by construction
(documented at `:172-174`). The branch is dead defensive code whose only
job is to avoid an out-of-bounds index; it never reaches a value-bound
decision. There is thus no input for which behavior is wrong, so no RED
test can be written.

### Disposition

Routed here per the /loop rule (non-reproducible → tried-and-rejected
citing the finding ID), and — following the X8 precedent for misleading
comments — the inaccurate comment was corrected in the same change.
`src/scaling/value_bound.rs:180-189` now states that the all-zero
fingerprint is a never-consulted placeholder, that rejection on a length
mismatch comes from the length gate (not the fingerprint), and that
`mean_diag_0 = 0` makes condition 3 vacuous rather than rejecting. No
behavioral change; no failing test exists or is added. Not user-visible
(internal comment) → no CHANGELOG entry.

Evidence: `src/scaling/value_bound.rs:175-196` (precompute, defensive
arm), `:212-227` (length gate `:218-220`, conditions `:222-226`),
`:169-174` (caller contract). Journal: dev/journal/2026-06-10-01.org.

## 2026-06-10 — X14: dense-bench factor_nnz convention vs comments (finding X14, repo-review-2026-06-09.md)

### Finding (verbatim)

> **X14** `factor_nnz` for the dense bench path is n² with a comment
> claiming strictly-lower-triangle count (`bench.rs:1642-1645`); the
> code matches the multifrontal nrow·nelim convention, so the comment is
> what's wrong — but fill-parity readers need to know which.
> low/certain.

(The current tree puts the dense-path `factor_nnz` at `bench.rs:1662-1665`
and the field doc at `:1009-1013`.)

### What is actually there (deeper than the finding states)

The dense path sets `factor_nnz: Some(n²)` for a single fully-eliminated
`n×n` front (`nrow = nelim = n`). The old inline comment called this "the
strictly-lower-triangle entries (matches the multifrontal accounting on a
single supernode of size n)". Both clauses are wrong:

- `n²` is not the strictly-lower-triangle count (`n(n−1)/2`).
- The actual multifrontal accounting, `SparseFactors::factor_nnz()`
  (`src/numeric/factorize.rs:959-969`), is the *triangular*
  `Σ nelim·(nelim+1)/2 + (nrow−nelim)·nelim`. For a single full front of
  size n that is `n(n+1)/2`, not `n²`. So `n²` does **not** match the
  multifrontal accounting on a single supernode of size n; it is ~2× it.

`n²` is in fact the *rectangular* `nrow·nelim` product — which is the
convention the `factor_nnz` field doc (`bench.rs:1009`) literally states
("total `nrow * nelim` across supernodes"). So the real situation is a
convention split inside the bench harness:

| path        | code                                   | convention            |
|-------------|----------------------------------------|-----------------------|
| dense       | `n²` (`:1665`)                          | rectangular nrow·nelim |
| sparse      | `sp_factors.factor_nnz()` (`:1951`)     | triangular            |

The field doc matches the dense path; `factor_nnz()`'s implementation
matches the sparse path; the doc and the implementation disagree with
each other. The fill-parity report (`bench.rs:487-494`) therefore counts
dense-path matrices rectangularly and sparse-path matrices triangularly —
a ~2× convention skew between rows of the same report. The old field doc
also wrongly claimed `factor_nnz` is "`None` on the dense path"; the dense
path sets `Some(n²)`.

### Why this is not reproducible as a defect fix here

`factor_nnz` is a *diagnostic* metric feeding the fill-parity report; it
has no bearing on solver correctness (inertia, residuals, factor values
are all independent of it). Unifying the two paths to one convention is a
deliberate design choice with no external oracle:

- the codebase is internally split (field doc = rectangular,
  `factor_nnz()` = triangular), so there is no single in-repo "truth" to
  assert against;
- the right cross-solver convention to compare against MUMPS / SSIDS
  `factor_nnz` is a separate, unsettled question;
- changing the dense-path value would silently move historical
  fill-parity baselines for every dense-path matrix.

A test could only pin one arbitrary convention against itself
(characterization), not reproduce a defect. Out of scope for a bug-fix
loop anchored on reproducing solver defects.

### Disposition

Routed here per the /loop rule (non-reproducible → tried-and-rejected
citing the finding ID). Following the X8 / X13 precedent for misleading
comments, both inaccurate comments were corrected in the same change:

- `bench.rs:1662-1665` now states the dense path reports the rectangular
  `nrow·nelim = n²` front-cell count, explicitly contrasts it with the
  triangular `SparseFactors::factor_nnz()` (`n(n+1)/2` for a single full
  front, used by the sparse path), and warns fill-parity readers the two
  paths are not directly comparable.
- `bench.rs:1009-1013` field doc corrected to describe both conventions
  and to state `factor_nnz` is `Some` on both paths (it is not `None` on
  the dense path).

The underlying convention split (dense rectangular vs sparse triangular,
and the field-doc-vs-`factor_nnz()` disagreement) is documented here for a
future deliberate decision; the `n²` value itself is left unchanged (no
oracle, diagnostic-only, would move baselines). Not user-visible (bench
diagnostic + internal comments) → no CHANGELOG entry.

Evidence: `src/bin/bench.rs:1009-1013` (field doc), `:1662-1665`
(dense-path value), `:1951` (sparse-path value), `:487-494` (fill-parity
report); `src/numeric/factorize.rs:959-969` (`factor_nnz()` triangular
formula); `src/dense/factor.rs:788,1247-1256` (dense `Factors` has no
`factor_nnz()`; `nrow`/`nelim` fields). Journal:
dev/journal/2026-06-10-01.org.

## 2026-06-10 — X16: Inertia struct "invariant: pos+neg+zero == n" doc (finding X16, repo-review-2026-06-09.md)

### Finding (verbatim)

> **X16** `Inertia::new` (`inertia.rs:12-18`) doesn't enforce the
> documented `pos+neg+zero == n` invariant; doc reads like a guarantee.
> low/certain.

### Why this is not reproducible as a defect

`Inertia` is a plain triple of `usize` counts (`positive`, `negative`,
`zero`). `Inertia::new(positive, negative, zero)` stores exactly what it is
given. There is no defective behavior to reproduce:

1. **`new` takes no `n`.** Its signature is
   `new(positive, negative, zero)` — there is no matrix dimension in scope,
   so it is structurally impossible for `new` to check `pos+neg+zero == n`.
   The "invariant" references an external `n` the constructor never sees.

2. **The "== n" invariant is violated by design for sub-blocks.** `Inertia`
   is intentionally used to describe sub-blocks, not just whole matrices.
   The 2×2 pivot classifier `classify_2x2_inertia`
   (`src/dense/factor.rs:4313-4332`) returns `Inertia::new(1,1,0)`,
   `Inertia::new(2,0,0)`, `Inertia::new(0,0,2)`, etc. — every one has
   `total() == 2`, the order of the 2×2 block, **not** the global matrix
   order `n`. `count_2x2_inertia_val` (`:4259`) does likewise. So adding
   `assert!(pos+neg+zero == n)` to `new` is not just impossible (no `n`) but
   semantically wrong: it would reject the type's legitimate sub-block use.

3. **No caller relies on a false guarantee.** Every whole-matrix
   construction site (`src/dense/factor.rs:1158,1786,2281,2515`, the bench
   harness, etc.) passes counts that already sum to the relevant dimension;
   `total()` (`inertia.rs:21-23`) recomputes the sum on demand. Nothing reads
   back an enforced invariant that could be wrong.

A test could only assert that `new` stores what it is given
(`Inertia::new(1,1,1).total() == 3`) — which passes, i.e. characterizes
correct behavior; it cannot go RED on a defect. The real issue is purely a
doc-comment that *reads* like an enforced guarantee when the relationship is
a caller-upheld contract (and a tautology with `total()` for whatever
(sub)matrix the inertia describes).

### Disposition

Routed here per the /loop rule (non-reproducible → tried-and-rejected citing
the finding ID). Following the X8 / X13 / X14 precedent for misleading
comments, the doc was corrected in place (`src/inertia.rs:1-9`, `:11-13`):
the struct doc now states `total()` equals the dimension of whatever
(sub)matrix the inertia describes, explicitly notes the 2×2-block use where
`total() == 2`, and that counts are caller-supplied and unvalidated; the
`new` doc states the counts are stored as given and not validated against any
dimension. No behavioral change; no enforcement added (it would break the
sub-block use). Not user-visible (internal doc) → no CHANGELOG entry.

Evidence: `src/inertia.rs:1-30`; `src/dense/factor.rs:4259,4313-4332`
(2×2-block inertias with `total()==2`); whole-matrix sites
`src/dense/factor.rs:1158,1786,2281,2515`. Journal:
dev/journal/2026-06-10-01.org.

## 2026-06-10 — O4: non-aggressive Pass-2 stale-mark `as usize` cast (finding O4, repo-review-2026-06-09.md)

### Finding (verbatim)

> **O4** Non-aggressive Pass-2 casts a possibly-stale mark difference
> straight to usize with no guard (`algo.rs:407`, AMF branch
> `:962-977`); the invariant holds today, but a regression wraps to
> ~2⁶⁴. Add `debug_assert!(we >= ws.wflg)`. low/possible.

### What is there

In `crates/feral-ordering-core/src/quotient_graph/algo.rs`, the
non-aggressive element sub-pass of Pass-2 computes the external degree
contribution of a live element `e` as `we - ws.wflg`, where
`we = ws.w[e]` and `ws.wflg` are both `i32`:

- AMD accumulator, `:406-408`:
  `let dext = (we - ws.wflg) as usize; deg += dext;`
- AMF accumulator, `:995-1000`:
  `let dext = we - ws.wflg; ... deg += dext as usize;`

If a live element (`we != 0`) ever had `we < ws.wflg`, the `i32`
subtraction is negative and the `as usize` cast sign-extends it to ~2^64,
corrupting `deg`.

### Why this is not reproducible as a defect

The algorithm maintains the invariant `we >= ws.wflg` for every live
element in the non-aggressive pass; the review confirms "the invariant
holds today" and rates the finding "low/possible". `ws.wflg` is the current
flag baseline and live elements carry a mark `>=` it by construction.
Driving `we < ws.wflg` requires directly corrupting the internal workspace
(`ws.w`, `ws.wflg`), which no public ordering entry point exposes — the
producers take a `CscPattern` and own the workspace privately. So no test
can reproduce the wrap on a real input; a test could only manufacture the
violation by reaching into private state, which characterizes the guard,
not a defect. (The aggressive AMF branch is already safe: `:972` guards
`if dext > 0` before using `dext`.)

### Disposition

Routed here per the /loop rule (non-reproducible → tried-and-rejected citing
the finding ID). Per the finding's explicit recommendation and the X12/X16
precedent (non-reproducible findings with a concrete low-risk in-code
hardening), a `debug_assert!` documenting the `we >= ws.wflg` invariant was
added at both non-aggressive cast sites (AMD `:407`, AMF `:1000`). It is
debug/test-only (compiled out in release), changes no behavior, and never
fires on current inputs — the full feral-ordering-core / feral-amd /
feral-amf / feral-metis / feral-scotch test suites pass with it in place. It
converts a silent ~2^64 wrap on a future regression into an immediate,
located assertion failure. Not user-visible (internal debug guard) → no
CHANGELOG entry.

Evidence:
`crates/feral-ordering-core/src/quotient_graph/algo.rs:402-413` (AMD
non-aggressive branch), `:990-1006` (AMF non-aggressive branch), `:966-988`
(aggressive AMF branch already guarded by `if dext > 0` at `:972`). Journal:
dev/journal/2026-06-10-01.org.

## 2026-06-10 — O5: dense-threshold formula in `dense_alpha` docs (finding O5, repo-review-2026-06-09.md)

### Finding (verbatim)

> **O5** Dense-threshold formula deviates from its own doc for small n
> / negative alpha (`workspace.rs:204-210`): `.max(16)` overrides the
> documented n−2 for n < 18; `min(max(16,x),n)` ≠ documented
> `max(16,min(n,x))`. No practical effect. low/certain.

### What is there

`crates/feral-ordering-core/src/quotient_graph/workspace.rs:212-217`:

    let dense = if opts.dense_alpha < 0.0 {
        n.saturating_sub(2)                          // n - 2
    } else {
        (opts.dense_alpha * (n as f64).sqrt()) as usize
    };
    let dense = dense.max(16).min(n);                // min(max(16, raw), n)

The three `dense_alpha` doc comments
(`feral-amd/src/lib.rs:40-45`, `feral-amf/src/lib.rs:46-50`,
`feral-ordering-core/src/quotient_graph/mod.rs:50-55`) describe this as
`max(16, min(n, dense_alpha * sqrt(n)))` and say a negative value "sets the
threshold to `n - 2`". Two inaccuracies: (1) the clamp nesting is reversed —
code is `min(max(16, raw), n)`, doc is `max(16, min(n, raw))`; (2) the
negative-alpha branch is also passed through `max(16)`/`min(n)`, so it is not
a bare `n - 2` for `n < 18`.

### Why this is not reproducible as a defect — the code is canonical-correct

The code matches the faer / SuiteSparse AMD reference exactly. Verified
against faer 0.24.0 (the version feral references), `amd.rs:173-179`
(confirmed by the faer-expert agent reading the cargo-registry source):

    let dense = if alpha < 0.0 { n - 2 } else { (alpha * sqrt(n)) as usize };
    let dense = Ord::max(dense, 16);     // max(16) first
    let dense = Ord::min(dense, n);      // min(n) second

i.e. `min(max(16, raw), n)`, with both clamps applied unconditionally to both
branches — identical to feral's `dense.max(16).min(n)`. The nesting order is
load-bearing: faer's form guarantees `dense <= n`, whereas the doc's
`max(16, min(n, raw))` would give `16 > n` for `n < 16`.

So the implementation is the canonical one and the doc transcription is the
artifact that is wrong. There is no behavioral defect to reproduce:

- positive branch: `min(max(16, x), n)` and `max(16, min(n, x))` agree for
  all `n >= 16`; they diverge only for `n < 16`, where the threshold (>= 16)
  already exceeds the maximum possible vertex degree `n - 1`, so no vertex is
  ever classified dense differently;
- negative branch: the code matches the faer reference, which is the oracle.

A test could only characterize the code against itself or against faer
(confirming correct behavior), not drive a RED defect.

### Disposition

Routed here per the /loop rule (non-reproducible → tried-and-rejected citing
the finding ID). Per the X8/X13/X14/X16 precedent for inaccurate docs, the
three `dense_alpha` doc comments are corrected to the actual formula
`min(max(16, floor(dense_alpha * sqrt(n))), n)`, noting the clamp order
matches faer `amd.rs:173-179` and that the negative-alpha branch uses a raw
`n - 2` with the same clamps (exactly `n - 2` for `n >= 18`). No code change
(the implementation is canonical-correct). Not user-visible (doc comments
only) → no CHANGELOG entry.

Evidence: `crates/feral-ordering-core/src/quotient_graph/workspace.rs:212-217`
(code); doc sites `feral-amd/src/lib.rs:40-45`, `feral-amf/src/lib.rs:46-50`,
`feral-ordering-core/src/quotient_graph/mod.rs:50-55`; faer 0.24.0
`sparse/linalg/amd.rs:173-179` (external oracle, faer-expert verified).
Journal: dev/journal/2026-06-10-01.org.

## 2026-06-10 — O6: `n_clear_flag` stat hard-coded 0 vs "every field populated" doc (finding O6, repo-review-2026-06-09.md)

### Finding (verbatim)

> **O6** `n_clear_flag` stat hard-coded 0 in both feral-amd
> (`lib.rs:115`) and feral-amf (`lib.rs:121`) while the stats docs
> claim "every field is populated" in debug builds. low/certain.

(Current tree: the hard-coded `0` is at `feral-amd/src/lib.rs:119` and
`feral-amf/src/lib.rs:123`.)

### What is there

`AmdStats` / `AmfStats` expose `pub n_clear_flag: u32`, documented as
"Number of mark-array generation-counter resets"; `AmdStats`'s struct doc
(`feral-amd/src/stats.rs:5-7`) claims "In debug builds every field is
populated". Both `amd_order_full` and the AMF equivalent set
`n_clear_flag: 0` literally. The shared `OrderDiagnostics`
(`quotient_graph/mod.rs:73-79`) carries `ncmpa`, `n_mass_elim`,
`n_supervar_merge`, `ndense`, `flops` — there is no clear-flag field, so the
stat has nothing to read from and is a disconnected constant.

### Why this is not reproducible as a defect

The reset it would count (`clear_flag`, `quotient_graph/workspace.rs:49-59`)
fires only when `wflg < 2` (the one-time lift from the initial `wflg = 0`,
over an all-ones `w`) or `wflg >= wbig`, where `wbig = i32::MAX - n`
(`workspace.rs:99-100`). During elimination `wflg` increases by at most
`lemax <= n` per pivot across at most `n` pivots (~`n^2` total), so reaching
the `~2^31` ceiling requires `n` on the order of tens of thousands. On every
practically unit-testable input the true reset count is `0`.

Consequences:

- A black-box test cannot distinguish a hard-coded `0` from a correctly
  computed `0`; both are `0` for all testable inputs. No RED state exists.
- Forcing a non-zero value needs an `n ~ 46k` matrix with the right fill —
  not a unit test.
- The counter is feral-specific; SuiteSparse / faer AMD do not expose it, so
  there is no external oracle for its value. Wiring it up and asserting a
  value would be self-authored impl + self-authored oracle in one session
  (forbidden by the project rule), and would require changing `clear_flag`'s
  signature across its four hot-loop call sites (algo.rs:280, 501, 891, 1101)
  for a value that is ~always `0`.

### Disposition

Routed here per the /loop rule (non-reproducible → tried-and-rejected citing
the finding ID). Per the X16/O4/O5 precedent for non-reproducible findings
with an inaccurate doc, the docs are corrected rather than the code:

- `feral-amd/src/stats.rs`: the `n_clear_flag` field doc now states the
  field is currently always `0` and not wired to a backing counter, with the
  `wbig = i32::MAX - n` ceiling explaining why the reset is a near-never
  event; the struct-level "every field is populated" claim is amended to
  carve out `n_clear_flag`.
- `feral-amf/src/stats.rs`: the `n_clear_flag` field doc gets the same
  correction.

No code change; the hard-coded `0` is left as-is (it is the correct value on
every input the crate can be tested against). Not user-visible (doc comments
only) → no CHANGELOG entry.

Evidence: `feral-amd/src/lib.rs:117-126` and `feral-amf/src/lib.rs` (stats
assembly, `n_clear_flag: 0`); `feral-amd/src/stats.rs:5-13`,
`feral-amf/src/stats.rs:5-13` (docs); `quotient_graph/mod.rs:73-79`
(`OrderDiagnostics` has no clear-flag field);
`quotient_graph/workspace.rs:49-59` (`clear_flag`), `:99-100`
(`wbig = i32::MAX - n`). Journal: dev/journal/2026-06-10-01.org.

## 2026-06-10 — O9: metis `two_hop_pass` O(n^2) on hubs (finding O9, repo-review-2026-06-09.md)

### Finding (verbatim)

> **O9** metis `two_hop_pass` is O(n²) on hub graphs
> (`coarsen.rs:148-204`) — its own motivating case; rescans
> neighbor-of-neighbor lists from the start per spoke; `mark` allocated
> and never used. medium-low/certain (perf).

### Why this is not reproducible as a correctness test

O9 is a performance finding, not a wrong-answer defect. `two_hop_pass`
produces a correct matching; the complaint is that on a star/hub graph each
self-matched spoke `v` rescans its hub neighbour's entire adjacency from the
start to find a self-matched partner, so ~n spokes each scan ~n hub-neighbours
=> O(n^2). A unit test cannot turn "quadratic instead of linear" into a
deterministic pass/fail without a timing/benchmark oracle, which the spec's
tests-first lifecycle does not provide for this kind of finding (and timing
assertions are flaky). So there is no RED state to write.

### Why the O(n^2) rescan is not rewritten here

The current algorithm pairs each self-matched `v` (in increasing vertex
order) with the *first* self-matched 2-hop neighbour found while scanning
`v`'s neighbours in adjacency order and each neighbour's adjacency in order.
The pairing is therefore order-sensitive and deterministic. A genuinely
O(nnz) rewrite (METIS Match_2Hop buckets unmatched vertices by a
representative neighbour and pairs within buckets) would choose *different*
partners, changing `cmap`, the coarse graph, and the final permutation. That
would break the crate's determinism contract and the
`coarsen_is_deterministic_with_seed` test, and shift ordering quality on
every input — far out of proportion to an opportunistic low-severity perf
fix. The two-hop pass also only fires when SHEM's reduction ratio exceeds the
two-hop threshold (default 0.85), i.e. on the already-rare hard-to-match
levels, bounding the practical impact.

### Disposition

Routed here per the /loop rule (non-reproducible -> tried-and-rejected citing
the finding ID). Per the X16/O4/O5/O6 precedent, the finding's safe, explicitly
recommended sub-fix is applied: the dead `mark` array — allocated `vec![-1; n]`
and only ever touched by a `let _ = &mut mark;` lint-silencer — is removed,
along with the stale comment describing its non-use. This is a pure cleanup
with no behaviour change; the existing two-hop tests
(`coarsen_grid_8x8_halves_vertices`, `coarsen_is_deterministic_with_seed`,
`coarsen_hierarchy_shrinks_monotonically`) remain the regression guard. The
O(n^2) scan structure is left as-is.

Evidence: `coarsen.rs:148-204` (two_hop_pass); the `mark` declaration and the
`let _ = &mut mark;` no-op were the only references to it. Journal:
dev/journal/2026-06-10-01.org.

## 2026-06-10 — O11: metis FM heap seeded with all n vertices (finding O11, repo-review-2026-06-09.md)

### Finding (verbatim)

> **O11** metis FM heap seeded with all n vertices each pass
> (`fm_refine.rs:56-61`): Ω(n log n) per pass × 10 passes × every
> level, boundary-only seeding is the standard. Acknowledged trade;
> flagged for cost. low/certain.

### Why this is not reproducible as a correctness test

O11 is a cost finding, not a wrong-answer defect — the reviewer marks
it "Acknowledged trade; flagged for cost." `refine_bisection` produces
a correct, balance-respecting FM result; the complaint is that it seeds
the gain heap with every vertex (`for (v, &g) in gain.iter()...`)
instead of only the current boundary, so each pass pays Ω(n log n) heap
inserts where METIS pays Ω(boundary). A unit test cannot turn
"asymptotically more heap work" into a deterministic pass/fail without a
timing/benchmark oracle, which the tests-first lifecycle does not
provide for this kind of finding (and timing assertions are flaky).
There is no RED state to write.

### Why boundary-only seeding is not adopted here

METIS-style boundary-only seeding is not a drop-in: it requires lazily
re-inserting a vertex the first time a neighbour move makes it a
boundary vertex. That changes the order in which equal-gain vertices
are visited (the heap no longer contains the interior vertices that
currently tie-break by index), so it changes the FM move trajectory and
therefore the final labels on inputs where the best-balanced prefix is
reached through a different sequence. That would shift ordering output
and break the crate's determinism contract and the FM/ND determinism
tests — out of proportion to a low/certain cost finding the reviewer
already flagged as an acknowledged trade. Interior vertices are not
free correctness risk in the current code: they have
gain = -internal_degree ≤ 0, so they sit at the bottom of the max-heap
and are popped only after the positive-gain boundary moves that reduce
the cut.

### Disposition

Routed here per the /loop rule (non-reproducible -> tried-and-rejected
citing the finding ID). Per the X16/O4/O5/O6/O9 precedent, the safe
low-risk sub-fix is applied: the all-n seeding cost trade — previously
documented only in this review — is now acknowledged in the code at the
seeding loop (`fm_refine.rs`), noting METIS's boundary-only +
lazy-reinsertion alternative, the Ω(boundary) vs Ω(n log n) cost, why
interior vertices are harmless (gain ≤ 0), and the simplicity-over-speed
rationale at FERAL's target sizes. No behaviour change; existing FM
tests remain the regression guard.

Evidence: `fm_refine.rs:56-61` (heap seeding loop). Journal:
dev/journal/2026-06-10-01.org.

## 2026-06-11 — O12: metis dense-quotient comment cites debunked HSL_MC68/ICNTL(6)/SSIDS basis (finding O12, repo-review-2026-06-09.md)

### Finding (verbatim)

> **O12** metis doc drift: `lib.rs:219` still cites HSL_MC68/ICNTL(6)/
> SSIDS for the dense-quotient path while the `MetisOptions` doc
> (`lib.rs:105-121`) explains that belief was audited and found wrong.
> low/certain.

### Why this is not reproducible as a correctness test

O12 is documentation drift, not a behavioural defect. The inline
`// Fix A` comment in `metis_order_full` (lib.rs:218-220) asserted the
dense-quotient path is the "Same technique as HSL_MC68 / MUMPS
ICNTL(6) / SSIDS", but the canonical `MetisOptions::dense_quotient_enabled`
doc (lib.rs:105-121) already records a 2026-04-27 audit of the MUMPS and
SPRAL sources that found that belief wrong: ICNTL(6) is MC64 matching,
MUMPS defers dense rows inside QAMD (`MUMPS_QAMD`, THRESM/HEAD(N)), and
SSIDS does not special-case dense rows at all — neither solver
pre-strips the graph. There is no runtime behaviour to assert here (the
two doc sites describe the same `dense_quotient_enabled` code path,
which is unchanged and default-off); a unit test cannot pin prose.

### Disposition

Routed here per the /loop rule (non-reproducible -> tried-and-rejected
citing the finding ID). Per the O6 doc-correction precedent, the
recommended low-risk sub-fix is applied: the stale inline comment is
rewritten to state that the HSL_MC68/ICNTL(6)/SSIDS equivalence was
audited and found wrong, and to point at
`MetisOptions::dense_quotient_enabled` for the full finding, so the two
doc sites no longer contradict each other. Comment-only; no behaviour
change, no public-API surface change -> no CHANGELOG.

Evidence: `lib.rs:218-220` (old comment) vs `lib.rs:105-121` (audited
finding). Journal: dev/journal/2026-06-10-01.org.

## 2026-06-11 — O14: scotch band FM is dead code while the crate doc advertises it (finding O14, repo-review-2026-06-09.md)

### Finding (verbatim)

> **O14** scotch band FM is dead code while `lib.rs:12-13` advertises
> it; `band_fm.rs` is `#[allow(dead_code)]`, unreachable from
> `node_nd.rs`; projection-loop variable names swapped
> (`band_fm.rs:76-81` — functionally correct, editor trap).
> low/certain.

### Why this is not reproducible as a correctness test

O14 has no behavioural defect to pin with a RED→GREEN test:

1. The `band_fm` module is unreachable from the production ND path —
   `node_nd.rs` contains no reference to it (`grep band` is empty), and
   `mod band_fm;` carries `#[allow(dead_code)]` (lib.rs:36-37). Dead
   code cannot be exercised by a library test, so there is nothing for
   a new test to drive.
2. The projection-loop variable names at `band_fm.rs:76-81` are
   *swapped but functionally correct*. `BandGraph::orig_of_sub` is
   indexed by sub-vertex and yields the original-graph vertex (struct
   doc, band_fm.rs:138-140), so `.enumerate()` produces
   `(sub_index, orig_vertex)` — yet the loop bound them as
   `(orig_v, &sub_v)`. The body `labels[sub_v] = sub_labels[orig_v]`
   therefore reads `labels[orig_vertex] = sub_labels[sub_index]`, which
   is exactly the intended projection. The existing module test
   `out_of_band_labels_preserved` already asserts the projection is
   correct and passes, so there is no failing behaviour to reproduce —
   only a naming trap that misleads a human reader.

### Disposition

Routed here per the /loop rule (non-reproducible → tried-and-rejected
citing the finding ID). Per the X16/O4/O5/O6/O9 precedent, the safe
low-risk sub-fixes are applied:

- `band_fm.rs:76-81`: the projection loop is renamed to
  `(sub_i, &orig_v)` with the body `labels[orig_v] = sub_labels[sub_i]`
  and the anchor guard keyed on `sub_i`. Pure rename — the generated
  code is identical, and all six `band_fm` tests still pass.
- `lib.rs:10-12`: the "Adaptive refinement (boundary / halo / band FM)"
  bullet is corrected so it no longer advertises band FM as part of the
  active pipeline; band FM is noted as implemented and unit-tested but
  not yet wired into the default ND driver.

The module itself is kept (it is tested and backed by the research note
`dev/research/scotch-band-fm.md`); wiring it into `node_nd` is a
behavioural change with its own benchmarking and is out of scope for a
low/certain documentation finding. Comment/rename only; no behaviour
change, no public-API surface change → no CHANGELOG.

Evidence: `crates/feral-scotch/src/lib.rs:36-37` (`#[allow(dead_code)]
mod band_fm;`), empty `grep band crates/feral-scotch/src/node_nd.rs`,
`band_fm.rs:76-81` (projection loop), `band_fm.rs:138-140` (orig_of_sub
contract), `band_fm.rs:488-511` (`out_of_band_labels_preserved` guard).
Journal: dev/journal/2026-06-10-01.org.

## 2026-06-11 — O15: scotch AMD leaves ignore supervariable weights on compressed graphs (finding O15, repo-review-2026-06-09.md)

### Finding (verbatim)

> **O15** scotch AMD leaves on the compressed graph ignore
> supervariable weights (`node_nd.rs:54-62`, `amd_leaf:281-302`):
> weight-7 supervariables treated as unit vertices, skewing
> degree-based pivots on heavily compressed inputs. Expansion still a
> valid permutation. medium-low/certain (code) / possible (impact).

### Why this is not reproducible as a correctness test

The code defect is real and certain: with `opts.compress` on,
`scotch_nd_order` builds `cg.graph` whose `vwgt` carries supervariable
sizes (node_nd.rs:54-62), those weights ride down through bisection
into leaf subgraphs, and `amd_leaf` (node_nd.rs:281-302) hands the leaf
to `feral_amd::amd_order` via `graph_to_csc_pattern`, which emits only
the adjacency structure and drops `vwgt` entirely (node_nd.rs:308-331).
AMD therefore scores a weight-7 supervariable as a unit vertex.

But there is no RED→GREEN cycle available:

1. **No correctness defect.** The finding itself notes "expansion still
   a valid permutation" — the leaf still emits a bijection over its
   vertices and `expand_perm` lifts it to a valid permutation of the
   original matrix. There is no invariant to violate, so no assertion
   can be made RED on the current code.
2. **No oracle for the quality loss.** The only effect is suboptimal
   degree-based pivoting (more fill) on heavily-compressed inputs. To
   assert "this ordering is worse than the weight-aware one" requires a
   weight-aware AMD to compare against — and `feral_amd` exposes no
   weighted entry point: every public function (`amd_order`,
   `amd_order_opts`, `amd_order_full`, …) takes a bare `CscPattern`
   plus `AmdOptions { dense_alpha, aggressive }`, with no `vwgt`
   parameter. AMD does its *own* internal supervariable merging
   (`n_supervar_merge`) but starts from unit mass and cannot know a
   leaf vertex already stands for 7 original rows.

The proper fix — a weight-aware AMD leaf (threading `vwgt` into the
minimum-degree metric) — is a cross-crate feature addition to
`feral_amd` that needs its own research note, tests, and benchmarks per
the FERAL feature lifecycle. That is out of proportion to a
medium-low/certain finding whose impact is rated only "possible," and
is out of scope for a single review-fix iteration.

### Disposition

Routed here per the /loop rule (non-reproducible → tried-and-rejected
citing the finding ID). Per the X16/O4/O5/O6/O9 precedent, the safe
low-risk sub-fix is applied: `amd_leaf` gains a doc comment recording
that supervariable weights from top-level compression are intentionally
dropped at the leaf, why correctness still holds (valid permutation),
the quality caveat on heavily-compressed inputs, and that a weighted
AMD leaf is the future fix (pointing back here). No behaviour change, no
public-API surface change → no CHANGELOG. No test is added: there is no
violated invariant to pin, and the quality oracle (weighted AMD) does
not yet exist.

Evidence: `node_nd.rs:54-62` (compressed-graph dispatch carries vwgt),
`node_nd.rs:281-302` (`amd_leaf`), `node_nd.rs:308-331`
(`graph_to_csc_pattern` drops vwgt), `feral-amd/src/lib.rs:68-176` (no
weighted entry point). Journal: dev/journal/2026-06-10-01.org.
## 2026-06-11 — O17: kahip `apply_degree2` rescans from vertex 0 per chain (finding O17, repo-review-2026-06-09.md)

### Finding (verbatim)

> **O17** kahip `apply_degree2` rescans from vertex 0 per chain
> (`data_reduction.rs:307-309`): O(n²) worst case on long paths. Off by
> default (Rule 1 only); will bite when Rule 2 is enabled. low-medium/certain.

### Why this is not reproducible as a correctness test

O17 is a performance finding, not a wrong-answer defect. `apply_degree2`
collapses degree-2 chains correctly; the complaint is that the seed scan
`(0..n).find(|v| alive && !skip && degree==2)` restarts from index 0 on every
outer iteration, so a graph that is one long degree-2 chain pays ~n scans of
~n vertices => O(n²). A unit test cannot turn "quadratic instead of linear"
into a deterministic pass/fail without a timing/benchmark oracle, which the
spec's tests-first lifecycle does not provide for this kind of finding (and
timing assertions are flaky). So there is no RED state to write. This mirrors
the O9 disposition (metis `two_hop_pass` O(n²)).

### Why the O(n²) scan is not rewritten here

The finding's implied fix — a non-rewinding scan cursor (start the next seed
search at the previous seed instead of 0) — is **not** a behaviour-preserving
drop-in, and code inspection proves it:

The simplicial collapse branch (`data_reduction.rs:399-405`) removes the chain
interior and, when `u ∼ w` already (`simplicial == true`), adds **no**
compensating `(u, w)` edge. Removing `path[0]` deletes the `u–path[0]` edge, so
endpoint `u`'s degree drops by one — a degree-3 branch endpoint becomes degree
2. The backward walk from `seed` can land on such a `u` at an index *below*
`seed`. The current from-0 scan always picks the lowest-index eligible vertex,
so it collapses that newly-degree-2 `u` *within the same `apply_degree2` call*;
a cursor that had advanced past `u` would instead defer it to the next
fixed-point round (`reduce_graph` loop, `data_reduction.rs:215-230`). The
chain still collapses eventually, but the `Degree2Path` ops are emitted in a
**different order**, and op-stack order determines the reconstructed
permutation (`expand_permutation` replays the stack in reverse). So a naive
cursor changes the produced ordering — out of proportion to an opportunistic
low-severity perf fix, and against "correctness before performance."

The order-preserving O(n log n) fix is a min-index worklist: a binary heap
keyed by vertex index, push a vertex when its degree falls to 2, pop the
lowest index with a lazy `alive && !skip && degree==2` staleness recheck. That
reproduces "lowest-index eligible seed every iteration" exactly while avoiding
the rescan. It is deferred because Rule 2 is **test-only** today — the driver
(`node_nd.rs:45`) runs `ReduceOptions::conservative()` (Rule 1 only);
`ReduceOptions::full()` is `#[cfg(test)]`. The O(n²) cost is therefore latent
and never on a production path.

### Disposition

Routed here per the /loop rule (non-reproducible -> tried-and-rejected citing
the finding ID). Per the X16/O4/O5/O6/O9 precedent, the finding's safe
sub-fix is applied: a code comment at the seed-scan site documenting the
O(n²), proving why a cursor is unsafe (the simplicial degree-drop above), and
recording the correct min-index-worklist fix for whoever enables Rule 2. This
is a pure documentation change with no behaviour change; the existing Rule 2
tests (`path_n5_collapses_to_branch_endpoints`,
`degree2_compression_fires_on_isolated_chain`,
`k_2_3_collapses_via_degree2_then_closed_twins`,
`triangle_with_tail_collapses_via_rule1_then_closed_twins`) remain the
regression guard. The O(n²) scan structure is left as-is.

Evidence: `data_reduction.rs:307-309` (from-0 seed scan), `:399-405`
(simplicial collapse drops endpoint degree, no compensating edge),
`:215-230` (fixed-point driver), `:166-184` (conservative vs full presets),
`node_nd.rs:45` (driver uses conservative). Journal:
dev/journal/2026-06-10-01.org.
## 2026-06-11 — O19: kahip flow stranded-vertex branch corrupts gap histogram (finding O19, repo-review-2026-06-09.md)

### Finding (verbatim)

> **O19** kahip `flow.rs:271-279` stranded-vertex branch would corrupt
> the gap histogram (sets height without decrementing height_count) —
> unreachable today (excess implies a residual reverse edge); add a
> debug_assert or comment. low/certain (code) / dead (impact).

### Why this is not reproducible as a correctness test

The branch is dead code. `discharge` reaches the relabel step only after the
`if excess[u] == 0 { return }` guard (`flow.rs:254`), so `excess[u] > 0` holds.
An active vertex always has at least one residual out-edge: the edge that
delivered its excess left a residual *reverse* edge `u -> v` in `adj[u]` with
`residual() > 0`. The relabel scan (`flow.rs:265-269`) therefore always finds a
finite `new_height`, so the `new_height == usize::MAX` branch at `:271` cannot
be entered while the vertex is active. No input can drive an active vertex to
have zero residual out-edges, so there is no RED state to construct — this is a
"dead impact" finding, as the reviewer marks it.

### What the branch would do if reached

It sets `height[u] = 2 * n` and returns *without* the
`height_count[old_h] -= 1` that the normal relabel path performs at
`flow.rs:286-287`. That would over-count `old_h` in the gap histogram and
corrupt gap detection. The pre-existing inline comment ("This can happen for
vertices that can't reach sink") was also wrong: it cannot happen for an
*active* vertex.

### Disposition

Routed here per the /loop rule (non-reproducible -> tried-and-rejected citing
the finding ID). Per the finding's explicit recommendation ("add a debug_assert
or comment") and the X16/O4/O5/O6/O9 sub-fix precedent, the safe in-code
improvement is applied: a `debug_assert_ne!(new_height, usize::MAX, ...)` after
the relabel scan pins the active-vertex invariant (it fires in debug builds if
the "unreachable" branch is ever about to be taken), and the misleading inline
comment is replaced with one stating the branch is an unreachable defensive
fallback and noting the latent histogram-corruption it would cause. The
`height_count` decrement is deliberately *not* added: the branch is unreachable,
so adding bookkeeping there would be untestable dead-code behaviour, which the
tests-first lifecycle forbids. The existing flow test suite (`flow.rs:333+`,
~15 tests driving `push_relabel`) is the guard — a green run proves the
debug_assert never fires.

Evidence: `flow.rs:254` (active-vertex guard), `:265-269` (relabel scan),
`:271-279` (stranded branch, no `height_count` decrement), `:286-287` (the
decrement the normal path performs). `cargo test -p feral-kahip` green with the
debug_assert in place. Journal: dev/journal/2026-06-10-01.org.
## 2026-06-11 — O20: thrice-copied ND driver scaffolding (finding O20, repo-review-2026-06-09.md)

### Finding (verbatim)

> **O20** Thrice-copied ND driver scaffolding
> (recurse/connected_components/extract_by_*/build_induced/
> graph_to_csc_pattern/invert_iperm) across the metis/scotch/kahip
> `node_nd.rs`, already drifted: kahip sorts neighbors before the
> diagonal splice, metis/scotch rely on induced sortedness; metis
> validates the inverse perm inline, scotch/kahip have a duplicate-
> position check metis lacks. Consolidate into feral-ordering-core.
> medium-low/certain.

### Why the consolidation itself is not reproducible / not done here

The core recommendation — move six near-identical helper functions out of the
three `node_nd.rs` files into `feral-ordering-core` — is a structural refactor,
not a wrong-answer defect. There is no input that produces an incorrect result
today, so there is no RED state to write. A blind multi-crate move also risks
introducing the very drift it removes (the three copies have already diverged
in small ways), and the per-finding atomic-commit discipline plus
"correctness before performance" weigh against a large behaviour-touching
refactor of three working drivers inside one /loop iteration. The consolidation
is therefore deferred and recorded here.

### The two named drift points

**Drift A — sortedness (benign, documented only).** metis `graph_to_csc_pattern`
(`node_nd.rs:254-279`) splices the diagonal in at the first neighbour `> v`,
relying on `adjncy[lo..hi]` being sorted; kahip (`:219-228`) calls
`neighbors.sort_unstable()` defensively. This is benign today: `build_induced`
preserves sorted adjacency for metis/scotch, and `CscPattern::new` rejects
unsorted rows (O3), so any violation fails loudly rather than silently. No code
change — adding a defensive sort to metis/scotch would be untestable dead
defence.

**Drift B — duplicate-position check (harmonized, the testable slice).** metis
inverted `iperm → perm` *inline* in `nd_order` (old lines 54-61): it
initialised `perm = vec![0; n]`, range-checked `new_pos`, and wrote
`perm[new_pos] = old` — with **no** duplicate-position check. scotch
(`invert_iperm`, `:433-450`) and kahip (`:328-345`) initialise `vec![-1; n]`
and reject `perm[np] >= 0` ("… produced duplicate position"). So metis would
silently emit a non-bijection if upstream ever produced a duplicate position
(and its `0`-init makes the corruption undetectable post-hoc). This was
harmonized: metis now has an `invert_iperm` mirroring its siblings exactly
(crate name aside), with the range and duplicate-position checks. It is
behaviour-neutral on every reachable input (a valid bijection writes each
position once) and is a strict step toward the eventual consolidation.

### Disposition

Routed here per the /loop rule (non-reproducible core -> tried-and-rejected
citing the finding ID). Per the X16/O4/O5/O6/O9 sub-fix precedent, the safe,
testable slice is applied: the metis `invert_iperm` extraction + duplicate
check, guarded by a new RED->GREEN unit test
(`invert_iperm_rejects_duplicate_positions`). Drift A is documented as benign.
The cross-crate consolidation into `feral-ordering-core` remains open.

Evidence: metis `node_nd.rs:54-61` (old inline inversion, no dup check),
scotch `:433-450` / kahip `:328-345` (the dup check metis lacked), metis
`:254-279` (sortedness reliance), kahip `:219-228` (defensive sort).
`cargo test -p feral-metis` green with the new test. Journal:
dev/journal/2026-06-10-01.org.
## 2026-06-11 — O21: AMD/AMF inner-loop `wf = 0` sentinel ambiguity (finding O21, repo-review-2026-06-09.md)

### Finding (verbatim)

> **O21** AMD/AMF inner-loop duplication (documented decision, ~600 LoC) is
> already asymmetric: AMF writes `wf = 0` for dead elements, indistinguishable
> from the first-touch sentinel 0, so a live element with true contribution 0 is
> recomputed every iteration (`algo.rs:968`). Harmless; illustrates the drift
> risk. low/certain.

### Code

`crates/feral-ordering-core/src/quotient_graph/algo.rs`, `finalize_step_amf`.
Pass-1 (line 955) resets `ws.wf[e] = 0` on first touch — the lazy-cache "not yet
computed this iteration" sentinel. Pass-2 element sub-pass (aggressive 983-988,
non-aggressive 1015-1018) computes `if ws.wf[e] == 0 { ws.wf[e] =
amf_wf_surface(dext, degree[e]) }` then `wf4 += ws.wf[e]`, where
`amf_wf_surface(dext, degree) = dext*(2*degree - dext - 1)` (line 664-666).

### Why the core finding is not reproducible as a defect

A live element with `dext > 0` has a genuine surface of 0 exactly when
`dext == 2*degree(e) - 1` (e.g. `dext = 1, degree = 1` → `1*(2-1-1) = 0`).
`amf_wf_surface` then returns 0, so `wf[e]` stays 0. The next member that
touches `e` in the same Pass-2 sees `wf[e] == 0`, treats it as uncached, and
recomputes `amf_wf_surface` — obtaining the **same** 0. `wf4 += 0` is unchanged.
`amf_wf_surface` is a pure function of `(dext, degree[e])`, both stable across a
single Pass-2, so the recompute is deterministically equal to the cached value.

Therefore the accumulated triple `(deg, wf3, wf4)`, the quantized RMF score, and
the resulting elimination permutation are **identical** whether the value is
cached or recomputed. No input produces a wrong answer, and no observable
(output / permutation / flop-accounted result) distinguishes "cached once" from
"recomputed per touch". The only effect is a handful of extra integer multiplies
on the rare `dext == 2*deg-1` element. This is a missed micro-optimization, not a
correctness defect — the same disposition as the O9 / O11 / O17 performance
findings: there is no RED state to write, so it is non-reproducible and routed
here per the /loop rule (citing the finding ID).

### Why a distinguishing sentinel was rejected (not applied)

The obvious "fix" — use a sentinel such as `-1` for "uncached" so a genuine 0 is
distinguishable — was rejected. The same `wf` array is reused for variable
scores: the supervariable merge takes `wf[i] = max(wf[i], wf[j])` (doc item 5)
and re-insertion quantizes `wf[i]` into the RMF bucket (item 6). A `-1` sentinel
would have to be guaranteed never to leak into either the `max` or the bucket
quantization for any index, across both the AMD and AMF code paths. That adds
real correctness risk to the working ordering to save a few integer ops on a rare
element. "Correctness before performance, always" (CLAUDE.md constraint) settles
it against the change.

The broader finding framing — that the ~600 LoC of duplicated AMD/AMF inner-loop
code is itself a drift hazard — is a structural-refactor observation (the same
class as O20's cross-crate consolidation), not a defect with a reproducing test,
and is likewise deferred rather than undertaken inside a single /loop iteration.

### Disposition

Routed here per the /loop rule (non-reproducible core → tried-and-rejected citing
the finding ID). Per the X16 / O4 / O5 / O6 / O9 / O17 sub-fix precedent, the
safe behaviour-neutral slice is applied: a comment block at the Pass-1 sentinel
reset documenting that `0` is deliberately overloaded as both "uncached" and a
possible genuine surface value, that the resulting recompute is benign (same
value), and why a distinguishing sentinel was rejected; plus a one-line pointer
at each of the two Pass-2 cache-check sites. No behaviour change.

Evidence: `algo.rs:955` (first-touch `wf[e] = 0` sentinel), `:983-988` /
`:1015-1018` (the `if wf[e] == 0` recompute sites), `:664-666`
(`amf_wf_surface`, zero at `dext == 2*deg-1`). `cargo test -p
feral-ordering-core` green (comment-only, proves no behaviour change). Journal:
dev/journal/2026-06-10-01.org.

## 2026-06-18 — Step 2 (dense bump workspace) rejected for the McCormick LP regime — bumps are wide but ultra-sparse

**Context.** discopt#229 / `dev/research/bump-elimination-speedup-2026-06-18.md`
proposed two speedups for `SparseLu::eliminate_bump`: Step 1 (sub-diagonal pivot
index, shipped 902e5d7, 15.8× end-to-end) and Step 2 (scatter the bump into a
dense workspace and do contiguous SAXPY elimination). The issue framed Step 2 as
"replace sparse-merge elimination with contiguous SAXPY work," premised on a
dense-spike bump.

**Why rejected.** Measured the actual bump structure on the real casctanks update
trace (1702 updates) via temporary `FERAL_BUMP_STATS` instrumentation in
`eliminate_bump` (now reverted):

- avg bump width = 731 (max 2157 ≈ m) — wide, as the issue reported;
- but for wide bumps (width > 500, 924 of 1702 updates) the `[r,h]×[r,h]` block
  **density = 0.23%** (`block_nnz / width²`);
- only **~23.9 row_subs (Axpy ops) per update**, total merge work ~233 entries.

So a "wide" bump is a far-reaching but nearly-empty spike — ~1230 scattered
entries over a 731×731 region, with only ~24 columns actually needing
elimination. A dense workspace would scatter that into a 731×731 = 534k-cell
dense array and touch all of it, vs the current ~24 sparse row_subs — **~100–1000×
more work.** Step 2's premise (dense-spike bump, contiguous SAXPY) does not hold
for this workload; implementing it would **regress** casctanks, not speed it up.

This also explains Step 1's 15.8×: the old O(bump²) **scan** probed ~731²/2 ≈ 267k
cells per update to find the ~24 that needed work. Removing the scan (Step 1) was
the correct and sufficient fix for the sparse-wide-bump regime; the residual
elimination work is already near-minimal.

**Not generally rejected.** A dense path could still help a genuinely *dense*-spike
basis (the journal's 2026-06-08 tridiagonal/`L⁻¹`-dense worst case). If such a
workload appears, Step 2 should be width-AND-density gated (dense path only when
block density exceeds a threshold), never width-only. For the McCormick LP regime
that motivated discopt#229, Step 1 stands alone.

**Evidence.** `FERAL_BUMP_STATS` aggregate over
`FERAL_LU_TRACE=.../casctanks_trace.txt` (full trace): avg_width=731.3,
max_width=2157, avg_density=0.1346 (all bumps) / 0.0023 (width>500),
avg_axpy=23.9, avg_merge_work=233.4. Step 1 end-to-end: casctanks LP solve
82.4 s → 5.2 s debug (15.8×), optimum −167.751 unchanged. Journal:
dev/journal/2026-06-18-01.org.

## 2026-06-30 — Auto-preprocess "smaller-fill-wins" race (issue #91)

**Tried.** For `OrderingPreprocess::Auto`, when the predicate recommends
`LdltCompress`, race None vs LdltCompress and keep whichever has the smaller
`factor_nnz_estimate` (ties → keep LdltCompress).

**Rejected.** Regressed inertia on near-singular corpus KKTs. `LdltCompress`'s
MC64-matched 2×2 pivots give the oracle-correct inertia on twirism1 and sawpath
even though that ordering costs slightly more fill (twirism1 +15 %: 26683 →
30782 est). Smaller-fill-wins flipped both to the leaner `None` ordering, which
misclassified near-zero pivots: twirism1 (432,313,0) → (434,311,0), sawpath
(789,670,116) → (789,671,115). `tests/issue65_mc64_fallback.rs` failed:
`explicit_infnorm_is_respected_no_fallback` and
`twirism1_iter0_auto_stays_infnorm_no_spurious_fallback`.

**Symptom.** `assertion left == right failed ... got Inertia { positive: 789,
negative: 671, zero: 115 }` (want 789,670,116); twirism1 got (434,311,0) (want
432,313,0).

**Replaced by.** Keep `LdltCompress` unless it inflates fill past a 2×
catastrophe ceiling (`PREPROCESS_FILL_INFLATION_LIMIT`). Preserves the
inertia benefit on the +15 % cases, still catches qap15's 6.3× misfire.
The numerical benefit of `LdltCompress` is not visible in symbolic fill, so a
fill-only race is the wrong criterion; only a *runaway* fill increase signals a
predicate misfire.

## 2026-06-30 — B-1a panel packing for the dense trailing update (issue #91)

**Tried.** Pack the eliminated panel (columns `k..k+n_elim` of `head`, column
stride `nrow`) into a contiguous `[n_elim × span]` buffer once per
`apply_schur_panel_range`, then feed the *same* strided kernels from it (tight
stride `span`), gated to large fronts (`nrow>128`). Byte-exact by construction.

**Rejected — net slowdown on qap15.** Byte-exact parity held (blocked_ldlt
21/21, inertia/nnz_L unchanged) but it was *slower* everywhere: sequential
factor loop 1747 → 1976 ms (+13%), the 2955×2955 root front 736 → 818 ms
(+11%), parallel default 771 → 945 ms (+22%).

**Why.** The root's early panels have `span ≈ nrow`, so packing does not reduce
the K-stride — it just adds an alloc + copy. More fundamentally, profiling of
the root shows it is **DST-bandwidth-bound, not panel-bound**: the ~70 MB
trailing block (2955×2955 f64) is streamed ~46 times (once per rank-64 panel),
which dwarfs the ~1.5 MB panel (already L2-resident). Packing the *source* panel
optimizes the wrong operand.

**Implication for the plan.** The effective lever is reducing DST traffic:
cache-blocked / recursive dense-root factorization (Phase C — reuse a
cache-sized trailing tile across many panels) or a larger panel width (more
flops per DST stream). A source-side pack (B-1a) is off the table. FMA remains a
+23% option but is a reproducibility-policy change (kept opt-in), not a
bit-exact win.

## 2026-07-01 — UPDATE: packed micro-kernel succeeds where B-1a source-pack failed (issue #99)

The 2026-06-30 "B-1a panel packing" entry above rejected source-panel packing as a
net slowdown and concluded the root front is DST-bandwidth-bound. That conclusion
was **specific to the variant tried** — packing the source into a tighter stride
but *feeding the same strided kernels*, which keep the per-`q` `as_simd` + strided
access. It does **not** generalize to a proper packed micro-kernel.

A different design — pack the panel into `q`-contiguous MR=8×NR=4 micro-panels and
run a register-tiled kernel with a **contiguous inner `q`-loop**
(`apply_schur_panel_range_packed`) — is **22–26× faster in isolation and
byte-exact** (`examples/bench_schur_micro`), and gives 8–10× on real dense fronts.
So the bottleneck was strided-`q` cache latency, not DST bandwidth, on this
hardware. Not a rejection — a correction of scope. See
`dev/research/issue-99-dense-front-fma-gate.md` UPDATE 3 and `dev/decisions.md`
2026-07-01 (packed BLAS-3). The B-1a *source-into-strided-kernel* variant remains
rejected; the packed micro-kernel is the shipped design.

## 2026-07-10 — BG rescue-after-TinyPivot for the FT update (issue #112)

**Tried.** Implemented issue #112's requested design literally: run the plain
Forrest–Tomlin bump sweep first; on `TinyPivot` (the exact-`0.0` class), roll
back row `r` and re-run the sweep with Bartels–Golub row interchanges
(`FtOp::Swap` physical row-content swaps; retry gated by
`update_pivot_search`, default true), expecting the re-selected pivot order
to commit where the fixed order cancelled.

**Symptoms / why it cannot work.** Every candidate regression matrix had the
rescue fail exactly like FT (computed `0.0` or sub-ztol), e.g. the m=4
absorption basis: rescue's final diag `1.39e-17` = `λ·2⁻³⁵` with `λ ~ 2⁻²⁰`
from the dominated-diagonal swaps. Root cause is a proved identity, not a
bug: any interchange variant's working row is **exactly proportional**
(`W'_k = λ_k·W_k`) to the fixed-order one — swaps only rescale the carried
row (`λ` resets to `−piv/vrc` per swap, `|λ| ≤ 1` under domination swaps) —
so (a) the true final pivot is `λ·t_FT ≤ t_FT` (determinant identity
`∏pivots·final = det(bump)`), (b) skip patterns coincide (proportional
zeros), and (c) FP absorption is scale-invariant (ulp is relative), so a
rescue re-computation absorbs the same bit. A pivot the fixed order computed
as exactly `0.0` by absorption is unrecoverable by ANY within-bump
row-operation reordering; retry-commit values would be noise (signal/noise
= `t_FT/(ε·I) ≤ 1` on every path). Numeric evidence: float + exact-Fraction
sweep replays across four hand constructions (journal 2026-07-10-01,
research note §UPDATE).

Also rejected en route: classic **Kahan** compensation for the sweep
accumulator (its `y = v − c` pre-subtraction re-absorbs the compensation
into the next 2²⁰-scale addend — computed `0.0` again; verified
numerically); the **Neumaier** two-sum variant works and shipped. And three
regression-matrix constructions whose base or replacement was numerically
singular for every path (±1 cascade to 2³⁴: `σ_min(B') = 1.5e-16`; diag-4
cascade: rescue-true `4.5e-13 <` ztol; spike-poison m=6: fresh LU burns the
4e6 spike entry and deflates its tail pivot to 0) — any single-shot
absorption reproducer necessarily has `σ_min(B') ⪅ δ·∏retained`, so the
"fresh factor succeeds" oracle is unsatisfiable without a multi-update
imbalance history.

**Shipped instead.** Always-on Neumaier-compensated scatter (recovers the
true pivot bit-for-bit on the regression basis) + `update_pivot_search` as an
always-on opt-in trajectory variant (bounded multipliers across chains),
default false. See `dev/research/issue-112-bg-update.md` §UPDATE and
`dev/decisions.md` 2026-07-10.

## 2026-08-09 — Stage 3: explicit pulp SIMD for the eager rank-1 pivot update (reverted)

**What was tried.** `do_1x1_pivot`'s fused scale + rank-1 trailing
update + argmax (the small-front eager path's hot loop, duplicated in
the static-floor branch) was refactored into a shared helper and given
an explicit-SIMD route: one `schur_kernel::rank1_trailing_nofma` pulp
dispatch per pivot step covering all trailing columns (per-element
`mul → sub`, byte-exact — proven by an eager-path A/B parity test and
the full 83-suite run incl. golden digests), with a work gate at
1024 multiply-subtracts and a scalar argmax re-scan of column k+1.
Motivation: the feral/MA57 small-bucket loss (4.23x geomean),
panel-frag attribution (scal% up to 89.8% on sawpath), and the
pounce#552 chain-KKT report.

**Symptoms / numbers.** Perf was FLAT everywhere:
- Warm fixture medians (3-run, sequential, x86_64 AVX2 container)
  unchanged vs Stage-2 within noise: AVION2 34→34, SWOPF 152→153,
  HYDCAR20 ~210→206-213, HAHN1 ~742→754-762, CRESC100 ~606→589-621,
  chain1200 ~867→863-878, twirism1 ~2100→2138-2195, sawpath ~698→671-726.
- Direct eager-driver A/B (gate default vs forced-off) on square
  fronts where the gate demonstrably fires: n=512 11.21 vs 11.23 ms,
  n=200 0.88 vs 0.89 ms, n=96 0.11 vs 0.11 ms. Zero effect at any size.

**Why it failed.** Unlike the packed tile walk (whose guarded loop
structure defeated autovectorization — Stage 1), the eager path's
plain `for i in j..n { a[j*n+i] -= a[k*n+i]*alpha }` loops are
textbook-autovectorizable, and the eager path's remaining time is
pivot search + memory traffic, not multiply-subtract throughput.
Explicit lanes duplicated what LLVM already did. This matches the
2026-05-16 finding (pulp == scalar == manual unroll at lengths 3..128)
at the whole-front scale.

**What was kept.** The de-duplication refactor (shared scalar
`rank1_scale_update_argmax`, byte-identical, golden digests unchanged)
stays; the pulp kernel, its gate/env var, the dedicated parity test,
and the A/B example were removed.

**Lesson.** The small-front/MA57 gap is NOT lane width in the eager
update. Remaining suspects, in evidence order: per-front fixed
overhead (assembly/scatter/build-row, 8.8-14.8% on the small
fixtures), pivot-search scans, `scalar_pivot_step` in blocked fronts,
and the delayed-pivot cascade (per-factor-cost-cluster mechanism A).
Any retry of eager-path SIMD must first show a front-level profile
where the update loops are >30% of eager time AND not already
vectorized in the disassembly.

## 2026-08-09 — Warm-starting the ∞-norm scaling iteration across factorizations

**What was tried.** The PR #150 review ranked "scaling warm-start" as the
largest remaining line item: the warm prologue is 15-39% of
factorization, ∞-norm equilibration is 63-81% of that, and it is the one
prologue component that does NOT warm across calls (permute collapses
4443->539 us on clnlbeam; scaling stays flat at 3909 us). Hypothesis:
seed `compute_infnorm`'s iteration with the previous factorization's `d`
instead of `1.0`, since IPM values drift smoothly, and cut the iteration
count.

**Symptoms / numbers.** Zero iteration reduction on every fixture
measured (`examples/probe_kr_warmstart.rs`, ±5% value perturbation
standing in for one IPM step):

| fixture | cold iters | warm iters |
|---|---:|---:|
| clnlbeam-like n=100000 | 2 | 2 |
| grid250 n=62500 | 2 | 2 |
| chain12000 | 10 | 10 |
| sparseqpL n=105000 | 10 | 10 |
| HYDCAR20 | 10 | 10 |
| twirism1_kkt | 10 | 10 |

**Why it failed.** Two regimes, neither reachable by warm-starting.
Matrices either converge in 2 iterations already (nothing to save), or
they hit `max_iter = 10` WITHOUT converging. The per-iteration trace
shows clean linear convergence at ratio ~1/2 — the known rate for
Ruiz-style ∞-norm equilibration (`d <- d/sqrt(row_max)`), which is what
`compute_infnorm` implements despite the "Knight-Ruiz" label. At the cap
`max_dev` is still ~1.4e-2 against `tol = 1e-8`; reaching that tolerance
needs ~30 iterations. So the tolerance is unreachable by construction and
the loop always runs the full 10 passes — a fixed cost that no starting
point can reduce. The 5e-2 cold-vs-warm spread on the same matrix
confirms neither run is near a fixed point.

**Correction to the code's own comment.** `compute_infnorm` says "Most
matrices converge in 2-4 iterations; a few pathological ones need all
10." On this fixture set 4 of 6 hit the cap, and they do not *need* 10 —
they are *truncated* at 10 and never converge.

**What was kept.** `examples/probe_kr_warmstart.rs` as the reproducer and
the convergence-trace tool. No production code changed.

**Lesson.** "Component X doesn't warm across calls" is not by itself
evidence that a cache would help — the cost may be iteration-count-bound
rather than starting-point-bound. Measure the iteration count before
building the cache. Full analysis and the ranked alternatives:
dev/research/scaling-warm-start-2026-08-09.md.

---

## 2026-08-09 — Amalgamation retune after 0.15.0: `nemin` lower, and a cost-model merge guard

**What.** Post-release queue item 2 (`dev/plans/release-0.15.0-checklist.md`
§4.2), motivated by the PR #150 review: 90% of clnlbeam's supernodes are
≤8 columns and they are 35.9% of its loop. Two levers measured, both
rejected. Harness:
`crates/feral-diagnostics/src/bin/diag_nemin_post_simd.rs` (paired
alternating A/B per decisions.md 2026-08-09, `min_us`, sign test), 61
parity fixtures + 4 structured KKTs, x86_64 AVX2.

**Lever 1 — retune `nemin`. Rejected on its own pre-registered
criterion.** The criterion, fixed before the first run: ships only if
≥5% geomean with ≥8/10 sign test on ≥2 fixture classes and no fixture
regressing >2%.

Geomean vs `nemin=16`, time / nnz — 61 parity, then 4 structured:

| arm | 1 | 4 | 8 | 32 | 64 |
|---|---|---|---|---|---|
| parity | 1.21/0.67 | 1.02/0.83 | 0.986/0.89 | 1.02/1.19 | 1.15/1.65 |
| structured | 1.33/0.37 | 0.99/0.51 | 0.925/0.68 | 1.13/1.60 | 1.53/2.74 |

`nemin=8` is genuinely better on both axes on the structured set, but
the parity geomean is 1.4% (not 5%) and individual fixtures regress well
past 2%: CERI651A_0000 1.163 (2/15 wins, 178→207 µs — an effect, not
noise), DEGENLPB_0046 1.119, BQPGASIM_0012 1.100, HS85_0176 1.056.

This also **re-confirms the 2026-05-16 rejection** (issue #10 lever 5)
after the kernel rewrite that was its only plausible escape hatch: every
arm above 16 loses on time and inflates fill on every class. The faster
trailing update did not buy back the fill. The queue item's stated
direction — raise `nemin` to fuse small supernodes — is dead twice over.

**Lever 2 — cost-model merge guard. Implemented, measured, rejected on
accuracy.** The size rule (`child_ncol < nemin && parent_ncol < nemin`)
never asks what a merge costs. Front height is
`col_counts[first_col].max(ncol)`, so past the natural row count every
extra column is a triangle of pure fill; clnlbeam's `col_counts=2` chain
links inflate 23× at `nemin=16` and the rule cannot see it. Added
`SupernodeParams::merge_flop_budget: Option<u128>` — merge only if
`Δflops ≤ budget`, with
`flops(ncol,nrow) = Σ_{k<ncol}(nrow-k)² = S(nrow)-S(nrow-ncol)`.

It works on the axes it was designed for. Clean interior optimum at
budget 30–60 (structured geomean 0.945/0.549 and 0.935/0.619), no speed
regressions on parity (0.980–0.997 time, 0.866–0.940 nnz), and it
dominates `nemin` where it counts: sparseqp holds 0.493× fill at the
same 0.947× time that `nemin=8` needed 0.676× fill for.

**Then the residuals.** Single unrefined solve, `b=1`, so ∞-norm
residual is also relative. Degradations >10× at budgets 15–125:

| fixture | default | guarded |
|---|---|---|
| HATFLDG_0003..0006 | 7.1e-15 | up to 7.4e-08 |
| VESUVIOU_0030 | 1.9e-06 | 5.4e-03 |
| MEYER3NE_0220 | 2.7e-08 | 8.0e-07 |

Seven digits on HATFLDG. Inertia unchanged everywhere, and on
HATFLDG/VESUVIOU **the fill is identical to the default** — so this is
not a fill effect. Thinner supernodes shrink Bunch-Kaufman's pivot
candidate pool inside each front and it takes worse pivots on
ill-conditioned matrices. `nemin` has the same defect (VESUVIOU 2789× at
`nemin=8`, MEYER3NE 83× at `nemin=4`), which is what makes it a property
of the direction rather than of this rule.

**Why rejected.** "Correctness before performance, always" is a hard
constraint. 2–7% of factor time and 11–45% of fill does not buy seven
digits of residual. Neither my pre-registered criterion nor the queue
item thought to check the axis that decided it — recorded here because
the next person to have this idea will not think to check it either.

The knob stays in-tree defaulting to `None` (bit-identical default path)
as the reproduction apparatus, with the accuracy result in its doc
comment. Research note:
`dev/research/amalgamation-cost-model-2026-08-09.md`.

**Also redirects the target.** pounce#552's re-measurement against a
released 0.15.0 (comment 5232409020) shows clnlbeam more than halved
(8.05× → 3.54× vs MA57) and **no longer the worst case** — `dtoc1nd` is,
at 3.77×, and it is a dense-front matrix (nnz/dim 23.0, fronts of 33–64
columns). Amalgamation is a chain-KKT lever aimed at a problem that has
largely receded.

## 2026-08-09 — efficiency cores as the explanation for the post-0.14.0 chain regression

**Rejected.** The proxy note `chain-kkt-ma57-gap-2026-08-09.md`
hypothesized that on Apple silicon rayon treats efficiency cores as
equivalent to performance cores, so a coarse task from #150's
task-per-subtree coarsening landing on an E-core stalls the whole
factorization, where 0.14.0's per-supernode spawning let work-stealing
rebalance. It predicted the regression "shrinks or inverts at
`RAYON_NUM_THREADS=4`".

Swept `RAYON_NUM_THREADS` = 1, 2, 4, 8, 10 against the default on six
real chain KKTs, 15 paired runs each, on an M4 Pro (10P + 4E — *more*
efficiency cores than the 4P+4E M2 the hypothesis came from, so the
predicted effect should be larger). Every ratio vs the default landed
within 6% of 1.0; the only significant one was `marine_1600` at t1
(0.961, 0/15, p = 0.0001), in the wrong direction to support the
hypothesis. `steering_12800` at one thread: 1.003 (9/15, p = 0.6072).
`dtoc1nd`, the matrix that actually regressed, at t4: 1.005 (7/15,
p = 1.0000).

The knob was verified live, not assumed: feral reads the global rayon
pool (`rayon::current_num_threads()`, `src/numeric/factorize.rs:3262`),
and the same variable moved the proxy matrices by up to 65%.

Consequence worth carrying forward: single-threaded main matches
all-threads main on every one of these matrices, so **#150's 1.20x to
2.05x gains on the large chains are not parallelism gains**. Do not
build on the assumption that they are.

Full data: `dev/research/chain-kkt-corpus-2026-08-09.md`, Result 3.

## 2026-08-09 — a tighter search bound as the MC64 Hungarian lever

**Rejected: the lever does not exist.** `mc64-condition1-cost-share-2026-08-09.md`
closed by recommending a Hungarian search bound and described it as
"same-output-less-work, needs no gate and no residual argument". **That
characterization is wrong.** Code inspection, before any code was written,
found two independent reasons:

1. The classic shortest-augmenting-path bound is **already implemented**.
   `csp` — the cost of the best augmenting path found so far — prunes the
   root scan (`src/scaling/hungarian.rs:541`), terminates the main loop
   (`:570`), and prunes the inner column scan (`:591`). There is no missing
   standard bound to add.
2. Any *further* truncation ends a search before its shortest augmenting path
   is proven, giving a suboptimal matching and therefore a different scaling
   vector. That is a numerics change requiring a corpus inertia/residual study
   and human approval — not a free win.

See also the pre-existing entry 2026-06-06-03 (line ~2313), which proves no
*per-column reduced-cost* bound can ever prune the inner scan (`vj + lb_tight
= dq0`, and `q0` was popped only because `dq0 < csp`) and instructs future
sessions not to retry it. That directive stands.

What the measurement pointed at instead was memory layout, not fewer scans:
per-edge-scan cost differs 4.5x between corpus families on identical code
(pinene_3200_0006 2.94 ns/scan, 87 rows touched per search, L1-resident;
nql180_0002 14.4 ns/scan, 53,411 rows touched per search, far past L1). The
inline-key heap that followed is bit-identical and won 4-5% on nql180.

Full data: `dev/research/mc64-hungarian-search-bound-2026-08-09.md`.

## 2026-08-09 — `build_cost_graph` as an MC64 optimization target

**Rejected on measurement.** Timed at 8-12 ms per iterate. That is ~20% of
pinene's *cheapest* iterate but **0.4%** of nql180's — and nql180 is where the
MC64 time actually is. Optimizing it cannot move the corpus. The instrumented
timer was reverted and is not in any commit.

## 2026-08-09 — array fusion projected from a microbenchmark

**Not rejected, but the projection was wrong and is recorded so it is not
reused.** A standalone microbenchmark of split-array vs fused-record reads
predicted **1.68-1.9x**. The real end-to-end win from the inline-key heap was
**4-5%**, because the split reads are only ~2.3 ns of nql180's ~13 ns/scan.
Microbenchmarks of one memory access pattern do not predict a loop that also
does heap sifting and comparison work; scale by the measured share of the loop
before believing them.

## 2026-08-13 — gating the dense-bump route on `bump_lo > 0 || bump_hi < m` alone

**Rejected on a broken test.** Reviewing PR #160, the dense-bump route was found
to fire on symbolics that never triangularized (`natural`, `with_order`,
`analyze_amd_only` all report `(0, m)`), packing a whole sparse basis into an
`m²` f64 buffer. The first fix gated on "the bump is a proper sub-block":

```rust
let peeled = bump_lo > 0 || bump_hi < m;
```

This broke the PR's own `(0, 16, 0)` differential case in
`tests/lu_dense_bump.rs` — a genuinely dense 16x16 basis with no triangular
border, which peels to nothing and so reports `(0, m)`, but for which the dense
kernel is exactly right. Failure was
`the dense arm fell back to sparse (seed 5) - test is vacuous`.

The distinction the index test cannot make is *why* the bump is the whole basis:
a default from a constructor that never looked, or a measurement from a peel
that found nothing. Those want opposite answers. Replaced with a
`SparseLuSymbolic::triangularized` provenance flag plus a `bump_dim <=
dense_threshold` allowance for the unpeelable case.

Note the provenance flag *alone* was also insufficient — `analyze` on a
tridiagonal m=3000 peels nothing, sets `triangularized = true`, and hit the same
cliff at 297 ms vs 1.5 ms sparse. Both guards are load-bearing.
## 2026-08-13 — sparse permuted marshalling around the LU triangular solves

**Rejected on measurement (issue #161B).** Having made the gather-form
triangular sweeps reach-limited, a hyper-sparse `ftran` on `m = 4000` still cost
~20 us for a solution with a p50 of 3 nonzeros, so I looked for the next `O(m)`
term. The hypothesis was the four permuted gather/scatters around the core solve
(`out[k] = rhs[perm[k]]`, `rhs[qcol[k]] = s[k]`, and their `btran` mirrors):
each reads or writes through a permutation, so all `m` accesses are random. The
replacement was a `fill(0.0)` plus a *sequential* scan plus one random access
per nonzero, gated on the same switch as the reach route so the feature-off path
stayed byte-identical.

It does nothing. Interleaved A/B on the same basis, toggling only that gate:

```
                     ftran mean   btran mean   dense-rhs fallback
  sparse marshal      33.6 us      31.6 us        0.90x
  dense marshal       31.3 us      33.0 us        0.93x
```

The two arms straddle each other (ftran favours dense marshalling, btran favours
sparse) — that is noise, not a signal, and the dense-rhs fallback is if anything
slightly worse with it.

**Why the hypothesis was wrong.** The permuted access is random but it is random
*within a 32 KB buffer*, which sits in L1/L2 — so it was never paying the cache
misses the reasoning assumed. What the phase probe actually showed is that the
residual cost is spread evenly across all ~6 `O(m)` linear passes
(`ftran_partial` alone, which is just the `P`-gather plus `lsolve`, was 10.3 of
the 22.1 us), at roughly 2-3 us per pass on this machine. There is no single
term left to remove: getting below the `O(m)` floor needs a **sparse-rhs entry
point**, not a cheaper way to walk a dense one.

The code was reverted. `dev/research/hyper-sparse-solves-2026-08-13.md` records
the floor and names the API change that would lift it.

## 2026-08-13 — caching `SparseLuSymbolic` across refactorizations in a simplex

**Rejected on measurement.** While attributing discopt issue #1008, the analysis
phase turned out to dominate: on QPLIB_3775, `LuSymbolic` was **1048.5 ms across
64 factorizations** against `LuNumeric` **184.6 ms** — 5.7x the numeric
factorization spent choosing a column order. Since `SparseLu::factor` only
validates `symbolic.m == a.m`, a stale ordering is *legal*, so the obvious fix
was to compute the ordering once and reuse the handle on every refactorization.

Probed in discopt behind `DISCOPT_LU_SYM_REUSE` (a `sym_cache: Option<SparseLuSymbolic>`
on `FeralLU`, never merged). On QPLIB_3775 with `analyze_triangularized`:

| arm | factorizations | LuNumeric | wall |
|---|---|---|---|
| tri=1, reuse=0 | 64 | 184.6 ms | **1.193 s** |
| tri=1, reuse=1 | 1112 | 18381 ms | **137.835 s** |

**115x slower.** tri=0 reuse=1 did not finish inside a 300 s timeout. A simplex
basis is not structurally stable across 64 pivots: the stale ordering explodes
fill, the fill blows the numeric factorization, and the resulting instability
triggers a refactorization storm (64 → 1112) that feeds back on itself.

The conclusion is not "reuse harder" — it is that the ordering **must** be
recomputed on every refactorization, and therefore must be cheap. That is what
makes the `analyze` vs `analyze_triangularized` cost (4.3-12.4x, measured
standalone) a first-order effect rather than an amortizable one.
