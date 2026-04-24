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
