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
