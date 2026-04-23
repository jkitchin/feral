# Rook Pivoting as Rescue Path — Research Note

**Status:** Pre-implementation research note (mandatory per CLAUDE.md)
**Date:** 2026-04-23
**Related plan:** `dev/plans/phase-2.4.3-rook-rescue.md`
**Key references:** citep:bunch1977stable (baseline BK-partial),
citep:bunch1971direct (Bunch-Parlett complete pivoting),
citep:duff2020aptp (APTP pivoting analysis),
citep:duff2005symmetric (symmetric pivoting for indefinite systems),
citep:duffreid1996zeros (rook original),
citep:ashcraft1998accurate (termination analysis),
citep:duff2004ma57 (MA57 reference implementation),
citep:foster1997rook (O(n^{0.5 log n}) growth bound).

## 1. Scope

This note covers the algorithmic foundation for rook pivoting as a
rescue path — not as a standalone factorization strategy. The design
decision (rescue vs top-level) is covered in the plan; this note covers
the numerical algorithm itself, why it has the growth properties it
does, how the "ghost-row" adaptation works for the frontal case, and
what termination safeguards are needed.

## 2. Baseline: BK-Partial Growth

Bunch-Kaufman 1977 partial pivoting searches a single column to decide
between a 1×1 pivot at `(k, k)` and a 2×2 pivot at `{k, r}` where `r`
is the off-column row-max. The decision rule uses `gamma0 = max_{i>k}
|a[i, k]|`:

- If `|a[k, k]| >= alpha * gamma0`, take 1×1 at k.
- Else search row `r` for `gamma_r = max_{j != r} |a[r, j]|`.
  - If `|a[k, k]| * gamma_r >= alpha * gamma0²`, take 1×1 at k.
  - If `|a[r, r]| >= alpha * gamma_r`, swap r into k, take 1×1.
  - Else take 2×2 at `{k, r}`.

The `alpha = (1 + √17) / 8 ≈ 0.6404` constant is chosen so element
growth per step is bounded.

**Growth bound.** BK-partial has element growth of at most
`(1 + 1/alpha)^{n-1} ≈ 2.57^{n-1}` (Bunch 1971). In practice growth is
nearly always modest, but pathological KKT systems — especially those
with ill-scaled equality constraint blocks — can exhibit near-worst-
case growth. On those matrices, the column-relative threshold test in
`try_reject_1x1_frontal` correctly flags the bad pivot and delays it,
but delay cascades degrade the ordering quality.

## 3. Rook Pivoting

Rook (also called "complete partial" or "refined BK") widens the
search from one column to a path of alternating column/row scans. The
key insight: a pivot that is *both* a column-max and a row-max (up to
the `alpha` factor) has element-growth bound `1 + 1/alpha` per step
*independently of n* — that is, the growth factor is constant
instead of exponential (Foster 1997, later refined by Ashcraft-Grimes-
Lewis 1998).

**Algorithm (Duff-Reid 1996):**

```
Given trailing submatrix A[k:n, k:n]:
  i, j := k, k                       # start at diagonal
  loop:
    gamma_col := max_{r > j} |A[r, j]|
    if |A[i, j]| >= alpha * gamma_col:
      return Pivot1x1(i, j)          # accept
    i_new := argmax_{r > j} |A[r, j]|
    gamma_row := max_{c != j, c >= k} |A[i_new, c]|
    if |A[i_new, i_new]| >= alpha * gamma_row:
      return Pivot1x1(i_new, i_new)  # swap i_new into pivot
    if |A[i_new, j]| >= alpha * gamma_row:
      return Pivot2x2(i_new, j)      # 2x2 at (i_new, j)
    (i, j) := (i_new, i_new) ... continue
```

**Termination.** The sequence of `|A[i_k, j_k]|` values visited by the
rook loop is strictly increasing (each step moves to a larger absolute
value along the chosen row or column). Since the submatrix has a
finite maximum, the loop terminates. Ashcraft-Grimes-Lewis 1998 prove
an O(n) worst-case iteration bound and report an empirical mean of
~1.5 iterations on sparse indefinite KKT test problems.

**Degenerate case.** If the submatrix has equal-valued entries at the
path of iterations, the increasing-sequence argument fails. Practical
implementations (MA57, SSIDS) cap the loop at 8 iterations and fall
through to a last-resort 1×1 at the current position. FERAL will do
the same; Test 4 in the plan exercises this case.

## 4. Why Rook Helps KKT Specifically

Three reasons rook is particularly well-suited to the KKT/saddle-point
matrices FERAL targets:

1. **Zero diagonals.** KKT systems have structural zeros on the
   diagonal corresponding to equality constraints. BK-partial's
   column-only search cannot distinguish a truly zero diagonal from a
   tiny-but-nonzero one; the column-relative threshold rejects both.
   Rook's row-then-column search finds off-diagonal pivots quickly in
   these cases and forms a 2×2 block.

2. **Arrow/border structure.** The dense border of a KKT matrix
   (constraint Jacobian rows) has large off-diagonals relative to its
   diagonal. BK-partial sees these as "column-dominated" and delays
   pivots, which the multifrontal driver then pushes up the etree.
   Rook's row scan finds a pivot in the border that clears the
   threshold locally, eliminating the column in place.

3. **Ill-scaled primal blocks.** When the Hessian block has widely
   varying scales (e.g., some variables in SI units, others normalized),
   BK-partial's column search can pick a pivot that is numerically
   dominated by the *next* column's entries. Rook catches this via
   the row scan and swaps in a better-scaled pivot. MC64 scaling
   (already in FERAL) addresses the same problem prophylactically, but
   rook handles the residual cases where scaling is imperfect.

This analysis is consistent with Duff-Pralet 2005 and Hogg-Scott 2008,
which report that MA57 (rook-based) achieves significantly tighter
iterative-refinement convergence on KKT benchmarks than MA27 (BK-
partial-based).

## 5. Frontal Adaptation: Ghost Rows

FERAL's frontal matrices have two row regions:

- Rows `[0, ncol)`: fully-summed, eligible for pivoting.
- Rows `[ncol, nrow)`: contribution rows, passed to parent supernode,
  not eligible for pivoting at this front.

BK-partial's `gamma0` search in FERAL already scans both regions
(ghost rows contribute to the threshold test but not to pivot
placement — see `column_offdiag_max` at `src/dense/factor.rs:1668`).
Rook must do the same:

- **`gamma_col`** includes rows `[k, nrow)` (fully-summed + ghost).
  This is the growth-bound input.
- **Pivot candidate row `i_new`** is restricted to `[k, ncol)`. If the
  column-max lies in a ghost row, rook cannot place that pivot; either
  fall back to the best fully-summed row or abandon the rescue.
- **`gamma_row`** is scanned over `[k, ncol)` only — ghost rows have
  no "row max" semantics because they are not being pivoted.

This is a feral-specific asymmetry. MA57 handles it via a `nelim`/`nabove`
split with identical effect. SSIDS's `block_ldlt` uses the same
pattern at `src/ssids/cpu/kernels/ldlt_app.hxx`.

## 6. Interaction with the Column-Relative Threshold

FERAL's current `pivot_threshold` (MUMPS `CNTL(1)`, default 0.01 under
MC64) rejects a pivot when `|d| <= u * col_max`. In the rescue design:

1. BK-partial runs. If it picks a pivot that clears the threshold,
   accept and continue. Rook does not fire.
2. If BK-partial's pivot fails the threshold, rook fires. Rook searches
   for a pivot that clears the *rook-style* criterion
   (`|a[i,j]| >= alpha * gamma_col` AND `|a[i,j]| >= alpha * gamma_row`).
   This criterion is strictly stronger than the BK threshold for the
   usual `alpha ≈ 0.64` and `u ≈ 0.01` values — rook's pivot always
   clears the threshold if it exists.
3. If rook finds no pivot (submatrix is numerically singular in this
   region), fall through to existing delay/reject logic.

Consequence: rook rescue never violates the column-relative threshold
guarantee. It only *enlarges* the set of pivots that satisfy it.

## 7. Element Growth Under Rescue

A subtle point: rescue changes the overall growth bound. The
factorization's worst-case growth is now:

- For pivots taken by BK-partial (common case): `(1 + 1/alpha)^k`
  where `k` is the count of BK-partial pivots. Same as before.
- For pivots taken by rook (rescue): `1 + 1/alpha` per step,
  constant.

Since rook fires only when BK-partial would reject, the overall bound
is no worse than BK-partial's alone. In practice, rook-rescued pivots
have *better* growth because they are chosen from a larger candidate
set. This is the numerical quality argument for the rescue design
independent of the perf argument.

## 8. Comparison to Complete Pivoting (Bunch-Parlett)

Bunch-Parlett 1971 scans the entire remaining submatrix for every
pivot — O(n²) per step, O(n³) total overhead on top of factorization.
Growth bound is optimal at `(1 + 1/alpha)` per step. Rook achieves
the same growth bound asymptotically with O(n) per step via the
bounded-iteration argument, which is why rook is the standard in
production solvers and complete pivoting is not.

FERAL should not implement complete pivoting. If rook is insufficient
on some matrix class, the correct escalation is iterative refinement
(already in FERAL via `solve_sparse_refined`), not complete pivoting.

## 9. Test Oracle Strategy

Rook pivots differ from BK-partial pivots on rescue-triggering
matrices, so existing BK77-based tests do not apply as oracles.
Oracle sources for the new tests:

1. **Hand-traced examples.** Construct small matrices (4×4, 5×5) where
   the rook path is short enough to verify by pencil. Document the
   expected pivot sequence in the test body. This is the primary
   correctness oracle for Steps 3–4 of the plan.
2. **MUMPS / MA57 output.** For the CRESC100 / GAUSS2 regression
   matrices, use MUMPS's factor as the reference. MUMPS uses a
   rook-like refinement strategy so its pivot choices should
   approximately agree. Compare inertia exactly, residual within
   `1e-10`, factor time qualitatively.
3. **Sylvester invariance.** Inertia is invariant under congruence
   transformation, so any valid factorization of the same matrix must
   produce the same inertia regardless of pivoting strategy. This is
   the hard test — even small rook bugs that miscount a 2×2 block
   produce detectable inertia errors.

## 10. Non-Goals

- **Not** a general-purpose rook implementation. Only the rescue
  variant, which assumes BK-partial already ran and failed its
  threshold test. This simplifies the code (no separate pivot search
  driver) and keeps the fast path free of rook cost.
- **Not** a whole-panel rook. Blocked-panel path continues to use
  BK-partial; rook fires in the scalar fallback only. A hypothetical
  panel-native rook would require carrying rook swap state through
  the peek-ahead and is deferred to Phase 2.4.4.
- **Not** complete pivoting. See Section 8.
- **Not** a replacement for MC64 scaling. Rook and scaling are
  complementary — scaling reduces how often rook fires, rook handles
  the residual cases scaling cannot.

## 11. Open Questions — Resolved 2026-04-23 (pre-Step-4)

1. **Rook 2×2 vs BK-partial's 2×2 criterion — strictly replace.**
   Confirmed by tracing `scalar_pivot_step` at `src/dense/factor.rs`:
   BK-partial's 2×2 decision tree (steps §3.Case-3 / §3.Case-2 /
   §3.Case-2×2 at lines 232–307) is taken *before* any 1×1 rejection
   test. Once `scalar_pivot_step` has called `do_2x2_pivot` successfully,
   `try_reject_1x1_frontal` never runs for that column pair — so rook
   rescue is never invoked on 2×2 candidates BK-partial accepted. The
   only path into rook rescue is: BK-partial chose 1×1, the column-
   relative threshold rejected it, `try_reject_1x1_frontal` returned
   `Delayed`/`Rejected`. At that point BK-partial's 2×2 criterion is
   irrelevant; rook applies its own Duff-Reid criterion (plan §"Rook
   Search Algorithm" step 7).

2. **`pivot_threshold` default — keep 0.01 through Step 5.** Deferred
   to Step 7 per plan §"Implementation Order". Changing the threshold
   in the same commit as rook rescue would couple two independent
   decisions and complicate the regression attribution. Step 7 of the
   plan measures CRESC100/GAUSS2 with the unchanged 0.01 threshold;
   if rook alone doesn't close the tail, a separate decision-log entry
   raises the threshold to 0.1 and re-measures. Record in
   `dev/decisions.md` at Step 5.

3. **SSIDS 2×2 det floor — rook must apply the same floor to its 2×2
   candidates.** The cancellation-aware det floor at
   `src/dense/factor.rs:1409-1440` (SSIDS `ldlt_tpp.cxx:98-106` port)
   protects BK-partial's 2×2 path from accepting near-singular blocks
   where `|a11*a22| ≈ |a21^2|`. Rook's step-7 acceptance criterion
   (`|A[i', j_prev]| >= alpha * gamma_row`) bounds the off-diagonal
   magnitude but not the determinant — rook can accept a 2×2 whose
   determinant cancels catastrophically. The kernel at Step 4 must
   apply the same `max_piv`/`detpiv0`/`detpiv1` test to any 2×2 it
   proposes. If rook's 2×2 fails the det floor, rook falls through to
   step-8 (iterate) just as if step-7 had failed. This is an
   implementation requirement, not an open question. Cross-check
   against MA57 Fortran source during Step 4 review.

## 12. References (added)

Added to `dev/references.bib` 2026-04-23 as prereq to Step 1:

- `duffreid1996zeros` — Duff & Reid, "Exploiting zeros on the
  diagonal in the direct solution of indefinite sparse symmetric
  linear systems." ACM TOMS 22(2), 1996. The original rook
  publication in the sparse-direct context.
- `ashcraft1998accurate` — Ashcraft, Grimes & Lewis, "Accurate
  symmetric indefinite linear equation solvers." SIAM J. Matrix
  Anal. Appl. 20(2), 1998. Termination proof and empirical
  iteration counts (mean ~1.5); 8-iteration safeguard.
- `duff2004ma57` — Duff, "MA57 — a code for the solution of
  sparse symmetric definite and indefinite systems." ACM TOMS
  30(2), 2004. Reference implementation of rook-as-rescue in a
  production solver; documents frontal ghost-row handling.
- `foster1997rook` — Foster, "The growth factor and efficiency of
  Gaussian elimination with rook pivoting." JCAM 86(1), 1997.
  O(n^{0.5 log n}) growth bound; empirical equivalence to complete
  pivoting.
