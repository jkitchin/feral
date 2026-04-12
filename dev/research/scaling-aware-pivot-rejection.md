# Scaling-aware pivot rejection for the sparse multifrontal kernel

**Status:** Pre-implementation research note for Phase 2.2.2.
**Date:** 2026-04-12
**Related:** Phase 2.2.1 landing commit `8a95825` (MC64 scaling), diagnostic
`dev/debugging/2026-04-12-acopp30-regression.md` (commit `3d0716b`),
prior research note `dev/research/mc64-scaling.md`, prior research note
`dev/research/dense-ldlt.md` (Bunch-Kaufman baseline).
**Key references:** citet:bunch1977stable, citet:duff1983multifrontal,
citet:hogg2013pivoting, citet:duff2001mc64, citet:duff2005symmetric.

---

## 1. Problem statement

Phase 2.2.1 landed MC64 symmetric matching-based scaling (strategy 5 in
citet:duff2001mc64, symmetric averaging per citet:duff2005symmetric). On
six of the seven sanity-panel matrices it improved the relative residual
by two to ten orders of magnitude. On the seventh — `ACOPP30_0000`,
n=209 — it **regressed** the residual from `2.84e+16` (the Phase 1
Identity-scaling baseline) to `2.27e+46`, a thirty-order blow-up. The
diagnostic report pinpointed the failure mode exactly:

> MC64 equilibrates `A → D·A·D` so pivots cluster around 1. In doing
> so it pushes the worst pivots from `~1e-8` (the Phase 1 KKT
> constraint-block `delta_c` floor) down to `~3.6e-10`, which is still
> six orders of magnitude above feral's `zero_tol = f64::EPSILON`. The
> Phase 1 `ZeroPivotAction::ForceAccept` accepts anything above
> `EPSILON` as a genuine pivot and divides by it in the D-solve. Five
> such `3.6e-10`-class pivots cascading through the forward L-sweep,
> the D-solve, and the backward L^T-sweep compound rounding error into
> the `1e30` range — exactly the observed 30-order residual blow-up.
> (`dev/debugging/2026-04-12-acopp30-regression.md`, §"Root cause".)

Four of the `mc64_regression.rs` tests are now `#[ignore]`-gated with
this exact symptom family (residuals `1e+2`–`1e+46` instead of
`<1e-8`). The MC64 scaling vector itself is correct; the assembly and
solve wrappers are correct; the bug is in the **pivoting strategy's
failure to reject the small pivots that MC64 deliberately exposes**.
This note designs the minimum fix.

**Failure mode in one line.** Small pivots that survive both the
`zero_tol` 1×1 threshold and the `zero_tol_2x2` determinant threshold
cascade through LDL^T because the absolute thresholds are orders of
magnitude below the scaled matrix's natural magnitude scale, and
because small 1×1 magnitudes can hide inside 2×2 blocks whose
determinant is healthy (`|det| ≈ 0.08` on the ACOPP30 offenders, well
above `zero_tol_2x2 = f64::EPSILON² ≈ 4.9e-32`).

---

## 2. Background: threshold partial pivoting in canonical solvers

All three reference solvers (MUMPS, SSIDS, faer) use **threshold
partial pivoting** for symmetric indefinite LDL^T. The canonical
formulation comes from citet:bunch1977stable Theorem 3 (Bunch-Kaufman
pivot selection with `α = (1 + √17)/8 ≈ 0.6404`) extended with the
"relative threshold" test introduced in citet:duff1983multifrontal
for the multifrontal setting: a candidate pivot is accepted only when
its magnitude is at least a fraction `u` of the largest entry in its
column/row in the trailing submatrix. citet:duff1983multifrontal
§4.3 then describes the **delayed-pivot mechanism**: pivots that fail
the threshold test are postponed to the parent node in the assembly
tree, where additional rows/columns have become available. MUMPS and
SSIDS both implement this; faer does not (faer is a dense kernel and
delayed pivoting is a multifrontal concept).

### 2.1 MUMPS — `dfac_front_aux.F:DMUMPS_FAC_I_LDLT`

The symmetric indefinite front-factorization kernel lives in
`ref/mumps/src/dfac_front_aux.F`, in the subroutine `DMUMPS_FAC_I_LDLT`
starting at line 1147. The parameter `UU` (alias `UULOC`) is the
pivot threshold `u`, and `SEUIL` is an absolute null-pivot floor
related to `DKEEP(1)`.

**1×1 acceptance test** (`dfac_front_aux.F:1494–1495`):

```fortran
IF ( abs(PIVOT).GE.UULOC*max(RMAX,AMAX)
 &     .AND. abs(PIVOT) .GT. max(SEUIL,tiny(RMAX)) ) THEN
```

Here `AMAX` is the max magnitude in the **left half** of the pivot
row (within the already-assembled part of the front) and `RMAX` is
the max magnitude in the **bottom half** of the pivot column
(trailing submatrix). The pivot is accepted when its magnitude is at
least `u` times the larger of these, AND exceeds the absolute floor
`SEUIL`. Both conditions must hold. If either fails, control falls
through to the 2×2 path.

**2×2 acceptance test** (`dfac_front_aux.F:1590–1606`):

```fortran
DETPIV = A(POSPV1)*A(POSPV2) - A(OFFDAG)**2
ABSDETPIV = abs(DETPIV)
IF (SEUIL.GT.RZERO) THEN
  IF (sqrt(ABSDETPIV) .LE. SEUIL ) THEN
    GOTO 460              ! delay
  ENDIF
ENDIF
...
IF ((abs(A(POSPV2))*RMAX+AMAX*TMAX)*UULOC.GT.ABSDETPIV
 &     .OR. (ABSDETPIV .EQ. RZERO) )  THEN
  GO TO 460               ! delay
ENDIF
IF ((abs(A(POSPV1))*TMAX+AMAX*RMAX)*UULOC.GT.ABSDETPIV
 &     .OR. (ABSDETPIV.EQ. RZERO) ) THEN
  GO TO 460               ! delay
ENDIF
```

The 2×2 test asks: is `|det|` at least `u` times the larger of two
growth-bound expressions? This is the Duff-Reid 1×1-within-2×2
stability criterion in algebraic form. `RMAX` and `TMAX` are the
trailing-column maxima for the two candidate rows; `AMAX` is the
larger off-diagonal within the candidate pair's fully-summed region.
If either inequality fails, the pivot is delayed (`GOTO 460`).

**Delayed pivot handling.** When `GOTO 460` fires, the pivot position
is not eliminated at this node. MUMPS's mechanism: the pivot's row
and column are left in place in the frontal matrix, but their indices
are transferred to the parent node's "delayed" list via the `INOPV`
flag (`dfac_front_LDLT_type1.F:433–443`, where `INOPV=-1` signals
"delayed pivots were written"). The parent node's frontal is resized
to accommodate them as additional fully-summed columns during
assembly.

**Default thresholds.** `ref/mumps/src/dini_defaults.F:1093` sets
`CNTL(1) = -1.0`, and `ref/mumps/src/dfac_driver.F:472–480` resolves
the "auto" value to `0.01` for symmetric indefinite (`KEEP(50) .ne. 1`
and `KEEP(19) .eq. 0`). The threshold is then capped at `0.5` for any
symmetric case at `dfac_driver.F:495–497`. So the effective default
is **`UU = 0.01`** — a candidate pivot must dominate its column by at
least a factor of 100 to be accepted.

**Interaction with MC64.** MC64 is applied at analysis time and
produces `ROWSCA` / `COLSCA` which are multiplied into each frontal
entry as it is assembled (`dfac_dist_arrowheads_omp.F:1021–1024`).
The threshold `UU = 0.01` then operates on the already-scaled
entries, which is exactly the point: after symmetric equilibration,
the off-diagonals are bounded by 1 and most diagonal entries are
close to 1, so rejecting pivots at `|d| < 0.01 · 1 = 0.01` is the
designed use of the equilibration. No scaling-dependent adjustment of
`UU` is needed — the threshold is on the scaled magnitudes.

### 2.2 SSIDS — `ldlt_tpp.cxx` and `block_ldlt.hxx`

SSIDS has two pivot-selection kernels. The "clean up small fronts"
kernel is `ldlt_tpp.cxx` (threshold partial pivoting, scalar), and
the blocked kernel for normal-sized fronts is `block_ldlt.hxx`. Both
use the same test structure.

**`ldlt_tpp_factor`** (`ref/spral/src/ssids/cpu/kernels/ldlt_tpp.cxx:
164–270`). The function signature includes `const T u, const T small`
(line 167). The main loop iterates over candidate pivot positions
`nelim+1..n`. For each candidate `p`:

1. Build the 2×2 candidate with its row-max partner `t`
   (`ldlt_tpp.cxx:207–211`).
2. Run `test_2x2(t, p, maxt, maxp, ..., u, small, &d)` — a 30-line
   helper at lines 89–119.
3. If 2×2 fails, try `p` as a 1×1 pivot: `if (fabs(a[p*lda+p]) >=
   u*maxp)` at line 226.
4. If both fail, continue to the next candidate column.
5. When the inner loop exhausts (`p >= n`, line 239), fall back to
   `p = nelim` as a last-resort 1×1. If that also fails, `break`
   — the return value `nelim` signals "only `nelim` pivots were
   eliminated; the rest are delayed to the parent". This is SSIDS's
   delayed-pivot protocol: the caller looks at the return value and
   passes the remaining rows/columns up the assembly tree.

The `test_2x2` function is the critical part (`ldlt_tpp.cxx:89–119`):

```cxx
double maxpiv = std::max(fabs(a11), std::max(fabs(a21), fabs(a22)));
if(maxpiv < small) return false;                     // below abs floor
double detscale = 1/maxpiv;
double detpiv0 = (a11*detscale)*a22;
double detpiv1 = (a21*detscale)*a21;
double detpiv = detpiv0 - detpiv1;
if(fabs(detpiv) < std::max(small,
                           std::max(fabs(detpiv0/2), fabs(detpiv1/2))))
   return false;                                      // cancellation guard
...
double x1 = fabs(d[0])*maxt + fabs(d[1])*maxp;        // D^-1 * column norms
double x2 = fabs(d[1])*maxt + fabs(d[3])*maxp;
return ( u*std::max(x1, x2) < 1.0 );                  // growth bound
```

The 2×2 decision has **three** checks, not one:
- (a) the absolute max within the 2×2 block is above `small`;
- (b) the determinant is well-conditioned (not vulnerable to
  cancellation — fabs(detpiv) must be at least half the larger of
  the two terms being subtracted);
- (c) the Duff-Reid growth bound: `u · max(x1, x2) < 1`, where `x1`
  and `x2` are the row-sums of `|D^{-1}| · (column maxima)` —
  equivalent to asking that the rank-2 update to the trailing block
  grows by at most `1/u`.

Check (b) is the SSIDS-specific elaboration that neither MUMPS nor the
original BK77 paper include. It catches the case where `detpiv` is a
small difference of two near-equal large numbers, even if `u · x < 1`
would accept — a cancellation guard against catastrophic loss of
precision. Feral should replicate this.

**`block_ldlt`** (`ref/spral/src/ssids/cpu/kernels/block_ldlt.hxx:
286–409`) — the compact version used for mid-block pivots. It uses a
simplified 1×1 test: if 2×2 fails, try `a11` or `a22` as a 1×1 with
`|a_kk / a_{k+1,k}| < u` as the rejection criterion (`block_ldlt.hxx:
337, 343`). This is `|a_kk| < u · |a_{k+1,k}|` algebraically —
identical to `fabs(akk) >= u * maxp` when `a_{k+1,k}` happens to be
the column max. When neither 1×1 works, `pivsiz = 0` is set and the
column is left in place for a later attempt.

**Default thresholds.** `ref/spral/src/ssids/datatypes.f90:260–262`:

```fortran
real(wp) :: small = 1e-20_wp ! Minimum pivot size
real(wp) :: u = 0.01
```

SSIDS default `u = 0.01` (same as MUMPS) and `small = 1e-20`. The
absolute floor is **four orders tighter** than `f64::EPSILON ≈
2.22e-16`, which means SSIDS essentially trusts the threshold test to
do the work and uses `small` only as an absolute-zero sanity check.

**Interaction with MC64.** SSIDS does **not** adjust `u` when scaling
is active — the threshold is on the scaled magnitudes. This is the
same protocol as MUMPS. However, SSIDS does not default to applying
scaling at all: `options%scaling = 0` (none). The user must opt in
via `options%scaling = 1` (MC64 match-based), `2` (auction), or `3`
(MC64 match + ordering). When scaling is requested, it is applied in
`assemble.hxx:64` as `node.lcol[k] = rscale * aval[src] * cscale` —
identical structure to MUMPS. `options%action = .true.` by default
(SSIDS's analog of `ForceAccept`: if threshold pivoting cannot find
any pivot and the matrix is singular, proceed with warning and zero
out the column instead of failing).

### 2.3 faer — `bunch_kaufman/factor.rs`

faer's dense Bunch-Kaufman lives in
`ref/faer-rs/faer/src/linalg/cholesky/bunch_kaufman/factor.rs`. It
implements textbook BK77 with `α = (1 + √17)/8` (line 271 and 710),
no user-configurable threshold, and no delayed pivoting (faer is a
dense kernel — delayed pivoting has no meaning there). The 1×1 test
is `A[(i0, i0)].real().abs() >= &alpha * &gamma_i` at line 546/817
and the LAPACK 3-way extension at line 586/841.

faer is relevant only as a confirmation that feral's existing dense
`src/dense/factor.rs` kernel is already correct **for the dense case**
— the issue is purely at the multifrontal frontal-kernel boundary,
where the `u` threshold must be applied on the scaled entries.

### 2.4 Takeaway for feral

The canonical design is **`u`-relative thresholding on the scaled
matrix, no scaling-dependent adjustment of `u`, plus delayed pivoting
for the unmatched ones**. The absolute floor `small` exists only as
a tiny-value sanity check (`1e-20` in SSIDS). MC64's job is to make
the absolute magnitudes sensible so that `u = 0.01` is a meaningful
threshold; threshold pivoting's job is to reject the small pivots
that MC64 deliberately exposes. Feral currently does neither: it has
no `u`-relative test, and its `zero_tol` is the absolute floor,
orders of magnitude too low.

---

## 3. Design options

Four options ordered by increasing complexity. Each is described in
terms of the algorithm, the files it would touch, its expected impact
on the four regression matrices (`ACOPP30_0000`, `CRESC132_0000`,
`CHWIRUT1_0000`, `CRESC100_0000`), and its risks.

### Option A — Column-relative `zero_tol` (minimal)

**Algorithm.** Keep the existing BK77 pivot-selection decision tree
in `src/dense/factor.rs` and `src/numeric/factorize.rs` exactly as is
(α, 1×1-vs-2×2 choice). Replace the absolute `zero_tol` check with a
column-relative one that depends on the trailing column max.
Specifically, whenever a 1×1 pivot is being committed, compute
`max_col = max_{i>k} |a[i,k]|` (which the BK algorithm already
computes as `γ₀`) and reject the pivot if

```
|a[k,k]| < u * max_col         (where u = 0.01 by default)
```

The existing `alpha`-based stability check of BK77 already ensures
`|a[k,k]| >= alpha * max_col ≈ 0.64 * max_col` for accepted 1×1
pivots, so **when the 1×1 branch is reached on a scaled matrix the
test normally succeeds**. The failure mode is when `alpha * max_col`
itself is tiny — the BK tree then accepts the 1×1 because the
relative criterion is satisfied (near-zero divided by near-zero), and
feral commits a catastrophically small pivot to D. Option A catches
this by adding a second requirement: the column max must itself be
"big enough" — specifically, the committed pivot must be above
`u · max_col_abs` where `max_col_abs` is the max of **all** entries
in the eliminated column (including the pivot itself). This is the
"absolute above a fraction of the column norm" criterion of SSIDS's
`ldlt_tpp_factor` line 226.

For 2×2 blocks, implement the Duff-Reid growth bound from SSIDS
`test_2x2`:

```
|det| >= u * max(|a22| * max_col_k + |a21| * max_col_{k+1},
                 |a21| * max_col_k + |a11| * max_col_{k+1})
```

Reject → call `on_zero_pivot`. With `ZeroPivotAction::ForceAccept`,
the rejected 1×1 pivot is written as zero and the column is zeroed;
the rejected 2×2 pivot is similarly zeroed (both diagonal entries
and both columns). The solve then correctly skips these positions
(it already checks `|d| <= zero_tol`).

**No delayed pivoting.** Rejected pivots are flushed to zero via the
existing `ForceAccept` path. On ACOPP30 this yields the same
effective behavior as the Identity path's 5 `1e-8` pivots: they get
flushed to zero instead of amplifying the solve. The solve drops
from `2.27e+46` back to `~1e+16` (the Identity baseline). The other
5 panel matrices retain their 2–10 order improvements because their
ACOPP30-class "tiny-pivot explosion" was not the blocker — their
residuals were dominated by the scaling mismatch that 2.2.1 already
fixed. For CRESC132/CHWIRUT1/CRESC100, which improved but did not
cross the acceptance threshold, Option A is expected to push them
further toward acceptance but may still leave some above `1e-8`.

**Files changed.**
- `src/dense/factor.rs` — add `u_thresh: f64` to
  `BunchKaufmanParams`, extend `count_1x1_inertia` and `do_2x2_pivot`
  to take the column max into account, implement `test_2x2` helper
  mirroring SSIDS.
- `src/numeric/factorize.rs` — thread the `u_thresh` field through
  `BunchKaufmanParams` into the frontal kernel; the frontal kernel
  already has the column max in `gamma0` / `gamma_r` variables from
  the existing pivot search.
- `src/lib.rs` — re-export if new public types are added.

**Estimated effort.** 1–2 days (hand-computed test first, then
implementation, then regression validation).

**Risk.** Rejecting more pivots changes the inertia. The 139
currently-passing tests include `sparse_postorder.rs` and
`threshold_consistency.rs::sparse_solve_skips_zero_pivots_rank_deficient`,
both of which were tuned against the current `zero_tol = EPSILON`
behavior. Specifically, the rank-deficient test relies on feral
accepting the `A[0,0]=2, A[1,0]=1, A[1,1]=1, A[2,1]=1, A[2,2]=2`
matrix's structurally rank-deficient column with a particular
inertia. Raising the rejection threshold may change what feral calls
"zero" vs "positive" on that matrix. Mitigation: verify by running
with `u = 0.01` and checking whether the inertia changes. If it
does, gate the test on `ScalingStrategy::Identity`.

**Also, Option A does not emit "delayed pivots"** — it just
drops them to zero. On a matrix where MUMPS would have delayed 5
pivots to the parent node and then successfully eliminated them
there (with the larger assembled column context), feral will flush
them to zero and flag `needs_refinement`. The refined solve then
relies on iterative refinement to converge, which works when the
zeroed pivots correspond to genuinely rank-deficient rows but fails
if they were threshold-rejected only temporarily (and would have
been fine at the parent).

### Option B — Threshold-only with BK decision tree (middle)

**Algorithm.** Same as Option A, but additionally re-derive the
1×1-vs-2×2 decision tree with threshold-aware semantics. Current
feral code uses BK77 Theorem 3 unchanged (α ≈ 0.6404). In Option B,
the decision tree is extended:

- If `|a[k,k]| >= u · max_col` **and** `|a[k,k]| >= alpha · max_col`
  — accept 1×1.
- Else if the BK swap-to-r step yields `|a[r,r]| >= u · max_col_r`
  **and** `|a[r,r]| >= alpha · max_col_r` — accept 1×1 at r.
- Else compute the 2×2 candidate `{k, r}` and run SSIDS's `test_2x2`.
- If 2×2 fails — reject (ForceAccept as in A, or delay in Option C).

This differs from A in being slightly more discriminating about
which 1×1 to pick, and in matching SSIDS's `ldlt_tpp_factor` more
literally. For feral it is probably over-engineering at this phase,
because the BK77 tree is already a good heuristic and feral's
dense tests validate against BK77 exactly. Moving to SSIDS's tree
would invalidate the existing exact BK77 validation tests.

**Files changed.** Same as A, plus significant rewriting of
`src/dense/factor.rs` pivot selection logic.

**Risk.** Diverges feral from its clean BK77 derivation. The
existing dense tests validate against exact BK77 pivot sequences from
citet:bunch1977stable's examples; those tests would fail or require
rewriting. High cost for marginal benefit over Option A.

### Option C — Delayed pivoting (textbook)

**Algorithm.** Full Duff-Reid 1983 delayed pivoting. Pivots that fail
the threshold test are **not** zeroed; instead, they are postponed
to the parent node in the elimination tree. This requires:

1. The frontal factorization kernel (`factor_frontal` in
   `src/dense/factor.rs:364`) returns a count of "eliminated" columns
   that may be **less than** the requested `ncol`. Rows/columns
   `nelim..ncol` are transferred back as part of the contribution
   block. (This mirrors SSIDS's `ldlt_tpp_factor` return value.)
2. The assembly step at the parent node in `src/numeric/factorize.rs`
   extends the parent's frontal to accommodate the delayed columns
   from each child. This is where the "contribution block extension"
   in Duff-Reid 1983 §4.3 happens.
3. Symbolic analysis must allocate extra space in the parent frontal
   for possible delayed columns, or the numeric phase must be
   prepared to resize dynamically.
4. The elimination tree handling of delayed columns must maintain
   the symbolic postorder correctness — specifically, the column
   counts and supernode structure must allow for the extra columns.

This is the "right" answer but is weeks of work and a substantial
refactor of the frontal machinery. Phase 2.2.2 budget does not
support it; it belongs in Phase 2.3 or later.

**Files changed.** `src/dense/factor.rs` (`factor_frontal` API
change), `src/numeric/factorize.rs` (assembly path rewrite),
`src/symbolic/mod.rs` (optional frontal size slack), new
`delayed_pivots.rs` helper for tracking.

**Risk.** Large refactor; high chance of regressing the 139 passing
tests. Timing uncertainty ±100%. Correct but Phase 2.3 work.

### Option D — Hybrid: threshold + modest regularization on fallback

**Algorithm.** Option A, plus: if a 1×1 pivot is rejected and the
`ForceAccept` branch fires, instead of writing zero, write
`sign(a[k,k]) · max(u · max_col_abs, SEUIL)` where `SEUIL` is an
absolute floor analogous to MUMPS's `DKEEP(1)` (typically `1e-20`).
This is MUMPS's `CSEUIL` mechanism from
`dfac_front_aux.F:819–828`. The effect is that rejected pivots are
**not** zeroed but are instead replaced by a small regularized value
that keeps the factorization non-singular. The solve then divides
by this small regularized value but the magnitude is bounded (since
it is chosen against the column norm), and iterative refinement
usually cleans up the error.

This is slightly more invasive than Option A but has one key
advantage for KKT matrices: IPOPT expects the solver to return a
factorization even when the matrix is ill-conditioned; regularizing
a rejected pivot instead of zeroing it preserves the inertia more
predictably. Zeroing a pivot makes it count as "zero inertia", which
IPOPT interprets as a near-singular constraint and triggers
regularization of its own — potentially double-regularizing. A
MUMPS-compatible approach that matches canonical behavior is to
write the regularized value.

However: Option D changes the inertia reported by feral on the
ACOPP30 path. The ACOPP30 regression report's Option 1 experiment
already ruled this out: setting `zero_tol = 1e-8` in the MC64 path
gave inertia `(62, 140, 7)` — two fewer zeros than Option A would
produce, but still **not** the canonical MUMPS `(71, 137, 1)` or
IPOPT's `(72, 137, 0)`. Option D would yield yet another inertia
count depending on the sign of the regularized value. Until we
understand the Phase 1 `zero_tol` mismatch more thoroughly — and
specifically whether the 5 rejected pivots should be counted as
positive, negative, or zero — Option D risks hiding a correctness
issue rather than exposing it.

**Files changed.** Same as A, plus a `regularize_on_reject: bool`
flag and one new helper that chooses the regularized value.

**Risk.** Masks the inertia gap that Phase 2.2.2 should be helping
to expose, not hide. Decision deferred to a later phase.

### Recommendation: **Option A**

Option A is the minimum change that correctness requires:

1. It closes the ACOPP30 regression by ensuring the
   `~3.6e-10`-class pivots are rejected instead of inverted. The
   residual drops from `2.27e+46` back to `~1e+16` (the Identity
   path's baseline), which is the pre-MC64 behavior — same
   correctness regime, no regression.
2. It probably also closes CHWIRUT1 and CRESC100 (residual
   `8.5e+02` and `1.4e+02` post-MC64) by rejecting small pivots
   that the current `zero_tol = EPSILON` lets through. CRESC132's
   residual (`1.4e+05`) is probably dominated by something else and
   may not be fully resolved by Option A alone.
3. It is 1–2 days of focused work plus validation, matching the
   Phase 2.2.2 budget (§13.2 of the spec's per-session cap).
4. It does not diverge feral's pivot selection from the clean BK77
   derivation that the existing dense tests validate against. The
   new check is a superset of BK77's check, not a replacement — a
   pivot still needs `|a[k,k]| >= alpha · γ₀` AND additionally
   `|a[k,k]| >= u · max_col_abs` and `|d[k,k]|` above an absolute
   floor. All of BK77's acceptance guarantees carry over.
5. It matches SSIDS's `ldlt_tpp_factor` behavior on the 1×1 branch
   exactly (line 226), and SSIDS's `test_2x2` on the 2×2 branch
   (lines 89–119). The literature citation trail (BK77 + Duff-Reid
   83 + SSIDS source) is clean.
6. It preserves `ForceAccept` semantics: the Phase 1 rank-deficient
   test (`sparse_solve_skips_zero_pivots_rank_deficient`) continues
   to rely on rejected pivots being flushed to zero, which Option A
   still does.

Full delayed pivoting (Option C) is the textbook-correct answer but
belongs in Phase 2.3. Option A captures the essential fix without
the large frontal-assembly refactor.

---

## 4. Data model changes

### 4.1 `BunchKaufmanParams`

Add one field. From `src/dense/factor.rs:6–29`:

```rust
pub struct BunchKaufmanParams {
    pub alpha: f64,
    pub zero_tol: f64,
    pub zero_tol_2x2: f64,
    pub on_zero_pivot: ZeroPivotAction,

    /// Pivot threshold `u` (SSIDS, MUMPS `CNTL(1)`). A candidate 1×1
    /// pivot is rejected when |a[k,k]| < u * max_col_abs. A 2×2
    /// candidate is rejected when the Duff-Reid growth bound fails.
    /// Default: 0.01, matching SSIDS `options%u` and MUMPS CNTL(1).
    ///
    /// Setting `u = 0.0` disables threshold pivoting and recovers
    /// the Phase 1 behavior (backwards-compat for the dense BK77
    /// validation tests).
    pub pivot_threshold: f64,
}
```

Default `0.01` when used via `SupernodeParams::default()` (the
sparse path); default `0.0` when used via `BunchKaufmanParams::
default()` directly (the dense path), to avoid regressing the
dense BK77 validation tests which were hand-computed against
un-thresholded BK77.

**Alternative:** a `PivotThresholdStrategy` enum. Not worth the
extra complexity at this phase — a single `f64` is enough to
express "on with u=0.01" and "off" and "tighter than default".
An enum buys nothing.

### 4.2 `ZeroPivotAction`

No new variant needed. `ForceAccept` semantics remain: when the
threshold test fails, call `ZeroPivotAction::ForceAccept` which
zeros the column. If `Fail`, return `NumericallyRankDeficient`. The
existing action dispatch is sufficient.

### 4.3 `SparseFactors` / `Factors` telemetry

Add one counter:

```rust
pub struct Factors {
    // ... existing fields ...
    /// Number of pivots rejected by the threshold test
    /// (|a[k,k]| < u * max_col_abs, or 2×2 Duff-Reid bound failure).
    /// Zero when `pivot_threshold == 0.0`. Always ≥ `inertia.zero`
    /// because threshold-rejected pivots land in `inertia.zero`.
    pub n_threshold_rejected: usize,
}
```

Analogous field on `FrontalFactors`. Useful for diagnostics and for
verifying in tests that the rejection path fires when expected. Not
load-bearing for correctness.

### 4.4 `SymbolicFactorization` / `ScalingInfo`

No change. `SymbolicFactorization::scaling_info` is already populated
by Phase 2.2.1 and `factorize_multifrontal` can read it to decide
whether to enforce the threshold (yes, on any scaled path; the
dense BK77 tests pass `u = 0.0` explicitly).

---

## 5. Test strategy

### 5.1 Hand-computable miniature reproductions

**Test 1: 3×3 tiny-pivot hidden by scaling.** Construct

```
A = diag(1e-10, 1, 1)    (symmetric, no off-diagonals)
```

Pre-scaling, pivot magnitudes are `1e-10, 1, 1`; the first is
tiny. Post-MC64 symmetric scaling, the scaling vector is
`s = (1e5, 1, 1)` (so `D·A·D = diag(1e-10·1e10, 1, 1) = diag(1, 1,
1)` — identity). The scaled pivot is now exactly 1, which passes any
threshold test, so **this particular case is not the failure
mode**. It is included to verify that healthy-after-scaling pivots
are accepted.

**Test 2: 4×4 ACOPP30-style pivot compounding.** Construct a
matrix where scaling cannot recover the pivots because they are
coupled through off-diagonals:

```
A = [[ 1e-5, 1, 0, 0],
     [ 1, 1e-5, 1e-3, 0],
     [ 0, 1e-3, 1, 1],
     [ 0, 0, 1, 1]]
```

This has mixed magnitude rows. After MC64 the scaled matrix should
have one or two sub-`0.01` pivots that Option A must reject. The
exact scaling and the expected post-scaling pivot magnitudes come
from running MC64 on the matrix by hand (or using the existing
`compute_scaling` helper and checking the result). Assert:
- Without Option A (`pivot_threshold = 0.0`): solve produces
  `‖x‖∞ > 1e5` (amplification).
- With Option A (`pivot_threshold = 0.01`): solve produces
  `‖x‖∞ < 1e2` (clean).
- Inertia differs by exactly the number of threshold-rejected
  pivots.

The oracle is MUMPS (via `../ripopt/rmumps`, testing reference only
per the hard rules). Get the reference inertia and residual from
`rmumps::solve` and assert feral matches to within the documented
tolerance. **Important:** the oracle comes from an external solver,
not from feral's own pre-Option-A path — per the hard rule about
test oracles.

**Test 3: already-healthy matrix.** Construct a well-conditioned
5×5 indefinite matrix (e.g., the BK77 Example 2). Option A must
**not** reject any of its pivots. Verify the inertia and residual
are unchanged from the Phase 1 baseline. This guards against Option
A being too aggressive.

### 5.2 Regression test revival

The four tests in `tests/mc64_regression.rs` are currently
`#[ignore]`-gated:

- `acopp30_0000_residual_under_1e_8_after_mc64` — target `< 1e-8`
- `cresc132_0000_residual_under_1e_6_after_mc64` — target `< 1e-6`
- `chwirut1_0000_residual_under_1e_8_after_mc64` — target `< 1e-8`
- `cresc100_0000_residual_under_1e_8_after_mc64` — target `< 1e-8`

Phase 2.2.2 acceptance criterion: after Option A lands, these four
tests move from `#[ignore]` to `#[test]`. The tests' current targets
may still be too strict — `< 1e-8` assumes the residual drops to
the canonical MUMPS regime, which requires threshold pivoting
**plus** delayed pivoting to fully match. Option A alone is likely
to hit `< 1e-4` to `< 1e-6` on these matrices, not `< 1e-8`. The
test targets may need relaxation after measuring the post-Option-A
residuals — if so, update the targets based on the measured values
(not by lowering the expected bar without evidence), and document
the relaxation in `decisions.md` and the Phase 2.2.2 session
checkpoint. Do not loosen a tolerance without a residual number to
justify it.

**Minimum acceptable behavior:** ACOPP30 must drop **at least** to
the Identity baseline `2.84e+16` (i.e., below the post-MC64
catastrophic `1e+46`). Below that, a post-Option-A residual in
`[1e-4, 1e-8]` is acceptable for landing Phase 2.2.2, with the
understanding that Phase 2.3 (delayed pivoting) will drive it the
rest of the way to `~1e-14`.

### 5.3 No-regression guarantee

All 139 currently-passing tests must still pass. Specific risks:

- `tests/threshold_consistency.rs::sparse_solve_skips_zero_pivots_rank_deficient`
  — exercises a rank-deficient 3×3 matrix. May or may not see its
  inertia change under Option A; verify and, if needed, change the
  test's `ScalingStrategy` to `Identity` so it exercises the Phase 1
  path rather than the Phase 2.2.1+2.2.2 path.
- `tests/threshold_consistency.rs::factors_carry_zero_tol_from_params`
  — asserts that `factors.zero_tol` matches `params.zero_tol`. Must
  still hold; add an analogous assertion for `pivot_threshold`.
- `tests/sparse_postorder.rs` — exercises an edge-case postorder on
  matrices with tricky supernode structure. Should be unaffected by
  pivot rejection logic, but verify.
- `tests/dense_ldlt.rs` — the BK77 validation tests. These use
  `BunchKaufmanParams::default()` which must continue to set
  `pivot_threshold = 0.0` for the dense path. Verify by re-running.
- `tests/kkt_matrices.rs`, `tests/kkt_hardening.rs`, `tests/stress_tests.rs`,
  `tests/property_tests.rs` — should be unaffected; run them.
- `tests/sparse_refined.rs` — verify that refined solves on matrices
  with threshold-rejected pivots still track best-iterate correctly.
- `tests/mc64_end_to_end.rs`, `tests/mc64_scaling.rs` — Phase 2.2.1
  tests; should be unaffected but verify.

### 5.4 Validation: sanity panel re-run

Re-run `examples/triage_large_cresc132.rs` on the 7-matrix sanity
panel. Record pre-Option-A and post-Option-A residuals for all seven
matrices in `dev/validation/phase-2.2.2-threshold-sweep.md`. Flag
any residual that got **worse** after Option A — that would indicate
a bug in the implementation.

---

## 6. Risk register

**R1. Threshold too strict.** `u = 0.01` may reject pivots that
are borderline-acceptable, changing the inertia feral reports.
*Mitigation:* Run Option A with `u = 0.01, 0.001, 0.0001` on
ACOPP30 and see which threshold brings the inertia closest to MUMPS
`(71, 137, 1)` without regressing residuals on the other 6 sanity
matrices. Default to the value that works best across the full
panel. SSIDS and MUMPS both use `0.01` which is a strong prior —
unless evidence says otherwise, keep `0.01`.

**R2. Threshold too loose.** `u < 0.001` may miss ACOPP30's
tiny-pivot cascade. The ACOPP30 offenders are at `~3.6e-10`, and
the column max is `~1` post-MC64, so `u * max_col ≈ 0.01`
decisively rejects them; `u = 1e-4` would still reject at
`|d| < 1e-4`, which is `1e6` times the actual pivot — also
fine. Only `u < 1e-9` lets them through. Unlikely to be the active
risk.

**R3. 2×2 criterion interacts with the deferred 2×2 trace fix.**
Phase 2.1.2 left a known bug in feral's 2×2 inertia computation
(the trace-vs-a00 sign classification). Raising `zero_tol_2x2`
magnitudes may expose this bug more clearly. *Mitigation:* Track
explicitly in the Phase 2.2.2 checkpoint; if the inertia disagreements
concentrate on 2×2 blocks, investigate. Consider whether the 2×2
trace fix should be folded into Phase 2.2.2 (see Open Question 4).

**R4. `sparse_solve_skips_zero_pivots_rank_deficient` breakage.**
This test exercises a structurally rank-deficient matrix and
asserts that feral zeros the rank-deficient pivot and produces a
clean residual via the compatible-RHS property. The scaling path
on this matrix may not trigger Option A's new rejection logic, but
verify by running. *Mitigation:* If it breaks, switch the test's
`SupernodeParams::scaling_strategy` to `Identity` so it exercises
only the Phase 1 code path.

**R5. The dense BK77 validation tests (`tests/dense_ldlt.rs`) break
under the new pivot rejection.** These tests hand-compute pivot
sequences from BK77 examples where the magnitudes are `O(1)` and
`u = 0.01` should not change the decisions. *Mitigation:* Default
`BunchKaufmanParams::default()` to `pivot_threshold = 0.0` (disabled)
so the dense direct path preserves exact BK77 behavior. Only the
sparse path via `SupernodeParams` opts into `u = 0.01`.

---

## 7. Open questions

1. **Default threshold value.** SSIDS and MUMPS both use `0.01`.
   citet:hogg2013pivoting §3.2 recommends `0.01` for indefinite
   matrices with well-conditioned scaling and a tighter `0.001` for
   poorly scaled problems — but MC64 is supposed to make the latter
   case the former. Propose `0.01` as the feral default; confirm
   on the sanity panel before landing.

2. **Does threshold pivoting fix the ACOPP30 inertia disagreement?**
   Currently feral reports `(62, 142, 5)` where MUMPS reports
   `(71, 137, 1)` and IPOPT `(72, 137, 0)`. The 5 force-accepted
   zero pivots in feral correspond to 9-ish inertia disagreement
   vs MUMPS. Threshold pivoting alone (Option A) is expected to
   change these 5 from "force-accepted" to "rejected-and-zeroed",
   which is still counted as zero inertia. So the inertia gap is
   likely **not** closed by Option A alone — the 9-off is because
   MUMPS **delays** those pivots and successfully eliminates them
   at the parent node (Option C). Phase 2.2.2 may land Option A
   with feral still reporting `(62, 142, 5)` on ACOPP30, and that
   is acceptable as long as the residual is below the minimum
   acceptable bar. Track whether any inertia improvement happens
   as a secondary observation; do not block landing on it.

3. **Column max scope: pivot column only or full frontal block?**
   SSIDS computes `max_col` as the max in the **single** pivot
   column (see `find_rc_abs_max_exclude` at `ldlt_tpp.cxx:75`).
   MUMPS computes separate `RMAX` (pivot column) and `AMAX` (pivot
   row within the fully-summed region), then uses `max(RMAX, AMAX)`
   for the 1×1 test. Feral's dense `factor()` already computes
   `γ₀ = max |A[i,k]|` (column only, `column_offdiag_max`) and
   `γ_r = max |A[i,r]|` for the symmetric row. The SSIDS approach
   (column only) is simpler and seems sufficient; defer the full
   MUMPS approach to a follow-up if validation shows the column-only
   is too loose.

4. **Should the deferred 2×2 trace-vs-a00 fix be folded into
   Phase 2.2.2?** The Phase 2.1.2 retrospective noted that feral's
   2×2 inertia classification has a latent bug in the
   trace/determinant branch logic. Phase 2.2.2 touches 2×2 pivot
   acceptance. The two are logically adjacent. *Recommendation:*
   Keep them separate. Phase 2.2.2 is about **rejecting** pivots,
   not **classifying accepted** ones. Fold the trace fix into a
   Phase 2.2.3 "2×2 classification correctness" sub-phase after
   2.2.2 lands. This keeps the Phase 2.2.2 commit scope small and
   reviewable, and ensures the test failures from each fix don't
   cross-contaminate.

---

## 8. Literature citations

All four required references already exist in `dev/references.bib`:
- `bunch1977stable` (line 42 of the bib)
- `duff1983multifrontal` (line 77)
- `hogg2013pivoting` (line 96)
- `duff2001mc64` (line 361) — from Phase 2.2.1
- `duff2005symmetric` (line 401) — from Phase 2.2.1

No additions needed. Cite them in the research note via
`citet:bunch1977stable` etc. as used above and in the rest of
`dev/research/`.

---

## 9. Estimated effort

- Research note (this document): **2 hours**, done.
- Implementation plan (follow-up
  `dev/plans/scaling-aware-pivot-rejection.md`): **1 hour**.
- Implementation of Option A:
  - Add `pivot_threshold` to `BunchKaufmanParams`: 30 min.
  - Extend `do_1x1_pivot` with column-relative check: 1 hour.
  - Implement `test_2x2` helper mirroring SSIDS: 1–2 hours.
  - Thread through `factor_frontal` and `factorize_multifrontal`:
    1 hour.
  - Update `SparseFactors` telemetry: 30 min.
- Tests:
  - Hand-computed 3×3 and 4×4 tests: 1–2 hours.
  - Un-ignore the 4 `mc64_regression.rs` tests and verify: 1 hour.
  - No-regression run of the 139 existing tests: 30 min.
- Validation:
  - Sanity-panel re-run: 30 min.
  - Inertia gap investigation (Open Question 2): 1 hour.
- Session checkpoint and journal: 30 min.

**Total realistic: 10–14 hours**, one long session.

---

## References

bibliography:../references.bib
bibliographystyle:plain
