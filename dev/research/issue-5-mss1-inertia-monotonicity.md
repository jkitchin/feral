# Issue #5 — MSS1 BK inertia non-monotone under δ_w·I perturbation

**Status:** pre-implementation research note
**Date:** 2026-05-10
**Issue:** https://github.com/jkitchin/feral/issues/5
**Author of report:** John Kitchin (downstream from `ripopt`'s
PDPerturbationHandler analog)
**Related notes:**
- `dev/research/2x2-bk-inertia-accounting.md` (which inertia rule fires where)
- `dev/research/scaling-aware-pivot-rejection.md` §2.1 (MUMPS `SEUIL` semantics)
- `dev/research/issue-2-kkt-pivot-default.md` (same α-boundary instability on `arki0003`)
- `dev/research/inertia-triage-2026-04-27.md` (MUMPS vs SSIDS pivoting near singularities)
**Key files:**
- `src/dense/factor.rs:1410, 2098, 436` — α-test sites (`|akk| >= alpha · gamma0`)
- `src/dense/factor.rs:1894-1912` — `count_2x2_inertia_val` (trace-based, used by sparse path)
- `src/dense/factor.rs:2386-2393` — force-accept-with-sign branch (where stabilization will live)
- `src/dense/factor.rs:47-127` — `panel_diag` counters (instrumentation handles)

## 1. Symptom

`Solver::factor` (sparse multifrontal path) returns `(pos, neg, 0)`
inertia counts that wander non-monotonically as a uniform positive
diagonal perturbation `δ_w · I` is added to the (1,1) block of an
MSS1 KKT (n=163, n_x=90, n_eq=73, jacobian rank ~62 — high
redundancy from the SDP relaxation structure).

Reported trace from `ripopt` with `ScalingStrategy::Identity`,
`zero_tol=1e-10`, `zero_tol_2x2=1e-20`, `on_zero_pivot=ForceAccept`:

```
δ_w=0       (98+, 65-, 0)
δ_w=1e-4    (98+, 65-, 0)
δ_w=1e-2    (96+, 67-, 0)
δ_w=1       (99+, 64-, 0)   ← worse than δ_w = 0
δ_w=1e2     (98+, 65-, 0)
δ_w=1e4     (95+, 68-, 0)
δ_w=1e6     (96+, 67-, 0)
δ_w=1e8     (99+, 64-, 0)
δ_w=1e10    (95+, 68-, 0)
δ_w=1e12    (90+, 73-, 0)   ← target hit
```

Operational consequence: ripopt's `PDPerturbationHandler` waits
for the inertia signal to match `(n+, m-, 0) = (90, 73, 0)` and
ramps δ_w until it does. With this trace it ramps to 1e12 — at
which point the resulting Newton step has `|dy|_∞ ≈ 5×10¹¹` and
the IPM diverges. Reference (Ipopt+MA57 on the same problem)
regularises with δ_w ≈ 100 and converges in 116 iterations.

## 2. Which path fires

`ripopt::FeralLdl::factor` → `feral::Solver::factor` →
`factorize_multifrontal` → `factorize_multifrontal_supernodal_with_workspace`.
Per-supernode the path goes through `factor_frontal_blocked_in_place`
which calls `lblt_panel_frontal` (block kernel) and falls back to
`scalar_pivot_step` (line 1331, 1625). Both ultimately call
`count_2x2_inertia_val` (the trace-based, Sylvester-correct rule
at `:1894-1912`).

So the **dense-only** `count_2x2_inertia` known-bug at
`src/dense/factor.rs:2216-2265` (sign-of-`a00` rule) is **not**
involved — it sits behind the dense `factor()` entry which the
sparse path never calls. The in-source TODO and the audit in
`2x2-bk-inertia-accounting.md` §1b stand; they are unrelated to
this report.

## 3. Why feral wanders

Two pieces of evidence pin the cause to BK 1×1/2×2 boundary
flipping:

### 3.1 All sweep entries report `0:0 zeros`

No pivot ever lands in feral's small-pivot branch. Both
`zero_tol = 1e-10` (line 2372) and `zero_tol_2x2 = 1e-20` (the
SSIDS-mirrored absolute floor) are far below anything that
arises in MSS1 even at δ_w = 0 — the matrix's smallest non-zero
diagonal magnitude is 1.0. So every pivot is committed as
either + or −; wandering must come from sign re-attribution as
the BK 1×1-vs-2×2 choice flips, not from threshold rejections.

### 3.2 KKT structure puts the α-test on a knife edge

The MSS1 KKT under Identity scaling has:
- (1,1)-block diagonals: 1 (Hessian) + δ_w
- (1,1) off-diagonals: from H, mostly small or zero
- (1,2)-block (J^T): unit-magnitude entries (edge constraints
  `x_i x_j + y_i y_j = c` linearise at `x = y = 1` to four
  `1`s; the spherical constraint to `2`s)
- (2,2)-block: 0 (no δ_c)

The α-test in `scalar_pivot_step` (line 2098) and the panel
inline path (line 1410) is `|akk| >= alpha · gamma0` with
`alpha = (1 + √17)/8 ≈ 0.6404`. For a constraint row with
`gamma0 ≈ 1` (a J^T column max) and `|akk| ≈ |δ_w|`:

- δ_w « 0.64 → α-test fails → 2×2 candidate
- δ_w » 0.64 → α-test passes → 1×1 with sign of the diagonal

Across the 163 columns, each individual column has its own
critical δ_w determined by which J^T row it is coupled to.
There are 63 columns at the rough threshold (the constraints
themselves), and AMD ordering interleaves them with H columns
in the supernode tree, so `|akk|` and `gamma0` see the
elimination Schur updates in different orders for different δ_w.
Each column flipping its bucket can shift (pos, neg) by ±1:

- 1×1 accepted: contributes `+1` if `d > 0` else `-1`
- 2×2 accepted with `det > 0`: contributes `(2, 0)` or `(0, 2)`
  per `count_2x2_inertia_val` based on `trace > 0`
- 2×2 accepted with `det < 0`: contributes `(1, 1)`

So a column transitioning "1×1 with `d > 0`" → "2×2 with
`det > 0`, `trace > 0`" gains a positive (and the partner
column was previously a 1×1 with its own sign, so the partner
flip is independent). Net (pos, neg) drift of ±8 across the
sweep is fully consistent with this mechanism.

The pattern in §1 — non-monotone, bouncing, settling only at
δ_w ≈ 1e12 — is exactly the signature: at 1e12 every diagonal
dominates every off-diagonal, the α-test passes uniformly, all
163 pivots are 1×1 with sign of `δ_w` (positive on the n_x
columns, negative on the m_eq columns where the (2,2) block is
zero so `d` ends up negative after Schur updates from the
−1/δ_w factor), and (pos, neg) hits the target.

## 4. Why MUMPS doesn't show the same symptom

MUMPS's `dfac_front_aux.F:DMUMPS_FAC_I_LDLT` 1×1 acceptance test
(line 1494-1495) has an absolute null-pivot floor `SEUIL`, derived
from `DKEEP(1)` (~`ε^(2/3) · ‖A‖_∞` by default). Pivots below
`SEUIL` are perturbed (static pivoting) and counted into
`INFOG(28)` as "zero". That is the source of the large `0:N`
triples in the reporter's `rmumps` trace — a feature, not a
different pathology.

feral has no equivalent. `zero_tol` is an *absolute* `1e-10`,
not a norm-relative threshold. The same physically-tiny pivots
that MUMPS would call zero get force-accepted by feral with
whatever sign the value happens to have at commit time, which
depends on pivot ordering and Schur-update magnitudes.

## 5. Design space

Three options in increasing scope.

### Option A — verify-only (suggestion #1 from the issue)

Reproduce the wandering on MSS1_0000 in feral standalone (no
ripopt dependency), with `panel_diag` counters enabled. For
each δ_w in the sweep, log:
- per-pivot inertia bucket (1×1+, 1×1−, 2×2 with each (pos,neg) tally)
- panel_full / panel_partial / fallback_2x2_* counter snapshot

If the diagnosis is right, `fallback_2x2_*` and `inline_2x2_*`
counts should shift non-monotonically with δ_w in the same
pattern as (pos, neg). This gates Option B / C — no code change,
only instrumentation, and the resulting trace is the regression
test fixture.

**Files touched.** New `src/bin/issue5_inertia_sweep.rs`
debug binary. Reads `MSS1_0000.mtx` from the corpus, configures
the BK params to match ripopt, sweeps δ_w, dumps inertia +
counter snapshot to stdout. No production code change.

**Effort.** 30 minutes.

### Option B — norm-relative pivot floor (recommended fix)

Add a `pivot_floor_strategy: PivotFloorStrategy` field to
`BunchKaufmanParams`:

```rust
pub enum PivotFloorStrategy {
    /// Absolute zero_tol only (current behaviour). Backwards-compat
    /// for the dense BK77 validation tests.
    AbsoluteOnly,
    /// MUMPS SEUIL analog: route pivots below
    ///   max(zero_tol, eps_relative · matrix_infnorm)
    /// into the zero bucket regardless of sign. `eps_relative`
    /// defaults to `f64::EPSILON.powf(2.0/3.0)` ≈ 3.7e-11.
    NormRelative { eps_relative: f64 },
}
```

Default for `BunchKaufmanParams::default()` stays `AbsoluteOnly`
(dense path unchanged). Default for `NumericParams::default()`
becomes `NormRelative { eps_relative: 3.7e-11 }` to match the
MUMPS-equivalent profile that issue #2's `pivot_threshold = 1e-8`
default already started moving toward.

Implementation: thread the matrix infnorm (already computed for
scaling — `compute_scaling`'s `ScalingInfo` carries it) through
`NumericParams` into `BunchKaufmanParams`'s pivot kernel, and
extend the force-accept-with-sign branch at
`src/dense/factor.rs:2386-2393` to bucket below-floor pivots
into `zero` instead of `pos`/`neg`. Same change in the rook-rescue
fast path (`try_reject_1x1_with_rook_rescue:2422-2500`) and in
the panel inline 2×2 acceptance (`src/dense/factor.rs:1579-1582`).

Test acceptance: after Option B, the MSS1 sweep should report
non-decreasing positive count at each δ_w step (or at minimum,
the count of non-monotone steps should drop from 9 to ≤ 2).
Inertia at δ_w = 0 may legitimately move from `(98, 65, 0)` to
something like `(90, 73, X)` with `X` zeros — matching MUMPS.

**Files touched.**
- `src/dense/factor.rs` — extend `BunchKaufmanParams`, add
  `PivotFloorStrategy`, plumb the floor into the pivot kernel.
- `src/numeric/factorize.rs` — pass matrix infnorm from
  `compute_scaling` into the dense kernel call.
- `tests/regressions/issue_5_mss1_inertia_monotonicity.rs` —
  new test.
- (Possibly) re-baseline `dev/research/inertia-triage-2026-04-27.md`
  numbers if any non-MSS1 corpus matrix changes inertia bucket.

**Effort.** 1–2 days plus regression validation.

**Risk.** Routing more pivots into `zero` changes the inertia
feral reports on non-MSS1 matrices. Validation must re-run the
full corpus inertia gate. The 113 mismatch matrices in
`inertia-triage-2026-04-27.md` are the most likely place to see
movement. Hard requirement: MUMPS-vs-feral consensus must improve
or hold; never regress.

### Option C — caller inertia hint API (suggestion #3 from the issue)

Expose a `tentative_inertia: Option<Inertia>` parameter that the
kernel uses to bias borderline pivot decisions toward the
caller-supplied target. Heavier; not what canonical solvers do
(MA57 / MUMPS / SSIDS all rely on the floor in B, not on hints).
Defer unless B turns out to be insufficient.

## 6. Recommendation

1. Land Option A first — instrumented standalone reproducer at
   `src/bin/issue5_inertia_sweep.rs`, plus a regression test
   fixture at `tests/regressions/issue_5_mss1_inertia_monotonicity.rs`
   that asserts the *current* wandering pattern. The test is a
   negative control: it locks in the symptom so Option B's fix
   is detectable. Mark with `#[ignore]` and a comment pointing
   at this note until B lands; then flip the assertion.
2. Land Option B as the durable fix. Validate against the
   corpus inertia gate and the existing 208 tests.

Option C stays in the queue, untouched, until evidence forces it.

## 7. Open questions

1. **Should the floor be per-front (using assembled-front infnorm)
   or global (matrix infnorm)?** SSIDS uses `small = 1e-20` flat;
   MUMPS uses `SEUIL` derived from `DKEEP(1)` which is computed
   once per matrix. Global is simpler and matches MUMPS; per-front
   is defensible for matrices with widely varying block magnitudes
   but introduces a per-front prologue cost. Default to global
   unless evidence demands per-front.

2. **What does `eps_relative` default to?** `ε^(2/3) ≈ 3.7e-11`
   matches MUMPS's heuristic. SSIDS uses a flat `1e-20` and trusts
   the `u`-relative threshold to do the work. For feral, given
   we already have `pivot_threshold = 1e-8` as the issue #2 default,
   the right combination needs an empirical sweep on the corpus.

3. **Does this affect ripopt's `pivtol_max = 0.5` cap?** No
   directly — `pivtol_max` is a column-relative cap that bounds
   `pivot_threshold` from above. The norm-relative floor is an
   absolute floor, orthogonal to `pivot_threshold`. Both can fire.

4. **Does this need to change the inertia of any matrix that
   currently passes the MUMPS-consensus gate?** Probably yes —
   matrices currently in the "feral matches none of the oracles"
   bucket (2 of 102 in `inertia-triage-2026-04-27.md`) may move
   into the "feral matches MUMPS" bucket. Matrices currently in
   "feral matches SSIDS but not MUMPS" may move toward "feral
   matches MUMPS instead". This is the intended direction. Flag
   any matrix that moves *away* from oracle consensus as a
   regression and investigate.

## 8. Out of scope

- Fix B (dense `count_2x2_inertia` `a00`-rule bug): independent,
  tracked at `2x2-bk-inertia-accounting.md` §6. Does not affect
  the sparse path under report.
- Migrating ripopt's `feral_direct.rs` to drop the explicit
  `zero_tol = 1e-10` override: ripopt-side cleanup, post-fix.
- `pivot_threshold` default re-tuning: issue #2 already set
  `1e-8` on `NumericParams::default()`; do not re-litigate here.

## References

- Wächter & Biegler 2006, *On the implementation of an
  interior-point filter line-search algorithm*, §3.1
  `[citet:wachter2006implementation]`.
- Bunch & Kaufman 1977, *Some stable methods for calculating
  inertia and solving symmetric linear systems*
  `[citet:bunch1977stable]`.
- Duff & Reid 1983, *The multifrontal solution of indefinite
  sparse symmetric linear equations*, §4.3 (delayed pivots)
  `[citet:duff1983multifrontal]`.
- Hogg & Scott 2013, *Pivoting strategies for tough sparse
  indefinite systems*, §3 `[citet:hogg2013pivoting]`.
- MUMPS 5.8.2 source `dfac_front_aux.F:1494-1495` (`SEUIL`),
  `dini_defaults.F:1093` (`DKEEP(1)` derivation),
  `dfac_driver.F:472-497` (threshold resolution).
- SPRAL SSIDS source `ldlt_tpp.cxx:89-119` (`test_2x2`),
  `:226` (1×1 threshold), `datatypes.f90:260-262` (defaults).

## 9. Addendum 2026-05-10 — Option B invalidated by empirical sweep

After landing the standalone reproducer (Option A), two diagnostic
sweeps in `src/numeric/factorize.rs::tests::issue_5_mss1_*_sweep_diagnostic`
demonstrate that **Option B does not address the wandering symptom**.

| `zero_tol`  | `pivot_threshold` | n+ trace                                |
|-------------|-------------------|------------------------------------------|
| 1e-10       | 0.0               | [93, 91, 94, 90, 92, 90, 90, 90, 90, 90] |
| 1e-2        | 0.0               | [93, 91, 94, 90, 92, 90, 90, 90, 90, 90] |
| 1e-10       | 1e-8              | [96, 98, 97, 91,101, 98, 94, 98, 90, 90] |
| 1e-10       | 0.5               | [93, 95, 96, 92, 93, 98, 95, 94, 93, 90] |

The wandering pivots have magnitudes O(1) and are not affected by any
absolute floor in the range we tested. Raising `pivot_threshold`
*amplifies* the wander rather than damping it (more 2×2 pivots get
rejected and re-routed through 1×1 splits, but the splits are equally
ambiguous on a rank-deficient J and BK lacks the inertia continuity
property under perturbation).

What is happening: with the (2,2)-block −1e-8 stripped, J's
structural rank deficiency (rank ≈ 45 of m=73) makes BK's pivot
choice ambiguous — multiple valid factorizations exist, each with
slightly different (pos, neg, zero) categorizations. Floating-point
perturbations from δ_w propagate through the Schur updates and
deterministically (but unstably) select different pivot orderings.
The resulting inertia jitters ±5 around the structurally correct
(90, 45, 28) without any individual pivot being "wrong" by the
α-test or growth-bound criteria.

When the corpus matrix's (2,2) block −1e-8 is **kept** (Ipopt's
default static δ_c regularization), every δ_w in the sweep produces
clean (90, 73, 0). That is Ipopt's static regularization doing its
job: breaking the rank-deficiency makes the BK pivot choice
deterministic.

### What the canonical solvers actually do

Investigated 2026-05-10 via `mumps-expert` and `ipopt-expert` agents.
Cross-checked against MUMPS 5.8.2 source and Ipopt 3.14's MA57
wrapper.

**MUMPS 5.8.2** (`ref/mumps/src/dfac_front_aux.F:1590-1619`,
`DMUMPS_FAC_I_LDLT`):

- 2×2 blocks are **atomic**. No eigenvalue split.
- Acceptance tests (in order): (a) `sqrt(|DETPIV|) ≤ SEUIL` — null
  test using `sqrt(|det|)` as a *proxy* for `|λ_min|`; (b)
  Duff-Reid 2×2 growth bound `(|d22|·RMAX + AMAX·TMAX)·UULOC > |DETPIV|`
  and its symmetric pair. With the SYM=2 default `CNTL(3)=CNTL(4)=0`
  the `sqrt(|det|)` test is effectively `|DETPIV|=0` only — so a
  near-singular 2×2 with `det≈ε` and `λ_max=O(1)` is **accepted** and
  contributes `(1+,1-,0)` whenever `det<0`, i.e., exactly the
  non-monotone bookkeeping observed here. The `sqrt(|det|) ≤ SEUIL`
  bound is loose precisely because `λ_min·λ_max = det` ⇒
  `sqrt(|det|) = sqrt(|λ_min|·|λ_max|)` overestimates `|λ_min|` for
  unbalanced pivots.
- Inertia: Sylvester via `det` and `sign(D22)`
  (`dfac_front_aux.F:1613-1619`):
  ```
  IF (DETPIV < 0)        NNEGW += 1   ! (1+, 1-)
  ELSE IF (D22 < 0)      NNEGW += 2   ! (0,  2-)
  ELSE                   /* none */    ! (2+, 0)
  ```
  Matches feral's `count_2x2_inertia_val`.
- Rank-revealing mode (`ICNTL(56)≠0` → `KEEP(19)≠0`,
  `dfac_front_aux.F:1513-1517`): rejects any candidate with
  `max(AMAX, RMAX, |PIVOT|) ≤ SEUIL` and defers to parent. This is
  the closest MUMPS comes to "indefinite-aware mode for KKT".
- Static pivoting (`CNTL(4)=SEUIL>0`, see `:1326-1328`,
  `:821-823`): replaces too-small diagonals with `±CSEUIL` rather
  than rejecting them. Counted in `NBTINYW`.

**MA57** (Duff 2004 ACM TOMS 30(2) §3.3, plus Ipopt wrapper at
`ref/Ipopt/src/Algorithm/LinearSolvers/IpMa57TSolverInterface.cpp`):

- 2×2 blocks **atomic**. No eigenvalue split.
- `CNTL(1) = pivtol` (Ipopt default `1e-8`, max `1e-4`,
  `:200-213`); `CNTL(2) = sqrt(eps)` tiny-pivot threshold.
- `INFO(24) = NEIG` reported via Sylvester (sign of `det` + sign of
  `D22`).
- Ipopt's `IncreaseQuality()` (`:821-835`):
  ```
  pivtol_ = min(pivtolmax_, pivtol_^0.75)
  ```
  geometric ramp toward `1e-4`. This is MA57's primary lever
  against wandering inertia.

**Ipopt's escalation stack** when MA57 returns `WRONG_INERTIA`
(`ref/Ipopt/src/Algorithm/IpPDFullSpaceSolver.cpp`):

1. **Too few negatives** (singular signal, `:541`):
   `IncreaseQuality()` (bump `CNTL(1)`), retry. If quality saturated:
   `PerturbForSingularity` (escalate `δ_c`, `:568`).
2. **Otherwise wrong** (`:579`): `PerturbForWrongInertia`
   (escalate `δ_w`).
3. **Inertia-free escape hatch** (`:592-639`,
   `neg_curv_test_tol`, default 0 = off): Zavala-Chiang curvature
   test on the computed step. Used when the linear solver doesn't
   report inertia (e.g. PARDISO selective inertia, iterative
   methods).

### Synthesis

Neither MA57 nor MUMPS implements eigenvalue-aware 2×2 splitting.
Their strategy is to keep the linear-solver inertia output
faithful (Sylvester via det/trace, atomic 2×2) and push the
correction loop one level up into the IPM driver. Implementing
Option 3 (eigenvalue-aware split) in feral would break this
symmetry and create inertia outputs that diverge from the canonical
Fortran solvers — violating the corpus consensus invariant in
`CLAUDE.md` and `dev/research/inertia-triage-2026-04-27.md`. **Do
not pursue Option 3.**

Option 2 (force-1×1 mode) is similarly non-canonical and would
generate inertia outputs no oracle would corroborate. Defer
indefinitely.

### Decision (2026-05-10)

1. **No feral-side code change for issue #5.** The reproducer test
   and diagnostic sweeps (committed `c002bec`, `cc8a45b`) document
   the symptom and demonstrate why the in-kernel levers don't
   touch it.
2. **Recommendation is upstream:** ripopt's `PDPerturbationHandler`
   should follow Ipopt's escalation stack. The cleanest fix is
   `PerturbForSingularity`-style δ_c bump on the first
   `WRONG_INERTIA` signal where neg < expected (the structural
   rank-deficiency case), in addition to the existing `δ_w`
   escalation.
3. **Track-but-don't-block:** Option B (norm-relative pivot floor /
   MUMPS SEUIL analog) remains a defensible *general* improvement
   for matching MUMPS's null-pivot bookkeeping on the broader
   corpus. If pursued, it must be validated against the consensus
   inertia gate (no regressions on the 154k-matrix corpus).
   Independent of issue #5; do not bundle.
4. **Leave the test in place as a regression guard.** It locks in
   the symptom under the `Identity / ForceAccept / pivot_threshold=0`
   configuration. If a future change happens to flip it to monotone
   non-decreasing, the assertion message directs the next agent to
   flip the assertion accordingly.

Issue #5 is closed on the feral side. Recommended user-facing
disposition: respond on the GitHub issue with this analysis and
suggest the ripopt-side `PerturbForSingularity` fix.
