# Task #19: Dense ACOPP30 Gap — Expert Consultation and Decision

**Date**: 2026-04-14
**Status**: Decision recorded. Action: reroute bench dense validation through `factor_frontal`.

## The gap

On the 154588-matrix KKT bench corpus, feral's dense path (`src/dense/factor.rs::factor`)
has a correctness gap specific to 67 ACOPP30 variants (AC optimal power flow
KKT systems, n=209 each). The pattern is uniform:

| metric            | dense `factor()`   | sparse `factorize_multifrontal` | MUMPS 5.8.2 oracle |
|-------------------|--------------------|---------------------------------|--------------------|
| inertia           | (72, 137, 0)       | (71, 137, 1)                    | (71, 137, 1)       |
| relative residual | ~2.7e-2            | ~1.4e-14                        | ~1.4e-9            |

The dense path matches a sidecar-provided inertia that disagrees with MUMPS +
SSIDS, while the sparse path and both canonical solvers agree on the other
side. Per `dev/research/shared-failure-triage.md`, the sidecar is wrong for
these 67 matrices — the canonical `(71, 137, 1)` is correct.

## What was already tried

Two fixes were attempted in session 2026-04-13-05 (committed as rejected in
`dev/tried-and-rejected.md`):

1. **Duff-Reid u backstop**: clamp `u` to `max(pivot_threshold, sqrt(eps))` so
   the 2×2 growth bound fires at u=0. Fails standalone because when the 2×2
   is rejected, the fallback `do_1x1_pivot(a, k, ...)` divides by `a[k,k] = 0`.
2. **Reducible-column floor**: extend `if gamma0 == 0.0` to
   `if gamma0 ≤ sqrt(eps)` and force-zero the diagonal below the same floor.
   Closes ACOPP30 on the triage (2.8e-2 → 1e-13) but regresses the full bench
   by +6998 failures (dense match 99.0% → 94.5%). Root cause: the absolute
   sqrt(eps) threshold assumes ||A||∞ ~ 1 and the bench corpus is not
   equilibrated.

The diagnostic trace at ACOPP30_0026 k=58:

```
2x2 block = [[ 0       , -4.16e-15 ],
             [ -4.16e-15, -6.08e-9 ]]
|det|     = 1.73e-29   (passes count_2x2_inertia eps² floor by 350×)
|L21|     ~ 10²⁹       (destroys trailing submatrix)
```

## Expert consultation (2026-04-14)

Three reference-solver experts consulted in parallel: MUMPS 5.8.2 (Fortran),
SPRAL SSIDS (C++/Fortran), and faer (Rust). Each was asked whether their solver
has a separate "dense single-front" code path, what it does with the specific
2×2 block feral is hitting, and whether closing the gap is worth pursuing.

### MUMPS (mumps-expert)

**"MUMPS has no analogue to your `factor()` entry point. Every KKT that MUMPS
solves goes through the delayed-pivot multifrontal path."**

Key findings:

- `dfac_front_LDLT_type1.F` is the *only* type-1 frontal routine. A dense input
  becomes a single-node tree; there is no dense-specific kernel.
- At the root node, MUMPS sets `AVOID_DELAYED=.TRUE.` (`cfac_par_m.F:353,582`),
  which forces `STATICMODE=.TRUE.` and `SEUIL_LOC = max(SEUIL, epsilon(SEUIL))`
  (`dfac_front_LDLT_type1.F:129-134`).
- The Duff-Reid 2×2 test at `dfac_front_aux.F:1599-1606` has an **explicit
  hardwired rejection** when `ABSDETPIV == RZERO`, independent of `UULOC`:
  ```fortran
  IF ((abs(A(POSPV2))*RMAX+AMAX*TMAX)*UULOC.GT.
   &    ABSDETPIV .OR. (ABSDETPIV .EQ. RZERO) )  THEN
     GO TO 460      ! reject this pivot
  ENDIF
  ```
- On rejection, `INOPV=1 → -1` routes to the forced-1×1 branch at
  `dfac_front_aux.F:1246-1270`, which accepts the diagonal as a 1×1 pivot and
  **replaces `|A(APOS)| < SEUIL` with `±SEUIL`** (static pivoting à la SuperLU),
  incrementing `NBTINYW`.
- MUMPS always applies a fill-reducing ordering (METIS/AMD) in `ana_set_ordering.F`;
  it does not degenerate to natural order even for small n=200.
- MUMPS never factors a 2×2 whose `|det|` is smaller than the Duff-Reid bound —
  not even at the root. Iterative refinement cannot rescue an L21 scaled by 10²⁹.

**Verdict**: *"Deprecate the dense entry point for KKTs and route everything
through `factor_frontal`."*

### SPRAL SSIDS (spral-expert)

**"SSIDS has no separate dense path. `ldlt_tpp.cxx`'s own header comment is
*'Simple LDL^T with threshold partial pivoting. Intended for finishing off
small matrices, not for performance'*. That IS the dense kernel."**

Key findings:

- `ldlt_tpp::test_2x2` at `ldlt_tpp.cxx:89-119` applies **three independent
  guards**, including a scale-invariant cancellation-aware determinant floor:
  ```
  |detpiv| < max(small, |detpiv0|/2, |detpiv1|/2)    // line 106
  ```
  For feral's ACOPP30 block with `detpiv0 = 0` and `detpiv1 ≈ 3e-21` after
  scaling, this test fails and the 2×2 is rejected — **no absolute
  sqrt(eps) tuning needed**. The critical insight: `detscale = 1/maxpiv`
  (`ldlt_tpp.cxx:101`) makes the comparison scale-invariant by construction.
- On rejection, `ldlt_tpp` advances `p`, tries a different pair, and if every
  `(t,p)` combination fails, breaks out with `nelim < n`. At the root, leftover
  columns are simply un-eliminated — SSIDS does not force-accept them.
- SSIDS ships `options.scaling` with 4 nonzero modes for ill-conditioned KKTs.
  Ruiz (mode 4, similar to feral's Knight-Ruiz) is the weakest; **matching-based
  MC64 scaling (modes 1-3) is what's recommended for indefinite KKT workloads
  like AC optimal power flow** (`docs/Fortran/ssids.rst:383-406`).
- SSIDS uses `options.small = 1e-20` as an absolute "is this zero" floor, but
  all stability logic is expressed as ratios against the local pivot block or
  trailing submatrix max — never against a precomputed matrix norm.

**Verdict**: *"Delete or stub `factor()`. Port the scale-invariant `test_2x2`
cancellation-aware det floor into `factor_frontal`. Add MC64-style matching
scaling for KKT workloads."*

### faer (faer-expert)

**"faer's dense BK kernel has no growth bound, no determinant floor, and no
regularization. It would hit the exact same pathology on a natural-order KKT."**

Key findings:

- faer's `faer::linalg::cholesky::bunch_kaufman::factor` implements pure
  Bunch-Kaufman partial pivoting with only the `alpha` test (line 271-273).
  No Duff-Reid, no cancellation floor, no `|det|` check.
- faer's sparse `factorize_supernodal_numeric_intranode_lblt`
  (`sparse/linalg/cholesky.rs:3470`) calls the **exact same** dense kernel
  (`cholesky_in_place` at line 3658). The only thing that makes faer's sparse
  path survive KKTs is **AMD preordering in the symbolic phase** — not a better
  kernel.
- faer has `LdltRegularization` (`ldlt/factor.rs:664-694`) with signed dynamic
  regularization for the pure-LDLT path, but **no `LbltRegularization`**. The
  LBLT path is deliberately unguarded.
- There is no "dense BK with fill-reducing preorder" API in faer. Pathological
  KKTs are expected to go through the sparse supernodal entry point.

**Verdict**: *"Don't fix the dense path. It's a leaf kernel, it has no user,
and the pathology is a fundamental limitation of partial-pivoting BK on
natural-order KKT. Route pathological KKTs through the sparse supernodal path.
If you must add a safety net, use signed dynamic regularization at 1×1 steps
(mirroring `LdltRegularization`) rather than patching the 2×2 rejection."*

## Consensus

All three experts independently converged on the same recommendation:

1. **None of the three references have a separate dense path.** MUMPS is
   multifrontal always; SSIDS's `ldlt_tpp` is explicitly "finishing off small
   matrices"; faer's dense BK is a leaf kernel called by the sparse driver.
2. **feral's dense `factor()` combines four things none of the references
   combine**: natural column order (no AMD/METIS), `pivot_threshold = 0.0`,
   no `|det| == 0` explicit rejection, no static pivoting floor.
3. **The two failed patch attempts are the signature of an under-constrained
   kernel**, not a single localizable bug. Each isolated fix trades ACOPP30
   for regressions elsewhere.
4. **The right architectural move is to deprecate the dense `factor()` entry
   point** as a bench validation oracle and route dense validation through
   `factor_frontal`. Keep `factor()` for unit tests on well-conditioned BK
   examples from the Bunch-Parlett / Ashcraft-Grimes-Lewis literature.

## Decision

Close task #19 as **"not a real bug — kernel under-constrained by design"**.

The ACOPP30 residual pattern is a faithful report that applying unpivoted
dense Bunch-Kaufman at `u=0` to a natural-order KKT matrix with near-null
Hessian rows adjacent to constraint multiplier rows will not produce a stable
factorization. The references agree. feral's sparse path already handles
these matrices correctly (1e-14 residuals). The fix is to stop pretending
the dense path should be a second validation oracle.

## Action plan

**Step 1 (this session)**: Add `factor_single_front` to `src/dense/factor.rs`
as a thin wrapper that applies Knight-Ruiz equilibration then delegates to
`factor_frontal(may_delay=false)`. Reroute `src/bin/bench.rs`'s dense
validation loop through this wrapper using `params_kkt_sparse`
(pivot_threshold=0.01). Verify the full bench shows:

- "dense" inertia match improves from 99.0% → matching sparse (~99.0%)
- "dense" worst residual drops from 2.80e-2 (ACOPP30_0026) → comparable to sparse
- The 67 ACOPP30 mismatches disappear from the shared failures section
- No new regressions

**Step 2 (separate future task)**: Port SSIDS's scale-invariant det floor
`|detpiv| < max(small, |detpiv0|/2, |detpiv1|/2)` into `factor_frontal`'s
2×2 test. This hardens the *sparse* production path too, not just the
dense bench. Scale-invariant by construction — no regression trap from
absolute thresholds.

**Step 3 (future Phase 2.4 item)**: Implement MC64-style matching-based
scaling for the sparse analysis phase. Knight-Ruiz is insufficient for
ACOPP30-class KKTs; SSIDS docs explicitly recommend matching-based scaling
for AC optimal power flow workloads.

## Files cited

### MUMPS 5.8.2
- `src/dfac_front_LDLT_type1.F:129-134,418-437`
- `src/dfac_front_aux.F:1246-1270,1590-1606,1660-1669`
- `src/cfac_par_m.F:353,582`

### SPRAL SSIDS
- `src/ssids/cpu/kernels/ldlt_tpp.cxx:89-119,166-270,239-260`
- `src/ssids/cpu/kernels/block_ldlt.hxx:213-218,289-414`
- `src/ssids/cpu/factor.hxx:75-113`
- `src/ssids/datatypes.f90:243-276`
- `docs/Fortran/ssids.rst:383-406`

### faer
- `faer/src/linalg/cholesky/bunch_kaufman/factor.rs:27-36,271-273,785-901,903-912`
- `faer/src/linalg/cholesky/ldlt/factor.rs:664-694`
- `faer/src/sparse/linalg/cholesky.rs:3470-3719,4155-4229`
