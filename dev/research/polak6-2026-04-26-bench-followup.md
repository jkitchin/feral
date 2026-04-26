# POLAK6_0021 — 2026-04-26 Bench Follow-up

**Date:** 2026-04-26
**Authorized by:** `dev/sessions/2026-04-26-02.md` §"Next Session Should" item 4
("POLAK6_0021 sparse residual blow-up (2.99e8). Drill before declaring sparse
residual stable on `kkt`").
**Builds on:** `dev/research/polak6-triage-2026-04-19.md` (root cause: matrix
is mathematically ambiguous at f64 precision; consensus excluded).
**Diagnostic source:** `src/bin/polak6_diag.rs` (extended this session with the
exact bench code path).
**Sidecar:** `data/matrices/kkt/POLAK6/POLAK6_0021.{json,verdict.json}`.

## TL;DR

The 2026-04-26-02 end-to-end bench surfaced `POLAK6_0021` as the worst sparse
residual (`2.99e8`). This is **not a new failure**: it is the same matrix the
2026-04-19 triage classified as **excluded from oracle consensus** (`rmumps`,
`feral`, `mumps`, `ssids` all give different inertias). It became visible
today only because the bench finally completed end-to-end after the OOM fix.

The 2.99e8 number is also **larger than what the verdict file records**
(feral residual 1.9e-16 there), because the bench uses the **actual Ipopt
sidecar RHS** with magnitudes up to 8.2e41, while the verdict-time
diagnostic used synthetic RHS. The relative residual on a RHS with
||b|| ≈ 1e42 is dominated by f64 rounding in the `b - A·x` computation
itself.

The 99.8% sparse residual rate on the kkt corpus is unchanged from the
prior baseline. Dense BK passes this matrix cleanly (residual 9.21e-17)
because dense BK pivots freely; sparse multifrontal can't reach the
(1, 4) hyperbolic 2×2 across non-contiguous supernode columns.

## 1. Bench-path reproduction

Augmented `src/bin/polak6_diag.rs` with the exact bench path: `Auto`
scaling + sidecar RHS + `solve_sparse_refined`. Output:

```
shape: n=9 stored_nnz=32
raw |diag|: min=1.000e-4  max=1.326e42  range=1.325e46
diag_only=4 / n=9 = 0.444 (>= 0.30 → adaptive routes to MC64)

--- bench-path reproduction (Auto + sidecar RHS + refined solve) ---
  sidecar RHS len=9
  RHS |b|: min=3.305e-3 max=8.239e41
  inertia=(4, 4, 1)  expected=(5, 4, 0)
  direct solve  ||b - A x|| / ||b|| = 1.305e13
  refined solve ||b - A x|| / ||b|| = 2.988e8
  bench tol     = n * eps * 1e6   = 1.998e-9
  PASS=NO
```

Refinement halves the magnitudes (1.31e13 → 2.99e8) but cannot recover
from a factor with a wrong-sign pivot. The factor is non-singular —
`ZeroPivotAction::ForceAccept` replaced the zero-pivot row with a
regularized value — but the resulting sign-pattern is wrong, so
refinement converges to a non-solution.

## 2. Bench / verdict disconnect

The bench compares factored inertia against `data/matrices/kkt/POLAK6/POLAK6_0021.json`,
which records `rmumps`'s `(5, 4, 0)`. The verdict file
(`POLAK6_0021.verdict.json`) explicitly classifies this matrix as
**excluded** from consensus:

```
"consensus_inertia": null,
"inertia_agreement": "none",
"inertia_dissenters": ["feral", "mumps", "ssids"],
"verdict": "excluded",
"oracles": {
  "rmumps": { "inertia": [5, 4, 0] },
  "feral":  { "inertia": [5, 1, 3] },
  "mumps":  { "inertia": [5, 1, 3] },
  "ssids":  { "inertia": [6, 3, 0] }
}
```

The bench therefore reports a "failure" against an oracle the verdict
says is **not authoritative** for this matrix. This wasn't visible in
prior sessions because the sparse loop never completed — the bench's
worst-residual top-10 list always lived in dense-only territory
(`ERRINBAR_0824`, `1.87e-4`).

## 3. Three threads not pursued tonight

These are all logged in `dev/sessions/2026-04-26-02.md` for future
consideration. None is small.

### 3a. Higher pivot threshold

Setting `pivot_threshold = 1.0` (full partial pivoting) would force
the sparse path to delay any pivot whose column-relative magnitude
falls below the threshold, giving the (1, 4) 2×2 a chance to land at
a parent supernode. Cost: substantial fill-in increase across the
corpus, since any pivoting decision is more conservative. Need to
characterize the corpus-wide fill-in vs residual tradeoff before
adopting.

### 3b. 2×2 pivots across non-contiguous supernode columns

The sparse multifrontal currently considers 2×2 pivots only within
the contiguous supernode column range. The dense BK kernel that the
multifrontal calls into has no such restriction. Lifting it requires
a dense-kernel API change so it can return a "pivot candidate" hint
plus a delayed-pivot flag, with the multifrontal driver tracking
non-contiguous pivot pairs across supernodes. Architectural — Phase
3.x candidate.

### 3c. Regularization-aware scaling

Ipopt's KKT regularization is **structurally identifiable**: the
constraint-block diagonal carries `-delta_c` exactly (here 1e-4) and
the variable-block regularization adds `delta_w` to the Hessian
diagonal. A sidecar-aware scaling step could mark these rows as
"pre-regularized, do not rescale" and let the solver focus its
scaling decision on the un-regularized portion. Cleanest in the
sense that it removes the pathological 1e46 dynamic range from the
scaling input. Limitation: only applies when the bench knows the
regularization values (i.e. via sidecar — production solver
generally doesn't).

## 4. Suggested bench enhancement (small, defensible)

The bench's "worst residual" reporting and the failure-list dump
should consult `verdict.json` and **filter out matrices marked
`excluded`** from worst-case statistics, while still reporting them
in a separate "consensus-excluded outliers" section. This avoids
chasing the non-failure that POLAK6_0021 represents under
oracle-disagreement, without losing visibility of the matrix.

Concretely:
- Load `verdict.json` alongside `.json` in `load_kkt_dir`.
- `Failure::is_excluded: bool` populated from `verdict.verdict == "excluded"`.
- Per-corpus "Worst residual" excludes excluded matrices.
- Add a separate "Consensus-excluded outliers" subsection in the
  failure analysis dump that lists them with their dissent pattern.

This is a one-session change. Keeping it scoped to the bench means
no production-code surface is touched.

## 5. Why this is not "tonight's bug"

The 99.8% sparse residual rate on `kkt` is unchanged from
2026-04-25. The 2.99e8 worst-case is a single matrix on the boundary
of f64 representability that *all four oracles disagree on*. Calling
this a feral-correctness bug requires accepting `rmumps` as ground
truth in a case where the verdict file explicitly does not. The
2026-04-19 triage already worked through this and rejected the
"matched but bad" framing; today's observation reaffirms that
conclusion under the actual bench harness.

Documented in `dev/journal/2026-04-26-02.org` @ 19:30.
