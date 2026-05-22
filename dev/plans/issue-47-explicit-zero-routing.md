# Plan — issue #47: explicit zeros flip the scaling-strategy router

Date: 2026-05-21
Journal: `dev/journal/2026-05-21-04.org` §17:30 → §20:30
Related: issue #45 (MC64 catastrophic-spread guard),
`dev/plans/mc64-scaling-spread-guard.md`

## Problem (empirically pinned)

POUNCE's CHO `parmest` backend refills a fixed KKT sparsity pattern each
IPM iterate; equality-constraint `(2,2)` slots that are mathematically
zero arrive as explicit stored `0.0`. `probe_explicit_zeros` on the
iter-0 CHO KKT (`n=43332`):

```
route stripped             diag_only=0      (0.000)  max_col_nnz=10861  -> InfNorm
route explicit zeros kept  diag_only=21660  (0.500)  max_col_nnz=10866  -> Mc64Symmetric
```

`pick_scaling_strategy` (`src/scaling/mod.rs:601`) is a **structural**
router: it counts *stored* entries. A constraint column of the zero-`(2,2)`
block, stored lower-triangle, holds only its diagonal. Kept: that
diagonal is an explicit `0.0` → `nnz_col=1`, counted `diag_only`.
Stripped: the `0.0` is gone → `nnz_col=0`, not `diag_only`. So
`diag_only/n` is 0.500 vs 0.000, `has_slack_mass` (≥0.30) flips, and the
two matrices route to different strategies.

The kept matrix then runs MC64, degenerates (issue #45 ≈3e82 blow-up),
falls back to InfNorm with `ScalingInfo::Mc64FallbackToInfnorm`, and the
B2 scaling cache — which only populates on `Applied` — never populates.
The full ~345 ms Hungarian reruns every warm refactor (warm 15 ms
stripped vs 370 ms kept). That wasted per-factor Hungarian is the whole
~2× POUNCE slowdown.

## Root cause

An explicit stored `0.0` is not coupling and not mass. `pick_scaling_strategy`
is a structural router that nonetheless lets value-only zeros change its
decision. The scaling strategy is a *numeric* choice, so the router
*should* be value-aware — `nnz_col` / `diag_only` / `max_col_nnz` must
count structurally meaningful (nonzero) entries.

This was checked against MUMPS 5.8.2 and SPRAL SSIDS (journal §19:10):
neither strips zeros from the matrix or the *analysis* pattern (ordering
/ etree must stay value-blind — the pattern is the contract). Both
handle zeros inside the scaling/matching kernels. feral's
`mc64::build_cost_graph` already drops explicit zeros; the one remaining
value-blind spot is this router. The fix matches the reference behaviour:
keep analysis value-blind, make the scaling decision value-aware.

## Fix — value-aware `pick_scaling_strategy` (Option A, root cause)

`src/scaling/mod.rs`, `pick_scaling_strategy` only. Replace the per-column
`nnz_col = end - start` and `row_idx[start] == j` test with a scan that
skips `values[k] == 0.0`:

- `nnz_col` = count of stored entries with `value != 0.0`.
- a column is `diag_only` iff `nnz_col == 1` and that one nonzero entry
  is on the diagonal.
- `max_col_nnz` tracks the nonzero count.

O(n+nnz), no allocation (the doc comment already claims O(n+nnz)).
Behaviour is identical for any matrix with no explicit zeros — the entire
existing validation panel (clnlbeam, VESUVIO, ACOPP30, …) is unaffected.
The kept CHO KKT then routes to InfNorm, identically to the stripped one.

Rejected alternative — Option B (negative cache: remember MC64-degenerate
fingerprints, skip the warm-refactor Hungarian). Localized, no #45 test
churn, but it treats the symptom (wasted recompute) not the cause
(mis-routing). A is the root-cause fix; recorded in journal §20:30.

## Tests first

Oracle: hand calculation — "an explicit stored `0.0` carries no
structural mass / no coupling." Admissible per CLAUDE.md (hand
calculation is an external oracle source). Routing is a discrete
heuristic, not a numerical result.

New tests in `src/scaling/mod.rs` `mod tests`:

- `pick_scaling_strategy_explicit_zero_diag_not_slack_mass` — build an
  arrow KKT two ways: (1) constraint columns store an explicit-zero
  diagonal, (2) the same columns are structurally empty. Both must route
  identically (→ InfNorm), and a third variant with *nonzero* constraint
  diagonals must route to `Mc64Symmetric`. This is the #47 reproduction
  in miniature.
- `pick_scaling_strategy_explicit_zero_offdiag_ignored` — an explicit-zero
  off-diagonal entry must neither inflate `max_col_nnz` nor disqualify a
  `diag_only` column.

## Issue #45 test fallout (must be handled in the same change)

`build_synth_kkt` (the #45 spread-guard oracle, `mod.rs:700`) pushes
explicit-zero state and constraint diagonals (lines 735, 750). Under
value-aware routing it now routes to InfNorm, so T2/T3's precondition
`assert_eq!(pick_scaling_strategy(&csc), Mc64Symmetric)` would fail and
T4 would no longer reach the guarded path.

Resolution: give `build_synth_kkt` an `nslack` parameter that appends
`nslack` degree-1 columns with a *nonzero* diagonal (value 1.0),
disconnected from the parameter/state/constraint chain. These model the
bound slacks of a bound-constrained parameter-estimation KKT (real slack
mass + zero equality duals) — a strictly more realistic KKT than the
pure-equality original. They:

- restore `Mc64Symmetric` routing under value-aware counting (real
  `diag_only` mass);
- do not perturb the chain submatrix — MC64's matching decomposes over
  the disconnected slack block (each slack matches itself, `log|1|=0`),
  so the chain potentials, and hence `scaling_spread`, are unchanged;
- T2/T3 already re-assert the spread oracle inline
  (`assert!(mc_spread > 1/EPS …)` / `< 1/EPS`), so they are
  self-validating — if the augmentation perturbs the spread the tests
  fail loudly rather than passing on a stale oracle.

`nslack` is sized so `nslack/(168+nslack) ≥ 0.30` with margin (120 →
0.417). T4 (`base=1.1`) factors+solves; 120 unit columns add 120 trivial
positive pivots and the residual identity oracle still holds.

The `MC64_SPREAD_GUARD` itself (`mod.rs:412`) is unchanged and stays as
defense-in-depth.

## Session verification

- `probe_explicit_zeros` — kept matrix must now route to InfNorm,
  `scaling_info=Applied`, `mc64_cache_hits` 1,2,3, warm refactor ~15 ms
  (matching stripped). This is the #47 fix proven on the real KKT.
- Full `cargo test` — all `pick_scaling_strategy_*`, Policy-4, MSS1 /
  ACOPP30 / HS75, and #45 T1–T4 tests green.
- `cargo run --bin bench --release` — inertia-match and residual-pass
  percentages must not regress (corpus `.mtx` files carry no explicit
  zeros → routing unchanged for them).

## Risk

Low. The change only differs from current behaviour on matrices that
store explicit zeros; the parity corpus has none. The kept CHO KKT
converges to the *same* InfNorm scaling the stripped one already uses
(residual 5e-9, unchanged). The #45 test augmentation is self-validating
via the existing inline spread assertions.

## Out of scope

- Issue #44 (NARX_CFy large-front per-factor cost) — distinct, stays open.
- A negative cache for MC64-degenerate fingerprints — recorded as Option
  B, not needed once routing is value-aware.
