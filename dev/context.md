# FERAL Context (auto-generated)

Generated: 2026-05-20T22:51:17Z

## Latest Session
File: dev/sessions/2026-05-20-02.md
```
# Session 2026-05-20-02

## Goal
Fix the defect underlying GitHub issue #45: the CHO `parmest` KKT factors
successfully with correct inertia but the back-solve returns garbage
(residual ~7e11) while `factor()` silently reports `Success`. Issue #46
(LDLᵀ ~160× slower than MA57) was explicitly scoped OUT as a separate,
larger performance effort — "correctness before performance, always".

## Accomplished

### Root cause isolated (issue #45)
A long diagnostic chain (full trace in `dev/journal/2026-05-20-02.org`)
overturned two earlier wrong hypotheses (duplicate-coordinate doubling;
ordering needs a complete diagonal) and converged on the real defect:

- `ScalingStrategy::Auto` applies an MC64 symmetric scaling vector whose
  own spread `max|s| / min|s|` exceeds `1/EPS ≈ 4.5e15`. On the CHO KKT
  MC64 produced spread ≈ **3e82** (`min 2.89e-42 .. max 8.88e40`).
- Such a scaling is degenerate to working precision: `D = diag(s)` is
  singular, `D·A·D` underflows, Bunch-Kaufman force-accepts exact-`0.0`
  1×1 pivots (`min pivot mag = 0.00e0`), the solve is garbage, and
  `factor()` still returns `Success` with correct inertia.
- The existing issue-#24 guard missed it: `compute_scaling_auto_with_cache`
  had a fast-path `if raw_diag_range(matrix) >= RAW_GUARD(1e6) { return mc64 }`
  that committed to MC64 *without* inspecting the produced vector. The CHO
  KKT is ill-conditioned (raw range ≥ 1e6) so it took that fast-path and
  the `mc_off` catastrophe diagnostic was never reached.
- Diagnosis note: `dev/research/kkt-mc64-scaling-blowup-2026-05-20.md`.

### Fix — MC64 catastrophic-spread guard
`src/scaling/mod.rs`, `compute_scaling_auto_with_cache`:
- New `const MC64_SPREAD_GUARD: f64 = 1.0 / f64::EPSILON` (≈ 4.5036e15).
  Corpus max MC64 spread is 3.27e15 (ssine) — a 67-order gap below the
  CHO catastrophe; the guard catches CHO and clears the whole corpus.
- New `Mc64FallbackReason::Mc64ScalingDegenerate` variant.
- The MC64 branch now computes `(mc_vec, mc_info)` **once**, then
  `if scaling_spread(&mc_vec) > MC64_SPREAD_GUARD` returns the
  already-computed InfNorm vector tagged `Mc64ScalingDegenerate`.
  Placed **before** the `raw_diag_range` fast-path so it fires regardless
  of raw conditioning — that fast-path was the #45 bypass.
- `src/bin/bench_one_matrix.rs`: exhaustive `Mc64FallbackReason` match
  extended with `"mc64_scaling_degenerate"`.

### Verification
- Real CHO KKT via `probe_issue45_ordering` (added an `Auto` row to its
  scaling loop): `completed Auto` went from **relres 7.149e11 → 2.455e-8**,
  inertia (21672,21660,0) unchanged. `Auto` now == `InfNorm` on the
  diagonal-completed CHO KKT (the POUNCE live-KKT form that triggers #45).
  #45 closed.
```

## Git Status
```
eb77966 test(issue46): ground-truth probes for the zero-(2,2)-block cascade
070840b fix(ldlt): break the zero-(2,2)-block KKT delayed-pivot cascade (#46)
d432086 docs(session): checkpoint 2026-05-20-02 — issue #45 MC64 spread guard
6bda61d test(probe): add issue #45/#46 diagnostic and oracle probes
b017beb fix(scaling): guard against catastrophic MC64 scaling spread (#45)
```

## Test Status
```
