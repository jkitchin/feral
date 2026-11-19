# Issue #237 step 4 — opt-in InfNorm (Knight–Ruiz) scaling reuse

## Problem

After PR #238 the ∞-norm scaling is still recomputed on every
factorization: 32–36 ms of a 182 ms factorization at 16 threads on the
#237 KKT (reporter's measurement), 73–84 ms of 254–279 ms here (chain
KKT, MetisND, 4 threads). It is cap-bound (10 passes, never converges
after the first IPM iterate), so warm starts and lower caps do not help
(tried-and-rejected 2026-08-09, 2026-08-15). The remaining lever is to
not recompute it: #202 Defect 1 and #237 proposal 1.

## Constraints from history

- `dev/decisions.md` (#49 entry): the MC64 cache once replayed a stale
  InfNorm vector *unintentionally*; called "wrong by construction",
  benign in every measured residual/inertia. The fix gated that cache to
  MC64 only. A reuse here must be explicit, opt-in, and guarded.
- Default behaviour must stay bit-identical: this container cannot run
  the corpus gate (inertia 100 %, residual partitions), which any change
  to default numerics needs.

## Why reuse is safe for inertia

`D·A·D` with any positive diagonal `D` has the inertia of `A`
(Sylvester). A stale `D` is a valid scaling; the solve undoes it
(`ScalingStrategy::External` → `ScalingInfo::Applied`). What can change
is numerical quality: pivot choices, growth, residual before refinement.
The #237 reporter measured `Identity` (no scaling at all) on their KKT:
same iteration count, refined residual no worse. A recent `D` is much
closer to a fresh one than `Identity` is.

## Design

`Solver::with_infnorm_reuse(Option<InfNormReuse>)`, default `None`
(unchanged behaviour, bit-identical).

`InfNormReuse { max_age: usize, max_row_ratio: f64 }`:
- `max_age`: consecutive factorizations that may reuse one computed
  vector before a fresh computation is forced.
- `max_row_ratio` (guard, ≥ 1): before reusing, one KR pass of the
  cached `d` over the current values computes the row ∞-norms of
  `D·A·D`. Reuse only if every nonzero row norm lies in
  `[1/ρ, ρ]`. A fresh KR at its cap leaves rows within ~1–3 % of 1, so
  ρ bounds how much worse than fresh the reused scaling may be. Cost:
  one pass (≈1/10 of a fresh computation), using the step-3 parallel
  pass on a pool. `f64::INFINITY` disables the guard.

Engages only when the effective strategy is `InfNorm` (explicit, or
`Auto` pinned to it after the first call). Cache keyed on the pattern
fingerprint; cleared on pattern change and on
`invalidate_symbolic_cache`. On a hit the vector is injected as
`ScalingStrategy::External` (the MC64 cache's mechanism). A counter
`infnorm_reuse_hit_count()` mirrors `mc64_cache_hit_count()`.

Interaction: the MC64 cache only engages on MC64 routes (#49 gate:
`External` → not cached as MC64). The #65 MC64 retry and the sticky
`Auto` pin are unchanged: the pin is decided on the first call, which
never reuses.

## Tests (oracles)

- Guard on hand matrices: `diag(4, 9)` with `d = (1/2, 1/3)` → rows
  (1, 1) pass at ρ = 1.01; with `d = (1, 1)` → rows (4, 9) fail at ρ = 2.
- Default `None`: stored scaling equals a fresh `compute_infnorm` of
  each matrix bitwise; hit count 0.
- Reuse on, small drift: hit; stored scaling equals the previous call's
  bitwise; inertia equals the hand / analytic oracle.
- One row scaled ×100: guard rejects; scaling equals fresh
  `compute_infnorm` bitwise.
- `max_age`: forced fresh computation after `max_age` reuses.
- Pattern change: cache cleared, fresh computation.
- Chain KKT harness: inertia equals the analytic oracle with reuse on;
  refined residual reported.
