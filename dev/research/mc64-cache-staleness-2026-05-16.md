# MC64 cache staleness across warm `Solver::factor` calls

**Date:** 2026-05-16
**Issue:** #38 (Failure A); incidentally bears on #37
**Status:** Fixed in `db20166` (one-shot cache invalidation in `Solver::factor`).

## Summary

`OrderingPreprocess::LdltCompress` populates
`SymbolicFactorization::cached_mc64` at symbolic time with the iter-0
Hungarian matching, dual variables `(u, v)`, and column maxes `cmax`.
The numeric phase consumes that cache via
`compute_scaling_with_cache` on every subsequent `factor()` call.
MC64's matching and dual variables are value-dependent; in an IPM
driver — which calls `factor()` repeatedly on the same pattern with
new values — this means iter-0 scaling is silently applied to
iter-N matrices.

On the `rocket_12800.nl` reproducer (`/tmp/rkt_*.bin`, 18 dumps) the
result is silently wrong inertia starting at IPM iter #010 (38400 →
38395), a steady drift through iter #014 (→ 38145), and an
explosion at iter #017 (43.2 s wall, `negative = 31720`) — while a
fresh `Solver::new()` on the same matrix file factors in 1.6 s with
the correct `negative = 38400`. Forcing `ScalingStrategy::InfNorm`
(no cache path) keeps warm Solver stable at `negative = 38400` and
~27 ms per call, confirming the cache is the root cause.

## Failure mechanism

1. `ScalingStrategy::default() == Auto`.
2. `Auto` on arrow-KKT matrices routes to `Mc64Symmetric`
   (`pick_scaling_strategy`); the rocket KKT qualifies because it
   has 38 400 degree-≤2 slack columns out of `n = 89 601`.
3. The same degree distribution triggers
   `pick_ordering_preprocess → LdltCompress` (n ≥ 128, ≥30 % cols
   with degree ≤ 2). MC64 compresses the column graph by collapsing
   matched pairs, then runs the user-chosen ordering on the
   compressed graph.
4. `LdltCompress` stores its MC64 cache:
   ```rust
   let cache = crate::scaling::compute_mc64_cache(matrix)?;
   // ... use cache.perm to compress the pattern ...
   cached_mc64 = Some(cache);
   ```
   (`src/symbolic/mod.rs:582-599`).
5. `compute_scaling_with_cache(matrix, Mc64Symmetric, Some(cache))`
   takes the `Some` branch:
   ```rust
   ScalingStrategy::Mc64Symmetric => match cache {
       Some(c) => Ok(mc64::scaling_from_cache(c)),
       None    => mc64::compute_symmetric(matrix),
   },
   ```
   (`src/scaling/mod.rs:259-262`). `scaling_from_cache(c)` reuses
   `c.u + c.v` and `c.cmax` to rebuild the scaling vector — none
   of those reflect iter-N's values.
6. Result: the iter-N matrix is scaled by `D_0`, factored under BK,
   and the wrong scaling pushes BK into different pivot choices.
   On well-conditioned small matrices this is harmless (Sylvester
   keeps inertia invariant under any symmetric non-singular
   scaling). On real arrow-KKTs the mis-scaling rejects pivots that
   would have been safe, triggers a delayed-pivot cascade, and the
   factor cost blows up.

## Why the symptom is "drifting inertia" not "obviously wrong solve"

Mathematically `D_0 (D_0 A_N D_0)^{-1} D_0 = A_N^{-1}` exactly when
`D_0` is non-singular, so a clean factor of `D_0 A_N D_0` would
round-trip the correct solve. The corruption enters because BK
pivoting depends on relative magnitudes — and `D_0 A_N D_0` is
typically far more ill-conditioned than `D_N A_N D_N`, so BK's
pivot-rejection threshold fires on benign pivots, delays them, and
the resulting "factor" is no longer a proper Bunch-Kaufman
factorization. Once enough pivots are delayed, the diagonal-block
count stops faithfully reporting the original matrix's inertia.

## Diagnostic evidence

**Warm Solver, default `Solver::new()` config (CB=off, scaling=Auto,
parallel=on):**

| call | factor | neg | comment |
|---:|---:|---:|---|
| #000 | 0.320 s | 38400 | cold (incl. symbolic) |
| #009 | 0.023 s | 38400 | warm stable |
| #010 | 0.022 s | 38395 | **inertia drift starts** |
| #014 | 0.024 s | 38145 | drift continuing |
| #015 | 0.056 s | 37513 | cost picks up |
| #016 | 2.093 s | 35900 | cost explodes |
| #017 | 43.216 s | 31720 | runaway |

**Side-by-side warm vs `FRESH=1` (rebuild Solver each call):**

| call | warm factor | fresh factor | warm neg | fresh neg |
|---:|---:|---:|---:|---:|
| #014 | 0.024 s | 1.206 s | 38145 | **38400** |
| #015 | 0.056 s | 1.222 s | 37513 | **38400** |
| #016 | 2.093 s | 1.194 s | 35900 | **38400** |
| #017 | 43.216 s | 1.640 s | 31720 | **38400** |

**Localisation:** PAR=0 re-runs reproduce the inertia drift and cost
runaway identically, ruling out parallel-pool warm state.

**InfNorm control:** same warm Solver, `ScalingStrategy::InfNorm` (no
MC64 cache path) recomputes scaling each call — stable
`negative = 38400` and ~27 ms factor across all 18 calls. (Identity
is a confounded probe: no cache, but no scaling either, so the raw
ill-conditioned matrix cascades for matrix reasons. InfNorm is the
clean control.)

## Fix options considered

1. **Drop the cache entirely.** Loses the ~70 % symbolic-overhead
   amortisation that originally motivated the cache (compression
   reuses MC64's matching; without the cache the numeric phase has
   to recompute). That amortisation assumed re-factor on the *same*
   matrix, which never happens in IPM — so the savings were
   theoretical in the actual workload.
2. **Recompute the cache lazily on each `factor()` when scaling
   resolves to `Mc64`.** Equivalent to (1) for IPM. No savings on
   genuine "factor the same matrix twice" use cases, but there are
   no such use cases in tree.
3. **Cache-with-revalidation.** Cheap value-hash at `factor()`
   entry; invalidate on mismatch. Adds branching complexity for
   uncertain win; the revalidation check itself must touch most of
   the matrix to be meaningful, defeating the savings.
4. **MC64 warm-start.** Treat the cache as an initial dual-variable
   guess; iterate Hungarian's augmenting-path search from there.
   Most aggressive option; requires understanding whether the
   matching itself is robust against value drift or whether warm
   starts can land in spurious local optima. Research scope.

## Fix landed

**Option chosen: closest to (1) — one-shot cache invalidation in
`Solver::factor` after every numeric call.** The cache remains
valid for the *first* numeric call after symbolic (values match by
construction; the symbolic phase built the cache from the same
matrix the numeric phase is about to factor). All subsequent calls
fall through to a fresh `mc64::compute_symmetric(matrix)` against
current values.

```rust
// src/numeric/solver.rs:452-468
// Issue #38: invalidate the one-shot MC64 cache that the symbolic
// phase populated for the immediately-following numeric reuse. ...
if let Some(s) = self.last_symbolic.as_mut() {
    s.cached_mc64 = None;
}
```

**Cost:** one extra MC64 (~100–200 ms on `n ≈ 1e5`) per warm
refactor when scaling resolves to `Mc64Symmetric`. The compression
side of the original amortisation is unaffected (`LdltCompress`
still runs once at symbolic time and re-uses the supernode
topology); only the scaling-side reuse is dropped.

## Regression test

`numeric::solver::tests::mc64_cache_invalidated_after_factor_issue_38`
inspects `last_symbolic.cached_mc64` after one `factor()` call and
asserts it is `None`. Direct field check rather than behavioural:
Sylvester's law keeps inertia invariant under any symmetric scaling
on well-conditioned small matrices, so the downstream wrong-inertia
symptom only manifests on large arrow-KKTs — a 4×4 reproducer is
insensitive. Verified the test fails when the fix is removed
(panics on the assertion) and passes when restored.

## Relationship to #37

#37 was the symmetric reverse of #17: pinene_3200 under default
CB=off takes ~94 s per IPM iteration because BK rejects safe pivots
in the wide dense trailing supernode that pinene's collocation
discretisation produces. That mechanism is **independent of the
MC64 cache** — it's a single-factor matrix-conditioning issue, not
a warm-state issue. The cache-staleness fix does not rescue pinene
single-factor cost.

Probe (`src/bin/probe_pinene_issue38_fix.rs`, uncommitted) ran the
10 pinene_3200_NNNN.mtx dumps through one warm `Solver::new()`
under the fix:

| iter | factor | inertia |
|---:|---:|---|
| 0 | 0.983 s | OK (64000 / 63995 / 0) |
| 1 | 0.917 s | OK |
| 2 | 0.706 s | OK |
| 3 | 0.872 s | OK |
| 4 | 0.940 s | OK |
| 5 | 451.871 s | OK |
| 6+ | killed @ 600 s wall | (incomplete) |

Inertia stays correct throughout — no silent corruption, unlike
rocket_12800. The late-iter cost explosion (iter 5: 452 s vs
iter 4: 0.94 s) is the same wide-supernode cascade the #37 issue
body diagnosed, not a stale-cache symptom. The MC64-cache fix
neither rescues nor regresses pinene_3200; #37 remains closed under
the per-problem `feral_cascade_break=yes` mitigation. The probe is
an investigation tool, not a regression test — pinene corpus is
gitignored and the per-factor cost varies by orders of magnitude
depending on ordering choice.

#37 stays closed (status quo: pounce surfaces `feral_cascade_break`
as a regular ipopt option; per-problem opt-in is the chosen
mitigation). Fixing the underlying wide-supernode cascade is a
separate, larger investigation (see issue body §"Suggested
direction" options 2 and 3).
