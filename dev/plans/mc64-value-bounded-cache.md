# Plan — value-bounded MC64 scaling cache (Track B2)

**Status:** pre-implementation
**Date:** 2026-05-21
**Research note:** `dev/research/mc64-value-bounded-cache-2026-05-17.md`
**Parent plan:** `dev/plans/per-factor-cost-cluster.md` (Track B2)
**Advances:** #38 residual (rocket_12800 per-factor cost)

## Goal

Eliminate the per-call MC64 Hungarian on warm `Solver::factor`
replays. B1 pinned `rocket_12800`'s per-factor cost: prologue is
99.8 % the `scaling` sub-phase, and `scaling` is the MC64 Hungarian
(`compute_scaling(Mc64Symmetric)` = 4111 ms vs `InfNorm` = 5.4 ms).
The Hungarian reruns every IPM iter because the symbolic-scope
`cached_mc64` is cleared after the first factor (the #38 fix
`db20166`).

Target: `rocket_12800` warm factor time drops from ~6.5 s toward the
MA57 band; inertia bit-unchanged on every matrix where the cached
scaling is reused.

## Code-inspection findings

1. **Injection point.** `factorize_*_with_workspace`'s prologue calls
   `compute_scaling_with_cache(matrix, &params.scaling, cached_mc64)`.
   `ScalingStrategy::External(Vec<f64>)` already exists and its arm
   in `compute_scaling_with_cache` returns `s.clone()` in O(n) with
   `ScalingInfo::NotApplied`. So the Solver can inject a cached
   scaling purely by setting `effective_params.scaling =
   External(cached)` — **no numeric-phase plumbing required.**

2. **Capture point.** `SparseFactors.scaling: Vec<f64>` is the
   user-order vector that was applied; `SparseFactors.scaling_info`
   is `ScalingInfo::Applied` exactly when the MC64 Hungarian ran to
   completion on a non-singular matrix. After a fresh `factor()` the
   Solver reads `last_factors.scaling` / `.scaling_info`.

3. **`cached_mc64` is orthogonal.** The new cache lives at Solver
   scope and is additive over `SymbolicFactorization::cached_mc64`
   (the one-shot post-symbolic cache, still cleared per #38). The
   #38 regression test only inspects `cached_mc64` — untouched here.

4. **rocket_12800 has a fully-populated, all-nonzero diagonal**
   (measured: 89601/89601 explicit diagonal entries, none zero,
   `min|diag|=5.6e-9`, `max|diag|=6.3e3`). The IPM regularises the
   KKT (2,2) block (`-δ_c·I`), so the value-bound check's
   diagonal-dominance framing applies directly to the target matrix.
   This was a feasibility risk (a structurally-zero (2,2) block
   would degrade the check) — **disconfirmed empirically** for
   rocket. See Deviation 1 for the general-robustness handling.

## Architecture

```
Solver field:
    mc64_cache_enabled: bool                  // builder-controlled
    mc64_scaling_cache: Option<Mc64ScalingCache>

Mc64ScalingCache {                            // private to solver.rs
    fingerprint: PatternFingerprint,
    scaling: Vec<f64>,                        // user-order D0
    validity: Mc64CacheValidity,
}
```

`factor()` flow additions:

- **Step 2 (pattern change):** also clear `mc64_scaling_cache`.
- **Before the numeric driver, after `effective_params` is built:**
  ```
  let scaling_cache_hit =
      mc64_cache_enabled
      && mc64_scaling_cache matches fingerprint
      && mc64_value_bound_passes(matrix, &cache.scaling, &cache.validity);
  if scaling_cache_hit {
      effective_params.scaling = ScalingStrategy::External(cache.scaling.clone());
  }
  ```
- **After `Ok((factors, inertia))`, before storing `last_factors`:**
  - if `scaling_cache_hit` → keep the existing cache untouched.
  - else if `factors.scaling_info == ScalingInfo::Applied` → install
    a fresh cache: `{ fingerprint, scaling: factors.scaling.clone(),
    validity: precompute_mc64_validity(matrix, &factors.scaling) }`.
  - else (InfNorm / fallback / Identity / External / partial) → set
    `mc64_scaling_cache = None`. Caching a 5 ms InfNorm buys nothing,
    and a `PartialSingular` scaling is degenerate.

`scaling_cache_hit` must be captured **before** the borrow of
`factors` in the `Ok` arm. The `effective_params.scaling` mutation
only touches the local clone — `self.numeric_params` is never
mutated, matching the existing cascade-break auto-arm contract.

## Builder

`pub fn with_mc64_cache(mut self, on: bool) -> Self` — sets
`mc64_cache_enabled`. **Default `true`.** Rationale: the IPM warm-
replay use case is the dominant `Solver` consumer and the whole
point of B2; the value-bound check makes reuse correctness-gated.
Tests that probe MC64 behaviour directly opt out with
`with_mc64_cache(false)`.

## The value-bound check (new module `src/scaling/value_bound.rs`)

Pure, `pub(crate)`, O(nnz), allocation-bounded. Follows the research
note §"The value-bound validity check".

```
struct DominanceStats {           // one O(nnz) sweep of D·A·D
    max_ratio: f64,               // max over qualifying rows
    n_off_dominant: usize,        // count(ratio > 1) over qualifying rows
    min_diag: f64,                // min |scaled diag| over qualifying rows
    mean_diag: f64,               // mean |scaled diag| over qualifying rows
}
fn scaled_dominance_stats(matrix, scaling) -> DominanceStats

struct Mc64CacheValidity {
    r0: f64,                      // max(1.0, baseline max_ratio)
    n_off_dominant_0: usize,
    mean_diag_0: f64,
}
fn precompute_mc64_validity(matrix, scaling) -> Mc64CacheValidity
fn mc64_value_bound_passes(matrix, scaling, &Mc64CacheValidity) -> bool
```

`mc64_value_bound_passes` rejects (returns `false`) if any of:

1. `stats_N.max_ratio > GROWTH_FACTOR * validity.r0`
2. `stats_N.n_off_dominant as f64 > GROWTH_COUNT * (n_off_dominant_0 as f64)`
3. `stats_N.min_diag < EPS_DIAG * validity.mean_diag_0`

Constants: `GROWTH_FACTOR = 2.0`, `GROWTH_COUNT = 1.5`,
`EPS_DIAG = 1e-12` (research note defaults).

### Deviation 1 — qualifying rows exclude structurally-zero diagonals

The research note assumes a fully-populated diagonal. A general KKT
with a structurally-zero (2,2) block has rows with zero diagonal and
nonzero off-diagonals → ratio `+∞`, which makes `r0 = +∞` and
neuters conditions 1 & 2. **All three conditions are therefore
computed only over "qualifying rows": rows whose diagonal entry is
structurally present and nonzero in the cache-baseline matrix.** For
rocket_12800 every row qualifies (full nonzero diagonal) → no
behaviour change on the target. For zero-(2,2) KKTs the check
becomes a well-defined Hessian-block-drift measure rather than a
degenerate one. The asymmetry is safe: a too-strict check only
forces a fresh (correct) MC64; a too-lenient one is what the local
replay validation gate catches.

`Mc64CacheValidity` stores `mean_diag_0` (baseline, stable) so
condition 3 compares against a fixed reference rather than the
drifting current mean.

## Tests-first sequence

External oracles, per the hard rule (no impl + oracle in one
session without an external oracle):

1. **`scaled_dominance_stats` — hand calculation.** A 3×3 symmetric
   CSC with known values and a known scaling vector; hand-compute
   `diag_scaled`, `off_max_scaled`, per-row ratio, and assert all
   four `DominanceStats` fields. Oracle = hand calculation.
2. **`mc64_value_bound_passes` — boundary cases.** Construct
   validity stats and a matrix so each of the three conditions is
   the lone trigger; assert pass on the in-bound matrix and reject
   on each out-of-bound one. Oracle = hand-derived thresholds.
3. **Zero-diagonal exclusion (Deviation 1).** A matrix with one
   structurally-zero diagonal row: assert that row does not pull
   `r0` to `+∞` and does not appear in `n_off_dominant`.
4. **Cache hit (integration).** Factor the same matrix twice through
   one `Solver` with `Mc64Symmetric`; assert the second factor's
   inertia and solve residual equal the first. Oracle = first
   (fresh-MC64) factor.
5. **Cache hit equals cache off.** Factor a 2-call sequence with
   `with_mc64_cache(true)` and again with `with_mc64_cache(false)`;
   assert identical inertia and identical solve on both calls.
   Oracle = the cache-off path (existing fresh-MC64 code).
6. **Pattern miss.** Two structurally-different matrices through one
   `Solver`; assert both factor correctly and the cache is rebuilt.
7. **Value-drift miss.** Start from a well-scaled matrix; perturb
   diagonal entries on the second matrix to break diagonal dominance
   under the cached scaling; assert the value-bound check rejects
   and inertia stays correct. Oracle = the fresh-MC64 result on the
   perturbed matrix.
8. **#38 regression.** `mc64_cache_invalidated_after_factor_issue_38`
   must stay green unchanged (the new cache does not touch
   `cached_mc64`).

## Validation

`probe_kkt_replay` warm-replays a KKT corpus through one `Solver`
and checks per-call inertia against the JSON oracle. Run it
cache-on vs cache-off on every local corpus —
`rocket_12800` (2), `pinene_3200` (10), `robot_1600` (7),
`marine_1600` (18), `NARX_CFy` (3), `arki0003` (66): 106 KKT
matrices. **Acceptance:**

- (a) per-call inertia bit-identical cache-on vs cache-off on all
  106 matrices (the inertia hard rule);
- (b) `rocket_12800` warm factor time drops materially (Hungarian
  amortised across the 2 local replays — full 18-call corpus is a
  B3 sweep item);
- (c) no regression on the other corpora's replay totals.

This local replay is both the calibration feedback loop and the
acceptance test: the research note's "find the iter-10 drift point"
calibration needs the full 18-matrix rocket corpus (not present
locally), but the inertia-parity gate does not — a too-lenient
threshold surfaces as an inertia mismatch here. If any inertia
drifts, tighten the constants or default the builder to `false`
pending the B3 sweep.

## Out of scope

- Plan B2 options 2 (re-route rocket off MC64) and 3 (speed up the
  Hungarian) — only pursued if the cache validation fails.
- MC64 warm-start of the fresh-recompute branch.
- Moving MC64 to symbolic scope (MA57 architecture).

## Exit

`cargo test` + `clippy` clean; the 8 test groups above pass;
`probe_kkt_replay` acceptance (a)–(c) met; `probe_rocket_profile`
re-run records the new prologue/scaling split; journal + checkpoint
+ `per-factor-cost-cluster.md` B2 status updated.
