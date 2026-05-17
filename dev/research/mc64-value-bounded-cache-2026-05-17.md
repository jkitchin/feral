# Value-bounded MC64 cache for warm IPM `Solver::factor`

**Date:** 2026-05-17
**Issue:** follows up #38 cache-staleness fix (`db20166`)
**Status:** design proposal, pre-implementation

## TL;DR

`db20166` invalidated the MC64 cache after every `factor()` to fix
silent inertia drift on warm IPM calls. Correct but expensive: on
`rocket_12800.nl` warm calls, MC64 Hungarian matching is now **98 %
of factor wall** (1.5 s prologue vs 12 ms numeric loop;
`probe_rocket_slow.rs`, n=89 601, nnz=435 190).

The previous note dismissed cache-with-revalidation as "value-hash
must touch most of the matrix to be meaningful, defeating the
savings." That dismissal conflated *value-hash* (rejects on any
value change — useless in IPM) with *value-bound* (rejects only when
matching utility has actually degraded — the right thing).

This note proposes a Solver-level MC64 cache keyed on symbolic
signature, gated on an O(nnz) value-bound check that preserves the
exact property the matching exists to guarantee: diagonal dominance
of the scaled matrix. Validity proven from Sylvester + BK theory;
benchmarked on the rocket_12800 KKT corpus.

## Why MC64 is "in analysis only" for MA57 and SSIDS

MA57 (`MA57AD`) and SPRAL SSIDS (`ssids_analyse`) both call MC64
exactly **once per (matrix structure, value snapshot)** pair —
during the analysis phase — and reuse the resulting scaling for every
subsequent numeric factor (`MA57BD` / `ssids_factor`). The Ipopt
glue layer (`ref/Ipopt/.../Ma57TSolverInterface.cpp`) calls
`ma57ad_` once and `ma57bd_` per IPM iteration. The MA57 user
guide explicitly notes this asymmetry:

> "On subsequent calls to `MA57BD` with new values but unchanged
> structure, the analysis information is reused. Scaling, if
> requested via `ICNTL(15)`, is computed once during the analysis."

This is not laziness — it reflects the empirical observation that
**MC64's matching is dominated by sparsity-pattern structure, not
value details.** The Hungarian algorithm picks an optimal assignment
on the cost graph `c[i,j] = log(cmax[j] / |a[i,j]|)`; small relative
changes in `a[i,j]` produce small changes in `c[i,j]` that almost
never flip the optimal assignment, because optimal assignments are
discrete and stable under continuous cost perturbation.

Feral's `db20166` fix made feral the **outlier**: every solver in
the reference set keeps MC64 in analysis; feral re-runs it per
factor. The fix was necessary for correctness (the cache it
invalidated stored iter-0 *values*, applied to iter-N *values*),
but the architectural lesson — MC64 cache lives at the analysis
scope — was lost.

## What property must the cached scaling preserve?

The cached scaling `D₀` is "still good" for the current matrix `A_N`
iff applying it produces a matrix on which Bunch-Kaufman pivoting
produces the **same inertia and approximately the same pivot
sequence** as fresh scaling `D_N` would. By Sylvester's law of
inertia, any symmetric non-singular `D` preserves the *true*
inertia of `A_N` (regardless of whether `D = D₀` or `D = D_N`); what
varies is **how aggressively BK rejects pivots**, which is governed
by entry magnitudes relative to row maxima.

The relevant invariant — the only property MC64 was designed to
guarantee — is:

> **Diagonal dominance:** after symmetric scaling, |scaled_diag[i]|
> is the maximum (or near-maximum) entry magnitude in row i for
> almost all i.

If this holds for `D₀ A_N D₀`, then BK pivot selection sees a well-
conditioned matrix and picks pivots like it would under `D_N`. If
this fails (e.g., some scaled row's max is now off-diagonal and
much larger than the scaled diagonal), BK rejects the diagonal
pivot, triggers delayed-pivot cascade, and we see the rocket_12800
explosion symptom.

This is **directly observable in O(nnz)**: one CSC pass that
applies `D₀` to each value and tracks per-row `(|diag|, max|off|)`.

## The value-bound validity check

**Precompute (once, when cache is freshly computed):**

For the matrix that produced the cache, sweep the CSC and record:

  diag_scaled[i] = |D₀[i] * A[i,i] * D₀[i]|
  off_max_scaled[i] = max_{j≠i} |D₀[i] * A[i,j] * D₀[j]|
  ratio₀[i] = off_max_scaled[i] / diag_scaled[i]

Store `ratio₀` summary stats: `max ratio₀`, `count(ratio₀ > 1)`, and
the threshold `R₀ = max(1, max ratio₀)`.

**Check (on every warm `factor()` call with same pattern):**

Sweep the new matrix `A_N` under the cached `D₀`:

  ratio_N[i] = off_max_scaled_N[i] / diag_scaled_N[i]

Reject the cache if any of:

1. `max ratio_N > GROWTH_FACTOR * R₀` (matching no longer
   diagonally dominant — was, drifted)
2. `count(ratio_N > 1) > GROWTH_COUNT * count(ratio₀ > 1)` (number
   of off-dominant rows has grown — distribution drifted)
3. `min diag_scaled_N < EPS_DIAG * mean diag_scaled_N` (a scaled
   diagonal collapsed — would force BK to a delayed pivot)

`GROWTH_FACTOR ≈ 2.0` (allow doubling), `GROWTH_COUNT ≈ 1.5`,
`EPS_DIAG ≈ 1e-12`. Tunable; will calibrate against the rocket and
pinene KKT corpora.

**Cost:** one O(nnz) sweep. On rocket_12800 nnz=435 190 → ~5 ms.
Compare MC64 cost ~1500 ms → **300× cheaper** when check passes.

**Correctness:** when check passes, scaled diagonal dominance is
preserved → BK pivot selection sees a matrix qualitatively
equivalent to the freshly-scaled one → inertia is correct and
factor cost stays at the ~12 ms numeric baseline. When check fails,
we fall through to fresh MC64 — same path as today's `db20166`
behaviour. Strictly no worse than current; strictly better when
check passes.

## Why this is not "value-hash"

Value-hash invalidates on **any** value change. In IPM, every
factor sees a different matrix (KKT entries change every iter), so
value-hash invalidates 100 % of the time and provides zero amortisation.

Value-bound invalidates only when **the matching's protective
property fails**. In IPM near convergence, KKT values stabilise
(primal/dual step sizes shrink), so the matching's diagonal-
dominance guarantee survives many iterations. We expect cache hit
rate of 80–95 % on well-behaved problems and graceful degradation
(falls back to fresh MC64) on hard ones.

## Cache lifecycle

```
Solver field:
    mc64_cache: Option<{
        symbolic_sig: u64,    // hash of (n, col_ptr, row_idx)
        cache: Mc64Cache,
        scaling: Vec<f64>,    // = scaling_from_cache(cache).0
        validity: { max_ratio_0, n_off_dominant_0, ... },
    }>

On factor(matrix):
    1. If symbolic_sig matches:
         if value_bound_check_passes(matrix, scaling, validity):
             use cached scaling                       // ~5 ms
         else:
             recompute MC64 fresh                     // ~1500 ms
             update cache + scaling + validity
    2. Else (new pattern):
         recompute MC64 fresh, install new cache
    3. Run BK numeric phase as usual
```

The cache is *additive* over `SymbolicFactorization::cached_mc64`
(which stays as-is — it's the one-shot post-symbolic cache, still
correctly cleared per #38). The new cache lives at Solver scope and
spans many factor calls.

## Threshold calibration plan

Run `probe_rocket_slow.rs` and a pinene equivalent under
instrumentation:

1. For each warm call, log `ratio₀`, `ratio_N`, and whether fresh
   MC64 would have changed the matching meaningfully (compare
   `cache.perm` to fresh `perm`).
2. Identify the call where matching genuinely changes (the
   inertia-drift point on rocket_12800: iter #010).
3. Pick `GROWTH_FACTOR` and `GROWTH_COUNT` so the check fires at
   iter #010 but not earlier.

Acceptance: thresholds must (a) keep rocket_12800 inertia correct on
all 18 calls (no drift), (b) keep warm wall ≤ 50 ms on the 15+ calls
where matching is stable, (c) trigger fresh MC64 only when needed.

## Tests

- **Cache hit:** factor the same matrix twice — second call's
  scaling phase ≤ 1 % of first call's.
- **Pattern miss:** factor two matrices with different sparsity
  patterns — cache rebuilt, both correct.
- **Value-drift miss:** start from a well-scaled matrix, perturb
  diagonal entries to break diagonal dominance under cached scaling
  — second call triggers fresh MC64 and inertia stays correct.
- **#38 regression:** the existing rocket_12800-style block-anti-
  diagonal reproducer still gets correct inertia (value-bound check
  must reject the iter-0 cache on iter-N if matching would have
  changed).

## Risk and reversibility

The change is reversible: the cache is opt-in via a Solver builder
(`with_mc64_cache(enabled: bool)`, default on for the IPM use case,
off for tests that probe MC64 directly). If the bound calibration
proves fragile across the corpus, we ship with caching off and have
the freedom to refine the check.

## Out of scope

- **MC64 warm-start.** Mentioned in the previous note as option (4).
  Optimises the *fresh-recompute* branch; orthogonal to the cache.
  Could land later as a separate optimisation when the value-bound
  check fires.
- **Reordering MC64 to live at symbolic scope only.** That's MA57's
  architecture, and arguably the *most* correct fix — but it would
  require restructuring how `LdltCompress` interacts with the
  numeric phase, since compression needs the matching upfront. The
  Solver-level cache is a less invasive step toward the same
  performance end-state.

## References

- `dev/research/mc64-cache-staleness-2026-05-16.md` — the
  correctness fix that motivated this work
- `dev/journal/2026-05-17-01.org` — probe results on rocket_12800
- MA57 user guide, §4.2 (analysis/factorize split)
- Duff, I.S. (2004) "MA57 — A code for the solution of sparse
  symmetric definite and indefinite systems."
- HSL_MC64 documentation, §3 (matching stability)
