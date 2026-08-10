# Condition 1 rejects on a metric that measures the barrier trajectory. The fix is not a better metric.

**Date:** 2026-08-09
**Machine:** Apple M2, `Mac14,2`, 4P + 4E cores, macOS. All timings
`with_parallel(false)`, `with_profiling(true)`.
**Status:** measurement complete; no implementation. The design
recommendation at the end is a proposal, not a landed change.

## Why

`tried-and-rejected.md` (2026-05-21, "B2 value-bounded MC64 scaling
cache: gate metric confounded by IPM δ") rejected the value-bounded
MC64 scaling cache. Today nql180 spends 11.2 s of 17 s in scaling,
pinene_3200 79%, marine_1600 51% — all rejecting on condition 1. That
entry is the starting point for re-opening it, not a reason to skip
it.

The entry rests on two claims. Both are examined below with fresh
measurements; one is stale, one is true but narrower than it reads.

## The cost claim is stale, and it inverted

> Even with a perfect gate, B2 targets <2 % of the cost. pinene_3200's
> 10 iters total 493.9 s; iters 6-9 are 64.8/77.8/135.7/208.2 s (the
> cost-cluster blowup, 98 %). The MC64 Hungarian is ≤6 s total.

Today, the same 10 iterates of pinene_3200 total **1.48 s**, of which
MC64 is **1.19 s — 79%**. The delayed-pivot cost-cluster blowup that
dwarfed MC64 in May has since been fixed (~325× faster on this
family), and MC64 went from a rounding error to the dominant term.
The May reasoning was correct when written; its premise no longer
holds.

## The metric claim is true, and it indicts the diagonal specifically

> The KKT (2,2)-block rows carry a tiny δ-regularized diagonal (≈1e-8)
> against ≈1 off-diagonals, so their off/diag ratio is ≈1/δ. As the
> interior-point method drives δ→0, the ratio explodes 1e8→1e10 — the
> gate is measuring the IPM's barrier trajectory, not whether `D₀` is
> still a usable scaling.

This is confirmed, and it is a complaint about dividing by the
diagonal. MC64's own objective never mentions the diagonal: the
matching maximises the product of matched magnitudes, and the scaling
drives matched entries to ≈1 with all others ≤1. So the obvious
repair is a diagonal-free statistic on the same scaled matrix.

**That repair does not work.** `diag_scaled_entry_spread` computes
today's `max_ratio` and entry-magnitude statistics of `D₀·A_N·D₀` in
the same sweep:

| family | speedup | max_ent drift vs iter 0 | p999_ent |
|---|---|---|---|
| marine_1600 | 1.63× | 5.2e12 | 1.030 |
| nql180 | 2.10× | 3.9e6 | 3.9e6 |
| pinene_3200 | 5.09× | 1.1e1 | 5.698 |
| ex4_2_160 | 0.75× | 1.0e4 | 1.0e4 |

`marine_1600` is the best-behaved family in the set by outcome and the
worst by entry drift; `ex4_2_160` is the one clear regression and its
drift is eight orders milder. Entry magnitude is **anti-correlated**
with whether reuse is a good idea. `p999_ent` separates marine (a
handful of outlier entries) from ex4_2_160 (a broad shift), but nql180
also has `p999 == max` and is a 2.10× win.

No value proxy tried predicts the outcome.

## Why no value proxy predicts the outcome: there is no numerical outcome

`diag_trajectory_reuse` drives **two warm solvers** over the same
trajectory in one process — one on `Auto`, one pinned to
`External(d0)` with iterate 0's own MC64 vector. The warm arm matters:
`Solver::factor` clears `symbolic.cached_mc64` after every call
(issue #38, `src/numeric/solver.rs:1374`), so only a warm solver pays
the Hungarian inside `ProfileReport::total_us`. Every earlier probe in
this session that built a fresh solver per iterate was billing that
work to `analyze()` and measuring the wrong thing.

Whole `kkt-mittelmann` corpus, 37 families with ≥2 iterates, 233
iterates:

    corpus fresh 34.66 s -> reuse 24.47 s
    aggregate 1.42x, geomean 1.164x
    inertia differs on 0 iterates

Zero. Including `rocket_12800`, the matrix #38 was filed on.

### Why that is not luck

`dev/journal/2026-05-16-30.org` §17:25 records #38's mechanism
explicitly: `pick_ordering_preprocess` triggers `LdltCompress`,
`LdltCompress` populates `SymbolicFactorization::cached_mc64` via
`mc64::compute_matching`, and the stale **matching** is applied to
iterate N. The scaling vector is downstream of that. Two distinct
objects share the name "MC64 cache":

* `symbolic.cached_mc64` — the matching/permutation. Feeds
  `LdltCompress`; changes the elimination structure. This is what #38
  corrupted, and what `mc64_cache_invalidated_after_factor_issue_38`
  (`src/numeric/solver.rs:2285`) guards.
* `Solver::mc64_scaling_cache` — the scaling vector. Values only.
  This is what the value bound gates.

`ScalingStrategy::External` reuses only the second. It cannot reach
`LdltCompress` and cannot change the elimination structure.

The in-tree guard `mc64_cache_rejected_on_value_drift_issue_38_guard`
(`src/numeric/solver.rs:2511`) asserts only that the gate *rejects* on
`tridiag(4, 10, 1)` → `tridiag(4, 10, 50)`. It never establishes that
reuse would have been wrong. A sweep of that same family against an
independent Sturm-sequence oracle (Golub & Van Loan 4th ed. §8.4.1)
found reuse == fresh == oracle at every drift magnitude from off=1 to
off=1e12, including off=50.

### What this is not

Absence of a counterexample across 233 iterates is not a safety proof.
Every family here has a fixed pattern, and the probe's fingerprint
check is `csc.n != d0.len()` — n only. `rocket_12800` changes nnz
between iterates (332793 → 435190), so the External arm there applied
`D₀` across a pattern change the real cache would reject outright.
That makes rocket a *harder* correctness test than the design would
ever attempt, and it also means rocket's 1.33× is not a payoff the
real design could bank.

## What does predict the outcome

Sorting the 37 families by the fraction of factor time fresh MC64
actually consumes:

| scal% band | families | speedup range |
|---|---|---|
| ≤ 5.4% | all four regressions | 0.75× – 1.08× |
| ≥ 12.6% | every family, no exception | 1.13× – 5.09× |

The four regressions sit at 5.4% (ex4_2_160), 3.3% (cont5_1_l), 3.0%
(arki0009), 0.6% (qcqp1000-2c). The band 5.4% → 12.6% is empty, so
any threshold in 6–12% gives the identical partition. The constant is
not load-bearing.

Gating on measured cost share at 10%:

    15 of 37 families reuse
    corpus 34.66 s -> 24.32 s, 1.43x

That banks essentially all of the unconditional 1.42× while dropping
every regression.

## Recommendation

Replace the value-bound gate's role, not its metric.
`scaling_us / total_us` is already measured per factorization
(`PrologueBreakdown::scaling_us`, `ProfileReport::total_us`), so the
gate needs no value proxy, no numerical justification, and no
additional O(nnz) sweep. The 2026-05-21 entry searched for a metric
separating "safe" from "unsafe"; on this evidence the scaling-vector
path has no unsafe side, and the quantity that needs separating is
profitable from unprofitable.

Open before implementing:

1. **The regression that reuse causes is real and unexplained.**
   `ex4_2_160` reuse costs +45% on iterates 2–9 with *correct*
   inertia, and `nql180_0001` costs +66%. Something downstream —
   most likely delayed pivots — gets more expensive under a stale
   scaling. A cost-share gate sidesteps it rather than explaining it.
2. Profiling is not free and is off by default; a shipped gate needs
   the cost share available without `with_profiling(true)`.
3. Correctness rests on 233 iterates with no counterexample. Before
   landing, either an argument that the scaling vector cannot alter
   inertia, or an adversarial search for a case where it does.

## The independent, strictly-safe lever this surfaced

The Hungarian is genuinely value-dependent, measured standalone via
`diagnose_mc64_matching` (no solver, no cache), on a pattern that
never changes:

| pinene_3200 | match_us | augment | touched | edge_scans |
|---|---|---|---|---|
| 0001 | 46,852 | 57,426 | 1,340,479 | 8,148,431 |
| 0006 | 229,183 | 77,994 | 6,816,372 | 70,268,885 |

4.9× in wallclock. `augment_searches` moves only 1.36×;
`main_loop_edge_scans` moves 8.6×. The search *count* is stable —
each shortest-augmenting-path search wanders much further as δ→0. On
nql180, 895 searches perform 222M edge scans on n=259,681, i.e.
~227,000 scans per search, ~87% of n.

Bounding that search is same-output-less-work, needs no gate and no
residual argument, and helps the families where reuse is unprofitable
too. It is independent of everything above.
