# The MA57 comparison on the real chain KKTs

**Date:** 2026-08-09
**Machine:** Apple M4 Pro, `Mac16,11`, 10 performance + 4 efficiency cores, macOS.
**Supersedes:** `chain-kkt-ma57-gap-2026-08-09.md` in part — its
Results 2 and 3 only. Its Result 1 (the MA57 gap) stands; see the
correction below.

> **CORRECTION (supersedes Result 1 below).** The "no gap" result
> was produced with a misconfigured oracle and is wrong. `ma57_bench.F`
> sets `ICNTL(15) = 1`, which makes MA57 compute MC64 scaling *inside*
> `MA57BD` — the region the oracle times. feral computes its MC64 in
> **analysis**, which `factor_us` excludes. The comparison charged MA57
> per-factorization for work feral had already amortized and was not
> being billed for.
>
> Re-run with `ICNTL(15) = 0`, 15 pairs, `min factor_us`:
>
> | matrix | feral | ma57 | MA57 faster by | wins | p |
> |---|---|---|---|---|---|
> | dtoc1nd | 14,050 | 2,539 | 5.53x | 0/15 | 0.0001 |
> | clnlbeam | 21,979 | 5,384 | 4.08x | 1/15 | 0.0010 |
> | rocket_12800 | 22,266 | 5,955 | 3.74x | 0/15 | 0.0001 |
> | marine_1600 | 33,131 | 10,973 | 3.02x | 0/15 | 0.0001 |
> | steering_12800 | 34,517 | 17,097 | 2.02x | 0/15 | 0.0001 |
> | dtoc2 | 81,725 | 51,309 | 1.59x | 0/15 | 0.0001 |
>
> MA57 is faster on **all six**. Three independent checks say this is
> the real number:
>
> - issue #153 measured the same six through pounce and got 1.29x-3.77x in
>   MA57's favor. Its MA57 times (clnlbeam 7.27 ms, rocket 8.79,
>   steering 18.96) match `ICNTL(15)=0` (5.38, 5.96, 17.1), not the
>   oracle's (201, 554, 292).
> - The proxy conclusion this note claimed to supersede — 1.4x-2.4x
>   slower — was right in direction and close in magnitude. A correct
>   finding was overturned by a worse measurement.
> - The steering accuracy caveat was the same artifact. With scaling
>   off MA57 gets `1.59e-14`, not `4.53e-08`.
>
> Results 2 and 3 are unaffected: they compare feral builds to each
> other and never touch the oracle. The #150 attribution, the dtoc1nd
> regression, and the refuted E-core hypothesis all stand.
>
> What the checking missed: `factor_us` was verified to cover `MA57BD`
> only, and that was treated as sufficient. The boundary of the timed
> region was checked; the contents were not.

## Why

The proxy note measured block-tridiagonal stand-ins because the machine
it ran on had no `data/matrices/`. It closed with three ordered steps
for a machine that did: bisect the regression, sweep threads, re-run
the gap. This is that run. All three are done, and the answers are not
the ones the proxies predicted.

## Setup

Six real KKT systems, one iterate per family, from
`data/matrices/kkt-mittelmann` (see
`external_benchmarks/chain_proxy/real_corpus_mtx.py` for the selection
rules). Paired alternating A/B per `dev/decisions.md` (2026-08-09):
every arm timed once per pair, `min factor_us` over 15 pairs as the
per-arm statistic, exact two-sided sign test over pairs. All arms read
the same matrices and the same synthesized RHS.

`steering_12800` had no `.mtx` in the corpus and was regenerated for
this run — see **Regenerating steering_12800** below.

`dtoc2` uses iterate `_0000`. `dtoc2_0001` and `_0002` are singular:
feral reports `inertia_zero = 4`, MA57 reports `103`, both `rel_res =
NaN`. Two independent solvers agreeing on rank deficiency means the
matrix, not the solver, so those iterates cannot carry a timing.

## Result 1 — the gap against MA57 is not a gap

`main` vs HSL MA57, `min factor_us` over 15 pairs. Ratio > 1 means
feral is faster.

| matrix | n | main | ma57 | ratio | wins | p |
|---|---|---|---|---|---|---|
| rocket_12800 | 89,601 | 22,893 | 535,659 | **23.40** | 15/15 | 0.0001 |
| clnlbeam | 99,999 | 21,739 | 210,481 | **9.68** | 15/15 | 0.0001 |
| steering_12800 | 115,201 | 33,923 | 273,494 | **8.06** | 15/15 | 0.0001 |
| marine_1600 | 76,807 | 33,373 | 42,471 | 1.27 | 15/15 | 0.0001 |
| dtoc2 | 103,920 | 81,163 | 61,245 | 0.76 | 0/15 | 0.0001 |
| dtoc1nd | 9,685 | 14,297 | 4,561 | 0.32 | 0/15 | 0.0001 |

feral is faster on four of six, by up to 23x, and the two losses are
`dtoc2` (1.3x) and `dtoc1nd`, the smallest matrix in the set by an
order of magnitude.

The proxy note reported "feral is slower than HSL MA57 on all five
proxies, 1.4x to 2.4x" and called the largest, most chain-like proxy
the worst case. On the real matrices the largest and most chain-like
are where feral wins hardest. The proxies inverted the result.

**One caveat belongs with the 8.06x.** On `steering_12800` MA57's
residual is `4.53e-08` against feral's `3.52e-14`, on identical matrix
and RHS — six orders of magnitude, and the only place in the run where
either solver is worse than `1e-13`. feral's sidecar reports `refined
yes`, so some of that is iterative refinement the MA57 oracle does not
do. `factor_us` is numeric factorization only, so the timing stands,
but the honest statement is that feral is 8x faster *and* far more
accurate there, against an MA57 solve that is looser than feral's.

## Result 2 — the regression is real, and it is on the small matrix

Ratios vs `v0.14.0` (c05eb77). > 1 means the arm is faster.

| matrix | #149 SIMD | #150 parallel | v0.15.0 | main |
|---|---|---|---|---|
| clnlbeam | 0.966 | 2.048 | 2.054 | 2.048 |
| steering_12800 | 0.986 | 1.646 | 1.644 | 1.669 |
| dtoc2 | 0.999 | 1.401 | 1.438 | 1.434 |
| rocket_12800 | 0.979 | 1.200 | 1.255 | 1.214 |
| marine_1600 | 0.953 | 0.956 | 0.953 | 0.943 |
| **dtoc1nd** | 0.960 | **0.764** | **0.762** | **0.750** |

All the large-matrix gains are 15/15 at p = 0.0001, and so is the
`dtoc1nd` loss.

Three things this settles:

1. **#150 (`fad5670`, task-per-subtree coarsening) is the mover, and it
   is a large win** — 2.05x, 1.65x, 1.43x, 1.20x on the four largest.
2. **The one regression is on `dtoc1nd`, n = 9,685** — a 25% loss, 0/15,
   p = 0.0001, introduced by #150 and carried unchanged into v0.15.0 and
   main. The proxy note predicted losses on the two *largest*,
   chain-structured matrices. It got the existence of a regression right
   and its location exactly backwards.
3. **v0.15.0 and main are indistinguishable** (2.054/2.048,
   1.644/1.669, 1.438/1.434, 1.255/1.214, 0.953/0.943, 0.762/0.750).
   The proxy note flagged that its "new" arm was three commits past the
   tag and that #156 had since changed the parallel default, and left
   tag-vs-main open. It does not matter: neither #155's amalgamation
   guard nor #156's `available_parallelism()` default moves anything
   here.

#149's SIMD kernel is neutral-to-negative on this corpus — below 1.0 on
all six, significantly so on `clnlbeam` (0.966, 0/15, p = 0.0001) and
`rocket_12800` (0.979, 2/15, p = 0.0074). On the proxies it was helping;
here it is not. That is a separate thread worth pulling.

## Result 3 — thread count does nothing, and the E-core hypothesis is dead

`main` at `RAYON_NUM_THREADS` = 1, 2, 4, 8, 10 against the default
(all 14). Ratios vs `main-all`.

| matrix | t1 | t2 | t4 | t8 | t10 |
|---|---|---|---|---|---|
| clnlbeam | 0.972 | 0.973 | 0.983 | 0.992 | 0.976 |
| dtoc1nd | 0.992 | 1.024 | 1.005 | 0.998 | 0.987 |
| dtoc2 | 1.008 | 1.018 | 0.995 | 1.018 | 0.998 |
| marine_1600 | 0.961 | 1.056 | 1.029 | 1.026 | 1.015 |
| rocket_12800 | 1.030 | 1.029 | 1.024 | 1.049 | 1.041 |
| steering_12800 | 1.003 | 0.997 | 0.993 | 1.018 | 1.020 |

Every ratio is within 6% of 1.0, and only `marine_1600` at t1 (0.961,
0/15, p = 0.0001) is significant. **Restricting feral to a single thread
costs essentially nothing on any of these matrices.**

The proxy note's untested hypothesis was that rayon treating efficiency
cores as equivalent to performance cores lets a coarse task stall the
factorization, predicting "the loss shrinks or inverts at
`RAYON_NUM_THREADS=4`". It does not move at all, on a 10P+4E machine
where the effect should be *larger* than on the 4P+4E M2 the hypothesis
came from. The hypothesis is refuted; recorded in
`tried-and-rejected.md`.

The knob is real, not inert: feral reads the global rayon pool
(`rayon::current_num_threads()`, `src/numeric/factorize.rs:3262`,
`src/numeric/solve.rs:632`), and the same env var moved the proxies by
up to 65%.

**The consequence is the interesting part.** If single-threaded main
matches all-threads main, then #150's 2.05x on `clnlbeam` is not a
parallelism win. Whatever task-per-subtree coarsening bought — lower
spawn overhead, better locality, a different traversal order — it is
not multi-core execution. Any follow-up that treats #150 as "the
parallel win" and tries to extend it by parallelizing harder is
starting from a false premise.

## Regenerating steering_12800

`data/matrices/kkt-mittelmann/steering_12800/` held only a zero-byte
`.solver.log`. The harvest path in
`scripts/harvest-mittelmann-kkt.sh` is dead: ripopt commit `76d3575`
(2026-04-28, "A7.6 (set 2) delete dead condensed-path helpers")
bulk-deleted `dump_kkt_matrix`, `write_kkt_mtx_file`,
`write_kkt_json_sidecar` and `collect_kkt_lower_triangle_entries`
along with the `if let Some(ref dump_dir) = options.kkt_dump_dir` call
site, but left `SolverOptions::kkt_dump_dir` (`options.rs:714`) and its
CLI wiring (`ripopt_ampl.rs:607`) in place. So `kkt_dump_dir=` is still
accepted on the command line and silently writes nothing — the harvest
reports "OK ... 0 mtx" and exits 0.

pounce dumps the same systems. The replacement path is

```sh
pounce steering_12800.nl --dump kkt:1-3 --dump-dir <dumpdir>
scripts/harvest-pounce-kkt.py --dump-dir <dumpdir> --name steering_12800
```

The conversion is checked against an external oracle rather than
trusted: pounce reports `num_neg_evals_actual = 51201`, and feral and
MA57 independently agree on the converted matrix (`inertia_neg =
51201`, `inertia_pos = 64000`, `inertia_zero = 0`). feral solves
pounce's own dumped RHS to `rel_res = 3.66e-14`.

This also unblocks the other 41 Mittelmann families that were never
harvested.

## What this means for the open items

- **The MA57 gap is not the story it was.** On four of six real chain
  KKTs, including the three largest, feral is 8x to 23x faster. Any
  number quoted to pounce should come from this table, not from the
  proxies and not from pounce#552's pre-SIMD figures.
- **`dtoc1nd` is the concrete regression to chase**, and it is cheap:
  one small matrix, a known culprit commit, 25% on the table.
- **`dtoc2` and `dtoc1nd` are where MA57 still wins.** Both are the
  losses; neither is large. That is the shape of the remaining gap.
- **#149's SIMD kernel is negative on this corpus.** It was a win on
  the proxies and on the release's own measurements. Worth its own
  bisect before more kernel work.
- The scaling-reuse item stays on hold, but for a new reason: the
  premise that feral is broadly behind MA57 on this matrix class is
  false.

## Limits

- One iterate per family, six families, one machine, one architecture
  (aarch64 / Apple silicon). No x86_64 arm was run.
- `factor_us` is numeric factorization only. Analysis dominates wall
  clock on these matrices (`clnlbeam` analyse 4.59 s vs factor 38 ms),
  and pounce#552 reports end-to-end solve time, so this does not
  directly answer the report.
- The MA57 oracle does not do the iterative refinement feral does; see
  the `steering_12800` residual caveat under Result 1.

## Result 4 — analysis dominates, and one stage is all of it

Found while checking whether the corrected MA57 gap left any
optimization work. `analyse_us` vs `factor_us` for `main`, same run:

| matrix | n | nnz | analyse_us | factor_us | analyse/nnz |
|---|---|---|---|---|---|
| clnlbeam | 99,999 | 259,993 | 5,001,272 | 26,008 | 19.24 |
| rocket_12800 | 89,601 | 435,190 | 2,705,801 | 29,910 | 6.22 |
| marine_1600 | 76,807 | 414,399 | 1,048,521 | 42,817 | 2.53 |
| steering_12800 | 115,201 | 409,591 | 130,389 | 41,795 | 0.32 |
| dtoc2 | 103,920 | 961,230 | 276,116 | 101,528 | 0.29 |
| dtoc1nd | 9,685 | 217,270 | 42,464 | 19,278 | 0.20 |

A 60x spread in cost per nonzero across matrices of comparable size.
`clnlbeam` has the *fewest* nonzeros of the large five and the most
expensive analysis; `steering_12800` is larger (n = 115,201) with
comparable nnz and analyses 38x faster. Not a constant factor.

Per-stage breakdown via `diag_symbolic_stages_argv`:

| matrix | `ldlt_compress` | share of symbolic | `ordering` |
|---|---|---|---|
| clnlbeam | 4,515,372 | 99.3% | 3,706 |
| rocket_12800 | 2,289,795 | 98.3% | 3,932 |
| steering_12800 | 23,539 | 33.6% | 6,172 |
| dtoc1nd | 2,834 | 23.1% | 2,118 |

`ldlt_compress` is `compute_mc64_cache` — the Duff-Pralet symmetric
matching (`src/symbolic/mod.rs:1211`). On `clnlbeam` it takes 4.5 s
while the fill-reducing ordering beside it takes 3.7 ms, a factor of
1,200. Every other symbolic stage is 1-15 ms and scales sanely.

`src/symbolic/mod.rs:1190` already records the same shape on the pf22
powerflow KKT (MC64 ~53 s vs `amd_order` ~0.3 s, issue #80), so this is
the second corpus where MC64 dominates analysis by three orders of
magnitude.

This also reframes Result 1 honestly: feral's MC64 is not free, it is
billed to analysis, where `factor_us` does not see it. Whether
amortizing it across an IPM's refactorizations is legitimate depends on
the caller — but 4.5 s of analysis dominates any end-to-end number
regardless of how fast the factorization becomes.

## Limits on the corrected comparison

Neither MA57 configuration is a clean apples-to-apples arm.
`ICNTL(15) = 1` charges MA57 per-factorization for scaling feral
amortizes into analysis. `ICNTL(15) = 0` gives feral MC64 scaling that
MA57 does not get, so feral's arm is doing strictly more numerical
work. The corrected table uses `ICNTL(15) = 0` because that matches
what pounce measures through `SparseSymLinearSolverInterface` (issue
#153) and agrees with it to within the harness difference.

The clean experiment, not yet run: pre-scale each matrix with MC64
offline and run both solvers with scaling off, so the two arms
factorize identical numbers.
