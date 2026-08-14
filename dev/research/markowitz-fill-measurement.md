# Threshold-Markowitz LU: measured fill and cost (#167)

Date: 2026-08-14. Branch `feat/167-threshold-markowitz`.
Plan: `dev/plans/threshold-markowitz-lu.md`. Implementation: `src/lu/markowitz.rs`.

## What was measured

`examples/basis_refactor` factors a preserved corpus of 16 LP bases four ways and
reports fill as `factor_nnz()/nnz(B)` (strict-lower `L` plus `U` with diagonal)
and wall-clock for the full analyze+factor:

- `AMDfull` — the shipped static path: AMD on the AᵀA column-intersection
  pattern, then Gilbert-Peierls with partial pivoting.
- `peel` — `analyze_triangularized`: Suhl-Suhl triangularization, AMD on the
  residual bump only.
- `denseBump` — `peel` plus the dense-bump route (`dense_bump_max_dim`).
- `markowitz` — `SparseLu::factor_markowitz`, threshold-Markowitz with
  `u = 0.1`, `max_search = 8`. Its symbolic column is 0 by construction, not by
  omission: there is no separate analyze phase.

3 reps, release build, `--release` example binary. Corpus at `/private/tmp/feral-fill`.

## Result

```
basis                            m   nnz(B)   bump |  AMDfull    peel  MARKOW |     t_AMD   t_peel  t_dense   t_mark
QPLIB_0343_rlt0               5102    12752     99 |    1.03x   1.03x   1.00x |     1.80m    0.67m    0.58m    1.43m
QPLIB_0343_rlt1               5202     7868     18 |    1.01x   1.00x   1.00x |     0.92m    0.38m    0.35m    0.97m
QPLIB_0911_rlt0               5150    37133     29 |   12.92x   1.00x   1.00x |   168.82m    1.10m    1.10m    4.32m
QPLIB_0911_rlt1               5150    37133     29 |   12.92x   1.00x   1.00x |   179.63m    1.07m    1.07m    4.30m
QPLIB_0975_rlt0               5110    17330     14 |    1.39x   1.00x   1.00x |     9.75m    0.66m    0.63m    1.89m
QPLIB_0975_rlt1               5110    17330     14 |    1.39x   1.00x   1.00x |     9.79m    0.67m    0.63m    1.89m
QPLIB_1055_rlt0               3300    20617     18 |    1.10x   1.00x   1.00x |     1.54m    0.60m    0.57m    1.78m
QPLIB_1055_rlt1               3300    20617     18 |    1.10x   1.00x   1.00x |     1.54m    0.60m    0.56m    1.79m
QPLIB_1143_rlt0               3308    19355     50 |    1.43x   1.01x   1.00x |     6.94m    0.70m    0.64m    1.98m
QPLIB_1143_rlt1               3628    30315    624 |    8.33x   5.66x   1.31x |    56.72m   31.77m    8.09m    4.87m
QPLIB_1157_rlt0               3273     9197     41 |    1.03x   1.03x   1.00x |     1.22m    0.43m    0.39m    1.05m
QPLIB_1157_rlt1               3937    29085    584 |    6.28x   5.97x   1.74x |    45.13m   31.16m   11.43m    6.94m
QPLIB_1451_rlt0_cap1000       7392    39569    391 |    1.30x   1.21x   1.00x |    16.36m    1.61m    1.92m    5.27m
QPLIB_1451_rlt0_cap100000     7392    63720   1350 |   24.90x   3.21x   1.14x |  1066.64m   12.08m   32.23m   10.98m
QPLIB_1451_rlt0_cap300        7392    22420     78 |    1.51x   1.22x   1.00x |     5.94m    1.08m    0.90m    3.01m
QPLIB_1451_rlt0_cap3000       7392    56213     67 |    7.86x   1.01x   1.00x |   119.81m    1.73m    1.73m    7.22m

n=16  geomean fill:  AMDfull 2.77x  peel 1.38x  MARKOWITZ 1.06x
geomean total time vs AMDfull:  peel 9.68x  denseBump 10.99x  markowitz 4.92x
```

## Reading

**Fill.** 2.77x -> 1.06x geomean, and 1.00x (zero fill) on ten of the sixteen
bases. This cross-validates the Python oracle from the #167 investigation, which
predicted 1.11x; the Rust implementation is at or slightly better than the
oracle, so both are measuring the same thing.

**Wall-clock is mixed and that is the honest headline.** Markowitz is the
fastest arm on exactly the class of basis discopt #1008 is written around --
large residual bumps:

- QPLIB_1143_rlt1 (bump 624): 4.87m vs denseBump 8.09m, peel 31.77m, AMDfull 56.72m
- QPLIB_1157_rlt1 (bump 584): 6.94m vs denseBump 11.43m, peel 31.16m, AMDfull 45.13m
- QPLIB_1451_rlt0_cap100000 (bump 1350): 10.98m vs peel 12.08m, denseBump 32.23m

It loses to the peel on near-triangular bases, where the peel's linear scan is
simply cheaper than running the general pivot search 5150 times:

- QPLIB_0911_rlt0 (bump 29): peel 1.10m vs markowitz 4.32m (still 39x faster than AMDfull)
- QPLIB_1451_rlt0_cap3000 (bump 67): peel 1.73m vs markowitz 7.22m

So the geomean (peel 9.68x, markowitz 4.92x) is dominated by the many
near-triangular bases in this corpus, not by a general slowness.

## Two cost experiments, both recorded because one of them failed

**Suhl-Suhl singleton fast path** (column with one live entry, row with one live
entry passing the threshold, taken off a stack instead of through the count
heaps). Fill 1.07x -> 1.06x, wall geomean 3.43x -> 3.50x. Essentially free but
essentially no help: the heap traffic was not where the time was. Kept, because
it does improve fill slightly and costs nothing, but it did not do the job it
was added to do.

**In-place rank-1 update.** The first implementation rebuilt each touched column
into a fresh `Vec` per (pivot, column) pair. On a 5150-column basis that is
millions of allocations. Rewriting it to walk the column once in place --
updating entries the pivot column touches, `swap_remove`ing exact cancellations
and the consumed pivot row, appending unmet rows as fill -- moved the wall
geomean 3.50x -> 4.92x with fill unchanged. This was the real cost, not the
search.

## What is not claimed

- No claim about numerical behaviour in a simplex trajectory. The measured
  stability cost from the plan stands: `max|U|/max|B| = 81.8`, `max|L| = 9.70`
  at u=0.1 on QPLIB_1157_rlt1, against SuperLU's 2.56/1.00. Less fill is bought
  with more growth, and #163/#166 are the standing evidence that growth on these
  bases is not a free parameter.
- No claim that this should replace the shipped path. `factor_markowitz` is an
  additional entry point; `SparseLu::factor` is untouched.
- Peel-then-Markowitz-on-the-residual-bump is still out of scope. The wall-clock
  split above is the argument *for* it -- take the peel's cheap linear scan for
  the triangular part, the Markowitz search only for the bump -- and it should
  be its own issue, measured on its own.
