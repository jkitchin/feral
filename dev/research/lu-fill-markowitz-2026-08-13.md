# The LP-basis fill is an artifact of static ordering, not intrinsic to the bases

**Date:** 2026-08-13 · **Origin:** discopt #1008 · **Harness:** `dev/probes/markowitz-fill/`

## Question

discopt #1008 attributes 72.6% of its LP wall to `SparseLu::factor` and shows the
per-LP share tracking fill directly (fill 1.0 → ~5% of wall; `QPLIB_1451_rlt0` at
fill 19.1x → 74.8%). Its closing question, left explicitly to feral:

> whether this fill is intrinsic to these bases or an artifact of static ordering
> under partial pivoting (the usual remedy being threshold-Markowitz pivoting,
> choosing pivots dynamically against both sparsity and stability).

## Answer

**Artifact.** On a 16-basis corpus of real discopt simplex bases, threshold-Markowitz
reaches geomean fill **1.11x** and never exceeds 1.89x. feral's shipped ordering is
geomean **3.00x** with a tail to 24.90x.

```
basis                         m   nnz(B)   bump |  AMDfull     peel   COLAMD   MARKOW | M vs best feral
QPLIB_1157                 3937    29376    611 |    6.49x    6.74x    7.26x    1.79x |           3.64x
QPLIB_3852                 1760     4003     45 |    1.08x    1.01x    1.07x    1.01x |           1.00x
QPLIB_1451_rlt0_cap100000   7392    63720   1350 |   24.90x    3.21x   19.44x    1.14x |           2.82x
QPLIB_1451_rlt0_cap3000    7392    56213     67 |    7.86x    1.01x   14.95x    1.00x |           1.01x
QPLIB_0343_rlt0            5102    12752     99 |    1.03x    1.03x    1.02x    1.00x |           1.03x
QPLIB_0343_rlt1            5202     7868     18 |    1.01x    1.00x    1.01x    1.01x |           1.00x
QPLIB_0911_rlt0            5150    37133     29 |   12.92x    1.00x    8.59x    1.00x |           1.00x
QPLIB_0911_rlt1            5150    37133     29 |   12.92x    1.00x    8.59x    1.00x |           1.00x
QPLIB_0975_rlt0            5110    17330     14 |    1.39x    1.00x    1.24x    1.00x |           1.00x
QPLIB_0975_rlt1            5110    17330     14 |    1.39x    1.00x    1.24x    1.00x |           1.00x
QPLIB_1055_rlt0            3300    20617     18 |    1.10x    1.00x    1.10x    1.00x |           1.00x
QPLIB_1055_rlt1            3300    20617     18 |    1.10x    1.00x    1.10x    1.00x |           1.00x
QPLIB_1143_rlt0            3308    19355     50 |    1.43x    1.01x    7.12x    1.00x |           1.01x
QPLIB_1143_rlt1            3628    30315    624 |    8.33x    5.66x    9.60x    1.32x |           4.30x
QPLIB_1157_rlt0            3273     9197     41 |    1.03x    1.03x    1.04x    1.00x |           1.03x
QPLIB_1157_rlt1            3937    29085    584 |    6.28x    5.97x    6.51x    1.89x |           3.15x

n=16  geomean fill:  AMDfull 3.00x  peel 1.52x  COLAMD 3.24x  MARKOWITZ 1.11x
geomean Markowitz advantage over feral's best ordering: 1.37x  (min 1.00x, max 4.30x)
```

`AMDfull` = `SparseLuSymbolic::analyze_amd_only`, which is what
`SparseLuSymbolic::analyze` does today (`LuOrderingParams::default()` is
`triangularize: false`) and therefore what discopt runs. `peel` =
`analyze_triangularized` (#160, merged, **unreleased**). `COLAMD` =
`scipy.sparse.linalg.splu(permc_spec="COLAMD", diag_pivot_thresh=0.1)`. `MARKOW` =
`dev/probes/markowitz-fill/markowitz.py` at `u = 0.1`. All four count strict-lower
`L` plus `U` including the diagonal, matching `SparseLu::factor_nnz()`.

## Two separable findings

### 1. Most of the gap is code we have already written and have not shipped

The peel takes geomean fill 3.00x → 1.52x, and on the bases that are nearly
triangular it closes the gap outright: `QPLIB_0911` 12.92x → 1.00x (5150 columns
peeling to a bump of 29), `QPLIB_1451_rlt0` at 3000 iterations 7.86x → 1.01x.

discopt does not get this. It pins crates.io `feral 0.15.1`, which predates #160,
and even on current `main` the default ordering is whole-basis AMD. So discopt's
statement that "feral's `analyze` already triangularizes and runs AMD on the
residual bump, so 19.1x fill is *after* a fill-reducing ordering" is **wrong on
both halves**: the version it links has no peel, and the entry point it calls does
not select one.

### 2. The residual bump is where Markowitz is actually needed

On the four bases whose bump exceeds 500 columns — `QPLIB_1157` (611),
`QPLIB_1157_rlt1` (584), `QPLIB_1143_rlt1` (624), `QPLIB_1451_rlt0_cap100000`
(1350) — the peel leaves 3.21–6.74x and Markowitz reaches 1.14–1.89x, a
**2.82–4.30x** fill difference. These are exactly the instances #1008 is written
around.

The bump grows along the simplex trajectory, which is why a shallow sample hides
this. On `QPLIB_1451_rlt0` the bump is 67 columns at 3000 iterations and 1350 at
25 088, and the peel goes from closing the gap entirely (7.86x → 1.01x) to leaving
a 2.82x one (24.90x → 3.21x). Bases sampled early in a solve are not evidence
about the bases a solve spends its time on. Under that issue's own measured cost law
(numeric cost ∝ fill² × nnz(B)) a 3.6x fill difference is an order of magnitude of
numeric LU work, against `LuNumeric` at 72.6% of LP wall.

## Why the earlier "fill theory is dead" measurement could not have found this

#1008 ran SuperLU (COLAMD) against feral on 12 final bases, got median 0.94x, and
cancelled the planned ordering work on it. That comparison is uninformative by
construction: **SuperLU is the same algorithm class as feral** — a static
fill-reducing column permutation followed by partial pivoting. On this corpus it
is in fact *worse* than feral (geomean 3.24x vs 3.00x), and both sit 2.7–2.9x
above the achievable 1.11x. Agreement between two members of a class says nothing
about the alternative to the class.

The same reading is available inside the SuperLU arm alone: `diag_pivot_thresh`
0.1 vs 1.0 moves fill by under 2% on every basis here (`QPLIB_1157`: 7.26x vs
7.50x). The pivoting threshold is not what sets the fill. The static column order
is.

## Cost of the alternative

Markowitz is not free and this note does not claim it is.

* **Stability.** Threshold pivoting bounds `max|L|` by `1/u` and nothing bounds
  growth as tightly as partial pivoting does. Measured on `QPLIB_1157` at
  `u = 0.1`: `max|U|/max|B| = 81.8`, `max|L| = 9.70` (at its 1/u bound), against
  SuperLU partial pivoting's `2.56` and `1.00`. At `u = 0.01` growth is 353x for
  a further 0.4% of fill — not worth it. feral already has the machinery to live
  with this (`LuParams::max_growth`, `should_refactor_growth()`).
* **Architecture.** Markowitz picks pivots from *numeric* values, so it cannot be
  expressed in feral's current split of `SparseLuSymbolic::analyze` followed by
  `SparseLu::factor`. It is a new factorization path, not a new ordering — the
  symbolic phase disappears into the numeric one rather than being replaced.
  (That is not purely a cost: symbolic analysis is 5.0% of discopt's LP wall today
  and would stop being a separate line item.)
* **Search cost.** The oracle here is Python and its speed means nothing. A real
  implementation needs the standard count-bucket structure with the Suhl search
  cutoff; that work is real and is not estimated in this note.

## Recommended order of work

1. **Release the peel and make it reachable.** Cut a release carrying #160 and
   decide the default-vs-parameter question in #165. This is finished code and it
   is where geomean 3.00x → 1.52x lives.
2. **Then** implement threshold-Markowitz for the bump. Everything above 500 bump
   columns is still 2.8–4.3x off, and the corpus says that is where the wall is.

Step 1 must not be reported as closing #1008: on `QPLIB_1157`, the instance the
issue is built around, the peel changes fill by *nothing* (6.49x → 6.74x, slightly
worse) and only Markowitz moves it.

## Provenance

Bases are real discopt simplex bases: the captured #1008 root relaxation LPs
re-solved through `discopt._rust.solve_lp_warm_csc_py`, with the final basis
reconstructed from the returned `basic_vars` against `[A | I]`. Every basis passes
a direct gate on that indexing — each basic index `>= n_struct` must be exactly
`e_{j-n_struct}` — plus a nonsingularity check, before it is written.
`QPLIB_1157_basis` and `QPLIB_3852_basis` are the in-tree fixtures from
`tests/data/lu_bases/`, written by feral itself during a discopt solve.

**#1008's 19.1x for `QPLIB_1451_rlt0` is reproduced**, as a range rather than a
point. That figure is a mean over ~600 factorizations of one full solve. Sampling
that trajectory at 3000 and at 25 088 iterations gives AMD-only fill of 7.86x and
24.90x, which brackets it. The same instance was the reason for the earlier
caveat in this note that only fill up to 12.92x had been seen; that caveat is
withdrawn.
