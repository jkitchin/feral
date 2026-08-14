# Plan: threshold-Markowitz LU (issue #167)

**Created:** 2026-08-14
**Research note:** `dev/research/lu-fill-markowitz-2026-08-13.md` (read first)
**Oracle:** `dev/probes/markowitz-fill/markowitz.py`
**Tracking issue:** #167

## Why

On 16 real discopt LP bases the shipped ordering (AMD on the AᵀA pattern, then
partial pivoting) reaches geomean fill **3.00x**; a threshold-Markowitz oracle on the
same bases reaches **1.11x** and never exceeds 1.89x. SuperLU/COLAMD is 3.24x — the
same algorithm class, so it could not have detected the headroom. The Suhl–Suhl peel
(#160) takes 3.00x to 1.52x but does nothing on exactly the bases discopt #1008 is
written around: on QPLIB_1157 it is 6.49x → 6.74x where Markowitz is 1.79x.

Under #1008's measured cost law (numeric cost ∝ fill² · nnz(B)) a 3.6x fill difference
is an order of magnitude of numeric LU work, against `LuNumeric` at 72.6% of LP wall.

## What this is not

Not a new *ordering*. Markowitz picks pivots from numeric values, so it does not fit
the `SparseLuSymbolic::analyze` → `SparseLu::factor` split at all: the symbolic phase
dissolves into the numeric one. It is a second factorization path that produces the
**same output object**.

## The one architectural commitment

`markowitz_factor` must fill exactly the fields `SparseLu::factor` fills — `L` in CSC
over pivot positions, `U` as per-row `(col_position, value)` with the diagonal first,
`perm`/`perm_inv`, `qcol`/`qcol_inv`, `uperm`/`uperm_inv` identity, `u_above`. Then
every downstream consumer — the triangular solves, the hyper-sparse route (#161B), the
Forrest–Tomlin update, `refactor()`, the growth monitors — works unchanged and is
covered by its existing tests. If that invariant holds, the blast radius of this
feature is one new module plus one entry point.

## Storage (differs from the oracle, same algorithm)

The oracle is dict-of-dict row-major. In Rust:

- **Active submatrix column-major with values** — `cols[j]: Vec<(row, val)>`.
- **Row-wise index only** — `row_cols[i]: Vec<usize>`, allowed to carry stale and
  duplicate entries; dedup with a dense mark array when the pivot row is gathered.
- **Exact counts** `rcnt[i]`, `ccnt[j]` maintained on every fill/cancel — the Markowitz
  cost is wrong if these drift.
- **Column-oriented rank-1 update**: for each alive column `j` in the pivot row,
  `col_j -= u_j · l`, done with a scatter workspace. Finding `u_j` costs a scan of
  `col_j`, which the update pays for anyway, so the row-major/column-major mismatch is
  free.

Pivot search reads column values directly, which is what the search actually needs
(`colmax` and the threshold test are per column).

## Pivot rule (identical to the oracle, which is the test target)

Cost `(rcnt[i]-1)·(ccnt[j]-1)`; threshold `|a_ij| ≥ u·max_k|a_kj|`; tie-break on larger
`|a_ij|`; scan alive columns in increasing count order; break early on the valid bound
`(c-1)·(minr-1) > best_cost`; stop on cost 0; Suhl cutoff after `max_search` columns
examined with a candidate in hand.

## The peel is subsumed

A singleton column has cost 0 and is taken immediately, so Markowitz performs the
Suhl–Suhl triangularization as a special case. That is why the oracle reaches 1.00x on
the 93–99% triangularizable bases. No separate peel step — and this is checkable:
on those bases the Markowitz fill must equal the peel's.

## Phases

1. **Tests first.** `PBQ = LU` residual on hand matrices and on random sparse matrices;
   `max|L| ≤ 1/u`; an arrow matrix where static ordering fills in and Markowitz must
   not; structural singularity → `SingularBasis`; solve-through-the-factor round trip.
2. **Implement** `src/lu/markowitz.rs` + `SparseLu::factor_markowitz`.
3. **Params.** `LuParams::markowitz_threshold` (0.1) and `markowitz_max_search` (8).
   Default factorization path unchanged — this is opt-in, like the dense bump.
4. **Corpus measurement** against the 16 bases in `/private/tmp/feral-fill`, reported
   against the oracle's own column. Rust fill must match the oracle's within noise; a
   large disagreement is a bug in one of the two, not a result.
5. **Cost.** Time it against `factor`. Fill is not the deliverable, wall is. If the
   search cost eats the fill win on this corpus, that is the finding and it gets
   reported as such.

## Stability, measured not assumed

Threshold pivoting bounds `max|L| ≤ 1/u` and bounds growth much more weakly than
partial pivoting. On QPLIB_1157 at `u = 0.1` the oracle gives `max|U|/max|B| = 81.8`
and `max|L| = 9.70` against SuperLU's 2.56 and 1.00. `max_growth` /
`should_refactor_growth()` already exist for this. `u = 0.01` buys 0.4% of fill for
353x growth and is not worth taking — the default stays 0.1.

## Out of scope

Applying Markowitz only to the residual bump after a peel. It is a search-cost
optimization, not a fill one (phase 5 decides whether it is needed).
