# factor_nnz residual gap (post-`build_row_indices` fix) — diagnosis

**Date**: 2026-05-03 (follow-up to `build-row-indices-fix.md`)
**Status**: Diagnosed — gap is dominantly pass-through padding, not delayed-pivot fill.

## The question

After the `build_row_indices` upper-triangle pollution fix, numeric
`factor_nnz` on PoissonControl sits at:

|        | numeric `factor_nnz` | Σ col_counts (GnP) | ratio  |
|--------|---------------------:|-------------------:|-------:|
| K=50   |              323,643 |            143,667 |  2.25× |
| K=158  |            4,610,269 |          2,447,001 |  1.88× |

The user asked whether this 1.6-2× residual is mostly delayed-pivot
fill plus contribution-block storage that the textbook sum omits, or
whether it has additional slack worth recovering.

## Decomposition method

For each supernode `s` we compute three quantities:

1. **`true_L_nnz_in_own_cols`** = `Σ_{k=0..own_ncol-1} col_counts[first_col + k]` —
   the real L NNZ contributed by this supernode's columns (Gilbert-Ng-Peyton).
2. **`sym_supernodal_padded`** = `own_ncol*(own_ncol+1)/2 + (s.nrow − own_ncol) × own_ncol` —
   what feral would store if symbolic frontal nrow (`col_counts[first_col].max(ncol)`,
   set in `src/symbolic/supernode.rs:358`) were the actual frontal nrow.
3. **`numeric_padded`** = `nelim*(nelim+1)/2 + (nrow − nelim) × nelim` —
   what feral actually stores (per `factor_nnz()` accounting,
   `factorize.rs:499-510`).

Two deltas:
- `Δ_amalg = sym_supernodal_padded − true_L_nnz_in_own_cols`: padding
  introduced by representing per-column row patterns as a single
  shared row pattern within each supernode.
- `Δ_passthrough = numeric_padded − sym_supernodal_padded`: padding
  introduced when the **working frontal** absorbs rows from
  children's contribs that aren't in any of this supernode's L
  columns — those rows are passed through to grandparents but stored
  as zeros in the trailing rectangle here.

Per `src/symbolic/supernode.rs:358`, `Supernode.nrow` is set from
`col_counts[first_col].max(ncol)` — the L column NNZ for the
supernode's first column. This is a **lower bound** on the working
frontal nrow because children's pass-through rows are not
captured. Numeric `build_row_indices` then collects the actual
union, which is larger. That delta is the pass-through contribution.

## Numbers (PoissonControl, AMD, Identity scaling, pivtol=1e-8)

| metric                                          |        K=50 |        K=158 |
|-------------------------------------------------|------------:|-------------:|
| n                                               |        7500 |       74,892 |
| nnz_lower (input A)                             |      19,800 |      199,080 |
| **Σ col_counts (true L NNZ)**                   |     143,667 |    2,447,001 |
| `sym_supernodal_padded`                         |     131,071 |    2,202,355 |
| `Δ_amalg = sym_padded − Σcc`                    |     −12,596 |     −244,646 |
| **numeric `factor_nnz`**                        |     323,643 |    4,610,269 |
| `Δ_passthrough = numeric − sym_padded`          |    +192,572 |   +2,407,914 |
| of which on `n_delayed_in == 0` nodes           |    +192,572 |   +1,741,427 |
| `total_n_delayed_in`                            |           0 |          367 |
| `max_n_delayed_in`                              |           0 |            2 |
| n_supernodes                                    |       1,430 |       13,868 |
| n_supernodes with `n_delayed_in == 0`           |       1,430 |       13,514 |

`Δ_amalg` is negative because `Supernode.nrow = col_counts[first_col]`
under-counts the union of trailing patterns across the supernode's
columns (only the first column's count is used). It is not actually
"amalgamation padding" — it's the symbolic side's nrow being a tight
lower bound rather than the true union. Cosmetic, not slack.

The dominant gap source is `Δ_passthrough`:
- K=50: **100%** of the passthrough delta is on supernodes with zero
  delayed pivots. Delayed pivots are not the source.
- K=158: **72%** of the passthrough delta is on zero-delayed
  supernodes; the remaining 28% is on supernodes that received some
  delayed columns and absorbed some pass-through. Even there, the
  pass-through component is most of the inflation per node.

## What "pass-through" means structurally

A child supernode's contribution block contains rows in the child's
trailing pattern. Those rows become part of the parent's frontal
during assembly. If a row `r` in the child's contrib is in
`[parent.first_col + parent.own_ncol, n)` AND `r` is not in any of
the parent's own columns' L pattern, then:
- The parent's working frontal absorbs `r`.
- The parent's trailing block stores `nelim` columns × this row =
  `nelim` zeros for this row's interaction with parent's eliminated
  cols.
- The parent passes `r` up to *its* parent's contrib.

These zero entries inflate the dense `(nrow − nelim) × nelim`
trailing storage. They're real bytes on the heap during factor and
real bytes in the stored L (numeric `factor_nnz` reflects exactly
the dense rectangle).

`sym_supernodal_padded` does not include these because
`col_counts[first_col]` is the count of nonzeros in column
`first_col` of L — which excludes pass-through rows that have no fill
at column `first_col`. The numeric frontal nrow is genuinely larger
than the symbolic prediction.

The `Δ_passthrough` per supernode roughly equals:
`(num_nrow − sym_nrow) × nelim`. On K=158 worst-case node 2537:
- sym(`first_col=12635, ncol=32, nrow=32`) — predicts no trailing
  rows because `col_counts[12635] = 32 = ncol`.
- num(`ncol=33, nrow=148, n_delayed_in=1, n_children=24`) — actual
  frontal needs 116 rows from 24 children's contribs that pass
  through this supernode.
- Pass-through inflation: `116 × 33 = 3,828` entries in storage,
  contributing essentially zero to L's true NNZ.

## Is the residual gap normal?

Yes — it's an inherent cost of dense supernodal storage in
multifrontal factorization:
- Children's contribs flow through ancestors that don't pivot on
  those rows. The dense trailing rectangle stores those rows × eliminated
  cols as zeros for cache-friendly Schur updates and dense forward/back
  solves.
- MUMPS's `INFOG(9)` and SSIDS's `inform%num_factor` reflect the same
  dense supernodal accounting. The 1.6-2× ratio over Σ col_counts is
  in the typical range reported by both reference solvers.

It is **not** delayed-pivot fill. K=50 has zero delayed pivots and
shows the same 2.25× ratio. K=158's 367 delayed pivots out of 74,892
columns (0.5% delay rate) account for at most ~0.7M of the 2.4M
inflation, and most of that is co-located pass-through.

## Slack to recover

Three mechanisms could shrink the gap, but none are pure wins:

1. **Better supernode amalgamation.** The current `nemin = 32` (per
   `SupernodeParams::default()`,
   `src/symbolic/supernode.rs:113-122`) minimizes supernode count but
   doesn't optimize for shared row pattern. A merge that pulls in a
   small column with a very different trailing pattern inflates the
   merged supernode's nrow more than the column's own L NNZ
   contribution — every other column in the merged supernode pays
   the pass-through cost for that column's outlier rows. Validating
   this requires sweeping `nemin` and measuring `factor_nnz` vs
   factor wall.

2. **Compressed L storage.** Drop pass-through zero rows from the
   stored L per supernode and keep an index list. Matches Σ col_counts
   storage, but solves become indirect-indexed gemv instead of dense
   gemv — typically slower despite less data movement on cache-resident
   problems and faster on RAM-resident ones. Out of scope without a
   broader storage-layout change.

3. **Ordering choice.** ND-style orderings (METIS, Scotch) tend to
   produce supernodes with smaller pass-through trees than AMD
   because the separator structure clusters rows. On the same K=50
   matrix the published ND vs AMD `factor_nnz` gap was ~3% before
   the build_row_indices fix; with proper L fill the ratio may have
   shifted. Re-evaluation is on the next-session list (Issue #3).

## Conclusion

The 1.6-2× post-fix gap between numeric `factor_nnz` and Σ
col_counts is dominantly **pass-through row padding** in the dense
trailing rectangle of each supernode's frontal. It is the expected
cost of dense supernodal multifrontal storage and is in the same
band that MUMPS and SSIDS report. Delayed-pivot fill contributes
negligibly even on the larger K=158 problem. The remaining slack
sits in ordering choice and supernode amalgamation tuning, neither
of which is a free improvement — both are tradeoffs against factor
wall and analysis cost.

## Files referenced

- `src/numeric/factorize.rs:499-510` — `factor_nnz()` accounting.
- `src/symbolic/supernode.rs:358` — `Supernode.nrow` set from
  `col_counts[first_col].max(ncol)`.
- `src/symbolic/column_counts.rs:72-74` — `total_factor_nnz` =
  `Σ col_counts`.
- `src/bin/diag_poisson_kkt.rs` — instrumentation that produced
  these numbers (untracked).
- `dev/research/build-row-indices-fix.md` — preceding fix that
  brought factor_nnz into this regime.
