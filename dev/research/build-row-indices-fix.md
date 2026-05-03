# build_row_indices upper-triangle pollution — diagnosis and fix

**Date**: 2026-05-03
**Author**: factor session — Phase A2 follow-up
**Status**: Fixed in factorize.rs:2257-2298 + debug_assert at factorize.rs:1469-1485
**Test**: tests/build_row_indices_trailing_invariant.rs (8 tests)

## Symptom

On PoissonControl K=158 (n=74,892, nnz_lower=199,080) feral's
`factor_nnz` was 46.7M while the textbook L-fill prediction
(Σ col_counts via Gilbert-Ng-Peyton) was 2.4M — a **19× inflation**.
The 234× factor_nnz / nnz_lower ratio was far above MUMPS-typical
10–50× and explained a ~650× factor-time gap vs. MUMPS on the same
matrix.

Diagnostic instrumentation in `src/bin/diag_poisson_kkt.rs` showed the
worst-case inflation at the etree root supernode (idx=1429,
first_col=7467, ncol=33): symbolic predicted nrow=33, numeric reported
nrow=3545, with 3512 of those rows < first_col. Trailing rows below
the supernode's own column range are structurally invalid in a
multifrontal frontal — they correspond to columns that should have
been eliminated by ancestors of those rows.

## Root cause

`build_row_indices` (factorize.rs:2241-2308) collected trailing rows
as:

```rust
for j in first_col..first_col + own_ncol {
    for k in full_pattern.col_ptr[j]..full_pattern.col_ptr[j + 1] {
        let r = full_pattern.row_idx[k];
        if !build_seen[r] { ... build_trailing.push(r); }
    }
}
```

`full_pattern = matrix.symmetric_pattern()` is the **fully symmetrized**
A pattern (csc.rs:186-239), constructed by adding each `(i, j)` AND
`(j, i)` for every off-diagonal entry. Iterating column j therefore
yields both lower-triangle entries (r > j, legitimate downstream rows)
and upper-triangle entries (r < j, columns already eliminated by
ancestors of those rows in the etree).

The symbolic side does the right thing — `column_counts_gnp`
(column_counts.rs:135) explicitly skips `partner <= i`, counting only
lower-triangle interactions. The textbook L-fill (Σ col_counts) is
correct. Numeric build_row_indices was the only place the upper
triangle leaked in.

The pollution propagates upward through child contrib blocks: each
parent rebuilds its frontal, again iterates the symmetric pattern,
again pulls in upper-tri rows from its own_cols, and additionally
absorbs whatever rogue rows its children carried in their contribs.

## Fix

Two changes to `build_row_indices`, both filtering
`r < first_col + own_ncol`:

1. **Native pattern collection** (factorize.rs:2274-2287) — guards the
   loop over `full_pattern.col_ptr[j]` against upper-triangle rows.
   This is the source of the pollution.

2. **Children's contrib trailing** (factorize.rs:2289-2298) —
   defensive. With change 1 in place, no clean child can produce
   trailing rows below its parent's first_col + own_ncol, but the
   filter guards against historical or future regressions.

Plus a debug-only invariant assertion right after `build_row_indices`
returns (factorize.rs:1469-1485) that verifies every trailing row is
≥ first_col + own_ncol. The assertion fires on the unfixed code (we
verified by adding the assert before the filter changes) and stays
silent after the fix.

## Why it was a performance bug, not a correctness bug

- The polluting rows are upper-triangle entries: A[r, j] for r < j.
  Numeric assembly only writes lower-triangle interactions into the
  frontal, so the polluting rows received zeros during assembly.
- Bunch-Kaufman pivoting on a row of zeros either picks a tiny/zero
  diagonal (caught by the threshold or zero-pivot policy) or skips it.
  At a root supernode with `n_delayed_in == 0` the rogue rows are dead
  weight that get absorbed into the contrib block but never affect any
  pivot decision.
- The full feral test suite (216 lib + integration tests) and inertia
  parity gates pass identically before and after the fix — confirming
  the bug was purely structural inflation, not numerical wrongness.

## Before / after numbers

PoissonControl K=50, AMD ordering, ScalingStrategy::Identity,
pivtol = 1e-8 (issue #2 default):

|             | factor_nnz | factor time | inertia        |
|-------------|-----------:|------------:|---------------:|
| before fix  | 1,363,445  | 231,075 µs  | (+5000, −2500) |
| after fix   |   323,643  |   3,542 µs  | (+5000, −2500) |
| symbolic Σ col_counts | 143,667 |   —    |        —       |

K=158 (n=74,892):

|             | factor_nnz | factor time | inertia          |
|-------------|-----------:|------------:|-----------------:|
| before fix  | 46,734,661 | seconds     | (+49928, −24964) |
| after fix   |  4,610,269 |  85,099 µs  | (+49928, −24964) |
| symbolic Σ col_counts | 2,447,001 |    —   |          —       |

Factor time dropped 65× on K=50 and roughly 100× on K=158. Inertia
unchanged. `factor_nnz` after fix sits at 1.6–1.9× the symbolic
prediction; the gap is delayed-pivot fill plus the per-supernode
contribution-block storage that Σ col_counts does not include.

## Test fixtures

`tests/build_row_indices_trailing_invariant.rs` covers four matrices
all sized n > 16 to bypass the dense fast-path
(`should_use_dense_fast_path`, factorize.rs:685, gates at
N_TINY = 16):

- `tridiag_spd(30)` — chain etree, simplest fill pattern
- `poisson_2d_spd(5)` (n=25) — 2D 5-point stencil, branching etree
- `small_kkt_saddle(20, 5)` (n=25) — saddle-point with J coupling
  and −δ_c·I on equality rows; mirrors Ipopt augmented system
- `disjoint_tridiag(12)` (n=24) — two disconnected tridiagonal blocks;
  guards against phantom upper-triangle rows in disjoint forests

Each fixture asserts (a) the trailing-row floor invariant on every
supernode (`r >= first_col + own_ncol`), and (b) on delayed-free
supernodes, `numeric.nrow == symbolic.nrow`. The tests fail on the
unfixed code and pass after the filter.

## Related

- Issue #2 (1e-8 pivot default): orthogonal — about pivot threshold,
  not fill pattern.
- Issue #3 (ScotchND silent fallback): cosmetic; ND ordering on the
  same matrix saved only ~3% factor_nnz vs AMD before this fix; with
  proper L fill, the AMD vs ND choice matters less for KKT.
- The remaining 1.6–1.9× gap between numeric factor_nnz and Σ
  col_counts is expected: contribution-block storage and delayed-pivot
  inflation. A separate research note can quantify each component if
  needed.

## Files touched

- src/numeric/factorize.rs:1469-1485 (debug_assert)
- src/numeric/factorize.rs:2257-2298 (filter on r < first_col + own_ncol)
- tests/build_row_indices_trailing_invariant.rs (new, 8 tests)
