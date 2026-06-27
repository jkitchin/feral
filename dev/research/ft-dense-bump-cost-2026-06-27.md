# FT update cost on dense-inverse bases — confirmation + fix (issue #89)

Date: 2026-06-27
Refs: issue #89, #87 (the prior FT row-elimination fix, `a9cea82`),
`dev/research/ft-row-elimination-design-2026-06-21.md`.

## Claim under test

On `main` after the #87 fix, `SparseLu::update()` cost scales with the **factor
fill** (`factor_nnz`), not with the bump / entering-column nnz. On set-covering
LP bases (sparse columns, ~6 nnz, but a *dense inverse*) a single column
replacement costs ~O(factor_nnz) → the update dominates a revised-simplex solve.
Second claim: `eta_ops()` undercounts the true per-update cost, so a caller's
"refactor when work is large" heuristic never fires in time.

## Confirmation (reproducer B, `examples/ft_dense_bump.rs`, release, this host)

Entering column fixed at 8 nonzeros throughout:

```
m=  256  factor_nnz/m= 146.4  update_avg=    1460.8 us/upd  avg_last_eta_ops= 210
m=  512  factor_nnz/m= 291.7  update_avg=    9647.1 us/upd  avg_last_eta_ops= 414
m= 1024  factor_nnz/m= 548.3  update_avg=   49620.5 us/upd  avg_last_eta_ops= 765
m= 2048  factor_nnz/m=1115.3  update_avg=  349883.3 us/upd  avg_last_eta_ops=1516
```

- `factor_nnz` ~ m² (the inverse is dense). `update_avg` grows ~ m^2.6, i.e.
  ~ `factor_nnz^1.3` — confirms cost ∝ fill, NOT the fixed entering nnz.
- `avg_last_eta_ops` grows only ~ O(m), so **µs per recorded eta-op** is
  7 → 23 → 65 → 231 as m doubles — i.e. the per-op build cost grows ~m. The
  recorded eta op-count undercounts true build work by exactly the row density.

Both claims confirmed.

## Root cause (why this is fundamental, not a bug)

`eliminate_pivot_row` (`src/lu/sparse_update.rs`) is a single-row sparse forward
sweep: for each eliminated sub-diagonal column `c` of the pivotal row it scatters
`row_r -= mult · u_rows[c]`, costing `nnz(u_rows[c])`. On a dense-inverse basis
`U` is dense, so each `u_rows[c]` is O(m) and the bump spans O(m) ranks ⇒
O(m²) = O(factor_nnz) per update. This is the irreducible cost of triangularizing
one row against a dense `U`; there is **no asymptotically cheaper Forrest–Tomlin
path** when the bump is dense. Worse, the spike ρ = B⁻¹·aₙₑw is itself dense on
these bases, so even *computing* it (`compute_spike`: forward L-solve + eta
replay) is already O(factor_nnz) — the 8-nnz entering column does not bound
anything once the inverse is dense.

Answer to issue Q1: no cheaper FT path; the caller must refactor aggressively.
Answer to issue Q2 (the actionable fix): `eta_ops()` measures *solve-replay*
cost (one O(1) axpy per op) — correct for warm solves but the wrong signal for
scheduling refactorization. We need a counter that reflects the true per-update
**build** cost.

## Fix

1. Add an accurate per-update work counter `last_update_work` (and cumulative
   `update_work` since the last factor/refactor) measuring the actual scalar
   multiply-add operations the update performs:
   - the spike solve in `compute_spike` (L forward-solve flops + eta-replay
     flops), and
   - the row-elimination scatters in `eliminate_pivot_row`.
   These are the dominant, fill-proportional terms. Exposed via
   `last_update_work()` / `update_work()`.
2. `should_refactor()` — advisory predicate (does not change `update()`'s
   behaviour): true once cumulative `update_work()` since refactor reaches
   `factor_nnz()`, i.e. the update chain has cost about one refactorization.
   Callers (e.g. discopt's simplex driver) drive refactor scheduling off this
   instead of `eta_ops()`.
3. Document the dense-bump reality at the `update_sparse` doc-comment.

`eta_ops()` is retained unchanged — it is the correct *solve* cost witness used
by the scalability guard test. The new counters are the *update* cost witness.

## Oracle for the new counter (external, hand calculation)

A 3×3 hand-built bump with a known elimination: replacing a column so the spike
forces exactly one sub-diagonal elimination against a known-length upper row lets
the elimination scatter count be hand-derived, independent of the counter's own
bookkeeping. The dense-scaling test asserts the *qualitative* invariant the
counter exists to expose — `last_update_work ≫ last_eta_ops` and grows with
`factor_nnz` — which the pre-fix `eta_ops` cannot satisfy.
