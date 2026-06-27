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

The *arithmetic* lower bound is real: the spike ρ = B⁻¹·aₙₑw is dense on these
bases, so even computing it (`compute_spike`: forward L-solve + eta replay) is
Ω(factor_nnz) — the 8-nnz entering column does not bound anything once the inverse
is dense. And `eliminate_pivot_row` triangularizes one row against `U` at
O(factor_nnz). For a genuinely **dense factor** (`factor_nnz/m ≈ m/2`) that is
Θ(m²) and unbeatable by any FT variant — you must form the dense spike.

**BUT the first version of this note over-generalized** that to "no asymptotically
cheaper FT path, period." Two corrections:

1. **"Dense inverse" ≠ "dense factor."** The real `sc2000x800` bases have a dense
   inverse but *sparse factors* (`factor_nnz/m` = 2–39). There the update is
   Θ(factor_nnz) ≈ Θ(fill·m), **not** Θ(m²); the issue's own table (0.01→0.72 ms
   as fill 2→39) is *linear in fill*. So "O(m²) is fundamental" is the wrong
   framing for the bases that actually hurt.

2. **The dominant cost was not the arithmetic at all — it was `u_above` index
   churn**, and that part was a genuine, fixable bug (see below).

## The real bottleneck: `u_above` reindex was O(m³) (profiled)

Per-phase timing of one update on the dense reproducer (m=2048, ns→µs/update):

```
spike (FTRAN)        1.3 ms
set_column_r         1.8 ms
eliminate_pivot_row  2.8 ms   <- the actual Θ(factor_nnz) arithmetic
commit (u_above)   537   ms   <- 99% of the time
```

`commit` scaled 1.7→13.6→76→537 ms over m=256..2048 — **~m^2.7, toward O(m³)**,
while the arithmetic stayed at ~Θ(factor_nnz). Root cause: the commit block
re-indexed *every* changed row wholesale —

```rust
for (i, old_row) in saved.iter() {       // O(bump) ≈ O(m) changed rows
    self.unindex_above(i, old_row);      // O(rowlen) binary-search + Vec::remove
    self.index_above(i, new_row);        // O(rowlen) binary-search + Vec::insert
}                                         //   each shift O(|u_above[c]|)
```

On a dense bump that is O(bump · rowlen · shift) ≈ O(m·m·m) = **O(m³)** of
sorted-`Vec` memmove traffic, for entries that did not change: of the O(m) changed
rows, all but row `r` only gained/lost their single column-`r` entry.

## Fix (this is the performance fix, beyond the counter)

`set_column_r` only rewrites column `r`; `eliminate_pivot_row` only rebuilds row
`r`. No other row's column membership changes (the FT win). So the commit needs
only:

- **(a)** `u_above[r] = supp \ {r}` — column `r`'s holders are exactly the spike
  support (one clear+extend, O(|supp|)); and
- **(b)** unindex/index **row `r` alone** against its old/new column set.

This drops commit from O(m³) to O(m²) = the arithmetic order. Validated by the
`lu-ft-invariant-check` feature (rebuilds expected `u_above` from `U`, asserts
exact equality after every update): all 28 `lu` tests pass with it on.

**Measured (dense reproducer, release):**

```
m      update_avg before → after      speedup   time/work after
256        1461 → 112 µs                13×       2.6 ns/work-unit
512        9647 → 394 µs                24×       2.6
1024      49620 → 1300 µs               38×       2.6
2048     349883 → 5612 µs               62×       2.6
```

Scaling after the fix is ~m² (constant 2.6 ns per counted multiply-add at every
m) — the update now sits on the Θ(factor_nnz) arithmetic floor.

## Answers to the issue questions (revised)

- **Q1 (cheaper FT path):** For a *dense factor*, no — Θ(factor_nnz)=Θ(m²) is the
  floor. For the *sparse-factor/dense-inverse* bases in the issue, the per-update
  cost is Θ(factor_nnz) ≪ Θ(m²) and now near-optimal after removing the O(m³)
  index churn. A *product-form* FT (fixed U₀ + row etas, never mutating U) could
  further avoid fill compounding across a chain, trading into a growing eta chain
  — a larger redesign, not pursued here.
- **Q2 (true cost signal):** `eta_ops()` measures *solve-replay* cost (O(1)/op),
  the wrong signal for refactor scheduling. Added `last_update_work` /
  `update_work` (true build flops) and `should_refactor()`.

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
