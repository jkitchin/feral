# Sparse-tail levers D.3 (small-dense) + D.4 (tiny-n) — scoping note

**Date:** 2026-04-19 (session 04 continuation).
**Predecessor:** `sparse-tail-perf-2026-04-19.md` §5, §9 (D.1
exhausted per ex-ante exit clause — corpus 0.48→0.46, below the
≤0.40 target; FactorWorkspace fully landed).

## Purpose

Pivot from D.1 (workspace reuse) to D.3/D.4 per the plan exit
clause. This note inventories the tail that D.1 could not
touch, scopes D.3 and D.4 independently, and proposes a
narrowest-first sequence. It does **not** authorize code.

## 1. Current tail, post-D.1

From `cargo run --release --bin bench` after the step-6 commit:

```
ratio          count   geomean   p50    p90    p99   max
factor/MUMPS  153560    0.46    0.33   1.84   3.48 128.34
```

Top-10 by factor-ratio vs MUMPS:

| name              |   n  | feral(µs) | MUMPS(µs) |  ratio |
|-------------------|-----:|----------:|----------:|-------:|
| CRESC50_0331      |  306 |    353698 |        87 |  4065x |
| LEWISPOL_1156     |   15 |      2458 |        11 |   223x |
| HAHN1_0178        |  715 |     29267 |       197 |   148x |
| NET1_0371         |  214 |      7732 |        55 |   140x |
| LEWISPOL_1164     |   15 |      1339 |        11 |   121x |
| METHANL8LS_0899   |   31 |      2312 |        20 |   115x |
| HAHN1_0177        |  715 |     21442 |       196 |   109x |
| HAHN1_0136        |  715 |     19622 |       195 |   100x |
| HAHN1_0132        |  715 |     19045 |       191 |   100x |
| HAHN1_0181        |  715 |     18856 |       191 |    99x |

The tail splits into three structural classes:

- **Arrow-KKT class** (CRESC50, HAHN1_*, NET1): n = 214–715,
  dense root frontal, dominated by scalar BK on the dense root.
  These are **Phase 2.4.1/2.4.2 territory** (blocked BK + SIMD
  Schur kernel). Not addressable by D.3 or D.4.
- **Tiny-n class** (LEWISPOL n=15, METHANL8LS n=31): small matrices
  where the multifrontal scaffolding dominates. **D.4 target.**
- **Small-dense class** (TRO3X3 n=69 37% dense, visible in the
  alloc_probe panel): still sparse-eligible but multifrontal is
  overkill. **D.3 target.** After D.1, TRO3X3_0013 reports 83 µs
  vs MUMPS (no MUMPS time captured in current top-10 because its
  ratio dropped out of the top-10 — verify separately).

## 2. D.3 — dense fast-path for small-dense matrices

**Claim.** Matrices with `n ≤ N_max` and density ≥ ρ_min benefit
from going directly to the dense BK kernel (`factor` in
`src/dense/factor.rs`) and skipping symbolic analysis + contribution
assembly entirely.

**Gate proposal.** At the `Solver::factor` entry, compute
`density = nnz / (n * (n+1) / 2)` for the lower triangle. If
`n ≤ 256` **and** `density ≥ 0.25`, route to a
`dense_fast_factor` that materializes `SymmetricMatrix` from
`CscMatrix::to_dense()` and calls the existing BK kernel.

**Impact estimate.** Narrow. TRO3X3 class is the clearest win:
TRO3X3_0013 (n=69, nnz=1764, density ≈ 75% of lower triangle)
should drop from ~80 µs to ≤20 µs — the dense kernel's own
timing. Estimated corpus coverage: 50–500 matrices.

**Risk.** Crossover tuning. Wrong thresholds push bad-fit
matrices to the dense path where they cost more. Mitigation:
pick thresholds from measurement, not intuition, and add a
parity-style test confirming the two paths produce identical
factors on an in-range matrix.

**Cost.** Small. The dense path exists; we need a gate, a
wrapper that produces the solver-visible state in the same
shape as `SparseFactors`, and tests.

**What D.3 cannot fix.** The Class-B arrow-KKT tail (CRESC50,
HAHN1, NET1). Those have `n = 200–700` but dense ROOT frontal —
they need the dense-kernel speedup, not a routing decision.
Phase 2.4 territory.

## 3. D.4 — tiny-matrix fast-path (n ≤ N_tiny)

**Claim.** Matrices with `n ≤ ~10` eat a ~100 µs flat overhead
in the multifrontal path (symbolic + scaling + frontal setup)
that dwarfs the actual floating-point work. A specialized path
that skips all of the multifrontal machinery and inlines a
single BK sweep on a small stack-resident buffer would fix this.

**Gate proposal.** `n ≤ 16` at `Solver::factor` entry routes to a
`tiny_factor(matrix, params) -> (SparseFactors, Inertia)` that:
- densifies to a stack array (e.g. `[f64; 256]` with `n*n ≤ 256`)
- runs the existing scalar BK kernel inline
- synthesizes a single-supernode `SparseFactors` for solve
  compatibility

**Impact estimate.** ~4–8 top-10 outliers fixed (LEWISPOL,
METHANL8LS class). Corpus geomean moves only slightly — many
tiny matrices were already passing perf targets — but the tail
gets cleaner and IPM problems with ≤16-variable subproblems
benefit linearly.

**Risk.** Maintenance. A second BK integration point doubles
the surface for pivoting bugs. Mitigation: the small-n path
should be a thin wrapper that calls the *same*
`dense::factor::factor` and only owns the densify + solve-state
synthesis.

**Cost.** Medium. The synthesis step (turning a dense BK result
into a single-supernode `SparseFactors`) is the non-trivial
bit. Needs its own parity tests and `solve_sparse` has to tolerate
the synthesized factor shape.

**What D.4 cannot fix.** Everything above n=16. Same arrow-KKT
tail as D.3.

## 4. Sequence recommendation

**D.3 first.** Rationale:

- Narrower implementation (a routing gate + a thin wrapper on
  an existing kernel), lower integration-bug risk.
- The `SparseFactors` synthesis for the dense path is a
  prerequisite for D.4 anyway — D.4 has the same shape but
  uses a smaller threshold and (possibly) a stack buffer.
- D.3 has a larger corpus coverage estimate (50–500 matrices
  vs ~4–8 outliers). Not because it's more transformative —
  both levers are narrow — but because D.3 is cheap per
  matrix it touches.

**D.4 second.** Only if D.3's synthesis path proves sound and
the remaining LEWISPOL/METHANL8LS tail still shows >100× ratio.
Skip D.4 if D.3 already captures them at the `n ≤ 16, density
≥ 0.25` gate.

**Phase 2.4 deferred.** Arrow-KKT tail is the biggest
blast-radius lever but 4–6 sessions per `tasks.org`. Revisit
after D.3 lands and we have a cleaner picture of the non-arrow
tail.

## 5. Ex-ante success target for D.3

- TRO3X3_0013: ≤ 25 µs (from 83 µs post-D.1, 77 µs MUMPS).
- Corpus factor/MUMPS geomean: 0.46 → ≤ 0.44.
- No regression on any matrix *outside* the gate — the
  oracle-matched branch path is byte-identical.

If TRO3X3 doesn't drop to ≤25 µs, either the gate is wrong or
the `CscMatrix::to_dense` densify is itself the cost center —
measure before widening the gate.

## 6. Not authorized in this note

- Implementation.
- Choosing final thresholds (that's stage-1 of the plan after
  authorization — profile a panel of in-range / at-boundary /
  out-of-range matrices to pick `N_max` and `ρ_min`).
- Changes to `Solver` beyond a gate at the top of `factor()`.

Next step on authorization: a short `dev/plans/sparse-tail-d3.md`
that names the gate API, the dense-path synthesis function, and
the test strategy (parity vs multifrontal on in-range matrices,
correctness on a boundary case, no-regression on out-of-gate).
