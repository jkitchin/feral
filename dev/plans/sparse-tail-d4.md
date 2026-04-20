# D.4 plan — tiny-n fast-path

**Authorized by:** `dev/research/sparse-tail-d3-d4-2026-04-19.md` §3,
§4, and the 2026-04-19-04 checkpoint "next session" item #1.
**Date opened:** 2026-04-20 (session 01).
**Prerequisite:** D.3 landed in session 2026-04-19-04. The
`dense_fast_factor` synthesis path and the
`should_use_dense_fast_path` gate predicate are already in
`src/numeric/factorize.rs`.

## Why this plan exists

The post-D.3 sparse-factor top-10 vs MUMPS contains five entries
with `n ≤ 11` (current bench run 2026-04-20-01):

| name              |   n  | feral(µs) | MUMPS(µs) |  ratio |
|-------------------|-----:|----------:|----------:|-------:|
| HS73_0308         |   7  |   118     |    9      | 13.1×  |
| PALMER1E_0484     |   8  |   129     |   10      | 12.9×  |
| HATFLDH_0083      |  11  |   106     |   10      | 10.6×  |
| PALMER1A_0034     |   6  |   130     |   13      | 10.0×  |
| KIRBY2LS_0274     |   5  |    99     |   10      |  9.9×  |
| HEART6LS_0418     |   6  |    96     |   10      |  9.6×  |

Prior work in `dev/tried-and-rejected.md` (Auto-ordering entry)
confirms the diagnosis: at n ≤ 10, `factorize_multifrontal` is
dominated by symbolic-phase overhead — not by the actual
floating-point work. D.4 skips symbolic analysis entirely for
tiny matrices.

The HS85_0022 diagnosis (2026-04-20-01 journal) gives a concrete
phase breakdown at n=68: symbolic 13 µs, numeric 25 µs — symbolic
is 36% of the pipeline at that size, and the fraction rises as
n shrinks. `dense_fast_factor` already skips symbolic; D.4
broadens the gate so it also fires on very-sparse tiny matrices.

## Goal

Route matrices with `n ≤ N_TINY` to the existing `dense_fast_factor`
unconditionally, independent of density. The existing
`should_use_dense_fast_path` gate stays as-is for the D.3 class
(n ≤ 128, ρ ≥ 0.25); D.4 adds a second disjunct for the tiny
class.

**Measurable win target:**

- Each of the six top-10 tiny-n rows above: factor ratio vs MUMPS
  ≤ 3× (from 10–13× today). The dense path's own wall time at
  n ≤ 11 is comfortably under 30 µs, so the synthesis overhead
  is what sets the floor.
- Corpus factor/MUMPS geomean: no worse than current (≈0.37).
  The class is too narrow to move the geomean visibly, but it
  cleans up the p99/max tail.
- No regression on any matrix outside the combined gate — the
  out-of-gate branch remains bit-identical.

## Non-goals

- Stack-buffer densify. The research note mentions it as a
  follow-up if synthesis overhead is still visible post-D.4.
  First cut reuses `CscMatrix::to_dense` on the heap; if the
  stage-3 bench still shows tiny-n above 3× MUMPS, a stack
  buffer is the next lever, not this plan.
- Pooling the dense scratch via `FactorWorkspace`. Same
  follow-up lane.
- Retuning `N_MAX` or `ρ_MIN`. Those are D.3 thresholds.
- Extending to the METHANL8LS class (n=31). That matrix is
  outside both the current D.3 gate and the proposed D.4 gate.
  Revisit only if the post-D.4 corpus still shows it in the
  top-10.

## Gate

Extend the existing predicate with a tiny-n disjunct. Constants
live next to the D.3 constants so future sweeps can move all
three together.

```rust
// src/numeric/factorize.rs
#[inline]
pub fn should_use_dense_fast_path(n: usize, nnz_lower: usize) -> bool {
    const N_TINY: usize = 16;  // D.4 — density-independent
    const N_MAX:  usize = 128; // D.3 — density-gated
    const RHO_NUM: usize = 1;  // ρ_MIN = 1/4
    const RHO_DEN: usize = 4;

    if n == 0 { return false; }
    if n <= N_TINY { return true; }                   // D.4 disjunct
    if n > N_MAX   { return false; }
    let lower_cells = n * (n + 1) / 2;
    nnz_lower * RHO_DEN >= lower_cells * RHO_NUM       // D.3 disjunct
}
```

Justification for `N_TINY = 16`:
- Captures all six observed top-10 tiny-n rows (max n=11).
- Matches the research-note proposal of 16.
- `n*n = 256` cells — densify is ≈2 KB, cheap even without a
  stack buffer.
- Leaves margin above the observed outliers so the next
  LEWISPOL-class (n=15) outlier also falls in.

## Implementation

Single predicate change; no new function surface. The existing
`dense_fast_factor` handles any `n ≥ 1` correctly — it allocates
a dense `SymmetricMatrix` sized `n*n`, runs `factor_frontal` on
the whole matrix, and synthesizes a one-node `SparseFactors`.
It does not assume the input is dense, only that the dense
factor kernel is the right choice.

The interaction with `FactorWorkspace` is unchanged — gate-hit
calls bypass the workspace on both disjuncts, per the D.3
design.

## Tests

All tests go in `tests/tiny_fast_path.rs`, a new integration
file. Each test is written against the current public API;
nothing in this section changes D.3's `tests/dense_fast_path.rs`
or the existing parity tests.

1. **Gate tiny-in, sparse**: synthetic n=8 matrix with 6 nnz
   (diag + 2 off-diag), density ≈ 0.17 (below D.3 ρ_MIN=0.25).
   `should_use_dense_fast_path(8, 6)` must return `true` now;
   previously returned `false`. Guards the gate change itself.
2. **Gate n=17 sparse**: just outside N_TINY. At density < 0.25
   the gate must still return `false` — this is the
   discriminator between D.4 and the unchanged D.3 predicate.
3. **Solve parity on a real tiny matrix**: pick one of HS73_0308,
   PALMER1E_0484, or HEART6LS_0418 from
   `data/matrices/kkt/`. Factor via the D.4 gate (default path)
   AND via `factorize_multifrontal_supernodal_with_workspace`
   (gate-bypass). Solve both against the sidecar RHS. Assert:
     - Inertia byte-equal (non-negotiable per CLAUDE.md hard rule).
     - `‖x_fast - x_mf‖∞ / (‖x_mf‖∞ + 1e-300) ≤ 1e-10`.
4. **Zero-pivot tiny**: synthetic n=4 indefinite matrix where
   the kernel's zero-pivot path fires. Both paths must produce
   byte-equal inertia under `ZeroPivotAction::ForceAccept`. This
   test is already written for D.3 (see
   `tests/dense_fast_path.rs::…zero_pivot…` if present); if it
   uses n=4 it already exercises the D.4 gate. If not, add it
   here.
5. **Gate boundary n=16**: synthetic sparse n=16 (nnz ≤ 10) is
   gated in now. Factor + solve must round-trip to tolerance.
6. **Determinism**: factor the same in-gate tiny matrix twice;
   assert bit-equal `SparseFactors`. (D.3 already has this
   downstream; if the existing test is parameterized over
   matrices it can just add a tiny case; otherwise add here.)

Test 3 is the primary correctness oracle. Tests 1 and 2 are
gate-truth unit tests that can run without loading any sidecar.

## Measurement plan

### Stage 1 — the six top-10 rows

Extend `bin/d3_probe.rs` (or a new `bin/d4_probe.rs`) to time
the six observed tiny-n rows: HS73_0308, PALMER1E_0484,
HATFLDH_0083, PALMER1A_0034, KIRBY2LS_0274, HEART6LS_0418.
For each, report:

- pre-D.4 full cold wall time (via `factorize_multifrontal`
  with the old gate — still applicable because none of these
  are in the D.3 gate at ρ < 0.25).
- post-D.4 full cold wall time (via `factorize_multifrontal`
  with the new gate — routes to `dense_fast_factor`).
- phase breakdown of the dense path on one representative
  row (symbolic-skipped: compute_scaling, densify,
  factor_frontal, synthesis).

Expected: each row drops below 30 µs; ratio vs MUMPS below 3×.

### Stage 2 — corpus bench

Full 154 588-matrix `cargo run --release --bin bench`. Acceptance:

- Tiny-n rows named above drop out of the top-10.
- Geomean no worse than 0.37 pre-D.4 (tolerance: ± 0.01 for
  run-to-run noise given that the bench is single-shot per
  matrix).
- No matrix outside the combined gate worse by > 20% vs its
  pre-D.4 timing (same 20% noise budget as the D.3 plan).

## Rollout

1. Plan (this file) — commit.
2. Tests (red) — commit. Tests 1–6 in `tests/tiny_fast_path.rs`.
   Tests 1, 2, 5 fail because the predicate hasn't been
   broadened; test 3 fails because gate-hit factors return
   different byte-identity vs multifrontal on a sparse tiny
   matrix (currently routes to multifrontal on both sides).
3. Broaden `should_use_dense_fast_path` — commit. Tests green.
4. Stage 1 measurement on the six named rows + commit a short
   results file under `dev/results/lever-d4/stage1.md`.
5. Stage 2 corpus bench — commit results + decision (D.4 done
   vs widen N_TINY vs pursue stack-buffer follow-up).
6. Session checkpoint.

## Risks

- **`dense_fast_factor` on very-sparse input silently pessimizes
  something downstream.** Mitigation: test 3 is the oracle —
  byte-equal inertia + tight solve residual on a real matrix.
  If parity fails, the synthesis is wrong, not the gate.
- **Widened gate also captures unintended cases that regress.**
  Mitigation: stage 2's "no matrix worse by > 20%" check. The
  N_TINY = 16 threshold is narrow by construction.
- **CscMatrix::to_dense allocates each call on tiny inputs.**
  At n=16 that's a 2 KB allocation — cheap, well below the
  ≈15 µs the matrix currently pays for symbolic. If
  stage-1 shows it dominating anyway, the stack-buffer
  follow-up is the answer.
- **The six observed rows were from one bench run.** Run
  noise: stage-1 probe should average ≥ 50 cold reps to
  avoid replacing one single-shot artifact with another.

## What this plan does not do

- Touch `dense_fast_factor` itself. The function stays as-is;
  only the gate predicate changes.
- Authorize a stack-buffer or pooled-scratch follow-up.
- Revisit D.3 thresholds.
- Expand the D.4 class beyond `n ≤ N_TINY = 16`.
