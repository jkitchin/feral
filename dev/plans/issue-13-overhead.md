# Issue #13 — Per-supernode fixed-overhead reduction

## Context

Re-profile of long-tail corpus (2026-05-12, `cargo run --bin diag_supernode_cost --release`):
ns/sup 600–1900 dominates ns/nnz 30–165 on CRESC100 / ACOPR30 / HAIFAM / KIRBY2.
Per-supernode fixed overhead is the current bottleneck. Issue #13 is the parking
issue that gates re-engaging #9 (32×32 SIMD) and the APP work in #10.

## Acceptance (verbatim from issue #13)

- `cargo run --bin diag_supernode_cost --release` shows ns/sup reduced on at least
  the CRESC100 / ACOPR30 / KIRBY2 cluster.
- `cargo run --bin bench --release` shows dense small-frontal p90 < 1.30 (currently
  1.33) **or** medium p90 < 1.60 (currently 1.70). Either alone is sufficient.
- No correctness regression: 154428/154481 inertia match holds, 99.8%+ residual.
- Bit-exact contract on `tests/blocked_ldlt.rs`.

## State of the three issue targets (verified 2026-05-12)

| target | status | notes |
|---|---|---|
| Workspace pooling | partially done; W-3a removed the big `nrow*nrow` clone but per-call still allocates `perm`, `subdiag`, `d_panel` + extract `l`, `d_diag`, `contrib`, `perm_inv`, `d_subdiag` | `dev/plans/phase-2.9.2-factor-frontal-arena.md` exists but never landed |
| `SymmetricMatrix::validate()` bypass | **already done** in `factor_frontal_blocked_in_place` (line 1062–1064 documents the skip) | stale issue item — hot path doesn't validate per-front |
| `extend_add` direct writes | not done; `numeric/factorize.rs:2612-2616` still branches `parent_i >= parent_j` per cell and uses `set`/`get` | real target |

## Phase A — Internal scratch pool (subdiag + d_panel)

**Scope.** Add a caller-supplied `FactorScratch` struct holding `subdiag: Vec<f64>`
and `d_panel: Vec<f64>` — the two internal-only working buffers that never leave
`factor_frontal_blocked_in_place`. `perm` is left as a fresh alloc-and-move in this
phase (changing that touches `FrontalFactors` and is Phase C scope).

**API.**

```rust
#[derive(Default, Debug, Clone)]
pub struct FactorScratch {
    pub subdiag: Vec<f64>,
    pub d_panel: Vec<f64>,
}

impl FactorScratch {
    pub fn new() -> Self { Self::default() }
}
```

New entry point:

```rust
pub fn factor_frontal_blocked_in_place_with_scratch(
    matrix: &mut SymmetricMatrix,
    ncol: usize,
    may_delay: bool,
    params: &BunchKaufmanParams,
    scratch: &mut FactorScratch,
) -> Result<FrontalFactors, FeralError>
```

Existing `factor_frontal_blocked_in_place` becomes a thin wrapper that allocates
`FactorScratch::default()` and delegates. Every existing call site continues to
work; the hot multifrontal driver opts into the `_with_scratch` variant.

**Kernel prologue.** Replace:

```rust
let mut subdiag = vec![0.0; nrow];
let mut d_panel = vec![0.0f64; bs];
```

with:

```rust
scratch.subdiag.clear();
scratch.subdiag.resize(nrow, 0.0);
scratch.d_panel.clear();
scratch.d_panel.resize(bs, 0.0);
let subdiag = scratch.subdiag.as_mut_slice();
let d_panel = scratch.d_panel.as_mut_slice();
```

Then `subdiag` and `d_panel` are reborrowed as `&mut [f64]` throughout the kernel
(they already are slices in the call signatures inside `lblt_panel_frontal` and
`scalar_pivot_step`). The extract step's `subdiag[..nelim].to_vec()` still allocates
the returned `d_subdiag` — that's Phase C.

**Wiring.** `FactorWorkspace` (`src/numeric/factorize.rs:739`) gains a
`pub factor_scratch: FactorScratch` field. Hot-path call sites at:

- `factorize.rs:913` (the n×n dense fast path in `factor_dense_fast`)
- `factorize.rs:1696` (`factor_one_supernode`)
- `factorize.rs:1849` (`factor_one_small_leaf`)

switch from `factor_frontal_blocked_in_place(...)` to
`factor_frontal_blocked_in_place_with_scratch(..., &mut ws.factor_scratch)`.

**Parity test.** `tests/factor_scratch_parity.rs` (new):

- Pick 3 matrices: 4×4 Bunch-Kaufman fixture, 32×32 random indefinite (fixed seed),
  128×128 dense.
- For each: factor via `factor_frontal_blocked_in_place` (owning) and via
  `factor_frontal_blocked_in_place_with_scratch` (a) fresh scratch, (b) warm scratch
  from a prior unrelated call (different `nrow`).
- Assert byte-identical `l`, `d_diag`, `d_subdiag`, `perm`, `perm_inv`, `contrib`,
  `inertia`, `n_delayed`, `needs_refinement`.

**Gate.** `tests/blocked_ldlt.rs` integration tests must remain bit-exact —
that's the strongest end-to-end gate.

**Expected outcome.** 2 allocs removed per supernode (subdiag + d_panel). On
matrices with 200–500 supernodes, that's 400–1000 fewer malloc/free pairs per
factor. Whether this moves ns/sup measurably is the empirical question Phase A
answers.

## Phase B — `extend_add` direct writes

**Scope.** Replace branchy `SymmetricMatrix::set`/`get` calls in
`numeric/factorize.rs::extend_add` (lines 2595-2619) with direct slice writes
into `frontal.data` using a precomputed lower-triangle column-major index.

**Current code (line 2602-2616):**

```rust
for ci in cj..cdim {
    let parent_i = parent_row_map[contrib.row_indices[ci]];
    if parent_i == usize::MAX { continue; }
    let val = contrib.data[cj * cdim + ci];
    if val == 0.0 { continue; }
    if parent_i >= parent_j {
        frontal.set(parent_i, parent_j, frontal.get(parent_i, parent_j) + val);
    } else {
        frontal.set(parent_j, parent_i, frontal.get(parent_j, parent_i) + val);
    }
}
```

**Replacement.** `SymmetricMatrix` stores column-major lower triangle; for cell
`(i, j)` with `i >= j`, the linear index is `j * n + i`. Direct write:

```rust
for ci in cj..cdim {
    let parent_i = parent_row_map[contrib.row_indices[ci]];
    if parent_i == usize::MAX { continue; }
    let val = contrib.data[cj * cdim + ci];
    if val == 0.0 { continue; }
    let (row, col) = if parent_i >= parent_j {
        (parent_i, parent_j)
    } else {
        (parent_j, parent_i)
    };
    frontal.data[col * frontal.n + row] += val;
}
```

The `if parent_i >= parent_j` branch is preserved — that's the symmetric-storage
canonicalisation, not a SymmetricMatrix-internal branch. The savings are:
- removes 2 function-call frames (`set`/`get`) per cell
- removes the redundant `i >= j` assertion `set`/`get` perform on every call
- removes the `Index/IndexMut`-style branching inside `set`/`get`

**Parity test.** `tests/extend_add_direct_parity.rs` (new): randomized contribution
blocks vs the old `set`/`get` path on a held-out small reference implementation.
Or simpler: rely on the multifrontal integration tests in `tests/factorize.rs`
which exercise `extend_add` end-to-end on dozens of small matrices.

**Gate.** Multifrontal factorization tests (`tests/factorize.rs`) and bench corpus
inertia match (154428/154481) must hold.

**Expected outcome.** Per #13's research note, ~38 children per front contribute
dense Schur blocks. Each `set`/`get` pair costs an integer comparison + assertion
check + index calculation. With cdim up to 64+ for medium fronts, that's
~38 * (64*65/2) = 79000 redundant branches per medium front. Concrete win is
empirical.

## Phase C (deferred unless A+B miss acceptance)

Pool the return-struct allocations (`l`, `d_diag`, `d_subdiag`, `contrib`,
`perm`, `perm_inv`) by either:
1. Changing `FrontalFactors` to borrow scratch (ABI break — affects every caller)
2. Adding a `take_into_factors(&mut self, ...)` API that swaps capacity
3. Pre-sizing `FrontalFactors` with `with_capacity` from scratch hints

Defer the design choice until Phase A+B measurements show whether Phase C is
needed to hit acceptance.

## Risk

- **Bit-parity violation.** Mitigation: parity tests at every phase plus the
  pre-existing `tests/blocked_ldlt.rs` byte-equal integration tests.
- **Hot-path call sites missed.** Mitigation: `factor_frontal_blocked_in_place`
  stays as a wrapper, so any missed site continues to work (just keeps allocating).
  Grep after wiring confirms intent.
- **Bench numbers don't improve.** Honest reporting per CLAUDE.md ("When benchmark
  numbers are worse than the previous session: report this explicitly"). If
  Phase A+B miss acceptance, move to Phase C.

## Files touched (estimate)

| file | LOC delta |
|---|---|
| `src/dense/factor.rs` | +60 / -10 |
| `src/numeric/factorize.rs` | +15 / -10 |
| `tests/factor_scratch_parity.rs` (new) | +120 |
| `dev/journal/2026-05-12-07.org` | +entries |
| `dev/sessions/2026-05-12-07.md` | +checkpoint |
| `CHANGELOG.md` | +entries (Unreleased) |
