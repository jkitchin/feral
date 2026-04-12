# Implementation plan: MC64 matching-based scaling (Phase 2.2.1)

## Goal

Implement pure-Rust MC64-style matching-based global scaling in
feral's sparse multifrontal pipeline, matching the behavior of
canonical Fortran MUMPS 5.8.2 (default for `SYM=2`) and canonical
SPRAL/SSIDS (`options%scaling=1`). Close the residual gap exposed
by the Phase 2.1.2 sanity check.

**Design document:** `dev/research/mc64-scaling.md` (committed
`d7ec681`). This plan assumes the reader has read the research
note; it does not re-derive the algorithm.

**Target matrices:** all KKT matrices with `n > 500` already in the
corpus. Success criterion: residuals within 2–3 orders of magnitude
of canonical MUMPS and SSIDS on the 7-matrix sanity panel
(`CHWIRUT1`, `HAHN1`, `GAUSS2`, `CRESC100`, `MUONSINE`, `VESUVIO`,
`CRESC132`). Aspirational: residuals within 1 order of magnitude.

## Scope

**In scope:**

- A new `src/scaling/` module with a pure-Rust Hungarian algorithm
  kernel and an MC64 wrapper.
- Integration into `symbolic_factorize` so scaling is computed once
  per analysis.
- Integration into `factorize_multifrontal` so scaling is applied
  during frontal assembly (not as a separate pre-transform).
- Integration into `solve_sparse` and `solve_sparse_refined` so the
  RHS is pre-scaled and the solution is post-scaled.
- Unit tests on hand-computable 3×3 matrices.
- Regression tests against canonical oracles on the sanity panel.
- Full corpus re-run.

**Out of scope (deferred to later Phase 2 tasks):**

- Symmetric 2×2 pivot pair detection from the matching cycles
  (MUMPS's `DMUMPS_SYM_MWM`). This is a pivoting optimization, not a
  correctness fix. Phase 2.3.2 work.
- The deferred `count_2x2_inertia` trace-vs-`a00` fix
  (`dev/tried-and-rejected.md`). That is Phase 2.2.2 and is
  deliberately sequenced after MC64 scaling so it can be
  re-evaluated with the new scaling in place.
- Performance optimization of the Hungarian kernel beyond a
  working heuristic initialization. Liu-style column counts or
  micro-optimization of the shortest-path inner loop are Phase
  2.5 work.
- Auction-based alternative scaling (SSIDS's `options%scaling=2`).
  We implement MC64 because that is what MUMPS uses by default and
  what the research note justifies.
- `Ruiz` equilibration as a fallback. Research note §10 rejects
  this; identity scaling is the only fallback.

## Dependencies

- `dev/research/mc64-scaling.md` — the design document.
- `ref/mumps/src/dana_mtrans.F` — MUMPS Fortran MC64, for
  algorithmic cross-checks. Particularly `DMUMPS_MTRANSW` at
  lines 791–1081 (strategy 5 with dual variables).
- `ref/spral/src/scaling.f90` — SSIDS's Rust-friendly reference
  implementation of the Hungarian kernel. Particularly
  `hungarian_match` at lines 938–1171 and `hungarian_wrapper` at
  597–801 (input preprocessing and output unwinding). This is the
  closest reference in spirit to what the Rust port should produce.
- The existing feral data flow from `src/symbolic/mod.rs`,
  `src/numeric/factorize.rs`, `src/numeric/solve.rs`.

## Test-first order

Per CLAUDE.md, tests come before implementation, and the oracle
must come from an external source rather than being written in
the same session as the implementation.

### Test oracles available before implementation starts

1. **Hand-computed 3×3 matrices.** Small enough that the matching
   and the symmetric scaling can be derived analytically from first
   principles. These are the unit tests. The oracles are hand
   derivations, not reference code.
2. **Duff & Koster 2001 worked example.** citet:duff2001mc64 §4
   works through a specific small matrix to illustrate the algorithm.
   If that example is reproducible, it is a second external oracle
   independent of hand derivation.
3. **SPRAL `hungarian_match` output, captured as a regression
   snapshot.** For any given input matrix, run SPRAL's Fortran
   implementation and capture the output duals into a JSON file.
   These become test fixtures that the Rust port must match. This
   is not strictly an "external" oracle (since SPRAL is one of
   the reference implementations the Rust port is modeled on) but
   it is independent of the same-session implementation.
4. **ACOPP30_0000 residual, measured on the existing triage
   example.** The current feral residual of `3.15e-2` is the
   pre-fix baseline. Canonical MUMPS produces `5.0e-14` on the
   same matrix. Any MC64 implementation that does not bring
   feral's residual within 4 orders of magnitude of MUMPS's on
   this matrix has a bug that blocks Phase 2.2.1 completion.
5. **7-matrix sanity panel residuals.** Captured in Phase 2.1.2
   and recorded in `dev/decisions.md` (2026-04-12 entry).

### Tests written before implementation

- `tests/mc64_scaling.rs` (new) with the following cases, all
  failing on commit of the test file (since the code doesn't exist
  yet):
  - `hungarian_match_3x3_diagonal`: a diagonal matrix should match
    to identity and produce zero duals.
  - `hungarian_match_3x3_permutation`: a pure permutation matrix
    (nonzero on an off-diagonal pattern) should produce the
    corresponding matching.
  - `hungarian_match_3x3_hand_computed`: a 3×3 with a specific
    non-trivial structure whose matching is derivable by hand.
    Assert on exact matching and exact dual values (to a tight
    floating-point tolerance).
  - `mc64_symmetric_scaling_diagonal_unit`: for the identity
    matrix, the scaling should be identity (`s_i = 1` for all i).
  - `mc64_symmetric_scaling_wide_dynamic_range`: a matrix with
    entries spanning 8 orders of magnitude should produce a
    scaling that brings the max-magnitude entry to exactly 1.
  - `mc64_symmetric_scaling_singular_warns`: a structurally
    singular matrix should return a `ScalingInfo::PartialSingular`
    warning rather than panicking or returning zeros.
- `tests/mc64_regression.rs` (new, `#[ignore]` until data is
  available in CI):
  - `acopp30_0000_residual_under_1e_10_after_scaling`: load
    `data/matrices/kkt/ACOPP30/ACOPP30_0000.mtx`, run
    `solve_sparse_refined`, assert the residual is below `1e-10`.
    Pre-fix baseline: `3.15e-2`. Canonical MUMPS: `5.0e-14`.
  - `cresc132_0000_residual_under_1e_6_after_scaling`: load
    `data/matrices/kkt/CRESC132/CRESC132_0000.mtx`, run sparse
    solve, assert residual below `1e-6`. Pre-fix baseline:
    `2.39e+08`.

### Gate point

Before implementing the Hungarian kernel, write all of the above
tests, confirm they fail to compile (because the module does not
exist) and then fail to pass (because the module is a stub
returning placeholder values). Only then start implementing.

## Module layout

```
src/
├── scaling/
│   ├── mod.rs              (public types, dispatch)
│   ├── hungarian.rs        (Hungarian matching kernel)
│   └── mc64.rs             (MC64 wrapper: input transform, call
│                            Hungarian, output unwind, symmetric
│                            averaging)
├── symbolic/
│   └── mod.rs              (modified: compute scaling, store it)
├── numeric/
│   ├── factorize.rs        (modified: apply scaling during assembly)
│   └── solve.rs            (modified: pre-scale RHS, post-scale x)
└── lib.rs                  (modified: `pub mod scaling;` and
                              public re-exports)
```

### `src/scaling/mod.rs`

```rust
//! Global scaling for sparse symmetric indefinite matrices.
//!
//! Implements MC64 matching-based scaling (Duff & Koster 2001,
//! Duff & Pralet 2005) using a pure-Rust Hungarian algorithm.
//! See `dev/research/mc64-scaling.md` for the design document.

use crate::error::FeralError;
use crate::sparse::csc::CscMatrix;

mod hungarian;
mod mc64;

/// User-facing scaling strategy selector.
#[derive(Debug, Clone, PartialEq)]
pub enum ScalingStrategy {
    /// MC64-style symmetric matching-based scaling. Default.
    Mc64Symmetric,
    /// Identity scaling (no-op). Use for regression testing and
    /// for inputs where matching is not appropriate.
    Identity,
    /// User-supplied pre-computed scaling vector in user-order
    /// indexing. Length must equal `n`.
    External(Vec<f64>),
}

impl Default for ScalingStrategy {
    fn default() -> Self {
        ScalingStrategy::Mc64Symmetric
    }
}

/// Diagnostic information about how the scaling was computed.
#[derive(Debug, Clone, PartialEq)]
pub enum ScalingInfo {
    /// MC64 matching ran to completion.
    Applied,
    /// MC64 matching found a partial solution; unmatched rows and
    /// columns fall back to identity scaling. `n_unmatched` is the
    /// number of variables that could not be matched.
    PartialSingular { n_unmatched: usize },
    /// No scaling applied (e.g., user requested Identity).
    NotApplied,
}

/// Compute the symmetric scaling vector for a sparse symmetric
/// matrix stored in CSC with only the lower triangle, following
/// `strategy`.
///
/// Returns a vector of length `n` in user-order indexing such that
/// applying `D = diag(scaling)` symmetrically as `D · A · D`
/// produces a scaled matrix with off-diagonals bounded by 1 in
/// absolute value and largest magnitudes on the diagonal.
pub fn compute_scaling(
    matrix: &CscMatrix,
    strategy: &ScalingStrategy,
) -> Result<(Vec<f64>, ScalingInfo), FeralError> {
    match strategy {
        ScalingStrategy::Identity => Ok((vec![1.0; matrix.n], ScalingInfo::NotApplied)),
        ScalingStrategy::External(s) => {
            if s.len() != matrix.n {
                return Err(FeralError::InvalidInput(format!(
                    "external scaling has length {} but matrix has n={}",
                    s.len(),
                    matrix.n,
                )));
            }
            Ok((s.clone(), ScalingInfo::NotApplied))
        }
        ScalingStrategy::Mc64Symmetric => mc64::compute_symmetric(matrix),
    }
}
```

### `src/scaling/hungarian.rs`

Core Hungarian matching kernel. The skeleton follows
`ref/spral/src/scaling.f90::hungarian_match` (lines 938–1171) in
spirit. Signature:

```rust
/// A sparse non-negative cost graph for the Hungarian algorithm.
///
/// The cost matrix is stored in CSC format on a full symmetric
/// pattern. Costs must be non-negative (the MC64 wrapper ensures
/// this via per-column normalization). Only finite costs are
/// valid; the graph must not contain NaN or ±∞ entries.
pub(crate) struct CostGraph {
    pub n: usize,
    pub col_ptr: Vec<usize>,
    pub row_idx: Vec<usize>,
    pub cost: Vec<f64>,
}

/// Result of a Hungarian matching run.
pub(crate) struct Matching {
    /// `perm[j]` is the row matched to column `j`. `usize::MAX`
    /// sentinel for unmatched columns.
    pub perm: Vec<usize>,
    /// Dual variable for row `i` (length `n`).
    pub u: Vec<f64>,
    /// Dual variable for column `j` (length `n`).
    pub v: Vec<f64>,
    /// Number of columns successfully matched (≤ n). Less than n
    /// means the matrix is structurally singular on this cost
    /// graph.
    pub n_matched: usize,
}

/// Solve the minimum-cost perfect bipartite matching problem via
/// the shortest-augmenting-path Hungarian algorithm. At
/// termination the dual variables satisfy `u[i] + v[j] ≤ cost[i,j]`
/// for every edge, with equality on matched edges.
///
/// Algorithm reference: citet:duff2001mc64 §4.
/// Source reference: `ref/spral/src/scaling.f90:938–1171`.
pub(crate) fn hungarian_match(cost: &CostGraph) -> Matching {
    // 1. Greedy initialization (see `hungarian_init_heurisitic`).
    // 2. Main loop over unmatched columns:
    //    a. Build shortest-path tree from column j.
    //    b. Follow reduced costs (cost[i,j] - u[i] - v[j]).
    //    c. Terminate when an unmatched row is reached.
    //    d. Update duals to preserve complementary slackness.
    //    e. Flip matching along augmenting path.
    // 3. Return `Matching { perm, u, v, n_matched }`.
    todo!()
}
```

### `src/scaling/mc64.rs`

MC64 wrapper that handles input preprocessing and output
unwinding, then returns a symmetric scaling vector.

```rust
use super::hungarian::{hungarian_match, CostGraph};
use super::ScalingInfo;
use crate::error::FeralError;
use crate::sparse::csc::CscMatrix;

/// Compute the MC64 symmetric scaling for a sparse symmetric
/// matrix (lower triangle only in the input CSC).
///
/// Steps (mirrors `ref/spral/src/scaling.f90::hungarian_wrapper`):
///
///  1. Expand the pattern to a full symmetric graph.
///  2. Drop explicit zero entries (they would become -∞ under log).
///  3. Compute `c[k] = log |a[k]|` for each remaining nonzero.
///  4. For each column j, compute `C[j] = max_k c[k]` and replace
///     each `c[k]` by `C[j] - c[k]`. Now all costs are ≥ 0 and the
///     minimum in each column is exactly 0.
///  5. Call `hungarian_match(cost_graph)`.
///  6. On success: the returned duals `u[i]`, `v[j]` satisfy
///     `u[i] + v[j] ≤ C[j] - log|a[i,j]|`. Unwinding the `C[j]`
///     normalization gives
///     `u[i] + (v[j] - C[j]) ≤ -log|a[i,j]|`, or equivalently
///     `-(u[i] + v[j] - C[j]) ≥ log|a[i,j]|`. The unscaled row
///     and column duals for scaling purposes are therefore
///     `u'[i] = -u[i]` and `v'[j] = C[j] - v[j]`. (Verify this
///     sign convention against `scaling.f90:169` during
///     implementation — SPRAL's exact sign handling is the
///     authoritative reference.)
///  7. Symmetric averaging: `s[i] = exp((u'[i] + v'[i]) / 2)`.
///  8. Safety guards: clamp any dual > log(f64::MAX) ~ 709 to
///     finite values; rewrite any s[i] == 0 to s[i] = 1.
///  9. If the matching is partial (n_matched < n), set s[i] = 1
///     for every unmatched row/column and return
///     `ScalingInfo::PartialSingular`.
pub(crate) fn compute_symmetric(
    matrix: &CscMatrix,
) -> Result<(Vec<f64>, ScalingInfo), FeralError> {
    todo!()
}
```

### `src/symbolic/mod.rs` — integration

Add two new fields to `SymbolicFactorization`:

```rust
pub struct SymbolicFactorization {
    // ... existing fields ...

    /// Symmetric scaling vector (exp of the symmetric dual
    /// average). Stored in pivot-order indexing so it can be
    /// applied directly from the assembly loop without an extra
    /// indirection through `perm`.
    pub scaling: Vec<f64>,

    /// Diagnostic info about how `scaling` was produced.
    pub scaling_info: crate::scaling::ScalingInfo,
}
```

`symbolic_factorize` gains a new parameter for the scaling
strategy (or `SupernodeParams` is extended with a `scaling` field).
The pipeline becomes:

```rust
pub fn symbolic_factorize(
    matrix: &CscMatrix,
    snode_params: &SupernodeParams,
) -> Result<SymbolicFactorization, FeralError> {
    let n = matrix.n;

    // Phase 2.2.1: Compute global scaling BEFORE ordering.
    // Scaling is a congruence transformation and does not affect
    // the sparsity pattern, so it is independent of ordering and
    // can run in any order. We run it first for clarity.
    let strategy = snode_params.scaling.clone();
    let (scaling_user_order, scaling_info) =
        crate::scaling::compute_scaling(matrix, &strategy)?;

    // ... existing pipeline: AMD → postorder → permute → etree
    //     → column counts → supernodes ...

    // Permute scaling into pivot-order indexing so that
    // factorize_multifrontal can look it up directly by pivot
    // position without a permutation indirection per entry.
    let mut scaling = vec![0.0; n];
    for (new_idx, &old_idx) in perm.iter().enumerate() {
        scaling[new_idx] = scaling_user_order[old_idx];
    }

    Ok(SymbolicFactorization {
        // ... existing fields ...
        scaling,
        scaling_info,
    })
}
```

### `src/numeric/factorize.rs` — apply during assembly

The frontal assembly loop gains a scaling multiplication. Current
code (simplified from the existing implementation):

```rust
// existing, without scaling:
for k in csc.col_ptr[col]..csc.col_ptr[col + 1] {
    let row = csc.row_idx[k];
    let val = csc.values[k];
    frontal_scatter(local_i, local_j, val);
}
```

becomes:

```rust
// with scaling (mirrors MUMPS's dfac_dist_arrowheads_omp.F:1023
// and SSIDS's assemble.hxx:64):
for k in csc.col_ptr[col]..csc.col_ptr[col + 1] {
    let row = csc.row_idx[k];
    let val = csc.values[k] * symbolic.scaling[row] * symbolic.scaling[col];
    frontal_scatter(local_i, local_j, val);
}
```

One subtlety: the row and column indices `row` and `col` at this
point in the code are in pivot-order indexing (they came from the
permuted CSC `permuted.row_idx` / `permuted.col_ptr`), and
`symbolic.scaling` is also in pivot-order indexing, so the lookup
is direct. Verify this alignment during implementation by adding a
debug assertion that `scaling.len() == symbolic.n`.

### `src/numeric/solve.rs` — pre and post scaling

Existing `solve_sparse` and `solve_sparse_refined` do forward and
backward sweeps with a permutation step. The pre- and post-scaling
go right at the permutation boundaries:

```rust
// In solve_sparse (and analogously in solve_sparse_refined):

// Step 1: Permute RHS to pivot order and pre-scale in one pass.
let mut y = vec![0.0; n];
for (new_idx, &old_idx) in factors.perm.iter().enumerate() {
    y[new_idx] = rhs[old_idx] * factors.scaling[new_idx];
    //                          ^^^^^^^^^^^^^^^^^^^^^^^^^
    //                          NEW: pre-scale
}

// Step 2: L-solve, D-solve, L^T-solve (unchanged from existing).

// Step 3: Un-permute and post-scale in one pass.
let mut x = vec![0.0; n];
for (new_idx, &old_idx) in factors.perm.iter().enumerate() {
    x[old_idx] = y[new_idx] * factors.scaling[new_idx];
    //                        ^^^^^^^^^^^^^^^^^^^^^^^
    //                        NEW: post-scale (same vector!)
}
```

Note: **same scaling vector on both ends**, not its inverse. See
research note §"Application at solve time" for the derivation.

The `SparseFactors` struct gains a `scaling: Vec<f64>` field so
that solve can find it; or we pass a reference to
`SymbolicFactorization` into solve. The latter is cleaner but
requires plumbing a lifetime; the former is simpler. Recommend the
former: clone or move the scaling vector into `SparseFactors` at
the end of `factorize_multifrontal`.

For `solve_sparse_refined`, the residual computation `r = b - A x`
must use the **original unscaled** `CscMatrix` passed in by the
caller. This is already the case in the current implementation —
`solve_sparse_refined(matrix, factors, rhs)` takes the matrix as a
parameter and the residual loop uses `matrix.symv(...)`. No change
needed beyond verifying this in a unit test.

## Implementation steps (ordered)

### Step 1 — Create the module skeleton (~30 min)

Files: `src/scaling/mod.rs`, `src/scaling/hungarian.rs`,
`src/scaling/mc64.rs`, and `src/lib.rs` update.

- Stub all three files with the types and function signatures from
  the module layout above.
- `hungarian_match` returns a trivial identity matching
  (`Matching { perm: (0..n).collect(), u: vec![0.0; n], v: vec![0.0; n], n_matched: n }`).
- `compute_symmetric` returns identity scaling.
- Module compiles cleanly, all existing tests still pass.

### Step 2 — Write the failing tests (~1 hour)

Files: `tests/mc64_scaling.rs`, `tests/mc64_regression.rs`.

- Unit tests on 3×3 matrices per §"Tests written before
  implementation" above.
- Regression tests on ACOPP30_0000 and CRESC132_0000, gated
  `#[ignore]` if the data files are not present.
- Confirm tests fail (most will, because the stubs return
  identity).

### Step 3 — Implement Hungarian kernel (~6–10 hours)

Files: `src/scaling/hungarian.rs`.

Recommend reading SPRAL's `hungarian_match` (230 Fortran lines)
alongside `hungarian_init_heurisitic` (about 120 lines) before
typing. The Rust port will be somewhat longer than the Fortran
because of borrow-checker considerations around the heap
operations and the dual update.

Sub-steps:

1. **Greedy initialization.** Port
   `hungarian_init_heurisitic`. For each column, scan for the
   smallest-cost edge whose row is unmatched; claim it. This
   typically matches 40–80% of rows in one pass and makes the
   subsequent shortest-path iteration much faster.
2. **Main loop.** For each unmatched column, run a
   shortest-augmenting-path search. The search uses a binary
   min-heap keyed on the reduced cost. Fortran uses a hand-rolled
   heap in the `q` array; Rust should use `std::collections::
   BinaryHeap<Reverse<(f64, usize)>>` or a custom heap with
   decrease-key (the Fortran version's heap supports decrease-key
   via the `l` inverse-index array).
3. **Dual update.** After finding the shortest augmenting path,
   update the dual variables along the path so that complementary
   slackness is preserved. This is the tricky part — the update
   depends on the search tree state, not just the path.
4. **Matching flip.** Flip the matching along the augmenting
   path.
5. **Termination condition.** Continue until every column has
   been considered. If the matching is partial at the end (some
   column has no augmenting path), return the partial matching
   with `n_matched < n`.

Test against the unit tests in `tests/mc64_scaling.rs` as
implementation progresses. Do not move on to step 4 until all
3×3 unit tests pass.

### Step 4 — Implement MC64 wrapper (~2 hours)

File: `src/scaling/mc64.rs`.

1. **Pattern expansion.** Given a lower-triangle CSC, build a
   full symmetric CSC pattern. Existing feral code has
   `CscMatrix::symmetric_pattern()` in `src/sparse/csc.rs`; reuse
   it.
2. **Log transform and column-max normalization.** For each
   entry, `c[k] = log |a[k]|`. For each column, compute
   `C[j] = max_k c[k]` and replace each entry by `C[j] - c[k]`.
   Drop any entry with `a[k] == 0.0` before the log (would be
   `-∞`).
3. **Call `hungarian_match`** on the normalized cost graph.
4. **Unwind the normalization.** Recover the row and column
   duals with respect to the original (unnormalized) cost
   matrix. Verify the sign convention against
   `ref/spral/src/scaling.f90:169` and
   `hungarian_wrapper` output handling.
5. **Safety guards.**
   - Clamp any dual > `LOG_HUGE = 709.0` to finite values.
   - If `s[i] = 0` exactly, rewrite to `s[i] = 1`.
   - For unmatched rows/columns (partial matching), set
     `s[i] = 1` and return `ScalingInfo::PartialSingular`.
6. **Symmetric average.** `s[i] = exp((u'[i] + v'[i]) / 2)`.

Test against the unit tests in `tests/mc64_scaling.rs`. All
`mc64_symmetric_scaling_*` tests should now pass.

### Step 5 — Integrate into `symbolic_factorize` (~1 hour)

File: `src/symbolic/mod.rs`.

- Add the `scaling` and `scaling_info` fields to
  `SymbolicFactorization`.
- Add a `scaling: ScalingStrategy` field to `SupernodeParams` with
  `Mc64Symmetric` as the default.
- Call `compute_scaling` at the top of `symbolic_factorize`
  before the ordering pipeline.
- Permute the scaling vector into pivot-order at the end.
- All existing tests in `tests/sparse_postorder.rs` should still
  pass because the scaling default behavior is the new canonical
  one and the tests do not assert on `scaling_info`.

### Step 6 — Integrate into `factorize_multifrontal` (~1 hour)

File: `src/numeric/factorize.rs`.

- Thread `symbolic.scaling` into the assembly loop.
- Multiply `val * scaling[row] * scaling[col]` when scattering
  each entry into the frontal.
- Store `scaling` in `SparseFactors` (clone from `symbolic`) so
  the solve can find it.

### Step 7 — Integrate into `solve_sparse` and `solve_sparse_refined` (~1 hour)

File: `src/numeric/solve.rs`.

- Add pre-scale step at the RHS permutation boundary.
- Add post-scale step at the solution un-permutation boundary.
- Verify that `solve_sparse_refined`'s residual computation still
  uses the unscaled `matrix` argument (it does, by construction).

### Step 8 — Validation sweep (~2–4 hours)

1. **Unit tests.** `cargo test --test mc64_scaling` — all 6 unit
   tests pass.
2. **Regression tests.** `cargo test --test mc64_regression
   --ignored` — ACOPP30 residual below 1e-10, CRESC132 residual
   below 1e-6.
3. **Sanity panel.** Re-run
   `cargo run --release --example triage_large_cresc132`. All 7
   matrices should have residuals within 2–3 orders of magnitude
   of their MUMPS/SSIDS oracles. Capture the output for the
   commit message.
4. **Full bench.**
   `FERAL_EMIT_SIDECARS=1 cargo run --release --bin bench`. Record
   aggregate residual pass count for comparison with the
   pre-MC64 baseline.
5. **Consensus re-run.** `python3 external_benchmarks/consensus/
   compute_consensus.py data/matrices/kkt`. Expected: the 26
   Definitive feral failures from commit `199bbe9` drop
   substantially.
6. **Inertia check.** All 121 existing Rust tests in
   `tests/` still pass. If any inertia assertion fails, investigate
   before committing — scaling should not change inertia on the
   matrices the test suite covers, but interactions with
   `ForceAccept` could surface the Phase 2.2.2 2×2 trace bug.

### Step 9 — Documentation (~1 hour)

- Update `README.md` Status section: remove the "sparse path
  produces wrong residuals at n > 500" caveat if the sanity panel
  and consensus re-run confirm the fix.
- Update `CHANGELOG.md` under `[Unreleased]`:
  - Add an entry under `### Added` for "MC64 matching-based
    global scaling via a pure-Rust Hungarian algorithm
    implementation in `src/scaling/`".
  - Remove or update the "Known issues" entry about the n>500
    correctness gap.
- Append to `dev/decisions.md` a new entry documenting the
  Phase 2.2.1 completion with the before/after residual numbers.
- Update `dev/plans/phase-2-planning.md` §2.2.1 to mark the task
  complete with a link to the commit.

### Step 10 — Commit

Ideally as a small number of atomic commits:

1. "Phase 2.2.1: scaffolding for MC64 scaling module"
   (skeleton + failing tests)
2. "Phase 2.2.1: Hungarian matching kernel"
   (hungarian.rs with unit tests passing)
3. "Phase 2.2.1: MC64 wrapper and input/output plumbing"
   (mc64.rs with unit tests passing)
4. "Phase 2.2.1: integrate MC64 scaling into symbolic and numeric
   pipelines" (the main wiring change)
5. "Phase 2.2.1: validation sweep and documentation update"
   (README, CHANGELOG, decisions.md, session file)

Per CLAUDE.md, each commit has a body explaining *why*, not just
*what*. Each commit has the expected test coverage running clean
before the commit is made.

## Acceptance criteria

The implementation is complete and Phase 2.2.1 can be closed when
all of the following are true:

1. `cargo test --all-targets` passes (all 121 existing tests plus
   the new `tests/mc64_scaling.rs` unit tests).
2. `cargo clippy --all-targets -- -D warnings` clean.
3. `pre-commit run --all-files` clean.
4. `tests/mc64_regression.rs --ignored` passes on ACOPP30_0000 and
   CRESC132_0000 (both present in the data dir).
5. The 7-matrix sanity panel `triage_large_cresc132` reports
   residuals within 3 orders of magnitude of the MUMPS oracle on
   every matrix in the panel.
6. The full corpus consensus re-run reports a meaningful drop in
   the Definitive feral failures list (from 26 to ≤ 10 is a
   concrete target, ≤ 5 is aspirational).
7. `README.md`, `CHANGELOG.md`, and `dev/decisions.md` are
   updated to reflect the closure of the `n > 500` residual gap.

## Rollback plan

If the implementation reveals a fundamental issue that cannot be
addressed in this session:

- The changes are contained to `src/scaling/`, a handful of new
  fields in `SymbolicFactorization` and `SparseFactors`, and
  assembly-time and solve-time multiplications in
  `factorize_multifrontal` / `solve_sparse`.
- `git revert` on the relevant commits takes feral back to the
  pre-scaling state cleanly. No data files are deleted or
  modified.
- If only part of the work lands (e.g., Hungarian kernel but not
  integration), the uncommitted files can be saved as a WIP
  branch and the implementation resumed in a later session. The
  research note `dev/research/mc64-scaling.md` is the design
  document and does not depend on the code; it can stay
  committed.

## Risks (inherited from research note)

- **R1: Hungarian runtime.** `O(n² log n)` is acceptable at
  `n ≤ 5000` but may become a bottleneck at `n ≥ 10⁴`. Mitigation:
  measure on CRESC132 during validation; if matching exceeds 10%
  of total factorization time, optimize the heap and shortest-
  path inner loop before proceeding. Optimization work is Phase
  2.5, not blocking 2.2.1.
- **R2: Sign convention mismatch with SPRAL.** The unwinding of
  the column-max normalization in the MC64 wrapper is the place
  most likely to have a subtle off-by-sign error. Mitigation:
  cross-check with SPRAL's `hungarian_wrapper` output handling
  line-by-line before writing the Rust version.
- **R3: ±1 inertia errors may not go away.** The sanity panel's
  inertia errors are a separate bug (the deferred 2×2 trace fix)
  and may remain visible after scaling lands. This is expected
  and is explicitly Phase 2.2.2 work. It should **not** block
  Phase 2.2.1 closure.
- **R4: Sparse-solve rounding accumulation at scale.** It is
  possible that scaling closes most of the residual gap but not
  all — the per-supernode `Vec` allocations in `solve_sparse`
  may themselves accumulate rounding at large `n`. If the sanity
  panel is close but not within 3 orders, investigate the solve
  path before declaring a second bug. Mitigation: the research
  note's test plan includes a "round-trip test" on a small SPD
  matrix that isolates this concern.
- **R5: Over-scaling.** If the Hungarian matching produces
  extreme dual variables on a badly conditioned matrix, the
  resulting `s_i = exp(u_i)` can be `f64::MAX`-scale even after
  the safety clamp. Mitigation: the safety clamp sets such rows
  to `s_i = 1`, which is identity scaling for that row — weaker
  than MC64's ideal output but never wrong.

## Estimated effort (recap from research note)

| Step | Hours |
|------|------:|
| 1. Module skeleton | 0.5 |
| 2. Failing tests | 1.0 |
| 3. Hungarian kernel | 6–10 |
| 4. MC64 wrapper | 2.0 |
| 5. Symbolic integration | 1.0 |
| 6. Numeric integration | 1.0 |
| 7. Solve integration | 1.0 |
| 8. Validation | 2–4 |
| 9. Documentation | 1.0 |
| 10. Commits | 0.5 |
| **Total realistic** | **16–22** |
| **Worst case with debugging** | **25–30** |

Probably 3–4 focused sessions.

## What happens next (not part of this plan)

After Phase 2.2.1 is complete and merged, the next Phase 2 tasks
are:

- **Phase 2.2.2** — re-evaluate the deferred `count_2x2_inertia`
  trace-vs-`a00` fix with MC64 scaling in place. This is quick
  (1–2 hours) because the code change is already drafted in
  `dev/tried-and-rejected.md`.
- **Phase 2.2.3** — triage the 88 sparse-only failures from the
  dense ∩ sparse cross-comparison, now with scaled numerics.
- **Phase 2.2.4** — re-run full consensus and publish the delta
  against the pre-Phase-2 baseline.

Then Phase 2.3 (pivoting improvements) begins.
