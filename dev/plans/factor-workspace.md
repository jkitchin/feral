# FactorWorkspace API plan

**Authorized by:** alloc-probe evidence in
`dev/research/sparse-tail-perf-2026-04-19.md` §9. Hypothesis confirmed.
**Date opened:** 2026-04-19 (session 04).

## Goal

Reduce per-call allocation in `factorize_multifrontal` by pooling
scratch buffers in a caller-owned `FactorWorkspace`, so IPM-style
re-factorizations reuse the same heap memory across calls.

**Measurable win target** (corpus `factor/MUMPS` geomean on
154 588 matrices, sparse path):

- baseline: 0.48
- post-workspace: ≤ 0.40

Plus for the two failing family geomeans:

- AVION2 (2682 matrices): 1.61 → ≤ 1.0
- BATCH (2054 matrices): 1.85 → ≤ 1.2

If the corpus A/B delivers less than ~15% geomean improvement,
treat the lever as exhausted and pivot to D.3/D.4.

## Non-goals (this iteration)

- Pooling `factor_frontal` internals (`a`, `subdiag`, `l`, `d_diag`,
  `d_subdiag`, `perm`, `perm_inv`, `contrib`). Four of those seven
  are retained in `FrontalFactors` and survive past the call — pooling
  them requires changing the ownership model of `FrontalFactors`, a
  larger refactor. Defer until we measure whether the
  first-pass win is sufficient.
- Pooling the `ContribBlock` storage. Same reason —
  parent consumption boundary is non-trivial to thread through
  without disturbing the postorder traversal.
- Multi-thread workspace sharing. Single-threaded only.

## Scratch sites targeted (the first-pass scope)

Measured allocation sites per supernode in the
`factorize_multifrontal` loop (not inside `factor_frontal`):

| site                         | where | per-snode | retained? |
|------------------------------|-------|-----------|-----------|
| `row_map = vec![MAX; n]`     | factorize.rs:259 | 1 | scratch |
| `build_row_indices` internals| factorize.rs:457 | ~5 + BTreeSet nodes | scratch |
| `SymmetricMatrix::zeros`     | factorize.rs:284 | 1 | scratch |
| `contrib_row_indices`        | factorize.rs:347 | 1 | retained* |

\* `contrib_row_indices` is stored on `ContribBlock` and consumed
by parent — its lifetime is bounded but not to the current snode.
Include in the workspace later.

`build_row_indices` is the worst offender: it allocates an
`own_cols` (collect), `delayed_cols` Vec, `is_fully_summed` Vec of
size n, a `BTreeSet` (each insert potentially allocates a tree
node), and a `result` Vec that grows via extend. Replacing the
BTreeSet with an `is_seen` marker + unsorted Vec + final sort
alone should drop the per-call alloc count by ~5–20 depending on
trailing-set size.

## API surface

```rust
// src/numeric/factorize.rs

/// Caller-owned scratch pool for sparse numeric factorization.
/// Safe to reuse across multiple `factorize_multifrontal` calls
/// with different matrix sizes — each field grows monotonically
/// via `resize(.., usize::MAX)` / `clear(); reserve(..)` as needed.
pub struct FactorWorkspace {
    /// Global→local row-index map, maintained in all-`usize::MAX`
    /// state outside the per-supernode critical section.
    row_map: Vec<usize>,
    /// Scratch for building the frontal's row_indices.
    delayed_cols: Vec<usize>,
    trailing: Vec<usize>,
    is_fully_summed: Vec<bool>,
    /// Scratch storage for the frontal SymmetricMatrix values.
    frontal_values: Vec<f64>,
}

impl FactorWorkspace {
    pub fn new() -> Self { /* zero-length */ }
    /// Pre-size for a given `n` to skip growth on the first call.
    pub fn with_capacity_for_n(n: usize) -> Self { ... }
}

pub fn factorize_multifrontal(
    matrix: &CscMatrix,
    symbolic: &SymbolicFactorization,
    params: &NumericParams,
) -> Result<(SparseFactors, Inertia), FeralError> { ... }

/// Workspace-reusing variant. Identical semantics; each scratch
/// allocation inside factorize_multifrontal becomes a `resize` or
/// `reserve+clear` on the matching field of `ws`.
pub fn factorize_multifrontal_with_workspace(
    matrix: &CscMatrix,
    symbolic: &SymbolicFactorization,
    params: &NumericParams,
    ws: &mut FactorWorkspace,
) -> Result<(SparseFactors, Inertia), FeralError> { ... }
```

The non-`_with_workspace` entry point stays byte-identical to the
current behavior — it creates a fresh `FactorWorkspace` internally
and calls the underlying implementation. This preserves all 118
existing tests and the 154 588-matrix corpus behavior by
construction.

`Solver` (src/numeric/solver.rs) gains a `workspace:
FactorWorkspace` field and calls the `_with_workspace` variant, so
IPM re-factorizations reuse the workspace across iterations.

## Testing strategy (must come before the implementation change)

1. **Parity tests**: 3 tests asserting that every matrix in a
   panel (AVION2_0000, BATCH_0000, LAKES_1199, VESUVIO_0000, plus
   one KKT with delayed pivots) produces byte-identical
   `SparseFactors` between the no-workspace and with-workspace
   paths. Oracle = the no-workspace path on the same input.
2. **Cross-matrix reuse test**: feed the same workspace through
   (AVION2_0000, then BATCH_0000, then VESUVIO_0000, then
   AVION2_0000 again). Assert each call's result matches the
   one-shot no-workspace result. This catches residual state
   (un-cleared `row_map`, dirty scratch).
3. **No-workspace path unchanged**: the existing test suite (118
   lib + 6 threshold_consistency + dense tests) must remain green.
4. **`cargo clippy -- -D warnings` + `cargo fmt --check`**: the
   usual gate.

Tests land in `src/numeric/factorize.rs` tests module and a new
`tests/factor_workspace_parity.rs` integration test.

## Measurement plan

### Stage 1 — micro: alloc_probe
Extend `src/bin/alloc_probe.rs` to accept a `--workspace` flag;
rerun and tabulate pre/post for the 10-matrix panel. Expected:
allocs/snode drops from 17–23 to ≤10.

### Stage 2 — micro: profile_sparse timing
Rerun `profile_sparse` with the workspace path. Expected:
AVION2_0000 99 → ≤50 µs, BATCH_0000 78 → ≤40 µs.

### Stage 3 — corpus: bench
Full 154 588-matrix run via `cargo run --release --bin bench`.
Report corpus geomean, p90, p99, and per-family geomeans for
AVION2 / BATCH / HS118. This is the acceptance gate.

## Rollout

1. Plan (this file) — commit.
2. Parity tests (red) — commit.
3. `FactorWorkspace` struct + no-workspace variant calling the
   unified impl — commit (tests green, behavior unchanged).
4. `_with_workspace` variant pooling `row_map` + frontal values —
   commit (parity tests green). Measure stage 1.
5. `build_row_indices` rewrite (BTreeSet → seen+sort, workspace
   buffers) — commit. Measure stage 1 + 2.
6. `Solver` wired to workspace — commit. Measure stage 3.
7. Session checkpoint.

## Risks

- `FactorWorkspace` API leaks into the public surface. Keep the
  no-workspace entry point as the default and mark the
  `_with_workspace` variant as the high-perf path. Future
  consolidation if the default becomes a trivial wrapper.
- Tests may be sensitive to bit-level differences if any pooled
  buffer is read before fully written. Parity test at stage 3
  catches this — run against the full corpus, not just 10
  matrices.
- Reused scratch may hide real bugs (e.g., a missed clear leaves
  stale data that happens to produce correct results on the
  current matrix). The cross-matrix reuse test at stage 2 is the
  mitigation.
