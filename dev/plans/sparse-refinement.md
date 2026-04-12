# Plan: Sparse Iterative Refinement (`solve_sparse_refined`)

## Goal

Implement `solve_sparse_refined` to mirror the dense
`solve_refined` and switch the bench harness to use the refined
solve on both paths. This brings feral into compliance with
FERAL-PROJECT-SPEC.md §1709 ("Use `solve_refined()` for all solves
in Phase 1b") and is the next step toward Phase 1b exit after the
postorder fix.

Research: dev/research/sparse-multifrontal.md §"Phase 1b solve
convention" already specifies refinement as the default. No new
research note needed.

## Scope

In scope:
- Add `pub fn solve_sparse_refined(matrix, factors, rhs)` to
  `src/numeric/solve.rs` mirroring `src/dense/solve.rs::solve_refined`.
- Re-export from `src/lib.rs`.
- Update `src/bin/bench.rs`:
  - Dense path uses `solve_refined(&dense_matrix, &factors, &rhs)`
    instead of `solve(&factors, &rhs)`.
  - Sparse path uses `solve_sparse_refined(&csc, &sp_factors, &rhs)`
    instead of `solve_sparse(&sp_factors, &rhs)`.
- Add a unit test for `solve_sparse_refined` that verifies it
  improves accuracy on a known ill-conditioned KKT.

Out of scope:
- The Phase 1b exit validation document (next step).
- Refactoring the existing `solve` / `solve_sparse` interfaces.
- Refinement step-count tuning (use the same `max_steps = 3` and
  convergence criterion as dense).

## Algorithm (mirrors `src/dense/solve.rs::solve_refined`)

```rust
pub fn solve_sparse_refined(
    matrix: &CscMatrix,
    factors: &SparseFactors,
    rhs: &[f64],
) -> Result<Vec<f64>, FeralError> {
    let n = factors.n;
    if rhs.len() != n {
        return Err(FeralError::DimensionMismatch { expected: n, got: rhs.len() });
    }

    let mut x = solve_sparse(factors, rhs)?;

    let max_steps = 3;
    let n_sqrt = (n as f64).sqrt();
    let threshold = f64::EPSILON * n_sqrt;

    for _ in 0..max_steps {
        // r = b - A·x
        let mut ax = vec![0.0; n];
        matrix.symv(&x, &mut ax);
        let mut r = vec![0.0; n];
        for i in 0..n { r[i] = rhs[i] - ax[i]; }

        // δx = A⁻¹·r
        let dx = solve_sparse(factors, &r)?;

        // ||δx|| / ||x|| < threshold ?
        let dx_norm = norm2(&dx);
        let x_norm  = norm2(&x);

        for i in 0..n { x[i] += dx[i]; }

        if x_norm > 0.0 {
            if dx_norm / x_norm < threshold { break; }
        } else if dx_norm < threshold {
            break;
        }
    }

    Ok(x)
}

fn norm2(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}
```

This is *exactly* the algorithm from `src/dense/solve.rs::solve_refined`,
substituting `solve_sparse` for `solve` and `CscMatrix::symv` for
`SymmetricMatrix::symv`. The convergence criterion, max-step count,
and threshold are identical so the dense and sparse paths produce
matched-precision answers on the same inputs.

## Test-First

### Test 1: Refinement matches dense on a small KKT

```rust
#[test]
fn solve_sparse_refined_matches_dense_refined_small_kkt() {
    // Same matrix as bordered_kkt_4x4 from sparse_postorder.rs
    // Expected: solve_sparse_refined gives the same answer as
    // dense solve_refined to machine precision.
}
```

### Test 2: Refinement reduces residual on ill-conditioned input

```rust
#[test]
fn solve_sparse_refined_improves_accuracy() {
    // Use a moderately ill-conditioned 6x6 KKT (same as
    // two_constraint_bordered_kkt with smaller diagonals).
    // Compute residual from solve_sparse, then from solve_sparse_refined,
    // assert refined < initial * 100  (i.e., at least one order of
    // magnitude improvement, looser bound to avoid flakiness).
}
```

### Test 3: Refinement on a well-conditioned matrix is a no-op

```rust
#[test]
fn solve_sparse_refined_well_conditioned_no_change() {
    // SPD diagonal matrix → refinement should converge in 0 steps.
    // Just verify the result is correct and matches solve_sparse.
}
```

## Implementation Steps

1. Read `src/dense/solve.rs::solve_refined` (already done).
2. Read `src/numeric/solve.rs::solve_sparse` to confirm signature
   and that it returns `Result<Vec<f64>, FeralError>`.
3. Confirm `CscMatrix::symv` exists with signature
   `(&self, x: &[f64], y: &mut [f64])`.
4. Write the three tests in `tests/sparse_postorder.rs` (or a new
   `tests/sparse_refined.rs` — TBD by file size).
5. Confirm tests fail (Test 1 and Test 2) on `main` because the
   function doesn't exist yet (compile error counts as fail).
6. Implement `solve_sparse_refined` + private `norm2` helper in
   `src/numeric/solve.rs`.
7. Re-export from `src/lib.rs`.
8. Tests pass.
9. Update `src/bin/bench.rs`:
   - Add `solve_refined` and `solve_sparse_refined` to imports.
   - Replace dense `solve(...)` call with
     `solve_refined(&entry.matrix, &factors, &rhs)`.
   - Replace sparse `solve_sparse(...)` call with
     `solve_sparse_refined(&entry.csc, &sp_factors, &rhs)`.
10. `cargo test`, `cargo clippy -- -D warnings`.
11. `cargo run --release --bin bench` — record numbers.
12. Commit.

## Acceptance Criteria

1. `solve_sparse_refined` exists and is exported.
2. All three new tests pass.
3. `cargo test` (full suite) passes — no regressions.
4. `cargo clippy -- -D warnings` clean.
5. Bench numbers improve or stay flat:
   - Dense inertia: ≥ 99.2% (current pre-refinement)
   - Dense residual: ≥ 99.6% (current pre-refinement)
   - Sparse inertia: ≥ 99.3% (current post-postorder)
   - Sparse residual: ≥ 99.7% (current post-postorder)
6. Worst residuals on both paths should drop. The remaining gap may
   not close to 100% — refinement cannot help with structural rank
   deficiency or wrong inertia from BK kernel decisions on
   ill-conditioned matrices. Whatever the new numbers are, they
   become the baseline for the Phase 1b exit validation document.

## Risk

- **Test flakiness.** Iterative refinement convergence depends on
  the BK pivot quality. If a test asserts convergence in 0 steps and
  the matrix is borderline, the test could be flaky. Mitigation: use
  the same fixtures that already work in the dense `solve_refined`
  property tests.

- **Bench numbers might get worse.** If refinement somehow
  destabilizes (e.g. on a matrix where the residual was already
  small but x is wrong because of rank deficiency), the residual
  pass count could drop. Investigate any regression before
  committing.

- **Worst-residual matrix may change.** ERRINBAR_0824 is the current
  sparse worst at 3.14e-4. After refinement, a different matrix may
  surface as the new worst. Note in the commit body.
