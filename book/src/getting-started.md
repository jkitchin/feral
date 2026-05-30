# Getting started

Add feral to your `Cargo.toml`:

```toml
[dependencies]
feral = "0.8"
```

feral is primarily a **sparse** symmetric indefinite solver. The
ergonomic entry point is the [`Solver`](./api.md) type: build a matrix,
factor it once, then solve it against one or many right-hand sides.

## Sparse quickstart

```rust,ignore
use feral::Solver;
use feral::numeric::solver::FactorStatus;
use feral::sparse::csc::CscMatrix;

// Build a symmetric matrix from triplets of its LOWER triangle
// (row, col, value), `n` rows/cols. Duplicate entries are summed.
let n = 5;
let rows = [0, 1, 2, 3, 4, 1, 2, 3, 4];
let cols = [0, 0, 0, 0, 0, 1, 2, 3, 4];
let vals = [10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
let a = CscMatrix::from_triplets(n, &rows, &cols, &vals)?;

// Factor once. Pass `None` to skip the inertia check, or
// `Some(expected)` to verify the matrix has the inertia you expect.
let mut solver = Solver::new();
let status = solver.factor(&a, None);
assert_eq!(status, FactorStatus::Success);

// The factorization carries a certified inertia (n_pos, n_neg, n_zero).
println!("inertia: {:?}", solver.inertia());

// Solve A·x = b for a single right-hand side.
let b = [1.0, 2.0, 3.0, 4.0, 5.0];
let x = solver.solve(&b)?;
```

`solver.factor` returns a [`FactorStatus`](./api.md); match on it to
distinguish `Success`, `Singular`, `WrongInertia`, and `FatalError`.
Once factored, the `Solver` keeps the factor, so any number of solves
reuse it.

## Many right-hand sides (batched solve)

The same factorization can be solved against many right-hand sides at
once. `solve_many` shares the supernodal traversal across columns, so it
is substantially cheaper than looping a single-RHS solve — and for wide
`nrhs` it runs each supernode's dense panel as register-blocked
TRSM + GEMM kernels.

`B` and `X` are **column-major** `n × nrhs` matrices, stored as flat
slices of length `n * nrhs` (column `c` occupies `[c*n .. (c+1)*n]`):

```rust,ignore
let nrhs = 64;
// b_many[c*n + i] is row i of right-hand-side column c.
let b_many: Vec<f64> = make_columns(n, nrhs);
let x_many = solver.solve_many(&b_many, nrhs)?;   // length n * nrhs
```

This is the path behind batched KKT back-solves, `jax.jacrev` over a
solve, sensitivity analysis, and parameter sweeps. On 2-D Laplacians it
is roughly 3–6× faster per RHS than looping single-RHS solves (the exact
factor depends on size, CPU SIMD width, and cache). See GitHub issue #57
and the [`02_multi_rhs_batched`
notebook](https://github.com/jkitchin/feral/tree/main/python/examples/notebooks).

## Iterative refinement

With `ZeroPivotAction::ForceAccept` (the default), an unrefined solve can
leave a residual on near-singular pivots. `solve_refined` runs a few
steps of iterative refinement against the original matrix and returns
the best iterate:

```rust,ignore
let x = solver.solve_refined(&a, &b)?;                       // single RHS
let x_many = solver.solve_many_refined(&a, &b_many, nrhs)?;  // batched
```

`solve_many_refined` keeps per-column best-iterate convergence but, for
wide right-hand sides, refines through the same batched panel kernel as
`solve_many` — one batched solve per refinement step over the still-
unconverged columns — so the refined path amortizes too (issue #58).

## Dense path

For small dense systems there is a direct API that mirrors the sparse
one. `factor` returns the factors **and** the certified inertia:

```rust,ignore
use feral::{factor, solve, BunchKaufmanParams, SymmetricMatrix};

// Lower-triangle entries (row >= col) of an n×n symmetric matrix.
let a = SymmetricMatrix::from_lower_triangle(n, &entries);
let (factors, inertia) = factor(&a, &BunchKaufmanParams::default())?;
let x = solve(&factors, &b)?;
```

## More

- Runnable Rust programs:
  [`examples/`](https://github.com/jkitchin/feral/tree/main/examples)
  exercises the dense and sparse paths, scaling, and refinement.
- Python bindings and notebooks: see [Python bindings](./python.md).
- Inertia guarantees on singular matrices: see
  [Inertia semantics](./inertia.md).
