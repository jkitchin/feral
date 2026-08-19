# Plan: caller-capped refinement + in-place solves (issue #178)

Research note: `dev/research/refinement-cap-2026-08-19.md`.

Two independent deliverables. Item 1 is the substantive one.

## Item 1 — `RefineOptions` cap

### New public type (`src/numeric/solve.rs`, re-exported at crate root)

```rust
pub const DEFAULT_REFINE_MAX_STEPS: usize = 10;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RefineOptions { pub max_steps: usize }

impl Default for RefineOptions { /* max_steps: DEFAULT_REFINE_MAX_STEPS */ }
impl RefineOptions { pub fn with_max_steps(max_steps: usize) -> Self }
```

### Threading

`solve_sparse_refined_core` gains an `opts: RefineOptions` parameter and
uses `opts.max_steps` in place of the hard-coded `let max_steps = 10`.
All three existing wrappers pass `RefineOptions::default()`, so they are
unchanged by construction.

New `_opts` entry points, each the existing one plus a trailing `opts`:

- `solve_sparse_refined_opts`
- `solve_sparse_refined_parallel_opts`
- `solve_sparse_refined_with_diagnostics_opts`
- `solve_sparse_many_refined_opts`

`solve_sparse_many_refined` gains the same parameter internally and keeps
its four-argument signature as a `Default` wrapper.

`Solver` gains `solve_refined_opts(matrix, rhs, opts)` and
`solve_many_refined_opts(matrix, rhs, nrhs, opts)`. `Solver::solve_many_refined`'s
`BLAS3_REFINE_THRESHOLD` dispatch threads `opts` into both arms — the wide
batched path and the narrow per-column loop.

### `max_steps = 0` fast path

After the initial solve, when `opts.max_steps == 0` and diagnostics are
off, return immediately — before the residual matvec. This makes `k = 0`
the same *computation* as `solve_sparse`, not just the same answer.
Under diagnostics, `steps[0]` is still emitted (the matvec is the point
there), and the loop simply does not run.

Same treatment in the multi-RHS path: return the initial batched solve
before the per-column residual sweep.

## Item 2 — in-place entry points

### Free functions (`src/numeric/solve.rs`)

- `solve_sparse_into(factors, rhs, x_out)` — public wrapper over the
  existing `pub(super) solve_sparse_into_ws`, building the workspace.
- `solve_sparse_refined_core` writes the best iterate into a caller
  `x_out: &mut [f64]` and returns only `Option<RefinementDiagnostics>`;
  `x_out` *is* the `best_x` storage, so the refined path loses an
  `n`-length allocation. The allocating wrappers become
  `let mut x = vec![0.0; n]; core(..., &mut x)?; Ok(x)`.
- `solve_sparse_many_refined_into(matrix, factors, rhs, nrhs, x_out, opts)`
  — same restructuring for the multi-RHS loop.

### `Solver` methods

| allocating | in-place |
|---|---|
| `solve` | `solve_into(rhs, x_out)` |
| `solve_refined` | `solve_refined_into(matrix, rhs, x_out, opts)` |
| `solve_many` | `solve_many_into(rhs, nrhs, x_out)` |
| `solve_many_refined` | `solve_many_refined_into(matrix, rhs, nrhs, x_out, opts)` |

The refined `_into` variants take `opts` rather than spawning a second
`_into_opts` name; `RefineOptions::default()` reproduces the allocating
twin exactly.

Every `_into` validates `x_out.len()` and returns
`FeralError::DimensionMismatch` — never a panic, never a silent partial
write. `NoFactor` precedence is unchanged (checked before dimensions, as
the allocating twins do).

Aliasing `rhs` / `x_out` is statically unrepresentable in safe Rust
(`&[f64]` and `&mut [f64]` into one allocation cannot coexist). Documented
on each `_into` method; no runtime check, because there is no reachable
state to check for.

## Tests (written before the implementation)

Mapping to issue #178's verification list:

1. **Cap is honoured.** On an ill-conditioned matrix whose default run
   returns `steps.len() == 11`, the same solve under `k = 1` returns
   `steps.len() == 2` — exactly one correction. Also `k = 3` -> 4.
2. **Cap is a cap, not a target.** A well-conditioned system that exits
   early under `k = 10` returns the identical step count and identical
   `x` under a much larger `k` — raising the cap adds no work.
3. **`k = 0` equals `solve_sparse` bit-for-bit** on the same factor
   (`to_bits()` comparison, not a tolerance).
4. **Best-iterate contract under any `k`.** For `k` in `0..=10`, the
   returned residual norm is `<=` the unrefined solve's, and the returned
   iterate matches the diagnostics' `returned_step`.
5. **All four for the multi-RHS path**, with per-column independence:
   a two-column RHS whose columns need different step counts must behave
   the same capped as the single-RHS refiner does per column.
6. **In-place bit-identity.** Each `_into` variant equals its allocating
   twin bit-for-bit (`to_bits()`), on the same factor and RHS.
7. **In-place dimension errors.** A wrong-length `x_out` returns
   `DimensionMismatch`, and a no-factor solver returns `NoFactor` first.

The ill-conditioned matrix for tests 1 and 4 must be one that genuinely
runs the full budget under default options; the test asserts
`steps.len() == 11` on the default run first, so it fails loudly rather
than silently vacuously if the matrix stops being hard.

## Out of scope

- Changing any default. `DEFAULT_REFINE_MAX_STEPS` stays 10.
- The C ABI. `feral_solve` already solves in place into its `rhs` buffer
  and exposes no refinement knob; adding one is a separate ask.
- pounce-side acceptance (issue #178 items 6 and 7) — the reporter runs
  those against a build of this branch.
