# F1.0 — Multi-RHS Solve: Research Note

**Date:** 2026-04-27
**Phase:** F1.0 of `dev/plans/kkt-feature-gaps.md`
**Goal:** Decide the public API and the per-supernode kernel
shape for batched right-hand-side solves before writing F1.1
code.

## Why we want this

Mehrotra predictor-corrector IPM does two solves per Newton
iteration with the same factor: the predictor (affine) step and
the corrector step. Today feral's downstream caller has to invoke
`Solver::solve` twice, paying a workspace allocation each time
and missing the trsm/gemm batching that BLAS-style kernels enable
at small column counts.

Other consumers of multi-RHS:
- Sensitivity analysis: solve `A · X = J` for a Jacobian-RHS matrix.
- Warm-start parametric LP/QP: same factor, sequence of perturbed
  RHSes.
- Trust-region methods: candidate-step + tentative-step batching.

## Reference solver behavior

### MUMPS (Fortran 5.8.2)

User manual §3.10. Multi-RHS is controlled by:

- `NRHS` — number of columns.
- `LRHS` — leading dimension of the dense RHS matrix on entry
  (must satisfy `LRHS ≥ N`). Allows the caller to pass a sub-block
  of a larger matrix without copying.
- `ICNTL(20)` — RHS format dispatch:
  - 0 (default): dense RHS, column-major.
  - 1: sparse RHS in CSC form (efficient for sensitivity-analysis
    workloads where most RHS columns are unit vectors).
  - 2/3: distributed (irrelevant on shared-memory).
- `ICNTL(27)` — block size for solve. MUMPS internally tiles the
  multi-RHS solve in blocks of `ICNTL(27)` columns to control
  workspace usage on out-of-core executions. Default is `NRHS`
  for in-core (no tiling).

Internal kernel: per-front, MUMPS calls level-3 BLAS `DTRSM` for
the triangular solve and `DGEMM` for the off-diagonal updates.
The block size `nb` matches the factor block size `KEEP(8)`
(default 32).

### SSIDS (C++/Fortran)

User guide §6.1. Multi-RHS via:

```c
void ssids_solve(int nrhs, double *x, int ldx, void *akeep,
                 void *fkeep, struct ssids_options *options,
                 struct ssids_inform *inform);
```

`x` is column-major `n × nrhs` with leading dimension `ldx ≥ n`.
SSIDS processes the columns one supernode at a time, calling
`dtrsm`+`dgemm` per node when `nrhs > 1`, falling through to
single-column `dtrsv`+`dgemv` for `nrhs == 1`.

### Ipopt's MUMPS adapter

`IpMumpsSolverInterface.cpp::MultiSolve` packs the predictor +
corrector RHSes into a single `2*N` buffer, calls MUMPS with
`NRHS=2`, then unpacks. The current Ipopt-MUMPS path in
production never goes above `NRHS=2`, but the API supports it.

## Decisions

### D1. Layout: column-major, `ldx >= n`

Match MUMPS and SSIDS exactly. The RHS / solution buffer is
column-major `n × k` with optional leading-dimension `ldx ≥ n`
support deferred to F1.4 (the simple `ldx == n` form covers IPM;
sub-block support is a sensitivity-analysis convenience).

```rust
pub fn solve_sparse_many_into(
    factors: &SparseFactors,
    rhs: &[f64],          // column-major, len = n*k (Phase F1.1-F1.3)
    nrhs: usize,
    x_out: &mut [f64],    // column-major, len = n*k
    ws: &mut SolveManyWorkspace,
) -> Result<(), FeralError>;
```

`ldx > n` (sub-block view) is a function signature change that
adds a `ldb`/`ldx` parameter pair and a stride argument inside
the gather/scatter loops. Defer.

### D2. Workspace shape

`SolveWorkspace` (single-RHS) holds:
```rust
y: Vec<f64>,            // length n
w: Vec<f64>,            // length max_nrow
scaled_rhs: Vec<f64>,   // length n or 0
```

`SolveManyWorkspace` (multi-RHS) becomes:
```rust
y: Vec<f64>,            // length n * nrhs
w: Vec<f64>,            // length max_nrow * nrhs
scaled_rhs: Vec<f64>,   // length n * nrhs or 0
```

The per-supernode `w` is the gather/scatter buffer; widening it
to `k` columns column-major `[max_nrow x k]` is the only shape
change downstream code sees. Allocation cost grows linearly with
`k`; for `k ≤ 8` and `max_nrow ≤ 1000` this is ≤ 64 KB extra.

### D3. Kernel dispatch

Three regimes, all hit by the same outer code:

- `nrhs == 1`: scalar inner loops (today's code) via a thin
  wrapper that elides the column dimension.
- `2 ≤ nrhs ≤ 8`: column-batched scalar loops. The forward solve's
  inner update
  ```rust
  for i in (j+1)..nrow { w[i] -= ff.l[j*nrow+i] * w_j; }
  ```
  becomes
  ```rust
  for i in (j+1)..nrow {
      let l_ij = ff.l[j*nrow+i];
      for c in 0..nrhs {
          w[c*nrow + i] -= l_ij * w[c*nrow + j];
      }
  }
  ```
  This is a length-`k` stride-1 axpy. The compiler auto-vectorizes
  when `k` is a runtime value bounded by 8 in the IPM hot path.
- `nrhs > 8`: defer to F1.4. Real BLAS-3 trsm/gemm shapes start
  paying off above `k ≈ 16`; the IPM hot path doesn't go there.

### D4. Symmetric solve identity

For symmetric A factored as `P A Pᵀ = L D Lᵀ`, the solve is
identical column-by-column. There is no benefit to a "batched
symmetric solve" beyond the per-supernode trsm/gemm — the outer
loop is still over supernodes. This is consistent with both MUMPS
and SSIDS.

### D5. Public API surface

```rust
// New free functions, mirroring solve_sparse / solve_sparse_into_ws.
pub fn solve_sparse_many(
    factors: &SparseFactors,
    rhs: &[f64],
    nrhs: usize,
) -> Result<Vec<f64>, FeralError>;

pub fn solve_sparse_many_into(
    factors: &SparseFactors,
    rhs: &[f64],
    nrhs: usize,
    x_out: &mut [f64],
    ws: &mut SolveManyWorkspace,
) -> Result<(), FeralError>;

// Solver convenience: solve k columns against the most recent factor.
impl Solver {
    pub fn solve_many(&self, rhs: &[f64], nrhs: usize)
        -> Result<Vec<f64>, FeralError>;
    pub fn solve_many_refined(
        &self,
        matrix: &CscMatrix,
        rhs: &[f64],
        nrhs: usize,
    ) -> Result<Vec<f64>, FeralError>;
}
```

`nrhs == 0` returns an empty `Vec`, matching `solve_sparse(n=0)`.
`nrhs == 1` is a fast-path forward to `solve_sparse_into_ws`.

### D6. Refinement composition

`solve_many_refined` runs the refinement loop **per column**,
not all-at-once. Justification:
- Each column has its own residual norm; convergence is per-column.
- The K columns may take different iteration counts (one converges
  in 0 steps, another in 5).
- Doing per-column refinement matches the MUMPS+Ipopt
  predictor-corrector pattern exactly: predictor refines to its
  target; corrector refines to its (different) target.

This means F1.3 does not need a "batched refinement" kernel; it
loops `solve_refined` over columns sharing the same factor and
workspace.

## Code-touch map (anticipating F1.1 implementation)

The change in `src/numeric/solve.rs`:

1. New struct `SolveManyWorkspace` (analogous to today's
   `SolveWorkspace` but with `nrhs` baked into its allocations).
2. New `fn solve_sparse_core_many_into` extracted from
   `solve_sparse_core_into` with the inner loop widened to `k`.
   Single-column path is `solve_sparse_core_many_into(..., 1)`.
3. New `fn solve_sparse_many_into` analogous to the current
   `solve_sparse_into_ws`, with the pre/post-scaling extended
   to `k` columns.
4. New free fn `solve_sparse_many` allocating the result buffer
   and forwarding to the `_into` form.
5. New `Solver::solve_many` and `Solver::solve_many_refined` in
   `src/numeric/solver.rs`.

The current `solve_sparse_core_into` is preserved (and forwards
to the multi-RHS form with `nrhs == 1`) so the iterative-
refinement code path doesn't change shape.

## Test plan (F1.1)

1. **Equivalence**: on the existing 5×5 panel, generate three
   independent RHSes; assert `solve_many` of the stacked RHS
   matches three calls to `solve` column-by-column to within
   `100 * f64::EPSILON` per entry.
2. **Edge cases**: `nrhs == 0`, `nrhs == 1`, `n == 0`, dimension
   mismatch (rhs.len() != n*nrhs).
3. **Refinement parity**: `solve_many_refined` of two columns
   matches two calls to `solve_refined` to within machine eps.
4. **Workspace reuse**: after one `solve_sparse_many_into` call
   with k=2 followed by another with k=2, the second result is
   correct (no stale workspace state).
5. **Scaling-active path**: the MC64 pre/post-scaling correctly
   applies to all k columns. Use a deliberately ill-scaled small
   matrix from the parity panel.

## Bench plan (F1.2)

Wire into the existing bench harness through a new
`FERAL_BENCH_NRHS=k` env knob. For each matrix in the
small-frontal panel record:

- `solve_us_k1` — single-column baseline (existing number)
- `solve_us_k4` — `nrhs=4` batched
- `solve_us_k8` — `nrhs=8` batched

Acceptance: at `k=4` the per-column cost (`solve_us_k4 / 4`)
should be ≤ 75% of `solve_us_k1` on small-frontal matrices —
i.e., real amortization from sharing the supernodal traversal
overhead across columns. If the multiplier is ≥ 1.0× per column,
the batching kernel needs work before F1.3 lands.

## Open questions (close before F1.1)

1. **Should `Solver::solve_many` accept `&mut [f64]` to match the
   existing `solve` (which returns `Vec<f64>`)?** Decision: no.
   `solve_many` returns `Vec<f64>`; callers wanting in-place can
   use `solve_sparse_many_into`. Mirrors the single-RHS surface.

2. **Do we expose `nrhs` to the bench harness as a CLI flag or
   env var?** Env var `FERAL_BENCH_NRHS=k`, consistent with the
   existing `FERAL_ORDERING`/`FERAL_SCALING`/`FERAL_BENCH_DUMP`
   pattern.

3. **Does iterative refinement need cross-column termination?**
   No (per D6). Per-column refinement is correct and matches the
   predictor-corrector use case.

## References

- MUMPS user manual 5.8.2, §3.10 (multi-RHS), §3.11 (sparse RHS),
  `ICNTL(20)`, `ICNTL(27)`
- SSIDS user guide, §6.1 (`ssids_solve`), §6.6 (solve_inquiry)
- Ipopt 3.14 source: `IpMumpsSolverInterface.cpp::MultiSolve`
- Mehrotra 1992: "On the implementation of a primal-dual interior
  point method", SIAM J Optim 2(4)
- Higham 2002: "Accuracy and Stability of Numerical Algorithms"
  §13 (LDLᵀ multi-RHS solve stability)
