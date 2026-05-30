# Issue #58 — Batched iterative refinement for the multi-RHS solve

**Date:** 2026-05-30
**Issue:** #58 — `solve_many_refined` loops per-RHS, bypasses the BLAS-3
panel kernel.
**Builds on:** issue #57 fix #2 (`dev/research/issue-57-blas3-panel.md`,
commit 9c2c716) — the register-blocked panel kernels that wide
`solve_sparse_many` now uses.

## Symptom

`Solver::solve_many_refined` (`src/numeric/solver.rs:1248`) loops over
columns calling the **single-RHS** `solve_sparse_refined` per column:

```rust
for c in 0..nrhs {
    let xc = solve_sparse_refined(matrix, factors, &rhs[c*n..(c+1)*n])?;
    out[c*n..(c+1)*n].copy_from_slice(&xc);
}
```

So every solve inside refinement is single-RHS — the BLAS-3 panel kernel
that issue #57 fix #2 added is never reached. With refinement on (the
default), batched multi-RHS is 3–7× slower per RHS than the *unrefined*
batched path, and can be slower than looping single-RHS. In pounce's
`jax.jacrev` backward pass (`kkt_solve_many`) this is ~25 ms vs ~4 ms at
n=512, N=64 — the exact path fix #2 was meant to accelerate.

## Fix: batch the refinement loop

Refine on a **batched residual + batched solve** instead of per column.
Mirror `solve_sparse_refined_core` but with `solve_sparse_many` inside:

```text
X = solve_sparse_many(factors, RHS, nrhs)        // one batched panel solve
loop:
    R = RHS − A·X            (per-column symv; one pass over CSC per col)
    freeze columns whose ‖r_c‖/‖b_c‖ < ε·√n (or that diverge/plateau)
    dX_active = solve_sparse_many(factors, R_active, n_active)  // batched
    X_active += dX_active
return best-iterate X per column
```

### What is preserved exactly (so nothing breaks)

1. **Per-column best-iterate + done tracking.** The single-RHS refiner
   returns the *best* iterate seen (smallest ‖r‖), and stops a column on
   convergence, on a 2-strike plateau, or on >100× divergence. This is a
   correctness guard: on near-singular columns refinement can amplify
   error, and best-iterate guarantees the result is no worse than the
   unrefined solve. We keep this **per column** — not the issue's
   simpler global-norm loop, which would re-add `dX` to already-converged
   columns and could regress a near-singular column. Same constants:
   `max_steps = 10`, `max_stagnant_steps = 2`, threshold `ε·√n`,
   divergence factor `100`, relative target `‖r‖/‖b‖`.
2. **Active-column compaction.** Each step gathers only the un-converged
   columns into the batched solve, so the batched path solves **≤** the
   columns the per-column loop would, at a **lower** per-RHS cost. This
   removes the only way batching could regress: a heterogeneous batch
   (most columns converge in 1 step, a few need 10) where solving the
   full batch every step would do more work than per-column. With
   compaction the common case (all columns converge together in 1–2
   steps) and the heterogeneous case are both bounded by the per-column
   work.
3. **Residual via the tested `CscMatrix::symv`,** called per column.
   Each column of `X`/`R` is a contiguous slice, so this is cache-
   friendly, and it reuses the existing tested SpMV (no new SpMV bug
   surface). For sparse KKT (the target workload) the residual is a
   small fraction of the solve. A single-pass batched SpMV is a possible
   future lever for dense inputs.

### Threshold dispatch

`Solver::solve_many_refined` dispatches:

- `nrhs < BLAS3_REFINE_THRESHOLD` (= 16) → the **current** per-column
  loop, unchanged. This keeps the IPM predictor-corrector (`nrhs = 2`)
  and other narrow refined solves on the literal, proven code path.
- `nrhs ≥ 16` → `solve_sparse_many_refined`.

16 (not 32) because the batched **solve** amortizes the supernodal
traversal from `nrhs ≈ 16` even below the `BLAS3_NRHS_THRESHOLD = 32`
panel-kernel crossover (issue #57 bench: ratio 0.37 at `nrhs = 16`), and
the batched refinement is provably safe there (see bit-exactness below).

## Bit-exactness / accuracy

- **`16 ≤ nrhs < 32`: bit-identical to the per-column loop.** There
  `solve_sparse_many` runs the rank-1 row-major kernels, whose per-column
  output equals `solve_sparse` of that column bit-for-bit (fix #1); the
  rank-1 kernels treat columns independently, so a column's result does
  not depend on its batch position or the compacted set. With the
  per-column logic replicated exactly, batched-refined == per-column-
  refined bit-for-bit.
- **`nrhs ≥ 32`: equal to the refinement residual target.** There the
  BLAS-3 back-substitution differs from single-RHS by float
  reassociation (~κ·eps), so the initial/correction solves and hence the
  refinement trajectory differ slightly. But both converge to the same
  relative-residual target `‖r‖/‖b‖ < ε·√n`, so the final solutions agree
  to ~κ(A)·(target) — machine precision on well-conditioned systems.

## Oracle & acceptance (CLAUDE.md external-oracle rule)

Oracle = the pre-existing single-RHS `solve_sparse_refined`.
`solve_sparse_many_refined` of a stacked RHS must equal `k` independent
`solve_sparse_refined` calls per column:

1. `nrhs = 24` (in the `[16, 32)` bit-identical band): assert
   `max|batched − per-column| == 0` (tolerance not loosened — exact).
2. `nrhs = 64` (BLAS-3 band): assert the batched-refined solution matches
   the per-column oracle to a tolerance tied to the residual target, and
   that the batched residual `‖R‖/‖RHS‖` meets `ε·√n` per column.
3. Existing `solve_many_refinement_per_column_parity` (nrhs = 2) stays on
   the per-column path → unchanged.
4. `cargo test --lib` ≥ 317; clippy `-D warnings` clean; fmt clean.
5. Bench: refined batched per-RHS time approaches unrefined batched (the
   issue's regression closes); report measured numbers.

## Results (2026-05-30)

Implemented as designed; `cargo test --test multi_rhs` 13 pass including
the exact bit-identical band (`max|batched − per-column| == 0` at
nrhs=24 SPD and nrhs=20 indefinite). Rust `bench_multirhs`: refined
batched/per-column ratio ~0.30–0.42 (~2.5–3×).

**Allocation overhead found and fixed (the "53 µs").** A notebook
showcase exposed that the *Python* `Solver.solve_refined(2-D)` path
showed ~1× while the Rust API showed ~2.5×, on the same matrix, RHS,
scaling, and with 0 correction steps. A probe ruled out the factor,
scaling, and the pyo3 marshaling (the `solve`/`solve_refined` 2-D
wrappers are byte-identical; `inner()` is a cheap reference). Root cause:
`solve_sparse_many_refined` allocated three `n × nrhs` Vecs up front
(`best_x.clone` + `r` + `r_act` ≈ 3 MB/call). Cheap in the standalone
binary (allocator reuse), but ~50 µs/RHS under the Python process where
~1 MB blocks hit `mmap`/page-faults each call.

Fix: compute the initial residual into a length-`n` scratch, build the
active set, and **return early before allocating the wide buffers** when
no column needs refinement (the common case); size the gather buffer to
the active set. Python refined batch dropped 87 → 30 µs/RHS (matching
the Rust API), and the win now surfaces through Python: ~2.86× at
n=2025, and ~2.2–2.5× in the heat-conduction notebook (n=1600). The 1×
was removable overhead, not a fundamental limit.

## Out of scope / follow-ups

- **Single-pass batched SpMV for the residual.** The per-column residual
  is now the only un-batched cost; on dense Hessians (matvec ≈ solve) it
  caps the speedup. This is the remaining lever.
- Reusing one `SolveManyWorkspace` across refinement steps (compaction
  changes `nrhs` per step, so each step still allocates a workspace;
  refinement is 1–2 steps in practice, so this is minor).
