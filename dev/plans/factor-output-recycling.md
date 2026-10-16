# Plan: recycle per-front factor outputs on warm refactors

Research: `dev/research/factor-output-recycling-2026-10-16.md`.

1. **Tests first** (`tests/factor_output_recycling.rs`, plus a unit test
   in `solver.rs`):
   - (a) A warm `Solver` factoring iterate k+1 is byte-identical to a
     fresh `Solver` factoring it. This covers every `NodeFactors` buffer
     and the solve result. Run it with the sequential and parallel
     drivers, on fixtures with and without delayed pivots, and with
     values chosen so the delay count changes between iterates
     (different buffer sizes).
   - (b) The unit test asserts recycling actually engages: a hit counter
     is greater than 0 on a warm refactor and 0 on a cold one. It fails
     before the implementation.
2. Add `FactorScratch::out: FactorOutputBufs`, a `Default` set of
   recyclable output Vecs. Use it in both BK kernels (eager and blocked)
   for `l`, `d_diag`, `d_subdiag`, `perm` and `perm_inv`.
3. Add `FactorWorkspace::recycled: Vec<Option<RecycledNode>>`, filled by
   `recycle_factors(SparseFactors)`, which `Solver` calls right before
   the numeric driver. The drivers install set `i` before supernode `i`:
   sequential via `take`, parallel via a per-node `Mutex` slot vector.
   `row_indices` reuse is part of the same install.
4. Pool the `contrib_row_indices` buffer through a single slot, like
   `contrib_pool`. This is a separate commit.
5. Measure with `diag_153_chain_x86 mtx` across the 77-matrix corpus,
   against glibc (before), mimalloc (the ceiling) and the patched glibc
   build. Paired alternating runs, 3 rounds.
6. Gates:
   - `cargo test` passes, including the golden digests;
   - clippy is clean;
   - no inertia or nnz_l change on the corpus;
   - `cargo run --bin bench --release` passes.
