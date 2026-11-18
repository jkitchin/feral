# Plan — issue #237 step 1: permute warm path

Research: `dev/research/issue-237-permute-warm-path-2026-11-18.md`.

1. Harness `examples/issue237_permute_warm.rs` (done): baseline
   permute 74.8 ms, factor 1280.6 ms (sequential, profiled).
2. Tests first (`src/numeric/factorize.rs` test module):
   - warm == cold bitwise, with a `-0.0` entry;
   - warm == cold bitwise, with duplicate `(i,j)`/`(j,i)` entries;
   - lost buffer → correct result, cache rebuilt;
   - `pattern_verified` cleared after a driver call.
3. `PermuteCache`: replace `permuted_col_ptr`/`permuted_row_idx` with
   `permuted: Option<CscMatrix>` + `value_map_bijective: bool`.
4. Warm branch: take the buffer, overwrite values, return it.
5. All three drivers (schur-inner, sequential, parallel) call
   `ws.restore_permuted(permuted)` before their final return.
6. `FactorWorkspace::pattern_verified` set by `Solver::factor_impl`,
   consumed in the drivers, used to skip the `O(nnz)` compare.
7. `cargo test --release`, harness after; digest must match the
   baseline build at equal `iters`.
8. Measure the remaining scatter; decide on the parallel gather.
