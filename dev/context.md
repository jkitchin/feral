# FERAL Context (auto-generated)

Generated: 2026-07-02T11:32:41Z

## Latest Session
File: dev/sessions/2026-07-02-01.md
```
# Session 2026-07-02-01

## Goal
Implement issue #107: add `OrderingMethod::External(Vec<usize>)` so callers can
inject a precomputed fill-reducing permutation into symbolic factorization,
instead of only selecting an ordering *algorithm*. Mirror the existing
`ScalingStrategy::External(Vec<f64>)`.

## Accomplished
- **Research + plan first** (`dev/research/issue-107-external-ordering.md`,
  `dev/plans/issue-107-external-ordering.md`), then tests, then implementation.
- **`OrderingMethod::External(Vec<usize>)`**: a 0-based new-to-old permutation
  of `0..n` (same convention as `SymbolicFactorization::perm`). Injected at the
  single `run_external_ordering` point — for `External` it returns the user perm
  directly instead of calling an ordering crate; the rest of the pipeline
  (postorder → etree → column counts → supernodes) is byte-identical to any
  other method.
- **`Copy` → `Clone`.** A `Vec` field is not `Copy`, so `OrderingMethod` drops
  `Copy` (kept `Clone`), matching `ScalingStrategy`. Hand-written `Debug` prints
  `External { len: N }` so `NumericFactorization::summary` doesn't dump the perm.
- **Validation up front** (`validate_external_perm`): wrong length /
  out-of-range / duplicate → `FeralError::InvalidInput`, never a panic, no
  `unwrap`.
- **`External` forces `OrderingPreprocess::None`.** `LdltCompress` reorders a
  compressed super-graph (dim `ncmp ≤ n`) and cannot consume a full-length user
  perm; guarded so `External` also skips the preprocess-`Auto` fill race.
- **`Copy`-removal fallout** (mechanical clones): `AutoRace` race loop,
  preprocess-`Auto` `run` closure, `Solver::factor` (2), the three numeric
  constructors, `src/bin/bench.rs` (3 match sites), `examples/bench_qap15.rs`,
  `python/src/*` (`ordering_to_str` now takes `&OrderingMethod` + `External`
  arm), and ~9 `feral-diagnostics` bins that reused a `method` binding.
- **Tests** (oracles external/hand): `tests/issue107_external_ordering.rs` — 6
  green. Saddle-point KKT inertia (2,1,0) by the saddle inertia theorem
  (Nocedal & Wright Lemma 16.1); SPD tridiag inertia (n,0,0); RHS `b = K·x_true`
  so the solution is known; identity and reversed external orderings both solve
  to the oracle with identical inertia (ordering changes fill, not the answer);
  validation rejects the three malformed inputs. `src/symbolic` units pin the
  bijection perm + forced `None` preprocess, the validation rejections, and the
  compact Debug.
- **Suite green:** feral **395 lib** + all integration tests pass;
  `cargo test -p feral-diagnostics` green; `cargo clippy --all-targets -- -D
  warnings` clean on both feral and `feral-diagnostics`; `cargo fmt --check`
  clean. Default (non-External) path unchanged.

## Benchmark Results
```
cargo run --bin bench --release (built-in fixtures; no data/ corpus present)
  KKT dense: Inertia match 1/1 (100%), Residual pass 1/1, worst 1.14e-15
  vs MUMPS:  Inertia match 2/2 (100%), Residual pass 2/2, worst 1.26e-16
  8 matrices benchmarked; no failures (dense or sparse).
```

## Git Status
```
a1dd7b5 release: feral v0.12.0 (#106)
1165d4d issue #102 follow-up: escalate ordering to LdltCompress on pivot growth (#105)
d6fe546 issue #102: fix parallel dense-front re-entrant deadlock (0% CPU stall) (#104)
bc24fc9 issue #99: packed BLAS-3 dense trailing update (byte-exact) + opt-in FMA-large-fronts gate (#103)
f54b335 docs: session checkpoint 2026-07-01-03 (issue #91 dense-kernel follow-up) (#100)
```

## Test Status
```
test symbolic::tests::schur_symbolic_tail_invariant_reversed_user_order ... ok
test symbolic::tests::schur_symbolic_tail_invariant_user_order ... ok
test symbolic::tests::symbolic_factorize_amf_produces_valid_perm ... ok
test symbolic::tests::symbolic_factorize_auto_produces_valid_perm ... ok
test symbolic::tests::symbolic_factorize_default_uses_amf_for_small_matrices ... ok
test symbolic::tests::symbolic_factorize_external_produces_valid_perm ... ok
test symbolic::tests::symbolic_factorize_kahip_produces_valid_perm ... ok
test symbolic::tests::symbolic_factorize_metis_produces_valid_perm ... ok
test symbolic::tests::symbolic_factorize_scotch_produces_valid_perm ... ok
test symbolic::tests::test_contrib_sizes_nonnegative ... ok
test symbolic::tests::test_perm_inverse_consistency ... ok
test symbolic::tests::test_symbolic_factorize_basic ... ok
test symbolic::tests::test_symbolic_factorize_dense ... ok
test symbolic::tests::test_symbolic_factorize_kkt ... ok
test symbolic::tests::issue_3_scotchnd_on_kkt_recurses_after_o13 ... ok
test symbolic::tests::issue_3_auto_on_kkt_routes_via_pick_default_method ... ok
test scaling::hungarian::tests::mc64_hungarian_no_quadratic_heap_realloc_regression ... ok

test result: ok. 395 passed; 0 failed; 6 ignored; 0 measured; 0 filtered out; finished in 1.95s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-07-02-01.md)

cargo run --bin bench --release (built-in fixtures; no data/ corpus present)
  KKT dense: Inertia match 1/1 (100%), Residual pass 1/1, worst 1.14e-15
  vs MUMPS:  Inertia match 2/2 (100%), Residual pass 2/2, worst 1.26e-16
  8 matrices benchmarked; no failures (dense or sparse).
No performance change expected or observed: `External` is a new opt-in ordering
source; the default `Auto` path is byte-exact.

```

## Recent Decisions
diverge from the scaling precedent. The `Copy` loss is mechanical (clone at the
`AutoRace` race loop, the preprocess-`Auto` race, `Solver::factor`, the three
numeric constructors, and the `feral-diagnostics` bins that reuse a `method`
binding) and was absorbed.

**Why `External` forces `OrderingPreprocess::None`.** `LdltCompress` reorders an
MC64-compressed super-graph of dimension `ncmp ≤ n`; a full-length user permutation
cannot be applied to it. So `External` bypasses the preprocess-`Auto` fill race and
pins `resolved_preprocess = None`, regardless of the requested (or default `Auto`)
preprocess. Scaling is unaffected — it is computed independently in the numeric
phase from `ScalingStrategy`; only the MC64 *cache-reuse* symbolic-time shortcut is
skipped, which is a performance optimization, not a correctness input.

**Soundness.** Numeric factorization, pivoting, and inertia are untouched — a
factorization under any valid ordering is exact. A bad user ordering only costs
fill/time. The permutation is validated as a bijection of `0..n` up front
(`validate_external_perm`): wrong length, out-of-range index, or duplicate returns
`FeralError::InvalidInput` (never a panic, no `unwrap`). Programmatic-only: no
string parsing, matching scaling's `External`.

**Evidence.** `tests/issue107_external_ordering.rs` (identity + reversed orderings
solve to the hand oracle with saddle-point inertia (2,1,0) and SPD inertia (n,0,0);
validation rejects the three malformed inputs); `src/symbolic` units
(`symbolic_factorize_external_produces_valid_perm` pins the bijection + forced
`None` preprocess; `external_perm_validation_rejects_bad_input`;
`ordering_method_external_debug_is_compact`). Full suite green: feral 395 lib + all
integration, `feral-diagnostics` builds/tests, clippy `--all-targets` clean on both,
`cargo fmt --check` clean. Default (non-External) path unchanged. See
`dev/research/issue-107-external-ordering.md` and
`dev/plans/issue-107-external-ordering.md`.

## Recent Tried-and-Rejected
+23% option but is a reproducibility-policy change (kept opt-in), not a
bit-exact win.

## 2026-07-01 — UPDATE: packed micro-kernel succeeds where B-1a source-pack failed (issue #99)

The 2026-06-30 "B-1a panel packing" entry above rejected source-panel packing as a
net slowdown and concluded the root front is DST-bandwidth-bound. That conclusion
was **specific to the variant tried** — packing the source into a tighter stride
but *feeding the same strided kernels*, which keep the per-`q` `as_simd` + strided
access. It does **not** generalize to a proper packed micro-kernel.

A different design — pack the panel into `q`-contiguous MR=8×NR=4 micro-panels and
run a register-tiled kernel with a **contiguous inner `q`-loop**
(`apply_schur_panel_range_packed`) — is **22–26× faster in isolation and
byte-exact** (`examples/bench_schur_micro`), and gives 8–10× on real dense fronts.
So the bottleneck was strided-`q` cache latency, not DST bandwidth, on this
hardware. Not a rejection — a correction of scope. See
`dev/research/issue-99-dense-front-fma-gate.md` UPDATE 3 and `dev/decisions.md`
2026-07-01 (packed BLAS-3). The B-1a *source-into-strided-kernel* variant remains
rejected; the packed micro-kernel is the shipped design.

## Source Files
```
src/bin/bench.rs
src/capi.rs
src/dense/block_ldlt32.rs
src/dense/equilibrate.rs
src/dense/factor.rs
src/dense/matrix.rs
src/dense/mod.rs
src/dense/rook.rs
src/dense/schur_kernel.rs
src/dense/solve.rs
src/error.rs
src/inertia.rs
src/io/mod.rs
src/io/mtx.rs
src/io/sidecar.rs
src/lib.rs
src/lu/condition.rs
src/lu/dense_factor.rs
src/lu/dense_matrix.rs
src/lu/dense_solve.rs
src/lu/dense_update.rs
src/lu/mod.rs
src/lu/scaling.rs
src/lu/sparse_factor.rs
src/lu/sparse_matrix.rs
src/lu/sparse_solve.rs
src/lu/sparse_symbolic.rs
src/lu/sparse_update.rs
src/numeric/condition.rs
src/numeric/factorize.rs
src/numeric/mod.rs
src/numeric/solve.rs
src/numeric/solver.rs
src/ordering/amd.rs
src/ordering/elimination_tree.rs
src/ordering/mod.rs
src/ordering/postorder.rs
src/ordering/schur.rs
src/scaling/hungarian.rs
src/scaling/infnorm.rs
src/scaling/mc64.rs
src/scaling/mod.rs
src/scaling/value_bound.rs
src/sparse/csc.rs
src/sparse/mod.rs
src/symbolic/column_counts.rs
src/symbolic/ldlt_compress.rs
src/symbolic/mod.rs
src/symbolic/profiler.rs
src/symbolic/small_leaf.rs
src/symbolic/supernode.rs
```

## Test Files
```
tests/amf_corpus_oracle.rs
tests/auto_strategy.rs
tests/blocked_ldlt.rs
tests/build_row_indices_trailing_invariant.rs
tests/column_renumbering.rs
tests/column_renumbering_parity.rs
tests/d4_solve_2x2_gate.rs
tests/d6_contrib_uninit.rs
tests/d7_block32_dispatch_pooled.rs
tests/delayed_pivoting.rs
tests/dense_fast_path.rs
tests/dense_ldlt.rs
tests/factor_scratch_parity.rs
tests/factor_workspace_parity.rs
tests/factors_ld_export.rs
tests/fine_grained_delay.rs
tests/fma_opt_in_roundtrip.rs
tests/growth_flag.rs
tests/issue102_intrafront_deadlock.rs
tests/issue102_ordering_escalation.rs
tests/issue107_external_ordering.rs
tests/issue52_stats.rs
tests/issue64_arrow_ordering.rs
tests/issue65_mc64_fallback.rs
tests/issue67_thin_ordering.rs
tests/issue91_preprocess_misfire.rs
tests/issue99_fma_front_gate.rs
tests/issue_15_cascade_arm_gate.rs
tests/issue_17_robot_1600_cascade_off.rs
tests/issue_18_narx_cfy_cascade_off.rs
tests/issue_2_kkt_ls_init.rs
tests/issue_38_static_pivot.rs
tests/issue_46_saddle_kkt_cascade.rs
tests/issue_55_delay_budget.rs
tests/issue_55_n_tiny_counter.rs
tests/kkt_hardening.rs
tests/kkt_matrices.rs
tests/large_matrix_smoke.rs
tests/ldlt_compress.rs
tests/lu_dense.rs
tests/lu_ft_widebump.rs
tests/lu_scaling.rs
tests/lu_sparse.rs
tests/lu_update_alloc_probe.rs
tests/lu_update_casctanks.rs
tests/maxfromm_parity.rs
tests/mc64_end_to_end.rs
tests/mc64_scaling.rs
tests/multi_rhs.rs
tests/n2_static_pivot_scaling.rs
tests/n3_parallel_profiler.rs
tests/n4_mc64_retry_latch.rs
tests/parallel_parity.rs
tests/parity.rs
tests/pivot_rejection.rs
tests/pounce_interface.rs
tests/profiler_smoke.rs
tests/property_tests.rs
tests/rook_rescue.rs
tests/rook_rescue_kkt.rs
tests/small_leaf_parity.rs
tests/solver_with_ordering.rs
tests/sparse_postorder.rs
tests/sparse_refined.rs
tests/sqd_fast_path.rs
tests/stress_tests.rs
tests/symbolic_profiler.rs
tests/threshold_consistency.rs
tests/tiny_fast_path.rs
```
