# FERAL Context (auto-generated)

Generated: 2026-06-11T19:42:44Z

## Latest Session
File: dev/sessions/2026-06-11-02.md
```
# Session 2026-06-11-02

## Goal

Make feral's Python interface more complete, in four areas requested by
the user, under a hard backward-compatibility constraint (every change
purely additive — no existing signature, name, or default changed; the
prior pytest suite must pass unmodified):

1. Expose the unsymmetric LU basis engine (`src/lu/`).
2. Surface the unexposed LDLᵀ tuning knobs (ordering, mc64-cache,
   profiling, partial-singular warning, auto-cascade-break).
3. Give access to the factored L/D (numeric) and the symbolic structure.
4. Add introspection (pivot magnitudes, mc64 counters, factor stats,
   profile reports, scaling info, cache invalidation) and conversion
   conveniences.

Rust-core support (`SparseFactors::ldlt_export`, `Solver::symbolic`
accessor) landed in the previous session as commit ce37da9; this session
is the Python binding on top of it.

## Accomplished

Split the single-file binding into internal modules
`python/src/{common,errors,matrix,solver,factors,symbolic,lu,introspect}.rs`
(extension still `feral._feral`, public surface unchanged), then added:

- **LU basis engine** (`lu.rs`): `LuMatrix` (general square CSC;
  `from_dense`/`from_triplet`/`from_columns`/raw CSC ctor; `matvec`/
  `matvec_transpose`) and `LuFactor`, an auto-routing dense/sparse
  factor (`should_use_dense_lu`, `force_dense` override). `ftran`/`btran`,
  product-form `update`/`update_sparse`/`refactor`, and the `P A Q = L U`
  factor (`perm`, `qcol`, `l_array`/`u_array`, `factor_nnz`, `eta_ops`).
  New exceptions `SingularBasisError` (`<FactorError`) and
  `NeedsRefactorError` (`<FeralError`).
- **Numeric factor access** (`factors.rs`): `Solver.factors()` →
  `Factors` snapshot with `l_csc()`, `d_blocks()`, `to_scipy_l()`, and
  `perm`/`perm_inv`/`scaling`/`needs_refinement`/`ordering`.
- **Symbolic** (`symbolic.rs`): `Solver.symbolic()` → `SymbolicAnalysis`
  and the standalone `feral.analyze(a, ordering=...)` (no numeric work),
  exposing resolved ordering, `etree_parent` (roots `-1`),
  `num_supernodes`, `col_counts`, `factor_nnz_estimate`.
- **Knobs** (`solver.rs`): `Solver(...)` kwargs `ordering`, `mc64_cache`,
  `profiling`, `partial_singular_warning`, `auto_cascade_break` (each an
  `Option` sentinel — when unset the builder is not invoked, so the
  default constructor reproduces prior behavior exactly) + an `ordering`
  getter (resolved method via `symbolic()`/`factors()`).
- **Introspection** (`introspect.rs`): `min_pivot_magnitude`/
  `max_pivot_magnitude`, the four MC64 counters, `scaling_info`
  (`ScalingInfo`), `last_factor_stats` (`FactorStats`), `profile_report`
```

## Git Status
```
ce37da9 feat(core): add LDLt factor export and symbolic accessor
9148ec6 Merge pull request #83 from jkitchin/claude/elegant-hawking-0nmp2y
079aebe docs(session): checkpoint 2026-06-11-01 — repo-review residuals cleared
b99abdd docs: correct ~10 stale doc/comment sites (repo-review item 4)
995456f docs(n4): record MC64-retry latch inertia tradeoff; track deferred N3/N5 facets
```

## Test Status
```
test symbolic::tests::symbolic_factorize_scotch_produces_valid_perm ... ok
test symbolic::tests::test_perm_inverse_consistency ... ok
test symbolic::tests::test_symbolic_factorize_dense ... ok
test symbolic::tests::test_symbolic_factorize_basic ... ok
test symbolic::tests::test_symbolic_factorize_kkt ... ok
test numeric::factorize::tests::issue_5_mss1_iter0_inertia_wanders_under_delta_w_sweep ... ok
test symbolic::tests::is_arrow_bordered_rejects_many_hubs ... ok
test symbolic::tests::choose_adaptive_routes_arrow_to_amf ... ok
test scaling::tests::auto_keeps_mc64_on_vesuvia_0000 ... ok
test symbolic::tests::issue_3_scotchnd_on_kkt_recurses_after_o13 ... ok
test symbolic::tests::choose_adaptive_rules ... ok
test scaling::tests::auto_keeps_mc64_on_vesuviou_0000 ... ok
test numeric::factorize::tests::issue_5_mss1_zero_tol_sweep_diagnostic ... ok
test symbolic::tests::issue_3_auto_on_kkt_routes_via_pick_default_method ... ok
test numeric::factorize::tests::issue_5_mss1_pivot_threshold_sweep_diagnostic ... ok
test scaling::tests::pick_scaling_strategy_routes_clnlbeam_to_infnorm ... ok
test scaling::hungarian::tests::mc64_hungarian_no_quadratic_heap_realloc_regression ... ok

test result: ok. 371 passed; 0 failed; 6 ignored; 0 measured; 0 filtered out; finished in 1.51s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-06-11-02.md)


Not re-run this session. No Rust-core source changed (the core support
landed in ce37da9 last session); all work was in the standalone python/
workspace and docs. The numeric bench is unchanged from session
2026-06-11-01.

```

## Recent Decisions
no tracking entry; recording them here so future sessions can find them (the
verifier's item 6). These are open performance facets, not rejected approaches.

**N3 (parallel driver, `factorize.rs`).** The profiler facet was fixed (the
default parallel dispatch no longer silently returns an empty
`with_profiling(true)` report). Still open on the parallel driver — the
`Solver` default:
- `pattern_reused_hint` / the issue-56 Lever A.2 warm-refactor **permute
  cache** never engages: the parallel driver uses plain `permute_csc_values`,
  so the cache built for *large* matrices is bypassed on exactly those matrices.
- `params.small_leaf` is ignored by the parallel driver (benign today since the
  default is off, but a drift trap if a caller sets it and silently gets the
  sequential-only behavior).

**N5 (per-call allocation churn).** One facet was addressed; still open:
- **Parallel-workspace churn** (`factorize.rs`, ~the `num_threads + 1` fresh
  `FactorWorkspace` construction): the parallel driver allocates per-thread
  workspaces (row_map `n×usize`, build_seen `n` bools, per-snode contrib
  options) plus two mutex-wrapped stores per `factor()`; the sequential path
  pools all of this. `phase_thread_ws_ns` telemetry measures the cost but
  nothing amortizes it.
- **Warm-permute clone** (`factorize.rs`, ~the warm permute path): clones
  `col_ptr` + `row_idx` (`O(nnz)` memcpy) on every warm factor, though the
  structure is immutable per pattern.

These are deferred, not rejected: the correct fix is to pool the parallel
workspaces on the `Solver` (mirroring the sequential pooling) and to borrow the
immutable structure in the warm path rather than clone it. No reproducing test
is meaningful for a pure allocation-churn change; they are guarded by the
existing bit-exactness tests between the sequential and parallel drivers.

## Recent Tried-and-Rejected
The broader finding framing — that the ~600 LoC of duplicated AMD/AMF inner-loop
code is itself a drift hazard — is a structural-refactor observation (the same
class as O20's cross-crate consolidation), not a defect with a reproducing test,
and is likewise deferred rather than undertaken inside a single /loop iteration.

### Disposition

Routed here per the /loop rule (non-reproducible core → tried-and-rejected citing
the finding ID). Per the X16 / O4 / O5 / O6 / O9 / O17 sub-fix precedent, the
safe behaviour-neutral slice is applied: a comment block at the Pass-1 sentinel
reset documenting that `0` is deliberately overloaded as both "uncached" and a
possible genuine surface value, that the resulting recompute is benign (same
value), and why a distinguishing sentinel was rejected; plus a one-line pointer
at each of the two Pass-2 cache-check sites. No behaviour change.

Evidence: `algo.rs:955` (first-touch `wf[e] = 0` sentinel), `:983-988` /
`:1015-1018` (the `if wf[e] == 0` recompute sites), `:664-666`
(`amf_wf_surface`, zero at `dext == 2*deg-1`). `cargo test -p
feral-ordering-core` green (comment-only, proves no behaviour change). Journal:
dev/journal/2026-06-10-01.org.

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
tests/column_renumbering_parity.rs
tests/column_renumbering.rs
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
tests/issue_15_cascade_arm_gate.rs
tests/issue_17_robot_1600_cascade_off.rs
tests/issue_18_narx_cfy_cascade_off.rs
tests/issue_2_kkt_ls_init.rs
tests/issue_38_static_pivot.rs
tests/issue_46_saddle_kkt_cascade.rs
tests/issue_55_delay_budget.rs
tests/issue_55_n_tiny_counter.rs
tests/issue52_stats.rs
tests/issue64_arrow_ordering.rs
tests/issue65_mc64_fallback.rs
tests/issue67_thin_ordering.rs
tests/kkt_hardening.rs
tests/kkt_matrices.rs
tests/large_matrix_smoke.rs
tests/ldlt_compress.rs
tests/lu_dense.rs
tests/lu_scaling.rs
tests/lu_sparse.rs
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
tests/rook_rescue_kkt.rs
tests/rook_rescue.rs
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
