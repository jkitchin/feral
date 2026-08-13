# FERAL Context (auto-generated)

Generated: 2026-08-13T17:53:15Z

## Latest Session
File: dev/sessions/2026-08-13-02.md
```
# Session 2026-08-13-02

Journal: `dev/journal/2026-08-13-02.org`
Research note: `dev/research/hyper-sparse-solves-2026-08-13.md`

## Goal

Issue #161 part B — "triangular solves are not work-proportional". Filed as
*described, not implemented*: the issue's numbers size the prize but do not
demonstrate a fix. Part A of the same issue is already implemented and measured
in open PR #160 and was deliberately not touched here.

## Benchmark comparison to previous session — NOT AVAILABLE

Reported first, per protocol. `cargo run --bin bench --release` runs to
completion and exits 0, but **this container has no corpus**: every partition
reports `N/A` and "no matrices have oracle timings". This is the same situation
as session 2026-08-10-01, and it means **no LDLᵀ regression comparison against
2026-08-09-09 was possible.**

The change is confined to `src/lu/`, a separate factorization family that the
LDLᵀ corpus does not exercise, so the expected corpus effect is nil. That is an
argument for low risk, not evidence of no regression, and the next session with
a corpus should re-run before assuming otherwise. The open item from
2026-08-10-02 — re-bench `origin/main` alone to separate main's `Supernode.nrow`
change from noise — is still open and still blocked on the same missing corpus.

## Accomplished

### The gather/scatter split, which is the whole of issue #161B

Code inspection of the four sparse triangular kernels explains the issue's
headline number exactly. Two are **scatter** form (`L y = s`, `Uᵀ z = s`) and
already test for zero before doing work; two are **gather** form (`U w = s`,
`Lᵀ v = s`) and read every row of `U` / every column of `L` regardless. Issue
#161 measured a one-nonzero solution costing **0.74x** a fully dense one — half
the solve going to ~0 and half staying at full cost is what predicts a number
between 0.5x and 1.0x rather than 1.0x.

So the fix is specifically to the two gather kernels, and the prediction is that
it moves the sparse-rhs case and leaves the dense-rhs case alone.

### What landed

- `usolve` and `lt_solve` now compute the reach of the right-hand side's pattern
  in the factor's DAG and sweep only those positions, behind a density cap.
  `usolve` walks `u_above`, which the Forrest–Tomlin update already builds and
  maintains — nothing new was needed. `lt_solve` needed a row-wise index of `L`,
  built at factor time and valid for the life of the factor because the FT
  update never touches the base `L`.
```

## Git Status
```
bd8a0f7 docs(lu): sparse_hyper module doc named a test that does not exist
6682f48 fix(lu,python): clippy 1.97 lint and scipy index-order assumption
c546165 fix(bench,docs): sample stddev in basis_refactor; swapped LU doc comments
d3f110b Merge origin/main (PR #160, issue #161 part A) into the part-B branch
10476c0 docs: renumber this session to 2026-08-13-02
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

test result: ok. 420 passed; 0 failed; 6 ignored; 0 measured; 0 filtered out; finished in 2.08s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; no session checkpoint with bench)
```

## Recent Decisions
**Decision: the guard is evaluated on the rows the solution depends on.**
`ut_solve` skips rows where `s[i] == 0.0` (unconditionally, matching what
`lsolve` has always done); `usolve` on the reach-limited route skips rows
outside the reach. The dense fallback route keeps the full every-row check.

**Why this is sound and not merely cheaper.** A row `k` that is skipped has
`s[k] == 0` and no reached predecessor, so back substitution would assign it
`0 / U[k,k]`. If `U[k,k]` is healthy that is `0`, which is what leaving the
position alone already gives. If `U[k,k]` is zero the row states `0 = 0`: the
system is consistent and underdetermined there, and `0` remains *a* correct
solution component. No solve returns a different or wrong answer.

**What is genuinely given up.** The *diagnostic* that the factor is degenerate
somewhere the caller's right-hand side never touched. That was always incidental
— a solve is not a factor validity check — and the primary detection remains
where it belongs: at factor and update time, against the pivot tolerance, where
singularity is decided rather than stumbled over. A caller who wants the strict
old behavior sets `hyper_sparse_max_density = 0.0`, which restores the previous
solve exactly.

This is recorded as a decision rather than an implementation detail because it
narrows an always-on guard that an earlier repo review (L10) deliberately made
always-on. The narrowing is in *coverage per solve*, not in the guard itself:
a row that is reached is still checked in every build mode, and the two tests
that pin L10 (`zero_u_diagonal_errors_instead_of_inf`,
`misplaced_u_diagonal_errors_instead_of_silent_wrong_pivot`) still pass
unchanged, because a corrupted row that the solution depends on is still hit.

Full reasoning: `dev/research/hyper-sparse-solves-2026-08-13.md` § Semantics
that change.

## Recent Tried-and-Rejected
                     ftran mean   btran mean   dense-rhs fallback
  sparse marshal      33.6 us      31.6 us        0.90x
  dense marshal       31.3 us      33.0 us        0.93x
```

The two arms straddle each other (ftran favours dense marshalling, btran favours
sparse) — that is noise, not a signal, and the dense-rhs fallback is if anything
slightly worse with it.

**Why the hypothesis was wrong.** The permuted access is random but it is random
*within a 32 KB buffer*, which sits in L1/L2 — so it was never paying the cache
misses the reasoning assumed. What the phase probe actually showed is that the
residual cost is spread evenly across all ~6 `O(m)` linear passes
(`ftran_partial` alone, which is just the `P`-gather plus `lsolve`, was 10.3 of
the 22.1 us), at roughly 2-3 us per pass on this machine. There is no single
term left to remove: getting below the `O(m)` floor needs a **sparse-rhs entry
point**, not a cheaper way to walk a dense one.

The code was reverted. `dev/research/hyper-sparse-solves-2026-08-13.md` records
the floor and names the API change that would lift it.

## Source Files
```
src/bin/bench.rs
src/bin/perf_probe.rs
src/bin/probe_ft_eta.rs
src/bin/probe_lu_phases.rs
src/bin/probe_panel_frag.rs
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
src/lu/sparse_hyper.rs
src/lu/sparse_matrix.rs
src/lu/sparse_solve.rs
src/lu/sparse_symbolic.rs
src/lu/sparse_triangular.rs
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
tests/cb_solve_parity.rs
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
tests/golden_bits.rs
tests/growth_flag.rs
tests/issue102_intrafront_deadlock.rs
tests/issue102_ordering_escalation.rs
tests/issue107_external_ordering.rs
tests/issue112_bg_update.rs
tests/issue127_pipeline_split.rs
tests/issue128_supernode_nrow.rs
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
tests/lu_adversarial_inputs.rs
tests/lu_dense.rs
tests/lu_dense_bump.rs
tests/lu_dense_update_bg.rs
tests/lu_ft_widebump.rs
tests/lu_hyper_sparse.rs
tests/lu_scaling.rs
tests/lu_sparse.rs
tests/lu_sparse_rhs.rs
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
tests/static_assembly_maps.rs
tests/stress_tests.rs
tests/symbolic_profiler.rs
tests/task_plan_parity.rs
tests/threshold_consistency.rs
tests/tiny_fast_path.rs
```
