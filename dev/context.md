# FERAL Context (auto-generated)

Generated: 2026-10-16T20:14:05Z

## Latest Session
File: dev/sessions/2026-10-16-01.md
```
# Session 2026-10-16-01

## Benchmark note (read first): not comparable to 2026-10-10-01

This ran in a cloud container (x86_64 KVM guest) without the local KKT
corpus. `cargo run --bin bench --release` found only the 2 generated
synthetic-regression fixtures. Both pass inertia and residual, and no
oracle timings are available. No factor-ratio p90 could be computed, so
nothing here can be compared with the previous session. No solver source
changed this session.

## Goal

Review the #153 comment 6223093143 (pounce#1020: same instruction stream,
about 2.6x lower IPC on x86 cloud hosts). Run the measurements it asks
for on x86 Linux, then post the review.

## Accomplished

- `crates/feral-diagnostics/src/bin/diag_153_chain_x86.rs`: a synthetic
  hanging-chain KKT close to the reporter's (11875 vs 11874 supernodes,
  max front 19 vs 20). It has modes wall / profile / alloc / bump / once,
  plus a mimalloc arm behind a local cfg.
- Callgrind: 192.0M Ir per warm refactor, compared with the reporter's
  215.7M (x86) and 191.2M (aarch64). Same profile shape: infnorm 24.6%,
  glibc malloc 18-19%, memset/memcpy 8.4%.
- The scaling route on warm refactors is plain InfNorm. The pre-MC64 trial
  (`scaling/mod.rs:458`) does not fire.
- Allocation is the dominant x86 lever on this matrix:
  - mimalloc runs 1.7-1.8x faster than glibc (10/10 alternating rounds,
    identical inertia and nnz_l);
  - glibc `tcache_count=65535` gives about 1.3x;
  - a bump arena is 1.04-1.085x *slower* (no reuse, so cache-cold).
- 93,138 allocations (16.2 MB) per warm refactor, 7.84 per supernode,
  almost all of them per-front outputs re-allocated every refactor.
  Sites are listed in the journal.
- Details and every number: `dev/journal/2026-10-16-01.org`.

## Benchmark Results
```
KKT summary: 2 matrices (1 dense-eligible n <= 1000, 1 skipped n > 1000, 0 parse-skipped)
  Inertia match: 1/1 (100.0%)
  Residual pass: 1/1 (100.0%)
  Worst residual: 1.14e-15 (densecol_kkt_300_0000)

--- Sparse solver validation ---
Sparse solver: 2/2 total
  Inertia match vs MUMPS: 2/2 (100.0%)
  Residual pass: 2/2 (100.0%)
  Worst residual: 1.26e-16 (densecol_kkt_300_0000)
```

## Git Status
```
b9c6e45 diag: add per-file corpus mode to diag_153_chain_x86
7c8a6dd dev: session 2026-10-16-01 checkpoint (#153 x86 allocator review)
6d55fab diag: add diag_153_chain_x86 hanging-chain warm-refactor probe
6fbea33 Merge pull request #218 from jkitchin/release/0.19.0
d78ddc3 release: feral 0.19.0
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
test symbolic::tests::issue_3_auto_on_kkt_routes_via_pick_default_method ... ok
test scaling::hungarian::tests::mc64_hungarian_no_quadratic_heap_realloc_regression ... ok
test numeric::solve::tests::cb_core_profitable_matches_the_plan_gate ... ok

test result: ok. 456 passed; 0 failed; 7 ignored; 0 measured; 0 filtered out; finished in 2.64s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-10-16-01.md)

KKT summary: 2 matrices (1 dense-eligible n <= 1000, 1 skipped n > 1000, 0 parse-skipped)
  Inertia match: 1/1 (100.0%)
  Residual pass: 1/1 (100.0%)
  Worst residual: 1.14e-15 (densecol_kkt_300_0000)

--- Sparse solver validation ---
Sparse solver: 2/2 total
  Inertia match vs MUMPS: 2/2 (100.0%)
  Residual pass: 2/2 (100.0%)
  Worst residual: 1.26e-16 (densecol_kkt_300_0000)

Dense failure analysis: no failures

Sparse failure analysis: no failures

--- Dense ∩ Sparse failure overlap ---
Failed in BOTH dense and sparse:  0
Failed in dense only:             0
Failed in sparse only:            0

--- Dense perf vs oracles: no matrices have oracle timings ---

--- Sparse perf vs oracles: no matrices have oracle timings ---

--- Dense Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
bucket                    count      p90     target  verdict
small-frontal (<200)          0        -     <= 2.0      N/A
medium (<500)                 0        -     <= 3.0      N/A

--- Sparse Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
bucket                    count      p90     target  verdict
small-frontal (<200)          0        -     <= 2.0      N/A
medium (<500)                 0        -     <= 3.0      N/A
EXIT 0

```

## Recent Decisions

## 2026-10-10 — The delay budget bounds front growth, not supernode width (issue #215)

**Decision.** `assign_delayed_capacities` sets
`delayed_capacity = min(subtree_ncol − own_ncol, max(3·nrow, 4096 − nrow))`,
replacing `min(worst, max(4·own_ncol, 16))` from issue #55. Cascade-break
stays as the over-budget path, now acting as a memory backstop. Add
`FactorStats::inertia_uncertain()` (= `n_tiny`) so callers can tell
when the inertia was computed from floored pivots.

**Why.** Under cascade-break, pivots are floored to `sign(d)·eps`, and
when `|d|` is round-off that sign is noise. The old cap scaled with
`own_ncol`, so thin IPM-KKT supernodes hit it at front sizes far below
any memory concern. column25 `i011_s002` under 9 inertia-preserving
congruences was wrong on 3 with the old cap and 0 without it. The
pinene/nql180 blowups that motivated the old cap no longer reach any
budget at HEAD (cb 0). Across the kkt-mittelmann corpus, the new cap
admits every delay observed under unbounded delay.

**Cost.** `qcqp1500-1nc_0000`: +52 % nnz(L), ~4.6× factor time. The
other changed matrices are within ±3 % or faster.

**Alternatives rejected.**
- Auto re-analysis; see the withdrawal entry above.
- Unbounded delay with no backstop: exact, but nothing stops a future
  dense-front blowup.
- Swap-to-boundary: deferred as follow-up, since it reduces the volume
  of delays rather than fixing their sign handling.

Research: `dev/research/issue-215-delay-budget-exact-inertia.md`.

## Recent Tried-and-Rejected
(`diag_153_chain_x86`, n = 40006, Amf + LdltCompress). Callgrind shows
`mc64::compute_matching` once (the cold factor only). The warm
`ScalingInfo` is `Applied` with `mc64_cache_hit_count() == 0`, so the warm
route is `pick_scaling_strategy -> InfNorm` (line 437), not the line-458
trial. `compute_infnorm`'s 24.6% of Ir is the ordinary 10-sweep InfNorm
cost, which is already covered by the 2026-08-09 / 2026-08-15 entries.

**2. "glibc trims or mmaps per refactor and page faults explain the x86
slowness."** Falsified. The glibc default gives 139 minor faults and 0.3 ms
sys per 19 ms refactor. Setting `GLIBC_TUNABLES` trim_threshold and
mmap_threshold to 4G cuts faults to 5 and is no faster (min 19.15 / 17.72
ms vs 18.26 / 17.75).

**Instrument defect, recorded so it is not reused:** a pointer-bump arena
with no-op free is NOT a lower bound on allocator cost. On this matrix it
was 1.04-1.085x *slower* than glibc (paired, 3 x 60), because it never
reuses memory: 16.2 MB of cache-cold lines per refactor. mimalloc (which
reuses memory) was 1.7-1.8x *faster* than glibc. The #200 bump-allocator
result ("0.6-7.7%, allocation ruled out") used the bump instrument and
does not rule allocation out. See dev/journal/2026-10-16-01.org.

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
src/env.rs
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
src/lu/markowitz.rs
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
tests/cb_core_choice_ignores_env.rs
tests/cb_solve_parity.rs
tests/column_renumbering.rs
tests/column_renumbering_parity.rs
tests/d4_solve_2x2_gate.rs
tests/d6_contrib_uninit.rs
tests/d7_block32_dispatch_pooled.rs
tests/delayed_pivoting.rs
tests/dense_fast_path.rs
tests/dense_ldlt.rs
tests/env_knob_parsing.rs
tests/env_knob_scan.rs
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
tests/issue177_parallel_entry_point_core.rs
tests/issue178_refine_cap.rs
tests/issue178_solve_into.rs
tests/issue203_ordering_race.rs
tests/issue204_work_estimate.rs
tests/issue205_ordering_info.rs
tests/issue52_stats.rs
tests/issue64_arrow_ordering.rs
tests/issue65_mc64_fallback.rs
tests/issue67_thin_ordering.rs
tests/issue91_preprocess_misfire.rs
tests/issue99_fma_front_gate.rs
tests/issue_15_cascade_arm_gate.rs
tests/issue_17_robot_1600_cascade_off.rs
tests/issue_18_narx_cfy_cascade_off.rs
tests/issue_215_delay_growth_cap.rs
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
tests/lu_default_ordering.rs
tests/lu_dense.rs
tests/lu_dense_bump.rs
tests/lu_dense_update_bg.rs
tests/lu_ft_widebump.rs
tests/lu_hyper_sparse.rs
tests/lu_markowitz.rs
tests/lu_real_bases.rs
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
tests/pounce710_refine_cap_nrhs2.rs
tests/pounce_interface.rs
tests/profiler_smoke.rs
tests/property_tests.rs
tests/refined_solve_core_stability.rs
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
