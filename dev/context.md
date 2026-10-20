# FERAL Context (auto-generated)

Generated: 2026-10-20T00:33:12Z

## Latest Session
File: dev/sessions/2026-10-19-01.md
```
# Session 2026-10-19-01

## Benchmark note (read first): local corpus is the 2 synthetic fixtures

`cargo run --bin bench --release` found only the 2 generated
synthetic-regression fixtures, as in 2026-10-16-01. Both pass inertia and
residual, and no oracle timings are available, so no factor-ratio p90 can
be compared. The comparable evidence is the HEAD-vs-fix sweep over
`data/matrices/kkt-pounce` (370 matrices) below.

## Unfavourable results (read second)

- **brainpc1 and brainpc2 still time out (300 s) in pounce with the fix.**
  The factorization is no longer the cost. A mid-run `sample` puts almost
  all busy time in MC64 `hungarian_match_instrumented`, while the IPM keeps
  cycling through restoration (brainpc1 reached 1397 iterations).
- brainpc0 now solves (68.8 s, 1512 iterations, 709 restoration), but the
  issue reports about 1 s on pounce 0.12/feral 0.18. Baseline 0.20 timed
  out at 600 s.
- brainpc3 i38/1: the fix counts 6901 negative, eigvalsh gives 6900, and
  HEAD gave 6900. The matrix has 129 eigenvalues below n·eps·‖A‖, so it
  is numerically singular and falls under the singular-matrix exemption.
  It is still a data point against.

## Goal

Fix issue #221: brainpc3 KKT factor 0.038 s (0.18) → 9.4 s (0.19/0.20).

## Accomplished

- Root cause, from the reporter's own matrices dumped with pounce:
  - A zero-diagonal constraint column's BK argmax is a hub row (weight
    ~600, not fully summed), and `k+1` is uncoupled. So its matched state
    variable (diag ~1e-13), delayed into the same front from a split pair,
    was never tried as a 2×2 partner.
  - Rook walked the hub rows and failed.
  - About 2000 delays were carried through 428 fronts once #215's budget
    allowed it.
- Fix in `scalar_pivot_step`: fall back to the fully-summed argmax `r_fs`
  as the 2×2 partner, gated pre-swap by `rook::passes_2x2_gates`. This
  matches SSIDS `ldlt_tpp.cxx:207` (spral-expert confirmed).
- brainpc3, before → after:
  - i37: 9.08 s → 0.022 s, nnz(L) 12.88M → 402K
  - i35: 43.8 s with 277 floored pivots and cb 1 → 0.022 s, none
  - i38/i39: 2.9 / 2.4 s → 0.023 s
  - backward error improves on every one
- kkt-pounce corpus, 370 matrices:
  - inertia and n_tiny identical
  - nnz(L) −0.9 %/+0.03 % on 141
  - total time 69.0 s → 68.6 s
```

## Git Status
```
4c4fe67 fix(bk): try the fully-summed argmax as 2x2 partner (#221)
23624f1 Merge pull request #220 from jkitchin/release/0.20.0
7fc9da0 release: feral 0.20.0
352f7ab Merge pull request #219 from jkitchin/claude/admiring-lamport-bkdaki
410c71a dev: session checkpoint addendum for output recycling (#153)
```

## Test Status
```
test symbolic::tests::symbolic_factorize_scotch_produces_valid_perm ... ok
test symbolic::tests::test_contrib_sizes_nonnegative ... ok
test symbolic::tests::symbolic_factorize_metis_produces_valid_perm ... ok
test symbolic::tests::test_perm_inverse_consistency ... ok
test symbolic::tests::test_symbolic_factorize_dense ... ok
test symbolic::tests::test_symbolic_factorize_basic ... ok
test symbolic::tests::test_symbolic_factorize_kkt ... ok
test symbolic::tests::is_arrow_bordered_rejects_low_nnz_share_border ... ok
test symbolic::tests::is_arrow_bordered_rejects_many_hubs ... ok
test symbolic::tests::choose_adaptive_routes_arrow_to_amf ... ok
test numeric::solve::tests::cb_coarsening_threshold_is_arithmetically_inert ... ok
test symbolic::tests::issue_3_scotchnd_on_kkt_recurses_after_o13 ... ok
test symbolic::tests::choose_adaptive_rules ... ok
test numeric::solver::tests::warm_refactor_recycles_output_buffers_issue_153 ... ok
test symbolic::tests::issue_3_auto_on_kkt_routes_via_pick_default_method ... ok
test numeric::solve::tests::cb_core_profitable_matches_the_plan_gate ... ok
test scaling::hungarian::tests::mc64_hungarian_no_quadratic_heap_realloc_regression ... ok

test result: ok. 459 passed; 0 failed; 7 ignored; 0 measured; 0 filtered out; finished in 3.09s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-10-19-01.md)

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

## Recent Decisions

## 2026-10-19 — 2×2 partner falls back to the fully-summed argmax (issue #221)

**Decision.** In `scalar_pivot_step`, the 2×2 partner candidates are now
tried in this order:
1. BK's argmax `r`, if fully summed;
2. `k+1`, if coupled (#46);
3. the largest entry among the fully-summed rows (`r_fs`), only if
   `{k, r_fs}` passes the Duff-Reid growth bound and the SSIDS det floor,
   checked before any swap.

A rejected candidate leaves the old path unchanged.

**Why.** A ghost-row argmax (a hub coupling) plus an uncoupled `k+1`
made a zero-diagonal constraint delay even when its matched partner was
fully summed in the same front. On brainpc3 this carried about 2000
delays through 428 fronts once #215's budget allowed it: 9.4 s against
0.02 s. SSIDS's TPP kernel restricts the partner search to fully-summed
columns (`ldlt_tpp.cxx:207`), so this aligns feral with SSIDS.

**Cost.** None measured. The kkt-pounce corpus has identical inertia on
370/370 and nnz(L) within −0.9 %/+0.03 % on the 141 that changed.

**Alternatives rejected.**
- Cheaper rook scans: rook is about half the time, and the Schur work on
  the delay-swollen fronts remains.
- Tightening the #215 delay budget back toward 0.18: reintroduces
  cascade-break floored pivots (brainpc3 i35 HEAD: 277 floored, cb 1).

Research: `dev/research/issue-221-fully-summed-2x2-partner.md`.

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
tests/column_renumbering_parity.rs
tests/column_renumbering.rs
tests/d4_solve_2x2_gate.rs
tests/d6_contrib_uninit.rs
tests/d7_block32_dispatch_pooled.rs
tests/delayed_pivoting.rs
tests/dense_fast_path.rs
tests/dense_ldlt.rs
tests/env_knob_parsing.rs
tests/env_knob_scan.rs
tests/factor_output_recycling.rs
tests/factor_scratch_parity.rs
tests/factor_workspace_parity.rs
tests/factors_ld_export.rs
tests/fine_grained_delay.rs
tests/fma_opt_in_roundtrip.rs
tests/golden_bits.rs
tests/growth_flag.rs
tests/issue_15_cascade_arm_gate.rs
tests/issue_17_robot_1600_cascade_off.rs
tests/issue_18_narx_cfy_cascade_off.rs
tests/issue_2_kkt_ls_init.rs
tests/issue_215_delay_growth_cap.rs
tests/issue_221_fully_summed_2x2_partner.rs
tests/issue_38_static_pivot.rs
tests/issue_46_saddle_kkt_cascade.rs
tests/issue_55_delay_budget.rs
tests/issue_55_n_tiny_counter.rs
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
tests/kkt_hardening.rs
tests/kkt_matrices.rs
tests/large_matrix_smoke.rs
tests/ldlt_compress.rs
tests/lu_adversarial_inputs.rs
tests/lu_default_ordering.rs
tests/lu_dense_bump.rs
tests/lu_dense_update_bg.rs
tests/lu_dense.rs
tests/lu_ft_widebump.rs
tests/lu_hyper_sparse.rs
tests/lu_markowitz.rs
tests/lu_real_bases.rs
tests/lu_scaling.rs
tests/lu_sparse_rhs.rs
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
tests/pounce710_refine_cap_nrhs2.rs
tests/profiler_smoke.rs
tests/property_tests.rs
tests/refined_solve_core_stability.rs
tests/rook_rescue_kkt.rs
tests/rook_rescue.rs
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
