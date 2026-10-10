# FERAL Context (auto-generated)

Generated: 2026-10-10T19:59:52Z

## Latest Session
File: dev/sessions/2026-10-10-01.md
```
# Session 2026-10-10-01

## Benchmark note (read first): factor-ratio p90 is worse, and the cause is machine load

`cargo run --bin bench --release` on the full local KKT corpus gave the same
correctness lines as 2026-10-09-01: dense inertia 154429/154482, residual
154149/154482; sparse inertia vs MUMPS 154531/154590, residual 154258/154590;
48 both-path inertia mismatches; 78 sparse-only and 33 dense-only failures.
The factor-ratio p90 vs MUMPS is **worse**:

| bucket | 2026-10-09-01 | this session |
|---|---|---|
| dense small (<200) | 1.61 | 1.75 |
| dense medium (<500) | 2.00 | 2.18 |
| sparse small (<200) | 1.59 | 1.91 |
| sparse medium (<500) | 1.59 | 1.91 |

The bench calls `symbolic_factorize` + `factorize_multifrontal` directly
(src/bin/bench.rs:1924-1942) and never constructs a `Solver`, so this
session's change (all inside `Solver::factor`) cannot reach it. During the
run the load average was 80 on 14 cores: two unrelated python jobs at ~83%
CPU each, plus mds_stores and WindowServer. I attribute the shift to load
but did not confirm it with a quiet-machine rerun. The next session should
rerun the bench on an idle machine before comparing.

## Goal

Issue #215: pounce#999 `column25_integral` still stalls after #214. A
`Solver` reused across IPM iterates 0-4 reported neg 75597 at iterate 4 on
the reporter's machine; a fresh `Solver` reports 75596. The user chose the
fix "auto re-analyse on cascade-break".

## Accomplished

- **Reproduction (journal §10:14).** Rebuilt the iterates with pounce
  43cbb488 patched to this feral. The matrices match the issue (same min|d|
  per iterate). The +1 does **not** reproduce on this machine: warm and
  fresh both give 75596. The cached symbolic is the state that matters.
  `invalidate_symbolic_cache()` before it004 reproduces the fresh factor
  exactly; the MC64 cache never hits. The analysis is value-dependent under
  escalation-forced `LdltCompress` (25024 vs 24684 supernodes on the same
  pattern). CB counts here (warm 1, fresh 4) differ from the reporter's
  (6, 2).
- **Auto re-analysis (c8048bf).** Gate: reused analysis, `resolved_preprocess
  == LdltCompress`, `n_cascade_break_nodes > 0`, miss latch < 2. Re-run the
  symbolic, refactor, and keep the new factor if `(cb, tfa + tiny)` is
  lexicographically smaller. Opt-out:
  `with_auto_reanalyse_on_cb(false)`. Counters: `auto_reanalyse_count`,
  `auto_reanalyse_adopted_count`.
- **Stats fix (e7f4675).** The escalation re-entry reported
```

## Git Status
```
f825984 docs(journal): #215 growth-cap Mittelmann sweep result
e8b6153 fix: delay budget bounds front growth, not supernode width (#215)
b3a9b8f Merge pull request #216 from jkitchin/fix/215-cb-reused-analysis
7b7522f docs: session 2026-10-10-01 addendum (re-analysis withdrawn)
7fcb100 revert(solver): remove auto re-analysis on cascade-break (#215)
```

## Test Status
```
test symbolic::tests::symbolic_factorize_scotch_produces_valid_perm ... ok
test symbolic::tests::test_contrib_sizes_nonnegative ... ok
test symbolic::tests::test_perm_inverse_consistency ... ok
test symbolic::tests::test_symbolic_factorize_basic ... ok
test symbolic::tests::choose_adaptive_rules ... ok
test symbolic::tests::choose_adaptive_routes_arrow_to_amf ... ok
test symbolic::tests::test_symbolic_factorize_dense ... ok
test symbolic::tests::test_symbolic_factorize_kkt ... ok
test scaling::tests::auto_keeps_mc64_on_vesuvia_0000 ... ok
test scaling::tests::auto_keeps_mc64_on_vesuviou_0000 ... ok
test symbolic::tests::issue_3_scotchnd_on_kkt_recurses_after_o13 ... ok
test numeric::factorize::tests::issue_5_mss1_zero_tol_sweep_diagnostic ... ok
test symbolic::tests::issue_3_auto_on_kkt_routes_via_pick_default_method ... ok
test numeric::factorize::tests::issue_5_mss1_pivot_threshold_sweep_diagnostic ... ok
test scaling::tests::pick_scaling_strategy_routes_clnlbeam_to_infnorm ... ok
test scaling::hungarian::tests::mc64_hungarian_no_quadratic_heap_realloc_regression ... ok
test numeric::solve::tests::cb_core_profitable_matches_the_plan_gate ... ok

test result: ok. 458 passed; 0 failed; 7 ignored; 0 measured; 0 filtered out; finished in 1.88s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-10-10-01.md)

KKT summary: 156929 matrices (154482 dense-eligible n <= 1000, 108 skipped n > 1000, 2339 parse-skipped)
  Inertia match: 154429/154482 (100.0%)
  Residual pass: 154149/154482 (99.8%)
  Parse-skipped: 2339
  Worst residual: 2.46e-1 (POLAK6_0021)


--- Dense Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
bucket                    count      p90     target  verdict
small-frontal (<200)     147982     1.75     <= 2.0     PASS
medium (<500)            152145     2.18     <= 3.0     PASS

--- Sparse Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
bucket                    count      p90     target  verdict
small-frontal (<200)     153455     1.91     <= 2.0     PASS
medium (<500)            153560     1.91     <= 3.0     PASS

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
  derivative matrix (same δx = 3.33e-3 on all 75,796 primal diagonals).
  The nearest eigenvalue is +3.47e-8, which feral counts as negative. The
  reused analysis gives cb 2, n_tiny 19, tfa 32 and a +1 count. A
  **fresh** `Solver` is also +1, with cb 5, n_tiny 63 and tfa 116. Both
  candidates are wrong, so choosing between them cannot help.
- Regression: on the five-matrix sequence in the issue body, iterate 1
  became +1 (correct on fdef8b1). Re-analysis ran 4 times and was adopted
  2 times. The comparator's criterion does not track inertia correctness.
- Earlier, on the developer's machine, the replay was correct throughout,
  and both re-analyses were declined. That run had no wrong count to fix
  and so could not test the rule.

**Why it fails.** The wrong counts come from factors with cascade-break
nodes, whether the analysis is stale or fresh. Fewer CB nodes or fewer
floored pivots is a proxy for "less perturbation". It does not show
whether any floored pivot's sign was right.

**Do not re-propose** factor-comparison or re-analysis heuristics for
CB inertia. The direction left open is inertia-exact handling of
delay-budget overflow (delay instead of floor).

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
