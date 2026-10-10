# FERAL Context (auto-generated)

Generated: 2026-10-10T14:54:29Z

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
c8048bf feat(solver): re-analyse when a reused LdltCompress analysis hits cascade-break
e7f4675 fix(solver): report pattern_reused=false on an escalated first factor
fdef8b1 Merge pull request #214 from jkitchin/claude/modest-dirac-fzfm5b
9632f30 docs: session checkpoint 2026-10-09-01 (PR #214 review fixes)
a90e63d docs(changelog): mark #213 struct fields breaking, record verified numbers
```

## Test Status
```
test symbolic::tests::test_perm_inverse_consistency ... ok
test symbolic::tests::test_symbolic_factorize_basic ... ok
test symbolic::tests::test_symbolic_factorize_dense ... ok
test symbolic::tests::test_symbolic_factorize_kkt ... ok
test symbolic::tests::choose_adaptive_routes_arrow_to_amf ... ok
test numeric::factorize::tests::issue_5_mss1_iter0_inertia_wanders_under_delta_w_sweep ... ok
test symbolic::tests::issue_3_scotchnd_on_kkt_recurses_after_o13 ... ok
test symbolic::tests::choose_adaptive_rules ... ok
test scaling::tests::auto_keeps_mc64_on_vesuvia_0000 ... ok
test numeric::solve::tests::cb_coarsening_threshold_is_arithmetically_inert ... ok
test scaling::tests::auto_keeps_mc64_on_vesuviou_0000 ... ok
test numeric::factorize::tests::issue_5_mss1_zero_tol_sweep_diagnostic ... ok
test symbolic::tests::issue_3_auto_on_kkt_routes_via_pick_default_method ... ok
test numeric::factorize::tests::issue_5_mss1_pivot_threshold_sweep_diagnostic ... ok
test scaling::tests::pick_scaling_strategy_routes_clnlbeam_to_infnorm ... ok
test numeric::solve::tests::cb_core_profitable_matches_the_plan_gate ... ok
test scaling::hungarian::tests::mc64_hungarian_no_quadratic_heap_realloc_regression ... ok

test result: ok. 461 passed; 0 failed; 7 ignored; 0 measured; 0 filtered out; finished in 12.10s

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
after 2 consecutive non-adopted re-analyses (reset on adoption and on
pattern change). On by default, opt out with
`with_auto_reanalyse_on_cb(false)`. The user chose this option over
exact-inertia swap-to-boundary (multi-session) and a caller-driven
re-analysis.

**Why.** The `LdltCompress` analysis is value-dependent (its MC64
matching picks the 2×2 pairs), so on a drifting IPM sequence it goes
stale. On pounce#999 `column25_integral`, iterate 4 on a reused `Solver`
gave neg 75597 (true 75596) on the reporter's machine. Here the reused and
fresh analyses differ (25024 vs 24684 supernodes), and resetting the
symbolic cache alone reproduces the fresh factor exactly. Neither analysis
is better on every machine: here warm has cb 1 and fresh cb 4, while the
reporter saw warm cb 6 and fresh cb 2. So the rule compares the two factors
instead of always preferring one. A miss is "not adopted", not "CB remains",
because in the reporter's run the fresh analysis also has CB. Counting that
as a miss would have latched before iterate 4.

**Limits.** This is a mitigation. A factor that still engages
cascade-break can have a wrong inertia. A pattern-only analysis
(preprocess `None`) is never re-analysed because it would come back
identical. Cost per re-analysis is one `LdltCompress` symbolic (including
MC64) plus one factor, about 6 s on column25 (n=151392) against a 0.33 s
warm factor.

**Evidence.** Unit tests `issue_215_*` in `solver.rs` (Weyl-inertia
fixture, with a stale analysis simulated by zeroing `delayed_capacity`).
seq215 replay in `dev/journal/2026-10-10-01.org` §11:55. The
`pattern_reused` re-entry fix is pinned in
`tests/issue102_ordering_escalation.rs`, which fails with the fix reverted.

## Recent Tried-and-Rejected

**Why they fail, as far as it was chased.** `dtoc_wide`'s `max_front` is 31.
At that width the numeric phase is all per-front overhead and no arithmetic,
so the comparison measures supernode count rather than flops, and MetisND's
fewer-but-wider fronts win for reasons that have nothing to do with dtoc2.
Widening the chain from `(nx,nu) = (4,2)` to `(8,4)` moved `max_front` only
24 -> 31; the real dtoc2's avg_deg of 17.5 is not reachable by this
construction at the right `n`.

**This was already on the record.**
`external_benchmarks/chain_proxy/README.md` documents geometry-matched
proxies producing the wrong answer and says to prefer the real corpus. That
warning was read during this session and the proxies were built anyway.

**What survives.** Nothing about routing. The non-proxy measurements in the
same session stand on their own and are recorded in
`dev/research/issue-203-auto-routing-2026-09-17.md`: on the issue #203
patterns MetisND is 1.62x faster sequentially and 3.5x in parallel, and
`nnz_L`, `flop_proxy` and `max_front` were each checked as routing guards
and each mispredicts.

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
