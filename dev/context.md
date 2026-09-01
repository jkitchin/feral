# FERAL Context (auto-generated)

Generated: 2026-09-01T16:00:54Z

## Latest Session
File: dev/sessions/2026-09-01-01.md
```
# Session 2026-09-01-01

## Benchmark note (read first)

**No benchmark was run this session, and the numbers below are not fresh.**
This session reverted the code tree to the `v0.17.0` baseline; the resulting
tree is byte-identical to the tag except for an 18-line clippy `allow`. The
figures quoted are therefore the ones recorded at the 0.17.0 release
(`dev/sessions/2026-08-19-05.md`) and are cited, not re-measured. Re-running
the corpus to confirm that unchanged code produces unchanged numbers would
have bought nothing. The next session that touches solver code must run the
bench and report against these.

```
--- Dense Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
bucket                    count      p90     target  verdict
small-frontal (<200)     147982     1.58     <= 2.0     PASS
medium (<500)            152145     2.00     <= 3.0     PASS

--- Sparse Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
bucket                    count      p90     target  verdict
small-frontal (<200)     153455     1.58     <= 2.0     PASS
medium (<500)            153560     1.58     <= 3.0     PASS
```

## Goal

Park the post-0.17.0 development on a branch and return `main` to the 0.17.0
release baseline, after review found that none of it is used by pounce or
discopt.

## Accomplished

- **Parked 42 commits.** `v0.17.0` (2fbf9b7) to 91ace05 was 42 commits / 79
  files / +12384 / -340. Pushed to `origin` as branch `park/post-0.17` and
  annotated tag `park/post-0.17-tip`, both verified at 91ace05 by
  `git ls-remote` *before* any rollback began.

- **Reverted the code tree to 0.17.0** on branch `revert/park-post-0.17`
  (commit bb18f0a). Undone: #190 (`RefineOptions` targets, `RefineOutcome`),
  #192 (`Solver::reset_quality`), #194 (cooperative `factor` cancellation),
  multi-RHS solve perf, and the componentwise-refinement default with its
  breaking `*_refined_into` return-type change.

- **Three carve-outs kept**, each for a stated reason: the clippy 1.98 fixes
  (CI runs `stable` = 1.98, local is 1.93 — reverting them turns CI red on
  every future PR); the CI coverage infrastructure; and all of `dev/`, whose
  append-only logs the protocol forbids rewinding.

- **Evidence.** `cargo check --workspace --all-targets` clean.
```

## Git Status
```
bb18f0a revert: park post-0.17.0 development, resume from the 0.17.0 baseline
91ace05 docs: session checkpoint 2026-08-30-02 (#191, #193, #195 reviewed and landed)
d0b3000 Merge pull request #195 from jkitchin/claude/issue-194-p6gri8
7b42414 fix(solve): an interrupt during the MC64 retry must not be swallowed
942d130 Merge origin/main into claude/issue-194-p6gri8
```

## Test Status
```
test symbolic::tests::symbolic_factorize_kahip_produces_valid_perm ... ok
test symbolic::tests::test_contrib_sizes_nonnegative ... ok
test symbolic::tests::test_symbolic_factorize_basic ... ok
test symbolic::tests::symbolic_factorize_scotch_produces_valid_perm ... ok
test symbolic::tests::test_symbolic_factorize_dense ... ok
test symbolic::tests::test_perm_inverse_consistency ... ok
test symbolic::tests::test_symbolic_factorize_kkt ... ok
test numeric::solve::tests::issue175_overhead_term_is_scheduling_only ... ok
test symbolic::tests::is_arrow_bordered_rejects_many_hubs ... ok
test numeric::solve::tests::issue175_wide_thin_tree_is_not_scheduled_in_parallel ... ok
test symbolic::tests::choose_adaptive_routes_arrow_to_amf ... ok
test symbolic::tests::choose_adaptive_rules ... ok
test numeric::solve::tests::cb_coarsening_threshold_is_arithmetically_inert ... ok
test symbolic::tests::issue_3_scotchnd_on_kkt_recurses_after_o13 ... ok
test symbolic::tests::issue_3_auto_on_kkt_routes_via_pick_default_method ... ok
test numeric::solve::tests::cb_core_profitable_matches_the_plan_gate ... ok
test scaling::hungarian::tests::mc64_hungarian_no_quadratic_heap_realloc_regression ... ok

test result: ok. 444 passed; 0 failed; 7 ignored; 0 measured; 0 filtered out; finished in 2.84s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; no session checkpoint with bench)
```

## Recent Decisions
needs a revert-of-revert, which is a known and local inconvenience; a force-push
would have silently invalidated every branch cut from `main` since 2026-08-19.

**Three carve-outs are deliberately not reverted.**

1. *The clippy 1.98 fixes* (9c5dfac, 216a755). CI resolves
   `dtolnay/rust-toolchain@stable` to 1.98, which introduced
   `needless_late_init` and `chunks_exact_to_as_chunks`; the local toolchain is
   1.93. Reverting them turns CI red on every future PR while the pre-commit
   hook still passes locally — reintroducing exactly the local/CI drift those
   commits fixed. The `unknown_lints` guard is part of this: without it the
   `allow` is itself a hard error under `-D warnings` on any pre-1.98
   toolchain.
2. *CI coverage infrastructure* — `.github/workflows/coverage.yml`,
   `codecov.yml`, the README badge. Infrastructure, not solver code.
3. *All of `dev/`.* The session checkpoints, journals, and the append-only
   `decisions.md` / `tried-and-rejected.md` entries are the record of what was
   tried and learned. Reverting an append-only log to drop entries would
   violate the protocol in `CLAUDE.md`, and the record is *more* valuable after
   a park, not less: it documents why the code is on a shelf.

**Evidence.** `cargo test --workspace`: 1191 passed, 0 failed, 25 ignored.
`git diff v0.17.0..HEAD -- src/ python/ tests/ Cargo.toml` reduces to the
18-line `schur_kernel.rs` clippy allow and nothing else.

**Standing implication for future work.** The parked features were built to a
high standard — researched, measured, documented — and still went unused. The
gap was not quality but demand: none originated in a request from pounce or
discopt. Feature work on the solver should start from a consumer-demonstrated
need, not from an improvement that is available to make.

## Recent Tried-and-Rejected
**Replaced with.** Two fix-independent observables:
`mc64_retry_attempt_count() == 1` proves factorization #1 returned `Ok(..)`
(the issue-#65 gate keys on `Ok`, so the flag was set after it completed),
and `delay < call_elapsed` proves it was set before the call returned.
Together they pin the flag inside the retry without referencing the status
being asserted.

## 2026-08-29 — busy-wait shell pollers while a benchmark is live

**Tried.** `until grep -q ...; do :; done` to wait on a background job.

**Symptom.** Pinned a core, pushed load to 15.71 alongside `cargo test --all`
and a live A/B benchmark, and contaminated the branch arm of round 2. Phase
2.8.1 p90s are ratios against **on-disk MUMPS oracle sidecars**, so
contention inflates only feral's numerator — the metric is not
contention-symmetric and a loaded host reads as a regression. A single-shot
run under that load FAILED; the interleaved re-run was 8/8 PASS.

**Rule.** Any background waiter in this repo must `sleep`, never busy-poll.
Do not trust a single-shot p90 taken under load; interleave A/B.

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
