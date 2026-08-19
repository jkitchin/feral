# FERAL Context (auto-generated)

Generated: 2026-08-19T19:56:19Z

## Latest Session
File: dev/sessions/2026-08-19-03.md
```
# Session 2026-08-19-03

## BENCHMARK NUMBERS ARE NOT COMPARABLE TO LAST SESSION

Reported first, per the hard rule in CLAUDE.md. `cargo run --bin bench
--release` in this container found only the 8 synthetic matrices — the
external corpus and the MUMPS/SPRAL oracle timings are not mounted — so
both Phase 2.8.1 exit partitions report `N/A`, exactly as in
2026-08-19-02. **No comparison against 2026-08-15-02's 1.61 / 2.00 /
1.67 / 1.67 is possible from this run.** The numbers were not measured;
I am not claiming they held.

This is also the wrong benchmark for this change: nothing in it touches
factor or solve arithmetic. What the run does confirm is that the change
is inert where it should be — inertia 2/2 vs MUMPS, residual 2/2, worst
residual 1.26e-16 (`densecol_kkt_300_0000`), byte-for-byte the same
figures 2026-08-19-02 reported.

## Goal

Fix issue #176 — `FERAL_CB_THRESH` and `FERAL_PAR_TASK_MIN_FLOPS`
silently ignore unparseable values (e.g. `1e18`) instead of erroring.

## Accomplished

### The bug is one shape, copied eighteen times

Every numeric `FERAL_*` knob in the tree was read as

    std::env::var(NAME).ok().and_then(|v| v.parse().ok()).unwrap_or(DEFAULT)

`"1e18".parse::<u64>()` is `Err(InvalidDigit)`, `.ok()` discards it, and
`unwrap_or` puts the default back. The knob is set; the process behaves
as if it were unset; nothing is printed. The reporter's evidence:

    $ FERAL_PAR_TASK_MIN_FLOPS=1e18 pounce NARX_CFy.nl --no-sol max_iter=1
    task_plan: n_snodes=45736 n_tasks=21 seeds=11 cutoff=1000000 min_seeds=2

`cutoff=1000000` is `PAR_TASK_MIN_FLOPS`, the built-in default. Two perf
attributions in the sibling issue were taken from runs like this.

The sweep the issue asked for found the same shape at 18 sites (the
inventory is in `dev/research/env-knob-parsing-2026-08-19.md`), four of
them behind local `env_usize`/`env_f64` copies in `feral-diagnostics`.

### One parse policy, in one module

New `src/env.rs` (`feral::env`), `u64_var` / `usize_var` / `f64_var` plus
`_where` variants carrying a validity check. Each returns `Option<T>`,
so every call site keeps its own default expression — including
```

## Git Status
```
2b84177 docs: document the numeric FERAL_* knobs and their parse policy (#176)
12e3330 fix(knobs): route every numeric FERAL_* read through feral::env (#176)
647202a feat(env): one parse policy for the numeric FERAL_* knobs (#176)
45429f1 docs: research note and plan for the FERAL_* knob parse policy (#176)
ffb7599 Merge pull request #180 from jkitchin/claude/quirky-bardeen-c3ynyz
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
test numeric::solve::tests::cb_core_profitable_matches_the_plan_gate ... ok
test scaling::hungarian::tests::mc64_hungarian_no_quadratic_heap_realloc_regression ... ok

test result: ok. 437 passed; 0 failed; 6 ignored; 0 measured; 0 filtered out; finished in 2.93s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-08-19-03.md)


8 matrices benchmarked

KKT summary: 2 matrices (1 dense-eligible n <= 1000, 1 skipped n > 1000, 0 parse-skipped)
  Inertia match: 1/1 (100.0%)
  Residual pass: 1/1 (100.0%)
  Worst residual: 1.14e-15 (densecol_kkt_300_0000)

--- Sparse solver validation ---
Sparse solver: 2/2 total
  Inertia match vs MUMPS: 2/2 (100.0%)
  Residual pass: 2/2 (100.0%)
  Worst residual: 1.26e-16 (densecol_kkt_300_0000)

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

```

## Recent Decisions
   `feral_min_par_flops` help. The notation the docs teach must work.
   The integer parse is tried first, so `18446744073709551615` stays
   `u64::MAX` rather than round-tripping through 2^64.
2. **A refused value warns on stderr, once per `(name, value)`, then
   falls back.** Not an error: these knobs are read from inside a
   factorization whose `Result` is reserved for *numerical* failure, and
   turning an environment typo into a `FeralError` would hand pounce a
   numeric-looking failure for an environment problem. The warn-and-fall-
   back shape has precedent here — `FERAL_SCALING` has warned on an
   unrecognized value since the X5 follow-up.
3. **An above-range magnitude clamps to the type maximum** instead of
   falling back. `FERAL_CB_THRESH=1e30` means "no subtree can reach this
   cutoff"; falling back to the default there would be the reported bug
   again, with the operator's intent inverted rather than merely lost.
4. **Fractional input rounds half away from zero.** Truncation would
   make `FERAL_PAR_MIN_SEEDS=0.9` mean 0 — "always parallel" — the
   opposite of what the value asks for.

Boolean and enum knobs are out of scope: they match a literal vocabulary
rather than parsing a number.

A source-scan test (`tests/env_knob_parsing.rs`) fails the build if a new
`FERAL_*` read parses its own value locally, because the defect was one
shape copied to eighteen sites, not one site. Two diagnostics-only
comma-list knobs are exempted by name in that scan.

Consequence for the public API: `numeric::factorize::par_task_min_flops`
and `par_min_seeds` are `pub`, so a caller can confirm what value the
process resolved a knob to. #176 could not be diagnosed from outside the
process without that.

## Recent Tried-and-Rejected

**Rejected on measurement.** The predicate runs on every refined solve,
including the ones it rejects, and `CbTaskPlan::build` allocates three
`Vec<Vec<usize>>` of length `n_nodes` (`build_children`, `owned`,
`tr_children`). Cost of the verdict alone, versus the shared-vector
baseline it was supposed to preserve:

    chain_400     1.29x
    chain_2000    1.27x
    chain_20000   1.24x

That is the same 1.24-1.29x the design existed to avoid — the predicate
cost as much as the core it was declining. Replaced by a flat
`O(n_nodes)` computation (four `Vec`s of scalars, subtree costs folded
into parents using the postorder guarantee, no child lists), which
brings the rejected trees back to 1.00-1.03x.

The cost of that replacement is a second implementation of one gate.
`cb_core_profitable_matches_the_plan_gate` pins the two together across
six fixtures landing on both sides of the gate.

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
tests/issue178_refine_cap.rs
tests/issue178_solve_into.rs
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
