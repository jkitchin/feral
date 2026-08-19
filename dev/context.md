# FERAL Context (auto-generated)

Generated: 2026-08-19T13:38:18Z

## Latest Session
File: dev/sessions/2026-08-19-01.md
```
# Session 2026-08-19-01

## BENCHMARK NOT COMPARABLE THIS SESSION — corpus absent

Reported first, per the hard rule in CLAUDE.md, because the honest
statement is "not measured", not "unchanged".

`cargo run --bin bench --release` ran to completion, but
`data/benchmark-config.toml` is **not present in this container**, so the
harness fell back to its 8 synthetic matrices. Both Phase 2.8.1 exit
partitions report `count = 0`, verdict `N/A`:

    partition                 count   p90   target  verdict
    dense  small-frontal (<200)   0     -    <= 2.0    N/A
    dense  medium (<500)          0     -    <= 3.0    N/A
    sparse small-frontal (<200)   0     -    <= 2.0    N/A
    sparse medium (<500)          0     -    <= 3.0    N/A

There is therefore **no comparison against 2026-08-15-02's numbers**
(1.61 / 2.00 / 1.67 / 1.67). Not a regression and not an improvement —
not measured. What did run passed: 2/2 inertia match vs MUMPS, 2/2
residual pass, worst residual 1.26e-16 (densecol_kkt_300_0000).

The changes this session are additive API surface with `Default`-valued
wrappers proven bit-identical to the previous entry points, so the
factor-ratio partitions are not expected to move; but "not expected to
move" is a prediction, not a measurement, and it should be checked on a
machine that has the corpus.

## Goal

Fix [issue #178](https://github.com/jkitchin/feral/issues/178), filed
from [pounce#698](https://github.com/jkitchin/pounce/issues/698). Two
independent asks:

1. A caller-supplied cap on iterative-refinement correction steps. An
   interior-point host runs its own refinement loop over the same
   augmented system, so FERAL's inner 10-step budget is work whose
   residual nobody consults — measured at 60 % of back-solve time on a
   118 276-dimension KKT system.
2. In-place (`*_into`) solve entry points, so a host that owns its
   right-hand-side buffer stops paying an allocation plus copy-back per
   back-solve.

## Accomplished

Both, plus the research note and plan the lifecycle requires.

### Item 1 — `RefineOptions`

```

## Git Status
```
22765f6 docs(changelog): record the issue #178 refinement cap and in-place solves
ae169f9 feat(solver): capped and in-place solve entry points on Solver
f23137b feat(solve): make the refinement step budget a per-call option
eeee52a docs: research note and plan for a caller-capped refiner (issue #178)
6fb9d26 Merge pull request #174 from jkitchin/feat/scaling-router-invariance
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

test result: ok. 425 passed; 0 failed; 6 ignored; 0 measured; 0 filtered out; finished in 1.54s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-08-19-01.md)


FERAL benchmark harness
  ordering: default (symbolic_factorize heuristic)
  scaling: default (SupernodeParams::default)
Loading matrices from data/benchmark-config.toml ... not found

name                n   factor(μs)    solve(μs)        inertia
--------------------------------------------------------------
spd_10             10           21            1     (10, 0, 0)
spd_50             50           45            2     (50, 0, 0)
spd_100           100          220            5    (100, 0, 0)
spd_200           200          926           41    (200, 0, 0)
kkt_10_3           13            5            0     (10, 3, 0)
kkt_30_10          40           31            1    (30, 10, 0)
kkt_50_15          65           89            2    (50, 15, 0)
kkt_100_30        130          339            7   (100, 30, 0)

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
measurement above: pounce defaults `feral_refine` to *on* for a
documented case (`pinene_3200`) whose IPM tail stalls when the residual
floor left by cascade-break's L-factor perturbation goes uncorrected.
Zero steps loses that case. So the problem was never that 10 is too
many — it is that 10 and 0 were the only two values expressible. The
value that plausibly serves both is 1, and nobody could ask for it.

Two semantics follow from "cap, not target", and both are tested rather
than merely documented. The existing exits — the `ε·√n` relative
residual, the 100× divergence guard, the 2-strike plateau — keep
priority, so raising `max_steps` can never add work to a system that has
already converged. And the best-iterate contract holds under every value,
so no cap can return an answer worse than `solve_sparse`'s.

`max_steps = 0` returns before the residual matvec rather than after. The
answer would be identical either way; the cost would not, and a caller
opting out of refinement paying a `symv` and a `norm2` per solve would
address only half of what was reported.

Measured on this branch (20 reps, release, single trial): on
VESUVIO_0021, a pounce KKT that uses 4 corrections, the refined solve is
7.31× the bare solve at the default and 3.07× at `k = 1`; `k = 0` is
1.03×, i.e. inside the bare solve's own noise. The 2.4× per-call
reduction is the same direction and rough magnitude as pounce#698's
independently measured 2.6× per-iteration back-solve reduction.

Evidence: issue #178; pounce#698 Observation 5;
`dev/research/refinement-cap-2026-08-19.md`;
`dev/plans/issue-178-refine-cap-and-inplace.md`;
`dev/journal/2026-08-19-01.org`; `tests/issue178_refine_cap.rs`.

## Recent Tried-and-Rejected
**Refuted by measurement.** `diag_symbolic_stages_argv` on
KIRBY2_0007:

    TOTAL 1182 us
      ldlt_compress   972   82.2%
      renumber         57    4.8%
      ordering         32    2.7%

Ordering is 32 us — 2.7% of symbolic and ~3% of the reported
`factor_us`. Eliminating AMD cost entirely could not move the ratio.
The cost is `ldlt_compress` (the MC64 matching feeding Duff-Pralet
compression), which is a different subsystem from the one the
hypothesis named.

A second prediction in the same hypothesis — that feral was producing
more fill than MUMPS — is also refuted: the numeric driver is 127 us
and `num_c ~ num_n` (149 vs 143 us), so the factorization is not the
problem in either time or fill.

Superseded by `dev/research/ldlt-compress-cost-benefit-2026-08-15.md`.

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
