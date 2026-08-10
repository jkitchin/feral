# FERAL Context (auto-generated)

Generated: 2026-08-10T12:00:56Z

## Latest Session
File: dev/sessions/2026-08-10-02.md
```
# Session 2026-08-10-02

Journal: `dev/journal/2026-08-09-08.org` (the work began late on 2026-08-09;
the file number follows the journal, the session number follows
`dev/sessions/` which main had already advanced to `2026-08-10-01`).

## Goal

Take the MC64 Hungarian search bound — the lever recommended at the close of
the condition-1 dig-in (`dev/research/mc64-condition1-cost-share-2026-08-09.md`).
Then resolve the merge conflicts blocking PR #157 and get remote CI green.

## Benchmark comparison to previous session — TAIL IS SLIGHTLY WORSE

Reported first, per protocol. Against the last full-corpus bench
(`2026-08-09-09`; `2026-08-10-01` had no corpus in its container):

| factor/MUMPS | 2026-08-09-09 | this session | direction |
|---|---|---|---|
| geomean | 0.43 | **0.44** | worse |
| p50     | 0.30 | 0.30       | flat  |
| p90     | 1.54 | **1.57**   | worse |
| p99     | 3.30 | **3.45**   | worse |
| max     | 8.70 | **12.40**  | worse |

`solve/SSIDS` p90 also moved 2.44 → 2.50; `factor/SSIDS` geomean and both
`nnzL` rows are unchanged. All Phase 2.8.1 exit partitions still PASS.

I do not have an attributed cause, and I am not claiming one. What can be said:

- **Not the MC64 change.** `509f0ce` is verified bit-identical on 51 matrices,
  and it only touches the Hungarian heap, which does not run at all on the
  small CUTEst matrices that set these tails. The worst ratios are `KIRBY2`
  (n = 458) and `GROUPING` (n = 225).
- The merge brought in main's `Supernode.nrow` fix (`2026-08-10-01`), which is
  explicitly flagged there as a *behavior change to parallel dispatch*. That is
  the one plausible candidate in the diff, but it is unverified — nobody has
  re-benched main alone since it landed.
- Sub-millisecond matrices on a laptop are noisy; a single-matrix `max` moving
  8.70 → 12.40 on `KIRBY2_0007` (1476 us vs 1298-1142 us on its siblings) is
  within what this harness swings run to run.

Next session should re-run the bench on `origin/main` alone before spending
effort here, to separate main's change from noise.

## Accomplished

### The recommended lever does not exist (negative result)

My own prior note called the Hungarian search bound
```

## Git Status
```
5a9150d Merge origin/main into docs/session-2026-08-09-05
509f0ce perf(mc64): store the key inline in the Hungarian heap; bit-identical
fe8cc64 Merge pull request #158 from jkitchin/claude/fix-supernode-nrow
32d90ee docs: session checkpoint 2026-08-10-01 (Supernode.nrow fix)
fc84eb3 fix(symbolic): correct post-amalgamation Supernode.nrow (#128 item E)
```

## Test Status
```
test symbolic::tests::symbolic_factorize_scotch_produces_valid_perm ... ok
test symbolic::tests::test_perm_inverse_consistency ... ok
test symbolic::tests::test_symbolic_factorize_basic ... ok
test symbolic::tests::test_symbolic_factorize_dense ... ok
test symbolic::tests::test_symbolic_factorize_kkt ... ok
test symbolic::tests::is_arrow_bordered_rejects_many_hubs ... ok
test numeric::factorize::tests::issue_5_mss1_iter0_inertia_wanders_under_delta_w_sweep ... ok
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

test result: ok. 412 passed; 0 failed; 6 ignored; 0 measured; 0 filtered out; finished in 1.42s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-08-10-02.md)


=== Sparse perf vs canonical oracles (154588 matrices with oracle timings) ===

ratio               count    geomean        p50        p90        p99        max
factor/MUMPS       153560       0.44       0.30       1.57       3.45      12.40
solve/MUMPS        153560       0.07       0.08       0.14       0.66       2.78
factor/SSIDS       154500       0.04       0.03       0.32       1.01       2.49
solve/SSIDS        154500       0.93       1.00       2.50       8.33      30.25
nnzL/MUMPS         153560       0.61       0.58       0.75       4.50      23.11
nnzL/SSIDS         154500       0.88       1.00       1.00       4.50       5.00

Per-family factor geomean vs MUMPS (top 25 families by count):
family                  count    geomean        p50        max
HS118                    3000       0.91       0.94       1.60
BIGGSC4                  3000       0.42       0.45       0.60
MGH10LS                  3000       0.20       0.22       0.44
PALMER7A                 3000       0.28       0.30       0.70
ALLINITA                 3000       0.41       0.40       0.85
HS13                     3000       0.17       0.20       0.56
ALLINITC                 3000       0.19       0.20       0.30
HS89                     3000       0.20       0.20       0.30
HATFLDH                  3000       0.42       0.45       0.55
MCONCON                  3000       0.90       0.94       1.97
HS92                     3000       0.35       0.36       0.82
SSINE                    3000       0.27       0.27       0.33
HATFLDBNE                3000       0.39       0.40       0.83
HS90                     3000       0.20       0.20       0.33
DJTL                     3000       0.09       0.10       0.22
SSI                      3000       0.21       0.22       0.33
CONCON                   3000       0.86       0.89       1.72
HS91                     3000       0.25       0.27       0.40
PALMER5A                 3000       0.29       0.30       0.44
AVION2                   2682       1.48       1.52       2.11
CERI651ALS               2331       0.27       0.27       0.33
PFIT4                    2286       0.25       0.27       0.30
CERI651C                 2233       0.28       0.30       0.33
CERI651CLS               2227       0.27       0.27       0.40
BATCH                    2054       1.35       1.41       1.98

Top 10 worst factor-ratio vs MUMPS:
name                             n    feral(μs)    mumps(μs)      ratio
KIRBY2_0007                    458         1476          119      12.40
KIRBY2_0006                    458         1298          127      10.22
KIRBY2_0008                    458         1142          122       9.36
KIRBY2_0011                    458          932          120       7.77
KIRBY2_0009                    458          990          128       7.73
KIRBY2_0010                    458          987          133       7.42
GROUPING_0059                  225          725          116       6.25
GROUPING_0139                  225          701          113       6.20
GROUPING_0033                  225          692          112       6.18
GROUPING_0137                  225          674          111       6.07

--- Dense Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
bucket                    count      p90     target  verdict
small-frontal (<200)     147982     1.58     <= 2.0     PASS
medium (<500)            152145     2.00     <= 3.0     PASS

--- Sparse Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
bucket                    count      p90     target  verdict
small-frontal (<200)     153455     1.57     <= 2.0     PASS
medium (<500)            153560     1.57     <= 3.0     PASS

```

## Recent Decisions

    merged_nrow = child_group_ncol + parent_group_nrow

maintained as a running per-group value. This is the union *cardinality*, not
a bound, and it composes for chains under both amalgamation iteration orders.

Verified rather than assumed: compared against
`SymbolicFactorization::static_rows(i).len()` (the issue #125 static frontal
layout, an independent computation already pinned to both a from-scratch
`BTreeSet` recompute and `build_row_indices`) across 7 matrix families x 3
`nemin` values — **zero error on every supernode**. The pre-change proxy was
wrong on up to 40% of summed `nrow`.

**Accepted consequence, with a caveat.** `nrow` feeds
`estimate_assembly_flops`, so the `PAR_MIN_FLOPS` gate now sees true costs
and borderline matrices can flip from sequential to parallel (one flip
recorded: a 60x60 grid Laplacian at `nemin = 32`, 4.3M -> 12.2M estimated
flops). Numeric factors and inertia are byte-identical. The caveat is that
`PAR_MIN_FLOPS` was calibrated against the *understated* estimate, so the
constant itself may now be mis-placed; the flip is unverified on the real
corpus (absent from the container this landed in). Re-deriving the threshold
against corrected flops is open work, not something this change did.

The `merge_flop_budget` guard's merged-height model was corrected in lockstep
at both of its sites. It shared the understatement, which made merges look
cheaper than they are — the wrong direction for a guard meant to reject
expensive merges. The knob defaults to `None`, so the default path is
unaffected, but the sweep recorded in
`dev/research/amalgamation-cost-model-2026-08-09.md` was taken under the old
model and its numbers do not transfer.

## Recent Tried-and-Rejected
inline-key heap that followed is bit-identical and won 4-5% on nql180.

Full data: `dev/research/mc64-hungarian-search-bound-2026-08-09.md`.

## 2026-08-09 — `build_cost_graph` as an MC64 optimization target

**Rejected on measurement.** Timed at 8-12 ms per iterate. That is ~20% of
pinene's *cheapest* iterate but **0.4%** of nql180's — and nql180 is where the
MC64 time actually is. Optimizing it cannot move the corpus. The instrumented
timer was reverted and is not in any commit.

## 2026-08-09 — array fusion projected from a microbenchmark

**Not rejected, but the projection was wrong and is recorded so it is not
reused.** A standalone microbenchmark of split-array vs fused-record reads
predicted **1.68-1.9x**. The real end-to-end win from the inline-key heap was
**4-5%**, because the split reads are only ~2.3 ns of nql180's ~13 ns/scan.
Microbenchmarks of one memory access pattern do not predict a loop that also
does heap sifting and comparison work; scale by the measured share of the loop
before believing them.

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
tests/cb_solve_parity.rs
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
tests/lu_dense_update_bg.rs
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

(truncated from      354 lines to 350 line budget)
