# FERAL Context (auto-generated)

Generated: 2026-08-14T00:46:26Z

## Latest Session
File: dev/sessions/2026-08-13-04.md
```
# Session 2026-08-13-04

## Benchmark regression, reported first per protocol

The dense KKT p90 factor ratio vs MUMPS is **worse** than session 03:

| bucket | session 03 | this session |
|---|---|---|
| dense small-frontal (<200) | 1.57 | **1.61** |
| dense medium (<500) | 1.96 | **2.09** |
| sparse small-frontal (<200) | 1.55 | 1.54 |
| sparse medium (<500) | 1.55 | 1.54 |

Both buckets still PASS their targets. Nothing this session touched the LDL^T
KKT path — the changes are in the LU sparse-rhs solves and the LU symbolic — and
the sparse side is flat to slightly better, so this reads as run-to-run
variation: the run was in a different worktree (corpus symlinked in) and followed
a 90-minute discopt panel on the same machine. The tail moved the other way, and
by a lot more: session 03's worst factor ratio was CHWIRUT1_0216 at **76.18x**
and this run's is KIRBY2_0007 at **8.93x**, with CHWIRUT1 and CRESC entirely
absent from the top 10. That asymmetry is itself evidence the p90 delta is noise
rather than a change in the code. Not asserted as proven; a clean re-run on an
idle machine would settle it.

## Goal

Fix the two follow-ups the PR #162 review filed against feral: the missing
density guard on `ftran_sparse`/`btran_sparse` (issue #164) and the LU column
ordering not being a parameter (issue #165). The review's verdict on discopt
#1008 was that feral's remaining deficiency is the ordering cost, not the solves;
these are the two changes that close it out.

A cloud session had already responded to the review (12d1fcc, 8c2326f) but
explicitly deferred both of these — its commit body says "NOT DONE HERE".

## Accomplished

### Issue #164 — density guard on the sparse-rhs entry points (7f1682e)

`LuParams::sparse_rhs_max_density`, default `0.10`. Past that fraction of `m` the
reach DFS is abandoned mid-walk and the kernel sweeps the whole basis in natural
topological order. `1.0` disables the fallback, `0.0` forces it.
`SparseLu::sparse_rhs_fallbacks()` counts how often it fired.

Not the shape the issue suggested. "Skip the sort and sweep `0..m`" breaks two
invariants: `pattern` is the `O(touched)` reset list restoring `HyperWork`'s
all-zero-between-calls contract (a nonzero written outside it seeds the *next*
solve), and `u_solve_sparse`'s `SingularBasis` guard is deliberately narrowed to
the rows the solution depends on (widening it makes singularity depend on an
unrelated right-hand side's density). So the fallback marks the whole accumulator
```

## Git Status
```
255e2b7 docs: record what the #164 guard actually recovers
b8906a6 docs: session checkpoint 2026-08-13-03 (PR #162 review + #1008 verdict)
37df6df docs: record the #164 guard and the #165 ordering parameter
4a83bab feat(lu): make the column ordering a parameter, not a choice of constructor
7f1682e feat(lu): guard the sparse-rhs entry points on solution density
```

## Test Status
```
test symbolic::tests::symbolic_factorize_scotch_produces_valid_perm ... ok
test symbolic::tests::test_contrib_sizes_nonnegative ... ok
test symbolic::tests::test_perm_inverse_consistency ... ok
test symbolic::tests::test_symbolic_factorize_basic ... ok
test symbolic::tests::test_symbolic_factorize_dense ... ok
test symbolic::tests::test_symbolic_factorize_kkt ... ok
test numeric::factorize::tests::issue_5_mss1_iter0_inertia_wanders_under_delta_w_sweep ... ok
test symbolic::tests::choose_adaptive_routes_arrow_to_amf ... ok
test symbolic::tests::issue_3_scotchnd_on_kkt_recurses_after_o13 ... ok
test scaling::tests::auto_keeps_mc64_on_vesuvia_0000 ... ok
test symbolic::tests::choose_adaptive_rules ... ok
test scaling::tests::auto_keeps_mc64_on_vesuviou_0000 ... ok
test numeric::factorize::tests::issue_5_mss1_zero_tol_sweep_diagnostic ... ok
test symbolic::tests::issue_3_auto_on_kkt_routes_via_pick_default_method ... ok
test numeric::factorize::tests::issue_5_mss1_pivot_threshold_sweep_diagnostic ... ok
test scaling::tests::pick_scaling_strategy_routes_clnlbeam_to_infnorm ... ok
test scaling::hungarian::tests::mc64_hungarian_no_quadratic_heap_realloc_regression ... ok

test result: ok. 423 passed; 0 failed; 6 ignored; 0 measured; 0 filtered out; finished in 1.42s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-08-13-04.md)


factor/SSIDS       154500       0.04       0.03       0.32       0.94       2.13
solve/SSIDS        154500       0.92       1.00       2.29       8.00      39.75
nnzL/MUMPS         153560       0.61       0.58       0.75       4.50      23.11
nnzL/SSIDS         154500       0.88       1.00       1.00       4.50       5.00

Per-family factor geomean vs MUMPS (top 25 families by count):
family                  count    geomean        p50        max
HS118                    3000       0.92       0.95       1.13
ALLINITC                 3000       0.19       0.20       0.33
HS91                     3000       0.27       0.30       0.40
HATFLDBNE                3000       0.41       0.40       0.83
HS92                     3000       0.35       0.40       0.44
ALLINITA                 3000       0.41       0.40       0.92
CONCON                   3000       0.87       0.94       1.75
MGH10LS                  3000       0.21       0.22       0.25
HS90                     3000       0.20       0.20       0.33
DJTL                     3000       0.09       0.10       0.22
HS89                     3000       0.20       0.20       0.40
MCONCON                  3000       0.89       0.94       1.72
PALMER7A                 3000       0.29       0.30       0.33
PALMER5A                 3000       0.29       0.30       0.89
SSINE                    3000       0.27       0.27       0.40
HS13                     3000       0.17       0.20       0.70
HATFLDH                  3000       0.43       0.45       0.55
BIGGSC4                  3000       0.43       0.45       0.60
SSI                      3000       0.21       0.22       0.38
AVION2                   2682       1.46       1.50       2.07
CERI651ALS               2331       0.27       0.27       0.40
PFIT4                    2286       0.25       0.27       0.30
CERI651C                 2233       0.28       0.30       0.90
CERI651CLS               2227       0.27       0.27       0.40
BATCH                    2054       1.32       1.38       1.77

Top 10 worst factor-ratio vs MUMPS:
name                             n    feral(μs)    mumps(μs)      ratio
KIRBY2_0007                    458         1063          119       8.93
KIRBY2_0006                    458         1008          127       7.94
KIRBY2_0008                    458          903          122       7.40
KIRBY2_0009                    458          904          128       7.06
CHWIRUT2_0144                  159          303           43       7.05
ACOPP30_0090                   209          549           84       6.54
KIRBY2_0010                    458          784          133       5.89
KIRBY2_0011                    458          677          120       5.64
CHWIRUT2_0141                  159          235           43       5.47
MUONSINE_0000                 1537         1883          376       5.01

--- Dense Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
bucket                    count      p90     target  verdict
small-frontal (<200)     147982     1.61     <= 2.0     PASS
medium (<500)            152145     2.09     <= 3.0     PASS

--- Sparse Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
bucket                    count      p90     target  verdict
small-frontal (<200)     153455     1.54     <= 2.0     PASS
medium (<500)            153560     1.54     <= 3.0     PASS

```

## Recent Decisions

- A dense sweep writes nonzeros into positions that are not in `pattern`, and
  `pattern` *is* the O(touched) reset list that restores `HyperWork`'s
  all-zero-between-calls contract. A nonzero left outside it silently seeds the
  next solve.
- `u_solve_sparse`'s `SingularBasis` guard is deliberately narrowed to the rows
  the solution depends on (decisions.md, earlier today). Sweeping all `m` rows
  widens it, so whether a basis is reported singular would depend on an
  unrelated right-hand side's density. Preserving the narrowing needs a per-row
  "does this position matter" test, which is the reach being abandoned.

So the fallback marks the whole accumulator (making the reset list cover the
sweep) and fills `order` with the natural topological order over `0..m`. The four
kernels are untouched — they sweep whatever is in `order` — and the fallen-back
solve is then bit-for-bit the dense entry point it is falling back to, guard
width included. That is the right semantics for a fallback: it should be
indistinguishable from the thing it falls back to, not a third behavior.

The DFS is abandoned mid-walk rather than completed and then discarded, mirroring
`ReachWork::push`'s early abort on the dense route, so an over-cap solve pays a
bounded fraction of the reach. `over_cap` is monotone within a solve (`pattern`
only grows), so the second and later kernels of a fallen-back solve cost one
`pop`, not a re-walk.

`SparseLu::sparse_rhs_fallbacks()` exists because there was no valid witness that
the guard fired: `last_sparse_solve_work()` counts factor entries traversed,
which exceeds `m` on the reach path too. Without a dedicated counter the tests
could not tell an inert guard from a working one.

Evidence: PR #162 review, findings 1 and 4; `dev/journal/2026-08-13-04.org`.

## Recent Tried-and-Rejected
validates `symbolic.m == a.m`, a stale ordering is *legal*, so the obvious fix
was to compute the ordering once and reuse the handle on every refactorization.

Probed in discopt behind `DISCOPT_LU_SYM_REUSE` (a `sym_cache: Option<SparseLuSymbolic>`
on `FeralLU`, never merged). On QPLIB_3775 with `analyze_triangularized`:

| arm | factorizations | LuNumeric | wall |
|---|---|---|---|
| tri=1, reuse=0 | 64 | 184.6 ms | **1.193 s** |
| tri=1, reuse=1 | 1112 | 18381 ms | **137.835 s** |

**115x slower.** tri=0 reuse=1 did not finish inside a 300 s timeout. A simplex
basis is not structurally stable across 64 pivots: the stale ordering explodes
fill, the fill blows the numeric factorization, and the resulting instability
triggers a refactorization storm (64 → 1112) that feeds back on itself.

The conclusion is not "reuse harder" — it is that the ordering **must** be
recomputed on every refactorization, and therefore must be cheap. That is what
makes the `analyze` vs `analyze_triangularized` cost (4.3-12.4x, measured
standalone) a first-order effect rather than an amortizable one.

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
tests/lu_default_ordering.rs
tests/lu_dense_bump.rs
tests/lu_dense_update_bg.rs
tests/lu_dense.rs
tests/lu_ft_widebump.rs
tests/lu_hyper_sparse.rs
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

(truncated from      356 lines to 350 line budget)
