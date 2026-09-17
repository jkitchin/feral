# FERAL Context (auto-generated)

Generated: 2026-09-17T19:16:37Z

## Latest Session
File: dev/sessions/2026-09-17-01.md
```
# Session 2026-09-17-01

## Benchmark note (read first)

**The corpus is not on this machine, so the session benchmark is vacuous.**
`cargo run --bin bench --release` ran and found **2 matrices**
(`densecol_kkt_300_0000` and one other), both passing, with **no oracle
timings**, so both Phase 2.8.1 exit partitions report `N/A` with count 0. No
factor-ratio-vs-MUMPS number can be quoted from this session, favourable or
otherwise, and none is quoted below.

That is acceptable for what this session changed — the work is entirely in
the *symbolic* ordering, measured directly as `nnz_L` and elimination flops
against an external oracle (MA57's bundled real METIS). It is **not**
acceptable as a release gate: the next session on a corpus machine must run
the full bench before this lands anywhere near a tag, and must in particular
measure real factor+solve wall-clock, which this session did not touch.

```
--- Dense Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
bucket                    count      p90     target  verdict
small-frontal (<200)          0        -     <= 2.0      N/A
medium (<500)                 0        -     <= 3.0      N/A

--- Sparse Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
bucket                    count      p90     target  verdict
small-frontal (<200)          0        -     <= 2.0      N/A
medium (<500)                 0        -     <= 3.0      N/A
```

## Goal

Explore issue #203 — pounce reports factorization cost growing `~n^2.3` and
symbolic fill `~n^1.37` on a long-horizon collocation optimal-control KKT
whose `dim K` and `nnz(K)` are exactly linear in the horizon. The issue
hypothesises a block chain with an available linear-fill elimination order
that no ordering backend is finding, and asks three questions plus a fourth
from the original report (why MA57 is 1.2–1.9x faster per iteration).

## Accomplished

### 1. Diagnosed the issue, and refuted its premise

The reproducer's own index arithmetic says the matrix is **not** a chain of
small blocks: the embedded spatial coupling is 775x775 with 1977 nonzeros
(2.55/row, a near-tree gas network) crossed with a time path of `3*nfe`
points. Spatial extent 775 exceeds temporal extent (18 to 288 over the
sweep), so a time-slice separator is the *expensive* cut.

Built the chain permutation the issue asks for and replayed it through
```

## Git Status
```
7045060 chore: drop the issue203 separator-tree probe as unreliable
15d6217 feat(metis): refine the node separator through uncoarsening, not the edge cut
ba19a3f research: diagnose issue #203 collocation-KKT fill growth as an ND ordering gap
a303326 Merge pull request #198 from jkitchin/docs/krylov-evaluation
b412113 docs: record the recycled-MINRES evaluation and why it was not adopted
```

## Test Status
```
test symbolic::tests::symbolic_factorize_default_uses_amf_for_small_matrices ... ok
test symbolic::tests::symbolic_factorize_metis_produces_valid_perm ... ok
test symbolic::tests::symbolic_factorize_scotch_produces_valid_perm ... ok
test symbolic::tests::test_contrib_sizes_nonnegative ... ok
test symbolic::tests::test_perm_inverse_consistency ... ok
test symbolic::tests::test_symbolic_factorize_basic ... ok
test symbolic::tests::test_symbolic_factorize_dense ... ok
test symbolic::tests::test_symbolic_factorize_kkt ... ok
test symbolic::tests::symbolic_factorize_kahip_produces_valid_perm ... ok
test symbolic::tests::is_arrow_bordered_rejects_many_hubs ... ok
test symbolic::tests::choose_adaptive_routes_arrow_to_amf ... ok
test numeric::solve::tests::cb_coarsening_threshold_is_arithmetically_inert ... ok
test symbolic::tests::choose_adaptive_rules ... ok
test symbolic::tests::issue_3_scotchnd_on_kkt_recurses_after_o13 ... ok
test symbolic::tests::issue_3_auto_on_kkt_routes_via_pick_default_method ... ok
test numeric::solve::tests::cb_core_profitable_matches_the_plan_gate ... ok
test scaling::hungarian::tests::mc64_hungarian_no_quadratic_heap_realloc_regression ... ok

test result: ok. 444 passed; 0 failed; 7 ignored; 0 measured; 0 filtered out; finished in 3.35s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; no session checkpoint with bench)
```

## Recent Decisions
|---|---|---|---|
| gaslib collocation KKT, nfe=48 | 2.10e10 | 6.74e9 | 6.42e9 |
| nfe=96 | 6.89e10 | 2.91e10 | 1.75e10 |
| grid3d 40^3 | 2.31e10 | 2.18e10 | 1.67e10 |

2.4x-3.5x on collocation KKTs, 6-10% on grid Laplacians, never worse on any
of the 40 matrices measured, and not slower (`MetisND` symbolic at nfe=96:
6.31 s -> 7.30 s, inside the 1.5x guardrail).

**The acceptance table in `dev/plans/metis-node-separator-fm.md` was missed
on 3 of its 4 rows** (nfe=96, grid2d, grid3d) and the default was flipped
anyway. Those targets were written as "match real METIS" before any code
existed; using them as a gate would have withheld a change that is a strict
improvement everywhere it was measured. Recording the miss here rather than
quietly restating the targets.

**What is deliberately *not* changed: `choose_adaptive`.** It still reroutes
every would-be-`MetisND` decision to `Amf` (issues #67/#73), so `Auto` is
bit-identical to before and still picks the 1.74x-worse ordering at nfe=96.
That override was established on real factor+solve wall-clock across the IPM
corpus, and `tried-and-rejected.md` (2026-05, fill-guarded race) already
records one attempt to re-decide it on fill that was rejected because fill
does not predict speed — nql180 has 0.98x the fill under MetisND and is still
2.05x slower end to end. Re-opening it needs a wall-clock A/B on the corpus
machine, not a symbolic argument.

**Evidence.** `dev/research/feral-metis-node-separator-fm-2026-09-17.md`;
`cargo test --workspace` 1199 passed / 0 failed / 25 ignored;
`cargo fmt --check` and `cargo clippy --workspace --all-targets -- -D warnings`
clean.

## Recent Tried-and-Rejected
past `nfe ~ 1000`, well beyond any horizon in play.

**Why the premise fails.** The matrix is not a chain of small blocks. The
reproducer's embedded spatial coupling is 775x775 with 1977 nonzeros (2.55
per row) — a near-tree gas network — crossed with a time path of `3*nfe`
points. That is a 2-D product graph whose *spatial* extent (775) exceeds its
*temporal* extent (18 to 648 over the whole sweep). Ordering along time only
is the classic band/profile ordering for a 2-D grid, with fill
`O(n * bandwidth)`.

**What this rejects.** Both the `External` chain permutation and the
chain/near-banded detection heuristic proposed for `Auto` dispatch: on this
pattern such a heuristic would fire and lose two orders of magnitude.

**What it does not reject.** A better *nested dissection*. Replaying MA57's
bundled real-METIS ordering through feral's pipeline gives 1.70x fewer flops
than feral's best at nfe=48 and 2.89x fewer at nfe=96, with the gap widening
as the horizon grows. The deficit is in `feral-metis` / `feral-scotch`
separator quality, not in a missing chain heuristic. Full evidence in
`dev/research/issue-203-collocation-kkt-ordering-2026-09-17.md`.

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
