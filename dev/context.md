# FERAL Context (auto-generated)

Generated: 2026-05-16T18:33:13Z

## Latest Session
File: dev/sessions/2026-05-16-07.md
```
# Session 2026-05-16-07

## Goal

Verify whether GH #19 (parallel-assembly heuristic regresses
small-KKT IPM 12×, helps large-KKT 2.8×) is empirically resolved by
the four follow-up commits already on `main`, and close the issue
if the data supports it. The implementation was complete after
session `2026-05-15-06`, but the GH `gh issue close` call was
blocked by the auto-mode classifier in that session; this session
finishes the closeout.

## Accomplished

### Verified — verdict A (fixed). Closed GH #19.

Reproduced the original two-vector regression panel on Apple M4 Pro
(14 rayon workers) using the existing `probe_issue_19` binary
(`src/bin/probe_issue_19.rs`, committed `25926cc`). 10 reps per
config, median wall, three configs per matrix.

**robot_1600** (4 iters: 0000, 0001, 0003, 0006):

| iter | est_flops | seq    | gated  | forced | par/seq |
| ---- | --------- | ------ | ------ | ------ | ------- |
| 0000 | 4.75e6    | 7.21   | 6.52   | 8.99   | 0.80×   |
| 0001 | 1.13e7    | 12.76  | 8.80   | 8.71   | **1.45×** |
| 0003 | 1.13e7    | 18.67  | 14.39  | 14.48  | **1.30×** |
| 0006 | 9.43e6    | 13.90  | 17.56  | 9.65   | 1.44× (vetoed) |

Worst-case slowdown of default-gated vs optimal driver: 1.26× on
iter 0006 (deliberate safety-margin trade-off in `b12e03c`).
Original issue's worst case: 12×. Regression vector closed.

**henon120** (4 iters: 0000, 0001, 0003, 0005):

| iter | est_flops | seq    | gated  | forced | par/seq |
| ---- | --------- | ------ | ------ | ------ | ------- |
| 0000 | 9.83e5    | 22.16  | 19.35  | 73.73  | 1.15×   |
| 0001 | 6.13e9    | 655.00 | 104.74 | 119.01 | **6.25×** |
| 0003 | 6.13e9    | 433.79 | 98.18  | 97.49  | **4.42×** |
| 0005 | 6.13e9    | 447.61 | 104.37 | 116.37 | **4.29×** |

Original issue: parallel wins 2.8× on henon120. Current: parallel
wins 4.3-6.3× on the three steady-state iters. Improvement vector
retained and amplified.

### Artefacts

- `dev/research/issue-19-par-min-flops-verification.md` — full
```

## Git Status
```
c7479fd feat(#34): NumericParams::sqd_mode + Solver::with_sqd_mode builder
1a18336 chore(session): 2026-05-16-06 -- close #10 (hardware floor); merge agent-27
61002f8 research(issue-10): forced supernode amalgamation lever fails (5/5 levers exhausted)
d3e6199 docs(#34): SQD fast-path research note + bib + decisions
05aef71 merge(agent-27): M4 synthetic generators for saddle/wide-frontal/MC64/Stokes
```

## Test Status
```
test result: ok. 5 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.00s

     Running tests/tiny_fast_path.rs (target/debug/deps/tiny_fast_path-57494b3d3352096b)

running 5 tests
test test_gate_just_outside_n_tiny ... ok
test test_solve_parity_tiny_real_matrix ... ok
test test_gate_tiny_sparse_in ... ok
test test_gate_boundary_n_16 ... ok
test test_determinism_tiny ... ok

test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

   Doc-tests feral

running 1 test
test src/symbolic/profiler.rs - symbolic::profiler::SymbolicProfiler (line 27) ... ignored

test result: ok. 0 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.00s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-05-16-07.md)


`cargo run --bin bench --release` not re-run this session because no
production code changed; the only deltas are documentation +
research-note + CHANGELOG. The `probe_issue_19` results above are
the substantive measurements.

Hygiene gates:

cargo test --lib --release      -> 259 passed, 0 failed, 5 ignored
cargo clippy --all-targets --release -- -D warnings -> clean
cargo fmt --check               -> clean

```

## Recent Decisions
1. SmallLeafBatch driver removal — within noise.
2. MAXFROMM AMAX-scan cache — within noise.
3. Manual axpy SIMD tightening — pulp ties scalar within 1ns/call.
4. Ordering swap (Metis/Scotch ND) — 1.3–2.3× slower; no shape
   widening (`ncol_p90` invariant at 10.08 across all orderings).
5. Forced supernode amalgamation (`nemin ∈ {32, 64, 128}`) — shape
   widens 2× but factor time flat or regresses 36% on `clnlbeam`.

The rank-1 axpy kernel on `ncol=1..16` fronts is bandwidth-bound;
pulp saturates the vector ALU; AMD's elimination tree is already
shape-optimal under the nnz_L bound. No further per-pivot speedup
is available without changing the front structure in ways that
violate the nnz_L bound that motivated the ordering choice.

**Decision.** Keep `SupernodeParams { nemin: 16, .. }` as the
default. Keep `OrderingMethod::Amd` as the default. The opt-in
knobs `Solver::with_ordering(MetisND/ScotchND)` (shipped session 02)
and `SupernodeParams::nemin` (existing) stay available for
workloads where the elimination tree genuinely has fusion
opportunities. No APP-class kernel is shipped; future work that
*adds new front structure* (children-of-children amalgamation
across non-adjacent tree levels, or a kernel that handles
`ncol < tile-size` differently) is welcome as a fresh issue.

References:
- `dev/research/issue-10-maxfromm-phase2-corpus.md` (#1, #2)
- `dev/research/issue-10-ordering-supernode-shape.md` (#4)
- `dev/research/issue-10-amalgamation-floor.md` (#5)
- Commits: d3b031d, 61002f8.
- GH: https://github.com/jkitchin/feral/issues/10#issuecomment-4467668859

## Recent Tried-and-Rejected
regresses. `clnlbeam` regresses 36% at nemin=64 because chain-link
merges blow trailing-fill faster than the wider panel can amortize.

**Why it was rejected.** Closes the fifth and final architectural
lever for issue #10. All five (SLB driver removal, MAXFROMM AMAX
cache, manual axpy SIMD, ordering swap, this nemin sweep) come up
negative on the 1D-banded panel. The rank-1 axpy kernel on
`ncol=1..16` fronts is bandwidth-bound; pulp saturates the vector
ALU; AMD's elimination tree is already shape-optimal under the
nnz_L bound. A pilot run at `nemin ∈ {256, MAX}` hung on
`clnlbeam_0000` — a single near-dense front of order >n/2 collapsed
the dense LDL into a non-returning state. Sweep capped at 128.

Issue #10 closes as "hardware floor reached on the 1D-banded panel."
The opt-in knobs (`Solver::with_ordering`, `SupernodeParams::nemin`)
stay shipped for workloads where the elimination tree genuinely
has fusion opportunities — they just don't help here.

Documented in `dev/research/issue-10-amalgamation-floor.md`. A/B
binary: `src/bin/diag_nemin_amalgamation_panel.rs`. Commit 61002f8.

## Source Files
```
src/bin/alloc_probe.rs
src/bin/bench_axpy_small.rs
src/bin/bench_fma_phase3.rs
src/bin/bench_issue8.rs
src/bin/bench_one_matrix.rs
src/bin/bench_orderings.rs
src/bin/bench_solver_corpus.rs
src/bin/bench_solver_reuse.rs
src/bin/bench.rs
src/bin/blas3_prototype.rs
src/bin/calibrate_par_min_flops.rs
src/bin/d3_probe.rs
src/bin/d4_probe.rs
src/bin/diag_acopp30_residual.rs
src/bin/diag_acopr.rs
src/bin/diag_acopr14.rs
src/bin/diag_amalgamation.rs
src/bin/diag_amd_substages.rs
src/bin/diag_amf_vs_amd.rs
src/bin/diag_cascade_default_evidence.rs
src/bin/diag_cascade_ratio_distribution.rs
src/bin/diag_chainwoo_profile.rs
src/bin/diag_chainwoo.rs
src/bin/diag_clnlbeam_maxfromm.rs
src/bin/diag_clnlbeam_slb.rs
src/bin/diag_compress_costbenefit.rs
src/bin/diag_compress_profile.rs
src/bin/diag_compression_bench.rs
src/bin/diag_cond_parity.rs
src/bin/diag_dense_tail.rs
src/bin/diag_etree_shape.rs
src/bin/diag_factor_nnz_accounting.rs
src/bin/diag_fbrain3ls_pivtol_sweep.rs
src/bin/diag_fill_parity.rs
src/bin/diag_fill_tail.rs
src/bin/diag_inertia_mismatch.rs
src/bin/diag_leaf_profile.rs
src/bin/diag_max_ncol.rs
src/bin/diag_mc64_cycles.rs
src/bin/diag_mittelmann.rs
src/bin/diag_near_singular_sweep.rs
src/bin/diag_nemin_amalgamation_panel.rs
src/bin/diag_orbit2_quotient.rs
src/bin/diag_ordering_panel.rs
src/bin/diag_ordering_race.rs
src/bin/diag_par_firstdiff.rs
src/bin/diag_par_frontal_hash.rs
src/bin/diag_par_repeat.rs
src/bin/diag_parent_unique.rs
src/bin/diag_phase_b_nemin_sweep.rs
src/bin/diag_pinene_0009_profile.rs
src/bin/diag_pinene_amd.rs
src/bin/diag_pinene_pivot_cliff.rs
src/bin/diag_pinene_static_pivot.rs
src/bin/diag_poisson_kkt.rs
src/bin/diag_qcqp_knobs.rs
src/bin/diag_qcqp_profile.rs
src/bin/diag_robot1600_eigs.rs
src/bin/diag_schur_parity.rs
src/bin/diag_small_leaf_gate.rs
src/bin/diag_small_leaf.rs
src/bin/diag_sparse_memory.rs
src/bin/diag_split_tail.rs
src/bin/diag_strategy_compare.rs
src/bin/diag_supernode_cost.rs
src/bin/diag_swopf_w22x2.rs
src/bin/diag_symbolic_stages.rs
src/bin/dump_diff.rs
src/bin/hs85_diag.rs
src/bin/parallel_corpus_parity.rs
src/bin/polak6_diag.rs
src/bin/policy4_diag.rs
src/bin/probe_acopp30_64.rs
src/bin/probe_cascade_perturb.rs
src/bin/probe_deltac_sensitivity.rs
src/bin/probe_ir_trajectory.rs
src/bin/probe_issue_19.rs
src/bin/probe_panel_attribution.rs
src/bin/probe_scaling_policy4.rs
src/bin/produce_dense_schur.rs
src/bin/profile_hot.rs
src/bin/profile_sparse.rs
src/bin/profile_supernode_distribution.rs
src/bin/solve_microbench.rs
src/bin/vesuvio_diag.rs
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
tests/column_renumbering_parity.rs
tests/column_renumbering.rs
tests/delayed_pivoting.rs
tests/dense_fast_path.rs
tests/dense_ldlt.rs
tests/factor_scratch_parity.rs
tests/factor_workspace_parity.rs
tests/fma_opt_in_roundtrip.rs
tests/growth_flag.rs
tests/issue_15_cascade_arm_gate.rs
tests/issue_17_robot_1600_cascade_off.rs
tests/issue_18_narx_cfy_cascade_off.rs
tests/issue_2_kkt_ls_init.rs
tests/kkt_hardening.rs
tests/kkt_matrices.rs
tests/large_matrix_smoke.rs
tests/ldlt_compress.rs
tests/maxfromm_parity.rs
tests/mc64_end_to_end.rs
tests/mc64_scaling.rs
tests/multi_rhs.rs
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
tests/stress_tests.rs
tests/symbolic_profiler.rs
tests/threshold_consistency.rs
tests/tiny_fast_path.rs
```
