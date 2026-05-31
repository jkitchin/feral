# FERAL Context (auto-generated)

Generated: 2026-05-31T18:23:03Z

## Latest Session
File: dev/sessions/2026-05-31-03.md
```
# Session 2026-05-31-03

## Goal

Work through the perf-review (`dev/research/perf-review-2026-05-31.md`) levers
systematically on branch `claude/perf-levers`. For each lever: a research note,
a with/without A/B (performance + correctness), all other tests green. User
cadence: pause after each lever; default-ON once proven; git before/after for
pure-kernel swaps.

## Accomplished

Six levers triaged; **one implemented**, the rest deferred-with-rationale or
found already-done. All on `claude/perf-levers` (NOT merged to main).

- **Lever 1.1 — intra-front parallel Schur update: IMPLEMENTED** (`fbc3136`).
  Parallelizes the trailing Schur update of a single dense front via
  `tail.par_chunks_mut`, gated by `INTRAFRONT_MIN_AREA = 256*256` and a new
  `BunchKaufmanParams::intrafront_parallel` flag (default ON on the parallel
  driver only; the sequential driver is untouched → pounce#79
  no-oversubscription preserved). Bit-exact by construction (each trailing
  column reduced over ascending q on one thread; no cross-thread reduction).
  - A/B (`bench_intrafront`, FERAL_INTRAFRONT 0 vs 1, dense SPD fronts): speedup
    **1.2×–3.1×** (load-sensitive, bandwidth-bound), bit-exact every run.
  - Correctness: new gate `intrafront_parallel_schur_matches_serial`;
    `parallel_corpus_parity` full KKT corpus **0 mismatch / 41 220 factored**;
    full suite **572 passed, 0 failed**; clippy + fmt clean.
- **Lever 1.2 — cache blocking + L-panel packing: DEFERRED** (`eafd826`).
  Analysis + plan complete; restructures the hot bit-exact kernel for a
  ~10–30% bandwidth gain that is below the run-to-run noise floor on this shared
  machine. Revisit on idle hardware: 1.2a row-band blocking, then 1.2b packing.
- **Lever 2.1 — parallel-across-RHS solve: DEFERRED** (`5336914`). Design
  complete; bit-exactness entangled with the nrhs≥32 kernel dispatch (BLAS-3
  back-sub not bit-identical to rank-1), risky surgery for a narrow payoff
  (solve already < MUMPS; IPM uses nrhs=2).
- **Lever 2.2 — symbolic speedups: ALREADY IMPLEMENTED** (`2d576c4`, verify
  only). Both halves live since Phase 2.4.4: MC64 cached once + reused
  (symbolic/mod.rs:608/620, scaling/mod.rs:298-300); compression auto-dispatch
  via `OrderingPreprocess::Auto` + `pick_ordering_preprocess`. Perf-review
  over-stated remaining work; its "compRat≤0.7" gate is circular.
- **Levers 3.1 (FMA fallback) + 3.2 (wider NR): DEFERRED** (`56c6e48`). 3.2 is
  the same bandwidth wall as 1.2 (sub-noise-floor). 3.1 is ~0% on this arm64 host
  (decisions.md 2026-04-14) and flips inertia on ~30/154k matrices; already an
  opt-in `BunchKaufmanParams::fma` field for a future x86 measurement.

## Benchmark Results

8-matrix `bench` (the small synthetic harness; the corpus p90 bench is the
separate `bench_solver_corpus` walk), FERAL_INTRAFRONT OFF vs ON — all
sub-threshold, so both take the serial refactored path:
```

## Git Status
```
56c6e48 docs(lever-3.x): defer FMA fallback (3.1) and wider NR (3.2) with rationale
2d576c4 docs(lever-2.2): verify symbolic speedups already implemented (Phase 2.4.4)
5336914 docs(lever-2.1): defer parallel-across-RHS solve with design + rationale
eafd826 docs(lever-1.2): defer cache-blocking/packing with analysis + rationale
dc66907 docs(lever-1.1): report speedup as a load-sensitive range, not a best run
```

## Test Status
```
test symbolic::tests::symbolic_factorize_kahip_produces_valid_perm ... ok
test symbolic::tests::test_contrib_sizes_nonnegative ... ok
test symbolic::tests::test_perm_inverse_consistency ... ok
test symbolic::tests::test_symbolic_factorize_dense ... ok
test symbolic::tests::test_symbolic_factorize_basic ... ok
test symbolic::tests::test_symbolic_factorize_kkt ... ok
test symbolic::tests::symbolic_factorize_scotch_produces_valid_perm ... ok
test scaling::tests::auto_falls_back_to_infnorm_on_mss1_0009 ... ok
test numeric::factorize::tests::issue_5_mss1_iter0_inertia_wanders_under_delta_w_sweep ... ok
test symbolic::tests::issue_3_scotchnd_on_kkt_resolves_to_amd_when_bisection_degenerates ... ok
test symbolic::tests::choose_adaptive_rules ... ok
test scaling::tests::auto_keeps_mc64_on_vesuvia_0000 ... ok
test scaling::tests::auto_keeps_mc64_on_vesuviou_0000 ... ok
test numeric::factorize::tests::issue_5_mss1_zero_tol_sweep_diagnostic ... ok
test numeric::factorize::tests::issue_5_mss1_pivot_threshold_sweep_diagnostic ... ok
test symbolic::tests::issue_3_auto_on_kkt_routes_via_pick_default_method ... ok
test scaling::tests::pick_scaling_strategy_routes_clnlbeam_to_infnorm ... ok

test result: ok. 317 passed; 0 failed; 6 ignored; 0 measured; 0 filtered out; finished in 0.42s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-05-31-03.md)


8-matrix `bench` (the small synthetic harness; the corpus p90 bench is the
separate `bench_solver_corpus` walk), FERAL_INTRAFRONT OFF vs ON — all
sub-threshold, so both take the serial refactored path:

spd_10   off=35  on=29     kkt_10_3   off=3   on=3
spd_50   off=20  on=20     kkt_30_10  off=25  on=25
spd_100  off=78  on=68     kkt_50_15  off=47  on=47
spd_200  off=389 on=362    kkt_100_30 off=130 on=130

Within noise; **inertia identical** OFF vs ON on all 8. The small-front geomean
cannot regress from 1.1: fronts below the 65 536-area gate never enter the
parallel branch, and the serial path is the bit-exact refactor proven by
`parallel_corpus_parity` (0 mismatch). The full corpus p90 walk
(`bench_solver_corpus`, 169 591 matrices vs MUMPS oracle) is an hour+ run and
timing-noisy on this contended box; not run to completion — the structural
argument + corpus-parity bit-exactness are the load-robust evidence.

```

## Recent Decisions

## 2026-05-31 — Lever 2.2 (symbolic speedups) found already-implemented

The perf-lever sweep reached Tier-2 #2 (symbolic-phase speedups: cache MC64
across compress->scale, and auto-dispatch compression on predicted-tail
matrices). On inspection BOTH halves are already implemented in the codebase
("Phase 2.4.4", pre-dating this sweep): the MC64 matching is computed once and
cached (symbolic/mod.rs:605/614) and reused by the numeric phase
(scaling/mod.rs:298-300); compression auto-dispatch is the default via
OrderingPreprocess::Auto + pick_ordering_preprocess (mod.rs:347-369). The
perf-review (dev/research/perf-review-2026-05-31.md), written the same day by
the PR#59 analysis session, over-stated the remaining work by listing these as
future. Its further "tighter gate (compRat<=0.7)" idea is not viable as stated
(compRat requires running MC64 to compute, so it cannot gate whether to run
MC64). No code change; verification recorded in
dev/research/lever-2.2-symbolic-speedups.md.


## 2026-05-31 — Levers 3.1 (FMA fallback) and 3.2 (wider NR) deferred

Both Tier-3 levers deferred (dev/research/lever-3.x-deferred.md). 3.2 (wider
micro-kernel NR): perf-review says measure only after 1.1/1.2 land, but 1.2 is
deferred and 3.2 attacks the same memory-bandwidth wall — wider arithmetic width
does not help a bandwidth-bound kernel, and the gain is sub-noise-floor on this
shared machine. 3.1 (FMA boundary-safe fallback): this host is arm64, where FMA
measured ~0% (decisions.md 2026-04-14, 1.87->1.86) and flips inertia on ~30/154k
boundary matrices; high-complexity fallback for ~zero gain on the only available
hardware. fma is already an opt-in BunchKaufmanParams field for a future x86
measurement. The perf-lever sweep thus implements Lever 1.1 only (1.2/2.1
deferred-with-plan, 2.2 already implemented in Phase 2.4.4).

## Recent Tried-and-Rejected
**c-block outer / m-tile inner** would keep a small `B`-block
L1-resident and cut the dominant re-streaming by the factor `NR/MR = 2`.

Tried the swap. **Measured: no improvement.** n=1024 stayed at ratio
~1.0–1.2 (still a regression), and n=484/2025 were within noise of the
m-outer order. The loop order was not the bottleneck at these sizes.

Reverted to the simpler m-outer kernel (the comment claiming the swap
fixed n=1024 would have been false). The actual n=1024 cause was the
**stride-`n` gather/scatter** reading the column-major `y` — power-of-
two `n` aliased RHS columns into the same cache sets. Flipping the
internal `y` buffer to row-major (contiguous memcpy gather/scatter)
fixed it (ratio 1.2 → 0.33) and ~halved wide-solve time everywhere.
See `dev/research/issue-57-blas3-panel.md` Results and the
`dev/decisions.md` 2026-05-30 entry.

**Lesson.** Diagnose the bottleneck before micro-optimizing the kernel:
the transpose in the gather/scatter dominated, not the GEMM's operand
re-streaming. A loop-order change to the GEMM was wasted motion until
the layout (row-major `y`) was fixed.

## Source Files
```
src/bin/alloc_probe.rs
src/bin/bench_axpy_small.rs
src/bin/bench_dense_multirhs.rs
src/bin/bench_fma_phase3.rs
src/bin/bench_intrafront.rs
src/bin/bench_issue8.rs
src/bin/bench_multirhs.rs
src/bin/bench_one_matrix.rs
src/bin/bench_orderings.rs
src/bin/bench_solver_corpus.rs
src/bin/bench_solver_reuse.rs
src/bin/bench_sqd.rs
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
src/bin/diag_issue50_auto_validate.rs
src/bin/diag_issue50_inventory.rs
src/bin/diag_issue50_large_sparse_scan.rs
src/bin/diag_issue50_numeric_inventory.rs
src/bin/diag_issue50_symbolic.rs
src/bin/diag_leaf_profile.rs
src/bin/diag_max_ncol.rs
src/bin/diag_mc64_cycles.rs
src/bin/diag_mittelmann.rs
src/bin/diag_narx_kernel_gflops.rs
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
src/bin/diag_small_sparse_inventory.rs
src/bin/diag_sparse_memory.rs
src/bin/diag_split_tail.rs
src/bin/diag_strategy_compare.rs
src/bin/diag_supernode_cost.rs
src/bin/diag_swopf_w22x2.rs
src/bin/diag_symbolic_stages.rs
src/bin/dump_diff.rs
src/bin/feral_replay.rs
src/bin/feral_time.rs
src/bin/hs85_diag.rs
src/bin/parallel_corpus_parity.rs
src/bin/phase0_cb_on_revalidation.rs
src/bin/polak6_diag.rs
src/bin/policy4_diag.rs
src/bin/probe_acopp30_64.rs
src/bin/probe_cache_sequence.rs
src/bin/probe_cascade_perturb.rs
src/bin/probe_clnlbeam_refine.rs
src/bin/probe_clnlbeam_shape.rs
src/bin/probe_deltac_sensitivity.rs
src/bin/probe_dtoc2_mc64.rs
src/bin/probe_explicit_zeros.rs
src/bin/probe_f01.rs
src/bin/probe_fbrain.rs
src/bin/probe_fma_kernel.rs
src/bin/probe_front_concentration.rs
src/bin/probe_hang_loop.rs
src/bin/probe_intrafront_schur.rs
src/bin/probe_ir_trajectory.rs
src/bin/probe_issue_19.rs
src/bin/probe_issue45_ordering.rs
src/bin/probe_issue45.rs
src/bin/probe_issue46_preprocess.rs
src/bin/probe_issue46_supernode.rs
src/bin/probe_issue46.rs
src/bin/probe_issue49.rs
src/bin/probe_issue54_alpha_shift.rs
src/bin/probe_issue54_cascade.rs
src/bin/probe_issue54_ma57_alpha.rs
src/bin/probe_issue54.rs
src/bin/probe_kkt_replay.rs
src/bin/probe_marine_shape.rs
src/bin/probe_marine_time.rs
src/bin/probe_mc64_spread.rs
src/bin/probe_mc64_synth.rs
src/bin/probe_narx_factor.rs
src/bin/probe_narx_phases.rs
src/bin/probe_panel_attribution.rs
src/bin/probe_pinene_issue38_fix.rs
src/bin/probe_rkt_shape.rs
src/bin/probe_robot_profile.rs
src/bin/probe_rocket_profile.rs
src/bin/probe_rocket_residuals.rs
src/bin/probe_rocket_slow.rs
src/bin/probe_scaling_policy4.rs
src/bin/probe_static_pivot_inertia.rs
src/bin/probe_supernode_widths.rs
src/bin/probe_thomson_hessian.rs
src/bin/probe_value_determinism.rs
src/bin/probe_warm_cascade.rs
src/bin/probe_wide_supernode.rs
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

(truncated from      402 lines to 350 line budget)
