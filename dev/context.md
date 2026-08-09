# FERAL Context (auto-generated)

Generated: 2026-08-09T16:11:10Z

## Latest Session
File: dev/sessions/2026-08-09-03.md
```
# Session 2026-08-09-03

## Goal

Post-0.15.0 optimization queue item 2: the `nemin` amalgamation sweep
(`dev/plans/release-0.15.0-checklist.md` §4.2), motivated by the PR #150
review — 90% of clnlbeam's supernodes are ≤8 columns and they are 35.9%
of its factorization loop.

## Accomplished

**Both amalgamation levers measured and rejected. Nothing ships; the
default path is bit-identical to 0.15.0.**

Harness: `crates/feral-diagnostics/src/bin/diag_nemin_post_simd.rs` —
paired alternating A/B per `decisions.md` 2026-08-09 (all arms timed once
per pair, `min_us`, sign test), symbolic computed once per arm so only
the numeric phase is timed, inertia + true ∞-norm residual reported per
arm. 61 CUTEst KKT parity fixtures (n = 3 … 5314) + 4 structured KKTs
(clnlbeam_like n=100000, grid250, sparseqp_kkt, chain12000_kkt).
x86_64, 4 cores, AVX2. Criterion pre-registered before the first run.

### 1. The queue item's direction is dead, and re-confirmed

Geomean vs shipped `nemin=16`, time / factor_nnz:

| arm | 1 | 4 | 8 | 32 | 64 |
|---|---|---|---|---|---|
| 61 parity | 1.21/0.67 | 1.02/0.83 | 0.986/0.89 | 1.02/1.19 | 1.15/1.65 |
| 4 structured | 1.33/0.37 | 0.99/0.51 | 0.925/0.68 | 1.13/1.60 | 1.53/2.74 |

Every arm above 16 loses on time and inflates fill on every class. This
re-confirms the 2026-05-16 rejection (issue #10 lever 5) after the one
development that could have overturned it — that rejection turned
entirely on "the wider panel cannot amortize the fill", and the 0.15.0
kernel rewrite is exactly what changed the amortization rate. It did not
buy the fill back.

### 2. `nemin=8` wins but fails its pre-registered criterion

Better on both axes on structured KKTs (clnlbeam_like 0.925× time at
0.579× fill; chain12000 0.815/0.640). But the criterion was ≥5% geomean
with ≥8/10 sign test on ≥2 classes and no fixture regressing >2%: parity
geomean is 1.4%, and CERI651A_0000 regresses 16% (178→207 µs, 2/15 wins
— an effect, not noise), DEGENLPB_0046 12%, BQPGASIM_0012 10%.

### 3. Cost-model merge guard: implemented, works, rejected on accuracy

The size rule (`child_ncol < nemin && parent_ncol < nemin`) never asks
what a merge costs. Front height is `col_counts[first_col].max(ncol)`,
```

## Git Status
```
f39a5db perf(symbolic): amalgamation cost-model guard (opt-in) + nemin re-sweep
e2ba443 research: falsify the scaling warm-start hypothesis (post-0.15.0 item 1)
808babb release: feral v0.15.0 (#151)
fad5670 perf(parallel): task-per-subtree coarsening + profiler nanoseconds (#150)
e8e1c5a perf(kernel): explicit SIMD packed trailing update + x86 pulp dispatch fix (#149)
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

test result: ok. 409 passed; 0 failed; 6 ignored; 0 measured; 0 filtered out; finished in 2.89s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-08-09-03.md)


The corpus bench (`cargo run --bin bench --release`) is not runnable
here — `data/matrices` holds 2 files in this container, not the 154k
corpus. The default path is unchanged and bit-identical, so the release
gates are unaffected by this session. Local suite:

cargo test --release
84/84 test binaries, 802 passed, 0 failed
cargo fmt --check / cargo clippy -- -D warnings / clippy -p feral-diagnostics: pass

Medium-front throughput probe for the next item (`bench_dense_front`,
nofma serial, 30 reps, dense front of order n):

n=32   0.77 GFLOP/s     n=192   3.13
n=48   1.42             n=256   3.51
n=64   1.85             n=512   4.40
n=96   2.95             n=1024  5.05
n=128  3.45

```

## Recent Decisions
run medians. `min_us` per invocation is the preferred per-sample
statistic (least interfered). Cross-time comparison of numbers taken in
different sessions is not evidence at all.

**Why.** Measured on the issue-#148 chainW proxy (session 2026-08-09-03):
three `FERAL_PAR_TASK_MIN_FLOPS` settings that produce *identical* task
plans — same code path, byte-identical work — measured 139.6 / 259.1 /
155.5 ms, a 1.9x spread. Eight invocations of one fixed config spanned
min_us 124.7-163.2 ms (31%) and median_us 149.2-183.7 ms (23%). Two
conclusions had already been drawn from inside that band and were both
wrong: a claimed "chainW anomaly" (per-node spawning 20% faster than
sequential) and a claimed 5-18% regression from PR #150. Paired
re-measurement reversed both — 9/12 pairs favour the new code (median
ratio 0.961) and 9/10 favour coarse over fine-grained tasks (median
1.045, sign-test p~0.02).

**Relationship to the existing rule.** The 2026-04-14 entry ("any
bench-p90 delta smaller than ~5% must be confirmed with a 3-run
median") is necessary but NOT sufficient: three consecutive medians can
all land inside one drift excursion, which is exactly how both wrong
conclusions above were reached. Paired A/B supersedes it for container
measurement; the 3-run rule still applies to the corpus bench on a
quiet machine.

**Consequence for prior sessions.** Numbers in
dev/sessions/2026-08-09-01.md and -02.md were collected unpaired.
Those with large effects (dense-front kernel 2.7-7x, grid250, sparseqpL
- since re-confirmed paired at 10/10 and 9/10) stand; sub-10% fixture
deltas in those checkpoints should be treated as unresolved rather than
as measured wins until re-run paired.

## Recent Tried-and-Rejected
`nemin=8`, MEYER3NE 83× at `nemin=4`), which is what makes it a property
of the direction rather than of this rule.

**Why rejected.** "Correctness before performance, always" is a hard
constraint. 2–7% of factor time and 11–45% of fill does not buy seven
digits of residual. Neither my pre-registered criterion nor the queue
item thought to check the axis that decided it — recorded here because
the next person to have this idea will not think to check it either.

The knob stays in-tree defaulting to `None` (bit-identical default path)
as the reproduction apparatus, with the accuracy result in its doc
comment. Research note:
`dev/research/amalgamation-cost-model-2026-08-09.md`.

**Also redirects the target.** pounce#552's re-measurement against a
released 0.15.0 (comment 5232409020) shows clnlbeam more than halved
(8.05× → 3.54× vs MA57) and **no longer the worst case** — `dtoc1nd` is,
at 3.77×, and it is a dense-front matrix (nnz/dim 23.0, fronts of 33–64
columns). Amalgamation is a chain-KKT lever aimed at a problem that has
largely receded.

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
tests/lu_dense.rs
tests/lu_dense_update_bg.rs
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
