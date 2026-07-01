# FERAL Context (auto-generated)

Generated: 2026-07-01T10:55:37Z

## Latest Session
File: dev/sessions/2026-07-01-03.md
```
# Session 2026-07-01-03

## Goal

Issue #99: "loop over all levers until faer-level performance" on the dense-front
factorization throughput gap (~3.5× to faer on the qap15 conic KKT, dominated by
a 2955×2955 indefinite root).

## Reality check (reported up-front, per protocol)

The issue's premises are **not reproducible on this branch**, and I established
this before writing code:

- **PR #92 is open/unmerged.** It (issue #91) contains the `OrderingPreprocess::Auto`
  fill-verification fix that made qap15 tractable *and* the qap15 fixture
  generator + `bench_qap15` harness. This branch is cut from current `main` and
  lacks all of it — `INTRAFRONT_MIN_AREA` is still `256*256`, not #92's `256*128`.
- **The qap15 fixture, its generator, `examples/{profile,bench}_qap15.rs`, and the
  two research notes the issue cites for "the full diagnosis" exist on no remote
  branch** (unpushed/lost work). The end-to-end qap15 number that defines the
  target cannot be reproduced here.
- **This container has 4 cores; the issue's numbers are 10-core.** The
  parallel-scaling levers (1 assembly, 2 schur scaling) cannot be validated
  against the issue's targets here.
- The issue itself states **no byte-exact lever closes 3.5×**; faer-class needs a
  policy decision (default-on FMA / static-SQD) the owner deliberately deferred.

I attempted to ask the owner the fixture-strategy and policy questions via
`AskUserQuestion`; the harness failed to deliver it (permission-stream error).
Told to continue, I proceeded autonomously with the **defensible subset**:
additive, default-off, byte-exact-preserving, measurable-here work only — no
unilateral default flips, no unvalidatable parallel retunes.

## Accomplished

**Issue #99 Lever 3 (per-core kernel throughput) — delivered as an opt-in knob.**

- New `examples/bench_dense_front.rs`: self-contained synthetic indefinite front
  (no external fixture), factored through the real `factor_frontal_blocked` path,
  timing nofma/FMA × serial/intrafront with an inertia-equality gate. Fills the
  measurement-harness hole the issue's (missing) `bench_qap15` left.
- New `BunchKaufmanParams::fma_min_front_area: Option<usize>` (default `None`) +
  `effective_front_fma(params, nrow, ncol)` helper. Gated at the single dense
  front-factor entry `factor_frontal_blocked_in_place_with_scratch` by shadowing
  `params` with an fma-flipped clone **only when the gate fires** (unarmed path
  pays nothing). Both multifrontal drivers funnel through this entry, so one
  insertion covers all.
- New `Solver::with_fma_large_fronts(min_area)` — writes straight to
  `numeric_params.bk` (no `NumericParams` field / funnel needed; low churn).
- `None` default ⇒ strict no-op: the production cross-arch bit-exact contract is
```

## Git Status
```
b25f3f8 issue #99: opt-in per-front FMA size gate for large dense fronts (Lever 3)
a17fb7a issue #95: richer update() instability signal + growth-aware/dense-parity refactor recommendations (#97)
f9398fd issue #94: one-norm condition estimate for the unsymmetric LU factor (#98)
f114b5d issue #93: expose the LU element-growth factor via public getters (#96)
f037285 release: feral v0.11.3
```

## Test Status
```
test symbolic::tests::schur_symbolic_supernodes_cover_n ... ok
test symbolic::tests::schur_symbolic_tail_invariant_reversed_user_order ... ok
test symbolic::tests::schur_symbolic_tail_invariant_user_order ... ok
test symbolic::tests::symbolic_factorize_amf_produces_valid_perm ... ok
test symbolic::tests::symbolic_factorize_auto_produces_valid_perm ... ok
test symbolic::tests::symbolic_factorize_default_uses_amf_for_small_matrices ... ok
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

test result: ok. 391 passed; 0 failed; 6 ignored; 0 measured; 0 filtered out; finished in 2.38s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-07-01-03.md)


cargo run --bin bench --release  →  Sparse failure analysis: no failures
Dense ∩ Sparse failure overlap: 0 / 0 / 0
(no oracle timings loaded in this environment; perf partitions N/A)
Default path unchanged (gate defaults None) → residual gate identical to the
2026-07-01-02 baseline (2/2, worst residual 1.26e-16).

examples/bench_dense_front 2955 5 (the new issue-99 harness):
  nofma serial     25586.15 ms  0.34 GFLOP/s  1.00×
  nofma intrafront  8631.85 ms  1.00 GFLOP/s  2.96×
  fma   serial     15422.96 ms  0.56 GFLOP/s  1.66×
  fma   intrafront  5142.82 ms  1.67 GFLOP/s  4.98×
  inertia (+1478,−1477,0) identical across all four variants ✓

```

## Recent Decisions
small-front pivot-drift KKTs (ACOPP14_0001, ACOPP30_0004, FBRAIN3LS_0848/0851)
need nofma to keep their Bunch-Kaufman pivot classification. The existing `fma`
flag is all-or-nothing, so it cannot serve both. A front-size gate does: fast
kernel on the roots, reference kernel on the sensitive small fronts.

**Why opt-in / default `None`.** Enabling FMA changes cross-arch bit patterns
(single vs double rounding) on the gated fronts — the reproducibility policy the
owner deliberately kept opt-in (`dev/tried-and-rejected.md` 2026-04-14). This
session had no authorization to flip a default (the interactive policy question
could not be delivered — harness permission-stream failure), so the lever is
shipped as a knob with measured evidence, leaving the default-on decision to the
owner.

**Evidence.** `examples/bench_dense_front 2955 5` (4-core x86_64): FMA 1.66×
per-core serial (25.6 s → 15.4 s), 1.67× inside intrafront (8.6 s → 5.1 s),
inertia `(+1478,−1477,0)` identical across all four nofma/FMA × serial/intrafront
variants. `tests/issue99_fma_front_gate.rs` (4 tests): gate above threshold is
bit-identical to `fma=true`, below threshold bit-identical to nofma default,
inertia preserved, threshold is exactly `nrow*ncol` with `>=`. Full suite 734
passed / 0 failed; fmt + clippy `-D warnings` clean; bench residual gate
unchanged (default path byte-for-byte identical). Note:
`dev/research/issue-99-dense-front-fma-gate.md`.

**Not closed.** This does not reach faer-class throughput — the best variant is
1.67 GFLOP/s vs ~50–100 for a tuned BLAS-3 core. The structural gap is feral's
memory-bandwidth-bound rank-panel update vs a 2-D register-tiled GEMM
(`dev/plans/dense-kernel-blas3.md`), a multi-session rewrite. Levers 1 (adaptive
`INTRAFRONT_MIN_AREA`) and 2 (assembly parallelism) are parallel-scaling levers
that need the bench corpus + a representative core count to validate no-regression
— not possible on this 4-core box without the (unmerged-PR-#92) qap15 fixtures.

## Recent Tried-and-Rejected
more work.** Step 2's premise (dense-spike bump, contiguous SAXPY) does not hold
for this workload; implementing it would **regress** casctanks, not speed it up.

This also explains Step 1's 15.8×: the old O(bump²) **scan** probed ~731²/2 ≈ 267k
cells per update to find the ~24 that needed work. Removing the scan (Step 1) was
the correct and sufficient fix for the sparse-wide-bump regime; the residual
elimination work is already near-minimal.

**Not generally rejected.** A dense path could still help a genuinely *dense*-spike
basis (the journal's 2026-06-08 tridiagonal/`L⁻¹`-dense worst case). If such a
workload appears, Step 2 should be width-AND-density gated (dense path only when
block density exceeds a threshold), never width-only. For the McCormick LP regime
that motivated discopt#229, Step 1 stands alone.

**Evidence.** `FERAL_BUMP_STATS` aggregate over
`FERAL_LU_TRACE=.../casctanks_trace.txt` (full trace): avg_width=731.3,
max_width=2157, avg_density=0.1346 (all bumps) / 0.0023 (width>500),
avg_axpy=23.9, avg_merge_work=233.4. Step 1 end-to-end: casctanks LP solve
82.4 s → 5.2 s debug (15.8×), optimum −167.751 unchanged. Journal:
dev/journal/2026-06-18-01.org.

## Source Files
```
src/bin/bench.rs
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
tests/growth_flag.rs
tests/issue52_stats.rs
tests/issue64_arrow_ordering.rs
tests/issue65_mc64_fallback.rs
tests/issue67_thin_ordering.rs
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
tests/rook_rescue.rs
tests/rook_rescue_kkt.rs
tests/small_leaf_parity.rs
tests/solver_with_ordering.rs
tests/sparse_postorder.rs
tests/sparse_refined.rs
tests/sqd_fast_path.rs
tests/stress_tests.rs
tests/symbolic_profiler.rs
tests/threshold_consistency.rs
tests/tiny_fast_path.rs
```
