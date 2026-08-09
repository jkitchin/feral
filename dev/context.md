# FERAL Context (auto-generated)

Generated: 2026-08-09T02:38:45Z

## Latest Session
File: dev/sessions/2026-08-09-01.md
```
# Session 2026-08-09-01

## Goal

Evidence-based kernel performance pass (user request: vectorization/SIMD
opportunities, pure Rust, every change (1) correct, (2) faster,
(3) bit-identical across platforms). Staged plan: measure-first, then
packed-kernel SIMD, pack-buffer pooling, small-front eager-path SIMD.

Environment caveat: x86_64 4-core AVX2+AVX512 container; the 154k-matrix
corpus is NOT present, so evidence comes from parity suites +
bench_schur_micro / bench_dense_front / perf_probe on tests/data
fixtures + a new hicks-like chain fixture. **All perf numbers below are
x86; aarch64 M-series revalidation is pending (see follow-ups).**

## Accomplished

1. **x86 pulp dispatch fix (f0be9fa)** — `dispatch_nofma`/`dispatch_fma`
   called `k.with_simd(v3)` instead of `pulp::Simd::vectorize(v3, k)`,
   so every pulp kernel on x86 ran its AVX intrinsics as outlined
   function calls since Phase 2.4.2 (invisible on aarch64 where NEON is
   baseline). Strided kernel: 0.45 → 4.69-7.00 GFLOP/s (~10×).
   Bit-identical by construction; all suites + golden digests unchanged.

2. **Explicit pulp SIMD packed trailing update (41c35b5)** — the
   default BLAS-3 tile walk had compiled fully scalar on x86 (objdump:
   ymm=0, packed SSE=0). Moved into
   `schur_kernel::packed_schur_tiles_{nofma,fma}`, one dispatch per
   panel; scalar walk kept as `FERAL_PACKED_SIMD=0` fallback. Also
   repairs the opt-in FMA path (scalar `f64::mul_add` → libm call, was
   a 3× x86 slowdown since 2026-07-01). New `tests/golden_bits.rs`
   hardcoded digests = cross-arch tripwire.

3. **Work gate for degenerate panels (8358d1b)** — panels below 1024
   multiply-subtracts stay on the inline scalar walk
   (`FERAL_PACKED_SIMD_MIN_WORK` override): the ~100-200 ns dispatch
   boundary can't be amortized there (`examples/bench_packed_tiny`),
   and un-gated SIMD showed a warm-median artifact on HAHN1/AVION2
   that in-kernel timing proved was not in-kernel cost (journal 04:10).

4. **Pack-buffer pooling, issue #128 rest (484bda7)** — `PackPool` on
   `FactorScratch`, serial path only; dirty-pool byte-parity sweep in
   the unit test.

5. **Eager-path SIMD tried and rejected (f509a18)** — full
   implementation measured FLAT everywhere (eager n=512: 11.21 vs
   11.23 ms) because the plain eager loops already autovectorize;
   reverted same session per the pre-registered criterion, keeping the
   byte-identical de-duplication of `do_1x1_pivot`'s twin loops.
   Recorded in tried-and-rejected.
```

## Git Status
```
f509a18 refactor(kernel): dedup do_1x1_pivot update loops; record eager-SIMD rejection
484bda7 perf(kernel): pool packed-update A/B pack buffers in FactorScratch
8358d1b perf(kernel): work-gate the packed SIMD tile kernel; add tiny-call probe
41c35b5 perf(kernel): explicit pulp SIMD tile kernel for the packed trailing update
f0be9fa perf(kernel): route x86 pulp dispatch through Simd::vectorize
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

test result: ok. 407 passed; 0 failed; 6 ignored; 0 measured; 0 filtered out; finished in 2.95s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-08-09-01.md)


`cargo run --bin bench --release` (corpus absent — synthetic + 2
regression fixtures only): 8 synthetic matrices benchmarked; KKT
fixtures 1/1 dense inertia+residual pass (worst 1.14e-15), sparse 2/2
vs MUMPS (worst 1.26e-16). Phase 2.8.1 partitions N/A (no oracle
timings in container).

Kernel/fixture evidence (3-run medians, sequential, this container):

| measure | session start | session end |
|---|---:|---:|
| bench_dense_front 2955 nofma serial | 4236 ms (2.0 GF/s) | 1517-1556 ms (5.6 GF/s) |
| bench_dense_front 2955 nofma intrafront | 1798 ms (4.8 GF/s) | 637-679 ms (12.7-13.5 GF/s) |
| bench_dense_front 2955 fma intrafront | 4214 ms | 572-609 ms (14-15 GF/s) |
| bench_dense_front 512 nofma serial | 39.4 ms | 10.0-10.9 ms |
| strided micro 2048²·64 | 0.45 GF/s | 4.69 GF/s |
| twirism1 warm factor | 4176-4217 µs | 2066-2195 µs |
| hydcar20 | 295-300 µs | 206-218 µs |
| sawpath | 826-830 µs | 671-745 µs |
| vesuvio | 2108-2284 µs | 1857-2071 µs |
| chain1200 (new fixture) | 985 µs (post-Stage-1) | 847-961 µs |
| hahn1 | 839-842 µs | 739-762 µs |
| avion2 | 39 µs | 34-35 µs |

No fixture worse than session start. All byte-exactness gates green
throughout (407 lib tests, 83 test binaries, golden digests stable
across default / FERAL_PACKED_SIMD=0 / FERAL_PACKED_SCHUR=0).

```

## Recent Decisions
nofma-serial 4236 → 1517-1556 ms (2.7-2.8×, 2.0 → 5.6 GF/s);
nofma-intrafront 1798 → 637-679 ms (12.7-13.5 GF/s); fma-intrafront
4214 → 572-609 ms (14-15 GF/s, fastest config); n=512 3.9×. Warm
fixtures: twirism1 −46%, hydcar20 −26%, sawpath −18%. Small fixtures
restored to baseline-or-better by the work gate.

**aarch64 status.** NOT yet measured on M-series (this container is
x86). The structural bit-identity argument plus golden digests
guarantee correctness there; performance must be re-validated —
`FERAL_PACKED_SIMD=0` is the one-env-var mitigation if the NEON
codegen regresses vs the old autovectorized walk.

## 2026-08-09 — Pack-buffer pooling on the serial packed path (issue #128 rest)

**Decision.** `FactorScratch` carries a `PackPool {apack, bpack0,
bpack1}`; the serial packed dispatch path reuses it (mem::take'd
around the scratch borrows), the intra-front rayon path keeps
per-range fresh allocations (a shared pool cannot cross the parallel
split; multi-slot pooling is on the tried-and-rejected list).
`bpack0`/`bpack1` re-zero on reuse via `clear()+resize` (their zeros
are load-bearing for out-of-range column lanes); `apack` skips the
re-zero only when its length matches (every slot including padding is
overwritten). Parity test carries a deliberately dirty pool across all
shapes.

**Evidence (3-run warm medians).** chain1200 (hicks-like
block-tridiagonal synthetic, added after the pounce#552 chain-KKT
report) 985 → 847-961 µs (−12%); AVION2 −6%; twirism1 −6%; HYDCAR20
−4%; VESUVIO −4%; HAHN1 −2%. Byte-exact (dirty-pool parity sweep +
golden digests unchanged).

## Recent Tried-and-Rejected
plain `for i in j..n { a[j*n+i] -= a[k*n+i]*alpha }` loops are
textbook-autovectorizable, and the eager path's remaining time is
pivot search + memory traffic, not multiply-subtract throughput.
Explicit lanes duplicated what LLVM already did. This matches the
2026-05-16 finding (pulp == scalar == manual unroll at lengths 3..128)
at the whole-front scale.

**What was kept.** The de-duplication refactor (shared scalar
`rank1_scale_update_argmax`, byte-identical, golden digests unchanged)
stays; the pulp kernel, its gate/env var, the dedicated parity test,
and the A/B example were removed.

**Lesson.** The small-front/MA57 gap is NOT lane width in the eager
update. Remaining suspects, in evidence order: per-front fixed
overhead (assembly/scatter/build-row, 8.8-14.8% on the small
fixtures), pivot-search scans, `scalar_pivot_step` in blocked fronts,
and the delayed-pivot cascade (per-factor-cost-cluster mechanism A).
Any retry of eager-path SIMD must first show a front-level profile
where the update loops are >30% of eager time AND not already
vectorized in the disassembly.

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
tests/threshold_consistency.rs
tests/tiny_fast_path.rs
```
