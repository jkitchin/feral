# FERAL Context (auto-generated)

Generated: 2026-06-21T15:43:20Z

## Latest Session
File: dev/sessions/2026-06-19-01.md
```
# Session 2026-06-19-01

## Goal

Two pieces of work:

1. Cut release **v0.11.1** (patch) for the sparse LU Forrest–Tomlin
   bump-elimination perf already on `main`.
2. Answer "is there any reason to think memory allocation would speed up
   performance anywhere?" — and, since the answer was yes, act on it under a
   measure-then-pool gate.

## Accomplished

### v0.11.1 release

- Bumped all six version strings via `scripts/release-checklist.sh bump 0.11.1`;
  cut `CHANGELOG [0.11.1] - 2026-06-19` from `[Unreleased]`.
- Committed (`e5587b3`), tagged `v0.11.1`, pushed, created the GitHub release.
- `release.yml` (crates.io) and `python-wheels.yml` (PyPI) both green; verified
  crates.io `feral` max_version = 0.11.1 and PyPI `feral-solver` = 0.11.1.

### LU-update allocation pooling (the allocation question)

Investigation found the multifrontal **factor** path is already
allocation-optimized (FactorWorkspace etc.; two further pooling attempts were
falsified there), but the **LU basis-update** path (`src/lu/sparse_update.rs`)
— the code v0.11.1 touched — pooled nothing, unlike its sibling
`sparse_solve.rs`.

**Phase 0 (measure):** new `tests/lu_update_alloc_probe.rs` (counting
`#[global_allocator]`) on the casctanks wide-bump trace (discopt#229, in-tree
fixture m=2169, 144 updates): **~1804 allocs + 176 reallocs per update** against
an **85.8 µs/update** budget (`lu_update_trace` bench 12.351 ms / 144). Gate
passed; the bench delta was the arbiter (pooling has been falsified here before
when bookkeeping > malloc saved).

**Phase 1 Step 1 — bump-loop pools** (`9cf5a96`): `pivot_scratch`,
`targets_scratch`, `row_pool` (+`row_sub`→`row_sub_into`), `col_rows_pool`.
allocs/update 1804 → 636 (−65%); bench 12.351 → 9.955 ms (**−19.0%**, p<0.05).

**Phase 1 Step 2 — saved-row snapshot pool** (`edf1f0f`): `saved_scratch` +
`saved_pool`. allocs/update 636 → 82.5 (−95% vs baseline); bench 9.955 →
8.545 ms (**−14.8%**). Hardened the probe into a regression guard (<250
allocs/update).

**Cumulative:** 1804 → 82.5 allocs/update (−95%); **12.351 → 8.545 ms = −30.8%**
on the casctanks replay. Numerics **bit-identical** at every step
(`worst_true_residual=9.095e-13`, `worst_sparse_vs_dense=0.000e0`; `FtOp` eta
sequence and pivot choices unchanged). Full suite green; `cargo fmt` +
```

## Git Status
```
ebaeca6 issue #87 P2: Forrest-Tomlin row-elimination update (O(bump²) → O(bump))
a676aaf issue #87 P1: add uperm_inv logical-permutation order, route U-solves through it
a34367e issue #87: diagnose O(bump²) FT update, choose logical-permutation FT, add baseline probe
75b0322 release: feral v0.11.2
2ed962d docs(session): 2026-06-19-01 checkpoint — LU-update allocation pooling
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

test result: ok. 371 passed; 0 failed; 6 ignored; 0 measured; 0 filtered out; finished in 2.43s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-06-19-01.md)


lu_update_trace (casctanks FT-update chain, 144 updates, m=2169):
  baseline (v0.11.1):  12.351 ms
  after Step 1:         9.955 ms   (-19.0%)
  after Step 2:         8.545 ms   (-14.8%; -30.8% cumulative)

alloc probe (allocs / reallocs / bytes per update):
  baseline:  1804.2 / 176.3 / 128610
  Step 1:     636.5 /  68.7 /  63428
  Step 2:      82.5 /  80.7 /  23560

`cargo run --bin bench --release` (full KKT corpus; factorization path,
unaffected by the LU-update pooling — confirms no regression):

--- Dense Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
bucket                    count      p90     target  verdict
small-frontal (<200)     147982     1.58     <= 2.0     PASS
medium (<500)            152145     2.09     <= 3.0     PASS

--- Sparse Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
bucket                    count      p90     target  verdict
small-frontal (<200)     153455     1.48     <= 2.0     PASS
medium (<500)            153560     1.48     <= 3.0     PASS

Top worst factor-ratio vs MUMPS unchanged (KIRBY2_0007 9.95, CRESC132 6.69, …).

```

## Recent Decisions
once per solve; `U`'s stored indices, `L`, `P`, `Q`, and prior etas stay in fixed
pivot-position coordinates and are never relabeled. Removed `FtOp::Swap` — the FT
update does no in-bump magnitude pivoting.

**Why this over the alternatives.**
- The old scheme eliminated the dense spike *column*, touching O(bump) rows with
  cascading fill ⇒ O(bump²) work **and** an O(bump²) eta (which then slows every
  warm solve). Eliminating the pivotal *row* is O(bump) for sparse `U`, with an
  O(bump) eta.
- The symmetric permutation places the **old nonzero `U` diagonals** on the bump
  diagonal, so it dodges the zero-superdiagonal-pivot landmine that reverted the
  2026-06-08 column-shift Hessenberg attempt. This is the distinction that makes
  the route correct on a sparse `U`.
- A *physical* permutation was rejected: relabeling prior etas is O(k²·bump) over
  a chain, and encoding the cyclic shift as per-eta swaps reintroduces the PFI
  O(k·bump) solve blow-up. The logical `uperm` applied once per solve avoids both.
- The column-ordering lever (discopt#229's other suggestion) changes no
  asymptotic and is workload-specific; kept as a possible complement, not the fix.

**Stability.** FT has no in-bump pivoting, so a small bump diagonal can grow
elements; this is caught by the existing `growth`/`max_growth` monitor and routed
to `NeedsRefactor` (authoritative verdict = fresh factor). A Schork–Gondzio
"permute-when-possible" stability/sparsity refinement is recorded as future work,
not required for correctness.

**Evidence.** `lu_wide_bump_probe` dense-spike: per-update 44–148× faster
(m=4000: 10.2 s → 69 ms), eta O(m²)→O(m) (2.09M → ~120). `casctanks_ft_update`
144-chain: 16.88 ms → 1.66 ms (10.2×). Localized-spike (`lu_update_probe`) and
the full suite unchanged/green. Clean-room from Forrest–Tomlin 1972, Reid 1982,
Schork–Gondzio ERGO-17-002 (`BASICLU` is GPL — paper only).

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
