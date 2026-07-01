# FERAL Context (auto-generated)

Generated: 2026-07-01T01:13:24Z

## Latest Session
File: dev/sessions/2026-06-21-01.md
```
# Session 2026-06-21-01

## Goal

Issue #87: eliminate the residual **O(bump²)** Forrest–Tomlin row-elimination on
wide (dense-spike) McCormick simplex bases — the Step-2 follow-up of discopt#229
that was parked as a research spike. The user asked for the **most robust and
performant** solution, chosen after digging into the code and the literature, and
verified with before/after timing + scaling experiments.

## Accomplished

### Diagnosis (the key insight)

The current `SparseLu::update`/`eliminate_bump` is **not** a Forrest–Tomlin update
— it sets the spike into `U`'s column `r` and runs a full partial-pivoting bump
re-triangularization in fixed pivot order. It eliminates the dense spike
**column**, touching `O(bump)` rows with cascading fill, so both the work **and**
the recorded eta are `O(bump²)` on a dense spike. True FT eliminates the single
pivotal **row** after a symmetric permutation (the permuted diagonal pivots are
the old nonzero `U` diagonals — dodging the zero-superdiagonal landmine that
reverted the 2026-06-08 column-shift attempt), which is `O(bump)` for sparse `U`.

### Route chosen: logical-permutation Forrest–Tomlin

Carry one evolving `uperm` (pivot-position ↔ triangular rank), applied once per
solve; never relabel `U`'s indices or prior etas. Rejected the column-ordering
lever (no asymptotic change) and physical permutation (stales prior etas →
`O(k²·bump)`; cyclic shift as per-eta swaps → `O(k·bump)` solve blow-up).
Clean-room from Forrest–Tomlin 1972, Reid 1982, Schork–Gondzio ERGO-17-002
(`BASICLU` is GPL — paper only). Docs: `dev/research/ft-row-elimination-design-2026-06-21.md`,
`dev/plans/ft-row-elimination-2026-06-21.md`, `dev/decisions.md`.

### Implementation (P1+P2)

- **P1** (`a676aaf`): `uperm`/`uperm_inv`; `usolve`/`ut_solve` walk `U` in rank
  order (identity at factor ⇒ byte-identical).
- **P2** (`ebaeca6`): rewrote `sparse_update.rs` — `set_column_r` (spike into
  other rows), `shift_uperm` (symmetric cyclic shift, `O(bump)`),
  `eliminate_pivot_row` (single-row sparse forward sweep via a rank min-heap, one
  `FtOp::Axpy` per sub-diagonal). Widened `u_above` to all off-diagonal holders;
  diagonal-first-aware row helpers; removed `FtOp::Swap`.
- **Bug fixed during bring-up**: the eliminated pivot column's FP residual
  `vrc − mult·piv ≠ 0` re-enqueued the column forever (infinite loop / OOM, exit
  137). Fixed by clearing `rw[c]` exactly and skipping the pivot's own diagonal.

### Evidence (before → after, release, this host)

| probe | metric | BEFORE | AFTER |
|---|---|---:|---:|
```

## Git Status
```
7b4669b issue #94: LU-basis one-norm condition estimate via ftran/btran
01a1527 issue #94: extract shared Hager–Higham driver + HagerHighamOperator trait
f037285 release: feral v0.11.3
a5c789f issue #89: FT update u_above reindex O(m³)→O(m²) + true per-update cost counter (#90)
a9cea82 issue #87: Forrest–Tomlin row-elimination LU update (O(bump²) → O(bump)) (#88)
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

test result: ok. 381 passed; 0 failed; 6 ignored; 0 measured; 0 filtered out; finished in 1.90s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-06-21-01.md)


cargo run --bin bench --release  (symmetric indefinite gate — unaffected)
  Residual pass: 2/2 (100.0%)   Worst residual: 1.26e-16
  Dense/Sparse failure analysis: no failures

casctanks_ft_update/chain_144_updates_m2169
  BEFORE: 16.770–17.036 ms   AFTER: 1.6413–1.6748 ms   (10.2×)

lu_wide_bump_probe (dense-spike tridiagonal, AFTER):
   m    us/update   us/update/m   eta_ops/upd
  250     102.11       0.41           52
 1000     675.41       0.68          116
 4000   69415.61      17.35         ~120

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
