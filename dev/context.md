# FERAL Context (auto-generated)

Generated: 2026-05-16T16:37:14Z

## Latest Session
File: dev/sessions/2026-05-16-02.md
```
# Session 2026-05-16-02

## Goal
Continue the issue-10 "remaining lever" investigation started end of
session 01 (axpy SIMD kernel microbench), and clear independent
single-shot issues from the open queue.

When the user reported every push breaking on GH Actions, scope
expanded to a root-cause CI-hooks fix, then to merging the parallel-
agent deliveries on #25 and #24, and to closing out #33 §3
(`Solver::with_ordering`) since it was the smallest open lever
adjacent to #10's blocker.

## Accomplished

### "Remaining lever" axpy microbench — negative (commit `05722a3` co-bundled)
Built `src/bin/bench_axpy_small.rs` comparing `pulp` /
`scalar` / `unroll4` at lengths [3..128] with 50M iters/measure.
Result: pulp SIMD dispatch ties with plain scalar within 1ns/call
quantization at all small lengths; manual unroll4 is slower. The
compiler auto-vectorizes the scalar form as well as the explicit
SIMD dispatch. *Rules out kernel-call overhead as the bottleneck for
clnlbeam.*  Combined with the prior negative #33 SLB A/B and the
negative #10 MAXFROMM Phase 2 A/B, all three architectural levers
tried against the 1D-banded Mittelmann panel come up within noise.

### CI hooks self-heal (commit `05722a3`)
Root cause of "every push breaks on GH Actions": `core.hooksPath`
was set to `/Users/jkitchin/Dropbox/projects/feral/.git/hooks`
(stale from a prior clone location), pointing at a directory that
does not exist on this machine — so git silently bypassed every
local pre-commit hook. CI caught the fmt drift on every push.

Fix: added a self-healing guard at the top of
`dev/assemble-context.sh` that detects a `core.hooksPath` pointing
nowhere, auto-unsets it, and reinstalls pre-commit. Verified the
guard fires on a synthetic broken state. CLAUDE.md already
documented this exact failure mode as a doc note; the guard
promotes the doc into automation so future sessions cannot
inherit the broken state silently.

### Issue #25 — cascade-break defaults research note (commit `7f096c1`)
Worktree-isolated agent wrote `dev/research/cascade-break.md` (392
lines) deriving (or not deriving) the defaults `ratio = 0.5` and
`eps = 1e-10` from the published literature.

**Conclusion: empirical, not derivable.** `ratio = 0.5` was
calibrated on `pinene_3200_0009` in #8 and cross-validated against
the bimodal `n_delayed_in / expanded_ncol` distribution measured in
#15. Wächter & Biegler 2006 uses `κ⁻_w = 1/3` not `1/2`;
```

## Git Status
```
fa62918 test(scaling): relax MSS1_0009 reason check to either fallback variant
e789ec3 Merge branch 'worktree-agent-ab2727cb5b91921b5'
2efa315 feat(solver): Solver::with_ordering builder (#33 §3)
02e699a feat(scaling): surface MC64 -> InfNorm silent fallback (#24)
7f096c1 docs(cascade-break): document derivation status of ratio=0.5, eps=1e-10 (#25)
```

## Test Status
```
test result: ok. 5 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.00s

     Running tests/tiny_fast_path.rs (target/debug/deps/tiny_fast_path-0aeadf63730f046f)

running 5 tests
test test_gate_tiny_sparse_in ... ok
test test_gate_just_outside_n_tiny ... ok
test test_gate_boundary_n_16 ... ok
test test_determinism_tiny ... ok
test test_solve_parity_tiny_real_matrix ... ok

test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

   Doc-tests feral

running 1 test
test src/symbolic/profiler.rs - symbolic::profiler::SymbolicProfiler (line 27) ... ignored

test result: ok. 0 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.00s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-05-16-02.md)


`cargo run --bin bench --release` (tail -40):

Top 10 worst factor-ratio vs MUMPS:
name                             n    feral(μs)    mumps(μs)      ratio
MUONSINE_0000                 1537        11321          376      30.11
KIRBY2_0007                    458          963          119       8.09
KIRBY2_0006                    458          983          127       7.74
KIRBY2_0008                    458          788          122       6.46
KIRBY2_0009                    458          714          128       5.58
KIRBY2_0010                    458          680          133       5.11
CRESC132_0000                 5314        62572        12266       5.10
KIRBY2_0011                    458          610          120       5.08
GROUPING_0139                  225          472          113       4.18
KIRBY2_0012                    458          465          118       3.94

--- Dense Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
bucket                    count      p90     target  verdict
small-frontal (<200)     147982     1.37     <= 2.0     PASS
medium (<500)            152145     1.74     <= 3.0     PASS

--- Sparse Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---
bucket                    count      p90     target  verdict
small-frontal (<200)     153455     1.73     <= 2.0     PASS
medium (<500)            153560     1.73     <= 3.0     PASS

Both partitions PASS their p90 ratio gates against MUMPS, unchanged
from session 01. Worst-case list is consistent with the long-tail
matrices documented in #12 / #14. No regression from this session's
commits.

Microbench (`src/bin/bench_axpy_small.rs`, 50M iters/measure):
pulp ties scalar within 1ns at lengths 3..128 — see Accomplished
section above.

```

## Recent Decisions
this session via `probe_cascade_perturb`) is fully recoverable
with a single builder call. The `Solver::with_cascade_break_eps`
and `Solver::with_cascade_break` builders are unchanged. Tests
that exercise the gate continue to construct `NumericParams`
with explicit `Some(...)` values.

**Evidence.**
- `probe_cascade_perturb` on `robot_1600_0004` (n=24000):
  cb=off residual 6.24e-7; cb=default residual 1.06e-5;
  cb=fa residual 2.10e+2.
- `probe_cascade_perturb` on `pinene_3200_0009` (n=127995):
  cb=off factor 94 s, residual 2.27e-2; cb=default factor 33 ms,
  residual 7.99e-2 (with inertia preserved); cb=fa factor 36 ms
  but wrong inertia and residual 5.34e+3.
- `cargo test --lib --release` → 256 passed; integration tests
  pass; `cargo clippy --all-targets --release -- -D warnings` clean;
  `cargo fmt --check` clean.
- `cargo run --release --bin bench` Phase 2.8.1 dense+sparse
  small-frontal and medium buckets all PASS; bench numbers within
  noise of session 2026-05-15-06.

**References.**
- `dev/research/cascade-break-l-perturbation-2026-05-15.md` —
  the corrected forensics (the note's original "zero L" proposal
  was rejected; the note now records both the wrong premise and
  the right outcome).
- `dev/tried-and-rejected.md` — 2026-05-15 "Zero L on
  `PerturbToEps`" entry.
- `src/bin/probe_cascade_perturb.rs` — the probe that produced
  the residual numbers.

## Recent Tried-and-Rejected
[3, 4, 5, 6, 8, 10, 16, 32, 64, 128].

Result: pulp ties with plain scalar within 1ns/call quantization at
all lengths 3..128; manual unroll4 is *slower* (0.25-1.00x). The
compiler auto-vectorizes the scalar form as well as the explicit SIMD
dispatch. Kernel-call overhead is *not* the bottleneck.

Implication: the #10 Phase 2 corpus post-mortem's hypothesis that
"the next lever is the scalar rank-1 trailing-update kernel" was
also wrong. Three architectural levers tried against the 1D-banded
Mittelmann panel (#33 SmallLeafBatch driver overhead, #10 MAXFROMM
pivot selection, axpy kernel tightening) all come up within noise.

Remaining levers for the corpus: (a) `Solver::with_ordering(ScotchND)`
to widen the supernode shape (untested, just landed via #33 §3);
(b) supernode amalgamation (symbolic-side restructure); (c) accept
a hardware floor for sequential factorization on this shape.

References: `src/bin/bench_axpy_small.rs`, journal
`2026-05-16-02.org` 11:30 entry.

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
src/bin/diag_fill_parity.rs
src/bin/diag_fill_tail.rs
src/bin/diag_inertia_mismatch.rs
src/bin/diag_leaf_profile.rs
src/bin/diag_max_ncol.rs
src/bin/diag_mc64_cycles.rs
src/bin/diag_mittelmann.rs
src/bin/diag_orbit2_quotient.rs
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

(truncated from      352 lines to 350 line budget)

(truncated from      352 lines to 350 line budget)
