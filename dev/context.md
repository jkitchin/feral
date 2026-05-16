# FERAL Context (auto-generated)

Generated: 2026-05-16T17:12:39Z

## Latest Session
File: dev/sessions/2026-05-16-03.md
```
# Session 2026-05-16-03

## Goal

Close GitHub issue #31 (M6: inertia certification on near-singular —
sweep `eps_pow`, find detection boundary). Build a parametric sweep
over `near_singular_eps_<p>` for p ∈ {6..14}, identify the boundary at
which the default Bunch-Kaufman pivot threshold stops detecting the
null pivot, and either certify the current threshold or propose a
better criterion.

## Accomplished

- Extended `external_benchmarks/stress/synth.py` with a parametric
  generator family `near_singular_eps_<p>` for p ∈ {6..14}
  (seed = 100+p, n=100, one λ=10^-p, 99 healthy λ in [0.5, 3.0]).
  Existing `near_singular_eps9` / `near_singular_eps12` rows preserved.
- New diagnostic binary `src/bin/diag_near_singular_sweep.rs` that
  runs `Solver::new()` factor + `solve_refined` on each matrix and
  reports `(status, inertia, min|D_ii|, rel_res, pivtol)`.
- Wrote `dev/research/inertia-near-singular-certification.md` with the
  full sweep table, theory, bound argument, and the rejected
  alternative criterion.
- Added regression row `near_singular_eps_7` to
  `external_benchmarks/stress/manifest.tsv` (per the issue's
  acceptance criterion of "boundary + 1").

### Key finding

The issue's premise was wrong. The sweep shows feral reports
`inertia.zero == 0` for *every* p ∈ {6..14}, including the canonical
`near_singular_eps9` and `near_singular_eps12` matrices — they were
never actually being detected as null. The boundary is p = 6.

| p  | status   | (pos, neg, zero) | min &#124;D_ii&#124; | rel_res    |
|----|----------|------------------|----------------------|------------|
|  6 | Success  | (48, 52, 0)      | -1.25e+1             | 4.32e-16   |
|  9 | Success  | (59, 41, 0)      | -1.47e+1             | 4.55e-16   |
| 12 | Success  | (53, 47, 0)      | -2.12e+1             | 5.74e-16   |
| 14 | Success  | (43, 57, 0)      | -9.55e+0             | 2.14e-15   |

The factorization is stable (residuals 2-6 × 10^-16 after iterative
refinement) but the null space is invisible to a magnitude-based
pivot test. This is the expected behavior for a single small
eigenvalue dispersed across a random orthonormal basis Q — see
Higham 2002 Ch. 11 and the BK bound `|d_k| ≥ (1-α²) σ_min(A_22)`
which lower-bounds the pivot magnitude by the *trailing* submatrix's
smallest singular value, not the input matrix's.

## Benchmark Results
```

## Git Status
```
d3b031d research(#10): supernode-shape ordering A/B — NEGATIVE on Mittelmann panel
af53f4e chore(context): refresh dev/context.md with new test+bench excerpt
84238ac chore(session): 2026-05-16-02 -- CI hooksPath fix, #24/#25/#33 §3, axpy negative
fa62918 test(scaling): relax MSS1_0009 reason check to either fallback variant
e789ec3 Merge branch 'worktree-agent-ab2727cb5b91921b5'
```

## Test Status
```
test result: ok. 5 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.00s

     Running tests/tiny_fast_path.rs (target/debug/deps/tiny_fast_path-510ea496d233abcf)

running 5 tests
test test_gate_just_outside_n_tiny ... ok
test test_gate_tiny_sparse_in ... ok
test test_solve_parity_tiny_real_matrix ... ok
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
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-05-16-03.md)


Did not run `cargo run --bin bench --release` — this session only
added a diagnostic binary, a generator extension, a research note,
and a manifest row. No solver code touched, no benchmark-relevant
changes. The sweep binary's own output is in the research note.

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
src/bin/diag_near_singular_sweep.rs
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
