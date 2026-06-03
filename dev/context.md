# FERAL Context (auto-generated)

Generated: 2026-06-03T18:01:39Z

## Latest Session
File: dev/sessions/2026-06-03-05.md
```
# Session 2026-06-03-05

## Goal
Issue #71 (follow-up to session-04's "Next Session Should"): relocate the 144
throwaway `src/bin` diagnostic binaries out of the default build/test set so
root `cargo build` / `cargo test` / `cargo clippy` stop compiling them on
every invocation. Keep the single keeper (`bench.rs`) and the protocol
command `cargo run --bin bench --release` unchanged.

## Accomplished
- **New crate `crates/feral-diagnostics/`** (`publish = false`,
  `autobins = true`, depends on `feral` + the six ordering crates + serde/
  serde_json/pulp/rayon). All 144 diagnostics `git mv`'d into its `src/bin/`
  (history preserved); only `bench.rs` remains in the root package.
- **Workspace member** added in root `Cargo.toml`. Because the diagnostics
  crate is a member but **not** a dependency of `feral`, root commands
  without `-p`/`--workspace` compile the `feral` package only. Verified:
  root `cargo build` compiles **zero** diagnostics.
- **CI updated** (`.github/workflows/ci.yml`):
  - `stress-smoke` now selects `bench_one_matrix` (build) and
    `probe_fma_kernel` (run) with `-p feral-diagnostics`.
  - `check` job adds `cargo clippy -p feral-diagnostics --all-targets
    -- -D warnings` and `cargo test -p feral-diagnostics`, keeping the
    diagnostics lint-clean and their 2 test sets running where it is cheap
    (Linux), having dropped them from the slow local path.
- **Verification** (all green):
  - root `cargo build`: no diagnostics compiled.
  - `cargo build -p feral-diagnostics`: all 144 compile, **zero warnings**
    (after adding the six ordering-crate deps — see Abandoned/fix below).
  - `cargo clippy -p feral-diagnostics --all-targets -- -D warnings`: clean.
  - diag tests (targeted): `diag_cond_parity` 5 passed, `diag_schur_parity`
    4 passed.
  - `cargo build --bin bench` from root: resolves (protocol cmd unchanged).
  - `cargo run -p feral-diagnostics --bin probe_issue67_thin`: runs (usage).
  - root `cargo clippy -- -D warnings`: clean (now lib + `bench` only).
  - root `cargo test`: <RESULT PENDING — recorded below>.

## Benchmark Results
Not run. #71 is a **build-layout change only** — no library, kernel, or
solver source was modified (the move is `git mv` of binary crates plus
manifest/CI edits). The benchmark exercises the `feral` library, whose code
is byte-for-byte unchanged, so its inertia / residual / timing columns
cannot move. The last recorded numbers are session-04's #67 filtered bench
(137/137 inertia match vs MUMPS, worst residual 4.22e-12) and remain valid.

```
(no bench run — see rationale above; library unchanged from session-04)
```

## Decisions Made
```

## Git Status
```
3391d6a fix(ordering): thin-large default prefers AMF up to n≤100k (closes #67) (#70)
5bcfecc Merge pull request #68 from jkitchin/claude/issue-63-diagnosis
892d7ef Merge remote-tracking branch 'origin/main' into claude/issue-63-diagnosis
0673f1b Merge pull request #69 from jkitchin/claude/issue-65-mc64-scaling
cfd8f68 docs(session): checkpoint 2026-06-03-03 — MC64 scaling fallback (#65)
```

## Test Status
```
test symbolic::tests::test_contrib_sizes_nonnegative ... ok
test symbolic::tests::symbolic_factorize_scotch_produces_valid_perm ... ok
test symbolic::tests::test_perm_inverse_consistency ... ok
test symbolic::tests::test_symbolic_factorize_basic ... ok
test symbolic::tests::test_symbolic_factorize_dense ... ok
test symbolic::tests::test_symbolic_factorize_kkt ... ok
test numeric::factorize::tests::issue_5_mss1_iter0_inertia_wanders_under_delta_w_sweep ... ok
test symbolic::tests::is_arrow_bordered_rejects_many_hubs ... ok
test symbolic::tests::issue_3_scotchnd_on_kkt_resolves_to_amd_when_bisection_degenerates ... ok
test symbolic::tests::choose_adaptive_routes_arrow_to_amf ... ok
test symbolic::tests::choose_adaptive_rules ... ok
test scaling::tests::auto_keeps_mc64_on_vesuvia_0000 ... ok
test scaling::tests::auto_keeps_mc64_on_vesuviou_0000 ... ok
test numeric::factorize::tests::issue_5_mss1_zero_tol_sweep_diagnostic ... ok
test symbolic::tests::issue_3_auto_on_kkt_routes_via_pick_default_method ... ok
test numeric::factorize::tests::issue_5_mss1_pivot_threshold_sweep_diagnostic ... ok
test scaling::tests::pick_scaling_strategy_routes_clnlbeam_to_infnorm ... ok

test result: ok. 322 passed; 0 failed; 6 ignored; 0 measured; 0 filtered out; finished in 0.41s

```

## Benchmark
```
(skipped: pass --with-bench to re-run; sourced from dev/sessions/2026-06-03-05.md)

Not run. #71 is a **build-layout change only** — no library, kernel, or
solver source was modified (the move is `git mv` of binary crates plus
manifest/CI edits). The benchmark exercises the `feral` library, whose code
is byte-for-byte unchanged, so its inertia / residual / timing columns
cannot move. The last recorded numbers are session-04's #67 filtered bench
(137/137 inertia match vs MUMPS, worst residual 4.22e-12) and remain valid.

(no bench run — see rationale above; library unchanged from session-04)

```

## Recent Decisions
  disabled and all 144 binaries enumerated as explicit `[[bin]]` blocks
  with `required-features` — verbose and brittle. A separate crate is the
  idiomatic non-default-build mechanism.
- Deleting the diagnostics: they are the audit trail behind shipped
  decisions (probe_issue67_thin, probe_issue65_corpus, …) and are cheap to
  keep once out of the default build.

Constraints preserved:
- `bench.rs` stays in the root package so `cargo run --bin bench --release`
  (the session protocol command) is unchanged.
- CI `stress-smoke` selects `bench_one_matrix` / `probe_fma_kernel` with
  `-p feral-diagnostics`. The `check` job adds
  `cargo clippy -p feral-diagnostics --all-targets -- -D warnings` and
  `cargo test -p feral-diagnostics` so the diagnostics stay lint-clean and
  their 2 test sets keep running — the bar is kept where it is cheap
  (Linux CI) and dropped only on the slow local path (pre-commit clippy /
  local `cargo test`).
- `feral-diagnostics` is absent from the explicit release publish list and
  is marked `publish = false`.

No library or solver source changed; this is a build-layout change only.

Evidence: crates/feral-diagnostics/Cargo.toml, root Cargo.toml workspace
members, .github/workflows/ci.yml, dev/journal/2026-06-03-05.org. Verified:
root `cargo build` compiles no diagnostics; `cargo build -p
feral-diagnostics` compiles all 144 cleanly; `cargo clippy -p
feral-diagnostics --all-targets -- -D warnings` clean; the 2 diag test sets
pass (5 + 4); root `cargo clippy -- -D warnings` clean; `cargo run --bin
bench` and `cargo run -p feral-diagnostics --bin probe_issue67_thin` both
resolve.

## Recent Tried-and-Rejected
   destroyed), inertia scrambled. Strictly worse. Force-accept-and-report-zeros
   is the useful behavior: it signals singularity so pounce escalates δ_w.

3. Any principled "better inertia" change. The ordering that wins (metis)
   reports a MORE pessimistic, LESS correct inertia (neg 255 ≠ 252 expected) on
   the singular matrix; that makes pounce regularize earlier and escape a frozen
   2.30e-8 fixed point. There is no known-correct inertia change that fixes
   scrs8 — "correct" inertia (amf) is what under-regularizes into the stall.

4. Ordering-class heuristic (route this KKT class to metis/scotch). Not pursued:
   the issue itself calls it "papering over the symptom," and it risks the
   cascade-break don't-regress set (robot_1600, NARX_CFy, marine_1600,
   rocket_12800, pinene_3200).

Conclusion: the durable fix is the δ_w / inertia-acceptance interaction
(pounce-side or joint), not FERAL factorization accuracy. Full analysis:
dev/research/issue-63-nearsingular-ordering-diagnosis.md;
dev/journal/2026-06-03-02.org; probe src/bin/probe_issue63_nearsingular.rs.
Future sessions: do NOT re-attempt a FERAL-only fix for scrs8 without first
re-checking these four dead ends.

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
tests/column_renumbering_parity.rs
tests/column_renumbering.rs
tests/delayed_pivoting.rs
tests/dense_fast_path.rs
tests/dense_ldlt.rs
tests/factor_scratch_parity.rs
tests/factor_workspace_parity.rs
tests/fine_grained_delay.rs
tests/fma_opt_in_roundtrip.rs
tests/growth_flag.rs
tests/issue_15_cascade_arm_gate.rs
tests/issue_17_robot_1600_cascade_off.rs
tests/issue_18_narx_cfy_cascade_off.rs
tests/issue_2_kkt_ls_init.rs
tests/issue_38_static_pivot.rs
tests/issue_46_saddle_kkt_cascade.rs
tests/issue_55_delay_budget.rs
tests/issue_55_n_tiny_counter.rs
tests/issue52_stats.rs
tests/issue64_arrow_ordering.rs
tests/issue65_mc64_fallback.rs
tests/issue67_thin_ordering.rs
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
tests/sqd_fast_path.rs
tests/stress_tests.rs
tests/symbolic_profiler.rs
tests/threshold_consistency.rs
tests/tiny_fast_path.rs
```
