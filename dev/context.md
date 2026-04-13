# FERAL Context (auto-generated)

Generated: 2026-04-13T02:04:36Z

## Latest Session
File: dev/sessions/2026-04-12-02.md
```
# Session 2026-04-12-02 — Phase 2.2.1 + Phase 2.2.2

## Goal

Land MC64 matching-based scaling (Phase 2.2.1) and, if the resulting
regressions warrant, the minimum pivot-rejection machinery (Phase 2.2.2)
to close the ACOPP30 gap exposed by the Phase 2.1.2 sanity check.
Follow the feature lifecycle strictly: research → plan → tests → code →
validate. Delegate each step to an agent with its own context to keep
the main context clean across a large workload.

## Accomplished

### Phase 2.2.1 — MC64 matching-based scaling

1. **Hungarian matching kernel** (`4e3448a`) — pure-Rust shortest-augmenting-path
   bipartite matching with dual variables, mirroring SPRAL's
   `scaling.f90:810-1171`. Custom `IndexHeap` with decrease-key. Three-phase
   greedy initialization. 6 unit tests including hand-derived oracles.

2. **MC64 wrapper** (`321568e`) — `compute_symmetric` implements the 9-step
   Duff-Koster 2001 + Duff-Pralet 2005 pipeline: pattern expansion, log
   transform, column-max normalization, Hungarian call, dual unwinding,
   symmetric average `s[i] = exp((u[i] + v[i] - cmax[i]) / 2)`, safety
   guards. Three previously-ignored tests (`mc64_scaling.rs`) pass.
   Hand-oracle: `diag(2,3,5)` → `s = [1/√2, 1/√3, 1/√5]`, scaled diagonal
   `= [1,1,1]`.

3. **Symbolic integration** (`67954d9`) — `SymbolicFactorization` carries
   `scaling` (user-order), `scaling_pivot_order` (pivot-permuted),
   `scaling_info`. `ScalingStrategy::default()` flipped from `Identity` to
   `Mc64Symmetric`. No existing test broke — small test matrices either
   are well-scaled (MC64 produces near-identity) or use scale-invariant
   assertions.

4. **Assembly + solve-side scaling** (`0a13515`) — frontal assembly multiplies
   each scattered entry by `s_pivot[gi] * s_pivot[gj]` producing `D·A·D`.
   `solve_sparse` wraps the core with `b' = D·b; y = core(b'); x = D·y`
   (same vector both ends, not inverse). `solve_sparse_refined` inherits
   the behavior for free. `ScalingInfo::NotApplied` short-circuits the
   pre/post-scale on the happy path.

5. **Step 8 validation sweep** (`8a95825`, `3d0716b`) — catastrophic finding:
   ACOPP30_0000 residual went from pre-fix 2.84e+16 to post-MC64 **2.27e+46**
   — a 30-order regression. Six of seven sanity-panel matrices improved
   by 2-10 orders; ACOPP30 was the outlier. Diagnostic investigation
   traced the root cause to a design mismatch: MC64 shrinks the worst
   pivots from the Phase 1 `1e-8` KKT floor down to `~3.6e-10`; these
   survive `zero_tol = f64::EPSILON` and `ForceAccept` cascades them
   through the LDL^T solve. MC64 itself, the wrapper, symbolic plumbing,
```

## Git Status
```
f47316e Phase 2.2.2 Steps 7-9: validation sweep and ACOPP30 recovery
09955c2 Phase 2.2.2 Steps 5-6: MC64 callers opt in; green test sweep
1f7d878 Phase 2.2.2 Step 4: implement Duff-Reid 2x2 growth bound
2b086bd Phase 2.2.2 Step 3: implement column-relative 1x1 pivot threshold
286e506 Phase 2.2.2 Steps 1-2: pivot_threshold field and failing tests
```

## Test Status
```
test result: ok. 8 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.02s

     Running tests/threshold_consistency.rs (target/debug/deps/threshold_consistency-c660296127e8afca)

running 6 tests
test polak6_0021_residual_after_threshold_fix ... ignored
test factor_inertia_force_accept_implies_solve_skip_invariant ... ok
test factors_carry_zero_tol_from_params ... ok
test dense_solve_skips_zero_pivots_rank_deficient ... ok
test refinement_does_not_amplify_error_on_rank_deficient_matrix ... ok
test sparse_solve_skips_zero_pivots_rank_deficient ... ok

test result: ok. 5 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.00s

   Doc-tests feral

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

```

## Benchmark
```
