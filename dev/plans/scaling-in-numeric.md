# Plan: move scaling from symbolic to numeric (β refactor)

**Date:** 2026-04-19
**Drives:** `dev/research/pounce-integration-interface.md` W1 option β.
**Outcome:** symbolic factorization is value-agnostic;
numeric factor takes a scaling strategy and computes the
scaling itself; symbolic results become reusable across IPM
iterations of structurally-identical KKTs.

## Current state (verified)

`src/symbolic/mod.rs` lines 295-410:
- `symbolic_factorize_with_method` calls
  `crate::scaling::compute_scaling(matrix, &snode_params.scaling_strategy)?`
  at line 303.
- Stores `scaling`, `scaling_pivot_order`, `scaling_info` on
  `SymbolicFactorization`.
- `scaling_pivot_order` is built from `perm` + `scaling_user` after
  the postorder composition (line 394).

`src/symbolic/supernode.rs` lines 6-18:
- `SupernodeParams { nemin, scaling_strategy }`.

`src/numeric/factorize.rs`:
- Line 223: assembly reads `&symbolic.scaling_pivot_order`.
- Line 339-340: `SparseFactors.scaling`, `SparseFactors.scaling_info`
  cloned out of `SymbolicFactorization`.

`src/numeric/solve.rs`:
- Lines 73, 110, 112-128: solve path reads `factors.scaling_info`
  and `factors.scaling` for pre/post-scaling.

## Target state

```rust
// src/symbolic/supernode.rs
pub struct SupernodeParams {
    pub nemin: usize,
    // scaling_strategy REMOVED
}

// src/numeric/factorize.rs (new struct, public)
pub struct NumericParams {
    pub bk: BunchKaufmanParams,
    pub scaling: ScalingStrategy,
}

impl Default for NumericParams { ... }   // BK default, ScalingStrategy default

pub fn factorize_multifrontal(
    matrix: &CscMatrix,
    symbolic: &SymbolicFactorization,
    params: &NumericParams,
) -> Result<(SparseFactors, Inertia), FeralError>;
```

Inside `factorize_multifrontal`:
1. Call `compute_scaling(matrix, &params.scaling)` → produces
   `scaling_user`, `scaling_info`.
2. Build `scaling_pivot_order` from `symbolic.perm` + `scaling_user`
   (the existing line 394 logic, just relocated).
3. Use `&scaling_pivot_order` in the assembly loop where
   `symbolic.scaling_pivot_order` is read today.
4. Move `scaling_user` into the returned `SparseFactors.scaling`
   and `scaling_info` into `SparseFactors.scaling_info`.

`SymbolicFactorization` loses three fields:
- `scaling`
- `scaling_pivot_order`
- `scaling_info`

## Consequence: symbolic becomes cacheable

After this change, two `factorize_multifrontal` calls on
matrices with identical sparsity pattern but different values
can share one `SymbolicFactorization`. This is the IPM use case
and is the structural goal of the refactor.

We are NOT shipping the cache here — just removing the blocker.
The cache lives in the future `Solver` struct.

## Migration order

Each step must leave `cargo test --release --lib` green.

### Step 1: introduce `NumericParams` (additive)

- Add `NumericParams` to `src/numeric/factorize.rs`.
- Add `factorize_multifrontal_v2` (temp name) that takes
  `&NumericParams` and computes scaling internally, ignoring
  the symbolic's scaling fields.
- Keep the existing `factorize_multifrontal` signature unchanged;
  it remains the canonical entry until step 4.

### Step 2: thread the new entry through callers

- Update callers (one per commit, smallest first):
  - tests under `src/`
  - integration tests under `tests/`
  - `src/bin/bench.rs`
  - `src/bin/vesuvio_diag.rs`
  - `src/bin/polak6_diag.rs`
  - `src/bin/dump_diff.rs` (does not call factor; skip)
  - `src/bin/solve_microbench.rs`
  - any other binaries

  Each caller swaps `&BunchKaufmanParams + symbolic-scaling` for
  `&NumericParams` while still passing `scaling_strategy` through
  `SupernodeParams` so symbolic continues to compute the scaling.
  At this point both old and new paths exist; the call sites
  use the new one.

### Step 3: drop scaling from symbolic

- Remove `SupernodeParams::scaling_strategy`.
- Remove the three scaling fields from `SymbolicFactorization`.
- Remove the `compute_scaling` call from
  `symbolic_factorize_with_method` and the `scaling_pivot_order`
  build at the end.
- Remove the `PartialSingular` warning print (it migrates to
  `factorize_multifrontal_v2` — same eprintln, same condition).
- All callers must be on the v2 path before this commit lands.

### Step 4: rename v2 → canonical

- Delete the old `factorize_multifrontal` (it has no remaining
  callers).
- Rename `factorize_multifrontal_v2` to `factorize_multifrontal`.
- Single commit, small diff.

## Tests to preserve / extend

Existing tests that should keep passing without modification
(beyond signature updates):

- `src/numeric/factorize.rs` factor tests (multifrontal LDLᵀ
  correctness with identity, infnorm, mc64).
- `src/numeric/solve.rs` solve tests (round-trip).
- `tests/` MC64-driven inertia agreement tests.

New tests to add in step 1:

- `factorize_multifrontal_v2_matches_v1_on_identity_scaling`:
  factor the same KKT with both entry points + `Identity`;
  assert `SparseFactors` agree on `inertia`, `needs_refinement`,
  `scaling`, `scaling_info`, and per-node `frontal_factors`
  byte-equivalent.
- `factorize_multifrontal_v2_matches_v1_on_mc64`: same with
  `Mc64Symmetric`.
- `factorize_multifrontal_v2_with_two_strategies_on_one_symbolic`:
  build symbolic ONCE, factor twice with `Identity` then `InfNorm`;
  both factorizations succeed; the InfNorm pass produces a
  `scaling_info != NotApplied`. **This is the proof the refactor
  achieves the structural goal.**

## Risks

### R1. Performance regression: scaling now runs every numeric

Today, repeated calls to `factorize_multifrontal` with the same
symbolic skip recomputing scaling (it is cached on
`SymbolicFactorization`). After β, each numeric call recomputes
scaling. Until the future `Solver` cache lands, this is a
minor regression on repeat-factor workloads.

**Mitigation:** the bench harness calls `factorize_multifrontal`
once per matrix, so corpus-level cost is unchanged. Microbench
on the IPM-loop pattern (factor-same-pattern-N-times) is left
to the `Solver` work that consumes this refactor.

### R2. Test fixtures that build `SymbolicFactorization` manually

A grep for `SymbolicFactorization {` should turn up test
fixtures that construct the struct field-by-field. After step 3
those fixtures have three fewer fields to populate. The
mechanical fix is straightforward.

### R3. The `scaling_info` warning print location

The `eprintln!("warning: MC64 matching left ...")` currently
runs in symbolic. Moving it to numeric means the warning fires
exactly as often (still once per factor) but is associated with
the numeric phase in any caller's logging. Confirm no tests
assert on the symbolic-phase eprintln line ordering.

## Out of scope

- The `Solver` struct and `FactorStatus` / `increase_quality()` —
  follows in a separate plan once β is in.
- The cache itself (`Solver` will hold one `SymbolicFactorization`
  optional field).
- Any change to scaling defaults — `InfNorm` stays the production
  default per the lever-C 2026-04-19 decision.

## Acceptance

- `cargo test --release --lib`: all tests pass.
- `cargo clippy --release -- -D warnings`: clean.
- `cargo fmt --check`: clean.
- Bench harness corpus-level numbers (geomean factor/MUMPS,
  inertia/residual passes) unchanged within run-to-run noise.
- The new `..._with_two_strategies_on_one_symbolic` test passes
  — proves the structural goal.
