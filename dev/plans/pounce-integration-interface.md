# POUNCE integration interface — implementation plan

**Date:** 2026-04-19
**Spec ref:** `FERAL-PROJECT-SPEC.md` §2.12.1
**Research note:** `dev/research/pounce-integration-interface.md`
**Status:** plan. No code yet.

## Scope

Add a stateful `Solver` handle to FERAL that mirrors the Ipopt
`SymLinearSolver` contract: factor → check inertia → escalate quality →
re-factor. The free functions in `src/numeric/factorize.rs` and
`src/numeric/solve.rs` stay; `Solver` is a thin coordinator on top
that owns persistent quality state and a cached
`SymbolicFactorization` for refactor-on-same-pattern reuse.

The β refactor (commit `18b8bc0`) was the precondition: scaling now
lives in `NumericParams`, so stage-1 escalation can flip
`scaling_strategy` without invalidating the cached symbolic.

## W1 design call-out — resolved

The research note flagged W1 ("scaling change forces symbolic
recomputation") with three options α/β/γ and recommended (α). The β
refactor shipped the (β) option ahead of this plan, so:

- The cached `SymbolicFactorization` on `Solver` is reusable across
  every quality level. No symbolic invalidation on stage-1.
- Stage-1 escalation flips `numeric_params.scaling` from `Identity`
  to `InfNorm`; the next `factor()` call passes the new
  `NumericParams` to `factorize_multifrontal` against the cached
  symbolic.

No further user input needed on W1.

## File layout

```
src/numeric/solver.rs              (new)  — Solver, FactorStatus, QualityLevel
src/numeric/mod.rs                 (edit) — pub mod solver;
src/lib.rs                         (edit) — pub use numeric::solver::{Solver, FactorStatus};
tests/pounce_interface.rs          (new)  — unit + integration tests
```

No changes to `src/numeric/factorize.rs`,
`src/symbolic/supernode.rs`, or the existing free functions for the
first cut.

Drive-by docs cleanup: `SparseFactors::scaling` doc comment still
says "Cloned from `SymbolicFactorization::scaling`" (stale post-β).
Fix in the same commit.

## Public API

```rust
// src/numeric/solver.rs

/// Result of a single `Solver::factor` attempt.
#[derive(Debug, Clone)]
pub enum FactorStatus {
    /// Factorization succeeded. If `check_inertia` was supplied,
    /// the actual inertia matched.
    Success,
    /// Numerically singular: factor encountered a zero pivot under
    /// `ZeroPivotAction::Fail`, or scaling reported
    /// `PartialSingular`.
    Singular,
    /// Inertia was checked and disagreed with the expected count.
    /// Factor is still stored — `solve()` may proceed.
    WrongInertia { actual: Inertia, expected: Inertia },
    /// Unrecoverable error (dimension mismatch, alloc failure,
    /// symbolic-analysis failure).
    FatalError(FeralError),
}

/// Stateful linear-solver handle. Mirrors Ipopt SymLinearSolver.
///
/// Owns quality-escalation state and a cached SymbolicFactorization
/// so repeated `factor()` calls on structurally identical matrices
/// reuse the symbolic analysis.
pub struct Solver {
    numeric_params: NumericParams,
    snode_params: SupernodeParams,
    pivtol_max: f64,
    quality_level: QualityLevel,
    last_symbolic: Option<SymbolicFactorization>,
    last_factors: Option<SparseFactors>,
    last_inertia: Option<Inertia>,
    /// Sparsity pattern fingerprint of the cached symbolic (matrix
    /// `n` plus the `col_ptr` and `row_idx` lengths). Recomputed
    /// from `matrix` on every `factor()` and compared against the
    /// cached fingerprint; mismatch invalidates `last_symbolic`.
    last_pattern_fingerprint: Option<PatternFingerprint>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualityLevel {
    Baseline,
    ScalingEnabled,
    PivotRaised,
    Exhausted,
}

impl Solver {
    pub fn new() -> Self;
    pub fn with_params(np: NumericParams, sn: SupernodeParams) -> Self;
    pub fn factor(
        &mut self,
        matrix: &CscMatrix,
        check_inertia: Option<Inertia>,
    ) -> FactorStatus;
    pub fn solve(&self, rhs: &[f64]) -> Result<Vec<f64>, FeralError>;
    pub fn solve_refined(
        &self,
        matrix: &CscMatrix,
        rhs: &[f64],
    ) -> Result<Vec<f64>, FeralError>;
    pub fn increase_quality(&mut self) -> bool;
    pub fn num_negative_eigenvalues(&self) -> usize;  // panics if no factor
    pub fn provides_inertia(&self) -> bool { true }
    pub fn factors(&self) -> Option<&SparseFactors>;
    pub fn quality_level(&self) -> QualityLevel;
}

impl Default for Solver { fn default() -> Self { Self::new() } }
```

`PatternFingerprint` is a private `(usize, usize, usize)` of
`(n, col_ptr.len(), row_idx.len())`. This is conservative — two
patterns can collide on these three numbers — but for the IPM use
case the pattern is genuinely identical across iterations, so a
collision means the consumer is doing something the solver doesn't
support yet. If we need a tighter check later we can hash
`col_ptr` and `row_idx`; not for the first cut.

## Escalation state machine

| State | `increase_quality()` action | Next state | Returns |
|---|---|---|---|
| `Baseline` | If `numeric_params.scaling == Identity`, set to `InfNorm`. Otherwise apply stage-2 (pivot) rule. | `ScalingEnabled` (stage 1) or `PivotRaised` (stage 2) | `true` |
| `ScalingEnabled` | Apply stage-2 (pivot) rule. | `PivotRaised` or `Exhausted` (see below) | `true` |
| `PivotRaised` | Apply stage-2 (pivot) rule again. | `PivotRaised` or `Exhausted` | `true` |
| `Exhausted` | No-op. | `Exhausted` | `false` |

**Stage-2 (pivot) rule**, applied to `numeric_params.bk.pivot_threshold`:

1. If `pivot_threshold == 0.0`, set `pivot_threshold = 0.01`. (W5
   special case — the `^0.75` formula does not work from 0.)
2. Else set `pivot_threshold = min(pivtol_max, pivot_threshold.powf(0.75))`.
3. After the update, if `pivot_threshold >= pivtol_max - eps_cap`,
   transition to `Exhausted` for the *next* call. The current call
   still returns `true` because work was done.

Defaults: `pivtol_max = 0.5` (MA27 value), exponent `0.75` (MA27).
`eps_cap = 1e-12`.

The escalation method does NOT call `factor()`. It mutates
`numeric_params` and `quality_level` only; the next caller-driven
`factor()` picks up the new params.

## `factor()` flow

```
fn factor(&mut self, matrix, check_inertia) -> FactorStatus:
  1. Compute pattern fingerprint of `matrix`.
  2. If fingerprint mismatches `last_pattern_fingerprint`,
     invalidate `last_symbolic`, `last_factors`, `last_inertia`,
     and reset `last_pattern_fingerprint`.
  3. If `last_symbolic` is None, call
     `symbolic_factorize(matrix, &snode_params)`. On error return
     FatalError. Store the result in `last_symbolic` and stash the
     fingerprint.
  4. Call `factorize_multifrontal(matrix, &symbolic, &numeric_params)`.
     - On `FeralError::ZeroPivot` (Fail mode) → return Singular,
       clear `last_factors` / `last_inertia` so `solve()` errors.
     - On other errors → return FatalError, clear factor state.
  5. Stash factors and inertia.
  6. If `check_inertia == Some(expected)` and
     `inertia != expected`, return
     `WrongInertia { actual: inertia, expected }`. Factor stays
     stored — caller may call `solve()` or `factors()`.
  7. Otherwise return Success.
```

Notes:
- The existing `factorize_multifrontal` signature already returns
  `Result<(SparseFactors, Inertia), FeralError>`, so no change needed
  on that side.
- `PartialSingular` from MC64 should map to `Singular`. Today
  `factorize_multifrontal` returns this in `factors.scaling_info`,
  not as an error. The plan: in step 5, after stashing, inspect
  `factors.scaling_info` and treat `ScalingInfo::PartialSingular`
  as a Singular return. The factor is still stored so the consumer
  can introspect.

## `solve()` flow

```
fn solve(&self, rhs) -> Result<Vec<f64>, FeralError>:
  - If `last_factors` is None, return FeralError::NoFactor (new
    variant; trivial).
  - Otherwise call `solve_sparse(&factors, rhs)`.

fn solve_refined(&self, matrix, rhs) -> Result<Vec<f64>, FeralError>:
  - If `last_factors` is None, return FeralError::NoFactor.
  - Otherwise call `solve_sparse_refined(matrix, &factors, rhs)`.
```

`solve_refined` is exposed because POUNCE will not use it (per
§2.12.3 refinement happens at the primal-dual level), but having
it parallel to `solve()` makes the `Solver` a complete drop-in for
non-IPM consumers too.

## Test plan

### Unit tests (in `src/numeric/solver.rs`)

`U1` `increase_quality_baseline_identity_to_scaling_enabled`:
construct `Solver::with_params` where `numeric_params.scaling ==
Identity`. Assert `quality_level()` is `Baseline`. Call
`increase_quality()`; assert it returned `true`,
`numeric_params.scaling == InfNorm`,
`numeric_params.bk.pivot_threshold` unchanged,
`quality_level() == ScalingEnabled`.

`U2` `increase_quality_baseline_nonidentity_skips_to_pivot_raised`:
construct with `numeric_params.scaling == InfNorm` (default) and
`pivot_threshold == 0.0`. Call `increase_quality()`; assert it
returned `true`, `pivot_threshold == 0.01`,
`quality_level() == PivotRaised`. Stage 1 was a no-op.

`U3` `increase_quality_pivot_geometric_rule`:
start with `pivot_threshold == 0.01`,
`quality_level == PivotRaised`. Call `increase_quality()`; assert
`pivot_threshold == 0.01_f64.powf(0.75)` (≈ 0.0316), still
`PivotRaised` (cap not yet reached).

`U4` `increase_quality_caps_at_pivtol_max_then_exhausts`:
start with `pivot_threshold == 0.49`,
`pivtol_max == 0.5`, `quality_level == PivotRaised`. Call
`increase_quality()`; assert it returned `true`,
`pivot_threshold == 0.5`, `quality_level() == Exhausted`. Call
again; assert it returned `false`, `pivot_threshold == 0.5`,
state still `Exhausted`.

`U5` `increase_quality_exhausted_returns_false`:
construct, call `increase_quality()` repeatedly until `false` is
returned. Assert finite step count (< 20) and final state
`Exhausted`.

### Integration tests (in `tests/pounce_interface.rs`)

`I1` `factor_then_solve_baseline_no_inertia_check`:
2×2 SPD matrix, `Solver::new()`, factor with `check_inertia=None`.
Assert `Success`. Solve a simple RHS; assert correct answer.

`I2` `factor_with_correct_inertia_returns_success`:
diag(2, 3, 5), expected inertia (3, 0, 0). Factor with
`check_inertia=Some(expected)`. Assert `Success`.

`I3` `factor_with_wrong_inertia_returns_wronginertia_keeps_factor`:
diag(2, 3, 5), expected inertia (2, 1, 0) (deliberately wrong).
Factor with `check_inertia=Some(wrong)`. Assert `WrongInertia {
actual: (3, 0, 0), expected: (2, 1, 0) }`. `factors()` returns
`Some`. `solve()` returns the correct answer (the factor is
valid; only the inertia check failed).

`I4` `singular_under_fail_returns_singular_clears_factor`:
3×3 matrix with a structural zero pivot (e.g. `[[0, 1, 0], [1, 0,
0], [0, 0, 1]]` after permutation forces a pivoting failure;
construct so that `BunchKaufmanParams { on_zero_pivot: Fail, .. }`
trips). Factor; assert `Singular`. `factors()` returns `None`.
`solve()` errors.

`I5` `pattern_change_invalidates_symbolic`:
factor a 3×3 SPD `A`, then factor a 4×4 SPD `B` on the same
`Solver`. Both should `Success`. Verify (white-box) by exposing a
test-only `pub(crate) fn cached_symbolic_n(&self) -> Option<usize>`
or by checking `factors()` dimension.

`I6` `same_pattern_reuses_symbolic`:
factor diag(2, 3, 5), then factor diag(7, 11, 13) on the same
solver. Both `Success`. Use a `#[cfg(test)] pub(crate)` counter on
`Solver` to assert `symbolic_factorize` was called exactly once.
This is the cache-reuse property that motivates the whole class.

`I7` `quality_escalation_loop_terminates_with_correct_inertia`:
small KKT (n ≈ 6) where naive factor with `Identity` scaling and
`pivot_threshold = 0.0` gives wrong inertia. Construct the loop:
```
loop {
    match solver.factor(&kkt, Some(expected)) {
        Success => break,
        WrongInertia { .. } => {
            assert!(solver.increase_quality(), "exhausted before success");
        }
        Singular | FatalError(_) => panic!(),
    }
}
```
Assert termination in ≤ 6 iterations and final factor produces
the expected inertia. Use the same `bordered_kkt_4x4` matrix from
`tests/sparse_postorder.rs` (extended) or a hand-constructed
case.

`I8` `solver_lifetime_state_persists`:
construct one `Solver`. Factor; call `increase_quality()` twice.
Factor again; assert the second factor used the bumped
`pivot_threshold` (white-box via `solver.numeric_params.bk.pivot_threshold`
exposed via `pub(crate)` test accessor, or by observing different
inertia on a borderline matrix).

## Error handling

New variant `FeralError::NoFactor` for `solve()` / `solve_refined()`
called before a successful factor. Plain enum addition, no impact
on existing callers.

The existing `FeralError::ZeroPivot` already exists and is what
`factorize_multifrontal` returns under
`ZeroPivotAction::Fail`. `Solver::factor` maps it to
`FactorStatus::Singular`.

## Implementation sequence

1. **Step 1** — types and skeleton.
   - Add `Solver`, `FactorStatus`, `QualityLevel`,
     `PatternFingerprint`, `FeralError::NoFactor` (compiles, no
     logic). Re-export from `src/lib.rs`.
2. **Step 2** — `factor()` happy path.
   - Implement steps 1-5 of the `factor()` flow (no inertia check
     yet). Tests `I1`, `I5`, `I6` should pass.
3. **Step 3** — inertia check.
   - Implement step 6. Tests `I2`, `I3`.
4. **Step 4** — singular handling.
   - Map `ZeroPivot` and `PartialSingular` to `Singular`. Test `I4`.
5. **Step 5** — escalation state machine.
   - Implement `increase_quality()` and `quality_level()`. Tests
     `U1`-`U5`.
6. **Step 6** — solve methods.
   - Implement `solve()`, `solve_refined()`, `factors()`,
     `num_negative_eigenvalues()`.
7. **Step 7** — integration loop.
   - Test `I7`, `I8`. Verify the IPM-style loop terminates.

Each step is a separate commit with the relevant tests passing.

## Out of scope

- Iterative refinement of the augmented system. POUNCE's job per
  §2.12.3.
- A `reset_quality()` method. Ipopt has none; if POUNCE wants
  reset, it constructs a new `Solver`.
- Adaptive `pivtol_max` or `exponent` tuning. Ship MA27 defaults
  (0.5, 0.75); revisit when POUNCE produces evidence.
- Hash-based pattern fingerprint. The conservative `(n, col_ptr.len(),
  row_idx.len())` is enough for the IPM use case.
- Threaded factor / solve. `Solver` is `!Sync` (mutable cache
  state). If POUNCE needs parallel solves later, expose
  `factors() -> &SparseFactors` and let the caller spawn.
