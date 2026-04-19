# POUNCE integration interface (FERAL §2.12.1)

**Date:** 2026-04-19
**Spec ref:** `FERAL-PROJECT-SPEC.md` §2.12.1
**Status:** research note. No code yet.

## Why

POUNCE (the planned IPM outer loop) needs an interface modeled on
Ipopt's `SymLinearSolver`: factor a (perturbed) KKT, query inertia
against an expected count, ask the solver to "try harder" on
inertia/singularity failures, then re-factor. Today FERAL is fully
stateless — `factorize_multifrontal` is a free function — and has
no concept of "try harder." We need a small object-shaped surface
that owns the quality-escalation state.

This note pins down the contract from the Ipopt reference, the
shape of FERAL's mirror, and the design wrinkles before writing
the plan.

## Reference contract (Ipopt 3.14)

From `IpSymLinearSolver.hpp`:

```
enum ESymSolverStatus {
  SYMSOLVER_SUCCESS,
  SYMSOLVER_SINGULAR,
  SYMSOLVER_WRONG_INERTIA,
  SYMSOLVER_CALL_AGAIN,    // MA27 realloc; not applicable to FERAL
  SYMSOLVER_FATAL_ERROR
};

ESymSolverStatus MultiSolve(
    const SymMatrix& A,
    vector<rhs>& rhsV, vector<sol>& solV,
    bool  check_NegEVals,
    Index numberOfNegEVals);          // expected # of negatives, IN

Index NumberOfNegEVals() const;       // queried after WRONG_INERTIA
bool  IncreaseQuality();              // monotonic, returns false when exhausted
bool  ProvidesInertia() const;
```

`IncreaseQuality()` two-stage escalation
(`IpTSymLinearSolver.cpp:429-444`):

- **Stage 1 (wrapper):** if a scaling method exists, scaling is
  currently OFF, and `linear_scaling_on_demand=true`, turn scaling
  ON for the next factorization. Returns `true`. State
  (`use_scaling_`, `just_switched_on_scaling_`) is persistent.
- **Stage 2 (delegate to inner solver):** raise the pivot
  threshold. `IpMa27TSolverInterface.cpp:723-739`: if
  `pivtol_ == pivtolmax_` return `false`; else
  `pivtol_ = min(pivtolmax_, pivtol_^0.75)`. MUMPS uses `^0.5`
  (more aggressive, `IpMumpsSolverInterface.cpp:615-633`). Both
  are persistent across factorizations.
- **Exhaustion:** both stages exhausted → returns `false`.

Calling pattern in `PDFullSpaceSolver::SolveOnce`
(`IpPDFullSpaceSolver.cpp:490-591`):

1. Caller perturbs the matrix diagonal (`PerturbForSingularity`
   or `PerturbForWrongInertia`), always with the SAME underlying
   matrix values.
2. Call `Solve(...check_NegEVals=true, numberOfNegEVals=expected)`.
3. Dispatch on return:
   - `FATAL_ERROR` → abort.
   - `SINGULAR` with equality duals → perturb-for-singularity, retry.
   - `WRONG_INERTIA` with `NumberOfNegEVals() < expected` → first
     time only, try `IncreaseQuality()`; if refused, treat as
     singular.
   - Otherwise (`WRONG_INERTIA` with too many negatives, or
     `SINGULAR` without duals) → perturb-for-wrong-inertia, retry.
4. On `SUCCESS`, run iterative refinement on the full primal-dual
   system (POUNCE's job, not FERAL's per §2.12.3). If residuals
   stagnate and the per-system `augsys_improved_` latch is
   `false`, call `IncreaseQuality()` and re-solve once.

Key invariants the contract relies on:
- Quality state is **persistent** for the solver lifetime, not
  reset per call. Consumer tracks its own per-system "asked once"
  latch.
- `check_NegEVals` and `numberOfNegEVals` are passed **into** the
  solve so the solver can short-circuit before producing a
  solution.
- `ProvidesInertia()` must return `true` for `check_NegEVals=true`
  to be legal.

## Current FERAL surface

Free functions, no state object:

```rust
// Numeric factor (stateless function):
pub fn factorize_multifrontal(
    &CscMatrix,
    &SymbolicFactorization,
    &BunchKaufmanParams,
) -> Result<(SparseFactors, Inertia), FeralError>;

pub fn solve_sparse(&SparseFactors, &[f64]) -> Result<Vec<f64>, FeralError>;
pub fn solve_sparse_refined(&SparseFactors, &CscMatrix, &[f64], max_iters, tol)
    -> Result<Vec<f64>, FeralError>;
```

Quality dials already in place:

- `BunchKaufmanParams::pivot_threshold`: 0.0 default; bump to
  0.01 (MUMPS/SSIDS default) or higher to tighten.
- `BunchKaufmanParams::on_zero_pivot`: `Fail` (default) → maps
  to `Singular`; `ForceAccept` → masks singularity, used today.
- `SupernodeParams::scaling_strategy`: `InfNorm` default,
  `Auto`/`Mc64Symmetric`/`Identity`/`External` available.

Note: `scaling_strategy` is consumed during **symbolic**
factorization (`symbolic_factorize_with_method`), which produces
`SymbolicFactorization::scaling`. Numeric factor reads scaling
from there. **Implication:** stage-1 escalation (turn scaling on)
requires a fresh symbolic pass. This is a real design wrinkle —
see §Wrinkles below.

## Proposed FERAL mirror

A new `Solver` struct that owns the persistent state. The free
functions stay; `Solver` is a thin coordinator on top.

```rust
/// Result of a single factorization attempt.
#[derive(Debug, Clone)]
pub enum FactorStatus {
    /// Factorization succeeded. If inertia was checked, it matched.
    Success,
    /// Numerically singular (Fail-mode zero pivot, or symbolic
    /// PartialSingular from MC64).
    Singular,
    /// Inertia was checked and disagreed with the expected count.
    WrongInertia { actual: Inertia, expected: Inertia },
    /// Unrecoverable: dimension mismatch, alloc failure, etc.
    FatalError(FeralError),
}

/// Stateful linear-solver handle. Mirrors Ipopt SymLinearSolver.
pub struct Solver {
    // owned state, persistent across factor() calls:
    bk_params: BunchKaufmanParams,
    snode_params: SupernodeParams,
    pivtol_max: f64,                     // cap, default 0.5 (MA27)
    quality_level: QualityLevel,
    last_symbolic: Option<SymbolicFactorization>,
    last_factors: Option<SparseFactors>,
    last_inertia: Option<Inertia>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QualityLevel {
    Baseline,                            // factory defaults
    ScalingEnabled,                      // stage 1 fired
    PivotRaised,                         // stage 2 fired (any number of times)
    Exhausted,                           // pivtol == pivtol_max
}

impl Solver {
    pub fn new() -> Self;
    pub fn with_params(bk: BunchKaufmanParams, sn: SupernodeParams) -> Self;

    /// Factor the matrix. If `check_inertia` is `Some(expected)`,
    /// returns `WrongInertia` when actual != expected without
    /// invalidating the stored factor (caller may still call
    /// `solve` to inspect; matches MUMPS behavior under
    /// SYM=2).
    pub fn factor(
        &mut self,
        matrix: &CscMatrix,
        check_inertia: Option<Inertia>,
    ) -> FactorStatus;

    /// Solve A x = b using the most recent successful factor.
    /// Errors if no successful factor exists.
    pub fn solve(&self, rhs: &[f64]) -> Result<Vec<f64>, FeralError>;

    /// Two-stage quality escalation. Persistent.
    /// Returns false when both stages are exhausted. Mirrors
    /// IpTSymLinearSolver::IncreaseQuality.
    pub fn increase_quality(&mut self) -> bool;

    /// Number of negative eigenvalues from the last factor.
    /// Panics if no factor has been performed yet.
    pub fn num_negative_eigenvalues(&self) -> usize;

    /// Whether the solver provides inertia. Always true for FERAL.
    pub fn provides_inertia(&self) -> bool { true }
}
```

Two-stage escalation, FERAL values:

- **Stage 1** (`Baseline → ScalingEnabled`): if
  `snode_params.scaling_strategy == ScalingStrategy::Identity`,
  set it to `ScalingStrategy::InfNorm` (current default);
  invalidate `last_symbolic` so the next `factor()` re-runs
  symbolic with the new scaling. Return `true`. Skip if scaling
  was already non-Identity.
- **Stage 2** (`ScalingEnabled | Baseline → PivotRaised → ...`):
  if `bk_params.pivot_threshold == 0.0`, set to `0.01`;
  else `pivot_threshold = min(pivtol_max, pivot_threshold^0.75)`.
  When `pivot_threshold >= pivtol_max` after the update,
  transition to `Exhausted` and return `true` for *this* call;
  the *next* call returns `false`.
- Defaults: `pivtol_max = 0.5` (MA27 value, conservative),
  exponent `0.75` (MA27, less aggressive than MUMPS).

The escalation does NOT call `factor()` — it only mutates state.
The next caller-driven `factor()` picks up the new params.

## Wrinkles

### W1. Scaling change forces symbolic recomputation

Stage 1 of escalation flips `scaling_strategy`. Today the scaling
vector is materialized inside `symbolic_factorize_with_method`
(it is a `Vec<f64>` in `SymbolicFactorization`). So enabling
scaling means tossing the cached symbolic and rebuilding it.

Three options:

- **(α) Tolerate the recompute.** Stage 1 fires at most once per
  solver lifetime; the IPM outer loop calls FERAL many times
  with the same pattern, so amortized cost is fine. This matches
  the Ipopt model where the inner solver is opaque about whether
  it actually re-does symbolic.
- **(β) Refactor scaling to live in numeric.** Move the scaling
  computation out of symbolic into a separate cached field on
  `Solver`. Lets stage 1 fire without invalidating symbolic.
  Bigger change, touches `SparseFactors::scaling`, the solve
  path, and every test that builds `SymbolicFactorization`
  manually.
- **(γ) Always run with scaling on.** Make `Identity` an
  explicit caller opt-in (not the path POUNCE follows) and make
  stage 1 always a no-op. Then stage 1 is dead code we delete.

**Recommendation: (α) for the first cut.** It is the smallest
change, matches the Ipopt model, and stage 1 firing means the
caller asked for an upgrade — paying one extra symbolic for it
is reasonable. (β) becomes interesting only if profiling shows
stage 1 fires often enough to matter; (γ) is appealing but
requires a separate scaling-default decision (the lever-C
2026-04-19 work just decided to keep `InfNorm` as default, so
`Identity` is already not the POUNCE path; this points toward
(γ) being the eventual answer once we are confident).

### W2. `WrongInertia` — keep the factor or discard it?

Ipopt's `SYMSOLVER_WRONG_INERTIA` does NOT invalidate the factor;
the consumer queries `NumberOfNegEVals()` and may still call
`Solve` if it wants to inspect. FERAL should do the same: store
the factor and the actual inertia, return `WrongInertia` from
`factor()`, but allow `solve()` to proceed against it. The
consumer is expected to perturb-and-refactor, not solve, but the
contract should not block the solve.

### W3. `Singular` — Fail vs ForceAccept

`BunchKaufmanParams::on_zero_pivot` controls this today. The
contract maps cleanly:

- `Fail` → `factor()` returns `FactorStatus::Singular`.
- `ForceAccept` → `factor()` returns `Success` and sets
  `needs_refinement=true` (existing behavior). POUNCE then
  decides via residual whether to upgrade quality or perturb.

POUNCE's `PerturbForSingularity` only fires on *real* singular
returns, so `Fail` is the contract-correct default. `ForceAccept`
remains as an opt-in for adversarial / research workloads.

### W4. Threading the matrix in for solve

Ipopt's `Solve` takes both the matrix and the RHS each call.
FERAL's `solve_sparse` only needs the factor (the matrix is not
re-touched by triangular solve). For iterative refinement, the
matrix IS needed. Decision: `Solver::solve` takes only the
factor (the common case). If POUNCE wants augmented-system
refinement (it doesn't, per §2.12.3 — refinement happens at the
primal-dual level), it can call `solve_sparse_refined` directly
against the cached factor. We expose `Solver::factors()`
returning `Option<&SparseFactors>` for this.

### W5. Stage 2 first call: 0.0 → 0.01 jump

The MA27 formula `pivot_threshold^0.75` does not work from 0.0.
The first stage-2 call needs an explicit "from 0.0 jump to 0.01"
rule (matching MUMPS/SSIDS default), then subsequent calls follow
the geometric rule. This is a one-line special case in
`increase_quality`, but worth pinning in the plan.

### W6. Inertia-check short-circuit

Ipopt's `check_NegEVals=true` lets the solver short-circuit and
return `WRONG_INERTIA` before doing the solve. In FERAL the
inertia is known the moment `factorize_multifrontal` returns —
there is no separate solve cost wrapped into factor — so the
"short-circuit" is just: factor, then compare `inertia.negative`
to `expected.negative`, and route the return. No structural
change to `factorize_multifrontal` is needed; the comparison
lives in `Solver::factor`.

## What this does NOT decide

- The actual `pivtol_max` value: 0.5 (MA27) vs more aggressive.
  Initial value 0.5; tune later if POUNCE evidence accumulates.
- The exponent for stage 2: 0.75 (MA27) vs 0.5 (MUMPS). 0.75
  initial, less aggressive; both stages get exhausted in 4-6
  steps from 0.01 anyway.
- Whether to expose a `reset_quality()` method. Ipopt does NOT
  have one — the consumer creates a new solver per problem if
  they want a reset. FERAL should match (no reset method) until
  there is evidence to add it.
- The wider iterative-refinement story (§2.12.3) — that is
  POUNCE's domain.

## Test plan (high level — for the plan note)

- Unit: `increase_quality` from each `QualityLevel` produces the
  documented next state.
- Unit: stage 1 is no-op if scaling already non-Identity.
- Unit: stage 2 first call sets `pivot_threshold = 0.01` from 0.0;
  subsequent calls follow the geometric rule; cap at `pivtol_max`.
- Unit: `Exhausted` returns `false` from the next
  `increase_quality` call.
- Unit: `factor` with `check_inertia=Some(expected)` returns
  `WrongInertia { actual, expected }` and stores the factor
  (verify `solve` still works against it, modulo correctness).
- Unit: `factor` with `check_inertia=None` always returns
  `Success` on a non-singular matrix.
- Integration: a small KKT (n≈10) where naive factor gives wrong
  inertia. Verify the IPM-style loop (factor → check → bump
  quality → re-factor) terminates with `Success` and correct
  inertia.
- Integration: solver lifetime — multiple `factor()` calls on the
  same `Solver` share state; calling `increase_quality()` between
  them changes the next factor's behavior.

## Files to add / change

- `src/numeric/solver.rs` (new): `Solver`, `FactorStatus`,
  `QualityLevel`. Re-exported from crate root.
- `src/lib.rs`: add `pub use numeric::solver::{Solver, FactorStatus};`.
- `tests/pounce_interface.rs` (new): the integration tests above.

No changes to `src/numeric/factorize.rs`,
`src/symbolic/supernode.rs`, or the existing free functions for
the first cut.

## Next step

Write the plan note (`dev/plans/pounce-integration-interface.md`)
covering: file layout, struct field-by-field, escalation state
machine table, test list with explicit assertions, and the
single design call-out for the user (W1 α/β/γ).
