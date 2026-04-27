# ripopt Integration Plan for F1/F2/F3

**Date:** 2026-04-27
**Status:** Companion to `dev/plans/kkt-feature-gaps.md`. Documents how
ripopt — the Rust port of Ipopt that consumes feral via the
`LinearSolver` trait — will adopt each new capability.
**Audience:** anyone who needs to land an end-to-end change spanning
feral and ripopt.

## Why this document exists

feral's three-phase roadmap (multi-RHS, condition estimate, Schur
complement) is justified primarily by ripopt's needs. This plan names
the concrete consumption sites in ripopt so the feral API shape can be
co-designed with the caller, and so a future agent landing an integration
change knows exactly which file:line to touch.

The asymmetry: feral changes are *additive* — the existing
single-RHS surface stays bit-for-bit identical, so ripopt continues to
work without any change. Each ripopt adoption is its own optional
follow-up landed only when the cost/benefit pencils out.

## ripopt's current consumption surface

`../ripopt/src/linear_solver/mod.rs:573` defines the trait:

```rust
pub trait LinearSolver {
    fn factor(&mut self, matrix: &KktMatrix) -> Result<Option<Inertia>, SolverError>;
    fn solve(&mut self, rhs: &[f64], solution: &mut [f64]) -> Result<(), SolverError>;
    fn provides_inertia(&self) -> bool;
    fn min_diagonal(&self) -> Option<f64> { None }
    fn increase_quality(&mut self) -> bool { false }
}
```

The `feral_direct.rs` adapter wraps `feral::numeric::solver::Solver`.
Single-RHS only. No condition number, no Schur.

Solve-call sites in the IPM hot path (`../ripopt/src/ipm.rs`):

| line | site                            | purpose                                  |
|------|---------------------------------|------------------------------------------|
| 2760 | newton_step                     | primary direction solve                  |
| 3028 | solve_for_direction             | direction solve with (δ_w, δ_c) regul.   |
| 3229 | try_mehrotra_predictor          | **affine (predictor) solve**             |
| 3500 | post-Mehrotra rebuild           | corrector RHS replacement → next solve   |
| 5029 | restoration NLP solves          | restoration phase direction              |

## F1 — Multi-RHS adoption

### What ripopt gets

The Mehrotra predictor-corrector path at
`../ripopt/src/ipm.rs:3217-3248` does two solves with the same factor:
the affine direction (line 3229) and then the corrector direction
(after RHS replacement at line 3500). Today these go through two
independent `solve` calls, each paying a workspace allocation and
missing the supernodal-traversal amortization.

After F1.1 lands in feral, ripopt's adoption is:

1. Extend `LinearSolver` with a default-impl `solve_many`:
   ```rust
   fn solve_many(&mut self, rhs: &[f64], nrhs: usize,
                 solution: &mut [f64]) -> Result<(), SolverError> {
       // Default: loop over single solve; backends override.
       let n = rhs.len() / nrhs;
       for c in 0..nrhs {
           self.solve(&rhs[c*n..(c+1)*n], &mut solution[c*n..(c+1)*n])?;
       }
       Ok(())
   }
   ```
   Existing backends keep working unchanged; the feral adapter
   overrides to call `Solver::solve_many`.

2. Refactor `try_mehrotra_predictor` so the affine RHS is *not*
   solved on its own. Instead, after the affine solve completes,
   build the corrector RHS from the affine result, pack both into
   a `2*n` buffer, and submit a single `solve_many(packed, 2)`
   call. This re-orders the data flow but does not change the
   math.

   Caveat: the corrector RHS depends on `mu_aff` which depends on
   the affine *solution*. So the predictor still has to be solved
   first. The bundling opportunity is for the *next* IPM iteration
   when a centrality correction is layered on; or, if Gondzio
   multiple correctors are added (Gondzio 1996), all corrector
   RHSes are known once the predictor has resolved and *those* can
   be batched. The minimal Mehrotra predictor-corrector pattern
   does not benefit; multi-corrector schemes do.

3. The simpler immediate use is the **higher-order corrector**
   in `mehrotra_corrector_step` if/when ripopt grows that path.
   At that point both corrector RHSes are computed before any
   solve, and a `solve_many(rhs, k)` call is a clean win.

**ripopt change cost:** ~30 lines (trait extension + adapter
override + Mehrotra refactor). Performance gain depends on the
corrector strategy — pure predictor-corrector is unchanged; multi-
corrector or higher-order Mehrotra get O(1.5×–2×) speedup on the
batched solves.

### Compatibility

- Existing `LinearSolver::solve` callers untouched.
- Backends without batched-solve support (banded, dense LDLᵀ
  wrappers) get the looped default impl free.
- The feral adapter is the one backend that overrides — and only
  if F1.1's per-column cost beats the loop default per the F1.2
  acceptance gate (≤ 0.75× per-column at k=4).

## F2 — Condition estimate adoption

### What ripopt gets

`../ripopt/src/ipm.rs:3067` (`try_solve_with_correction`) drives δ_w
escalation by a fixed schedule when inertia is wrong or the solve
fails:

```rust
loop {
    dir_result = kkt::solve_for_direction(kkt_system, lin_solver,
                                          ic_delta_w, ic_delta_c);
    if /* inertia wrong */ {
        ic_delta_w *= delta_w_inc_fast;  // Ipopt's default 8.0
        continue;
    }
    ...
}
```

This matches Ipopt's heuristic but is blind to the actual
conditioning of the factored system. With F2's
`Solver::estimate_condition_1norm` returning κ̂_1(L D Lᵀ), ripopt
can:

1. **Detect ill-conditioned factors** even when inertia is
   correct — Ipopt's filter line search regularly accepts steps
   from factors with κ ~ 10¹⁵ that produce no usable direction.
   Logging κ̂ at every iteration gives ripopt's diagnostics page
   the same observability that MUMPS provides via `RINFO(7)/(8)`.

2. **Trigger early δ_w bump:** if the factor is well-conditioned
   *and* the inertia is wrong, the inertia error is structural
   (genuine indefiniteness in the wrong direction) and δ_w should
   escalate aggressively. If the factor is ill-conditioned *and*
   inertia is wrong, δ_w should escalate gently because the
   condition number reflects an underlying scaling issue that
   more regularization cannot fix.

3. **Surface to user-facing logs:** ripopt's `IntermediateData`
   already exposes per-iteration solver info; adding `kappa_1_est`
   is a single field addition.

### Implementation in ripopt

Add a default-impl method to `LinearSolver`:

```rust
fn estimate_condition_1norm(&self) -> Option<f64> { None }
```

The feral adapter returns `Some(κ̂)`; other backends return `None`.
At the call site in `try_solve_with_correction`, log the value
when present. The adaptive δ_w heuristic is a **future** change;
the first adoption is just observability.

**ripopt change cost:** ~10 lines (trait method + adapter override
+ logging field).

## F3 — Schur complement adoption

### What ripopt gets

Two distinct use cases:

#### Use case A: structured KKT elimination

For NLPs with a clear two-block structure — e.g., bounds-only
problems where the slack-block is diagonal, or stochastic two-stage
problems where the second-stage block is block-diagonal — Ipopt
3.14 has a "structured" reduction strategy (Ip ConditionalRegOp).
ripopt today flattens these to a generic sparse augmented system.

With `Solver::factor_with_schur(matrix, schur_indices)`, ripopt
can implement a structured-problem detector in `kkt.rs`:

1. Identify the slack/second-stage block by analyzing the KKT
   sparsity pattern at problem-build time.
2. Pass the block's variable indices as `schur_indices` to feral.
3. Solve the reduced (Schur complement) system with whatever
   solver fits the dense block's size.
4. Recover the eliminated variables via `solve_with_partition`.

This is a substantial refactor in ripopt — ~200 lines spanning
`kkt.rs`, `ipm.rs`, and a new `structured_kkt.rs`. Justified only
if ripopt grows a stochastic-programming or large-bounds workload;
not on the path to general NLP performance.

#### Use case B: sensitivity analysis

Ipopt's `sIPOPT` extension (Pirnay-López 2012) computes parametric
sensitivities by solving against the final KKT factor with new
RHSes derived from parameter perturbations. ripopt's
`sensitivity.rs` does this today via repeated single-RHS solves.

With Schur, sensitivity directions can be computed by reducing
the sensitivity system to the parameter-block alone — much smaller
than the full augmented system. Same pattern as use case A:
identify the parameter block, factor with Schur, solve in reduced
space.

**ripopt change cost (sensitivity only):** ~50 lines added to
`sensitivity.rs`.

### Implementation in ripopt

Trait extension is non-trivial — Schur breaks the simple
"factor + solve" trait pattern because the solve has to know
which variables were eliminated. Two options:

1. **Wide trait:** add `factor_with_schur` and
   `solve_with_partition` methods to `LinearSolver` with default
   impls returning `Err(Unsupported)`. Backends that don't
   implement Schur stay at default; feral overrides.

2. **Adapter wrapper:** keep `LinearSolver` simple; expose Schur
   via a separate `SchurCapableSolver` trait that the feral adapter
   *additionally* implements. ripopt detects the capability via
   downcast or a `as_schur(&self) -> Option<&dyn SchurCapable>`
   accessor.

Option 2 is cleaner — Schur is fundamentally a different solve
shape, and bolting it onto `LinearSolver` muddles the contract.
Recommend option 2; record the decision when F3.0 lands.

## Trait extension proposal

Cumulative end-state, post-F3:

```rust
pub trait LinearSolver {
    fn factor(&mut self, matrix: &KktMatrix) -> Result<Option<Inertia>, SolverError>;
    fn solve(&mut self, rhs: &[f64], solution: &mut [f64]) -> Result<(), SolverError>;
    fn provides_inertia(&self) -> bool;
    fn min_diagonal(&self) -> Option<f64> { None }
    fn increase_quality(&mut self) -> bool { false }

    // F1
    fn solve_many(&mut self, rhs: &[f64], nrhs: usize,
                  solution: &mut [f64]) -> Result<(), SolverError> {
        let n = rhs.len() / nrhs;
        for c in 0..nrhs {
            self.solve(&rhs[c*n..(c+1)*n],
                       &mut solution[c*n..(c+1)*n])?;
        }
        Ok(())
    }

    // F2
    fn estimate_condition_1norm(&self) -> Option<f64> { None }

    // F3 — separate trait, see above
    fn as_schur_capable(&self) -> Option<&dyn SchurCapableSolver> { None }
}
```

All additive, all defaulted. Existing ripopt backends compile
unchanged.

## API stability promise

The single-RHS feral surface
(`solve_sparse`, `solve_sparse_into`, `Solver::solve`,
`Solver::solve_refined`) is *frozen* by F1.1 onwards. Any change
to those signatures is a breaking change to ripopt and bumps
feral's MAJOR version.

Multi-RHS, condition, and Schur are additive; they extend feral's
public surface without changing existing signatures.

## Sequencing

| feral phase   | ripopt phase                              | gate |
|---------------|-------------------------------------------|------|
| F1.1-F1.3     | trait `solve_many` default                | F1.2 acceptance gate (≤ 0.75× per-col at k=4) |
| F1.4 (rayon)  | (no ripopt change — internal)             | -    |
| F2.1-F2.2     | trait `estimate_condition_1norm` + log    | F2.2 ≤ 5% solve cost overhead |
| F3.1-F3.3     | `SchurCapableSolver` trait + sensitivity  | F3.3 reduced-system correctness vs full-block |

Each row is independently shippable. ripopt does not have to wait
for all three to land before adopting F1.

## Open questions (close before adoption begins)

1. **Should `LinearSolver::solve_many` take `nrhs` or infer from
   slice lengths?** Decision: take `nrhs` explicitly. Matching
   feral's signature reduces translation surface in the adapter
   and makes the dim-mismatch error site obvious.

2. **Is the Mehrotra refactor worth it given the predictor →
   corrector dependency?** Probably not in isolation. Plan F1
   adoption *jointly* with a Gondzio multi-corrector experiment
   so the batched-solve infrastructure has a real workload.

3. **Should F2's κ̂ value influence the δ_w schedule, or just be
   logged?** First version: log only. Adaptive δ_w is a research
   project (cf. Wächter & Biegler 2006 §3.1) — observability first,
   policy second.

## References

- `dev/plans/kkt-feature-gaps.md` — feral-side phasing
- `dev/research/multi-rhs.md` — F1.0 design note
- `../ripopt/src/linear_solver/mod.rs` — `LinearSolver` trait
- `../ripopt/src/ipm.rs:3217` — Mehrotra predictor site
- `../ripopt/src/ipm.rs:3067` — δ_w escalation site
- `../ripopt/src/sensitivity.rs` — parametric sensitivity site
- Gondzio 1996: "Multiple centrality corrections in a primal-dual
  method for linear programming"
- Pirnay-López 2012: "Optimal sensitivity based on IPOPT"
- Wächter & Biegler 2006 §3.1 — δ_w schedule
