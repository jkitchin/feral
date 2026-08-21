# Issue #190 — a residual *target* for iterative refinement

Research note. Written before any implementation, per CLAUDE.md.
Predecessor: `dev/research/refinement-cap-2026-08-19.md` (issue #178),
which added the step cap this note argues is the wrong knob.

## The claim under test

Issue #190 says the per-call step cap shipped by #178 is unusable as a
lever for an interior-point host, because the *target* the loop is
driving toward is unreachable on the matrices that matter, so the loop
always runs to the cap and every cap value is a different arbitrary
answer.

## Verification of the issue's code claims

All five check out against the tree at `7de1a93`. The issue cites
`numeric/solve.rs:2574` and `:2749`; those line numbers predate
`a0b4d64`, which added ~250 lines above them. Located by content
instead:

| claim | site | verdict |
|---|---|---|
| convergence threshold hardwired `ε·√n` | `solve.rs:2693` (single), `:2872` (many) | confirmed |
| `RefineOptions` carries only `max_steps` | `solve.rs:2118` | confirmed |
| ForceAccept can leave a residual (quoted) | `solve.rs:2146` | confirmed verbatim |
| `solve_refined_into` discards diagnostics | `solver.rs:1676` | confirmed |
| `solve_many_refined_into` discards diagnostics | `solver.rs:1824` | confirmed |

And the arithmetic: at `n = 118 276`, `ε·√n = 7.636e-14`. A relative
residual that small is not reachable on a near-singular KKT, so the
mechanism the issue describes — every step improves slightly, the
2-strike rule never fires, the divergence guard never fires, the loop
runs to `max_steps` — follows.

## Finding 1: the cap has no principled value, and the sweep shows it

From #190's corpus sweep, `max_steps` is not monotone in outcome:
`cresc4` under L-BFGS is `InfeasibleProblemDetected` at `k = 1` — a
false infeasibility verdict on a feasible problem in 32 iterations —
but fine at `k = 0`, `2`, and `10`. `eigena2` is `Optimal` at `k = 5`,
`SolvedToAcceptableLevel` at `k = 10`, and `ErrorInStepComputation` at
`k ∈ {0,1,2,3,4}`.

That is what a knob looks like when it does not control the quantity the
caller cares about. Under a target the loop cannot reach, the
best-iterate contract returns whichever of the `k+1` non-converged
iterates happened to have the smallest `‖r‖₂`, and the issue's argument
that this has no reliable relationship to the quality of the host's
Newton step is sound: `‖r‖₂` is measured on the *condensed* symmetric
system, and the host refines against the unreduced one (Wächter &
Biegler §3.10).

Note what this does **not** say. It does not say refinement is useless —
`max_steps = 0` loses the badly-scaled LP outright (`RestorationFailed`).
It says the *stopping rule* is the thing without a caller-visible knob.

## Finding 2: `‖r‖₂/‖b‖₂` is a weaker measure than the issue's own goal

#190 states the goal precisely: the host "wants the backend to return
what a stable `LDLᵀ` would have returned — a backward-stable solve of
the system it factorized." That is a statement about *backward* error,
and the normwise relative residual is only a proxy for it.

The direct measure is the componentwise backward error of Arioli,
Demmel & Duff (1989):

    ω = max_i |r_i| / (|A|·|x| + |b|)_i

`ω ≤ u` means the computed `x` is the exact solution of `(A+δA)x =
b+δb` with `|δA| ≤ ω|A|` and `|δb| ≤ ω|b|` — componentwise, relative,
entry by entry. That is exactly the contract Ipopt's design assumes of
a backend and that MA27/MA57/MUMPS deliver by threshold pivoting. It is
also what those solvers report: MA57 `RINFO(6..8)`, MUMPS `RINFO(7)`
and `RINFO(8)`, both driven by their refinement controls.

Two properties matter here:

1. **It has a principled stopping point.** `ω ≲ u` is "the solve is
   honest", full stop. No caller has to guess a number. A normwise
   target requires the caller to invent one — which #190 accepts
   ("that becomes something we can sweep"), but a sweep is only needed
   because the criterion has no canonical value.
2. **It is scale-aware where the normwise measure is not.** The
   denominator is formed per row from the actual magnitudes present in
   that row. #190's own corpus includes a badly-scaled LP at data scale
   `1e11`; a single `‖r‖₂/‖b‖₂` number over such a system is dominated
   by the largest-magnitude rows and says nothing about the small ones.

Cost: `(|A|·|x|)_i` is one extra pass over `A` per step, the same shape
as the residual matvec already being run. **Zero extra solves.** This is
the reason to prefer it over anything derived from `kappa_1_est`.

### The denominator needs the LAPACK/ADD guard

Naïve `ω` is fragile: a row where `(|A||x| + |b|)_i` underflows toward
zero produces a spuriously enormous ratio, and refinement cannot fix it,
so the criterion would never fire — reproducing the exact failure mode
#190 reports for `ε·√n`. ADD handle this with an `ω₁/ω₂` split; LAPACK's
`dgerfs` implements the practical form, and that is what this note
adopts:

    safe1 = (n+1) * f64::MIN_POSITIVE
    safe2 = safe1 / f64::EPSILON

    d_i = (|A|·|x| + |b|)_i
    if d_i > safe2:  term = |r_i| / d_i
    else:            term = (|r_i| + safe1) / (d_i + safe1)
    ω = max_i term

Adopting a published formula verbatim also solves the oracle problem:
CLAUDE.md forbids writing the implementation and the test oracle in the
same session without an external source. The formula above **is** the
external source, and expected values can be produced independently in
Python from it.

## Finding 3: do not plumb `RefinementDiagnostics` through the hot path

#190's second ask is to surface the step count through
`Solver::solve_refined_into` / `solve_many_refined_into`. The obvious
reading — return `RefinementDiagnostics` — is wrong, and expensively so.

`solve.rs:2596` gates `kappa_1_est` on `with_diagnostics`:

```rust
let (anorm_1, kappa_1_est) = if with_diagnostics && n > 0 {
    let a1 = matrix_norm_1(matrix);
    let inv1 = estimate_inverse_norm_1(factors)?;
    (a1, a1 * inv1)
} else { (0.0, 0.0) };
```

`estimate_inverse_norm_1` is Hager–Higham; the struct's own doc comment
puts it at **3–5 extra solves per refinement run**. On #190's 58 014-
variable benchmark, back-solve is already 48.962 s at `max_steps = 10`.
Routing the host's normal call through the diagnostics path would
multiply precisely the cost the issue exists to cut.

The issue itself allows the cheap answer ("even just `usize`"). Every
value a host needs is already computed inside the loop and thrown away:
how many corrections ran, the relative residual of the returned iterate,
and — the one that actually answers "did the budget bind?" — *which
exit fired*. A small `Copy` struct carrying those three costs nothing.

The four exits are already distinct in the code and worth naming:
target reached, cap reached, 2-strike plateau, 100× divergence guard.
"Ran to the cap" and "bottomed out after two non-improving steps" are
very different diagnoses for a host, and back-solve wall time cannot
distinguish them.

## Design

### `StopCriterion`

```rust
pub enum StopCriterion {
    /// Today's rule: ‖r‖₂ < ε·√n·‖b‖₂. Default; bit-for-bit historical.
    EpsSqrtN,
    /// ‖r‖₂/‖b‖₂ ≤ t. #190's ask.
    RelativeResidual(f64),
    /// ω ≤ t, componentwise backward error (ADD 1989 / LAPACK dgerfs).
    BackwardError(f64),
}
```

`RefineOptions` becomes `{ max_steps, stop }`. `Default` is
`{ DEFAULT_REFINE_MAX_STEPS, EpsSqrtN }`, so every existing call is
unchanged in behavior *and* in arithmetic.

`EpsSqrtN` keeps the strict `<` it has today so the default path is
bit-identical. The two target variants use `≤`, so a target of `0.0`
means "only an exact zero residual stops you" rather than "unreachable".

Fallout: `RefineOptions` currently derives `Eq`; `f64` inside the enum
forces that to `PartialEq` only. Combined with the new field breaking
`RefineOptions { max_steps: k }` struct-literal construction, this is a
minor bump. Both breaks land together, once.

### `RefineOutcome`

```rust
pub struct RefineOutcome {
    pub steps: usize,             // corrections actually performed
    pub relative_residual: f64,   // ‖r‖₂/‖b‖₂ of the returned iterate
    pub stop: RefineStop,
}

pub enum RefineStop { Converged, MaxSteps, Stagnated, Diverged }
```

Returned from the four `*_into` entry points — `solve_sparse_refined_into`,
`solve_sparse_many_refined_into`, `Solver::solve_refined_into`,
`Solver::solve_many_refined_into` — changing `Result<(), _>` to
`Result<RefineOutcome, _>`. Callers writing `f(..)?;` are unaffected.
The `Vec`-returning convenience forms keep their signatures.

Multi-RHS aggregates over columns rather than allocating a per-column
vector: `steps` and `relative_residual` are maxima, and `stop` takes the
first match in the order `MaxSteps > Diverged > Stagnated > Converged` —
`MaxSteps` ranks first because "did the budget bind" is the question
being asked.

### Priority of exits is unchanged

The cap stays a cap. A target that is met at step 1 returns at step 1
under `max_steps = 10`; a cap of 10 never *adds* work. This was #178's
invariant and it survives verbatim, because the criterion is evaluated
at the top of the loop body exactly where `relative_reached` is today.

## What this does not fix

#190 observes that the architecturally correct setting for the host is
`max_steps = 0`, and that FERAL cannot offer it because
`ZeroPivotAction::ForceAccept` lets the raw solve leave a residual
against the very system it factorized — which is why the badly-scaled LP
dies at `k = 0`. A stopping-criterion knob does not change that. The
real answer there is threshold pivoting with delayed pivots, as
MA27/MA57/MUMPS do. Out of scope for #190; recorded so it is not lost.

Nor does this note claim a target will make the corpus sweep come out
well. It makes the sweep *possible*, which is the stated ask. The
numbers are pounce-side work.

## References

- [feral#190](https://github.com/jkitchin/feral/issues/190)
- [pounce#698](https://github.com/jkitchin/pounce/issues/698) obs. 5,
  [pounce#710](https://github.com/jkitchin/pounce/issues/710)
- `dev/research/refinement-cap-2026-08-19.md` — issue #178, the step cap
- `dev/research/issue-58-batched-refinement.md` — the multi-RHS refiner
- Arioli, Demmel & Duff, "Solving sparse linear systems with sparse
  backward error", SIMAX 10(2), 1989
- LAPACK `dgerfs` — the `safe1`/`safe2` guarded `BERR` formula
- Wächter & Biegler 2006, §3.10 — why the host refines a different system
- MA57 `RINFO(6..8)`; MUMPS `ICNTL(10)`, `RINFO(7..8)`
