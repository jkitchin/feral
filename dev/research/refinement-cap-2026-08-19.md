# Caller-capped iterative refinement (issue #178)

Date: 2026-08-19. Motivating report: [feral#178], drawn from [pounce#698].

## The question

FERAL's sparse iterative refinement runs a fixed budget of up to 10
correction steps. Is that budget right for *every* caller, and if not,
what is the smallest interface change that lets a caller pick its own?

## Why a fixed budget is defensible, and where it stops being so

The 10-step budget is not arbitrary. It is MUMPS's `ICNTL(10)` default,
and `dev/journal/2026-04-18-06.org` measured that below 10 some
near-rank-deficient KKT matrices (CERI651C/ELS, HAHN1, MEYER3NE) bounce
in and out of the machine-precision basin before settling. The same
session added the 2-strike `max_stagnant_steps` plateau exit so the
easy-case cost is not 10 steps for everyone. Both are still in place.

That reasoning holds for a caller who solves `Ax = b` and *keeps the
answer*. It does not hold for a caller that is itself iterating on the
same system.

An interior-point method is exactly that caller. Ipopt's
`PDFullSpaceSolver` runs its own refinement loop over the augmented
system (`min_refinement_steps = 1`, `max_refinement_steps = 10`,
`residual_ratio_max = 1e-10`), computing the residual after each of its
own back-solves and deciding from it whether to continue. When each of
*those* back-solves is a `Solver::solve_refined`, the two loops nest:
up to 10 x 11 = 110 substitution passes and 100 matvecs for one
augmented-system solve.

The loops are not redundant in a way that cancels. The outer loop owns
the convergence criterion — it computes the residual ratio the host
actually cares about. The inner loop drives a residual nobody consults
toward a tolerance nobody set, and the host pays for every step of it.

## Evidence

From pounce#698, Observation 5. pounce 0.10.0 against feral 0.15.1, one
direct-collocation NLP: 59 939 variables, 58 271 equality constraints,
augmented system 118 276 x 118 276, ~1.09 M nonzeros,
`hessian_approximation = limited-memory`. Same build, same options; only
pounce's `feral_refine` switch differs.

| | refine on | refine off | MA57 |
|---|---|---|---|
| Status | Optimal | Optimal | Optimal |
| Iterations | 186 | 191 | 170 |
| Back-solve | 147.3 s | 58.3 s | 22.3 s |
| Factorization | 92.3 s | 82.1 s | 27.6 s |
| Wall | 487.2 s | 387.9 s | 267.5 s |

Turning the inner loop off cut back-solve 60% and wall time 20% *despite
five more IPM iterations*. Per-iteration back-solve 0.792 s -> 0.305 s.

Two things are worth stating plainly rather than glossed:

- This is not the stagnation exit being absent. The 2-strike rule from
  2026-04-18-06 is present in the measured 0.15.1 and the cost above is
  what remains after it. Both `max_steps = 10` sites and the 2-strike
  exit are unchanged on `main` at 0.16.0.
- The accuracy case for the inner loop is *not visible* in these
  numbers. The unrefined run reached an objective closer to the MA57
  reference (7.4664668011e+01 vs 7.4678926542e+01, reference
  7.4661943200e+01) than the refined run. The reporter does not read a
  mechanism into the sign of that and neither do we; the honest summary
  is that these runs do not show the inner loop buying accuracy.

Nor is it a threading artefact: with `FERAL_PARALLEL=0` the back-solve is
marginally *faster* serial (0.732 vs 0.792 s/iter). Threads help the
factorization (1.21x), not this.

## Why the default must not change

pounce defaults `feral_refine` to on for a documented reason: the
`pinene_3200` model's IPM tail stalls when the residual floor left by
cascade-break's L-factor perturbation goes uncorrected. That is a real
case where *some* correction is required. Zero steps loses it.

So the shape of the problem is not "10 is too many" — it is that the
only two values available are 10 and 0. A cap of `k = 1` is the value
neither loop can currently express, and is very plausibly enough for
`pinene_3200` at a tenth of the cost. Establishing that is pounce-side
work (issue #178 acceptance items 6 and 7); FERAL's job is to make `k`
expressible.

Conclusion: add a per-call cap, default it to today's 10, change no
default anywhere.

## Interface shape

Three candidates were considered.

1. **A bare `max_steps: usize` parameter** on new `*_capped` entry
   points. Smallest possible diff. Rejected because the next tunable
   anyone asks for (a caller-set residual target, a stagnation budget)
   forces either another parameter on every entry point or a second
   round of renaming.
2. **A `Solver`-level setting** (`set_refine_max_steps`). Rejected: the
   ask is explicitly per-call, and a stateful cap on a `&self` solve
   would make `solve_refined` non-reentrant in a way it is not today.
3. **A small options struct**, `RefineOptions`, `Copy`, `Default`
   reproducing today's behavior. Chosen. It matches the established
   repo convention (`NumericParams`, `SupernodeParams`,
   `BunchKaufmanParams`, `LuParams` are all params structs), and adding
   a field later is backward compatible for callers that construct it
   from `Default`.

`RefineOptions::max_steps` counts *correction* steps, not total
substitution passes: the initial solve always happens, so a call runs at
most `1 + max_steps` passes. This matches how `RefinementDiagnostics`
already numbers its steps (`steps[0]` is the unrefined solve), so the
observable the reporter proposed to verify against needs no
reinterpretation.

`max_steps = 0` must be the same computation as an unrefined solve, not
merely the same answer. That means the residual matvec and its norm have
to be skipped too, not computed and discarded — otherwise `k = 0` costs
a matvec more than `solve_sparse` and the "all-or-nothing" complaint is
only half addressed. The one exception is the diagnostics entry point,
which must still emit `steps[0]`; there the matvec is the point.

## Precedence: a cap is a cap, not a target

The existing exits — the `eps*sqrt(n)` relative-residual target, the
100x divergence guard, the 2-strike plateau — all keep priority over
`max_steps`. A well-conditioned system that converges in one step must
still return after one step under `k = 10`. This falls out of the loop
structure (`for step in 1..=max_steps` with the convergence test at the
top of the body), but it is worth an explicit test because it is the
property that makes the cap safe to hand to callers: raising `k` can
never *add* work on a system that has already converged.

Similarly the best-iterate contract is unaffected. The returned `x` is
the iterate with the smallest `||r||_2` seen, which under any `k >= 0`
includes the unrefined solve. A cap therefore cannot return an answer
worse than `solve_sparse`'s — the guarantee stated in
`solve_sparse_refined`'s doc comment holds for every `k`.

## The in-place ask

Independent and much smaller. Every `Solver` solve entry point returns an
owned `Vec`; a host that already owns its RHS buffer pays an allocation
plus a copy-back per back-solve — `dim x nrhs` doubles, ~946 KB at the
dimension above. The internals already have the shape:
`solve_sparse_into_ws` and `solve_sparse_many_into` take an out-slice.

Two notes on doing this honestly:

- The refined paths should write the *best iterate* directly into the
  caller's slice rather than filling a `Vec` and copying at the end.
  Using `x_out` as the `best_x` storage removes one `n`-length
  allocation per call from the refined path itself, not just from the
  caller. The working iterate `x` still needs its own buffer.
- Aliasing `rhs` and `x_out`: issue #178 asks for it to be "supported or
  rejected explicitly rather than silently wrong". In safe Rust it is
  neither — it is *unrepresentable*, because `&[f64]` and `&mut [f64]`
  into the same allocation cannot coexist. That is a stronger guarantee
  than a runtime check, and the right response is to document it rather
  than to add an unreachable branch. A C-ABI caller passing one pointer
  twice is a different question, and the C ABI (`feral_solve`) already
  solves in place into its own `rhs` buffer, so it does not arise there.

## References

- [feral#178](https://github.com/jkitchin/feral/issues/178)
- [pounce#698](https://github.com/jkitchin/pounce/issues/698), Observation 5
- `dev/journal/2026-04-18-06.org` — the 2-strike plateau exit and the
  cap=2 / cap=3 / two-tier / 1-strike / 2-strike bench panel
- Wachter & Biegler 2006; Ipopt `IpPDFullSpaceSolver.cpp` (the host loop)
- MUMPS `ICNTL(10)` — origin of the 10-step default
