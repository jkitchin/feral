#+title: MUMPS perturbation alignment audit
#+date: 2026-05-27

* Scope

Phase A3 of the issue #55 implementation plan
(~/.claude/plans/feral-is-a-cached-raccoon.md~). Confirms that the
*formulae* FERAL applies when a pivot is rounded to a static floor are
already MUMPS-aligned, and documents the single remaining divergence
that motivates Phase B.

This is a paperwork commit, not a code change: every FERAL site cited
below was inspected for alignment with the MUMPS reference
(MUMPS 5.8.2). No formula was found that needed adjustment as part of
Phase A. The Phase A1 ~n_tiny~ counter
(commit subject: "Phase A1: thread n_tiny through dense factor +
FactorStats") closes the *accounting* gap (FERAL had no
~INFO(25)~-equivalent counter); the *trigger* gap is closed by
Phase B.

* The three things MUMPS does and FERAL must match

For each pivot it perturbs, MUMPS does three things:

1. *Replace the pivot* with a static-floored value: $\tilde d =
   \mathrm{sign}(d)\cdot\max(|d|, \tau)$ with the convention
   $\mathrm{sign}(0) = +1$.
2. *Count the pivot's inertia* by the sign of the *perturbed* value
   $\tilde d$ — never as a "zero", and never by the sign of the
   original near-zero $d$.
3. *Bump a diagnostic counter* (~INFO(25) = NBTINYW~) that is
   reported back to the caller but does not gate any acceptance
   check.

The audit walks each of these three contracts against FERAL.

* 1. Pivot-replacement formula

** MUMPS 5.8.2 reference

~MUMPS_REPLACE_TINY_PIVOT~ (~dfac_front_aux.F~, around lines 1251-1331
in the FERAL `mumps-5.8.2/` mirror) implements:

#+begin_src fortran
PIVOT = SIGN(MAX(ABS(PIVOT), SEUIL), PIVOT)
#+end_src

with the Fortran ~SIGN(A, B)~ semantics: ~|A| · sign(B)~ where
~sign(0) == +1~. So a strict-zero pivot is replaced with ~+SEUIL~.

** FERAL site

~perturb_to_floor~ in ~src/dense/factor.rs:639-646~:

#+begin_src rust
fn perturb_to_floor(d: f64, abs_floor: f64) -> f64 {
    let mag = d.abs().max(abs_floor);
    if d < 0.0 { -mag } else { mag }
}
#+end_src

The branch ~if d < 0.0 { -mag } else { mag }~ treats both ~d == 0.0~
and ~d > 0.0~ as the positive case, which is exactly the Fortran
~SIGN(_, 0.0) == +1~ contract. *Match.*

For 2×2 blocks, ~perturb_2x2_to_floor~
(~src/dense/factor.rs:664-695~) follows the analogous convention:
push the small-magnitude eigenvalue $\lambda_{\min}$ to
$\pm\tau$, preserving its current sign if nonzero; if
$\lambda_{\min} = 0$ use the sign of $\lambda_{\max}$ (keeps the
~(+,−)~ / ~(+,+)~ / ~(−,−)~ signature unambiguous); if both
eigenvalues are zero, push positive. The "both zero" sub-case is the
2×2 analogue of MUMPS's ~SIGN(_, 0.0) == +1~. *Match.*

* 2. Inertia-by-perturbed-sign

** MUMPS reference

After ~MUMPS_REPLACE_TINY_PIVOT~ runs, the inertia accumulator looks
at ~PIVOT~ (the now-perturbed value) and increments either
~INFO(12) = NEG~ or the positive bucket according to ~PIVOT > 0 / <
0~. There is no "zero" bucket for the post-perturbation path: a
zero pivot was just replaced by ~+SEUIL~ and counts as positive.

** FERAL site

~count_1x1_inertia~ ~PerturbToEps~ branch
(~src/dense/factor.rs:4621-4632~):

#+begin_src rust
ZeroPivotAction::PerturbToEps { abs_floor } => {
    let d_new = perturb_to_floor(d, abs_floor);
    a[k * stride + k] = d_new;
    *needs_refinement = true;
    *n_tiny += 1;
    if d_new > 0.0 {
        *pos += 1;
    } else {
        *neg += 1;
    }
    Ok(())
}
#+end_src

The MA57-style ~static_pivot_floor~ branch a few lines above
(~src/dense/factor.rs:4597-4607~) is identical in structure: replace,
count by sign of replaced value, no path to ~zero~ bucket.

2×2 perturbations apply ~perturb_2x2_to_floor~ in the do_2x2_pivot /
scalar_pivot_step paths; the perturbed diagonals are written back
into the L storage *before* ~count_2x2_inertia~ runs, so the 2×2
inertia counter classifies the perturbed block (one positive, one
negative for the saddle case; both positive for indefinite-collapsed;
both negative for negative-collapsed). *Match.*

Note the parallel path for ~ForceAccept~ (a strict-zero pivot
*kept* as zero, no perturbation): that branch increments ~*zero += 1~,
which is the SSIDS-aligned convention recorded under issue #54.
~ForceAccept~ is *not* equivalent to MUMPS perturbation — it is the
"accept the singularity" path, which MUMPS does not have. This
distinction matters when reading the per-supernode trace in the
Phase 0 re-validation note
(~dev/research/cb-on-default-revalidation-2026-05-27.md~).

* 3. Diagnostic counter

** MUMPS reference

~INFO(25) = NBTINYW~ counts ~MUMPS_REPLACE_TINY_PIVOT~ calls and is
returned alongside ~INFOG(12)~ (negative inertia) and other status
counters. It is reported but not gated on; the factor proceeds
regardless.

** FERAL site

Pre-Phase-A: the FERAL ~FrontalFactors~ struct exposed
~needs_refinement~ (set on any perturbation event) but no scalar
event count. There was no way for a caller to ask "how many pivots
were perturbed on this solve?"

Post-Phase-A: ~FrontalFactors::n_tiny~
(~src/dense/factor.rs:1220~ in the public struct definition) is a
~usize~ incremented at every ~perturb_to_floor~ /
~perturb_2x2_to_floor~ call site. It aggregates upward through
~SparseFactors::n_tiny()~
(~src/numeric/factorize.rs:1036-1046~) and is surfaced on
~FactorStats::n_tiny~ (~src/numeric/solver.rs:71-78~), reachable
via the existing ~Solver::last_factor_stats()~ accessor. *Match
(Phase A1 closed the gap).*

* The one remaining divergence: trigger condition

MUMPS only enters the ~MUMPS_REPLACE_TINY_PIVOT~ branch under two
trigger conditions:

1. ~uu == 0.0~ (user disabled pivoting entirely, equivalent to
   FERAL ~pivot_threshold = 0~ with no on-zero-pivot fallback).
2. The post-~GO TO 630~ delay path is exhausted: the pivot reached
   the root of its frontal stack with no eligible Schur partner.
   In MUMPS this happens when the analysis-time per-front delay
   capacity is full, *not* on a numeric-time heuristic.

FERAL's cascade-break trigger
(~src/numeric/factorize.rs:2248-2258~):

#+begin_src rust
let cascade_break = match params.cascade_break_ratio {
    Some(r)
        if !is_root[snode_idx]
            && params.allow_delayed_pivots
            && expanded_ncol > 0
            && symbolic.n >= CASCADE_BREAK_MIN_N =>
    {
        (n_delayed_in as f64) / (expanded_ncol as f64) >= r
    }
    _ => false,
};
#+end_src

fires when the *numeric-time* ratio ~n_delayed_in / expanded_ncol~
crosses an empirical threshold (default ~r = 0.5~). This perturbs
pivots that MUMPS would have delayed and accepted cleanly elsewhere
in the tree. The two known consequences of this divergence are
documented in
~dev/research/cb-on-default-revalidation-2026-05-27.md~ §
"Addendum — mechanism for the two new failures":

- *LP-shape KKT* (nuffield2_trap_iter1): CB's
  ~PerturbToEps~ event breaks the MC64-co-located saddle-pair
  invariant the issue #46 mechanism relies on. The
  constraint-half of the saddle gets perturbed-and-eliminated, the
  surviving Lagrange-half has no off-diagonal coupling, and
  ~count_1x1_inertia~ on a strict-zero diagonal increments the
  ~zero~ bucket instead of giving a 2×2.
- *Late-IPM converged KKT* (marine_1600_0017): the perturbation
  rounds a noise-floor pivot to whatever sign IEEE rounding gave it
  — a single sign flip on an already-near-singular pivot. MUMPS
  doesn't see this because it would delay rather than perturb at
  this iterate.

Both failures are *trigger-condition* failures, not formula
failures. No change to ~perturb_to_floor~,
~count_1x1_inertia~ PerturbToEps branch, or
~perturb_2x2_to_floor~ would close them. Phase B
(symbolic-time ~delayed_capacity~ on ~Supernode~ + CB rewire to fire
only when delay is *structurally* exhausted) is what closes them, by
construction: CB then fires under exactly the MUMPS trigger
condition, and the issue #46 saddle-partner path runs first whenever
the delay budget hasn't been hit.

* Frozen conventions

The following conventions are now considered locked under Phase A4
(see ~dev/decisions.md~ entry of 2026-05-27):

- ~perturb_to_floor~ formula ~sign(d) · max(|d|, τ)~ with
  ~sign(0) = +1~.
- ~perturb_2x2_to_floor~ as currently implemented, including the
  $\lambda_{\min} = 0$ tie-breaker via $\mathrm{sign}(\lambda_{\max})$.
- ~count_1x1_inertia~ PerturbToEps branch: count by sign of the
  perturbed value, increment ~n_tiny~, never increment ~zero~.
- ~ForceAccept~ branch on strict-zero: increment ~zero~, do *not*
  increment ~n_tiny~ (ForceAccept is the "accept the singularity"
  path, not a perturbation; the inertia accounting is the SSIDS
  convention from issue #54).
- ~n_tiny~ is diagnostic only — not part of any acceptance gate,
  not surfaced in error messages, not in CI assertions beyond the
  Phase A5 regression tests that lock ~n_tiny == 0~ on the CB-off
  default path.

Re-running this audit is mandatory before any change to
~perturb_to_floor~, ~perturb_2x2_to_floor~, the inertia-counting
branches in ~count_1x1_inertia~ / ~count_2x2_inertia~, or the
~n_tiny~ wiring on ~FrontalFactors~ / ~SparseFactors~ /
~FactorStats~.

* References

- MUMPS 5.8.2: ~dfac_front_aux.F~ ~MUMPS_REPLACE_TINY_PIVOT~
  (lines ~1251-1331); ~dini_defaults.F~ ~INFOG(12)~ / ~INFO(25)~
  accounting (lines ~875-876, 919-920).
- FERAL: ~src/dense/factor.rs:639-695~ (perturb helpers),
  ~src/dense/factor.rs:4576-4632~ (count_1x1_inertia 1×1 branches),
  ~src/dense/factor.rs:1158-1224~ (FrontalFactors struct,
  including ~n_tiny~),
  ~src/numeric/factorize.rs:1036-1046~
  (~SparseFactors::n_tiny~ accessor),
  ~src/numeric/factorize.rs:2248-2279~ (cascade-break trigger),
  ~src/numeric/solver.rs:71-78~ (~FactorStats::n_tiny~ field).
- Phase 0 evidence:
  ~dev/research/cb-on-default-revalidation-2026-05-27.md~ — 39/41
  historical regressions resolved, 2 new failures traced to the
  trigger divergence audited above.
- Plan: ~/.claude/plans/feral-is-a-cached-raccoon.md~ §
  "Phase A — Diagnostic counter and alignment audit".
