# Issue #221 follow-up: when should `increase_quality` decline a rung?

Date: 2026-10-20. Context: PR #222 comment 6294656538 (pounce side).

## Problem

After #221's fixes, the remaining pounce regressions (nql180 and
brainpc2 time out; brainpc1 needs a re-solve) all disappear with
`feral_increase_quality=no`. pounce could not find a rule of its own
that separates the models the ladder rescues (`deb7`, `pooling_rt2stp`,
a 12-variable watchdog model) from the ones it costs. The reason is that
pounce sees only symptoms: an inertia count and refinement stalls. It
does not see whether the factorization was the cause.

feral can see that. It asked: can feral decline a rung it knows will not
help, or say "this count is certain at this threshold"?

## What a rung is for

Threshold pivoting with parameter `u` accepts a 1×1 pivot only when
`|a_kk| >= u · max_i |a_ik|`, which is exactly `|L_ik| <= 1/u` for every
multiplier in that column. The 2×2 test bounds the multipliers of the
pivot block the same way (Duff-Reid; the SSIDS determinant floor). The
element-growth bound that makes the factorization backward stable comes
only from `|L| <= 1/u` (Higham, *Accuracy and Stability*, §11.1.1;
Duff-Reid 1983). MA57 answers Ipopt's `IncreaseQuality` by raising
`pivtol`, which tightens that bound.

So a rung is a request to recompute the factor with a tighter multiplier
bound. Two facts follow:

1. **A factor that already meets the next rung's bound cannot be
   improved by it.** If no pivot was floored (`n_tiny == 0`) or
   force-accepted below the threshold (`n_threshold_fail_accepts == 0`),
   and the realized `max |L_ij| <= 1/u_next`, then the current factor is
   already a valid threshold-`u_next` factorization: every pivot it took
   passes the stricter test. Refactoring at `u_next` may choose other
   pivots, but its stability guarantee is no stronger than the one
   already held. The symptom (inertia or refinement) does not come from
   multiplier growth, and the rung is a lateral move.
2. **A rung that makes the realized factor worse has failed.** In
   feral, unlike MA57, a higher `u` means more delays. Fronts that cannot
   delay (root, cascade-break) then force-accept or floor pivots, so the
   realized `|L|` and the violation counts can *grow* when `u` rises. That
   is the opposite of what the rung is for.

## Measurements (pounce 68e438d0 + this branch, every escalation logged)

`viol` = `n_tiny + n_threshold_fail_accepts` of the factor that pounce
judged insufficient; `1/max|L|` is its realized threshold.

| model | rung | u (factor) | viol | 1/max\|L\| | next u | admissible at next? |
|---|---|---|---|---|---|---|
| nql180 | 1 | 1e-8 | 0 | 1.4e-6 | 1e-6 | **yes** |
| nql180 | 4 | 4.2e-4 | 3652 | 3.4e-6 | | no (made worse) |
| nql180 | 7 | 3.8e-2 | 30554 | 3.8e-6 | | no (made worse) |
| deb7 | 1 | 1e-8 | 6 | 1.3e-8 | 1e-6 | no |
| deb7 | 2 | 1e-6 | 0 | 2.0e-6 | 3.2e-5 | no |
| brainpc2 | 1 | 1e-8 | 0 | 1.0e-8 | 1e-6 | no |
| brainpc2 | 2 | 1e-6 | 35 | 5.7e-14 | | no (made worse) |
| pooling (exact) | 1 | 1e-8 | 1 | 6.1e-3 | 1e-6 | no |

On nql180 every rung leaves the count at 130079 (one short). The
factorization at 1e-8 is clean and already meets 1e-6. Each later rung
adds forced pivots (0, then 3652, then 30554) and delays (788, then
484003), at about 5× the cost per factor. On brainpc2 the first rung is
legitimate in MA57 terms (`max |L| ≈ 1e8 = 1/u`), but the factor at the
raised threshold is far worse: 35 violations and `max |L| ≈ 1.8e13`. deb7
improves at every rung (6 → 0 violations, `max |L|` 7.9e7 → 5e5), which
is why pounce saw "the second rung is the one that pays".

## Proposal

In `Solver::increase_quality`, before a stage-2 bump, measure the last
factor's `(viol, max |L|)`:

- **(A) Admissible:** `viol == 0 && max|L| · u_next <= 1`. Decline:
  return `false`, change nothing. Pounce already treats a declined rung
  as "go to δ_c".
- **(B) Failed rung:** judge each bump once, at the next call, against
  the factor that prompted it. If the factor at the raised threshold is
  worse (more violations, or equal violations and a larger `max |L|`),
  return the threshold to where the ladder started, mark the ladder
  `Exhausted`, and return `false`. Retrying would only repeat the failure
  at the cost of another factorization.

  Two refinements came from prototyping on pounce:
  - *Back to the ladder start, not one rung.* With a one-rung undo,
    nql180 stopped at 3.2e-5 (rungs 1e-8 → 1e-6 → 3.2e-5 → 4.2e-4, the
    last one worse, with 5548 violations). That left 70k delays per
    factor: 224 s and 33 iterations, Optimal. The earlier rungs had only
    raised delays (788 → 3459 → 70061) and never cured the one-short
    count.
  - *Judged once.* With a persistent record, pooling_rt2stp compared a
    later iterate (2 floored pivots) with the factor that prompted a rung
    three calls earlier. A clean factor in between had declined the next
    rung, so the stale comparison undid the whole ladder.

Expose the realized `max |L|` in `FactorStats` (`max_abs_l`) so callers
can see the threshold the factor actually meets. That answers the
"inertia-confidence" question with a measured quantity rather than a
verdict.

Cost: one pass over L, O(nnz(L)), only when `increase_quality` is
called. That is about the cost of one triangular solve.

## What this does not claim

- (A) does not prove the inertia is correct. It proves a higher threshold
  cannot make the factor more stable. Inertia errors from a
  well-conditioned perturbation, `‖E‖ < min |λ|`, are excluded only
  together with a backward-error bound, and pounce's refinement already
  measures that.
- (B) compares factors of different matrices (successive IPM iterates).
  On the measured cases the differences are orders of magnitude (0 vs 35
  violations; 1e8 vs 1e13), not noise. A model whose escalations
  improve the factor (deb7) is unaffected.
