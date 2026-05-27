#+title: CB-on default re-validation (Phase 0 of issue #55)
#+date: 2026-05-27

* Scope

Phase 0 of the issue #55 implementation plan. #55 records that the
historical inertia regressions which forced ~cascade_break_ratio~ /
~cascade_break_eps~ to default ~None~ may no longer reproduce on
current HEAD. If true, flipping the default is a one-line fix for
the pinene_3200 / nql180 / cascade-victim performance class.

This note records the head-to-head evidence.

* Method

Harness: ~src/bin/phase0_cb_on_revalidation.rs~. Configures the
solver with ~cascade_break_ratio = Some(0.5)~ and
~cascade_break_eps = Some(1e-10)~ forced on, then factors each
matrix in the historical regression corpus and compares the reported
inertia to the sidecar JSON oracle (or, for ~nuffield2_trap_iter1~
which has no sidecar, the hardcoded oracle from issue #54: positive
13447, negative 13202, zero 0).

Corpus (41 matrices):
- robot_1600 iters 0-6 (cited regression: iter 3, issue #17)
- NARX_CFy iters 0-2 (cited regression: mid-IPM iters, issue #18)
- marine_1600 iters 0-17 (cited regression: iter 4, issue #48)
- rocket_12800 iters 0-1 (both-mode failure, issue #38)
- pinene_3200 iters 0-9 (CHO cascade target, issue #46)
- nuffield2_trap iter 1 (LP-shape KKT, issue #54)

Build: HEAD on ~main~ as of 2026-05-27. Release build.

* Results

Summary: pass=39, fail=2, skip=0, total=41.

** Historical regressions that DO NOT reproduce at HEAD

All cited iterations from the four closed issues that motivated the
CB-off default pass cleanly:

| Issue | Matrix         | Cited iter | Result      |
|-------+----------------+------------+-------------|
| #17   | robot_1600     | iter 3     | PASS  0.05s |
| #18   | NARX_CFy       | iters 0-2  | PASS  ~0.4s |
| #38   | rocket_12800   | iters 0-1  | PASS  ~3s   |
| #46   | pinene_3200    | iter 9     | PASS  1.52s |
| #48   | marine_1600    | iter 4     | PASS  0.47s |

The cited regressions for #17, #18, #38, #46, #48 are stale. All
five matrices pass with CB-on under the original ratio/eps
parameters. This confirms #55's hypothesis from the pinene notes.

For pinene_3200 specifically the recovered factor times (~1.0-1.6s
across iters 0-9, headed by 1.52s on iter 9) are vastly better than
the 88s CB-off baseline cited in #55 and #46 — the perf win is real
and at production scale.

** New regressions that ARE reproducible at HEAD under CB-on

Two cases fail. Both are bugs to resolve, not blockers to filing
this report.

*** marine_1600_0017 — off-by-1 sign jitter at late-IPM iter

Got ~(38416, 38391, 0)~ vs expected ~(38415, 38392, 0)~. Single
sign flip: one negative-class pivot ended up classified positive.

Sidecar context: ~delta_c = 4.7e-11~, ~delta_w = 7.7e-13~ — both
essentially at the cancellation noise floor. Iters 0-16 of the same
problem pass cleanly. This is the "sign jitter on borderline pivots
that the IPM has already perturbed nearly to noise" failure mode #55
path (2) was supposed to address.

#55 path (2) framing assumed the perturbation was sign-flipping; the
MUMPS/SSIDS reference audit showed FERAL's ~perturb_to_floor~
(~src/dense/factor.rs:639-646~) is already sign-preserving in the
MUMPS sense. So the remaining failure here is not a perturbation
sign bug — it is that the *original* pivot's sign at iter 17 is
itself noise-dominated due to the IPM's converged-state
near-singularity. The right fix is to *not perturb* that pivot at all
— delay it — which is exactly what Phase B (symbolic-time delay
budget so CB doesn't fire opportunistically) accomplishes.

*** nuffield2_trap_iter1 — 591 zero pivots from CB on LP-shape KKT

Got ~(13544, 12514, 591)~ vs expected ~(13447, 13202, 0)~. 591
pivots ended up in the zero bucket and the negative count is off by
688. This is a structurally different failure mode from
marine_1600_0017: ~ForceAccept~ is firing 591 times because the
LP-shape KKT has a structurally zero (3,3) block (issue #54), and
CB's force-acceptance bypasses the saddle-partner mechanism the
issue #46 fix put in place for the zero-(2,2)-block case.

This is itself diagnostic: it shows that CB's ~ForceAccept~ path is
not aware of structural saddle pivoting that the delay path handles
correctly. Phase B's "CB engages only when delay budget exhausted"
rewire would prevent CB from intercepting these pivots in the first
place; the existing #46 saddle-partner path would handle them via
2x2 pivots.

* Decision

Do NOT flip ~FeralConfig::default()~ to CB-on at this time.

Reasoning:
- 39 of 41 cases pass; 5 of 5 cited historical regressions are
  resolved. The case for flipping is strong on the headline metric.
- BUT the two new failures (~marine_1600_0017~,
  ~nuffield2_trap_iter1~) would each be a new regression on
  problems pounce currently solves correctly. Per the user's
  constraint ("fix that does not compromise everything else"),
  introducing new failures to fix the cascade-victim perf class is
  not an acceptable trade.
- Phase B (symbolic-time delay budget + CB rewire) closes both new
  failures by construction: marine_1600_0017 by letting delay
  continue at iter 17 instead of perturbing, and nuffield2_trap by
  keeping CB out of the zero-(3,3)-block path entirely. Flip the
  default after Phase B.
- Phase A (~n_tiny~ counter, instrumentation, MUMPS-alignment audit)
  is independently useful and unblocked; proceed.

* Next steps

1. Phase A1-A2: add ~n_tiny~ counter to ~FactorStats~ and expose via
   ~Solver::last_factor_stats~.
2. Phase A2.5: per-front delay instrumentation; run trace over the
   nuffield2_trap, marine_1600_0017, nql180, pinene_3200,
   clnlbeam, qcqp1500-1nc, robot_c set to size B2's capacity
   constant.
3. Phase A3-A5: MUMPS alignment note, decision log entry, regression
   coverage.
4. Phase B: symbolic-time delay budget, root-supernode width cap,
   CB rewire. Then re-run this harness; if all 41 pass with the
   rewired CB, flip the default per Phase 0.4.

* Files

- Harness: ~src/bin/phase0_cb_on_revalidation.rs~
- Raw output: see Phase 0 session log (run produced 41 lines of
  pass/fail/skip table; failures quoted above).

* Addendum — mechanism for the two new failures (per-supernode trace)

Per-supernode pos/neg/zero added to the existing
~FERAL_TRACE_SUPERNODE=1~ printout (factor.rs ff.inertia fields). Ran
each failing case through a one-shot probe and aggregated by
~cb=true~ / ~cb=false~. Both failures share a single mechanism:
CB's ~ForceAccept~ / ~PerturbToEps~ path bypasses the saddle-partner
and delay paths that MUMPS uses for the same pivots.

** nuffield2_trap_iter1 (591 zeros, sign imbalance +97/-688)

Trace aggregates:
- 32 ~cb=true~ supernodes processed 992 expanded columns
  (619 of them delayed-in). They produced pos=653, neg=339, zero=0
  — a +314 net positive bias from PerturbToEps perturbations,
  which the MUMPS-aligned ~perturb_to_floor~ leaves with whatever
  sign the round-off noise had.
- The 591 zeros all come from ~cb=false, may_del=true~ supernodes
  whose constraint columns (zero (2,2)-block diagonal in nuffield2's
  LP-shape KKT) now have ~gamma0 == 0~ in the panel BK loop. With
  the partner already perturbed-and-eliminated by CB upstream, the
  surviving constraint column has no off-diagonal coupling left, so
  ~count_1x1_inertia~ on a strict-zero diagonal increments ~zero~.
- Sums match: (13544, 12514, 591) = (13447 + 97, 13202 − 688, 591),
  zero net change in dimension.

Conclusion: CB and the issue #46 saddle-partner mechanism are
mutually exclusive. CB's eps perturbation breaks the MC64-co-located
~(k, k+1)~ saddle-pair invariant that ~scalar_pivot_step~
(~src/dense/factor.rs:3417-3425~) depends on. ~scalar_pivot_step~
gets the constraint half of the pair without its partner and has no
recovery path other than counting it as zero.

** marine_1600_0017 (single sign flip)

Trace aggregates:
- 69 ~cb=true~ supernodes, 0 zeros across all of them.
- Total ~zero == 0~ matches the oracle; only one pivot changed sign.
- The ~delta_c = 4.7e-11~ and ~delta_w = 7.7e-13~ context from the
  sidecar puts borderline pivots at the cancellation noise floor.
  ~cascade_break_eps = 1e-10~ is one order of magnitude above
  ~delta_c~, so ~perturb_to_floor~ rounds the borderline pivot to
  whatever sign IEEE rounding noise gives.
- Iters 0-16 of the same problem pass cleanly because they aren't
  yet in the IPM's near-converged regime where the perturbations
  drive pivots into the noise floor.

Conclusion: same mechanism as #54 ~ForceAccept~ path noticed
originally, but here amplified by CB instead of being absorbed by
delay. MUMPS doesn't see this because at iter 17 it would delay the
pivot rather than perturb it; the delay path eventually accepts it
via the natural Schur progression with the correct sign.

** Shared root cause

Both failures are instances of the *trigger* divergence the plan's
Phase A audit pre-identified: FERAL's ~perturb_to_floor~ is already
MUMPS-aligned, but CB fires the perturbation branch in cases where
MUMPS would delay. The downstream effects then differ by problem
class:

- LP-shape KKT (nuffield2): perturbation breaks saddle-pair locality
  → orphaned constraint columns count as zero.
- Late-IPM converged KKT (marine_1600_0017): perturbation rounds
  noise-floor pivots to either sign → sign jitter.

Both are closed by construction in Phase B: ~delayed_capacity~
ensures CB only fires when the delay budget is *structurally*
exhausted at this supernode, which means the issue #46 saddle path
runs first and the borderline pivots get delayed and absorbed
naturally upstream.

** Implication for Phase A scope

Phase A should keep its current scope — ~n_tiny~ counter,
instrumentation, MUMPS-alignment audit. The investigation above
confirms that neither failure is a bug in ~perturb_to_floor~ /
~count_1x1_inertia~ / ~count_2x2_inertia~ that Phase A could fix
with a small change. The mechanism is the CB *trigger condition*,
which is structurally what Phase B rewires. No Phase A patch can
close these failures.
