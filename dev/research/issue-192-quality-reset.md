# Issue #192 — bounding the lifetime of `increase_quality`

Date: 2026-08-29

## Problem

`Solver::increase_quality` is a one-way ratchet. `quality_level` moves
`Baseline → ScalingEnabled → PivotRaised → Exhausted` and there is no
path back, so an escalation chosen for *one* hard factorization governs
**every** factorization for the remaining life of the `Solver`.

That is the right shape for an escalation that is monotone in
robustness. FERAL's is not monotone in *trajectory*:

- Stage 1 flips `ScalingStrategy::Identity → InfNorm`.
- Stage 2 raises `bk.pivot_threshold` by `threshold^0.75`, capped at
  `pivtol_max`.

Both change *which pivots are taken*. The resulting factorization is
different, not uniformly better. Because the change persists, the whole
downstream trajectory of the caller's algorithm is different too —
including any restoration sub-problem's factorizations.

## Why Ipopt gets away with never reverting

Ipopt's `IncreaseQuality` contract (`IpPDFullSpaceSolver.cpp:296`) is
"this factorization cannot deliver — escalate and refactor". MA57
answers it by raising `pivtol` toward `pivtolmax`
(`IpMa57TSolverInterface.cpp:832`). That *is* monotone in robustness:
a higher threshold is strictly more conservative, so leaving it raised
for the rest of the solve can only make later factorizations safer.
Never reverting is therefore harmless there, and Ipopt exposes no
reset. FERAL's stage 1 has no MA57 counterpart at all, and stage 2
lands in a different pivot regime rather than a strictly safer one.

## Downstream evidence (from the issue; measured in pounce gh#850)

On `square_flowsheet_resto`, wiring `increase_quality` through cost
both solve arms:

| leg | with escalation | without |
|---|---|---|
| exact Hessian | `RestorationFailed`, 131 iters | `Optimal`, 99 |
| limited-memory | 3000 iters, at the cap | `Optimal`, 178 |

The rung fires exactly twice: once in the main solve at iteration 25,
once inside the restoration sub-solve at `76r`.

Three measured facts rule out the obvious workarounds:

1. **Scoping it out of restoration is not enough.** Allowing only the
   first (main-solve) firing still loses the leg.
2. **A firing count does not discriminate.** `deb7` and
   `square_flowsheet_resto` each fire it exactly twice on their exact
   legs — one gains 16% of its iterations, the other loses its verdict.
3. **Declining to escalate is not the answer either.** A 12-variable
   watchdog model ends `Solved_To_Acceptable_Level` at `obj = 3.7e-6`
   with the escalation and at `obj = 3.42` against `f* = 0` without it,
   and the rung buys 15–25% of the iterations on five other models.

So the escalation is genuinely useful *and* genuinely destructive, and
the distinguishing variable is **for how long**, not **whether**.

## Code inspection

`increase_quality` mutates exactly two pieces of state:

- `numeric_params.scaling` (stage 1, `solver.rs:1908`)
- `numeric_params.bk.pivot_threshold` (stage 2, `bump_pivot_threshold`)

plus `quality_level` itself. It touches nothing else — not
`last_symbolic`, not `mc64_scaling_cache`, not `last_factors`, and none
of the independent latches (`ordering_escalated`, `auto_arm_latched`,
`mc64_retry_not_adopted`). The symbolic cache is already documented as
escalation-invariant: the β refactor moved scaling from the symbolic to
the numeric phase, so a scaling flip does not invalidate it.

Consequently an exact inverse of the escalation is: restore those two
fields, set `quality_level = Baseline`, and touch nothing else. The
inverse is symmetric with the forward operation in what it leaves
alone, which is the property that keeps it cheap and predictable.

## Prior deferral

`dev/research/pounce-integration-interface.md` (§"What this does NOT
decide") recorded:

> Whether to expose a `reset_quality()` method. Ipopt does NOT have one
> — the consumer creates a new solver per problem if they want a reset.
> FERAL should match (no reset method) until there is evidence to add
> it.

Issue #192 supplies that evidence. This note takes up the deferral; it
does not reverse a decision.

## Design

### Where the baseline comes from

The values to restore are the caller's *factory* parameters, not
`NumericParams::default()`. A caller may build with
`Solver::with_params(np, sn)` and/or `.with_scaling(...)`, so the
correct baseline is whatever was in place immediately before the first
escalation fired.

Capture it **lazily, on the transition out of `Baseline`**, rather than
in `with_params`. Two reasons:

1. The builders (`with_scaling`, …) are consuming and run *after*
   `with_params`, so a construction-time snapshot would record
   pre-builder values and a reset would silently discard the caller's
   configuration.
2. It makes the round trip exact by construction: the snapshot is taken
   at the exact instant the ladder starts, so restoring it returns the
   solver to a state it demonstrably occupied.

After a reset the snapshot is cleared, so a later re-escalation
re-captures and walks the identical ladder. `reset` → `increase` is
indistinguishable from a fresh `Solver` with the same params, which is
the property downstream needs when it re-baselines at a major-iteration
or restoration boundary.

### API

```rust
/// Revert every escalation applied by `increase_quality`.
pub fn reset_quality(&mut self) -> bool;
```

Returns `true` if an escalation was undone, `false` if the solver was
already at `Baseline` (a no-op). The boolean lets a caller that
re-baselines unconditionally at a loop boundary log only the firings
that mattered, without first reading `quality_level()`.

Valid from any level, `Exhausted` included — a caller that exhausted
the ladder on one system is exactly the caller that most needs the next
system to start clean.

### Considered and not taken: a scoped guard

The issue also lists a guard form (escalate for the next `factor()`
only, then fall back). A `QualityGuard<'a>(&'a mut Solver)` with
`DerefMut` and a restoring `Drop` would work, but:

- It fixes the scope at *one* `factor()`, and the Ipopt contract loop
  escalates repeatedly until the factorization delivers. The natural
  scope is the caller's own retry loop, whose shape FERAL does not know.
- It is strictly expressible in terms of the reset, so shipping the
  reset first costs the guard nothing if evidence for it later arrives.

The issue itself names the reset as "the smallest useful increment",
and states the downstream intent: re-baseline at whatever boundary
makes sense "without FERAL having to know anything about the caller's
algorithm". A reset satisfies that; a guard would encode an assumption
about the caller's loop. Ship the reset.

## Test oracle

The oracle is external to the implementation: the reset must restore
the parameter values the *caller* supplied. Those are known
independently of any FERAL code path —

- `Solver::new()` starts at `pivot_threshold = 1e-8`, MA27's `cntl(1)`
  default (issue #2), with `ScalingStrategy::Auto`.
- `Solver::with_params(np, _)` starts at whatever `np` carries; tests
  pass a deliberately non-default value so a reset that restored
  `NumericParams::default()` instead of the caller's parameters would
  fail.

No assertion in the plan is read off the implementation's own output.
