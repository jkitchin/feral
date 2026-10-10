# Issue #215 — making delay-budget overflow inertia-exact

Date: 2026-10-10
Session: 2026-10-10-01
Supersedes: the fix direction in `issue-215-reused-analysis-cb.md`.
Auto re-analysis is rejected; see `tried-and-rejected.md`, 2026-10-10.

## STATUS — research note, fix not yet chosen

## 1. Mechanism

`numeric/factorize.rs` checks each non-root supernode for
`n_delayed_in > delayed_capacity`. When the check trips and CB is armed
(the default), the supernode runs with `may_delay = false` and
`ZeroPivotAction::PerturbToEps`. Every pivot that fails the threshold is
then set to `sign(d)·1e-10·‖DAD‖∞`. When `|d|` is at round-off level,
`sign(d)` is noise, so the inertia count can be off by the number of
noise-signed pivots.

The capacity is `min(subtree_ncol − own_ncol, K·own_ncol)`, with
`K = DELAY_CAPACITY_MULTIPLIER = 4` and a floor of 16
(`symbolic/supernode.rs`). The first term is a hard bound: no node can
receive more delays than that. The second term is a policy, added in
issue #55 because unbounded delay produced effectively dense root fronts
and >100 GB allocations on pinene_3200 and nql180
(`symbolic-delay-budget-2026-05-27.md`). The budget does not size any
buffer, since fronts are assembled at the expanded size. With CB off and
the budget exceeded, the code returns `DelayBudgetExceeded`.

## 2. Evidence

All runs: fresh `Solver`, `pivot_threshold = 1e-8`, `Auto` ordering,
M4 Pro. Load average was 40-80, so the timings are noisy, but both
configurations ran under the same conditions.

### 2.1 column25 decisive pair (reporter's `i011_s002`)

Oracle: the reporter's `eigsh` count of 75,596 negative eigenvalues
(nearest eigenvalue +3.47e-8), together with Sylvester's law. Each seed
is a diagonal congruence D·A·D with `d_i ∈ [0.7, 1.4]` (xorshift), so
the inertia is unchanged.

| config                     | wrong seeds | cb      | n_tiny  | nnzL        | time   |
|----------------------------|-------------|---------|---------|-------------|--------|
| default (K=4, CB on)       | 3 / 9 (+3, +1, +1) | 2–12 | 29–145 | 9.49–9.70M | ~25 s |
| K=2^40, CB off             | 0 / 9       | 0       | 0       | 9.04–9.08M  | ~11 s  |

On this matrix the budget is the cause of the wrong counts and also
makes the factorization slower. The unscaled matrix is correct here and
+1 on the reporter's x86/AVX-512 machine. Which machine gets it right is
decided by round-off, so congruence seeds are a better local reproducer
than the raw file.

### 2.2 kkt-mittelmann, all 618 matrices

- 613 factor. The other 5 files are corrupt (truncated, nnz mismatch, or
  containing `inf`).
- 0 wrong counts under the default, out of 598 matrices with a sidecar
  oracle.
- The budget binds (cb > 0) on only 6. On every other matrix the two
  configurations produce bit-identical factors.

| matrix               | default cb / n_tiny / nnzL / t | unbounded cb / n_tiny / nnzL / t |
|----------------------|--------------------------------|----------------------------------|
| marine_1600_0009     | 1 / 0 / 2.493M / 0.42 s        | 0 / 0 / 2.520M / 0.35 s          |
| marine_1600_0011     | 4 / 1 / 2.524M / 0.57 s        | 0 / 0 / 2.595M / 0.41 s          |
| marine_1600_0012     | 1 / 0 / 2.467M / 0.48 s        | 0 / 0 / 2.502M / 0.36 s          |
| marine_1600_0013     | 1 / 7 / 2.182M / 0.50 s        | 0 / 0 / 2.183M / 0.31 s          |
| marine_1600_0016     | 1 / 2 / 2.368M / 0.45 s        | 0 / 0 / 2.373M / 0.39 s          |
| qcqp1500-1nc_0000    | 16 / 2263 / 5.14M / 2.36 s     | 0 / 0 / 7.81M / 10.8 s           |

All counts are correct under both configurations. qcqp1500-1nc_0000 is
the cost case for lifting the budget: +52 % fill and about 4.6× the
time. Its default factor floors 2263 pivots, so its correct count is
luck, in the same way as column25 seeds 0, 2 and 3.

pinene_3200 (iterations 0–9) and nql180_0006 never reach the budget at
HEAD (cb 0). The blowups that motivated K = 4 have since been absorbed by
#46's `LdltCompress` pairing and related fixes, so lifting K cannot
change those matrices today.

## 3. Candidate fixes

1. **Lift the per-node policy cap.** Keep only the hard subtree bound
   (or use a much larger K). CB then never fires below the root. This is
   inertia-exact wherever the root front itself does not floor. Cost:
   the qcqp1500-1nc row above. Risk: a future pinene-like matrix grows
   dense fronts with no backstop.
2. **Lift the cap and add a memory backstop.** Allow delay until the
   expanded front exceeds a generous multiple of its symbolic size, such
   as front area > c·symbolic area. Only past that point engage CB. This
   follows MUMPS, which delays freely and treats workspace exhaustion
   (ICNTL(14)) as the only limit. The backstop must be per-node, not a
   global running total, to stay deterministic under the parallel tree.
3. **Report the uncertainty.** Add a stats field, and/or an inertia
   status, counting pivots floored with `|d| ≤ tol` under CB, so callers
   like pounce know the count may be off by up to k. This is orthogonal
   to fixes 1 and 2 and needed by fix 2, since its backstop can still
   fire.
4. **Swap-to-boundary** (`kkt-cascade-amplifier-2026-05-21.md`, Fix 1).
   Delay only the failing column, not the whole supernode tail. This
   reduces how much is delayed and so the cost of fixes 1 and 2. It is a
   kernel change in both the scalar and panel paths, so it is follow-up
   work.

Recommendation: fix 2 combined with fix 3. Set the backstop so that it
does not fire on any of the 613 matrices or on the column25 seeds,
calibrated against the qcqp1500-1nc and column25 front growth.

## 4. Open questions

- What front-area growth do the qcqp1500-1nc_0000 and column25 CB nodes
  actually reach? This sets c.
- Is qcqp1500-1nc's 4.6× time acceptable? It is one matrix in 613, and
  only on IPM iteration 0.
- The root front still runs with `may_delay = false`. Do any of the six
  matrices floor pivots at the root under the unbounded configuration?
  The aggregate n_tiny of 0 in every unbounded run says no.
