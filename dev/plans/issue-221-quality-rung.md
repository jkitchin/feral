# Plan: evidence-based `increase_quality` rungs (issue #221, PR #222)

Research: `dev/research/issue-221-quality-rung-admissibility.md`.

## Change

1. `SparseFactors::max_abs_l()`: realized `max |L_ij|`, one pass over L.
   The unit diagonal and the in-block coupling of 2×2 pivots are excluded.
   Surfaced as `FactorStats::max_abs_l`.
2. `Solver::increase_quality`, stage 2 only (the stage-1 scaling flip is
   unchanged):
   - **Judge the previous rung once.** `rung_before` holds the stability
     `(violations, max |L|)` of the factor that prompted the last bump.
     The next call consumes it. If the current factor is worse (more
     violations, or equal violations and a larger `max |L|`), restore
     `ladder_start`, set `Exhausted`, and return `false`.
   - **Decline an admissible rung.** If the current factor has no
     violations and `max |L| · u_next <= 1`, return `false` and change
     nothing.
   - Otherwise bump as before, recording `ladder_start` (first bump only)
     and `rung_before`.
   - Without a stored factor (`last_factors == None`), nothing changes, so
     U1–U5 hold by construction.

## Tests (hand oracles), `tests/issue_221_quality_rung.rs`

- `max_abs_l` on `tridiag(b, 1, b)` in natural order equals the hand
  recurrence `l_k = b / d_k`; on a diagonal matrix it is 0.
- A clean factor (dominant tridiagonal) declines rung 1e-8 → 1e-6.
- Floored pivots (diagonal with zeros, `PerturbToEps`) keep the rung.
- A rung followed by more floored pivots is undone, back to the ladder
  start, even after two rungs. The ladder is then exhausted.
- A rung that is not worse is kept, and the ladder climbs.
- A rung is judged only by the next call (the pooling_rt2stp pattern).

## Acceptance (pounce 68e438d0 + this branch, `solve_quality_audit=no`)

- nql180, brainpc1, brainpc2 solve within 300 s.
- deb7 exact unchanged (Optimal/180). pooling_rt2stp and
  square_flowsheet_resto keep their verdicts.
- qcqp1000-2c and qcqp1500-1c are unaffected (they never escalate).
- `cargo test --release` green.
