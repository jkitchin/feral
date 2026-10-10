# Issue #215: cascade-break on a reused value-dependent analysis

Date: 2026-10-10. Status: research note before implementation.

## Problem

pounce#999 `column25_integral.nl` stalls after #214. On a `Solver` reused
across IPM iterates 0-4 (same pattern), iterate 4 reported neg 75597 on the
reporter's machine; a fresh `Solver` reports the true 75596. Both factors
engage cascade-break (CB). Under CB, #214 floors a threshold-failing pivot to
`sign(d)·1e-10·‖DAD‖`, so a pivot whose sign comes from round-off can flip
the inertia. CB is a budget overflow path, not an inertia guarantee.

## Findings (journal 2026-10-10-01 §10:14)

- Rebuilt matrices match the issue bit for bit (same min|d| per iterate).
  The +1 does not reproduce on this machine (Apple M4 Pro). The CB counts
  differ from the reporter's (warm 1 vs 6, fresh 4 vs 2), so the outcome is
  platform-sensitive at round-off margins.
- The state that matters is the cached symbolic analysis. Calling
  `invalidate_symbolic_cache()` before it004 reproduces the fresh factor
  exactly. The MC64 scaling cache hits 0 times on this sequence.
- The analysis is value-dependent. Ordering escalation (#102 follow-up)
  forces `OrderingPreprocess::LdltCompress`, whose MC64 matching is computed
  from the values. The analyses at it000 and it004 differ (25024 vs 24684
  supernodes) on the same pattern. The 2026-05-17 note's claim that "the
  symbolic is pattern-only" does not hold on this path.
- Fresh is not uniformly better: here fresh has more CB nodes, while in the
  reporter's run warm had more. Any fix that blindly prefers one analysis
  regresses one of the two runs.

## Options considered

1. Exact inertia under budget overflow (swap-to-boundary delays instead of
   flooring). Correct in principle, but it needs delayed-capacity growth in
   the numeric phase. That is multi-session work and was not chosen.
2. A user-facing cheap re-analysis (`invalidate_symbolic_cache()` already
   exists). It leaves detection to the caller. Pounce has no signal for when
   to call it.
3. **Auto re-analyse on CB (chosen by the user).** When a factor on a
   *reused* analysis whose preprocess is `LdltCompress` engages CB, redo the
   analysis on the current values, refactor, and keep the factor with fewer
   CB nodes. The tie-break is fewer `n_threshold_fail_accepts + n_tiny`, and
   the incumbent wins exact ties. This is a mitigation: it lowers the chance
   of a round-off sign flip by choosing the factor that floored fewer
   pivots. It does not prove the inertia is exact.

## Design constraints

- Fire only when the analysis was reused (`pattern_reused`) and is
  value-dependent (`resolved_preprocess == LdltCompress`). A pattern-only
  analysis would come out identical, so a re-analysis would be wasted.
- Bound the cost. If CB survives the re-analysis on every iterate, paying
  for one extra symbolic and numeric factor per iterate is too much. Use a
  per-pattern miss latch: after 2 consecutive re-analyses that are not
  adopted, stop re-analysing on this pattern. Reset the latch on adoption
  and on a pattern change. (The first draft counted "winner still has CB"
  as a miss. In the reporter's run the fresh analysis also has CB, so that
  rule would have latched before iterate 4. See journal §11:55.)
- Keep the rest of the warm state consistent. If the new factor wins, adopt
  the new symbolic, factors, inertia and `prev_max_n_delayed_in`. The
  scaling does not depend on the analysis, so the MC64 cache needs no
  change.
- Observability: add a counter for re-analyses run and one for re-analyses
  adopted. Add an opt-out builder method; the default is on.
- Related stats bug: the escalation re-entry into `factor()` keeps the
  pattern fingerprint, so `last_factor_stats().pattern_reused` reads true on
  a fresh Solver's first factor. Thread the outer value through the re-entry.
