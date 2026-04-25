# Phase 2.13 — New Tail Diagnostic Scope

**Status:** Scoping. Open questions, no implementation yet.

**Predecessor:** Phase 2.12 (commits `8a20315` + `cd9af06`) flipped
`AmalgamationStrategy::default()` from `Adjacency` to `Renumber`.
ACOPR30/CRESC100 dropped out of the corpus Top-10 worst sparse factor
ratio. The new Top-10 worst is dominated by KIRBY2_* (max 10.64×
MUMPS) and MUONSINE_* (max 7.12×).

A diagnostic dive on `KIRBY2_0007` and `MUONSINE_0000` under both
strategies (recorded in `dev/journal/2026-04-25-03.org`, 20:00 entry)
revealed two **distinct** root causes:

## 2.13a — Shape-dispatched amalgamation strategy

**Finding.** MUONSINE_0000 (n=1537, path-like etree) was 1.4× MUMPS
under Adjacency. Renumber default merged the chain into a single
ncol=32 root frontal that costs 1008µs by itself — total 5.5× MUMPS.
**Renumber actively regressed this family.**

KIRBY2 also has a near-path tree but without the ncol=32 root.

**Hypothesis.** Renumber wins on bushy IPM-KKT trees (ACOPR30/
CRESC100) and loses on path/near-path trees (MUONSINE/KIRBY2). A
cheap shape predicate at symbolic-time can pick per-matrix:
- multi-child internal node count / total internal nodes
- max child count
- depth-of-deepest-chain / supernode count

**Goal.** `AmalgamationStrategy::Auto` (parallel to
`OrderingPreprocess::Auto`, default since Phase 2.4.4). Default
becomes `Auto`; the predicate dispatches between Renumber and
Adjacency. Success criterion: corpus median sparse factor ratio
within ±2% of Adjacency baseline AND tail ACOPR30/CRESC100 keep the
60-67% factor reduction AND MUONSINE regression eliminated.

**Open questions.**
1. What is the cheapest shape predicate that separates these cases?
   Need a 1-pass-on-etree statistic that costs <5% of symbolic.
2. Should the predicate run before or after the merge prediction?
   (Cheap-first: shape predicate runs in O(n); merge prediction runs
   only if shape says "bushy".)
3. Is there a third strategy (e.g., Renumber-but-only-merge-non-path
   subtrees) that beats both? Probably premature; ship Auto first.

## 2.13b — Symbolic-phase setup overhead

**Finding.** KIRBY2_0007 (n=458, ratio 9.5× MUMPS):
- Numeric phase: 235µs (1.8× MUMPS — fine for this size)
- Symbolic phase: 924µs (6× MUMPS's *entire* factor 122µs)

The bench rolls symbolic+numeric into `factor_us`, so the headline
9.5× ratio is dominated by analyze-phase setup, not kernel work.

**Hypothesis.** AMD ordering, etree, supernode partition, Renumber,
small-leaf grouping, and (when on) MC64 each pay a per-call constant
that is small for n=4000 but dominates at n=458. The 924µs symbolic
on a 458-row matrix means O(2µs/row), which is large — MUMPS does
its analyze in a small fraction of that.

**Goal.** Identify which symbolic stage(s) carry the setup cost and
either:
- (i) cache symbolic across factorizations of the same pattern (an
  IPM iterates the *same* sparsity pattern with different numerical
  values; symbolic should run once);
- (ii) skip a stage when shape predicates predict no benefit (e.g.,
  small-leaf grouping does nothing useful when n_supernodes < 50);
- (iii) shrink a single-stage constant if one stage dominates.

**Open questions.**
1. Per-stage breakdown: of the 924µs on KIRBY2_0007, what fraction
   is AMD vs etree vs supernode-partition vs Renumber vs
   small-leaf vs MC64?
2. Does feral expose an analyze-once API path? (`SparseSolverState`
   in `dev/plans/policy-traits-api.md` may already have this scope —
   check.)
3. ~~Is the bench's `factor_us` measurement methodology correct?~~
   **Resolved 2026-04-25.** MUMPS oracle uses `JOB=4` (analyze +
   factor combined), feral bench times symbolic + numeric together
   as a single block (`src/bin/bench.rs:1445-1476` with explicit
   comment). The methodology is apples-to-apples. The 9.5× ratio
   on KIRBY2_0007 is real; the gap is primarily feral-amd taking
   770µs on n=458 vs MUMPS getting analyze+factor done in 122µs.

## Suggested execution order

1. ~~**Add per-stage symbolic profiler.**~~ **Done 2026-04-25**
   (commit `2143658`). `SymbolicProfiler` lives at
   `src/symbolic/profiler.rs`; instrumentation covers 14 stages in
   `symbolic_factorize_with_method`.

2. ~~**Confirm KIRBY2 hypothesis.**~~ **Done 2026-04-25.** 5-run
   median per-stage breakdown:

   - KIRBY2_0007 (n=458): `ordering` 773µs / 85.5%, everything else
     <50µs each. AMD per-call cost is the entire problem.
   - MUONSINE_0000 (n=1537): `ordering` 440µs / 46.9%, `postorder`
     206µs / 21.9%, `renumber` 159µs / 16.9%.

   Hypothesis refined to: **AMD per-call setup cost dominates
   symbolic on small-n.**

3. ~~**Verify bench methodology.**~~ **Done 2026-04-25.** MUMPS
   oracle uses `JOB=4` (analyze + factor combined); feral bench
   times symbolic + numeric together. Apples-to-apples.

4. **Implement `AmalgamationStrategy::Auto`.** Cheap shape predicate
   from the diagnostic, dispatch between Renumber and Adjacency.
   Test with parity tests + corpus bench. Success criteria above.

5. **Decide between symbolic caching vs AMD per-call shrink.**
   Both target the dominant stage on small-n; caching also targets
   IPM-iteration economics broadly. Recommend a 5-call AMD
   sub-stage profiler probe before committing to the fix path —
   if AMD's 770µs is dominated by allocations and hash sizing,
   per-call shrink is the cheap win; if it's dominated by O(n α n)
   work, symbolic caching is the only meaningful lever.

## Out of scope

- Re-flipping the default away from Renumber. User decided to stay
  with Renumber default; Phase 2.13a fixes MUONSINE properly via
  Auto rather than reverting.
- New corpus matrices. Use the existing 153k for both phases.
- Numeric kernel work. The KIRBY2 numeric phase at 1.8× MUMPS is
  already good; do not chase it.

## References

- `dev/journal/2026-04-25-03.org` 20:00 entry: full diagnostic numbers
- `dev/research/phase-2.12-column-renumbering.md`: Phase 2.12 research
- `dev/decisions.md` 2026-04-25 Phase 2.12 entries: default-flip
  reasoning
- `src/bin/diag_strategy_compare.rs`: 5-run profiler probe template
- `src/numeric/factorize.rs:72-219`: `Profiler` types to extend
