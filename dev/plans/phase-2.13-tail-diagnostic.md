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

5. ~~**Decide between symbolic caching vs AMD per-call shrink.**~~
   **Done 2026-04-25.** Sub-stage probe
   (`src/bin/diag_amd_substages.rs`) overturned the framing.
   The 770 µs `ordering` stage on KIRBY2_0007 is **MC64**, not
   AMD. 5-run median attribution:

   - mc64                    : 827 µs (94.2%)
   - build_supermap          :  12 µs (1.4%)
   - compress_pattern        :  15 µs (1.7%)
   - AMD entire (prep + ws_new + run_el + fin_p + post): 20 µs (2.3%)
   - expand_permutation      :   0 µs

   AMD itself takes ~20 µs on n=458; the AMD `run_elimination`
   inner loop is 16 µs of that. Per-call shrink targets a slice
   that is at most 2-3% of the bottleneck — rejected on those
   grounds. Symbolic caching is still attractive for IPM workloads
   (one analyze amortized across 30+ iterations) but does not
   help the corpus bench (1 factor per matrix).

   The cheapest single fix is **Phase 2.13c — tighten
   `pick_ordering_preprocess`'s LdltCompress gate** so KIRBY2-class
   small-n matrices do not pay the ~830 µs MC64 cost when AMD on
   the uncompressed pattern would only have cost ~25 µs.

## 2.13c — Tighten LdltCompress gate (paused — data overturns the framing)

**Status:** paused 2026-04-25. The cost/benefit probe
(`src/bin/diag_compress_costbenefit.rs`) shows the proposed change
would regress the matrices it was meant to fix.

5-run-median total wall-time (symbolic + numeric), in microseconds:

| matrix         |   n  | None | Compress | delta | verdict |
|----------------|-----:|-----:|---------:|------:|---------|
| KIRBY2_0007    |  458 | 1209 |     1045 |  -164 | compress wins |
| MUONSINE_0000  | 1537 | 2093 |     1354 |  -739 | compress wins |
| ACOPR30_0067   |  564 |  594 |      810 |  +216 | None wins |
| CRESC100_0000  |  806 |  642 |      851 |  +209 | None wins |
| LAKES_0000     |  324 |  247 |      258 |   +11 | neutral |
| NELSON_0000    |  387 |  294 |      298 |    +4 | neutral |
| SWOPF_0000     |  175 |  157 |      155 |    -2 | compress |

The step-5 sub-stage probe correctly identified that 800 µs of the
KIRBY2 `ordering` stage is MC64. What it missed is that the
*numeric* stage on the non-compressed ordering balloons by an
amount that cancels the MC64 cost: 1028 µs → 245 µs on KIRBY2,
1612 µs → 619 µs on MUONSINE. Compression is *already* the
better choice for those matrices; the 9.5× MUMPS headline reflects
that.

The actual gate flaw is a different shape:

- Current predicate (`n >= 128 && low_degree_cols/n >= 0.30`) is
  *necessary* — catches KIRBY2, MUONSINE, SWOPF wins.
- Not *sufficient* — mis-fires on ACOPR30, CRESC100 where
  compression triggers but does not pay back in numeric savings.
- Tightening on `n` is wrong (would lose KIRBY2 and MUONSINE).

The right discriminator is something the current predicate does
not compute (probably `ncmp/n` after the compression, or a finer
structural test). But:

- ACOPR30/CRESC100 are *not* in the corpus Top-10 worst-ratio
  under the current Phase-2.12+2.13a defaults. Phase 2.12 already
  cut their factor 60-67% via Renumber.
- The absolute saving from fixing the mis-fire is ~200 µs per
  matrix.

The honest conclusion: Phase 2.13c as framed is rejected. The
corpus tail is not a gate-design problem. KIRBY2 will continue to
show ~8-9× MUMPS per single factor; the workload-realistic answer
is **symbolic caching** so MC64's 800 µs amortizes to ~25 µs per
IPM iteration. That is now the recommended next direction.

If a future session wants to revisit a finer gate, the open
follow-up is:

- (a) Extend the cost/benefit probe to print `ncmp/n` for each
  matrix so the structural difference between compress-wins and
  compress-loses cases is visible.
- (b) Find a cheap predicate that gates ACOPR30/CRESC100 out
  without losing KIRBY2/MUONSINE.
- (c) Only then change `pick_ordering_preprocess`.

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
