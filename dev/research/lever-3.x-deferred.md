# Levers 3.1 (FMA fallback) and 3.2 (wider micro-kernel NR) — deferred

**STATUS: DEFERRED (2026-05-31).** Both Tier-3 levers from
`dev/research/perf-review-2026-05-31.md` are deferred with rationale; neither
implemented. They are the lowest effort/reward in the sweep on this hardware.

## Lever 3.2 — wider micro-kernel NR (4 → 6/8)

The perf-review (§2 Tier-3 #2) already qualifies this: *"diminishing returns on
the small fronts that dominate, more remainder code. Measure only after 1.1/1.2
land."* Two reasons it is deferred:

1. **Gated on 1.2, which is itself deferred.** Wider NR shares more of the
   L-panel load across accumulators — but the trailing Schur update is
   *memory-bandwidth-bound* (the wall Lever 1.1 hit and Lever 1.2 was written to
   address). Adding arithmetic-throughput width does not help a bandwidth-bound
   kernel until the bandwidth problem (1.2: cache blocking + packing) is solved.
2. **Sub-noise-floor + unmeasurable here.** Like 1.2, any gain is in the 10–20%
   band, below the run-to-run noise floor on this shared/contended machine
   (Lever 1.1's identical A/B swung 1.2×–2.5× under load). Cannot be validated.

Revisit alongside 1.2 on idle hardware with hardware counters.

## Lever 3.1 — FMA with a boundary-safe fallback

The perf-review (§2 Tier-3 #1) rates this "platform-dependent / low priority,
high complexity, conditional payoff — keep opt-in." On **this** machine it is
worse than low-priority — it is ~zero-gain with real correctness risk:

1. **~0% payoff on arm64 (this box).** `uname -m` = `arm64`. Prior measurement
   (decisions.md 2026-04-14): nofma vs FMA is 1.87 → 1.86 on the Apple-Silicon
   target — noise. The 4 non-FMA accumulators already saturate the NEON pipes.
   FMA is only *potentially* material on x86 AVX-512, which cannot be tested
   here.
2. **Correctness risk.** FMA changes rounding (single rounding of `a*b+c` vs
   two), which flips inertia on ~30/154k boundary matrices
   (tried-and-rejected 2026-04-14). A boundary-safe fallback
   (detect-pivot-within-k·eps, fall back to scalar) is *required* and is
   high-complexity — a lot of careful code to defend a benefit that is ~0% on
   the only hardware available.
3. **Already opt-in.** `BunchKaufmanParams::fma` exists and defaults `false`;
   the infrastructure is present should an x86 gap ever be measured. Nothing to
   build now.

Revisit only if/when an x86 AVX-512 host shows a measured factor-time gap large
enough to justify the boundary-safe fallback work.

## Net

The perf-lever sweep implements **Lever 1.1** (intra-front parallel Schur);
1.2/2.1 are deferred-with-plan, 2.2 was already implemented (Phase 2.4.4), and
3.1/3.2 are deferred here. 1.1 is the one net-new win and the right stopping
point for this hardware.
