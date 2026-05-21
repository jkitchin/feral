# FERAL Context (auto-generated)

Generated: 2026-05-21T18:19:11Z

## Latest Session
File: dev/sessions/2026-05-21-01.md
```
# Session 2026-05-21-01

## Goal

Implement Track B2 of the per-factor cost cluster plan — eliminate the
per-call MC64 Hungarian on `rocket_12800` via a value-bounded MC64
scaling cache. Mid-session the goal pivoted (with human approval) to
**Track A**: investigate the `pinene_3200` iter 6-9 factor-time
explosion, which is 98 % of that problem's wall time.

## Accomplished

### B2 — value-bounded MC64 scaling cache (landed, then pivoted off)

- Wrote `dev/plans/mc64-value-bounded-cache.md`; implemented
  `src/scaling/value_bound.rs` (value-bound pure functions, 10 tests)
  and the Solver-scope `Mc64ScalingCache` in `src/numeric/solver.rs`
  (`with_mc64_cache` builder, cache-hit injection via
  `ScalingStrategy::External`, 5 integration tests).
- The cache is **correct and fully tested** but has **no measured
  corpus payoff**: the gate metric (diagonal dominance of `D·A·D`) is
  confounded by the IPM δ-regularization trajectory, and the MC64
  Hungarian it eliminates is < 2 % of factor cost. Ships as latent
  infrastructure. See `decisions.md` / `tried-and-rejected.md`.

### Fixed — `External` scaling 10× solve bug (pre-existing, latent)

- B2's integration test `mc64_cache_hit_bit_matches_cache_off` caught
  it: `ScalingStrategy::External` paired a real scaling vector with
  `ScalingInfo::NotApplied`. The factor applies `D·A·D`
  unconditionally; the solve keys un-scaling off
  `scaling_info != NotApplied` — so an `External` solve returned
  `D⁻¹A⁻¹D⁻¹b` (exactly 10× on `tridiag(6,10,1)`).
- Fix: the `External` arm now returns `ScalingInfo::Applied`.
  `NotApplied` is now exclusively `Identity` (genuine all-ones).
  Verified bit-identical across repeated calls; 302 lib tests + full
  suite green. Committed `c990def`.

### Track A — pinene_3200 iter 6-9 blowup characterized (A1)

- Established this is **issue #8**, and that the warm-state hypothesis
  was already disproved (2026-05-17): the cascade is structural to the
  iter-N matrix's *numeric content*, standalone ≈ warm.
- Ran `diag_pinene_pivot_cliff` (per-supernode 2×2 / delayed-pivot
  stats) on iterates 0008 and 0009 (n=127995, nnz=733k). **Direct
  evidence — the root front is a fully dense block, ~14 % of n:**

  | metric            |   0008 |   0009 |
  |-------------------|-------:|-------:|
  | root front nelim  |  15446 |  17538 |
```

## Git Status
```
70f2e44 diag(dense): localize pinene KKT cascade — amplifier × two triggers
d3d93d2 docs(trackA): localize pinene cascade to the 2x2 stability gate
76174bd docs(session): checkpoint 2026-05-21-01 — B2 landed, pivot to Track A
c990def feat(scaling): value-bounded MC64 scaling cache (B2) + fix External 10× bug
9512c0a perf(profiler): instrument numeric prologue sub-phases (Track B1)
```

## Test Status
```
