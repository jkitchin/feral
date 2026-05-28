# FERAL Context (auto-generated)

Generated: 2026-05-28T12:49:26Z

## Latest Session
File: dev/sessions/2026-05-28-01.md
```
# Session 2026-05-28-01

## Goal

Continuation of issue #56 ("Dense Thomson-Hessian per-iter throughput:
1.57×–2.34× slower than MUMPS on elec*"). Branch
`issue-56-thomson-hessian-throughput` already had Lever A (permute
cache + pre-built permuted_pattern) and the Phase 2/3 probe scaffold
from prior sessions; this session was charged with finishing the
remaining throughput levers identified by the per-phase probe,
re-validating, closing #56, and merging.

## Accomplished

- **Lever B (`c33f023`) — fused single-pass `contribextract` write.**
  Phase 2 sub-phase drill-down localized 27–32 % of total wall at
  n=200 to `dense_bookkeeping`, with 1435 µs (10 % of total wall) in
  `contribextract` and 289 µs of that in the resize-zero pre-pass.
  Replaced `resize(cdim², 0.0) + lower-triangle overwrite` (every
  lower-triangle cell written twice) with
  `reserve + unsafe set_len + single-pass loop` writing each cell
  exactly once (zero for upper-triangle, a-value for lower-triangle).
  Bit-identical to the prior contrib block — `extend_add` reads only
  `ci ≥ cj`, root-Schur extractor canonicalizes via transpose, and
  `parallel_corpus_parity` binary-compares the full buffer. Long-form
  safety comment at the `factor_frontal` site (f64 has no Drop →
  briefly-uninitialized bytes are sound; loop writes every cell
  before any read).

  Re-measurement at n=200 (darwin/aarch64, 9 warm reps):
  - factor min (par on): 19731 → 18648 µs (−5.5 %)
  - factor min (par off): 18528 → 16597 µs (−10 %)
  - contribextract: 1435 → 850 µs (−41 %)
  - contribzerofill: 289 → 3 µs (≈ gone)
  - factor min at n=100 (par on): 2977 → 2753 µs (−8 %)
  - n=50: within noise

- **Phase 3 probe extension (`b0b9b7e`) — already on branch from prior
  session, formalized this session.** `per_phase_breakdown_via_solver`
  drives `Solver::factor()` and snapshots `mc64_cache_hit_count()` /
  `scaling_info()` per rep. Finding: MC64 scaling cache is **correctly
  inactive** on Thomson — `pick_scaling_strategy` picks InfNorm, and
  InfNorm is deliberately not cached per #49 (replays a stale iter-0
  scaling on a drifted iter-N matrix). The Phase 2 prologue
  `scaling_us` is real recurring per-iter work, not a cache miss.

- **Lever C (`5de817b`) — vectorize InfNorm inner loop (hoist + pulp
  SIMD).** Live instrumentation of the KR loop revealed it never
  converges on Thomson: `max_dev` decays geometrically at ratio
  0.5/iter and plateaus at 6.77e-3 — six orders away from the 1e-8
```

## Git Status
```
5de817b feat(#56): Lever C — vectorize InfNorm inner loop (hoist + pulp SIMD)
c33f023 feat(#56): Lever B — fused single-pass contribextract write
b0b9b7e probe(issue56): Phase 3 — drive factor() through Solver to verify mc64 cache
14d1d58 feat(#56): Lever A — permute structure cache + pre-built permuted_pattern
21bb2f4 probe(#56): Thomson per-phase breakdown — bookkeeping > kernel
```

## Test Status
```
