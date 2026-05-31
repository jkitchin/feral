# Lever 2.2 — symbolic-phase speedups

**STATUS: ALREADY IMPLEMENTED (verified 2026-05-31). No net-new work.**

Source: `dev/research/perf-review-2026-05-31.md` §2 Tier-2 #2. The perf-review
listed two symbolic-speedup items as future work; on inspection **both are
already implemented** ("Phase 2.4.4", landed before this lever sweep began). The
perf-review over-stated the remaining work. This note records the verification.

## Sub-item A — kill the "double-Hungarian" (cache MC64 across compress→scale)

**Implemented.** When the `LdltCompress` ordering preprocess runs its MC64
symmetric matching, the matching is computed **once** and cached on the returned
`SymbolicFactorization`, and the numeric phase reuses it for `Mc64Symmetric`
scaling instead of recomputing:

- compute once + cache: `src/symbolic/mod.rs:605` (`compute_mc64_cache`) →
  `:614` (`cached_mc64 = Some(cache)`).
- consume in numeric: `src/scaling/mod.rs:298-300` —
  `ScalingStrategy::Mc64Symmetric => match cache { Some(c) =>
  mc64::scaling_from_cache(c) /* O(n) */, None => compute_symmetric(matrix) }`.

So on matrices where compression runs **and** scaling resolves to
`Mc64Symmetric`, the Hungarian runs once, not twice. (Residual: when `Auto`
scaling picks `InfNorm` instead, the compression MC64 has no sharing partner —
but that is inherent, not a missing optimization; the cache cannot help a path
that does not run MC64.)

## Sub-item B — auto-dispatch compression on predicted-tail matrices

**Implemented.** `OrderingPreprocess::Auto` is the default
(`src/symbolic/supernode.rs:108,116`) and resolves per-matrix via
`pick_ordering_preprocess` (`src/symbolic/mod.rs:347-369`): compress only when
`n >= 128` **and** ≥30% of columns are degree-≤2 (the arrow-KKT slack
signature). This mirrors `pick_scaling_strategy`'s `Auto`. Cheap O(nnz) shape
scan; no MC64 needed to decide.

## The perf-review's "tighter gate (compRat ≤ 0.7)" is not viable as stated

The perf-review floated gating on the *MC64 compression ratio* (compRat ≤ 0.7).
That is circular: the compression ratio is only known **after** running the MC64
matching, so using it to decide *whether* to run MC64 defeats the purpose. The
implemented cheap shape predicate (degree-≤2 fraction, computable without MC64)
is the correct design. No change recommended.

## Conclusion

Lever 2.2 requires no implementation — it is done and on `main`. Calibration
evidence and the geomean-stability check are in
`dev/research/phase-2.4.4-compression-auto-dispatch.md` and the 2026-04-23
session. Marking the lever complete (already-landed).
