# Scaling-aware `LdltCompress` skip — design analysis + negative result

**Status:** Research note (precedes implementation, per spec §5.1). The
proposed change was investigated and **rejected** on empirical evidence; no
implementation landed.
**Date:** 2026-06-06
**Author:** session 2026-06-06-04
**Related:**
- `dev/research/mc64-dense-column-2026-06-06.md` (session -03): the inner-loop
  fast path, closed with an impossibility proof. This note closes the *other*
  dense-column follow-up lead (the "safe win", audit §8.1 option (b)).
- `dev/research/scaling-audit-2026-06-06.md` §8.1.
- Issue #80 (MC64 preprocessor cost).
- `src/symbolic/mod.rs` (LdltCompress branch, `pick_ordering_preprocess`),
  `src/symbolic/ldlt_compress.rs` (`build_supermap`),
  `src/scaling/mod.rs` (`pick_scaling_strategy`, `compute_scaling_with_cache`),
  `src/numeric/solver.rs` (`factor()` phase order).

---

## 1. The proposed "safe win"

From session -03's checkpoint and the user request: in the symbolic
`LdltCompress` branch the MC64 matching is run once per pattern
(`compute_mc64_cache`, mod.rs:755) and cached for `Mc64Symmetric` numeric
scaling. The cache is reused by numeric scaling **only** when the resolved
strategy is `Mc64Symmetric` (`compute_scaling_with_cache`, scaling/mod.rs:342).
The idea: when the resolved scaling will *not* reuse the cache
(Identity/InfNorm/External), the symbolic MC64 is "purely speculative" — so
skip it.

## 2. Two design facts established first

### 2.1 Symbolic runs *before* scaling resolution, and is cached

In `Solver::factor()` (solver.rs:738) the phase order is:
- Step 3 (line 805): `symbolic_factorize_with_method(...)` — **only on a cache
  miss** (`self.last_symbolic.is_none()`). Symbolic is cached per pattern
  fingerprint; the LdltCompress MC64 runs **once per pattern**, not per factor.
- Step 3.75 (line 942): sticky-Auto resolution.
- Step 3.8 (line 962): value-bounded MC64 scaling-cache reuse.

So symbolic is blind to the resolved scaling. To make it scaling-aware we would
compute the predicate from `self.numeric_params.scaling` (plus
`pick_scaling_strategy(matrix)` when `Auto`) *before* the symbolic call and
plumb a `will_reuse_mc64: bool` into `symbolic_factorize_with_method`. On the
first factor (the only one where symbolic runs) `auto_picked_strategy` and the
value-bounded cache are both empty, so `pick_scaling_strategy` is the exact
predictor of what numeric scaling will choose. Plumbing is feasible.

### 2.2 The MC64 in `LdltCompress` is **load-bearing for compression**, not speculative

`LdltCompress` is a port of MUMPS `ICNTL(12)=2` (Duff-Pralet symmetric
matching + quotient-graph compression, ldlt_compress.rs:1-14). `build_supermap`
**walks the MC64 matching permutation's cycle structure** to contract 2-cycles
into super-variables (ldlt_compress.rs:39-77). The MC64 matching *is* the
compression input — there is no compression without it. "Skip the speculative
MC64" therefore really means "skip compression and fall through to the
uncompressed ordering" — which **changes the ordering** (not correctness:
inertia is exact under any ordering; the corpus residual gate still applies).

So the proposed change is only a win if, for the affected matrices, compression
is not worth its MC64 cost. That is an empirical question.

## 3. Empirical bucket analysis

`probe_compress_scaling_bucket` (first `.mtx` per family, all three corpus
roots, 1006 families) tallies the two independent routing decisions:

```
LdltCompress chosen   : 376
  + Mc64 scaling      : 118   (MC64 shared with scaling — gate keeps compression)
  + non-Mc64 scaling  : 258   (TARGET bucket — scaling won't reuse the cache)
```

The 258-family target bucket is **not vacuous** and is **not all small**: it
includes large dense-column matrices where MC64 is genuinely expensive —
INDEFM (n=100000, max_col_deg=100000), SINQUAD2 (5000/5000),
ex8_2_3 (18791/3132), ex8_2_2 (9453/1894), ORTHREGF (6405/1601),
ROSEPETAL (3000/2001) — all `InfNorm`-scaled. So the lead looked promising:
rocket-like dense columns whose MC64 would not be reused.

## 4. The cost/benefit measurement refutes the premise

`probe_compress_costbenefit_argv` (symbolic+numeric, `None` vs `LdltCompress`,
5-run median, `OrderingMethod::Amd`, release):

| matrix | n | max_col_deg | tot_None (µs) | tot_Compress (µs) | Δ% | verdict |
|---|---|---|---|---|---|---|
| ROSEPETAL | 3000 | 2001 | 5 965 926 | 1 452 187 | **−75.7%** | compress WINS |
| ex8_2_2 | 9453 | 1894 | 215 758 | 214 777 | −0.5% | compress |
| ex8_2_3 | 18791 | 3132 | 957 786 | 968 299 | +1.1% | neutral |
| INDEFM | 100000 | 100000 | 5 030 355 | 5 147 508 | +2.3% | neutral |
| SINQUAD2 | 5000 | 5000 | 17 232 | 18 210 | +5.7% | None |
| ORTHREGF | 6405 | 1601 | 6 224 | 11 936 | **+91.8%** | None wins |
| CRESC100 | 806 | 201 | 509 | 751 | +47.5% | None (sub-ms) |

The crux is **ROSEPETAL vs ORTHREGF** — both large, both a near-dense column,
both `InfNorm` (won't reuse MC64), structurally similar, **opposite verdicts**:

- **ROSEPETAL**: compression pays 0.68 s of MC64 (sym 0.027 s → 0.67 s) but the
  compressed ordering gives an **8× numeric speedup** (num 5.72 s → 0.77 s),
  netting a **75% total win**. Reproducible (−75.7%, −75.0% on reruns).
- **ORTHREGF**: compression's MC64 adds ~5.6 ms with **zero numeric benefit**
  (num 1.67 ms ≈ 1.66 ms), a **90% total loss**. Reproducible (+91.8%, +89.4%).

## 5. Conclusion — the scaling-reuse signal is the wrong predictor

The value of `LdltCompress` is its **numeric fill reduction**, which is
- sometimes the single largest performance lever (ROSEPETAL: 5.97 s → 1.45 s),
- entirely **independent of the scaling choice**, and
- **not predicted** by max column degree, MC64 cost, or n (ROSEPETAL's MC64 is
  68× more expensive than ORTHREGF's, yet ROSEPETAL is the win).

"Scaling won't reuse the MC64 cache" therefore carries **no information** about
whether compression pays off. A gate keyed on it would **regress the
fill-reduction wins** (ROSEPETAL by ~4×, ex8_2_2 slightly) to save a few
milliseconds on the overhead-only losses (ORTHREGF, SINQUAD2, the sub-ms small
matrices). That is not a safe win — it trades a large, real regression for a
small, marginal saving. **Rejected; not implemented.**

## 6. What the real lever is (separate, harder, not this task)

The losses (ORTHREGF, and the KIRBY2/MUONSINE cases that motivated
`diag_compress_costbenefit`) are real: `LdltCompress` is sometimes pure MC64
overhead with no fill benefit. The correct fix is a **compression cost/benefit
gate** that estimates fill reduction vs MC64+ordering cost — **orthogonal to
scaling**. The current cheap proxy is `pick_ordering_preprocess`'s low-degree
fraction (≥30% cols nnz≤2), which has false positives (ORTHREGF) and would have
false negatives if tightened naively (it must keep ROSEPETAL). Building a
cheap, reliable benefit predictor that separates ROSEPETAL from ORTHREGF is an
open problem (the obvious structural features do not separate them) and a
distinct workstream from the dense-column follow-up. Not pursued here.

A trivially-safe micro-optimization exists but is not worth the plumbing:
when scaling won't reuse the cache, `cached_mc64` could be dropped to `None`
after `build_supermap` to free its O(n) vectors (read by nothing thereafter).
Pure memory hygiene, no time saved, no ordering change. Deferred — it would
couple symbolic to the scaling decision for a few MB on the largest matrices.

## 7. Status of the dense-column follow-up

Both leads are now closed with negative results:
- **inner-loop fast path** (option (a)) — impossibility proof + SPRAL
  confirmation (`mc64-dense-column-2026-06-06.md`).
- **scaling-aware symbolic skip** (option (b)) — refuted here: the
  scaling-reuse signal does not predict compression cost/benefit; skipping
  would regress fill-reduction wins.

The dense-column MC64 cost is inherent (matches SPRAL); the symbolic
`LdltCompress` MC64 is load-bearing for a fill reduction that is often the
dominant performance lever, independent of scaling. Issue #80's reported
problem (the pf22 near-tree heap-realloc) remains fixed and guarded by
`6699f09`.
