# Phase 2.6.5 — LDLᵀ-aware ordering preprocessing

**Status:** pre-implementation research.
**Date:** 2026-04-21.
**Plan:** `dev/plans/phase-2.6.5-ldlt-compressed-graph.md` (to be written).
**Paper(s):** Duff & Pralet 2005 "Strategies for scaling and pivoting for
sparse symmetric indefinite problems", SIAM J. Matrix Anal. Appl.
27(2):313–340. Companion to the MC64 scaling already used by `feral`.
**MUMPS reference:** `ICNTL(12) = 2` (compressed-graph ordering)
implemented in `mumps/src/dana_LDLT_preprocess.F`; dispatch in
`dana_driver.F:3683-3900`.

## The authoritative algorithm (from the MUMPS source)

The MUMPS expert report answers the question "what does ICNTL(12) actually
do" authoritatively. The feral phase plan's characterisation ("collapses
the constraint block of a bordered KKT matrix into a single
super-variable") is **inaccurate**. The real algorithm is:

1. Run the symmetric weighted matching `DMUMPS_SYM_MWM` — MC64 on
   `A + Aᵀ` followed by 2-cycle extraction — producing a symmetric
   permutation `PIV` in which large off-diagonals are clustered into
   2×2 diagonal blocks. Each 2-cycle in the permutation is a
   **tentative 2×2 pivot**.
   (`dana_LDLT_preprocess.F:270`, `dana_aux.F:1925`.)
2. Build the super-variable map `ICMP`: each 2-cycle pair
   `(p1, p2)` maps to one super-variable; each 1-cycle (fixed point)
   is its own super-variable.
   (`dana_LDLT_preprocess.F:145-165`.)
3. Contract the symmetric adjacency graph: for every edge `(i,j)` in
   the pattern, produce an edge `(ICMP[i], ICMP[j])` in the compressed
   graph; drop self-loops and deduplicate. Result is `NCMP × NCMP`
   in CSR-ish form.
   (`dana_LDLT_preprocess.F:199-264`.)
4. Hand the compressed graph to the ordering kernel (AMD / METIS /
   SCOTCH / etc.) to get a super-permutation of length `NCMP`.
5. Expand the super-permutation back to length `n`. Each
   super-variable slot in the super-permutation emits its 1 or 2
   originals **adjacent** in the output, so the downstream
   factorization sees each tentative 2×2 pivot pair in consecutive
   columns.
   (`DMUMPS_EXPAND_PERMUTATION`, `dana_LDLT_preprocess.F:88-121`.)

There is **no KKT-specific logic**. Bordered KKT matrices
`[H Jᵀ; J 0]` benefit incidentally: the zero `(2,2)` block forces
MC64 to match constraint rows with their Jacobian columns, which
produces 2-cycles, which the compression contracts.

## Why this matters for feral (corpus survey)

`src/bin/diag_mc64_cycles.rs` classifies the MC64 matching cycle
structure on every matrix in `data/matrices/kkt/`. Definitions:

- `compRat = n_compressed / n`
- `n_compressed = n1 + pairs_from_short + pairs_from_long`
  where `n1` = 1-cycles, `pairs_from_short = #{j : perm[j] ≠ j,
  perm[perm[j]] = j} / 2`, `pairs_from_long = ⌊k/2⌋` per cycle of
  length `k ≥ 3` (matches MUMPS `DMUMPS_SYM_MWM` which decomposes
  long cycles into pairs + a singleton).

Top-10 worst-ratio matrices (`factor_us` / MUMPS) on the current
bench — all ten have strong compression leverage:

| matrix          |    n |   n1 | pairs | compRat |
|-----------------|-----:|-----:|------:|--------:|
| MUONSINE_0000   | 1537 |  513 |   512 |   0.667 |
| CRESC100_0000   |  806 |  406 |   198 |   0.752 |
| KIRBY2_0007     |  458 |  156 |   151 |   0.670 |
| HAHN1_0259      |  715 |  237 |   239 |   0.666 |
| KIRBY2_0006     |  458 |  156 |   151 |   0.670 |
| KIRBY2_0008     |  458 |  156 |   151 |   0.670 |
| GAUSS2_0000     |  758 |  258 |   250 |   0.670 |
| VESUVIO_0011    | 3083 | 1033 |  1025 |   0.668 |
| VESUVIO_0019    | 3083 | 1033 |  1025 |   0.668 |
| VESUVIO_0013    | 3083 | 1033 |  1025 |   0.668 |

Every worst-ten matrix shrinks by ~33% under compression, which
means AMD/METIS runs on a graph of 2/3 the vertices. Fill reduction
at the ordering stage typically compounds into numeric-phase
speedup proportional to fill², so this is a real lever.

Full 169 585-matrix histogram of `compRat`:

```
bin[comp_rat]  count   frac
  [0.3, 0.4)    5139    3.0%
  [0.4, 0.5)    6047    3.6%
  [0.5, 0.6)   17604   10.4%
  [0.6, 0.7)   15598    9.2%   ← all top-10 worst matrices sit here
  [0.7, 0.8)   13676    8.1%
  [0.8, 0.9)   14298    8.4%
  [0.9, 1.0)    7010    4.1%
  [1.0, 1.0)   90213   53.2%   ← no compression possible
```

Read-out: ~47% of the corpus has *some* compression leverage, ~26%
has significant leverage (ratio ≤ 0.7). The remaining 53% are
cases where the matching is fully on-diagonal (fixed points only),
which happens for well-conditioned problems where the diagonal
dominates the off-diagonals — ordering on those is already cheap,
and compression is a no-op on them (same graph size, same
ordering).

## Why this is *not* documented-vacuous like 2.5.4

Contrast with Phase 2.5.4 (fill prediction), which closed as vacuous
because `factor_slack` had **no hot-path consumer**. Here the
leverage is real:

1. The MC64 matching is **already computed** by
   `Mc64Symmetric` / `Auto` scaling. The compression step re-uses
   the existing `perm` array — no new Hungarian call, no MC64
   symbolic cost on top of what scaling already pays. (The
   corresponding MUMPS cost breakdown dispatches matching once and
   feeds both `ICNTL(12)` and `ICNTL(8)` scaling from the same
   call.)
2. Compression reduces the vertex count going into AMD/METIS by
   the measured `compRat` (0.66–0.75 on the worst matrices), which
   directly shrinks ordering time and typically reduces fill in the
   numeric factors.
3. The expansion step is O(n) and has no algorithmic subtlety.
4. Our sparse p90 vs MUMPS is 1.59 (target 2.0, PASS). MUMPS runs
   with `ICNTL(12) = 2` as the SYM=2 default, so part of the 1.59×
   gap is precisely that we don't compress and MUMPS does.
   Matching this feature is legitimate parity work.

## Interaction with existing feral plumbing

- `src/scaling/mc64.rs:matching_perm` (added in this phase) exposes
  the matching as `(perm, n_matched)` without the scaling
  post-processing.
- `src/symbolic/mod.rs::symbolic_factorize_with_method` is the
  entry point. The compression preprocess sits between the
  `CscPattern` extraction and the `feral_amd::amd_order(&pat)` call
  at line 231.
- `feral_amd` (and the METIS/SCOTCH/KaHIP equivalents) produce a
  permutation of the *pattern they receive*. We feed them the
  compressed pattern, get back a super-permutation of length
  `NCMP`, and expand before the rest of the pipeline
  (postorder / etree / column counts / supernode detection) runs.
  **Everything downstream of the ordering call operates on the
  expanded permutation; no data structure needs to be aware of the
  compression.**
- The MC64 matching can be structurally partial (unmatched
  columns). Unmatched columns become singletons by default (same
  as in MUMPS). Scaling already handles the partial case via
  `ScalingInfo::PartialSingular`; we re-use that logic.

## Opt-in vs default

Decision for this phase: **opt-in**. Add a new field to
`SupernodeParams` (or a dedicated `OrderingPreprocess` enum) that
defaults to `None`; callers get the behavior only by asking for it.
Once the bench shows no regression across the 169 585-matrix corpus
(target: dense and sparse p90 unchanged or better; no residual
regression beyond the existing `PartialSingular` tolerance), a
follow-up session can flip the default.

This mirrors how MC64 scaling landed (Phase 2.2.1 opt-in first,
flipped to default in Phase 2.2.3 after corpus evidence).

## Interaction with 2×2 pivoting in the numeric phase

MUMPS uses the pair structure as a *hint* to the numeric phase:
after compression+ordering, the paired columns sit adjacent in
pivot order, so `DMUMPS_FAC_ASM` tries the tentative 2×2 pivot
first during Bunch-Kaufman. If it passes the pivot test, the pair
is promoted to a real 2×2 block without further search.

Feral's BK kernel (`src/dense/factor.rs`) already discovers 2×2
pivots at elimination time via the BK regular-pivot test — it does
**not** need the ordering-stage hint to find them. Compression
still benefits us for the ordering / fill reason, but the "2×2
hint" channel is absent from the port scope.

## Implementation scope (handoff to plan)

The plan file will break this into:

1. **Expose the matching.** `src/scaling/mc64.rs:matching_perm`
   (done in this note as the survey prerequisite) and the
   `pub fn mc64_matching` wrapper.
2. **Compression module.** `src/symbolic/ldlt_compress.rs`:
   - `build_icmp(perm: &[usize]) -> (icmp, pairs, singletons)`
     where `icmp[i]` is the super-variable id of original `i`.
   - `compress_pattern(pat, icmp) -> CscPattern` contracting edges.
   - `expand_permutation(super_perm, pairs, singletons) -> Vec<usize>`
     emitting paired originals adjacent.
3. **Opt-in wiring.** New field on `SupernodeParams` or a new enum
   passed through `symbolic_factorize_with_method`. When on, run
   MC64 matching if not already available, build `icmp`, compress,
   order, expand, feed expanded perm to the rest of the pipeline.
4. **Tests.** Unit tests on toy matrices (a 4×4 with two known
   2-cycles, a 6×6 with a 3-cycle) verifying:
   - `icmp` is correct
   - `compress_pattern` preserves symmetric structure
   - `expand_permutation ∘ identity_super_perm` maps each super to
     the union of its originals
   - a small full round-trip: `symbolic_factorize(..., compress=true)`
     produces a valid permutation of `{0..n-1}` and the resulting
     factorization passes the inertia check
5. **Bench.** `diag_compression_bench` on the worst-10 tail
   matrices plus a 10 000-sample of the corpus histogram buckets
   comparing `factor_us` and `factor_nnz_estimate` with and without
   compression. Decide default based on the delta.

## References

- Duff & Pralet 2005 (cited above; see `dev/references.bib`).
- MUMPS 5.8.2 source files cited above.
- feral session 2026-04-18-08 (41-matrix ordering bakeoff).
- feral `src/scaling/mc64.rs` + `src/scaling/hungarian.rs`.
- feral `src/bin/diag_mc64_cycles.rs` (this note's survey tool).
