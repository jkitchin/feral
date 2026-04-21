# Phase 2.6.5 — LDLᵀ-aware ordering (compressed-graph) plan

**Status:** pre-implementation.
**Date:** 2026-04-21.
**Research:** `dev/research/phase-2.6.5-ldlt-aware-ordering.md`.

## Goal

Port the Duff-Pralet symmetric matching + quotient-graph compression
that MUMPS runs under `ICNTL(12) = 2`, as an **opt-in** preprocessing
step for feral's `symbolic_factorize_with_method`. Expected benefit:
~33% vertex-count reduction on the ordering graph for the ~47% of
the KKT corpus with nontrivial MC64 cycles, directly reducing
ordering time and typically reducing fill in the numeric phase.

## Non-goals

- Not KKT-specific — the algorithm operates on the MC64 matching,
  not on bordered-KKT structure. Saddle-point matrices benefit
  incidentally.
- Not flipping the default in this phase — land opt-in, collect
  corpus evidence, flip in a follow-up.
- Not porting the "tentative 2×2 pivot hint" channel to the BK
  kernel — feral's BK already finds 2×2s at elimination time.

## Step list

### Step 1 — Expose the MC64 matching as a public helper

Already done as part of the survey:

- `src/scaling/mc64.rs:matching_perm` — runs the cost-graph build +
  `hungarian_match` and returns `(perm, n_matched)` without the
  symmetric-average post-processing that produces the scaling
  vector.
- `src/scaling/mod.rs:mc64_matching` — public wrapper.

The scaling code already runs this inside `compute_symmetric`, so
`Mc64Symmetric` / `Auto` callers will (in a later integration step)
be able to cache the matching and hand it to the compression path
without a second Hungarian call.

### Step 2 — Compression module

New file `src/symbolic/ldlt_compress.rs`:

```rust
/// ICMP[i] = super-variable id of original variable i.
pub struct SuperMap {
    pub icmp: Vec<usize>,
    pub pairs: Vec<(usize, usize)>,
    pub singletons: Vec<usize>,
}

pub fn build_supermap(perm: &[usize]) -> SuperMap;
pub fn compress_pattern(pat: &CscPattern, map: &SuperMap) -> CscPattern;
pub fn expand_permutation(super_perm: &[usize], map: &SuperMap) -> Vec<usize>;
```

Invariants:

- `map.icmp.len() == n` (original dimension).
- `map.pairs.len() * 2 + map.singletons.len() == n`.
- Super-variable ids are contiguous `[0, ncmp)` where `ncmp =
  pairs.len() + singletons.len()`.
- `compress_pattern(..).n == ncmp`.
- `compress_pattern` preserves symmetry: the output pattern is
  symmetric in the CscPattern sense (full-pattern, not
  lower-triangle). It drops self-loops and deduplicates.
- `expand_permutation(super_perm, map)` returns a length-`n`
  permutation of `0..n` where each super `s` in `super_perm`
  contributes its originals consecutively, pair originals in
  the order `(p.0, p.1)`.

### Step 3 — Cycle extraction

The MC64 `perm` encodes a column-to-row matching. On a full
matching of a symmetric matrix, the permutation's cycle structure
consists of:

- **1-cycles** (`perm[j] = j`): on-diagonal match → singleton.
- **2-cycles** (`perm[j] ≠ j`, `perm[perm[j]] = j`): a pair.
- **k-cycles (k ≥ 3)**: decomposed into `⌊k/2⌋` pairs +
  (1 singleton if k is odd). For a cycle `j0 → j1 → j2 → ... → jk-1 → j0`,
  emit pairs `(j0, j1)`, `(j2, j3)`, ... and leftover if odd.
  This matches `DMUMPS_SYM_MWM`'s decomposition.
- **Unmatched** (`perm[j] == usize::MAX`): treat as singleton.

`build_supermap` walks the permutation graph, marks visited nodes,
and emits the `pairs` / `singletons` arrays in the order variables
are discovered. Deterministic.

### Step 4 — Pattern compression

Algorithm (mirrors `DMUMPS_LDLT_COMPRESS` but in Rust idiom):

```
ncmp = pairs.len() + singletons.len()
out_entries: Vec<(usize, usize)> = []
for each edge (i, j) in the full symmetric pattern:
    si = icmp[i]; sj = icmp[j]
    if si == sj: continue   // self-loop, drop
    out_entries.push((min(si,sj), max(si,sj)))  // store half
sort + dedup -> out_entries'
build CscPattern from out_entries', symmetrised
```

Implementation detail: to avoid O(nnz) sort for large matrices, use
a per-column hash-set accumulator keyed by super-variable id. The
accumulator is a `Vec<bool>` of length `ncmp` reset per-column
(same marker pattern as AMD's mark-array).

### Step 5 — Permutation expansion

```
out = Vec::with_capacity(n)
for &s in super_perm:
    if s < pairs.len():
        let (p0, p1) = map.pairs[s]; out.push(p0); out.push(p1);
    else:
        out.push(map.singletons[s - pairs.len()]);
```

Result is a permutation of `0..n`; adjacency of pair originals is
preserved.

### Step 6 — Wiring into the symbolic pipeline

New field on the ordering path. Two options:

- **(A)** Add `preprocess: OrderingPreprocess` to
  `SupernodeParams`. Default `None`. Enum has `LdltCompress`.
- **(B)** Add a new method name, e.g. `AmdCompressed`,
  `MetisCompressed`, to `OrderingMethod`. Explodes the enum.

Go with **(A)** — one orthogonal flag × any ordering method.

Wiring in `symbolic_factorize_with_method`:

```rust
let resolved_method = choose_adaptive(&pat, method);
let (final_perm, final_perm_inv) = if params.preprocess == LdltCompress {
    let (matching, _) = mc64_matching(matrix)?;
    let map = build_supermap(&matching);
    let cpat = compress_pattern(&pat, &map);
    let super_perm = run_ordering(resolved_method, &cpat)?;
    let perm = expand_permutation(&super_perm, &map);
    let perm_inv = invert(&perm);
    (perm, perm_inv)
} else {
    let perm = run_ordering(resolved_method, &pat)?;
    let perm_inv = invert(&perm);
    (perm, perm_inv)
};
```

Where `run_ordering` is a small helper that dispatches on
`resolved_method`.

### Step 7 — Tests

`tests/ldlt_compress.rs`:

1. **4×4 known-pair**: construct a matrix with zero diagonals at
   positions 2 and 3 and off-diagonals connecting 0↔2 and 1↔3.
   MC64 must match (2,0) and (3,1). `build_supermap` must produce
   `pairs = [(0,2), (1,3)]`, `icmp = [0,1,0,1]`, `ncmp = 2`.
2. **6×6 with a 3-cycle**: constructed so the matching is
   `0→1→2→0` plus 3,4,5 on-diagonal. Must produce `pairs = [(0,1)]`,
   `singletons = [2, 3, 4, 5]`.
3. **Pattern round-trip**: `expand_permutation(build_supermap.super_perm_iota)`
   is the identity permutation of `0..n`, for the construction
   where `super_perm_iota[s] = s`.
4. **End-to-end on a tiny KKT**: pick a tiny corpus matrix (e.g.
   HS13 KKT at some iteration, `n` ≈ 10) and factor with and
   without compression; compare `factor_nnz_estimate` and verify
   inertia matches bit-exact.

### Step 8 — Corpus bench

New bin `src/bin/diag_compression_bench.rs`: for a stratified
sample of the corpus by `compRat` bucket, run symbolic+numeric
with and without compression, record:

- `symbolic_us` delta (should be small but non-zero — compression
  adds a Hungarian call + graph contraction).
- `factor_nnz_estimate` delta (expect reduction on matrices with
  `compRat < 0.9`).
- `factor_us` delta (expect improvement where NNZ reduction is
  large).
- inertia parity (must be 0 mismatches).

Decision criterion for flipping default in a follow-up session:
geomean factor_us improvement ≥ 5% on the `compRat ≤ 0.7` bucket
AND 0 inertia regressions AND no residual regressions worse than
the existing `PartialSingular` tolerance.

### Step 9 — Gate on bench

If bench shows clear improvement: commit as opt-in; update
`CHANGELOG.md`; queue a follow-up "flip default" task. If bench
shows improvement only on narrow bucket: still commit as opt-in
with the empirical caveat in the field's doc comment. If bench
regresses: do not commit; update the research note with the
null result and close as "documented-could-regress".

## Risk register

- **R1: our AMD already supervariable-detects.** `feral-amd`'s
  mark-array AMD already merges variables with identical closed
  neighborhoods internally (different mechanism from MC64 2-cycles
  though). Possible overlap → less than expected benefit.
  Mitigation: the corpus bench is the arbiter; if the improvement
  on `compRat ≤ 0.7` is less than ~5% geomean, document and close.
- **R2: MC64 matching order-nondeterminism.** The Hungarian
  implementation is deterministic on ties, so the matching is
  stable — no new determinism surface area.
- **R3: partial-matching case.** Unmatched columns get singleton
  treatment. On the ~1% of corpus with partial matching, compression
  does nothing for unmatched cols but still compresses matched
  cycles. No extra failure modes expected.
- **R4: scope creep into the numeric phase.** Do *not* add any
  2×2 pivot hints. Pure ordering-stage preprocessing.

## Out-of-scope (for this phase)

- Integrating the matching computation between scaling and
  compression (to avoid two Hungarian calls when both `Mc64Symmetric`
  scaling and `LdltCompress` preprocess are requested). A future
  micro-opt.
- Flipping the default. Gate on bench data in a follow-up session.
- Porting `ICNTL(12) = 3` (constrained ordering for AMF). No AMF
  in feral, and the constrained variant is not widely used in the
  MUMPS default stack anyway.
