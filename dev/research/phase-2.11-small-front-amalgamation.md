# Phase 2.11 — Small-front amalgamation on the tiny-IPM tail

**Status:** Pre-implementation research note for Phase 2.11.
**Date:** 2026-04-25
**Related:**
- `dev/research/reference-solver-comparison.md` (defines the gap)
- `dev/research/phase-2.9-small-leaf-subtree.md` (existing
  numeric-side mitigation)
- `dev/plans/phase-2.10-supernode-profiler.md` (the measurement
  Phase 2.11 starts from)
- `dev/journal/2026-04-25-03.org` (Phase 2.10 results)
- `dev/research/phase-2.2.3-plateau.md` §"larger refactor"
  comment in `src/symbolic/supernode.rs:222-230` referencing the
  SSIDS column-renumbering approach.

---

## 1. The gap, in numbers

Reference-solver comparison (`reference-solver-comparison.md` §1):

| matrix         |   n  | MUMPS μs | FERAL μs | FERAL/MUMPS |
|----------------|-----:|---------:|---------:|-------------|
| ACOPR30_0067   |  564 |     144  |    1214  |     8.43×   |
| CRESC100_0000  |  806 |     200  |    1233  |     6.17×   |
| BATCH_0000     |  121 |      83  |      70  |     0.84×   |

We beat MUMPS on BATCH (n=121); we trail by 8.5× on ACOPR30
(n=564). The gap is real on tiny IPM KKTs. SSIDS is in the same
boat as us (4-8× slower than MUMPS in the same regime), so this is
specifically about MUMPS's per-invocation tuning, not a vendor-BLAS
moat.

---

## 2. Phase 2.10 evidence: where the time goes

`cargo run --release --bin profile_supernode_distribution`, median
of 5 runs:

| matrix         | snodes |  ≤8  |  9-16 | 17-32 | 33-64 | loop_us | total_us |
|----------------|-------:|-----:|------:|------:|------:|--------:|---------:|
| ACOPR30_0067   |    341 | 332  |    8 |     1 |     0 |    1060 |     1222 |
| CRESC100_0000  |    600 | 599  |    0 |     1 |     0 |    1003 |     1304 |
| LAKES_0000     |    216 | 214  |    1 |     0 |     1 |     340 |      432 |
| NELSON_0000    |    256 | 256  |    0 |     0 |     0 |      98 |      191 |
| SWOPF_0000     |     42 |  38  |    2 |     2 |     0 |     ~83 |      big |

Two observations:

1. **The loop is dominated by tiny supernodes**: 97-100% of fronts
   have ≤8 columns on every tail matrix.
2. **The loop is not the only cost**: prologue + epilogue is
   13-49% of `total_us`. NELSON's 191μs total breaks down as
   98μs loop + 93μs prologue/epilogue. SWOPF's bimodal profile
   (4 medium fronts carry ~95% of loop time) is the exception.

For a meaningful gap close on ACOPR30/CRESC100 we need to either
(a) shrink the per-supernode loop cost, or (b) shrink the number
of supernodes — i.e., amalgamate harder.

---

## 3. Diagnostic: why feral's amalgamation under-merges

`cargo run --release --bin diag_amalgamation` (default `nemin=32`):

| matrix         | n_snodes | leaves | multi-child | max children | est. blocked sibling-merges |
|----------------|---------:|-------:|------------:|-------------:|----------------------------:|
| ACOPR30_0067   |     341  |    235 |          14 |           69 |                         234 |
| CRESC100_0000  |     600  |    411 |         189 |          223 |                         410 |
| LAKES_0000     |     216  |    181 |          34 |           56 |                         180 |
| NELSON_0000    |     256  |    129 |           1 |          129 |                         128 |
| SWOPF_0000     |      42  |     35 |           6 |           14 |                          34 |

Read the rightmost column as: "estimated number of sibling-merges
the SSIDS rule would accept but the adjacency check at
`supernode.rs:204-236` blocks." On every tail matrix this number is
≈ leaf count.

**Mechanism.** `find_supernodes` checks for adjacency before
merging:

```rust
let s_first = snode_first_col[root_s];
let s_ncol  = snode_ncols[root_s];
let p_first = snode_first_col[root_p];
if s_first + s_ncol != p_first {
    continue;  // not adjacent → cannot merge
}
```

In a postordered etree, every parent's columns come *after* all
its descendants', so when a parent has N children, only one of the
N can have its last column equal to `parent.first_col − 1`. The
other N−1 are blocked from merging into the parent even when they
satisfy the SSIDS size rule (`child_ncol < nemin AND parent_ncol
< nemin`). This is documented in the source as a known limitation;
the comment cites SSIDS's `core_analyse.f90:644-685` as the canonical
fix.

**NELSON_0000 is the textbook bushy IPM KKT:** 256 supernodes, of
which 129 are leaves and 127 are internal-chain singletons. ONE
internal node has 129 children (`max children = 129`). The other
126 internal nodes are chain-of-1 singletons up the spine. The
adjacency check rejects 128 candidate sibling-merges into the
bordering parent.

---

## 4. Phase 2.9 (small-leaf batching) is firing but producing
1-leaf groups

The diagnostic also confirms small-leaf grouping is enabled and
runs but produces near-degenerate groups:

| matrix         | n_leaves | n_groups | avg leaves/group |
|----------------|---------:|---------:|------------------|
| ACOPR30_0067   |     235  |    105   | 2.24             |
| CRESC100_0000  |     411  |    189   | 2.17             |
| LAKES_0000     |     181  |     34   | 5.32             |
| NELSON_0000    |     129  |    127   | 1.02             |
| SWOPF_0000     |      35  |      6   | 5.83             |

NELSON gets 127 groups for 129 leaves — almost no batching at all.
Why? `find_small_leaf_groups` requires *consecutive* qualifying
supernodes in postorder; **a non-leaf or over-size snode breaks the
group**. On a bushy tree the postorder interleaves leaf siblings
with their (often singleton chain) parents, breaking groups
between every leaf.

So Phase 2.9 was a real win on the cleaner small-leaf-cluster
shape (LAKES, SWOPF) but barely helps the heavily-bushy IPM KKTs
that drive the MUMPS gap.

---

## 5. What MUMPS does (per the 2026-04-25 four-agent investigation)

From `dev/journal/2026-04-25-{01,02}.org` and the `mumps-expert`
agent's read of MUMPS 5.8.2:

1. **Aggressive amalgamation (`KEEP(197)=1`, `NEMIN=8` default).**
   MUMPS merges siblings into the parent unconditionally for
   small fronts; the cost-vs-fill heuristic is biased toward
   fewer fronts. The net effect on a bushy IPM KKT is to collapse
   each parent's many tiny-leaf children into the parent itself,
   eliminating dozens of frontal-matrix allocations.
2. **AMF ordering (not METIS) for `n ≤ 10000`.** Smaller, denser
   front trees per ordering. Out of scope for Phase 2.11.
3. **Single-shot kernel for `NASS < 24`.** A specialized
   small-front factorization that bypasses the standard
   blocked-LDLᵀ kernel and avoids per-front allocator round
   trips. Approximately what feral's small-leaf batching aims at,
   but firing on more of the tree (any small front, not just true
   leaves at the bottom).
4. **Driver-level tuning.** Workspace is preallocated by the
   analysis phase and re-used across the factorization; no
   prologue allocator activity.

(1) is the symbolic-time fix and the largest single lever. (3) is
a numeric-time fix that complements (1).

---

## 6. Implementation options

### Option A — SSIDS-style column renumbering during amalgamation

The "correct" generalization of feral's current adjacency-gated
merge: when a parent has multiple small children that all satisfy
the merge rule, emit a column permutation that places those
children's columns *contiguous* with the parent's first column,
then merge them all into one supernode whose `first_col..ncol`
range is contiguous by construction.

**Effort.** Medium-large refactor. Touches:

- `src/symbolic/supernode.rs::find_supernodes`: extend the merge
  loop to renumber instead of skipping.
- `src/symbolic/mod.rs::symbolic_factorize_with_method`: the new
  permutation must be composed with the existing AMD+postorder
  composition before re-permuting the matrix and rebuilding the
  etree/column-counts. Either re-run those steps after the merge
  (cheap — O(n)) or carefully prove the etree is invariant under
  the renumbering of within-supernode columns (it should be since
  merged columns share row structure exactly, but proving and
  testing this is non-trivial).

**Reward.** Targets the dominant cost. Likely closes much of the
ACOPR30/CRESC100 gap.

**Risk.** Breaks invariants downstream (frontal assembly, solve
gather/scatter, `small_leaf_groups`). Each downstream consumer of
`Supernode::first_col` and the column permutation must be audited.

### Option B — Keep `find_supernodes` honest, fix `find_small_leaf_groups` to merge across non-leaf neighbours

Lift the "non-leaf breaks the group" rule in `small_leaf.rs`. The
batched numeric path can in principle handle non-adjacent leaves
(via gap tracking) and tiny chain-internal singletons (treat them
as leaves with empty contribution). This is a numeric-side fix
that reuses the symbolic structure as-is.

**Effort.** Smaller. Touches `find_small_leaf_groups` and the
batched numeric driver (`numeric/factorize.rs` small_leaf branch).

**Reward.** Closes some of the gap by amortizing the per-leaf
dispatch cost across more leaves. Does *not* reduce the actual
front count — feral still allocates 256 frontal matrices for
NELSON, just dispatches them faster.

**Risk.** Lower than Option A. The hard part is the gap-tracking /
chain-internal-as-leaf logic in the numeric path, which is local
to the small_leaf branch.

### Option C — Aggressive sibling pre-amalgamation as a pre-pass

Before `find_supernodes`'s main merge loop, run a separate pass
that identifies parents with ≥2 small children and absorbs them
into the parent by emitting a tight column permutation. This is
Option A scoped to the multi-child case only, leaving the
adjacent-child path unchanged.

**Effort.** Medium. Smaller than Option A but with the same
downstream-permutation issue.

**Reward.** Same shape as Option A on the bushy-tree workloads.

**Risk.** Same as Option A.

### Option D — MUMPS-style numeric fast-path for `nrow < 24`

Add a fast small-front kernel in the numeric phase that bypasses
the allocator (uses a thread-local arena), inlines the BK pivot
loop, and skips the contribution-block bookkeeping when the
parent is also small (assemble directly into the parent's
arena slot).

**Effort.** Medium. Numeric-side, doesn't touch symbolic.

**Reward.** Reduces per-supernode loop cost. Does *not* reduce
prologue/epilogue or supernode count.

**Risk.** Numeric correctness must be proven against the standard
kernel via parity tests. Allocator changes have a history of
regressions.

---

## 7. Recommendation

Start with **Option B** (tighten `find_small_leaf_groups`),
because:

1. It is the smallest, lowest-risk intervention.
2. The diagnostic shows small_leaf groups produce 1.0–2.2 leaves
   per group on the worst-case tail; lifting the postorder
   adjacency requirement should multiply that 2-5×.
3. If Option B produces a meaningful (≥30%) close on
   ACOPR30/CRESC100, that's a solid Phase 2.11 deliverable.
4. If Option B's payoff is small, the result tells us the
   per-front *allocation/dispatch* cost is not the binding
   constraint and we should pivot to Option A in Phase 2.12.

Defer Options A and C to a follow-up phase. They are larger and
the diagnostic does not yet prove the front-count reduction is
worth the refactor over the (smaller, surer) Option B win.

Defer Option D until after Option B; both are numeric-side and
should not be combined in one PR.

---

## 8. Success / rejection criteria for Phase 2.11 (Option B)

**Success** (any of the following, on the median of 5 runs):

- ACOPR30_0067: `total_us` ≤ 800 (≥34% reduction from 1222).
- CRESC100_0000: `total_us` ≤ 850 (≥35% reduction from 1304).
- NELSON_0000 small_leaf coverage: ≥10× more leaves per group
  (currently 1.02, target ≥10.0 average).
- No parity regressions on the existing test suite.

**Rejection**:

- Any parity failure (bit-equal outputs required for the parity
  tests).
- Inertia mismatch on any matrix with definitive verdict.
- Total time *increase* on any tail matrix (regression).

If rejected because Option B's payoff is too small (<10% on
ACOPR30/CRESC100), record the result honestly and escalate to
Option A in Phase 2.12.

---

## 9. Out of scope for Phase 2.11

- AMF ordering port (would be Phase 2.12).
- Allocator changes (already reverted in Phase 2.9.2 — see
  `dev/decisions.md`).
- Restructuring the symbolic pipeline to avoid postorder-induced
  bushiness (would require revisiting AMD output handling).
- Generalizing the merge rule beyond SSIDS / `nemin`.
