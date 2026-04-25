# Phase 2.12 — SSIDS-style column renumbering for amalgamation

**Status:** Pre-implementation research note for Phase 2.12.
**Date:** 2026-04-25
**Related:**
- `dev/research/phase-2.11-small-front-amalgamation.md` (Option A
  rationale, deferred from Phase 2.11)
- `dev/tried-and-rejected.md` § 2026-04-25 Phase 2.11 (rejected the
  cheap fix; this note picks up where that left off)
- `src/symbolic/supernode.rs:204-236` (the adjacency check the
  renumbering removes)
- `src/symbolic/mod.rs:336-488` (the symbolic pipeline this note
  amends)
- SSIDS `src/core_analyse.f90:644-685` (canonical reference)

---

## 1. Where Phase 2.11 left us

Phase 2.10's profiler showed 90-100% of supernodes on the tiny-IPM
tail are ≤8 columns wide. Phase 2.11 attempted the cheapest mitigation
(flip `SmallLeafBatch` default Off→On) and rejected it on a 5-run
repeat: the effect is in measurement noise. The diagnostic
`diag_amalgamation` showed *why*: feral's amalgamation rejects 128-410
sibling-merges per tail matrix because of an adjacency constraint that
SSIDS sidesteps via column renumbering.

Concretely (from `diag_amalgamation` output):

| matrix         | snodes | leaves | multi-child | max children | est. blocked merges |
|----------------|-------:|-------:|------------:|-------------:|--------------------:|
| ACOPR30_0067   |    341 |    235 |          14 |           69 |                 234 |
| CRESC100_0000  |    600 |    411 |         189 |          223 |                 410 |
| LAKES_0000     |    216 |    181 |          34 |           56 |                 180 |
| NELSON_0000    |    256 |    129 |           1 |          129 |                 128 |
| SWOPF_0000     |     42 |     35 |           6 |           14 |                  34 |

Every "blocked merge" is a child supernode that would satisfy SSIDS's
size rule (`child_ncol < nemin AND parent_ncol < nemin`) but cannot
merge because the postorder column numbering puts it non-adjacent to
the parent (see `src/symbolic/supernode.rs:204-236`).

## 2. Why the adjacency check exists

In a postordered etree, every parent's columns come *after* all its
descendants'. When a parent has N children, only ONE child's last
column can equal `parent.first_col − 1`. Merging any other child into
the parent would create a supernode whose `first_col..first_col+ncol`
range jumps over columns belonging to the other children — a
correctness bug in every downstream consumer (frontal assembly, solve
gather/scatter, L storage).

The arrow matrix is the archetype: variables 0..n-2 all parented by
n-1. Only child n-2 is adjacent to parent n-1; children 0..n-3 are
blocked.

## 3. SSIDS's fix: renumber columns so merges are contiguous

SSIDS (`core_analyse.f90:644-685`) emits a column permutation π such
that the merged children's columns are placed immediately before the
parent's columns. The permutation is consistent with the etree (it
permutes within subtrees only), so the etree structure and column
counts are invariant under it (just relabeled).

The cleanest formulation, in our pipeline's terms: replace the
single postorder pass with a **merge-biased postorder**. When
descending into a parent's children, emit:

1. The non-merging children's subtrees first (in any postorder),
2. The merging children's subtrees next (any order among themselves),
3. The parent's columns last.

Because the merging children's subtrees are emitted last, their
column blocks land immediately before the parent's column block, and
the existing adjacency check at `supernode.rs:204-236` succeeds
naturally for every desired merge.

## 4. Algorithm sketch (the implementation strategy)

The pipeline stays the same up through "first postorder + first
etree + first col_counts", but inserts a re-postorder step before
`find_supernodes`:

```
Step 1: AMD perm                   (existing)
Step 2: post1 = postorder(etree₀)  (existing — unbiased postorder)
Step 3: perm₁ = AMD ∘ post1        (existing)
Step 4: re-permute → etree₁, col_counts₁  (existing)
Step 5: predict_merges(etree₁, col_counts₁, nemin)
Step 6: post2 = biased_postorder(etree₁, predicted_merges)
Step 7: perm₂ = perm₁ ∘ post2      (NEW)
Step 8: re-permute → etree₂, col_counts₂
Step 9: find_supernodes(etree₂, col_counts₂)  (existing — adjacency now natural)
Step 10: small_leaf_groups, etc.   (existing)
```

Step 5 is the prediction. We don't have supernodes yet — we have
columns. The size-rule predicate operates on supernode sizes. We need
to identify *fundamental* supernodes first (the merge-free baseline
from `find_supernodes` Step 1), then ask: under the size rule, which
fundamental child supernodes would merge with their parent?

Two ways to do this:

**(A) Two passes through `find_supernodes`.** First call ignores
adjacency entirely (or takes a `skip_adjacency_check: bool` flag),
returns the merge graph as data. Use that to drive the biased
postorder. Second call (after Step 8) is the real one with the
adjacency check enforced — but by construction, every desired merge
is now adjacent.

**(B) A separate `predict_merges` function** that re-implements the
fundamental-supernode detection and SSIDS size rule, but without
producing `Supernode` records. Cheaper but duplicates logic.

I prefer (A) for code-reuse and to make the prediction logic
exactly equal to the realized logic. The cost of running
`find_supernodes` twice is O(n) per call — negligible compared to
the rest of symbolic factorization.

## 5. Risks

### 5.1 Etree invariance under within-subtree relabeling

**Claim:** If π reorders columns within each parent's descendant
range (i.e., π is consistent with the etree as a "subtree
permutation"), then `etree(π(A))` is just the relabeling of
`etree(A)` under π. No structural change.

This is the standard CHOLMOD/SSIDS observation. Postorder itself is
such a permutation. A merge-biased postorder is also such a
permutation — biasing only changes the order children's subtrees
are visited, not which columns belong to which subtree.

**Test:** the existing parity test suite. If etree-structural
invariance fails, factor outputs will diverge bit-for-bit from the
reference solver.

### 5.2 col_counts invariance

Column counts are a function of the etree + the sparsity pattern.
Both are invariant under subtree permutation (the pattern is
relabeled but the underlying set of nonzero positions on the
permuted matrix is determined). col_counts₂ should equal col_counts₁
modulo the relabeling.

**Test:** assert that under nemin=1 (no merges) the biased postorder
returns col_counts identical to the unbiased postorder modulo
relabeling.

### 5.3 small_leaf_groups invariance

Small-leaf grouping operates on `find_supernodes`'s output. Phase
2.9's groups depend on postorder adjacency of *qualifying leaf
supernodes*. The biased postorder may change which leaves are
adjacent in postorder.

**Expected effect:** *better* leaf grouping. Currently NELSON has
1.02 leaves/group. After renumbering the bushy-parent's 129 leaf
children to be contiguous, those 129 leaves become adjacent in
postorder and group naturally. The leaves/group ratio should rise to
the `ncol_max=8` floor.

**Risk:** if some Phase 2.9 invariant relies on an interaction
between postorder and the original AMD ordering that the biased
postorder breaks, small_leaf coverage could drop or assemble could
fail. Will be caught by the existing parity tests.

### 5.4 Permutation composition correctness

Composing perm₁ ∘ post2 must produce a valid permutation on `0..n`.
The implementation must be: `perm₂[k] = perm₁[post2[k]]`. Easy to
get wrong (in/out direction inversion). Test with assert_eq on the
sorted bijection.

### 5.5 Performance regression on cleanly-postordered trees

When the unbiased postorder already respects desired merges (chain
trees, binary trees with one-child merges), the biased postorder is
identical and the second column-counts pass is wasted O(nnz · α(n))
work.

**Mitigation:** detect the no-bias-needed case and skip steps 5-8.
Predicate: if every parent has ≤1 desired-merge child, the unbiased
postorder is already biased-correct. This catches chain trees and
tridiagonals (where unbiased = biased trivially).

## 6. Scope decisions

**In scope for Phase 2.12:**

- `predict_merges` function (re-uses find_supernodes Step 1
  fundamental-supernode logic + SSIDS size rule, ignoring adjacency)
- `biased_postorder` function (modified DFS that visits
  desired-merge children last)
- Pipeline wiring in `symbolic_factorize_with_method`
- `SupernodeParams::amalgamation_strategy` enum gate (default
  `Adjacency` for parity; `Renumber` opts in)
- Tests: arrow matrix (1 supernode after merge), bushy NELSON-like,
  parity vs Adjacency on the existing test suite
- Benchmark via `diag_amalgamation` and `diag_small_leaf_gate` on
  the tail

**Out of scope:**

- Generalizing the merge rule beyond SSIDS / `nemin` (the rule
  itself is unchanged — only adjacency is)
- AMF ordering port
- Numeric-side changes (the renumbering is a symbolic-time fix
  only; the numeric path consumes the supernodes as before)
- Allocator changes

## 7. Success / rejection criteria

**Success** (all of):

1. Parity: factor outputs and inertia match the existing
   `Adjacency` strategy on every parity test (use a dual-run or
   bit-equal hash).
2. Front-count reduction: `diag_amalgamation` reports `n_snodes`
   reduction ≥ 50% on at least one of {ACOPR30, CRESC100, NELSON}.
3. End-to-end speedup: `total_us` (median of 5 runs) on
   `diag_small_leaf_gate` equivalent harness reduced by ≥ 20% on at
   least one of {ACOPR30, CRESC100} relative to the current main.
4. No regression on small-and-medium matrices (50-matrix corpus
   bench `cargo run --release --bin bench` median total_us within
   ±5%).

**Rejection** (any of):

- Parity failure (any test).
- Inertia mismatch on any matrix with definitive verdict.
- Total-time regression on any tail matrix.
- Front-count reduction < 30% on every tail matrix.

## 8. Risks specifically called out by the four-agent investigation

The 2026-04-25-{01,02} session-1 agent investigation flagged that
SSIDS itself is 4-8× slower than MUMPS on this regime, so even a
clean port of SSIDS's renumbering is *not* expected to close the
full gap. The honest performance ceiling for Phase 2.12 alone is
"meet SSIDS" not "meet MUMPS". A fast small-front numeric kernel
(MUMPS NASS<24, deferred to a later phase) is the second piece of
the puzzle.
