# Phase 2.9 — SmallLeafSubtree batching

> **Retrospective (2026-04-24).** Phase 2.9 and its follow-up
> Phase 2.9.2 both produced null / rejected results. The framing
> below ("10× vs MUMPS") is preserved for historical accuracy but
> is superseded by `dev/research/reference-solver-comparison.md`:
> FERAL is at-or-ahead of SSIDS across the corpus and the remaining
> MUMPS-gap on tiny IPM matrices (ACOPR30, CRESC100, HAIFAM_0082)
> is a narrow acknowledged deficit — not the universal gap this
> note originally assumed. A future assault on the MUMPS-gap should
> start from the instrumentation checklist in that newer note, not
> from this one.

## Problem

On long-tail IPM matrices (ACOPR30, CRESC100, HAIFAM) feral's factor is
~10× slower than MUMPS *with identical L structure*. The bottleneck is
not fill — AMD matches or beats METIS on these matrices (see
`diag_fill_tail` output, 2026-04-24), and the factor nnz sits below 1%
of the dense triangle. The bottleneck is per-front overhead.

## Evidence

Numbers are medians over 20 repetitions, release build, Darwin aarch64,
collected 2026-04-24 via `cargo run --release --bin diag_supernode_cost`.

Long-tail (ACOPF-class KKT):

| matrix           |   n | nsup | med-size | max-size | num_us |  ns/sup | ns/nnz |
|------------------|-----|------|----------|----------|--------|---------|--------|
| ACOPR30_0067     | 564 |  341 |        2 |       32 |   1170 |    3431 |    297 |
| ACOPR30_0185     | 564 |  341 |        2 |       32 |   1194 |    3502 |    303 |
| HAIFAM_0082      | 249 |  ~   |        ~ |        ~ |  ~3000 |   ~3500 |    ~   |

Bulk (feral already wins):

| matrix           |   n | nsup | med-size | max-size | num_us |  ns/sup | ns/nnz |
|------------------|-----|------|----------|----------|--------|---------|--------|
| HS118            |  22 |    1 |       22 |       22 |      4 |    4000 |     90 |
| HS92             |  12 |    1 |       12 |       12 |      3 |    3000 |     80 |

`ns/sup` is nearly identical (~3500 ns) across tail and bulk. Bulk wins
because it hits the dense fast-path or has 1–3 supernodes total; tail
loses because it has 340 supernodes and pays 3500 ns × 340 = 1.2 ms of
*fixed per-front overhead* on top of ~20k FLOPs of real arithmetic.

An `nemin` sweep on ACOPR30_0067 confirms amalgamation alone cannot
close the gap — the supernode count saturates at 340 regardless:

    nemin=1    nsup=498 max=16  num_us=3018
    nemin=8    nsup=347 max=16  num_us=1129
    nemin=16   nsup=342 max=18  num_us=1008   ← minimum
    nemin=32   nsup=341 max=32  num_us=1159   (current default)
    nemin=128  nsup=340 max=43  num_us=1038
    nemin=512  nsup=340 max=43  num_us=1041

## Expert consultation (2026-04-24)

Consulted `mumps-expert`, `spral-expert`, `ipopt-expert` in parallel on
what each does differently in the per-front critical path.

### MUMPS

No specialized small-front code, yet negligible per-front overhead
because:

* Stack-bump workspace (`S`/`IW` arrays with `IWPOS`/`LRLU` cursors).
  Zero heap allocations during numeric factorization.
  (`dfac_front_LDLT_type1.F` — frontal lives at `S(IWPOS:IWPOS+nrow²)`.)
* `KEEP(234)=1` in-place CB reuse: parent frontal overwrites the last
  child's contribution block memory instead of allocating a fresh
  frontal. (`dfac_asm.F`.)
* `KEEP(197)=1` symbolic-time aggressive tiny-front amalgamation:
  merges chains of small fronts past the usual merge rule.
  (`ana_orderings.F`.)
* `MUMPS_HAMF4` AMF ordering yields fatter supernodes on ACOPF graphs.

### SSIDS

Has an explicit fast path for exactly this regime, `SmallLeafNumericSubtree`
in `src/ssids/cpu/kernels/SmallLeafNumericSubtree.hxx` (paired with
`SmallLeafSymbolicSubtree.hxx`):

* At analysis time, groups the bottom of the elimination tree into
  *subtree-units* by cumulative flops (default threshold `4e6`).
* At numeric time: **one** factor-memory allocation per subtree,
  **one** `memset`, **one** OpenMP task per subtree (not per front).
* Precomputes `rlist → parent-offset` maps at analysis time so assembly
  is a straight scatter rather than a hash lookup.
* Subtrees are sized to live in L2 cache per thread.

This is the structural analogue of what feral needs.

### Ipopt

No structural exploitation of the KKT block pattern; flattens to a
generic triplet and hands it to MUMPS. The one optimization Ipopt
relies on is symbolic-factorization reuse across IPM iterations (the
sparsity pattern is invariant after iter 0). feral already supports
this via `SymbolicFactorization` reuse in
`factorize_multifrontal_with_workspace`, so this branch yields no new
leverage.

## Convergent fix vectors

Ordered by expected leverage:

1. **SmallLeafSubtree batching (SSIDS analogue)** — group consecutive
   small leaf supernodes by cumulative flops; one allocation, one
   scatter per group; precompute per-leaf layouts at symbolic time.
2. **Stack-arena workspace** (MUMPS analogue) — single bump allocator
   for the entire numeric phase; amortises across *all* supernodes, not
   only leaves. Complementary to (1).
3. **In-place CB reuse** — parent reuses last child's CB memory.
4. **Symbolic-time aggressive amalgamation** — relax the merge rule
   beyond `nemin` for chains of tiny fronts. Evidence shows the raw
   fundamental count saturates at 340 regardless of nemin, so this
   requires a different criterion (e.g. merge any front with `ncol ≤ k`
   whose parent is also small, even when it violates the postorder
   adjacency constraint — needs the SSIDS-style renumbering).
5. Inline unblocked factor path for `nrow ≤ small`.

## Why (1) first

* Documented 10× impact in SSIDS's own regression suite on ACOPF-class
  matrices.
* Targets the exact measured bottleneck (per-front dispatch + alloc).
* Symbolic-side change is localized (new `small_leaf_groups` field on
  `SymbolicFactorization`).
* Numeric-side change is a new code path on a new gate, not a rewrite
  of the existing driver — easy to parity-test by toggling the gate.
* Does not require the SSIDS-style supernode renumbering that full
  (4) needs.

## Design sketch

### Symbolic side

After `find_supernodes`, walk the supernode list in postorder and
greedily group consecutive *leaf* supernodes:

```rust
pub struct SmallLeafGroup {
    /// Indices into `SymbolicFactorization::supernodes`, in postorder.
    pub members: Vec<usize>,
    /// Sum of per-leaf `nrow * nrow` — arena size.
    pub arena_size: usize,
    /// Per-leaf offset into the arena. Length == members.len() + 1.
    pub offsets: Vec<usize>,
}

pub struct SymbolicFactorization {
    // ... existing fields ...
    pub small_leaf_groups: Vec<SmallLeafGroup>,
    /// For each supernode index: `Some(g)` if it belongs to
    /// `small_leaf_groups[g]`, else `None`.
    pub snode_group: Vec<Option<usize>>,
}
```

A supernode is a *small leaf* if:
* `children.is_empty()` (true leaf)
* `nrow ≤ SMALL_LEAF_NROW_MAX` (initial value: 16)
* `ncol ≤ SMALL_LEAF_NCOL_MAX` (initial value: 8)

A group closes when cumulative `nrow * nrow` exceeds
`SMALL_LEAF_ARENA_BUDGET` (initial value: 4096 f64s = 32 KB — an
order of magnitude below L2 on Apple Silicon) or when the next
supernode is not a small leaf.

Thresholds are initial calibration values. Must be tuned against the
full bench in a follow-up pass.

### Numeric side

New helper `factor_small_leaf_group(group, ...)`:

1. `ws.arena.clear(); ws.arena.resize(group.arena_size, 0.0)` — **one**
   allocation for the whole group.
2. For each member supernode:
   a. Scatter A entries directly into the arena slice
      `[offsets[k]..offsets[k+1]]`. Skip `build_row_indices` — the
      leaf's row layout is the supernode's own column range.
   b. Call `factor_frontal_blocked` on a `SymmetricMatrix` view over
      the arena slice. Factor is written back into the arena.
   c. Extract the contribution block into `contrib_blocks[i]`.
3. Group is done in one contiguous pass; the arena is reused across
   groups.

Dispatched from `factorize_multifrontal_supernodal_with_workspace`:
before the per-supernode loop, iterate over groups and do each group
as a batch; then skip those members in the main loop.

### Gate

New field on `NumericParams`:

```rust
pub enum SmallLeafBatch { Off, On }
```

Default `Off` until parity is verified and bench confirms the win.
Flip to `On` once ACOPR30 p90 drops and no bulk regressions appear.

## Parity oracle

Because the fast path operates only on true leaves (supernodes with
no children and no delayed columns from below), the output
`NodeFactors` are byte-identical to the sequential driver's output on
the same supernodes. The parity test is:

```
factorize_multifrontal(m, sym, {batch: Off}) ==
factorize_multifrontal(m, sym, {batch: On})
```

bit-exact on `node_factors[i].frontal_factors.l` and `.d_diag` for
every `i`, and on total `Inertia`. Run against ACOPR30, CRESC100,
HAIFAM and the existing factorize-workspace-parity corpus.

## Success criteria

* ACOPR30 `num_us` drops from ~1200 to ~250 (expected 3–5× on the
  group body; assembly + root unaffected).
* Bulk matrices (HS118, HS92, DJTL) show ≤5% regression or better.
* Full 154k bench shows sparse-factor p90 vs MUMPS dropping from
  current ~1.61 in the small-frontal bucket.

## Open questions

* Should groups span across small leaves *and* their immediate
  single-child parents? That would let one arena absorb one more
  level of assembly. Decision: **no** for v1 — parent needs
  `extend_add` which complicates the arena layout. Revisit as a
  phase 2.9.1 extension.
* Is `SMALL_LEAF_ARENA_BUDGET` the right shape parameter, or should
  we budget by flop count (SSIDS-style)? Decision: arena size is
  more directly tied to the per-front dispatch cost we're trying
  to amortise; flop count makes more sense as a parallel scheduling
  unit. Use arena size for v1.

## References

* SSIDS: `src/ssids/cpu/kernels/SmallLeafNumericSubtree.hxx`,
  `src/ssids/cpu/kernels/SmallLeafSymbolicSubtree.hxx`,
  `src/ssids/anal.F90` (analysis-side grouping).
* MUMPS: `src/dfac_front_LDLT_type1.F` (frontal kernel),
  `src/dfac_asm.F` (in-place CB assembly),
  `src/ana_orderings.F` (KEEP(197)).
* feral diagnostic: `src/bin/diag_supernode_cost.rs` (this session).
* feral evidence: `dev/journal/2026-04-24-*.org`.
