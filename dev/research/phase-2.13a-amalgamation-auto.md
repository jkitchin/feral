# Phase 2.13a — `AmalgamationStrategy::Auto`

## Motivation

Phase 2.12 flipped the amalgamation default to `Renumber`. The
flip is a win on bushy IPM-KKT etrees (ACOPR30/CRESC100/LAKES/
NELSON/SWOPF: 30-67% factor-time reduction) but a regression on
path-like etrees (MUONSINE_0000: 1.4× → 5.5× MUMPS, because
Renumber over-merged the path into a single ncol=32 root frontal
that costs 1008 µs alone).

Goal: a shape-dispatched `AmalgamationStrategy::Auto` (parallel to
`OrderingPreprocess::Auto`, default since Phase 2.4.4) that picks
`Renumber` when the etree shape is bushy and `Adjacency` when the
etree shape is path-like.

## Known answers (from Phase 2.12 diagnostic)

| matrix          | n     | best strategy | observed gap |
|-----------------|-------|---------------|--------------|
| ACOPR30_0067    | ~1.5k | Renumber      | -62% factor (Renumber/Adjacency=0.38) |
| CRESC100_0000   | ~3.0k | Renumber      | -67% factor |
| LAKES_0000      | ~600  | Renumber      | -68% factor |
| NELSON_0000     | ~700  | Renumber      | -25% factor |
| SWOPF_0000      | ~600  | Renumber      | -12% factor |
| MUONSINE_0000   | 1537  | Adjacency     | +290% factor under Renumber (regression) |
| KIRBY2_0007     | 458   | (close)       | Renumber slightly worse but symbolic dominates |

KIRBY2's symbolic phase dominates under either strategy (Phase
2.13b separately), so it's not a strong signal here. MUONSINE is
the unambiguous Renumber-loses case.

## Predicate candidates

Cheap statistics computable in O(n) from the etree alone (no
column counts, no supernode partition, no permuted pattern):

### (A) Branching factor

`mean_children = (n - n_leaves) / n_internal` where
`n_internal = n_nodes - n_leaves`. A pure path has
`mean_children ≈ 1`; a bushy tree has `mean_children > 1`.

### (B) Multi-child fraction

`multi_child_frac = n_multi_child_internal / n_internal`. On a
pure path: 0. On a bushy IPM-KKT tree: large.

### (C) Max child count

`max_children = max over nodes of |children|`. Path: 1. Bushy: >>1.

### (D) Path depth ratio

`max_chain_len / n_supernodes` — but this requires the supernode
partition, which is expensive at predicate time.

(D) is rejected as too expensive. (A), (B), (C) are O(n) on the
etree.

## Probe design

`src/bin/diag_etree_shape.rs` runs `symbolic_factorize` on each
known-answer matrix and prints (A), (B), (C) plus a few cross-checks.
The expected pattern is something like:

| matrix         | mean_children | multi_child_frac | max_children |
|----------------|---------------|------------------|--------------|
| MUONSINE_0000  | ≈1.00         | ≈0               | low          |
| ACOPR30_0067   | >1.5          | high             | high         |

If (B) cleanly separates Renumber-wins from Renumber-loses across
all 7 matrices with a single threshold, use (B). Otherwise fall
back to a combination.

## Default predicate (provisional, pending probe)

Hypothesis: `multi_child_frac < 0.05` → `Adjacency`, else
`Renumber`. This is a "near-path" detector. The 0.05 threshold
admits some leaves clustered at a single internal junction (which
is fine for Renumber) but rejects pure paths.

The actual threshold will be set by the probe output.

## Predicate runs once per call

The shape predicate runs **before** `find_supernodes` on the etree
that exists at that point in `symbolic_factorize_with_method`.
It's at most O(n) and replaces a stage that is itself ~10× more
expensive on every relevant matrix. Cost is negligible.

## Implementation outline

1. `pub enum AmalgamationStrategy` gains an `Auto` variant.
2. `Auto` becomes `#[default]`; `Renumber` stays as an
   explicit-opt-in (and the documented "what Auto picks for bushy
   trees"); `Adjacency` stays as the explicit-opt-in escape hatch.
3. `pub fn pick_amalgamation_strategy(etree: &EliminationTree) ->
   AmalgamationStrategy` returns either `Adjacency` or `Renumber`
   based on the predicate.
4. In `symbolic_factorize_with_method`: resolve `Auto` to the
   concrete dispatch result before the existing
   `matches!(snode_params.amalgamation_strategy, Renumber)`
   branch.

## Success criteria

(From the parent plan.)

- Corpus median sparse factor ratio within ±2% of Adjacency
  baseline (i.e., recover most of the +10% Renumber p50 cost).
- Tail ACOPR30/CRESC100 keep the 60-67% factor reduction.
- MUONSINE regression eliminated.

## Out of scope

- Predicate based on column counts or supernode partition — too
  expensive to pay before knowing the answer.
- Tuning Renumber's merge prediction itself. Auto's job is to pick
  the right strategy; refining either strategy is its own phase.
- Recovering KIRBY2 small-n cost. That's Phase 2.13b territory
  (AMD per-call shrink or symbolic caching).

## References

- `dev/plans/phase-2.13-tail-diagnostic.md` — parent plan
- `dev/research/phase-2.12-column-renumbering.md` — Renumber
  rationale
- `dev/sessions/2026-04-25-03.md` — Phase 2.12 result table
- `src/symbolic/supernode.rs` — `AmalgamationStrategy` enum
- `src/symbolic/mod.rs:475-507` — current Renumber dispatch site
- `src/symbolic/mod.rs:374-377` — `OrderingPreprocess::Auto`
  resolution as the structural template
