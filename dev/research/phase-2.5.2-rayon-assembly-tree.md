# Phase 2.5.2 — Rayon on the assembly tree: scoping

## Premise (from `dev/plans/phase-2-planning.md` §2.5.2)

> Use Rayon to parallelize the independent subtrees of the assembly
> tree. Sibling supernodes can be factored in parallel; only the
> join at the parent supernode is sequential. `ContribPool`
> transitions from a LIFO stack to a buddy allocator so contribution
> blocks from independently-running siblings do not collide.

## Current structure (pre-2.5.2)

`factorize_multifrontal_supernodal_with_workspace` in
`src/numeric/factorize.rs:383`:

- Single postorder loop over `symbolic.supernodes[0..n_snodes]`.
- Shared `contrib_blocks: Vec<Option<ContribBlock>>` indexed by
  supernode.
- Shared `FactorWorkspace` (row_map, frontal_values, build_*)
  accessed/mutated inside each iteration.
- Shared `node_factors: Vec<NodeFactors>` appended in postorder.
- Shared `total_inertia`, `needs_refinement` accumulators.

The planning doc's "ContribPool / LIFO stack" does not exist; the
current implementation is a plain `Vec<Option<ContribBlock>>`. So
the planning doc's buddy-allocator upgrade is not the right framing
for FERAL as-is.

## Parallelism model: independent subtrees of the etree forest

Observation: at any node `j`, the supernodes rooted at `j.children`
are independent — their factorization produces contrib blocks that
only affect `j`, not each other. So the classical "divide the etree
at the top into child-subtrees" scheme applies.

```
                 root (serial join)
                /     \
         subtree_A    subtree_B    ← fully independent
          /    \       /    \        (can factor in parallel)
         ...  ...    ...   ...
```

For a typical KKT matrix, a moderate fraction of the work lives in
independent subtrees below the root; the root supernode itself is
often a modest-sized dense front that must be factored last
(serial).

## Minimum-viable design (this session)

A non-destructive parallel driver on top of the existing sequential
body. Keep the sequential path as the default and reference
implementation — introduce a parallel variant only for corpora
where the work-available-to-parallelise threshold is met.

### Ingredients

1. **rayon** as a dependency.
2. A subtree-partition pass on the symbolic supernodes: for each
   root, collect its descendants; then for each root, enumerate
   the immediate-child subtree partition (the supernode indices
   reachable from each child, postorder-preserving).
3. A **per-subtree `FactorWorkspace`** — each parallel task owns
   its own workspace. Shared state (symbolic, scaling, permuted
   matrix pattern/values) is borrowed immutably.
4. A collection/merge step: the parallel tasks each produce a
   list of `(snode_idx, NodeFactors, Option<ContribBlock>,
   per-subtree inertia accumulator)`. After the tasks join, the
   main driver inserts them at the correct positions in the
   shared `node_factors` and `contrib_blocks` vectors, then
   factors the remaining ancestor supernodes (those whose
   children mix across subtrees — typically just the root and a
   handful near it).

### Threshold gate

Only engage parallelism when:
- `n_snodes >= N_PAR_MIN` (say 32) — below this, rayon overhead
  dominates.
- Root has `>= 2` children with non-trivial subtrees (each
  subtree has at least a handful of supernodes).

Otherwise, the parallel path delegates to the sequential body
verbatim.

## Risks and mitigations

- **Risk: bit drift.** Supernode postorder is changed by parallel
  scheduling. Floating-point sums differ, inertia for
  quasi-singular matrices can shift. Mitigation: the extend-add
  into a supernode is associative/commutative at the (row, col)
  level (a dense scatter by global index), so within a supernode
  the parent's frontal sum is deterministic regardless of child
  order. Children themselves do not interact; their contrib blocks
  are independent. Thus bit-exactness is expected across the full
  corpus.

- **Risk: workspace memory.** Per-subtree workspaces multiply peak
  memory. Mitigation: reuse the top-level `FactorWorkspace` for
  one subtree (e.g. the largest); allocate fresh workspaces only
  for the extra parallel tasks.

- **Risk: complexity blast.** Implementing a full parallel driver
  with partition, task dispatch, and merge is sizable. Mitigation:
  start with a trivial gate (only activate when the work
  partition is obvious and the gain is large), fall back to
  sequential otherwise.

## Exit criteria

1. All lib tests pass (129/129).
2. Full corpus bench: bit-exact inertia and residual match vs
   pre-change on all 154k KKT matrices.
3. No regression on dense or sparse factor/MUMPS p90.
4. Demonstrable speedup on a large, branchy matrix (CRESC132
   class) where the etree has meaningful independent subtrees.

## Fallback position

If the implementation does not reach bit-exact parity in a single
session, the plan is to:
- Leave the sequential path as the production path.
- Land the partition logic and per-subtree workspace infrastructure
  as scaffolding, gated off by a feature flag.
- Document what was learned for the next session.

Premature parallelism that regresses correctness is worse than
no parallelism. Falling back is explicit and acceptable.

## References

- `src/numeric/factorize.rs:383` (supernodal body)
- `src/numeric/factorize.rs:479-480` (contrib_blocks)
- `src/symbolic/supernode.rs` (Supernode::children)
- `dev/plans/phase-2-planning.md:505-513` (original 2.5.2 spec)
