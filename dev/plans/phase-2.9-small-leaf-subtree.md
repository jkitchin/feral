# Phase 2.9 — SmallLeafSubtree batching

Research backing: `dev/research/phase-2.9-small-leaf-subtree.md`.

Goal: close the 10× per-front-overhead gap on long-tail IPM matrices
(ACOPR30, CRESC100, HAIFAM) by batching true-leaf supernodes into
subtree-units that share a single arena allocation and a single
pattern-scatter pass.

## Step A — Symbolic: detect small-leaf groups

File: `src/symbolic/supernode.rs` (new module `small_leaf.rs` or
extend `supernode.rs`) and `src/symbolic/mod.rs`.

1. Add public struct `SmallLeafGroup { members, arena_size, offsets }`
   as described in the research note.
2. Add field `small_leaf_groups: Vec<SmallLeafGroup>` and
   `snode_group: Vec<Option<usize>>` to `SymbolicFactorization`.
3. Add free function:

   ```rust
   pub fn find_small_leaf_groups(
       supernodes: &[Supernode],
       params: &SmallLeafParams,
   ) -> (Vec<SmallLeafGroup>, Vec<Option<usize>>);
   ```

4. Wire it into `symbolic_factorize_with_method` after
   `find_supernodes`, before returning the `SymbolicFactorization`.
5. Params struct:

   ```rust
   pub struct SmallLeafParams {
       pub nrow_max: usize,    // default 16
       pub ncol_max: usize,    // default 8
       pub arena_budget: usize // default 4096 f64s
   }
   ```

   Place on `SupernodeParams` as `pub small_leaf: SmallLeafParams`.

6. Grouping algorithm:
   ```
   for each snode in postorder:
     if is_small_leaf(snode):
       if current_group.arena_size + snode.nrow² > budget:
         close current_group
       add snode to current_group
     else:
       close current_group
   close current_group
   ```
   A supernode is a leaf iff `children.is_empty()`. Small =
   `nrow <= nrow_max && ncol <= ncol_max`.

### Tests (unit, under `#[cfg(test)]` in the module)

* `small_leaf_grouping_tridiagonal`: 20-wide tridiagonal → 1 group of
  several tiny fronts.
* `small_leaf_grouping_large_front_breaks_group`: chain of small leaves
  followed by a non-leaf should leave the non-leaf out.
* `small_leaf_grouping_budget_forces_split`: budget=4 forces two
  groups out of what would otherwise be one.
* `snode_group_mapping`: for each member, `snode_group[idx] == Some(g)`.

## Step B — Numeric: batched leaf factor path

File: `src/numeric/factorize.rs`.

1. Add `ws.leaf_arena: Vec<f64>` to `FactorWorkspace`.
2. Add `fn factor_small_leaf_group(...)` that:
   * Resizes `ws.leaf_arena` to `group.arena_size`; zeroes it.
   * For each member:
     - Scatter permuted `A` entries into
       `ws.leaf_arena[offsets[k]..offsets[k+1]]` using
       `row_indices = (first_col..first_col+nrow)`. Apply scaling in
       place.
     - Construct a `SymmetricMatrix` view backed by that slice.
     - `factor_frontal_blocked(view, ncol, may_delay=true, &bk)`.
     - Extract the contribution block into `contrib_blocks[idx]`.
     - Write `NodeFactors` into the per-group `Vec<Option<NodeFactors>>`
       slot at `idx`.
3. In `factorize_multifrontal_supernodal_with_workspace`:
   * Before the main postorder loop, iterate over
     `symbolic.small_leaf_groups` and batch-process each.
   * In the main loop, **skip** any supernode whose
     `snode_group[idx].is_some()`.
4. Construct `node_factors: Vec<NodeFactors>` in strict postorder by
   merging group-produced nodes with loop-produced nodes using
   `snode_group`.

**Critical correctness points**

* The leaf's frontal dimension must match what the existing path
  computes. Verify by calling `build_row_indices` on a leaf in a
  debug assert and comparing against the naive
  `(first_col..first_col+nrow)` construction. If they differ for any
  leaf (because the full symmetric pattern adds trailing rows that
  aren't in the local range), **fall back** to the scalar path for
  that leaf (disqualify it from grouping in symbolic).
* Scaling vector indexing: uses `scaling_pivot_order[row_index]`,
  same as the scalar path.
* Inertia, `needs_refinement` accumulation: identical semantics to
  the scalar path.

## Step C — Gate

1. Add enum and field to `NumericParams`:

   ```rust
   #[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
   pub enum SmallLeafBatch {
       #[default]
       Off,
       On,
   }

   pub struct NumericParams {
       ...
       pub small_leaf: SmallLeafBatch,
   }
   ```

2. In `factorize_multifrontal_supernodal_with_workspace`, dispatch to
   the group path **only when** `params.small_leaf == On`.
3. Default stays `Off` — zero behavioral change for existing callers
   until we explicitly flip it.

## Step D — Parity tests

File: `tests/small_leaf_parity.rs`.

Run the same factorize with `small_leaf: Off` and `small_leaf: On`;
assert bit-exact parity on:

* `node_factors[i].frontal_factors.l` (full Vec<f64>)
* `node_factors[i].frontal_factors.d_diag`
* `node_factors[i].frontal_factors.d_subdiag`
* `node_factors[i].frontal_factors.perm`
* Total `Inertia`

Corpus:

* A handful of small fixed matrices (block-diagonal, tridiagonal).
* ACOPR30_0067 — the archetype.
* CRESC100_0000 — long-tail bulk.
* HAIFAM_0082 — different tree shape.
* A matrix from the existing `tests/factor_workspace_parity.rs` corpus
  to catch any cross-phase regression.

Plus a focused unit test of `factor_small_leaf_group` on a crafted
3-leaf group with known-answer arithmetic.

## Step E — Bench

File: `src/bin/diag_small_leaf.rs`.

For a fixed set of long-tail matrices, report `num_us` under:

* `small_leaf: Off` (current default)
* `small_leaf: On`
* computed speedup

And run the 154k-matrix bench (`cargo run --release --bin bench`)
with the gate on vs off to confirm no bulk regression.

## Step F — Flip default

Flip `SmallLeafBatch::default()` from `Off` to `On` when:

* All parity tests pass.
* Long-tail ACOPR30 `num_us` drops ≥ 3×.
* No individual bulk family (HS118, HS92, DJTL, ...) regresses > 5%
  on its geomean ratio vs MUMPS.

Record the flip in `dev/decisions.md`.

## Out of scope for this phase

* Stack-arena allocator spanning *all* supernodes (MUMPS-style). This
  requires rethinking the contribution-block ownership graph. Defer to
  phase 2.9.2.
* In-place CB reuse. Defer to phase 2.9.3.
* Aggressive symbolic amalgamation past postorder-adjacency. Requires
  SSIDS-style supernode renumbering — defer to phase 2.9.4.
* Extending groups across single-child parents. Defer to phase 2.9.1.
