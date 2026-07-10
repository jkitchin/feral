# Issue #131 implementation plan — 2026-07-10

Design + bit-exactness analysis: `dev/research/issue-131-parallelism-design-2026-07-10.md`.
User scope decisions (2026-07-10): solve gap → **full contribution-block
rewrite** (accept residual re-baseline, show diffs before any tolerance change);
assembly gap → **#125 static maps first, then column-partitioned parallel
assembly**.

Sequencing (bit-exact/autonomous track first, sign-off track last):

## Part 1 — #125 static assembly maps (bit-exact, autonomous)

Precompute per-supernode static row_indices at symbolic time; use them in the
numeric path when `n_delayed_in == 0`.

**Key fact (bit-exactness):** `build_row_indices`' output is a function of only
static structure whenever the node's `n_delayed_in == 0` — the delayed set is
empty, and the trailing set is `own-pattern-reach ∪ children's-contrib-trailing`,
then **sorted** (so numeric pivot order `ff.perm` cannot affect it). A child's
contrib-trailing equals its static separator iff the child eliminated all its
columns (child `n_delayed == 0`), which is exactly the `n_delayed_in == 0`
condition on the parent. So `static == build_row_indices` bit-for-bit under the
gate; delayed-pivot fronts fall back to `build_row_indices`.

Steps:
1. Storage: flat `static_row_offsets: Vec<usize>` (len n_snodes+1) +
   `static_row_indices_flat: Vec<usize>` on `SymbolicFactorization`.
2. Compute in one postorder pass (supernodes are already child-before-parent):
   for each node, trailing = sorted-dedup of (own-col pattern reach ≥ own_last) ∪
   (each child's static separator ≥ own_last); layout `[own cols | trailing]`.
   Mirror `build_row_indices` filters exactly.
3. Numeric: in `factor_one_supernode` (both drivers), when `n_delayed_in == 0`
   use the static slice; else `build_row_indices`.
4. Tests first: `tests/static_assembly_maps.rs` — for a zero-delay matrix,
   symbolic static slice == `build_row_indices` for every node; full suite green
   (numeric output unchanged ⇒ existing parity tests pass).
5. Bench: prologue (`BUILDROW_NS`) drop via perf_probe.

## Part 2 — #131 Gap B: column-partitioned parallel assembly (bit-exact)

On top of Part 1: partition a front's destination columns into disjoint chunks;
each thread scatters original entries + extend-adds every child's entries whose
destination column is in its chunk. Disjoint columns ⇒ disjoint `f_data` memory
⇒ bit-exact (each cell keeps serial child-order `+=`). Gate on front area/flops
and only under the parallel driver (`intrafront_parallel`). Tests: parity with
3+ children, disjoint/overlapping ranges; determinism (repeat == identical bits).
Bench: assembly-phase scaling on grid220.

## Part 3 — #131 Gap A: contribution-block solve rewrite (needs sign-off)

Rewrite both solve cores (`solve_sparse_core_into`,
`solve_sparse_core_many_into`) to contribution-block form: each node produces a
private contribution to its parent's RHS rows; parent sums children in fixed
child-order (mirrors factor `extend_add`). Then siblings solve concurrently into
private blocks; forward = leaves-up, backward = root-down; serial within small
subtrees (flop threshold). Carry the assembly tree into `SparseFactors`.

Bit-exactness: new-serial == new-parallel (byte-identical, the #131 contract).
NOT bit-identical to the *old* serial solve (arithmetic reorder). **Before
changing any residual-test tolerance, show the user the exact diffs** (hard
rule). Tests: `tests/parallel_solve_parity.rs` (serial==parallel bits;
post-delayed-pivot factors; determinism across repeats). Bench: solve-phase
thread-scaling on grid220 (baseline 13.5 ms fully serial) + nuffield2.
