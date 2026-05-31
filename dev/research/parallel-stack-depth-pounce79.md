# Parallel multifrontal worker-stack depth (pounce#79)

Investigation note. **Question:** does the parallel task-graph driver
`run_parallel_task` (`src/numeric/factorize.rs`) keep worker-stack depth
O(1) regardless of elimination-tree height, or can a deep / path-like
tree drive native stack depth ∝ tree height (and so overflow a rayon
worker's default ~2 MiB stack)?

**Answer:** depth is **O(1) in tree height** — the leaf→root climb is
trampolined through rayon's task queue, not native call-stack recursion.
No behavioral change is needed; this note records the structural argument
and the measurements, and a regression guard was added.

## Origin

A downstream consumer (pounce's batched QP, `solve_qp_batch_parallel`)
hit rayon worker-stack overflows running feral's parallel factorizer
*nested inside its own `par_iter`* (parallel-over-parallel) and worked
around it with a 64 MiB-stack pool. pounce#79 then switched the inner
backend to `Solver::with_parallel(false)` (the sequential driver is a
flat postorder loop — `while snode_idx < n_snodes` in
`factorize_multifrontal_supernodal_with_workspace` — and runs fine on
default stacks). This note covers the remaining question about feral's
*standalone* parallel path.

## Structural argument: the climb is trampolined, not recursive

`run_parallel_task` (`factorize.rs:3069`) is seeded once per leaf inside
`rayon::scope(|scope| …)` (`:2951`). Its entire body is
`scope.spawn(move |s| { … })`. At the bottom of that closure, when a task
finishes its supernode and decrements its parent's pending counter to
zero, it "recurses":

```rust
if let Some(parent_idx) = parents[snode_idx] {
    let prev = pending[parent_idx].fetch_sub(1, Ordering::AcqRel);
    if prev == 1 {
        run_parallel_task(s, parent_idx, …);   // factorize.rs:3232
    }
}
```

But `run_parallel_task` only calls `scope.spawn(...)` and returns — it
does **not** factor the parent on the current frame. The parent's
factorization runs in a *freshly spawned task* that rayon picks up after
the current task's frame has popped. So the native call stack never nests
parent-on-child: each task is a small constant frame plus the per-front
dense-kernel scratch (bounded by front size, also independent of tree
height). A depth-h chain produces h *sequential spawns over time*, not h
*nested frames*.

This contrasts with the documented overflow, which came from
**oversubscription** (an outer `par_iter` each running an inner parallel
factor), not from tree depth. The fix for that is a serial inner backend
(`with_parallel(false)`), not a bigger stack.

## Measurements

A throwaway probe (`src/bin/probe_stack_need.rs`, removed after use)
factored matrices on the parallel driver inside a rayon pool with a
CLI-chosen `stack_size`, exiting 0 on success and crashing (SIGSEGV) if
the stack were exhausted. If the climb were native recursion, a deep tree
would need stack ∝ depth; instead:

| matrix   | n        | supernode-tree height | smallest worker stack that still factors |
|----------|---------:|----------------------:|------------------------------------------|
| bratu3d  | 27 792   | 154                   | factors even at requested **1 KiB** (OS rounds up to pthread min) |
| c-big    | 345 241  | **1521**              | OK at **32 KiB** (full sweep 4096→32 KiB all OK) — far under the 2 MiB default |

Tree heights came from a second throwaway probe over the corpus: every
optimization/KKT matrix (ACOPP, CRESC132, VESUVIO, MUONSINE, …) has
supernode-tree height ≤ 9; the deep cases are large 3D/FEM problems
(c-big 1521, bratu3d 154), and even those need a trivially small stack.
A standalone tridiagonal SPD chain (n = 8000) under the default ordering
amalgamates to ~500 supernodes (supernode-tree height ~500) and factors
fine. All findings are exactly what the trampoline predicts: stack need
tracks **per-front frame size**, which is O(1) in height, not height.

## Decision

Depth is already O(1) → **document + guard, no behavioral change**:

1. Doc comment on `run_parallel_task` explaining the trampoline and the
   O(1)-in-height property, with the measured evidence, and noting that
   `ensure_parallel_pool` (`src/numeric/solver.rs`) therefore needs no
   enlarged `stack_size`.
2. Regression test
   `tests/parallel_parity.rs::deep_chain_tree_no_stack_overflow`: a
   tridiagonal SPD system, n = 8000 (default ordering → deep supernode
   chain, height ~500). It factors on default worker stacks and asserts
   SPD inertia (8000, 0, 0) plus bit-exact inertia + factor parity with
   the sequential driver. If a future refactor reintroduced native
   leaf→root recursion, a deep chain would crash this test
   (SIGSEGV/SIGABRT) rather than fail an assertion.

The iterative-worklist rewrite the task offered as a fallback is
**unnecessary** — the spawn-based scheduler is already the iterative
form; converting it would churn a bit-exact-tested numeric path for no
behavioral gain.

## Parity preserved

`solver_parallel_factor_matches_sequential` (`src/numeric/solver.rs`)
and all six existing `parallel_parity_*` corpus tests still pass; the new
deep-chain test adds bit-exact parity at the chain shape.
