# Issue #131 — parallelism gaps: design + bit-exactness analysis — 2026-07-10

Follows the measure-first investigation (session 05) that found #131 — unlike
#130/#132/#133 — has **genuine measured headroom**: on a bushy nested-dissection
tree (grid220, n=48400) factor scales only ~1.99× on 4 cores and the solve is
**100% serial** (13.5 ms, 6–13% of factor); on a path-like tree (nuffield2)
neither scales (1.11× factor, solve 0.3% of factor). This note takes the next
step required by the protocol — code inspection + design — before any code.

The headline finding: **the solve gap (Gap A) cannot be made bit-exact against
today's serial solve without rewriting the solve core to a contribution-block
formulation.** The assembly gap (Gap B) is bit-exact-safe but second-order. Both
are strongly tree-shape-dependent, and the primary IPM/KKT workloads often have
path-like trees where tree parallelism gives ~nothing. Details below.

## Gap A — tree-parallel forward/backward solve

### How the serial solve works today (`src/numeric/solve.rs:236`)

`solve_sparse_core_into` is a shared-global-vector formulation:

```
for node in node_factors (postorder):        # forward
    gather   w[i] = y[row_indices[perm[i]]]   for i in 0..nrow   # incl. separators
    L-solve  w[i] -= L[j,i]*w[j]              for j<nelim, i>j    # updates separators too
    D-solve  w[0..nelim] *= D^-1
    scatter  y[row_indices[perm[i]]] = w[i]   for i in 0..nrow   # incl. separators
```

The gather/scatter covers **all** `nrow` rows, not just the `nelim` eliminated
ones. The separator rows (`nelim..nrow`) are ancestor variables, and the
scatter is a read-modify-write of `y` at those entries (`y[sep] = y[sep] -
Σ L[j,i] w_j`).

### Why naive subtree parallelism races

A variable `v` eliminated at node `p` appears as a **separator row in every
descendant of `p`** that reaches it. Two sibling subtrees `A`, `B` (children of
some `q`) are both descendants of `q` and of every ancestor of `q`, so both
contain nodes with `v ∈ separators` for every `v` eliminated at `q` or above.
Running `A` and `B` on different threads means both do `y[v] -= …` via the
gather/scatter — a lost-update race on shared `y` entries. This is **not** an
edge case; the parent's pivot rows receive contributions from *all* children by
construction. Confirmed by reading the gather/scatter at `solve.rs:270-323`.

### Why it can't be made bit-exact by private accumulation

The obvious fix — each subtree accumulates a private delta `dA` for shared
entries, then `y[v] -= dA` — is **not bit-exact**. Serial computes
`((y0 - a1) - a2) - b1 …` (individual subtractions in postorder). Lumping
`dA = a1 + a2` and doing `y0 - dA` reorders the floating-point adds; by
non-associativity `y0 - (a1+a2) ≠ (y0 - a1) - a2`. The established contract for
#131 is *byte-identical* serial-vs-parallel, so any lumped reduction violates it.

The only way to stay bit-exact with the current arithmetic is to apply each
individual separator subtraction to `y` in the exact serial (postorder) order —
which serializes precisely the shared-separator writes. Those writes are a large
fraction of the total solve work (they are the `i>=nelim` half of every L-solve
inner loop). So even a "compute in parallel, apply serially" scheme has a big
serial fraction — Amdahl caps it low. The dominant root front is the *last*
node (no tree parallelism there regardless).

### What a real speedup would require

Rewrite both solve cores to the **contribution-block formulation** MUMPS/SSIDS
use: each node produces a private contribution to its parent's RHS rows; the
parent sums children in fixed order (exactly mirroring the factor's `extend_add`,
which is bit-exact because the child-order reduction is serial at the parent).
Siblings then solve concurrently into private blocks. This is a genuine parallel
win — but it **changes the arithmetic order versus today's serial solve**, so it
is *not* bit-exact against the current serial path. To keep a bit-exactness
contract we would rewrite the serial solve to the same formulation, then assert
serial-new == parallel-new. That is real surgery on the bit-exact-tested numeric
core (risks the ~1e-15 residual assertions in `tests/`), for a payoff that is
~8 ms on grid220 and ~0 on path-like trees.

## Gap B — parallel assembly (`extend_add`, `factorize.rs:4053`)

`extend_add` does `f_data[col*fn_ + row] += val` with
`(row,col)=(max(pi,pj),min(pi,pj))` — canonicalised to the parent's lower
triangle. Bit-exact-safe parallelism **is** possible here: partition the
destination frontal **columns** into disjoint chunks, one per thread; each
thread scatters original entries + extend-adds every child's entries landing in
its column range. Disjoint `col` ⇒ disjoint `f_data` memory ⇒ no race; the `+=`
into each cell keeps its serial child-order. Bit-exact by construction.

Caveats that make it second-order:
- Assembly is `O(nrow²)` per front; the factor it feeds is `O(nrow³)` and is
  **already** intra-front parallelised (Lever 1.1, `apply_blocked_schur_panel`).
  So parallel assembly chases the smaller term.
- The canonicalisation means "entries landing in column range `[c0,c1)`" is not
  a contiguous slice of the child loop; efficient column-partitioned assembly
  wants the per-front static assembly maps that **#125** (still open) proposes.
  Building them on the fly each factor partly eats the gain.

## Tree-shape dependence (applies to both gaps)

Measured 4-core scaling (session 05):

| matrix    | tree shape          | factor 1t→4t | solve fraction |
|-----------|---------------------|--------------|----------------|
| grid220   | bushy nested-diss.  | 205→103 ms (1.99×) | 6–13% (serial) |
| nuffield2 | path-like           | 7.37→6.62 s (1.11×) | 0.3% (serial) |
| dense1400 | single front        | flat (routes serial) | — |
| arrow     | path + big root     | flat | — |

Tree parallelism (both gaps) helps **only bushy trees**. Many KKT/optimization
matrices — the primary consumer — factor to path-like or lightly-branched trees
(nuffield2 is a KKT trap matrix). So #131's expected value is concentrated on a
subset of the corpus and is near-zero on an important part of the target
workload.

## Recommendation

Ordered by value-per-risk:

1. **Gap B parallel assembly, column-partitioned** — bit-exact-safe, issue-first,
   but second-order (chases `O(nrow²)` behind already-parallel `O(nrow³)`).
   Cleanest with #125's static maps landed first.
2. **Gap A tree-parallel solve** — needs the contribution-block rewrite of the
   solve core to get real, bit-exact parallelism; larger and riskier (touches the
   bit-exact numeric core), payoff ~8 ms on bushy trees / ~0 on path trees.
3. **Panel parallelism** — the issue itself defers this last ("consider after
   #129's measurement"); #129 measured *not justified*
   (`panel-fragmentation-2026-07-10.md`), so the panel-tiling motivation is weak.

This is recorded per the measure-first + correctness-before-performance rules so
the scope decision is made on evidence, not on the optimistic headroom headline
alone.
