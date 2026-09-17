# Plan: node-separator FM refinement in feral-metis

**Issue:** #203. **Research:**
`dev/research/feral-metis-node-separator-fm-2026-09-17.md`.

Replace `feral-metis`'s "edge-cut FM through uncoarsening, one greedy
separator pass at the end" with METIS's "node separator from the coarsest
level, FM-refined at every level".

> **Outcome (2026-09-17): P1 and P2 landed, P3 flipped the default, P4 not
> started.** The acceptance table below was met on 1 of its 4 rows — nfe=48
> beat its target, nfe=96 / grid2d / grid3d all missed. The default was
> flipped anyway because the change is never worse on any of the 40 matrices
> measured and is 2.4x-3.5x better on the target class; the reasoning and the
> miss are recorded in `dev/decisions.md` (2026-09-17). Measured numbers are
> in `dev/research/feral-metis-node-separator-fm-2026-09-17.md`.

## Acceptance

The gate is symbolic fill and flops, measured by replaying each permutation
through feral's own symbolic pipeline (`issue203_fill_probe`), so the numbers
are comparable across solvers.

| graph | metric | now | target | oracle |
|---|---|---|---|---|
| grid3d 40^3 | flops | 2.31e10 | <= 1.9e10 | real METIS 1.67e10 |
| grid2d 300x300 | flops | 4.69e8 | <= 3.7e8 | real METIS 3.33e8 |
| gaslib nfe=48 | flops | 2.10e10 | <= 9.0e9 | real METIS 6.42e9 |
| gaslib nfe=96 | flops | 6.89e10 | <= 2.5e10 | real METIS 1.75e10 |

Plus, as guardrails:

* no regression worse than 1.05x in `factor_nnz_estimate` on any matrix in
  `tests/data/parity` or `tests/data/large`;
* `MetisND` symbolic time no worse than 1.5x its current value on nfe=96;
* two runs with the same seed produce bit-identical permutations.

The oracle is external (MA57's bundled real METIS, via `ma57_analyse`), which
satisfies the protocol's "no implementation and oracle in the same session"
rule: the target numbers above were measured before any code was written.

## Phases

### P1 — `refine_separator_fm` (new, `fm_refine.rs`)

Real FM on the node separator, one-sided, with:

* `edeg[v][A]`, `edeg[v][B]` maintained incrementally for separator vertices;
* a bucket priority queue keyed by `gain = vwgt[v] - edeg[v][other]`, with
  deterministic tie-breaking (lowest vertex id wins);
* negative-gain moves allowed, `limit = min(3 * n_sep, 300)` consecutive
  non-improving moves before the pass stops;
* an undo log, and rollback to the best `(sep_weight, balanced)` state seen;
* balance constraint `pwgts[to] + vwgt[v] <= (1 + max_imbalance) * total / 2`;
* passes alternate the target side, `max_passes` of them.

Returns the final separator weight. Leaves `refine_separator` in place —
`P1` is additive so the two can be compared arm to arm.

**Tests (written first):**

* a path graph `0-1-...-n-1`: the minimum separator is a single vertex; FM
  must find it from a deliberately bad 5-vertex separator;
* a 2-D grid `k x k`: the separator must come out at `<= k` (a straight cut
  exists) from a ragged initial separator;
* a barbell (two cliques joined by one edge): separator must be one vertex;
* balance is never violated for any `max_imbalance` in {0.0, 0.03, 0.2, 0.4};
* the result is a valid separator (`is_valid_separator`) after every pass;
* determinism: two runs on the same input agree bit for bit;
* never worse than the input: `refine_separator_fm` output weight is `<=` the
  input separator weight (the rollback guarantees this).

### P2 — separator through the uncoarsening hierarchy (`node_nd.rs`)

Restructure `multilevel_node_bisection`:

```
coarsen -> levels
at the coarsest level: niparts edge bisections + refine_bisection, keep best
construct_separator(coarsest)          <-- moved here from the finest level
refine_separator_fm(coarsest)
for level in rev:
    project the 3-way labels through cmap to the finer graph
    refine_separator_fm(finer)
```

The projection is the existing `cmap` copy — it is label-agnostic, so it
carries `PART_SEP` unchanged.

**Tests:** `is_valid_separator` holds at every level (debug assertion plus an
explicit test on a grid); the ND driver's existing tests still pass; a grid
graph's top separator does not grow relative to the current code.

### P3 — option, measurement, default

Add `MetisOptions::node_refine: bool`. Land it **default `false`**, measure
both arms on the four acceptance graphs plus the parity corpus, and flip the
default to `true` in the same session only if the acceptance table is met and
no guardrail is broken. If it is not met, the arm stays opt-in and the result
goes in `tried-and-rejected.md` with the numbers.

### P4 — downstream

Only if P3 flips the default: re-check `pick_default_method` /
`choose_adaptive`, since a better `MetisND` changes where the `Amf`/`MetisND`
boundary should sit. Out of scope for the first PR; note it in the session
checkpoint.

## Non-goals

* Two-sided node FM (`FM_2WayNodeRefine2Sided`). One-sided is METIS's own
  NodeND default.
* `FM_2WayNodeBalance`. The balance constraint inside the FM pass covers the
  common case; a separate balancing pass is only needed when the projected
  partition arrives already infeasible, which cannot happen while every level
  enforces the same bound.
* Graph compression. Measured to be a no-op on the issue #203 pattern.
* Coarsening (SHEM) changes. If P3 misses the targets, that is the next
  suspect, not this PR.
