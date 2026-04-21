# Phase 2.5.4 — Fill prediction: scoping and finding

## Premise (from `dev/plans/phase-2-planning.md` §2.5.4)

> Improve the `factor_slack` heuristic (currently `1.2×` the predicted
> NNZ). Use the SSIDS approach of tracking the actual delay rate and
> adjusting dynamically.

## Finding: 2.5.4 has no hot-path leverage as specified

An exhaustive grep shows `factor_nnz_estimate` and `factor_slack`
live purely in the symbolic output metadata:

```
src/symbolic/mod.rs:127  pub factor_nnz_estimate: usize,
src/symbolic/mod.rs:129  /// Slack factor applied to factor_nnz_estimate. Default 1.2.
src/symbolic/mod.rs:130  pub factor_slack: f64,
src/symbolic/mod.rs:349  let factor_slack = 1.2;
src/symbolic/mod.rs:356  factor_nnz_estimate: (factor_nnz as f64 * factor_slack) as usize,
src/bin/bench_orderings.rs:47  Some((sym.factor_nnz_estimate as u64, us))
src/bin/hs85_diag.rs:104  println!("  factor_nnz_est    = {}", sym.factor_nnz_estimate);
```

Consumers outside those three sites: **none**. The value is stored
on `SymbolicFactorization` and referenced only by the ordering
bench-harness (`bench_orderings`) and the HS85 diagnostic printer,
plus three test asserts (`factor_nnz_estimate > 0`).

The field does **not** drive allocation of:

- `NodeFactors.frontal_factors.l` — populated by
  `factor_frontal_blocked` per supernode; no slack prediction
  involved.
- `ContribBlock.data` — cloned from `ff.contrib` per supernode;
  no pool, no preallocation, no slack.
- `FactorWorkspace.frontal_values` — grows monotonically to the
  largest `nrow * nrow` seen; slack not consulted.
- `FactorWorkspace.dense_values` — new (Phase 2.5.x pool);
  grows to `n * n`; slack not consulted.
- `SparseFactors.node_factors` — sized to `n_snodes`, not NNZ.

Likewise `peak_contrib_bytes` (compute_peak_contrib) is metadata
only, never preallocated against — there is no `ContribPool` in
the current code, so the prediction is not used.

## Why: feral's memory strategy is per-supernode grow-as-needed

FERAL does not preallocate a single `factor_nnz`-sized L buffer
and assign slices of it to supernodes (which is the SSIDS/MUMPS
pattern). Instead each `NodeFactors` owns its own `l: Vec<f64>`
populated inside `factor_frontal_blocked`. A dynamic slack
estimate cannot make the allocator cheaper because no single
allocation sized by `factor_slack * factor_nnz` exists.

Similarly, contribution blocks live in
`Vec<Option<ContribBlock>>` indexed by supernode: each is a
separate `Vec<f64>` cloned from the kernel's `ff.contrib` and
dropped when the parent consumes it. No pool means no slack to
improve.

## The genuine fill-prediction lever would be a pool

SSIDS's approach works because it uses a **single large allocation**
sized to `factor_slack × factor_nnz`, and sub-allocates supernode
frontal L chunks out of it. Delay overruns force a reallocation of
the whole buffer. Dynamic slack tracking reduces those reallocations.

FERAL would need to:
1. Introduce a `FactorPool` / `Vec<f64>` arena owned by
   `FactorWorkspace` or `SparseFactors`.
2. Sub-allocate each supernode's `L`/`D` as a sliced view into
   the arena.
3. Have a grow-on-overrun policy whose heuristic is `factor_slack`.

That is a significant refactor (touches the `FrontalFactors`
ownership model in the dense kernel) and is Phase 2.7 / Phase 3
material, not a 2-to-4-hour Phase 2.5.4 tweak.

## Decision

Close 2.5.4 as documented-vacuous. The named lever
(`factor_slack` 1.2 → adaptive) has no hot-path consumer in the
current codebase. Keep `factor_slack` at 1.2 — it is correct
for the one consumer (bench metadata reporting) and a dynamic
adjustment would add drift without runtime benefit.

The real fill-prediction lever — a pooled factor arena — is a
Phase 3 item depending on a redesign of how `FrontalFactors`
owns its numeric storage. Record the deferral; do not implement
the adaptive-slack tweak in isolation.

## References

- `src/symbolic/mod.rs:127-130,349-357` (fields, default 1.2)
- `src/bin/bench_orderings.rs:47` (ordering-bench consumer)
- `dev/plans/phase-2-planning.md:523-529` (original 2.5.4 spec)
- `src/numeric/factorize.rs:140-175` (FactorWorkspace — no
  factor-nnz-sized field)
