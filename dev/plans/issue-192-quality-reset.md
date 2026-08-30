# Plan — issue #192: `Solver::reset_quality()`

Research note: `dev/research/issue-192-quality-reset.md`.

## Scope

Add a way to bound the lifetime of `increase_quality`'s escalation.
One new public method on `Solver`, its Python binding, and tests.

Out of scope: changing the escalation ladder itself (the rungs, the
`0.75` exponent, `pivtol_max`), and any scoped-guard API — see the
research note for why the guard is deferred rather than shipped.

## State to add

```rust
/// Factory values of the two parameters `increase_quality` mutates,
/// snapshotted on the transition out of `Baseline`.
struct QualityBaseline {
    scaling: ScalingStrategy,
    pivot_threshold: f64,
}
```

One new `Solver` field: `quality_baseline: Option<QualityBaseline>`,
`None` at construction and after every reset.

## Behaviour

| current level | `reset_quality()` | resulting level | returns |
|---|---|---|---|
| `Baseline` | no-op | `Baseline` | `false` |
| `ScalingEnabled` | restore `scaling` + `pivot_threshold` | `Baseline` | `true` |
| `PivotRaised` | restore `scaling` + `pivot_threshold` | `Baseline` | `true` |
| `Exhausted` | restore `scaling` + `pivot_threshold` | `Baseline` | `true` |

`increase_quality` gains one line: when it acts from `Baseline`, it
snapshots the two fields before mutating either.

Invariants the reset preserves (it touches nothing else, mirroring
what `increase_quality` leaves alone):

- `last_symbolic` / `symbolic_call_count` — the symbolic cache is
  scaling-invariant since the β refactor, so a reset must not force a
  re-analysis any more than an escalation does.
- `last_factors`, `last_inertia` — neither operation factors.
- `mc64_scaling_cache`, `ordering_escalated`, `auto_arm_latched`,
  `mc64_retry_not_adopted` — independent latches, not quality state.

## Tests (RED before implementation)

Unit tests in `src/numeric/solver.rs`, using the existing
`solver_with_scaling` helper. Oracle in every case is the parameter
value the *caller* supplied, not anything the implementation computes.

- `r1_reset_quality_at_baseline_is_noop` — fresh solver: returns
  `false`, level stays `Baseline`, both params unchanged.
- `r2_reset_quality_undoes_stage_one_scaling_flip` — Identity scaling,
  escalate (→ `ScalingEnabled`, InfNorm), reset: returns `true`,
  scaling is `Identity` again, level `Baseline`.
- `r3_reset_quality_undoes_pivot_ladder` — InfNorm scaling, escalate
  three times, reset: `pivot_threshold` back to the constructed `0.0`,
  level `Baseline`.
- `r4_reset_quality_from_exhausted_restarts_identical_ladder` —
  escalate to `Exhausted`, record the visited `(level, threshold)`
  sequence, reset, escalate again; the second sequence equals the
  first. Pins "reset + re-escalate == fresh solver".
- `r5_reset_quality_restores_caller_params_not_defaults` — construct
  with a deliberately non-default `pivot_threshold` (`0.25`), escalate,
  reset, assert `0.25`. Fails if the reset restores
  `NumericParams::default()`.
- `r6_reset_quality_preserves_builder_configured_scaling` — build with
  `.with_scaling(Identity)` *after* `with_params` carried a different
  strategy, escalate, reset, assert `Identity`. Pins the lazy snapshot
  against a construction-time one.

Integration test in `tests/pounce_interface.rs`:

- `i9_reset_quality_rebaselines_without_invalidating_symbolic` —
  factor, escalate twice, factor, reset, factor. Asserts the third
  factor still `Success`, `pivot_threshold()` is back at the `1e-8`
  MA27 baseline, `quality_level()` is `Baseline`, and
  `symbolic_call_count()` never moved off 1.

Python test in `python/tests/test_ipm.py` (or `test_basic.py`,
wherever `increase_quality` is exercised): escalate then
`reset_quality()`, assert the level code returns to `QUALITY_BASELINE`.

## Steps

1. Research note + this plan. *(done)*
2. Write the tests above; confirm they fail to compile / fail RED.
3. Implement `QualityBaseline`, the field, the snapshot in
   `increase_quality`, and `reset_quality`.
4. `cargo test`, `cargo fmt`, `cargo clippy --all-targets -D warnings`.
5. Python binding + Python test; build and run if maturin is available,
   otherwise note it as untested-here and rely on CI.
6. `CHANGELOG.md` (user-visible API addition), `decisions.md` (the
   deferral in `pounce-integration-interface.md` is now taken up).
7. Benchmark: no numeric path changes, so the exit partition should be
   unmoved. Run it anyway per the session protocol.
