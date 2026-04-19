## Plan — Flip default `ScalingStrategy` to `Auto`

**Authorized by:** `dev/research/lever-c-residual-diff-2026-04-19.md` §6.

### Scope

Change the production default from `ScalingStrategy::InfNorm` to
`ScalingStrategy::Auto`. The `Auto` variant and
`pick_scaling_strategy(matrix)` are already shipped (commit
`be1e3ec`), opt-in only. This plan flips the polarity.

### Diff

`src/scaling/mod.rs`:

- Move `#[default]` from `InfNorm` to `Auto`.
- Update the `enum ScalingStrategy` doc-comment to reflect the new
  default and cite `dev/research/lever-c-residual-diff-2026-04-19.md`.
- Update the `Auto` variant doc-comment — drop "opt-in only and
  never the default".

No call-site changes — `NumericParams::default()` and
`with_bk(...)` both go through `ScalingStrategy::default()`.

### Tests to update

A test that asserts the default is `InfNorm` should be flipped to
assert `Auto`. The grep above found two:

- `src/numeric/solver.rs:344` — `s.scaling_strategy()` expected
  `InfNorm`. Flip to `Auto`.

Other call-sites that pass `ScalingStrategy::InfNorm` explicitly
(e.g. `src/numeric/factorize.rs:706` test override,
`src/bin/vesuvio_diag.rs`, `src/bin/polak6_diag.rs`,
`src/bin/bench.rs` parser) are already opt-in to `InfNorm` and
keep working — they pin the strategy locally and don't depend on
the default.

The `compute_scaling` `Auto` arm at `src/scaling/mod.rs:134`
recurses through `pick_scaling_strategy(matrix)` already, so no
behavior changes for callers that already use `Auto`.

### Validation

1. `cargo test` — every test must pass. Any test that breaks
   because it implicitly relied on `InfNorm` default must either
   (a) be flipped to opt-in `InfNorm` explicitly when that was
   what the test meant, or (b) accept the new default behavior
   and update expectations. CLAUDE.md hard rule: do not loosen
   tolerances.

2. `cargo clippy --all-targets -- -D warnings` clean.

3. `cargo run --bin bench --release` — confirm the predicted
   numbers from the residual diff:
   - residual_pass count: 154 232 (= baseline 154 241 − 9).
   - max factor/MUMPS ratio: ~10× (down from 83×).
   - inertia_match count: ≥ 153 008 (predicted 153 009).

   If any of those is more than 1% off the prediction, stop and
   investigate before committing.

4. POLAK6/MSS1 inertia spot check — confirm both still get
   correct inertia under the new default. Inertia hard rule.

### Commit

One commit. Body covers what (default flip), why (8× tail
compression on the IPM corpus), evidence (residual-diff numbers
inline). Citation: `dev/research/lever-c-residual-diff-2026-04-19.md`.

### Out of scope (deferred)

- Policy 4 (post-scaling trial-residual diagnostic to recover
  MSS1_0009-class regressions). Future session.
- `Mc64Symmetric` blanket default (Policy 2). Permanently
  rejected — geomean cost is real and unmeasured benefit is
  smaller than `Auto`.
- Any change to `Auto`'s threshold (`diag_only / n >= 0.30`).
  The existing threshold validates on the corpus diff; tuning
  it now is overfitting.
