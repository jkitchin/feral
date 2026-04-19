## Plan — Policy 4: MC64 → InfNorm fallback inside `Auto`

**Authorized by:** `dev/research/policy-4-scaling-fallback.md` §5.

### Scope

Inside `compute_scaling(matrix, &ScalingStrategy::Auto)`, when
`pick_scaling_strategy(matrix)` returns `Mc64Symmetric`, run a
cheap diagnostic and fall back to `InfNorm` if MC64 produces a
catastrophically worse-conditioned scaling.

This change is invisible to callers: `Auto` still picks the right
scaling, `pick_scaling_strategy` is still public, no new
`ScalingStrategy` variant.

### Design

**Diagnostic** (in `src/scaling/mod.rs`):

```rust
/// Returns `mc_off / in_off` and `mc_off` after applying the two
/// candidate scalings. O(nnz) per scaling.
fn mc64_vs_infnorm_offratio(
    matrix: &CscMatrix,
    mc64_scaling: &[f64],
    infnorm_scaling: &[f64],
) -> (f64, f64) { ... }
```

**Routing rule** (the `Auto` arm of `compute_scaling`):

```rust
ScalingStrategy::Auto => {
    let picked = pick_scaling_strategy(matrix);
    if matches!(picked, ScalingStrategy::Mc64Symmetric) {
        let (mc, info_mc) = mc64::compute_symmetric(matrix)?;
        let (in_, _) = infnorm::compute_infnorm(matrix);
        let (ratio, mc_off) = mc64_vs_infnorm_offratio(matrix, &mc, &in_);
        const MC_OFF_GUARD: f64 = 1e6;
        const RATIO_GUARD: f64 = 1e5;
        if mc_off > MC_OFF_GUARD && ratio > RATIO_GUARD {
            // MC64 is catastrophically worse; fall back to InfNorm.
            Ok((in_, ScalingInfo::NotApplied))
        } else {
            Ok((mc, info_mc))
        }
    } else {
        compute_scaling(matrix, &picked)
    }
}
```

Notes:
- Wastes one `compute_infnorm` call per arrow-KKT matrix (~500
  matrices in corpus). InfNorm is cheap.
- Uses public-but-internal `infnorm::compute_infnorm` and
  `mc64::compute_symmetric` directly to avoid double-dispatch
  through `compute_scaling`. The recursive call we have today
  through `compute_scaling(matrix, &picked)` becomes a flat
  match.
- `ScalingInfo::NotApplied` for the fallback path matches the
  InfNorm convention (InfNorm doesn't report `Applied`).

### Tests

In `src/scaling/mod.rs` `#[cfg(test)] mod tests`:

1. `auto_falls_back_to_infnorm_on_mss1_0009` — read
   `data/matrices/kkt/MSS1/MSS1_0009.mtx`, call
   `compute_scaling(&csc, &Auto)`, factor with
   `factorize_multifrontal`, solve a deterministic RHS, assert
   residual ≤ 1e-9 (matches InfNorm baseline 6.3e-12).
2. `auto_keeps_mc64_on_vesuvia_0000` — read VESUVIA_0000, call
   `compute_scaling(&csc, &Auto)`, assert the resulting scaling
   vector has range > 1e6 (signature of MC64 having been used; an
   InfNorm scaling on this matrix has range ~1e8 too — see the
   diag table — so we use a different distinguishing test:
   factor + assert nelim count low or factor time low). Simpler
   alternative: factor + assert no delays.
3. `auto_keeps_mc64_on_vesuviou_0000` — same shape as test 2.
4. `auto_keeps_mc64_on_hs75_0000` — read HS75_0000, factor + solve,
   assert residual ≤ 1e-12 (the MC64 win).
5. `auto_falls_back_when_diagnostic_threshold_crossed` — small
   synthetic 4×4 indefinite matrix where we know MC64 produces a
   ruinous off-ratio. (May skip if hand-crafting one is non-
   trivial; the corpus tests above cover the production cases.)

CLAUDE.md hard rule: tests use the corpus matrices (external
oracles), not hand-rolled test matrices that I've also written.

### Validation

1. `cargo test` — all green, including the new tests.
2. `cargo clippy --lib --bins -- -D warnings` clean.
3. `cargo fmt --check` clean.
4. `cargo run --release --bin bench` — confirm:
   - sparse residual_pass: **154 233** (was 154 232).
   - sparse inertia_match: 153 009 (unchanged).
   - MSS1_0009 residual: ≤ 1e-9 (was 9.96e-7).
   - VESUVIA_0000 factor: ≤ 30 ms (was 24 ms; allow noise).
   - Worst factor/MUMPS: ≈ 10× (unchanged).
5. `cargo run --release --bin policy4_diag` — sanity-check the
   feature table is unchanged (the diagnostic is read-only).

### Commit

One commit. Body: what (Auto fallback), why (recover MSS1_0009),
evidence (predicted vs actual residual_pass count, MSS1 residual
recovery). Cite research note + plan.

### Out of scope

- Trial-factor + retry (Option A.2 from the research note).
- Threshold tuning if the rule misfires — fix in a follow-up
  commit, not pre-emptively.
- Public API change for the diagnostic (consumers want `Auto` to
  Just Work).
