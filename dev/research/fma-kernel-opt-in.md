# Research note: FMA-enabled Schur kernel as an opt-in path (issue #8)

**Status:** research-phase note per CLAUDE.md feature lifecycle. No code yet.

**Motivation issue:** [#8](https://github.com/jkitchin/feral/issues/8) — sequential factor on `pinene_3200` (n=64k optimal-control NLP) does not complete in 600s, vs ipopt+MA57 finishing the full IPM in 2.6s. Profile attached on the issue shows 60%+ of CPU in `axpy2_minus_unroll4_nofma` (the **deliberately non-FMA** trailing-update kernel). The `_nofma` design exists for cross-arch bit-exactness; the cost is documented as ~2× the FMA-using variant.

**Goal:** add an opt-in FMA path that closes the per-kernel throughput gap on banded optimal-control problems, while keeping the bit-exact path as the default so existing users with bit-identity invariants are unaffected.

---

## Problem statement

The hot inner kernels of the multifrontal trailing update — `axpy_minus_unroll4_nofma`, `axpy2_minus_unroll4_nofma`, and the panel-strided variants `schur_panel_minus_nofma_strided{,_dual,_quad}` — are written with explicit `mul + sub` (or `mul + add + sub` for rank-2) instead of `mul_add`. The comment block at `src/dense/schur_kernel.rs:498-518` records the rationale: cross-arch bit-exactness with the scalar reference `naive_axpy_minus` / `naive_axpy2_minus`, because FMA's single-rounding step is one ULP off from separate-multiply-then-subtract per multiply-accumulate.

This invariant is paid for in throughput. On Apple M-series NEON the cost is two pipe slots per element instead of one — the documented ~2× — and on x86_64 V3 (AVX2+FMA) similar. For optimal-control problems with banded KKT Jacobians and wide trailing updates (pinene, marine, gasoil, robot), this 2× factor is the difference between completing a factor in seconds vs minutes.

Several FERAL users (notably the IPM stack consuming feral via crates.io) do not need cross-arch bit-exactness — they need to converge a Newton step. The opt-in lets them trade the invariant for throughput.

## Current architecture

### Kernel surface (`src/dense/schur_kernel.rs`)

**Production-wired non-FMA kernels:**
- `axpy_minus_unroll4_nofma` (line 541): rank-1 trailing column update. Cross-arch via `dispatch_nofma` helper at line 520 (aarch64 NEON or x86 V3 pulp).
- `axpy2_minus_unroll4_nofma` (line 600): rank-2 trailing column update.
- `schur_panel_minus_nofma_strided` (line 738): batched n_elim-fused rank-1 column update over a panel. Issued for the trailing-update phase of blocked LDLᵀ.
- `schur_panel_minus_nofma_strided_dual` (line 946): processes two destination columns per pulp dispatch.
- `schur_panel_minus_nofma_strided_quad` (line 1230): processes four destination columns per pulp dispatch — the workhorse on wide trailing updates.

**FMA-using siblings (currently dead code):**
- `axpy_minus_unroll4` (line 320): rank-1, uses `simd.mul_add_f64s(neg_a, s, d)`. **Gated `#[cfg(target_arch = "aarch64")]`** — only built on ARM right now.
- `axpy2_minus_unroll4` (line 392): rank-2, FMA. Also aarch64-only.

No FMA variant exists for any of the `schur_panel_minus_*_strided*` family. Those need to be written.

### How non-FMA kernels are called

Call sites (verified by grep on the production tree):
- `src/dense/factor.rs:1813` — `axpy2_minus_unroll4_nofma` (eager rank-2 update path)
- `src/dense/factor.rs:1835` — `axpy_minus_unroll4_nofma` (eager rank-1 update path)
- `src/dense/factor.rs:1918, 1937` — same two, inside `lblt_panel_frontal`
- `src/dense/factor.rs:2036, 2068, 2094` — `schur_panel_minus_nofma_strided_{quad,dual,base}` dispatch chain (the "apply blocked schur panel" routine)
- `src/dense/factor.rs:2666, 2705` — final fallback axpys
- `src/dense/block_ldlt32.rs:127, 154, 175, 222` — 32×32 register-resident driver (issue #9 land)

Every call site is in either `src/dense/factor.rs` or `src/dense/block_ldlt32.rs`. All are reached through `NumericParams`-aware entry points (`factor_frontal_blocked_in_place_with_scratch` and its small-leaf cousins).

### API surface

`NumericParams` (`src/numeric/factorize.rs:27-58`) is the right home for the flag. It already carries:
- `bk: BunchKaufmanParams`
- `scaling: ScalingStrategy`
- `small_leaf: SmallLeafBatch` (Off/On enum)
- `profiler: Option<Arc<Mutex<Profiler>>>`
- `parallel_telemetry: Option<Arc<AtomicLockStats>>`

`Solver` (`src/numeric/solver.rs:133+`) already has `with_parallel(bool)` builder — mirror this pattern.

The C ABI (`src/capi.rs`) currently has no surface for `NumericParams` tweaks. Setting the FMA flag through the C shim would need a new `feral_set_fma(s: *mut FeralSolver, enabled: i32) -> i32` entry point — straightforward; postpone until v0.4 if the Rust API is enough for the first cut.

## Design space

### Option A: Bool flag (`fma: bool`)

Add `pub fma: bool` to `NumericParams`. Each call site does:

```rust
if params.fma {
    schur_kernel::axpy_minus_unroll4(dst, src, alpha);
} else {
    schur_kernel::axpy_minus_unroll4_nofma(dst, src, alpha);
}
```

**Pros:** simplest. One branch per call site, predicted away after the first dispatch. Zero overhead when off.
**Cons:** the choice space might grow (e.g., a future "MA57-style block-sized panels" or "use crates.io BLAS" option that we explicitly reject under the pure-Rust constraint). Bool doesn't scale.

### Option B: Enum policy (`FmaPolicy { BitExact, Fma }`)

Add `pub fma: FmaPolicy` (defaulting to `BitExact`). Same branch shape at each call site. More extensible.

**Pros:** future-proof. Matches the `SmallLeafBatch::{Off, On}` precedent on the same struct.
**Cons:** marginal extra code vs bool. None substantive.

### Option C: Function-pointer indirection

Store `axpy_minus_unroll4_ptr: fn(&mut [f64], &[f64], f64)` on the solver and switch via that. Avoids per-call branch.

**Pros:** clean dispatch.
**Cons:** function pointers defeat `#[inline(always)]` — the kernel body wouldn't inline into the call site, killing the very ILP we're trying to expose. Inlining matters more than the eliminated branch.

**Choice: Option B (enum policy).** Matches the existing `SmallLeafBatch` pattern, costs nothing extra, leaves room for future kernel variants.

## Concrete plan (research-phase, not yet a commitment)

### Phase 1 — kernel work

1. **Cross-arch the existing aarch64-only FMA kernels.** Replace the `#[cfg(target_arch = "aarch64")]` gate on `axpy_minus_unroll4` and `axpy2_minus_unroll4` with a `dispatch_fma` helper mirroring `dispatch_nofma` (V3 try_new on x86_64 with V2/scalar fallback; NEON baseline on aarch64; generic `pulp::Arch::new().dispatch` elsewhere). Internal tests sweep both paths.
2. **Write the FMA siblings of the panel kernels.** Three new functions:
   - `schur_panel_minus_fma_strided`
   - `schur_panel_minus_fma_strided_dual`
   - `schur_panel_minus_fma_strided_quad`
   These mirror the `_nofma` bodies line-for-line, swapping `simd.sub_f64s(d, simd.mul_f64s(a, s))` for `simd.mul_add_f64s(-a, s, d)`. (Subtle: `mul_add(-a, s, d) == d - a*s` in one rounding step, matching the contract.)
3. **Internal bit-correctness gates for the FMA path.** Two test classes:
   - **Within-FMA bit-exactness:** the `_dual` and `_quad` kernels must produce the same result as `n_elim`/`n_col` sequential calls to the FMA rank-1 kernel. The existing `_nofma` tests at `schur_kernel.rs:1864+` (rank-2 ↔ two rank-1 calls) and `:2100+` (quad ↔ four sequential single-column calls) give the template — copy them with FMA names.
   - **Cross-policy tolerance:** FMA path vs `_nofma` path should agree to within `n * 1 ULP` where `n` is the number of multiply-accumulates per element. A loose absolute tolerance test (e.g., `|fma - nofma| < n * f64::EPSILON * |result|`) catches algorithmic divergence without false-positiving on the 1-ULP FMA rounding.

### Phase 2 — plumbing

1. **`NumericParams::fma` field.** Add `pub fma: FmaPolicy` with default `BitExact`. Define `pub enum FmaPolicy { #[default] BitExact, Fma }` next to `SmallLeafBatch`.
2. **`Solver::with_fma(policy)` builder.** Mirror `with_parallel` exactly. Passes through to `numeric_params.fma`.
3. **Dispatch shims.** Add `axpy_minus_unroll4_dispatch(dst, src, alpha, policy)` and three more, that match on `policy` and call the right kernel. Replace every production call site in `factor.rs` and `block_ldlt32.rs` with the dispatch shim. The branch is hoistable above tight loops; the shim is `#[inline(always)]`.
4. **(Deferred to v0.4)** C ABI surface: `feral_set_fma(s, enabled)`. Not blocking the v0.3.1 cut.

### Phase 3 — verification

1. **Inertia gate.** Run the full 153k-matrix corpus with `--fma=on`. Acceptance: at most a handful of borderline-zero-pivot matrices may shift inertia by ±1 negative eigenvalue; document each. If any matrix loses correctness (residual blowup, or inertia gap >1), the FMA path is wrong, not the data.
2. **Residual gate.** Same corpus, residuals must remain within 10× of the BitExact path (FMA's single-rounding is slightly *better* than two-rounding on this metric, not worse — should pass trivially).
3. **Performance gate on Mittelmann targets.** The acceptance criterion *for the issue*:
   - `pinene_3200` must complete a single factor in ≤ 5s (vs current >600s).
   - `marine_1600` must complete a full IPM solve in ≤ 30s (vs current 324.8s).
   - The KKT 153k-corpus bench: small-frontal p90 must improve (target ≤ 1.20; baseline 1.40); medium p90 must improve (target ≤ 1.60; baseline 1.88). PASS / FAIL gates same as Phase 2.8.1.
4. **NLP comparison re-run.** Re-run `external_benchmarks/nlp_comparison/run.py` with the FMA shim active for the feral build. Update REPORT.md.

### Phase 4 — release

1. CHANGELOG entry under Unreleased ("Added — opt-in FMA Schur kernel path").
2. `dev/decisions.md` entry: scope of bit-exactness invariant clarified — now a per-policy property, not a feral-wide invariant.
3. Bump 0.3.0 → 0.3.1 (semver patch — additive opt-in, default behavior unchanged).
4. Tag, push, re-archive on Zenodo.

## Effort estimate

| Phase | Work | Estimate |
|---|---|---|
| 1 | Three new panel kernels + cross-arch the existing two + internal tests | 1.5 days |
| 2 | Plumbing: NumericParams field, Solver builder, dispatch shims, call-site swaps | 1 day |
| 3 | Inertia gate + residual gate + Mittelmann verification + NLP rerun | 1 day |
| 4 | Changelog, decisions entry, version bump, release | 0.5 day |
| **Total** | | **~4 days, parallelizable to ~3** |

## Open questions

1. **Does the FMA path materially change inertia on borderline-zero-pivot matrices?** Expected to be rare (<10 matrices in 153k corpus) and to shift by at most 1 negative eigenvalue. Verified empirically in Phase 3; if it shifts more, the cross-policy tolerance is too loose and we need to tighten the zero-pivot threshold for the FMA path specifically.
2. **Should the policy be per-supernode or solver-wide?** Solver-wide for v0.3.1. A future per-supernode policy (e.g., "use FMA on supernodes with nrow > 128") could close more of the gap without touching the small-supernode invariant, but adds complexity. Defer.
3. **C ABI exposure.** The Ipopt shim currently has no path to set the FMA flag — a user wanting feral+FMA from C++ has to wait for `feral_set_fma`. Decision: defer to v0.4 unless a downstream user blocks on it. The Rust users (pounce) get the win immediately.
4. **Should the default flip to FMA in a future major bump?** Plausibly. If field experience shows the bit-exactness invariant is rarely depended on, v1.0 could default to FMA with `BitExact` as the opt-in. Not for v0.3.1.

## References

- Issue [#8](https://github.com/jkitchin/feral/issues/8) — the motivating profile + ask.
- `src/dense/schur_kernel.rs:498-518` — bit-exactness invariant statement.
- `src/dense/schur_kernel.rs:320-388` — existing `axpy_minus_unroll4` (FMA, aarch64-only, dead code).
- `src/dense/schur_kernel.rs:541-597` — existing `axpy_minus_unroll4_nofma` (production).
- `src/dense/schur_kernel.rs:1864+` — rank-2 ↔ rank-1 bit-exactness test template.
- `src/dense/schur_kernel.rs:2100+` — quad ↔ sequential single-column bit-exactness test template.
- `src/numeric/factorize.rs:27-58` — `NumericParams`.
- `src/numeric/solver.rs:133-166` — `Solver` builder pattern (`with_parallel` as the precedent).
- `dev/decisions.md` 2026-04-14 (Phase 2.4.2 entry) — original `_nofma` decision.
- `dev/tried-and-rejected.md` Phase 2.4.2 — bench data behind the ~2× cost claim.
