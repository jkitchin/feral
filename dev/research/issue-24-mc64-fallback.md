# Issue #24 — Surfacing MC64 → InfNorm silent fallback

Date: 2026-05-16
Author: agent session 2026-05-16-NN
Issue: https://github.com/jkitchin/feral/issues/24 (M2 of `dev/plans/robustness-roadmap.md`)

## Problem

`src/scaling/mod.rs::compute_scaling_auto_with_cache` silently swaps
the user-selected (or `Auto`-routed) MC64 matching-based scaling for
the cheaper InfNorm (Knight-Ruiz) scaling under two distinct conditions:

1. **Pre-MC64 InfNorm-spread guard** (lines 298-301). When
   `scaling_spread(in_vec) < IN_SPREAD_GUARD` (1e3), the routine
   commits to InfNorm without ever calling the Hungarian kernel.
2. **Post-MC64 Policy 4 ratio guard** (lines 311-332). MC64 runs,
   but if its scaled `max(|off|/|diag|)` exceeds `MC_OFF_GUARD = 1e6`
   AND the ratio to InfNorm exceeds `RATIO_GUARD = 1e5` AND raw
   `diag_range < RAW_GUARD`, the routine throws the MC64 vector
   away and uses InfNorm.

Both branches return `ScalingInfo::Applied` carried from
`infnorm::compute_infnorm`, indistinguishable from the case where
InfNorm was the user's explicit choice. There is also no signal on the
non-`Auto` path — but on that path the caller already knows they
asked for MC64 and InfNorm did not run, so the silent-fallback
concern is `Auto`-specific.

Symptoms (from issue #24):
- Subtle accuracy regressions that disappear after manually picking
  `ScalingStrategy::Mc64` (the regression IS the fallback firing).
- "Feral got the wrong answer" reports are undiagnosable because
  there is no trace of which scaling actually ran.

## MC64 failure modes

From the MC64 paper @duff2001mc64 and SSIDS commentary
(`ref/spral/src/scaling.f90:597-801`) the matching-based scaling
can fail in three structural ways:

1. **No perfect matching exists** (structurally singular). The
   Hungarian kernel returns `n_matched < n`, and we currently
   already surface this via `ScalingInfo::PartialSingular`.
2. **All entries of a column are zero or non-finite**, so
   `cmax[j] = -Inf` and the dual variables for that index are
   meaningless. `scaling_from_cache` falls back to `s[j] = 1.0`
   silently — the row is then identity-scaled, and at most we
   currently flag the matrix as `PartialSingular` if any column
   was actually unmatched. A matrix that is matched but has one
   all-zero column survives as `Applied`.
3. **Numerical breakdown / catastrophic mis-scaling**. The dual
   variables produce an `s` vector that, when applied, leaves
   the off-diagonal magnitudes orders of magnitude larger than
   the diagonal (the MSS1_0009 case: `max(|off|/|diag|) ≈ 7.8e14`
   vs InfNorm's `≈ 2.0e8`). The Policy 4 fallback was added in
   2026-04-XX to catch this; see
   `dev/research/policy-4-scaling-fallback.md`.

Failure modes 1 and 2 are already surfaced via `PartialSingular`.
Mode 3 — the Policy 4 fallback — is the gap this issue targets.
A fourth case (pre-MC64 InfNorm trial wins) is not strictly a
"failure" — MC64 never ran — but for `Auto` callers it is
operationally identical: "you asked for Auto, you got InfNorm,
MC64 was on the table but did not contribute."

## What signal to add

A new `ScalingInfo` variant, `Mc64FallbackToInfnorm`, with a
`reason` enum:

```rust
pub enum Mc64FallbackReason {
    /// `Auto` picked MC64 by shape, but the pre-MC64 InfNorm trial
    /// produced a tight scaling vector
    /// (`max|s|/min|s| < IN_SPREAD_GUARD`), so the matching was
    /// skipped — InfNorm was already good enough.
    InfNormSpreadAcceptable,
    /// MC64 ran but produced a scaled `max(|off|/|diag|)` far
    /// worse than InfNorm on a matrix where the raw |diag| range
    /// was small enough that MC64 had no room to recover. Policy 4
    /// ratio guard.
    Mc64WorseThanInfnorm,
}

pub enum ScalingInfo {
    Applied,
    PartialSingular { n_unmatched: usize },
    Mc64FallbackToInfnorm { reason: Mc64FallbackReason },
    NotApplied,
}
```

This is the minimum change that lets a caller distinguish the four
operational cases (clean MC64 / partial-singular / fell-back-to-InfNorm /
user said no scaling) without overloading existing variants.

Downstream surfacing:
1. `SparseFactors::scaling_info` already carries `ScalingInfo`, so the
   factor object preserves the new variant for `Solver` consumers.
2. Add `Solver::scaling_info()` accessor returning
   `Option<&ScalingInfo>`. (`provides_inertia`-style read-only window
   on the last factor's diagnostics.)
3. `bench_one_matrix` writes `mc64_fallback yes` whenever
   `scaling_info` is `Mc64FallbackToInfnorm { .. }`, and
   `mc64_fallback no` on every other status (so the field is always
   present, just like `refined yes`). Adding a `mc64_fallback_reason`
   key with the reason discriminant gives the sidecar reader enough
   to triage without re-running.

## Why this mechanism over alternatives

Alternatives considered and rejected:

- **`eprintln!` warning** inside `compute_scaling_auto_with_cache`.
  Rejected: the codebase only uses `eprintln!` in `src/bin/*`
  (diagnostic drivers), never in `src/` library code. Solver-library
  callers do not want stderr noise; the bench harness wants
  structured data, not log scraping. A grep confirms no
  `tracing` / `log` infrastructure exists in `src/`.
- **Add a `tracing` / `log` dependency**. Rejected per task
  guidance and CLAUDE.md: minimal-dependency philosophy, and a
  one-bit signal does not justify pulling a logging crate into a
  pure-Rust solver.
- **Extend `FeralError` with a non-fatal warning channel**.
  Rejected: `FeralError` is an error type, not a diagnostic stream.
  Returning a `Result<(T, Vec<Warning>), FeralError>` from every
  scaling/factor entry would be a wide-blast-radius signature
  change for one field.
- **`mc64_fallback: bool` counter on `Solver`** (as the issue
  suggests as `mc64_fallback` counter telemetry). Adopted as a
  *supplement* to the `ScalingInfo` variant, not a replacement —
  the variant is needed anyway so `SparseFactors` (which doesn't
  go through `Solver`) carries the information for direct
  `factorize_multifrontal_*` callers like `bench_one_matrix`.
  The `Solver::mc64_fallback_count()` accessor is the second
  surface, mirroring `Solver::symbolic_call_count()`'s style.

The `ScalingInfo` enum already exists and is the natural place
for this signal; adding a new variant is the smallest change that
threads through every existing pathway (`compute_scaling`,
`compute_scaling_dense_fast`, `compute_scaling_with_cache`,
`SparseFactors::scaling_info`) without touching their signatures.

## Test plan

1. Unit test in `src/scaling/mod.rs`: construct a matrix that
   triggers the `InfNormSpreadAcceptable` branch (a well-equilibrated
   arrow-KKT, where `pick_scaling_strategy` returns MC64 but
   InfNorm's spread is small). Assert
   `compute_scaling(_, Auto)` returns the new variant.
2. Unit test for the Policy 4 ratio-guard branch
   (`Mc64WorseThanInfnorm`) — gated on the MSS1_0009 fixture; skip
   if absent, mirroring the existing
   `auto_falls_back_to_infnorm_on_mss1_0009` test pattern.
3. Solver-level test: factor a fallback-triggering matrix and
   assert `Solver::scaling_info()` is the new variant and
   `Solver::mc64_fallback_count()` is 1.

## Scope guardrails

- Do not modify the MC64 algorithm or the fallback policy itself.
- Do not change the default `ScalingStrategy::Auto`.
- Do not add a logging crate.
- Do not break existing `ScalingInfo` consumers — every existing
  match arm continues to compile (the new variant means the
  three call sites in `factorize.rs` that match
  `PartialSingular` get an `_` wildcard for the new variant
  without behaviour change; this is acceptable because the
  fallback is *not* a singularity condition).

## References

- @duff2001mc64 — MC64 paper (canonical algorithm).
- @duff2005symmetric — symmetric averaging for LDLᵀ.
- `dev/research/policy-4-scaling-fallback.md` — the Mc64WorseThanInfnorm
  trigger.
- `dev/research/acopp30-plateau-2.md` — the IN_SPREAD_GUARD trigger.
- `dev/plans/robustness-roadmap.md` M2.
