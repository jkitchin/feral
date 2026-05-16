#+TITLE: ACOPP30 residual plateau (round 2): MC64 mis-routing
#+DATE: 2026-05-16

* Problem

Issue #23 (M2) originally tracked 8 ACOPP30 matrices stuck at
rel_res 1.9e-2..2.6e-2 after IR. The SSIDS scale-invariant 2×2
cancellation-aware determinant floor (already landed at
`src/dense/factor.rs:1737, :2505`, research note
`dev/research/ssids-scale-invariant-det-floor.md`) closed all 8.
A full 105-matrix sweep via `src/bin/diag_acopp30_residual.rs`
under default `Solver` then revealed 6 NEW plateau matrices on
iterations 59, 63, 64, 65, 66, 67:

| iter | rel_raw  | rel_ref  | inertia (feral)   | inertia (json) |
|------|----------|----------|-------------------|----------------|
| 0059 | 1.28e-02 | 1.88e-06 | (71, 137, **1**)  | (72, 137, 0)   |
| 0063 | 1.56e-01 | 1.56e-01 | (71, 137, **1**)  | (72, 137, 0)   |
| 0064 | 1.74e-01 | 1.74e-01 | (71, 137, **1**)  | (72, 137, 0)   |
| 0065 | 9.07e-02 | 9.07e-02 | (71, 137, **1**)  | (72, 137, 0)   |
| 0066 | 1.49e-01 | 1.49e-01 | (71, 137, **1**)  | (72, 137, 0)   |
| 0067 | 9.07e-02 | 9.07e-02 | (71, 137, **1**)  | (72, 137, 0)   |

All six report `min|D| = 0`: feral emits a literal zero pivot
(ForceAccept path), and IR cannot recover. External oracles:
MUMPS reaches ~1e-5, SSIDS reaches ~1e-13. `cond1(A) ≈ 3e16`.

* Hypothesis tree (and the answer)

Built `src/bin/probe_acopp30_64.rs` to sweep `NumericParams`
knobs on iter 0064:

| variant                | inertia     | pivot pattern | rel_ref  |
|------------------------|-------------|---------------|----------|
| A: default Auto        | (71,137,1)  | 115×1 + 47×2  | 1.74e-1  |
| B: pivot_thresh=0      | (71,137,1)  | 115×1 + 47×2  | 1.74e-1  |
| D: pivot_thresh=1e-10  | (71,137,1)  | 115×1 + 47×2  | 1.74e-1  |
| E: PerturbToEps 1e-8   | (72,137,0)  | 115×1 + 47×2  | 1.40e-1  |
| G: scaling=MC64        | (71,137,1)  | 115×1 + 47×2  | 1.74e-1  |
| **H: scaling=InfNorm** | (71,138,0)  | 101×1 + 54×2  | **1.43e-14** |
| **I: scaling=Identity**| (71,138,0)  | 101×1 + 54×2  | **1.12e-14** |
| J: MC64 + Perturb 1e-8 | (72,137,0)  | 115×1 + 47×2  | 1.40e-1  |

Same pivot pattern under A/B/D/G/J → same factor → same broken
solve. InfNorm and Identity produce a completely different pivot
pattern (101×1+54×2) and factor to working precision. PerturbToEps
masks the literal zero but doesn't fix the underlying L.

**Root cause:** `ScalingStrategy::Auto` routes to MC64 (matrix has
arrow-KKT shape, `diag_only/n >= 0.3`), and Policy 4's first guard
at `src/scaling/mod.rs:284` skips the diagnostic when
`raw_diag_range >= 1e6`. ACOPP30 has `raw_drng = 1.06e10`, so MC64
is used unconditionally. On this family MC64 produces a
catastrophic scaling: factor zero pivot, no recovery.

The existing Policy 4 fallback (`mc_off > 1e6 AND mc_off/in_off
> 1e5`) cannot catch this case because both `mc_off` and `in_off`
are `+∞` — both scalings produce some dead diagonals, so the
off-diagonal-ratio metric is uninformative.

* The right discriminator: `in_spread`

Extended probe (`src/bin/probe_scaling_policy4.rs`) measured
`max|s|/min|s|` of both scaling vectors on a 9-matrix panel:

| matrix         | raw_drng | mc_spread | in_spread | MC64 res | InfNorm res |
|----------------|---------:|----------:|----------:|---------:|------------:|
| ACOPP30_0064   |  1.06e10 |   3.62e08 |  **1.63** | 5.18e-4  | **1.75e-16** |
| MEYER3NE_0220  |  4.77e19 |   6.91e09 |   6.64e09 | 1.22e-16 | 1.22e-16    |
| MSS1_0009      |  5.12e01 |   3.91e06 |  **1.09** | 1.86e-15 | 1.58e-15    |
| VESUVIA_0000   |  4.73e14 |   1.95e10 |   7.60e07 | 1.16e-15 | 2.84e-16    |
| VESUVIO_0000   |  1.00e10 |   1.53e08 |   1.30e04 | 3.36e-16 | 4.89e-16    |
| VESUVIOU_0000  |  8.92e13 |   1.03e06 |   8.43e05 | 7.61e-16 | 1.90e-15    |
| HS75_0000      |  1.00e11 |   9.39e06 |     20.8  | 1.31e-16 | 4.20e-17    |
| CRESC132_0000  |  1.00e11 |   2.32e03 |     859   | 1.49e-16 | 1.08e-16    |
| MUONSINE_0000  |  1.14e10 |   6.19e03 |     99.6  | 8.95e-17 | 1.12e-16    |

**Signal:** when `in_spread < 1e3`, InfNorm has already nearly
equilibrated the matrix with one Knight-Ruiz pass, and MC64's
heavy matching is gratuitous (and on ACOPP30, catastrophic).

| in_spread bracket | matrices                | MC64 outcome   |
|-------------------|-------------------------|----------------|
| < 100             | ACOPP30, MSS1, HS75, MUONSINE | mixed (ACOPP30 BAD) |
| 100..10k          | CRESC132, VESUVIO       | tie            |
| > 10k             | VESUVIA, VESUVIOU, MEYER3NE | MC64 strictly wins |

A threshold of `in_spread < 1e3` catches the 6 ACOPP30 plateaus
without flipping any of the matrices Policy 4 was designed to keep
on MC64 (MEYER3NE 6.64e9, VESUVIA/VESUVIO/VESUVIOU all > 1e4).
MUONSINE/HS75 land in the "tie" range — switching is a residual
no-op (both achieve 1e-16); the only cost is losing whatever MC64
delay-pivot speedup may exist on those matrices.

* Validation

`DIAG_SCALING=infnorm cargo run --release --bin diag_acopp30_residual -- --all`
reports **105/105 ACOPP30 matrices passing rel_ref < 1e-10**
(strictly better than the current Auto default at 99/105).

* Proposed fix

In `src/scaling/mod.rs::compute_scaling_auto_with_cache`, add a
pre-MC64 InfNorm trial:

  - Run InfNorm Knight-Ruiz first (cheap; the existing diagnostic
    path already computes it conditionally).
  - If `max|s|/min|s| < IN_SPREAD_GUARD (1e3)`, return InfNorm
    immediately, skipping MC64 entirely.
  - Otherwise, fall through to existing logic
    (`raw_drng >= RAW_GUARD → MC64 unconditionally`, else run the
    `mc_off / in_off` ratio test).

Cost: one extra InfNorm computation on the matrices that today hit
the `raw_drng >= 1e6` fast-MC64 path. On the validation panel this
saves an MC64 call on ACOPP30 and pays one InfNorm on
MEYER3NE/VESUVIA/etc — Knight-Ruiz on KKT-scale matrices is
typically faster than MC64 matching, so net likely neutral or
positive even before the correctness win.

Alternative considered: raise `RAW_GUARD` from `1e6` to `1e12`.
Would route ACOPP30 (raw_drng=1.06e10) through the diagnostic,
but the diagnostic's `mc_off/in_off` ratio test breaks on inf/inf
and would not catch ACOPP30 even then. Rejected.

Alternative considered: PerturbToEps with `abs_floor=1e-8`. Masks
the literal zero pivot but does not change the broken pivot
pattern; residual stays at 1.4e-1. Rejected.

Alternative considered: residual oracle (factor under both, pick
the better). Doubles factor cost on every `Auto` invocation.
Rejected as too expensive.

* References

- `dev/research/lever-c-residual-diff-2026-04-19.md` — origin of
  Policy 4 (the existing `mc_off/in_off` ratio fallback).
- `dev/research/lever-c-adaptive-scaling.md` — origin of the
  `pick_scaling_strategy` arrow-KKT heuristic.
- `dev/research/policy-4-scaling-fallback.md` — 17-matrix
  validation panel underlying the current `RAW_GUARD`,
  `MC_OFF_GUARD`, `RATIO_GUARD` constants.
- `dev/research/task-19-dense-acopp30-expert-consultation.md` —
  expert consultation that flagged MC64 matching scaling as the
  Phase 2.4 Step 3 path for the original 8 suspects (now obsolete
  since the det floor closed them).
- `dev/research/ssids-scale-invariant-det-floor.md` — the change
  that closed the original 8 ACOPP30 plateaus.
