# Lever C — Corpus Bench Across Policies 1–3

**Date:** 2026-04-19
**Source plan:** `dev/plans/lever-c-adaptive-scaling.md`
**Source note:** `dev/research/lever-c-adaptive-scaling.md`
**Code:** commit `be1e3ec` (`ScalingStrategy::Auto` + `pick_scaling_strategy` + `FERAL_SCALING` env).
**Raw bench output:** `dev/results/lever-c/bench-{baseline,mc64,adaptive}.txt`.

## Procedure

Three back-to-back invocations of `cargo run --bin bench --release`,
identical except for the `FERAL_SCALING` env var:

| run     | env                       | scaling resolved per matrix |
|---------|---------------------------|-----------------------------|
| Policy 1 | (unset)                   | `InfNorm` (default)          |
| Policy 2 | `FERAL_SCALING=mc64`      | `Mc64Symmetric` blanket      |
| Policy 3 | `FERAL_SCALING=adaptive`  | `Auto` → MC64 if `diag_only/n ≥ 0.30`, else InfNorm |

Same corpus (154 588 IPM KKT matrices), same ordering dispatch, same
numeric pivot threshold, same hardware. No tolerance was changed.

## Headline numbers

`factor/MUMPS` ratio — same column semantics in all three runs:

| metric        | Policy 1 (InfNorm) | Policy 2 (MC64) | Policy 3 (Adaptive) |
|---------------|-------------------:|----------------:|--------------------:|
| count         |             153 560 |          153 560 |              153 560 |
| geomean       |               0.42 |            0.49 |                0.47 |
| p50           |               0.33 |            0.40 |                0.36 |
| p90           |               1.76 |            1.98 |                1.88 |
| p99           |               3.51 |            3.94 |                3.73 |
| **max**       |          **83.21** |       **10.10** |            **9.98** |

Sparse-path correctness counts (out of 154 588):

| metric                       | Policy 1 | Policy 2 | Policy 3 |
|------------------------------|---------:|---------:|---------:|
| inertia match vs MUMPS       |  153 008 |  153 007 |  153 009 |
| residual pass                |  154 241 |  154 225 |  154 232 |
| **worst sparse residual**    | **2.69e-4** | **1.31e13** | **1.31e13** |
| worst-residual matrix        | ERRINBAR_0824 | POLAK6_0021 | POLAK6_0021 |

## Reading the data

### Tail compression: real and large

The session-08 tail outliers (VESUVIOU 83×, VESUVIO 79×, MUONSINE
59×, CRESC132 43×) are **gone** from the top 10 under both Policy 2
and Policy 3. The new worst is `KIRBY2_0007` at 10.1× and 9.98×
respectively — a 8× compression of the worst-case ratio. This is
the lever-C win.

### Center-of-mass regression: real and small

Geomean factor/MUMPS slips 0.42 → 0.49 (Policy 2) / 0.47 (Policy 3).
This is the cost of the MC64 symbolic overhead landing on ~150 000
small matrices that don't need it. Policy 3 recovers some of that
because the adaptive routing keeps small InfNorm-friendly matrices
on the InfNorm path, but the heuristic does not catch every one of
them — e.g. small matrices with `diag_only/n ≥ 0.30` still pay the
MC64 cost.

### POLAK6_0021 — a hard correctness regression

Both Policy 2 and Policy 3 break correctness on `POLAK6_0021` (n=9):

```
expected inertia: (5, 4, 0)
   feral inertia: (3, 4, 2)        # 2 zeros — structural singularity
worst residual:   1.31e13          # vs InfNorm baseline 9.21e-17
```

Under InfNorm the same matrix factors to inertia `(5, 4, 0)` and
solves with residual ~1e-16. Under MC64 the matching produces a
scaling that puts the matrix into a singular-looking shape — the BK
factorization completes but with two zero pivots, and the back-solve
produces a 13-digit-too-large residual. Inertia is wrong by
construction; residual reflects that.

This is a concrete numerical regression, not a perf regression. It
overrides the headline tail-compression win. CLAUDE.md hard rule
"Inertia must be exactly correct — no tolerance on inertia counts"
applies: a default that loses inertia on a matrix the previous
default handled correctly is unacceptable, regardless of geomean.

The adaptive policy doesn't dodge this because POLAK6_0021's
shape (n=9, presumably ≥ 3 diag-only rows) trips the
`diag_only/n ≥ 0.30` heuristic and gets routed to MC64.

## Decision (per plan §"Validation procedure")

> Highest residual-pass + inertia-pass count wins; geomean is the
> tie-breaker. Tolerances are NOT loosened.

Counts:

- Inertia match: Policy 3 (153 009) > Policy 1 (153 008) > Policy 2 (153 007)
- Residual pass: Policy 1 (154 241) > Policy 3 (154 232) > Policy 2 (154 225)

Policy 1 (baseline) wins on residual pass. Policy 3 wins on inertia
match by **+1**. Neither margin is decisive on its own. The deciding
factor is the **worst-residual blow-up to 1.31e13** under both
Policy 2 and Policy 3 — that is a 17-orders-of-magnitude regression
on a matrix Policy 1 handles correctly.

**Recommendation: do NOT flip the production default.** Keep
`InfNorm` as the `SupernodeParams::default` scaling strategy.
`ScalingStrategy::Mc64Symmetric` and `ScalingStrategy::Auto` remain
opt-in via the user-facing API. Lever C is shipped as opt-in with
this measurement as the documentation; it is not a transparent win.

## What unblocks Policy 4 and a possible default flip

The data justifies pursuing Policy 4 (try-MC64-fallback-to-InfNorm)
in a future session, but only after a separate investigation into
**POLAK6_0021** (and any other matrix where MC64 silently produces a
worse-conditioned scaling than InfNorm). The hypothesis is:

1. MC64 matching can succeed but yield a scaling that is *worse-
   conditioned* than InfNorm on small matrices with strong
   diagonal-pivot structure.
2. The current `ScalingInfo::PartialSingular` fallback only catches
   the singular-matching case, not this "matched but bad" case.
3. A robust adaptive policy needs a *post-scaling* condition check —
   e.g. min-abs-diagonal of the scaled matrix, or a sample
   condition-number proxy — to fall back to InfNorm when MC64
   produces a numerically suspect result.

That investigation should produce a small triage binary
(`src/bin/polak6_diag.rs`) with the same shape as `vesuvio_diag.rs`
and a research note before any code change.

## Lever-C status after this session

- `ScalingStrategy::Auto` shipped, opt-in, documented.
- `pick_scaling_strategy(matrix)` shipped, public, unit-tested
  (5 tests).
- `FERAL_SCALING={infnorm,mc64,adaptive,identity}` shipped on the
  bench harness.
- Production default unchanged.
- Measurement note (this file) records why the default is unchanged.
- The next-session menu now contains a POLAK6 triage as the
  prerequisite for Policy 4.

## Files touched

- `dev/results/lever-c/bench-{baseline,mc64,adaptive}.txt` (new) —
  full bench output for each policy.
- `dev/research/lever-c-corpus-bench-2026-04-19.md` (this file).

No production code change in this measurement step.
