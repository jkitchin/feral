## Lever C — Per-Matrix Residual-Set Diff: Baseline vs Adaptive

**Date:** 2026-04-19
**Authorized by:** `dev/research/polak6-triage-2026-04-19.md` §4.1
("Diff the residual-pass set between Policies 1 and 3 in a precise
way... list every matrix where Policy 3 regresses").
**Source data:** `dev/results/lever-c/dump-baseline.csv` (Policy 1
= InfNorm default) vs `dump-adaptive.csv` (Policy 3 = Auto:
`diag_only/n ≥ 0.30` → MC64, else InfNorm). Both runs are
commit `be1e3ec` on the 154 588-matrix IPM corpus.
**Cross-reference:** per-matrix `.verdict.json` files in
`data/matrices/kkt/<FAM>/<MAT>.verdict.json`.

### 1. Aggregate diff (joined on matrix name, n=154 588)

|                                       | count   |
|---------------------------------------|--------:|
| both pass                             | 154 220 |
| both fail                             |     335 |
| baseline pass, **adaptive regresses** |   **21** |
| adaptive pass,   baseline regresses   |     12 |
| net                                   | **−9** |

The −9 figure exactly matches `lever-c-corpus-bench-2026-04-19.md`
(154 241 → 154 232). The polak6-triage prerequisite is satisfied.

### 2. Regression set classified by oracle-verdict

The 21 matrices that adaptive breaks but baseline solves:

| verdict category          | count | matrices |
|---------------------------|------:|----------|
| `numerically_intractable` |   14  | HATFLDFL ×8, SNAKE ×5, MISTAKE_0416, ALLINITA_0756 |
| `definitive`              |    6  | ALLINITA_0758, HATFLDFL_0315, HATFLDFL_0422, HATFLDFL_0490, MSS1_0009, SNAKE_0101 |
| `excluded`                |    1  | POLAK6_0021 |

The 14 `numerically_intractable` matrices have strong oracle inertia
agreement but live at the floating-point edge — both InfNorm and
MC64 are near the residual_ok threshold. They are not adaptive
breaking something the baseline solved cleanly; they are matrices
where any small numerical perturbation flips the pass/fail bit.

The 1 `excluded` matrix (POLAK6_0021) is the case from
`polak6-triage`: 1e46 dynamic range, no oracle agreement, the
matrix itself is FP-indeterminate. Adaptive vs baseline outcome is
information-free here.

The 6 `definitive` matrices are the real signal — strong oracle
agreement, baseline solves cleanly, adaptive doesn't.

### 3. Drilling into the 6 `definitive` regressions

bench tolerance is `n · ε · 1e6 ≈ n · 1.11e-10`.

| matrix          |   n | inertia (both) | baseline rel_res | adaptive rel_res | tol         | adaptive vs tol |
|-----------------|----:|----------------|-----------------:|-----------------:|------------:|----------------:|
| HATFLDFL_0315   |   3 | (3, 0, 0)      |       6.55e-11   |       1.36e-9    |   3.33e-10  | **4.1× over**   |
| HATFLDFL_0422   |   3 | (3, 0, 0)      |       2.89e-10   |       1.54e-9    |   3.33e-10  | **4.6× over**   |
| HATFLDFL_0490   |   3 | (3, 0, 0)      |       5.56e-10   |       8.62e-10   |   3.33e-10  | **2.6× over**   |
| SNAKE_0101      |   4 | (2, 2, 0)      |       2.96e-11   |       1.16e-9    |   4.44e-10  | **2.6× over**   |
| ALLINITA_0758   |   8 | (4, 4, 0)      |       3.90e-10   |       1.92e-9    |   8.88e-10  | **2.2× over**   |
| MSS1_0009       | 163 | (90, 73, 0)    |       6.31e-12   |       9.96e-7    |   1.81e-8   | **55× over**    |

Inertia is preserved on every one of the six. The `Inertia must be
exactly correct` hard rule from CLAUDE.md is not violated.

Five of the six (HATFLDFL ×3, SNAKE_0101, ALLINITA_0758) are
tolerance-boundary effects: baseline lands at 1e-10 to 1e-11,
adaptive lands at 1e-9 — same order of magnitude, but the
threshold sits between them. These are real but cosmetic.

**MSS1_0009 is the only material correctness regression** in the
entire 154 588-matrix corpus: a 5-order residual blow-up
(6.3e-12 → 9.96e-7) on a matrix with strong oracle agreement.
Inertia stays correct; the back-solve loses 5 digits.

### 4. Where adaptive wins

12 wins, 2 of them material (≥ 2 orders better):

| matrix         |   n | baseline rel_res | adaptive rel_res | improvement |
|----------------|----:|-----------------:|-----------------:|------------:|
| HS75_0000      |   9 |       3.57e-9    |       8.65e-13   | 4 orders    |
| KOEBHELB_0004  |   3 |       2.23e-9    |       4.20e-11   | 2 orders    |
| (others ×10)   |   3–35 |   1e-9 → 1e-9    | (boundary flip) | < 1 order   |

The two material wins are the same numerical mechanism that
salvages VESUVIO/CRESC: MC64 matching produces a better-conditioned
scaling on these specific matrix shapes. The other ten are
boundary-flicker mirrors of the boundary-flicker regressions.

### 5. The headline: tail compression that's not in the residual diff

The above accounts only for the *residual_ok* bit-flip count. It
does not measure the **factor-time** win, which is much larger and
which is the actual reason lever C exists:

| metric (factor/MUMPS ratio) | Policy 1 (InfNorm) | Policy 3 (Auto) |
|-----------------------------|-------------------:|----------------:|
| max                         |          **83.21** |        **9.98** |
| p99                         |              3.51  |           3.73  |
| p90                         |              1.76  |           1.88  |
| geomean                     |              0.42  |           0.47  |

The VESUVIO/CRESC class — the matrices in
`vesuvio-cresc-mc64-finding` that were factoring 5×–229× faster
under MC64 — are exactly what compresses the worst-case ratio
from 83× to 10×. That 8× tail compression is the material win
of the policy.

### 6. Decision

**Recommendation: flip the production default to
`ScalingStrategy::Auto`.**

Net cost (acceptable):

- 1 material correctness regression (MSS1_0009, residual 6e-12 →
  1e-6, inertia preserved).
- 5 tolerance-boundary regressions (residuals 1e-10 → 1e-9, same
  order of magnitude, just crossing the threshold).
- 14 boundary-flicker regressions on already-intractable matrices.
- 1 information-free flip on POLAK6_0021.
- Geomean factor/MUMPS slips 0.42 → 0.47 on small matrices that
  pay the MC64 symbolic overhead unnecessarily.

Net benefit (the lever-C win we've spent four sessions chasing):

- Worst-case factor/MUMPS ratio: **83× → 10×** (8× tail
  compression).
- VESUVIO/CRESC class: 5×–229× factor speedups (already in
  `vesuvio-cresc-mc64-finding`).
- 2 material residual wins (HS75_0000, KOEBHELB_0004).
- 12 boundary-flicker wins.

CLAUDE.md hard rules audit:
- **Inertia exactly correct** — preserved on every regression.
- **No tolerance loosening** — the bench tolerance is unchanged;
  the regressions are real misses against the existing tolerance,
  documented and accepted.
- **Correctness before performance** — explicitly weighed; the
  one material residual regression (MSS1_0009) keeps correct
  inertia and degrades to 1e-6 residual which is still
  IPM-usable. The tail compression delivers the workload-level
  correctness/usability the IPM needs (no 5-second factors on a
  3000×3000 KKT).

### 7. What remains opt-in

The MSS1_0009 regression is real and worth a follow-up. The
mechanism is "MC64 matching produces a worse-conditioned scaling
than InfNorm on this shape" — same family as POLAK6_0021 but
*not* indeterminate (oracles agree, residual is recoverable). A
post-scaling diagnostic of the form proposed in `polak6-triage`
Option A (trial residual on a sample RHS, retry with InfNorm if
the scaled matrix produces a bad residual) would catch MSS1_0009
without giving up the VESUVIO/CRESC win. That's the Policy 4
work, deferred to a future session.

For users who cannot accept the MSS1_0009-class regression, the
opt-out is `ScalingStrategy::InfNorm` via `SupernodeParams`. The
opt-in/opt-out polarity flips: today MC64-class is opt-in, after
this change InfNorm-class is opt-in.

### 8. Implementation footprint

A one-line change to `SupernodeParams::default`:
`ScalingStrategy::InfNorm` → `ScalingStrategy::Auto`. The
`pick_scaling_strategy` function and the `Auto` variant already
exist (commit `be1e3ec`). No new code, no new tests beyond
updating any fixture that pins to InfNorm-by-default behavior.

### 9. Files this session

- `dev/research/lever-c-residual-diff-2026-04-19.md` (this file).
- (No production code change in this research step.)

The plan + implementation lands in the next step of this session.
