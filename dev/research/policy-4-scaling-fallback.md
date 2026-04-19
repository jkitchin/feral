## Policy 4 — MC64-with-fallback for the MSS1_0009 case

**Date:** 2026-04-19
**Authorized by:** `dev/research/lever-c-residual-diff-2026-04-19.md`
§7 ("MSS1_0009 regression is real and worth a follow-up... A
post-scaling diagnostic of the form proposed in `polak6-triage`
Option A would catch MSS1_0009 without giving up the
VESUVIO/CRESC win").
**Diagnostic source:** `src/bin/policy4_diag.rs` (this session,
throwaway).
**Prior:** `dev/research/polak6-triage-2026-04-19.md` §3 enumerated
three Option candidates (A trial-residual, B raw-range pre-filter,
C accept loss). C is the current default. A vs B is the question
this note resolves.

### 1. Goal

Recover the MSS1_0009 residual regression (6e-12 → 1e-6 under
default `Auto`) without sacrificing the VESUVIO/CRESC factor-time
wins. Specifically: keep `pick_scaling_strategy` routing the
arrow-KKT class to MC64, but bail out to InfNorm on the `MSS1_0009`
failure mode.

### 2. Feature-table from `policy4_diag`

Format: `mc_off` and `in_off` are the maximum per-column
`|off-diagonal| / |diagonal|` ratio after applying MC64 vs InfNorm
scaling. `mc/in` is the multiplicative ratio.

| matrix         |   n | raw_drng | in_off    | mc_off    |   mc/in | label / outcome      |
|----------------|----:|---------:|----------:|----------:|--------:|----------------------|
| MSS1_0009      | 163 |  5.12e1  |  2.00e8   |  7.81e14  | 3.9e6   | **REGR-mat (target)** |
| HATFLDFL_0315  |   3 |  1.79e5  |  1.00     |  1.00     | 1.00    | regr-edge             |
| HATFLDFL_0422  |   3 |  7.81e5  |  1.00     |  1.00     | 1.00    | regr-edge             |
| HATFLDFL_0490  |   3 |  1.69e6  |  1.00     |  1.00     | 1.00    | regr-edge             |
| SNAKE_0101     |   4 |  4.71e18 |  2.62e3   |  2.58e3   | 0.98    | regr-edge             |
| ALLINITA_0758  |   8 |  1.94e25 |  1.00     |  1.00     | 1.00    | regr-edge             |
| POLAK6_0021    |   9 |  1.33e46 |  1.95e45  |  2.92e41  | 1.5e-4  | excluded (indet.)     |
| HS75_0000      |   9 |  1.00e11 |  1.00e10  |  1.08e9   | 0.108   | **WIN-mat** (4-order)|
| KOEBHELB_0004  |   3 |  1.78e7  |  1.01     |  1.00     | 0.99    | WIN-mat (2-order)     |
| VESUVIA_0000   |3083 |  4.73e14 |  1.22e11  |  4.84e12  | 39.7    | **WIN-perf**          |
| VESUVIO_0000   |3083 |  1.00e10 |  2.69e11  |  2.54e13  | 94.4    | WIN-perf              |
| VESUVIOU_0000  |3083 |  8.92e13 |  1.00e10  |  1.05e14  | 1.05e4  | WIN-perf              |
| MUONSINE_0000  |1537 |  1.14e10 |  7.12e9   |  4.36e13  | 6.1e3   | WIN-perf              |
| CRESC132_0000  |5314 |  1.00e11 |  ∞        |  ∞        | n/a     | WIN-perf              |

### 3. Option B (raw-range pre-filter) is dead

The `polak6-triage` Option B proposed: skip MC64 if
`range(|raw diag|) > 1e10`. The data refutes it:

- VESUVIA_0000:   raw_drng 4.73e14 → would be filtered out → loses the
  84× → 9.4× win.
- VESUVIOU_0000:  raw_drng 8.92e13 → filtered → loses the win.
- HS75_0000:      raw_drng 1.00e11 → filtered → loses the 4-order
  residual win.

Any threshold that catches MSS1_0009 (raw_drng=51) is below all
the wins — the pre-filter rule has exactly inverted polarity for
this corpus. **Option B is rejected.**

### 4. Option A — sub-variants

The principled answer is "measure the residual, don't predict it."
But the cheapest version of "measure" varies:

#### A.1 — post-scaling `off/diag` ratio comparison

Cheap diagnostic: compute both MC64 and InfNorm scalings, compare
their `off/diag` ratios, fall back to InfNorm when MC64 is
catastrophically worse.

Candidate rule: `mc_off > 1e6 ∧ mc_off / in_off > 1e5`.

Coverage on the 14-matrix validation panel:

| matrix         |  mc_off    | mc/in    | rule fires? | desired  | match? |
|----------------|-----------:|---------:|-------------|----------|:------:|
| MSS1_0009      |  7.81e14   | 3.9e6    | **yes**     | fallback | ✓      |
| 5× regr-edge   | ≤ 2.62e3   | ≈ 1      | no          | keep MC64 | ✓     |
| POLAK6_0021    | 2.92e41    | 1.5e-4   | no          | keep MC64 | ✓     |
| HS75_0000      | 1.08e9     | 0.108    | no          | keep MC64 | ✓     |
| KOEBHELB_0004  | 1.00       | 0.99     | no          | keep MC64 | ✓     |
| VESUVIA_0000   | 4.84e12    | 39.7     | no          | keep MC64 | ✓     |
| VESUVIO_0000   | 2.54e13    | 94.4     | no          | keep MC64 | ✓     |
| VESUVIOU_0000  | 1.05e14    | 1.05e4   | no          | keep MC64 | ✓     |
| MUONSINE_0000  | 4.36e13    | 6.1e3    | no          | keep MC64 | ✓     |
| CRESC132_0000  | ∞          | n/a      | no          | keep MC64 | ✓     |

The threshold has comfortable margin: the highest "keep MC64"
ratio is VESUVIOU at 1.05e4, the lowest "fallback" is MSS1 at
3.9e6 — **2.5 orders of magnitude separation.** A 1e5 ratio
threshold sits in the middle of the gap.

Cost: one extra `compute_scaling` call (InfNorm) + one
`scaled_diag_and_offratio` pass (O(nnz)) per Auto-routes-to-MC64
matrix. InfNorm is the cheap scaling (single pass equilibration),
typically faster than MC64 itself. Net overhead < 2× on the
arrow-KKT subset, ~0% elsewhere.

Risk: validated against 14 matrices. Generalization to the
~500-matrix arrow-KKT subset and the full IPM corpus needs a
post-implementation residual-diff confirmation. The rule is
single-matrix-derived; the threshold MAY need adjustment.

#### A.2 — trial-factor + trial-residual

Run MC64, factor, sample-solve a deterministic RHS, check
`||r|| / ||b||`. If above 1e-6, refactor with InfNorm.

Pros: failsafe, no threshold tuning, directly measures what we
care about.

Cons: the *factor* (not the scaling) is the expensive operation.
A retry doubles factor cost on the retry set. The retry set size
is bounded above by the count of Auto-routes-to-MC64 matrices that
have problems — but we don't know that count without running it,
and false positives on the trial-residual threshold are possible.

Cleanest place: integrate with `Solver::increase_quality()`'s
escalation state machine. The current `Baseline` →
`ScalingEnabled` → `PivotRaised` ladder doesn't have a
"scaling strategy alternative" rung; adding one fits the pattern.

### 5. Recommendation

Ship A.1 (cheap diagnostic) inline in `pick_scaling_strategy` →
new function `pick_scaling_strategy_with_fallback(matrix)` (or
fold the logic in). When `pick_scaling_strategy` would route to
MC64, run the diagnostic and fall back to InfNorm if the rule
fires. Keep `Auto` as the default behavior.

#### Implementation correction (added during landing)

The original 2-condition rule (`mc_off > 1e6 ∧ mc_off/in_off > 1e5`)
false-positives on MEYER3NE_*: those matrices have `raw_drng ≈ 1e19`
where MC64 is genuinely needed even though its scaled `mc_off` looks
catastrophic. The shipped rule adds a third condition:

> **`raw_diag_range(matrix) < 1e6`** — fall back only when the raw
> matrix's diagonal already spans a narrow range. If the raw range
> is wide, MC64's bad-looking `mc_off` is reflecting genuine
> ill-conditioning, not its own malfunction, and InfNorm would do
> no better.

Validated on a 17-matrix panel (the 14 above plus the 3 MEYER3NE
parity fixtures): fires only on MSS1_0009. See `policy4_diag` table.

A second bug found during landing: the fallback initially returned
`ScalingInfo::NotApplied` (from a misreading of the InfNorm
convention). InfNorm actually returns `Applied`; `NotApplied` makes
`numeric::solve` skip the pre/post scaling steps, so the InfNorm
vector was being computed but never applied. The MSS1_0007–0013 set
regressed to residual ≈ 2.4e-3 in the bench until the fix landed.
The shipped code forwards `ScalingInfo::Applied` from
`infnorm::compute_infnorm`, restoring the predicted +1
residual_pass result.

Why A.1 over A.2:

- **Zero factor-cost overhead.** Diagnostic runs in
  `compute_scaling`, before any factorization happens. The Solver's
  escalation ladder stays unchanged.
- **Composes with `Solver` cache.** Auto's resolved strategy is
  computed once (or memoized); the trial-residual variant would
  need to factor under each strategy speculatively, breaking the
  symbolic cache.
- **The data supports the threshold.** 2.5 orders of magnitude
  separation between target and nearest "keep MC64" matrix is a
  comfortable safety margin.
- **Cheap to revisit.** If a future matrix exposes the rule's
  limit, A.2 is still on the menu — A.1 doesn't preclude it.

If the corpus re-bench shows the rule misfiring (regressing
VESUVIO/CRESC), revert to A.2 in a follow-up.

### 6. Test fixtures

Unit tests for the new diagnostic:
- MSS1_0009 → routes to InfNorm (was MC64).
- VESUVIA_0000 → still routes to MC64.
- VESUVIOU_0000 → still routes to MC64.
- HS75_0000 → still routes to MC64 (the win).
- A small synthetic matrix with mc_off ratio just below 1e5 → keeps
  MC64. Just above → falls back. Catches threshold-edge regressions.

The matrices are in `data/matrices/kkt/`; the test reads them via
`read_mtx`.

### 7. Validation — corpus re-bench expectations

Predicted deltas vs current Auto baseline (residual_pass 154 232,
inertia 153 009, worst factor/MUMPS ≈ 10):

| metric                  | current Auto | A.1 prediction |
|-------------------------|-------------:|---------------:|
| sparse residual_pass    |      154 232 | **154 233 (+1)** |
| sparse inertia_match    |      153 009 |        153 009 |
| MSS1_0009 residual      |      9.96e-7 | **6.31e-12** (back to InfNorm baseline) |
| VESUVIA_0000 ratio      |        9.41× |          9.41× (unchanged)             |
| VESUVIOU_0000 perf      |          fast |           fast (unchanged)             |
| worst factor/MUMPS      |          ~10 |             ~10 |

The +1 residual_pass count is small but the value-per-bit-flip is
high: MSS1_0009 is the lone material residual regression in the
default flip.

### 8. What this note explicitly does not propose

- No change to `pick_scaling_strategy`'s routing threshold
  (`diag_only/n ≥ 0.30`). That threshold validates well.
- No trial-factor / trial-residual ladder in `Solver`. Reserved
  for the case where A.1 misfires.
- No new public `ScalingStrategy` variant. The new logic lives
  inside `Auto`'s resolver and is invisible to callers.
- No change to InfNorm's behavior on matrices `Auto` already
  routes to InfNorm. Only the `Auto → MC64` arm is affected.

### 9. Files this session

- `src/bin/policy4_diag.rs` (new, diagnostic only).
- `dev/research/policy-4-scaling-fallback.md` (this file).

The plan and implementation land in the next steps of this session.
