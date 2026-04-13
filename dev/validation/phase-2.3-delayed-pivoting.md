# Phase 2.3 — Delayed Pivoting Validation Report

**Date:** 2026-04-13
**Head commit under test:** `8f3fce0` (Session 04 addendum: refinement-termination fix)
**Status:** WIN — parity panel 11/28 → **27/28**, worst sparse residual
`2.31e+11 → 2.50e-4` (**15 orders of magnitude**). The single
remaining ignored matrix (SSI_2597) is a pathological
factorization-level case, correctly deferred to Phase 2.4.

## Scope

Phase 2.3 set out to add SPRAL SSIDS-style delayed pivoting to
feral's multifrontal kernel so that rejected pivots in non-root
supernodes get a second chance at their parent, where child
contributions have been assembled and the block is more likely
to pivot cleanly. The plan had 10 steps; everything in
`dev/plans/phase-2.3-delayed-pivoting.md` landed, plus two
emergent fixes that fell out of the panel run and are
documented under the corresponding session checkpoints.

## Summary

| Arc stage                         | Parity | Sparse residual pass | Sparse inertia match | Worst sparse residual |    Commit |
| --------------------------------- | -----: | -------------------: | -------------------: | --------------------: | --------: |
| Phase 2.2 exit (pre-Phase-2.3)    |  11/28 |                    — |                    — |             `2.31e11` |         — |
| Session 02 — Steps 1–4 kernel     |  11/28 |               152571 |               149820 |             `2.31e11` | `bd5e0e2` |
| Session 03 — Steps 5–6 assembly   |  11/28 |               154113 |               152987 |              `1.19e0` | `28ff3b1` |
| Session 04 — Step 7 (u=0.01)      |  14/28 |               154137 |               152987 |              `7.06e2` | `6245952` |
| Session 04 — sign preservation    |  22/28 |               154237 |               153009 |            `3.22e-4` | `a630977` |
| Session 04 — refinement stop fix  |  27/28 |               154329 |               153009 |            `2.50e-4` | `ed07ee3` |

The 2 emergent fixes (sign preservation at root fallback and the
refinement-termination rewrite) were not in the original plan
but were necessary to close the parity panel. Both are
documented in `dev/decisions.md` and their rationale is in the
session 04 checkpoint and addendum.

## Parity panel — matrices flipped

16 matrices moved from `#[ignore]` to passing across Phase 2.3.
Grouped by which fix closed them:

### Closed by Step 7 (restore `pivot_threshold = 0.01` for sparse callers) — 3 matrices

| Matrix          |    n | Pre-2.3 failure mode                           |
| --------------- | ---: | ---------------------------------------------- |
| BQPGASIM_0012   |   50 | residual `4.77e+0` on non-delayed ForceAccept  |
| HATFLDBNE_2138  |    8 | residual `2.11e-1`                             |
| HATFLDBNE_2140  |    8 | residual `2.11e-1`                             |

(Other matrices in the panel that eventually passed also
benefited from `u=0.01`, but their primary close came from one
of the fixes below.)

### Closed by the sign-preservation fix (commit `a630977`) — 8 matrices

Symptom pre-fix: inertia mismatch of the form `(p, q-1, 1)` vs
MUMPS `(p, q, 0)`. The root-fallback ForceAccept branch was
converting small-but-nonzero pivots to zero, losing the sign
and producing one extra `inertia.zero` instead of an extra
`inertia.negative`.

| Matrix          |    n | Pre-fix inertia   | MUMPS inertia     |
| --------------- | ---: | ----------------- | ----------------- |
| CERI651A_0000   |  190 | (129, 60, 1)      | (129, 61, 0)      |
| CERI651A_0165   |  190 | (128, 61, 1)      | (129, 61, 0)      |
| CERI651A_0166   |  190 | (128, 61, 1)      | (129, 61, 0)      |
| DEGENLPA_0065   |   35 | (20, 14, 1)       | (20, 15, 0)       |
| DEGENLPB_0045   |   35 | (20, 14, 1)       | (20, 15, 0)       |
| DEGENLPB_0046   |   35 | (20, 14, 1)       | (20, 15, 0)       |
| DEGENLPB_0047   |   35 | (20, 14, 1)       | (20, 15, 0)       |
| PALMER2ANE_0000 |   74 | (52, 22, 1)       | (52, 23, 0)       |

### Closed by the refinement-termination fix (commit `ed07ee3`) — 5 matrices

Symptom pre-fix: feral residual within `1.1×–5.1e2×` of MUMPS's
residual, all above the `K=10` parity gate. Diagnosed via
`examples/triage_residual_margin.rs` as premature early-stop on
the `|dx|/|x|` threshold — under `ForceAccept` the trajectory
is non-monotone, corrections produce small `dx` without reducing
`r`, so the old termination was a false convergence signal.

| Matrix          |    n | Pre-fix residual | MUMPS residual | Ratio     |
| --------------- | ---: | ---------------: | -------------: | --------: |
| AVION2_0510     |  269 |         3.86e-14 |       2.54e-15 |    15.2×  |
| HAHN1_0004      |  736 |         3.23e-13 |       2.99e-14 |    10.8×  |
| MEYER3NE_0253   |   19 |         2.50e-14 |       1.99e-15 |    12.5×  |
| CERI651C_0746   |  190 |         4.29e-13 |       8.40e-16 |      510× |
| CERI651ELS_1482 |  375 |         1.28e-12 |       2.50e-15 |      512× |

CERI651C_0746's extended trajectory is the clearest evidence:

```
step 0:  4.29e-13
step 1:  4.29e-13
step 2:  4.29e-13
step 3:  4.29e-13  ← old loop exited here
step 4:  2.67e-17  ← machine-precision basin
step 5:  8.57e-13
```

The residual is flat for 4 steps (triggering the `|dx|/|x|`
stop) then drops 4 orders at step 4. Best-iterate tracking
captures it under the new criterion.

## Remaining frontier — SSI_2597

One matrix stays ignored: `parity_ssi_2597`.

- n=3
- Contains a denormal value `-2.96e-322`
- Dynamic range ~10³³¹
- Extended refinement trajectory is stuck at `1.80e-13`
  regardless of iteration count (10, 20, or more)
- MUMPS residual `1.15e-16`, ratio `1564×`

This is a factorization-level limitation, not a refinement
limitation. The factor is producing a `1.80e-13` residual as
its asymptotic floor; no amount of iterative refinement will
push it below that. Deferred to Phase 2.4 where the working
hypothesis is that a Knight-Ruiz-style equilibration or explicit
denormal handling in the kernel is needed to close the gap.

## dense_vs_sparse evidence

`cargo run --release --example dense_vs_sparse`. All four
matrices called out in Success Criterion 2 of the plan
(HYDCAR20, METHANL8, SWOPF, HATFLDG) match between the dense
and sparse paths and agree with MUMPS on inertia:

| Matrix                |    n | MUMPS inertia   | Feral sparse residual | Feral sparse inertia match |
| --------------------- | ---: | --------------- | --------------------: | -------------------------: |
| HYDCAR20_0000         |  198 | (99, 99, 0)     |              4.53e-15 |                        YES |
| METHANL8_0000         |   62 | (31, 31, 0)     |              3.78e-16 |                        YES |
| SWOPF_0000            |  175 | (83, 92, 0)     |              1.32e-15 |                        YES |
| HATFLDG_0005          |   50 | (25, 25, 0)     |              6.42e-16 |                        YES |
| HATFLDBNE_1586        |    8 | (4, 4, 0)       |              2.55e-16 |                        YES |
| **ACOPP30_0000**      |  209 | (71, 137, 1)    |          **4.27e-15** |              **YES (±0)** |
| CHWIRUT1_0000         |  645 | (431, 214, 0)   |              1.16e-14 |                        YES |
| CRESC100_0000         |  806 | (606, 200, 0)   |              8.39e-16 |                        YES |

Notable: **the sparse path now matches MUMPS on ACOPP30_0000**
with residual `4.27e-15`, while the *dense* path still reports
inertia `(72, 137, 0)` and residual `2.74e-2`. This is the
ACOPP30 regression recovery story from Phase 2.2.2 going one
step further under delayed pivoting — the sparse multifrontal
path now correctly identifies the singular column that the
dense `|d| < zero_tol` rejection misses.

CHWIRUT1, CRESC100, and CRESC132 (the Phase 2.2.2 regression
targets that Phase 2.2.2 left at plateaued residuals) all now
pass in the full sparse solver at machine precision. This
closes the gap that Phase 2.2.2's validation report explicitly
flagged as a Phase 2.3 deliverable.

## Bench deltas (full KKT corpus, 154588 matrices)

| Metric                     | Phase 2.2 baseline | Session 02 | Session 03 | Session 04 | Session 04 + addendum |
| -------------------------- | -----------------: | ---------: | ---------: | ---------: | --------------------: |
| Inertia match              |             149820 |     149820 |     152987 |     153009 |                153009 |
| Residual pass              |             152571 |     152571 |     154113 |     154237 |                154329 |
| Worst sparse residual      |           `2.31e11` |  `2.31e11` |    `1.19e0` |   `3.22e-4` |              `2.50e-4` |
| Dense KKT residual pass    |             154141 |     154141 |     154141 |     154141 |                154141 |
| Dense KKT worst residual   |           `2.80e-2` |  `2.80e-2` |  `2.80e-2` |  `2.80e-2` |              `2.80e-2` |
| Sparse-only failure count  |               3328 |       3328 |        203 |         64 |                    28 |

The dense path is unchanged across all of Phase 2.3 (as designed
— the `u = 0.01` flip is scoped to `params_kkt_sparse` and
`BunchKaufmanParams::default()` still has `pivot_threshold = 0.0`).
Every sparse improvement is due to the multifrontal kernel's
new delay capability, not a dense-path change leaking through.

## Success criteria check

| # | Criterion                                                        | Result |
| - | ---------------------------------------------------------------- | ------ |
| 1 | Parity count ≥ baseline + 7 matrices                             | 27/28 (+16) ✓ |
| 2 | `dense_vs_sparse` matches on HYDCAR20, METHANL8, SWOPF, HATFLDG   | ✓ |
| 3 | `tests/delayed_pivoting.rs` has 5 passing tests                   | 6 passing ✓ |
| 4 | `BunchKaufmanParams::default()` still has `pivot_threshold = 0.0` | ✓ |
| 5 | Sparse-path callers explicitly set `pivot_threshold = 0.01`       | ✓ |
| 6 | Full bench sparse residual pass rate increases                    | 154237 → 154329 (within phase); full-phase 152571 → 154329 ✓ |
| 7 | No existing passing test regresses                                | ✓ (full `cargo test --release` green) |
| 8 | Validation report committed                                       | this document |

All 8 criteria met.

## What was committed in Phase 2.3

Commits on `main`, earliest first:

- `bd1c6e4` — Phase 2.3 setup: research note + implementation plan
- `29ccf83` — Steps 1–3: delayed-pivoting kernel plumbing
- `7fb3779` — Step 4: wire `may_delay` through `factorize_multifrontal`
- `c84b7c9` — Step 4 fixup: revert `may_delay` flag flip
- `bd5e0e2` — Session 2026-04-13-02 checkpoint
- `0364e6d` — Steps 5+6: parent-side delay assembly + solve `nelim` fix
- `28ff3b1` — Session 2026-04-13-03 checkpoint
- `6245952` — Step 7: restore `pivot_threshold = 0.01` for sparse callers
- `a630977` — sign-preservation fix (emergent, root fallback)
- `97c598e` — Session 2026-04-13-04 checkpoint
- `ed07ee3` — refinement-termination fix (emergent, max_steps 3→10 + residual stop)
- `8f3fce0` — Session 2026-04-13-04 addendum
- (this commit) — Step 9 validation report

Files added/modified in `src/`:

- `src/numeric/factorize.rs` — `PivotOutcome::{Accepted, Rejected, Delayed}`,
  delayed-column bookkeeping, `may_delay = !is_root` flag propagation.
- `src/numeric/multifrontal.rs` — parent-side assembly of delayed
  columns from child frontal matrices.
- `src/numeric/solve.rs` — `nelim` used for the backward sub,
  refinement-termination rewrite (max_steps=10, residual-based stop).
- `src/dense/factor.rs::try_reject_1x1_frontal` — sign-preservation
  branch in `ForceAccept` at root fallback.
- `src/dense/solve.rs::solve_refined` — same refinement rewrite
  applied for consistency.

Files added/modified in tests:

- `tests/delayed_pivoting.rs` — 6 new unit and integration tests.
- `tests/pivot_rejection.rs` — updated for the sign-preservation
  semantics (old assertions about zero counts no longer apply).
- `tests/parity.rs` — 16 matrices moved from `#[ignore]` to
  passing; panel comment updated to 27/28.
- `examples/triage_residual_margin.rs` — new triage script used
  to diagnose the refinement early-stop bug.

Documents:

- `dev/research/delayed-pivoting.md` (Phase 2.3 Step 0)
- `dev/plans/phase-2.3-delayed-pivoting.md` (Phase 2.3 Step 0)
- `dev/sessions/2026-04-13-02.md`, `03.md`, `04.md` + addendum
- `dev/journal/2026-04-13-02.org`, `03.org`, `04.org`
- `dev/decisions.md` — two new entries (pivot-threshold split,
  root-fallback sign preservation)
- `CHANGELOG.md` — Phase 2.3 `Fixed` entries for delayed pivoting
  and refinement termination

## Open questions for Phase 2.4

1. **SSI_2597 denormal-value factorization floor.** Is the
   `1.80e-13` asymptote set by accumulation of denormal products
   in the kernel, or by a catastrophic cancellation that
   pre-scaling would fix? Plan:
   - Instrument the per-step frontal residual on SSI_2597.
   - Try Knight-Ruiz on the sparse path (currently default
     scaling only does symmetric inf-norm) and see if the floor
     moves.
   - If scaling does not help, consider a denormal-flushing
     pass or a single-column iterative refinement on the root
     front.

2. **Dense KKT ACOPP30 gap.** The sparse path now handles
   ACOPP30 at machine precision but dense plateaus at `2.74e-2`.
   The dense and sparse paths use the same pivot rejection
   logic; the gap must be in how the dense path's `ForceAccept`
   sequences interact with its non-delayed elimination. Not a
   Phase 2.3 deliverable but newly visible because the sparse
   side cleared its end of the gap.

3. **Is 64 sparse-only failures the floor, or is there a next
   order of improvement?** Session 04 bench shows 64 matrices
   that fail sparse but pass dense. Worth a one-session pass to
   classify them and see whether a common root cause exists.

## Verdict

**Phase 2.3: WIN.** Every plan success criterion met. Parity
panel improved from 11/28 to 27/28. Worst sparse residual
improved by 15 orders of magnitude. Dense path unchanged (no
regressions). Two emergent fixes landed cleanly, with decision
entries and a clear test signal. The single remaining parity
failure (SSI_2597) has a cleanly-scoped Phase 2.4 hypothesis
and does not block closing Phase 2.3.

Phase 2.3 is **closed**.
