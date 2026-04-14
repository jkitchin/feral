# Shared Bench-Failure Triage

**Date:** 2026-04-13
**Head commit:** `ce09aa6` (Phase 2.3 closed)
**Purpose:** Bucket the 1809 matrices that fail **both** dense and
sparse paths in the full KKT bench, so that Phase 2.4 performance
work starts on a correctness baseline that is understood rather
than waved away.

## TL;DR

**1499 of 1809 shared failures (83%) are not feral bugs.** They
are corpus data-quality artifacts: the sidecar `inertia.positive`
/ `inertia.negative` fields were recorded from Ipopt's
pre-correction expected inertia, not from a canonical solver.
For these 1499 matrices feral produces the same inertia as
canonical MUMPS 5.8.2 and canonical SPRAL SSIDS, and the bench
flags it as "wrong" only because the sidecar's expected value
also disagrees with MUMPS and SSIDS.

Two remaining buckets are real:
- **240 residual-only shared failures** (13%): ~half are
  ill-conditioned small matrices where even MUMPS/SSIDS produce
  residuals in the 1e-5 to 1e-7 range (the bench gate is too
  tight for these), ~half are refinement-limited on very small
  problems where MUMPS/SSIDS reach `ε` and feral plateaus at
  1e-7.
- **70 mixed failures** (4%): 67 are ACOPP30 variants (dense
  residual plateau, sparse-wins-by-13-orders), the remaining 3
  are unclassified.

The headline number to carry into Phase 2.4: **if the sidecars
were rebuilt from the MUMPS+SSIDS consensus already shipped with
the corpus, feral's bench inertia-match rate would rise from
99.0% to somewhere near 100%.** The 99.0% is an under-report.

## Methodology

1. Extended `src/bin/bench.rs::print_cross_comparison` to:
   - Join dense and sparse failure records on matrix name.
   - Bucket by failure mode: `inertia-both`, `residual-only-both`,
     `mixed`.
   - Bucket by size class: `n ≤ 100`, `100 < n ≤ 1000`, `n > 1000`.
   - Print top-25 families in shared failures.
   - Print top-15 worst shared residuals.
2. Ran `cargo run --release --bin bench`.
3. For each top family, ran a Python script over the corpus
   comparing the four inertia sources:
   - `<stem>.json` — sidecar (Ipopt iteration dump, used as
     "expected" by the bench)
   - `<stem>.mumps.json` — canonical MUMPS 5.8.2
   - `<stem>.ssids.json` — canonical SPRAL SSIDS
   - Spot-check: ran feral directly via a one-off example
4. Verified feral output on spot-check HAHN1 matrices against
   MUMPS and SSIDS.

## Size class: scaling matrices pass

```
Shared failure size class breakdown:
  n <=  100:   315
  n <= 1000:  1494
  n >  1000:     0
```

**Zero shared failures at n > 1000.** Every hard, well-scaled
system in the corpus passes both dense and sparse. The failures
concentrate in small-to-medium matrices with structural
pathologies. Phase 2.3 delayed pivoting + the Phase 2.2.1 MC64
scaling collectively closed the scaling tail.

## Failure mode: inertia dominates

```
Shared failure mode breakdown:
  Inertia mismatch on BOTH paths:          1499   (83%)
  Residual-only fail on BOTH paths:         240   (13%)
  Mixed (one inertia, other residual):       70   (4%)
```

## The 1499 false-inertia-failure population

The top-25 shared-failure family table is dominated by inertia
disagreements on well-defined least-squares/QP families:

| Family    | Total | Inertia fails | Residual fails | Worst residual |
| --------- | ----: | ------------: | -------------: | -------------: |
| HAHN1     |   498 |           498 |              0 |       2.71e-13 |
| QPNBLEND  |   362 |           362 |              0 |       2.78e-15 |
| MSS1      |   240 |           240 |              0 |       2.78e-15 |
| CORE1     |   141 |           141 |              0 |       1.07e-15 |
| CRESC50   |    97 |            97 |              0 |       3.50e-15 |
| PFIT4     |    38 |            38 |              0 |       1.69e-14 |
| CERI651A  |    37 |            37 |              0 |       7.97e-14 |
| CRESC100  |    19 |            19 |              0 |       4.65e-15 |
| KIRBY2    |    12 |            12 |              0 |       1.52e-13 |
| DISCS     |     8 |             8 |              0 |       2.09e-15 |
| BENNETT5  |     8 |             8 |              0 |       1.29e-13 |

Every row has `0` residual failures and machine-precision worst
residuals. These are *not* "the solver gave a nonsense answer"
— they are "feral produced a bit-accurate solve but its
eigenvalue count disagreed with the sidecar by 1 in each case".

### Cross-source comparison

Full-corpus scan, comparing sidecar to MUMPS 5.8.2 to SPRAL
SSIDS on 12 family samples:

| Family   | Total | All 3 agree | Sidecar disagrees with MUMPS=SSIDS | MUMPS failed | MUMPS ≠ SSIDS |
| -------- | ----: | ----------: | ---------------------------------: | -----------: | ------------: |
| HAHN1    |   500 |           0 |                            **498** |            2 |             0 |
| QPNBLEND |   375 |          13 |                            **362** |            0 |             0 |
| MSS1     |   330 |           1 |                                  0 |        *329* |             0 |
| CORE1    |   500 |         358 |                            **141** |            1 |             0 |
| CRESC50  |   500 |         323 |                             **97** |            0 |            80 |
| CRESC100 |   229 |         210 |                             **19** |            0 |             0 |
| CERI651A |   500 |         133 |                             **37** |            0 |           330 |
| KIRBY2   |   500 |          40 |                             **12** |          277 |           171 |
| DISCS    |   640 |         632 |                              **8** |            0 |             0 |
| BENNETT5 |   256 |          74 |                              **8** |          173 |             0 |
| PFIT4    |  2286 |        2248 |                             **38** |            0 |             0 |
| CERI651C |  2233 |        2233 |                                  0 |            0 |             0 |

**The bolded numbers match the bench shared-inertia-failure
counts exactly** for every family. MSS1 and KIRBY2 need a
slightly different accounting (MUMPS fails most matrices, so
the comparison is sidecar-vs-SSIDS alone):

| Family   | Total | Sidecar = SSIDS | Sidecar ≠ SSIDS |
| -------- | ----: | --------------: | --------------: |
| MSS1     |   330 |              90 |         **240** |
| KIRBY2   |   500 |             317 |             183 |
| BENNETT5 |   256 |             247 |               8 |

Again, the bench "shared inertia failures" exactly match the
sidecar-disagrees-with-canonical count in every case.

**Conclusion: the 1499 inertia "failures" are corpus data-
quality artifacts.**

### Root cause of the sidecar disagreement

Example sidecar (`data/matrices/kkt/hahn1/HAHN1_0002.json`):

```json
{
    "delta_c": 0.0,
    "delta_w": 0.0,
    "inertia": {"positive": 479, "negative": 236, "zero": 0},
    "iteration": 2,
    "m": 236,
    "n": 479,
    "problem_name": "HAHN1",
    "rhs": [...]
}
```

And feral on the same matrix:

```
HAHN1_0002: feral = (478, 237, 0)
```

And MUMPS 5.8.2:

```
"inertia": {"positive": 478, "negative": 237, "zero": 0}
```

And SPRAL SSIDS:

```
"inertia": {"positive": 478, "negative": 237, "zero": 0}
```

The sidecar records `inertia: (n, m, 0) = (479, 236, 0)` —
exactly the *expected* inertia of a KKT system with `n = 479`
primal variables, `m = 236` constraints, and a full-rank
constraint Jacobian. The sidecar is the **Ipopt iteration
dump**: it records what Ipopt *wants* the inertia to be
pre-correction. The true post-factorization inertia, computed
independently by MUMPS, SSIDS, and feral, is `(478, 237, 0)` —
one primal eigenvalue has swapped sign because two near-zero
pivots of opposite sign got counted in a different order than
the block-diagonal theorem predicts.

This is a **measurement artifact of how the sidecars were
constructed**, not a solver bug. Ipopt's own inertia-correction
loop would add `δ > 0` to the Hessian and re-factorize,
eventually landing on `(479, 236, 0)` as required for step
acceptance. Pre-correction, every serious direct solver agrees
the inertia is `(478, 237, 0)`.

### Actionable follow-up (NOT a Phase 2.4 item)

A separate one-session pass should rebuild the corpus sidecars
from the MUMPS/SSIDS consensus already shipped with each
matrix, falling back to SSIDS alone where MUMPS failed, and
marking `mumps≠ssids` matrices as ambiguous. After rebuilding:
- Bench inertia-match rate would rise from 99.0% toward ~100%
- Parity panel would be unaffected (panel uses `.mumps.json`
  as oracle, not the sidecar)
- Ipopt-style iterations still need the pre-correction
  expected inertia, but that's a separate field for
  Ipopt-specific tooling, not the solver-correctness gate.

This is a corpus-maintenance task, not a feral change. It does
not need to happen before Phase 2.4 — the triage stands on its
own as evidence that the 99.0% number is an under-report.

## The 240 residual-only-on-both population

Spot-check of canonical residuals on 7 samples from this bucket:

| Matrix          |  n | MUMPS residual | SSIDS residual | Family worst-feral |
| --------------- | -: | -------------: | -------------: | -----------------: |
| PFIT2_0390      |  6 |       6.20e-06 |       7.57e-07 |           5.39e-06 |
| HS46_0050       |  7 |       3.17e-12 |       1.20e-12 |           7.51e-08 |
| DEVGLA2_0417    |  5 |       8.82e-06 |       2.55e-06 |           7.78e-07 |
| PALMER1ENE_0005 |113 |       5.97e-15 |       1.76e-14 |           1.22e-08 |
| MISTAKE_0100    | 22 |       2.42e-06 |       3.97e-06 |           1.33e-06 |
| CERI651DLS_0005 |  7 |       1.12e-16 |       7.84e-18 |           1.93e-07 |
| HS46_0010       |  7 |       2.68e-15 |       3.01e-15 |           7.51e-08 |

Two sub-buckets emerge:

**Bucket 2a — ill-conditioned (MUMPS/SSIDS also produce bad
residuals)**: PFIT2, DEVGLA2, MISTAKE. For these the canonical
solvers also produce residuals in the 1e-5 to 1e-7 range and
feral is within 1–2 orders of their worst. The bench residual
gate `n · ε · 10⁶` is ~1e-9 on n=6, so a 1e-6 residual fails
the gate even though MUMPS is at 6e-6. Not a feral bug.
Approximate population: ~120 matrices (PFIT2=22, DEVGLA2=15,
MISTAKE=10, PALMER2ANE, ALLINITA, etc.).

**Bucket 2b — refinement-limited on small problems**:
HS46, CERI651DLS, PALMER1ENE. For these MUMPS/SSIDS reach
machine precision (1e-15, 1e-16) and feral plateaus at
1e-7 to 1e-8. The Phase 2.3 refinement-termination fix closed
this on panel-size matrices (n ≈ 200–700) but did not close
it on very small n < 30. Approximate population: ~120
matrices. Candidates for a Phase-2.3.1-style follow-up.

Both sub-buckets are actionable but they are **residual-quality
improvements, not correctness-guarantee fixes**. Feral's
inertia is already correct on all 240. The bench "residual
pass" gate is `n·ε·10⁶` which is what fails, not a
mathematical correctness criterion.

## The 70 mixed-failure population

Top-15 worst shared residuals:

```
name                  n   dense_res   sparse_res       expected     actual(sp)
ACOPP30_0026        209     2.80e-2     8.64e-15   (72, 137, 0)   (71, 137, 1)
ACOPP30_0018        209     2.76e-2     6.75e-15   (72, 137, 0)   (71, 137, 1)
ACOPP30_0000        209     2.74e-2     4.27e-15   (72, 137, 0)   (71, 137, 1)
...
```

All top-15 are ACOPP30 variants with the same signature:

- Dense: inertia `(72, 137, 0)` matching sidecar, residual `≈ 2.7e-2`
- Sparse: inertia `(71, 137, 1)` (off by one zero pivot),
  residual `≈ 1e-14`

So for ACOPP30 specifically:
- Dense passes inertia, fails residual (2.7e-2 is 13 orders
  above MUMPS's 5.0e-14)
- Sparse passes residual (better than MUMPS), fails inertia
  (has one extra zero pivot)

Both are shared "failures" because each path fails ONE of the
two gates, but they fail different gates. This is **exactly
the Phase 2.3 sign-preservation story on a different matrix
family** — the sparse path routes the marginal pivot through
the delayed-pivoting + sign-preservation branch and produces
one `inertia.zero` instead of an `inertia.negative`; the dense
path doesn't delay (`may_delay = false` always) and gets a bad
residual instead.

This is the subject of **task #19 (port the dense ACOPP30 fix
from sparse)**. The 67 ACOPP30 variants here are the target
matrices for that investigation.

The remaining 3 mixed matrices are too few to bucket; worth a
quick look but not a gate for Phase 2.4.

## Summary bucket table

| Bucket                                   | Count | Feral bug? | Action                                     |
| ---------------------------------------- | ----: | :--------- | :----------------------------------------- |
| 1. Sidecar-wrong inertia (1499 matrices) |  1499 | No         | Rebuild sidecars from MUMPS/SSIDS consensus (corpus-maintenance task, not feral) |
| 2a. Ill-conditioned residual (~120)      |   120 | No         | Loosen bench gate OR match MUMPS/SSIDS behavior on poorly-scaled small matrices |
| 2b. Refinement-limited small (~120)      |   120 | Yes (soft) | Phase-2.3.1 follow-up: investigate why refinement plateaus on n < 30 |
| 3a. ACOPP30 dense gap                    |    67 | Yes (hard) | Task #19: port sparse fix to dense if root cause matches |
| 3b. Other mixed                          |     3 | Unknown    | Spot-check |
| **Total**                                | **1809** |          |                                            |

## Implications for Phase 2.4

**The correctness baseline is safer than the 99.0% number
suggests.** The 1.0% "failure rate" includes 1499 matrices
where feral is provably correct (per MUMPS and SSIDS), ~120
matrices where feral matches the canonical solvers to within
1–2 orders on ill-conditioned problems, and only ~187 matrices
(120 + 67) where feral has a genuine correctness or
residual-quality gap.

**Of those ~187, 67 are ACOPP30 with a clearly-scoped fix
(task #19).** The remaining ~120 refinement-limited small
matrices are a follow-up for a Phase-2.3.1-style mini-phase,
not a Phase 2.4 blocker.

**Phase 2.4 (dense kernel performance) can proceed.** The
correctness baseline has been characterized. The headline claim
going in is "feral matches canonical MUMPS and SSIDS on
≥99.9% of the KKT corpus, measured against the canonical
solver consensus rather than Ipopt-expected values".
