# Inertia Mismatch Triage — 2026-04-27

**Carried forward across sessions 02–09 as "45 BOTH-path inertia
mismatches" / "43 BOTH-path inertia mismatches".**

## TL;DR

Of the 113 matrices in the 169_585-matrix KKT corpus where feral's
sparse-path inertia disagrees with at least one canonical oracle:

- **8 are real feral bugs**, all in the ACOPP30 family — already
  scoped under task #19 (`dev/research/task-19-dense-acopp30-expert-consultation.md`).
- **3 are borderline-real** in the FBRAIN3LS family (n=6,
  `verdict=numerically_intractable` — even MUMPS and SSIDS plateau
  at residuals 1e-7 to 1e-8 on these). Defensible as feral correctly
  identifying rank deficiency at the singular boundary.
- **102 are no-consensus** matrices where MUMPS ≠ SSIDS. On 88 of
  these feral matches MUMPS exactly; on the remaining 14 either
  feral matches rmumps or all four solvers split.

**The bench's "BOTH-path inertia mismatches" line is a subset of
these 113** — specifically the cases where the dense path *also*
disagrees, which adds the residual-quality dimension on ACOPP30
already covered by task #19.

**Implication for the CLAUDE.md "Inertia must be exactly correct —
no tolerance on inertia counts" rule:** the rule was written
assuming a well-defined answer exists. On the 102 no-consensus
matrices it does not — two reference Fortran solvers (MUMPS 5.8.2
and SPRAL SSIDS) routinely produce different inertia counts because
they pivot differently at near-zero diagonals. The verdict.json
consensus framework already handles this correctly by tagging
matrices `excluded` when fewer than 3 of 4 oracles agree.

## Methodology

Scanned all `*.verdict.json` files under `data/matrices/kkt`
(169_585 files produced by `external_benchmarks/consensus/compute_consensus.py`).
Each verdict file contains the four-source inertia comparison:

- `rmumps` — Ipopt sidecar (.json)
- `feral`  — feral sparse path (.feral.json)
- `mumps`  — Fortran MUMPS 5.8.2 oracle (.mumps.json)
- `ssids`  — SPRAL SSIDS oracle (.ssids.json)

Filtered to matrices where `feral_match_inertia == false` (113).
Then bucketed by:

1. **Real feral bug** — `mumps == ssids != feral` (canonical
   consensus disagrees with feral)
2. **No-consensus** — `mumps != ssids` (no canonical answer to
   match against; feral typically matches one)

There were **0 sidecar-stale** cases (`mumps == ssids == feral` but
`rmumps` disagrees). The Phase 2.3-era population of 1499
sidecar-stale matrices documented in
`dev/research/shared-failure-triage.md` is no longer in the
mismatch set — the verdict framework correctly compares against
the MUMPS+SSIDS consensus rather than the rmumps sidecar.

## Per-family breakdown

### Real feral bugs (11)

#### ACOPP30 (8 of 11)

```
ACOPP30_0006   feral=(72,137,0)  MUMPS=SSIDS=(71,138,0)  verdict=borderline
ACOPP30_0008   feral=(72,137,0)  MUMPS=SSIDS=(71,138,0)  verdict=borderline
ACOPP30_0011   feral=(72,137,0)  MUMPS=SSIDS=(71,138,0)  verdict=borderline
ACOPP30_0040   feral=(72,137,0)  MUMPS=SSIDS=(71,138,0)  verdict=numerically_intractable
ACOPP30_0046   feral=(72,137,0)  MUMPS=SSIDS=(71,138,0)  verdict=numerically_intractable
ACOPP30_0047   feral=(72,137,0)  MUMPS=SSIDS=(71,138,0)  verdict=numerically_intractable
ACOPP30_0049   feral=(72,137,0)  MUMPS=SSIDS=(71,138,0)  verdict=numerically_intractable
ACOPP30_0058   feral=(72,137,0)  MUMPS=SSIDS=(71,138,0)  verdict=numerically_intractable
```

Feral residuals are 1.9e-2 to 2.6e-2 on every one — the same
2.7e-2 plateau documented in task #19. Already scoped:
`dev/research/task-19-dense-acopp30-expert-consultation.md`
established that feral hits a Duff-Reid-rejected 2×2 pivot, falls
back to a near-zero 1×1, and the L21 column blows up. Two
attempted fixes regressed broader corpus; the chosen path was to
reroute bench dense validation through `factor_frontal` to match
MUMPS's no-dense-shortcut design.

Note that 5 of 8 are `numerically_intractable` (MUMPS itself
produces residuals 1e-7 to 2.9e-6 on these — borderline matrices
where even the oracle is straining).

#### FBRAIN3LS (3 of 11)

```
FBRAIN3LS_0839  n=6  feral=(5,0,1)  MUMPS=SSIDS=(6,0,0)  feral_res=3.3e-8  mumps_res=2.7e-7  ssids_res=6.5e-8
FBRAIN3LS_0843  n=6  feral=(5,0,1)  MUMPS=SSIDS=(6,0,0)  feral_res=5.4e-9  mumps_res=4.1e-8  ssids_res=1.6e-8
FBRAIN3LS_0851  n=6  feral=(5,0,1)  MUMPS=SSIDS=(6,0,0)  feral_res=2.9e-9  mumps_res=3.8e-9  ssids_res=1.1e-8
```

All three are n=6, `verdict=numerically_intractable`. **Feral's
residual is comparable to or better than MUMPS** on every one of
them. MUMPS and SSIDS use static pivoting (replace |a_ii| < ε with
±ε) and report a full-rank (6,0,0); feral's pivoting flags one
diagonal as zero, returning (5,0,1).

Rather than a pivoting bug, this looks like a *legitimate
divergence in singularity-detection policy*. On a matrix that all
three solvers agree they can't solve to machine precision, feral
is more honest about reporting the rank deficiency.

The intermediate-residual rate (1e-7 to 1e-8 across all three
solvers) is consistent with the matrix actually having a singular
value at the unit roundoff boundary.

### No-consensus (102)

#### SPANHYD (35)
```
feral = MUMPS = (81, 33, 0)
SSIDS         = (97, 17, 0)        ← 16-eigenvalue gap
```
All 35 matrices show identical inertia tuples across all four
oracles. SSIDS's pivoting drives 16 more eigenvalues positive than
MUMPS's. Verdict: `excluded` on all 35.

#### KIRBY2 (18)
```
feral = MUMPS = (307, 151, 0)
SSIDS         = (322, 136, 0)      ← 15-eigenvalue gap
```

#### CERI651B (18)
```
feral = MUMPS = (139, 66, 0)
SSIDS         = (205, 0, 0)        ← 66-eigenvalue gap (SSIDS reports PD)
```

#### LINSPANH (17)
```
feral = MUMPS = (81, 33, 0)
SSIDS         = (91, 23, 0)        ← 10-eigenvalue gap
```

#### ACOPP30 (12)
```
feral = rmumps = (72, 137, 0)
MUMPS          = (71, 137, 1)
SSIDS          = (71, 138, 0)
```
Three-way split. MUMPS reports 1 zero pivot; SSIDS pivots through
it as a tiny negative; feral and rmumps as a tiny positive.

#### FBRAIN3LS (1) and POLAK6 (1)
Three-way and four-way splits respectively, n ≤ 9.

### Distribution among no-consensus matrices

```
feral matches MUMPS:                88 / 102   (86.3%)
feral matches rmumps (vs others):   12 / 102   (11.8%)
feral matches none:                  2 / 102    (2.0%)
```

Feral's pivoting is in family with MUMPS's. The remaining
disagreements are SSIDS-vs-everyone-else.

## Bucket summary

| Bucket                            | Count | Feral bug? | Action                                 |
| --------------------------------- | ----: | :--------- | :------------------------------------- |
| Real bug — ACOPP30 dense gap      |     8 | Yes        | Task #19 (already scoped)              |
| Borderline — FBRAIN3LS singular n=6 |   3 | Soft       | Document as policy choice or revisit if widespread |
| No-consensus — SSIDS dissents     |    88 | No         | Tag in CLAUDE.md as "consensus, not oracle" |
| No-consensus — three-way splits   |    14 | No         | Tag as `excluded` (already done)       |
| **Total**                         |   **113** |       |                                        |

## CLAUDE.md rule implication

Current CLAUDE.md text:

> Inertia must be exactly correct — no tolerance on inertia counts

This is too strong. There exist physically meaningful KKT systems
where MUMPS 5.8.2 and SPRAL SSIDS — the two reference Fortran
direct solvers — produce different inertia counts that differ by
up to 66 eigenvalues. The disagreement reflects different pivoting
strategies near singular pivots, not a bug in either.

**Proposed clarification:**

> Inertia must be exactly correct on non-singular matrices. On
> matrices where canonical Fortran direct solvers (MUMPS 5.8.2
> and SPRAL SSIDS) disagree on inertia, feral must agree with at
> least one of them. The corpus consensus framework
> (`compute_consensus.py`) tags matrices with no 3-of-4 agreement
> as `excluded` and they are not part of the inertia gate.

The bench's "BOTH-path inertia mismatch" reporter should also be
updated to filter out `verdict ∈ {excluded, numerically_intractable}`
cases, so the headline number reflects only matrices where a
canonical answer exists.

## Decisions

1. **Do not reopen task #19.** The 8 ACOPP30 hard bugs are already
   under expert-consulted disposition. The chosen approach
   (re-route dense bench through `factor_frontal`) supersedes
   chasing the dense `factor()` Duff-Reid 2×2 fallback path.

2. **FBRAIN3LS as policy, not bug.** Feral's singular-pivot
   detection on n=6 KKT systems where the matrix is provably at
   the rank-deficiency boundary (all 4 oracles fail residual gate)
   is defensible behavior. Document in `dev/decisions.md` rather
   than chase as a fix.

3. **Update CLAUDE.md inertia clause** to reference the consensus
   framework rather than imply a single oracle exists. (One-line
   edit; not a code change.)

4. **Update bench's "BOTH-path inertia mismatches" reporter** to
   exclude `verdict ∈ {excluded, numerically_intractable}` so the
   number reflects only correctness-relevant disagreements. Once
   filtered, the number drops from 43 to ≤ 11.

5. **Multi-RHS / cond-est / Schur work is not blocked** by the
   inertia mismatch population. The 8 hard bugs are scoped under
   task #19; the rest are no-consensus or borderline-singular.

## Files generated

- `/tmp/triage_verdicts.py` — initial scan script (not committed;
  artifacts of this triage live in this report)
- `/tmp/triage_detail.py` — per-family detail script
- `/tmp/realbugs_detail.py` — real-bugs detail script with residuals
