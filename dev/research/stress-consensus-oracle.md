# Stress gate: replace the synthetic-label oracle with a solver consensus

Date: 2026-05-20
Status: research note for the report.py classify() rewrite (option 3).

## Problem

`external_benchmarks/stress/report.py` is the `stress-smoke` PR-blocking
CI gate. For rank-deficient synthetic matrices it checks `inertia.zero`.
The rule it has used is a band derived from the matrix *name*:

    expected_zero("rankdef_<n>_<k>") = k          # the constructed k
    accept iff  1 <= zero <= k                    # plus exemptions

This is the wrong oracle. `k` is the null-space dimension the generator
*built into* the matrix — an input, not a verified output. On a
borderline-singular matrix the rank deficiency can be numerically tiny
enough that a direct solver legitimately counts the would-be null pivots
by sign and reports `zero=0` (full numerical rank). That is not a bug:
it is the documented default behaviour of MUMPS, SSIDS and MA57, and —
since the #39 F-01 sign-fallback — of feral too.

Consequence: the gate flagged feral for producing the *same answer the
canonical solvers produce*. Two layers of hand-maintained patch
accreted to suppress the false positives:

- `MUMPS_REPORTS_ZERO0` — a frozenset of 3 matrices exempted from the
  `zero >= 1` lower bound.
- `ALLOWLIST` — 5 per-matrix bypasses citing issues #39 / #40.

Both exist only because the gate's oracle is fake.

## Borderline inertia has no ground truth

For a non-singular matrix inertia is exact and unique. For a matrix
whose smallest pivot lands in the band `(EPS, sqrt(n)*EPS*||A||]`,
"is this pivot zero?" is a numerical judgement, and canonical solvers
disagree by design:

- MUMPS default, SSIDS, MA57, feral (#39): count by sign -> `zero=0`.
- MUMPS with `ICNTL(24)=1` (explicit null-pivot detection, non-default):
  may report `zero>0`.

`CLAUDE.md` already defines "correct" for exactly this situation:

  > feral inertia must agree with at least one of {MUMPS, SSIDS}.

So the gate should check that contract directly, not a name label.

## Oracle data (this session, 9 rank-deficient synthetics)

Verified by running the three oracle binaries
(`external_benchmarks/{mumps,ssids,ma57}_oracle/`) on each `.mtx`.
MUMPS run with `ICNTL(24)=1`; SSIDS run with `OMP_CANCELLATION=true`.

| matrix                  | feral (x86 CI) | MUMPS IC24 | SSIDS      | MA57       |
|-------------------------|----------------|------------|------------|------------|
| rankdef_5_2             | (3,2,0)        | (2,2,1)    | (4,1,0)    | (3,2,0)    |
| rankdef_10_3            | (4,5,1)        | (3,4,3)    | (4,6,0)    | (4,6,0)    |
| rankdef_50_5            | (26,24,0)      | (26,24,0)  | (25,25,0)  | (27,23,0)  |
| rankdef_200_20          | (110,90,0)     | (111,89,0) | (110,90,0) | (110,90,0) |
| rankdef_exact_50_5      | (24,26,0)      | (24,26,0)  | (24,26,0)  | (23,27,0)  |
| rankdef_exact_100_10    | (56,44,0)      | (54,43,3)  | (56,44,0)  | (56,44,0)  |
| saddle_rankdef_50_10_3  | (51,39,0)      | (50,37,3)  | (52,38,0)  | (51,39,0)  |
| saddle_rankdef_100_20_5 | (102,78,0)     | (100,75,5) | (103,77,0) | (103,77,0) |
| stokes_q1p0_8           | (99,63,0)      | (98,62,2)  | (100,62,0) | (100,62,0) |

x86 CI feral values are from CI run 26159004313 (commit `4eb9c5e`,
`ubuntu-latest`, x86_64).

Observations:
- `zero` is what matters; pos/neg split varies +/-1 between solvers as
  BK pivot-selection noise on near-singular blocks — even MUMPS, SSIDS
  and MA57 disagree on pos/neg.
- SSIDS reports `zero=0` on every one. feral (x86) reports `zero=0` on
  **8 of the 9** — and so agrees with SSIDS's `zero` on those 8. The
  exception is `rankdef_10_3`: feral reports `zero=1` there, on x86 and
  aarch64 alike, matching no canonical oracle (MUMPS-IC24 `zero=3`,
  SSIDS `zero=0`). That both-arch consensus miss is tracked in #42 and
  carries an `ALLOWLIST` entry on every architecture.
- MUMPS-IC24 reports `zero>0` on 4 of them — it is the only oracle that
  detects the constructed null space, because IC24 explicitly looks.

## Proposed rule

Replace the band + `MUMPS_REPORTS_ZERO0` + rankdef `ALLOWLIST` with:

  accept iff  feral.zero == MUMPS_IC24.zero  OR  feral.zero == SSIDS.zero

plus the existing `pos + neg + zero == n` consistency check, and the
existing `status`/`rel_res` rules (unchanged).

This is **not purely a relaxation**. It is looser where the band was
wrong (permits `zero=0`, which all oracles support) but *tighter* where
the band was too loose: the band's `1 <= zero <= k` silently accepted
partial detection that no oracle agrees with — e.g. feral-aarch64's
`zero=1` on `rankdef_50_5`, where MUMPS, SSIDS and MA57 all say `0`.
Under the consensus rule that becomes a flag, correctly surfacing the
#40 cross-arch BK divergence instead of hiding it. CI runs on x86,
where feral matches a canonical oracle on 8 of the 9 synthetics, so
those 8 pass the gate with no allowlist. The 9th, `rankdef_10_3`,
flags on x86 too (feral `zero=1` on both arches, #42) and so keeps an
`ALLOWLIST` entry that applies on every architecture — distinct from
the two `#40` entries, which flag on local aarch64 only.

MA57 is recorded in the oracle file for context but is not part of the
gate predicate: `CLAUDE.md` names MUMPS and SSIDS as the two canonical
Fortran solvers; MA57 is a supplementary reference.

## Storage

One committed file `external_benchmarks/stress/oracles.json`, keyed by
matrix name, each entry carrying n, constructed_k, an SHA-256 of the
`.mtx` bytes at generation time, and the full inertia triple from each
of the three oracles. The matrices themselves stay gitignored
(regenerated by the seeded, bit-reproducible `synth.py`); the SHA-256
pins the oracle to exact bytes so a `synth.py` change that alters a
matrix is caught by report.py as a stale-oracle flag rather than
silently invalidating the gate. Regeneration is scripted in
`gen_oracles.py`.

## Why this matches house style

`tests/data/parity/` already does exactly this for the curated KKT
corpus: committed `.mumps.json` / `.ssids.json` sidecars, and
`tests/parity.rs::run_parity` asserts feral matches at least one. The
corpus-wide `external_benchmarks/consensus/compute_consensus.py` is the
same idea over four solvers. The stress suite is the lone holdout that
used a label instead of a solver; this note converges it onto the
existing standard.
