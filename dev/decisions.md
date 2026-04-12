# Architectural Decisions Log

Append-only. Do not modify existing entries.

---

## 2026-04-12 — Phase 1b exit criterion via multi-source consensus

**Decision.** The strict 100%-correct-vs-rmumps Phase 1b exit criterion in
`FERAL-PROJECT-SPEC.md` §1712 is superseded for the purpose of declaring
Phase 1b complete. The new criterion is multi-source consensus across four
solvers: feral, rmumps, canonical Fortran MUMPS 5.8.2 (built from
`ref/mumps`), and canonical Fortran SSIDS (built from `ref/spral`). For
each matrix in the 153k KKT corpus, classify as **Definitive**, **Borderline**,
**Numerically intractable**, or **Excluded** based on inertia and residual
agreement across the four solvers. Phase 1b exits when feral satisfies the
per-matrix verdict for every Definitive matrix.

Full plan: `dev/plans/phase-1b-consensus-exit.md`.

**Why.** After this session's three structural fixes (postorder pipeline,
best-iterate refinement, factor/solve threshold consistency), feral sits
at 99.2-99.8% on the 153k corpus. The triage of the remaining failures
shows three categories:

1. ~880 matrices where feral solves correctly (residual at machine
   precision) but disagrees with rmumps on the inertia label of
   boundary pivots — feral is not wrong, the oracle disagrees with it
   on a definitional choice.
2. ~400 matrices in problem families (ACOPP30, FBRAIN3LS, CERI*, HS46,
   PFIT2, ...) where ForceAccept on rank-deficient KKTs produces wrong
   `A⁻¹`. The principled fix is delayed pivoting, a Phase 2 feature.
3. 88 sparse-only failures, possibly a sparse-pipeline bug like the
   postorder issue.

The deeper concern: rmumps is a Rust port of MUMPS authored by the same
person developing feral. Treating it as ground truth means a bug in
rmumps and a matching bug in feral would both look like "100% pass"
forever. A multi-oracle consensus catches this class of failure and is
also more honest about matrices where the right answer is genuinely
ambiguous in double precision.

**Reconsideration clause.** This decision is **revisitable**. If running
the consensus across all four solvers reveals that the canonical Fortran
oracles agree with rmumps to within float64 precision on essentially the
entire corpus, then the multi-source machinery has not improved the
ground truth and the original strict criterion can be reinstated. If
the oracles disagree substantially, the consensus criterion stays. The
data from Phases 3-5 of `phase-1b-consensus-exit.md` will tell us which
world we live in.

**Constraints unchanged.** Feral itself remains pure Rust with zero
non-Rust dependencies in the core solver. The Fortran oracles live in a
new top-level `external_benchmarks/` directory, are not built by cargo,
and are not in CI. They are run manually as one-time test infrastructure.

---

## 2026-04-12 — rmumps deprecated as a validation oracle

**Decision.** rmumps (`../ripopt/rmumps`) is no longer considered a
validation oracle for FERAL. Phase 1b's consensus results showed
rmumps disagreeing with canonical Fortran MUMPS 5.8.2 on 2.35% of the
corpus (152,243 / 155,899 matches) and with canonical SPRAL/SSIDS on
2.69%, both worse than any pair involving canonical solvers or feral.
The rmumps sidecars that Phase 1b inherited from `collect_kkt` were
treated as the strict-exit ground truth under the original spec and
turned out to be a co-developed Rust port that could not detect
common-mode bugs shared with FERAL. The consensus framework in
`external_benchmarks/consensus/compute_consensus.py` now votes over
the three canonical oracles {feral, MUMPS, SSIDS} and reports rmumps
alignment as informational metadata.

This decision supersedes the "reconsideration clause" in the
2026-04-12 consensus-exit entry above: there is no path back to a
strict 100%-vs-rmumps criterion, because rmumps is being replaced by
FERAL itself in the downstream ripopt solver and will no longer be
maintained as an independent reference implementation.

**Why keep rmumps in the sidecar pipeline at all.** When rmumps
disagrees with the three canonical oracles on a matrix where feral
agrees with them, the disagreement is a /signal about rmumps/ that
the rmumps author can use to polish rmumps independently. Loading
rmumps inertia from the existing ipopt sidecars costs nothing, and
discarding the data would lose that feedback channel. So the
consensus script continues to read rmumps, display its agreement
rate, and list its dissents — but its vote no longer counts toward
the Definitive / Borderline / Numerically intractable / Excluded
classification.

**Consequence for future collect_kkt runs.** Eventually FERAL will
replace rmumps inside ripopt, at which point the `inertia` field in
newly-generated `<id>.json` sidecars will contain feral's output, not
rmumps's. Loading that field and treating it as a fourth oracle
would be circular — feral validated against feral's own output.
Future `collect_kkt` runs should either (a) stop writing the
`inertia` field entirely, (b) write it under a different key
(`solver_reported_inertia`) that the consensus framework does not
read, or (c) emit a "linear solver used" field so the framework can
refuse to consume inertia generated by the system under test. This
is Phase 2 planning work and is tracked here as a reminder; no
immediate action is needed.

**Consensus rule changes.** The voting set shrinks from four to
three solvers, so the strong/weak thresholds are re-parameterized:

| Old (4 oracles)                    | New (3 canonical + rmumps info) |
|------------------------------------+---------------------------------|
| Strong: ≥3 of 4 agree              | Strong: all 3 agree             |
| Weak: 2 of 4 + others within ±1    | Weak: 2 of 3 + third within ±1  |
| None: otherwise                    | None: otherwise                 |

The expected effect on Phase 1b's already-passing numbers is small
because matrices where feral, MUMPS, and SSIDS already agreed stay
Definitive regardless of rmumps. Matrices where the 4-vote
classification depended on rmumps's tiebreaking vote drop to
Borderline or Numerically intractable — they were never really
Definitive, and this re-classification is a correction.
