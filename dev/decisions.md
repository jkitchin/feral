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

---

## 2026-04-12 — Phase 1 exit numbers do not generalize beyond n ≤ 500

**Decision.** Phase 1 is not re-opened procedurally. The Phase 1b
exit session file (`dev/sessions/2026-04-12-01.md`) stands as an
accurate record of what was measured under the criterion in effect
at the time. However, the Phase 1 exit numbers are recorded here
as explicitly **not predictive** of feral's behavior on matrices
with n > 500, and the work that closes that gap is treated as
Phase 2 correctness work (not Phase 2 performance work) per the
ordering in `dev/plans/phase-2-planning.md` §2.2.1.

**Why.** The Phase 2.1.2 sanity check, run on the morning of
2026-04-12 immediately after the Phase 1 exit, lifted the
`if mtx.n > 500 { continue; }` filter in `src/bin/bench.rs` and
ran feral's sparse multifrontal pipeline on seven representative
large matrices already present in the existing KKT corpus
(CHWIRUT1 n=645 through CRESC132 n=5314). The pipeline ran to
completion without crashing on any of them and produced:

| Matrix      |    n | consensus inertia | feral inertia    | feral residual | canonical residual |
|-------------|-----:|-------------------|------------------|---------------:|-------------------:|
| CHWIRUT1    |  645 | (431, 214, 0)     | (431, 214, 0)    |        1.4e+09 |            ~1e−13  |
| HAHN1       |  715 | (479, 236, 0)     | (478, 237, 0)    |        1.4e+14 |            ~3e−14  |
| GAUSS2      |  758 | (508, 250, 0)     | (507, 251, 0)    |        1.3e+09 |            ~5e−16  |
| CRESC100    |  806 | (606, 200, 0)     | (606, 200, 0)    |        2.5e+04 |            ~6e−15  |
| MUONSINE    | 1537 | (1025, 512, 0)    | (1026, 511, 0)   |        3.5e+03 |            ~1e−15  |
| VESUVIO     | 3083 | (2058, 1025, 0)   | (2057, 1026, 0)  |        5.6e+14 |            ~1e−12  |
| CRESC132    | 5314 | (2660, 2654, 0)   | (2658, 2656, 0)  |        2.4e+08 |            ~1e−11  |

Two separate defects visible in this data:

1. *Residual bug.* Independent of the inertia bug. CHWIRUT1 and
   CRESC100 have correct inertia but still produce residuals many
   orders of magnitude worse than canonical solvers. Cause:
   feral's sparse path applies no global scaling before
   factorization, while canonical MUMPS and SPRAL/SSIDS both
   apply MC64 matching-based scaling by default for symmetric
   indefinite matrices. Phase 1 saw a weaker version of this on
   ACOPP30 (12 orders of magnitude worse than MUMPS; see
   `dev/phase1-retrospective.org` §"The ACOPP30 residual gap").
   At larger n the defect produces results no reasonable residual
   tolerance can accept. This is the primary Phase 2.2.1 work
   item.

2. *Inertia bug.* ±1 error in positive and negative counts on 5
   of 7 test matrices — classic signature of the deferred
   `count_2x2_inertia` trace-vs-a00 fix firing on near-singular
   2×2 blocks. At n ≤ 500 this bug mostly showed up on ACOPP30
   (Borderline under the consensus); at larger n it fires on
   most KKT matrices with near-singular blocks. This is
   Phase 2.2.2 work, re-evaluated against canonical MUMPS
   rather than the rmumps oracle that regressed it in
   Phase 1b.

**Re-reading Phase 1's residual pass rate.** The Phase 1 bench
tolerance was `n · ε · 10⁶`, which at n = 500 evaluates to
≈ 1.1 × 10⁻⁷. On small matrices, feral was producing residuals
around 10⁻⁷ to 10⁻⁸ while canonical solvers produced 10⁻¹³ to
10⁻¹⁶ on the same inputs — 5 to 9 orders of magnitude worse, but
within the loose absolute tolerance. The Phase 1 "99.7% sparse
residual pass rate" was therefore a measurement of *whether feral
met an absolute tolerance*, not a measurement of *whether feral
was producing answers comparable to canonical solvers*. The
former claim is accurate as stated. The latter is what a casual
reader of the exit summary would assume, and that assumption does
not hold.

**What this changes.** Nothing about the Phase 1b exit commit or
session file is undone. The retrospective
(`dev/phase1-retrospective.org`) already documents the scope caveat
in its "honest assessment of success" section; that caveat is now
a concrete failure mode with measurements attached, and the README
and CHANGELOG have been updated to reflect the revised
interpretation. The Phase 2 plan ordering (`dev/plans/phase-2-planning.md`)
remains correct: Phase 2 opens with measurement infrastructure
(which surfaced the bug in its first hour), followed by the
deferred correctness fixes (MC64 scaling as Phase 2.2.1 and the
trace fix as Phase 2.2.2), followed by pivoting and performance
work. The sanity check the plan called for in §2.1.2 did exactly
what a gate is supposed to do, which was to stop us from
proceeding with corpus expansion on top of a broken sparse path.

**Commitment.** Feral's README will not advertise scale-related
correctness (n > 500 matrices, production KKT workloads, or
performance parity with canonical solvers) until Phase 2.2.1 is
complete and the sanity check panel is re-run with residuals
within 2–3 orders of magnitude of canonical solvers. This is not
a target to aspire to after Phase 2; it is a precondition for
advertising feral as a working sparse solver at all.

## 2026-04-12 — Phase 2.2.2: `pivot_threshold = 0.01` default for MC64 callers

**Decision.** `BunchKaufmanParams::pivot_threshold` defaults to
`0.0` (disabled) for backward compatibility with the dense BK77
tests and the Phase 1 threshold-consistency suite. All MC64
callers opt in explicitly at `u = 0.01`:

- `tests/mc64_regression.rs::ldlt_params`
- `src/bin/bench.rs::params_kkt`
- `examples/triage_large_cresc132.rs`

This mirrors MUMPS `CNTL(1)` default `0.01` and SSIDS `options%u`
default `0.01`, both of which are cited in the Phase 2.2.2
research note (`dev/research/scaling-aware-pivot-rejection.md`
§2). The value is not tuned — we inherit the canonical default on
the reasoning that both Fortran MUMPS and SSIDS have empirical
evidence on much larger corpora than feral has, and reproducing
their setting is a sounder starting point than picking our own.

**Rationale.** MC64 scaling (Phase 2.2.1) equilibrates row and
column norms to `O(1)`, which intentionally shrinks the worst
pivots to be close to the `zero_tol` absolute floor. The original
`BunchKaufmanParams` had no column-relative check, so any pivot
above `zero_tol` was accepted, including pivots that were
`O(10⁻⁴⁷)` relative to their column maximum. On ACOPP30_0000 this
produced 5 effectively-zero forced pivots under `ForceAccept` and
a `2.27e+46` residual — a 30-order regression vs the unscaled
baseline. Phase 2.2.2's column-relative clause (`|a_kk| ≥ u ·
col_max`) rejects these pivots before they reach `ForceAccept`,
and the solve then sees a proper rank-deficient factor rather
than 5 forced zeros interacting with the exp-scaled rescale.
ACOPP30_0000 residual drops `2.27e+46 → 1.076e-1` (47 orders).

The 6 other sanity-panel matrices show no change, because their
pivot streams are already well-conditioned at the absolute
`zero_tol` — the column-relative rejection has nothing to fire
on. This is evidence that Phase 2.2.2 is a *correctness fix*
rather than a general-purpose improvement.

**Explicit deferral: delayed pivoting → Phase 2.3.** Phase 2.2.2
implements MUMPS-style column-relative rejection only. It does
*not* implement SPRAL SSIDS's delayed-pivot mechanism
(`ldlt_tpp.cxx`, where a rejected pivot is carried forward to the
parent front rather than forced-accepted). Three of the four
`tests/mc64_regression.rs` targets (CRESC132, CHWIRUT1, CRESC100)
did not improve under `u = 0.01` and plateau at `1e+02 – 1e+05`;
full closure of their residual gap is expected to require delayed
pivoting in Phase 2.3 plus a separate investigation of
solve-side rounding / refinement convergence on large KKT
systems. The 4 regression tests remain `#[ignore]`'d with updated
Post-2.2.2 status comments. No test tolerances were loosened.

**Commitment.** The README sparse-status section is *not* updated
by Phase 2.2.2. The broader MC64 residual gap remains open. Phase
2.2.2 closes the ACOPP30 correctness regression but does not
promote feral to "competitive on KKT matrices"; that claim still
waits on Phase 2.3. Validation evidence:
`dev/validation/phase-2.2.2-pivot-rejection.md`.

---

## 2026-04-13 — Phase 2.2.3 adjacency fix; drop bench nemin override

**Decision.** In `src/symbolic/supernode.rs::find_supernodes`, the
step-2 amalgamation loop now refuses to merge a child supernode
into its parent unless the child's effective column range is
immediately followed by the parent's column range in the postorder
column numbering (`snode_first_col[root_s] + snode_ncols[root_s] ==
snode_first_col[root_p]`). This is the minimal correctness fix
for a bug where the loop updated `snode_first_col[root_p] = min(...)`
without checking contiguity, producing merged supernodes that
claimed a contiguous column range but actually owned
non-contiguous columns. Variables were eliminated multiple times
with inconsistent state in the downstream code paths
(`build_row_indices`, the A-scan, `elim_cols` construction).

Full analysis: `dev/research/phase-2.2.3-plateau.md`,
`dev/validation/phase-2.2.3-supernode-adjacency.md`.

**Second decision: drop the `nemin=10000` override from
`src/bin/bench.rs`.** That override (commit `81e686c`, "Multi-
supernode solve has a known issue") used `nemin=10000` to force
so much amalgamation under the buggy loop that the claimed column
range became `[0, n)` — trivially contiguous — producing a
degenerate configuration where the sparse path reduced to a
dense LDLᵀ wrapped in sparse plumbing. That configuration is what
produced the historical 99.8% sparse residual pass rate on the
153k–154k KKT corpus. **The 99.8% rate is obsolete and should
never be cited again.** The honest Phase 2.2.3 rate under the
default `nemin=32` is 74.2% inertia match / 77.9% residual pass,
with a worst residual of 2.32e+12 on HYDCAR20_0000. The 22-point
drop reveals the real surface area of the multi-supernode code
path and defines the correctness-closing work for Phases 2.3–2.4.

**Why the minimal fix over the SSIDS-style renumbering.** SSIDS
handles non-adjacent sibling merging by emitting a permutation
`sperm` that renumbers columns so every amalgamated supernode is
contiguous by construction (`src/core_analyse.f90:644-685`). This
is strictly better for fill and flops on arrow-like trees and
would probably close the ACOPP30 regression this session
introduced. But it is a substantially larger refactor touching
the symbolic analysis pipeline end-to-end, and shipping a
correct-but-slower supernode amalgamation today unblocks three
plateau matrices (CHWIRUT1, CRESC100, CRESC132) that now all
beat the canonical MUMPS oracle. Logged as follow-up.

**Commitment.** The README and any future user-facing documents
should cite the post-Phase-2.2.3 numbers, not the historical
99.8%. Phase 2.3 (delayed pivoting) remains on the roadmap and
is expected to help ACOPP30; the SSIDS-style renumbering is
logged as Phase 2.2.4 or as prerequisite work for Phase 2.3. No
test tolerances were loosened. All 146 non-ignored tests pass.
