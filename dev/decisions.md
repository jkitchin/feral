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

## 2026-04-13 — Phase 2.3: pivot-threshold split between dense and sparse callers

**Decision.** Sparse multifrontal callers use
`BunchKaufmanParams::pivot_threshold = 0.01` (SSIDS / MUMPS default
`u`). The dense `factor()` path and all dense benchmarks use
`pivot_threshold = 0.0` via an explicit override.
`BunchKaufmanParams::default()` stays at `0.0`.

**Why.** The column-relative threshold test `|d| >= u*col_max` only
pays off when rejected pivots have somewhere to go — delayed
pivoting at non-root supernodes gives them a landing zone at the
parent. The dense BK kernel has no delayed-pivoting machinery
and runs under Knight-Ruiz ∞-norm equilibration, which handles
column scaling at preprocess time. Using `u = 0.01` in the dense
kernel would trade equilibration-handled cases for a hard
column-relative rejection with nowhere to go, regressing the
99.0% dense KKT rate. Sparse, by contrast, has delayed pivoting
(Phase 2.3 Steps 5+6) and MC64 scaling that do not equilibrate
to the dense kernel's precision, so the threshold earns its
keep.

**Scope.** `src/bin/bench.rs` carries two configs:
`params_kkt_dense` (0.0) for the dense sweep, `params_kkt_sparse`
(0.01) for the sparse sweep. `examples/*.rs` that exercise both
paths similarly carry two configs. Library clients constructing
`BunchKaufmanParams::default()` get `0.0` and are unchanged by
Phase 2.3 — they must explicitly opt into `0.01` if they want
the sparse-path behavior.

**Evidence.** Dense KKT rate: 152979/154481 (99.0%) unchanged
before and after. Sparse KKT rate: 152987 → 153009 inertia match,
154113 → 154237 residual pass, 1.19e0 → 3.22e-4 worst residual,
203 → 64 sparse-only failures. Full measurements in
`dev/sessions/2026-04-13-04.md`.

## 2026-04-13 — Phase 2.3: preserve pivot sign at root-supernode fallback

**Decision.** When the column-relative threshold test rejects a
1×1 pivot at a root supernode (`may_delay = false`), and the
`ForceAccept` zero-pivot policy is in effect,
`src/dense/factor.rs::try_reject_1x1_frontal` accepts the pivot
with its correct sign (`d > 0 → pos += 1`, else `neg += 1`) and
flags `needs_refinement = true` — only `|d| <= zero_tol ≈ eps`
is counted as a zero pivot. The 2×2 fallback routes through the
same path.

**Why.** Converting a small-but-clearly-nonzero pivot into a
zero loses inertia information and produces residuals that
iterative refinement cannot recover, because the pivot is driven
to exactly 0 instead of being preserved with its noisy-but-
nonzero value. This is exactly the DEGENLPA_0065 failure mode:
the reference reports `(20, 15, 0)` and feral reported
`(20, 14, 1)` with a 7.06e2 residual. MUMPS always reports
`n0 = 0` in the default configuration (INFOG(28) is only
computed when ICNTL(24)=1), so the reference oracle never
reports zero pivots — the comparison is partly a measurement
artifact on top of the real sign-loss bug. SSIDS handles the
same case by breaking at the root and leaving the pivot
un-eliminated (the outer multifrontal driver reassembles it);
sign preservation is a strictly smaller change that captures
the correctness gain without touching the root-break logic.

**Evidence.** Parity 14/28 → 22/28 (flipped CERI651A×3,
DEGENLPA_0065, DEGENLPB_0045/0046/0047, PALMER2ANE_0000).
Sparse worst residual 7.06e2 → 3.22e-4 (six orders of
magnitude). Full measurements in
`dev/sessions/2026-04-13-04.md`. No test tolerances were
loosened. The `factor_frontal_root_force_accepts_without_delay`
unit test was updated to use `d = 0` exactly (matching the
absolute-zero branch), and
`factor_frontal_root_accepts_small_pivot_with_sign` was added to
cover the new sign-preserving branch with a clearly-negative
pivot.

---

## 2026-04-14 — Accepted pulp 0.22.2 as the SIMD backbone for Phase 2.4.2

**What.** Added `pulp = { version = "0.22.2", default-features = false,
features = ["x86-v3"] }` as a runtime dependency in `Cargo.toml`. pulp is
a pure-Rust portable SIMD abstraction crate (MIT/Apache-2.0, authored by
sarah-quinones, the author of faer) that wraps `core::arch::x86_64::*` and
`core::arch::aarch64::*` intrinsics behind a safe trait-based interface
(`pulp::WithSimd`, `pulp::Simd`, `pulp::Arch::dispatch`). It does
CPU-feature detection at runtime and dispatches to the best monomorphized
variant (AVX-512 / AVX2 / SSE2 / NEON / wasm SIMD / scalar fallback).
The pinned version exactly matches faer's `0.22.2`.

**Why.** The Phase 2.4.1a null result established empirically that scalar
loop reordering cannot produce a Schur-update speedup; faer-expert
confirmed that faer's entire blocked Bunch-Kaufman advantage lives in a
pulp-dispatched register-blocked SIMD GEMM at `bunch_kaufman/factor.rs:684`.
The Phase 2 exit criterion (dense factor p90 ≤ 2× MUMPS) therefore
requires a vectorized inner kernel. Options evaluated in
`dev/research/phase-2.4.2-simd-schur-kernel.md`:

1. **Hand-rolled `core::arch::x86_64` AVX2/FMA + `core::arch::aarch64`
   NEON intrinsics**, gated by `#[cfg(target_arch)]` and
   `#[target_feature]`, dispatched via `is_x86_feature_detected!`. This
   keeps zero new deps but introduces `unsafe` blocks into `src/`, two
   separate kernels to maintain, and no path to AVX-512 without a third
   kernel. Estimated time well beyond the Phase 2.4.2 budget.
2. **pulp.** One kernel, cross-arch for free, no `unsafe` in feral source,
   AVX-512 scaling automatic, ~10× less code, already audited at scale
   inside faer.

pulp wins on every practical axis. The only cost is one more crate in
the dependency graph and one more external project we trust — both
acceptable since pulp is pure Rust, widely deployed, and does not
violate the CLAUDE.md "zero non-Rust deps in the core solver" rule
(which exists to rule out BLAS, LAPACK, and Fortran, not pure-Rust
utility crates).

**Interface boundary.** The entire pulp dependency is confined to
`src/dense/schur_kernel.rs`, which exposes two `pub(crate)` functions:

- `axpy_minus(dst: &mut [f64], src: &[f64], alpha: f64)`
- `axpy2_minus(dst: &mut [f64], src0: &[f64], alpha0: f64, src1: &[f64], alpha1: f64)`

No other file in `src/` references `pulp`. Callers use only these two
functions. This keeps the dep swappable.

**Replacement trigger (future work).** If feral ever needs to ship as a
zero-external-dep crate — e.g., embedded, hardened, or compliance
environments that restrict supply-chain surface — replace pulp with
hand-rolled AVX2/FMA and NEON kernels at that time. The swap is
mechanical because of the interface boundary above: rewrite the two
functions in `src/dense/schur_kernel.rs` using `core::arch` intrinsics
with `#[target_feature]` + `is_x86_feature_detected!`, and delete the
pulp line from `Cargo.toml`. No call sites change. Tracked as a future
activity but not scheduled.

**Evidence.** Full research note at
`dev/research/phase-2.4.2-simd-schur-kernel.md`; implementation plan
at `dev/plans/phase-2.4.2-simd-schur-kernel.md`. Phase 2.4.1a
post-mortem establishing the necessity of a SIMD kernel is in
`dev/tried-and-rejected.md`. Commit introducing the dep: see Phase
2.4.2 Step 1 commit message.

## 2026-04-14 — Phase 2.4.3: Schur SIMD kernel must use separate mul + sub, not FMA

**Decision.** The production `do_1x1_update` / `do_2x2_update` hot-path
wiring uses `axpy_minus_unroll4_nofma` / `axpy2_minus_unroll4_nofma`,
the 4-way-unrolled pulp kernels whose inner body issues separate
`simd.mul_f64s` + `simd.sub_f64s` instead of a fused
`simd.mul_add_f64s`. FMA variants (`axpy*_minus_unroll4`) remain in
`schur_kernel.rs` and the microbench but are not called from
production code.

**Why.** Phase 2.4.2 wired the FMA variants into factor.rs and hit both
Phase 2.8 exit targets (dense p90 2.27 → 1.87, sparse p90 3.18 → 2.82)
but regressed sparse inertia from 153009 → 153005 and sparse residual
pass from 154329 → 154303 on 154588 KKT matrices. Per-matrix triage
identified the 4 inertia regressions as single-pivot boundary flips
on ACOPP14_0001, ACOPP30_0004, FBRAIN3LS_0848, FBRAIN3LS_0851 — all
caused by the well-known 1-ULP difference between one-rounding FMA
and two-rounding mul+add at pivots whose Schur-updated value lies
within a ULP of 0 or `zero_tol`. Full writeup in
`dev/tried-and-rejected.md` 2026-04-14 Phase 2.4.2 entry.

Non-FMA unroll4 fixes the root cause by reproducing the scalar loop's
rounding exactly:

| loop form                      | effective expression                           |
|--------------------------------|------------------------------------------------|
| scalar `d[i] -= α*s[i]`        | `round(d − round(α·s))` (two roundings)        |
| FMA `mul_add_f64s(−α, s, d)`   | `round(−α·s + d)` (one rounding)               |
| nofma `sub(d, mul(α, s))`      | `round(d − round(α·s))` (two roundings)        |

The nofma lane-wise operation is bit-identical to the scalar loop, so
any number of independent unrolled accumulators produce bit-identical
results across the length sweep. Verified by `assert_eq!` bit-exact
tests at `src/dense/schur_kernel.rs`
`axpy{,2}_minus_unroll4_nofma_is_bit_exact_vs_scalar` over lengths
{0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127,
128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024} — the length
sweep crosses every plausible SIMD register boundary (SSE2 f64x2,
NEON f64x2, AVX2 f64x4, AVX-512 f64x8) plus one-past-boundary sizes.

**Measured end-to-end result.** Full KKT bench (154588 sparse, 154481
dense, M-series aarch64), baseline commit `ce09aa6`:

| metric                  | baseline | nofma   |      Δ   |
|-------------------------|---------:|--------:|---------:|
| dense factor/MUMPS p90  |     2.27 |    1.86 |  −18.1%  |
| sparse factor/MUMPS p90 |     3.18 |    2.82 |  −11.3%  |
| dense factor geomean    |     0.23 |    0.22 |   −4.3%  |
| sparse factor geomean   |     0.67 |    0.63 |   −6.0%  |
| dense inertia match     |   152911 |  152911 |    0     |
| sparse inertia match    |   153009 |  153009 |    0     |
| dense residual pass     |   154207 |  154207 |    0     |
| sparse residual pass    |   154329 |  154329 |    0     |

Both Phase 2.8 exit targets (dense ≤ 2.0, sparse ≤ 3.0) hit. Zero
correctness regressions — every match and pass count is bit-identical
to the pre-kernel scalar baseline at commit `ce09aa6`. The bit-exact
rounding guarantee at the unit-test level translates to bit-exact
pivot classification at the factorization level.

**Cost in perf vs FMA.** Dense p90 moved 1.87 → 1.86 (FMA → nofma),
sparse p90 stayed at 2.82. Nofma is not measurably slower than FMA
end-to-end on the M-series NEON pipe — two operations (mul, sub) can
issue in parallel with the 4 independent accumulators, so the
apparent 2× instruction-count penalty is absorbed by ILP. On an
AVX-512 x86 machine the FMA-vs-nofma gap may be larger and the
decision may need to be revisited; for now the Apple Silicon
development target shows zero performance cost from the correctness
fix.

**Interface boundary.** The pulp boundary established in the
2026-04-14 Phase 2.4.2 decision is unchanged. `src/dense/schur_kernel.rs`
still exposes only the axpy-style functions; factor.rs calls them via
the `schur_kernel::` path with no direct pulp reference.

**Open question.** If a future target shows a material FMA-vs-nofma
gap and a way to preserve bit-exact rounding is found (e.g., a
correction term for the second rounding, or a detect-and-fall-back
near `zero_tol`), revisit. Not scheduled.

**Evidence.** Bench output `/tmp/feral_bench_nofma.txt`; 4 ULP4 +
2 bit-exact unit tests pass under `cargo test --lib schur_kernel`;
Phase 2.4.2 Step 5 triage (the failed FMA wiring) documented in
`dev/tried-and-rejected.md` 2026-04-14 Phase 2.4.2 entry.
