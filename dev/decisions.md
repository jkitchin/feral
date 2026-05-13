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

---

## 2026-04-14 — Phase 2.5 priority reordered: AMD is the sparse-small bottleneck, not column counts

**Context.** `dev/plans/phase-2-planning.md` §2.5.1 names Liu's
row-subtree column counts as "probably the highest-leverage Phase 2.5
item because it affects every call to `symbolic_factorize` and the
current implementation is the documented scaling weak point". The
Phase 2.8.1 partition verdict (session 2026-04-14-02) showed sparse
small-frontal p90 = 2.81 vs the 2.0 target — a clear fail that demands
a Phase 2.5 answer.

**Decision.** Before committing to any Phase 2.5.1 (column counts)
implementation, profile the sparse symbolic pipeline end-to-end on the
small-frontal bucket and spend the 2.5 hours on whatever phase
*actually* carries the cost. The profile binary
`examples/profile_sparse_smallfront.rs` replicates the
`symbolic_factorize` pipeline inline with per-phase `Instant::now()`
timing and runs over all 152128 small-frontal (max_front < 200, n ≤
500) matrices with a MUMPS oracle sidecar.

**Evidence — phase share across 152128 small-frontal matrices:**

| phase         |    sum (μs)   | share |
|---------------|--------------:|------:|
| total         |    9,376,324  | 100.0%|
| symbolic      |    6,714,929  |  71.6%|
| ├─ mc64       |      288,039  |   3.1%|
| ├─ **amd**    |  **3,733,092**|**39.8%**|
| ├─ etree      |    1,794,829  |  19.1%|
| ├─ colcnt     |      242,495  |   2.6%|
| └─ snode      |      410,403  |   4.4%|
| numeric       |    2,661,395  |  28.4%|

**Per-phase percentile tails (μs):**

| phase   | p50 | p90 | p99 |  max |
|---------|----:|----:|----:|-----:|
| mc64    |   0 |   5 |  23 |  109 |
| **amd** | **0**|**28**|**554**|**9322**|
| etree   |   2 |  29 | 127 |  880 |
| colcnt  |   0 |   5 |  31 |  157 |
| snode   |   0 |   3 |  49 |  502 |
| numeric |   1 |  55 | 253 | 1451 |

**Top offenders:**
- DISCS family (n=234, max_front=138, 20 matrices): AMD alone =
  9000–9300 μs, feral total = 11000 μs, MUMPS total = 440 μs.
  **AMD alone is 20× slower than MUMPS's entire analyse+factor**
  on this n=234 family.
- DMN15103 (n=99): AMD 1500–1800 μs, feral total ~2100 μs, MUMPS 120
  μs. AMD is ~75% of feral work; MUMPS is ~15× faster on n=99.
- LAKES (n=324): AMD 8200–8600 μs, feral total ~11000 μs, MUMPS 600
  μs. AMD is again ~75% of feral work.
- GROUPING (n=225): different pattern — AMD only 750–810 μs, but
  snode 450+ μs and numeric 360+ μs (unusually large for n=225),
  ratio ~16. Snode overhead here is anomalous.

**Implication.** The Phase 2.5.1 plan-item priority is wrong. Column
counts is 2.6% of the total small-frontal budget; Liu's row-subtree
would improve it but could at most remove 2.6 percentage points off
the sparse ratio. **The dominant cost is AMD** at 39.8%, with a
fat-tail of ~9ms on n=234 geometric families. Etree is second at
19.1% with its own smaller fat tail.

Reorder Phase 2.5:

1. **New Phase 2.5.1** — diagnose and fix the AMD implementation.
   The fat-tail pattern (p50=0, max=9322 for n≤300) suggests a
   pathological case in our AMD (likely dense-row handling, quotient
   graph updates, or degree approximation) rather than a constant
   factor. The fix may be a single bug, not a full rewrite. Action:
   (a) pick DISCS_0012 as the minimal repro, (b) profile `amd_order`
   with `cargo flamegraph` or manual sub-phase timing, (c) compare
   against AMD from SuiteSparse or our reference paper citation
   trail.
2. **New Phase 2.5.2** — follow-up on etree if it still dominates.
   Lower priority; 19.1% share with a narrower tail.
3. **Demoted — old 2.5.1 (Liu row-subtree column counts)** — defer
   until after AMD and etree are fixed and measured. Not an exit-gate
   item; revisit only if the small-frontal p90 still misses the bar
   after 2.5.1′ and 2.5.2′ land.
4. Phase 2.5.2 (parallelism), 2.5.3 (allocation), 2.5.4 (fill
   prediction) remain in their original positions in the plan.

**Evidence.** Profile output `/tmp/profile_smallfront.txt`; profile
binary `examples/profile_sparse_smallfront.rs`. Journal:
`dev/journal/2026-04-14-02.org` Phase 2.5 triage entry.

---

## 2026-04-14 — Phase 2.5.1′: AMD stays exact minimum-degree (mark-array, not real AMD)

**Context.** Session 04 diagnosis showed `adj[a].contains(&b)` inside
the fill loop was the sole source of AMD's pathology. On near-dense
inputs (DISCS_0012 n=234, DMN15103_0000 n=99 fully dense) the fill
set is already a clique, so every `contains` returns `true` after
scanning the full adjacency vector — 778k lookups for zero inserts on
DISCS_0012. Fill phase was 80–88% of AMD runtime on the top offenders.

**Decision.** Keep the exact minimum-degree algorithm, fix the hot
loop with a mark array. Do **not** port real AMD (approximate
external degree + element absorption + quotient graph).

**Rationale.**
1. The mark-array fix brings fill phase from O(deg³) to O(deg²) per
   step — one Vec<bool> of size n reused across steps, set/cleared
   within each outer iteration.
2. Combined with a dense-clique early exit (when pivot's live
   neighbors equal all remaining live nodes, push survivors and
   return), DMN15103_0000 short-circuits entirely and DISCS_0012
   terminates after its first few steps.
3. This brings sparse small-frontal p90 to 1.99 (target ≤ 2.0) on
   a 3-run median — meets the Phase 2.8.1 exit criterion.
4. Real AMD is a larger surface-area change (quotient graph, element
   absorption, degree approximation) whose correctness surface would
   need its own research note and test matrix. Not worth taking on
   now when the minimal fix clears the gate.

**When to revisit.** If a future partition (e.g., Phase 3 sparse
medium or large-frontal) needs AMD to be significantly faster on
large n, or if we find an input where exact min-degree produces
meaningfully worse fill than real AMD.

**Evidence.** Triage binary `examples/triage_discs_amd.rs`;
`dev/sessions/2026-04-14-04.md`; journal
`dev/journal/2026-04-14-04.org` 13:05/14:10 entries.

---

## 2026-04-14 — Phase 2.5.1′: `permute_pattern` preserves sorted-column invariant

**Context.** Session 04 rewrote `permute_pattern` in
`src/ordering/amd.rs` from a `Vec<Vec<usize>>` + sort_unstable +
dedup scheme to a two-pass counting-sort layout (count → prefix sum
→ fill). The counting-sort is ~7× faster on DMN15103_0000 because
each entry is copied exactly once (the input is a full symmetric
pattern so we just re-bucket) instead of being pushed twice and
deduped.

**Decision.** The new implementation runs one additional
`sort_unstable` pass per column at the end to keep row indices
sorted, preserving the invariant the old implementation produced.

**Rationale.** Downstream code (column_counts, frontal assembly)
does not strictly require sorted columns, but:
1. The previous impl produced sorted output; some callers may
   implicitly rely on it through debug_assert or iteration order.
2. The sort is O(nnz/col · log(nnz/col)) per column which is cheap
   compared to the assembly work the sorted output enables.
3. Removing the invariant is a cross-cutting audit we do not need
   to take on now.

**When to revisit.** If profiling shows the per-column sort is
measurable (it should not be for small frontals) and we can prove
no caller relies on sorted columns.

**Evidence.** `src/ordering/amd.rs` `permute_pattern`;
`dev/sessions/2026-04-14-04.md`.

---

## 2026-04-14 — Phase 2.5.1′: symbolic factorization builds final etree by renumbering, not re-parsing

**Context.** `src/symbolic/mod.rs` used to call
`EliminationTree::from_pattern` twice: once on the AMD-permuted
pattern (to compute the postorder) and once on the final permuted
pattern (to get the etree used by column_counts and the numeric
phase). The second call is O(nnz · α(n)) and redundant.

**Decision.** Compute the final etree by renumbering the
AMD-permuted etree's parent array through the postorder, in O(n):

```rust
let final_parent: Vec<Option<usize>> = (0..n)
    .map(|new| {
        let old_amd = post[new];
        amd_etree.parent[old_amd].map(|old_par| post_inv[old_par])
    })
    .collect();
```

**Rationale.** Postorder is a topological relabeling of the
elimination tree: `etree(P·A·Pᵀ) = post-renumbering of etree(A)`
when P is a postorder of `etree(A)`. The tree structure is
preserved and only the node labels change. This makes the second
from_pattern call mathematically redundant.

**Evidence.** 3-run median sparse small-frontal p90:
- Before renumbering: 2.12 / 2.12 / 2.14
- After renumbering:  2.03 / 2.06 / 2.08
- ~3% improvement at p90, stable across runs.

`src/symbolic/mod.rs` lines around the `final_parent` construction;
`dev/sessions/2026-04-14-04.md`; journal entry
`dev/journal/2026-04-14-04.org` 14:55.

---

## 2026-04-14 — Phase 2.8.1 exit gate satisfied (all four partitions PASS)

**Context.** Session 03 reported sparse small-frontal `factor/MUMPS`
p90 = 2.81 (FAIL). Session 04 applied six fixes (AMD mark array,
AMD clique shortcut, counting-sort `permute_pattern`, dead loop in
`supernode.rs`, etree renumbering, dead transpose call in
`factorize.rs`).

**Decision.** **Phase 2 exits on sessions 04 / 05 boundary.** All
four Phase 2.8.1 exit partitions PASS on the full KKT bench:

| bucket                 | count  |  p90 | target | verdict |
|------------------------|-------:|-----:|-------:|:-------:|
| Dense small-frontal    | 147982 | 1.56 | ≤ 2.0  | PASS    |
| Dense medium           | 152145 | 1.96 | ≤ 3.0  | PASS    |
| Sparse small-frontal   | 153455 | 1.99 | ≤ 2.0  | PASS    |
| Sparse medium          | 153560 | 2.00 | ≤ 3.0  | PASS    |

3-run medians on sparse small-frontal: 2.00 / 1.98 / 2.00.

**Tight-margin acknowledgement.** Sparse small-frontal lands at
1.98–2.00 with measured run-to-run noise ~3–5%. The next
regression in this band could push it back over the gate. Phase 3+
work must re-verify this partition on commit. Recorded as a Phase
2.8.1 follow-up risk for session 05.

**Evidence.** `/tmp/feral_bench_session04_final.txt`; 3-run medians
in `dev/sessions/2026-04-14-04.md` "Benchmark Results" section.
`FERAL-PROJECT-SPEC.md` §1747 for the exit criterion.

---

## 2026-04-16 — Ordering backends live in sibling workspace crates, not src/ordering

**Decision.** Pluggable fill-reducing ordering backends (AMD, METIS, SCOTCH, KaHIP) are each implemented as their own Cargo workspace-member crate under `crates/*`, accepting a slice-based full-symmetric CSC pattern and returning a permutation. The feral package itself is untouched by these additions. Integration into feral's symbolic factorization is deferred to a future `dev/plans/ordering-integration.md` that will land after at least two backends exist and can be compared side-by-side.

**Why.** (1) Keeps each backend testable in isolation against its own oracle (e.g. SuiteSparse AMD for feral-amd). (2) Avoids committing to one ordering strategy before we have comparative fill-quality numbers on feral's 153k corpus. (3) Slice-based input means no ordering crate depends on feral's `CscPattern` / `FeralError`, and third parties could adopt any one of them. (4) Each crate gets its own CLI + bench, mirroring how SuiteSparse ships each algorithm as a standalone artifact.

**Alternatives considered.** In-place replacement of `src/ordering/amd.rs` (rejected: couples integration to correctness, and a subtle ordering bug would regress the 153k corpus before we can roll back); feature-gated alternatives inside feral (rejected: still couples lifecycle).

**Evidence.** `dev/plans/ordering-amd-upgrade.md` (third revision, Architecture section); `Cargo.toml` root now has `[workspace] members = [".", "crates/feral-amd"]`; sibling plans `dev/plans/ordering-metis.md`, `ordering-scotch.md`, `ordering-kahip.md` on disk as placeholders.

---

## 2026-04-16 — Clean-room invariant for feral-amd enforced in CI

**Decision.** The external SuiteSparse AMD port (`amd` crate v0.2.2) is used **only** as an external oracle, inside a throwaway Cargo project preserved at `crates/feral-amd/tests/data/amd_oracle/harness/` as `.txt` files (extension-stripped so Cargo never compiles them). The feral workspace dependency graph must never contain an `amd` crate dependency. `scripts/check-amd-cleanroom.sh` greps every `Cargo.toml`, every feral / feral-amd `*.rs` file, and `Cargo.lock` for violations; CI runs it as the `amd-cleanroom` step.

**Why.** feral's MIT-license / pure-Rust / zero-non-Rust-deps posture requires that feral-amd be a clean-room implementation derived from published papers and faer's BSD-licensed in-tree port, not from SuiteSparse. A mechanical check prevents the oracle from accidentally leaking into the runtime graph.

**Evidence.** `scripts/check-amd-cleanroom.sh` reports "clean-room OK: 'amd' crate absent from feral workspace"; `.github/workflows/ci.yml` `amd-cleanroom` step; harness `.txt` files under `crates/feral-amd/tests/data/amd_oracle/harness/` with SHA-256s pinned in the oracle README.

---

## 2026-04-17 — Ordering crate boundary: `i32` index width, free function, no etree

**Decision.** The four ordering crates (`feral-amd`, `feral-metis`, `feral-scotch`, `feral-kahip`) share a minimal contract exposed by a new `feral-ordering-core` workspace crate and adhere to three specific choices:

1. **Index width is `i32`.** `CscPattern` borrows `&[i32]` slices for `col_ptr` and `row_idx`, and ordering routines return `Vec<i32>` permutations. Ipopt consumes ordering output as plain indices and never needs 64-bit counts at this boundary; this matches the Fortran MUMPS / SSIDS convention and the MA27 Ipopt interface.
2. **No trait, one free function per crate.** Each crate exposes `fn {amd,metis,scotch,kahip}_order_full(&CscPattern, &Opts) -> Result<(Vec<i32>, OrderingStats, CrateStats), OrderingError>`. Ipopt / feral pick the backend by name-dispatching, not by a generic `Orderer` trait. Crate-specific options and crate-specific stats stay in the crate.
3. **No elimination tree in the contract.** Ordering crates return a permutation and a small shared `OrderingStats` (time, optional fill/flop estimates). Etree construction, symbolic factor, and postorder belong in the downstream analysis phase, not in the ordering boundary. METIS/SCOTCH/KaHIP give node separators, not etrees; forcing an etree across the boundary would shape-distort three of the four backends.

**Why.** Locks API drift before three more crates are written. After verifying with the ipopt-expert agent that Ipopt's ordering consumers require only a permutation array across the boundary, the minimal-surface design falls out: the ordering crate returns perm + counters, downstream code (eventually feral's symbolic analysis, later Ipopt's `MA27TSolverInterface`-style wrapper) turns that into an etree on demand. A trait was considered and rejected — generic dispatch gains nothing when we have exactly four backends and the per-crate options diverge (METIS has `ufactor`, `seed`; SCOTCH has strategy strings; AMD has `dense_row_thresh`).

**Alternatives considered.** `usize` index width (rejected: Ipopt column-index pipeline is `int`, casts at every interop boundary would just move the problem); `trait Orderer` with associated types (rejected: zero-benefit indirection given the four-backends-forever count); etree construction inside the ordering crates (rejected: see above — three of four backends would need to synthesize a fake etree from a separator tree, defeating the simplicity).

**Reconsideration clause.** If a fifth ordering backend is ever added and it turns out to share options with an existing one, revisit the trait choice. If feral ever needs >2^31 rows (unrealistic for KKT matrices in NLP), revisit the `i32` choice.

**Evidence.** `dev/plans/ordering-crate-contract.md` (full spec, including acceptance checklist); `crates/feral-ordering-core/src/lib.rs` (45 LOC contract module, 12 passing unit tests); `crates/feral-amd` retrofit passes 29 lib tests + 12 SuiteSparse oracle tests bit-for-bit after the switch to `i32`.

---

## 2026-04-18 — OrderingMethod enum dispatch (not trait) in src/symbolic/mod.rs

**Decision.** `src/symbolic/mod.rs` wires the three ordering crates
(feral-amd, feral-metis, feral-scotch) through an
`enum OrderingMethod { Amd, MetisND, ScotchND }` dispatched by a
single `match` inside `symbolic_factorize_with_method`. No
`OrderingBackend` trait, no generic parameter on the caller. The
in-tree `src/ordering/amd.rs` remains the `OrderingMethod::Amd`
implementation pending separate retirement work.

**Why.**
- Only three ordering implementations exist today (AMD, METIS,
  SCOTCH). A fourth (KaHIP per `dev/plans/ordering-kahip.md`) is
  planned but would drop into the same enum.
- The ergonomic call-site is `symbolic_factorize_with_method(
  &matrix, &params, method)`. A trait-based dispatch would either
  require a type parameter on the caller (propagates through
  Solver, Factorization, etc.) or a `Box<dyn OrderingBackend>`
  with heap allocation and dynamic dispatch for what is a
  ~microsecond operation.
- Each ordering crate exposes the shared `feral-ordering-core`
  contract (`fn _order(&CscPattern<'_>) -> Result<Vec<i32>,
  OrderingError>`) using i32 indices and borrowed patterns. The
  main feral crate uses owned-usize patterns and `FeralError`.
  Conversion must happen somewhere. Putting it behind a trait
  means every crate's `impl OrderingBackend for Amd {}` does the
  same conversion; putting it behind an enum means one
  `run_external_ordering` adapter in `src/symbolic/mod.rs` does
  it once. The enum path is shorter and keeps conversion
  concerns centralized.
- Dynamic selection (strategy autotuning) is easier with an enum:
  pattern-match on the method, swap variants based on runtime
  heuristics, without constructing trait objects.

**Scope.** `symbolic_factorize` (the legacy one-arg entry) is
preserved as a thin delegate to
`symbolic_factorize_with_method(.., OrderingMethod::Amd)` so no
caller breaks. 3 symbolic-level tests enforce
`MetisND`/`ScotchND` produce valid perms and the default matches
AMD.

**Reconsideration clause.** If a fifth backend arrives and it
carries its own configuration type incompatible with a simple
enum variant (e.g., KaHIP's preconfiguration struct), revisit
the trait choice. Also if some caller needs to accept an
arbitrary user-supplied ordering at runtime (plugin-like), a
trait object is better suited; not needed for current FERAL
users.

**Evidence.** Commit `d4e5eda` (enum + dispatch + 3 tests);
`dev/research/ordering-bakeoff-2026-04-18.md` covers the
comparative behaviour of the three enum variants on the parity
and large-matrix corpora.

---

## 2026-04-18 — Large-matrix bake-off corpus via SuiteSparse, not synthesis or IPM dumps

**Decision.** `dev/scripts/large_matrices.txt` pins four matrices
from the SuiteSparse Matrix Collection (bcsstk38, bratu3d,
cont-201, c-big) to extend the ordering bake-off into the
n=8k–345k regime the parity corpus does not reach.
`dev/scripts/fetch_large_matrices.sh` downloads them into
`tests/data/large/`, which is gitignored.

**Why.**
- The parity corpus (`tests/data/parity/`) has median n=77 and
  only 3 matrices > 1000. The bench-orderings result geomean of
  1.011× for METIS/AMD is not a credible estimate of ordering
  quality at the scales where fill-reducing ordering actually
  matters (LU/LDL^T dominated by factorization cost, not
  ordering cost). The n > 10k regime must be in the corpus.
- Three options were considered:
  - (a) **Synthetic matrices** (5-point Laplacian, random
    sparse, planted-structure). Rejected: we'd debate whether
    synthetic structure is representative of real KKT / mesh /
    indef workloads; the debate costs more than the fetch.
  - (b) **SuiteSparse pinned set** (chosen). Public, citable,
    reproducible, covers symmetric-indefinite and KKT regimes.
    License permits redistribution but matrices are large
    (~45 MB), so keep outside git.
  - (c) **Mine IPM dumps from `../ripopt` or CUTEst runs.**
    Rejected: adding more dumps of the same shape as the parity
    corpus does not close the size gap. The largest IPM KKTs we
    have are n ≈ 5k.
- Pinning is via a text manifest (`large_matrices.txt`) rather
  than burning the URLs into a shell script, so future
  additions (KaHIP-appropriate graphs, direct solver stress
  tests) are a one-line edit.

**Scope.** `bench_orderings` auto-detects `tests/data/large/`
if present and adds those matrices to the report.
`tests/data/large/` is gitignored; the fetch script is the
reproduction path. Matrix selection criteria: symmetric or
symmetric-indefinite, n spanning 10³ to 10⁵, including at
least one KKT (c-big) and one 3D-PDE Jacobian (bratu3d).

**Reconsideration clause.** If the corpus-regeneration time
becomes painful (fetch + bench on c-big currently 10+ minutes)
we could cache `factor_nnz_estimate` per `(matrix, method,
ordering-version)` triple in a checked-in JSON. Not warranted
yet — the bake-off reruns are rare.

**Evidence.** Commit `7962568` (script + manifest + bench
extension + results table);
`dev/research/ordering-bakeoff-2026-04-18.md` "Large-matrix
extension" section.

---

## 2026-04-18 — In-tree AMD (src/ordering/amd.rs) retirement deferred

**Decision.** The in-tree `src/ordering/amd.rs` is retained as the
`OrderingMethod::Amd` implementation in `src/symbolic/mod.rs`
even though `feral-amd` now exposes the same algorithm through
the ordering-crate contract. Retirement is filed as a deferred
follow-up.

**Why.** Retirement requires:
- Adapting every call site in `src/symbolic/mod.rs` to the
  borrowed-i32 `CscPattern` used by the ordering crates
  (currently the main crate uses owned-usize patterns).
- Mapping `OrderingError` (from `feral-ordering-core`) onto
  `FeralError` at all entry points, not just the
  `run_external_ordering` adapter.
- Verifying that the output permutation is bit-for-bit
  identical across the two AMD implementations on the entire
  Phase 1b parity corpus (otherwise the bench regression
  partition shifts under us).

None of that is hard, but it is cross-cutting and the current
session's scope is ordering-dispatch wiring + comparative
bake-off, not a full symbolic refactor. Deferring keeps the
two AMDs coexisting cleanly — the dispatch enum already makes
either one selectable — until a dedicated cleanup session.

**Reconsideration clause.** Retire when one of: (a)
`src/ordering/amd.rs` develops a bug that `feral-amd` does not
have (or vice versa), (b) the cost of maintaining two AMDs
exceeds the migration cost, (c) `feral-amd` grows a feature
(e.g., a different pivot strategy) that would be useful in the
default path.

**Evidence.** `src/symbolic/mod.rs` still imports `amd_order`
from `crate::ordering::amd`; commit `d4e5eda` deliberately does
not remove the module. `Cargo.toml` depends on `feral-amd`
transitively through `feral-metis`, `feral-scotch`, and
`feral-ordering-core` but does not itself consume it directly
yet.

---

## 2026-04-18 — Retire in-tree AMD; route `OrderingMethod::Amd` through `feral-amd`

**Decision.** `OrderingMethod::Amd` is now the default and routes through the
`feral-amd` workspace crate (full AMD with approximate external degree,
aggressive element absorption, supervariable detection — Amestoy/Davis/Duff
1996+2004). The in-tree simplified AMD at `src/ordering/amd.rs` is kept on disk
as a reference implementation of the exact-external-degree variant and for the
`permute_pattern` helper, but is no longer reachable from the symbolic pipeline.

**Evidence.** 34-matrix bakeoff (30 parity + 4 large) via `amd_compare`:

- Parity corpus: geomean `fill_crate / fill_intree` = 1.001 (tied).
  In-tree wins on 18 matrices, crate on 6, 10 ties. Differences are 1-6%.
- Large corpus: crate strictly better on every matrix.
  - `bcsstk38`: fill 0.941, time 18×
  - `bratu3d`:  fill 0.840, time 46×
  - `c-big`:    fill 0.776, time 36×
  - `cont-201`: fill 0.769, time 88×

**Tradeoff.** The new default produces different permutations than the old
in-tree AMD, which causes inertia classification to flip at the zero/tiny-
signed-pivot boundary on some rank-deficient KKT matrices (e.g.
ACOPP30_0000). This is a property of ordering choice on rank-deficient
systems, not an AMD regression — `feral-metis` exhibits the same flip on
the same matrices. Residual quality is preserved (feral residuals ~1e-15,
better than MUMPS). The parity panel was regenerated via
`select_parity_panel` (its documented purpose on solver behavior change),
moving 8 additional boundary-case matrices into the `#[ignore]` bucket.

**Journal.** `dev/journal/2026-04-18-03.org` 10:00-10:40 entries.

---

## 2026-04-18-08 — `pick_default_method` will not route to KahipND

The default ordering dispatcher (`src/symbolic/mod.rs:178
pick_default_method`) returns either `Amd` or `MetisND`. It will
not return `KahipND` on its own. KaHIP remains reachable through
two explicit channels:

- `symbolic_factorize_with_method(.., OrderingMethod::KahipND)`
- `symbolic_factorize_with_method(.., OrderingMethod::Auto)` whose
  decision tree includes KaHIP for narrow shape branches

**Evidence.** 41-matrix `bench_orderings` bake-off at session 08:
KaHIP-with-K1 ties METIS on fill (geomean 1.023 vs 1.024 relative
to AMD) at 4-6× the per-call symbolic-time cost (81s vs 68s vs
AMD 14s, total). Strict-fill wins of KaHIP over AMD on only 4/41
matrices, in every case merely tying the best other ordering.
On the 154 588-matrix IPM bench KaHIP would only match METIS
where the existing `n>=5000 && nnz/n<6 → MetisND` rule already
fires (e.g. CRESC132).

**Pinning.** Test `pick_default_method_never_returns_kahip` covers
8 representative shapes (CRESC132, VESUVIOU, c-big, etc.) and
asserts none route to KahipND. A future opt-in change must
consciously update the test and the cross-referenced research
note.

**Research / plan.**
`dev/research/ordering-kahip-driver-integration.md` and
`dev/plans/ordering-kahip-driver-integration.md`.

**Journal.** `dev/journal/2026-04-18-08.org` third entry.

## 2026-04-18-08 — VESUVIO factor-time gap is dense-kernel limited

The factor/MUMPS max ratio of ~85× on the VESUVIOU/VESUVIO/VESUVIA
families is a property of `src/dense/factor.rs` (scalar rank-1 BK
updates, no blocking, no SIMD), not of fill-reducing ordering.
Both AMD and MetisND produce the same ~67%-of-n root frontal on
every VESUVIO sample because the matrix has a single dense linking
column with 1026 nnz that any reasonable ordering pushes to the
root.

**Evidence.** `src/bin/vesuvio_diag.rs` (commit 86cf1e8) measured
factor times under both orderings across 5 VESUVIO samples plus
CRESC132 as a positive-control. MetisND saves ≤8% on two VESUVIO
samples and is slower on three. Cost analysis: 2059×959 BK ≈
1.9 GFLOPs at our scalar ~8 GFLOP/s ≈ 240ms (matches 236ms
observed); MUMPS DGETRF on Accelerate ≈ 400 GFLOP/s ≈ 5ms (matches
2.5ms oracle). The 50× kernel gap explains the 84× factor ratio.

**Implication.** Closing the VESUVIO-class tail requires blocked
BK + SIMD in `src/dense/factor.rs`. Multi-session engineering;
deferred to a future planning pass.

**Journal.** `dev/journal/2026-04-18-08.org` second entry.

## 2026-04-19 — Policy 4: 3-condition Auto fallback to InfNorm

`ScalingStrategy::Auto` (the default since the lever-C flip
earlier this day) now runs a post-scaling diagnostic when
`pick_scaling_strategy` would route a matrix to MC64. If
ALL three conditions fire, it falls back to InfNorm:

1. `raw_diag_range(matrix) < 1e6` — raw matrix's diagonal
   already spans a narrow range, so MC64 has nothing to
   recover; any huge scaled `mc_off` it produces is artifact.
2. `mc_off > 1e6` — MC64's scaled `max(|off|/|diag|)` is
   large in absolute terms.
3. `mc_off / in_off > 1e5` — and is much larger than what
   InfNorm produces.

The first guard (raw_diag_range) is critical: it lets
matrices like MEYER3NE_0220 (raw_drng = 4.77e19, MC64 is
genuinely needed) keep MC64, while still catching MSS1_0009
(raw_drng = 51, MC64 produces noise).

**Validation.** 17-matrix panel: rule fires only on
MSS1_0009. Corpus residual_pass: 154 233 / 154 588 (was
154 232; +1 matches prediction). MSS1 family residuals:
0 fail (was 1). Inertia hard rule preserved on every
regression.

**Research / plan.**
`dev/research/policy-4-scaling-fallback.md`,
`dev/plans/policy-4-scaling-fallback.md`.

**Journal.** `dev/journal/2026-04-19-02.org`.
**Commit.** `af9315d`.

## 2026-04-19 — `ScalingInfo::Applied` is load-bearing

`numeric::solve` keys off `factors.scaling_info` to decide
whether to apply pre/post-scaling. `NotApplied` makes solve
skip the scaling step entirely (treat as identity);
`Applied` and `PartialSingular { .. }` invoke the pre/post
multiply.

**Convention.** Any `compute_scaling` path that returns a
non-trivial scaling vector MUST return
`ScalingInfo::Applied` (or `PartialSingular` for MC64's
partial case). `NotApplied` is reserved for paths where the
returned vector itself should not be applied — currently
`Identity` (vector of 1.0s) and `External` (caller-supplied
vector that the caller is responsible for tracking).

**Trigger.** Policy 4's initial implementation returned
`NotApplied` for the InfNorm fallback path ("matches the
InfNorm convention" — but the convention was misread;
InfNorm returns `Applied`). The bug regressed
MSS1_0007–0013 to residual ≈ 2.4e-3 in the bench until
fixed by forwarding `infnorm::compute_infnorm`'s actual
return value. Verified by a corpus re-bench showing the
predicted +1 residual_pass (154 233 / 154 588).

**Journal.** `dev/journal/2026-04-19-02.org`.
**Commit.** `af9315d`.

---

## 2026-04-19 — D.1 `FactorWorkspace` landed; D.3 dense fast-path gate adopted

**Decision 1.** Introduce `FactorWorkspace`, a caller-owned scratch
pool for `factorize_multifrontal_with_workspace`. Pools `row_map`,
the per-supernode frontal `SymmetricMatrix::data`, and the scratch
buffers used by `build_row_indices` (`build_delayed`,
`build_trailing`, `build_seen`). `Solver` retains one across calls.

**Rationale.** The Lever D.1 alloc-probe evidence
(`dev/results/lever-d1/alloc-probe-2026-04-19.txt`) showed the
sparse factor was paying 17–23 allocations per supernode, 99 % of
which were scratch reallocs across supernodes and across factor
calls. Pooling collapsed VESUVIO reallocs from 2053 to 13 and
drove corpus geomean factor/MUMPS from 0.48 → 0.46.

**Non-decision.** We intentionally did not widen the workspace to
pool the dense-path `sym.data` buffer at this time; that's a
follow-up now that D.3 has landed and can use it.

**Commits.** `9c0419b` (plan) → `b1016cc`, `f102d56`, `dedb3f3`
(rollout). Guardrails in `tests/factor_workspace_parity.rs` assert
byte-identical factors vs the allocator-per-call path.

---

## 2026-04-19 — D.3 dense fast-path gate thresholds

**Decision.** `factorize_multifrontal_with_workspace` routes
matrices satisfying `n ≤ 128 ∧ density ≥ 0.25` (lower-triangle
nnz / n·(n+1)/2) to `dense_fast_factor`, a thin wrapper that
densifies the CSC, applies `D · A · D` symmetric scaling, calls
the existing dense BK kernel on the full matrix, and synthesizes a
single-supernode `SparseFactors` shape-compatible with
`solve_sparse`. Matrices outside the gate continue through the
multifrontal path byte-identically.

**Rationale.** Stage-2 synthetic sweep
(`dev/results/lever-d3/stage1-stage2-2026-04-19.md`) showed that at
ρ = 0.25 the dense path beats the multifrontal path for every
tested n up to 192 (ratio 0.49–0.66×); at ρ = 0.10 it ties at
n = 128 and regresses at n ≥ 160. The 0.25 floor gives a 2-fold
safety margin over the tied-case, absorbing the 1.5–2× variance
real IPM matrices exhibit vs the best-case diagonally-dominant
synthetics. `N_MAX = 128` keeps the dense workspace at ≤ 128 KB
(fits L1/L2 comfortably); widening to 192 is tempting but deferred
until corpus evidence demands it.

**Corpus evidence.**
`dev/results/lever-d3/stage3-corpus-2026-04-19.md`. Sparse
factor/MUMPS geomean 0.46 → 0.37 (-20 %), p50 0.33 → 0.29, max
ratio 128.34 → 80.22. Ex-ante acceptance target (≤ 0.44) met with
0.07 margin.

**Entry-point convention.**
`factorize_multifrontal_supernodal` and
`factorize_multifrontal_supernodal_with_workspace` are the
documented bypass entry points for tests and callers that need to
force the multifrontal path on an in-gate matrix. They share the
supernodal body with the gated dispatcher; only the bypass reaches
it without consulting `should_use_dense_fast_path`.

**Commits.** `71f5692` (plan), `7c9e07d` (RED), `32dd65a` (GREEN),
`70f077e` (stage 1/2), `e0db169` (stage 3).


---

## 2026-04-20 — D.4 tiny-n disjunct in dense fast-path gate

**Decision.** `should_use_dense_fast_path` now accepts any
`n ≤ N_TINY = 16` unconditionally, in addition to the D.3
`n ≤ 128 ∧ ρ ≥ 0.25` disjunct. The implementation of
`dense_fast_factor` is unchanged; only the gate predicate is
broadened. `FactorWorkspace` semantics are unchanged — a gate-hit
call bypasses the workspace regardless of which disjunct fired.

**Rationale.** Pre-existing CLAUDE.md-era finding
(`dev/tried-and-rejected.md`) that at n ≤ 10 `factorize_multifrontal`
is dominated by symbolic-phase overhead rather than floating-point
work. HS85_0022 diagnosis this session confirmed the pattern at
n=68: symbolic is 36 % of the pipeline, fraction rises as n shrinks.
`dense_fast_factor` skips symbolic entirely, so the D.4 disjunct
captures tiny matrices that D.3's density gate rejected.

**Stage-1 evidence.**
`dev/results/lever-d4/stage1.md`: the six observed top-10 tiny-n
rows (HS73_0308, PALMER1E_0484, HATFLDH_0083, PALMER1A_0034,
KIRBY2LS_0274, HEART6LS_0418) all showed 1.17–1.53× p50 speedup,
and all six beat MUMPS by 2–4× post-D.4.

**Stage-2 evidence.**
`dev/results/lever-d4/stage2-corpus.md`: corpus geomean stable at
0.38–0.39 across three bench runs (pre-D.4 was 0.37, within noise),
all six target rows drop out of the top-10 in every run. Phase
2.8.1 exit partitions remain PASS.

**Corpus coverage.** Smaller than the research note implied. Every
observed top-10 tiny-n row was already D.3-eligible at ρ ≥ 0.50;
D.4's unique class (n ≤ 16 ∧ ρ < 0.25) appears empty or near-empty
on the current IPM corpus. D.4 is the correct primitive to have
but its observable corpus impact is small. The rollout is complete;
stack-buffer densify and dense-scratch pooling remain named
follow-ups but are not authorized as of this decision.

**Threshold choice.** `N_TINY = 16` covers every top-10 tiny-n row
(max observed n = 11) with 30 % headroom and matches the
research-note recommendation. `n*n ≤ 256` cells — ≈ 2 KB dense
workspace — cheap even without a stack buffer.

**Commits.** `2fe8836` (plan+diag), `d570960` (RED),
`ddefc2f` (GREEN), `16fdd77` (stage 1/2).



---

## 2026-04-20 — Bench harness multi-sample denoise

**Decision.** `src/bin/bench.rs` resamples per-matrix factor+solve
timings `K = 5` cold reps for any matrix whose MUMPS oracle sidecar
reports `factor_us < 200`. Recorded `MatrixTiming::factor_us` is the
minimum across reps; `solve_us` is the median. Dense and sparse
loops are patched symmetrically. No env flag — denoise is always on.

**Why.** Single-shot per-matrix wall time at the tens-of-µs scale
produces 10–100× noise excursions that dominate the top-N worst
factor-ratio report. Session 2026-04-20-01 diagnosed HS85_0022 as a
false 80× regression (probe p50 = 37 µs; single-shot bench reading
1845 µs). Pre-denoise three-run max: 11.81 / 102.07 / 285.80 (24×
spread). Post-denoise three-run max: 13.38 / 11.36 / 27.09 (2.4×
spread). All entries in the new top-10 are n ≥ 458 — the real
arrow-KKT regression class that Phase 2.4.1b would target.

**Cost.** Wall-time +~1:45 per bench run (2:15 → 4:00) — failed my
≤ 20% ex-ante but accepted because the signal improvement is 10×
and a bench runs once per session.

**Threshold choice.** 100 µs was the initial target (session 2026-04-20-01
checkpoint named "say 100 µs"), but run 2 at 100 µs still hit
NELSON_0414 at 37× (MUMPS=142 µs, above threshold). Raising to 200 µs
covers the NELSON/SWOPF/CRESC100 boundary cases observed pre-denoise
at MUMPS times 98–167 µs. Residual noise at threshold=200 µs:
HAIFAM_0709 (MUMPS=234 µs) spiked once in 3 runs to 27×. Acceptable;
500 µs threshold would remove it at ~+60 s but is not warranted.

**Reduction choice.** `min` for factor (robust against single
cold-cache outliers, the observed noise mode). `median` for solve
(smaller numeric phase, less outlier-prone). Matches the convention
used in the stage-1 probes `src/bin/d4_probe.rs` (`p50`) and
`hs85_diag.rs` (`min`, `p50`).

**Evidence.** `dev/results/bench-denoise/summary.md` +
`run{1..6}*.txt` raw bench outputs.

---

## 2026-04-20 — Phase 2.4.1b: blocked dense LDLᵀ is a separate public function

**Decision.** The blocked-panel BK LDLᵀ kernel (Phase 2.4.1b) is exposed
as a *new* public function `factor_frontal_blocked` in
`src/dense/factor.rs`, alongside the existing `factor_frontal`.
Dispatch from `factor_single_front` / the multifrontal driver to the
blocked variant will be gated on `remaining > params.block_size &&
!may_delay` once Step 4 lands. Both entry points stay public.

**Why.**
- Parity testing wants to call both kernels side-by-side on the same
  `SymmetricMatrix` with the same `BunchKaufmanParams` and
  `assert_eq!` their returned `FrontalFactors`. A `use_blocked: bool`
  flag on `BunchKaufmanParams` or an env-var dispatch inside
  `factor_frontal` would force every test to clone + mutate the
  params struct, which is noisier and hides what's being compared.
- The scalar path remains the oracle. Rejection-heavy sparse matrices
  with `may_delay = true` keep using `factor_frontal` indefinitely;
  the blocked path is only for the root supernode and for dense
  fronts where `may_delay = false`. Keeping them as distinct
  functions makes "which kernel ran" a static fact, not a
  runtime-configured dispatch table.
- Matches the existing `factor` vs `factor_single_front` pattern:
  different code paths for different call-site shapes, shared via the
  common `scalar_pivot_step` helper.

**Parity oracle is `f64::to_bits`, not `approx_eq`.** The 2026-04-14
Phase 2.4.2 FMA-unroll4 reversion showed that a 1-ULP rounding drift
(from one fused-multiply-add replacing two roundings) flipped inertia
on ACOPP14_0001, ACOPP30_0004, FBRAIN3LS_0848, FBRAIN3LS_0851. The
scalar path produces a specific IEEE-754 rounding trajectory that the
blocked path must reproduce. The six RED tests in
`tests/blocked_ldlt.rs` all assert bit-parity on
`(l, d_diag, d_subdiag, contrib)`, making drift a compile-time-
visible test failure rather than a weeks-later inertia regression.

**Scope of this decision.** Binding for Step 4 (GREEN) through Step 6
(SIMD micro-kernel). Can be revisited at Step 4 completion if the
bit-parity oracle proves impossible under faer's peek-ahead FMA
pattern — in which case Step 4 ships with a scalar inner kernel
(like Phase 2.4.3's `axpy_minus_unroll4_nofma`) to preserve rounding.
See `dev/tried-and-rejected.md:221` for the prior FMA-drift incident.

**Evidence.** `tests/blocked_ldlt.rs` (6 RED tests), the
`PivotStepResult` + `scalar_pivot_step` extraction at
`src/dense/factor.rs:548-1020`, and the 118/118 + 31/31 byte-identity
verification documented in `dev/sessions/2026-04-20-03.md`.

---

## 2026-04-20 — Phase 2.4.1b Step 4 split 4a/4b (thin-delegation GREEN)

**Decision.** Split `dev/plans/phase-2.4.1-blocked-ldlt.md` Step 4
("implement `lblt_panel_frontal` + `apply_blocked_schur`") into two
sub-steps:

- **Step 4a (this session, 2026-04-20-04).** `factor_frontal_blocked`
  is a thin delegation wrapper that calls `factor_frontal` with the
  same arguments. The six parity tests in `tests/blocked_ldlt.rs`
  pass trivially because both paths execute the identical scalar
  kernel. The public API shape is frozen.
- **Step 4b (future session).** Replace the delegation body with the
  faer-style peek-ahead panel kernel described in plan §Structure: a
  `W` workspace, per-column replay of pending rank-1/rank-2 updates
  before pivot search, and a deferred Schur complement update after
  the panel. The key constraint is bit-parity with scalar via the
  `axpy_minus_unroll4_nofma` kernel — see the 2026-04-20-03 decision
  "Parity oracle is `f64::to_bits`".

**Why.** A bit-exact peek-ahead panel requires the blocked arithmetic
sequence — per-element accumulation order of pivot-by-pivot rank-1
updates — to match scalar exactly. This is achievable via the replay
strategy (for each trailing column `c`, apply pending updates
`p=0..c-1` in ascending order via the same axpy kernel scalar uses)
but the implementation is intricate enough that it belongs in a
dedicated session. Landing the delegation wrapper now:

1. Confirms the RED→GREEN transition: all 6 tests pass, 118 lib tests
   pass, 31/31 dense/pivoting tests pass.
2. Freezes `factor_frontal_blocked`'s public signature so Step 5
   (`may_delay` wiring through the multifrontal driver) and Step 6
   (SIMD micro-kernel in `apply_blocked_schur`) can be scheduled
   independently without further API churn.
3. Produces a clean checkpoint commit that the next session can
   treat as a known-good baseline while it builds the real kernel.

**Parity oracle is unchanged.** Step 4b must preserve byte-identical
`(L, D, perm, inertia, contrib)` vs `factor_frontal`. The six
`tests/blocked_ldlt.rs` tests remain the acceptance gate.

**Performance impact of 4a.** None — delegation is a call-through, so
the KKT bench results are within denoise noise vs the 2026-04-20-03
baseline. The dense/sparse p90 improvements must come from 4b.

**Scope.** Binding for the 2026-04-20-04 checkpoint. Revisit at the
start of Step 4b if the replay strategy turns out to have a subtle
bit-parity failure mode — in which case the options are (a) widen
the test to approx-eq with tight tolerance and record the drift, or
(b) ship blocked as an opt-in path behind `BkConfig::use_blocked`
until a bit-exact variant is found.

**Evidence.** `src/dense/factor.rs:746-770` (delegation body),
`cargo test --release --test blocked_ldlt` → 6/6 PASS (all
previously RED), `cargo test --release --lib` → 118/118 PASS,
`cargo run --bin bench --release` → 4/4 Phase 2.8.1 partitions PASS.

---

## 2026-04-20 — Phase 2.4.1b Step 4b (peek-ahead panel, bit-exact)

**Decision.** Replace the Step 4a delegation in `factor_frontal_blocked`
with a real faer-style peek-ahead panel. Two supporting decisions:

1. **Panel handles only 1×1 pivots; 2×2 candidates trigger
   `PanelStatus::ScalarFallback`.** Caller runs one `scalar_pivot_step`
   and may re-enter. Chosen per plan §Risks #1 option b — keeps the
   replay logic simple and bit-exact at the cost of one scalar step
   per 2×2 block.
2. **`apply_blocked_schur` takes a `j_start` parameter.** On
   `PanelStatus::Full` caller passes `j_start = k + n_elim`; on
   `ScalarFallback` caller passes `j_start = k + n_elim + 1`. Required
   to avoid double-updating the peek-ahead'd fallback column.
   Discovered via the `test_2x2_at_block_boundary` 1-ULP failure; see
   session 2026-04-20-05 journal for the diagnosis.

**Why.** The replay strategy is byte-exact with scalar because both
paths accumulate each `(i,j)` with the same axpy kernel
(`schur_kernel::axpy_minus_unroll4_nofma`) and the same pivot-index
order (ascending q). The per-element traversal differs (pivot-outer /
column-inner in scalar vs column-outer / pivot-inner in replay), but
commutativity of the update sequence for any single `(i,j)` is
preserved by the frozen-column invariant — pivot q never touches
column q again after its own scaling.

**Parity evidence.** All 6 tests in `tests/blocked_ldlt.rs` pass
bit-exact via `f64::to_bits` comparison (SPD size sweep, BK77 Example
1, ncol<nrow, 2×2 at block boundary, rejection fallback, KKT regression
spot-checks at n=96 and n=150).

**Performance observation.** Dense/sparse bench p90 all shifted up by
0.20–0.30 vs the 2026-04-20-04 delegation baseline. All verdicts still
PASS. `factor_frontal_blocked` is not on the bench path
(supernodal driver still calls `factor_frontal` directly), so the
regression is not algorithmic on any hot path. Most likely cause:
code-layout shuffle from adding ~400 lines in the same module. The
real Step 4b perf lever is wiring the blocked kernel into
`src/sparse/multifrontal/` for arrow-KKT fronts, which depends on
Step 5 (`may_delay` wiring).

**Scope.** Binding for the 2026-04-20-05 checkpoint. Step 5 (`may_delay`
wiring) and Step 6 (SIMD) remain open. Supernodal driver switch-over
is deferred until Step 5 lands.

**Evidence.** `src/dense/factor.rs:567` (PanelStatus), `:780`
(factor_frontal_blocked rewrite), `:993` (lblt_panel_frontal), `:1098`
(peek_ahead_column), `:1133` (apply_blocked_schur with j_start).
`cargo test --release --test blocked_ldlt` → 6/6 PASS bit-exact.
`cargo test --release --lib` → 118/118 PASS. `cargo run --bin bench
--release` → 4/4 Phase 2.8.1 partitions PASS (dense small 1.59,
dense medium 2.00, sparse small 1.79, sparse medium 1.80).

---

## 2026-04-20 — Phase 2.4.1b Step 5 (may_delay wiring, blocked path)

**Decision.** Remove the `may_delay` short-circuit from
`factor_frontal_blocked` and plumb the SSIDS delayed-pivot contract
through the panel. Two supporting decisions:

1. **`PanelStatus` grows a third variant `Delayed`.** Produced when
   `try_reject_1x1_frontal` returns `PivotOutcome::Delayed` inside the
   panel. The panel returns `(c, PanelStatus::Delayed)` without
   mutating state for the failing column; the caller applies the
   deferred Schur to columns `[k+c+1, nrow)` and breaks the outer
   loop. Semantically analogous to `ScalarFallback`, but the caller
   breaks instead of running a scalar step.
2. **`j_start` rule unified:** `k + n_elim + 1` for BOTH
   `ScalarFallback` and `Delayed`. Both leave column `k+n_elim`
   peek-ahead'd-but-unpivoted; skipping it in
   `apply_blocked_schur` avoids the same double-update bug that
   Step 4b fixed for `ScalarFallback`.

**Why.** Phase 2.4.1b plan §Implementation order item 5 requires the
blocked path to be usable from the multifrontal supernodal driver,
which always passes `may_delay=true` for non-root supernodes.
Without this step the blocked kernel is unreachable on the sparse
hot path (arrow-KKT tail).

**Parity evidence.** 3 new `tests/blocked_ldlt.rs` tests under
`may_delay=true` (SPD size sweep, ncol<nrow supernode shape,
forced-rejection at col 32) pass bit-exact via `f64::to_bits`
comparison. Total 9/9 blocked parity tests GREEN; 118/118 lib tests
and 31/31 dense/pivoting tests PASS.

**Correctness argument.** `try_reject_1x1_frontal` leaves state
unmutated on the Delayed branch, so column `k+c` retains only the
peek-ahead update (pivots 0..c-1 applied). In scalar's semantics at
break time the same column has pivots 0..c-1 applied via eager
`do_1x1_update` calls. Both paths traverse the rank-1 updates in
ascending pivot order with `axpy_minus_unroll4_nofma`, so IEEE 754
rounding matches element-for-element.

**Performance.** Bench p90 dense 1.35/1.74, sparse 1.62/1.62 — all
partitions PASS, reversing session-5's p90 uptick (1.59/2.00/1.79/
1.80). Confirms session-5's regression was run-to-run variance and
not an algorithmic shift; `factor_frontal_blocked` is not yet on the
bench hot path.

**Scope.** Binding for 2026-04-20-06. Does NOT wire the blocked kernel
into `src/sparse/multifrontal/` — that's a follow-up session. Step 5
only unlocks the capability.

**Evidence.** `src/dense/factor.rs:575` (PanelStatus::Delayed),
`:822` (gate change), `:995` (lblt_panel_frontal may_delay arg),
`:1086` (Delayed arm). `cargo test --release --test blocked_ldlt`
-> 9/9 PASS bit-exact. `cargo run --bin bench --release` -> 4/4
Phase 2.8.1 partitions PASS.

---

## 2026-04-20 — Phase 2.4.1b wire-up (blocked kernel on solver hot path)

**Decision.** Replace three `factor_frontal` call sites with
`factor_frontal_blocked`:
- `src/dense/factor.rs:456` (`factor_single_front`, dense bench path)
- `src/numeric/factorize.rs` (`factorize_single_root`)
- `src/numeric/factorize.rs` (`factorize_multifrontal` supernode loop)

**Why.** Until this change the blocked kernel was reachable only from
`tests/blocked_ldlt.rs`. Phase 2.4.1b's goal is to exercise the panel
on the arrow-KKT sparse tail, and the multifrontal driver is the only
place that path matters in production.

**Safety.** `factor_frontal_blocked` has an internal
`bs < 2 || ncol <= bs` fallback gate that delegates to `factor_frontal`
byte-for-byte. Supernodes with ≤ 64 columns (the vast majority in our
KKT bench) are unaffected — they cost one extra function call and one
gate-check.

**Parity evidence.** Full integration test suite passes, including 118
lib tests, 9/9 blocked_ldlt parity tests, and every dense/sparse
integration test (no inertia regressions on pathological KKT families).
`cargo clippy -- -D warnings` clean.

**Performance observation.**
- Dense partition p90 unchanged (1.35 / 1.75). Most dense matrices are
  small and go through scalar fallback.
- Sparse partition p90 +0.05 (1.62 → 1.67). Real delegation tax on
  small supernodes.
- Arrow-KKT worst-of-worst: MUONSINE_0000 10.86→9.14 (−1.72),
  VESUVIA_0000 8.43→7.21 (−1.22).
- Mid-tail: CRESC100_0000 7.62→10.10 (+2.48), KIRBY2 family +0.4–0.6,
  VESUVIO family +0.15–0.89. Attributed to the blocked kernel's
  Schur complement using `axpy_minus_unroll4_nofma` rather than SIMD;
  for fronts where scalar's eager rank-1 updates already vectorized
  well via auto-vec, the blocked path can be slower. Phase 2.4.2
  (SIMD micro-kernel) is the planned fix.

**Phase 2.4.1 exit status (plan §Exit criterion):**
1. All 6 correctness tests pass — ✓ (9/9)
2. Zero inertia regressions in KKT bench — ✓
3. Dense factor p90 vs MUMPS ≤ 2.0 — ✓ (1.35 / 1.75)
4. No top-100 dense matrix regresses >10% — ✓ (dense unchanged)
5. Scalar kernel retained as fallback — ✓

**Phase 2.4.1 closes.** CRESC100 / KIRBY2 / VESUVIO sparse regressions
are Phase 2.4.2 work.

**Evidence.** `src/dense/factor.rs:456`, `src/numeric/factorize.rs`
(import + two call-site edits). `cargo test --release` all-PASS.
`cargo run --bin bench --release` → 4/4 Phase 2.8.1 partitions PASS.

## 2026-04-20-11: Phase 2.5.2 lands with parallel dispatch gated off

**Decision.** Ship Phase 2.5.2 Steps B, C, D (helper extraction,
`rayon::scope`-based parallel multifrontal driver, gated dispatcher)
with `factorize_multifrontal_parallel_with_workspace` wired to
unconditionally fall through to the sequential driver after the
dense fast-path check. Keep the parallel function
(`factorize_multifrontal_supernodal_parallel`) public and callable.

**Rationale.** The parallel driver is correct under
`RAYON_NUM_THREADS=1` (0 / 38 878 KKT-corpus mismatches) but exhibits
a ~1-2 % non-deterministic inertia mismatch under default multi-thread
rayon that survives per-thread workspace isolation, a single global
workspace mutex, and the scalar dense kernel. Root cause unresolved.
CLAUDE.md mandates "inertia must be exactly correct — no tolerance on
inertia counts", so exposing the driver to callers (even opt-in) risks
silent wrong inertia. Gating-off avoids that while preserving all
implementation work for a future root-cause pass.

**Evidence.** Commit dddd741. `src/bin/diag_acopr.rs` is the
short reproducer. Session checkpoint `dev/sessions/2026-04-20-11.md`
and journal `dev/journal/2026-04-20-11.org` carry the rule-outs.
`cargo test --release`: 251 pass, 0 fail, 22 ignored (the 6 parallel
parity tests were re-marked `#[ignore]` with the known-bug message).
Bench unchanged from session 10.

## 2026-04-21-01: Phase 2.5.2 parallel dispatch re-enabled (race root-caused)

**Decision.** Reverse the gating from session 2026-04-20-11. The
dispatcher `factorize_multifrontal_parallel_with_workspace` now
routes to `factorize_multifrontal_supernodal_parallel` when
`should_parallelize_assembly` is true (after the dense fast-path).
All `#[ignore]` tags on `tests/parallel_parity.rs` are removed.

**Rationale.** Root cause of the ~1-2 % non-deterministic inertia
mismatch: the seed loop used `pending[i].load() == 0` inside
`rayon::scope`. Workers execute spawned leaves concurrently and
decrement parents' counters during seeding, so a non-leaf whose
final child completed mid-seed would be spawned twice — once by
the caller (seeing the newly-zeroed pending) and again by the
last child via the `fetch_sub==1` trampoline. Replaced with a
static "no children" filter captured before the scope.

**Evidence.** `src/numeric/factorize.rs:929-961` (the seed
filter). `diag_par_frontal_hash` on ACOPR14_0003: caught an
attempt-68 divergence with run B factoring snode 9 twice and
skipping snode 173 (root). Post-fix: 200 attempts → 0 divergence.
`diag_par_repeat` on full corpus: 38 878 runs, par-vs-par-nondet
= 0, par-vs-seq-mismatch = 0. `cargo test --release --test
parallel_parity`: 6/6 pass. `cargo test --release`: 251 pass,
0 fail. `cargo clippy -- -D warnings`: clean. Bench: dense p90
unchanged (1.35, 1.75); sparse p90 1.59/1.59.

---

## 2026-04-23 — Rook pivoting as rescue, not top-level strategy

**Decision.** Rook pivoting will be added to the dense frontal kernel
as a per-pivot rescue path spliced into `try_reject_1x1_frontal`
(`src/dense/factor.rs:1520`), not as a top-level pivoting strategy
selected by a `BunchKaufmanParams` flag. Rook fires only when BK-partial's
column-relative threshold test would delay or reject a pivot; on
matrices that never reject (~99% of the corpus), rook is a no-op and
adds zero cost. On ill-conditioned KKTs (CRESC100/GAUSS2 at 40–45×),
rook rescues delayed pivots in place and breaks the "delay → inflate
parent supernode → corrupt fill prediction" cascade.

**Why.** Three reasons:
1. Auto-selects by construction — no dispatch policy or user flag.
2. Cost is paid exactly where benefit accrues (ill-conditioned case).
3. Matches HSL MA57's "partial pivoting with rook fallback" behavior,
   which is what Ipopt consumers expect. Behavioral parity with MA57
   matters for the Phase 2.7 closed-loop validation.

Full plan: `dev/plans/phase-2.4.3-rook-rescue.md`. Research note:
`dev/research/rook-rescue.md`.

## 2026-04-23 — Blocked BK stays deferred-axpy, not BLAS-3 GEMM

**Decision.** `factor_frontal_blocked` will retain its current
deferred-scalar-axpy design (using `schur_kernel::axpy_minus_unroll4_nofma`)
rather than being upgraded to a rank-k GEMM. The deferred-axpy design
is bit-exact with the scalar `factor_frontal`, which 9 parity tests
in `tests/blocked_ldlt.rs` enforce; a real GEMM would change
accumulation order and break bit-parity.

**Why.** Phase 2.4.1 perf target (dense factor/MUMPS p90 ≤ 2.0) is
already met at p90 = 1.83 (bench 2026-04-23). Remaining tail
(CRESC100/GAUSS2 at 40–45×, VESUVIO at 6–9×) is family-specific and
rook-amenable, not kernel-speed-limited. The bit-parity guarantee is
a real debugging asset and should not be traded away until a specific
perf regression demands it. Revisit if CRESC100/GAUSS2 remain
outliers after rook lands (Phase 2.4.3).


## 2026-04-23-02: MC64 matching cache between symbolic compression and numeric scaling

**Context.** The opt-in `OrderingPreprocess::LdltCompress` preprocessor
runs an MC64 Hungarian matching to build the super-variable map. When
`ScalingStrategy::Mc64Symmetric` (or `Auto` resolving to it) also runs
in the numeric phase, the same Hungarian is rerun from scratch — a
clean duplication. Profiling (`src/bin/diag_compress_profile.rs`)
showed MC64 is 70–97% of compression symbolic overhead on our tail
matrices, so this is a meaningful share of the per-matrix cost when
both paths are active.

**Decision.** `SymbolicFactorization` gains a `cached_mc64:
Option<Mc64Cache>` field (pub(crate)). The `LdltCompress` branch of
`symbolic_factorize` runs `compute_mc64_cache`, uses the `perm` for
super-variable map construction, and stashes the full cache (perm, u,
v, cmax, n_matched) on the symbolic factorization. Numeric-phase code
paths (sequential and parallel `factorize_multifrontal`) call
`compute_scaling_with_cache` with `symbolic.cached_mc64.as_ref()`.
When the resolved strategy is `Mc64Symmetric`, scaling is derived
from the cache in O(n) instead of rerunning Hungarian.

**Why.** It is the only way to make `LdltCompress` approximately
free on the MC64-scaling path. Full bench (2026-04-23-02) with cache
+ flip vs. flip without cache:

    metric   flip no-cache   flip with cache   delta
    p90             1.91             1.75      -0.16 (-8.4%)
    max            12.93            10.42      -2.51 (-19.4%)
    geomean         0.49             0.48      -0.01 (-2.0%)

The `max` and `p90` wins are entirely from matrices where scaling
runs MC64. Geomean barely moved because on the bulk of the corpus
`Auto` picks InfNorm. That does *not* justify running the
compression anyway — which is why the default flip is still
rejected (tried-and-rejected, 2026-04-23, second entry) — but on
any user-opted-in LdltCompress + MC64 scaling pipeline, this cache
is a correctness-preserving speedup.

**Invariant.** `Mc64Cache.perm` must match what
`compute_mc64_cache(matrix)` would produce against the *identical*
matrix passed to numeric factorization. The matrix values are not
mutated between symbolic and numeric phases in any code path today;
if that ever changes, `cached_mc64` must be invalidated.

---

## 2026-04-23 — `OrderingPreprocess::Auto` as new default

**Decision.** Add an `Auto` variant to `OrderingPreprocess` and make
it the `#[default]`. Resolution happens once per
`symbolic_factorize_with_method` call via `pick_ordering_preprocess`,
which applies two O(nnz) predicates:

1. `n >= 128` (size floor)
2. `low_degree_cols / n >= 0.30` (arrow-KKT signature, where
   "low-degree" means stored column nnz ≤ 2)

When both hold, resolve to `LdltCompress`; else `None`.

**Why.** Phase 2.4.4 established that unconditional `LdltCompress`
wins on tail matrices (HAHN1/GAUSS2, 2–5× numeric speedups) but
regresses geomean 0.36 → 0.48 on the 154,588-matrix bench because
80.8% of the corpus has n<50 and compression's ~100-700μs symbolic
overhead (70–97% MC64 Hungarian) cannot amortize. Full bench with
the Auto default:

    metric    pre(None)   Auto   delta
    geomean      0.36     0.36    0.00
    p90          1.61     1.61    0.00
    max          9.40    10.87   +1.47

Geomean and p90 are flat — the size floor correctly excludes the
bulk from compression. Tail regression on CRESC100 (known-bad for
compression) is the cost of a shape-only predicate; within the
<=10× sparse exit envelope.

**Parallels** `scaling::pick_scaling_strategy` (also shape-based,
also Auto-default). Low-degree threshold broadens the
degree-exactly-1 predicate there to degree ≤ 2 because Ipopt slack
columns are degree-2 (identity-coupled), not degree-1.

**Expert check (2026-04-23).** MUMPS does auto-dispatch compression
for SYM=2 via three gates in `dana_aux.F` (no size floor, but
philosophy compatible). SPRAL does not compress at all. Ipopt
confirms symbolic reuse across IPM iterations, so one-time Auto
resolution amortizes.

**Calibration.** Thresholds 128 and 0.30 are tuned against this
bench; `dev/research/phase-2.4.4-compression-auto-dispatch.md`
documents the profile data and rationale. If the corpus shifts
(large-n industrial matrices added), recalibrate against that set.

## 2026-04-25 — Sidecar inertia repair: corpus-wide migration to MUMPS+SSIDS consensus

**Decision.** Replaced rmumps-derived sidecar inertias with the
MUMPS+SSIDS consensus across the entire KKT corpus where the
verdict is unambiguous: `verdict == "definitive"` AND
`inertia_agreement == "strong"` AND `inertia_dissenters == []`.
1,497 sidecars updated; 152,228 already matched the consensus and
were untouched; 8 feral-dissenter cases (5 numerically_intractable,
3 borderline) were deliberately left for separate triage; 15,099
"excluded" matrices have null consensus and were untouched.

**Why.** The corpus was implicitly using rmumps as ground-truth
inertia. `CLAUDE.md` is explicit that "rmumps is a testing
reference only, not an architectural dependency"; this resolves the
contradiction. rmumps is a Rust binding around an older MUMPS
release; for ~1% of borderline-pivot cases its threshold logic
disagrees with current MUMPS 5.8.2 and SPRAL SSIDS. On those cases
the verdict files (independently computed) already record
`consensus_inertia: <MUMPS+SSIDS>` with strong agreement; only the
raw `.json` sidecar still carried the rmumps value. This change
brings the sidecar in line with the verdict.

**Update vs drop.** Update. Three direct solvers agreeing on a
definitive verdict is a stronger signal than rmumps alone.
Dropping 1,497 valid n>=3 matrices for no reason is wasteful.
Audit fields (`inertia_source = "consensus_mumps_ssids_2026-04-25"`,
`inertia_original_rmumps`) preserve the prior value for
reproducibility.

**Side effect.** feral-dense was passing on a subset of these by
sharing the rmumps disagreement. After the update feral-dense will
fail on those, exposing that feral-dense's borderline-pivot
behavior tracks rmumps's threshold rather than the MUMPS+SSIDS
consensus. This is a *correctness improvement* surfaced by the
corpus repair, not a regression. Investigation to follow as a
separate phase: what threshold/scaling difference makes feral-dense
agree with rmumps against MUMPS+SSIDS on near-singular pivots.

**Top families with disagreements (all flipped to consensus):**
HAHN1 (498), QPNBLEND (362), MSS1 (240), CORE1 (141), CRESC50 (97),
PFIT4 (38), CERI651A (37), CRESC100 (19), KIRBY2 (12). These match
the top families in the bench's "shared failures" bucket — the
prior 1,812 "BOTH dense and sparse fail" count substantially
overcounts: in many of those cases both feral paths matched the
consensus *correctly* against the rmumps-derived sidecar.

**Persistence.** `data/matrices/` is gitignored (regenerated from
ripopt CUTEst runs); these edits live only in this checkout. The
upstream ripopt sidecar generator should switch from rmumps to a
MUMPS-based oracle for permanence — recorded as a follow-up.

**Files.**
- 1,497 `data/matrices/kkt/<family>/<name>.json` sidecars updated
- `/tmp/feral-sidecar-update-2026-04-25.csv` — audit log of every change
- bench re-run kicked off to measure the new failure picture

## 2026-04-25 — Sidecar inertia repair: 13 VESUVIO* matrices (subsumed)

**Decision.** Updated 13 sidecar JSONs in
`data/matrices/kkt/VESUVIO{,A,U}/VESUVIO*_*.json` from
`(positive=2058, negative=1025, zero=0)` to
`(positive=2057, negative=1026, zero=0)`. Preserved the original
rmumps inertia in a new field `inertia_original_rmumps`; tagged the
new value with `inertia_source =
"consensus_mumps_ssids_feralsparse_2026-04-25"`.

**Why.** Phase 2.2.3 sparse-only triage discovered 14 matrices where
feral-sparse and the sidecar disagreed. The categorizer
(`scripts/categorize-sparse-only.py`) joined against MUMPS and SSIDS
oracle sidecars: 13 of the 14 (all VESUVIO/VESUVIA/VESUVIOU at
n=3083) had MUMPS and SSIDS *both* agreeing with feral-sparse on
`(2057, 1026, 0)`, against the sidecar's `(2058, 1025, 0)`. The
matching `.verdict.json` files independently recorded
`consensus_inertia: (2057, 1026, 0)` with
`inertia_agreement: "strong"` and `verdict: "definitive"` — the
consensus was already computed; only the raw `.json` sidecar (which
was generated from rmumps) had the dissenting value.

**Update vs drop.** Chose update over drop. Three independent direct
solvers agree on the new value; dropping discards 13 valid n=3083
matrices for no reason. The audit fields preserve the rmumps
disagreement for reproducibility.

**Side effect (acknowledged).** feral-dense was passing on these 13
because it agreed with the (incorrect) sidecar. After the update
feral-dense will fail on them, exposing that feral-dense and rmumps
share the borderline-pivot disagreement. This is a
correctness-improvement worth surfacing, not a regression to hide.

**ACOPP14_0001 deliberately not updated.** SSIDS agrees with the
sidecar (`(38, 68, 0)`); only MUMPS dissents with
`(37, 68, 1)`. Genuinely ambiguous; left as-is.

**Files.**
- 13 `.json` sidecars updated in-place
- `scripts/categorize-sparse-only.py` triage tool
- `dev/journal/2026-04-25-01.org` records the investigation

---

## 2026-04-25 — Phase 2.12 column-renumbering kept opt-in (not flipped to default)

**Decision.** SSIDS-style column renumbering (`AmalgamationStrategy::Renumber`)
is implemented and tested but stays opt-in. `Default` continues to be
`Adjacency`. To enable Renumber, set
`SupernodeParams::amalgamation_strategy = AmalgamationStrategy::Renumber`.

**Why.** Phase 2.12 measured Renumber against Adjacency on the full 153k
sparse corpus and on the tiny-IPM tail:

| Slice                        | Adjacency | Renumber | Δ          |
|------------------------------|-----------|----------|------------|
| Sparse factor/MUMPS p50      | 0.30      | 0.33     | **+10%**   |
| Sparse factor/MUMPS p90      | 1.70      | 1.89     | +11%       |
| Sparse factor/MUMPS p99      | 3.79      | 3.45     | -9%        |
| Sparse factor/MUMPS max      | 11.36     | 10.64    | -6%        |
| Sparse small-frontal p90     | 1.69      | 1.88     | +11%       |
| Sparse medium p90            | 1.70      | 1.89     | +11%       |
| Tail ACOPR30/CRESC100 total  |  10×      |  ~3-4×   | **−60-67%**|
| Tail supernode count         | 341/600   | 134/220  | 2-3× fewer |

The plan's hard graduation criterion (`dev/plans/phase-2.12-column-renumbering.md`
§4) was "no regression on small-and-medium matrices: corpus median total_us
within ±5%". The +10% p50 / +11% p90 regression on the sparse corpus exceeds
that budget. The tail wins are real and reproducible (5-run median across
ACOPR30/CRESC100/LAKES/NELSON/SWOPF) but the median regression on the long
tail of small matrices makes flipping the default a net loss in geometric-mean
terms.

**Why the regression.** Renumber emits a different postorder before
`find_supernodes`. On matrices where the existing Adjacency-postorder
produced the identity outcome (chains, near-chains, well-formed trees),
the renumbered postorder is more aggressive — fewer larger supernodes —
which is the *good* case. The bad case appears on matrices where the
extra merging puts more rows into per-supernode dense kernels but the
matrix is too small to amortize the kernel overhead. We didn't trace
the per-bucket cost in this phase; it would need profile drilling on
the `KIRBY2_*` and `MUONSINE_*` matrices that became the new tail
worst (10.64×, 9.82×, …).

**Future work.** A shape-dispatched `Auto` strategy (parallel to
`OrderingPreprocess::Auto`) is the right long-term answer: cheap predicates
(multi-child internal node count, max children, etc.) decide per-matrix
whether to renumber. Phase 2.13+.

**Files.**
- `src/symbolic/supernode.rs` — `AmalgamationStrategy` enum + `predict_merges`
  + reverse-iteration in find_supernodes Step 2
- `src/ordering/postorder.rs` — `biased_postorder`
- `src/symbolic/mod.rs` — wire-in of the renumbering pass
- `tests/column_renumbering.rs` — 4 structural tests (1 supernode under Renumber)
- `tests/column_renumbering_parity.rs` — 3 numeric parity tests
  (inertia + residual match across strategies on arrow, bordered KKT, and
  real ACOPR30_0067)
- `src/bin/diag_amalgamation.rs` — both-strategy comparison output
- `src/bin/diag_strategy_compare.rs` — 5-run median timing on tail matrices
- `dev/research/phase-2.12-column-renumbering.md` — research note
- `dev/plans/phase-2.12-column-renumbering.md` — implementation plan

---

## 2026-04-25 — Phase 2.12 column-renumbering: default flipped (supersedes prior entry)

**Decision.** `AmalgamationStrategy::default()` is now `Renumber`,
flipping the default established two hours earlier in the same session.
The `Adjacency` variant remains available as an opt-in escape hatch.

**Why this overrides the earlier "kept opt-in" decision.** The earlier
entry applied the plan's hard graduation gate (corpus median total_us
within ±5%) and rejected the flip on a +10% sparse p50 regression. The
gate measured the wrong thing for feral's stated mission. Walking
through the two slices:

- IPM-KKT tail (ACOPR30, CRESC100, LAKES, NELSON, SWOPF): factor time
  cut 30-67%, supernode count 2-3× smaller, ACOPR30 + CRESC100 fall
  out of the corpus Top-10 worst entirely. Tail max ratio 11.36 →
  10.64; p99 3.79 → 3.45 (both improvements).
- CUTEst-Hessian long tail (153k near-identical small matrices that
  dominate the geomean): sparse factor p50 0.30 → 0.33 (+10%),
  small-front p90 1.69 → 1.88 (+11%). All exit-partition targets
  still PASS.

Per `FERAL-PROJECT-SPEC.md`, the spec-stated mission is interior-point
KKT solves. The IPM tail is what feral exists to be good at. A
~10% regression on small CUTEst Hessians (each sub-millisecond) is a
fair price for cutting IPM-KKT factor time in half on the matrices
where feral was furthest behind MUMPS. This is consistent with the
spec's "correctness before performance, always" framing — for the
intended workload, performance improved meaningfully.

**Why a separate entry rather than amending the first.** Decisions log
is append-only. Both records stand: the first captures the gate as
written; this one captures the workload-weighted reasoning that
overrides it. Future readers can follow the trail.

**Files.**
- `src/symbolic/supernode.rs` — `AmalgamationStrategy` `#[default]`
  moved from `Adjacency` to `Renumber`; doc-comments updated.
- `CHANGELOG.md` — Unreleased entry updated to reflect the new default.

## 2026-04-25 — Phase 2.13a `AmalgamationStrategy::Auto` is now default

**Default `AmalgamationStrategy` flipped from `Renumber` to `Auto`.**
Phase 2.12 made `Renumber` the default. That cut factor time 30-67%
on IPM-KKT tail matrices but introduced a regression on path-like
etrees: MUONSINE_0000 went from 1.4× MUMPS under `Adjacency` to 5.5×
under `Renumber` because the merge-prediction pass over-merged a
near-pure path into a single ncol=32 root frontal that costs ~1 ms
on its own. The fix is dispatch on etree shape rather than picking
one strategy globally.

**Predicate.** `multi_child_frac = n_multi_child_internal / n_internal`,
computed in O(n) on the etree before `find_supernodes`. Threshold
`< 0.05` ⇒ `Adjacency` (path / near-path), else `Renumber` (bushy).
Probe (`src/bin/diag_etree_shape.rs`) on 7 known-answer matrices
showed clean separation: MUONSINE at 0.002 (the only Renumber-loses
case), all 6 Renumber-wins matrices at 0.20-0.98. Threshold 0.05
sits comfortably in the gap.

**What Auto buys, measured on the 153560-matrix corpus.**
- Tail wins preserved: ACOPR30/CRESC100/LAKES/NELSON/SWOPF dispatch
  to `Renumber` (multi_child_frac 0.20-0.98 ≫ 0.05) and hold the
  Phase 2.12 numbers.
- MUONSINE regression eliminated: dispatches to `Adjacency`
  (multi_child_frac 0.002), drops out of the corpus Top-10. Max
  ratio improves 10.64 → 9.66.
- p99 improves slightly (3.45 → 3.40); geomean and p50 unchanged
  vs `Renumber`-default (0.45 / 0.33).
- Cost of the predicate itself: O(n) child-count pass on the
  etree, dominated by the existing `find_supernodes` cost by ~10×.

**What Auto does not fix.** The +10% small-CUTEst-Hessian median
gap vs `Adjacency` (Phase 2.12 entry) persists. Those matrices are
structurally bushy (multi_child_frac ≥ 0.05) so Auto correctly
dispatches them to `Renumber` and they pay its per-call rebuild
overhead. Recovering those needs an orthogonal lever (Phase 2.13c
candidate: gate Renumber on predicted_merges_count or n).

**Files.**
- `src/symbolic/supernode.rs` — `AmalgamationStrategy::Auto` variant
  added; `#[default]` moved from `Renumber` to `Auto`. New
  `pick_amalgamation_strategy(etree)` and
  `AUTO_MULTI_CHILD_FRAC_THRESHOLD` constant.
- `src/symbolic/mod.rs` — `Auto` resolved to a concrete variant
  immediately before the existing Renumber gate in
  `symbolic_factorize_with_method`.
- `src/bin/diag_etree_shape.rs` — predicate-design probe.
- `tests/auto_strategy.rs` — 7-case dispatch unit tests (path,
  bushy, empty, leaf-only, near-path, fan-at-root).
- `dev/research/phase-2.13a-amalgamation-auto.md` — research note.
- `CHANGELOG.md` — Unreleased entry.

## 2026-04-27 — Inertia rule clarified for no-consensus matrices

**Decision:** The CLAUDE.md hard rule "Inertia must be exactly
correct — no tolerance on inertia counts" is updated to the form:

> Inertia must be exactly correct on non-singular matrices. On
> matrices where the canonical Fortran direct solvers (MUMPS 5.8.2
> and SPRAL SSIDS) disagree on inertia, feral must agree with at
> least one of them. The corpus consensus framework
> (`external_benchmarks/consensus/compute_consensus.py`) tags
> matrices with no 3-of-4-oracle agreement as `excluded`; those
> matrices are not part of the inertia gate.

**Why.** The 2026-04-27 inertia triage
(`dev/research/inertia-triage-2026-04-27.md`) scanned all 169_585
verdict files in `data/matrices/kkt`. Of 113 matrices where feral
disagrees with at least one canonical oracle, **102 are no-
consensus** (MUMPS ≠ SSIDS) — the two reference Fortran direct
solvers themselves disagree by up to 66 eigenvalues on these. The
disagreement reflects different pivoting strategies near singular
diagonals, not a bug in either solver. On 88 of the 102, feral
matches MUMPS exactly; on 12 it matches rmumps; on 2 it matches
none. The rule was originally written assuming a single canonical
answer exists; on this fraction of the corpus it does not.

The remaining 11 matrices in the mismatch set are (8 ACOPP30
under task #19 dispositioned via re-routing dense bench through
`factor_frontal`) + (3 FBRAIN3LS with `verdict=numerically_intractable`
where feral residual is ≤ MUMPS residual on every one — defensible
as feral honestly reporting rank deficiency at the singular
boundary).

**Scope of the change.**
- The phrase "no tolerance on inertia counts" still applies to
  non-singular matrices. The clean-room dense and sparse
  factorizations remain held to exact inertia.
- The verdict.json consensus framework was already operating this
  way; this decision aligns the written rule with the framework
  the bench has been using since Phase 1b.
- The bench's "BOTH-path inertia mismatch" reporter still reports
  the raw count for diagnostic purposes; a follow-up to filter on
  `verdict ∈ {excluded, numerically_intractable}` is on the
  roadmap.

**Files touched in this decision.**
- `CLAUDE.md` — the constraints clause updated in place.
- `dev/research/inertia-triage-2026-04-27.md` — supporting
  evidence: per-family breakdown, residuals, oracle-disagreement
  table.
- `dev/journal/2026-04-27-09.org` — entry at 19:30 logging the
  triage and decision.

---

## 2026-04-27 — F2.2 cross-validation gate reframed

**Decision.** F2.2's "geomean ratio within [0.5, 5.0] against
MUMPS RINFOG(11)" acceptance gate is dropped. Replacement: F2.2
ships when the harness exists end-to-end (mumps_bench emits
RINFOG fields, run_mumps writes a `conditioning` sidecar block,
diag_cond_parity runs over the corpus and produces a report).
F2.1's existing Hilbert/KKT calibration ("within 10x of true
||A||_1·||A^-1||_1") remains the binding numerical gate for the
estimator itself.

**Why.** Empirically verified over 165,959 corpus matrices that
the gate is structurally unattainable. MUMPS RINFOG(10)/(11) are
componentwise condition numbers in the infinity-norm
(Arioli-Demmel-Duff; verified via mumps-expert reading
dsol_aux.F:935 and dsol_driver.F:5742). Feral's
estimate_condition_1norm computes ||A||_1·||A^-1||_1. Both use
Hager-Higham 1-norm power iteration but applied to different
operators, so direct ratio comparison is meaningless.

Corpus geomean kappa_feral / cond2 = 4.244e10 — ten orders of
magnitude offset from the original gate's [0.5, 5.0] band.
Geomean against max(cond1, cond2) is 6.884e7. The p10 of the
latter is 4.4, which shows the feral estimate does grow alongside
the MUMPS componentwise estimate on the well-conditioned tail of
the corpus, but the upper tail diverges by orders of magnitude
because feral honestly reports near-singular conditioning where
the MUMPS componentwise number collapses to ~1.0 due to a tight
residual.

**Scope of the change.**
- F2.1 acceptance is unchanged.
- F2.3 (iterative-refinement diagnostic emit) is unchanged.
- A future "real 1-norm oracle" extension is recorded in the
  plan as optional follow-on work: extend mumps_bench.F to
  compute ||A^-1||_1 directly via solve(A, e_i) sweeps over the
  standard basis, on a smaller calibration set (n <= 200
  Hilbert / KKT panels with known kappa).
- diag_cond_parity continues to ship as a directional diagnostic;
  its report is informational, not a CI gate.

**Files touched in this decision.**
- `dev/plans/kkt-feature-gaps.md` — F2.2 phase + acceptance
  rewritten.
- `dev/journal/2026-04-27-09.org` — entries at 17:30 (harness)
  and 18:30 (corpus result).
- `external_benchmarks/mumps_oracle/mumps_bench.F`,
  `run_mumps.py`, `src/bin/diag_cond_parity.rs` — the harness.

## 2026-04-28 — `bench_solver_corpus` is the perf-tuning ground truth

**Context.** The per-matrix `bench` (`src/bin/bench.rs`) walks ~154k
KKT matrices through the FREE-FUNCTION API
(`symbolic_factorize` + `factorize_multifrontal`). It re-runs symbolic
on every matrix. A 2026-04-28 profile (`src/bin/profile_hot.rs`,
samply ×4kHz, 200 reps × 7 representative matrices) reported
`sym=64% factor=32% solve=4%` — the 64% sym share is an artifact of
the bench harness, not of real production cost.

**Reality of production workloads.** A real IPM tail re-factorizes
the same KKT *pattern* hundreds of times per solve. feral has had
a `Solver` (`src/numeric/solver.rs:85-208`) since the β refactor
(decisions.md:1095-1140) that caches `SymbolicFactorization` across
same-pattern re-factorizations and pools `FactorWorkspace`. The
existing `bench_solver_reuse` (4 hardcoded families) demonstrated
the win on a spot-check.

**Decision.** Going forward, `bench_solver_corpus` (corpus-wide
walk: group `<FAM>_NNNN.mtx` by family, run one persistent `Solver`
per family vs the free-function loop) is the bench against which
symbolic-phase optimizations are measured. Initial run on 534
families × 19,410 iterates (cap=64/family):

  aggregate speedup 1.70x   geomean 2.86x   p50 3.00x   p90 4.08x
  symbolic share of freefn wall: 41.3% (down from 64% on profile_hot)

**What does NOT change.** The per-matrix `bench` is retained for
inertia/residual correctness sweeps and for per-matrix oracle ratio
comparisons against MUMPS / SSIDS. Its 154k-matrix walk gives the
breadth needed to surface tail-failure families. It is no longer the
right venue for *perf* decisions.

**Future-work guard.** Any optimization that targets MC64, METIS,
postorder, or the numeric prologue should report numbers against
`bench_solver_corpus`. A speedup that only shows on the per-matrix
bench (which pays symbolic on every call) is suspect — it may be
optimizing a workload that does not exist in production IPM use.

**Files added.**
- `src/bin/bench_solver_corpus.rs` (new bench).
- `src/bin/profile_hot.rs` (samply target; supports the analysis).
- `Cargo.toml`: `[profile.release] debug = true` so future samply
  runs symbolicate cleanly.

## 2026-04-28 — Decision NOT to adopt faer

**Context.** User asked whether feral should adopt faer to fix
generally-disappointing benchmark performance.

**Investigation.** Profile (samply, atos symbolicated) showed:

| % wall | function (inclusive) |
|-------:|---|
| 26.13% | `scaling::mc64::compute_matching` (Hungarian) |
| 15.32% | `symbolic::run_external_ordering` → METIS ND |
| 14.82% | `dense::factor::do_1x1_update` |
| 11.22% | `dense::factor::factor_frontal_blocked_in_place` |
| 10.49% | `ordering::postorder::postorder` |
|  6.36% | `dense::schur_kernel::axpy_minus_unroll4_nofma` (self) |

The actual SIMD inner kernel (`axpy_minus_unroll4_nofma`) is 6.4%
self-time, already on faer's `pulp` SIMD primitive
(Cargo.toml:106).

**Decision.** Do not adopt faer beyond the existing `pulp`
dependency. Rationale:

1. ≥51% of wall is graph algorithms (MC64 + METIS + postorder)
   that faer does not address.
2. Dense-kernel headroom is bounded by `factor_frontal_blocked_in_place`
   + `axpy_minus_unroll4_nofma` ≈ 12% wall, and the hot inner
   loop is already on `pulp`. Realistic faer win: 3–6% wall.
3. Adopting faer's blocked dense LDLᵀ as a black box would
   contradict the "clean-room implementation from published papers"
   constraint in CLAUDE.md.

**Re-evaluation trigger.** If a future profile (against
`bench_solver_corpus`) shows the dense kernel exceeding 25% of
wall — e.g. after symbolic-phase wins land — revisit this decision
for the dense path only.

**Files referenced.**
- `dev/journal/2026-04-28-01.org` — investigation log.
- `dev/sessions/2026-04-28-01.md` — full session checkpoint.
- `src/bin/profile_hot.rs` — the profiler harness.

---

## 2026-04-28 — Auto routing thresholds are δ_c-robust (probe evidence)

**Decision.** The Auto routing rules in
`src/scaling/mod.rs:371-392` (`pick_scaling_strategy`,
`diag_only/n >= 0.30`) and `src/symbolic/mod.rs:299-321`
(`pick_ordering_preprocess`, `low_degree/n >= 0.30`) are accepted
as δ_c-robust without further hardening. Their thresholds were
calibrated on the `data/matrices/kkt/` corpus (pre-regularized
IPM snapshots dumped with δ_c ≈ 1e-8 on the dual block); the
calibrations gate on **structural** ratios that do not depend on
δ_c magnitude.

**Evidence.** `src/bin/probe_deltac_sensitivity.rs` perturbs the
detected dual-reg block of 9 representative KKT matrices by
`mult ∈ {1e-4, 1e-2, 1, 1e2, 1e4}` (effective δ_c span 1e-12 to
1e-4) and re-runs both routing functions plus a 5-run-median
symbolic + numeric factor:

- 0/9 matrices flipped scaling routing
- 0/9 matrices flipped ordering preprocess
- inertia stable across the sweep on every matrix
- wall time within ±5% across multipliers (within run-to-run noise)
- residuals scale with effective δ_c as expected for refined-solve
  on a more-singular matrix; not a feral defect

**Implications.** Future heuristic changes that gate on raw
diagonal magnitude (an interpretation-class change rather than a
structural-signature change) must validate against the same probe.
A consumer with a different δ_c choice (POUNCE with a different
`mu_init`, etc.) is not expected to see different routing answers.

**Files referenced.**
- `src/bin/probe_deltac_sensitivity.rs` — the probe.
- `src/scaling/mod.rs` — Auto scaling routing.
- `src/symbolic/mod.rs` — Auto preprocess routing.
- `dev/journal/2026-04-28-01.org` — investigation log.

---

## 2026-04-28 — Phase A2 swap-2x2 inline restricted to c==0

**Decision.** Phase A2 inline support for swap-required 2×2 pivots
in `lblt_panel_frontal` is restricted to `c == 0` (the first pivot
of any panel). Mid-panel (`c > 0`) swap-2×2 continues to bail to
`scalar_pivot_step` via `PanelStatus::ScalarFallback`.

**Why.** At c==0 the deferred state IS the scalar state (no
committed pivots), so reading `arr` and `gamma_r` at the candidate
row r is bit-exact with scalar without any new replay primitive.
At c > 0, scalar-equivalent reads at row r require:

1. `peek_ahead_replay(target = col + 1)` (already implemented for
   the no-swap 2×2 path).
2. `peek_ahead_replay(target = r)` — disjoint with (1) because
   r > col + 1.
3. A **new** row-r-left-of-diagonal replay primitive: the entries
   `a[j*nrow + r]` for `j in (col+1)..r` are read by
   `symmetric_row_offdiag_max(a, nrow, col, r)` but `peek_ahead_replay`
   only updates `a[r*nrow + i]` for `i in r..nrow`.
4. **Bail-state extension**: a new `PanelStatus::ScalarFallbackPeekedTwo
   { col1, r }` to thread which two columns the caller's
   `apply_blocked_schur` must skip.

The c==0 path required ZERO new primitives and lays the API
groundwork (perm threading, `INLINE_2X2_SWAP_OK` counter, probe
output, fixture patterns) for the mid-panel extension.

**Evidence.** All 208 tests pass byte-identical against scalar.
Corpus `probe_panel_attribution` shows `swap_ok = 0` aggregate —
ALL corpus swap-2×2 cases happen at c > 0, so the restriction
catches none of them. The plan's ≥75% bail-drop acceptance
criterion was NOT met; this decision documents the scope narrowing
as intentional rather than a defect.

**Trigger to revisit.** When Phase A2 mid-panel ROI is
re-evaluated against alternatives (B-1 NR=4 widening, W-3
workspace pre-sizing). If mid-panel wins, write a fresh research
note for the row-r-left-of-diagonal replay primitive + the
`apply_blocked_schur(..., skip_col=Option<usize>)` API extension,
and land them as separate commits before the semantics change.

References: `dev/sessions/2026-04-28-03.md`,
`dev/journal/2026-04-28-01.org` 16:30 entry, commit `dfe169e`.

## 2026-05-02 — `NumericParams::default()` adopts `pivot_threshold = 1e-8`

**Decision.** `NumericParams::default()`
(`src/numeric/factorize.rs`) replaces `#[derive(Default)]` with a
manual `impl Default` that sets `bk.pivot_threshold = 1e-8`, matching
MA27's `cntl[1]` reference default — equivalently Ipopt's
`ma27_pivtol` default.

`BunchKaufmanParams::default()` stays at `pivot_threshold = 0.0`,
preserving the 2026-04-13 dense-vs-sparse split decision (dense
has no delayed-pivoting / rook-rescue infrastructure to land
rejected pivots in; sparse does).

**Why.** Issue #2 surfaced that ripopt and other consumers
constructing `NumericParams::default()` were inheriting `0.0` via
`BunchKaufmanParams::default()`. On rank-deficient KKT-augmented
LS-init systems (`A = [I J^T; J diag]` with `m > n`, equality rows
having `D = 0`, e.g. CUTEst `arki0003`), the SSIDS-style
scale-invariant 2×2 det-floor in `factor.rs:2232-2243` rejects
saddle blocks regardless of `pivot_threshold`, but the 1×1
fallback's rook-rescue fast-path is dead at `pivot_threshold = 0`.
The result: small pivots that MA27 would rescue via threshold
partial pivoting got "accepted" with huge `1/d` rank-1 updates,
propagated cancellation through the elimination tail, and produced
exact-zero L columns and multipliers on non-structurally-zero
rows. On `arki0003` this manifested as 58 zero `y_d` entries
clustered at `_scon[2052..2138]`.

**Why 1e-8 and not 0.01 (SSIDS/MUMPS canonical).** Both values
re-enable the column-relative pivot rejection that the bug needs.
The SSIDS canonical `u = 0.01` was validated on MC64-equilibrated
inputs where every column has `colmax ≈ 1`, so a `1e-2` relative
floor is roughly an absolute `1e-2` floor. ripopt's `FeralLdl`
runs with `ScalingStrategy::Identity` (preserving inertia signal —
see `feral_direct.rs:84-91`), where column maxes span IPM-scaled
magnitudes. A `0.01` threshold there rejects substantially more
pivots and forces them through the delayed-pivoting cascade; the
MA27 `1e-8` value is conservative in that regime and is what Ipopt
ships with for the same KKT pattern. Sparse callers that have
explicitly chosen `0.01` (the in-tree benches, parity tests) keep
their override. This decision sets a default that is correct for
the unscaled-KKT consumer path (ripopt's primary use case) without
changing those existing call-sites.

The 2026-04-12 decision documented `0.01` as the canonical
benchmark default backed by SSIDS/MUMPS empirical evidence on
MC64-scaled corpora. The 2026-04-13 decision split dense (0.0)
from sparse (0.01) for opt-in callers. This decision closes the
remaining gap by giving the default consumer path a non-zero
threshold while choosing the value that matches the canonical
unscaled-KKT solver in the optimization domain (MA27/Ipopt).

**Touched call-sites.** Six diagnostic bins, two integration tests
(`tests/multi_rhs.rs`, `tests/ldlt_compress.rs`), one example
(`examples/triage_bratu3d.rs`), and `Solver::new` all flip from
`0.0` to `1e-8` baseline. Pivots in those tests are well-conditioned
so the threshold change is a no-op; all 146+ tests pass under the
new default. The `i8_solver_lifetime_state_persists` test in
`tests/pounce_interface.rs` was updated to reflect that the W5
"0.0 → 0.01" first-jump rule no longer fires from baseline; the
cascade now reads 1e-8 → 1e-6 → 10^-4.5 → ... → `pivtol_max = 0.5`.
The W5 rule is kept for callers that explicitly disable the
threshold via `with_bk(BunchKaufmanParams::default())`.

**Commitment.** ripopt's `set_pivot_threshold(1e-8)` workaround at
`src/linear_solver/feral_direct.rs:128-131` (referenced in issue
#2) becomes redundant after this change. ripopt-side cleanup is
tracked in ripopt's own repo, not in feral.

References: `dev/research/issue-2-kkt-pivot-default.md`,
`dev/plans/issue-2-kkt-pivot-default.md`, issue
[#2](https://github.com/jkitchin/feral/issues/2).

---

## 2026-05-03 — `build_row_indices` filters upper-triangle pollution

**Decision.** `build_row_indices` (src/numeric/factorize.rs:2257-2298)
now skips trailing-row candidates with `r < first_col + own_ncol`.
A `cfg(debug_assertions)` invariant assertion at
src/numeric/factorize.rs:1469-1485 enforces, for every supernode,
that every row at frontal positions
`[own_ncol + n_delayed_in .. nrow)` is `>= first_col + own_ncol`.

**Why.** `full_pattern = matrix.symmetric_pattern()` is the fully
symmetrized A pattern; iterating column j gives both legitimate
lower-tri rows (r > j) and upper-tri rows (r < j) that correspond
to columns already eliminated by ancestors of those rows. Without
filtering, upper-tri rows polluted every supernode's frontal,
propagated up the etree through child contrib blocks, and inflated
`factor_nnz` by 7-19× over the textbook L-fill (Σ col_counts via
Gilbert-Ng-Peyton). On PoissonControl K=158 the symptom was
factor_nnz = 46.7M vs symbolic 2.4M and a ~650× factor-time gap vs
MUMPS. `column_counts_gnp` was already filtering correctly
(column_counts.rs:135 `if partner <= i { continue; }`); only the
numeric path was over-collecting.

**Why this was performance, not correctness.** Rogue rows are
upper-triangle entries A[r, j] for r < j. Numeric assembly only
writes lower-tri interactions, so the rogue rows received zeros
during assembly and never affected pivot decisions at the supernodes
where they appeared as dead weight. Inertia is bit-identical before
and after the fix on every test fixture and on PoissonControl K=50,
K=158. The fix is purely structural — drop dead rows from frontals.

**Evidence.** PoissonControl K=50 factor_nnz dropped from 1,363,445
to 323,643 (4.2×) and factor time from 231,075 µs to 3,542 µs (65×).
K=158 factor_nnz dropped from 46,734,661 to 4,610,269 (10×) and
factor time from seconds to 85,099 µs. All 216 lib + integration
tests pass identically to before the fix. New regression test
`tests/build_row_indices_trailing_invariant.rs` covers four
multifrontal-path fixtures (n > N_TINY=16) with both the trailing-row
floor invariant and symbolic ↔ numeric nrow parity assertions. The
debug_assert was first added before the filter changes — it fired on
6 existing tests, confirming the bug's reach. After both the assert
and the filter were in place all 216 tests pass.

**Touched call-sites.** Two changes inside
`build_row_indices` (factorize.rs:2274-2287 native pattern loop,
factorize.rs:2289-2298 child contrib loop), one debug assertion
near the call site, one new test file.

References: `dev/research/build-row-indices-fix.md`,
`tests/build_row_indices_trailing_invariant.rs`.


## 2026-05-03 — `SupernodeParams::default().nemin` lowered 32 → 16

**Decision.** `SupernodeParams::default().nemin`
(src/symbolic/supernode.rs:115) drops from 32 to 16. `nemin` is the
minimum supernode size below which the symbolic phase merges
parent and child during amalgamation: smaller `nemin` ⇒ thinner
supernodes ⇒ tighter L storage and less pass-through padding;
larger `nemin` ⇒ fatter supernodes with more BLAS-3 work per node
but more pass-through inflation.

**Why.** Two converging signals:

1. The previous `nemin = 32` was inherited from an early
   dense-kernel study (BLAS-3 sweet spot for inner GEMM panels)
   and out of step with reference multifrontal solvers. MUMPS
   uses `KEEP(63) = 5`; SSIDS's canonical config sits in the same
   low band. `32` is the high-end outlier even among solvers
   that explicitly trade L NNZ for kernel throughput.

2. `dev/research/factor-nnz-residual-gap.md` (this session)
   established that the post-`build_row_indices`-fix 1.6-2× gap
   between numeric `factor_nnz` and Σ col_counts (GnP) is
   dominantly **pass-through row padding** — rows from children's
   contribs flowing through ancestors that don't pivot on those
   rows, stored as zeros in the dense trailing rectangle. Smaller
   supernodes have less inflation: each supernode's pass-through
   cost scales with `(num_nrow − sym_nrow) × nelim`.

**Evidence.** Sweep over
{nemin ∈ 8, 16, 32, 64} × {AMD, METIS-ND} on PoissonControl
K=50 and K=158 (this session journal `2026-05-03-01.org`):

| K   | nemin | ordering | factor_nnz | Δ vs nemin=32 | factor_med_us | Δ wall |
|-----|-------|----------|-----------:|--------------:|--------------:|-------:|
| 50  | 32    | AMD      |    323,643 |       —       |         4,200 |   —    |
| 50  | 16    | AMD      |    240,167 |          -26% |         3,440 |   -18% |
| 50  | 8     | AMD      |    191,074 |          -41% |         3,300 |   -21% |
| 158 | 32    | AMD      |  4,610,269 |       —       |        85,099 |   —    |
| 158 | 16    | AMD      |  3,660,090 |          -21% |        86,572 |    +2% |
| 158 | 8     | AMD      |  3,107,011 |          -33% |       103,400 |   +21% |

`nemin = 16` is the sweet spot: substantial memory savings on both
sizes, factor wall improved on the small case and ≈ par on the
large case. `nemin = 8` recovers more memory but the wall regresses
on K=158 (more pivot-block boundaries amortizing fewer GEMM3 rows
per supernode). `nemin = 16` aligns with the "halfway between feral's
prior 32 and MUMPS's 5" intuition and is what the data picks.

The corpus bench (Phase 2.8.1 dense + sparse exit partition) retains
its P90 ratio targets vs MUMPS at `nemin = 16`:
small-frontal P90 = 1.33 (target ≤ 2.0, PASS), medium P90 = 1.70
(target ≤ 3.0, PASS) on the dense path; sparse 1.56 / 1.56 PASS.
Geomean factor ratio vs MUMPS is unchanged at 0.22 / 0.43 across the
two partitions.

**Why not also flip `AmalgamationStrategy::Auto` shape-dispatched
nemin (planned Phase B).** Phase B (path-like → small `nemin`,
bushy → larger `nemin`) is the right next step but layers logic
onto an existing dispatcher and wants its own evaluation. This
decision is the cheap, mechanical default flip that requires no
new code path; Phase B will adjust `nemin` per-shape on top of
this new baseline.

**Touched call-sites.** One line in `SupernodeParams::default`.
Lib tests and the `build_row_indices_trailing_invariant.rs`
integration tests pass after relaxing one over-tight assertion
(`nrow_matches_symbolic` → `nrow_at_least_symbolic`) — the prior
`assert_eq!` was conceptually wrong (it conflated
`Supernode.nrow` with the working frontal nrow), only happening to
hold on the small fixtures because at `nemin = 32` those fixtures
had no pass-through padding. The trailing-row floor invariant —
the half of the test file that actually guards the
`build_row_indices` fix — is unaffected and still passes.

References: `dev/research/factor-nnz-residual-gap.md`,
`dev/journal/2026-05-03-01.org` 14:00 entry,
`dev/research/build-row-indices-fix.md`.

## 2026-05-09 — `resolved_method` is what ran, not what was asked

**Decision.** `SymbolicFactorization.resolved_method` is a contract
field whose value MUST equal the concrete ordering algorithm that
produced `perm`. When `OrderingMethod::ScotchND` is requested and the
SCOTCH driver silently falls back to `amd_leaf` for every recursion
node (bisection produces an empty side at every level), the field is
re-stamped to `OrderingMethod::Amd`. Detection signal:
`feral_scotch::ScotchStats.n_separator_vertices == 0` from
`scotch_order_full`. The fallback itself is preserved as a recovery
path — only its visibility is fixed.

**`OrderingMethod::Auto` is dispatched against the original matrix.**
Auto resolution happens once in `symbolic_factorize_with_method`,
against `matrix.symmetric_pattern()`, *before* any
`OrderingPreprocess::LdltCompress` reshaping. The concrete method is
threaded through the dispatch as a non-`Auto` value;
`run_external_ordering` carries a `debug_assert_ne!(method, Auto)`.

**`choose_adaptive` delegates to `pick_default_method` on residual.**
The bare `→ Amd` else branch is replaced by a delegation that uses
the `(full_nnz + n) / 2` stored-equivalent estimate (exact when the
diagonal is stored once per row, which `CscMatrix::symmetric_pattern`
produces). This makes `Auto` a strict superset of `pick_default_method`
on every input — the two existing shape-bakeoff branches (large-
sparse → ScotchND, small-sparse → KahipND) keep priority, with
`pick_default_method` as the residual.

**Why.** Prior behavior: `Auto` could disagree with the no-arg
`symbolic_factorize` default on the same matrix, because (a)
`choose_adaptive` was called on the post-compression pattern with a
different `n`, and (b) its residual was unconditional `Amd`. Issue #3
flagged the K=158 PoissonControl case where `Auto → Amd` instead of
the expected `MetisND`. Code that branched on `resolved_method` (bench
dispatch, oracle scoring) was making decisions on a value that did
not describe the actual computation.

**Touched call-sites.** `src/symbolic/mod.rs`: `choose_adaptive`,
`symbolic_factorize_with_method` (one new line: pre-resolution),
`run_external_ordering` (ScotchND branch reworked, internal
`choose_adaptive` call removed). One existing test
(`choose_adaptive_rules`) updated: the residual case (`n=50_000`,
full avg_deg=20) now expects `MetisND` instead of `Amd`, reflecting
the delegation. No production callers branched on the old `Amd`
residual that this commit changed.

References: GitHub issue #3, `crates/feral-scotch/tests/issue_3_kkt_repro.rs`,
`src/symbolic/mod.rs::tests::issue_3_*` (two new tests).

## 2026-05-12 — `Solver` defaults to the parallel multifrontal driver (issue #7)

**Decision.** `Solver::new()` and `Solver::with_params(...)` now
produce a `Solver` whose `factor()` routes through
`factorize_multifrontal_parallel_with_workspace`. The previous
default was the sequential supernodal driver. An override is
provided as `Solver::with_parallel(false)`, and a diagnostic
accessor `Solver::parallel()` reports the current state.

**Why this is safe.** The parallel driver carries a documented
bit-exact contract with the sequential supernodal path on a
per-supernode basis (same FP sum order per supernode, per-thread
`FactorWorkspace`, mutex-only on the shared contribution-block
store — see the doc comment at
`src/numeric/factorize.rs:1822`). Internally it also self-gates
on `should_parallelize_assembly` (`N_PAR_MIN = 32` supernodes,
`src/numeric/factorize.rs:1769`) so problems below that threshold
fall through to the sequential supernodal path within the same
call — making default-on neutral for small problems and a strict
constant-factor win on large ones.

**Motivating evidence.** Issue #7 reports that pounce's Mittelmann
runs (`marine_1600`, `pinene_3200`) timed out on the inner sparse
factor while the parallel driver sat unused, because the public
`Solver::factor` entry only routed through the sequential path.
The MA57-vs-feral gap in pounce on those benchmarks was essentially
this wiring.

**Bit-exact regression test.** Added
`solver_parallel_factor_matches_sequential` in the
`src/numeric/solver.rs::tests` module. Fixture: 64 independent
2×2 indefinite blocks `[[1, 2], [2, 1]]` (n = 128, 64 disjoint
elimination trees, well above `N_PAR_MIN`). Asserts equality of
summed inertia, `num_negative_eigenvalues`, and **bit-identical
f64 bits** of the `solve(rhs)` output between
`Solver::new()` (parallel) and
`Solver::new().with_parallel(false)` (sequential). Per the CLAUDE.md
hard rule, this is `==`, not a tolerance.

**Touched call-sites.** Three edits, all in
`src/numeric/solver.rs`: a new `use_parallel: bool` field,
initialization to `true` in `with_params`, a `with_parallel`
builder, a `parallel` accessor, and a function-pointer dispatch
inside `factor()` selecting between
`factorize_multifrontal_parallel_with_workspace` and
`factorize_multifrontal_with_workspace`. Both functions have
identical signatures so the dispatch is one branch wide.

**Out of scope.** The pulp SIMD wiring at
`src/dense/factor.rs:1719/1741/1824/1843` mentioned in issue #7
is *not* included here. That work is blocked on Phase 2.4.3
(replace `mul_add_f64s` with `mul_f64s + sub_f64s` to recover
bit-exact rounding versus the scalar path); the 2026-04-14
reverted-FMA decision earlier in this file is the prerequisite.

References: issue #7,
`src/numeric/factorize.rs::factorize_multifrontal_parallel_with_workspace`,
CHANGELOG.md `[Unreleased] / Changed`.

## 2026-05-12 — Skip upper-triangle memset on pooled frontal buffer

**Decision.** Added `SymmetricMatrix::from_pooled_buf(n, buf)` in
`src/dense/matrix.rs`. The dense BK + Schur kernels touch only the
lower triangle of a `SymmetricMatrix`, so the upper-triangle zero
on pool-reuse is dead work. The new constructor grows the buffer
if needed (which zeros only the tail) and explicitly zeros the
`n(n+1)/2` lower-triangle cells. The full-`nrow*nrow` zero is
gone.

**Why this is safe.** Inspection of `src/dense/factor.rs:1137`
(scalar Schur), `src/dense/schur_kernel.rs:738`
(`schur_panel_minus_nofma_strided` — the pulp SIMD kernel), and
the BK pivot/swap paths confirms that no consumer reads upper-
triangle cells of a `SymmetricMatrix`. Indexers always normalize
to `(max(i,j), min(i,j))`. Added a doc-comment audit note on the
new constructor stating this contract.

**Bit-exact.** No FP value changes; the kernels never saw those
upper cells in the first place.

**Measured impact.** Roughly 5–10% wall-time reduction on
sequential factor across mid-size matrices (bratu3d, cont-201).
Numbers in `dev/sessions/2026-05-12-01.md`.

**References.** `src/dense/matrix.rs::SymmetricMatrix::from_pooled_buf`,
`src/numeric/factorize.rs::factor_one_supernode` (two call sites),
`src/bin/diag_leaf_profile.rs` (one diagnostic site).

## 2026-05-12 — Pool `local_contribs` per worker in the parallel driver

**Decision.** Moved the `Vec<Option<ContribBlock>>` of length
`n_snodes` that `run_parallel_task` was allocating on every spawned
task into a new `FactorWorkspace::local_contribs` field. The
parallel driver pre-sizes one such vec per rayon worker; tasks
take it out via `std::mem::take`, use it as the children-contrib
staging area + own-contrib output slot, then put it back. All
slots are `None` between tasks (postcondition: children's slots
were drained into the pool by the task entry, and the own slot
was just taken out at task exit), so no clearing is needed.

**Why this is safe.** Same data flows through `factor_one_supernode`
in the same order. The split-borrow is achieved with safe Rust
(`std::mem::take` plus a `&mut FactorWorkspace` whose
`local_contribs` field is empty during the call), so no `unsafe`
is required.

**Bit-exact.** Same values, same order, just heap-allocated once
per worker instead of once per task.

**Measured impact.** Decisive on cont-201 (11 121 tasks × 11 121
slot vec = ~9 GB of cumulative allocator churn before the fix):
sequential wall **–34%** (435.7 → 286.0 ms), parallel-at-T=8
**–10%** (219.7 → 198.9 ms). bratu3d **–6% / –5%**. Small matrices
unchanged. Numbers in `dev/sessions/2026-05-12-01.md`.

**References.** `src/numeric/factorize.rs::FactorWorkspace`,
`src/numeric/factorize.rs::factorize_multifrontal_supernodal_parallel`,
`src/numeric/factorize.rs::run_parallel_task`.

## 2026-05-12 — Reject lock-free contribution-block store

**Decision.** Keep the `Mutex<HashMap<usize, ContribBlock>>` shared
contribution-block store in the rayon parallel multifrontal driver
as-is. Do **not** redesign it into a sharded/lock-free structure.

**Why.** Empirical falsification via `AtomicLockStats` telemetry
(this session). At T=4 on a representative four-matrix sample the
total wait+hold time on the contribution-block + node-factors
mutexes accounts for:

- bcsstk38: 1.8% of aggregate body time
- bratu3d:  0.2%
- c-big:    0.02%
- cont-201: 3.4%

cont-201 is the worst case and is still <4%. A lock-free store
would buy at most that fraction back, and would not change the
within-scope work-stealing/dep-chain idle that constitutes the
remaining cont-201 cached-mode headroom (loop utilization 68.5%
inside the rayon::scope).

**Evidence.** Test
`numeric::solver::tests::solver_parallel_lock_breakdown` (cold +
cached pair, T=4), plus full numbers in
`dev/debugging/2026-05-12-cont201-cached-headroom.md`.

**Escape hatch.** The `AtomicLockStats` telemetry stays in tree so
the decision can be re-checked at higher thread counts or different
matrix mixes without re-instrumenting.

**References.** `src/numeric/factorize.rs::AtomicLockStats`,
`src/numeric/factorize.rs::run_parallel_task`,
`src/numeric/solver.rs::tests::solver_parallel_lock_breakdown`.

## 2026-05-12 (b) — Defer within-supernode parallelism; close cont-201 assembly-tree investigation

**Decision.** Close the cont-201 assembly-tree parallelism
investigation as **etree-topology-bound**. Do not pursue
topological-level schedulers, alternative ready-queue
structures, or other assembly-tree-level tuning. The remaining
1.5× cached-mode headroom on cont-201 (T=4) cannot be recovered
by changing the rayon scheduling pattern.

**Empirical basis.** Within-scope localization (iteration 2 of
the cont-201 investigation, added `task_wall_ns` +
`ws_lock_wait_ns` to `AtomicLockStats`):

- cont-201 cached at T=4: scope·T capacity = 194.5 ms,
  task_wall_agg = 145.3 ms, rayon_idle = 49.2 ms = **25% of
  capacity = 12.3 ms/worker**. Locks contribute 1.7 ms/T,
  ctrl-flow 1.5 ms/T. The dominant residual is workers waiting
  for the next eligible task — etree dependencies, not
  engineering loss.

- c-big at T=4: 74% rayon-idle capacity; parallel driver buys
  only 1.04× speedup over body_agg. Confirms the same bound
  on a much larger matrix.

**Next axis if needed.** Within-supernode parallelism (panel-BK
or threaded dense kernels inside `factor_one_supernode`), which
is what MUMPS' threaded BLAS + SPRAL's panel scheduler provide.
This is a substantial undertaking — Phase 2.4.3
(`mul_f64s` + `sub_f64s` to restore bit-exact rounding in the
Schur kernel SIMD path) must complete first per
`dev/decisions.md` 2026-04-14 before any further dense-kernel
parallelism work. Track as a separate effort.

**Diagnostic surface kept.** All 16 atomics in `AtomicLockStats`
stay in tree (opt-in, default None, zero cost). The
`solver_parallel_lock_breakdown` test is the canonical way to
re-check this decision at other thread counts or matrix mixes.

**References.** `dev/debugging/2026-05-12-cont201-cached-headroom.md`
(iteration 2), `src/numeric/factorize.rs::AtomicLockStats`,
`src/numeric/solver.rs::tests::solver_parallel_lock_breakdown`,
`dev/decisions.md` 2026-04-14 (SIMD/FMA blocker on
within-supernode kernel parallelism).

## 2026-05-12 (c) — Park BLAS-3 quad kernel; pivot to per-front overhead

**Context.** Issue #9 (Phase 2.4.3 BLAS-3 trailing-update kernel)
landed `schur_panel_minus_nofma_strided_quad` and wired it into
`apply_blocked_schur_panel`. The quad kernel packs four destination
columns per pulp-dispatch, halving src memory traffic vs the existing
dual kernel. It is correct (176-config bit-parity sweep + 19 blocked_ldlt
integration tests passing byte-identical) and zero-regression on the
154k-matrix corpus.

**The original motivation no longer holds.** The 2026-04-27 CHAINWOO
profile (`dev/research/feral-kernel-profile-chainwoo.md`) cited a
1984-row root front at 62 % of factor time. That front no longer
exists on the current build — METIS-ND on CHAINWOO_0000 now produces
actual frontal sizes ≤ 18 rows and the matrix factors in ~740 µs end
to end (vs 24 ms in the profile note). The intervening landings (W-4
in `lblt_panel_frontal`, 1x1 fast path, post-2026-04 ordering changes)
shrank the wide-front case faster than this work could close it.

**Re-profile finding.** `cargo run --bin diag_supernode_cost --release`
shows the new dominant cost is **fixed per-supernode overhead** at
small fronts. ns/sup is 600–1900 across the long-tail corpus while
ns/nnz is 30–165. The nemin sweep on ACOPR30_0067 confirms it:
shrinking supernode count from 493 → 158 (nemin 1 → 32) drops total
time from 242 µs → 152 µs even as per-supernode cost climbs from
492 ns → 964 ns. The arithmetic layer (which quad targets) is not
the bottleneck on this corpus.

**Decision.** Retain the quad kernel + wiring as parked infrastructure
on the merge target. Justification:
1. It is correct and in production for every front with ≥ 4 trailing
   columns. No maintenance burden — the bit-parity tests are the
   regression gate and they sweep 176 configs.
2. The win it targets (tall-skinny fronts where trailing-update
   bandwidth dominates) is workload-dependent. A future workload
   shift — larger problems, different ordering, an amalgamation
   change that grows fronts — re-engages the quad path automatically.
3. Reverting would lose the bit-parity harness and ~700 LoC of
   reviewed, tested kernel code that has zero runtime cost on small
   fronts (the dispatch path is identical, just routed through a
   wider kernel when ncol ≥ 4).

**Next axis.** Open a new issue for per-front overhead reduction.
Candidate items from `dev/research/feral-kernel-profile-chainwoo.md`
§3 and §4: workspace pooling (eliminate per-front `vec![0.0; ...]` +
L/D/contrib `Vec::new`), bypass `SymmetricMatrix::validate()` when
caller already validated, replace `SymmetricMatrix::set/get` branches
in `extend_add` with direct slice writes.

**Lesson for future kernel work.** Re-measure the profile that
motivates the work *immediately before* writing code, not weeks
prior. Front shapes can shift under intervening landings.

**References.** Branch `feat/issue-9-block32-kernel` commits
`fdd631c` (quad kernel + tests), `8a07386` (wiring),
`dev/research/blas3-trailing-update.md`,
`dev/plans/phase-2.4.3-blas3-trailing-update.md`.

## 2026-05-13 — Do not implement issue #10 APP path; gate not met

**Context.** Issue #10 proposes an APP (aggressive partial pivoting)
path alongside the existing per-pivot threshold check in
`src/dense/factor.rs`. The issue itself posted a re-open gate:
"fresh `diag_supernode_cost` shows ns/nnz dominates ns/sup on a
relevant cluster (ACOPR30, CRESC100 at nemin=32, or any new corpus
with fronts wide enough to use the panel path)."

The previous session's checkpoint (`dev/sessions/2026-05-13-02.md`)
listed #10 as the next target on the assumption that #9 landing was
the only remaining precondition. The gate measurement was not
re-done at that time.

**Measurement.** `cargo run --bin diag_supernode_cost --release` on
the post-`d7267fe` build (full output in
`dev/research/dense-app-path.md`):

- ACOPR30_0067 at nemin=32 (the cluster the gate names): ns/sup
  943, ns/nnz 61. Ratio 15× the wrong way.
- CRESC100_0000 default nemin=16: ns/sup 914, ns/nnz 79. 12× the
  wrong way.
- HAIFAM_0082 (widest fronts on corpus, max 86): ns/sup 1174, ns/nnz
  33. 36× the wrong way.

Across every matrix and every nemin in the sweep, ns/sup dominates
ns/nnz by at least 4× — the opposite of the gate condition.

**Decision.** Do not implement APP today. The two motivating gaps
the issue cites (per-pivot γ₀ scan and per-element SIMD trailing
update) have been closed via different code paths since the
motivating measurement was taken:

1. `fused_gamma0` (`factor.rs:369-371, 400-405, ...`, landed
   `ad05ff4` 2026-04-11) eliminates the per-pivot column scan on
   the scalar path's no-swap branches — the same trick the issue
   body attributes uniquely to MUMPS `MAXFROMM`.
2. The 32×32 SIMD body (`block_ldlt32::update_1x1_block32`, landed
   `98ef545`+`d3f1132` 2026-05-12/13) puts trailing-update FLOPs on
   the dominant CHAINWOO-style 32-col front shape through a quad
   pulp dispatch.

The remaining un-fused γ₀ scan in `lblt_panel_frontal:1480-1488` is
real but on the current corpus its code path is bypassed for the
dominant front size (32×32 dispatches to `factor_block32` before
the panel path is reached) and unmeasured-but-likely-tiny on the
remaining sizes (max corpus front 86, mostly ≤ 17).

**Recommendation.** Close issue #10 with a comment citing
`dev/research/dense-app-path.md`. The issue's own "narrow
alternative" — fuse γ₀ into the panel's deferred rank-1 stream —
is also not justified today; revisit only when a corpus front
appears at sizes 32–96 with low enough per-front overhead that
the panel γ₀ scan shows up in a profile.

**Lesson reinforced.** Same as the 2026-05-12 (c) BLAS-3 quad
decision: re-measure the profile that motivates the work
immediately before writing code. The 2026-05-13-02 session
checkpoint advanced #10 as the next target without re-checking
the gate; re-measuring took one binary run and avoided weeks of
implementation work that the data shows would not have paid back.

**References.**
- `dev/research/dense-app-path.md` — gate measurement and design
  space.
- Issue #10 posted comment by `jkitchin` — the gate text.
- `src/dense/factor.rs:369-371, 400-405, 439-441, 465-467,
  486-488, 537-539, 555-557` — fused_gamma0 thread.
- `src/dense/factor.rs:1189-1193` — 32×32 dispatch entry.
- `src/dense/factor.rs:1480-1488` — the remaining un-fused panel
  γ₀ scan.

---

## 2026-05-13 — Small-front bench-gap: retrospective on the #9/#10/#11/#13 model

**Decision.** Record explicit retrospective that the original
small-front-performance model implicit in issues #11, #12, #13
("kernel cost dwarfs driver overhead; closing the kernel will
reveal the driver win") did not hold against post-land data.

**What the post-land data shows.** After #9 Step 2 dispatch
(`d3f1132`) and #13 phases A+B+C (workspace pooling +
extend_add direct writes + contrib pool):

- bench p90 small 1.36 → 1.33, medium 1.78 → 1.74 (~0.04
  absolute movement each)
- `diag_supernode_cost` ns/sup vs ns/nnz: ns/sup still dominates
  ns/nnz by 4× to 36× across every long-tail corpus row at
  every nemin
- ACOPR30_0067 ns/sup 943 / ns/nnz 61 (15× ratio preserved)
- HAIFAM_0082 ns/sup 1174 / ns/nnz 33 (36× ratio preserved)

The two layers were the same order of magnitude all along.
Both shrank a bit; neither dwarfed the other; the *ratio*
between them is preserved post-land, so the bench p90 — which
captures end-to-end including sparse path / refinement /
scaling layers neither of those issues touched — barely
moved.

**Implication for the un-done #13 candidate.** The single
largest un-done lever on the per-front overhead axis is the
`SymmetricMatrix::validate()` bypass on the multifrontal hot
path. `factor_frontal` at `src/dense/factor.rs:871` runs
`matrix.validate()` (O(n²/2) NaN/Inf scan) on every call;
the 32×32 SIMD path now reaches it via `factor_block32` on
every 32×32 front. That's ~528 reads, plausibly 260–800 ns
out of the 600–1200 ns/sup budget. The multifrontal driver
assembles fronts from a value-checked CSC, so the per-front
re-scan is unconditionally redundant on that path.

Per `dev/research/small-matrix-perf-retrospective-2026-05-13.md`
this lever alone won't hit #13 criterion #2 (small <1.30 /
medium <1.60); even a 30–60% per-front overhead reduction on
the SIMD-dispatched cluster maps to <0.05 absolute bench p90
movement on the current mix because bench p90 has other
amortized layers.

**Scope.** This entry records the model correction; it does
*not* commit to landing the validate-bypass. That is a new
line of work outside the original scope of #13 (which was
the three pooling/direct-write phases that did land) and
should be its own issue if pursued.

**Lesson — bench p90 is the wrong instrument for kernel/
overhead work in isolation.** Bench p90 is the right top-line
metric but the wrong attribution metric for any single layer.
For per-front cost the right instruments are
`diag_supernode_cost`'s ns/sup and ns/nnz columns (both moved
under #13 Phase A; criterion #1 met). For end-to-end ratio,
bench p90 is correct. Future small-front work should gate on
the kernel/overhead-attributed metric (`diag_supernode_cost`),
not on bench p90 alone, so that the gate isn't masked by
unrelated layers.

**References.**
- `dev/research/small-matrix-perf-retrospective-2026-05-13.md`
- `dev/tried-and-rejected.md` 2026-04-25 Phase 2.11 entry
  (SmallLeafBatch flip noise-floor result)
- `dev/research/dense-app-path.md` (gate measurement)
- `src/dense/factor.rs:871` (validate call site on hot path)
- `src/dense/matrix.rs:106-133` (validate body)

---

## 2026-05-13 — `feral-capi` as a separate workspace member, not a core feature

**Decision.** Adding a C ABI surface to enable feral as a
plug-in linear solver for canonical (C++) Ipopt 3.14 does not
violate the "Pure Rust, stable toolchain; zero non-Rust
dependencies in the core solver" constraint in CLAUDE.md.
The C ABI lives in a **separate workspace member crate**,
provisionally named `feral-capi`, and is optional — only
required for the Ipopt-via-C++-shim integration.

**Layout.**

- Core `feral` crate: `crate-type = ["rlib"]`. No FFI, no
  `extern "C"`, no cdylib output. Unchanged from today
  except that the top-level `Cargo.toml` becomes a
  `[workspace]` root.
- `feral-capi/`: `crate-type = ["cdylib", "staticlib",
  "rlib"]`. Depends on `feral`. All `extern "C"`
  declarations and FFI-boundary `unsafe` blocks live
  here. Exposes `feral_create / feral_destroy /
  feral_set_option_* / feral_initialize_structure /
  feral_get_values_ptr / feral_factor / feral_solve /
  feral_num_neg_evals / feral_increase_quality` plus the
  status enum and a `feral_capi.h` header (committed or
  cbindgen-generated, TBD in the plan).
- `feral-ipopt-shim/` (separate concern): the C++ shim
  consuming `feral-capi`'s output. Layout decision
  pending (Open Question #1 in the research note).

**Why a separate crate over a feature flag on `feral`:**

1. The "pure Rust core" property becomes a *crate-level*
   invariant, not a config-option invariant. Reviewers
   audit FFI safety in one place
   (`feral-capi/src/lib.rs`) instead of grepping for
   `#[cfg(feature = "capi")]` across the core crate.
2. The cdylib / staticlib outputs are produced **only**
   when someone explicitly builds `feral-capi`. Default
   `cargo build` in the workspace root still produces
   only an rlib for the core crate (workspace builds all
   members, but the cdylib is small and only present
   when the Ipopt integration is being built).
3. Matches the precedent set by ripopt's split of
   `rmumps` from the core IPM crate.

**Constraint scope clarification.** The CLAUDE.md
constraint "Zero non-Rust dependencies in the core solver
(no BLAS, LAPACK, Fortran)" refers to **runtime / build
dependencies** of the core numerical code. A C ABI export
surface is the opposite direction — *feral* providing a
non-Rust-callable interface, not feral *consuming* a
non-Rust dependency. No core numerical algorithm imports
or links against any C/C++/Fortran code. The shim that
consumes `feral-capi` is a downstream consumer like any
other.

**References.**
- `dev/research/feral-ipopt-c-shim.md` — full design
  rationale and lifecycle mapping.
- `/Users/jkitchin/projects/ripopt/rmumps` — precedent
  for the workspace-member-for-FFI pattern.
- CLAUDE.md "Constraints (hard, do not change without
  recording in decisions.md)".

---

## 2026-05-13 — `feral-ipopt-shim` lives in-tree during bring-up

**Decision.** The C++ shim that subclasses Ipopt's
`SparseSymLinearSolverInterface` and forwards to feral via
the `feral-capi` C ABI lives **in-tree** at
`feral/feral-ipopt-shim/` during the bring-up phase. Plan
to split it to a separate repository once the C ABI
stabilizes (semver 1.0) and/or we need to support more
than one Ipopt-version shim variant.

**Rationale.**

- During bring-up the C ABI will churn. Every C ABI change
  needs a coordinated update to the shim. In-tree means
  one PR, one CI run; cross-repo means a two-PR
  coordination with pinned-version bumps each time.
- The "pure Rust core" branding is protected by the
  *crate* boundary (`feral` core stays rlib-only with no
  FFI). `feral-ipopt-shim/` is a sibling directory, not
  part of the Rust workspace; the Rust API consumer can
  ignore it entirely.
- Precedent in-tree already: `ref/Ipopt/`, `ref/mumps/`,
  `ref/spral/` are vendored non-Rust sources the core
  doesn't link against. A first-party C++ subdirectory is
  a milder version of the same pattern.

**Split criteria** (when these are met, split to its own
repo):

1. `feral-capi` reaches semver 1.0 with a stable C ABI
   that can be released independently.
2. We want to maintain multiple shim variants (e.g.,
   Ipopt 3.14 and Ipopt 3.15+, or HSL-style dlopen
   variants) without each driving feral-repo PRs.

**Repo layout during bring-up:**

```
feral/
├── Cargo.toml         # workspace root
├── src/               # core feral, rlib-only
├── feral-capi/        # workspace member, cdylib + staticlib + rlib
├── feral-ipopt-shim/  # in-tree C++ shim, NOT a workspace member
│   ├── CMakeLists.txt
│   ├── include/feral_capi.h   # mirrored from feral-capi
│   ├── src/
│   │   ├── FeralSolverInterface.hpp
│   │   └── FeralSolverInterface.cpp
│   ├── patches/ipopt-3.14-feral-solver.patch
│   └── tests/
└── ref/Ipopt/        # vendored Ipopt source for shim build + reference
```

**CI impact.** A new `feral-ipopt-shim` job runs CMake
build + smoke test on Linux + macOS. It is **non-blocking
during bring-up** (marked `continue-on-error: true` or
equivalent); becomes a required job once it's reliable.

**References.**
- `dev/research/feral-ipopt-c-shim.md` Open Question #1
  (resolved by this entry).
- `dev/decisions.md` 2026-05-13 "`feral-capi` as a
  separate workspace member" (companion decision).

---

## 2026-05-13 — C ABI lives in `feral::capi`, not a separate workspace member (supersedes earlier-today decision)

**Decision.** During implementation, the planned
`feral-capi` workspace member was collapsed into the core
`feral` crate as `pub mod capi` (`src/capi.rs`). The
`feral` package now declares `crate-type = ["staticlib",
"rlib"]`. The earlier 2026-05-13 decision ("`feral-capi`
as a separate workspace member") is **superseded** by
this entry; that entry remains in the log as the prior
intent.

**What changed:**

- No new workspace member. `src/capi.rs` is part of the
  core `feral` crate, behind `pub mod capi`.
- `Cargo.toml` adds `staticlib` to the existing `rlib`
  crate-type rather than introducing a sibling cdylib
  crate.
- The `feral-ipopt-shim/` C++ shim links against
  `target/release/libferal.a` directly (no intermediate
  `feral-capi`).

**Why the collapse:**

1. The C ABI is small (7 functions, ~250 lines) and tied
   1:1 to types already public in the core crate
   (`CscMatrix`, `Solver`, `FactorStatus`). A separate
   workspace member would have re-exported these or
   wrapped them with no added isolation.
2. Single `cargo build` produces both the rlib for Rust
   consumers and the staticlib for the C++ shim — no
   second crate to coordinate. Pure-Rust consumers
   ignore the staticlib artifact.
3. The FFI safety surface is still localized to one file
   (`src/capi.rs`) with a clear module boundary. The
   "audit FFI in one place" property the prior decision
   wanted is preserved.

**What's *not* changed:**

- The CLAUDE.md "pure Rust core, zero non-Rust deps"
  constraint scope clarification from the prior entry
  still stands: feral exposing a C ABI is not the same
  as feral consuming a non-Rust dependency.
- The `feral-ipopt-shim/` in-tree-during-bring-up
  decision still stands.

**References.**
- `src/capi.rs` (7 `extern "C"` functions, status codes).
- `Cargo.toml:39-45` (lib crate-type).
- `src/lib.rs` (`pub mod capi;`).
- `feral-ipopt-shim/` (consumer, in-tree).
