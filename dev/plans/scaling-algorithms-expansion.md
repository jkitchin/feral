# Scaling Algorithms Expansion — Plan

**Status:** draft
**Created:** 2026-04-23
**Research note:** `dev/research/` (see §2 below — a research note will be
written before Step 2 lands, per CLAUDE.md mandate).
**Crucible concept:** `.crucible/wiki/concepts/matrix-scaling-algorithms.org`
**Primary sources (ingested):**
- `ruiz2001` — Ruiz 2001 RAL-TR-2001-034, `.crucible/sources/external/pdfs/ruiz-2001-equilibrate.pdf`
- `bertsekas2001` — Bertsekas 2001 auction encyclopedia article,
  `.crucible/sources/external/pdfs/bertsekas-auction-algorithms.pdf`
- `reid2023` — MC29 HSL specification, `.crucible/sources/external/pdfs/mc29-hsl-spec.pdf`
**Bib entries:** `dev/references.bib` §"Norm-balancing scaling" — 6 entries
(`ruiz2001equilibrate`, `knight2014symmetry`, `curtis1972automatic`,
`bertsekas1988auction`, `bertsekas2001auctionencycl`, `sinkhorn1967doubly`).

---

## 1  Motivation

FERAL's scaling module today has two production algorithms:

- `InfNorm` — labelled "Knight-Ruiz ∞-norm iterative equilibration" at
  `src/scaling/infnorm.rs:1`, but structurally it is the Ruiz 2001
  symmetric iteration (Ruiz Algorithm 2.1 with `DR = DC` when `A = A^T`).
  The comment is imprecise; the algorithm is correct.
- `Mc64Symmetric` — Duff-Koster matching + diagonal-maximization scaling
  at `src/scaling/mc64.rs`, selected by `pick_scaling_strategy` when the
  arrow-KKT shape heuristic fires (`diag_only / n ≥ 0.3`).

Two practical gaps that keep biting us:

1. **No safe fallback for matrices where neither of the two existing
   strategies balances well.** MSS1_0009 is a known case; it forced the
   Policy 4 fallback rule (`auto` diagnostic + ratio guards) as
   workaround. A genuine least-log-squares scaling (Curtis-Reid / MC29)
   is the classical "safe" answer and we do not have one.

2. **No rigorous implementation of the highest-performance algorithm
   in the scaling literature.** The current `InfNorm` is Ruiz 2001 in
   spirit but does not document the convergence contract, does not
   apply the symmetric-preserving fixed-point criterion from the
   paper, and cannot be claimed to equilibrate "asymptotically" to
   rate 1/2. A first-class Ruiz 2001 kernel that cites the paper,
   matches its notation, and tests against the paper's stated
   asymptotic rate would close that gap.

Per the user's request (2026-04-23): add to the plan the
highest-performance scaling algorithm, plus one safe alternative.

## 2  Algorithm selection

### 2.1 Highest-performance: Ruiz 2001 ∞-norm

**Paper:** Ruiz (2001), "A Scaling Algorithm to Equilibrate Both Rows
and Columns Norms in Matrices", RAL-TR-2001-034.

Cite key: `ruiz2001equilibrate` (`dev/references.bib`), ingested as
`ruiz2001` in Crucible.

Properties relevant to FERAL (from the paper abstract + §3):

- Scales row and column ∞-norms to 1 simultaneously via alternating
  diagonal square-root normalization.
- **Preserves symmetry** on symmetric input (Algorithm 2.1 with
  `DR = DC`). This is *the* reason to prefer it over Sinkhorn-Knopp
  for our symmetric-indefinite use case.
- **Asymptotic linear convergence rate 1/2** for the ∞-norm variant.
  Each iteration halves the distance to the fixed point.
- Converges in **one iteration** on diagonally dominant matrices
  (paper §3, para following Algorithm 2.1).
- After one iteration all entries ≤ 1 in absolute value (paper §3
  eq. between (3.2) and (3.3)).
- Extensions in paper §5 to 1-norm (requires extra hypotheses) and
  2-norm (same framework).

**Decision:** upgrade `src/scaling/infnorm.rs` to be a first-class,
documented Ruiz 2001 implementation. Rename to `ruiz.rs` and expose a
new `ScalingStrategy::Ruiz` variant; retain `InfNorm` as an alias for
one release for migration. Add the stopping criterion from paper
eq. (2.3) and the convergence-rate regression test from §3.

### 2.2 Safe alternative: Curtis-Reid 1972 / MC29

**Papers:**
- Curtis and Reid (1972), "On the Automatic Scaling of Matrices for
  Gaussian Elimination", IMA JAM 10(1).
- HSL MC29 package specification (Reid/STFC).

Cite keys: `curtis1972automatic`, and HSL MC29 ingested as `reid2023`
in Crucible.

Properties:

- Minimizes `Σ_{a_ij ≠ 0} (log|a_ij| + r_i + c_j)²` via a few CG
  iterations on the normal equations. Result: logs of nonzeros are
  close to zero, i.e. magnitudes close to 1.
- **Deterministic, no iteration bound concerns** — converges in a
  small, bounded number of CG steps (typically ≤ 2·rank).
- MC29 HSL spec verbatim: "Use of this method gives far better
  results on sparse matrices than scaling to equilibrate row norms."
  That quote is the reason it is "safe" — it works when norm-balancing
  fails. Policy 4 (`compute_scaling_auto`) exists precisely because
  norm-balancing fails on MSS1_0009; a proper Curtis-Reid scaling is
  the principled alternative to the diagnostic-based workaround.
- MC30 is the symmetric variant (referenced in Ruiz 2001 §1).

**Decision:** add `src/scaling/curtis_reid.rs` implementing the
symmetric Curtis-Reid scaling (MC30-equivalent). Expose as
`ScalingStrategy::CurtisReid`. Evaluate on the MSS1_0009 /
VESUVIO / CRESC panel to see whether Curtis-Reid can replace the
Policy 4 diagnostic.

### 2.3 Rejected (for now)

- **Knight-Ruiz-Uçar 2014** (`knight2014symmetry`): the 1-norm
  variant of Ruiz with a proved linear convergence bound. Strictly
  stronger theorems than Ruiz 2001 but the algorithm is
  substantively the same iterate with different row-norm; once we
  have a clean Ruiz kernel, enabling the 1-norm variant is a
  one-flag extension. Defer to post-safe-candidate.
- **Bertsekas auction** (`bertsekas1988auction`,
  `bertsekas2001auctionencycl`): would replace the Hungarian
  matching inside `mc64.rs` with the ε-scaling auction. Real gain
  is parallelism. Defer — the Hungarian is not on the critical path
  of any bench matrix today.
- **Sinkhorn-Knopp** (`sinkhorn1967doubly`): breaks symmetry per
  iteration; Ruiz 2001 §3 explicitly argues against it for symmetric
  inputs. Reject.

## 3  Implementation order

All steps tests-first; research note required before Step 2.

1. **Research note** `dev/research/scaling-ruiz-curtis-reid.md`
   (MANDATORY before any code): cover (a) exact paper correspondence
   for the current `infnorm.rs` code; (b) where it deviates from
   Ruiz Algorithm 2.1 (the Jacobi-vs-sequential question, which
   eq. (2.3) stopping criterion it uses); (c) the Curtis-Reid normal
   equations and the CG iteration count needed for convergence on
   our test panel. **Gate:** research note reviewed before Step 2.

2. **Rename `infnorm.rs` → `ruiz.rs`** with the paper-matching
   notation, paper-cited convergence test (rate 1/2), and
   `ScalingStrategy::Ruiz` exposed alongside a deprecated-alias
   `InfNorm`. No behavioral change — byte-identical scaling vectors
   for the existing test corpus.

3. **Add `curtis_reid.rs`** with the normal-equations CG kernel.
   Tests: (a) MC29 published example (find one in
   `reid2023` spec); (b) scaled-matrix log-sum-squared compared to
   the dense oracle; (c) MSS1_0009 residual comparison vs Ruiz +
   MC64.

4. **Wire `ScalingStrategy::CurtisReid`** into `compute_scaling` and
   extend the bench panel to report residual / factor time / nelim
   across all four strategies (Ruiz, MC64Symmetric, CurtisReid, Auto).

5. **Policy-5 study** (separate session): decide whether Curtis-Reid
   lets us retire Policy 4 ratio-guard fallback. If yes, write
   `dev/plans/policy-5-curtis-reid-default.md` before flipping.

## 4  Exit criteria

- `ScalingStrategy::Ruiz` / `ScalingStrategy::CurtisReid` both
  compile and pass unit tests.
- No regressions on the current bench panel — geomean within 2%
  of 0.21, p90 ≤ 1.90, max ≤ 20 (same gate as Phase 2.4.3).
- Inertia hard rule holds across every matrix in the parity corpus.
- For MSS1_0009: Curtis-Reid must match or beat the Policy 4
  InfNorm fallback (residual ≤ 2×10⁻⁸) while keeping inertia.

## 5  Non-goals

- No interaction with rook-rescue / Phase 2.4.3. These are
  independent work streams.
- No changes to the Auto routing rule (§`pick_scaling_strategy`)
  beyond adding the new strategy names; routing stays
  shape-heuristic until Policy 5 is scoped.
- No public `External` API change — users who pass external
  scaling vectors keep their contract.

## 6  Open questions (to resolve in the research note)

1. `infnorm.rs` updates `d` using Jacobi sweeps (accumulated
   row_max, then divide-by-sqrt). Ruiz Algorithm 2.1 is written as
   sequential `A^{(k+1)} = DR^{-1} A^{(k)} DC^{-1}`. Confirm these
   are mathematically equivalent for symmetric input (we believe
   yes — `DR = DC` makes the order irrelevant — but the paper
   statement is sequential).

2. The `infnorm.rs` `tol = 1e-8` and `max_iter = 10` picks — do
   they match the paper's eq. (2.3) ε-criterion with ε = 1e-8 in
   the worst case of 10 iterations? With rate 1/2 per iteration,
   10 iterations bounds the distance at 2⁻¹⁰ ≈ 10⁻³, which is
   **weaker** than the tol claim. Either `max_iter` should be
   raised to ~30 or the tol relaxed to 10⁻³. This is a correctness
   question — not a tuning one — and must be resolved in the note.

3. Curtis-Reid: MC30 vs MC29 — the symmetric variant vs the
   unsymmetric one. FERAL only needs the symmetric path (congruence
   scaling `D·A·D`), but the HSL implementations differ in the
   normal-equations structure. Confirm MC30's equations are what
   we implement.
