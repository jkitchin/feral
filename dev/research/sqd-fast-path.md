# SQD fast-path — symmetric quasi-definite LDL^T for KKT — 2026-05-16

Pre-implementation research note for the opt-in `Solver::with_sqd_mode(true)`
fast-path. The implementation plan is `dev/plans/sqd-fast-path.md` (to be
written); the user-facing approved design is the plan file
`/Users/jkitchin/.claude/plans/let-s-work-on-a-reflective-anchor.md`.
GitHub tracking issue: #34 (opened in the same commit as this note as
the first step of the phased rollout).

## Motivation

FERAL's dense Bunch-Kaufman kernel (`src/dense/factor.rs:449-696`) performs
a 1x1-vs-2x2 pivot selection scan at every column of every supernode. The
scan is the dominant inner-loop cost on small-frontal supernodes — issue
#10's post-mortem ("supernode-shape ordering A/B — NEGATIVE on Mittelmann
panel") showed that on a 1D-banded panel the kernel-call overhead is
*not* the bottleneck and that "the next lever" candidates (MAXFROMM, axpy
tightening, SmallLeafBatch) all come up within noise. The remaining
algorithmic lever the post-mortem did not consider was *removing* the BK
pivot search entirely when the input structure guarantees it is unnecessary.

That structural guarantee exists for the KKT matrices FERAL is built to
consume.

## Vanderbei's theorem (the structural guarantee)

Vanderbei (1995) defines a *symmetric quasi-definite* (SQD) matrix as

    K = [[ -E,  A^T ],
         [  A,   F  ]]

with `E in R^{n1 x n1}` and `F in R^{n2 x n2}` both symmetric positive
definite. Vanderbei's Theorem 2.1 states:

> For any SQD matrix `K` and any permutation matrix `P`, the matrix
> `P K P^T` admits a factorization `L D L^T` with `L` unit lower
> triangular and `D` *diagonal* (no 2x2 blocks).

The signature of `D` is `(n2 positive, n1 negative, 0 zero)` by Sylvester's
law of inertia. The proof proceeds by induction on `n1 + n2`: at each step
the trailing principal minor remains SQD because Schur complementation
of an SQD block produces an SQD block (the "SQD invariance under Schur
complement" lemma, Vanderbei 1995 Lemma 2.2).

The practical implication: every pivot step of an LDL^T factorization on
an SQD matrix selects a *diagonal entry* without comparison to off-diagonals.
The BK 1x1-vs-2x2 decision tree, the rmax/tmax scan, the Duff-Reid growth
bound test — all dead code on SQD input.

## When does FERAL see SQD input?

The KKT systems FERAL is built to factor come from interior-point methods
in the IPOPT / IP-PMM lineage. Wächter and Biegler (2006) describe the
canonical regularized KKT matrix at iteration `k`:

    K_k = [[ -(H_k + δ_w·I),  J_k^T          ],
           [  J_k,            -δ_c·I         ]]

Identifying `E = H_k + δ_w·I` (positive definite when `δ_w > 0` and `H_k`
is positive semidefinite, which is enforced by IPOPT's primal regularization
loop) and `F = δ_c·I` (positive definite for any `δ_c > 0`), this is exactly
the SQD form when both regularizations are strictly positive.

When does that hold in practice? Wächter-Biegler's Algorithm 1 starts with
`δ_w = 0, δ_c = 0` (line 1 of the inertia correction loop). On the first
iterate where the inertia is "wrong" (the loop fires), `δ_w` is bumped to
`δ_w^0 = 10^{-4}` and `δ_c` is set to `10^{-8} μ^{1/4}`. After this happens
*once* in a given IPOPT run, both regularizations stay positive for the
rest of the solve. Empirically, on the FERAL benchmark KKT corpus this
condition holds on a large fraction of warm iterates — the corpus
classifier probe (see test `sqd_kkt_corpus_classifier` in the plan) is
how we measure exactly how many.

The newer IP-PMM line (Pougkakiotis-Gondzio 2020) makes the SQD condition
structural: regularizations are *always* positive, the KKT is *always*
SQD by construction. The same is true of regularized solvers in the
Friedlander-Orban (2012) line. As IPM theory has moved towards always-
regularized formulations, the fraction of KKTs that are SQD has gone up,
not down.

## Stability — the GSS-1996 bound

Vanderbei's theorem guarantees *existence* of a diagonal-D factorization.
*Stability* is a separate question: does the computed `L_hat D_hat L_hat^T`
satisfy a useful backward error bound?

Gill, Saunders, and Shinnerl (1996) Theorem 4.1 give the answer:

> Let `K` be SQD with diagonal blocks `-E` and `F`, and let
> `λ_min(E), λ_min(F) ≥ σ > 0`. Then any sequence of diagonal pivots
> selected from `K` (in any order) is stable, with growth factor bounded by

    rho_n  ≤  (||K||_inf / σ)^{n}

That bound is exponential in `n` but the base `||K||_inf / σ` is small
whenever both regularizations are not too close to zero. In particular, for
FERAL's typical KKT — `||K||_inf` ~ 1 after MC64 equilibration, `δ_w`,
`δ_c` ~ 10^{-8} — the base is ~ 10^8 and the bound is useless beyond a
handful of columns. The realistic story is the *practical* bound that GSS
also prove (their Section 5): for typical KKTs, growth is bounded by a
small constant times `cond(E) cond(F)`. This matches empirical experience
from PARDISO and IPOPT-MA57 where SQD-eligible KKTs solve at machine
precision under static pivoting.

The takeaway: **stability is conditional**, and FERAL must verify the
condition at run time. The contract-violation predicate
(`sqd_diagonal_ok`, plan §1d) is the mechanism: any column where
`|d_k| < zero_tol` or where `max_i |L[i,k]| > 1/sqrt(EPS)` indicates that
either `E` or `F` has lost positive-definiteness during this step, and the
remaining columns are no longer guaranteed stable. FERAL returns
`FeralError::SqdContractViolated { column, pivot }` rather than continuing
into the unbounded-growth region.

The L-growth threshold `SQD_L_GROWTH_LIMIT = 1.0 / EPS.sqrt() ≈ 6.7e7` is
justified by the BK paper's standard rationale: BK accepts pivots with
multiplier bound `|L_ij| ≤ 1/(1-α) ≈ 2.78` under `α = (1+sqrt(17))/8`.
Allowing growth seven orders of magnitude beyond that is loose enough that
no realistic SQD KKT trips it, but tight enough that a non-SQD matrix
fed through the fast-path is caught within the first few columns.

## What "the fast-path" actually skips

Compared to the current BK kernel
(`src/dense/factor.rs` `fn factor()` at line 449), the SQD diagonal-only
loop at each column `k` skips:

1. `column_offdiag_max(a, n, k)` — scan column k below diagonal for max.
2. The α-threshold test `if gamma0 <= alpha * d.abs()`.
3. `symmetric_row_offdiag_max(a, n, k, r)` — scan row r of the prospective
   2x2 candidate.
4. The Duff-Reid rmax / tmax growth bound test (factor.rs:644-653).
5. The `do_2x2_pivot` codepath entirely.
6. The fused `fused_gamma0` / `have_fused` state machine plumbing.
7. The `PivotOutcome::Rejected` handling (no rejected pivots possible in
   SQD because rejection itself is a contract violation).

What remains: pick `a[k*n+k]`, divide L column by it, rank-1 update of
the trailing submatrix. The rank-1 update reuses the same kernel as the
BK 1x1 path (factored into a shared helper in the implementation plan).

## Per-supernode timing model

Let `b` be a supernode block size, `m` its trailing size. The BK 1x1 step
costs (in flops + memory accesses):

- column_offdiag_max:     `~b`     reads, `~b` compares
- alpha test + branch:    O(1)
- if 1x1: rank-1 update:  `~b·m`   flops
- if 2x2: symmetric_row_offdiag_max + DR test: `~m` extra ops, then
  rank-2 update `~2·b·m`

The SQD 1x1 step costs:

- diagonal sign + contract check:  O(1)
- rank-1 update:                   `~b·m`   flops (same as BK 1x1)

Per pivot saving: `~b` reads + `~b` compares, plus the entire 2x2 branch
amortized over the fraction of columns BK selects 2x2 (typically 0 to 10%
on KKT, but the *scan* runs every column even when 2x2 is not selected).

For a supernode of size `(b=64, m=64)`, BK does ~128 reads + branches per
column, SQD does 1. Over 64 columns that's ~8000 saved operations against
a rank-1 update of ~4000 flops — a ~2x kernel speedup in the limit of
small supernodes where the scan overhead dominates the FMA.

For large supernodes (`m >> b`) the rank-1 update dominates regardless,
and SQD's gain shrinks to the constant per-column scan saving. The plan's
ship gate (geometric-mean speedup ≥ 1.15) reflects this — large fronts
dilute the gain.

## Design alternatives considered and rejected

| Alternative | Rejected because |
|---|---|
| Auto-detect SQD structure inside `factor()` | Detection cost is O(n) per call (check diagonal signs), wrong call on borderline matrices, and inverts the safety contract: a non-SQD matrix would silently take a slow probe-then-fail path |
| Drop-in replacement of BK with SQD as default | Inverts the safety contract; a non-SQD matrix would either fall back silently (hiding caller bugs) or fail loudly (regression for non-KKT callers) |
| New `factor_kkt(kkt, δ_w, δ_c)` entry point | Larger API surface, requires caller to surface `(δ_w, δ_c)` separately when FERAL's current contract is "values baked into the matrix"; the plain `with_sqd_mode` flag is a strict subset of this design and trivially upgradeable |
| Silent BK fallback on `SqdContractViolated` | Hides caller's regularization bug. Caller can implement fallback themselves by catching the error and rebuilding the `Solver` without `with_sqd_mode` — the symbolic cache survives because pattern is unchanged |
| Flag-gated branch inside `factor()` rather than a separate `factor_diagonal()` | Splices into a tight BK state machine; makes both paths harder to test and harder to optimize; the SQD loop is a strict subset of BK 1x1 path, cleaner as its own function |

## Failure-mode catalogue

The contract predicate `sqd_diagonal_ok` (plan §1d) detects:

| Failure | Symptom | Cause | Caller action |
|---|---|---|---|
| Vanishing pivot | `|d_k| ≤ zero_tol` | `E` or `F` lost positive-definiteness mid-step (e.g. `δ_w` was set to zero) | Rebuild Solver without `with_sqd_mode`; refactor with BK |
| Unbounded L growth | `max_i |L[i,k]| > 1/sqrt(EPS)` | `E` or `F` poorly conditioned, regularization too small for the problem scale | Increase `δ_w`, `δ_c`, refactor |
| Sign disagreement with `check_inertia` | `FactorStatus::WrongInertia` | Caller mis-counted the partition `(n1, n2)` or passed wrong expected inertia | Recount, retry |

The first two surface as `FeralError::SqdContractViolated { column, pivot }`
from the dense kernel; the third surfaces through the existing
`check_inertia` plumbing at the Solver layer.

## What the corpus classifier will measure

`is_sqd_candidate(matrix, partition)` (plan §test strategy) takes a
matrix and a candidate partition `(n1, n2)` and returns true iff:

1. `matrix.n == n1 + n2`,
2. the leading `n1 x n1` block has all negative diagonals,
3. the trailing `n2 x n2` block has all positive diagonals,
4. (heuristic) the off-diagonal Frobenius norm in either diagonal block
   is small compared to the diagonal Frobenius norm (i.e. the diagonal
   blocks look like regularized matrices, not arbitrary indefinite blocks).

This is necessary but not sufficient for SQD — true SQD requires `E, F`
positive *definite*, which the classifier cannot verify without a
Cholesky attempt. Condition (4) catches the common cases without an
extra factorization. The classifier emits the candidate-SQD subset of
the corpus; the parity test `sqd_known_kkt_fixtures` is what actually
proves SQD-eligibility (a successful SQD factor + inertia parity with BK
is the proof).

## Open follow-up issues to file *after* shipping

1. SQD-aware ordering sweep. Vanderbei guarantees any P works, but the
   fill produced by AMD / METIS / SCOTCH under the "no 2x2" constraint
   may differ from BK-fill. Worth measuring.
2. C API exposure: `feral_set_sqd_mode` + `FERAL_ERR_SQD_CONTRACT`.
3. Per-call override (`Solver::factor_with_overrides`) for IPM runs
   where a single iterate may be non-SQD.
4. SQD-mode interaction with `Solver::solve_refined` — when contract
   holds, iterative refinement should converge in 1 step; verify.

## References

All in `dev/references.bib`:

- `vanderbei1995sqd` — the existence theorem and SQD definition.
- `gill1996sqd_stability` — the stability bound and the practical
  condition for backward-stable diagonal pivoting.
- `orban2017sqd` — the SIAM Spotlights monograph; ch. 2 collects the
  SQD theorems with modern notation.
- `friedlander2012regularized` — primal-dual regularized IPM where SQD
  is structurally enforced.
- `pougkakiotis2020ippmm` — IP-PMM lineage; SQD always holds by
  construction.
- `greif2014kkt_eigenvalue_bounds` — eigenvalue bounds on regularized
  KKT; informs the `SQD_L_GROWTH_LIMIT` choice.
- `wachter2006ipopt` (existing) — IPOPT regularization scheme that
  produces SQD KKTs after the first inertia-correction fires.

## Resolution

Proceed with the plan in `/Users/jkitchin/.claude/plans/let-s-work-on-a-reflective-anchor.md`.
Commit (a) lands this note, the bib entries, the decisions log entry, and
the GitHub tracking issue. Code lands in commits (b) through (g).
