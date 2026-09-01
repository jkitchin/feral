# Evaluation: adaptive recycled Krylov solves for FERAL and POUNCE

**Date:** 2026-09-01
**Status:** evaluation complete — recommendation is *do not build as proposed*
**Source document:** `~/Desktop/feral-krylov.org` (proposal, 2026-09-01, 433 lines)
**Related:** `pounce/dev-notes/research/interior-cg-matrix-free.md` (pounce's own
prior design for the same capability, status "design / proposed. Not yet
implemented", roadmap area C5)

This note records why the proposal was not adopted, and — more importantly —
what evidence would have to change for that answer to flip. It is written for a
future session that encounters the same idea, so the numerical results below are
given with the code that produced them.

---

## 1. What was proposed

An optional MINRES path inside FERAL with Krylov-subspace recycling across
POUNCE's interior-point iterations. Four phases: (0) offline Lanczos study of
spectral persistence; (1) standalone MINRES plus an SPD "absolute-value factor"
preconditioner; (2) breakdown-safe deflated/augmented recycling with bounded
persistent state; (3–4) POUNCE `observe` / `recovery` / `adaptive` modes.

The proposal's §1 executive decision: *"The feature belongs primarily in Feral."*

The document is well constructed — staged exit gates, an explicit falsification
gate at Phase 0, a risk table, and a test plan with external oracles. The
objections below are not about its rigour.

---

## 2. Bottom line

Three findings, in descending order of weight.

1. **The seam is wrong, and POUNCE's own prior note disagrees with it.**
2. **Phase 1's preconditioner destroys Phase 2's premise, by construction.**
3. **The problem it targets is measured not to occur on this corpus.**

---

## 3. Finding 1 — the seam disagreement

POUNCE already has a design note for this capability:
`pounce/dev-notes/research/interior-cg-matrix-free.md`. It places the Krylov
solver at the **`AugSystemSolver`** seam
(`pounce/crates/pounce-algorithm/src/kkt/aug_system_solver.rs:72`). The proposal
places it at **`SparseSymLinearSolverInterface`**
(`pounce/crates/pounce-linsol/src/sparse_sym_iface.rs:75`), one level lower —
inside FERAL.

That choice decides the architecture:

| Seam | μ, iteration, phase, δ perturbations, residuals |
|---|---|
| `AugSystemSolver` (pounce's note) | all in scope |
| `SparseSymLinearSolverInterface` (the proposal) | **none** — `multi_solve` takes seven scalars and no context object |

μ is live at `pd_full_space_solver.rs:1505` (`data.borrow().curr_mu`) and the
iteration count at `ipopt_data.rs:79`, three frames above the seam the proposal
chose. This is exactly why the proposal's §6 has to invent a
`LinearSolveContext` and push that data back down. The higher seam needs no such
plumbing.

Corollary: the executive decision is likely backwards. The parts the proposal
assigns to FERAL — sequence identity, iteration metadata, phase-driven resets —
are POUNCE-side concepts being pushed into a layer that was deliberately built
to know nothing about them (`FERAL-PROJECT-SPEC.md` §2.12.1: *"FERAL sees a
generic symmetric indefinite matrix"*).

---

## 4. Finding 2 — the |D| preconditioner and the recycle space cancel

The proposal's §4.1 offers the absolute-value factor preconditioner

    M = P Dₛ⁻¹ L |D| Lᵀ Dₛ⁻¹ Pᵀ

as the SPD preconditioner MINRES requires, replacing each 1×1/2×2 pivot block by
its matrix absolute value.

**Measured result: `M⁻¹K` is sign-similar to the identity.**

```
n = 55 saddle KKT, barrier diagonal spanning 1e-6 .. 1e6
M symmetric positive definite? min eig = 0.0155
eigenvalues of M⁻¹K: min = -1.000000  max = 1.000000
distinct |eigenvalues| (rounded): [1.]
count near +1: 35   near -1: 20   (N = 55)
||(M⁻¹K)² − I||_F = 1.54e-10
```

Since `(M⁻¹K)² = I`, the minimal polynomial has degree 2 and **preconditioned
MINRES converges in at most two iterations in exact arithmetic**. Confirmed
against `scipy.sparse.linalg.minres`: 2 iterations at every barrier value tested.

This is not a lucky spectrum; it is what `|D|` *is*. Writing `K = L D Lᵀ` and
`M = L |D| Lᵀ`, we get `M⁻¹K = L⁻ᵀ|D|⁻¹D Lᵀ`, similar to `sign(D)`.

### Why this matters

- **With a current-matrix factorization no Krylov subspace is built**, so Phase 2
  has nothing to extract Ritz vectors from. Phase 1 defeats Phase 2.
- The proposal's risk table lists *"Direct factors are so accurate that Krylov
  adds work"* as a risk to be mitigated by measurement. It is not a risk to be
  measured — it follows from the algebra.
- The only regime where the Krylov layer does real work is an **inexact**
  preconditioner, i.e. a factorization frozen at an earlier iterate. That is also
  the only regime with a real prize, since it would save *factorization* time
  rather than solve time.

### Reproduction

```python
import numpy as np
from scipy.linalg import ldl, eigh
np.random.seed(0)
n, m = 40, 15
A = np.random.randn(n, n); W = (A + A.T)/2
J = np.random.randn(m, n)
Sig = np.diag(10.0**np.random.uniform(-6, 6, n))
K = np.block([[W + Sig, J.T], [J, -1e-8*np.eye(m)]])
L, D, perm = ldl(K, lower=True)

def matabs_blockdiag(D):
    Da = np.zeros_like(D); i = 0
    while i < D.shape[0]:
        if i+1 < D.shape[0] and abs(D[i+1, i]) > 0:
            w, V = eigh(D[i:i+2, i:i+2])
            Da[i:i+2, i:i+2] = V @ np.diag(np.abs(w)) @ V.T; i += 2
        else:
            Da[i, i] = abs(D[i, i]); i += 1
    return Da

M = L @ matabs_blockdiag(D) @ L.T
P = np.linalg.solve(M, K)
print(np.unique(np.round(np.abs(np.linalg.eigvals(P).real), 6)))   # -> [1.]
print(np.linalg.norm(P @ P - np.eye(K.shape[0])))                  # -> ~1e-10
```

---

## 5. Finding 3 — a stale factorization degrades immediately (synthetic)

Since the only useful regime is an inexact preconditioner, the follow-up question
is how long a frozen factorization stays useful. Model: 80×80 KKT, factorization
frozen at μ=1e-1, applied at later μ with 12 variables driven toward their bounds.

| μ | cond(K) | MINRES its | spectrum of `M₀⁻¹K` |
|---|---:|---:|---|
| 1e-1 | 1.99e+05 | 2 | [1.00e+00, 1.00e+00] |
| 1e-2 | 2.66e+04 | 200+ | [1.97e-02, 1.14e+03] |
| 1e-3 | 2.15e+05 | 200+ | [1.47e-02, 9.50e+03] |
| 1e-4 | 1.38e+06 | 200+ | [1.41e-02, 6.24e+04] |
| 1e-5 | 8.67e+06 | 200+ | [1.41e-02, 3.96e+05] |
| 1e-6 | 5.46e+07 | 200+ | [1.41e-02, 2.50e+06] |

Control (refactorize at every μ): 2 iterations throughout.

**Caveat, important.** This is a synthetic model with an aggressive 10× μ drop
per step and a harsh active-set model (`x·μ^0.9` on 12 variables). Real
consecutive IPM matrices are closer together than this. The result **bounds the
sensitivity; it does not settle the question**. If factorization reuse is ever
pursued, it must be measured on captured POUNCE sequences, not on this model.

---

## 6. Finding 4 — "recovery mode" loses to plain refinement (synthetic)

The proposal's most defensible mode is `recovery`: run Krylov only after
stationary iterative refinement fails its residual contract. Tested directly.
Both methods cost **one triangular solve per step**, so step count is a fair
comparison.

Setup: near-singular KKT (near-dependent constraint row), κ(K)=1.85e16, one pivot
perturbed to a floor to mimic `ZeroPivotAction::ForceAccept`. Metric is the
Arioli–Demmel–Duff componentwise backward error ω, the same quantity #190/#191
adopted.

| triangular solves | refinement ω | prec-MINRES ω |
|---:|---:|---:|
| 0 | 2.581e-16 | 1.315e-01 |
| 1 | 1.936e-16 | 1.315e-01 |
| 2 | 1.601e-16 | 5.667e-16 |
| 3 | 1.546e-16 | 2.085e-16 |
| 4 | 1.657e-16 | 1.865e-16 |
| 5+ | ~1.6e-16 | 3.895e-15 |

Refinement floor 9.46e-17; MINRES floor 1.87e-16; √ε target 1.49e-08.

Stationary refinement matched or beat preconditioned MINRES at every step count,
and MINRES drifted *worse* after step 4. One synthetic case, not a corpus — but
it is the proposal's own use case and it did not reproduce.

---

## 7. Where the time actually goes

POUNCE's per-phase breakdown (`pounce/dev-notes/ma57-batched-backsolve.md`,
118,276-row KKT, seconds per iteration):

| | Ipopt/MA57 | POUNCE (current) |
|---|---:|---:|
| numeric factorization | 0.1344 | **0.1702** |
| back-solve | 0.1055 | 0.0856 |
| linear algebra total | 0.2465 | 0.2662 |
| solver internal | 0.2833 | 0.3533 |

Linear algebra is ~75% of solver-internal time, so the target area is real. But
**factorization is 2× the back-solve**, and
`pounce/dev-notes/performance-engineering.md:149` states the priority outright:
*"The factorization is the bottleneck — address it first."*

A Krylov path that still factors for inertia cannot touch factorization cost. It
competes only with the back-solve, at one triangular solve per iteration — the
same unit cost as the refinement step it would replace. **The ceiling is the
smaller half of the smaller half.**

Corpus-level numbers point the same way: `solve/MUMPS` geomean 0.08 vs
`factor/MUMPS` geomean 0.44 (one solve per factor). Note the countervailing
datum: against SSIDS, feral's *solve* is its weakest ratio (geomean 1.05, p90
3.50, p99 12.00, max 53×). There is headroom in the back-solve — but the fix for
that is faster triangular solves, not a Krylov method layered on top of them.

---

## 8. The premise: refinement does not stall here

- **Issue #190 measurement** (7 large matrices, well-scaled RHS): refinement
  takes **0–2 steps, never at the 10-step cap**. There is no wasted-iteration
  budget for a Krylov method to reclaim.
- **Issue #30 measurement** (28-matrix stress manifest): 17/28 need no
  refinement, 7 converge in exactly 1 useful step, 4 stagnate above `ε·√n`. For
  the stagnating four, *MUMPS floors at the same residual* (`stokes64`: feral
  3.37e-14, MUMPS ≈3.4e-14). The floor is intrinsic to the matrix geometry, not
  recoverable by a different iteration.
- Refinement is recorded as asymptoting gracefully, never diverging.

### What POUNCE actually asked for

Every downstream request ran in the opposite direction — less work in the solve,
or a different stopping rule, never "it will not converge":

| Issue | Request |
|---|---|
| #178 (pounce#698) | **Less** refinement — back-solve 147.3s → 58.3s when disabled |
| #190/#191 | A different *stopping criterion* (normwise → componentwise ω) |
| #192 (pounce gh#850) | **Bound** `increase_quality`'s lifetime |
| #194 | **Interrupt** a running factorization (48.8s against a 5s budget) |

All four shipped in the post-0.17.0 window; all four were parked unused on
2026-09-01 (see `dev/decisions.md`, 2026-09-01 entry). A speculative subsystem
with no originating consumer request has a weaker prior than any of them did.

---

## 9. The inertia gate — and it is backwards from the proposal's assumption

`pounce/crates/pounce-algorithm/src/kkt/pd_full_space_solver.rs:1616`:

```rust
let check_inertia = self.neg_curv_test_tol <= 0.0 || !self.aug_solver.provides_inertia();
```

Returning `provides_inertia() -> false` forces `check_inertia = true` — the
*opposite* of standing the check down. The inertia-free path that exists
(Zavala–Chiang curvature test, `:1711-1748`) is **gated on
`provides_inertia() == true`** and `neg_curv_test_tol` defaults to `0.0`, i.e.
off.

So any design that hopes to *skip factorizations* — the only way to reach the
dominant cost — requires enabling and validating an inertia-free IPM in POUNCE
first. That is a change to the global convergence argument, in POUNCE, before
FERAL writes a line. `provides_inertia()` is hardcoded `true` in all five places
the interface is specified, and `FERAL-PROJECT-SPEC.md` §10.8 calls exact inertia
*"not negotiable"*.

---

## 10. What FERAL would have to build

Repo-wide grep for `minres|gmres|krylov|arnoldi|bicgstab|matrix-free` returns
**one hit and it is a false positive** ("matrix free-function"). The single
"Lanczos" hit is the CUTEst matrix `LANCZOS1_0029`; every "recycling" hit is
`Vec<f64>` buffer pooling. This ground has never been tried and never been
rejected — **silence, not precedent**.

Missing outright: dense QR (Householder/MGS), any symmetric eigensolver above 2×2
(`sym2_eigenvalues`, `src/dense/factor.rs:5043`, is private and 2×2-only), block
orthogonalization, Lanczos tridiagonalization, Givens rotations for a MINRES QR
update, harmonic-Ritz extraction, and public dot/axpy/norm primitives (`norm2` is
private at `src/numeric/solve.rs:2862`). The only abstract operator trait is
`HagerHighamOperator` (`src/numeric/condition.rs:61`), which is *inverse*-apply
and has no forward `apply`.

Structural friction:

- `D⁻¹` is **fused into the forward pass** (issue #126, `src/numeric/solve.rs:320-348`)
  and open-coded in four places; none takes D as a parameter. Substituting `|D|`
  needs new entry points.
- `SparseFactors` / `NodeFactors` / `FrontalFactors` do not implement `Clone`, and
  `Solver` exposes `factors()` but no `factors_mut()` — a caller cannot build a
  `|D|` variant of the stored factor.
- `FeralError` is **not** `#[non_exhaustive]`; a new variant is a breaking change
  that also touches `src/capi.rs` and `python/src/errors.rs`.
- Every `Solver` solve method takes `&self`; a persistent recycle space needs
  `&mut self` or interior mutability.
- Determinism is contractually pinned (`tests/golden_bits.rs` plus nine parity
  suites). A Krylov path is a new source of trajectory variation.
- `FERAL-PROJECT-SPEC.md` §1.2: *"One engine, not multiple solvers"* and
  *"pluggable strategies, not pluggable solvers."* A Krylov path is a second
  solver, not a variation point.

What *does* exist and would be reused: `CscMatrix::symv` (`src/sparse/csc.rs:327`,
public, correct for lower-half storage), the dense Bunch–Kaufman factor/solve for
small `r × r` systems, and `SchurBlock::solve` (`src/numeric/factorize.rs:906`).

---

## 11. What POUNCE would have to add

1. **A metadata channel.** `multi_solve` (`sparse_sym_iface.rs:94`) takes seven
   scalars and no context.
2. **A phase concept.** There is no phase enum; restoration is a whole second
   algorithm instance (`pounce-restoration/src/resto_alg_builder.rs`). The
   proposal's `SolvePhase::Restoration` has no source today.
3. **A structure epoch.** Closest existing: `StdAugSystemSolver::struct_sig`
   (`std_aug_system_solver.rs:56`), a fingerprint compare rather than a counter.
4. **Forwarding through four decorator layers** — `PooledFeralBackend`
   (`pounce-algorithm/src/batch.rs:301`) forwards every trait method by hand; an
   unforwarded method silently reverts to its default. Precedent: gh#825 shipped
   nine options that were "accepted and discarded" with no observable symptom.
5. **Trajectory neutrality as a merge gate** — `scripts/sweep-fixtures.sh`, 91
   fixtures × 2 legs, a pre-merge human obligation, not CI. Only `observe` mode is
   provably sweep-neutral, and even adding a telemetry column shifts every line by
   one field.

---

## 12. Recommendation

**Do not build Phases 2–4.**

**Run a measurement first — but not the one the proposal specifies.** Its Phase 0
measures spectral persistence (principal angles between consecutive candidate
subspaces), which only matters once recycling is assumed to be the mechanism. The
prior question is cheaper and the tool already exists: `pounce/scripts/qbench.py`
already reports the `print_timing_statistics` phase breakdown plus `n_factors` /
`n_pattern_reuse`. Point it at the known-hard families — `laptime` (126,028-dim
KKT), `nug12`, `square_flowsheet_resto`, `pooling_rt2stp`, the gh#698 118k KKT —
and obtain:

1. factor vs back-solve vs refinement split per family;
2. **refactorizations per IPM iteration**;
3. how often the refinement ladder reaches its `q` / `s` / `S` rungs.

Item 2 is the one to watch. The recorded pathologies are `nug12` thrashing δ_x to
its 1e20 ceiling (`std_aug_system_solver.rs:543`), gh#592 burning five rungs to
land at δ_w=1e2 where Ipopt accepted at 1e-4 (`pd_perturbation.rs:531-536`), and
`eigena2` returning 64/58/62 negatives for the same matrix
(`pounce-feral/src/lib.rs:1131-1143`). Those are **inertia-reliability and
regularization-policy** problems that cost whole extra factorizations. A Krylov
path fixes none of them, and they plausibly dominate the back-solve.

**If one thing is built from this document, build Phase 1 only — and call it a
diagnostic, not a solver.** Standalone MINRES plus the `|D|` preconditioner
behind a non-default feature. The ±1 result of §4 makes it a principled
factorization-quality probe: any departure from 2-iteration convergence measures
exactly how inexact the factorization is. That is cheap, requires no POUNCE
changes, no metadata channel, and no inertia story — and it would give an
independent cross-check precisely on the `eigena2`-class cases where we currently
cannot tell whether a reported inertia is signal or noise.

---

## 13. What would change this conclusion

Stated so a future session can test rather than re-argue:

1. **A measured family where back-solve + refinement dominates factorization**
   *and* refinement regularly exhausts its step budget. #190 measured 0–2 steps on
   everything available locally; the gh#698 118,276-dim KKT is a POUNCE *runtime*
   matrix and is **not in the local corpus**, so the premise is untested at its own
   scale rather than refuted at it. Capturing that sequence is the single most
   valuable next artifact.
2. **POUNCE enabling and validating `neg_curv_test_tol > 0`** (the inertia-free
   curvature test). This unlocks skipping factorizations, which is the only route
   to the dominant cost. Without it, the ceiling stays the back-solve.
3. **A measured case where a factorization frozen across ≥2 IPM iterations still
   preconditions well** on real captured KKT sequences — contradicting §5's
   synthetic bound.
4. **A consumer request.** None of #178/#190/#192/#194 originated here, and all
   four went unused anyway. A Krylov subsystem with no originating request starts
   below that bar.

---

## 14. References

- Paige & Saunders, MINRES: <https://doi.org/10.1137/0712047>
- Parks et al., recycling Krylov subspaces (GCRO-DR): <https://doi.org/10.1137/040607277>
- Gaul et al., deflated/augmented Krylov framework: <https://doi.org/10.1137/110820713>
- Choi, Paige & Saunders, MINRES-QLP: <https://web.stanford.edu/group/SOL/reports/SOL-2010-3.pdf>
- Gill, Murray, Ponceleón & Saunders, preconditioners for indefinite systems in
  optimization — the origin of the `L|D|Lᵀ` absolute-value preconditioner and the
  ±1 spectrum result independently re-derived in §4.
- Chiang & Zavala, inertia-free filter line search — the relevant prior art for
  §9, and the prerequisite for any factorization-skipping design.

In-repo:

- `dev/research/ir-convergence-policy.md` — issue #30, the 28-matrix refinement study
- `dev/research/refinement-cap-2026-08-19.md` — issue #178, the nested-loop cost
- `dev/decisions.md` 2026-08-21 — the MA57/MUMPS componentwise gap and its resolution
- `dev/decisions.md` 2026-09-01 — the post-0.17.0 park and its standing implication
- `pounce/dev-notes/research/interior-cg-matrix-free.md` — the pre-existing design at the higher seam
- `pounce/dev-notes/ma57-batched-backsolve.md` — the only per-phase timing breakdown
