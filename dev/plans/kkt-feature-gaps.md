# KKT Feature-Gap Roadmap

**Date:** 2026-04-27
**Status:** Authoring (Phase 0)
**Scope:** Three KKT-relevant gaps identified in the production-
readiness assessment of session 2026-04-27-09:
- F1 — multi-RHS API
- F2 — condition-number estimate
- F3 — Schur complement extraction

Each is independently scoped; they ship in sequence F1 → F2 → F3
because F2's 1-norm condition estimator benefits from F1's batched
triangular solve, and F3 reuses the partial-factorization plumbing
that none of the prior phases needed.

## Context

The 2026-04-27-09 production-readiness audit listed 14 features
present in MUMPS or SSIDS but absent from feral, then triaged them
by KKT-relevance. The three above were the only gaps that
constrain feral as a KKT solver (vs. as a generic large-scale
solver — those are scale/perf gaps for a separate roadmap).

Sources:
- MUMPS user manual §3 (multi-RHS), §3.10 (Schur complement)
- SSIDS user guide §6 (multi-RHS via `ssids_solve` columnar
  interface)
- Ipopt's `IpMumpsSolverInterface.cpp` for how a production IPM
  consumes these features
- `dev/research/inertia-triage-2026-04-27.md` — confirms no
  load-bearing inertia bug blocks this work

## F1 — Multi-RHS API

### Motivation

Mehrotra predictor-corrector IPM solves two right-hand sides
against the same factor in every Newton iteration: the predictor
(affine) step and the corrector step. Without a batched API the
caller pays per-call workspace overhead and loses opportunity for
column-batched triangular solve kernels. Sensitivity analysis,
parametric LP/QP, and warm-start methods also want this.

Today: `solve_sparse(factors, rhs: &[f64]) -> Vec<f64>` is single-
column only. The internal `SolveWorkspace` is already a step
toward amortization (used by `solve_sparse_refined` for 11 solves
in succession), so the shape of the change is incremental.

### API sketch

```rust
/// Solve A·X = B for X, where B and X are column-major n×k.
/// Equivalent to `k` independent calls to `solve_sparse` but
/// reuses workspace and admits column-batched kernel optimization.
pub fn solve_sparse_many(
    factors: &SparseFactors,
    rhs: &[f64],          // column-major, len = n*k
    nrhs: usize,
) -> Result<Vec<f64>, FeralError>;

/// Same with caller-owned workspace (zero alloc on hot path).
pub fn solve_sparse_many_into(
    factors: &SparseFactors,
    rhs: &[f64],
    nrhs: usize,
    x_out: &mut [f64],
    ws: &mut SolveManyWorkspace,
) -> Result<(), FeralError>;

impl Solver {
    pub fn solve_many(&self, rhs: &[f64], nrhs: usize)
        -> Result<Vec<f64>, FeralError>;
    pub fn solve_many_refined(&self, matrix: &CscMatrix,
        rhs: &[f64], nrhs: usize)
        -> Result<Vec<f64>, FeralError>;
}
```

### Internals

Extend `SolveWorkspace` to hold a `k`-wide gather buffer per
supernode (`w: Vec<f64>` becomes `w: Vec<f64>` of length
`max_nrow * k`). The forward/backward supernodal traversal is
unchanged in shape; the per-supernode triangular solve and
Schur-update kernels become column-batched:

- Per-supernode forward: instead of `trsv(L_diag, y_supernode)`
  call `trsm(L_diag, Y_supernode_k_columns)`.
- Per-supernode update: instead of `gemv(L_off, y_supernode)` call
  `gemm(L_off, Y_supernode_k_columns)`.

The kernels exist in `src/dense/` for the dense path; reuse via
the same shapes that `factor_frontal` already calls.

### Phases

- **F1.0** — research note `dev/research/multi-rhs.md` covering
  MUMPS's `ICNTL(20)` (sparse-RHS/dense-RHS dispatch) and SSIDS's
  `ssids_solve(nrhs, x, ldx, ...)` to verify the column-major
  layout choice and the `ldx > n` (leading-dimension) flexibility
  question.
- **F1.1** — `solve_sparse_many_into` scalar implementation
  (loop k times, share workspace). Tests: equivalence with `k`
  independent `solve_sparse` calls on a 5×5 panel.
- **F1.2** — column-batched forward/backward kernels. Bench the
  panel sizes that matter for IPM (k=1, 2, 4, 8). Target: factor=
  trsm/gemm beats the loop by ≥ 30% at k≥4 on small fronts.
- **F1.3** — `Solver::solve_many` + `solve_many_refined`. Tests
  against the existing parity panel (run k=1 through it as a
  sanity check that `solve_many` matches `solve` exactly).
- **F1.4** — Optional: rayon over RHS columns when k is large.
  Defer until a real workload asks for it.

### Acceptance gate

- All current solve tests pass (parity panel, kkt_matrices,
  sparse_postorder).
- `solve_many(rhs, k)` produces column-equivalent output to `k`
  independent `solve(rhs_i)` calls within machine precision.
- Bench harness records solve_us at k=1, 2, 4, 8 for the small-
  frontal panel; k=4 must be ≤ 3× the k=1 cost (i.e. real
  amortization, not a wrapper).

## F2 — Condition-number estimate

### Motivation

Ipopt's regularization-escalation loop reads MUMPS's INFOG-based
conditioning signals to decide when to bump `δ_w` (Hessian
perturbation) or `δ_c` (constraint relaxation). Without a
condition-number estimate, feral's downstream IPM has only a
heuristic δ-ladder with no feedback signal. MUMPS exposes
estimate-of-`||A⁻¹||₁` via `RINFOG(11)` (when `ICNTL(11)=2`);
SSIDS exposes the same via `solve_inquiry`. (See
`dev/research/condition-estimate.md` for the field-name
correction — earlier draft of this plan said `INFOG(40)`.)

### API sketch

```rust
impl Solver {
    /// Hager-Higham 1-norm condition estimate of A.
    /// Returns ||A||₁ · ||A⁻¹||₁ approximation. Cost: 4-5 solves
    /// against the stored factor.
    pub fn estimate_condition_1norm(&self, matrix: &CscMatrix)
        -> Result<f64, FeralError>;
}
```

### Internals

Hager 1984 / Higham 1988 1-norm power iteration. Algorithm:

1. Initialize `x = (1/n, 1/n, ..., 1/n)`.
2. Compute `y = A⁻¹ x` (one solve).
3. Compute `ξ = sign(y)`.
4. Compute `z = A⁻ᵀ ξ` (one solve).
5. If `||z||_∞ ≤ z·x`, return `||y||₁`.
6. Else replace `x` with `e_j` where `j = argmax|z_i|` and goto 2.

Three to five iterations terminate in practice for KKT systems.
For symmetric A, `A⁻ᵀ = A⁻¹`, so the inner loop uses a single
solve per iteration (not two). LAPACK's `DGECON` is the textbook
reference; SSIDS implements it in `core_solve.f90::condition_est`.

### Dependencies

- F1 lands first. The 1-norm estimator wants 1 RHS at a time, but
  having `solve_into` (the workspace-amortized form) makes the
  inner loop allocation-free. F1.1 is sufficient; F1.2 is not
  required.

### Phases

- **F2.0** — research note `dev/research/condition-estimate.md`
  covering Hager's algorithm, the symmetric specialization, and
  cross-validation against MUMPS's `INFOG(40)`.
- **F2.1** — Standalone `condition_estimate(factors, A,
  workspace)` function. Test: matrices with known condition
  numbers (Hilbert, KKT panels with known singular values) within
  10× of true value (Hager is a lower bound; 10× is the textbook
  conservative gate).
- **F2.2** — Expose via `Solver::estimate_condition_1norm`.
  Cross-validate against MUMPS sidecar conditioning data on the
  full corpus where MUMPS provides `RINFOG(11)`. F2.2 must extend
  `external_benchmarks/mumps_oracle/mumps_bench.F` with
  `ICNTL(11)=2` and write `RINFOG(11)` to verdict.json.
- **F2.3** — Wire into iterative-refinement termination as a
  diagnostic. Don't change behavior yet — just emit the estimate
  alongside the residual at each refinement step. (A later
  decision is whether to use it for adaptive δ; that's an Ipopt-
  facing change, out of scope here.)

### Acceptance gate

- `estimate_condition_1norm` returns within 10× of the true
  `||A||₁·||A⁻¹||₁` on the Hilbert/KKT calibration set.
- Cross-validation report comparing feral's estimate to MUMPS's
  `INFOG(40)` on N ≥ 1000 corpus matrices, geomean ratio within
  [0.5, 5.0].
- No regression on the Phase 2.8.1 bench partition (the estimator
  is invoked only when explicitly requested).

## F3 — Schur complement extraction

### Motivation

Block-elimination IPM, multi-stage stochastic NLP, and PDE-
constrained optimization need to factor a sub-block, form the
Schur complement explicitly, and either return it to the caller
(for further factorization or eigenvalue analysis) or solve
against it. MUMPS's `ICNTL(19)`+`LISTVAR_SCHUR` returns S as a
dense matrix; SSIDS does not currently expose it (open issue
upstream).

The classic use case is the augmented-system → reduced-system
transition in IPM: factor the diagonal slack-block, eliminate it,
form the smaller dense Schur in (primal, dual) space, factor that
densely. Without explicit Schur extraction the IPM has to factor
the full augmented system every iteration.

### API sketch

```rust
impl Solver {
    /// Factor A with a designated Schur block. After factor()
    /// returns, the Schur complement S = A22 - A21·A11⁻¹·A12 is
    /// available via `schur_complement()`. The Schur block is
    /// the *last* `nschur` indices of the (post-permutation)
    /// pivot order; callers select Schur variables via the
    /// `schur_indices` parameter on `factor_with_schur`.
    pub fn factor_with_schur(
        &mut self,
        matrix: &CscMatrix,
        schur_indices: &[usize],
    ) -> FactorStatus;

    /// Extract the Schur complement as an n_schur × n_schur dense
    /// symmetric matrix (column-major). Available only after
    /// `factor_with_schur`.
    pub fn schur_complement(&self) -> Option<&[f64]>;
}
```

### Internals

The multifrontal factorization already builds a postorder over
the elimination tree. The Schur API requires:

1. Constraining the ordering so the Schur indices are the **last**
   `nschur` columns of the eliminated permutation. Cheapest
   approach: post-process the AMD/AMF/MetisND ordering to push
   `schur_indices` to the tail.
2. Stopping the factorization at the boundary between the "fully
   eliminated" and "Schur" supernodes. The fully-eliminated
   prefix proceeds normally; the Schur tail's frontal updates
   accumulate but the diagonal is **not** eliminated.
3. Assembling the resulting Schur tail into a dense block S of
   size `nschur × nschur` and returning it.

This is the **partial-factorization** primitive. MUMPS's
`dfac_root_par_m.F` is the Fortran reference (look at how
`KEEP(60)` controls the root-node behavior: `KEEP(60)=1` means
"factor everything", `KEEP(60)=2` means "Schur, do not factor
last-block diagonal").

### Dependencies

- F1 is helpful (multi-RHS solve against the Schur block) but
  not required.
- F2 is independent.

### Phases

- **F3.0** — research note `dev/research/schur-complement.md`
  covering MUMPS's `KEEP(60)`/`ICNTL(19)` plumbing, the dense
  vs. sparse Schur output decision, and the variable-selection
  semantics (caller passes original-index list; feral re-orders
  internally).
- **F3.1** — Ordering hook: `OrderingMethod` learns to accept a
  Schur tail. The output permutation maps Schur indices to the
  last `nschur` positions; the elimination tree is built so the
  Schur supernode sits at the root. Tests: small KKT example
  with hand-computed Schur.
- **F3.2** — Numeric hook: `factor_with_schur` runs the
  multifrontal factorization but stops eliminating when the
  current supernode is the Schur tail. The Schur block
  accumulates updates from descendants and is returned as a
  dense column-major buffer. Tests: against the F3.1 ordering
  hook, verify `S = A22 - A21·A11⁻¹·A12` to machine precision on
  a 10×10 example.
- **F3.3** — Cross-validation against MUMPS's Schur output on
  N ≥ 100 corpus matrices where Schur extraction is requested for
  a non-trivial sub-block. Reuse the existing oracle harness;
  add a `mumps_schur_oracle.py` driver.
- **F3.4** — `solve_against_schur(s_matrix, rhs)` convenience —
  optional, defer if no caller asks.

### Acceptance gate

- Hand-computed Schur on a 10×10 KKT example matches feral's
  output to ≤ 100·ε relative error.
- Cross-validation rollup: feral Schur vs. MUMPS Schur on
  N ≥ 100 corpus matrices, max relative entry-wise error
  ≤ 10⁻¹⁰.
- No regression on Phase 2.8.1 bench partition (factor without
  Schur is the default; the new path is opt-in).

## Sequencing

```
F1 (multi-RHS)         ──┐
                         ├──▶ F2 (cond-est)
                         │
                         └────────────────▶ F3 (Schur)
```

F2 starts after F1.1 lands (allocation-amortized solve), not after
F1.4. F3 has no hard dependency on the others; it ships when F2
is done so the team is not context-switching across all three.

Estimated work breakdown, in sessions:
- F1: 2–3 sessions (research + scalar + batched + tests)
- F2: 2–3 sessions (research + standalone + integration + cross-val)
- F3: 4–6 sessions (research + ordering hook + numeric hook +
  cross-val; this is the biggest scope)

Total: 8–12 sessions. F1 alone unblocks Mehrotra IPM, so it has
the highest individual leverage.

## Out of scope

- Sparse-RHS API (MUMPS `ICNTL(20)=1`). Useful for sensitivity
  analysis, not for vanilla IPM Newton-step solves. Defer.
- A⁻¹ entries / selected-inverse. Different algorithm class
  (Takahashi recursion, etc.). Defer.
- Determinant / log-determinant. Niche. Defer.
- f32 or mixed-precision. Performance feature for very large
  KKT, separate roadmap.
- Complex arithmetic. Real KKT only.
- MPI / OOC / GPU / BLR. Scale-perf gaps, separate roadmap.
- Explicit null-space vectors. Ipopt detects degeneracy via
  inertia + δ-ladder; not asked of the linear solver.

## Files this roadmap will touch

- `src/numeric/solve.rs` — multi-RHS kernels (F1)
- `src/numeric/solver.rs` — `Solver::solve_many`,
  `Solver::estimate_condition_1norm`, `Solver::factor_with_schur`,
  `Solver::schur_complement` (F1, F2, F3)
- `src/numeric/factorize.rs` — `factor_with_schur` plumbing (F3)
- `src/symbolic/mod.rs` — Schur-tail ordering hook (F3.1)
- `src/numeric/condition.rs` (new) — Hager 1-norm estimator (F2)
- `src/numeric/schur.rs` (new) — Schur-block accumulator (F3)
- `tests/multi_rhs_*.rs` (new) — F1 tests
- `tests/condition_estimate_*.rs` (new) — F2 tests
- `tests/schur_*.rs` (new) — F3 tests
- `dev/research/multi-rhs.md` (new) — F1.0 research note
- `dev/research/condition-estimate.md` (new) — F2.0 research note
- `dev/research/schur-complement.md` (new) — F3.0 research note
- `external_benchmarks/mumps_oracle/run_mumps_schur.py` (new) —
  F3.3 cross-validation oracle

## References

- Hager 1984: "Condition Estimates", SIAM J Sci Stat Comput 5(2)
- Higham 1988: "FORTRAN codes for estimating the one-norm of a
  real or complex matrix", ACM TOMS 14(4)
- Mehrotra 1992: "On the implementation of a primal-dual interior
  point method", SIAM J Optim 2(4)
- MUMPS user manual 5.8.2 §3.10 (Schur complement) and §3.11
  (multi-RHS)
- SPRAL SSIDS user guide §6.1 (`ssids_solve`) and §6.6
  (`solve_inquiry`)
- Wächter & Biegler 2006: "On the implementation of an interior-
  point filter line-search algorithm for large-scale nonlinear
  programming", Math Program 106(1)
