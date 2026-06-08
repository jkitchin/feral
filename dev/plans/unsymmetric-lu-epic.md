# Epic: Unsymmetric LU as a Simplex Basis Engine (issue #81)

**Created:** 2026-06-08
**Research note:** `dev/research/unsymmetric-lu.md` (read first)
**Tracking issue:** #81

## Why this plan exists

feral is a symmetric indefinite LDLᵀ solver. Issue #81 adds a *new factorization family*: an
unsymmetric LU built to drive a revised-simplex basis — cheap rank-1 column-replacement
updates and warm ftran/btran, not just one-shot factor/solve. The downstream consumer is the
`BasisEngine` trait in `pounce-simplex` (a separate repo). The eta/product-form approach
`pounce-simplex` uses today is correct for tiny OBBT LPs but degrades at scale; this epic is
the large-scale graduation, worked independently in feral, which already owns the
sparse-ordering / scaling / robustness machinery this reuses.

## Bounding reality

`pounce-simplex` is **not present** in this environment (only `/home/user/feral`). The
end-to-end acceptance criteria of #81 (BasisEngine drop-in, GLOBALLib no-regression, netlib
win) therefore cannot be built or run here. feral delivers the self-contained engine with an
API *shaped* to the BasisEngine seam; the integration is Phase 8, done later in the pounce
repo.

## Phases

### Phases 1–6 — THIS PR (`claude/open-issue-review-UCoRw`)

- **P1 Dense LU + Bartels–Golub update.** `GeneralMatrix`; right-looking LU with threshold
  partial pivoting (`PB Q = LU`, `Q=I` dense); ftran/btran/ftran_partial; rank-1 dense
  column-replacement update with eta replay; `SingularBasis`/`NeedsRefactor`.
- **P2 Sparse LU (Gilbert–Peierls).** General `SparseColMatrix`; DFS-symbolic + numeric
  left-looking LU with partial pivoting; sparse ftran/btran.
- **P3 Column ordering.** `transpose_pattern`/`ata_pattern`; `Q` via `feral_amd::amd_order`
  on the AᵀA pattern (reusable symbolic handle). Stand-in for true COLAMD.
- **P4 Sparse Forrest–Tomlin update.** Row-eta file on row-wise U; spike→Hessenberg
  bump→single row eta; refactor budget.
- **P5 Scaling.** Two-sided ∞-norm equilibration (adapt `infnorm.rs`) + unsymmetric MC64 as a
  driver over the existing `hungarian_match` kernel, with partial-matching→InfNorm fallback.
- **P6 Iterative refinement.** `ftran_refined`/`btran_refined` looping on `r = a − Bx`.
- **Router** (cross-cutting): `should_use_dense_lu(m, nnz)` with `LuParams` override.

Correctness gate (all in this PR): hand-worked exact cases + equation-residuals +
dense↔sparse agreement + scaling/refinement recovery + adversarial, in `tests/lu_dense.rs`
and `tests/lu_sparse.rs`.

### P6.5 Sparse Forrest–Tomlin update — DONE

The sparse update is now a true Forrest–Tomlin / Bartels–Golub–Reid update: the spike is
folded into `U`'s column `r` and the bump `[r,h]` is re-triangularized by **in-place sparse
Gaussian elimination with partial pivoting**, recorded as a replayable `FtEta` (swaps +
axpys) applied between the `L`- and `U`-solves. Partial pivoting resolved the zero-pivot
landmine (the spiked column's diagonal pivot is often zero in a sparse `U`; a sub-diagonal
spike entry is a nonzero pivot, and the swap goes into the eta so the base `L` is never
permuted). `U` is updated in place; the base `L` is fixed; no dense `τ`, no `O(k·n)`
chain. Warm-solve cost is **bump-local, independent of `n`** for localized spikes (probe:
eta work flat ≈5100 and ftran overhead flat ≈10µs across n=1000..8000; guard test asserts
n-independence). Worst case (dense spike, e.g. tridiagonal) degrades to product-form-like
cost — inherent. Commits: 8738279 (storage), 0fc767c (update), 2b5a5f5 (proof).

### Deferred (follow-up sessions)

- **P7 Reference benchmarks.** SuiteSparse unsymmetric corpus; factor time + solve accuracy
  vs UMFPACK/KLU/SuperLU where licensing allows (the way the LDLᵀ side checks vs MUMPS). The
  update-vs-refactor crossover microbench (`m ∈ {10,100,1k,10k}`, sparsity sweep) is seeded
  in P1–P6 and matured here. Also: faithful COLAMD (replace the AMD-on-AᵀA stand-in), and
  optional Markowitz pivoting (citep:suhl1990computing) for fill on large bases.
- **P8 Downstream (pounce repo, OUT OF SCOPE here).** Implement `BasisEngine` behind the
  feral LU engine; `DenseBasis` as the correctness oracle; no GLOBALLib small-regime
  regression; measured netlib large-regime win; peak fill + refactor-count reporting.

## API shape (matches `BasisEngine`, for P8)

    factor(cols, m, params) -> Result<Self, FeralError>
    ftran(&mut self, rhs)    /  btran(&mut self, rhs)        (in-place, reusable scratch)
    ftran_partial(&mut self, rhs)                            (the spike for update)
    ftran_refined / btran_refined                            (opt-in refinement)
    update(&mut self, leaving_row, entering_col_ftran) -> Result<(), FeralError(NeedsRefactor)>
    refactor(&mut self, cols)  /  updates_since_refactor()
    (sparse) reusable symbolic handle across structurally-similar bases

## Non-goals

Inertia (LU is unsymmetric). Symmetric KKT solves stay on the LDLᵀ path. No change to any
existing symmetric code path — the `lu` module is purely additive.
