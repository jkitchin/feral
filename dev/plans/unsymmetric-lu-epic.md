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

### Deferred (follow-up sessions)

- **P6.5 Sparse Forrest–Tomlin update (the warm-solve scalability fix).** The sparse update
  is currently a product-form update of `U` storing a *dense* `τ` per eta, so warm `ftran`
  degrades `O(k·n)` over `k` updates (measured: `crates/feral-diagnostics/src/bin/lu_update_probe.rs`,
  75µs→3033µs at n=5000, k=0→400). **Stage 1 is done** (`U` stored row-wise CSR, commit
  23af110) — the storage FT needs. **Stage 2 (the in-place update) is the hard part**, with a
  documented correctness landmine: the naive column-shift Bartels–Golub produces an
  upper-Hessenberg `U` whose diagonal pivots are the *old superdiagonal* `U[k,k+1]`, which are
  frequently **zero in a sparse `U`** (a dense `U` always has them, which is why the dense
  `DenseLu` update works without in-bump pivoting). A correct sparse update therefore needs
  **either** the symmetric-permutation FT (pivots = the original nonzero `U` diagonals, but
  the bottom-row elimination must be folded as an L-side row-eta and the accumulating
  permutation tracked) **or** in-bump threshold partial pivoting with row-permutation tracking
  (Reid sparse BGR, citep:reid1982sparsity). Both keep solves `O(nnz)` (no eta chain) and the
  update `O(bump)`. This is the genuine graduation; the product-form remains correct and
  `max_updates`-bounded until it lands.
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
