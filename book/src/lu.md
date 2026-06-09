# Unsymmetric LU basis engine

Most of feral is a *symmetric* indefinite LDLᵀ solver with certified inertia.
The `feral::lu` module is a **separate factorization family**: an *unsymmetric*
LU built to drive a **revised-simplex basis**. It is additive — the LDLᵀ solver
and every one of its code paths are untouched — and it deliberately does **not**
compute inertia (an unsymmetric basis has no symmetric eigenvalue structure to
certify).

The distinguishing requirement is not the one-shot factor/solve but the **rank-1
update**: a simplex iteration replaces one basic column, and the engine folds
that change into the existing factors in `O(nnz)` rather than refactoring.

## The factorization

For a square nonsingular basis `B` (`m × m`), feral computes

```text
P B Q = L U
```

where `P` is a row permutation (threshold partial pivoting, for stability), `Q`
is a fill-reducing column permutation (sparse path only; `Q = I` on the dense
path), `L` is unit lower triangular, and `U` is upper triangular. The two hot
operations of revised simplex follow directly:

```text
ftran:  solve  B x = a      (forward transformation,  B⁻¹a)
btran:  solve  Bᵀ x = a     (backward transformation, B⁻ᵀa)
```

## Dense path

For the small bases that dominate (e.g. OBBT bases on a handful of variables),
[`DenseLu`](./api.md) factors a general column-major matrix with right-looking
LU and threshold partial pivoting:

```rust,ignore
use feral::{DenseLu, LuParams};

// `cols[j]` is column j of the m×m basis (length m each).
let m = 3;
let cols = vec![
    vec![2.0, 4.0, 8.0],
    vec![1.0, 3.0, 7.0],
    vec![1.0, 3.0, 9.0],
];
let mut lu = DenseLu::factor(&cols, m, LuParams::default())?;

// ftran / btran overwrite the right-hand side in place.
let mut x = vec![1.0, 2.0, 3.0];
lu.ftran(&mut x)?;          // x ← B⁻¹ x
let mut y = vec![1.0, 0.0, 0.0];
lu.btran(&mut y)?;          // y ← B⁻ᵀ y
```

## Sparse path

For larger bases, [`SparseLu`](./api.md) is a left-looking Gilbert–Peierls LU
with an output-sensitive depth-first reach (so the factor cost is `O(flops)`,
not `O(n²)`). The fill-reducing column order is a reusable symbolic handle
computed by [`SparseLuSymbolic`](./api.md) — feral's in-tree AMD run on the
`AᵀA` (column-intersection) pattern, a stand-in for COLAMD. Because the pattern
is invariant under the row permutation and scaling, the same order is valid for
numerically different but structurally identical bases:

```rust,ignore
use feral::{SparseColMatrix, SparseLu, SparseLuSymbolic, LuParams};

let b: SparseColMatrix = /* general CSC basis */;
let symbolic = SparseLuSymbolic::analyze(&b)?;   // reusable across refactors
let mut lu = SparseLu::factor(&b, &symbolic, LuParams::default())?;

let mut x = vec![/* … length m … */];
lu.ftran(&mut x)?;
```

`should_use_dense_lu(m, nnz, &params)` mirrors the symmetric router: tiny bases
go dense unconditionally, small dense-enough bases go dense by a density gate,
and the rest go sparse.

## Rank-1 column-replacement update

This is the reason the engine exists. `update` (and `update_sparse`, which takes
the entering column already in sparse form) replaces one basic column and folds
the change into the factors in place:

```rust,ignore
let leaving_slot = 2;                 // basis column to evict
let entering = vec![0.5, 0.0, 1.5];   // new column aₙₑw
match lu.update(leaving_slot, &entering) {
    Ok(()) => { /* factors now reflect the new basis */ }
    Err(feral::FeralError::NeedsRefactor) => {
        // Budget or stability limit reached; `lu` is unchanged.
        lu.refactor(&b, &symbolic)?;  // (DenseLu::refactor takes the new columns)
    }
    Err(e) => return Err(e),
}
```

- **Dense** updates use Bartels–Golub re-triangularization (spike → cyclic
  column shift to upper-Hessenberg → Gauss sweep, folded into `L`/`U`).
- **Sparse** updates use a Forrest–Tomlin / Bartels–Golub–Reid scheme: the bump
  is re-triangularized by sparse Gaussian elimination with partial pivoting,
  recorded as a replayable *eta* and applied between the `L`- and `U`-solves.
  The work is **bump-local**, so a warm solve after a localized update stays
  sparse instead of degrading toward a full re-solve.

An update returns [`NeedsRefactor`](./api.md) — leaving the factorization
**unchanged** — when the update count (`max_updates`) or growth monitor
(`max_growth`) trips, so the caller can refactor rather than accept an unstable
factor. A vanished bump pivot returns
[`SingularBasis`](./api.md) so the simplex can repair the basis instead of
receiving a garbage solve.

## Robustness: scaling and refinement

The wrong-answer bugs that historically bit downstream simplex code were *all*
scaling/tolerance, never the update math — so the robustness layer is
load-bearing. [`LuParams::scaling`](./api.md) selects a two-sided strategy:

- `LuScaling::InfNorm` — Knight–Ruiz ∞-norm equilibration (separate row/column
  scalings).
- `LuScaling::Mc64` — unsymmetric MC64 (max-weight bipartite matching) that
  places large entries on the diagonal, with a partial-matching fall back to
  `InfNorm`.

Scaling wraps the core solve; the factorization factors the scaled matrix
`D_row Π B D_col`. For ill-conditioned bases, `ftran_refined` / `btran_refined`
run residual-based iterative refinement against the original basis:

```rust,ignore
let params = LuParams {
    scaling: feral::LuScaling::Mc64,
    refine_steps: 2,
    refine_tol: 1e-14,
    ..LuParams::default()
};
let mut lu = SparseLu::factor(&b, &symbolic, params)?;
let mut x = a.clone();
lu.ftran_refined(&b, &mut x)?;   // drives the true residual ‖Bx − a‖ down
```

## Scope

The LU engine is a Rust API today; it is **not** exposed through the
[Python bindings](./python.md) yet (no inertia, different update-centric
surface). The downstream `BasisEngine` integration and reference (UMFPACK/KLU)
benchmarks are tracked as later phases — see `dev/plans/unsymmetric-lu-epic.md`
in the repository.
