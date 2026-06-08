# Unsymmetric LU Factorization as a Simplex Basis Engine

**Status:** Pre-implementation research note (mandatory before any LU code, per CLAUDE.md)
**Date:** 2026-06-08
**Issue:** #81 (Unsymmetric LU with rank-1 factorization updates for large-scale simplex bases)
**Related spec sections:** new factorization family — not covered by the symmetric spec
**Key references:** citep:gilbert1988sparse, citep:forrest1972updated, citep:bartels1969stable,
citep:reid1982sparsity, citep:suhl1990computing, citep:davis2004colamd, citep:duff2001mc64,
citep:knight2014symmetry, citep:higham2002accuracy

---

## 1. Overview and motivation

feral is today a *symmetric* indefinite LDLᵀ solver with certified inertia. Issue #81 adds a
**new, separate factorization family**: an *unsymmetric* LU designed from the start to drive
a **revised-simplex basis**. The distinguishing requirement is not the one-shot factor/solve
but the **rank-1 factorization update**: the simplex replaces one basic column per iteration
and must update `B⁻¹` cheaply (≈O(nnz)) rather than refactor (O(factor)).

The two hot operations in revised simplex are

    ftran:  solve  B x = a      (forward transformation, B⁻¹a)
    btran:  solve  Bᵀ x = a     (backward transformation, B⁻ᵀa)

and the structural operation is

    update: swap basic column q out for an entering column, folding the change
            into the factors in place.

This note covers the algorithmic foundation for the dense path, the sparse path, both update
mechanisms, ordering, scaling, and refinement. Inertia is **not** part of LU — an unsymmetric
basis has no symmetric eigenvalue structure to certify.

Non-goals here: the downstream `BasisEngine` integration in `pounce-simplex` (a different
repo, absent from this environment); benchmarking against UMFPACK/KLU. Those are later phases.

---

## 2. The factorization: PAQ = LU with threshold partial pivoting

For a square nonsingular basis `B` (m×m), we compute

    P B Q = L U

where `P` is a row permutation (from partial pivoting, for stability), `Q` is a column
permutation (from fill-reducing ordering, sparse path only — `Q = I` on the dense path), `L`
is unit lower triangular, and `U` is upper triangular. With `B = P⁻¹ L U Q⁻¹`, solves become

    ftran  B x = a:   z = P a;  L y = z;  U w = y;  x = Q w
    btran  Bᵀ x = a:  Bᵀ = Q⁻ᵀ Uᵀ Lᵀ P⁻ᵀ;  solve in reverse with transposed triangles.

### 2.1 Threshold partial pivoting

Strict partial pivoting selects, at step k, the row of maximum magnitude in column k and
swaps it to the diagonal — element growth bounded by `2^(m-1)` (rarely attained). For a
*sparse* factorization, always taking the row-max can force unnecessary fill. **Threshold
partial pivoting** relaxes this: accept any candidate pivot `|a_pk|` satisfying

    |a_pk| ≥ u · max_i |a_ik|,   u ∈ (0, 1]

where `u` is the pivot threshold. `u = 1` is strict partial pivoting (max stability); smaller
`u` trades a controlled growth factor (≤ `1 + 1/u` per step, citep:higham2002accuracy) for
freedom to pick a sparser pivot row. Phase-1 default `u = 1.0` on the dense path; the sparse
path uses `u` to prefer the diagonal/sparser pivot when it is within threshold. Singularity:
if no candidate exceeds `zero_pivot_tol`, the column is numerically null → `SingularBasis` or
a `PerturbToEps` floor (mirrors `ZeroPivotAction`).

---

## 3. Dense LU and the dense rank-1 update (Bartels–Golub)

### 3.1 Dense factorization

Right-looking outer-product LU into a single packed `m·m` column-major buffer: the strict
lower triangle holds the unit-`L` multipliers, the diagonal and upper triangle hold `U`. At
step k: pivot search in column k (rows k..m), threshold test, row swap across the whole row
(and record in `perm`), compute multipliers `L(i,k) = a(i,k)/a(k,k)`, rank-1 trailing update
`A(k+1:,k+1:) -= L(k+1:,k) · U(k,k+1:)`. Chosen for clarity and ease of hand-verification;
this is the correct-but-not-yet-optimal dense kernel.

### 3.2 The spike

Replacing basic column at position `q` by a new column `aₙₑw`: the entering column expressed
in the current `L`-coordinate system is the **spike**

    s = L⁻¹ P aₙₑw

i.e. ftran *stopped before the U-solve* (`ftran_partial`). Overwriting column `q` of `U` with
`s` makes `U` upper-triangular **except** column `q`, which now has entries below the diagonal
down to row m−1.

### 3.3 Re-triangularization (Bartels–Golub, citep:bartels1969stable)

Cyclically shift columns `q+1..m−1` of `U` left by one and move the spike column to position
m−1 (tracked in the explicit column permutation `Q = qcol`). The result is **upper-Hessenberg**:
one nonzero subdiagonal on positions `q..m−1`. Sweep `k = q..m−2` eliminating each subdiagonal
entry `H(k+1,k)` with a 2-row Gauss elimination.

**As implemented (supersedes the eta sketch above):** the dense path does **not** keep an
eta-replay file. Each elimination is folded *in place* into the factors — the row op
`row_{k+1} -= mult·row_k` on `U`, and the matching column op `col_k += mult·col_{k+1}` on `L`
(this is `L ← L·M⁻¹`, `U ← M·U` for the elementary `M = I − mult·e_{k+1}e_kᵀ`). Because
`col_{k+1}` of unit-lower `L` is supported on rows `≥ k+1`, the column op leaves `L`'s unit
diagonal intact and keeps it unit-lower; `U` stays upper-triangular. The cost is therefore the
dense `O(m²)` re-fold per update, not `O(m)` — still well below an `O(m³)` refactor, and the
dense regime (tiny OBBT bases) does not need the sparser eta file that the **sparse** path (§4)
uses. The work is done on clones of `L`, `U`, `Q`, committed only on success, so a
`NeedsRefactor` leaves `self` unchanged. This choice is recorded in `dev/decisions.md`
(2026-06-08, "Dense update representation").

**No in-bump pivoting on the dense path.** The dense sweep takes the diagonal as the pivot with
no threshold row interchange; instability is caught by the growth monitor (`max_growth`) and a
vanishing bump pivot returns `NeedsRefactor`, rather than being repaired by a swap. Consequently
the original `perm` is never touched by an update (it still defines the factor-time `P B Q = L U`),
so the permutation-composition landmine that an eta-with-swaps scheme would face does not arise
here. (The **sparse** Forrest–Tomlin path §4 *does* pivot inside the bump and records the swaps
in its eta — see §4.3.)

### 3.4 Refactor trigger

`update` returns `NeedsRefactor` (leaving `self` unchanged so the caller can `refactor`) when
`updates_since_refactor + 1 ≥ max_updates` OR the growth monitor exceeds `max_growth`. A
degenerate spike pivot under `Fail` returns `SingularBasis{column: q}` so the simplex can
repair the basis rather than receive a garbage solve.

---

## 4. Sparse LU (Gilbert–Peierls) and the sparse Forrest–Tomlin update

### 4.1 Gilbert–Peierls left-looking LU (citep:gilbert1988sparse)

Factor column by column. For column j, the nonzero pattern of column j of L is the set of
rows reachable from `nnz(A(:,j))` in the directed graph of the already-computed L — found by
a depth-first search yielding a topological order. The sparse triangular solve `L(:,1:j-1) x =
A(:,j)` is then performed only over that reachable set, giving the column of U (above the
pivot) and the candidate column of L (below). Threshold partial pivoting picks the pivot from
the candidate L entries; the chosen row is recorded in `P`. Total cost is O(flops) — the
defining property that makes sparse partial pivoting practical. This is the KLU/SuperLU
kernel.

### 4.2 Column ordering Q (fill reduction)

Partial pivoting precludes a purely a-priori symbolic factorization, but a good **column**
order still bounds fill. The faithful tool is COLAMD (citep:davis2004colamd): approximate
minimum degree on the column-intersection graph `pattern(AᵀA)` *without forming AᵀA*. Phase-1
reuses what feral already owns: form the `AᵀA` pattern explicitly and run the in-tree
symmetric `feral_amd::amd_order` on it (the Hungarian/contract plumbing at
`src/symbolic/mod.rs:535-569` is the template for the i32-contract conversion). Documented as
a stand-in for true COLAMD. `Q` is the **reusable symbolic handle**: across numerically
different but structurally identical bases, recompute only the numeric factor.

### 4.3 Forrest–Tomlin update (citep:forrest1972updated)

The sparse specialization of Bartels–Golub that preserves sparsity. U is held with a row-wise
view (and row/column permutations) so the spike's elimination touches only the spike row and
the eliminated rows. On a column replacement: the spike enters U; permute the spike column to
the right end of the bump and its row to the bottom (upper-Hessenberg bump); eliminate the
single resulting subdiagonal with **one row eta** appended to a growing **row-eta file**.
ftran applies the row etas forward, btran transposed in reverse. Refactor when the eta-file
size or a fill/round-off budget (`max_updates`/`max_growth`) trips, or a stability monitor
demands it (citep:reid1982sparsity gives the sparsity/stability-aware permutation choice and
the budget rationale; citep:suhl1990computing gives the practical refactor cadence).

The dense (§3) and sparse (§4) updates share the `EtaUpdate`/replay control flow deliberately:
the sparse phase swaps the *storage* (sparse row-eta vectors vs dense Gauss records), not the
solve structure.

---

## 5. Robustness layer

### 5.1 Scaling

Two-sided scaling `D_row B D_col` puts the factorization on a well-balanced matrix. Two
strategies, selected by `LuScaling`:

- **∞-norm equilibration** (Knight–Ruiz, citep:knight2014symmetry): iterate row and column
  ∞-norm balancing until convergence. The unsymmetric case keeps *separate* `d_row`, `d_col`
  (the symmetric `src/scaling/infnorm.rs` collapses them); we adapt that code.
- **Unsymmetric MC64** (citep:duff2001mc64): a maximum-weight bipartite matching on the
  per-column log-magnitude graph that places large entries on the diagonal and yields
  `D_row = diag(exp u)`, `D_col = diag(exp v)/cmax` from the matching duals, plus a row
  permutation. **Key reuse:** feral's Hungarian solver `hungarian_match(&CostGraph) ->
  Matching` (`src/scaling/hungarian.rs:448`, `pub(crate)`) is *already a general unsymmetric
  bipartite matcher* — it returns `perm[j] = row matched to column j` and the duals `u`, `v`.
  Unsymmetric MC64 is therefore a thin *driver*: build the cost graph from the general basis,
  call the existing kernel, read off the permutation and duals. (The symmetric MC64 in
  `src/scaling/mc64.rs` symmetrizes the matching per citep:knight2014symmetry — we skip that
  symmetrization for the unsymmetric basis.) Partial matching ⇒ fall back to ∞-norm
  (mirrors the symmetric `Mc64FallbackToInfnorm` policy).

The factor consumes a `LuScale { row_perm, d_row, d_col }` and factors `D_row P B D_col`;
ftran/btran apply the scaling around the solve. The MC64 row permutation composes with the
pivot permutation — a flagged correctness point with a dedicated test.

The wrong-answer bugs that historically bit `pounce-simplex` were *all* scaling/tolerance,
never the update math (per the issue), so this layer is load-bearing, not optional.

### 5.2 Iterative refinement

For ill-conditioned bases a single solve loses digits. Refinement loops on the true residual:

    x = ftran(a)
    repeat up to refine_steps:
        r = a − B x            (computed with the matrix, not the factors)
        if ‖r‖/‖a‖ < refine_tol: stop
        δ = ftran(r);  x += δ

`r` uses `GeneralMatrix`/`SparseColMatrix` `matvec` — an oracle-free convergence check, since
it drives the *actual* residual down rather than comparing to a self-computed truth. btran is
symmetric. Exposed as opt-in `ftran_refined`/`btran_refined`.

---

## 6. Auto dense/sparse routing

Mirror `should_use_dense_fast_path` (`src/numeric/factorize.rs:1296`): `should_use_dense_lu(m,
nnz)` returns dense for tiny `m` unconditionally and for small dense-enough bases (density
gate), sparse otherwise, with a `LuParams` override. The simplex hits the dense regime
constantly via OBBT (GLOBALLib LPs are 1–43 vars), so the engine must not pay sparse overhead
on a 15×15 basis.

---

## 7. Test and oracle plan

Per CLAUDE.md, implementation and oracle cannot both be authored this session unless the
oracle is *external*. Two authoritative external oracles plus oracle-free property checks:

1. **Hand-worked exact** (factors/solutions computed on paper, hardcoded):
   - 2×2 `B=[[0,2],[3,4]]` — forces a pivot swap; known `P`, `L`, `U`, and `x` for a chosen
     RHS.
   - 3×3 `[[2,1,1],[4,3,3],[8,7,9]]` — textbook getrf, `P=I`, integer `L`,`U`.
   - 3×3 needing a mid-factor swap — assert `‖PBQ − LU‖ < 1e-12`.
2. **Equation-residual property checks** (verify the equation, not a self-computed truth):
   ftran `‖Bx−a‖/‖a‖ < 1e-10`; btran `‖Bᵀx−a‖/‖a‖ < 1e-10`; post-update `< 1e-8`;
   k-update chain; `ftran_partial` == hand-built spike; refactor-matches-updated-factor.
3. **Dense↔sparse agreement** — same basis through both paths agrees to ~1e-9 (a consistency
   check; not the sole oracle, since both paths are new this session).
4. **Robustness** — ill-scaled basis: `None` gives a large residual, `InfNorm`/`Mc64` recover
   it; ill-conditioned basis: `ftran_refined` drives the residual below `refine_tol`; MC64
   partial-matching → InfNorm fallback.
5. **Adversarial** — repeated columns → `SingularBasis`; near-singular `PerturbToEps`
   succeeds finite; `max_updates` budget → `NeedsRefactor` with `self` unchanged.

Files: `tests/lu_dense.rs`, `tests/lu_sparse.rs` (structure mirrors `tests/dense_ldlt.rs`).

---

## 8. Why this reuses feral, and what is genuinely new

Reused: error type, column-major/CSC storage idioms, the AMD ordering entry + i32-contract
conversion, the `should_use_dense_fast_path` router idiom, and — crucially — the Hungarian
matching kernel (already unsymmetric-capable) and the ∞-norm equilibration code.

New: the general (unsymmetric) dense and CSC basis types; the LU factor structs; Gilbert–
Peierls symbolic+numeric; the two rank-1 update mechanisms and their eta replay; the
unsymmetric MC64 *driver*; the two-sided equilibration adaptation; refinement; the router.
Inertia is absent by design.
