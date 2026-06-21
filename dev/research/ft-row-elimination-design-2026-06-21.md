# Forrest–Tomlin row-elimination: design & route selection (issue #87)

Date: 2026-06-21
Status: **design chosen — closes the open spike**
[[asymptotic-bump-update-spike-2026-06-18]]. Implementation plan:
`dev/plans/ft-row-elimination-2026-06-21.md`.

Resolves the open question in `dev/research/asymptotic-bump-update-spike-2026-06-18.md`
("pick and land one route") for the residual **O(bump²)** FT row-elimination on
wide McCormick spikes (issue #87, Step-2 of discopt#229).

---

## 1. Diagnosis — the current update is *column* elimination, not Forrest–Tomlin

`eliminate_bump` (`src/lu/sparse_update.rs`) sets the spike into `U`'s column `r`
(`set_column_r`) and then runs **left-to-right partial-pivoting Gaussian
elimination over the whole bump `[r, h]`** (`eliminate_bump`). That is a
*Bartels–Golub full re-triangularization in fixed pivot order*, **not** a
Forrest–Tomlin update. It eliminates the **spike column**: the first pivot column
`r` has one entry per spike row, so it subtracts the pivot row from `O(bump)`
target rows, and the fill cascades through columns `r+1..h`. On a **dense spike**
(`autocorr_bern`: every structural column is dense) the bump is `[r, m-1]` and the
work — and the recorded eta — are `O(bump²)`.

Two costs, both `O(bump²)`:
- the elimination itself (issue #87: ~6.6 ms/update at m=3005, ≈ a full refactor);
- the **eta** it records, which then slows *every subsequent solve* until refactor.

### Measured worst case (BEFORE — `lu_wide_bump_probe`, this host)

Tridiagonal basis (so `L`,`U` are bidiagonal = **sparse**), early slot replaced by
a **dense** column ⇒ dense spike, full-width bump, sparse `U` — the `autocorr_bern`
regime distilled:

| m | factor_nnz | µs/update | µs/update/m | eta_ops/update |
|---:|---:|---:|---:|---:|
| 250 | 748 | 4 476 | 17.9 | 30 752 |
| 500 | 1 498 | 20 286 | 40.6 | 124 002 |
| 1000 | 2 998 | 80 407 | 80.4 | 402 980 |
| 2000 | 5 998 | 1 693 684 | 846.8 | 964 895 |
| 4000 | 11 998 | 10 246 482 | 2 561.6 | 2 089 427 |

16× in `m` ⇒ **2 290×** in time (≈ m^2.8). `factor_nnz` stays `O(m)` (≈3m) — the
fill is **not** in `U`; the blow-up is the transient elimination + the eta. This is
the issue #87 fingerprint reproduced with no external dependency.

## 2. The fix — eliminate the *pivotal row*, after a symmetric permutation

Forrest–Tomlin (1972), Reid (1982), and the modern Schork–Gondzio variant
(ERGO-17-002, the algorithm in `BASICLU` — GPL, so *paper only*, clean-room) all
do the opposite of what feral does today:

1. **Symmetrically permute** column `r` *and* row `r` to the end of the bump
   (cyclic shift of `[r, h]`). The bump becomes upper triangular **except one
   row** (the old row `r`, now last) carrying sub-diagonal entries, and one dense
   last column (the spike — genuinely part of the new factor).
2. The permuted diagonal pivots are the **old `U` diagonals** `U[j,j]`
   (`j ∈ [r, h)`) — all **nonzero**. This is *why the symmetric permutation
   dodges the zero-pivot landmine* that killed the column-shift attempt
   (journal 2026-06-08-01.org, 19:30): that attempt left the spike in place and
   used the **old superdiagonal** (frequently zero) as the pivot.
3. **Eliminate that single row** by a sparse forward sweep against the (sparse)
   triangular part: one row-eta. Cost `O(nnz(spike) + fill in one row)` = `O(bump)`
   for sparse `U`, **not** `O(bump²)`. The eta is one row + the spike column.

Schork–Gondzio refinement (adopt): re-triangularize **by permutation alone**
whenever the spiked digraph is acyclic, and fall back to the row-elimination only
on the residual cyclic part — zero arithmetic / zero fill in the common case.

### Why this is `O(bump)` here, concretely

For the tridiagonal probe: after the symmetric permutation the spike is the dense
last bump column (`O(m)` genuine nonzeros, inserted once); the old row `r` had two
entries (`U[r,r]`,`U[r,r+1]`) ⇒ **one** elimination axpy. Update `O(m)`, eta `O(m)`
— vs `O(m²)` today.

## 3. Why not the other two routes

- **Route 1, column-ordering lever** (keep churned/dense columns late). Workload-
  specific heuristic, partly driver-side (discopt), changes no asymptotic, and is
  fragile to the LP structure. Useful as a *complement*, not the robust fix the
  issue asks to "pick and land." Rejected as primary.
- **Route 3 as *physical* permutation** (relabel `U`'s column indices + prior etas
  each update). Two killers: (a) staling prior etas needs relabeling the whole eta
  history → `O(k²·bump)` over a chain; (b) encoding the cyclic shift as per-eta
  `FtOp::Swap`s adds `O(bump)` swaps per eta → solve cost `O(k·bump)`, re-creating
  the PFI blow-up the FT eta exists to avoid. Measured today: the dense-spike eta
  already reaches 2.09M ops; we must *shrink* the eta, not grow it. Rejected.

## 4. Chosen route — logical-permutation Forrest–Tomlin (Route 2 ∪ Route 3-logical)

Carry a single evolving permutation `uperm` (pivot-position ↔ triangular rank) and
apply it **once** per solve; never relabel `U`'s stored indices or the prior etas.

Invariant kept: `P A Q = L G U`, `G = E₁⁻¹…Eₜ⁻¹`. New: `U` is upper triangular
*in `uperm` order* (it is identity at factor time, so all existing tests/solves are
unchanged until the first update). Each update appends one row-eta (axpys in fixed
pivot-position coordinates — the existing `FtOp` model, unchanged) and composes the
cyclic shift into `uperm` (`O(bump)`).

Touch list:
- **Add** `uperm`, `uperm_inv` to `SparseLu` (identity at factor).
- **`usolve`/`ut_solve`**: traverse rows in `uperm` order; each row's pivot is its
  `uperm`-diagonal. (At identity `uperm` this is the current loop — keep the fast
  path.)
- **`eliminate_bump` → `ft_update`**: spike-insert, permute-to-triangular
  (Schork–Gondzio symbolic pass), single-row elimination, compose `uperm`, record
  the row-eta. The `FtOp` eta and `compute_spike` are reused.
- ftran/btran outer structure, `L`, `P`, `Q`, scaling, refinement: **unchanged**
  (etas and `L` stay in fixed pivot-position coordinates).

### Correctness story (the spike's exit criterion)

- **Zero-pivot landmine**: gone — permuted pivots are the nonzero old `U`
  diagonals (§2.2), not the old superdiagonal.
- **Stored-state consistency**: `uperm` is the *only* new evolving state; `U`'s
  indices and all prior etas are immutable across updates, so no relabel/staleness.
  A new differential invariant test asserts `U` is upper-triangular under `uperm`
  after every update and that `‖U(uperm-order)‖` reconstructs `P A Q (G L)⁻¹`.
- **Stability**: FT does not pivot for magnitude inside the bump, so a tiny
  permuted diagonal can grow elements. Keep the existing `growth`/`max_growth`
  monitor over changed rows and the `NeedsRefactor` contract; the authoritative
  singularity/stability verdict remains a fresh refactor. Schork–Gondzio's
  permutation freedom is also used to prefer larger pivots when triangularizing.
- **Oracle**: `DenseLu` and the residual checks `‖Bx−a‖`, the existing
  dense↔sparse parity and 25-update FT-chain tests, and the in-tree `casctanks`
  trace are all external oracles (no self-oracle in one session).

## 5. Validation plan (before/after + scaling — issue #87 and user ask)

- `lu_wide_bump_probe` exponent drops from ≈2.8 to ≈1.0; `eta_ops/update` from
  `O(m²)` to `O(m)`. Re-run the table above AFTER.
- `casctanks_ft_update` bench (`tests/data/lu_trace/casctanks.txt`, 144 updates):
  BEFORE = **16.88 ms**; expect a large drop.
- Differential: every existing `lu_sparse`/`lu_scaling`/`lu_update_*` test green,
  plus a new wide-bump differential (dense↔sparse residual after each of a chain of
  dense-column updates) and the `uperm`-triangularity invariant test.

## References

- Forrest & Tomlin 1972 (`citep:forrest1972updated`); Reid 1982
  (`citep:reid1982sparsity`); Suhl & Suhl 1990 (`citep:suhl1990computing`).
- Schork & Gondzio, "Permuting Spiked Matrices to Triangular Form and its
  Application to the Forrest–Tomlin Update", ERGO-17-002 (2017) — paper only;
  `BASICLU` is GPL and is **not** consulted as source.
- Huangfu & Hall 2015, "Novel update techniques for the revised simplex method"
  (HiGHS update technology) — corroborates FT eta-size advantage over PFI.
