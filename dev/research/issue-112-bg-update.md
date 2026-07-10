# Issue #112: pivot-searching (Bartels–Golub) bump update for `SparseLu`

Date: 2026-07-10
Status: research → plan (`dev/plans/issue-112-bg-update.md`)
Prior art in-tree: `dev/research/ft-row-elimination-design-2026-06-21.md`
(which explicitly left "in-bump stability pivoting" as future work).

## 1. Problem

`SparseLu::update_sparse` restores triangularity after a column replacement by
the Forrest–Tomlin scheme: a symmetric cyclic `uperm` shift moves the leaving
column/row to the bottom of the bump, then the single pivotal row `r` is
eliminated in **fixed rank order** against the (unchanged) bump rows
(`eliminate_pivot_row`). The pivots of that sweep are the *retained* `U`
diagonals; the update never re-selects a pivot.

Issue #112 (measured downstream in discopt): on dense McCormick-lifted node
LPs this fails with `RefactorCause::TinyPivot` at magnitude **exactly 0.0** on
~every pivot (`st_e36`: 11 714 events, 100% of FT failures; `Growth` and
`Singular` never fire), forcing an O(m²) refactor per pivot. 120/120 captured
bases are full-rank and well-conditioned (cond ≤ 3.4e6, σ_min ≥ 1.5e-4); a
from-scratch refactor (partial pivoting) always succeeds.

## 2. Where the zero comes from

The failing check is the **final diagonal** of the rebuilt row `r`
(`sparse_update.rs:498`, `diag = rw[r]`). The intermediate pivots
(`sparse_update.rs:449-461`) are old `U` diagonals — nonzero by the factor's
`ztol` gate and every prior commit's final-diagonal gate — so they essentially
never fire; the mid-sweep `TinyPivot` sites are corruption guards, not the
observed failure.

Two facts pin down the mechanism:

1. **The bump block is nonsingular whenever the replacement basis is.** In
   rank coordinates the spiked matrix is block upper triangular
   `[[T₁, ·, ·], [0, M, ·], [0, 0, T₃]]` around the bump block `M` (every bump
   row has support only at ranks ≥ its own; rows outside the bump are
   untouched). So `det(spiked U) = det(T₁)·det(M)·det(T₃)`, and `B'`
   nonsingular ⇒ `M` nonsingular.
2. **The final diagonal is mathematically `det(M) / ∏(retained bump
   diagonals)`** — the Schur complement of the upper-triangular part of `M` —
   and the retained diagonals are all nonzero. Hence on a nonsingular `B'` the
   true value is **never** zero.

So a computed value of exactly `0.0` is floating-point **cancellation**: the
fixed-order sweep can grow the working row unboundedly (its multipliers
`rw[c]/U[c,c]` are unbounded — nothing constrains the retained diagonal to
dominate the spike-row entry), and when intermediate terms reach magnitude
`~|true| / ε`, the final subtraction returns 0.0 (or garbage). This matches
the issue's offline analysis: "cancellation-to-zero in the retained pivot
order on a nonsingular basis". No threshold tweak fixes it — the computed
pivot is literally `0.0`, and the information needed to recover it is already
rounded away. The fix must **change the elimination order**, i.e. pivot.

## 3. Literature

- **Bartels & Golub 1969** (`citep:bartels1969stable`): the original stable
  simplex update — re-triangularize the spiked factor with **row
  interchanges** chosen by magnitude at each elimination step. Growth per step
  bounded (|mult| ≤ 1 with full interchanges); backward stable like partial
  pivoting.
- **Reid 1982** (`citep:reid1982sparsity`, LA05): sparsity-exploiting BG —
  threshold pivoting between the two candidate rows (diagonal row vs spike
  row) so the sparser choice is kept when stability permits.
- **Forrest & Tomlin 1972** (`citep:forrest1972updated`): drop the interchange
  freedom entirely, keep the retained diagonals as pivots — cheapest update
  (one row eta, no fill in other rows) but **unstable in exactly the way
  observed here**; production codes pair it with a bail-to-refactor, which is
  feral's current design.
- **Suhl & Suhl 1990** (`citep:suhl1990computing`): the modern LU kernel these
  updates sit on; threshold partial pivoting with sparsity bias.
- **Huangfu & Hall 2015** (`citep:huangfu2015novel`): surveys FT vs PFI/BG
  variants in a modern dual simplex (HiGHS); FT preferred *when it commits*,
  robustness comes from the fallback chain.
- **Schork & Gondzio 2017** (`citep:schork2017permuting`): permute-first FT
  variant (paper only — BASICLU is GPL, not consulted).

Mainstream sparse-LP codes all provide a pivot-searching update precisely for
this robustness (issue text); FT-then-refactor alone is the outlier.

## 4. Design

### 4.1 Constraint: keep the symmetric `uperm` invariant

The 2026-06-21 FT design hinges on `U` being upper triangular under a single
symmetric permutation `uperm` (rows and columns ranked identically, diagonal
of row `i` stored at column `i`, `usolve`/`ut_solve`/`compute_spike` all walk
`uperm`). A textbook BG row interchange is an *unsymmetric* permutation — it
would force separate row/column rank maps through every solve and invariant.

**Key idea: implement the interchange as a physical row-content swap.** At
elimination step `k` (pivot column `c = uperm_inv[k]`), the two candidate
pivot rows are the retained row `u_rows[c]` (diagonal `piv = U[c,c]`) and the
working row `W` (the dense scatter `rw`, destined for position `r`, entry
`v = rw[c]`). To swap:

- store `W` *as* `u_rows[c]` — its rank-`k` diagonal is its column-`c` entry
  `v`, so the diagonal-first, diagonal-at-own-position convention is
  preserved; every other entry of `W` has rank > `k`, so triangularity under
  the **unchanged** `uperm` is preserved;
- the displaced old row `c` becomes the new working row (destined for `r`),
  and its column-`c` entry `piv` is eliminated against the new pivot:
  `W' = old_row_c − (piv/v)·W`;
- record two eta ops: a new **`FtOp::Swap{a: c, b: r}`** (rows of `U` and the
  corresponding solve-vector entries permute together: `s[c] ↔ s[r]`),
  followed by the usual `FtOp::Axpy{target: r, src: c, mult: piv/v}`.

`L`, `P`, `Q`, `qcol`, and all prior etas stay fixed; `G` absorbs the swap
exactly as it absorbs the axpys. `uperm` is untouched beyond the existing
cyclic shift. `apply_forward` performs `y.swap(a,b)`; `apply_transpose` the
same (a transposition is its own transpose) in the existing reversed op walk;
`compute_spike`'s eta replay swaps `w[a]`, `w[b]` and marks both touched.

The final step is unchanged: the last working row lands at position `r` (rank
`h_rank`, diagonal column `r`), gated by the same `ztol` check.

### 4.2 When to swap: FT first, BG as in-update fallback

Per the issue ("FT = cheapest when it works; BG = robust when it doesn't"):

- **Pass 1 (unchanged FT):** run the existing fixed-order sweep (never swap).
  Zero behavior/cost change on every update that commits today.
- **Pass 2 (BG rescue):** only if pass 1 fails with `TinyPivot`, roll the row
  state back (the FT pass mutates nothing but `u_rows[r]`, already
  snapshotted) and re-run the sweep with threshold row interchanges: swap when
  `|piv| < |v|` (strict-larger candidate, τ = 1 — max stability; multipliers
  then satisfy `|mult| ≤ 1`, the classic BG growth bound). If the final
  diagonal still fails `ztol`, fall through to `NeedsRefactor` exactly as
  today — the refactor remains the authoritative verdict, and accuracy is
  never traded (growth monitor still gates the commit).

This keeps the hot path byte-identical, bounds the rescue cost at ~2× one
bump sweep (still ≪ the O(m²) refactor it replaces), and needs no tuning
parameter for the common case. Gated by a new `LuParams::update_pivot_search:
bool`, default **true** (opt-out restores bit-exact legacy behavior);
rescues are counted and exposed (`pivot_search_rescues()`) so drivers like
discopt can split the cause telemetry.

### 4.3 Bookkeeping consequences (BG pass only)

- **Rollback:** each row swapped into gets snapshotted (pre-update content)
  before being rewritten, deduped against the rows already saved at update
  start (`changed` is sorted then, so a binary search suffices; a position can
  be swapped at most once — each rank is popped once). Rollback and the
  growth-monitor scan then work unchanged over the extended saved/changed set.
- **`u_above`:** swapped rows change their column sets wholesale (they take
  the working row's full support, including a column-`r` entry), so the
  incremental issue-#89 refresh is wrong for them. The BG commit path rebuilds
  `u_above` from scratch — O(nnz(U)), acceptable because the alternative this
  path replaces is a full refactor (strictly more work). The FT commit path
  keeps the incremental refresh, untouched.
- **Scatter hygiene:** the swap gathers the working row out of `rw` — the
  existing duplicate-tolerant `rw_touched` makes that ambiguous, so the sweep
  gains a dedicated touched-mark buffer (pooled, like the others) making
  `rw_touched` duplicate-free; this also deletes the `dedup_by_key` wart in
  the final gather. The eliminated entry at the swap column is cleared exactly
  (`rw[c] = 0.0`, skip the displaced diagonal in the scatter) — same
  re-enqueue landmine as the 2026-06-21 bring-up (journal 2026-06-21).

### 4.4 Why the rescue actually fixes the observed failure

The cancellation requires intermediate working-row entries of magnitude
`~|true pivot| / ε` — i.e. unbounded multipliers. With τ = 1 interchanges
every multiplier is ≤ 1 in magnitude, so the sweep's growth is bounded by
2 per step (Wilkinson bound restricted to the two-row choice), the same
guarantee the refactor's partial pivoting provides — and the refactor
demonstrably succeeds on all 120 captured bases. A regression test encodes
the mechanism directly (see plan §tests): a hand-built basis whose fixed-order
sweep doubles the working row each step (Wilkinson-style ±1 cascade) until an
engineered rounding absorbs the true diagonal (`8 + 2⁻²⁰` against a `2³⁴`
intermediate) and the final subtraction returns exactly `0.0` — while the
matrix stays modest-entried and nonsingular (true diagonal `2⁻²⁰ ≫ ztol`),
and the BG pass (which swaps on the first dominated pivot) commits with a
residual matching the refactor's.

## 5. Alternatives considered

- **Unsymmetric rank maps (separate row/column `uperm`s):** the textbook
  formulation; rejected — it threads a second permutation through `usolve`,
  `ut_solve`, `compute_spike`, the invariant checker, and every triangularity
  argument, for zero benefit over the content-swap formulation, which is
  mathematically the same interchange.
- **Always-on threshold pivoting (Reid/Suhl–Suhl style, τ ≈ 0.1..0.5) in the
  single pass:** changes eta streams and fill on updates that commit fine
  today; harder to A/B, no benefit for the failing class (the rescue engages
  regardless), and contradicts "FT when it works". Possible later as a tuning
  knob on top of the same machinery (τ becomes a parameter).
- **Second symmetric shift to a deeper rank when the final diagonal is zero:**
  useless — a symmetric permutation preserves diagonal values; the zero
  diagonal just moves.
- **Extending `zero_pivot_tol` / retry with perturbation:** the issue measured
  the pivots at exactly `0.0` with the tolerance already at `1e-13`; a
  perturbed pivot would silently poison the factor (cf. the PerturbToEps
  lesson in `dev/tried-and-rejected.md` 2026-05-02).

## 6. References

`citep:bartels1969stable`, `citep:forrest1972updated`,
`citep:reid1982sparsity`, `citep:suhl1990computing`,
`citep:huangfu2015novel`, `citep:schork2017permuting` — all in
`dev/references.bib`. BASICLU (GPL) not consulted; clean-room from papers.
