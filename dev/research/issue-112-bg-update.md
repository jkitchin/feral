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

---

## UPDATE 2026-07-10 (same session): the rescue design is impossible; the fix is compensated accumulation

The plan above (§4.2, FT-first + BG-retry-on-TinyPivot) was implemented and
then **rejected on proof**, before commit. Three findings, each verified
numerically (float + exact-rational replays of the sweep on candidate
matrices):

### 1. Proportionality theorem: pivot re-ordering cannot recover an absorbed pivot

In the row-spike form, let `W_k` be the fixed-order (FT) working row before
step `k` and `W'_k` the working row of *any* interchange variant of the same
sweep. Induction over the two step types gives `W'_k = λ_k·W_k` **exactly (in
exact arithmetic)**: a no-swap step preserves `λ`, a swap at column `c`
resets `λ ← −U[c,c]/W_k[c]`. Consequences:

- the swapped path's **true** final diagonal is `λ_final · t_FT`, and under
  domination-triggered swaps `|λ| ≤ 1` monotonically — pivot search can only
  *shrink* the true final pivot (`∏(pivots)·final = det(bump)` makes this a
  determinant identity);
- both paths skip exactly the same columns (proportional zeros), so no
  interchange schedule can "route around" a poisoned column;
- FP absorption is **scale-invariant** (ulp is relative), so a λ-scaled
  recomputation absorbs the same bit at the same step. The absorbed
  information exists nowhere any within-bump row-operation order can reach.

Hence a rescue pass run *after* the FT sweep produced an exact-`0.0` can
never soundly commit: if the computed zero was absorption
(`|t_FT| ≤ ε·I`, `I` = intermediate growth), every interchange order's
signal-to-noise is the same `t_FT/(ε·I) ≤ 1`. The retry implementation was
removed; `dev/tried-and-rejected.md` has the entry.

### 2. The actual fix: Neumaier-compensated accumulation in the sweep

The exact-`0.0` is a *summation* artifact: `rw[·]` accumulates hits whose
intermediate magnitude exceeds `|true|/ε`, and one rounded add drops the true
value's bits. Making every scatter add a Neumaier (Kahan–Babuška) two-sum —
value carried as `rw[c] + comp[c]`, error of each add banked in `comp[c]`,
collapsed once at each read — retains those bits: on the m=4 regression
matrix the compensated sweep's final diagonal equals the true value `2⁻³⁵`
**bit-for-bit** where the plain sweep returns `0.0` (and scipy's fresh LU
returns ε-noise). The classic Kahan form does **not** work here: its
`y = v − c` pre-subtraction re-absorbs the compensation into the next large
addend (verified numerically). Cost: ~4 extra flops per scatter add,
`O(bump)` extra memory (one pooled `f64` array), no allocation.

This is always-on. It also sharpens the singular case: a compensated final
diagonal of `0.0`/sub-ztol is now trustworthy evidence of a genuinely
dependent replacement, not an artifact.

### 3. What pivot search is still for — and the limits of a distilled reproducer

`update_pivot_search` (default off) now runs the whole sweep with
Bartels–Golub interchanges from the start: every multiplier bounded by 1,
growth bounded across update chains — a *trajectory* choice that prevents the
imbalance (`∏(retained diagonals) ≫ det`) from building up over long update
chains, which is where the issue #112 storms come from in vivo. It is not,
and provably cannot be, a cure after the fact (finding 1). The `FtOp::Swap`
eta machinery (physical row-content swap keeps the symmetric `uperm`
invariant, diagonal-first storage, and all prior etas intact) is exactly as
designed in §4.1.

Distillation limit (test design): any *single-shot* reproducer of an
exact-absorption failure necessarily satisfies
`σ_min(B') ⪅ δ·∏(retained diagonals)` with `δ ≤ ε·I` — the replacement basis
is within `δ` of singular, so a from-scratch factorization of it sees an
ε-relative pivot and cannot serve as the residual oracle. The discopt
captures' healthy `σ_min ≥ 1.5e-4` coexists with the exact-`0.0` only because
their retained-`U` imbalance was built by a long *chain* of updates (the
`∏retained ≫ det` split lives in the factorization state, not in `B'`).
The regression test therefore asserts (a) the committed diagonal equals the
hand-computed true value exactly, and (b) backward-stable solve residuals —
and validation on the real captures (issue acceptance) remains a discopt-side
step, now with the compensated sweep expected to commit them.
