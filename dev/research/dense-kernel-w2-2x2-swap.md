# Research: W-2 swap-required 2×2 inline (Phase A2)

Date: 2026-04-28
Status: research note for the Phase A2 implementation
Related:
- `dev/plans/dense-kernel-w2-2x2-swap.md` (this session's plan)
- `dev/research/dense-kernel-attribution-2026-04-28.md` (lever attribution)
- `dev/journal/2026-04-28-01.org` 14:30 entry (probe output)
- `src/dense/factor.rs:1344-1652` (`lblt_panel_frontal`)
- `src/dense/factor.rs:1602-1679` (`peek_ahead_replay`)
- `src/dense/factor.rs:2552-2585` (`swap_rows_cols`)
- `src/dense/factor.rs:1947-2266` (`scalar_pivot_step`, the reference)

## 1. The case

Inside the panel at `col = k + c`, scalar BK fires a 2×2 pivot when
`akk = |a[col,col]| < alpha_bk * gamma0` where
`gamma0 = max_{i > col} |a[i, col]|` and `alpha_bk ≈ 0.6404`.

The argmax row `r` is the row index achieving `gamma0`. Two cases:

1. **`r == col + 1`** — the 2×2 spans consecutive columns. Phase A
   handles this inline. Probe shows ~17.4% of all bails are not this
   case but are still inline-rejected (swap-1×1 / LAPACK-1×1 wins).
2. **`r > col + 1`** — the 2×2 needs a symmetric swap: rows and
   columns `(col + 1, r)` exchange. Scalar calls
   `swap_rows_cols(a, nrow, col + 1, r, perm)`, then proceeds with
   the now-consecutive 2×2. Phase A bails to scalar
   (`PanelStatus::ScalarFallback`) on this case.

The probe measured 19/23 panel bails (82.6%) to be case 2. ACOPR30 is
84.6% case 2 by itself.

## 2. Why scalar can do the swap and the panel currently cannot

Scalar BK applies eager rank-1 (and rank-2) updates at every accepted
pivot:

```
for q in 0..k:
    A_t -= rank-{1 or 2 from pivot q}
```

So when scalar reaches column `col = k`, all trailing rows of all
trailing columns reflect the full rank-`k` state. `swap_rows_cols`
operates directly on this: swap diagonal `(p,p) ↔ (q,q)`, swap
column entries below row q, swap row entries left of column p, and
swap the cross-region.

The panel's deferred-Schur invariant is different. Define:
- `A`: original frontal matrix at panel entry (column `k`).
- `c`: number of pivots committed in the current panel so far.
- `P`: pending rank-`c` update from those pivots.
- `A_state`: the current memory state of `a[..]`.

Then for trailing rows/cols `(>= k + c)`:

```
A_state[i, j] = A[i, j]                     for i, j >= k + c    (1)
```

That is, the panel has NOT applied the rank-`c` update to the
trailing block. The committed L columns (col `k..k+c-1`, rows
`>= k + c`) hold scaled values, and `d_panel[0..c]` + `subdiag[k..k+c]`
hold the diagonal block. The pending update is:

```
A_t_correct[i, j] = A[i, j] - sum_{q=0..c} (L_p row i col q)
                                       * D_q
                                       * (L_p row j col q)^T   (2)
```

`peek_ahead_replay(a, nrow, k, c, target_col, d_panel, subdiag)`
implements equation (2) restricted to one column: it overwrites
`a[target_col * nrow + (target_col..nrow)]` so that, for that one
column, the trailing entries match what scalar would see at this
point in the factorization.

## 3. Key claim: deferred swap commutes with deferred update

**Claim.** Let `Π` be the symmetric permutation matrix swapping
rows/cols `p = col + 1` and `q = r > col + 1` in the trailing region.
Then for any `i, j >= k + c`:

```
(Π A_t_correct Π^T)[i, j]
   = (Π A Π^T)[i, j]
       - sum_{q=0..c} (Π L_p)_row_i_col_q * D_q * (Π L_p)_row_j_col_q^T
```

i.e., applying the symmetric swap to `A_t_correct` is identical to
applying the swap to both `A` (the deferred input) and to the rows
of `L_p` (the committed L block), then applying the rank-`c` update.

**Proof sketch.** The rank-`c` update is `A_t_correct = A - L_p D L_p^T`.
Conjugating by `Π`: `Π A_t_correct Π^T = Π A Π^T - (Π L_p) D (Π L_p)^T`.
The diagonal D is unaffected. ∎

**Consequence.** To "do the swap inside the deferred state" we need to
exchange:
1. Trailing entries of `a` for rows/cols `(p, q)` (touches the
   block `a[col, row]` for col, row >= k + c with at least one of
   col or row in {p, q}). This is exactly what `swap_rows_cols`
   already does.
2. Rows `p, q` of the committed L block (columns `0..c` in panel
   coords, i.e., `k..k+c-1` in matrix coords). `swap_rows_cols`
   already does this on line 2582:
   ```rust
   for j in 0..p { a.swap(j*n + p, j*n + q); }
   ```
   This loop's range `j in 0..p` includes the L-block range `j in 0..k+c`
   (since `p = k + c + 1 > k + c`). ✓

So `swap_rows_cols` operates correctly on the deferred state — its
semantics happen to be "permute everything left of column p and below
row p", which captures both regions (1) and (2) above.

**Caveat.** The swap takes `A_t_correct[p, ...]` ↔ `A_t_correct[q, ...]`
in the conjugated coordinates. It does NOT swap any `d_panel` or
`subdiag` values, because those are diagonal-block storage from
already-committed pivots and indexed by panel position, not matrix
row. The new pivot at `c+1` (the swapped row, now at panel col
`c+1`) reads `d_panel[c+1]` after we write it. ✓

## 4. The peek-ahead requirement

Inside `lblt_panel_frontal`, after deciding the 2×2 trigger, the code
needs to read:
- `a[r * nrow + r]` (= `arr` for the swap-1×1 reject test)
- `a[col * nrow + r]` (= a value used in growth-bound `tmax`)
- `a[r * nrow + i]` for various `i` (gamma_r computation)

In the no-swap case (`r == col + 1`), Phase A peek-ahead's column
`col + 1` once. In the swap case, both column `col + 1` AND column
`r` need to be in scalar-state before we can do the swap (because
the swap reads from both). After the swap, we proceed exactly as in
the no-swap path with `r_idx = col + 1` (which now holds the data
that was at `r`).

**Two-column peek-ahead.** Calling `peek_ahead_replay` twice:
```rust
peek_ahead_replay(a, nrow, k, c, col + 1, d_panel, subdiag);
peek_ahead_replay(a, nrow, k, c, r,       d_panel, subdiag);
```
is bit-exact with two scalar updates because:
- Each call writes only to its own target column (`a[target_col * nrow
  + target_col..nrow]`).
- Each call reads only from `a[q_col * nrow + ...]` for `q < c`, i.e.,
  from committed L columns. Those are not modified by either call.
- The two target columns (`col + 1` and `r`) are distinct.

So the two calls are order-independent and disjoint — no aliasing.

## 5. Trace: a hypothetical ACOPR30-style 4×4 sub-block

Take a 4×4 fully-summed sub-block typical of ACOPR30 KKT with
indefinite primal block (numbers chosen for illustrative clarity,
not a real ACOPR30 trace):

```
A =  [  1.0e-3  ─        ─        ─    ]   <- col 0
     [  3.0     2.0e-3   ─        ─    ]   <- col 1
     [  0.5     1.0      4.0e-3   ─    ]   <- col 2
     [  0.5     8.0      0.6      5.0e-3]   <- col 3
```

Lower triangular only; symmetric.

**Step 1, panel position c=0.** col = k + 0 = 0.
- `gamma0 = max(|3.0|, |0.5|, |0.5|) = 3.0` at row `r = 1`.
- `akk = 1.0e-3`, `alpha_bk * gamma0 ≈ 0.64 * 3 ≈ 1.92`. So
  `akk < alpha * gamma0` → 2×2 trigger.
- `r == col + 1 = 1`. NO SWAP. Phase A handles this.

OK that wasn't a swap. Let's permute the example so col 1 and col 3
are swapped (which is the natural ACOPR-style pattern when the
maximally-coupled row is the slack diagonal at the bottom):

```
A' = [  1.0e-3   ─        ─        ─    ]
     [  0.5      5.0e-3   ─        ─    ]
     [  0.5      0.6      4.0e-3   ─    ]
     [  3.0      8.0      1.0      2.0e-3]
```

Now col 0 has gamma0 at row 3:
- `gamma0 = 3.0`, `r = 3`.
- `akk = 1.0e-3 < 1.92`. 2×2 trigger.
- `r = 3 ≠ col + 1 = 1`. **SWAP REQUIRED.**

**What scalar does:**
1. `swap_rows_cols(a, 4, 1, 3, perm)`:
   - perm.swap(1, 3) → perm becomes [0, 3, 2, 1]
   - a.swap(diag entries 1↔3): (5e-3, 2e-3) become (2e-3, 5e-3)
   - swap cols 1 and 3 below row 3: nothing (row > q=3 is empty)
   - swap a[1*4 + 2] (col 1, row 2) with a[2*4 + 3] (col 2, row 3):
     0.6 ↔ 1.0
   - swap a[0*4 + 1] (col 0, row 1) with a[0*4 + 3] (col 0, row 3):
     0.5 ↔ 3.0
   Result:
   ```
   A'' = [ 1.0e-3   ─        ─        ─    ]
         [ 3.0      2.0e-3   ─        ─    ]
         [ 0.5      1.0      4.0e-3   ─    ]
         [ 0.5      8.0      0.6      5.0e-3]
   ```
   (= the original A from the start of the trace.)
2. Continue the no-swap 2×2 path on `A''[0:2, 0:2]`.

**What the panel must do (Phase A2):**
At `c = 0` there are no committed pivots, so `peek_ahead_replay` is a
no-op. Conceptually:
1. peek_ahead_replay onto col 1 — no-op (`n_committed = 0`).
2. peek_ahead_replay onto col 3 — no-op.
3. swap_rows_cols(a, 4, 1, 3, perm) — same as scalar.
4. Continue the no-swap 2×2 path. r_idx = col + 1 = 1.

**Numerically identical to scalar.** ✓

## 6. Trace: panel at c=2 (after a committed 1×1 + 1×1)

Same `A'` matrix, but suppose c=2 and we're looking for a third pivot
at col=2, with row 3 as the swap target. The committed pivots are at
panel positions 0 and 1, with `d_panel = [d0, d1]` and
`subdiag[k..k+2] = [0, 0]` (both 1×1).

Before the panel ever inspects col=2, eager scalar would have applied:
```
A_t_correct[i, j] = A'[i, j] - L_p[i, 0] * d0 * L_p[j, 0]
                            - L_p[i, 1] * d1 * L_p[j, 1]
for i, j >= 2.
```

Trailing entries `a[2*4+2]`, `a[2*4+3]`, `a[3*4+3]` and the symmetric
mirror `a[2*4+3]` need this update applied before the panel can read
them.

`peek_ahead_replay(a, 4, k=0, n_committed=2, target_col=2, d_panel,
subdiag)` writes the correct values into `a[2*4+2 .. 2*4+4]` (rows 2
and 3 of column 2). It does NOT touch column 3.

For the swap to read `a[3*4 + 3]` (gamma0 search), `a[3*4 + i]` (gamma_r
search), and `a[col=2 * 4 + 3]` correctly, we need column 3 also
peek-ahead'd:
`peek_ahead_replay(a, 4, k=0, n_committed=2, target_col=3, d_panel,
subdiag)`.

Now both columns 2 and 3 are in scalar-state at the trailing region.
The L-block rows (cols 0 and 1, rows 2 and 3) were never modified by
peek-ahead — they are the committed L data and remain canonical.

`swap_rows_cols(a, 4, p=3, q=3, perm)` is a no-op when p=q. Adjust:
suppose r=3 and col+1=3, then no swap. So this trace requires
`col+1=3` and `r > 3`, but nrow=4 means r can't be > 3. The trace
needs nrow ≥ 5 to be informative. Skipping the worked numerics — the
structural argument from §3 covers correctness.

## 7. State after a successful swap-2×2

Panel coords: c → c + 2. Matrix coords: pivot accepted at cols
`(k+c, k+c+1)`. `d_panel[c] = d11`, `d_panel[c+1] = d22`,
`subdiag[k+c] = d21`. Inertia incremented (sign of det → pos/neg, or
zero/refinement).

`perm` reflects the swap. Crucially, the swap in `perm` must already
be in place when `factor_frontal_blocked_in_place` later reads perm
to slot the L block back into the parent supernode's coordinate
system. This is the only externally-visible effect of the panel's
swap.

The L columns at `k+c` and `k+c+1` are the standard 2×2 inverse-scaled
columns, computed exactly as in Phase A's no-swap 2×2 path
(`scalar_pivot_step:1804-1820` for reference, lines `1495-1502` for
the panel mirror). No new arithmetic.

## 8. State after a bail AFTER the swap

The panel may execute the swap, peek-ahead the new col+1, and then
discover one of:
- swap-1×1 wins (rare)
- LAPACK-1×1 wins (rare)
- growth-bound or det-floor reject

In each case, return `ScalarFallbackPeekedNext`. The state is:
- perm: SWAPPED (committed). Caller must NOT undo.
- L cols 0..c+1 unchanged (L commits happen only on accept).
- col `k+c+1` (the swapped row, now at position k+c+1) has been
  peek-ahead'd.
- col `k+c` is in original deferred state (not peek-ahead'd in the
  swap branch, because we only peek-ahead'd col+1 and r before the
  swap; after the swap, `col+1` holds what was at r and is correct,
  and `col+0 = k+c` was never touched).

Wait — that's wrong for the `gamma0` re-scan if scalar runs after.
Scalar's first action on entering at column `k+c` is to re-scan
`gamma0` over `a[col*nrow + (col+1)..nrow]`. Column `k+c` is the
SAME col we were processing in the panel — its trailing entries
have NOT been peek-ahead'd (we peek-ahead'd only `col+1` and `r`
before swap). So scalar's gamma0 scan reads stale (un-rank-c-updated)
values.

**Fix.** Before bailing post-swap, peek-ahead `k+c` itself. Better:
peek-ahead `k+c` FIRST (this is what Phase A's no-swap path does for
free, because the loop top calls `peek_ahead_column(a, nrow, k, c,
...)` which peeks ahead `k+c` automatically). Confirm that the
swap-2×2 branch enters AFTER this peek-ahead. Reading
`src/dense/factor.rs:1367`:
```rust
peek_ahead_column(a, nrow, k, c, d_panel, subdiag);
```
Yes — this is called at the top of every iteration of the c-loop,
BEFORE the gamma0 scan and the swap branch. So `k+c` is always
peek-ahead'd. Only `k+c+1` and `r` need additional explicit
peek-aheads. ✓

So on bail-after-swap, `j_start = k + n_elim + 1` (the swapped
col+1's first un-peek-ahead'd column is k+n_elim+2; col+1 itself was
peek-ahead'd before we tested LAPACK-1×1 and growth/det). This is
exactly the `ScalarFallbackPeekedNext` semantics already in the
caller. ✓

## 9. Bound-only bail conditions remain

The existing `c + 1 < cap` and `col + 1 < ncol` checks in
`lblt_panel_frontal:1420` still apply unchanged. Even with swap-2×2
inline, the panel must have:
- Two `d_panel` slots (`c + 1 < cap`).
- Two columns inside the fully-summed range (`col + 1 < ncol`).
- An eligible swap target (`r < nrow`, always true since `r` was
  found in the gamma0 search over `(col+1)..nrow`).

These bound bails will be re-tagged as `FALLBACK_2X2_NEED_SWAP_OR_BOUND`
(the existing counter), separating them from the now-inline swap-2×2
case which gets a new `INLINE_2X2_SWAP_OK` counter.

## 10. What can go wrong

| failure mode | detection |
|---|---|
| Forgot to peek-ahead col+1 BEFORE swap | gamma_r scan reads stale a[r*nrow+i]; arr ≠ scalar's arr; bit-mismatch on first test_swap_2x2_inside_panel_bare |
| Forgot to peek-ahead r BEFORE swap | a[r*nrow+r] = stale; same symptom |
| swap_rows_cols called with col+1 > r | function asserts/swaps p<q internally — should be safe, but always pass `col+1` as p |
| perm not threaded through panel | compile error (caller passes; panel needs &mut [usize]) |
| Phase A2 and Phase A 2×2 share the same accept block (good); a bug in either breaks both | new tests exercise both paths |
| Accept then immediately followed by another swap-2×2 (chain) | test_swap_2x2_chain catches; no fundamental issue because each c iteration peek-ahead's k+c at the top |

## 11. Performance estimate

Probe-attributed swap-bail rate: 19/23 panel bails on the 9-matrix
mix. Each bail triggers `scalar_pivot_step` for one or two pivots,
then per-trailing-column rank-1 sweeps until the next panel restart.

Per the attribution note: `do_1x1_update` 3.6% wall + `do_2x2_update`
1.0% wall = 4.6% wall in scalar fallback. If swap-2×2 inline absorbs
83% of bails, expect ~80% of that 4.6% to migrate into the panel's
deferred path, where `apply_blocked_schur` runs the same arithmetic
but with rank-bs accumulation amortizing the per-column setup cost
(W-2 1×1 measured 3-6× speedup vs scalar; 2×2 should be similar).

Net wall savings estimate: 2-3% on aggregate, with concentrated
gains on ACOPR30 (~10-20% individually) and modest gains on
CRESC100/CRESC132/VESUVIO. Verified by acceptance criterion in the
plan.

## 12. References

- Bunch, J. R., & Kaufman, L. (1977). Some stable methods for
  calculating inertia and solving symmetric linear systems.
  Math. Comp., 31(137), 163-179. — the underlying pivot algorithm.
- LAPACK `dsytrf` (BSD) — reference implementation that does the
  swap eagerly in the unblocked path and amortizes via a delayed
  panel in the blocked path. The deferred-Schur structure here
  mirrors LAPACK's `dlasyf_rook` panel layout (rows committed,
  trailing block updated lazily).
- faer `bunch_kaufman/factor.rs:491` (`lblt_blocked_step`) —
  reference Rust implementation in a similar shape, but with FMA.
