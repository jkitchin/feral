# Plan: Phase A2 — W-2 swap-required 2×2 inline

Date: 2026-04-28
Status: PROPOSED — awaiting research note + tests-first kickoff
Owner: dense BK kernel
Successor to: Phase A (no-swap 2×2 inline) in `dev/plans/dense-kernel-blas3.md`
Inputs:
- `dev/research/dense-kernel-attribution-2026-04-28.md` (the lever rationale)
- `dev/journal/2026-04-28-01.org` 14:30 entry (probe output)
- `dev/research/dense-kernel-w2-2x2-and-blas3.md` (W-2 + B-1 design basis)
- `src/dense/factor.rs:1344-1652` (`lblt_panel_frontal`)
- `src/dense/factor.rs:2552-2585` (`swap_rows_cols`)

## 1. Why this plan exists

The 2026-04-28 panel-attribution probe (commit `97c16e5`) measured the
panel/scalar split and bail-reason histogram on a 9-matrix
representative mix. Result:

| metric                              | aggregate |
|-------------------------------------|----------:|
| pivots inline (panel)               |     86.1% |
| pivots scalar (post-bail + tail)    |     13.9% |
| panels with `n_elim < bs` (partial) |     23/39 |
| **bails: swap-required 2×2**        | **82.6%** |
| bails: swap-1×1 wins                |      8.7% |
| bails: LAPACK-1×1 wins              |      8.7% |
| bails: growth/det fail              |      0.0% |

83% of every panel bail is the case Phase A explicitly deferred as
"Out of scope: 2×2 with symmetric swap". Each bail triggers
`scalar_pivot_step` followed by per-trailing-column rank-1 sweeps
through `do_1x1_update`/`do_2x2_update` until the next panel can
restart — which the profile attribution measured at **4.6% of solver
wall**.

By contrast, the next planned step (B-1 NR=4 widening) targets
`schur_panel_minus_nofma_strided_dual` which is 1.4% wall total;
best-case ROI 0.7% wall. Phase A2 is 6× the lever.

## 2. Scope

Extend `lblt_panel_frontal` to handle the swap-required 2×2 case
inline, preserving the deferred-Schur invariant, instead of bailing.

### What "swap-required 2×2" means

Inside the panel at column `col = k + c`, scalar BK identifies a 2×2
pivot when `akk < alpha_bk * gamma0`. The argmax row `r` is the
off-diagonal of largest magnitude in column `col` over rows
`(col+1)..nrow`. When `r == col + 1`, the 2×2 spans consecutive
columns — Phase A handles this inline. When `r > col + 1`, scalar
calls `swap_rows_cols(a, nrow, col + 1, r, perm)` to bring row `r`
into position `col + 1`, then proceeds with the consecutive 2×2.

The panel currently bails because:
1. `peek_ahead_replay` only brings ONE column (`col + 1`) into scalar
   state. The swap target `r` has not been peek-ahead'd, so reading
   `a[r*nrow + r]` returns a partially-updated value.
2. `swap_rows_cols` swaps trailing entries in cols `col+1..r-1` row
   `col+1` with row `r`. These entries have not had the deferred
   rank-`c` update applied, but neither have rows `col+1` and `r`
   themselves — the swap is consistent at the `A_t - L_p D_p L_p^T`
   level *iff* `swap_rows_cols` also swaps the `L_p` rows. It does:
   line 2582 swaps `a[j*n + p]` with `a[j*n + q]` for `j in 0..p`,
   which includes `j in 0..c`. The deferred update remains valid.

### The work

For Phase A2 acceptance the panel must, on detecting `r > col + 1`:

1. **Peek-ahead row r** in addition to `col + 1`. Need a new helper
   `peek_ahead_two_columns` (or call `peek_ahead_replay` twice, with
   `r_idx = col + 1` and again with `r_idx = r`). Both calls share
   the same committed pivots `0..c` and write disjoint columns.
2. **Apply `swap_rows_cols(a, nrow, col + 1, r, perm)`** with the
   panel's perm threaded through.
3. **Continue down the existing no-swap 2×2 path** — gamma_r, swap-1×1
   reject, LAPACK-1×1 reject, growth/det checks, accept and record
   `d11/d21/d22` in `d_panel` + `subdiag`.
4. **Mark `ScalarFallbackPeekedNext`-equivalent state**: caller's
   `j_start = k + n_elim + 2` is correct when n_elim ends mid-panel
   on `col, col+1` consecutive. After a successful swap-2×2, the
   panel ends at `c += 2`; if it bails AFTER the swap, both `col+1`
   and (formerly) `r` (now at `col+1`) have peek-ahead'd state, so
   the existing `ScalarFallbackPeekedNext` semantics still apply for
   the new col+1. **The swap is permanent** — `perm` carries the
   change forward; the caller does not need to undo.

### The deferred-Schur invariant after a swap-2×2

After Phase A2 commits a swap-2×2 at panel position c:
- Rows `col, col+1` have committed L columns (rows `col+2..nrow`
  scaled by `D_2x2^-1`).
- Trailing rows `col+2..nrow` of trailing cols `col+2..ncol` still
  carry the *un*-rank-(c+2)-updated values, but with rows/cols
  `col+1 ↔ r_old` permuted in the deferred matrix. The rank-(c+2)
  update applied later at flush time will use the NEW (post-swap)
  L_p rows, which are the old rows in their new positions. Result is
  bit-equivalent to scalar.

### Out of scope (Phase A2 does NOT cover)

- Swap-1×1 (8.7% of bails). Different pivot path: scalar runs
  `try_reject_1x1_with_rook_rescue` then a 1×1 at row `r`. The panel
  has no equivalent. Defer.
- LAPACK-extension 1×1 wins (8.7%). Same — different pivot path,
  different threshold, panel-level rook rescue not implemented.
- Growth/det rejection (0% in the probe). Stays bailed.
- Rook-rescue on 1×1 rejection. Orthogonal — not panel-side.
- Multi-swap chains (a swap-2×2 immediately after another swap-2×2
  inside the same panel). Should "just work" because each `c`
  iteration recomputes gamma0/r from scratch on freshly peek-ahead'd
  state, but the test plan must include this case.

## 3. File-level changes

| file | function | change |
|---|---|---|
| `src/dense/factor.rs` | `lblt_panel_frontal` | new `perm: &mut [usize]` parameter; on `r > col + 1` and inline-2×2 budget OK, peek-ahead both `col+1` and `r`, call `swap_rows_cols`, continue to existing 2×2 accept path; on bail-after-swap return `ScalarFallbackPeekedNext` (col+1 was peek-ahead'd, swap was committed) |
| `src/dense/factor.rs` | `factor_frontal_blocked_in_place` | thread `perm` slice into the panel call; replace existing `swap_rows_cols` calls in scalar fallback section unchanged |
| `src/dense/factor.rs` | `peek_ahead_replay` | no signature change; just called twice for swap-2×2 |
| `src/dense/factor.rs` | `panel_diag` module | add `INLINE_2X2_SWAP_OK` counter incremented on every successful swap-2×2; `FALLBACK_2X2_NEED_SWAP_OR_BOUND` keeps bound-only triggers (`c+1 >= cap`, `col+1 >= ncol`) |
| `tests/blocked_ldlt.rs` | new tests | `test_swap_2x2_inside_panel_bare` (single swap-2×2 at panel-internal column), `test_swap_2x2_chain` (two swap-2×2 in a row), `test_swap_2x2_then_1x1` (swap-2×2 followed by clean 1×1), parity matrix from ACOPR30 frontal sub-block |
| `src/bin/probe_panel_attribution.rs` | format str | print `swap_ok` count next to `swap` bails |

LoC estimate: ~80 production, ~120 tests.

## 4. Acceptance criteria

- All existing `tests/blocked_ldlt.rs` tests pass byte-identical
  (same scalar reference comparison). Phase A's no-swap 2×2 path is
  unchanged.
- New `test_swap_2x2_inside_panel_bare`: 4×4 panel pattern that forces
  one swap-2×2 (constructed from a hand-permuted SSIDS-paper fixture).
  Byte-identical scalar vs. blocked, perm matches.
- New `test_swap_2x2_chain`: 6×6 with two consecutive swap-2×2 pivots.
  Byte-identical, both perm entries correct.
- New `test_swap_2x2_then_1x1`: swap-2×2 followed by a 1×1 within the
  same panel (the 1×1 must see the swapped state).
- Probe re-run on the 9-matrix mix shows `swap` bail count drops by
  at least 75% (allowing for swap-2×2 cases the panel still bails on
  due to bound exhaust).
- `cargo run --release --bin bench_solver_corpus` shows a ≥1% wall
  improvement on ACOPR30, CRESC100, CRESC132, VESUVIO, no regression
  on VESUVIA / HS118 / BATCH (already 100% inline).
- Inertia gate stays clean — no new oracle disagreements against
  MUMPS / SPRAL / faer / Eigen.

## 5. Risks and mitigations

| risk | mitigation |
|---|---|
| Wrong bit pattern after peek-ahead'ing two columns | peek_ahead_replay is column-local and writes only to its own column; calling twice with disjoint targets is identical to two separate scalar peek-aheads. New `test_swap_2x2_chain` exercises this. |
| `swap_rows_cols` walks rows that the panel hasn't initialized properly | The function operates on the full lower triangle; it does not depend on rank-c update state. The L_p row swap (j in 0..p, hits j in 0..c) preserves the deferred update equivalence. |
| Bail AFTER a successful swap leaves the matrix mid-state | Bail can happen in the swap-1×1, LAPACK-1×1, or growth/det branches. After swap, `col+1` is peek-ahead'd and the swap is committed (perm + L_p row + trailing cols). Returning `ScalarFallbackPeekedNext` means caller starts at `k + n_elim + 1` (the swapped col+1), which now has scalar-correct state. ✓ |
| Forgotten gate: bail when `c == cap - 1` before doing the swap | Same as Phase A's `c + 1 < cap` precondition; reuse without change. |
| Test fixtures drift from real-world swap patterns | One of the new tests uses a 6×6 sub-block sampled from an ACOPR30 frontal where the probe measured 84.6% swap-2×2. |

## 6. Sequencing

1. **Research note** (~1 hour): write `dev/research/dense-kernel-w2-2x2-swap.md`.
   Trace one ACOPR30 swap-2×2 by hand: which row swaps, what
   `swap_rows_cols` mutates, why peek-ahead'ing two columns + the
   swap is bit-exact with scalar.
2. **Test fixtures first** (~1 hour): construct three 4×4–6×6 test
   matrices with known swap-2×2 patterns. Verify scalar reference
   produces the expected `(L, D, perm)`. Tests fail.
3. **Threading `perm` into the panel** (~30 min): no-op refactor. All
   existing tests pass.
4. **Implementation** (~2 hours): swap-2×2 branch in
   `lblt_panel_frontal`; new counter; printf in probe.
5. **Tests pass + parity benchmark** (~30 min).
6. **Re-profile + checkpoint** (~30 min).

Total estimate: 5–6 hours of focused work, fits in one session.

## 7. Out of scope across A2

- Phase B-1 NR=4 widening — explicitly deferred per the
  attribution research note. ROI flipped after the diagnostic.
- Swap-1×1 / LAPACK-1×1-wins inline. 17.4% combined of bails;
  separate, harder lever (panel rook-rescue).
- Workspace pre-sizing (15.3% allocator wall). Parallel track.
