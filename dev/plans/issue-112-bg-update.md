# Plan: issue #112 — Bartels–Golub pivot-searching rescue for the FT update

Research: `dev/research/issue-112-bg-update.md`. Branch: `claude/issue-112-0885lr`.

## Scope

Sparse path only (`src/lu/sparse_update.rs`, `sparse_factor.rs`). The dense
update is out of scope (issue targets `update_sparse`); its behavior is
unchanged.

## Steps (tests first)

1. **Tests** (`tests/issue112_bg_update.rs`), written before the
   implementation, red on the current code:
   - `ft_fixed_order_cancels_to_exact_zero_on_nonsingular_basis`: hand-built
     m≈40 basis (construction in research note §4.4: upper-triangular base,
     ±1 cascade doubling the working row to 2³⁴, seed diagonal `8 + 2⁻²⁰`).
     With `update_pivot_search: false` the update must return
     `NeedsRefactor` with `last_refactor() == Some((TinyPivot, 0.0))`, while
     the replacement basis is verifiably nonsingular (fresh `SparseLu::factor`
     on B' succeeds; dense-LU oracle solves it).
   - `bg_rescue_commits_where_ft_cancels`: same basis, param on (default):
     update returns `Ok`, `pivot_search_rescues() == 1`, and for a spread of
     RHS the update-path solution residual `‖B'x − a‖∞` is ≤ the from-scratch
     refactor's residual on the same RHS (acceptance criterion from the
     issue), with ftran AND btran checked.
   - `bg_rescue_preserves_invariants_and_chains`: after the rescue, run
     further updates/solves; check uperm bijection, diagonal-first, rank
     triangularity (mirror of `uperm_triangular_invariant_holds_after_wide_bump_chain`).
   - `bg_swap_eta_roundtrip`: small (m=3..6) matrices where the rescue path is
     forced; ftran/btran against dense-LU oracle to pin `FtOp::Swap` replay
     (forward, transpose, and `compute_spike`) exactly.
   - Param off ⇒ bit-identical legacy behavior (existing suite already covers
     the legacy path; add an explicit `update_pivot_search: false` assertion in
     the regression test).
2. **`FtOp::Swap`** in `sparse_factor.rs`: variant + `apply_forward`
   (`y.swap`), `apply_transpose` (same swap; reversed walk already exists).
3. **`LuParams::update_pivot_search: bool`** (default `true`) + doc fix for
   the stale "always uses strict partial pivoting" sentence on
   `pivot_threshold`; `pivot_search_rescues: usize` counter + accessor on
   `SparseLu` (reset by factor/refactor).
4. **`eliminate_pivot_row(…, allow_swaps: bool, saved, changed_sorted)`**:
   - dedicated touched-mark pooled buffer (`ft_rw_mark: Vec<bool>`) so
     `rw_touched` is duplicate-free; drop the final-gather `dedup_by_key`.
   - swap branch (allow_swaps && `|piv| < |vrc|`): snapshot old row `c` into
     `saved` (dedup vs sorted prefix), gather `rw` into the new `u_rows[c]`
     (diagonal `(c, vrc)` first, tail column-sorted), push
     `Swap{a:c,b:r}` + `Axpy{target:r,src:c,mult:piv/vrc}`, transform
     `rw ← old_c − mult·rw` in place (scale gathered support by `−mult`,
     `rw[c] = 0` exactly, scatter old row `c` skipping its diagonal),
     return swapped positions alongside ops.
5. **`update_sparse` orchestration**: capture `diag0 = w[r]` and
   `last_refactor` before elimination; pass 1 FT (allow_swaps=false); on
   `TinyPivot` + param on: restore `u_rows[r]` from the snapshot, re-run with
   allow_swaps=true; on rescue success restore prior `last_refactor`, bump the
   rescue counter; extend `changed` with swapped rows before the growth scan;
   commit path rebuilds `u_above` wholesale when swaps occurred (helper
   `rebuild_u_above`), incremental otherwise.
6. **`compute_spike`**: replay `Swap` (swap + mark/touch both positions).
7. Docs: module header (`sparse_update.rs`), CHANGELOG Unreleased entry.
8. Full suite + clippy + fmt; run `lu`-relevant benches; session-end protocol.

## Risks / checks

- Swap must not re-enqueue the swap column (exact-zero clearing, same
  landmine as 2026-06-21 bring-up) — covered by the chain test looping.
- `last_refactor` must stay `None`-equivalent after a successful rescue
  (existing `should_refactor_growth_preempts_trip` asserts `is_none()`).
- Rollback correctness when the BG pass itself fails (growth trip or still
  tiny): all swapped rows restored — covered by asserting the factors solve
  identically to pre-update after a forced BG failure (growth cap 1.0… or
  reuse existing rollback tests with param on).
- Oracle discipline: the regression matrix is a hand calculation (documented
  in-test), the residual oracle is the pre-existing refactor path and the
  dense LU — no same-session self-oracle.
