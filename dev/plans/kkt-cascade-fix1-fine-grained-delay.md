# Plan — Fix 1: fine-grained delayed pivoting (swap-to-boundary)

**Status:** DONE — session 2026-05-21-02. Implemented (swap-to-boundary
in both BK driver loops, `delay_swap_to_boundary` helper); 2 new tests
in `tests/fine_grained_delay.rs` pass; full suite + clippy green;
`pinene_3200_0009` cascade broken — `n_delayed` 133648→11309, blowup
69×→1.51×, factor ~183 s→78 ms, inertia exact (64000,63995,0); bench
all four buckets PASS. Fix 2/3 not needed.
**Research note:** `dev/research/kkt-cascade-amplifier-2026-05-21.md`
**Track:** A2 of `dev/plans/per-factor-cost-cluster.md`
**Closes / advances:** #8 / #46 family (`pinene_3200` iter 6-9
factor-time explosion).

## Problem

The Bunch-Kaufman driver loops in `src/dense/factor.rs` do
`Delayed => break`: one delayed pivot forfeits the entire remaining
tail of the supernode (`n_delayed = ncol - nelim`). On `pinene_3200`
3936 scalar delay events become `n_delayed = 133648` (~34 columns
forfeited per event) → 69× fill blowup, 183 s factor. Config 2
(static pivoting, `n_delayed = 0`, healthy 1.25× factor) proves the
forfeited columns are pivotable — the forfeit throws away real work.

## Fix

Replace break-on-first-delay with **swap-to-boundary**: when the
pivot at column `k` delays, swap it with the last still-eligible
column (`ncol_eff - 1`), decrement `ncol_eff`, and keep eliminating
at `k`. Each stuck pivot forfeits exactly one column. Inertia-exact
by construction — this is real delayed pivoting (the stuck column is
promoted to the parent front and re-attempted with more context); no
force-accept, no perturbation.

### Mechanism (verified by code inspection — see journal 2026-05-21-02)

- At a `Delayed` return the front is clean: `PivotOutcome::Delayed`'s
  contract (`factor.rs:1115-1118`) is "kernel has not mutated any
  state for the failed pivot", so columns `[k..nrow)` are
  consistently updated through pivot `k-1`. A symmetric swap of
  column `k` with column `ncol_eff-1` (both in `[k,ncol_eff)`,
  un-eliminated) is valid.
- `swap_rows_cols(a, nrow, k, ncol_eff-1, &mut perm)` is the
  existing symmetric-swap helper; it updates `perm` and no-ops on
  `p == q`.
- The multifrontal driver already consumes a permuted contribution
  block: `factorize.rs:2267` builds contrib global indices as
  `row_indices[ff.perm[node_nelim + cj]]` — applies `ff.perm` over
  the whole contrib range. So permuting `perm[nelim..ncol)` is
  consumed correctly; delayed-column order within the block is
  irrelevant.
- swap-to-boundary only swaps within `[0,ncol)`, so `perm[ncol..nrow)`
  stays identity and contrib values stay consistent with indices.
- After the loop `nelim == ncol_eff`, so `n_delayed = ncol - nelim`
  is unchanged in form; the delayed columns occupy `[ncol_eff,ncol)`
  = the leading `n_delayed` positions of the contrib block, exactly
  the documented contract.
- Termination: each iteration advances `k` or decrements `ncol_eff`;
  `ncol_eff - k` strictly decreases → O(ncol), no infinite loop.

### Sites (both driver functions in `src/dense/factor.rs`)

1. **Plain driver** `factor_frontal_in_place_with_scratch_impl`,
   loop 1405-1425 — site 1423.
2. **Panel driver**, loop 1719-1846 — site 1751 (scalar tail),
   site 1841 (scalar fallback after panel), site 1844
   (`PanelStatus::Delayed`). For all three the front is clean at
   delay time (after `apply_blocked_schur` for the panel sites — see
   journal). `pinene` exercises only the scalar sites
   (`PANEL_DELAYED = 0`); the panel-delayed site is converted for
   consistency and covered by a panel-path test.

`may_delay == false` (root supernode) never returns `Delayed`, so
`ncol_eff` stays `== ncol` and behaviour is byte-identical there.

## Tests first (`tests/fine_grained_delay.rs`)

External oracle: fine-grained delayed pivoting forfeits exactly the
genuinely-stuck columns. The test matrices are built so exactly one
column is provably stuck (near-zero diagonal, only coupling is
out-of-front) and the rest are provably pivotable (positive
diagonal, no coupling). Break-on-first forfeits the pivotable tail;
swap-to-boundary does not.

- **T1 — plain driver.** 5×5, `ncol=4`: col 0/2/3 isolated SPD
  (diag 2/3/5), col 1 stuck (diag 1e-14, only coupling = trailing
  row 4). `factor_frontal(.., 4, true, ..)`. Assert `nelim == 3`,
  `n_delayed == 1`, `inertia == (3,0,0)`, `perm[3] == 1` (stuck
  column tracked to the boundary), contrib block holds the stuck
  column. Before fix: `nelim == 1`, `n_delayed == 3`.
- **T2 — panel driver.** ~12 columns, one early stuck column, the
  rest isolated SPD. `factor_frontal_blocked(.., ncol, true, ..)`
  routes through the panel (`ncol ≥ PANEL_MIN_NCOL = 8`). Assert
  `nelim == ncol-1`, `n_delayed == 1`.
- **Regression guards (existing suite).** `tests/delayed_pivoting.rs`
  stuck-pair tests are unchanged (both columns genuinely stuck —
  hand-traced). `tests/issue_46_saddle_kkt_cascade.rs` inertia must
  stay exact. Full `cargo test` must stay green — Fix 1 is
  byte-identical on any matrix with no delays.

## Benchmark / validation

- `probe_issue46_supernode pinene_3200_0009.mtx`: record `n_delayed`,
  `factor_nnz`, blowup, factor time, inertia before vs after. Target:
  `n_delayed` ≪ 133648, blowup ≪ 69×, inertia stays (64000,63995,0).
- `cargo run --bin bench --release`: no regression on the four
  exit-partition buckets.

## Out of scope

- Fix 2 (matching-aware growth-bound exemption) and Fix 3 (tighter
  co-location) — only if a residual cascade remains after Fix 1
  (research note §6).
