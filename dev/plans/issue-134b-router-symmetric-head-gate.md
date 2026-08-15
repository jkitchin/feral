# Plan — issue #134 item B: symmetric head gate in `pick_scaling_strategy`

**Research:** `dev/research/router-permutation-invariance-2026-08-15.md`
**Scope:** gate (b) only. Gate (a) deferred with evidence (89-family
MC64 loss, needs IPM-outcome data).

## Change

In `src/scaling/mod.rs::pick_scaling_strategy`, compute gate (b) —
`max_col_nnz > MAX_COL_NNZ_FOR_INFNORM` — on the **symmetric** degree
of each column rather than its stored lower-triangle nnz. Gate (a) is
untouched.

Symmetric degree of `j` = number of value-nonzero entries in row and
column `j` of the full matrix, diagonal counted once. Computed by
accumulating each stored off-diagonal `(i, j)` into both `deg[i]` and
`deg[j]`, and each diagonal into `deg[j]` alone.

Both the value-aware skip of explicit stored `0.0` (issue #47) and the
`n == 0` early return are preserved.

## Tests (written first)

1. `route_is_invariant_under_index_reversal` — build the existing
   arrow-KKT fixture, apply `P(i) = n-1-i` (remapping `(i,j)` with
   `i>=j` to `(n-1-j, n-1-i)` to stay lower-triangular), assert both
   forms route to `Mc64Symmetric`. **Fails before the change**: the
   reversed form routes `InfNorm` today.
2. `trailing_border_routes_to_mc64` — arrow KKT with the head in the
   *last* column (duals-last / IPOPT convention), assert
   `Mc64Symmetric`. Fails before the change.
3. `banded_stays_infnorm` — clnlbeam-shaped banded KKT, high
   `diag_only` ratio, small degree; assert `InfNorm` under both
   orderings. Guards the gate that separates clnlbeam from VESUVIO.
4. `explicit_zeros_do_not_change_route` — re-assert issue #47 against
   the symmetric counter: padding a column with stored `0.0` must not
   flip the route in either ordering.
5. Existing router tests must pass unchanged.

## Validation

- `cargo test` full suite.
- Re-run `probe_scaling_levers table` and confirm the shipped router
  now reports 890/1004 invariant and 15 gains / 0 losses.
- `cargo run --bin bench --release` — the 15 movers are small; expect
  no partition movement.

## Docs

- Update the `pick_scaling_strategy` doc comment: gate (b) is now
  symmetric degree; the "No allocations" claim is no longer true (one
  `n`-length `usize` accumulator); note that the 2026-05-17 panel is
  preserved because clnlbeam (sym 5) and ACOPP30 (sym 29) stay under 32.
- `CHANGELOG.md` Unreleased — user-visible routing change.
- Leave #134 item B open, retitled to the gate (a) remainder.
