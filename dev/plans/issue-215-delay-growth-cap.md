# Plan: issue #215, replace the K·own_ncol delay cap with a front-growth cap

Research: `dev/research/issue-215-delay-budget-exact-inertia.md`.
Approach approved by the user (2026-10-10): growth cap plus reporting.

## Change

`symbolic/supernode.rs::assign_delayed_capacities`:

    worst = subtree_ncol(s) − own_ncol(s)        (unchanged, hard bound)
    grow  = max(DELAY_GROWTH_ROWS · nrow(s),     (front rows ≤ (1+3)·nrow,
                DELAY_FRONT_ROW_FLOOR − nrow(s))  i.e. area ≤ 16× symbolic,
                                                  or front ≤ 4096 rows)
    delayed_capacity(s) = min(worst, grow)

Constants: `DELAY_GROWTH_ROWS = 3` and `DELAY_FRONT_ROW_FLOOR = 4096`. The
floor allows a dense front of 4096², about 134 MB. They replace
`DELAY_CAPACITY_MULTIPLIER` and `DELAY_CAPACITY_MIN_FLOOR`, which are
`pub`, so the change goes in CHANGELOG as breaking (next release is
already 0.19.0).

The trigger, the CB path and `DelayBudgetExceeded` all stay. CB becomes
the memory backstop rather than the normal overflow path.

Calibration (unbounded traces): every delayed node in qcqp1500-1nc_0000,
column25 i011_s002 seeds 0–8 (300k nodes), and marine_1600
9/11/12/13/16 satisfies the new cap, with 0 violations. The largest
d/nrow is 2.87, on a marine front of fewer than 200 rows.

## Report

Add `FactorStats::inertia_uncertain() -> usize`, returning `n_tiny`: the
number of pivots whose sign came from a value at or below the
perturbation floor. Its doc says that nonzero means the inertia is not
guaranteed. Under the default parameters only the CB path floors (the
#38 static-pivot floor is opt-in), so this measures exactly the #215
failure mode. It is a helper over an existing counter, with no new
field.

## Tests first (they must fail on main)

1. Unit test in `supernode.rs`. Oracle: hand-computed capacities.
   - On a small tree, check the new formula, including the worst-bound
     clamp and the floor regime.
   - Check that a node with own_ncol 16 and nrow 256 gets
     min(worst, 3840), where it gets 64 on main.
2. `tests/issue_215_delay_growth_cap.rs`, data-gated, skips when the file
   is absent:
   - Fixture: column25 i011_s002 (`tests/data/large/issue215_column25_i011.mtx`).
   - Oracle: the reporter's `eigsh` count of 75,596 negative, plus
     Sylvester's law.
   - Seeds 1, 4 and 8 of the D·A·D congruence. On main these are +3, +1
     and +1 with cb > 0. With the fix they must be exact with
     `n_cascade_break_nodes == 0`.
   - This test is slow (~10 s per factor), so mark it `#[ignore]`, as
     the other heavy corpus tests are.
3. In the same file, qcqp1500-1nc_0000 from the data corpus, with the
   sidecar inertia as oracle. Check exact inertia with cb == 0
   (currently cb 16, n_tiny 2263).
4. The existing `issue_55_delay_budget` tests, which zero the capacity
   by hand, must still pass. They exercise the backstop path.

## Validation

- `cargo test --release`.
- Re-run the 613-matrix kkt-mittelmann default sweep. Expect the same
  inertia everywhere, with only the 6 currently-binding matrices
  changing (cb → 0).
- The `cargo run --bin bench --release` gate.
