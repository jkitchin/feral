# Plan: TPP tail for stuck columns, plus a chain-cost backstop (issue #221)

## Problem

Measured on pounce's dumps (journal 2026-10-19-01, 2026-10-20 entries):

1. Long delay chains. The #215 budget bounds one front's size, not the
   cost of carrying the same delays up a chain of fronts:
   - qcqp1000-2c iter 85: chain ratio 224, 4.2 s, 3259 floored pivots,
     wrong inertia;
   - nql180 after quality escalation: 49 s vs 5.7 s on 0.18.
2. Slow forced fronts. In cascade-break and root fronts
   (`may_delay = false`), every threshold-failing pivot leaves the
   blocked panel (`factor.rs`, the issue #117 bail). It then pays a rook
   scan and a full unblocked rank-1 update of the trailing matrix,
   including the contribution block. With thousands of failing columns
   the front runs at level-2 speed (`axpy_minus_unroll4_nofma` 5247
   samples, rook 1708, blocked Schur 932 on qcqp1000-2c iter 85 under the
   backstop with a 1e-14 floor).
3. Floor. Backstopped CB fronts under the #213 floor (`cascade_break_eps`
   1e-10) gave wrong inertia (qc85 5103, q15c76 10500). A round-off floor
   (1e-14) gave exact counts but was slow, because of (2).

## Design (SSIDS structure: APP pass, then TPP on the leftovers)

SSIDS factors a node with a blocked a-posteriori pass and then runs
`ldlt_tpp_factor` (`ldlt_tpp.cxx`) on the columns that failed. The TPP
search:
- picks the 2×2 partner from uneliminated fully-summed columns only
  (:207);
- tests with full-column maxima (:75-86, :115-118, :225-226).
Every quantity it reads lives in fully-summed columns, which stay
current if updates to them are applied eagerly. The contribution-block
(CB) update can be deferred and applied blocked.

feral:

1. Blocked pass, unchanged, except that the front always runs it with
   delays enabled. A failing column is swapped to the end of the
   fully-summed range (`delay_swap_to_boundary`), which is cheap.
2. TPP tail on `[nelim, ncol)`:
   - For each remaining candidate `p`:
     - try a 2×2 with `t = argmax_{fully summed, uneliminated} |a(p,t)|`
       (Duff-Reid growth bound and SSIDS det floor, via
       `rook::passes_2x2_gates`);
     - otherwise try a 1×1 with `|a_pp| >= u · colmax(p)`.
     The first that passes is swapped into position and eliminated.
   - Updates to the remaining fully-summed columns are eager (full
     height). Updates to CB columns `[ncol, nrow)` are deferred and
     applied with `apply_blocked_schur` in chunks of at most `bs` pivots.
     A chunk never splits a 2×2.
   - If no candidate passes:
     - `may_delay`: stop, and the rest delays as now;
     - otherwise (CB or root): force-accept the least-bad candidate (max
       `|a_pp| / colmax`) through `try_reject_1x1_frontal`, which keeps
       the floor and counter semantics.
3. Chain-cost backstop. Each `ContribBlock` carries `sub_sym_area` and
   `chain_excess`. A front engages cascade-break when
   `chain_excess > max(DELAY_CHAIN_FLOOR, DELAY_CHAIN_GROWTH ·
   sub_sym_area)`, with K = 10 and floor 2^20. Calibration: column25
   guard 3.7, brainpc3 ≤ 2.8; runaways 36-224.
4. CB floor. Decide after (2) lands, from measurements. Candidates: keep
   1e-10, or a round-off floor.

## Tests first (external oracles)

- Frontal, `may_delay = true`: a front whose columns fail in order, but
  that pivots completely once a later fully-summed partner is
  considered. Oracle: inertia by construction (`A = Q·D·Qᵀ`, Sylvester),
  and the LDLᵀ + contribution reconstruction identity.
- Frontal, `may_delay = false`, many failing columns: nothing delayed,
  inertia by construction, reconstruction identity.
- Sparse: the brainpc-like and saddle KKTs (saddle-point theorem), the
  issue-215 and issue-221 tests, and the column25 guard set (Sylvester
  across congruence seeds).

## Acceptance

- Inertia:
  - column25 guard: 28/28 = 75597 with tiny 0;
  - kkt-pounce 370: identical inertia;
  - qc85 = 5107 and q15c76 = 10508 (expected counts).
- Time: qc85 and q15c76 at or below 0.18 (2.25 s / 8.4 s); nql180 at
  u = 0.01 well below 49 s.
- End to end, pounce 68e438d0 `solve_quality_audit=no`: the five models
  from the PR comment solve.
- `cargo test --release` green; golden-bit changes justified one by one.
