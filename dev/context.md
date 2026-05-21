# FERAL Context (auto-generated)

Generated: 2026-05-21T21:18:19Z

## Latest Session
File: dev/sessions/2026-05-21-02.md
```
# Session 2026-05-21-02

## Goal

Track A2 — implement **Fix 1: fine-grained delayed pivoting
(swap-to-boundary)** for the `pinene_3200` interior-point KKT
delayed-pivot cascade (issue #46 family). Replace the Bunch-Kaufman
driver loops' break-on-first-delay behaviour — which forfeits the
entire remaining supernode tail on the first delayed pivot — with
swap-to-boundary, so a delay forfeits exactly one column.
Correctness-critical (touches the pivot/inertia path); FERAL
lifecycle: research (done last session) → code inspection → plan →
tests-first → implement → benchmark.

## Accomplished

### Fix 1 — fine-grained delayed pivoting (swap-to-boundary) — DONE

- **Code inspection** of both BK driver functions in
  `src/dense/factor.rs`. Five de-risking findings (journal §15:35):
  `PivotOutcome::Delayed` leaves the front clean; `swap_rows_cols`
  is the existing symmetric-swap helper; `factorize.rs:2267` already
  consumes a permuted contribution block via `ff.perm`; swaps stay
  within `[0, ncol)` so `perm[ncol..nrow)` stays identity;
  termination is guaranteed (`ncol_eff - k` strictly decreases).
- **Plan** written: `dev/plans/kkt-cascade-fix1-fine-grained-delay.md`.
- **Tests first** — `tests/fine_grained_delay.rs`, two tests, oracle
  = Bunch & Kaufman 1977 pivot admissibility (fixtures built so
  exactly one column is provably stuck, every other provably
  pivotable). Both **failed before the fix** (`nelim: left 1,
  right 3` / `left 1, right 11`) — break-on-first forfeits the
  pivotable tail. Tests-first protocol satisfied.
- **Implementation** — added the `delay_swap_to_boundary` helper
  (`factor.rs:3977`) and converted all four delay sites: plain
  driver `factor_frontal_in_place_with_scratch_impl` (1 site), panel
  driver (scalar tail, scalar fallback, `PanelStatus::Delayed`).
  Each loop now carries `ncol_eff` (initially `= ncol`); a `Delayed`
  return calls `delay_swap_to_boundary` and drops the stale
  `cached_maxfromm`. Post-loop `nelim = k; n_delayed = ncol - nelim`
  unchanged (`nelim == ncol_eff` at exit). Byte-identical on any
  matrix with no delays; `may_delay == false` root supernode
  unchanged.
- **Validation** — full `cargo test` green (302 lib + all
  integration suites, 0 failed); `cargo clippy --all-targets -D
  warnings` clean. New tests pass; `delayed_pivoting` (6) and
  `issue_46_saddle_kkt_cascade` (1) regression guards stay green.

### Cascade broken on pinene_3200

`probe_issue46_supernode pinene_3200_0009.mtx` (n=127995, 63995 MC64
```

## Git Status
```
42434a5 fix(dense): fine-grained delayed pivoting kills the BK cascade amplifier (#46)
ef5fb7e docs(session): checkpoint 2026-05-21-01 — A2 amplifier diagnosis
70f2e44 diag(dense): localize pinene KKT cascade — amplifier × two triggers
d3d93d2 docs(trackA): localize pinene cascade to the 2x2 stability gate
76174bd docs(session): checkpoint 2026-05-21-01 — B2 landed, pivot to Track A
```

## Test Status
```
