# Plan: issue #224 — recover speed after #221 without touching correctness

Research: `dev/research/issue-224-post-221-performance.md`. Read it first;
the numbers below refer to its tables. Scratchpad root (dumps, harness,
results): `/private/tmp/claude-501/-Users-jkitchin-projects-feral/a1458847-2377-4832-a0c1-096910fb944e/scratchpad/i224/`
(call it `$I`).

Hard constraints (do not relitigate): inertia exact on non-singular
matrices; blocked/scalar bit-parity (`tests/blocked_ldlt.rs`,
`tests/golden_bits.rs`, `tests/issue_221_forced_front_parity.rs`); no
tolerance loosening; no reintroduction of anything in CHANGELOG 0.21.0 /
decisions 2026-10-19..21 / tried-and-rejected 2026-10-20..21 (in particular
no lower delay-chain backstop K, no 1e-10 CB floor, no APP panel).

Measurement rules: fixed matrices before end-to-end; sequential runs; each
timing claim from the harness `$I/ab/target/release/ab224` (`018|021`,
`--reuse`, `--serial`, `--seed S`) or `diag_221_rook_cost`; pounce runs with
a 300 s limit. For a local feral under test, point a copy of the harness at
`feral021 = { package = "feral", path = "/Users/jkitchin/projects/feral" }`.

## Step 0 — close the two open measurements (no code change)

0a. Scaled eigvalsh verdict for brainpc0 i95/s1 and brainpc1 i41/s1: read
    `$I/res/eig_scaled.txt` (produced by `$I/run_eig2.sh`; rerun
    `python3 -I $I/eig224s.py <file>` if absent, ~2–3 min each). Oracle:
    Sylvester (D·A·D has A's inertia). Record in the research note §3.3.
    Expected: 0.21's count (6901 on brainpc0) agrees; if the scaled oracle
    says 6902 on brainpc0, stop and report — that would be a 0.21 defect.
0b. The qcqp integration gap (§3.4). Run, sequentially:
    - `ab224 021 --reuse --serial $I/dump/f021-qcqp1000-2c/iter_*/kkt_solve_001.jsonl`
      and compare with `$I/res/qcqp.f021-qcqp1000-2c.b021.txt` (56.3 s).
    - `~/projects/pounce/target/release/pounce qcqp1000-2c.nl --no-sol
      solve_quality_audit=no print_timing_statistics=yes --json-output q.json`
      and read `LinearSolverSummary.n_factors` / `total_factor_secs`
      (`crates/pounce-linsol/src/summary.rs`).
    Decision: if `total_factor_secs` ≈ 56 s while `LinearSystemFactorization`
    ≈ 100 s, the 44 s is pounce's timer scope → report to pounce, feral steps
    1–3 still apply to the 56 s. If `total_factor_secs` ≈ 100 s and the
    `--serial` harness ≈ 100 s, the gap is thread-pool dispatch → Step 4.

## Step 1 — row-max caching in the rook rescue (bit-parity-safe)

Files: `src/dense/rook.rs` (`rook_rescue`, `sym_row_argmax`),
`src/dense/factor.rs` (`scalar_pivot_step`, `symmetric_row_offdiag_max`,
`try_reject_1x1_with_rook_rescue`).

What: within one `scalar_pivot_step` call, every row-max/argmax computed
for row `r` (`gamma_r` in the BK test, then rows visited by the rook walk)
is exact integer/max arithmetic on unchanged data, so memoising
`(gamma, argmax)` per row for the duration of the call cannot change any
value. Pass the already-computed `(gamma_r, r)` of the BK test into
`rook_rescue` (its first iteration recomputes exactly that row), and keep a
small per-call cache keyed by row for the ≤ `MAX_ROOK_ITER` rows the walk
visits (rows repeat when the walk oscillates between two rows).

Tests first (oracle: the existing kernel): a unit test in `rook.rs` that
runs `rook_rescue` old vs new on 200 random symmetric fronts (sizes
8–96, with ghost rows) and asserts identical `RookPivot` (kind, swaps);
the parity suites stay the oracle for the full factor. Nothing numeric
changes, so `tests/golden_bits.rs` must pass byte-identical.

Measure: `ab224`-style A/B on `$I/dump/f021-qcqp1000-2c/iter_120/kkt_solve_001.jsonl`
and `iter_083`, 5 repetitions each, plus `$I/dump/f018-brainpc0/iter_095/kkt_solve_001.jsonl`
(rook 1152 under 0.18, 7 under 0.21 — the rook count on 0.21 is small, so
expect little here unless `FERAL_221_PANEL_DIAG=1` shows rook scans in the
scalar steps dominating).

Accept: ≥ 5 % median factor-time reduction on the two qcqp matrices with
identical inertia, nnz_l, n_tiny, tfa, cb and identical golden digests;
kill: < 2 % or any parity test change.

## Step 2 — cheap flush when a panel bails with n_elim ≤ 1

Files: `src/dense/factor.rs` (`apply_blocked_schur`, `apply_blocked_schur_panel`,
the bail arms after `lblt_panel_frontal` in
`factor_frontal_blocked_in_place_with_scratch`).

What: on `ScalarFallback`/`Delayed` with `n_elim == 1` (and `n_elim == 2`
when the pair is two 1×1s), apply the pending rank-1 update(s) to the
trailing columns with a direct strided loop that evaluates
`a[i,j] -= (d·L_i)·L_j` in the same association and FMA setting the packed
tile kernel uses (read `schur_kernel::packed_schur_tiles_{fma,nofma}` for
the exact expression and order; the 2026-10-20 lazy-BK journal entry
confirms the per-pivot strided loop was bit-identical). No packing, no
tile dispatch.

Tests first (oracle: the present kernel): `tests/blocked_ldlt.rs` and
`tests/golden_bits.rs` already pin the blocked kernel's bits; add a test
that drives `factor_frontal_blocked_in_place_with_scratch` on fronts built
to bail after 0, 1 and 2 pivots (a tiny diagonal at column 1 or 2 with
`may_delay = true`) and compares `L`, `D`, `perm`, contrib byte-for-byte
with `factor_frontal` (the scalar reference). Kill criterion: any ulp
difference — then the strided loop is not reproducing the tile order and
the step is abandoned, not tuned.

Measure: `FERAL_221_PANEL_DIAG=1 FERAL_221_PHASES=1 target/release/diag_221_rook_cost
$I/dump/f021-qcqp1000-2c/iter_120/kkt_solve_001.jsonl` before/after:
`panel_partial`, `pivots_inline`, `SCHUR_NS`; then the `ab224` A/B on
iters 083, 120, 135 and on `$I/dump/f021-qcqp1000-2c/iter_000` (no delays:
must not regress).

Accept: ≥ 10 % factor-time reduction on the hard matrices, ≤ 1 % change on
iter_000, bit-identical outputs. Kill: < 3 %.

## Step 3 — refined-solve cost on floored-pivot factors (measure, maybe act)

0.21's refined solves cost 3× 0.18's on the 0.21 qcqp sequence (3.69 s vs
1.21 s), concentrated on matrices with `tiny` 150–700. Instrument with
`solve_refined_opts` + `RefinementDiagnostics` (steps taken, backward error
per step) on iters 083/120/135. If the extra cost is refinement steps that
do not improve the backward error (stall), the existing step/stall logic in
`src/numeric/solve.rs` is the place; oracle for any change is the backward
error after the solve (must not increase) on the 264 dumped qcqp matrices.
Kill: if the steps are all productive, leave it; the floored pivots are
the exact factor's price.

## Step 4 — only if Step 0b says the gap is dispatch

If pounce's `total_factor_secs` ≈ `--serial` harness time ≫ parallel
harness time, feral is running serially (or over-subscribed) inside
pounce. Check `Solver::with_parallel` resolution and `min_parallel_flops`
under pounce's process (pounce-feral passes `FeralConfig.parallel = None`
→ feral default from `available_parallelism`). This is a configuration
finding to hand to pounce, not a kernel change; do not change feral
defaults for it.

## Acceptance for the whole plan

- `cargo test --release` green; `pre-commit run --all-files` clean.
- kkt-pounce corpus (370): inertia, n_tiny, nnz_l identical to 0.21.0.
- column25 guard (if the fixtures are present): 75597 on all 28, tiny 0.
- Fixed-matrix: `$I/res/qcqp.f021-qcqp1000-2c.b021.txt` re-run with the
  local feral: inertia identical on 164/164, factor total ≤ 50 s (from
  56.3 s) for Steps 1+2 together.
- End-to-end (sequential, 300 s): qcqp1000-2c `LinearSystemFactorization`
  below 100.5 s with the same 163-iteration path (compare the iteration log
  with `$I/itdiff.py`; a different path means the change was not
  bit-identical — kill). brainpc0/1: no change expected from feral; report
  the 0b/0a findings to pounce (#224 thread) as the trajectory owner.

## Journal / checkpoint

Append to `dev/journal/2026-10-22-01.org` as you go; checkpoint per
CLAUDE.md. Record any abandoned step in `dev/tried-and-rejected.md` with
the measured numbers.
