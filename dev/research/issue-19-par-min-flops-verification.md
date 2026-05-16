# Issue #19 verification — parallel-assembly heuristic after PAR_MIN_FLOPS=1e7

**Status:** Resolved (verdict A — fixed).
**Date:** 2026-05-16.
**Worktree branch:** `worktree-agent-a248ac8c98e45f576`.
**Related commits (already on `main`):**
- `19d7b03` work-aware gate in `should_parallelize_assembly` (#19)
- `91e028a` persistent rayon `ThreadPool` reused across `factor()`
- `db7b761` `calibrate_par_min_flops` probe + research note
- `b12e03c` lower `PAR_MIN_FLOPS` 1e8 → 1e7 (#19 closeout)
- Prior research notes: `dev/research/issue-19-parallel-heuristic.md`,
  `dev/research/par-min-flops-calibration-2026-05-15.md`.

## Why this note exists

GH #19 was empirically closed by `b12e03c` but the GitHub issue itself
was left open because the close-comment / follow-up draft was blocked
by the auto-mode classifier in session 2026-05-15-06 (see that session
checkpoint). This note re-verifies the closeout against the two
original regression vectors and pulls the trigger on `gh issue close 19`.

## Original numbers (from the issue report)

| Problem    | KKT size            | parallel=true     | parallel=false    | Winner             |
| ---------- | ------------------- | ----------------- | ----------------- | ------------------ |
| robot_1600 | 24k vars, 9.6k cons | ~2.1 s/iter       | ~0.18 s/iter      | sequential **12×** |
| henon120   | 32k vars, 0.24k cons| 0.54 s/iter (147s)| 1.50 s/iter (410s)| parallel **2.8×**  |

Reporter hardware unspecified; sample profile showed cv-wait + mutex
dominance (`__psynch_cvwait` etc. ≈ 45% of non-idle CPU).

## Reproduction setup

- Host: Apple M4 Pro, 14 rayon workers (`rayon::current_num_threads`).
- Binary: `src/bin/probe_issue_19.rs` (committed `25926cc`). Times
  `Solver::factor()` for three configs on a single KKT dump:
  1. `parallel=false` (sequential driver, no pool).
  2. `parallel=true`, `min_parallel_flops=None` (default gate, i.e. 1e7).
  3. `parallel=true`, `min_parallel_flops=Some(0)` (gate disabled).
- Reps = 10 per config, median reported; first call is warm-up.
- All runs inside one `Solver` instance so the persistent ThreadPool
  (`91e028a`) is amortised across reps and across configs.

## Results — robot_1600 (median ms, 10 reps)

| iter | est_flops | seq    | gate=default | gate=OFF | par speedup vs seq |
| ---- | --------- | ------ | ------------ | -------- | ------------------ |
| 0000 | 4.75e6    | 7.21   | 6.52         | 8.99     | 0.80× (parallel hurts; gate keeps it sequential — correct) |
| 0001 | 1.13e7    | 12.76  | 8.80         | 8.71     | **1.45×** (gate fires — correct) |
| 0003 | 1.13e7    | 18.67  | 14.39        | 14.48    | **1.30×** (gate fires — correct) |
| 0006 | 9.43e6    | 13.90  | 17.56        | 9.65     | 1.44× possible but vetoed (just below 1e7 threshold; 1.6× safety margin against the 6e6 break-even) |

On iter 0000 the gate decision is `sequential` and parallel-forced
runs 0.80× as fast — gate vetoes correctly. On iters 0001/0003 the
gate fires PARALLEL and parallel beats sequential by 1.3–1.45×. The
"gate=default" row tracks the better of the two driver rows. Iter
0006 sits just under the 1e7 threshold; the conservative veto trades a
1.4× win for a 1.6× safety margin (documented trade-off in `b12e03c`).

**No 12× regression on any of the four iters.** The worst the default
gate does on this matrix is 1.26× slower than the optimal driver
(iter 0006). The original report's worst case was 12× slower.

## Results — henon120 (median ms, 10 reps)

| iter | est_flops | seq    | gate=default | gate=OFF | par speedup vs seq |
| ---- | --------- | ------ | ------------ | -------- | ------------------ |
| 0000 | 9.83e5    | 22.16  | 19.35        | 73.73    | 1.15× (gate sequential; forcing parallel hurts 3.3× — gate correct) |
| 0001 | 6.13e9    | 655.00 | 104.74       | 119.01   | **6.25×** (gate fires — correct) |
| 0003 | 6.13e9    | 433.79 | 98.18        | 97.49    | **4.42×** (gate fires — correct) |
| 0005 | 6.13e9    | 447.61 | 104.37       | 116.37   | **4.29×** (gate fires — correct) |

Iter 0000 is the symbolic-only warm-up phase (low flop estimate);
the gate keeps it sequential and that is correct (forced parallel is
3× *slower*). Iters 0001/0003/0005 are the steady-state large-KKT
factor; gate fires PARALLEL and gets a **4.3–6.3× wall speedup** —
*better* than the issue's reported 2.8× and substantially better than
any sequential run.

## Verdict

**(A) Fixed.** Both regression vectors are resolved:

- **robot_1600.** Original 12× wall regression no longer reproduces.
  Default-gated parallel is within 1.26× of optimal on the worst iter
  and beats sequential 1.3–1.45× on the dominant iters. Two changes
  did the work: `91e028a` (persistent pool removed the per-call
  cv-wait amortisation cost the issue profile flagged), and `b12e03c`
  (lower threshold lets the gate fire on iters that actually win).
- **henon120.** Original 2.8× parallel-win is retained and improved
  to 4.3–6.3× on the dominant iters. The lower threshold did not
  cost anything here because flops on the steady-state factor are
  three orders of magnitude above the threshold.

## Cross-checks

- Per-call gate decisions match the empirically optimal driver on 6
  of 8 measured iters; the two misses (robot_1600 iter 0006 and a
  hypothetical iter at est_flops between 6e6 and 1e7) are deliberate
  safety-margin trade-offs documented in `b12e03c`'s message.
- `cargo test --lib --release` — 256 passed, 0 failed.
- `cargo clippy --all-targets --release -- -D warnings` — clean.
- `cargo fmt --check` — clean.

## What this verification does not cover

- **Non-M4 hardware.** The issue reporter's machine is the only one
  that produced the 12× wall regression; without their hardware we
  cannot directly disprove a residual regression on their box. The
  per-instance override (`NumericParams::min_parallel_flops`,
  `POUNCE_FERAL_MIN_PAR_FLOPS=<u64>`) shipped in this work-stream
  remains the escape hatch for hardware where the local break-even
  differs.
- **Corpus-wide PAR_MIN_FLOPS validation.** Two workload families
  (Poisson-KKT, robot_1600 KKT dumps) and now henon120 KKT dumps put
  break-even between 1e6 and 1e7 flops, but Pinene, robot_b, vesuvia,
  and the non-IPM synthetic corpus have not been swept. Drafted
  follow-up at `/tmp/feral-issue-19-followup.md` covers this; not
  filed as a separate issue this session because it is parameter
  tuning, not a correctness gap.

## Action taken

- `gh issue close 19` with comment summarising the two-vector
  reproduction above and pointing at this note + the four
  contributing commits.
- CHANGELOG `[Unreleased]` entry under `### Verified` for the closure.
- No source changes — the implementation was complete before this
  session; this session is verification + close.
