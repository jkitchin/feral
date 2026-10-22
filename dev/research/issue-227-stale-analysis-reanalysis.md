# Issue #227 — re-analysing a stale value-dependent analysis: what the replays say

Date: 2026-10-22. Follow-up to #224 (`dev/research/issue-224-post-221-performance.md`).
Companion plan: `dev/plans/issue-227-reanalysis-policy.md`. Scratch root
(harness, dumps, results):
`/private/tmp/claude-501/-Users-jkitchin-projects-feral/a1458847-2377-4832-a0c1-096910fb944e/scratchpad/i227/`
(call it `$J`; the #224 material is in the sibling `i224/`).

## 0. Summary

1. **The "stale init analysis" of #224 is not what the issue says it is.**
   pounce's LS-multiplier init matrix (70 % explicit zeros) makes the
   preprocess-`Auto` fill race choose `None` (pattern-only ordering, no
   matching). That analysis is the *fastest* one for the first 76 IPM KKTs
   (0.06 s vs 0.10 s per factor under the first-KKT analysis). The 44 s
   #224 gap comes from the **ordering escalation** (#102): at iter 76 the
   pivot growth passes 1e24, feral re-analyses with `LdltCompress` on that
   hard matrix, and that analysis (MC64 matching built on a 1e-12..1e12
   matrix) is the one that carries the 85 hard KKTs at 0.9 s each. The
   first-KKT `LdltCompress` analysis carries them at 0.47 s.
2. **Fresher is not better.** On qcqp1000-2c the quality of a `LdltCompress`
   analysis degrades monotonically with the iterate it is built on
   (iter_120 under the analysis built on iter 0/1/2/5/10/30/75: 20k / 23k
   / 24k / 34k / 41k / 39k / 36k delays). No analysis built on a late
   matrix recovers the early one. "Re-analyse on the current matrix when
   stale" therefore cannot recover qcqp1000-2c in the init-first call
   order; it can only choose between analyses that already exist.
3. **The flops trigger works as a detector but predicted flops cannot
   rank candidates.** Actual/predicted flops jumps from 1.25 to 2.7–6.9
   within three factors where the hard regime begins; but the cheapest
   analysis by predicted flops (`None`, 3.5e8) costs 1.0e10 actual on
   iter_120 (29×) and reports a different inertia (5104 vs 5107) on 48 of
   165 factors. Delays, not fill, decide cost, and only a factorization
   sees delays. Adoption has to be judged by a **trial factorization of
   the same matrix**, not by predicted flops of the candidate.
4. **The rollback rule as specified would misfire.** It compares the new
   analysis's first factor with the old analysis's last factor — two
   different matrices. Across iter_078 → iter_079 the incumbent analysis's
   own cost rises 1.66× (5.9e9 → 9.9e9 flops); the prototype of the
   review-scoped `OnStale` adopted at 78 and rolled back at 79 for exactly
   that reason, a net loss of one analysis and one slower factor.
5. **The two models want opposite things from the same trigger.**
   qcqp1000-2nc in the kkt-first order runs the whole solve at ratio 1.00
   under an analysis that is 2.4× the cost of `None` (41 s vs 21 s); a
   cost trigger never fires. Only a predicted-vs-predicted probe (a fresh
   analysis's prediction against the current analysis's prediction) sees
   it, and a trial factor confirms it (4.5e8 vs 8.6e8 actual, same
   inertia).
6. **What recovers both models in both call orders** (prototype in `$J/r227`,
   `--policy proto`): keep a small store of displaced/probed
   value-dependent analyses; open a trial on three events (cost ratio,
   a geometric probe schedule, and the #102 escalation); candidates are
   the fresh analysis plus the retained ones; adopt the cheapest by
   actual flops on the *same* matrix, only with equal inertia and no
   more floored/zero pivots. Prototype, summed actual flops against
   `Never` on the same call order: qcqp1000-2c 0.75 (init-first) and
   0.98 (kkt-first), qcqp1000-2nc 1.00 and 0.54, clnlbeam 1.00; inertia
   identical to `Never` on every factor of every sequence. Numbers in §7.
7. Analysis is cheap here (0.07–0.25 s on n = 11k, median 0.18 s) against
   0.06 s easy / 0.4–1.1 s hard factors; a trial factor costs one factor.
   Everything the policy decides on (flops, inertia, pivot counts) is
   bit-exact across thread counts, so the policy is deterministic per
   input sequence.

## 1. What is value-dependent in a cached analysis (code inspection)

`src/symbolic/mod.rs::symbolic_prefix_concrete` (lines ~1270–1365):
under `OrderingPreprocess::LdltCompress` the pipeline is
`compute_mc64_cache(matrix)` → `build_supermap(&cache.perm)` →
`compress_pattern(&full_pattern, &map)` → `run_external_ordering(&cpat)` →
`expand_permutation(&super_perm, &map)`. The fill-reducing ordering is
computed on the **compressed** graph, whose vertices are the matching's
2-cycles (`ldlt_compress.rs::build_supermap`: pairs from `perm[perm[j]] == j`,
k-cycles split into ⌊k/2⌋ pairs). The expanded permutation places each
matched pair in adjacent positions so the BK kernel sees it as a
tentative 2×2 pivot. A new matching therefore means a new compressed
graph and a new ordering; the old full-length permutation is still a
valid permutation, but it keeps the *old* pairs adjacent — using it with a
new matching is `OrderingMethod::External` with preprocess `None`
(`resolve_prefix` forces that), which drops the pairing that is the
point of `LdltCompress`. **The review's correction 1 holds**: only the
scaling choice can be redone without a new symbolic analysis.

Also value-dependent:

- `OrderingPreprocess::Auto` itself (`symbolic_prefix_preprocess_auto`):
  if the structural predicate fires, `None` and `LdltCompress` prefixes
  are both computed and `LdltCompress` is kept unless its fill exceeds
  `PREPROCESS_FILL_INFLATION_LIMIT` (2×) of `None`. Because the
  `LdltCompress` fill depends on the matching, **the resolved preprocess
  flips with the values**: on qcqp1000-2c it is `LdltCompress` for
  iter_000–~009 (895k–1.85M nnz vs 576k) and `None` from iter_010 on
  (1.86M+ nnz > 2 × 576k). The init matrix resolves to `None` (1.87M).
- Ordering escalation (#102, `solver.rs` after the factor): `Auto`
  resolved `None`, predicate says `LdltCompress`, growth
  `max|piv|/min|piv| > 1e24` → `last_symbolic = None`, `ordering_escalated
  = true`, re-enter `factor_impl`, which forces `preprocess =
  LdltCompress` on the **current** matrix. This is the one value-dependent
  re-analysis feral already performs mid-sequence, and it is unconditional
  (no cost or adoption test).
- The sticky `Auto` scaling pick (#51, `auto_picked_strategy`), resolved
  on the first factor of a pattern. On both qcqp models
  `pick_scaling_strategy` returns `InfNorm` on every probed iterate
  (iter 0/5/30/78/120/160 and 0/50/156), so the pick is not a factor on
  these sequences. Level (a) of the review ("scaling-only re-pick") is an
  O(nnz) re-run of the picker and one factor; nothing in this data set
  exercises it.
- The one-shot `SymbolicFactorization::cached_mc64` (#38) is cleared after
  every factor (`solver.rs` "Issue #38: invalidate the one-shot MC64
  cache"), so it cannot go stale; the solver-scope `Mc64ScalingCache` is
  value-bounded (`mc64_value_bound_passes`). Neither needs a rule beyond
  "untouched".

`Solver::invalidate_symbolic_cache()` clears only `last_symbolic`,
`last_pattern_fingerprint`, `last_pattern`. The next `factor()` then takes
the `!cache_valid` branch and **resets every pattern-bound latch**
(`ordering_escalated`, `auto_picked_strategy`, `mc64_retry_not_adopted`,
`race_result`, `mc64_scaling_cache`, `prev_max_n_delayed_in`,
`auto_arm_latched`, and drops `last_factors` so #153 recycling restarts).
That is what the #224 `--inval1` measurement and pounce#1046 did. A policy
re-analysis must not go through that path; it needs its own targeted
replacement of `last_symbolic` (§8).

`ladder_start` / `rung_before` are not reset on a pattern change today.

## 2. Flop accounting

feral has a predicted figure only: `SymbolicFactorization::work_estimate()`
gives `factor_flops = Σ ncol·nrow²` over supernodes (`symbolic/mod.rs:919`);
pounce's `LinearSolverSummary.total_factor_flops` is that predicted number
summed over factors (`pounce-feral/src/lib.rs:944`), not a measurement.
There is no actual-work counter in the numeric phase.

The counter used here, and proposed for the implementation:

    F(ncol, nrow) = Σ_{k<ncol} (nrow − k)²        (supernode.rs::front_flops)
    predicted      = Σ_supernodes F(sn.ncol, sn.nrow)
    actual         = Σ_node_factors F(nf.nelim, nf.nrow)

`nf.nrow` includes the delayed columns the node received (`n_delayed_in`),
and `nelim ≤ ncol` is what was actually eliminated there, so delayed
pivots inflate `actual` at the parent and deflate it at the child exactly
as the real Schur work moves. `nelim` and `nrow` are part of the factor's
bit-exact contract (`tests/parallel_parity.rs`), so the ratio is identical
across thread counts and runs. It is O(n_supernodes) to compute, no
allocation, and it needs no change to the kernels.

What the counter does not see: the per-rejected-pivot overhead (rook
scans, panel bails, #224 Steps 1–2). On iter_120 of qcqp1000-2c:

| analysis | actual flops | delays | nnz(L) | t_fac (s, contended) |
|---|---|---|---|---|
| `None` (init / fresh Auto on a late iterate) | 1.02e10 | 13 100 | 7.49 M | 0.33 |
| `LdltCompress` on iter_000 (`A_c0`) | 1.11e10 | 20 280 | 7.86 M | 0.47 |
| `LdltCompress` on iter_076 (escalated, `A_c76`) | 1.47e10 | 39 000 | 9.02 M | 0.92 |
| `LdltCompress` on iter_120 itself | 1.46e10 | 38 303 | 9.09 M | 0.84 |

Flops and time agree on the ranking, not on the magnitude (the delay
count is the better time proxy). For the policy only the ranking of
candidates on the same matrix matters, and ties within the margin are
declines.

## 3. Harness, sequences, init reconstruction

- `$J/r227/` (Rust, links the working tree): `r227 [--policy never|inval1|
  onstale|proto] [--fresh [--fresh-pp auto|ldlt|none]] [--pp auto|ldlt|none]
  [--noesc] [--serial] [--solve] [--scaling ...] files…`. pounce-feral's
  defaults (`u = 1e-8`, Auto ordering/scaling, parallel), one `Solver`
  across the sequence. Per factor it prints predicted/actual flops, the
  ratio, delays, floored/forced counts, inertia, growth, and (with
  `--fresh`) the predicted flops of a standalone fresh analysis of the
  same matrix and its time. `--policy onstale` is the review-scoped
  rule (trigger ratio, adopt on predicted flops, rollback on the next
  factor); `--policy proto` is the design of §7.
- `$J/mkinit.py`: rebuilds pounce's LS-multiplier init KKT from an
  `iter_000` dump using the `StdAugSystemSolver` triplet layout
  (`W | D_x | D_s | J_c | D_c | J_d | −I | D_d`) and `LeastSquareMults`
  (`w_factor = 0`, `δ_x = δ_s = 1`, `δ_c = δ_d = 0` on the first attempt):
  W → 0, D_x/D_s → 1, D_c/D_d → 0, Jacobians kept. Byte-identical to the
  #224 reconstruction (`i224/s0b/init/kkt_init.jsonl`, 0 differing
  values), which reproduced pounce's summary to the digit.
- Sequences (pounce 0.13 @ 9958a24e built against this tree,
  `--dump kkt:all`, `solve_quality_audit=no`, 300 s cap; `$J/run_dumps.sh`):

| model | n | factors (init + dumps) | pounce `total_factor_secs` | pounce iterations |
|---|---|---|---|---|
| qcqp1000-2c (#224 dumps) | 11 114 | 1 + 164 | 100.8 (0.21.0) | 163 |
| qcqp1000-2nc | 11 112 | 1 + 175 | 11.2 | 289 (129 restoration) |
| pooling_rt2stp (fixture) | 150 | 1 + 797 | — | 214 |
| NARX_CFy | 96 765 | 1 + 939 | 130.9 | 378 |
| clnlbeam | 99 999 | 1 + 2 526 | 42.8 | 2 465 |

qcqp1500-1c (489 s of factor time in the issue table) exceeds the cap and
was not dumped. Restoration-phase solves are not dumped by pounce, so
sequences replay the dumped `kkt_solve_*` files in order; `n_factors`
agrees with init + dumps on every model.

Timings in §4–6 were taken with other replays and a dump run in progress
and are indicative only; §7's clean sequential numbers are the ones to
quote.

## 4. qcqp1000-2c

### 4.1 Init-first (what pounce does; `--policy never`)

| phase | analysis | predicted | actual | ratio | t_fac |
|---|---|---|---|---|---|
| init | `None` (Auto race: LdltCompress 1.87 M nnz > 2 × 576 k) | 3.53e8 | 4.41e8 | 1.25 | 0.14 |
| iter_000–iter_075 | same | 3.53e8 | 4.41e8 | 1.25 | 0.06 |
| iter_076 | **escalation** (#102, growth 2e24): `LdltCompress` on iter_076 | 2.15e9 | 2.22e9 | 1.03 | 0.6 |
| iter_078 | same | 2.15e9 | 5.89e9 | 2.7 | 0.7 |
| iter_080–iter_150 | same | 2.15e9 | 1.47e10 | 6.9 | 0.9–1.0 |

`symbolic_calls = 2`. 165/165 inertia (6007, 5107, 0). Sequence total
80.7 s (contended; #224 measured 99 s on 0.21.0 kernels).

### 4.2 KKT-first (`--policy inval1`, = pounce#1046)

| phase | analysis | predicted | actual | ratio | t_fac |
|---|---|---|---|---|---|
| iter_000–iter_075 | `LdltCompress` on iter_000 (`A_c0`) | 6.98e8 | 6.98e8 | 1.00 | 0.10 |
| iter_078 | same | 6.98e8 | 4.27e9 | 6.1 | 0.30 |
| iter_080–iter_150 | same | 6.98e8 | 1.11e10 | 16 | 0.47 |

165/165 inertia (6007, 5107, 0); no escalation (resolved preprocess is
already `LdltCompress`). Total 46.6 s. So in the easy phase `None` beats
`A_c0` by 0.04 s/factor (≈ 3 s over 76 factors) and in the hard phase
`A_c0` beats the escalated analysis by 0.45 s/factor (≈ 38 s over 85).

### 4.3 `None` throughout (`--noesc`)

34.2 s total, ratio up to 29, but inertia differs from the expected
(6007, 5107, 0) on 48/165 factors (iter_120: (6010, 5104, 0), 684
floored pivots). Scaled dense `eigvalsh` on iter_120
(`$J/res/eig_qcqp2c_i120_i078.txt`, `eig224s.py`): strict count 5107,
but 3351 eigenvalues inside the n·eps·‖DAD‖ band — numerically singular,
count undecidable (singular-matrix exemption). The `LdltCompress`
analyses report 5107 on every factor, which is what pounce expects and
what its trajectory was computed on. Escalation is doing its job here;
`None` is not a candidate for the hard regime.

### 4.4 A `LdltCompress` analysis is only as good as the matrix it was built on

`LdltCompress` analysis built on matrix k, applied to iter_120
(`$J/res/qcqp2c.ack.*.txt`, `--noesc --pp ldlt`):

| built on | predicted | nnz(L) on iter_120 | delays | t_fac |
|---|---|---|---|---|
| init (forced) | 2.14e9 | 9.09 M | 35 466 | 0.90 |
| iter_000 | 6.98e8 | 7.86 M | 20 280 | 0.47 |
| iter_001 | 8.45e8 | 8.08 M | 22 946 | 0.66 |
| iter_002 | 1.03e9 | 8.07 M | 23 656 | 0.71 |
| iter_005 | 1.94e9 | 8.91 M | 34 087 | 1.05 |
| iter_010 | 2.12e9 | 9.16 M | 40 937 | 1.17 |
| iter_030 | 2.16e9 | 9.08 M | 38 523 | 1.14 |
| iter_075 | 2.16e9 | 9.09 M | 35 896 | 1.12 |

The predicted flops of a fresh `LdltCompress` analysis along the sequence
(`--fresh --fresh-pp ldlt`) rise smoothly: 7.0e8, 8.5e8, 1.0e9, 1.4e9,
1.75e9, 1.94e9, 1.98e9, 2.07e9, 2.10e9, 2.12e9 (iter 0–9), then ≈ 2.15e9
for the rest. The matching built on the initial point is a different and
better one; nothing later reproduces it. This is the mechanism behind
"the init ordering was better on qcqp1000-2nc" too (§5): which matrix
builds the matching matters more than how recent it is.

### 4.5 Analysis cost

Standalone `symbolic_factorize_with_method(.., Auto)`: 0.07–0.25 s, median
0.18 s (165 matrices; the `Auto` race runs both prefixes when the
predicate fires). Factor: 0.06 s easy, 0.4–1.1 s hard. A trial factor on
a hard matrix is therefore 2–6 analyses; on an easy one, a third of one.

### 4.6 The review-scoped `OnStale` on this sequence (`--policy onstale`, R = 2, margin 0.2)

Init-first: trigger fires at iter_078 (ratio 2.7). The fresh analysis on
iter_078, under the escalation latch, is `LdltCompress` (predicted 2.15e9
< 5.9e9/1.2 → "adopt"). Next factor (iter_079): 9.76e9 > 5.89e9 → rollback
and latch. Under the incumbent analysis the same step costs 9.92e9: the
rollback judged the drift of the sequence, not the analysis. Net: one
analysis, one factor on an equivalent analysis, then back to the
incumbent. (`$J/res/qcqp2c.onstale.t2m02.txt`; the prototype
regenerates the old analysis by re-factoring the init matrix, which is
why `symbolic_calls = 6` there.)

Had the fresh analysis ignored the escalation latch (plain `Auto`), it
would have been `None` (predicted 3.5e8): adopted on predicted flops,
then wrong-count factors (§4.3) until escalation fired again — a thrash
that the per-pattern cap would stop, at the cost of several re-analyses
and a changed inertia report.

## 5. qcqp1000-2nc

| order | analysis | predicted | ratio (median) | total t_fac | inertia (6005, 5107, 0) |
|---|---|---|---|---|---|
| init-first (`never`) | `None` | 3.57e8 | 1.25 | 20.7 s | 163/176 (13 are inertia-correction retries) |
| kkt-first (`inval1`) | `LdltCompress` on iter_000 | 8.59e8 | 1.00 | 41.2 s | 163/176 |

No escalation on either path (growth stays below 1e24: 8.5e23 at the
end). The fresh `Auto` analysis is `LdltCompress` only on iter_000/001
(8.59e8) and `None` (3.57e8) from iter_010 on. In the kkt-first order the
factor matches its prediction on every iterate — a cost trigger never
fires (ratio exceeds 2 only on the last 6 factors) — yet a fresh analysis
would predict 2.4× less work, and a trial factor of iter_014 under `None`
costs 4.47e8 against 8.59e8 with the same inertia. This is the case a
predicted-vs-predicted probe catches and a cost trigger cannot.

## 6. pooling_rt2stp, NARX_CFy, clnlbeam

- pooling_rt2stp: n = 150, 798 factors in 0.58 s (init-first) / 0.75 s
  (kkt-first). The analysis is sub-millisecond and so is the factor; the
  issue's "delays 13 175 → 1 475" and the 242 → 111 iterations are the
  trajectory, not factor cost. Nothing for a cost policy to gate on;
  the policy must simply cost nothing here.
- NARX_CFy (n = 96 765, 940 factors): both call orders resolve
  `LdltCompress`; ratio 1.00 (max 1.26 / 1.04) throughout — no cost
  event would fire. Init-first Σ actual 5.68e11 flops, Σ delays 434 501,
  Σ nnz(L) 1.888e9; kkt-first 5.44e11, 469 206, 1.924e9. Inertia identical
  on 941/941. The 30 % per-iteration difference in the issue's table
  (126.5 → 89.3 s at 378 → 380 iterations) has no counterpart in flops,
  delays or fill here; the replay cannot confirm it. Analysis cost
  0.24–0.85 s (median 0.46 s) against a 0.15 s median factor: three
  factors per analysis, so a 10-probe schedule costs ≈ 3–5 % of the
  sequence.
- clnlbeam (n = 99 999, 2 527 factors): both orders `LdltCompress`, ratio
  1.00 everywhere; init-first Σ actual 3.15e10, Σ delays 4 630, Σ nnz(L)
  2.476e9; kkt-first 3.64e10, 55 216, 2.679e9 — the init analysis is the
  better one, by 15 % in flops, as the issue's nnz(L) column says. A
  fresh analysis on iter_1000 predicts 1.115e7 against the init's
  1.246e7 (−10 %): inside the margin, no trial. Analysis 0.08–0.10 s
  against 0.016–0.025 s factors (4–6 factors per analysis): probes must
  be off here, and a deterministic size guard does it —
  `predicted_flops / nnz(A)` is 42 on clnlbeam, 90 on pooling, 1 450 on
  NARX and 1 600 on qcqp; probing only above 100 keeps probes where a
  factor outweighs an analysis.

## 7. The design the data support, and its prototype

Mechanism (one, with three entry points):

- **Store.** Besides the cached analysis, keep at most two displaced or
  probed analyses whose resolved preprocess is value-dependent
  (`LdltCompress`). Pattern-only (`None`) analyses are not stored: a
  fresh one is identical and cheap.
- **Events** that open a trial, evaluated only at the base pivot threshold
  and never while `ladder_start`/`rung_before` are set:
  1. *cost*: `actual/predicted > trigger_ratio` (default 2.0) for the
     stored factor, once per analysis (latched on decline);
  2. *probe*: factor index k on the pattern is a power of two ≥ 2 — a
     fresh analysis under the solver's effective configuration
     (`LdltCompress` forced if escalated) is computed; it opens a trial
     only if its predicted flops are below `predicted/(1+margin)` of the
     current analysis, otherwise it is stored (if value-dependent) or
     dropped;
  3. *escalation* (#102): the candidates are the fresh `LdltCompress`
     analysis (today's behaviour) and the stored `LdltCompress` analyses.
- **Trial.** Each candidate factors the *current* matrix. The adopted
  analysis is the cheapest by actual flops that is cheaper than the
  incumbent's factor by the margin (default 25 %), **reports the same
  inertia**, and has no more floored or zero pivots (`n_tiny + n_zero`).
  Forced accepts are not compared: on iter_076 the first-KKT analysis
  forces 248 pivots where the escalated one forces 0, both at
  `n_tiny = 0` and the same inertia, and the `Never` policy in the
  kkt-first order uses that factor without complaint. For the escalation
  event the fresh analysis is the default and a stored one replaces it
  only under the same gate.
- The factor returned to the caller is the adopted candidate's (already
  computed). Declined candidates that are value-dependent go to the
  store; the displaced incumbent goes to the store.
- **Caps.** `max_trials_per_pattern` (default 8 trial factors; a cost
  event may spend up to 1 + store size, a probe 1, an escalation store
  size). Probes stop after the schedule passes `max_probe_index`
  (default 2^10). All state resets on a pattern change.

No rollback: a stored incumbent is a candidate at the next event, which
is the same thing judged on one matrix instead of two.

Prototype (`r227 --policy proto`, `$J/r227/src/main.rs::proto`; one
`Solver` per analysis so the store is emulated by Solvers, escalation
emulated in the harness with all Solvers at `with_ordering_escalation(None)`;
analysis materialisation in the emulation costs a factor the real
implementation does not pay, counted under `t_trial`):

Prototype parameters: trigger 2.0, margin 0.25, cap 8 trials, store 2,
probes at k = 2, 4, 8, …, no size guard (`$J/res/*.proto2.*.txt`,
`$J/anproto.py`). "Σ actual" is the deterministic flop count of the
factors handed to the caller; "overhead" is analyses + trial factors.

| sequence | reference (`Never`) Σ actual | prototype Σ actual | ratio | analyses / trials / adoptions | what happened | inertia vs `Never` |
|---|---|---|---|---|---|---|
| qcqp1000-2c, init-first | 1.128e12 | 8.47e11 | **0.75** | 8 / 4 / 1 | probe at k = 2 stores the first-KKT `LdltCompress` analysis (not trialled: 7.0e8 > 3.5e8); at the escalation (iter_076) the fresh escalated analysis costs 2.22e9 and the stored one 1.35e9 with the same inertia → stored adopted; the cost event at iter_077 trials `None` and a fresh `LdltCompress` and declines | 165/165 equal |
| qcqp1000-2c, kkt-first (ref = `Never` on that order) | 8.67e11 | 8.47e11 | 0.98 | 8 / 7 / 2 | probe at k = 4 adopts `None` (4.4e8 vs 7.0e8, same inertia) for the easy phase; the escalation at iter_076 brings the displaced first-KKT analysis back (1.35e9 vs 2.22e9) | 164/164 equal |
| qcqp1000-2nc, init-first | 8.35e10 | 8.35e10 | 1.00 | 8 / 2 / 0 | probes store two `LdltCompress` analyses; the cost event on the last 6 factors trials them and declines | 176/176 equal |
| qcqp1000-2nc, kkt-first | 1.578e11 | 8.59e10 | **0.54** | 8 / 3 / 1 | probe at k = 8 adopts `None` (4.47e8 vs 8.59e8, same inertia) | 175/175 equal |
| clnlbeam, init-first | 3.149e10 | 3.149e10 | 1.00 | 11 / 0 / 0 | no candidate within the margin; 11 analyses ≈ 2.4 s on a 42 s sequence — the `predicted/nnz ≥ 100` guard (§6) removes them | 2527/2527 equal |

Both call orders of both qcqp models end on the better of the two
analyses the issue's table was comparing, and the choice is made by
measurement on the matrix at hand, not by a fixed re-analysis point.
Timings (clean, sequential, this machine, parallel driver as pounce
runs it):

| sequence | `Never` t_fac | kkt-first reference (`inval1`) | prototype: factors handed to caller + analyses + trials = total |
|---|---|---|---|
| qcqp1000-2c, init-first | 113.5 s | 64.0 s | 44.9 + 1.1 + 3.2 = **49.2 s** |
| qcqp1000-2c, kkt-first | 68.1 s | — | 38.1 + 1.1 + 3.6 = **42.8 s** |
| qcqp1000-2nc, init-first | 10.6 s | 26.4 s | 10.7–11.3 + 0.8 + 0.8 = **12.3–12.9 s** |
| qcqp1000-2nc, kkt-first | 28.3 s | — | 14.1 + 0.8 + 1.3 = **16.2 s** |

(`$J/res/clean.*.txt`; the 2nc init-first line was re-run twice back to
back, `$J/run_batch6.sh`.) Wall time on this machine drifts by tens of
percent between runs minutes apart (the 2c `Never` sequence took 80.7 s,
then 113.5 s; the identical 2nc factor set 10.6 s, then 14.7 s, then
10.7 s), so the flop ratios above are the numbers to gate on and the
times are indicative. Two things the times do show: the prototype's
overhead is 2–4 s per sequence (8 analyses plus 2–7 trial factors),
which is 16–22 % of the cheap 2nc init-first sequence where nothing is
adopted — the "must not regress" case. Halving the probe schedule
(k = 2, 8, 32, 128, …; every adoption above happened at k = 2, 4 or 8, and
the k = 4 one moves to k = 8 unchanged) brings it to ≈ 1.2 s; the plan
uses that schedule and gates 2nc init-first at ≤ 1.15 × `Never` in time
and 1.00 × in flops.

## 8. Pattern-bound state: keep or reset on a policy re-analysis

| state | rule | why |
|---|---|---|
| `last_symbolic` | replaced by the adopted analysis; the displaced one goes to the store | the policy's object |
| `last_pattern`, `last_pattern_fingerprint` | keep | same pattern |
| `symbolic_call_count` | increments per analysis computed (probes included) | diagnostic honesty |
| `auto_picked_strategy` (#51) | keep | the pick is `InfNorm` on every probed iterate here; scaling is orthogonal to the ordering. A separate level-(a) event (re-pick + one trial factor) is specified in the plan but off by default until a sequence shows the pick flapping |
| `mc64_scaling_cache` (B2) | keep | keyed on pattern + value bound, independent of the analysis |
| `SymbolicFactorization::cached_mc64` (#38) | a fresh candidate carries the cache for the matrix it was built on; the trial factor consumes it and it is cleared as today; stored analyses carry `None` | unchanged invariant: never reused past one factor |
| `ordering_escalated` (#102) | keep; fresh candidates honour it | stability decision, not a cost one; §4.3 |
| `mc64_retry_not_adopted` (N4) | keep | rank is a property of the values, not of the analysis |
| `race_result` (#203) | keep | `self.ordering` already holds the winner; candidates use it |
| `prev_max_n_delayed_in`, `auto_arm_latched` | reset on adoption | observations of the displaced analysis's factors (pounce does not arm auto cascade-break; off by default) |
| `last_factors` / #153 recycling | pass `None` to `begin_recycle` for a trial and for the first factor after adoption | buffers are indexed by supernode; recycling across analyses is memory-only but pointless |
| `ladder_start`, `rung_before` | untouched; their presence blocks every event | #221 single-use rung judgement |
| `last_pattern_reused` | `true` (the pattern was reused) | #215 semantics; `reanalysed` is the new signal |
| `FactorStats` | add `reanalysed: bool` (this call adopted a new analysis), `reanalysis_count: u32` (adoptions on this pattern), and a `ReanalysisInfo` with the event kind and the trial outcomes | review's "missing" list |

## 9. Risks and limits

- A trial costs a factor. On a model whose factors are 10 s the worst case
  is `max_trials_per_pattern` × 10 s; the default cap of 8 bounds it,
  and `Never` stays the default.
- The probe schedule computes up to ~10 analyses per pattern. Where an
  analysis costs several factors (clnlbeam, pooling) that is pure
  overhead, and on pf22-class matrices (MC64 ≈ 53 s) unacceptable. The
  plan gates probes on `predicted_flops / nnz(A) >= 100` (§6), an exact
  integer test; pf22 (n = 2.8 M) should be checked against it before
  release, and the host can leave `Never` there regardless.
- Adoption is judged on one matrix. The data show cost rankings are
  stable within a regime (§4.4), and a later event re-judges.
- A different analysis can change the reported count on numerically
  singular iterates (§4.3); the inertia-equality gate at the trial
  matrix prevents adopting an analysis that disagrees there, not later.
  pounce should expect this, as the review already says.
- `Always` (re-analyse every factor) is a probe on every factor with the
  same trial gate; it is for measurement, not production.
