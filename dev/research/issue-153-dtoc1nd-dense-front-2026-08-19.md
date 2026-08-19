# Issue #153 item 2 — where dtoc1nd's dense-front time actually goes

Session 2026-08-19-05. Scope: the single surviving item of #153 after the
2026-08-19 rescope (items 1 and 3 were closed by prior sessions; see the
issue thread and `dev/journal/2026-08-19-05.org` 22:10).

The question: dtoc1nd is the worst downstream case in the #148 follow-up
table, and its time is not in scaling, ordering, or the sparse tail. Which
*phase*, on which *size class* of front, and is any of it recoverable?

## Instruments written for this

Neither existing probe could answer the question.
`profile_supernode_distribution` buckets loop time by `nrow` but on a
hardcoded matrix list and with no phase split; `diag_factor_phases` splits
phases but aggregates over every front. Neither can say *which size class*
a phase's time sits in.

- `crates/feral-diagnostics/src/bin/probe_front_bucket_phases.rs` — the
  front-size histogram weighted by loop time, with the phase breakdown
  computed inside each bucket, plus the bucket's mean `ncol`/`nrow` so the
  phase shares are interpretable. Sequential driver (the per-supernode
  deltas come from process-global counters, so a parallel driver would
  interleave them), warm workspace, median of N runs. `FERAL_BUCKET_BS`
  sets `BunchKaufmanParams::block_size`, so the phase shares can be read
  *as a function of the lever*.
- `crates/feral-diagnostics/src/bin/probe_panel_block_size.rs` — a paired
  alternating sweep over `block_size` (`dev/decisions.md` 2026-08-09
  methodology: every arm timed once per pair in order, `min_us` per arm,
  sign test over pairs), reporting per arm the inertia, the delayed-pivot
  count, a hash of every `d_diag`/`d_subdiag`/`l` bit in storage order,
  and the true relative residual.

## Finding 1 — the cost is 148 fronts, and it is 91% of the factor

`dtoc1nd_0010` (n = 9685, 482 supernodes), bucketed by `nrow`,
`FERAL_BUCKET_REPS=7`, default `block_size = 64`:

    bucket   count  time_ms  time%  flop%   avg_ns  ncol  nrow   asm%  dense%  panel%  schur%  tail%
     17-32     162    0.239    4.0    0.2     1476     1    26   36.8    54.6     0.0     0.0    0.0
     33-64     172    0.397    6.7    1.0     2311     2    50   19.2    75.2    11.1     0.5    0.0
    65-128     148    5.284   89.2   98.8    35701    62    88   14.4    84.9    53.5    19.5    0.0

The paying bucket is 148 fronts of mean shape `ncol = 62`, `nrow = 88`.
Time share and flop share agree (89% vs 99%), so this is not a bucket
whose cost is out of line with its work — it simply *is* the work.

Inside that bucket the split is anomalous against every other #153
fixture. Same probe, same settings:

    matrix                paying bucket   ncol  nrow   panel%  schur%
    dtoc1nd_0010          65-128            62    88     53.5    19.5
    clnlbeam_0001         17-32             17    19      6.4    23.9
    dtoc2_0000            33-64             16    46     13.4    17.2
    marine_1600_0010      33-64             17    47      7.5    27.3
    rocket_12800_0001     17-32             17    22     14.5    25.9
    steering_12800_0002   17-32             17    19     10.7    30.1

Every other fixture is panel 6-15% / Schur 17-30%. dtoc1nd is the other
way round, and the reason is shape: `bs = params.block_size.min(ncol)`
(`src/dense/factor.rs:2171`) and `block_size` defaults to 64, so a
62-column front runs as **one** panel and does all of its elimination in
the left-looking BLAS-2 kernel with no inter-panel BLAS-3 update at all.
dtoc1nd is the only fixture in the set with `ncol` in the 48-64 range —
every other one has `ncol` ≈ 17.

## Finding 2 — the packed-SIMD work gate is not involved

The standing hypothesis (issue text, and the doc comment this probe was
written under) was that dtoc1nd's front regime falls between the eager
small-front path and the packed-SIMD work gate
(`PACKED_SIMD_MIN_WORK = 1024`, `src/dense/factor.rs:3861`), so its
trailing updates run on the scalar walk.

**Falsified, structurally and by measurement.** The gate is
`simd_work = n_elim · (nrow − col_start) · ncol`; on the paying front that
is `62 · 88 · 62 ≈ 3.4e5`, which clears the 1024 threshold by 330×. So the
SIMD path is already taken. Forcing the question with
`FERAL_PACKED_SIMD_MIN_WORK` (median of 7, paying bucket avg_ns):

    min_work        avg_ns   panel%  schur%   note
    0                32910     54.9    17.9   SIMD always
    1024 (default)   32131     55.6    18.1
    1e12             37173     47.9    26.5   scalar always, +15.7%

Lowering the gate to 0 changes nothing — nothing was being rejected.
Raising it past every panel costs 15.7%, which confirms the SIMD kernel is
both engaged and paying on exactly these fronts. There is no missed
dispatch here to recover.

## Finding 3 — `block_size` moves the split exactly as predicted, and buys ~1%

If the 53.5% panel share is BLAS-2 work that ought to be BLAS-3, lowering
`block_size` should convert it. It does, cleanly and monotonically —
`FERAL_BUCKET_BS` sweep, median of 9, paying bucket:

    bs    avg_ns   panel%   schur%   tail%
     8     35016     14.7     57.2     2.3
    16     33449     23.2     48.8     0.0
    32     31367     36.0     36.3     0.0
    48     30633     43.7     28.0     0.0
    62     32199     53.4     18.4     0.3
    64     32348     54.1     17.9     0.0

The panel share falls 54% → 15% and the Schur share rises 18% → 57% as the
panel narrows. The mechanism is confirmed. **The wall clock is not.**
Across-process `avg_ns` from this sweep is not trustworthy at this effect
size — an earlier run of the same sweep put the minimum at `bs = 64`, this
one at `bs = 48`. The paired alternating sweep is the measurement of
record (single process, arms interleaved, 60 pairs):

    bs    min_us   vs_base   wins/60
    16      9769     1.004         6
    32      9673     0.994         9
    48      9604     0.987        43
    64      9733     1.000         2   <- default

`bs = 48` does win: 43 of 60 pairs against an expected 15 under the null,
which is not noise. But the effect is **1.3% of total factor time**, on a
matrix whose paying bucket is 91% of that time. Converting three quarters
of the panel share into BLAS-3 buys 1.3%. The 53.5% is not recoverable
time; the two kernels cost very nearly the same per flop at this front
shape.

It also does not generalize. `bs = 48` and `bs = 64` are *identical* for
any front with `ncol ≤ 48`, and that is every other fixture in the corpus
sample — `ncol` p90 is 1-19 everywhere except dtoc1nd (63):

    matrix           ncol p90   ncol max
    dtoc1nd_0010           63         63
    pinene_3200             19         29
    marine_1600             19         51
    nql180                  15         86
    gasoil_3200             16         21
    robot_1600              16         17
    clnlbeam                16         16
    dtoc2                    1         23
    svanberg                 1         18
    qcqp1500-1c              1        612
    cont5_2_4_l              1        308

On the two with any wide fronts at all, the paired sweep finds nothing:

    nql180_0000       bs=48  0.990  5/12 wins (tied with bs=64)
    qcqp1500-1c_0000  bs=48  0.994  3/12 wins

So `block_size = 48` is a 1.3% win on one matrix and a no-op on the rest.
That is below the bar for changing a global default. Not shipped; recorded
in `dev/tried-and-rejected.md`.

## Finding 4 — `block_size` is bit-neutral (worth keeping)

Every arm of every sweep above produced the same inertia, the same zero
delayed pivots, the same residual, and the **same hash over every `L` and
`D` bit in storage order**:

    dtoc1nd_0010    5960/3725/0  d0   9cb93f568423e6c0   resid 5.54e-6
    nql180_0000   129601/130080/0 d0  4f588093d6bac8c7   resid 7.55e-7
    qcqp1500-1c     1500/10508/0 d0   cfec17df1a4f8d38   resid 7.55e-7

across `bs ∈ {8,16,24,32,48,62,64}`. On these matrices `block_size` is
pure scheduling — it does not perturb the pivot sequence, so any future
retuning of it is a performance-only change. Caveat: none of the three
delays a pivot (`d0` throughout), so this is not yet established on a
matrix where `may_delay` actually fires.

## What is left

The dtoc1nd gap is *not* a mis-set threshold. Three candidate levers are
now eliminated: the packed-SIMD work gate (Finding 2), the panel/Schur
split (Finding 3), and — from the rescope — scaling and `nemin` (issue
thread). The remaining decomposition of the paying front is assembly 14.4%
/ dense 84.9% with the dense part genuinely spent in kernels that are
already SIMD and already near-optimally blocked. Any further win has to
come from the kernel itself at `ncol ≈ 62`, `nrow ≈ 88`, or from producing
fewer such fronts (ordering), not from a dispatch decision.
