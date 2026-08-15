# LdltCompress cost/benefit — the KIRBY2 factor-ratio outlier

Session 2026-08-15-02. Investigates the worst factor-ratio outlier in
the bench (KIRBY2_0007, 8.95x vs MUMPS) and finds it is not a
factorization problem at all.

## Summary

`LdltCompress` (the MUMPS `ICNTL(12)=2` Duff-Pralet compression
preprocessor) engages on **74%** of corpus families and is a **net
loss on 96%** of the families where its effect is measurable, at a
**median cost of 18.1%** of total factor time. On the worst case it
costs +696%. The MC64 matching it requires dominates symbolic time
while delivering no fill reduction.

An oracle gate keyed on "MC64 cost exceeds the achievable numeric
saving" separates losers from winners cleanly. Implementing it
requires solving a cost-prediction problem the 2026-06-06-04 session
identified and did not solve; a deterministic work-budget design is
proposed below that sidesteps prediction entirely.

## Evidence

### The outlier is symbolic, not numeric

`diag_factor_phases` on KIRBY2_0007 (n=458, nnz=1522) reports a
numeric driver wall of **127 us**, against a bench-reported
`factor_us` of 1065 us. Per-stage symbolic breakdown
(`diag_symbolic_stages_argv`):

    KIRBY2_0007   TOTAL 1182 us
      ldlt_compress   972   82.2%
      renumber         57    4.8%
      ordering         32    2.7%
      (all else <2% each)

Ordering is 2.7%. The initial hypothesis — that the five near-dense
hub rows (symmetric degree ~155 over a bed of 302 degree-2 and 151
degree-8 rows) blow up AMD's quotient graph — is **refuted**.

### The comparison is fair

Before attributing the gap, confirmed feral's `factor_us` (symbolic +
numeric, `src/bin/bench.rs:1926`) is apples-to-apples with the oracle:
`external_benchmarks/mumps_oracle/mumps_bench.F:191` sets
`MP%JOB = 4` inside the timed region, and JOB=4 is analyse +
factorize. MUMPS's 119 us does include analysis.

### Compression buys nothing here

`probe_compress_costbenefit_argv`, 5-run median, None vs LdltCompress:

    matrix          n  | sym_n num_n tot_n | sym_c num_c tot_c | delta   delta%
    KIRBY2_0007   458  |   155   143   297 |  1041   149  1191 |  +894  +301.0%
    KIRBY2_0006   458  |   151   153   300 |   924   144  1094 |  +794  +264.7%
    GROUPING_0243 225  |   104   122   226 |   364    97   461 |  +235  +104.0%
    GROUPING_0031 225  |   103   113   216 |   308   100   410 |  +194   +89.8%

`num_c ~ num_n` throughout — zero numeric benefit. Uncompressed,
KIRBY2_0007 is 297 us against the oracle's 119 us, i.e. **~2.5x
instead of 8.95x**.

### Corpus sweep

One iterate per family, 696 families with .mtx < 400 KB, 693 parsed:

    LdltCompress engaged        : 514 / 693  (74%)
    with tot_n >= 50us (signal) : 164  -> losers 157, winners 7
    loser delta% distribution   : p25 13.0  median 18.1  p75 26.2
                                  p90 52.3  max 696.4
    aggregate waste             : 121945 us
      GILBERT_0000 alone        :  87361 us (72% of the total)
      other 156 losers          :  34584 us (median 120 us each)

The aggregate figure is dominated by one pathological matrix, but the
**distributed** effect is the real finding: a median 18.1% tax on 96%
of the families where compression engages.

## The gate

Oracle form: skip compression when

    mc64_cost > T * num_n

where `num_n` is the uncompressed numeric time. The argument needs no
fill prediction: `num_n` is the *entire* numeric phase, so even a
compression that drove numeric time to exactly zero could not repay
an MC64 costing more than `T=1` times it.

Threshold sweep over the signal set (166 losers / 7 winners):

    T     skips losers   us recovered   winners broken
    0.25    126/166         114860        1  (HAHN1, 79 us, -9.7%)
    0.50     56/166         106307        1  (HAHN1)
    1.00     21/166          96991        0
    2.00      6/166          93553        0
    4.00      2/166          89940        0

Excluding GILBERT, to check the win is not one matrix: at T=0.25 the
gate skips 120/156 remaining losers and recovers 27436 us of 34584 us
(79%), breaking one 79 us winner. A ~347:1 trade.

### Validation against the 2026-06-06-04 counterexamples

`dev/tried-and-rejected.md:2364` rejected a *scaling-aware* compression
skip and named ROSEPETAL (win) vs ORTHREGF (loss) as the pair no cheap
feature separates. Re-measured:

    ORTHREGF_0000  6405 | 1692  1422  3183 | 5850 1420 7294 | +4111 +129.2%
    ex8_2_2_0000   9453 | 2355 144305 146601| 3313 138835 142148 | -4453 -3.0%
    SINQUAD2_0000  5000 | 1277  1665  2924 | 1582 1622 3199 |  +275  +9.4%

- ORTHREGF: mc64 4158 us vs num_n 1422 us -> ratio 2.92, **skipped**.
  Correct (it is a +129% loser).
- ex8_2_2: mc64 958 us vs num_n 144305 us -> ratio 0.007, **kept**.
  Correct (it is a -3.0% winner).
- SINQUAD2: ratio 0.18, **kept** — a conservative miss (it is a +9.4%
  loser). The gate errs toward keeping compression.

So the ceiling ratio does separate the pair that defeated the
scaling-reuse signal.

**ROSEPETAL could not be re-measured.** Its only iterate is truncated:
the header declares 2003000 entries and the file contains 1460226, so
`read_mtx` rejects it. `data/matrices/` is gitignored
(`.gitignore:20`), so the file is untracked and the corruption cannot
be dated or reverted. This is the same class of data defect as the
known `dtoc2_0001/0002` non-finite-value bug. Consequence: the
flagship counterexample of the 2026-06-06-04 rejection is currently
unreproducible. Its recorded numbers (MC64 0.68 s, numeric 5.72 s ->
0.77 s) do satisfy the ceiling gate — ratio 0.68/5.72 = 0.12, well
under every threshold in the sweep, so the gate would **keep**
ROSEPETAL — but that rests on recorded values, not a fresh
measurement.

## The implementation obstacle

The gate above is an **oracle**: both `mc64_cost` and `num_n` are
measured after doing the work. Neither is available at the decision
point.

`num_n` is tractable: the uncompressed ordering plus `col_counts` is
32 + 20 = 52 us on KIRBY2 (~5% of the MC64 cost), and column counts
give a standard flop prediction.

`mc64_cost` is **not** cheaply predictable. Across the signal set the
observed MC64 cost per row spans

    min 0.0303  p25 0.0560  median 0.0972  p75 0.2061  max 17.3509 us/row

a **573x spread**. Any estimator keyed on `n` (or `n` and `nnz`) will
be wrong by orders of magnitude on the tail — which is exactly where
the win is.

### Proposed design: deterministic work budget

Do not predict the MC64 cost — *bound* it:

1. Run the uncompressed ordering and `col_counts` first (~5% overhead).
2. Predict uncompressed numeric flops from the column counts.
3. Run the MC64 matching under a **work budget** proportional to that
   flop prediction, counted in augmenting-path edge scans — not wall
   clock.
4. If the matching exceeds its budget, abandon it and fall through to
   the uncompressed ordering already computed in step 1.

Counting work rather than time keeps the outcome deterministic and
machine-independent, which a wall-clock abort would not (cf. the
determinism probes `probe_value_determinism`, `diag_par_repeat`). The
fallback path is free because step 1 already produced a usable
ordering.

Open questions before implementing:
- Where in `mc64::compute_matching` a work counter can be threaded
  without disturbing the matching result on the non-aborted path.
- Whether abandoning changes `cached_mc64` state that the numeric
  phase or the #38 staleness guard depends on.
- Calibration of the budget constant against the T sweep above.

## Status

Research only. No code changed. Constraints from
`dev/tried-and-rejected.md:2364` respected: this does **not** gate on
the scaling strategy, and does not treat the MC64 as speculative —
`build_supermap` genuinely consumes the matching.
