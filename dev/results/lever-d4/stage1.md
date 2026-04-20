# D.4 stage-1 — tiny-n probe results

**Date:** 2026-04-20 (session 01).
**Binary:** `src/bin/d4_probe.rs`, 50 cold reps per row.

## Raw output

```
name                    n   nnz    rho     gate |    pre_min    pre_p50 |   post_min   post_p50 |    p50_x
--------------------------------------------------------------------------------------------------
HS73_0308               7    19  0.679    DENSE |    5.54us    6.00us |    3.75us    3.92us |    1.53x
PALMER1E_0484           8    36  1.000    DENSE |    5.04us    5.46us |    4.08us    4.25us |    1.28x
HATFLDH_0083           11    33  0.500    DENSE |    8.17us    8.42us |    5.58us    5.71us |    1.47x
PALMER1A_0034           6    21  1.000    DENSE |    4.00us    4.12us |    3.25us    3.33us |    1.24x
KIRBY2LS_0274           5    15  1.000    DENSE |    3.58us    3.79us |    2.96us    3.25us |    1.17x
HEART6LS_0418           6    21  1.000    DENSE |    4.08us    4.33us |    3.29us    3.42us |    1.27x
```

`pre_*` = forced multifrontal via `factorize_multifrontal_supernodal`.
`post_*` = gated dispatcher (`factorize_multifrontal`) with D.4.

## Findings

1. **D.4 delivers 1.17–1.53× speedup at the p50 of the cold
   distribution on all six observed top-10 tiny-n rows.** The
   speedup is smaller than hoped because the pre-D.4 multifrontal
   path on these tiny matrices was already fast (≤ 10 µs); D.4's
   win is that dense_fast_factor skips the 13 µs symbolic cost
   HS85-class probes showed at n=68, and that saving plus the
   trimmed scaling overhead lands at ~1 µs to ~2 µs wallclock
   per call.

2. **All six rows are ALREADY D.3-eligible** (ρ ≥ 0.50 in every
   case). That means the bench-harness behavior was
   dense_fast_factor → single-shot noise, not multifrontal
   slowness. The 9–13× ratios in the 2026-04-20-01 bench run
   were cold-cache outliers on the dense path, not a D.4 target
   class in the sense the research note described.

3. **D.4's unique class is `n ≤ 16 AND ρ < 0.25`** — sparse tiny
   matrices that D.3 rejected. None of the six named rows fall in
   that class. Whether such matrices exist in the IPM corpus and
   how their timings shift is the stage-2 question.

## Acceptance vs MUMPS

Using sidecar MUMPS timings for each row:

| row              | post_p50 | MUMPS | ratio | target |
|------------------|---------:|------:|------:|:------:|
| HS73_0308        |  3.92 µs |  9 µs | 0.44× | ≤ 3×   |
| PALMER1E_0484    |  4.25 µs | 10 µs | 0.43× | ≤ 3×   |
| HATFLDH_0083     |  5.71 µs | 10 µs | 0.57× | ≤ 3×   |
| PALMER1A_0034    |  3.33 µs | 13 µs | 0.26× | ≤ 3×   |
| KIRBY2LS_0274    |  3.25 µs | 10 µs | 0.33× | ≤ 3×   |
| HEART6LS_0418    |  3.42 µs | 10 µs | 0.34× | ≤ 3×   |

All six comfortably under the 3× ex-ante target. Feral beats
MUMPS by 2–4× on the p50 of the cold distribution on every one.

## Next step

Stage 2: run `cargo run --release --bin bench` and confirm:

- the six named rows drop out of the sparse top-10
- geomean does not regress from 0.37
- any new tiny-n row that appears in the top-10 is a D.4-unique
  class (ρ < 0.25, n ≤ 16) worth examining
