# `PAR_MIN_FLOPS` calibration probe — 2026-05-15

Issue #19 follow-up. The work-aware gate added in session
2026-05-15-03 introduced a flop-count threshold
(`PAR_MIN_FLOPS = 10⁸`) below which the parallel multifrontal
driver is rejected. The const was set "one decimal above
break-even" on a freehand estimate of "~100 µs rayon spawn +
~10 GFLOP/s sequential factor". Session 2026-05-15-04 deferred
empirical calibration to a follow-up. This is that follow-up.

## Method

`src/bin/calibrate_par_min_flops.rs`: sweeps the discrete
2D Poisson optimal-control KKT (`diag_poisson_kkt`'s problem,
no `δ_c`/`δ_w`) at `K ∈ {15, 20, 25, 30, 40, 50, 60, 80, 100,
130, 160}`, factors each through both the sequential supernodal
driver and the rayon parallel driver, and reports best-of-`reps`
wall time for each. Both drivers run inside a single persistent
`rayon::ThreadPool` (built with `rayon::current_num_threads()`
workers, matching the `Solver`-owned pool from session 04) to
amortise cv-wait wakeup. A single warm-up call per driver is
discarded before timing.

The probe calls
`factorize_multifrontal_supernodal_with_workspace` (sequential)
and `factorize_multifrontal_supernodal_parallel` (forced
parallel) directly, bypassing `should_parallelize_assembly`'s
gate. This isolates *driver performance* from *gate decision*:
the output answers "where *should* the gate fire" rather than
"does the current gate fire here".

Same `NumericParams` as `diag_poisson_kkt` (BK pivot 1e-8,
ForceAccept on zero, no scaling).

## Hardware

- Apple M4 Pro, 14 rayon threads (`rayon::current_num_threads`)
- macOS 26.3.1
- `rustc 1.93.1` (release, `lto = thin`, default codegen-units)

## Data (best-of-10)

| K   | n_kkt | n_snode | est_flops | seq (ms) | par (ms) | par/seq | regime           |
| --- | ----- | ------- | --------- | -------- | -------- | ------- | ---------------- |
| 15  | 675   | 223     | 1.3e5     | 0.20     | 0.44     | **2.18** | parallel hurts  |
| 20  | 1200  | 373     | 3.2e5     | 0.38     | 0.67     | **1.76** | parallel hurts  |
| 25  | 1875  | 580     | 6.7e5     | 0.64     | 0.94     | **1.48** | parallel hurts  |
| 30  | 2700  | 812     | 2.5e6     | 1.21     | 1.43     | 1.18    | tie              |
| 40  | 4800  | 1447    | 5.9e6     | 2.36     | 2.32     | 0.99    | tie (break-even) |
| 50  | 7500  | 2254    | 1.2e7     | 4.23     | 3.52     | **0.83** | parallel wins   |
| 60  | 10800 | 3247    | 2.3e7     | 7.31     | 5.41     | **0.74** | parallel wins   |
| 80  | 19200 | 5827    | 5.7e7     | 16.06    | 9.87     | **0.61** | parallel wins   |
| 100 | 30000 | 9061    | 1.1e8     | 30.23    | 16.12    | **0.53** | parallel wins   |
| 130 | 50700 | 15341   | 2.7e8     | 61.14    | 28.72    | **0.47** | parallel wins   |
| 160 | 76800 | 23313   | 5.4e8     | 114.28   | 47.63    | **0.42** | parallel wins   |

(`mc=y` for every row — every Poisson-KKT tree has at least one
multi-child supernode, so the structural gate would pass.)

## Findings

1. **Break-even (par/seq = 1.0)** sits at `est_flops ≈ 6×10⁶`
   (K=40 on Poisson-KKT). Below this the per-task rayon overhead
   exceeds the parallel arithmetic gain.

2. **Parallel wins ≥ 1.2×** at `est_flops ≥ 1.2×10⁷` (K=50).

3. **Parallel wins ≥ 1.35×** at `est_flops ≥ 2.3×10⁷` (K=60),
   the typical "meaningfully faster" bar.

4. **Current `PAR_MIN_FLOPS = 10⁸` is conservative by ~5×** on
   this hardware. It rejects parallel on K=50/60/80 where
   parallel beats sequential by 1.2×–1.65×.

5. The previous freehand estimate ("100 µs spawn + 10 GFLOP/s")
   was the right order of magnitude — break-even of 6×10⁶ flops
   on ~2.4 ms sequential time implies ~2.5 GFLOP/s effective and
   ~1 ms (not 100 µs) parallel overhead at the small end. The
   measured overhead is ~10× higher than the freehand guess; the
   measured GFLOPs is ~4× lower (these are sparse, not dense
   GEMM). The two errors approximately cancel, which is why
   10⁸ ended up only one decimal off rather than two.

## Recommendation

**Lower `PAR_MIN_FLOPS` from 10⁸ to 3×10⁷** based on this data:

- 5× above measured break-even (6×10⁶) — preserves the
  "meaningfully faster, not break-even" intent.
- Catches K=60 onward, where parallel wins 1.35×–2.4×, which
  the current const misses.
- K=50 (par wins 1.2×, the tightest margin) sits below 3×10⁷
  and stays sequential — the right call when measurement noise
  could swing the ratio either side of 1.2×.

## Caveats and follow-ups

1. **One workload family.** Calibration is on Poisson-KKT only.
   Other tree shapes (control-NLP small-KKT à la robot_1600,
   linear-network problems, mixed indefinite blocks) may have
   different overhead-vs-arithmetic balance. Issue #19's
   reporter saw a 12× wall regression on robot_1600 on
   non-M4 hardware; without their probe data we can't verify
   that 3×10⁷ is safe for their problem mix.

2. **One CPU.** M4 Pro is high-thread-count consumer silicon.
   M1 / M2 with fewer cores have *less* rayon overhead per
   spawn but *less* parallel headroom too — the crossover may
   shift either direction. The probe should be re-run on the
   reporter's hardware before the const is changed.

3. **One measurement style.** Best-of-N hides tail latency.
   IPM workloads care about mean/median across many factor
   calls. The expected effect is symmetric (both drivers see
   the same OS noise) but should be checked on real corpus.

4. **Action items.**

   - Post calibration data to issue #19 and ask the original
     reporter to run `cargo run --release --bin
     calibrate_par_min_flops -- --reps 10` on their hardware.
   - **Do not change `PAR_MIN_FLOPS` in this session.** The
     existing `NumericParams::min_parallel_flops` knob already
     lets consumers override; lowering the default needs
     cross-hardware verification.
   - Add the calibration probe to the standard "session bench"
     so regressions in driver overhead surface immediately.

## Reproduction

```sh
cargo run --release --bin calibrate_par_min_flops -- --reps 10
```

Knob: `--reps N` (default 3). Honours `RAYON_NUM_THREADS`.
