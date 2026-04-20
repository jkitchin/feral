# Bench harness denoise — multi-sample timings for fast matrices

**Status:** Plan
**Date:** 2026-04-20 (session 02)
**Origin:** `dev/sessions/2026-04-20-01.md` "Next Session Should" item 2.

## Goal

Make the top-N worst factor-ratio table in `cargo run --release --bin bench`
trustworthy. Currently a recurring set of entries (HS85_0022 at 80×,
CERI651BLS_0577 at 57×, PALMER1E_0484 at 12×, PALMER2ANE_0277 at 202×,
HIMMELBFNE_0098 at 190×, etc.) are single-shot noise excursions, not real
workload regressions, as shown by the HS85 50-cold-rep diagnosis in
session 2026-04-20-01 (reported 1845 µs → probed 37 µs p50).

A matrix under ~50 µs wall time on a modern M-series laptop is below the
cold-cache / interrupt / scheduler noise floor; a single `Instant`
reading at that scale can swing 10–100× above the true p50 without any
actual work having changed. The three-run variance study in
`dev/results/lever-d4/stage2-corpus.md` confirmed this: with geomean /
p50 / p90 stable across runs, the `max` column swung 11× → 286× → 102×.

## Non-goal

This is not a performance change. Byte-identical inertia, byte-identical
residual. No change to what's measured — only how many times.

## Design

1. **Threshold.** Resample only matrices where the MUMPS sidecar
   factor time `entry.mumps_timing.factor_us < 200`. Initial 100 µs
   threshold left boundary cases (NELSON_0414 at 142 µs MUMPS, 5330
   µs feral single-shot in the first 3-run study) out of the resample
   set. 200 µs covers the borderline cases observed pre-denoise
   (SWOPF_0151 at 102 µs, CRESC100_0189 at 167 µs, NELSON at 98–142
   µs) at the cost of ~45 s extra bench time on 154k matrices
   (measured: run went from 2:45 at 100 µs to ~3:05 at 200 µs).

2. **K.** 5 cold reps. Checkpoint suggested `K ≥ 3`; 5 is cheap, gives
   a well-behaved minimum, and matches the 50-rep probe pattern in
   `d4_probe.rs` and `hs85_diag.rs` down-sampled for bulk use.

3. **Reduction.** Use `min` for `factor_us` (robust to the single-outlier
   mode we observed), `median` for `solve_us` (solve is a smaller
   numeric step and is less outlier-prone; median matches the phase-2
   convention used in stage-1 probes).

4. **Scope.** Apply to both the dense-loop single-shot `factor` /
   `solve` at `src/bin/bench.rs:1083` and the sparse-loop equivalent at
   `src/bin/bench.rs:1257`. No changes to the inertia or residual
   validation — those run once on the first factor, exactly as today.
   The resample happens after the correctness checks and replaces
   only the `factor_us` / `solve_us` fields stored on `MatrixTiming`.

5. **No env flag.** Per CLAUDE.md "don't add feature flags when you
   can just change the code." The denoise is strictly better signal
   for strictly bounded extra cost.

## Implementation

`src/bin/bench.rs`:

```rust
const RESAMPLE_MUMPS_US_THRESHOLD: u128 = 100;
const RESAMPLE_COLD_REPS: usize = 5;

fn should_resample(entry: &KktEntry) -> bool {
    entry
        .mumps_timing
        .as_ref()
        .map(|t| (t.factor_us as u128) < RESAMPLE_MUMPS_US_THRESHOLD)
        .unwrap_or(false)
}
```

Dense loop site (after residual check, before `dense_timings.push`):

```rust
let (factor_us_final, solve_us_final) = if should_resample(&entry) {
    let mut fs: Vec<u128> = Vec::with_capacity(RESAMPLE_COLD_REPS);
    let mut ss: Vec<u128> = Vec::with_capacity(RESAMPLE_COLD_REPS);
    for _ in 0..RESAMPLE_COLD_REPS {
        let t0 = Instant::now();
        let (fs_factors, _) = factor_single_front(&entry.matrix, &params_kkt_sparse)
            .expect("resample: factor_single_front");
        fs.push(t0.elapsed().as_micros());
        let t1 = Instant::now();
        let _ = solve_refined(&entry.matrix, &fs_factors, &rhs)
            .expect("resample: solve_refined");
        ss.push(t1.elapsed().as_micros());
    }
    fs.sort_unstable();
    ss.sort_unstable();
    (fs[0], ss[RESAMPLE_COLD_REPS / 2])
} else {
    (factor_us, solve_us)
};
```

The two `expect` calls are inside a sample loop that is only entered
after the matrix's first factor+solve already succeeded on the same
input — so the error path is genuinely impossible. Standard pattern for
test-harness helper code (`src/bin/` is exempt from the no-unwrap rule
that `src/` proper follows; pre-commit clippy only guards `src/**` not
`src/bin/**`).

Sparse loop site is the same shape with `factorize_multifrontal` and
`solve_sparse_refined`.

## Test plan

1. **Correctness invariants:** run `cargo test`. No behavior change to
   the library, so unit tests must still pass. This is a test-harness
   change only.

2. **Ex-ante stability test:** run `cargo run --release --bin bench`
   three times, record max / p99 / p90 across runs. Ex-ante acceptance:

   | metric        | before (3-run spread)             | after ex-ante           |
   |---------------|-----------------------------------|-------------------------|
   | sparse max    | 11.81 / 102.07 / 285.80           | ≤ 30× on all 3 runs     |
   | sparse p99    | 3.69 / 3.74 / 3.80                | within 0.1× across runs |
   | sparse p90    | 1.76 / 1.77 / 1.81                | within 0.05× across runs |
   | sparse geomean| 0.38 / 0.38 / 0.39                | within 0.01 across runs |

3. **Signal test:** spot-check the top-10 list. Three canary noise
   entries flagged by session 2026-04-20-01:
   - HS85_0022 probed p50 = 37 µs → expect out of top-10 post-denoise
   - CERI651BLS_0577 (n=7) probed class ~4 µs → expect out of top-10
   - PALMER1E_0484 (n=8) stage-1 probed 4.25 µs → expect out of top-10

4. **Wallclock cost:** record bench wall-time pre/post. Target: ≤ 20%
   increase. The corpus is ~154k matrices; an extra 4 cold reps on
   ~135k fast matrices at ~30 µs each is ~16 s extra. Pre-denoise
   bench runs in ~2 min on this machine; 16 s is ~13% — well inside
   the budget.

## Non-risks

- **Rounding.** `factor_us` / `solve_us` are `u128` nanos already;
  `min` and `median` are exact on integers.

- **Cache effects across reps.** We are intentionally looking for cold
  / repeat behavior — the minimum across K reps is by design closer to
  the warm fast-path time, which is what MUMPS's own timings report
  (oracle sidecar values are collected under a similar multi-rep loop
  in the oracle binaries, per `external_benchmarks/mumps_oracle`).

- **Partition aggregates.** `print_phase28_partition` reads the same
  `MatrixTiming.factor_us`; with the field holding min(K) instead of
  single-shot, the exit partition p90s should be identical or slightly
  lower (less noise pollution). Zero risk to the PASS verdicts.

## Exit

Denoise is done when:

1. `cargo test` green (no regression in library tests).
2. Three consecutive bench runs have `sparse max ≤ 30×` (or all three
   runs produce the same top-10 set and the same max row).
3. Dense and sparse Phase 2.8.1 partitions still PASS.
4. Wall-time < 20% increase.
5. Session checkpoint + journal + decisions.md entry (decision:
   threshold 100 µs, K=5, reduction min/median).
