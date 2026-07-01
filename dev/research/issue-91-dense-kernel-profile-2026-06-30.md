# issue #91 follow-up — dense-kernel profile of the fixed qap15 factor — 2026-06-30

After the `OrderingPreprocess::Auto` fix (PR #92), qap15's residual gap to
faer is **entirely dense-kernel throughput**. Evidence: post-fix feral stores
*fewer* nonzeros than faer (9.25M vs 13.4M) yet is ~3× slower — so it is
per-flop MFLOP/s, not fill.

## Profile (sequential driver, `examples/profile_qap15.rs`, release)

Per-supernode profiler (`NumericParams::profiler`), `with_parallel(false)`:

```
fma=false  loop=1747 ms   n_supernodes=13352   nnz_L=9,247,544
  front-size (nrow)  count   sum_ms   %loop
    <=8             11716     48.9     2.8
    9-16             1579     33.5     1.9
    33-64              21      8.6     0.5
    >128               36   1656.0    94.8      <-- dominates
  top fronts (nrow x ncol -> ms):
    2955 x 2955  -> 736.6 ms  (42.2% of loop)   <-- the root front
    3313 x 108   -> 107.0 ms  ( 6.1%)
    3205 x 115   -> 106.8 ms  ( 6.1%)
    3090 x 108   ->  91.7 ms  ( 5.2%)
    3059 x 49    ->  48.0 ms  ( 2.7%)
    ... (tall-skinny 3000 x {16..115} tail)
fma=true   loop=1341 ms  (-23%);  2955x2955 root 736 -> 499 ms (-32%)
```

## Reading

- **The >128 bucket is 94.8 % of the loop; 36 fronts.** Everything ≤128 is
  ~5 %. Optimize the large fronts or nothing.
- **One 2955×2955 near-square root front is 42 % of the loop.** This is the
  assignment-structure Schur complement — a large dense LDLᵀ. It is the single
  highest-value target (Phase C "cache-blocked dense root" in
  `dev/plans/dense-kernel-blas3.md`).
- The rest of the >128 mass is tall-skinny fronts (nrow≈3000, ncol≈16–115):
  trailing-update-bound, the Phase B-1 DSYRK target.
- **FMA alone = 23 % overall / 32 % on the root**, with identical inertia
  `(+22275,−28605,0)` and identical `nnz_L` — but it changes bit patterns and
  was deliberately kept opt-in (pivot-classification drift on 4 small-front KKTs,
  `dev/tried-and-rejected.md` 2026-04-14). So FMA is *not* the correctness-safe
  lever.

## Throughput math

2955×2955 LDLᵀ ≈ 2955³/3 ≈ 8.6 GFLOP. At 736 ms (nofma) that is ~11.7 GFLOP/s;
499 ms (fma) ~17 GFLOP/s. A register-tiled, packed f64 kernel on this core
(Apple M-series NEON, 2 FMA pipes × 2 lanes × ~3.5 GHz ≈ 28 GFLOP/s peak with
FMA, ~14 without) should reach ~2–3× the nofma rate. So the bit-exact (no-FMA)
ceiling is ~25–30 GFLOP/s ⇒ the 2955² root ~300 ms, and the whole loop toward
faer's range — without touching rounding.

## Current kernel state

The trailing update (`apply_schur_panel_range`, `src/dense/factor.rs:3239`)
already tiles NR=4 columns (`schur_panel_minus_nofma_strided_quad`) with 4-way
row-SIMD and q-inner accumulation — an ~8×4 register tile. What it lacks vs a
real BLAS-3 kernel:

1. **No panel packing.** The n_elim eliminated columns are re-read *strided*
   from `head` (column stride = nrow ≈ 2955) for every trailing column-group,
   thrashing cache/TLB on the big fronts. The panel is reused across all
   trailing column-groups, so packing it once into a contiguous buffer is a
   bit-exact (values/order unchanged) cache win — the classic GEMM pack.
2. **No dedicated dense-root path.** The 2955² root runs the generic
   tall-front blocked path; a square-root path (Phase C) can block both
   dimensions.

## Plan (bit-exact, correctness-first — no rounding change)

Priority order, each byte-identical to the current `_nofma` output (parity
oracle = existing strided kernel + scalar reference):

- **B-1a — pack the L panel** into a contiguous `[n_elim × trailing]` scratch
  buffer once per panel; kernels read unit-stride from it. Bit-exact; targets
  the cache/TLB bottleneck. Bounded change, own parity test.
- **B-1b — widen/schedule the packed microkernel** (larger MR×NR with more
  independent accumulators) now that reads are contiguous.
- **C — cache-blocked dense-root** path for `nrow==ncol && ncol>=256` (the
  2955² root), blocking both dimensions over the packed kernel.

FMA-on-large is left as a separate, opt-in-gated lever (23 %) because it changes
bit patterns and the project keeps `nofma` the default; the bit-exact packed
kernel is preferred as the durable, inertia-safe win.

Harness: `examples/profile_qap15.rs` (per-front buckets + top fronts, nofma vs
fma). Note the numbers above are the *sequential* driver (~1.75 s); the parallel
default factor is ~0.67–0.77 s, but the per-front *attribution* is what matters.

## Update (same day) — B-1a packing measured and rejected; root is DST-bandwidth-bound

Implemented B-1a (pack the L panel, feed the existing kernels, byte-exact) and
measured: **net slowdown** — sequential loop 1747 → 1976 ms (+13%), the 2955²
root 736 → 818 ms (+11%), parallel default 771 → 945 ms (+22%). Byte-exact
parity held (blocked_ldlt 21/21, inertia/nnz_L unchanged). Reverted. See
`dev/tried-and-rejected.md` 2026-06-30.

**Corrected model of the bottleneck.** The panel (~1.5 MB) is already
L2-resident; packing it optimizes the wrong operand. The 2955×2955 root is
**DST-bandwidth-bound**: right-looking blocked factorization streams the ~70 MB
trailing block once per rank-`bs` panel (~46 passes at `bs=64`), so total DST
traffic ≈ Σ trailing-sizes ≫ panel size. The lever is reducing DST passes:

- **Phase C — cache-blocked / recursive dense-root** (`nrow==ncol && ncol large`,
  e.g. the 2955² front): factor in cache-sized tiles so a trailing tile is
  reused across many panels before eviction — O(n³/√cache) bandwidth instead of
  O(n³). Bit-exact (same arithmetic, reordered blocking). This is the real,
  durable win and the correct next build.
- **Larger panel width** (`bs` 64 → 96/128, capped by `MAX_N_ELIM=128`): more
  flops per DST stream, fewer passes. Cheaper to try; bounded by the panel
  factorization cost and 2×2-pivot handling. Worth a quick A/B before Phase C.
- **FMA-on-large**: still +23%, but a reproducibility-policy change (opt-in),
  not bit-exact — separate decision.

Source-side packing (B-1a/B-1b as originally scoped) is off the table.
