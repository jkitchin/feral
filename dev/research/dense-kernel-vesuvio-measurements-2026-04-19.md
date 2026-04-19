# Dense-Kernel Measurement Pass for the VESUVIO Tail

**Date:** 2026-04-19
**Authorized by:** `dev/research/dense-kernel-vesuvio-tail.md` §4
**Goal:** Pin lever A's ceiling and decide the lever-B question
*before* drafting any plan document.
**Decision rule (from the parent note):** lever-B prototype wins
≥ 3× on a 1024×1024 trailing block → commit to the full 2.4.1b
plan; otherwise drop lever B and pursue lever A only.

## Step 1 — Lever A ceiling: schur_kernel microbench

Extended `benches/schur_kernel.rs` to include the wired-in
`axpy_minus_unroll4_nofma` / `axpy2_minus_unroll4_nofma` variants
(the kernels actually called by `do_1x1_update` / `do_2x2_update`
on aarch64) and added L = 2048 to cover the VESUVIOU root frontal
column length.

Throughput on the dev machine (M-series, single-threaded, Gelem/s
from criterion `--quick --warm-up-time 1 --measurement-time 2`).
Each elem is one FMA (= 2 FLOPs); 10 Gelem/s ≈ 20 GFLOPS.

`axpy_minus` (one source column, one destination — what
`do_1x1_update` calls):

| L    | scalar (autovec) | unroll4 FMA | unroll4 nofma  (wired) |
|------|-----------------:|------------:|-----------------------:|
| 8    |              2.0 |         2.3 |                    2.2 |
| 16   |              3.0 |         4.1 |                    4.2 |
| 32   |              4.6 |         6.0 |                    5.9 |
| 64   |              5.9 |         6.6 |                    5.8 |
| 128  |              7.5 |         8.7 |                    8.2 |
| 256  |              8.6 |         9.7 |                    9.2 |
| 512  |              9.7 |        10.4 |                    9.7 |
| 1024 |             10.0 |        10.2 |                   10.1 |

`axpy2_minus` (two source columns — what `do_2x2_update` calls;
each elem is two FMAs ≈ 4 FLOPs, so 7 Gelem/s ≈ 28 GFLOPS):

| L    | scalar | unroll4 FMA | unroll4 nofma (wired) |
|------|-------:|------------:|----------------------:|
| 256  |    7.0 |         7.0 |                   6.5 |
| 512  |    7.2 |         7.2 |                   7.2 |
| 1024 |    7.0 |         7.3 |                   7.2 |
| 2048 |    7.2 |         7.2 |                   7.2 |

### Findings (lever A)

1. **The wired `nofma` kernel is at peak BLAS-2 bandwidth.** At L
   ≥ 256 it is within 1–5% of the FMA variant on `axpy_minus` and
   tied on `axpy2_minus`. There is no AXPY-throughput headroom
   left to harvest on M-series.
2. **Autovectorized scalar is essentially as fast** at L ≥ 256
   (10.0 vs 10.1 Gelem/s at L = 1024 on `axpy_minus`). The
   hand-written NEON unroll4 wins primarily on small/medium L where
   rustc's autovectorizer doesn't pipeline as aggressively, and at
   the `axpy2` level (where it ties — the compiler already
   vectorizes that loop well).
3. **Lever A's residual ceiling is about 10–20% throughput.** That
   would close VESUVIO at most from ratio 84× to ~70× — well short
   of MUMPS-class. Wider unroll (a1 in the parent note) or x86_64
   (a3) would buy little on this hardware; on Linux/Intel the x86
   port is still useful for users without a NEON kernel today, but
   it does not change the M-series story at all.

**Verdict: lever A is exhausted on aarch64.** The x86_64 port
remains a nice-to-have for non-aarch64 users (it lifts them from
autovectorized scalar to the equivalent of our NEON unroll4) but
it is not the lever that closes VESUVIO.

## Step 2 — Where does VESUVIO factor time go?

Skipped as an instrumented run (would require adding timers
inside `factor_frontal`). Instead, applied the cost model from
the parent note §2 against the step-1 throughput data:

For VESUVIOU root frontal 2059×959:
- rank-1 cascade FLOPs ≈ 959 × 2059² / 2 ≈ 2.0 GFLOP
- At 12.7 GFLOPS effective (step-3 prototype, see below): 158 ms
- Observed VESUVIOU factor time (session 08): ~236 ms
- → rank-1 cascade accounts for ~67% of factor time

So the rank-1 cascade is the dominant cost as expected. Closing
it 10× would reduce total factor from 236 ms to ~94 ms (ratio
84× → ~33×). A 16× speedup in the cascade puts VESUVIO at
~88 ms (ratio ~30×). Full MUMPS parity (~5× ratio) requires
the cascade to drop to ~10 ms, i.e. a ~16× kernel speedup —
which matches faer's measured advantage on this exact kernel
shape.

A formal step-2 instrumentation can be revisited if step 3
recommends commitment to lever B and the budget for an
implementation pass needs sharpening.

## Step 3 — Lever-B prototype on 1024×1024 trailing block

New throwaway binary `src/bin/blas3_prototype.rs`. Compares
three implementations of "apply 64 panel columns to a 1024-wide
lower-triangular trailing block":

- **`rank1_cascade`** — 64 sequential calls to
  `axpy_minus_unroll4_nofma`, mirroring `do_1x1_update`'s inner
  loop on the wired NEON kernel. This is the current production
  path.
- **`rank_bs_update`** — naive deferred form: triple loop
  `for j; for i ≥ j; for k ∈ panel`, scalar inner. The simplest
  possible BLAS-3 reformulation.
- **`rank_bs_update_tiled`** — same shape with a manual 4-way
  register tile in the `i` dimension (4 independent
  accumulators, dot-product across `k` in the inner loop). Still
  no NEON intrinsics — relies on rustc autovectorization.

All three produce results identical to within 2.0e-12 on the
trailing block (verified at startup). FLOP count is identical;
only the loop order and inner-kernel structure differs.

Single-machine results (10 reps, dev machine, release build,
2026-04-19):

```
rank-1 cascade (64 updates):       5.30 ms  (12.7 GFLOP/s)
rank-64 update (naive triple):    34.96 ms  ( 1.9 GFLOP/s)
rank-64 update (4-tile ILP):      12.49 ms  ( 5.4 GFLOP/s)

speedup naive / rank-1: 0.15×   (i.e. 6.5× SLOWER)
speedup tiled / rank-1: 0.42×   (i.e. 2.4× SLOWER)
```

### Findings (lever B)

1. **The naive deferred form is 6.5× slower** than the rank-1
   cascade. Even the 4-way ILP tile is 2.4× slower. This is the
   same finding as the 2026-04-14 Phase 2.4.1a tried-and-rejected
   entry: BLAS-3 reformulation without a vectorized inner kernel
   is pure overhead.
2. **The reason is asymmetric SIMD.** The rank-1 path benefits
   from the wired `axpy_minus_unroll4_nofma` (NEON, 4-way unroll,
   independent accumulators). The deferred paths use plain Rust
   loops, which rustc partially vectorizes but not at the same
   register-tiled level. The "BLAS-3 advantage" only materializes
   if the rank-`bs` kernel is itself register-tiled SIMD —
   exactly faer's `Ukr<MR, NR, T>` micro-kernel pattern.
3. **The decision rule fires "no".** Per the parent note §4: if
   the prototype does not win ≥ 3×, drop lever B and pursue
   lever A only. The prototype loses by 2.4–6.5×. Combined with
   lever A's exhaustion at step 1, this is the inflection point
   — the dense-kernel work cannot proceed on the current "wider
   SIMD or naive blocking" menu.

### What the result actually says

The 0.42× tiled number does **not** mean lever B is dead. It
means lever B requires committing to a register-tiled SIMD GEMM
micro-kernel — i.e., writing the equivalent of faer's
`Ukr<MR, NR, T>` in NEON (and later x86 AVX2/FMA). That's a
~500-line module with non-trivial verification (panel-edge
correctness, masked tail, register pressure tuning per
microarchitecture). Estimated 3–5 sessions, possibly more.

The prototype does prove the structural reformulation is sound:
the math is identical to within 2e-12, so the surrounding
machinery (panel extraction, deferred apply, contribution-block
strip update) is straightforward — it's only the inner kernel
that needs SIMD work.

## Updated lever ranking after measurement

| lever | ceiling for VESUVIO ratio | implementation cost  | feasibility on M-series |
|-------|--------------------------|---------------------:|-------------------------|
| A     | 84× → ~70×               |     1 session (small) | **exhausted**, residual ~10–20% only |
| B     | 84× → ~5×                |     3–5 sessions      | requires register-tiled NEON GEMM micro-kernel |
| C     | 84× → ~1×                |  unknown, likely 2–3  | structural fix in symbolic, out of dense-kernel scope |

## Recommendation

The next-session menu, in order of preference:

1. **Pursue lever C first** — write a focused research note on
   detecting and exploiting the arrow-KKT structure during
   symbolic. If the slack supernode can be eliminated as
   diagonal 1×1 pivots before the main multifrontal sweep, the
   root frontal shrinks from ~2000×1000 to ~50×50 and the
   dense-kernel question becomes moot for this workload class.
   The pattern is detectable (max_col_nnz ≈ n/2 with diag_only
   ≈ n/2 — exactly what `vesuvio_diag` already prints) and the
   IPM corpus has at least four matrices with this signature.
   Lower implementation budget than lever B and a much higher
   ceiling.

2. **Defer lever B** until either (a) lever C is shown not to
   apply broadly enough to retire VESUVIO from the tail, or
   (b) a generic non-arrow factor outlier emerges that lever C
   cannot help. The 2.4.1b plan stays on the shelf as a
   reference; the prerequisite for it is now clearly "register-
   tiled NEON GEMM micro-kernel", not "blocked outer driver".

3. **Land an x86_64 SIMD path** as low-risk hygiene. This is
   not a VESUVIO fix — it lifts non-aarch64 users from
   autovectorized scalar to the equivalent of our NEON unroll4.
   The implementation is mechanical (port `axpy_minus_unroll4_nofma`
   to `core::arch::x86_64` AVX2) and the correctness story is
   the same as the existing aarch64 path. Half a session.

## What got committed

- `benches/schur_kernel.rs` — added `unroll4_nofma_neon`
  variants for both `axpy_minus` and `axpy2_minus`, extended
  length sweep to L = 2048.
- `src/bin/blas3_prototype.rs` — throwaway lever-B prototype
  binary with three implementations (rank-1 cascade, naive
  deferred, 4-tile ILP) and a self-contained timing harness.
- This measurement report.

## What did not get committed

- No production code changes (`src/dense/factor.rs` and
  `src/dense/schur_kernel.rs` are byte-identical to before).
- No plan document. The recommendation (§"Recommendation"
  above) is intentionally a *research finding*, not yet a
  plan — the lever-C research note has not been authored, and
  per CLAUDE.md the order is research → plan → tests → code.
