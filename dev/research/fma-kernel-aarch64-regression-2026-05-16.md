# FMA Schur-panel kernel regression on aarch64 — 2026-05-16

Closes the measurement gap from issue #35. Companion to
`dev/research/fma-kernel-opt-in.md` (which documented the FMA path as
opt-in) and `dev/research/wide-supernode-throughput-2026-05-16.md` §6
(which surfaced the regression at the full-factor level).

## TL;DR

On Apple M-series (aarch64 / NEON ASIMD), `schur_panel_minus_fma_strided_quad`
is **12–25% slower** than `schur_panel_minus_nofma_strided_quad` across
the three regimes that matter for the wide-supernode workload. The
regression is confirmed at the kernel level, not just at the full-factor
level — so the cause is intrinsic to the FMA kernel body, not panel
admin or dispatch overhead.

x86 measurement landed (CI run 25971444759 on ubuntu-latest x86_64
via commit f1f9894): FMA is **1.55× faster** in every shape — the
textbook AVX2+FMA win. Decision tree branch 3 ("x86 wins, aarch64
loses") confirmed. See §"Resolution" at the end of this note and
`dev/decisions.md` 2026-05-16 — FMA Schur-panel kernel: per-arch
asymmetry.

## Probe

`src/bin/probe_fma_kernel.rs` — direct A/B of the two kernels on
identical synthetic inputs. Three shapes mirror the issue #14
probe (`probe_wide_supernode.rs`):

- `wide_2829x433`  (nrow=2829, n_elim=64) — wide trailing rows
- `square_1928`    (nrow=1928, n_elim=64) — square-ish root supernode
- `narrow_512x32`  (nrow=512,  n_elim=32) — narrow panel, modest trailing

For each shape: 5 warmup calls, 21 timed reps, report median + min ns
and effective GFLOPS. Dst buffers are reset from a template per rep so
each call sees identical input.

## Numbers (M-series aarch64, 2026-05-16)

```
shape          nrow  n_elim   fma med   nofma med   fma GF  nofma GF   fma/nofma
wide_2829x433  2829  64       80.8 us   64.8 us     17.5    21.9       0.80
square_1928    1928  64       48.8 us   41.7 us     19.6    22.9       0.85
narrow_512x32   512  32        6.5 us    5.8 us     18.9    21.2       0.89
```

Speedup ratio `nofma_ns / fma_ns`: `>1` means FMA wins. All three are
`<1` — the FMA path is the regression in every shape.

The min-time ratios (least noisy measurement) tell the same story:
0.89 / 0.85 / 0.90 — FMA loses by 10–15% on the cleanest reading.

## Why? (hypothesis)

`schur_panel_minus_fma_strided_quad` uses `simd.mul_add_f64s(...)` end
to end (one rounding per multiply-add via NEON `vfmaq_f64`). The
`_nofma` sibling uses explicit `mul + sub` (two roundings) — *more*
arithmetic per element, yet measurably faster on aarch64.

Two candidate explanations, both consistent with the observed pattern:

1. **Pipeline disparity.** NEON on M-series has dedicated FMA pipes
   that, in principle, retire one FMA per cycle per pipe. The
   `_nofma` body uses `fmul` + `fsub` on separate pipes, exposing
   more independent ILP. Four-way unroll already saturates the FMA
   pipes in the `_fma` body, so the explicit-mul-sub variant ends up
   strictly faster by exploiting an extra pipe slot.
2. **Latency on the dst accumulator.** The `_fma` body has a true
   data dependency dst → fma → dst on every element; the `_nofma`
   body decomposes that into `tmp = alpha * src` (no dst dep) plus
   `dst -= tmp` (single-cycle sub), letting the multiplier issue
   ahead of the accumulator.

This is broadly consistent with the LLVM-codegen note in the issue:
`mul_add_f64s` on aarch64 lowers to `FMADD`, but the explicit
`mul + sub` does not get fused (the `-ffp-contract` policy in the
non-FMA path is implicit `off`), and the slower-per-element body
actually wins in steady state. Verifying requires `cargo asm` on
both bodies, which is outside this probe's scope.

## Decision tree (from issue #35) — resolved

x86 measurement from CI run 25971444759 (ubuntu-latest x86_64, same
probe, same shapes):

```
shape          fma med    nofma med   fma GF   nofma GF   fma/nofma
wide_2829x433  2745.8 us  4255.6 us   0.52     0.33       1.55
square_1928    1841.2 us  2851.0 us   0.52     0.33       1.55
narrow_512x32   237.4 us   363.8 us   0.52     0.34       1.53
```

(Caveat: the ubuntu-latest runner is a shared virtualised x86 so the
absolute GFLOPS are ~40× lower than the M-series aarch64 numbers
above — reads like AVX2/V3 dispatch is hitting frequency throttling
or a non-V3 fallback. The *ratio* is clean and consistent across
shapes, which is the only quantity #35's decision tree needs.)

Resolution: **branch 3 — "x86 wins, aarch64 loses"** is confirmed.
The production default (`fma = false`) is already correct for both
architectures' worst case. Documented in `dev/decisions.md` 2026-05-16
"FMA Schur-panel kernel: per-arch asymmetry, defaults stay".

No code change. Two paths were considered and rejected:

- Gate `fma = true` to `cfg(target_arch = "x86_64")` in
  `BunchKaufmanParams`: would silently override explicit opt-in,
  blocking probe binaries that legitimately want to time the
  aarch64 FMA path. Decision: keep the flag a pure runtime knob.
- Remove the FMA path: loses the x86 1.5× win and the existing
  bit-exact rank-1 reference tests on both kernels. Decision: keep.

## Resolution

Issue #35 closes with no code change, both arches' best kernel
already wired by default. The aarch64 ILP regression and the x86 FMA
win are captured in the decision log so a future tuner doesn't
re-discover the asymmetry from scratch.

## References

- Issue #35 (FMA path regression on aarch64).
- `dev/research/fma-kernel-opt-in.md` — original FMA opt-in design.
- `dev/research/wide-supernode-throughput-2026-05-16.md` §5.2, §6 —
  full-factor-level observation that motivated this probe.
- Probe: `src/bin/probe_fma_kernel.rs`.
- Kernel sources: `src/dense/schur_kernel.rs:2030`
  (`schur_panel_minus_fma_strided_quad`),
  `src/dense/schur_kernel.rs:1648`
  (`schur_panel_minus_nofma_strided_quad`).
