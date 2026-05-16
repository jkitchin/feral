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

x86 measurement is pending (probe ships in this commit; CI run on a
Linux x86 runner will resolve the decision-tree branch).

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

## Decision tree (from issue #35) — current state

- **aarch64-only regression** → gate `fma = true` behind
  `cfg(target_arch = "x86_64")`. Awaiting x86 measurement. **(probable)**
- **Regression on both** → remove the FMA path entirely. **(possible)**
- **x86 wins, aarch64 loses** → no-op; per-arch default is already
  `fma = false`. Document the asymmetry in `dev/decisions.md`. **(possible)**

x86 verification: easiest path is a CI run of `cargo run --release
--bin probe_fma_kernel` on the existing `check` job (ubuntu-latest,
x86_64). Adding two lines to `.github/workflows/ci.yml` would surface
the numbers in CI logs for the cost of one extra release-mode build
step. Not done in this commit — let user prioritise against the
existing CI budget.

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
