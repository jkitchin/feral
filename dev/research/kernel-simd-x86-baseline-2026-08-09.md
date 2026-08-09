# Kernel SIMD x86 baseline — Stage 0 measurements (2026-08-09)

Session: 2026-08-09-01. Container: x86_64, 4 cores, Intel Xeon @ 2.80GHz,
AVX2 + AVX-512 + FMA. Corpus not present; evidence from parity fixtures +
isolated harnesses. Build: `cargo build --release --all-targets`, default
profile (`opt-level=3`, no target-cpu, `debug=true` = debuginfo only).

## Motivation (from dev/ history)

- `apply_schur_panel_range_packed` (src/dense/factor.rs:3621-3821) is the
  default dense trailing-update since issue #99. decisions.md 2026-07-01
  lists its untuned headroom: explicit-SIMD, pack-buffer pooling, tile
  retune, cache blocking.
- The feral/MA57 small bucket (n=102-888) is the one clear loss: 4.23x
  geomean (external_benchmarks/comparison/REPORT.md). The small-front
  scalar path is ~87% of DENSEFACTOR_NS on the many-small-front corpus
  (dev/research/panel-fragmentation-2026-07-10.md).

## Finding 1 — the packed kernel compiles to SCALAR code on x86 (premise check)

Method: `objdump -d target/release/examples/bench_dense_front` over the
`apply_schur_panel_range` region (packed kernel is inlined into it,
0x315b0..0x356e0).

| metric | count |
|---|---:|
| `ymm` references (AVX) | **0** |
| packed SSE2 arithmetic (`mulpd`/`subpd`/`addpd`) | **0** |
| scalar `mulsd`/`subsd` | 187 |
| `xmm` references (scalar use only) | 1122 |

LLVM did not autovectorize the MR-axis inner loop at all — not even to
2-wide SSE2. Meanwhile the pulp strided kernels DO get AVX2 via runtime
dispatch (110 `vmulpd`/`vsubpd` in the binary, in `core_arch::x86::avx`
thunks reached from `pulp::x86::V3` dispatch).

Implication: on x86 the production default trailing update runs scalar
f64 arithmetic. Stage 1 (move the tile loop into `schur_kernel.rs` under
pulp dispatch) upgrades it to 4-wide AVX2 — a larger headroom than the
planned "SSE2→AVX2 ≈ 2x" estimate. On aarch64 (the project's usual bench
host) NEON is baseline so LLVM may already vectorize there; the aarch64
story must be re-measured on M-series (see cross-platform protocol).

## Baseline numbers (3-run, this container)

Test suite: `cargo test --release` all green (407 lib + integration suites).

### bench_schur_micro (isolated kernel; byte-exact check passes on all runs)

| shape | strided GF/s | packed GF/s | speedup |
|---|---:|---:|---:|
| 2048x2048 ke=64 | 0.45-0.46 | 6.04-6.18 | 13.3-13.8x |
| 512x512 ke=64 | 0.46 | 6.33-6.35 | 13.7x |
| 96x96 ke=32 | 0.47 | 6.03 | 12.8x |

Run-to-run variance <3%. 6 GFLOP/s ~= scalar mul+sub ILP peak at 2.8 GHz —
consistent with Finding 1 (scalar codegen). AVX2 leaves ~3-4x on the kernel.

### bench_dense_front (production blocked path, rayon=4)

| config | n=2955 (3 runs, ms) | GF/s | n=512 (ms) |
|---|---|---:|---|
| nofma serial | 4236 / 4303 / 4244 | 2.0 | 38.8-39.9 |
| nofma intrafront | 1798-1801 | 4.78 | ~39.5 (gate off) |
| fma serial | 12825-12949 | 0.66 | 71.2-71.9 |
| fma intrafront | 4214-4314 | 2.0 | ~71.5 |

**Finding 2 — opt-in FMA is a ~3x SLOWDOWN on x86 since the packed kernel
became default (2026-07-01).** The packed FMA variant uses scalar
`f64::mul_add` (factor.rs:3767/:3789); at baseline codegen (no `fma`
target feature) that lowers to a libm `fma()` call per element. The
2026-07-01 x86 measurement showing fma-serial 1.66x faster than nofma was
taken on the pre-packed pulp strided path (real `vfmadd` via V3 dispatch).
The default nofma path is unaffected. Stage 1 repairs this as a side
effect: pulp `mul_add_f64s` reaches real FMA through V3 runtime dispatch.

### perf_probe warm-factor medians (sequential driver, 3 runs)

| fixture | n | iters | median_us (r1/r2/r3) |
|---|---:|---:|---|
| AVION2_0251 | 64 | 300 | 39 / 39 / 39 |
| SWOPF_0000 | 175 | 300 | 180 / 177 / 176 |
| CERI651A_0000 | 190 | 300 | 251 / 239 / 246 |
| HYDCAR20_0000 | 198 | 300 | 295 / 300 / 299 |
| ACOPP30_0000 | 209 | 300 | 223 / 222 / 223 |
| HAHN1_0004 | 715 | 150 | 842 / 840 / 839 |
| CRESC100_0000 | 806 | 150 | 652 / 726 / (noisy) |
| VESUVIO_0021 | 3083 | 30 | 2108 / 2284 / — |
| twirism1_kkt | 745 | 150 | 4217 / 4176 / — |
| sawpath_kkt | 1575 | 100 | 830 / 826 / — |

All inertia_stable=true. cresc100/vesuvio show ~5-8% run noise; per the
2026-04-14 methodology rule, claims below 5% on those need extra runs.

### probe_panel_frag attribution (gates Stage 3)

| matrix | n | frag% | scal% | schur% | panelf% |
|---|---:|---:|---:|---:|---:|
| AVION2_0251 | 64 | 75.0 | 14.8 | 20.2 | 8.5 |
| SWOPF_0000 | 175 | 85.7 | 37.1 | 30.6 | 15.4 |
| CERI651A_0000 | 190 | 100.0 | 28.1 | 3.2 | 3.3 |
| HYDCAR20_0000 | 198 | 95.5 | 45.6 | 29.3 | 15.1 |
| ACOPP30_0000 | 209 | 70.6 | 25.4 | 25.1 | 20.1 |
| HAHN1_0004 | 715 | 100.0 | 30.0 | 1.1 | 0.5 |
| CRESC100_0000 | 806 | 66.7 | 15.6 | 3.1 | 2.0 |
| twirism1_kkt | 745 | 92.0 | 44.4 | 26.9 | 10.2 |
| sawpath_kkt | 1575 | 99.5 | 89.8 | 8.2 | 1.1 |

Aggregate: panels full=19 partial=331 (frag 94.6%); pivots inline=648
scalar=592 (47.7% scalar); schur=13.6% panel-factor=4.3% of aggregate
dense-front time. Confirms the panel-fragmentation-2026-07-10 corpus
steer: on small/KKT fronts the scalar pivot path, not the Schur update,
is the dominant kernel bucket → Stage 3 (eager-path SIMD) is justified;
Stage 1 primarily helps medium/large fronts (vesuvio, dense roots).

## Stage ordering confirmed by Stage 0

1. Stage 1 explicit pulp SIMD in packed kernel — scalar→AVX2, also
   repairs the x86 opt-in-FMA regression (Finding 2).
2. Stage 2 pack-buffer pooling (small/medium fronts, alloc traffic).
3. Stage 3 eager-path SIMD (scal% dominates small fixtures).
4. Stages 4-6 conditional on post-Stage-1 profiles.
