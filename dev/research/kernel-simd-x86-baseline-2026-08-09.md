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

## Baseline numbers

(to be filled below as Stage 0 proceeds)
