# BLAS reference probes (issue #153)

These three Fortran programs measure the *external* side of the #153
comparison. They are diagnostics, not part of the build: feral's core
solver has zero non-Rust dependencies (CLAUDE.md) and nothing here is
linked into the crate. They exist so the "how much of the machine does
a tuned BLAS get on our shapes" question is answered by measurement
instead of by citation.

All three link the OpenBLAS that ships with CoinHSL — the same library
MA57 uses in `external_benchmarks/ma57_oracle` — so the feral-vs-MA57
numbers and these numbers come from one library on one core.

Build (adjust `HSL` to your CoinHSL bundle):

    HSL=$HOME/Dropbox/projects/CoinHSL.v2023.11.17.aarch64-apple-darwin-libgfortran5
    gfortran -O2 -o dgemm_peak  dev/scripts/dgemm_peak.f  -L$HSL/lib -Wl,-rpath,$HSL/lib -lopenblas
    gfortran -O2 -o dsytrf_peak dev/scripts/dsytrf_peak.f -L$HSL/lib -Wl,-rpath,$HSL/lib -lopenblas
    gfortran -O2 -o blas_front  dev/scripts/blas_front.f  -L$HSL/lib -Wl,-rpath,$HSL/lib -lopenblas
    OPENBLAS_NUM_THREADS=1 ./blas_front

| program | measures | feral counterpart |
|---|---|---|
| `dgemm_peak.f`  | `dgemm` on square shapes — the machine's ceiling | (none; upper bound only) |
| `dsytrf_peak.f` | LAPACK blocked Bunch-Kaufman, square, full elimination | `diag_153_dense_peak` |
| `blas_front.f`  | `dsytrf` + `dtrsm` + `dsyrk` partial front on the exact `(nrow, ncol)` shapes the corpus produces — the kernel MA57 actually runs | `diag_153_kernel_headroom` per-shape `kernel_us` |

Two things to keep in mind when reading the output.

`SYSTEM_CLOCK` has 1 us granularity here, which is longer than a whole
small front. `blas_front.f` therefore batch-times `REPS` iterations and
subtracts a copy-only calibration loop; `dgemm_peak.f` and
`dsytrf_peak.f` take a min over repetitions and are unreliable below
n ~ 64 — read their small-n rows as noise, not as data.

The MMac/s column uses feral's `sum_{k<ncol}(nrow-k)^2` model so it is
directly comparable to `diag_153_dense_peak`,
`diag_153_kernel_headroom` and `diag_200_work_vs_ma57`. That model
counts the *full* trailing square, while a symmetric update only
touches a triangle, so it over-counts real work by up to ~2x on wide
trapezoidal fronts. It over-counts both sides identically, so ratios
are sound; absolute MMac/s should not be compared against a vendor's
published GFLOP/s.
