//! Phase 2.4.2 Step 4 — microbenchmark comparing the scalar reference
//! AXPY loops against the pulp-dispatched SIMD kernels in
//! `src/dense/schur_kernel.rs`.
//!
//! This bench measures the kernels in isolation. The end-to-end
//! `do_1x1_update` comparison is deferred to Step 6 (full KKT bench),
//! because Step 5 is what wires the SIMD kernel into `factor.rs`.
//!
//! The Step 4 gate: the SIMD path must be ≥ 2× faster than the scalar
//! reference at L = 256 for `axpy_minus`. On Apple Silicon NEON (the
//! dev machine) a 2 doubles/cycle FMA ceiling suggests roughly 2×
//! speedup is plausible; on AVX2+FMA boxes we expect 4×, on AVX-512
//! 8×. The numbers for L ∈ {8, 16, 32} also calibrate the length
//! threshold for the Step 5 scalar-fallback guard.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use feral::dense::schur_kernel::{axpy2_minus, axpy_minus};
#[cfg(target_arch = "aarch64")]
use feral::dense::schur_kernel::{
    axpy2_minus_direct, axpy2_minus_unroll4, axpy2_minus_unroll4_nofma, axpy_minus_direct,
    axpy_minus_unroll4, axpy_minus_unroll4_nofma,
};

/// Minimal xorshift64 for reproducible bench inputs.
struct Xorshift64(u64);

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        Self(if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        })
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn next_f64(&mut self) -> f64 {
        let bits = (self.next_u64() >> 12) | 0x3FF0_0000_0000_0000;
        let x = f64::from_bits(bits) - 1.0;
        2.0 * x - 1.0
    }
}

/// Scalar reference — identical body to the test-module naive path.
/// Kept here (duplicated) so the bench does not depend on a test-only
/// helper. The compiler will autovectorize this loop to some extent
/// on release, which makes the comparison against pulp an honest
/// measurement of "pulp SIMD vs whatever the compiler does with a
/// straight loop".
#[inline(never)]
fn scalar_axpy_minus(dst: &mut [f64], src: &[f64], alpha: f64) {
    for i in 0..dst.len() {
        let tmp = alpha * src[i];
        dst[i] -= tmp;
    }
}

#[inline(never)]
fn scalar_axpy2_minus(dst: &mut [f64], src0: &[f64], alpha0: f64, src1: &[f64], alpha1: f64) {
    for i in 0..dst.len() {
        let t0 = alpha0 * src0[i];
        let t1 = alpha1 * src1[i];
        dst[i] -= t0 + t1;
    }
}

const LENGTHS: &[usize] = &[8, 16, 32, 64, 128, 256, 512, 1024, 2048];

fn make_vec(rng: &mut Xorshift64, len: usize) -> Vec<f64> {
    (0..len).map(|_| rng.next_f64()).collect()
}

fn bench_axpy_minus(c: &mut Criterion) {
    let mut group = c.benchmark_group("axpy_minus");
    let mut rng = Xorshift64::new(0xA1B2_C3D4_E5F6_0789);

    for &len in LENGTHS {
        let src = make_vec(&mut rng, len);
        let dst_init = make_vec(&mut rng, len);
        let alpha = rng.next_f64() * 1.5;

        group.throughput(Throughput::Elements(len as u64));

        let dst_s = dst_init.clone();
        group.bench_with_input(
            BenchmarkId::new("scalar", len),
            &(src.clone(), dst_s, alpha),
            |b, (s, d, a)| {
                let s = s.clone();
                let mut d = d.clone();
                b.iter(|| {
                    scalar_axpy_minus(black_box(&mut d), black_box(&s), black_box(*a));
                });
            },
        );

        let dst_p = dst_init.clone();
        group.bench_with_input(
            BenchmarkId::new("pulp", len),
            &(src.clone(), dst_p, alpha),
            |b, (s, d, a)| {
                let s = s.clone();
                let mut d = d.clone();
                b.iter(|| {
                    axpy_minus(black_box(&mut d), black_box(&s), black_box(*a));
                });
            },
        );

        #[cfg(target_arch = "aarch64")]
        {
            let dst_d = dst_init.clone();
            group.bench_with_input(
                BenchmarkId::new("direct_neon", len),
                &(src.clone(), dst_d, alpha),
                |b, (s, d, a)| {
                    let s = s.clone();
                    let mut d = d.clone();
                    b.iter(|| {
                        axpy_minus_direct(black_box(&mut d), black_box(&s), black_box(*a));
                    });
                },
            );

            let dst_u = dst_init.clone();
            group.bench_with_input(
                BenchmarkId::new("unroll4_neon", len),
                &(src.clone(), dst_u, alpha),
                |b, (s, d, a)| {
                    let s = s.clone();
                    let mut d = d.clone();
                    b.iter(|| {
                        axpy_minus_unroll4(black_box(&mut d), black_box(&s), black_box(*a));
                    });
                },
            );

            // Wired-in production kernel: unroll4 with separate mul+sub
            // (no FMA) so results are bit-identical to scalar.
            let dst_n = dst_init.clone();
            group.bench_with_input(
                BenchmarkId::new("unroll4_nofma_neon", len),
                &(src.clone(), dst_n, alpha),
                |b, (s, d, a)| {
                    let s = s.clone();
                    let mut d = d.clone();
                    b.iter(|| {
                        axpy_minus_unroll4_nofma(black_box(&mut d), black_box(&s), black_box(*a));
                    });
                },
            );
        }
    }

    group.finish();
}

fn bench_axpy2_minus(c: &mut Criterion) {
    let mut group = c.benchmark_group("axpy2_minus");
    let mut rng = Xorshift64::new(0xDEAD_BEEF_CAFE_BABE);

    for &len in LENGTHS {
        let src0 = make_vec(&mut rng, len);
        let src1 = make_vec(&mut rng, len);
        let dst_init = make_vec(&mut rng, len);
        let alpha0 = rng.next_f64() * 1.5;
        let alpha1 = rng.next_f64() * 1.5;

        group.throughput(Throughput::Elements(len as u64));

        let dst_s = dst_init.clone();
        group.bench_with_input(
            BenchmarkId::new("scalar", len),
            &(src0.clone(), src1.clone(), dst_s, alpha0, alpha1),
            |b, (s0, s1, d, a0, a1)| {
                let s0 = s0.clone();
                let s1 = s1.clone();
                let mut d = d.clone();
                b.iter(|| {
                    scalar_axpy2_minus(
                        black_box(&mut d),
                        black_box(&s0),
                        black_box(*a0),
                        black_box(&s1),
                        black_box(*a1),
                    );
                });
            },
        );

        let dst_p = dst_init.clone();
        group.bench_with_input(
            BenchmarkId::new("pulp", len),
            &(src0.clone(), src1.clone(), dst_p, alpha0, alpha1),
            |b, (s0, s1, d, a0, a1)| {
                let s0 = s0.clone();
                let s1 = s1.clone();
                let mut d = d.clone();
                b.iter(|| {
                    axpy2_minus(
                        black_box(&mut d),
                        black_box(&s0),
                        black_box(*a0),
                        black_box(&s1),
                        black_box(*a1),
                    );
                });
            },
        );

        #[cfg(target_arch = "aarch64")]
        {
            let dst_d = dst_init.clone();
            group.bench_with_input(
                BenchmarkId::new("direct_neon", len),
                &(src0.clone(), src1.clone(), dst_d, alpha0, alpha1),
                |b, (s0, s1, d, a0, a1)| {
                    let s0 = s0.clone();
                    let s1 = s1.clone();
                    let mut d = d.clone();
                    b.iter(|| {
                        axpy2_minus_direct(
                            black_box(&mut d),
                            black_box(&s0),
                            black_box(*a0),
                            black_box(&s1),
                            black_box(*a1),
                        );
                    });
                },
            );

            let dst_u = dst_init.clone();
            group.bench_with_input(
                BenchmarkId::new("unroll4_neon", len),
                &(src0.clone(), src1.clone(), dst_u, alpha0, alpha1),
                |b, (s0, s1, d, a0, a1)| {
                    let s0 = s0.clone();
                    let s1 = s1.clone();
                    let mut d = d.clone();
                    b.iter(|| {
                        axpy2_minus_unroll4(
                            black_box(&mut d),
                            black_box(&s0),
                            black_box(*a0),
                            black_box(&s1),
                            black_box(*a1),
                        );
                    });
                },
            );

            let dst_n = dst_init.clone();
            group.bench_with_input(
                BenchmarkId::new("unroll4_nofma_neon", len),
                &(src0.clone(), src1.clone(), dst_n, alpha0, alpha1),
                |b, (s0, s1, d, a0, a1)| {
                    let s0 = s0.clone();
                    let s1 = s1.clone();
                    let mut d = d.clone();
                    b.iter(|| {
                        axpy2_minus_unroll4_nofma(
                            black_box(&mut d),
                            black_box(&s0),
                            black_box(*a0),
                            black_box(&s1),
                            black_box(*a1),
                        );
                    });
                },
            );
        }
    }

    group.finish();
}

criterion_group!(benches, bench_axpy_minus, bench_axpy2_minus);
criterion_main!(benches);
