//! Diagnostic micro-bench for tiny packed-tile calls (session
//! 2026-08-09): times `schur_kernel::packed_schur_tiles_nofma` on the
//! degenerate shape observed on HAHN1 (nrow=10, col_start=9, ncol=1,
//! n_elim=7 — one active element) against a plain scalar walk, to
//! isolate per-call overhead of the pulp dispatch boundary.
//!
//! Usage: cargo run --release --example bench_packed_tiny [-- REPS]

use feral::dense::schur_kernel::{packed_schur_tiles_nofma, PACKED_MR, PACKED_NR};
use std::time::Instant;

fn main() {
    let reps: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1_000_000);

    let nrow = 10usize;
    let col_start = 9usize;
    let ncol = 1usize;
    let n_elim = 7usize;
    let rowspan = nrow - col_start;
    let npanels_i = rowspan.div_ceil(PACKED_MR);
    let npanels_j = ncol.div_ceil(PACKED_NR);

    let apack: Vec<f64> = (0..npanels_i * n_elim * PACKED_MR)
        .map(|i| (i as f64).sin())
        .collect();
    let bpack0: Vec<f64> = (0..npanels_j * n_elim * PACKED_NR)
        .map(|i| (i as f64).cos())
        .collect();
    let bpack1: Vec<f64> = Vec::new();
    let d_panel: Vec<f64> = (0..n_elim).map(|i| 1.0 + i as f64).collect();
    let subdiag_k = vec![0.0f64; n_elim];
    let mut block = vec![1.0f64; ncol * nrow];

    let t = Instant::now();
    for _ in 0..reps {
        packed_schur_tiles_nofma(
            &mut block, &apack, &bpack0, &bpack1, &d_panel, &subdiag_k, nrow, col_start,
        );
        std::hint::black_box(&mut block);
    }
    let simd_ns = t.elapsed().as_nanos() as f64 / reps as f64;

    // Scalar walk of the same tile (reference semantics).
    let mut block2 = vec![1.0f64; ncol * nrow];
    let col_end = col_start + ncol;
    let t = Instant::now();
    for _ in 0..reps {
        for pj in 0..npanels_j {
            let j0 = col_start + pj * PACKED_NR;
            let bbase = pj * (n_elim * PACKED_NR);
            for pi in 0..npanels_i {
                let i0 = col_start + pi * PACKED_MR;
                if i0 + PACKED_MR <= j0 {
                    continue;
                }
                let abase = pi * (n_elim * PACKED_MR);
                let mut acc = [[0.0f64; PACKED_MR]; PACKED_NR];
                for (jr, accj) in acc.iter_mut().enumerate() {
                    let j = j0 + jr;
                    if j >= col_end {
                        continue;
                    }
                    let colblk = (j - col_start) * nrow;
                    for (ir, a) in accj.iter_mut().enumerate() {
                        let i = i0 + ir;
                        if i < nrow && i >= j {
                            *a = block2[colblk + i];
                        }
                    }
                }
                let mut q = 0usize;
                while q < n_elim {
                    if d_panel[q] != 0.0 {
                        let a0 = &apack[abase + q * PACKED_MR..][..PACKED_MR];
                        let b0 = &bpack0[bbase + q * PACKED_NR..][..PACKED_NR];
                        for (jr, accj) in acc.iter_mut().enumerate() {
                            let bj = b0[jr];
                            for (ir, acci) in accj.iter_mut().enumerate() {
                                *acci -= bj * a0[ir];
                            }
                        }
                    }
                    q += 1;
                }
                for (jr, accj) in acc.iter().enumerate() {
                    let j = j0 + jr;
                    if j >= col_end {
                        continue;
                    }
                    let colblk = (j - col_start) * nrow;
                    for (ir, a) in accj.iter().enumerate() {
                        let i = i0 + ir;
                        if i < nrow && i >= j {
                            block2[colblk + i] = *a;
                        }
                    }
                }
            }
        }
        std::hint::black_box(&mut block2);
    }
    let scalar_ns = t.elapsed().as_nanos() as f64 / reps as f64;

    assert_eq!(
        block[9].to_bits(),
        block2[9].to_bits(),
        "paths diverged (byte check)"
    );
    println!("tiny shape (nrow=10 ncol=1 ke=7): simd {simd_ns:.1} ns/call, scalar {scalar_ns:.1} ns/call");

    // Frequency-license experiment (env FERAL_TINY_LICENSE=1): interleave
    // ONE tiny 256-bit kernel call with ~700 µs of scalar busywork per
    // "iteration", mimicking the HAHN1 warm-factor profile, and compare
    // the busywork's own wall time against a pure-scalar control. If the
    // isolated ymm burst taxes the whole iteration (license transition +
    // downclock window), the combo runs measurably slower than control.
    if std::env::var("FERAL_TINY_LICENSE").is_ok() {
        let iters = 500usize;
        let busy = |x0: f64| {
            // ~700 µs of scalar dependent work (not vectorizable).
            let mut x = x0;
            for i in 0..400_000u64 {
                x = (x * 1.000000001 + (i & 7) as f64).sqrt();
            }
            x
        };
        // Control: busywork only.
        let mut acc = 0.0f64;
        let mut ctrl = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t = Instant::now();
            acc += busy(acc.fract() + 1.0);
            ctrl.push(t.elapsed().as_micros() as u64);
        }
        // Combo: one tiny simd call, then the same busywork.
        let mut combo = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t = Instant::now();
            packed_schur_tiles_nofma(
                &mut block, &apack, &bpack0, &bpack1, &d_panel, &subdiag_k, nrow, col_start,
            );
            std::hint::black_box(&mut block);
            acc += busy(acc.fract() + 1.0);
            combo.push(t.elapsed().as_micros() as u64);
        }
        std::hint::black_box(acc);
        let stats = |mut v: Vec<u64>| {
            v.sort_unstable();
            let n = v.len() - 1;
            (v[0], v[n / 2], v[n * 9 / 10])
        };
        let (c0, c50, c90) = stats(ctrl);
        let (s0, s50, s90) = stats(combo);
        println!("license probe: control busywork  p0={c0} p50={c50} p90={c90} us");
        println!("license probe: simd+busywork     p0={s0} p50={s50} p90={s90} us");
    }
}
