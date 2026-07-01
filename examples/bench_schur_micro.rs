//! Isolated throughput microbenchmark for the dense trailing-update
//! kernel (issue #99, BLAS-3 follow-up).
//!
//! The full-front profile attributes ~94% of a dense factor to the
//! trailing Schur update yet measures only ~0.33 GFLOP/s — far below
//! even scalar peak. This harness isolates the *kernel* from the
//! factorization (pivot search, alpha precompute, per-panel setup) to
//! answer one question: is the kernel itself slow, or is it the
//! surrounding factorization overhead?
//!
//! It computes a rectangular rank-`ke` update `C[i,j] -= Σ_q L[i,q] *
//! alpha[j,q]` (`C` is m×n, `L` is m×ke, `alpha` is n×ke) two ways:
//!   * BASELINE — the production per-column strided kernel
//!     (`schur_panel_minus_nofma_strided`), one dispatch per column,
//!     reading the panel at column-stride `m` (the strided access the
//!     factor uses).
//!   * PACKED   — pack `L` into MR-row micro-panels (contiguous in the
//!     `q` loop) and run a plain MR×NR register-tiled kernel. Same
//!     per-element `mul → sub` order ⇒ byte-exact with BASELINE.
//!
//! Reports ms and GFLOP/s for each, plus a byte-exactness check. If
//! PACKED is much faster, the strided `q`-access is the bottleneck and a
//! packed micro-kernel is worth wiring in; if not, the ceiling is DST
//! bandwidth (→ Phase C cache-blocked factor).
//!
//! Usage: cargo run --release --example bench_schur_micro [-- M N KE REPS]

use feral::dense::schur_kernel::schur_panel_minus_nofma_strided;
use std::time::Instant;

fn mix(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
}

/// BASELINE: production per-column strided kernel, one call per column.
fn run_baseline(c: &mut [f64], l: &[f64], alpha: &[f64], m: usize, n: usize, ke: usize) {
    let mut alphas = vec![0.0f64; ke];
    for j in 0..n {
        for q in 0..ke {
            alphas[q] = alpha[j + q * n];
        }
        let dst = &mut c[j * m..j * m + m];
        // src_block = L (col-major, col-stride m); src_first_col 0,
        // src_row_offset 0, len m.
        schur_panel_minus_nofma_strided(dst, l, 0, ke, m, 0, m, &alphas);
    }
}

/// Pack `L` (m×ke col-major) into MR-row micro-panels: for panel p
/// (rows p*MR..), store q-major MR-contiguous blocks so the kernel's
/// inner q-loop reads sequential memory. `apack[p*(ke*MR) + q*MR + ir]`.
const MR: usize = 8;
const NR: usize = 4;

fn pack_a(l: &[f64], m: usize, ke: usize) -> Vec<f64> {
    let npanels = m.div_ceil(MR);
    let mut apack = vec![0.0f64; npanels * ke * MR];
    for p in 0..npanels {
        let i0 = p * MR;
        for q in 0..ke {
            for ir in 0..MR {
                let i = i0 + ir;
                let v = if i < m { l[i + q * m] } else { 0.0 };
                apack[p * (ke * MR) + q * MR + ir] = v;
            }
        }
    }
    apack
}

fn pack_b(alpha: &[f64], n: usize, ke: usize) -> Vec<f64> {
    let npanels = n.div_ceil(NR);
    let mut bpack = vec![0.0f64; npanels * ke * NR];
    for p in 0..npanels {
        let j0 = p * NR;
        for q in 0..ke {
            for jr in 0..NR {
                let j = j0 + jr;
                let v = if j < n { alpha[j + q * n] } else { 0.0 };
                bpack[p * (ke * NR) + q * NR + jr] = v;
            }
        }
    }
    bpack
}

/// PACKED: MR×NR register-tiled kernel over packed operands. Plain
/// arrays; LLVM autovectorizes the contiguous MR inner axis. Per
/// element `acc -= a*b` in ascending q — byte-exact with BASELINE.
#[allow(clippy::needless_range_loop)]
fn run_packed(c: &mut [f64], apack: &[f64], bpack: &[f64], m: usize, n: usize, ke: usize) {
    let np_j = n.div_ceil(NR);
    for pj in 0..np_j {
        let j0 = pj * NR;
        let bbase = pj * (ke * NR);
        let np_i = m.div_ceil(MR);
        for pi in 0..np_i {
            let i0 = pi * MR;
            let abase = pi * (ke * MR);
            // MR×NR accumulator tile, loaded from C.
            let mut acc = [[0.0f64; MR]; NR];
            for (jr, accj) in acc.iter_mut().enumerate() {
                let j = j0 + jr;
                if j >= n {
                    continue;
                }
                for ir in 0..MR {
                    let i = i0 + ir;
                    accj[ir] = if i < m { c[i + j * m] } else { 0.0 };
                }
            }
            for q in 0..ke {
                let a = &apack[abase + q * MR..abase + q * MR + MR];
                for jr in 0..NR {
                    let b = bpack[bbase + q * NR + jr];
                    let accj = &mut acc[jr];
                    for ir in 0..MR {
                        accj[ir] -= b * a[ir];
                    }
                }
            }
            for (jr, accj) in acc.iter().enumerate() {
                let j = j0 + jr;
                if j >= n {
                    continue;
                }
                for ir in 0..MR {
                    let i = i0 + ir;
                    if i < m {
                        c[i + j * m] = accj[ir];
                    }
                }
            }
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let m: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1500);
    let n: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1500);
    let ke: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(64);
    let reps: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(3);

    let mut s = 0xDEAD_BEEF_1234u64;
    let l: Vec<f64> = (0..m * ke).map(|_| mix(&mut s)).collect();
    let alpha: Vec<f64> = (0..n * ke).map(|_| mix(&mut s)).collect();
    let c0: Vec<f64> = (0..m * n).map(|_| mix(&mut s)).collect();

    let gflop = 2.0 * m as f64 * n as f64 * ke as f64 / 1e9;
    println!("bench_schur_micro: m={m} n={n} ke={ke} reps={reps}  ({gflop:.3} GFLOP/update)");

    // BASELINE
    let mut best_base = f64::INFINITY;
    let mut c_base = c0.clone();
    for _ in 0..reps {
        c_base.copy_from_slice(&c0);
        let t = Instant::now();
        run_baseline(&mut c_base, &l, &alpha, m, n, ke);
        best_base = best_base.min(t.elapsed().as_secs_f64());
    }
    println!(
        "  baseline (strided per-col) {:8.2} ms   {:6.2} GFLOP/s",
        best_base * 1e3,
        gflop / best_base
    );

    // PACKED (packing timed separately + fused)
    let mut best_pack = f64::INFINITY;
    let mut best_pack_all = f64::INFINITY;
    let mut c_pack = c0.clone();
    for _ in 0..reps {
        c_pack.copy_from_slice(&c0);
        let t_all = Instant::now();
        let apack = pack_a(&l, m, ke);
        let bpack = pack_b(&alpha, n, ke);
        let t_k = Instant::now();
        run_packed(&mut c_pack, &apack, &bpack, m, n, ke);
        best_pack = best_pack.min(t_k.elapsed().as_secs_f64());
        best_pack_all = best_pack_all.min(t_all.elapsed().as_secs_f64());
    }
    println!(
        "  packed kernel only         {:8.2} ms   {:6.2} GFLOP/s",
        best_pack * 1e3,
        gflop / best_pack
    );
    println!(
        "  packed incl. pack          {:8.2} ms   {:6.2} GFLOP/s",
        best_pack_all * 1e3,
        gflop / best_pack_all
    );

    // Byte-exactness check.
    let mut mism = 0usize;
    for (x, y) in c_base.iter().zip(&c_pack) {
        if x.to_bits() != y.to_bits() {
            mism += 1;
        }
    }
    if mism == 0 {
        println!("  byte-exact: baseline == packed ✓");
    } else {
        println!("  BYTE MISMATCH in {mism} / {} elements ✗", c_base.len());
    }
    println!(
        "  speedup (kernel only): {:.2}×   (incl. pack): {:.2}×",
        best_base / best_pack,
        best_base / best_pack_all
    );
}
