//! Verification probe for the perf-review Tier-1 #1 recommendation:
//! intra-front (node-level) parallelism of the trailing Schur update.
//!
//! Claims under test (dev/research/perf-review-2026-05-31.md). CORRECTNESS:
//! partitioning the trailing-column loop across threads is bit-exact for any
//! thread count / chunk size, because each output element a[i,j] is reduced
//! over the same pivot order on a single thread (no cross-thread reduction).
//! PERFORMANCE: the trailing update (the O(ncol*nrow^2) cost of a large front)
//! scales with cores.
//!
//! Method: build a representative dense front (column-major, the same layout
//! as `src/dense/factor.rs`), then run the production Schur kernel
//! (`schur_panel_minus_nofma_strided{,_dual,_quad}`) over the trailing columns
//! two ways — `apply_seq` (one contiguous pass, mirrors
//! `apply_blocked_schur_panel`) and `apply_par` (rayon `par_chunks_mut` over
//! disjoint column chunks). The chunk width is deliberately NOT a multiple of
//! 4 so the quad/dual/single grouping differs between the two paths; if the
//! result is still bit-identical, determinism cannot depend on grouping.
//!
//! Not wired into the library. `src/bin` is excluded from the published
//! crate. Run: `cargo run --release --bin probe_intrafront_schur`.

use feral::dense::schur_kernel::{
    schur_panel_minus_nofma_strided, schur_panel_minus_nofma_strided_dual,
    schur_panel_minus_nofma_strided_quad,
};
use std::time::Instant;

const MAX_N_ELIM: usize = 128;

/// Deterministic LCG → reproducible "random" matrix values.
struct Lcg(u64);
impl Lcg {
    fn next_f64(&mut self) -> f64 {
        // Numerical Recipes LCG constants.
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // map high bits to [-1, 1)
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
}

/// Build a front: column-major `nrow*nrow`. Columns `0..n_elim` are the
/// "already factored" pivot panel (the L columns the update reads);
/// columns `n_elim..nrow` are the trailing block the update writes.
/// `d` holds the n_elim pivot diagonals (all nonzero so no alpha is
/// skipped — worst-case work).
fn build_front(nrow: usize, n_elim: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = Lcg(seed);
    let mut a = vec![0.0f64; nrow * nrow];
    for col in 0..nrow {
        for row in col..nrow {
            a[col * nrow + row] = rng.next_f64();
        }
    }
    let mut d = vec![0.0f64; n_elim];
    for v in d.iter_mut() {
        let mut x = rng.next_f64();
        if x.abs() < 1e-3 {
            x += 1.0;
        }
        *v = x;
    }
    (a, d)
}

/// Process a contiguous range of trailing columns `[col_start, col_start+ncol)`
/// of `block` (column-major, `ncol` columns each of length `nrow`), reading
/// the read-only pivot panel `head` (length `n_elim*nrow`). Mirrors
/// `apply_blocked_schur_panel`: quad → dual → single fall-through. `k = 0`.
#[allow(clippy::too_many_arguments)]
fn apply_range(
    head: &[f64],
    block: &mut [f64],
    col_start: usize,
    ncol: usize,
    nrow: usize,
    n_elim: usize,
    d: &[f64],
) {
    let mut a0 = [0.0f64; MAX_N_ELIM];
    let mut a1 = [0.0f64; MAX_N_ELIM];
    let mut a2 = [0.0f64; MAX_N_ELIM];
    let mut a3 = [0.0f64; MAX_N_ELIM];

    let mut lc = 0usize; // column index local to `block`
    while lc + 3 < ncol {
        let j = col_start + lc;
        for q in 0..n_elim {
            let base = q * nrow;
            let dq = d[q];
            a0[q] = head[base + j] * dq;
            a1[q] = head[base + j + 1] * dq;
            a2[q] = head[base + j + 2] * dq;
            a3[q] = head[base + j + 3] * dq;
        }
        let (_done, rest) = block.split_at_mut(lc * nrow);
        let (c0, rest1) = rest.split_at_mut(nrow);
        let (c1, rest2) = rest1.split_at_mut(nrow);
        let (c2, c3) = rest2.split_at_mut(nrow);
        schur_panel_minus_nofma_strided_quad(
            &mut c0[j..],
            &mut c1[j + 1..nrow],
            &mut c2[j + 2..nrow],
            &mut c3[j + 3..nrow],
            head,
            0,
            n_elim,
            nrow,
            j,
            &a0[..n_elim],
            &a1[..n_elim],
            &a2[..n_elim],
            &a3[..n_elim],
        );
        lc += 4;
    }
    if lc + 1 < ncol {
        let j = col_start + lc;
        for q in 0..n_elim {
            let base = q * nrow;
            let dq = d[q];
            a0[q] = head[base + j] * dq;
            a1[q] = head[base + j + 1] * dq;
        }
        let (_done, rest) = block.split_at_mut(lc * nrow);
        let (c0, c1) = rest.split_at_mut(nrow);
        schur_panel_minus_nofma_strided_dual(
            &mut c0[j..],
            &mut c1[j + 1..nrow],
            head,
            0,
            n_elim,
            nrow,
            j,
            &a0[..n_elim],
            &a1[..n_elim],
        );
        lc += 2;
    }
    if lc < ncol {
        let j = col_start + lc;
        for q in 0..n_elim {
            a0[q] = head[q * nrow + j] * d[q];
        }
        let (_done, rest) = block.split_at_mut(lc * nrow);
        let len = nrow - j;
        schur_panel_minus_nofma_strided(
            &mut rest[j..nrow],
            head,
            0,
            n_elim,
            nrow,
            j,
            len,
            &a0[..n_elim],
        );
    }
}

/// Sequential: one pass over the whole trailing block.
fn apply_seq(a: &mut [f64], nrow: usize, n_elim: usize, d: &[f64]) {
    let (head, tail) = a.split_at_mut(n_elim * nrow);
    let ncol = nrow - n_elim;
    apply_range(head, tail, n_elim, ncol, nrow, n_elim, d);
}

/// Parallel: rayon over disjoint column chunks of the trailing block.
/// `chunk_cols` is intentionally not a multiple of 4.
fn apply_par(a: &mut [f64], nrow: usize, n_elim: usize, d: &[f64], chunk_cols: usize) {
    use rayon::prelude::*;
    let (head, tail) = a.split_at_mut(n_elim * nrow);
    let head: &[f64] = head;
    tail.par_chunks_mut(chunk_cols * nrow)
        .enumerate()
        .for_each(|(ci, block)| {
            let col_start = n_elim + ci * chunk_cols;
            let ncol = block.len() / nrow;
            apply_range(head, block, col_start, ncol, nrow, n_elim, d);
        });
}

fn bits_eq(a: &[f64], b: &[f64]) -> Option<usize> {
    if a.len() != b.len() {
        return Some(usize::MAX);
    }
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        if x.to_bits() != y.to_bits() {
            return Some(i);
        }
    }
    None
}

fn min_time<F: FnMut()>(reps: usize, mut f: F) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        f();
        let s = t.elapsed().as_secs_f64();
        if s < best {
            best = s;
        }
    }
    best
}

fn main() {
    let nrow = 2048usize;
    let n_elim = 64usize;
    let threads = rayon::current_num_threads();
    let (pristine, d) = build_front(nrow, n_elim, 0x1234_5678_9abc_def0);

    // FLOPs in one trailing update: sum over trailing cols of len*n_elim*2.
    let flops: f64 = {
        let mut f = 0.0;
        for j in n_elim..nrow {
            f += (nrow - j) as f64 * n_elim as f64 * 2.0;
        }
        f
    };

    println!("intra-front Schur update verification");
    println!("  front: nrow={nrow} n_elim={n_elim}  rayon threads={threads}");
    println!("  trailing-update flops/pass = {:.3e}\n", flops);

    // ---- CORRECTNESS: seq vs par, across chunk sizes and thread counts ----
    let mut ref_seq = pristine.clone();
    apply_seq(&mut ref_seq, nrow, n_elim, &d);

    let mut all_ok = true;
    for &chunk in &[64usize, 100, 137, 200, 512] {
        let mut buf = pristine.clone();
        apply_par(&mut buf, nrow, n_elim, &d, chunk);
        match bits_eq(&ref_seq, &buf) {
            None => println!("  [OK ] bit-exact: par(chunk={chunk}, T={threads}) == seq"),
            Some(i) => {
                all_ok = false;
                println!("  [BAD] chunk={chunk}: first differing element at flat index {i}");
            }
        }
    }
    // Thread-count invariance: 1, 2, 4 threads must all match seq.
    for &t in &[1usize, 2, 4] {
        if let Ok(pool) = rayon::ThreadPoolBuilder::new().num_threads(t).build() {
            let mut buf = pristine.clone();
            pool.install(|| apply_par(&mut buf, nrow, n_elim, &d, 137));
            match bits_eq(&ref_seq, &buf) {
                None => println!("  [OK ] bit-exact: par(chunk=137, T={t}) == seq"),
                Some(i) => {
                    all_ok = false;
                    println!("  [BAD] T={t}: first differing element at flat index {i}");
                }
            }
        }
    }
    println!(
        "\n  CORRECTNESS: {}\n",
        if all_ok {
            "PASS (bit-identical across all configs)"
        } else {
            "FAIL"
        }
    );

    // ---- PERFORMANCE: seq vs par wall time (best of N, fresh copy each rep) ----
    let reps = 7;
    let t_seq = min_time(reps, || {
        let mut buf = pristine.clone();
        apply_seq(&mut buf, nrow, n_elim, &d);
        std::hint::black_box(&buf);
    });
    let t_par = min_time(reps, || {
        let mut buf = pristine.clone();
        apply_par(&mut buf, nrow, n_elim, &d, 137);
        std::hint::black_box(&buf);
    });
    // Subtract clone cost so we time the update, not the memcpy.
    let t_clone = min_time(reps, || {
        let buf = pristine.clone();
        std::hint::black_box(&buf);
    });
    let seq = (t_seq - t_clone).max(1e-9);
    let par = (t_par - t_clone).max(1e-9);

    println!("  clone-only           : {:8.3} ms", t_clone * 1e3);
    println!(
        "  seq update           : {:8.3} ms  ({:6.2} GFLOP/s)",
        seq * 1e3,
        flops / seq / 1e9
    );
    println!(
        "  par update (T={threads})       : {:8.3} ms  ({:6.2} GFLOP/s)",
        par * 1e3,
        flops / par / 1e9
    );
    println!(
        "  speedup              : {:8.2}x  (ideal {threads}x)",
        seq / par
    );
}
