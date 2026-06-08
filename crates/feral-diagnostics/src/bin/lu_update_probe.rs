//! Warm-solve scaling probe for the sparse LU under update chains (issue #81).
//!
//! The Forrest–Tomlin update stores a bump-local eta (the elimination of the
//! spike), so warm `ftran` cost grows with the *bump* size, not `O(k·n)` like
//! the old product-form (dense `τ`). This probe demonstrates that on a
//! block-diagonal basis (localized spikes — the realistic LP regime): with a
//! fixed block size, the per-update eta and the `ftran` cost are independent of
//! `n`. A tridiagonal basis (dense `L⁻¹` ⇒ dense spike) is the honest worst
//! case where the bump spans the tail.
//!
//! Run: `cargo run -p feral-diagnostics --release --bin lu_update_probe`

use feral::lu::sparse_matrix::SparseColMatrix;
use feral::lu::{LuParams, SparseLu, SparseLuSymbolic};
use std::time::Instant;

/// Block-diagonal basis: `nblocks` diagonally-dominant dense blocks of size
/// `bs`. With natural ordering, `L` is block-diagonal so a within-block update
/// has a spike confined to its block.
fn block_diag(nblocks: usize, bs: usize, seed: u64) -> SparseColMatrix {
    let n = nblocks * bs;
    let mut state = seed;
    let mut rng = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((state >> 33) as f64) / (1u64 << 31) as f64 - 1.0
    };
    let mut cols: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for b in 0..nblocks {
        let base = b * bs;
        for cc in 0..bs {
            let j = base + cc;
            for rr in 0..bs {
                let i = base + rr;
                let v = if rr == cc { 5.0 + rng().abs() } else { rng() };
                cols[j].push((i, v));
            }
        }
    }
    SparseColMatrix::from_sparse_columns(n, &cols).expect("block_diag")
}

fn within_block_col(n: usize, bs: usize, slot: usize, seed: u64) -> Vec<f64> {
    let mut state = seed;
    let mut rng = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((state >> 33) as f64) / (1u64 << 31) as f64 - 1.0
    };
    let base = (slot / bs) * bs;
    let mut c = vec![0.0; n];
    for rr in 0..bs {
        let i = base + rr;
        c[i] = if base + (slot - base) == i {
            5.0 + rng().abs()
        } else {
            rng()
        };
    }
    c
}

fn avg_ftran_us(lu: &mut SparseLu, n: usize, reps: usize) -> f64 {
    let mut total = 0.0;
    for t in 0..reps {
        let mut rhs: Vec<f64> = (0..n).map(|i| 1.0 + ((i + t) % 7) as f64).collect();
        let t0 = Instant::now();
        lu.ftran(&mut rhs).expect("ftran");
        total += t0.elapsed().as_secs_f64() * 1e6;
    }
    total / reps as f64
}

fn main() {
    let bs = 25;
    let k = 50; // updates
    println!("Block-diagonal basis, block size {bs}, {k} within-block updates:");
    println!(
        "{:>8}  {:>14}  {:>14}  {:>16}",
        "n", "ftran@0(µs)", "ftran@k(µs)", "eta_ops_total"
    );
    for &nblocks in &[40usize, 80, 160, 320] {
        let n = nblocks * bs;
        let a = block_diag(nblocks, bs, 0x100 + nblocks as u64);
        let symbolic = SparseLuSymbolic::natural(n);
        let params = LuParams {
            max_updates: 100_000,
            ..LuParams::default()
        };
        let mut lu = SparseLu::factor(&a, &symbolic, params).expect("factor");
        let ft0 = avg_ftran_us(&mut lu, n, 30);
        for s in 0..k {
            let slot = ((s * 53 + 7) % nblocks) * bs + (s % bs);
            let col = within_block_col(n, bs, slot, 0xABC + s as u64);
            // Some updates may legitimately hit a singular within-block basis;
            // skip those for the timing demo.
            let _ = lu.update(slot, &col);
        }
        let ftk = avg_ftran_us(&mut lu, n, 30);
        println!(
            "{:>8}  {:>14.2}  {:>14.2}  {:>16}",
            n,
            ft0,
            ftk,
            lu.eta_ops()
        );
    }
    println!("(ftran@k and eta_ops should be ~flat in n — FT cost is bump-local, not O(n).)");
}
