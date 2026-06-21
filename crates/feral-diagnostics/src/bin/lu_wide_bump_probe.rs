//! Wide-bump update scaling probe (issue #87).
//!
//! Reproduces the `autocorr_bern` / `casctanks` regime that drives
//! `SparseLu::update` / `eliminate_bump` into its O(bump²) worst case: a
//! **sparse** `U` but a **dense entering column**, so the spike `ρ = G⁻¹L⁻¹Pa`
//! is dense and the bump spans (almost) the whole trailing factor.
//!
//! Construction: a tridiagonal basis (natural order ⇒ `L`,`U` bidiagonal, hence
//! sparse), updated by replacing an *early* slot with a dense column. Because
//! `L⁻¹` of a tridiagonal is dense lower-triangular, the spike is dense and the
//! bump is `[r, m-1]` — the full trailing block — while `U` itself stays sparse.
//! This isolates the residual super-linear *row-elimination* term (issue #87),
//! distinct from genuine fill.
//!
//! The headline number is **µs/update vs m** and the empirical exponent: the
//! current full-bump re-triangularization is ~O(m²); the Forrest–Tomlin
//! row-elimination target is ~O(m).
//!
//! Run: `cargo run -p feral-diagnostics --release --bin lu_wide_bump_probe`

use feral::lu::sparse_matrix::SparseColMatrix;
use feral::lu::{LuParams, SparseLu, SparseLuSymbolic};
use std::time::Instant;

/// Tridiagonal basis: diagonal 4, off-diagonals -1. Diagonally dominant, so the
/// natural-order factorization is stable and `L`,`U` are bidiagonal (sparse).
fn tridiag(m: usize) -> SparseColMatrix {
    let mut cols: Vec<Vec<(usize, f64)>> = vec![Vec::new(); m];
    for (j, col) in cols.iter_mut().enumerate() {
        if j > 0 {
            col.push((j - 1, -1.0));
        }
        col.push((j, 4.0));
        if j + 1 < m {
            col.push((j + 1, -1.0));
        }
    }
    SparseColMatrix::from_sparse_columns(m, &cols).expect("tridiag")
}

/// A dense, well-conditioned entering column: small spread of values with a
/// dominant entry at `slot` so the replacement basis stays nonsingular.
fn dense_col(m: usize, slot: usize, seed: u64) -> Vec<f64> {
    let mut state = seed;
    let mut rng = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((state >> 33) as f64) / (1u64 << 31) as f64 - 1.0
    };
    let mut c: Vec<f64> = (0..m).map(|_| 0.5 + rng()).collect();
    c[slot] = 50.0 + rng().abs();
    c
}

fn main() {
    let k = 20; // updates per size
    println!("Wide-bump (dense-spike) update scaling — tridiagonal basis, {k} updates:");
    println!(
        "{:>7}  {:>12}  {:>12}  {:>14}  {:>12}  {:>10}",
        "m", "factor_nnz", "us/update", "us/update/m", "eta_ops/upd", "committed"
    );
    let mut prev: Option<(usize, f64)> = None;
    for &m in &[250usize, 500, 1000, 2000, 4000] {
        let a = tridiag(m);
        // Natural order keeps L,U bidiagonal so the bump is dense-spike, not fill.
        let symbolic = SparseLuSymbolic::natural(m);
        let params = LuParams {
            max_updates: 1_000_000,
            max_growth: 1e30,
            ..LuParams::default()
        };
        let mut lu = SparseLu::factor(&a, &symbolic, params).expect("factor");
        let factor_nnz = lu.factor_nnz();

        // Replace early slots (cycling among the first few) with dense columns.
        let mut total = 0.0;
        let mut committed = 0usize;
        let mut eta_ops_acc = 0usize;
        for s in 0..k {
            let slot = s % 4; // early slots ⇒ wide bump [r, m-1]
            let col = dense_col(m, slot, 0xC0FFEE + s as u64 + m as u64 * 7);
            let t0 = Instant::now();
            let res = lu.update(slot, &col);
            total += t0.elapsed().as_secs_f64() * 1e6;
            if res.is_ok() {
                committed += 1;
                eta_ops_acc += lu.last_eta_ops();
            }
        }
        let per = total / k as f64;
        let eta_avg = if committed > 0 {
            eta_ops_acc as f64 / committed as f64
        } else {
            0.0
        };
        let exponent =
            prev.map(|(pm, pper)| (per / pper).log2() / ((m as f64) / (pm as f64)).log2());
        println!(
            "{:>7}  {:>12}  {:>12.2}  {:>14.4}  {:>12.0}  {:>10}{}",
            m,
            factor_nnz,
            per,
            per / m as f64,
            eta_avg,
            committed,
            exponent
                .map(|e| format!("   exp≈{e:.2}"))
                .unwrap_or_default(),
        );
        prev = Some((m, per));
    }
    println!(
        "(us/update/m flat ⇒ O(m); rising ⇒ super-linear. 'exp' is the local\n \
         scaling exponent d(log t)/d(log m): ~2 = O(m²), ~1 = O(m).)"
    );
}
