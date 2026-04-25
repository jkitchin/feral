//! RED tests for Lever D.4 — tiny-n fast-path. Spec in
//! `dev/plans/sparse-tail-d4.md`.
//!
//! State on RED commit: `should_use_dense_fast_path` still has the
//! D.3-only predicate (n <= N_MAX && density >= rho_MIN). The tests
//! here specify the post-D.4 contract. The predicate edit in the
//! GREEN commit is what turns them from RED to GREEN.
//!
//! Test map vs plan §Tests:
//!   test_gate_tiny_sparse_in                   -> §1
//!   test_gate_just_outside_n_tiny              -> §2
//!   test_solve_parity_tiny_real_matrix         -> §3 (primary oracle)
//!   test_gate_boundary_n_16                    -> §5
//!   test_determinism_tiny                      -> §6
//!
//! Test 4 (zero-pivot tiny) is deferred: the D.3 test suite's
//! `test_zero_column_force_accept` already exercises the zero-pivot
//! path via `dense_fast_factor` on a (larger-n) in-gate matrix,
//! and the kernel path is identical at tiny n. Adding a duplicate
//! n=4 case here would not change coverage.

use feral::numeric::factorize::{
    factorize_multifrontal, factorize_multifrontal_supernodal, should_use_dense_fast_path,
    NumericParams,
};
use feral::numeric::solve::solve_sparse;
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::{read_mtx, BunchKaufmanParams, CscMatrix, Inertia, ZeroPivotAction};
use std::path::Path;

fn default_params() -> NumericParams {
    NumericParams::with_bk(BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        pivot_threshold: 0.01,
        ..BunchKaufmanParams::default()
    })
}

fn load_csc(path: &str) -> CscMatrix {
    let mtx = match read_mtx(Path::new(path)) {
        Ok(m) => m,
        Err(e) => panic!("read_mtx({}) failed: {}", path, e),
    };
    match mtx.to_csc() {
        Ok(c) => c,
        Err(e) => panic!("to_csc({}) failed: {}", path, e),
    }
}

fn make_rhs(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|i| {
            s = s
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add(i as u64 + 1);
            ((s >> 32) as u32 as f64) / (u32::MAX as f64) - 0.5
        })
        .collect()
}

fn rel_linf(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "vector length mismatch in rel_linf");
    let mut diff = 0.0_f64;
    let mut refn = 0.0_f64;
    for (x, y) in a.iter().zip(b.iter()) {
        diff = diff.max((x - y).abs());
        refn = refn.max(y.abs());
    }
    if refn == 0.0 {
        diff
    } else {
        diff / refn
    }
}

fn assert_inertia_eq(a: &Inertia, b: &Inertia, ctx: &str) {
    assert_eq!(a.positive, b.positive, "{}: positive differs", ctx);
    assert_eq!(a.negative, b.negative, "{}: negative differs", ctx);
    assert_eq!(a.zero, b.zero, "{}: zero differs", ctx);
}

/// Build a CSC (lower triangle) of an n x n symmetric matrix with
/// `off_diag_per_col` sub-diagonal entries in each column. Diagonal
/// is `10*n`; sub-diagonals are `1 + 0.1*(i-j)`. Used for sparse
/// tiny-n cases.
fn sparse_tiny(n: usize, off_diag_per_col: usize) -> CscMatrix {
    let mut col_ptr = Vec::with_capacity(n + 1);
    let mut row_idx = Vec::new();
    let mut values = Vec::new();
    col_ptr.push(0);
    for j in 0..n {
        // Diagonal.
        row_idx.push(j);
        values.push(10.0 * (n as f64));
        // Up to off_diag_per_col strict-lower entries.
        let max_off = (n - 1 - j).min(off_diag_per_col);
        for k in 1..=max_off {
            let i = j + k;
            row_idx.push(i);
            values.push(1.0 + 0.1 * (k as f64));
        }
        col_ptr.push(row_idx.len());
    }
    CscMatrix {
        n,
        col_ptr,
        row_idx,
        values,
    }
}

// -----------------------------------------------------------------
// §1  gate predicate — tiny sparse MUST gate in post-D.4
// -----------------------------------------------------------------

#[test]
fn test_gate_tiny_sparse_in() {
    // n=8, 6 nnz → density 6/(8*9/2) = 6/36 ≈ 0.17, below the D.3
    // rho_MIN of 0.25. Pre-D.4 the gate rejects. Post-D.4 the tiny
    // disjunct must accept unconditionally.
    assert!(
        should_use_dense_fast_path(8, 6),
        "D.4: n=8 (<= N_TINY) must gate in regardless of density (nnz_lower=6)"
    );

    // Top-10 observed cases — each must gate in.
    for (n, nnz) in [
        (5usize, 5usize), // KIRBY2LS_0274 shape
        (6, 7),           // PALMER1A / HEART6LS shape
        (7, 8),           // HS73
        (8, 9),           // PALMER1E
        (11, 12),         // HATFLDH
    ] {
        assert!(
            should_use_dense_fast_path(n, nnz),
            "D.4: n={} (<= N_TINY) must gate in (nnz_lower={})",
            n,
            nnz
        );
    }

    // n = N_TINY exactly must still gate in.
    assert!(
        should_use_dense_fast_path(16, 16),
        "D.4: n=16 (= N_TINY) must gate in"
    );
}

// -----------------------------------------------------------------
// §2  gate predicate — just outside N_TINY, still sparse, must NOT gate in
// -----------------------------------------------------------------

#[test]
fn test_gate_just_outside_n_tiny() {
    // n=17, 17 nnz (diag only) → density 17/(17*18/2) = 17/153 ≈ 0.11.
    // Above N_TINY and below rho_MIN. Pre-D.4 rejects and post-D.4
    // must still reject; N_TINY cannot have silently widened to 128.
    assert!(
        !should_use_dense_fast_path(17, 17),
        "n=17 sparse must stay out-of-gate (rho < 0.25, n > N_TINY)"
    );
    assert!(
        !should_use_dense_fast_path(20, 25),
        "n=20 sparse must stay out-of-gate"
    );
    // Still respect D.3 for in-density cases above N_TINY.
    assert!(
        should_use_dense_fast_path(64, 600),
        "D.3: n=64, 600 nnz (rho ≈ 0.29) must still gate in"
    );
    // Still reject everything above N_MAX irrespective of density.
    assert!(
        !should_use_dense_fast_path(129, 100_000),
        "n=129 above N_MAX must still be rejected"
    );
}

// -----------------------------------------------------------------
// §3  solve parity on a real tiny sparse matrix
// -----------------------------------------------------------------

#[test]
fn test_solve_parity_tiny_real_matrix() {
    // HS73_0308 (n=7) is in the observed D.4 target class. Post-D.4
    // `factorize_multifrontal` routes it through `dense_fast_factor`;
    // pre-D.4 it runs the multifrontal path. In both states the
    // dispatcher output MUST match the direct
    // `factorize_multifrontal_supernodal` bypass — this test is the
    // idempotency oracle that catches a silently-wrong synthesis
    // post-GREEN.
    let path = "data/matrices/kkt/HS73/HS73_0308.mtx";
    if !Path::new(path).exists() {
        eprintln!("SKIP: {} not present (corpus is gitignored)", path);
        return;
    }
    let csc = load_csc(path);
    assert!(
        csc.n <= 16,
        "HS73_0308 expected tiny-n (n={}); fixture changed?",
        csc.n
    );
    let params = default_params();
    let sn = SupernodeParams::default();

    // Bypass (forced multifrontal) — oracle.
    let sym = match symbolic_factorize(&csc, &sn) {
        Ok(s) => s,
        Err(e) => panic!("oracle symbolic failed: {}", e),
    };
    let (f_oracle, i_oracle) = match factorize_multifrontal_supernodal(&csc, &sym, &params) {
        Ok(r) => r,
        Err(e) => panic!("oracle multifrontal failed: {}", e),
    };

    // Gated dispatcher — post-D.4 routes to dense_fast_factor.
    let (f_gated, i_gated) = match factorize_multifrontal(&csc, &sym, &params) {
        Ok(r) => r,
        Err(e) => panic!("gated factorize_multifrontal failed: {}", e),
    };

    assert_inertia_eq(&i_oracle, &i_gated, "D.4 tiny parity inertia");
    assert_eq!(f_oracle.n, f_gated.n, "D.4 tiny parity: factor n mismatch");

    let rhs = make_rhs(csc.n, 0xD4_C0DE);
    let x_oracle = match solve_sparse(&f_oracle, &rhs) {
        Ok(x) => x,
        Err(e) => panic!("oracle solve failed: {}", e),
    };
    let x_gated = match solve_sparse(&f_gated, &rhs) {
        Ok(x) => x,
        Err(e) => panic!("gated solve failed: {}", e),
    };

    let rel = rel_linf(&x_gated, &x_oracle);
    assert!(
        rel <= 1e-10,
        "D.4 tiny parity: solve rel-inf = {:.3e} > 1e-10 on HS73_0308",
        rel
    );
}

// -----------------------------------------------------------------
// §5  boundary — n = N_TINY must round-trip
// -----------------------------------------------------------------

#[test]
fn test_gate_boundary_n_16() {
    // Synthetic n=16 with only one off-diagonal per column → density
    // well below rho_MIN. Post-D.4 gate accepts; solve must pass.
    let csc = sparse_tiny(16, 1);
    assert!(
        should_use_dense_fast_path(csc.n, csc.row_idx.len()),
        "n=16 sparse must gate in post-D.4"
    );
    let params = default_params();
    let sn = SupernodeParams::default();
    let sym = match symbolic_factorize(&csc, &sn) {
        Ok(s) => s,
        Err(e) => panic!("boundary symbolic failed: {}", e),
    };
    let (factors, inertia) = match factorize_multifrontal(&csc, &sym, &params) {
        Ok(r) => r,
        Err(e) => panic!("boundary factorize failed: {}", e),
    };
    assert_eq!(factors.n, 16, "boundary factor n mismatch");
    assert_eq!(
        inertia.positive + inertia.negative + inertia.zero,
        16,
        "boundary inertia count mismatch (pos+neg+zero != n)"
    );

    // Round-trip residual against a random RHS.
    let rhs = make_rhs(16, 0xB0_5A16);
    let x = match solve_sparse(&factors, &rhs) {
        Ok(x) => x,
        Err(e) => panic!("boundary solve failed: {}", e),
    };
    // Residual = b - A x. CSC is lower triangle only; expand to symmetric.
    let mut r = rhs.clone();
    for j in 0..16 {
        let cs = csc.col_ptr[j];
        let ce = csc.col_ptr[j + 1];
        for idx in cs..ce {
            let i = csc.row_idx[idx];
            let v = csc.values[idx];
            r[i] -= v * x[j];
            if i != j {
                r[j] -= v * x[i];
            }
        }
    }
    let rnorm = r.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    let bnorm = rhs.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    let rel = if bnorm == 0.0 { rnorm } else { rnorm / bnorm };
    assert!(
        rel <= 1e-10,
        "boundary residual {:.3e} > 1e-10 on synthetic n=16",
        rel
    );
}

// -----------------------------------------------------------------
// §6  determinism — two gated factorizations produce bit-equal solves
// -----------------------------------------------------------------

#[test]
fn test_determinism_tiny() {
    let csc = sparse_tiny(8, 2);
    assert!(
        should_use_dense_fast_path(csc.n, csc.row_idx.len()),
        "n=8 sparse must gate in post-D.4"
    );
    let params = default_params();
    let sn = SupernodeParams::default();
    let sym = match symbolic_factorize(&csc, &sn) {
        Ok(s) => s,
        Err(e) => panic!("determinism symbolic failed: {}", e),
    };

    let (f_a, i_a) = match factorize_multifrontal(&csc, &sym, &params) {
        Ok(r) => r,
        Err(e) => panic!("determinism first call failed: {}", e),
    };
    let (f_b, i_b) = match factorize_multifrontal(&csc, &sym, &params) {
        Ok(r) => r,
        Err(e) => panic!("determinism second call failed: {}", e),
    };
    assert_inertia_eq(&i_a, &i_b, "determinism inertia");

    let rhs = make_rhs(csc.n, 0xDE_7E12);
    let x_a = match solve_sparse(&f_a, &rhs) {
        Ok(x) => x,
        Err(e) => panic!("determinism solve a failed: {}", e),
    };
    let x_b = match solve_sparse(&f_b, &rhs) {
        Ok(x) => x,
        Err(e) => panic!("determinism solve b failed: {}", e),
    };
    for (i, (a, b)) in x_a.iter().zip(x_b.iter()).enumerate() {
        assert!(
            a.to_bits() == b.to_bits(),
            "determinism: x[{}] bits differ ({} vs {})",
            i,
            a,
            b
        );
    }
}
