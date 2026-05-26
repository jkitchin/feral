//! RED tests for Lever D.3 — dense fast-path for small-dense
//! matrices. Spec in `dev/plans/sparse-tail-d3.md`.
//!
//! State on RED commit: `dense_fast_factor` and
//! `should_use_dense_fast_path` are stubs (one returns `Err`, the
//! other returns `false` unconditionally). The tests here compile
//! against the stub API and fail in exactly the places the GREEN
//! commit is expected to fix. The stubs + failing tests form the
//! authorized scope for the implementation commit.
//!
//! Test map (vs plan §Tests):
//!   test_gate_predicate_shape                    -> §4 / §1
//!   test_solve_parity_tro3x3                     -> §2 (primary oracle)
//!   test_cross_path_determinism_tro3x3           -> §6
//!   test_zero_column_force_accept                -> §5
//!
//! Tests §1 (gate-off snapshot) and §3 (at-boundary) are deferred
//! to the GREEN-follow-up commit because they require a
//! working dense path to validate — they check that the gated
//! dispatcher in `factorize_multifrontal_with_workspace`
//! correctly routes OR does NOT route. The stub dispatcher in
//! RED has no gate wired, so those tests would be vacuous.

use feral::numeric::factorize::{
    dense_fast_factor, factorize_multifrontal_supernodal, should_use_dense_fast_path,
    NumericParams, SparseFactors,
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

/// Skip helper for corpus-dependent tests: matrices in `data/matrices/`
/// are gitignored (CUTEst KKT corpus regenerates from ripopt runs;
/// see .gitignore) so CI hosts don't have them. Returns `true` when
/// the file is absent and the caller should `return` early.
fn skip_if_missing(path: &str) -> bool {
    if !Path::new(path).exists() {
        eprintln!(
            "SKIP: {} not present (data/matrices/ is gitignored; \
             regenerate via ripopt CUTEst harness)",
            path
        );
        true
    } else {
        false
    }
}

/// Deterministic RHS — irrational scalars keep cancellation honest.
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

/// Build a CSC (lower triangle) of a dense, diagonally-dominant
/// symmetric matrix. Deterministic. Used for the zero-column and
/// synthetic in-gate shape tests.
fn dense_diag_dominant(n: usize) -> CscMatrix {
    // Pattern: dense lower triangle, A(i, j) = 1.0 + 0.1 * (i - j) for
    // i > j, A(i, i) = 10.0 * (n as f64). Diagonally dominant enough
    // that BK pivots cleanly with 1x1 blocks only.
    let mut col_ptr = Vec::with_capacity(n + 1);
    let mut row_idx = Vec::new();
    let mut values = Vec::new();
    col_ptr.push(0);
    for j in 0..n {
        for i in j..n {
            row_idx.push(i);
            if i == j {
                values.push(10.0 * (n as f64));
            } else {
                values.push(1.0 + 0.1 * ((i - j) as f64));
            }
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

/// Same as `dense_diag_dominant(n)` but with column `k` forced to
/// zero (except the diagonal), producing an in-gate matrix that
/// triggers BK's zero-pivot path at column `k`.
fn dense_with_zero_column(n: usize, k: usize) -> CscMatrix {
    assert!(k < n, "k must be in range");
    let mut m = dense_diag_dominant(n);
    // Zero the diagonal AND off-diagonals of column k.
    let cstart = m.col_ptr[k];
    let cend = m.col_ptr[k + 1];
    for idx in cstart..cend {
        m.values[idx] = 0.0;
    }
    // Zero the row-k entries in columns j < k (lower triangle only).
    for j in 0..k {
        let jstart = m.col_ptr[j];
        let jend = m.col_ptr[j + 1];
        for idx in jstart..jend {
            if m.row_idx[idx] == k {
                m.values[idx] = 0.0;
            }
        }
    }
    m
}

#[test]
fn test_gate_predicate_shape() {
    // Out-of-gate by n (>128 under the planned threshold).
    assert!(
        !should_use_dense_fast_path(129, 10_000),
        "n=129 above N_MAX must not gate in"
    );
    assert!(
        !should_use_dense_fast_path(512, 20_000),
        "n=512 far above N_MAX must not gate in"
    );
    // Out-of-gate by density (very sparse). 100 nnz in a 128x128
    // matrix is well below 25% of 128*129/2 = 8256.
    assert!(
        !should_use_dense_fast_path(128, 100),
        "density 100/8256 ≪ 0.25 must not gate in"
    );
    // In-gate: n=64 with >=25% density. 64*65/2 = 2080, 25% = 520.
    assert!(
        should_use_dense_fast_path(64, 600),
        "n=64, 600/2080 ≈ 0.29 must gate in"
    );
    // Boundary: n=128, nnz = 25% of cells. 128*129/2 = 8256, 25% = 2064.
    assert!(
        should_use_dense_fast_path(128, 2064),
        "boundary n=128, density=0.25 must gate in"
    );
}

#[test]
fn test_solve_parity_tro3x3() {
    let path = "data/matrices/kkt/TRO3X3/TRO3X3_0013.mtx";
    if skip_if_missing(path) {
        return;
    }
    let csc = load_csc(path);
    assert!(csc.n <= 128, "TRO3X3_0013 expected in-gate (n={})", csc.n);
    let params = default_params();

    // Oracle arm: forced multifrontal path.
    let sn = SupernodeParams::default();
    let sym = match symbolic_factorize(&csc, &sn) {
        Ok(s) => s,
        Err(e) => panic!("oracle symbolic failed: {}", e),
    };
    let (f_oracle, i_oracle) = match factorize_multifrontal_supernodal(&csc, &sym, &params) {
        Ok(r) => r,
        Err(e) => panic!("oracle multifrontal failed: {}", e),
    };

    // Dense arm: direct call to the fast-path.
    let (f_dense, i_dense) = match dense_fast_factor(&csc, &params) {
        Ok(r) => r,
        Err(e) => panic!("dense_fast_factor failed: {}", e),
    };

    assert_inertia_eq(&i_oracle, &i_dense, "solve_parity_tro3x3");
    assert_eq!(
        f_oracle.n, f_dense.n,
        "factor n mismatch: oracle {} vs dense {}",
        f_oracle.n, f_dense.n
    );

    let rhs = make_rhs(csc.n, 0xCAFE_F00D);
    let x_oracle = match solve_sparse(&f_oracle, &rhs) {
        Ok(x) => x,
        Err(e) => panic!("oracle solve failed: {}", e),
    };
    let x_dense = match solve_sparse(&f_dense, &rhs) {
        Ok(x) => x,
        Err(e) => panic!("dense solve failed: {}", e),
    };

    let rel = rel_linf(&x_dense, &x_oracle);
    assert!(
        rel <= 1e-10,
        "solve parity rel-∞ = {:.3e} > 1e-10 on TRO3X3_0013",
        rel
    );
}

#[test]
fn test_cross_path_determinism_tro3x3() {
    // Factor the same in-gate matrix twice via the dense path.
    // Solves must be bit-equal (determinism floor).
    let path = "data/matrices/kkt/TRO3X3/TRO3X3_0013.mtx";
    if skip_if_missing(path) {
        return;
    }
    let csc = load_csc(path);
    let params = default_params();

    let (f_a, i_a) = match dense_fast_factor(&csc, &params) {
        Ok(r) => r,
        Err(e) => panic!("dense_fast_factor first call failed: {}", e),
    };
    let (f_b, i_b) = match dense_fast_factor(&csc, &params) {
        Ok(r) => r,
        Err(e) => panic!("dense_fast_factor second call failed: {}", e),
    };
    assert_inertia_eq(&i_a, &i_b, "cross_path_determinism");

    let rhs = make_rhs(csc.n, 0xDEAD_BEEF);
    let x_a = match solve_sparse(&f_a, &rhs) {
        Ok(x) => x,
        Err(e) => panic!("solve a failed: {}", e),
    };
    let x_b = match solve_sparse(&f_b, &rhs) {
        Ok(x) => x,
        Err(e) => panic!("solve b failed: {}", e),
    };
    assert_eq!(x_a.len(), x_b.len(), "solve length mismatch");
    for (i, (a, b)) in x_a.iter().zip(x_b.iter()).enumerate() {
        assert!(
            a.to_bits() == b.to_bits(),
            "cross_path_determinism: x[{}] bits differ ({} vs {})",
            i,
            a,
            b
        );
    }
    // Smoke field match.
    let _: &SparseFactors = &f_a;
    let _: &SparseFactors = &f_b;
}

#[test]
fn test_zero_column_force_accept() {
    // ForceAccept must route a zero-pivot column to the `zero`
    // inertia count without erroring. This mirrors the
    // multifrontal root-supernode behavior under
    // `ZeroPivotAction::ForceAccept` and is what the dense path
    // needs to match.
    let n = 32;
    let k = 5;
    let csc = dense_with_zero_column(n, k);
    let params = default_params();

    let (factors, inertia) = match dense_fast_factor(&csc, &params) {
        Ok(r) => r,
        Err(e) => panic!("dense_fast_factor on zero-column matrix failed: {}", e),
    };
    assert_eq!(factors.n, n, "factor n mismatch");
    // Issue #54 (SSIDS alignment): a strict-zero forced pivot is recorded
    // in the `zero` bucket, matching SSIDS `NumericSubtree.hxx:259-267`
    // and MA57's `info(24/25)` accounting. The remaining n-1
    // diagonally-dominant pivots are positive.
    assert_eq!(
        inertia.zero, 1,
        "strict-zero forced pivot → `zero`; got {}",
        inertia.zero
    );
    assert_eq!(
        inertia.negative, 0,
        "no negative pivots in this matrix; got neg={}",
        inertia.negative
    );
    assert_eq!(
        inertia.positive,
        n - 1,
        "remaining n-1 pivots should all be positive; got pos={}",
        inertia.positive
    );
}
