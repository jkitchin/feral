//! Issue #127 regression: splitting the symbolic pipeline so race losers
//! skip the tail stages must not change *which* candidate wins or the
//! factorization produced from it.
//!
//! Strategy (no golden constants): the race dispatchers must be
//! self-consistent — the `AutoRace` / preprocess-`Auto` result must be
//! byte-identical to the result of directly requesting the concrete
//! `resolved_method` / `resolved_preprocess` the race reports. That is
//! exactly the property the prefix/finish split must preserve: the winner
//! is finished from its own prefix, and no other candidate influences it.

use feral::symbolic::{
    symbolic_factorize, symbolic_factorize_with_method, OrderingMethod, OrderingPreprocess,
    SupernodeParams,
};
use feral::CscMatrix;

/// 2D 5-point grid Laplacian on a `k×k` grid (SPD, n = k²). Rich enough
/// that the four ordering backends can reach different fills, and all of
/// them succeed on it.
fn grid_laplacian(k: usize) -> CscMatrix {
    let n = k * k;
    let idx = |r: usize, c: usize| r * k + c;
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    for r in 0..k {
        for c in 0..k {
            let i = idx(r, c);
            let mut deg = 0.0f64;
            for (dr, dc) in [(0isize, 1isize), (1, 0)] {
                let (nr, nc) = (r as isize + dr, c as isize + dc);
                if nr >= 0 && nr < k as isize && nc >= 0 && nc < k as isize {
                    let j = idx(nr as usize, nc as usize);
                    // off-diagonal -1, stored once in the lower triangle
                    // (CscMatrix keeps only row >= col; the matrix is symmetric).
                    rows.push(i.max(j));
                    cols.push(i.min(j));
                    vals.push(-1.0);
                    deg += 1.0;
                }
            }
            // diagonal: degree + 4 (strictly diagonally dominant → SPD)
            rows.push(i);
            cols.push(i);
            vals.push(deg + 4.0);
        }
    }
    CscMatrix::from_triplets(n, &rows, &cols, &vals).expect("build grid laplacian")
}

/// Compare the fields that fully determine a symbolic factorization. `perm`
/// equality alone forces the whole downstream (deterministic) result to
/// match; the rest are cheap belt-and-suspenders checks.
fn assert_same_symbolic(
    a: &feral::symbolic::SymbolicFactorization,
    b: &feral::symbolic::SymbolicFactorization,
    ctx: &str,
) {
    assert_eq!(a.perm, b.perm, "{ctx}: perm differs");
    assert_eq!(a.perm_inv, b.perm_inv, "{ctx}: perm_inv differs");
    assert_eq!(
        a.factor_nnz_estimate, b.factor_nnz_estimate,
        "{ctx}: factor_nnz_estimate differs"
    );
    assert_eq!(
        a.resolved_method, b.resolved_method,
        "{ctx}: resolved_method differs"
    );
    assert_eq!(
        a.resolved_preprocess, b.resolved_preprocess,
        "{ctx}: resolved_preprocess differs"
    );
    assert_eq!(a.col_counts, b.col_counts, "{ctx}: col_counts differs");
    assert_eq!(
        a.supernodes.len(),
        b.supernodes.len(),
        "{ctx}: supernode count differs"
    );
    for (i, (sa, sb)) in a.supernodes.iter().zip(b.supernodes.iter()).enumerate() {
        assert_eq!(sa.first_col, sb.first_col, "{ctx}: supernode {i} first_col");
        assert_eq!(sa.ncol, sb.ncol, "{ctx}: supernode {i} ncol");
        assert_eq!(sa.nrow, sb.nrow, "{ctx}: supernode {i} nrow");
    }
}

/// The `AutoRace` winner must byte-match the concrete method it reports,
/// and its estimate must be the minimum over all raced candidates.
#[test]
fn autorace_matches_winning_concrete_method() {
    let candidates = [
        OrderingMethod::Amd,
        OrderingMethod::MetisND,
        OrderingMethod::ScotchND,
        OrderingMethod::KahipND,
    ];
    for &preprocess in &[OrderingPreprocess::None, OrderingPreprocess::Auto] {
        for k in [6usize, 10, 15] {
            let m = grid_laplacian(k);
            let params = SupernodeParams {
                preprocess,
                ..SupernodeParams::default()
            };

            let race = symbolic_factorize_with_method(&m, &params, OrderingMethod::AutoRace)
                .expect("autorace");

            // The concrete result for the reported winning method must match.
            let concrete =
                symbolic_factorize_with_method(&m, &params, race.resolved_method.clone())
                    .expect("concrete winner");
            assert_same_symbolic(
                &race,
                &concrete,
                &format!("autorace k={k} preprocess={preprocess:?}"),
            );

            // The winner's estimate must be the minimum across candidates.
            let min_est = candidates
                .iter()
                .filter_map(|c| {
                    symbolic_factorize_with_method(&m, &params, c.clone())
                        .ok()
                        .map(|s| s.factor_nnz_estimate)
                })
                .min()
                .expect("at least one candidate");
            assert_eq!(
                race.factor_nnz_estimate, min_est,
                "autorace k={k} preprocess={preprocess:?}: winner is not the argmin estimate",
            );
        }
    }
}

/// The preprocess-`Auto` winner must byte-match the concrete
/// preprocess it reports (proving the finish uses the winning arm's prefix).
#[test]
fn preprocess_auto_matches_resolved_arm() {
    for k in [6usize, 10, 15, 20] {
        let m = grid_laplacian(k);
        let auto_params = SupernodeParams {
            preprocess: OrderingPreprocess::Auto,
            ..SupernodeParams::default()
        };
        // Fix the ordering method so only the preprocess race varies.
        let auto = symbolic_factorize_with_method(&m, &auto_params, OrderingMethod::Amd)
            .expect("auto preprocess");

        let arm_params = SupernodeParams {
            preprocess: auto.resolved_preprocess,
            ..SupernodeParams::default()
        };
        let arm = symbolic_factorize_with_method(&m, &arm_params, OrderingMethod::Amd)
            .expect("resolved arm");

        assert_same_symbolic(&auto, &arm, &format!("preprocess-auto k={k}"));
    }
}

/// `symbolic_factorize` (default params → `Auto` preprocess, `Auto` method)
/// must equal the concrete resolution it reports for both dimensions.
#[test]
fn default_symbolic_matches_its_resolution() {
    for k in [8usize, 12] {
        let m = grid_laplacian(k);
        let auto = symbolic_factorize(&m, &SupernodeParams::default()).expect("default auto");

        let concrete_params = SupernodeParams {
            preprocess: auto.resolved_preprocess,
            ..SupernodeParams::default()
        };
        let concrete =
            symbolic_factorize_with_method(&m, &concrete_params, auto.resolved_method.clone())
                .expect("concrete resolution");
        assert_same_symbolic(&auto, &concrete, &format!("default k={k}"));
    }
}
