//! Finding N2 (`dev/research/repo-review-2026-06-09.md`): the
//! static-pivot floor is computed from the **unscaled** user matrix
//! (`static_pivot_floor = t · ‖A‖∞` in `solver.rs`, from
//! `matrix_inf_norm(matrix)`) but enforced by the BK kernels on pivots
//! of the **scaled** matrix `D·A·D`. Under a norm-normalizing scaling
//! (InfNorm / MC64) the unscaled and scaled ∞-norms can differ by orders
//! of magnitude, so the relative threshold `t` behaves like a wildly
//! different value in pivot space — drifting from the MA57 `cntl(1)`
//! analogy the docs promise. The F-01 null-pivot floor does this
//! correctly (scaled ∞-norm, `factorize.rs::scaled_matrix_infnorm`).
//!
//! ## Oracle: scale invariance under InfNorm equilibration
//!
//! Knight-Ruiz InfNorm scaling normalizes a global scalar out
//! completely: for `A` and `γ·A`, the equilibration produces
//! `D' = D / √γ`, so `D'·(γA)·D' = D·A·D` — the *identical* scaled
//! matrix. Choosing `γ = 2³⁰` (a power of two) makes both `γ·A` and the
//! `√γ` scaling exactly representable, so the scaled matrices are
//! bit-identical and the only thing that can differ between the two
//! factorizations is the static-pivot floor.
//!
//! Therefore the static-pivot *decision* (whether each pivot is
//! perturbed) MUST be identical for `A` and `γ·A`: same `needs_refinement`,
//! same inertia. This is a self-consistency oracle — no external solver
//! needed.
//!
//! Pre-fix the unscaled floor differs by `γ`:
//!   floor(A)  = t · ‖A‖∞      ≈ 1e-6 · O(1)   = O(1e-6)   → no pivot that small → no perturbation
//!   floor(γA) = t · ‖γA‖∞     ≈ 1e-6 · 2³⁰·O(1) ≈ 1e3      → every O(1) scaled pivot perturbed
//! so `A` reports `needs_refinement = false` and `γ·A` reports `true` —
//! the test's scale-invariance assertion fails. Post-fix (floor computed
//! from the scaled ∞-norm) both use `floor ≈ t · O(1) = O(1e-6)`, neither
//! perturbs, and the two agree.

use feral::numeric::factorize::NumericParams;
use feral::scaling::ScalingStrategy;
use feral::symbolic::supernode::SupernodeParams;
use feral::{CscMatrix, FactorStatus, Solver};

/// Build a CSC lower-triangle matrix from (row, col, val) triplets
/// (col-major, row >= col).
fn csc_lower(n: usize, triplets: &[(usize, usize, f64)]) -> CscMatrix {
    let mut cols: Vec<Vec<(usize, f64)>> = (0..n).map(|_| Vec::new()).collect();
    for &(r, c, v) in triplets {
        assert!(r >= c, "lower triangle: r={r} c={c}");
        cols[c].push((r, v));
    }
    let mut col_ptr = vec![0usize];
    let mut row_idx = Vec::new();
    let mut values = Vec::new();
    for col in cols.iter_mut() {
        col.sort_by_key(|x| x.0);
        for &(r, v) in col.iter() {
            row_idx.push(r);
            values.push(v);
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

/// Scale every stored value of a CSC matrix by a constant `γ`.
fn scale_values(mat: &CscMatrix, gamma: f64) -> CscMatrix {
    CscMatrix {
        n: mat.n,
        col_ptr: mat.col_ptr.clone(),
        row_idx: mat.row_idx.clone(),
        values: mat.values.iter().map(|v| v * gamma).collect(),
    }
}

fn solver_infnorm_static_pivot(t: f64) -> Solver {
    let np = NumericParams {
        // InfNorm (Knight-Ruiz) scaling normalizes a global scalar out,
        // which is exactly the interaction the Identity-scaling issue_38
        // tests deliberately bypass.
        scaling: ScalingStrategy::InfNorm,
        static_pivot_threshold: Some(t),
        ..NumericParams::default()
    };
    Solver::with_params(np, SupernodeParams::default())
}

/// Factor `mat` and return `(needs_refinement, positive, negative, zero)`.
fn factor_summary(mat: &CscMatrix, t: f64) -> (bool, usize, usize, usize) {
    let mut s = solver_infnorm_static_pivot(t);
    assert!(
        matches!(s.factor(mat, None), FactorStatus::Success),
        "factor must succeed"
    );
    let nr = s.factors().expect("factors").needs_refinement;
    let inertia = s.inertia().expect("inertia").clone();
    (nr, inertia.positive, inertia.negative, inertia.zero)
}

/// N2 reproduction: the static-pivot decision must be invariant under a
/// global scalar `γ` when InfNorm scaling is used, because `A` and `γ·A`
/// equilibrate to the identical scaled matrix.
#[test]
fn n2_static_pivot_floor_is_scale_invariant_under_infnorm() {
    let t = 1e-6;
    let gamma = (1u64 << 30) as f64; // 2^30, exactly representable

    // Indefinite 3×3 saddle-point KKT:
    //   [ 1  0  1 ]
    //   [ 0  1  1 ]
    //   [ 1  1  0 ]
    // ‖A‖∞ = 2; well-conditioned; InfNorm-scaled pivots are O(1).
    let a = csc_lower(
        3,
        &[
            (0, 0, 1.0),
            (1, 1, 1.0),
            (2, 0, 1.0),
            (2, 1, 1.0),
            (2, 2, 0.0),
        ],
    );
    let ga = scale_values(&a, gamma);

    let base = factor_summary(&a, t);
    let scaled = factor_summary(&ga, t);

    assert_eq!(
        base, scaled,
        "static-pivot decision must be invariant under γ·A with InfNorm \
         scaling (A→{base:?}, γ·A→{scaled:?}); the floor is being computed \
         from the unscaled ‖A‖∞ instead of the scaled ‖D·A·D‖∞ (N2)"
    );
}
