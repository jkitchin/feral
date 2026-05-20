//! Issue #46 — delayed-pivot cascade on a structurally-zero (2,2)-block
//! saddle-point / IPM KKT.
//!
//! ## The bug
//!
//! On a KKT `[[H, Bᵀ], [B, 0]]` whose (2,2) block is structurally
//! absent, FERAL's numeric Bunch-Kaufman kernel re-derived every pivot
//! blind via magnitude-argmax. For a zero-diagonal constraint column
//! `k`, BK picks `r` = the row of the largest off-diagonal coupling;
//! when that `r` is not fully summed (an out-of-front coupling) the
//! kernel could neither form a 2×2 nor 1×1 the zero diagonal, so it
//! *delayed* the column up the elimination tree. The delays cascaded —
//! on the POUNCE `cho` KKT this produced a 23× factor-nonzero blowup
//! and a ~160× end-to-end slowdown vs MA57. See
//! `dev/research/kkt-zero-2x2-block-cascade-2026-05-20.md`.
//!
//! ## The fix
//!
//! `scalar_pivot_step` now selects the 2×2 partner explicitly. The
//! MC64-matched saddle partner is co-located at the adjacent
//! fully-summed column `k+1`, so when the BK argmax row `r` is not
//! fully summed the kernel falls back to `k+1` as the 2×2 partner
//! instead of delaying.
//!
//! ## Oracle
//!
//! The matrix is a genuine saddle-point KKT with `H` SPD and `B` full
//! row rank. By the saddle-point inertia theorem (external math —
//! Benzi, Golub & Liesen, *Numerical solution of saddle point
//! problems*, Acta Numerica 2005, §3.4) its inertia is exactly
//! `(nv, nc, 0)`. The solve residual is an identity check. The fill
//! bound is a measurement oracle: a healthy factorization stays
//! within a small multiple of the symbolic no-delay estimate, whereas
//! the #46 cascade blew it up 20×+.

use feral::numeric::factorize::factorize_multifrontal;
use feral::scaling::ScalingStrategy;
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::{solve_sparse, CscMatrix, NumericParams};

/// Build the lower triangle of a saddle-point KKT in *constraints-first*
/// layout
///
/// ```text
///     A = [ 0   B ]      (n = nc + nv, nc < nv)
///         [ Bᵀ  H ]
/// ```
///
/// — a symmetric permutation of the textbook `[[H, Bᵀ], [B, 0]]`, so
/// inertia is unchanged. Constraints occupy indices `[0, nc)` and
/// variables `[nc, n)`. The constraints-first layout matters: MC64
/// pairs each constraint `c` with a variable of *higher* index, and
/// `build_supermap` canonicalises pairs as `(min, max)`, so the
/// co-located pair is expanded constraint-then-variable — the numeric
/// kernel reaches the zero-diagonal constraint column *first*, exactly
/// as it does on a real IPM KKT.
///
/// Structure:
///   - variable `0` (matrix index `nc`) is the "global" variable `g`,
///     coupled to *every* constraint — highest degree, eliminated last
///     at the root front;
///   - variable `1 + c` (matrix index `nc + 1 + c`) is constraint
///     `c`'s "local" variable, coupled only to constraint `c`;
///   - each constraint `c` couples to `g` with the *larger* magnitude
///     `2.0` and to its local variable with `1.0`. MC64 can co-locate
///     `g` with only one constraint; every other constraint's largest
///     coupling (`g`) is therefore an out-of-front trailing row when
///     its child panel is factored. BK's magnitude-argmax lands on
///     that out-of-front `g` — the exact #46 trigger;
///   - `H` is diagonal `4.0` (SPD) and the structurally-explicit `0.0`
///     is the (2,2) diagonal a POUNCE live KKT constraint carries.
///
/// `B` has full row rank (the local-variable columns form an `nc × nc`
/// identity block) and `H` is SPD, so `A` is non-singular with inertia
/// exactly `(nv, nc, 0)`.
fn saddle_kkt(nv: usize, nc: usize) -> CscMatrix {
    assert!(nc < nv, "saddle KKT needs nc < nv for distinct local vars");
    let n = nc + nv;
    let g = nc; // matrix index of the global variable
    let mut rows: Vec<usize> = Vec::new();
    let mut cols: Vec<usize> = Vec::new();
    let mut vals: Vec<f64> = Vec::new();
    let mut push = |r: usize, c: usize, v: f64| {
        debug_assert!(r >= c, "lower triangle only: ({r},{c})");
        rows.push(r);
        cols.push(c);
        vals.push(v);
    };

    // Constraint columns 0..nc: zero (2,2) diagonal plus the two B
    // couplings (rows g and the local variable, both > c).
    for c in 0..nc {
        push(c, c, 0.0); // structurally-explicit zero (2,2) diagonal
        push(g, c, 2.0); // coupling to global var g (out-of-front)
        push(nc + 1 + c, c, 1.0); // coupling to local var (partner)
    }
    // Variable columns nc..n: diagonal SPD H.
    for v in 0..nv {
        push(nc + v, nc + v, 4.0);
    }

    CscMatrix::from_triplets(n, &rows, &cols, &vals)
        .expect("saddle KKT triplets form a valid lower triangle")
}

/// `y = A·x` where `a` stores only the lower triangle of symmetric A.
fn symv_lower(a: &CscMatrix, x: &[f64]) -> Vec<f64> {
    let n = a.n;
    let mut y = vec![0.0; n];
    for j in 0..n {
        for k in a.col_ptr[j]..a.col_ptr[j + 1] {
            let i = a.row_idx[k];
            let v = a.values[k];
            y[i] += v * x[j];
            if i != j {
                y[j] += v * x[i];
            }
        }
    }
    y
}

/// Factor a structurally-zero-(2,2)-block saddle KKT through the
/// default sparse pipeline and assert: exact inertia, no delayed-pivot
/// cascade, and an accurate solve.
#[test]
fn issue_46_saddle_kkt_factors_without_delayed_pivot_cascade() {
    let nv = 300;
    let nc = 200;
    let n = nv + nc;
    let a = saddle_kkt(nv, nc);
    assert!(
        n >= 128,
        "n must clear MIN_N_FOR_COMPRESSION for LdltCompress"
    );

    // Symbolic analysis with default settings (preprocess = Auto,
    // which resolves to LdltCompress on this zero-(2,2)-block KKT).
    let sym = symbolic_factorize(&a, &SupernodeParams::default())
        .expect("symbolic analysis of saddle KKT");

    // Numeric factorization with `allow_delayed_pivots = true` (the
    // production path that exhibited the #46 cascade) and `Identity`
    // scaling. Identity isolates the Bunch-Kaufman kernel: the #46 fix
    // lives in `scalar_pivot_step`, and the kernel must factor a saddle
    // KKT without a delayed-pivot cascade *without leaning on the
    // InfNorm/MC64 rescaling* — feral's MC64 scaling is degenerate on
    // saddles (#45) and rejected, so the kernel cannot rely on it.
    let np = NumericParams {
        scaling: ScalingStrategy::Identity,
        ..NumericParams::default()
    };
    let (factors, inertia) =
        factorize_multifrontal(&a, &sym, &np).expect("numeric factorization of saddle KKT");

    // Oracle 1 — inertia. Saddle-point inertia theorem: H SPD +
    // B full row rank ⇒ inertia exactly (nv, nc, 0).
    assert_eq!(
        (inertia.positive, inertia.negative, inertia.zero),
        (nv, nc, 0),
        "saddle KKT inertia must be exactly (nv, nc, 0); got {inertia:?}",
    );

    // Oracle 2 — no delayed-pivot cascade. A healthy factorization
    // stays within a small multiple of the symbolic no-delay fill
    // estimate; the #46 cascade blew this up >20×. The 5× bound
    // decisively separates "healthy" (≤2× on this matrix) from a
    // cascade while leaving generous slack for 2×2-pivot bookkeeping.
    // Verified: the pre-#46 kernel cascades this matrix to a 61×
    // blowup (n_delayed=398, n_2x2=0); the fixed kernel holds it at
    // 0.83× (n_delayed=0, n_2x2=199). The 5× bound separates the two
    // decisively.
    let fnnz = factors.factor_nnz();
    let est = sym.factor_nnz_estimate.max(1);
    let blowup = fnnz as f64 / est as f64;
    assert!(
        blowup <= 5.0,
        "factor_nnz {fnnz} is {blowup:.1}× the symbolic estimate {est} \
         — delayed-pivot cascade (issue #46) regression",
    );

    // Oracle 3 — solve accuracy (identity check). Build b = A·x_true
    // for a known x_true, solve, and check the relative residual
    // ‖A·x − b‖ / ‖b‖.
    let x_true: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.5).sin() + 1.5).collect();
    let b = symv_lower(&a, &x_true);
    let x = solve_sparse(&factors, &b).expect("solve saddle KKT");
    let resid = symv_lower(&a, &x);
    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..n {
        num += (resid[i] - b[i]).powi(2);
        den += b[i] * b[i];
    }
    let relres = (num / den).sqrt();
    assert!(
        relres <= 1e-8,
        "saddle KKT solve relative residual {relres:.3e} exceeds 1e-8",
    );
}
