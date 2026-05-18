// Pivot-magnitude audit across F-01 test matrices.
//
// Purpose: before changing F-01 band semantics, dump per-pivot |d|
// values and the corresponding null_pivot_floor for each matrix that
// currently relies on F-01 firing. This lets us predict the regression
// of any proposed change without running the full test suite first.
//
// Matrices probed:
//   1. FBRAIN3LS_0839 — the new outlier (#39); F-01 mis-fires here.
//   2. rank-1 dyadic A = u·uᵀ, n=5, u=ones — the
//      `f01_rankdef_surfaces_at_least_one_zero_pivot` test target.
//   3. synth/rankdef_*.mtx — the original F-01 evidence corpus.
//
// For each matrix we print every eliminated pivot's |d| alongside
// EPS, the matrix-global null_pivot_floor, and whether the pivot
// would be reclassified from zero -> sign under sign-fallback in
// the band (EPS, null_pivot_floor].
use std::path::Path;

use feral::numeric::factorize::{factorize_multifrontal, NumericParams};
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::{read_mtx, BunchKaufmanParams, CscMatrix, ZeroPivotAction};

fn null_pivot_floor(csc: &CscMatrix, scaling: &[f64]) -> f64 {
    let n = csc.n;
    let mut row_sum = vec![0.0_f64; n];
    for j in 0..n {
        let s_j = scaling[j];
        for k in csc.col_ptr[j]..csc.col_ptr[j + 1] {
            let i = csc.row_idx[k];
            let m = (csc.values[k] * scaling[i] * s_j).abs();
            row_sum[i] += m;
            if i != j {
                row_sum[j] += m;
            }
        }
    }
    let infnorm = row_sum.into_iter().fold(0.0_f64, f64::max);
    (n as f64).sqrt() * f64::EPSILON * infnorm
}

fn probe(label: &str, csc: &CscMatrix) {
    let sym = match symbolic_factorize(csc, &SupernodeParams::default()) {
        Ok(s) => s,
        Err(e) => {
            println!("== {} == symbolic failed: {:?}", label, e);
            return;
        }
    };
    let bk = BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        ..BunchKaufmanParams::default()
    };
    let np = NumericParams::with_bk(bk);
    let (fac, inertia) = match factorize_multifrontal(csc, &sym, &np) {
        Ok(p) => p,
        Err(e) => {
            println!("== {} == factor failed: {:?}", label, e);
            return;
        }
    };
    let floor = null_pivot_floor(csc, &fac.scaling);
    let eps = f64::EPSILON;
    println!();
    println!(
        "== {} ==  n={}  inertia={}  floor=sqrt(n)·EPS·||A_scaled||={:.3e}",
        label, csc.n, inertia, floor
    );
    let mut would_flip_to_sign = 0usize;
    let mut still_zero = 0usize;
    let mut signed_above_floor = 0usize;
    for node in &fac.node_factors {
        let ff = &node.frontal_factors;
        for k in 0..ff.d_diag.len() {
            let d = ff.d_diag[k];
            let ad = d.abs();
            let in_strict = ad <= eps;
            let in_band = ad > eps && ad <= floor;
            let class = if in_strict {
                "STRICT_ZERO  (case a)"
            } else if in_band {
                would_flip_to_sign += 1;
                "F-01 BAND    (case a') -> would become sign under fallback"
            } else if ad <= 1e-10 {
                signed_above_floor += 1;
                "SIGN-ACCEPT  (case b, small)"
            } else {
                "SIGN-ACCEPT  (case b, normal)"
            };
            // The current code in fact only emits zero in the strict and
            // band cases; track them too for sanity.
            if in_strict || in_band {
                still_zero += 1;
            }
            println!("  k={:3}  |d|={:.3e}  d={:+.3e}  {}", k, ad, d, class);
        }
    }
    println!(
        "  summary: strict_or_band={}, would_flip_to_sign={}, small_signed_above_floor={}",
        still_zero, would_flip_to_sign, signed_above_floor
    );
}

fn dyadic_uut(n: usize) -> CscMatrix {
    // A = u·uᵀ with u = ones(n). Symmetric rank-1, eigenvalues (n, 0, …, 0).
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    for j in 0..n {
        for i in j..n {
            rows.push(i);
            cols.push(j);
            vals.push(1.0);
        }
    }
    CscMatrix::from_triplets(n, &rows, &cols, &vals).expect("dyadic csc")
}

type CscBuilder = Box<dyn Fn() -> CscMatrix>;

fn main() {
    let cases: Vec<(&str, CscBuilder)> = vec![
        (
            "FBRAIN3LS_0839 (issue #39)",
            Box::new(|| {
                read_mtx(Path::new("tests/data/parity/fbrain3ls/FBRAIN3LS_0839.mtx"))
                    .expect("read fbrain")
                    .to_csc()
                    .expect("csc fbrain")
            }),
        ),
        (
            "dyadic u·uᵀ n=5 (F-01 invariant test)",
            Box::new(|| dyadic_uut(5)),
        ),
        (
            "synth/rankdef_5_2",
            Box::new(|| {
                read_mtx(Path::new(
                    "external_benchmarks/stress/matrices/synth/rankdef_5_2.mtx",
                ))
                .expect("read rd5")
                .to_csc()
                .expect("csc rd5")
            }),
        ),
        (
            "synth/rankdef_10_3",
            Box::new(|| {
                read_mtx(Path::new(
                    "external_benchmarks/stress/matrices/synth/rankdef_10_3.mtx",
                ))
                .expect("read rd10")
                .to_csc()
                .expect("csc rd10")
            }),
        ),
        (
            "synth/rankdef_50_5",
            Box::new(|| {
                read_mtx(Path::new(
                    "external_benchmarks/stress/matrices/synth/rankdef_50_5.mtx",
                ))
                .expect("read rd50")
                .to_csc()
                .expect("csc rd50")
            }),
        ),
        (
            "synth/rankdef_200_20",
            Box::new(|| {
                read_mtx(Path::new(
                    "external_benchmarks/stress/matrices/synth/rankdef_200_20.mtx",
                ))
                .expect("read rd200")
                .to_csc()
                .expect("csc rd200")
            }),
        ),
    ];

    for (label, build) in &cases {
        let csc = build();
        probe(label, &csc);
    }
}
