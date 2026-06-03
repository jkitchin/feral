use std::path::Path;

use feral::dense::factor::factor;
use feral::dense::matrix::SymmetricMatrix;
use feral::numeric::factorize::{factorize_multifrontal, NumericParams};
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::{read_mtx, BunchKaufmanParams, ZeroPivotAction};

fn main() {
    let mtx = read_mtx(Path::new("tests/data/parity/fbrain3ls/FBRAIN3LS_0839.mtx")).expect("read");
    let csc = mtx.to_csc().expect("csc");

    // Path A: multifrontal (the parity-test path).
    {
        let sym = symbolic_factorize(&csc, &SupernodeParams::default()).expect("sym");
        let bk = BunchKaufmanParams {
            on_zero_pivot: ZeroPivotAction::ForceAccept,
            pivot_threshold: 0.01,
            ..BunchKaufmanParams::default()
        };
        println!("default null_pivot_tol = {:.3e}", bk.null_pivot_tol);
        let np = NumericParams::with_bk(bk.clone());
        let (fac, inertia) = factorize_multifrontal(&csc, &sym, &np).expect("factor");
        // Compute the per-supernode null_pivot_floor manually:
        // floor = sqrt(n) · EPS · ||A_scaled||_inf
        let n = csc.n as f64;
        // Row-sum infinity norm of scaled matrix (MC64+equilibration).
        let mut row_sum = vec![0.0_f64; csc.n];
        for j in 0..csc.n {
            let s_j = fac.scaling[j];
            for kk in csc.col_ptr[j]..csc.col_ptr[j + 1] {
                let i = csc.row_idx[kk];
                let m = (csc.values[kk] * fac.scaling[i] * s_j).abs();
                row_sum[i] += m;
                if i != j {
                    row_sum[j] += m;
                }
            }
        }
        let infnorm = row_sum.into_iter().fold(0.0_f64, f64::max);
        let floor = n.sqrt() * f64::EPSILON * infnorm;
        println!("scaled ||A||_inf  = {:.6e}", infnorm);
        println!(
            "null_pivot_floor  = sqrt({}) * EPS * ||A|| = {:.6e}",
            csc.n, floor
        );
        println!("d_diag[5]         = {:.6e}", 2.467786894e-16);
        println!(
            "in F-01 band?     : {} <= |d| <= {} ?  {}",
            f64::EPSILON,
            floor,
            2.467786894e-16 > f64::EPSILON && 2.467786894e-16 <= floor
        );
        println!("== multifrontal path ==");
        println!("inertia: {}", inertia);
        let ff = &fac.node_factors[0].frontal_factors;
        println!("d_diag at k=5: {:.17e}", ff.d_diag[5]);
        println!("EPSILON     : {:.17e}", f64::EPSILON);
        println!("|d| > EPS?  : {}", ff.d_diag[5].abs() > f64::EPSILON);
        println!("scaling = {:?}", fac.scaling);
        println!("perm = {:?}", ff.perm);
        for k in 0..ff.d_diag.len() {
            println!(
                "  k={}  d={:+.6e}  sub={:+.6e}",
                k, ff.d_diag[k], ff.d_subdiag[k]
            );
        }
        println!("frontal inertia = {:?}", ff.inertia);
        println!("zero_tol_2x2 = {:.3e}", ff.zero_tol_2x2);
    }

    // Path B: dense factor() directly on the un-scaled matrix.
    {
        let mut entries: Vec<(usize, usize, f64)> = Vec::new();
        for j in 0..csc.n {
            for k in csc.col_ptr[j]..csc.col_ptr[j + 1] {
                let i = csc.row_idx[k];
                if i >= j {
                    entries.push((i, j, csc.values[k]));
                }
            }
        }
        let sm = SymmetricMatrix::from_lower_triangle(csc.n, &entries);
        let bk = BunchKaufmanParams {
            on_zero_pivot: ZeroPivotAction::ForceAccept,
            pivot_threshold: 0.01,
            ..BunchKaufmanParams::default()
        };
        let (fac, inertia) = factor(&sm, &bk).expect("dense factor");
        println!();
        println!("== dense factor() directly on un-MC64'd matrix ==");
        println!("inertia: {}", inertia);
        for k in 0..csc.n {
            println!(
                "  k={}  d={:.17e}  sub={:+.6e}  perm[k]={}",
                k, fac.d_diag[k], fac.d_subdiag[k], fac.perm[k]
            );
        }
        println!("d_eq = {:?}", fac.d_eq);
    }
}
