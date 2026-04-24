//! Phase 2.2.3 — minimal multi-supernode repro.
//!
//! 6x6 arrow matrix, nemin=1 → residual 7.47e-1 (WRONG).
//! Same matrix under nemin=10000 (single supernode) → residual 1e-16.
//!
//! Prints symbolic structure + factor dump to pin the bug.

use feral::numeric::factorize::{factorize_multifrontal, NumericParams};
use feral::numeric::solve::solve_sparse;
use feral::scaling::ScalingStrategy;
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::{BunchKaufmanParams, CscMatrix, ZeroPivotAction};

fn params() -> NumericParams {
    NumericParams {
        bk: BunchKaufmanParams {
            on_zero_pivot: ZeroPivotAction::ForceAccept,
            pivot_threshold: 0.0,
            ..BunchKaufmanParams::default()
        },
        scaling: ScalingStrategy::Identity,
        small_leaf: Default::default(),
    }
}

fn arrow_6x6() -> (CscMatrix, Vec<f64>) {
    // [2 0 0 0 0 1]
    // [0 3 0 0 0 1]
    // [0 0 4 0 0 1]
    // [0 0 0 5 0 1]
    // [0 0 0 0 6 1]
    // [1 1 1 1 1 7]
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    for j in 0..6 {
        rows.push(j);
        cols.push(j);
        vals.push((j + 2) as f64);
    }
    for j in 0..5 {
        rows.push(5);
        cols.push(j);
        vals.push(1.0);
    }
    let m = CscMatrix::from_triplets(6, &rows, &cols, &vals).unwrap();
    let rhs = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    (m, rhs)
}

fn dump_run(m: &CscMatrix, rhs: &[f64], nemin: usize) {
    println!("\n=== nemin = {} ===", nemin);
    let snp = SupernodeParams {
        nemin,
        ..Default::default()
    };
    let sym = symbolic_factorize(m, &snp).unwrap();

    println!("perm        = {:?}", sym.perm);
    println!("perm_inv    = {:?}", sym.perm_inv);
    println!("supernodes  = {}", sym.supernodes.len());
    for (i, s) in sym.supernodes.iter().enumerate() {
        println!(
            "  #{}  first_col={}  ncol={}  nrow={}  children={:?}",
            i,
            s.first_col,
            s.ncol(),
            s.nrow,
            s.children
        );
    }

    let (fac, inertia) = factorize_multifrontal(m, &sym, &params()).unwrap();
    println!("inertia = {}", inertia);
    for (i, nf) in fac.node_factors.iter().enumerate() {
        println!(
            "  node {}: ncol={} nrow={} row_indices={:?}",
            i, nf.ncol, nf.nrow, nf.row_indices
        );
        println!("    perm={:?}", nf.frontal_factors.perm);
        println!("    D diag={:?}", nf.frontal_factors.d_diag);
        println!("    D sub ={:?}", nf.frontal_factors.d_subdiag);
        // L as dense matrix (column-major)
        let nrow = nf.frontal_factors.nrow;
        let ncol = nf.frontal_factors.ncol;
        println!("    L ({}x{}):", nrow, ncol);
        for i in 0..nrow {
            print!("      ");
            for j in 0..ncol {
                print!("{:>8.4} ", nf.frontal_factors.l[j * nrow + i]);
            }
            println!();
        }
        if nf.frontal_factors.contrib_dim > 0 {
            println!(
                "    contrib ({}x{}):",
                nf.frontal_factors.contrib_dim, nf.frontal_factors.contrib_dim
            );
            let cdim = nf.frontal_factors.contrib_dim;
            for i in 0..cdim {
                print!("      ");
                for j in 0..cdim {
                    let v = if i >= j {
                        nf.frontal_factors.contrib[j * cdim + i]
                    } else {
                        nf.frontal_factors.contrib[i * cdim + j]
                    };
                    print!("{:>12.5e} ", v);
                }
                println!();
            }
        }
    }

    let x = solve_sparse(&fac, rhs).unwrap();
    let n = m.n;
    let mut ax = vec![0.0; n];
    m.symv(&x, &mut ax);
    let mut rs = 0.0;
    let mut bs = 0.0;
    for i in 0..n {
        rs += (ax[i] - rhs[i]).powi(2);
        bs += rhs[i] * rhs[i];
    }
    let res = (rs / bs).sqrt();
    println!("x   = {:?}", x);
    println!("Ax  = {:?}", ax);
    println!("res = {:.3e}", res);
}

fn main() {
    let (m, rhs) = arrow_6x6();
    // Print the matrix itself
    println!("A (dense, symmetric):");
    let n = m.n;
    for i in 0..n {
        print!("  ");
        for j in 0..n {
            // Lookup A[i,j] from symmetric CSC (lower triangle)
            let (ii, jj) = if i >= j { (i, j) } else { (j, i) };
            let mut v = 0.0;
            for k in m.col_ptr[jj]..m.col_ptr[jj + 1] {
                if m.row_idx[k] == ii {
                    v = m.values[k];
                    break;
                }
            }
            print!("{:>5} ", v);
        }
        println!();
    }
    println!("b = {:?}", rhs);

    dump_run(&m, &rhs, 1);
    dump_run(&m, &rhs, 10000);
}
