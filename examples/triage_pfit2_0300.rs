//! Deep triage of PFIT2_0300 — the worst REAL_BUG candidate from the
//! Phase 2.2.3 sparse-only failure investigation.
//!
//! Failure summary:
//!   n=6, expected inertia (3,3,0), production sparse residual 3.55e-6,
//!   bench tolerance n*eps*1e6 = 1.33e-9 (2666× over).
//!   Dense path produces residual 2.89e-12 (PASS) — same factor quality;
//!   only the IR loop differs.
//!
//! Hypothesis from the broad triage (`triage_sparse_residual_edge.rs`):
//! `solve_sparse_refined`'s 2-strike stagnation rule (max_stagnant_steps=2,
//! src/numeric/solve.rs:366) exits at step 3 while the basin transition
//! happens at step 5. Extended IR confirms residual drops to 2.89e-12.
//!
//! This example prints:
//!   1. The matrix entries
//!   2. Dense factor: L, D, equilibration scaling, BK permutation
//!   3. Sparse factor (the single supernode): L, D, scaling
//!   4. Bit-identity of the two factors
//!   5. Step-by-step IR trajectory under the production rule and extended
//!   6. The exact step at which the production rule prematurely exits
//!
//! Run with: cargo run --release --example triage_pfit2_0300

use std::path::Path;

use feral::numeric::factorize::{factorize_multifrontal, SparseFactors};
use feral::numeric::solve::solve_sparse;
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::{
    factor as dense_factor, read_mtx, read_sidecar, solve_refined as dense_solve_refined,
    BunchKaufmanParams, CscMatrix, SymmetricMatrix, ZeroPivotAction,
};

fn norm2(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn rel_residual_csc(a: &CscMatrix, x: &[f64], b: &[f64]) -> f64 {
    let n = a.n;
    let mut ax = vec![0.0; n];
    a.symv(x, &mut ax);
    let mut rs = 0.0;
    let mut bs = 0.0;
    for i in 0..n {
        let r = ax[i] - b[i];
        rs += r * r;
        bs += b[i] * b[i];
    }
    if bs > 0.0 {
        (rs / bs).sqrt()
    } else {
        rs.sqrt()
    }
}

fn rel_residual_dense(a: &SymmetricMatrix, x: &[f64], b: &[f64]) -> f64 {
    let n = a.n;
    let mut ax = vec![0.0; n];
    a.symv(x, &mut ax);
    let mut rs = 0.0;
    let mut bs = 0.0;
    for i in 0..n {
        let r = ax[i] - b[i];
        rs += r * r;
        bs += b[i] * b[i];
    }
    if bs > 0.0 {
        (rs / bs).sqrt()
    } else {
        rs.sqrt()
    }
}

fn print_dense_lower(a: &SymmetricMatrix, label: &str) {
    println!("\n{} ({}x{}):", label, a.n, a.n);
    for i in 0..a.n {
        print!("  ");
        for j in 0..=i {
            print!("{:13.6e} ", a.get(i, j));
        }
        println!();
    }
}

fn params() -> BunchKaufmanParams {
    BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        pivot_threshold: 0.01,
        ..BunchKaufmanParams::default()
    }
}

fn sparse_np() -> feral::numeric::factorize::NumericParams {
    feral::numeric::factorize::NumericParams::with_bk(params())
}

/// Step-by-step IR using the same primitives as `solve_sparse_refined`
/// but reporting the per-step residual, the "would the 2-strike rule
/// exit here?" flag, and the value of `best_r_norm` at each point.
fn instrumented_refine(
    matrix: &CscMatrix,
    factors: &SparseFactors,
    rhs: &[f64],
    max_steps: usize,
    max_stagnant_steps: usize,
) {
    let n = matrix.n;
    let b_norm = norm2(rhs);

    let mut x = solve_sparse(factors, rhs).expect("solve_sparse");
    let mut r = vec![0.0; n];
    matrix.symv(&x, &mut r);
    for i in 0..n {
        r[i] = rhs[i] - r[i];
    }
    let mut r_norm = norm2(&r);

    let mut best_r_norm = r_norm;
    let mut stagnant_count: usize = 0;
    let mut prod_exit_step: Option<usize> = None;

    println!(
        "  {:>4} {:>14} {:>14} {:>14} {:>10} {}",
        "step", "||r||", "rel_res", "best", "stagnant", "production_2-strike"
    );
    println!(
        "  {:>4} {:>14.6e} {:>14.6e} {:>14.6e} {:>10} {}",
        0,
        r_norm,
        r_norm / b_norm,
        best_r_norm,
        "-",
        "(initial)"
    );

    for step in 1..=max_steps {
        let dx = solve_sparse(factors, &r).expect("dx");
        for i in 0..n {
            x[i] += dx[i];
        }
        matrix.symv(&x, &mut r);
        for i in 0..n {
            r[i] = rhs[i] - r[i];
        }
        r_norm = norm2(&r);

        let improved = r_norm < best_r_norm;
        if improved {
            best_r_norm = r_norm;
            stagnant_count = 0;
        } else {
            stagnant_count += 1;
        }
        let prod_would_exit = stagnant_count >= max_stagnant_steps;
        if prod_exit_step.is_none() && prod_would_exit {
            prod_exit_step = Some(step);
        }
        println!(
            "  {:>4} {:>14.6e} {:>14.6e} {:>14.6e} {:>10} {}",
            step,
            r_norm,
            r_norm / b_norm,
            best_r_norm,
            stagnant_count,
            if prod_would_exit {
                if prod_exit_step == Some(step) {
                    "EXIT (first time)"
                } else {
                    "(would have exited)"
                }
            } else if improved {
                "improved"
            } else {
                "non-improving"
            }
        );
    }
    if let Some(s) = prod_exit_step {
        println!("\n  → production 2-strike rule exits at step {}", s);
        println!(
            "  → best residual under production: see step {} 'best' column",
            s
        );
    } else {
        println!(
            "\n  → production 2-strike rule never triggers within {} steps",
            max_steps
        );
    }
}

fn main() {
    let base = "data/matrices/kkt/PFIT2/PFIT2_0300";
    let mtx_path = format!("{}.mtx", base);
    let sidecar_path = format!("{}.json", base);

    let mtx = read_mtx(Path::new(&mtx_path)).expect("mtx");
    let csc = mtx.to_csc().expect("csc");
    let dense_mat = mtx.to_dense();
    let sc = read_sidecar(Path::new(&sidecar_path)).expect("sidecar");
    let rhs = sc.finite_rhs().expect("rhs");
    let n = csc.n;
    let tol = (n as f64) * f64::EPSILON * 1e6;

    println!("=== PFIT2_0300 deep triage ===");
    println!("n = {}", n);
    println!(
        "expected inertia (sidecar) = ({},{},{})",
        sc.inertia.positive, sc.inertia.negative, sc.inertia.zero
    );
    println!("residual tolerance (bench) = n*eps*1e6 = {:.3e}", tol);
    println!("||b||₂ = {:.6e}", norm2(&rhs));

    print_dense_lower(&dense_mat, "Lower triangle of A");

    println!("\nRHS:");
    for (i, v) in rhs.iter().enumerate() {
        println!("  b[{}] = {:.6e}", i, v);
    }

    // ----- Dense path -----
    println!("\n=== Dense factor + solve_refined ===");
    let (df, di) = dense_factor(&dense_mat, &params()).expect("dense factor");
    println!("inertia = {}", di);
    println!(
        "d_eq (equilibration): {:?}",
        df.d_eq
            .iter()
            .map(|x| format!("{:.4e}", x))
            .collect::<Vec<_>>()
    );
    println!("perm: {:?}", df.perm);
    println!(
        "d_diag: {:?}",
        df.d_diag
            .iter()
            .map(|x| format!("{:.4e}", x))
            .collect::<Vec<_>>()
    );
    println!(
        "d_subdiag: {:?}",
        df.d_subdiag
            .iter()
            .map(|x| format!("{:.4e}", x))
            .collect::<Vec<_>>()
    );
    println!("L (unit lower, column-major nxn):");
    for i in 0..n {
        print!("  ");
        for j in 0..n {
            print!("{:11.4e} ", df.l[j * n + i]);
        }
        println!();
    }

    let x_d = dense_solve_refined(&dense_mat, &df, &rhs).expect("dense solve_refined");
    let rel_d = rel_residual_dense(&dense_mat, &x_d, &rhs);
    println!(
        "\ndense final rel_res = {:.6e}  ({})",
        rel_d,
        if rel_d <= tol { "PASS" } else { "FAIL" }
    );

    // ----- Sparse path -----
    println!("\n=== Sparse factor + solve_sparse_refined ===");
    let sym = symbolic_factorize(&csc, &SupernodeParams::default()).expect("sym");
    println!(
        "symbolic: {} supernodes, est_nnz_L = {}",
        sym.supernodes.len(),
        sym.factor_nnz_estimate
    );

    let (sf, si) = factorize_multifrontal(&csc, &sym, &sparse_np()).expect("sparse factor");
    println!("inertia = {}", si);
    println!(
        "sparse scaling vector: {:?}",
        sf.scaling
            .iter()
            .map(|x| format!("{:.4e}", x))
            .collect::<Vec<_>>()
    );
    println!("sparse scaling_info: {:?}", sf.scaling_info);
    let nf = &sf.node_factors[0];
    println!(
        "supernode 0: nrow={} ncol={} nelim={} n_delayed_in={}",
        nf.nrow, nf.ncol, nf.nelim, nf.n_delayed_in
    );
    let ff = &nf.frontal_factors;
    println!(
        "frontal d_diag: {:?}",
        ff.d_diag
            .iter()
            .map(|x| format!("{:.4e}", x))
            .collect::<Vec<_>>()
    );
    println!(
        "frontal d_subdiag: {:?}",
        ff.d_subdiag
            .iter()
            .map(|x| format!("{:.4e}", x))
            .collect::<Vec<_>>()
    );
    println!("frontal perm: {:?}", ff.perm);
    println!("frontal L (nrow={} x nelim={}):", ff.nrow, ff.nelim);
    for i in 0..ff.nrow {
        print!("  ");
        for j in 0..ff.nelim {
            print!("{:11.4e} ", ff.l[j * ff.nrow + i]);
        }
        println!();
    }

    // Plain (no IR) sparse solve
    let x_s_plain = solve_sparse(&sf, &rhs).expect("solve_sparse");
    let rel_plain = rel_residual_csc(&csc, &x_s_plain, &rhs);
    println!(
        "\nplain (no IR) rel_res = {:.6e}  ({})",
        rel_plain,
        if rel_plain <= tol { "PASS" } else { "FAIL" }
    );

    println!("\n--- Step-by-step IR (production rule: max_steps=10, max_stagnant=2) ---");
    instrumented_refine(&csc, &sf, &rhs, 10, 2);

    println!("\n--- Same trajectory, max_stagnant=4 (would PFIT2_0300 converge?) ---");
    instrumented_refine(&csc, &sf, &rhs, 10, 4);

    // Dense vs sparse L/D bit-identity (single supernode, nelim=n).
    if ff.nelim == n && ff.nrow == n {
        println!("\n=== Dense vs sparse factor comparison (both n×n single front) ===");
        let mut max_l_diff: f64 = 0.0;
        let mut max_d_diff: f64 = 0.0;
        for i in 0..n {
            for j in 0..n {
                let d_l = df.l[j * n + i];
                let s_l = ff.l[j * ff.nrow + i];
                max_l_diff = max_l_diff.max((d_l - s_l).abs());
            }
            max_d_diff = max_d_diff.max((df.d_diag[i] - ff.d_diag[i]).abs());
        }
        println!("max |L_dense - L_sparse| = {:.3e}", max_l_diff);
        println!("max |D_dense - D_sparse| = {:.3e}", max_d_diff);
        println!("perm dense:  {:?}", df.perm);
        println!("perm sparse: {:?}", ff.perm);
    } else {
        println!(
            "\n(skip L/D comparison: ff.nelim={} ff.nrow={} vs n={})",
            ff.nelim, ff.nrow, n
        );
    }
}
