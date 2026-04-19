//! Diagnose the VESUVIOU / VESUVIO factor-time outlier.
//!
//! Session 07 landed the CRESC132 fix (bordered-KKT → MetisND when
//! `n >= 5000 && nnz/n < 6`). VESUVIOU is the leftover top outlier:
//! n=3083 (below the threshold), factor/MUMPS ratio 80-85× across
//! samples. Structurally different from CRESC — shape-wise it's an
//! augmented-system IPM KKT but the density is identical (nnz/n ≈ 4.3).
//!
//! This binary answers: is VESUVIO {U,A,} factoring slowly because of
//! delayed pivots (bordered pathology), because the ordering is bad
//! (METIS would help), or because numerically it's just hard?
//!
//! For each matrix we report, under both Amd and MetisND:
//!   - symbolic max_nrow / total L nnz
//!   - numeric factor time
//!   - actual max_nrow (post-delay) and total delays
//!   - nnz-of-root, root nelim vs ncol (to see top-of-tree pathology)
//!   - inertia
//!
//! Plus basic shape features: zero diagonals, max row nnz, diag-only
//! rows (constraint slacks).
//!
//! Usage: `cargo run --release --bin vesuvio_diag`

use feral::numeric::factorize::factorize_multifrontal;
use feral::scaling::ScalingStrategy;
use feral::symbolic::{symbolic_factorize_with_method, OrderingMethod, SupernodeParams};
use feral::{read_mtx, BunchKaufmanParams, CscMatrix, ZeroPivotAction};
use std::path::Path;
use std::time::Instant;

fn shape_features(csc: &CscMatrix) {
    let n = csc.n;
    let mut zero_diag = 0usize;
    let mut diag_only_rows = 0usize;
    let mut max_col_nnz = 0usize;
    let mut total_nnz = 0usize;
    for j in 0..n {
        let start = csc.col_ptr[j];
        let end = csc.col_ptr[j + 1];
        let nnz = end - start;
        max_col_nnz = max_col_nnz.max(nnz);
        total_nnz += nnz;
        let mut has_diag = false;
        let mut has_offdiag = false;
        for k in start..end {
            let i = csc.row_idx[k];
            if i == j {
                has_diag = true;
                if csc.values[k] == 0.0 {
                    zero_diag += 1;
                }
            } else {
                has_offdiag = true;
            }
        }
        if !has_diag {
            zero_diag += 1; // structurally absent = zero
        }
        if has_diag && !has_offdiag {
            diag_only_rows += 1;
        }
    }
    println!(
        "  shape: n={} stored_nnz={} avg_deg={:.2} max_col_nnz={} zero_diag={} diag_only={}",
        n,
        total_nnz,
        total_nnz as f64 / n as f64,
        max_col_nnz,
        zero_diag,
        diag_only_rows,
    );

    // Column-degree distribution — buckets to expose arrow-KKT shape.
    // For lever-C analysis we want to know:
    //   - how many "trivial" columns (degree 1, just diagonal)
    //   - how many "thin" columns (degree 2-4, typical IPM Jacobian rows)
    //   - how many "medium" columns (degree 5-32, IPM Hessian rows)
    //   - how many "dense" columns (degree > 32 or > sqrt(n))
    //   - top-5 largest columns and their indices
    let mut bucket_1 = 0usize; // degree 1 (diagonal only)
    let mut bucket_2_4 = 0usize;
    let mut bucket_5_32 = 0usize;
    let mut bucket_dense = 0usize;
    let dense_threshold = (10.0 * (n as f64).sqrt()) as usize;
    let mut top5: Vec<(usize, usize)> = Vec::with_capacity(5);
    for j in 0..n {
        let nnz = csc.col_ptr[j + 1] - csc.col_ptr[j];
        if nnz <= 1 {
            bucket_1 += 1;
        } else if nnz <= 4 {
            bucket_2_4 += 1;
        } else if nnz <= 32 {
            bucket_5_32 += 1;
        } else {
            bucket_dense += 1;
        }
        if top5.len() < 5 {
            top5.push((nnz, j));
            top5.sort_by(|a, b| b.0.cmp(&a.0));
        } else if nnz > top5[4].0 {
            top5[4] = (nnz, j);
            top5.sort_by(|a, b| b.0.cmp(&a.0));
        }
    }
    println!(
        "  col_deg: deg=1: {}  deg=2-4: {}  deg=5-32: {}  deg>32: {} (dense_thresh={})",
        bucket_1, bucket_2_4, bucket_5_32, bucket_dense, dense_threshold,
    );
    print!("  top5_cols:");
    for (deg, col) in &top5 {
        print!(" col{}({})", col, deg);
    }
    println!();
}

fn run_one_method(csc: &CscMatrix, method: OrderingMethod, scaling: ScalingStrategy) {
    let snode_params = SupernodeParams {
        scaling_strategy: scaling.clone(),
        ..SupernodeParams::default()
    };
    let factor_params = BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        pivot_threshold: 0.01,
        ..BunchKaufmanParams::default()
    };
    let scale_tag = match scaling {
        ScalingStrategy::InfNorm => "infnorm",
        ScalingStrategy::Mc64Symmetric => "mc64",
        ScalingStrategy::Identity => "ident",
        ScalingStrategy::External(_) => "ext",
        ScalingStrategy::Auto => "auto",
    };

    let t_sym = Instant::now();
    let sym = match symbolic_factorize_with_method(csc, &snode_params, method) {
        Ok(s) => s,
        Err(e) => {
            println!("  {:?}: symbolic FAILED: {:?}", method, e);
            return;
        }
    };
    let sym_us = t_sym.elapsed().as_micros();

    // Symbolic shape
    let sym_max_nrow = sym.supernodes.iter().map(|s| s.nrow).max().unwrap_or(0);
    let n_snodes = sym.supernodes.len();

    let t_fac = Instant::now();
    let (factors, inertia) = match factorize_multifrontal(csc, &sym, &factor_params) {
        Ok(f) => f,
        Err(e) => {
            println!("  {:?}: numeric FAILED: {:?}", method, e);
            return;
        }
    };
    let fac_us = t_fac.elapsed().as_micros();

    // Numeric shape
    let mut actual_max_nrow = 0usize;
    let mut total_delays = 0usize;
    let mut total_nelim = 0usize;
    let mut total_ncol = 0usize;
    let mut root_nrow = 0usize;
    let mut root_nelim = 0usize;
    let mut root_ncol = 0usize;
    for nf in &factors.node_factors {
        let ff = &nf.frontal_factors;
        actual_max_nrow = actual_max_nrow.max(ff.nrow);
        total_nelim += ff.nelim;
        total_ncol += nf.ncol;
        total_delays += nf.ncol.saturating_sub(ff.nelim);
        if ff.nrow > root_nrow {
            root_nrow = ff.nrow;
            root_nelim = ff.nelim;
            root_ncol = nf.ncol;
        }
    }

    let avg_deg = csc.row_idx.len() as f64 / csc.n as f64;
    println!(
        "  {:<8?}/{:<8}: sym={}us fac={}us snodes={} sym_max_nrow={} actual_max_nrow={} total_delays={} \
         (attempted={}, elim={})  root={}x{}(nelim={}, {:.0}% of n)  inertia=({}/{}/{}) avg_deg={:.1}",
        method,
        scale_tag,
        sym_us,
        fac_us,
        n_snodes,
        sym_max_nrow,
        actual_max_nrow,
        total_delays,
        total_ncol,
        total_nelim,
        root_nrow,
        root_ncol,
        root_nelim,
        100.0 * root_nrow as f64 / csc.n as f64,
        inertia.positive,
        inertia.negative,
        inertia.zero,
        avg_deg,
    );
}

fn run(family: &str, sample: &str) {
    let p = format!("data/matrices/kkt/{}/{}{}.mtx", family, family, sample);
    let path = Path::new(&p);
    let mtx = match read_mtx(path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("SKIP {}: {}", p, e);
            return;
        }
    };
    let csc = match mtx.to_csc() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("SKIP {}: csc failed: {:?}", p, e);
            return;
        }
    };

    println!("== {}{} ==", family, sample);
    shape_features(&csc);
    run_one_method(&csc, OrderingMethod::Amd, ScalingStrategy::InfNorm);
    run_one_method(&csc, OrderingMethod::Amd, ScalingStrategy::Mc64Symmetric);
    run_one_method(&csc, OrderingMethod::MetisND, ScalingStrategy::InfNorm);
    run_one_method(
        &csc,
        OrderingMethod::MetisND,
        ScalingStrategy::Mc64Symmetric,
    );
    println!();
}

fn main() {
    println!("VESUVIO-family factor outlier diagnostic");
    println!("{}", "-".repeat(80));
    let cases: &[(&str, &str)] = &[
        ("VESUVIOU", "_0000"),
        ("VESUVIOU", "_0005"),
        ("VESUVIO", "_0000"),
        ("VESUVIO", "_0021"),
        ("VESUVIA", "_0000"),
        // MUONSINE: next factor outlier after VESUVIO. n=1537, 60.9× MUMPS
        // in the session-08 bench. Same diagnostic to rule in/out a dense-
        // kernel pathology vs an ordering-fixable one.
        ("MUONSINE", "_0000"),
        // Reference: CRESC132 numbers after the METIS heuristic, for
        // sanity.
        ("CRESC132", "_0000"),
    ];
    for (fam, samp) in cases {
        run(fam, samp);
    }

    let _ = CscMatrix::from_triplets;
}
