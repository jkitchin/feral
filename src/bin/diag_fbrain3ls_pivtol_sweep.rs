//! Issue #29: FBRAIN3LS BK `pivot_threshold` sweep.
//!
//! Sweeps `BunchKaufmanParams::pivot_threshold ∈
//! {0.0, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 0.01}` on the FBRAIN3LS panel
//! of interest (the three "borderline" matrices flagged in
//! `dev/research/inertia-triage-2026-04-27.md` plus 0788/0848) and
//! reports per (matrix, pivtol) the factor status, inertia tuple,
//! smallest |D|, and the JSON-RHS relative residual.
//!
//! Two passes per matrix:
//!   - PASS A: `on_zero_pivot=ForceAccept` (default sparse policy).
//!     The driver's F-01 override raises `null_pivot_tol` to
//!     `sqrt(n)·EPS·‖A‖_inf` for rank-deficient detection. On
//!     FBRAIN3LS (‖A‖ ~ 4e8) this floor is ~2e-7 and dominates the
//!     `pivot_threshold` gate, so the sweep is intentionally flat here
//!     — that flatness is the headline finding.
//!   - PASS B: `on_zero_pivot=Fail`. Disables the null_pivot_tol
//!     override (per `override_null_pivot_tol`'s `Fail` early-out),
//!     leaving `pivot_threshold` as the active acceptance gate so the
//!     sweep actually exercises BK pivot acceptance per
//!     Bunch-Kaufman 1977 §2 / Ashcraft-Grimes-Lewis 1998.
//!
//! See `dev/research/fbrain3ls-2x2-stability.md`.
//!
//! Usage:
//!     cargo run --release --bin diag_fbrain3ls_pivtol_sweep

use std::path::{Path, PathBuf};

use feral::numeric::factorize::{factorize_multifrontal, NumericParams, SparseFactors};
use feral::scaling::ScalingStrategy;
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::{
    read_mtx, read_sidecar, solve_sparse_refined, BunchKaufmanParams, CscMatrix, ZeroPivotAction,
};

const CORPUS: &str = "data/matrices/kkt/FBRAIN3LS";

/// Matrices selected for the sweep. Mix of "borderline" cases where
/// feral and MUMPS/SSIDS disagree on inertia and one definitive case
/// (0788) where the canonical Fortran solvers and feral all agree —
/// included as a baseline so the sweep proves it doesn't *break*
/// well-behaved members of the family.
const SAMPLES: &[&str] = &[
    "FBRAIN3LS_0788", // definitive, consensus (6,0,0), feral matches
    "FBRAIN3LS_0839", // numerically_intractable, feral (5,0,1) vs (6,0,0)
    "FBRAIN3LS_0843", // numerically_intractable, feral (5,0,1) vs (6,0,0)
    "FBRAIN3LS_0848", // excluded, three-way split
    "FBRAIN3LS_0851", // numerically_intractable, feral (5,0,1) vs (6,0,0)
];

const PIVTOLS: &[f64] = &[0.0, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-2];

fn matvec_lower_sym(csc: &CscMatrix, x: &[f64], out: &mut [f64]) {
    out.iter_mut().for_each(|v| *v = 0.0);
    for j in 0..csc.n {
        for k in csc.col_ptr[j]..csc.col_ptr[j + 1] {
            let i = csc.row_idx[k];
            let v = csc.values[k];
            out[i] += v * x[j];
            if i != j {
                out[j] += v * x[i];
            }
        }
    }
}

fn relative_residual(csc: &CscMatrix, x: &[f64], rhs: &[f64]) -> f64 {
    let n = csc.n;
    let mut ax = vec![0.0; n];
    matvec_lower_sym(csc, x, &mut ax);
    let mut num2 = 0.0;
    let mut den2 = 0.0;
    for i in 0..n {
        let d = ax[i] - rhs[i];
        num2 += d * d;
        den2 += rhs[i] * rhs[i];
    }
    let denom = den2.sqrt().max(1.0);
    num2.sqrt() / denom
}

/// Inspect every supernode's D and return (n_2x2, smallest |eigenvalue|).
/// The eigenvalues come from a closed-form 2x2 eigendecomposition for
/// blocks and from |d| for 1x1 pivots — matches `diag_acopp30_residual`.
fn d_stats(factors: &SparseFactors) -> (usize, f64) {
    let mut n_2x2 = 0usize;
    let mut min_abs = f64::INFINITY;
    for nf in &factors.node_factors {
        let ff = &nf.frontal_factors;
        let nelim = ff.nelim;
        let mut k = 0;
        while k < nelim {
            let two_by_two = k + 1 < nelim && ff.d_subdiag[k] != 0.0;
            if two_by_two {
                n_2x2 += 1;
                let a = ff.d_diag[k];
                let b = ff.d_subdiag[k];
                let c = ff.d_diag[k + 1];
                let trace = a + c;
                let det = a * c - b * b;
                let disc = (trace * trace - 4.0 * det).max(0.0).sqrt();
                let e1 = (trace - disc) * 0.5;
                let e2 = (trace + disc) * 0.5;
                min_abs = min_abs.min(e1.abs()).min(e2.abs());
                k += 2;
            } else {
                min_abs = min_abs.min(ff.d_diag[k].abs());
                k += 1;
            }
        }
    }
    (n_2x2, min_abs)
}

/// Compute the F-01 null-pivot floor that the multifrontal driver
/// installs when `on_zero_pivot != Fail`. Mirrors the private
/// `null_pivot_floor` in `numeric::factorize`; replicated here so the
/// diagnostic can report which gate is active without exposing a new
/// public surface.
fn null_pivot_floor_estimate(csc: &CscMatrix) -> f64 {
    let n = csc.n;
    // ‖A‖_inf for a symmetric matrix stored as its lower triangle.
    let mut row_sum = vec![0.0_f64; n];
    for j in 0..n {
        for k in csc.col_ptr[j]..csc.col_ptr[j + 1] {
            let i = csc.row_idx[k];
            let v = csc.values[k].abs();
            row_sum[i] += v;
            if i != j {
                row_sum[j] += v;
            }
        }
    }
    let infnorm = row_sum.into_iter().fold(0.0_f64, f64::max);
    (n as f64).sqrt() * f64::EPSILON * infnorm
}

fn run_pass(
    label: &str,
    csc: &CscMatrix,
    sym: &feral::symbolic::SymbolicFactorization,
    rhs: &[f64],
    expected: (usize, usize, usize),
    on_zero: ZeroPivotAction,
    tag: &str,
) {
    println!(
        "\n--- {label}  pass {tag}  on_zero_pivot={:?} ---",
        match &on_zero {
            ZeroPivotAction::Fail => "Fail",
            ZeroPivotAction::ForceAccept => "ForceAccept",
            ZeroPivotAction::PerturbToEps { .. } => "PerturbToEps",
        }
    );
    println!(
        "{:>9}  {:>9}  {:>3}  {:>15}  {:>13}  {:>13}  {:>6}",
        "pivtol", "status", "n2x2", "inertia", "min|D|", "rel_res", "match?"
    );

    for &u in PIVTOLS {
        let params = NumericParams {
            bk: BunchKaufmanParams {
                pivot_threshold: u,
                on_zero_pivot: on_zero.clone(),
                ..BunchKaufmanParams::default()
            },
            scaling: ScalingStrategy::Identity,
            ..NumericParams::default()
        };

        match factorize_multifrontal(csc, sym, &params) {
            Ok((factors, inertia)) => {
                let (n_2x2, min_abs) = d_stats(&factors);
                let inertia_tuple = (inertia.positive, inertia.negative, inertia.zero);
                let m = inertia_tuple == expected;

                let res_str = match solve_sparse_refined(csc, &factors, rhs) {
                    Ok(x) => format!("{:>13.3e}", relative_residual(csc, &x, rhs)),
                    Err(e) => format!("solve_err:{e}"),
                };
                let inertia_str = format!(
                    "({},{},{})",
                    inertia.positive, inertia.negative, inertia.zero
                );
                println!(
                    "{:>9.0e}  {:>9}  {:>3}  {:>15}  {:>13.3e}  {}  {:>6}",
                    u,
                    "OK",
                    n_2x2,
                    inertia_str,
                    min_abs,
                    res_str,
                    if m { "yes" } else { "NO" }
                );
            }
            Err(e) => {
                println!(
                    "{:>9.0e}  {:>9}  {:>3}  {:>15}  {:>13}  {:>13}  {:>6}",
                    u,
                    "FAIL",
                    "-",
                    "-",
                    "-",
                    format!("{e}"),
                    "-"
                );
            }
        }
    }
}

fn run_sample(label: &str) {
    let mtx_path = PathBuf::from(format!("{CORPUS}/{label}.mtx"));
    let json_path = PathBuf::from(format!("{CORPUS}/{label}.json"));
    let mtx = match read_mtx(&mtx_path) {
        Ok(m) => m,
        Err(e) => {
            println!("{label}: SKIP read_mtx: {e}");
            return;
        }
    };
    let csc = match mtx.to_csc() {
        Ok(c) => c,
        Err(e) => {
            println!("{label}: SKIP to_csc: {e}");
            return;
        }
    };
    let sc = match read_sidecar(&json_path) {
        Ok(s) => s,
        Err(e) => {
            println!("{label}: SKIP sidecar: {e}");
            return;
        }
    };
    let n = csc.n;
    let nnz = csc.row_idx.len();
    let rhs = match sc.finite_rhs() {
        Some(r) if r.len() == n => r,
        _ => {
            println!("{label}: SKIP RHS not finite or length mismatch");
            return;
        }
    };
    let expected = (sc.inertia.positive, sc.inertia.negative, sc.inertia.zero);

    let sym = match symbolic_factorize(&csc, &SupernodeParams::default()) {
        Ok(s) => s,
        Err(e) => {
            println!("{label}: SKIP symbolic: {e}");
            return;
        }
    };

    let nfloor = null_pivot_floor_estimate(&csc);
    println!(
        "\n=== {label}  n={n}  nnz={nnz}  expected_inertia=({},{},{})  \
         null_pivot_floor≈{nfloor:.3e} ===",
        expected.0, expected.1, expected.2
    );

    run_pass(
        label,
        &csc,
        &sym,
        &rhs,
        expected,
        ZeroPivotAction::ForceAccept,
        "A",
    );
    run_pass(
        label,
        &csc,
        &sym,
        &rhs,
        expected,
        ZeroPivotAction::Fail,
        "B",
    );
}

fn main() {
    println!("=== Issue #29 FBRAIN3LS pivot_threshold sweep ===");
    println!("Sweep: {PIVTOLS:?}");
    println!(
        "scaling=Identity (per default IPM-KKT recipe), \
         on_zero_pivot=ForceAccept (sparse multifrontal default)"
    );
    println!("rel_res uses solve_sparse_refined and JSON-supplied RHS");
    if !Path::new(CORPUS).exists() {
        eprintln!("CORPUS {CORPUS} not found — set up data/matrices symlink first");
        std::process::exit(2);
    }
    for &label in SAMPLES {
        run_sample(label);
    }
}
