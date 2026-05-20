//! Synthesize a small saddle-point KKT that reproduces the issue-#45
//! MC64 catastrophic-spread blow-up, for use as a committed regression
//! test oracle (`src/scaling/mod.rs` tests T2/T4).
//!
//! The real trigger is the CHO `parmest` KKT (n=43332) — far too large
//! to commit. This probe builds a faithful small analog and measures
//! whether it has the four properties a committed test needs:
//!
//!   1. routes to `Mc64Symmetric` via `pick_scaling_strategy`
//!      (`max_col_nnz > 32` ∧ `diag_only/n >= 0.30`);
//!   2. InfNorm scaling spread >= 1e3, so the `IN_SPREAD_GUARD`
//!      early-return does not pre-empt the MC64 branch;
//!   3. MC64 scaling spread > `1/EPS ≈ 4.5e15` (the blow-up);
//!   4. factors+solves correctly under InfNorm (so the fallback the
//!      guard performs yields a usable answer, not more garbage).
//!
//! Construction — a parameter-estimation KKT `[H Bᵀ; B 0]`:
//!   * `ntheta` dense "parameter" primal columns, graded H diagonal
//!     `1 .. theta_top` (genuine moderate ill-conditioning InfNorm
//!     reflects → property 2). Each couples to every constraint, so
//!     `max_col_nnz = 1 + nc > 32` → arrow head.
//!   * `nx` "state" primal columns, H diagonal 0 (completed). Each
//!     couples to constraint c (coeff 1) and constraint c-1 (coeff
//!     `base`). The zero diagonal forces MC64 to match states through
//!     B, and the constant `base` factor telescopes along the chain
//!     into a path-accumulated potential → MC64 spread `base^(nx/2)`.
//!     A constant ratio gives InfNorm a translation-invariant chain →
//!     uniform InfNorm scaling, so InfNorm does NOT accumulate.
//!   * `nc = nx` "constraint" dual columns, (2,2) block 0 (completed)
//!     → degree-1 diag-only columns → `diag_only/n ≈ 0.5`.
//!
//! Usage: cargo run --release --bin probe_mc64_synth

use feral::numeric::factorize::factorize_multifrontal;
use feral::numeric::solve::solve_sparse;
use feral::scaling::{compute_scaling, pick_scaling_strategy, ScalingStrategy};
use feral::symbolic::{symbolic_factorize_with_method, OrderingMethod, SupernodeParams};
use feral::{CscMatrix, NumericParams};

fn norm_inf(v: &[f64]) -> f64 {
    v.iter().fold(0.0_f64, |m, &x| m.max(x.abs()))
}

fn spread(s: &[f64]) -> f64 {
    let mut mn = f64::INFINITY;
    let mut mx = 0.0_f64;
    for &x in s {
        let a = x.abs();
        if a > 0.0 && a < mn {
            mn = a;
        }
        if a > mx {
            mx = a;
        }
    }
    if mn.is_finite() && mn > 0.0 {
        mx / mn
    } else {
        f64::INFINITY
    }
}

/// Build the parameter-estimation saddle-point KKT described above.
fn build_kkt(ntheta: usize, nx: usize, theta_top: f64, base: f64, pcoef: f64) -> CscMatrix {
    let nc = nx;
    let n = ntheta + nx + nc;
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    let con0 = ntheta + nx; // first constraint global index

    // Parameter columns: graded H diagonal + coupling to every constraint.
    for p in 0..ntheta {
        let hp = if ntheta > 1 {
            theta_top.powf(p as f64 / (ntheta - 1) as f64)
        } else {
            1.0
        };
        rows.push(p);
        cols.push(p);
        vals.push(hp);
        for c in 0..nc {
            rows.push(con0 + c);
            cols.push(p);
            vals.push(pcoef);
        }
    }
    // State columns: zero H diagonal + chain coupling to constraints.
    for s in 0..nx {
        let js = ntheta + s;
        rows.push(js);
        cols.push(js);
        vals.push(0.0);
        // constraint s-1 couples state s as its "c+1" end (coeff base)
        if s >= 1 {
            rows.push(con0 + s - 1);
            cols.push(js);
            vals.push(base);
        }
        // constraint s couples state s as its "c" end (coeff 1.0)
        rows.push(con0 + s);
        cols.push(js);
        vals.push(1.0);
    }
    // Constraint columns: zero (2,2) diagonal only.
    for c in 0..nc {
        let jc = con0 + c;
        rows.push(jc);
        cols.push(jc);
        vals.push(0.0);
    }
    CscMatrix::from_triplets(n, &rows, &cols, &vals).expect("synthetic KKT triplets are valid")
}

/// Factor with the given explicit scaling; return (relres, min pivot).
fn factor_solve(m: &CscMatrix, scaling: ScalingStrategy) -> (f64, f64) {
    let snode = SupernodeParams::default();
    let np = NumericParams {
        scaling,
        ..NumericParams::default()
    };
    let sym = match symbolic_factorize_with_method(m, &snode, OrderingMethod::Auto) {
        Ok(s) => s,
        Err(_) => return (f64::NAN, f64::NAN),
    };
    let (factors, _) = match factorize_multifrontal(m, &sym, &np) {
        Ok(fi) => fi,
        Err(_) => return (f64::NAN, f64::NAN),
    };
    let minp = factors.min_pivot_magnitude().unwrap_or(f64::NAN);
    let rhs = vec![1.0_f64; m.n];
    match solve_sparse(&factors, &rhs) {
        Ok(x) => {
            let mut ax = vec![0.0; m.n];
            m.symv(&x, &mut ax);
            let r: Vec<f64> = ax.iter().zip(&rhs).map(|(&a, &b)| a - b).collect();
            (norm_inf(&r) / norm_inf(&rhs).max(1.0), minp)
        }
        Err(_) => (f64::NAN, minp),
    }
}

fn probe(ntheta: usize, nx: usize, theta_top: f64, base: f64, pcoef: f64) {
    let m = build_kkt(ntheta, nx, theta_top, base, pcoef);
    let routed = pick_scaling_strategy(&m);
    let in_s = compute_scaling(&m, &ScalingStrategy::InfNorm)
        .map(|(v, _)| spread(&v))
        .unwrap_or(f64::NAN);
    let (mc_v, mc_info) =
        compute_scaling(&m, &ScalingStrategy::Mc64Symmetric).expect("MC64 scaling");
    let mc_s = spread(&mc_v);
    let (auto_v, auto_info) = compute_scaling(&m, &ScalingStrategy::Auto).expect("Auto scaling");
    let auto_s = spread(&auto_v);
    let (in_r, in_p) = factor_solve(&m, ScalingStrategy::InfNorm);
    let (mc_r, mc_p) = factor_solve(&m, ScalingStrategy::Mc64Symmetric);
    let (au_r, au_p) = factor_solve(&m, ScalingStrategy::Auto);
    println!(
        "ntheta={ntheta} nx={nx} theta_top={theta_top:.0e} base={base} pcoef={pcoef}  n={}",
        m.n
    );
    println!("  routed={routed:?}  mc64_info={mc_info:?}");
    println!("  spread: in={in_s:.3e}  mc={mc_s:.3e}  auto={auto_s:.3e}  auto_info={auto_info:?}");
    println!("  InfNorm : relres={in_r:.3e}  min_pivot={in_p:.3e}");
    println!("  Mc64    : relres={mc_r:.3e}  min_pivot={mc_p:.3e}");
    println!("  Auto    : relres={au_r:.3e}  min_pivot={au_p:.3e}");
    println!();
}

fn main() {
    println!("guard threshold 1/EPS = {:.4e}\n", 1.0 / f64::EPSILON);
    // T2 oracle: MC64 spread far above the guard (base^(nx/2) ≈ 4^40).
    probe(8, 80, 1e8, 4.0, 0.5);
    // T3 oracle: reaches the MC64 branch (in_spread ≥ 1e3) with an MC64
    // spread BELOW the guard → the new guard must NOT fire; MC64 vector
    // is used. Empirically mc_spread ≈ base^(2·nx); sweep small bases.
    probe(8, 80, 1e8, 1.10, 0.5);
    probe(8, 80, 1e8, 1.15, 0.5);
    probe(8, 80, 1e8, 1.30, 0.5);
}
