//! Phase 2.2.3 — diagnose the CHWIRUT1 / CRESC100 / CRESC132 plateau.
//!
//! Post-2.2.2 these three matrices plateau at residuals 8–16 orders
//! above canonical MUMPS despite MC64 scaling and column-relative
//! pivot rejection. The research note
//! `dev/research/phase-2.2.3-plateau.md` enumerates five hypotheses;
//! this binary tests the cheap ones:
//!
//!   H1 — refinement best-iterate locked at unrefined solve
//!        (log per-iteration ‖r‖₂, ‖dx‖₂/‖x‖₂, break reason)
//!   H2 — residual at arithmetic noise floor
//!        (report ‖A‖₁·‖x‖∞ / ‖b‖∞ and Arioli ω₁)
//!   H4 — multi-supernode backsolve bug
//!        (run each matrix with nemin=32 and nemin=10000)
//!
//! Run with:
//!     cargo run --release --example triage_plateau
//!     cargo run --release --example triage_plateau -- CHWIRUT1/CHWIRUT1_0000

use std::path::Path;
use std::time::Instant;

use feral::numeric::factorize::{factorize_multifrontal, SparseFactors};
use feral::numeric::solve::solve_sparse;
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::{read_mtx, read_sidecar, BunchKaufmanParams, CscMatrix, Inertia, ZeroPivotAction};

fn norm2(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}
fn norm_inf(v: &[f64]) -> f64 {
    v.iter().fold(0.0f64, |a, &x| a.max(x.abs()))
}

/// Compute y = |A| * |x| for a symmetric CSC matrix stored lower-triangular.
fn abs_symv(a: &CscMatrix, x: &[f64], y: &mut [f64]) {
    for yi in y.iter_mut().take(a.n) {
        *yi = 0.0;
    }
    for j in 0..a.n {
        for k in a.col_ptr[j]..a.col_ptr[j + 1] {
            let i = a.row_idx[k];
            let v = a.values[k].abs();
            let xj = x[j].abs();
            y[i] += v * xj;
            if i != j {
                let xi = x[i].abs();
                y[j] += v * xi;
            }
        }
    }
}

/// Normwise ‖A‖₁ (= max column 1-norm, but for symmetric A this equals
/// ‖A‖∞ = max row 1-norm, computed here by absolute-row-sum).
fn a_norm_1(a: &CscMatrix) -> f64 {
    let mut row_sum = vec![0.0f64; a.n];
    for j in 0..a.n {
        for k in a.col_ptr[j]..a.col_ptr[j + 1] {
            let i = a.row_idx[k];
            let v = a.values[k].abs();
            row_sum[i] += v;
            if i != j {
                row_sum[j] += v;
            }
        }
    }
    row_sum.iter().fold(0.0f64, |m, &s| m.max(s))
}

fn rel_residual(a: &CscMatrix, x: &[f64], b: &[f64]) -> f64 {
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

/// Arioli ω₁ = max_i |r_i| / (|A|·|x| + |b|)_i
fn arioli_omega(a: &CscMatrix, x: &[f64], b: &[f64]) -> f64 {
    let n = a.n;
    let mut ax = vec![0.0; n];
    a.symv(x, &mut ax);
    let mut abs_ax = vec![0.0; n];
    abs_symv(a, x, &mut abs_ax);
    let mut omega = 0.0f64;
    for i in 0..n {
        let ri = (ax[i] - b[i]).abs();
        let denom = abs_ax[i] + b[i].abs();
        if denom > 0.0 {
            omega = omega.max(ri / denom);
        }
    }
    omega
}

/// Instrumented copy of `solve_sparse_refined`. Returns the final x
/// and prints the per-iteration trajectory.
fn solve_refined_verbose(
    a: &CscMatrix,
    factors: &SparseFactors,
    b: &[f64],
    max_steps: usize,
) -> Vec<f64> {
    let n = factors.n;
    let b_norm = norm2(b);
    let a1 = a_norm_1(a);

    let mut x = solve_sparse(factors, b).expect("solve_sparse");
    let mut r = vec![0.0; n];
    let mut ax = vec![0.0; n];
    a.symv(&x, &mut ax);
    for i in 0..n {
        r[i] = b[i] - ax[i];
    }
    let mut r_norm = norm2(&r);
    let x_norm0 = norm2(&x);
    let omega0 = arioli_omega(a, &x, b);

    println!(
        "  step 0  ‖r‖₂={:.3e}  ‖r‖₂/‖b‖₂={:.3e}  ‖x‖₂={:.3e}  ‖A‖₁·‖x‖∞/‖b‖∞={:.3e}  ω₁={:.3e}",
        r_norm,
        if b_norm > 0.0 {
            r_norm / b_norm
        } else {
            r_norm
        },
        x_norm0,
        a1 * norm_inf(&x) / norm_inf(b).max(f64::MIN_POSITIVE),
        omega0,
    );

    let mut best_x = x.clone();
    let mut best_r = r_norm;
    let threshold = f64::EPSILON * (n as f64).sqrt();
    let divergence_factor = 100.0;
    let mut last_break = "max_steps";

    for step in 1..=max_steps {
        let dx = solve_sparse(factors, &r).expect("solve_sparse dx");

        let mut x_new = x.clone();
        for i in 0..n {
            x_new[i] += dx[i];
        }
        let mut r_new = vec![0.0; n];
        let mut ax_new = vec![0.0; n];
        a.symv(&x_new, &mut ax_new);
        for i in 0..n {
            r_new[i] = b[i] - ax_new[i];
        }
        let r_new_norm = norm2(&r_new);
        let dx_norm = norm2(&dx);
        let x_new_norm = norm2(&x_new);
        let omega_new = arioli_omega(a, &x_new, b);

        let ratio = if x_new_norm > 0.0 {
            dx_norm / x_new_norm
        } else {
            dx_norm
        };
        let improved = r_new_norm < best_r;
        println!(
            "  step {}  ‖r‖₂={:.3e}  ‖dx‖₂/‖x‖₂={:.3e}  ω₁={:.3e}  {} {}",
            step,
            r_new_norm,
            ratio,
            omega_new,
            if improved { "↓best" } else { "  " },
            if r_new_norm > best_r * divergence_factor {
                "DIVERGED"
            } else {
                ""
            }
        );

        if improved {
            best_r = r_new_norm;
            best_x = x_new.clone();
        }

        x = x_new;
        r = r_new;
        r_norm = r_new_norm;

        if ratio < threshold {
            last_break = "converged";
            break;
        }
        if r_norm > best_r * divergence_factor {
            last_break = "diverged";
            break;
        }
    }
    println!(
        "  break: {}   best ‖r‖₂={:.3e}   best/‖b‖₂={:.3e}",
        last_break,
        best_r,
        if b_norm > 0.0 {
            best_r / b_norm
        } else {
            best_r
        }
    );
    best_x
}

fn params() -> BunchKaufmanParams {
    // Phase 2.3: restored pivot_threshold = 0.01 now that delayed
    // pivoting is in place. Matches bench::params_kkt and the
    // parity panel template in select_parity_panel.rs.
    BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        pivot_threshold: 0.01,
        ..BunchKaufmanParams::default()
    }
}

fn read_oracle(path: &Path) -> Option<(Inertia, f64)> {
    let data: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()?;
    let inertia = data.get("inertia")?;
    let pos = inertia.get("positive")?.as_u64()? as usize;
    let neg = inertia.get("negative")?.as_u64()? as usize;
    let zero = inertia.get("zero")?.as_u64()? as usize;
    let residual = data.get("residual_2norm_relative")?.as_f64()?;
    Some((
        Inertia {
            positive: pos,
            negative: neg,
            zero,
        },
        residual,
    ))
}

fn triage(stem: &str) {
    let base = format!("data/matrices/kkt/{}", stem);
    let mtx_path = format!("{}.mtx", base);
    let json_path = format!("{}.json", base);
    let mumps_path = format!("{}.mumps.json", base);

    if !Path::new(&mtx_path).exists() {
        println!("=== {} ===\n  SKIP: {} not found\n", stem, mtx_path);
        return;
    }
    println!("=== {} ===", stem);

    let mtx = read_mtx(Path::new(&mtx_path)).expect("read mtx");
    let csc = mtx.to_csc().expect("to_csc");
    let sc = read_sidecar(Path::new(&json_path)).expect("read sidecar");
    let rhs = sc.finite_rhs().expect("finite rhs");
    let oracle = read_oracle(Path::new(&mumps_path));

    let a1 = a_norm_1(&csc);
    let b_inf = norm_inf(&rhs);
    let b2 = norm2(&rhs);
    println!(
        "  n={}  nnz={}  ‖A‖₁={:.3e}  ‖b‖∞={:.3e}  ‖b‖₂={:.3e}",
        csc.n,
        csc.values.len(),
        a1,
        b_inf,
        b2
    );
    if let Some((i, r)) = &oracle {
        println!("  MUMPS oracle: inertia={}, residual={:.3e}", i, r);
    }

    for nemin in [32usize, 10000] {
        println!("\n  --- nemin={} ---", nemin);
        let snp = SupernodeParams {
            nemin,
            ..Default::default()
        };
        let t = Instant::now();
        let sym = symbolic_factorize(&csc, &snp).expect("symbolic");
        println!(
            "  symbolic: {:.1}ms  {} supernodes",
            t.elapsed().as_secs_f64() * 1000.0,
            sym.supernodes.len(),
        );

        let t = Instant::now();
        let (factors, inertia) = factorize_multifrontal(&csc, &sym, &params()).expect("factor");
        println!(
            "  factor:   {:.1}ms  inertia={}  scaling_info={:?}",
            t.elapsed().as_secs_f64() * 1000.0,
            inertia,
            factors.scaling_info,
        );
        if let Some((iref, _)) = &oracle {
            println!(
                "    vs MUMPS: {}",
                if &inertia == iref {
                    "MATCH"
                } else {
                    "MISMATCH"
                }
            );
        }

        // Unrefined baseline
        let x_un = solve_sparse(&factors, &rhs).expect("solve");
        let res_un = rel_residual(&csc, &x_un, &rhs);
        let omega_un = arioli_omega(&csc, &x_un, &rhs);
        println!(
            "  unrefined:  ‖r‖₂/‖b‖₂={:.3e}  ω₁={:.3e}",
            res_un, omega_un
        );

        println!("  refined trajectory (max 6 steps):");
        let x = solve_refined_verbose(&csc, &factors, &rhs, 6);
        let res = rel_residual(&csc, &x, &rhs);
        let omega = arioli_omega(&csc, &x, &rhs);
        println!("  final:  ‖r‖₂/‖b‖₂={:.3e}  ω₁={:.3e}", res, omega);
    }
    println!();
}

fn main() {
    let default_stems = [
        "CHWIRUT1/CHWIRUT1_0000",
        "CRESC100/CRESC100_0000",
        // CRESC132 is included but its plateau mechanism may be H5 (inertia).
        "CRESC132/CRESC132_0000",
    ];
    let args: Vec<String> = std::env::args().collect();
    let stems: Vec<String> = if args.len() > 1 {
        args[1..].to_vec()
    } else {
        default_stems.iter().map(|s| s.to_string()).collect()
    };
    for s in &stems {
        triage(s);
    }
}
