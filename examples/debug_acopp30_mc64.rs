//! Phase 2.2.1 Step 8 triage: root-cause the ACOPP30 MC64 regression.
//!
//! Prints the scaling vector, its extrema, the ScalingInfo, feral
//! inertia with and without MC64, and key intermediate norms during
//! the solve so we can localize where the 1e46 magnitude appears.
//!
//! Not a test and not part of any suite. Run with:
//!
//!     cargo run --release --example debug_acopp30_mc64

use std::path::Path;

use feral::numeric::factorize::factorize_multifrontal;
use feral::numeric::solve::{solve_sparse, solve_sparse_refined};
use feral::scaling::{compute_scaling, ScalingInfo, ScalingStrategy};
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::{read_mtx, read_sidecar, BunchKaufmanParams, CscMatrix, ZeroPivotAction};

fn inf_norm(v: &[f64]) -> f64 {
    v.iter().fold(0.0f64, |m, &x| m.max(x.abs()))
}

fn two_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
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

fn summarize_vec(label: &str, v: &[f64]) {
    let mut min_fin = f64::INFINITY;
    let mut max_fin = f64::NEG_INFINITY;
    let mut n_nonfin = 0;
    let mut n_zero = 0;
    let mut n_neg = 0;
    let mut sum_log = 0.0f64;
    let mut n_log = 0usize;
    for &x in v {
        if !x.is_finite() {
            n_nonfin += 1;
            continue;
        }
        if x == 0.0 {
            n_zero += 1;
            continue;
        }
        if x < 0.0 {
            n_neg += 1;
        }
        let a = x.abs();
        if a < min_fin {
            min_fin = a;
        }
        if a > max_fin {
            max_fin = a;
        }
        sum_log += a.ln();
        n_log += 1;
    }
    let geomean = if n_log > 0 {
        (sum_log / n_log as f64).exp()
    } else {
        0.0
    };
    println!(
        "  {}: n={}  min|.|={:.3e}  max|.|={:.3e}  ratio={:.3e}  geomean={:.3e}  nonfin={}  zero={}  neg={}",
        label,
        v.len(),
        min_fin,
        max_fin,
        if min_fin > 0.0 { max_fin / min_fin } else { f64::NAN },
        geomean,
        n_nonfin,
        n_zero,
        n_neg,
    );
}

fn run_one_path(
    label: &str,
    csc: &CscMatrix,
    rhs: &[f64],
    params: &BunchKaufmanParams,
    strategy: ScalingStrategy,
) {
    println!("\n=== path: {} ===", label);
    let snode = SupernodeParams {
        scaling_strategy: strategy,
        ..SupernodeParams::default()
    };
    let sym = symbolic_factorize(csc, &snode).expect("symbolic");

    println!("  scaling_info: {:?}", sym.scaling_info);
    summarize_vec("scaling (user-order)", &sym.scaling);

    // Print first 10 entries
    let head: Vec<String> = sym
        .scaling
        .iter()
        .take(10)
        .map(|x| format!("{:.3e}", x))
        .collect();
    println!("  scaling[0..10] = [{}]", head.join(", "));

    let (factors, inertia) = factorize_multifrontal(csc, &sym, params).expect("factor");
    println!(
        "  feral inertia = {}, needs_refinement={}",
        inertia, factors.needs_refinement
    );

    // Walk node_factors and collect 1x1 pivot magnitudes (diagonal of D
    // across supernodes). Zero pivots are counted separately.
    let mut d_abs: Vec<f64> = Vec::new();
    let mut dets_abs: Vec<f64> = Vec::new();
    let mut zero_count = 0usize;
    let mut d_2x2 = 0usize;
    for node in &factors.node_factors {
        let ff = &node.frontal_factors;
        let mut k = 0;
        while k < ff.ncol {
            if k + 1 < ff.ncol && ff.d_subdiag[k] != 0.0 {
                let a = ff.d_diag[k];
                let b = ff.d_subdiag[k];
                let c = ff.d_diag[k + 1];
                let det = a * c - b * b;
                dets_abs.push(det.abs());
                if det.abs() > ff.zero_tol_2x2 {
                    // eigenvalues magnitudes approximated by trace/det
                    d_abs.push(a.abs().max(c.abs()));
                } else {
                    zero_count += 2;
                }
                d_2x2 += 1;
                k += 2;
            } else {
                let d = ff.d_diag[k];
                if d.abs() > ff.zero_tol {
                    d_abs.push(d.abs());
                } else {
                    zero_count += 1;
                }
                k += 1;
            }
        }
    }
    dets_abs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if !dets_abs.is_empty() {
        println!(
            "  2x2 |det| range: [{:.3e}, {:.3e}]  zero_tol_2x2={:.3e}",
            dets_abs[0],
            dets_abs[dets_abs.len() - 1],
            params.zero_tol_2x2,
        );
        println!(
            "  5 smallest 2x2 |det|: {:?}",
            &dets_abs.iter().take(5).collect::<Vec<_>>()
        );
    }
    d_abs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n_d = d_abs.len();
    if n_d > 0 {
        println!(
            "  D 1x1-accepted: {} (plus {} zero-accepted, {} 2x2 blocks)",
            n_d, zero_count, d_2x2
        );
        println!(
            "  |D| range: [{:.3e}, {:.3e}]  ratio={:.3e}",
            d_abs[0],
            d_abs[n_d - 1],
            d_abs[n_d - 1] / d_abs[0]
        );
        let p5 = d_abs[n_d * 5 / 100];
        let p50 = d_abs[n_d / 2];
        let p95 = d_abs[(n_d * 95).min(n_d * 100 - 1) / 100];
        println!(
            "  |D| quantiles: p5={:.3e} p50={:.3e} p95={:.3e}",
            p5, p50, p95
        );
        // Print the 5 smallest accepted pivots
        println!(
            "  5 smallest accepted |D|: {:?}",
            &d_abs.iter().take(5).collect::<Vec<_>>()
        );
    }

    // Plain solve
    let x_plain = solve_sparse(&factors, rhs).expect("solve");
    let mut ax = vec![0.0; csc.n];
    csc.symv(&x_plain, &mut ax);
    println!(
        "  plain: ||x||_inf={:.3e}  ||Ax||_inf={:.3e}  ||b||_inf={:.3e}  rel_res={:.3e}",
        inf_norm(&x_plain),
        inf_norm(&ax),
        inf_norm(rhs),
        rel_residual(csc, &x_plain, rhs),
    );
    println!(
        "  plain: ||x||_2={:.3e}  ||Ax-b||_2={:.3e}",
        two_norm(&x_plain),
        {
            let mut r = vec![0.0; csc.n];
            for i in 0..csc.n {
                r[i] = ax[i] - rhs[i];
            }
            two_norm(&r)
        }
    );

    // Refined solve
    let x_ref = solve_sparse_refined(csc, &factors, rhs).expect("ref");
    let mut ax2 = vec![0.0; csc.n];
    csc.symv(&x_ref, &mut ax2);
    println!(
        "  refined: ||x||_inf={:.3e}  ||Ax||_inf={:.3e}  rel_res={:.3e}",
        inf_norm(&x_ref),
        inf_norm(&ax2),
        rel_residual(csc, &x_ref, rhs),
    );
}

fn main() {
    let mtx_path = Path::new("data/matrices/kkt/ACOPP30/ACOPP30_0000.mtx");
    let json_path = Path::new("data/matrices/kkt/ACOPP30/ACOPP30_0000.json");

    let mtx = read_mtx(mtx_path).expect("read mtx");
    let csc = mtx.to_csc().expect("to_csc");
    let sc = read_sidecar(json_path).expect("read sidecar");
    let rhs = sc.finite_rhs().expect("finite rhs");
    let n = csc.n;

    println!("=== ACOPP30_0000 MC64 regression triage ===");
    println!("n={}, nnz={}", n, csc.nnz());
    println!(
        "IPOPT inertia: ({},{},{})  delta_w={:?}  delta_c={:?}",
        sc.inertia.positive, sc.inertia.negative, sc.inertia.zero, sc.delta_w, sc.delta_c
    );
    println!(
        "||b||_inf={:.3e}  ||b||_2={:.3e}",
        inf_norm(&rhs),
        two_norm(&rhs)
    );

    // Matrix magnitude analysis
    let mut min_diag = f64::INFINITY;
    let mut max_diag: f64 = 0.0;
    let mut zero_diag = 0usize;
    let mut max_off: f64 = 0.0;
    for col in 0..n {
        for k in csc.col_ptr[col]..csc.col_ptr[col + 1] {
            let row = csc.row_idx[k];
            let v = csc.values[k].abs();
            if row == col {
                if v == 0.0 {
                    zero_diag += 1;
                } else {
                    if v < min_diag {
                        min_diag = v;
                    }
                    if v > max_diag {
                        max_diag = v;
                    }
                }
            } else if v > max_off {
                max_off = v;
            }
        }
    }
    println!(
        "diag |.| range: [{:.3e}, {:.3e}] (ratio {:.3e}, zeros {})",
        min_diag,
        max_diag,
        max_diag / min_diag,
        zero_diag
    );
    println!("max |off-diag|: {:.3e}", max_off);

    let params = BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        ..BunchKaufmanParams::default()
    };

    // Compute scaling vector directly (user-order)
    let (scaling, info) = compute_scaling(&csc, &ScalingStrategy::Mc64Symmetric).expect("mc64");
    println!("\n--- direct compute_scaling(Mc64Symmetric) ---");
    println!("  info = {:?}", info);
    summarize_vec("scaling", &scaling);
    let nfin = scaling.iter().filter(|x| x.is_finite()).count();
    println!("  n_finite = {} / {}", nfin, n);

    // Find largest and smallest scaling entries with their indices
    let mut pairs: Vec<(usize, f64)> = scaling.iter().enumerate().map(|(i, &x)| (i, x)).collect();
    pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    println!(
        "  5 smallest:  {:?}",
        &pairs.iter().take(5).collect::<Vec<_>>()
    );
    println!(
        "  5 largest:   {:?}",
        &pairs.iter().rev().take(5).collect::<Vec<_>>()
    );

    // Paths
    run_one_path(
        "Identity (no MC64)",
        &csc,
        &rhs,
        &params,
        ScalingStrategy::Identity,
    );
    run_one_path(
        "Mc64Symmetric",
        &csc,
        &rhs,
        &params,
        ScalingStrategy::Mc64Symmetric,
    );

    // External: all-ones (should match Identity modulo ScalingInfo tag)
    run_one_path(
        "External([1.0; n])",
        &csc,
        &rhs,
        &params,
        ScalingStrategy::External(vec![1.0; n]),
    );

    // Hypothesis 4: run with MC64 but replace any "extreme" scaling entries
    // with 1.0, to test whether a few outlier entries drive the blowup.
    let mut scaling_capped = scaling.clone();
    let mut n_capped = 0usize;
    for s in scaling_capped.iter_mut() {
        if !s.is_finite() || *s == 0.0 || s.abs() > 1e6 || s.abs() < 1e-6 {
            *s = 1.0;
            n_capped += 1;
        }
    }
    println!(
        "\n--- capped MC64: {} extreme entries rewritten to 1.0 ---",
        n_capped
    );
    if n_capped > 0 {
        run_one_path(
            "MC64 capped (|s| in [1e-6,1e6])",
            &csc,
            &rhs,
            &params,
            ScalingStrategy::External(scaling_capped),
        );
    }

    // Hypothesis 6 (part a): run refinement manually with an unscaled oracle
    // residual, to see whether `solve_sparse_refined` is actually tracking
    // the best iterate on the MC64 path.
    {
        println!("\n--- manual refinement trace (MC64 path) ---");
        let snode = SupernodeParams {
            scaling_strategy: ScalingStrategy::Mc64Symmetric,
            ..SupernodeParams::default()
        };
        let sym = symbolic_factorize(&csc, &snode).expect("symbolic");
        let (factors, _) = factorize_multifrontal(&csc, &sym, &params).expect("factor");

        let mut x = solve_sparse(&factors, &rhs).expect("solve");
        let mut r = vec![0.0; n];
        let mut ax = vec![0.0; n];
        csc.symv(&x, &mut ax);
        for i in 0..n {
            r[i] = rhs[i] - ax[i];
        }
        let r0 = two_norm(&r);
        println!(
            "  step 0: ||r||_2={:.3e}  ||x||_inf={:.3e}  rel_res={:.3e}",
            r0,
            inf_norm(&x),
            rel_residual(&csc, &x, &rhs)
        );

        for step in 1..=5 {
            let dx = solve_sparse(&factors, &r).expect("solve");
            for i in 0..n {
                x[i] += dx[i];
            }
            csc.symv(&x, &mut ax);
            for i in 0..n {
                r[i] = rhs[i] - ax[i];
            }
            println!(
                "  step {}: ||dx||_2={:.3e}  ||r||_2={:.3e}  ||x||_inf={:.3e}  rel_res={:.3e}",
                step,
                two_norm(&dx),
                two_norm(&r),
                inf_norm(&x),
                rel_residual(&csc, &x, &rhs)
            );
        }
    }

    // Hypothesis probe: what if we loosen zero_tol on the MC64 path so
    // the very-small pivots are force-accepted rather than inverted?
    println!("\n--- MC64 + loosened zero_tol = 1e-8 ---");
    let loose = BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        zero_tol: 1e-8,
        zero_tol_2x2: 1e-16,
        ..BunchKaufmanParams::default()
    };
    run_one_path(
        "MC64 + zero_tol=1e-8",
        &csc,
        &rhs,
        &loose,
        ScalingStrategy::Mc64Symmetric,
    );

    // Same loosened threshold, no MC64 — control
    println!("\n--- Identity + loosened zero_tol = 1e-8 ---");
    run_one_path(
        "Identity + zero_tol=1e-8",
        &csc,
        &rhs,
        &loose,
        ScalingStrategy::Identity,
    );

    // Orthogonal check: ScalingInfo match
    match info {
        ScalingInfo::Applied => println!("\nScalingInfo::Applied (non-singular matching)"),
        ScalingInfo::PartialSingular { n_unmatched } => println!(
            "\nScalingInfo::PartialSingular {{ n_unmatched = {} }}",
            n_unmatched
        ),
        ScalingInfo::NotApplied => println!("\nScalingInfo::NotApplied"),
    }
}
