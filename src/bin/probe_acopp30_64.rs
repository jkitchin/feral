//! Probe binary for feral issue #23 (ACOPP30 plateau-2).
//!
//! Loads ACOPP30_0064.mtx, runs `factorize_multifrontal` under a few
//! NumericParams variants, and reports D-spectrum + min|eig|+ rel_res
//! for each, isolating which knob is responsible for the literal-zero
//! pivot that defeats iterative refinement.

use std::path::Path;

use feral::numeric::factorize::{factorize_multifrontal, NumericParams, SparseFactors};
use feral::numeric::solver::Solver;
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::{read_mtx, CscMatrix};

const SUSPECTS: &[usize] = &[59, 63, 64, 65, 66, 67];

fn parse_rhs(json: &str, n: usize) -> Vec<f64> {
    let key = "\"rhs\":[";
    let i = json.find(key).unwrap() + key.len();
    let rest = &json[i..];
    let end = rest.find(']').unwrap();
    let body = &rest[..end];
    let v: Vec<f64> = body
        .split(',')
        .filter_map(|s| s.trim().parse::<f64>().ok())
        .collect();
    assert_eq!(v.len(), n);
    v
}

fn eigs_of_d(factors: &SparseFactors) -> Vec<f64> {
    let mut eigs = Vec::new();
    for nf in &factors.node_factors {
        let ff = &nf.frontal_factors;
        let nelim = ff.nelim;
        let mut k = 0;
        while k < nelim {
            let two_by_two = k + 1 < nelim && ff.d_subdiag[k] != 0.0;
            if two_by_two {
                let a = ff.d_diag[k];
                let b = ff.d_subdiag[k];
                let c = ff.d_diag[k + 1];
                let trace = a + c;
                let det = a * c - b * b;
                let disc = (trace * trace - 4.0 * det).max(0.0).sqrt();
                eigs.push((trace - disc) * 0.5);
                eigs.push((trace + disc) * 0.5);
                k += 2;
            } else {
                eigs.push(ff.d_diag[k]);
                k += 1;
            }
        }
    }
    eigs
}

fn count_2x2(factors: &SparseFactors) -> (usize, usize) {
    let mut n1 = 0;
    let mut n2 = 0;
    for nf in &factors.node_factors {
        let ff = &nf.frontal_factors;
        let nelim = ff.nelim;
        let mut k = 0;
        while k < nelim {
            let two_by_two = k + 1 < nelim && ff.d_subdiag[k] != 0.0;
            if two_by_two {
                n2 += 1;
                k += 2;
            } else {
                n1 += 1;
                k += 1;
            }
        }
    }
    (n1, n2)
}

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

fn rel_res(csc: &CscMatrix, x: &[f64], rhs: &[f64]) -> f64 {
    let mut ax = vec![0.0; csc.n];
    matvec_lower_sym(csc, x, &mut ax);
    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..csc.n {
        let d = ax[i] - rhs[i];
        num += d * d;
        den += rhs[i] * rhs[i];
    }
    num.sqrt() / den.sqrt().max(1.0)
}

fn try_params(label: &str, csc: &CscMatrix, rhs: &[f64], params: NumericParams) {
    let sym = symbolic_factorize(csc, &SupernodeParams::default()).unwrap();
    let (factors, inertia) = match factorize_multifrontal(csc, &sym, &params) {
        Ok(p) => p,
        Err(e) => {
            println!("  [{label:<22}] factor FAILED: {e:?}");
            return;
        }
    };
    let mut eigs = eigs_of_d(&factors);
    eigs.sort_by(|a, b| a.abs().partial_cmp(&b.abs()).unwrap());
    let n_zero = eigs.iter().filter(|v| **v == 0.0).count();
    let n_below_1e10 = eigs.iter().filter(|v| v.abs() < 1e-10).count();
    let (n1, n2) = count_2x2(&factors);

    // Use Solver with same params for solve path.
    let mut solver = Solver::with_params(params.clone(), SupernodeParams::default());
    solver.factor(csc, None);
    let rhs_buf = rhs.to_vec();
    let x_raw = solver.solve(&rhs_buf).map(|x| x.to_vec()).unwrap();
    let raw = rel_res(csc, &x_raw, rhs);
    let x_ref = solver.solve_refined(csc, rhs).map(|x| x.to_vec()).unwrap();
    let refined = rel_res(csc, &x_ref, rhs);

    print!(
        "  [{label:<22}] inertia={inertia}  pivots=({n1}x1,{n2}x2)  zero={n_zero}  <1e-10:{n_below_1e10}"
    );
    println!("  raw={raw:>8.2e}  ref={refined:>8.2e}");
    print!("    smallest |eigs|: ");
    for (i, v) in eigs.iter().take(6).enumerate() {
        if i > 0 {
            print!(", ");
        }
        print!("{v:+.3e}");
    }
    println!();
}

fn raw_diag_range(csc: &CscMatrix) -> f64 {
    let mut lo = f64::INFINITY;
    let mut hi = 0.0f64;
    for j in 0..csc.n {
        for k in csc.col_ptr[j]..csc.col_ptr[j + 1] {
            if csc.row_idx[k] == j {
                let a = csc.values[k].abs();
                if a > 0.0 {
                    lo = lo.min(a);
                    hi = hi.max(a);
                }
            }
        }
    }
    if lo.is_finite() && lo > 0.0 {
        hi / lo
    } else {
        f64::INFINITY
    }
}

fn max_off_diag_ratio(csc: &CscMatrix, s: &[f64]) -> f64 {
    let mut diag_abs = vec![0.0f64; csc.n];
    let mut max_off = vec![0.0f64; csc.n];
    for j in 0..csc.n {
        for k in csc.col_ptr[j]..csc.col_ptr[j + 1] {
            let i = csc.row_idx[k];
            let v = (csc.values[k] * s[i] * s[j]).abs();
            if i == j {
                diag_abs[j] = v;
            } else {
                max_off[i] = max_off[i].max(v);
                max_off[j] = max_off[j].max(v);
            }
        }
    }
    let mut w = 0.0f64;
    for j in 0..csc.n {
        let r = if diag_abs[j] > 0.0 {
            max_off[j] / diag_abs[j]
        } else if max_off[j] > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };
        w = w.max(r);
    }
    w
}

fn process(idx: usize) {
    let path = format!("data/matrices/kkt/ACOPP30/ACOPP30_{idx:04}.mtx");
    let json_path = format!("data/matrices/kkt/ACOPP30/ACOPP30_{idx:04}.json");
    let mtx = read_mtx(Path::new(&path)).unwrap();
    let csc = mtx.to_csc().unwrap();
    let json = std::fs::read_to_string(&json_path).unwrap();
    let rhs = parse_rhs(&json, csc.n);
    println!("\nACOPP30_{idx:04}: n={} nnz={}", csc.n, csc.row_idx.len());

    let picked = feral::scaling::pick_scaling_strategy(&csc);
    let drng = raw_diag_range(&csc);
    let (mc_vec, _) =
        feral::scaling::compute_scaling(&csc, &feral::scaling::ScalingStrategy::Mc64Symmetric)
            .unwrap();
    let (in_vec, _) =
        feral::scaling::compute_scaling(&csc, &feral::scaling::ScalingStrategy::InfNorm).unwrap();
    let mc_off = max_off_diag_ratio(&csc, &mc_vec);
    let in_off = max_off_diag_ratio(&csc, &in_vec);
    println!(
        "  router: picked={:?}  raw_diag_range={:.2e}  mc_off={:.2e}  in_off={:.2e}",
        picked, drng, mc_off, in_off,
    );

    try_params("Auto (default)   ", &csc, &rhs, NumericParams::default());
    let h = NumericParams {
        scaling: feral::scaling::ScalingStrategy::InfNorm,
        ..NumericParams::default()
    };
    try_params("InfNorm          ", &csc, &rhs, h);
    let i = NumericParams {
        scaling: feral::scaling::ScalingStrategy::Identity,
        ..NumericParams::default()
    };
    try_params("Identity         ", &csc, &rhs, i);
}

fn main() {
    for &idx in SUSPECTS {
        process(idx);
    }
}
