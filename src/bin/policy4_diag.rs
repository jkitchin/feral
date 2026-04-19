//! Policy 4 triage: find a separating feature between matrices where
//! MC64 helps (VESUVIO/CRESC) vs. matrices where MC64 hurts
//! (MSS1_0009, the boundary-regression set from the lever-c residual
//! diff). The hypothesis is that one of the cheap diagnostics from
//! `polak6_diag.rs` discriminates them.
//!
//! For each matrix prints:
//!   - shape: n, nnz, diag_only/n
//!   - raw |diag|: min, max, range
//!   - InfNorm scaling: min, max, range; scaled |diag| min/max/range;
//!     scaled max(|off|/|diag|)
//!   - MC64    scaling: same fields
//!   - bench-corpus residual (for context, hard-coded from the dump)
//!
//! Usage: `cargo run --release --bin policy4_diag`. No production code change.

use feral::scaling::{compute_scaling, ScalingStrategy};
use feral::{read_mtx, CscMatrix};
use std::path::PathBuf;

fn matrix_diagonal(csc: &CscMatrix) -> Vec<f64> {
    let n = csc.n;
    let mut d = vec![0.0; n];
    for (j, dj) in d.iter_mut().enumerate().take(n) {
        for k in csc.col_ptr[j]..csc.col_ptr[j + 1] {
            if csc.row_idx[k] == j {
                *dj = csc.values[k];
            }
        }
    }
    d
}

fn scaled_diag_and_offratio(csc: &CscMatrix, scaling: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = csc.n;
    let mut diag = vec![0.0_f64; n];
    let mut max_off = vec![0.0_f64; n];
    for j in 0..n {
        for k in csc.col_ptr[j]..csc.col_ptr[j + 1] {
            let i = csc.row_idx[k];
            let v = csc.values[k] * scaling[i] * scaling[j];
            if i == j {
                diag[j] = v;
            } else {
                let av = v.abs();
                if av > max_off[i] {
                    max_off[i] = av;
                }
                if av > max_off[j] {
                    max_off[j] = av;
                }
            }
        }
    }
    let ratio: Vec<f64> = diag
        .iter()
        .zip(max_off.iter())
        .map(|(&d, &m)| {
            if d.abs() > 0.0 {
                m / d.abs()
            } else {
                f64::INFINITY
            }
        })
        .collect();
    (diag, ratio)
}

#[derive(Default)]
struct Features {
    n: usize,
    #[allow(dead_code)]
    nnz: usize,
    diag_only_ratio: f64,
    raw_diag_range: f64,
    infnorm_svec_range: f64,
    infnorm_scaled_diag_range: f64,
    infnorm_scaled_offratio: f64,
    mc64_svec_range: f64,
    mc64_scaled_diag_range: f64,
    mc64_scaled_offratio: f64,
}

fn extreme_range(values: &[f64]) -> f64 {
    let mut lo = f64::INFINITY;
    let mut hi = 0.0_f64;
    for &v in values {
        let a = v.abs();
        if a > 0.0 {
            lo = lo.min(a);
            hi = hi.max(a);
        }
    }
    if lo.is_finite() && lo > 0.0 {
        hi / lo
    } else {
        f64::INFINITY
    }
}

fn analyze(csc: &CscMatrix) -> Features {
    let mut f = Features {
        n: csc.n,
        nnz: csc.row_idx.len(),
        ..Default::default()
    };
    let mut diag_only = 0usize;
    for j in 0..csc.n {
        let len = csc.col_ptr[j + 1] - csc.col_ptr[j];
        if len == 1 && csc.row_idx[csc.col_ptr[j]] == j {
            diag_only += 1;
        }
    }
    f.diag_only_ratio = diag_only as f64 / csc.n as f64;
    let raw = matrix_diagonal(csc);
    f.raw_diag_range = extreme_range(&raw);
    if let Ok((s, _)) = compute_scaling(csc, &ScalingStrategy::InfNorm) {
        f.infnorm_svec_range = extreme_range(&s);
        let (d, r) = scaled_diag_and_offratio(csc, &s);
        f.infnorm_scaled_diag_range = extreme_range(&d);
        f.infnorm_scaled_offratio = r.iter().cloned().fold(0.0_f64, f64::max);
    }
    if let Ok((s, _)) = compute_scaling(csc, &ScalingStrategy::Mc64Symmetric) {
        f.mc64_svec_range = extreme_range(&s);
        let (d, r) = scaled_diag_and_offratio(csc, &s);
        f.mc64_scaled_diag_range = extreme_range(&d);
        f.mc64_scaled_offratio = r.iter().cloned().fold(0.0_f64, f64::max);
    }
    f
}

fn print_header() {
    println!(
        "{:<22} {:>6} {:>5} {:>10} {:>10}   {:>10} {:>10} {:>10}   {:>10} {:>10} {:>10}",
        "matrix",
        "n",
        "d/n",
        "raw_drng",
        "label",
        "in_svec",
        "in_drng",
        "in_off",
        "mc_svec",
        "mc_drng",
        "mc_off",
    );
    println!("{}", "-".repeat(140));
}

fn print_row(name: &str, label: &str, f: &Features) {
    println!(
        "{:<22} {:>6} {:>5.2} {:>10.2e} {:>10}   {:>10.2e} {:>10.2e} {:>10.2e}   {:>10.2e} {:>10.2e} {:>10.2e}",
        name,
        f.n,
        f.diag_only_ratio,
        f.raw_diag_range,
        label,
        f.infnorm_svec_range,
        f.infnorm_scaled_diag_range,
        f.infnorm_scaled_offratio,
        f.mc64_svec_range,
        f.mc64_scaled_diag_range,
        f.mc64_scaled_offratio,
    );
}

fn load(family: &str, name: &str) -> Option<CscMatrix> {
    let candidates = [
        format!("data/matrices/kkt/{}/{}.mtx", family, name),
        format!("tests/data/parity/{}/{}.mtx", family.to_lowercase(), name),
    ];
    for path_str in candidates {
        let path = PathBuf::from(&path_str);
        if path.exists() {
            return read_mtx(&path).ok().and_then(|m| m.to_csc().ok());
        }
    }
    None
}

fn main() {
    println!("Policy 4 triage — feature comparison across MC64-helps vs MC64-hurts matrices");
    println!();
    print_header();

    // MC64-hurts (regressions in lever-c residual diff)
    let regressions = [
        ("MSS1", "MSS1_0009", "REGR-mat"),
        ("HATFLDFL", "HATFLDFL_0315", "REGR-edge"),
        ("HATFLDFL", "HATFLDFL_0422", "REGR-edge"),
        ("HATFLDFL", "HATFLDFL_0490", "REGR-edge"),
        ("SNAKE", "SNAKE_0101", "REGR-edge"),
        ("ALLINITA", "ALLINITA_0758", "REGR-edge"),
        ("POLAK6", "POLAK6_0021", "REGR-excl"),
    ];
    for (fam, name, label) in &regressions {
        if let Some(c) = load(fam, name) {
            print_row(name, label, &analyze(&c));
        } else {
            println!("{:<22} <missing>", name);
        }
    }
    println!();

    // Parity-test matrices that may have flipped under Policy 4.
    let parity = [
        ("MEYER3NE", "MEYER3NE_0220", "PARITY"),
        ("MEYER3NE", "MEYER3NE_0259", "PARITY"),
        ("MEYER3NE", "MEYER3NE_0253", "PARITY"),
    ];
    for (fam, name, label) in &parity {
        if let Some(c) = load(fam, name) {
            print_row(name, label, &analyze(&c));
        } else {
            println!("{:<22} <missing>", name);
        }
    }
    println!();

    // MC64-helps wins
    let wins = [
        ("HS75", "HS75_0000", "WIN-mat"),
        ("KOEBHELB", "KOEBHELB_0004", "WIN-mat"),
        ("VESUVIA", "VESUVIA_0000", "WIN-perf"),
        ("VESUVIO", "VESUVIO_0000", "WIN-perf"),
        ("VESUVIOU", "VESUVIOU_0000", "WIN-perf"),
        ("MUONSINE", "MUONSINE_0000", "WIN-perf"),
        ("CRESC132", "CRESC132_0000", "WIN-perf"),
    ];
    for (fam, name, label) in &wins {
        if let Some(c) = load(fam, name) {
            print_row(name, label, &analyze(&c));
        } else {
            println!("{:<22} <missing>", name);
        }
    }
}
