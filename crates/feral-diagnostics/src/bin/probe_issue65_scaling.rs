//! Issue #65 probe: Auto scaling reports wrong inertia (spurious zero
//! pivots) on ill-conditioned indefinite KKTs; Mc64Symmetric fixes it.
//!
//! For each ScalingStrategy, factor the matrix and report inertia +
//! min|pivot| + status. Also report the shape stats
//! `pick_scaling_strategy` keys on (diag_only/n, max_col_nnz) and what
//! Auto resolves to, to see why Auto mis-routes.
//!
//! Run: `cargo run --release --bin probe_issue65_scaling -- <kkt.mtx> [expected_neg]`
//!
//! Throwaway diagnostic, not a test.

use feral::scaling::{pick_scaling_strategy, ScalingStrategy};
use feral::{read_mtx, CscMatrix, FactorStatus, Solver};
use std::path::Path;

fn shape_stats(m: &CscMatrix) -> (usize, usize) {
    let n = m.n;
    let mut diag_only = 0usize;
    let mut max_col_nnz = 0usize;
    for j in 0..n {
        let mut nnz_col = 0usize;
        let mut diag_nonzero = false;
        for k in m.col_ptr[j]..m.col_ptr[j + 1] {
            if m.values[k] == 0.0 {
                continue;
            }
            nnz_col += 1;
            if m.row_idx[k] == j {
                diag_nonzero = true;
            }
        }
        max_col_nnz = max_col_nnz.max(nnz_col);
        if nnz_col == 1 && diag_nonzero {
            diag_only += 1;
        }
    }
    (diag_only, max_col_nnz)
}

fn status_short(s: &FactorStatus) -> String {
    match s {
        FactorStatus::Success => "Success".to_string(),
        FactorStatus::Singular => "Singular".to_string(),
        FactorStatus::WrongInertia { actual, expected } => {
            format!("WrongInertia(act {actual:?} exp {expected:?})")
        }
        FactorStatus::FatalError(e) => format!("FatalError({e:?})"),
        FactorStatus::Interrupted => "Interrupted".to_string(),
    }
}

fn run(m: &CscMatrix, scaling: ScalingStrategy, expected_neg: Option<usize>) {
    let mut s = Solver::new().with_scaling(scaling.clone());
    let status = s.factor(m, None);
    let inertia = s.inertia().cloned();
    let minp = s.min_pivot_magnitude();
    let label = format!("{scaling:?}");
    let label = if label.len() > 14 {
        format!("{}…", &label[..13])
    } else {
        label
    };
    let instr = match &inertia {
        Some(i) => {
            let flag = match expected_neg {
                Some(e) if i.negative == e && i.zero == 0 => "✓",
                _ => "✗",
            };
            format!("({},{},{}) {flag}", i.positive, i.negative, i.zero)
        }
        None => "none".to_string(),
    };
    println!(
        "  {label:<15} inertia={instr:<18} min|piv|={:.4e}  status={}",
        minp.unwrap_or(f64::NAN),
        status_short(&status),
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: probe_issue65_scaling <kkt.mtx> [expected_neg]");
        return;
    }
    let mtx = Path::new(&args[1]);
    let expected_neg: Option<usize> = args.get(2).and_then(|s| s.parse().ok());
    let m = match read_mtx(mtx).and_then(|x| x.to_csc()) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("read failed: {e:?}");
            return;
        }
    };
    let (diag_only, max_col_nnz) = shape_stats(&m);
    let resolved = pick_scaling_strategy(&m);
    println!(
        "=== {} (n={}, nnz={}) ===",
        mtx.file_name().and_then(|s| s.to_str()).unwrap_or("?"),
        m.n,
        m.row_idx.len()
    );
    println!(
        "  router: diag_only/n = {}/{} = {:.3}  max_col_nnz = {}  (gates: ratio>=0.30, nnz>32)",
        diag_only,
        m.n,
        diag_only as f64 / m.n as f64,
        max_col_nnz
    );
    println!("  Auto resolves to: {resolved:?}");
    if let Some(e) = expected_neg {
        println!("  expected: (.,{e},0)");
    }
    for sc in [
        ScalingStrategy::Auto,
        ScalingStrategy::InfNorm,
        ScalingStrategy::Mc64Symmetric,
        ScalingStrategy::Identity,
    ] {
        run(&m, sc, expected_neg);
    }
}
