//! Issue #208 — which arm would AutoRace keep, under each candidate set,
//! and do the arms agree on inertia?
//!
//! `AutoRace` ranks on `factor_nnz_estimate`. This prints that figure and
//! the resulting inertia for every candidate on every matrix, so the
//! winner under the old set {Amd, MetisND, ScotchND, KahipND} and the new
//! set {Amd, Amf, MetisND} can be read off directly, along with whether
//! the two winners disagree about the matrix.

use feral::symbolic::OrderingMethod;
use feral::{read_mtx, FactorStatus, Solver};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let arms = [
        ("amd", OrderingMethod::Amd),
        ("amf", OrderingMethod::Amf),
        ("metis", OrderingMethod::MetisND),
        ("scotch", OrderingMethod::ScotchND),
        ("kahip", OrderingMethod::KahipND),
    ];
    println!(
        "{:<22} {:>8} {:>12} {:>22} {:>10}",
        "matrix", "arm", "nnz_est", "inertia", "status"
    );
    for p in &args {
        let Ok(mtx) = read_mtx(std::path::Path::new(p)) else {
            continue;
        };
        let Ok(m) = mtx.to_csc() else { continue };
        let name = std::path::Path::new(p)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("?");
        let mut best_old = (usize::MAX, "");
        let mut best_new = (usize::MAX, "");
        for (label, method) in &arms {
            let mut s = Solver::new().with_ordering(method.clone());
            let st = s.factor(&m, None);
            let ok = matches!(st, FactorStatus::Success);
            let nnz = s
                .work_estimate()
                .map(|e| e.factor_alloc_nnz)
                .unwrap_or(usize::MAX);
            let inr = s
                .inertia()
                .map(|i| format!("({}, {}, {})", i.positive, i.negative, i.zero))
                .unwrap_or_else(|| "-".into());
            println!(
                "{:<22} {:>8} {:>12} {:>22} {:>10}",
                name,
                label,
                nnz,
                inr,
                if ok { "ok" } else { "FAIL" }
            );
            if ok {
                if matches!(label, &"amd" | &"metis" | &"scotch" | &"kahip") && nnz < best_old.0 {
                    best_old = (nnz, label);
                }
                if matches!(label, &"amd" | &"amf" | &"metis") && nnz < best_new.0 {
                    best_new = (nnz, label);
                }
            }
        }
        println!(
            "  -> old set would keep {:<8} new set would keep {:<8} {}",
            best_old.1,
            best_new.1,
            if best_old.1 != best_new.1 {
                "*** DIFFERENT ***"
            } else {
                ""
            }
        );
    }
}
