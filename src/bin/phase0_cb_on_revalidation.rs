//! Phase 0 of the issue #55 plan: re-validate historical CB-on regressions
//! at HEAD. If all clear, flipping `FeralConfig::default()` to CB-on is a
//! one-line fix for the pinene_3200 / nql180 cascade-victim class.
//!
//! For each matrix in the historical regression corpus, runs the default
//! `Solver` with `cascade_break_ratio = Some(0.5)` and
//! `cascade_break_eps = Some(1e-10)` forced on, then compares the
//! reported inertia against the sidecar JSON oracle (or a hardcoded
//! oracle for matrices that don't carry one).
//!
//! Output: a pass/fail table per matrix. Exit code is 0 if all pass,
//! 1 if any regression.
//!
//! Usage:
//!     cargo run --release --bin phase0_cb_on_revalidation

use std::path::Path;
use std::time::Instant;

use feral::numeric::factorize::NumericParams;
use feral::numeric::solver::{FactorStatus, Solver};
use feral::symbolic::SupernodeParams;
use feral::{read_mtx, read_sidecar, Inertia};

#[derive(Debug)]
struct Case {
    label: String,
    mtx_path: String,
    oracle: Inertia,
}

#[derive(Debug)]
struct Outcome {
    pass: bool,
    detail: String,
}

fn cb_on_params() -> NumericParams {
    NumericParams {
        cascade_break_ratio: Some(0.5),
        cascade_break_eps: Some(1e-10),
        ..NumericParams::default()
    }
}

fn run_case(case: &Case) -> Outcome {
    let path = Path::new(&case.mtx_path);
    if !path.exists() {
        return Outcome {
            pass: true,
            detail: format!("SKIP (missing: {})", case.mtx_path),
        };
    }
    let mtx = match read_mtx(path) {
        Ok(m) => m,
        Err(e) => {
            return Outcome {
                pass: false,
                detail: format!("read_mtx error: {e:?}"),
            }
        }
    };
    let csc = match mtx.to_csc() {
        Ok(c) => c,
        Err(e) => {
            return Outcome {
                pass: false,
                detail: format!("to_csc error: {e:?}"),
            }
        }
    };
    let mut solver = Solver::with_params(cb_on_params(), SupernodeParams::default());
    let t0 = Instant::now();
    let status = solver.factor(&csc, Some(case.oracle.clone()));
    let dt = t0.elapsed().as_secs_f64();
    match status {
        FactorStatus::Success => {
            let got = match solver.inertia() {
                Some(i) => i.clone(),
                None => {
                    return Outcome {
                        pass: false,
                        detail: format!("Success but no inertia after {dt:.3}s"),
                    }
                }
            };
            if got.positive == case.oracle.positive
                && got.negative == case.oracle.negative
                && got.zero == case.oracle.zero
            {
                Outcome {
                    pass: true,
                    detail: format!(
                        "PASS  factor={dt:.3}s  inertia=({}, {}, {})",
                        got.positive, got.negative, got.zero
                    ),
                }
            } else {
                Outcome {
                    pass: false,
                    detail: format!(
                        "FAIL inertia mismatch  factor={dt:.3}s  got=({}, {}, {}) expected=({}, {}, {})",
                        got.positive,
                        got.negative,
                        got.zero,
                        case.oracle.positive,
                        case.oracle.negative,
                        case.oracle.zero,
                    ),
                }
            }
        }
        FactorStatus::WrongInertia { actual, expected } => Outcome {
            pass: false,
            detail: format!(
                "FAIL WrongInertia  factor={dt:.3}s  got=({}, {}, {}) expected=({}, {}, {})",
                actual.positive,
                actual.negative,
                actual.zero,
                expected.positive,
                expected.negative,
                expected.zero,
            ),
        },
        FactorStatus::Singular => Outcome {
            pass: false,
            detail: format!("FAIL Singular  factor={dt:.3}s"),
        },
        FactorStatus::FatalError(e) => Outcome {
            pass: false,
            detail: format!("FAIL FatalError({e:?})  factor={dt:.3}s"),
        },
    }
}

fn cases_from_sidecars(family: &str, iters: &[u32]) -> Vec<Case> {
    let mut out = Vec::new();
    for &i in iters {
        let base = format!("data/matrices/kkt-mittelmann/{family}/{family}_{i:04}");
        let mtx_path = format!("{base}.mtx");
        let json_path = format!("{base}.json");
        if !Path::new(&json_path).exists() {
            continue;
        }
        let sc = match read_sidecar(Path::new(&json_path)) {
            Ok(s) => s,
            Err(_) => continue,
        };
        out.push(Case {
            label: format!("{family}_{i:04}"),
            mtx_path,
            oracle: Inertia::new(sc.inertia.positive, sc.inertia.negative, sc.inertia.zero),
        });
    }
    out
}

fn hardcoded_nuffield2_trap() -> Case {
    // Issue #54: nuffield2_trap iter 1 has expected_neg = 13202, n = 26649.
    // dim = pos + neg + zero = 26649; positive = 26649 - 13202 = 13447.
    Case {
        label: "nuffield2_trap_iter1".to_string(),
        mtx_path: "dev/repros/issue-54/nuffield2_trap_iter1.mtx".to_string(),
        oracle: Inertia::new(13447, 13202, 0),
    }
}

fn main() {
    let mut all_cases: Vec<Case> = Vec::new();

    // Issue #17 — robot_1600. Iter 3 was the cited regression iteration;
    // include the full available range as a sweep.
    all_cases.extend(cases_from_sidecars(
        "robot_1600",
        &(0..=6).collect::<Vec<_>>(),
    ));

    // Issue #18 — NARX_CFy. Mid-IPM iters were the cited regression
    // (solve_001, solve_100, solve_400); only iters 0-2 are checked in.
    all_cases.extend(cases_from_sidecars("NARX_CFy", &[0, 1, 2]));

    // Issue #48 — marine_1600. Cited iter 4; sweep the full range.
    all_cases.extend(cases_from_sidecars(
        "marine_1600",
        &(0..=17).collect::<Vec<_>>(),
    ));

    // Issue #38 — rocket_12800. BOTH-mode failure historically.
    all_cases.extend(cases_from_sidecars("rocket_12800", &[0, 1]));

    // Issue #46 — pinene_3200. CHO cascade target; iter 9 was the
    // 88s-under-CB-off case.
    all_cases.extend(cases_from_sidecars(
        "pinene_3200",
        &(0..=9).collect::<Vec<_>>(),
    ));

    // Issue #54 — nuffield2_trap. Hardcoded oracle (no sidecar).
    all_cases.push(hardcoded_nuffield2_trap());

    println!(
        "phase0_cb_on_revalidation: {} historical regression cases\n\
         configuration: cascade_break_ratio=Some(0.5), cascade_break_eps=Some(1e-10)\n",
        all_cases.len()
    );

    let mut n_pass = 0usize;
    let mut n_fail = 0usize;
    let mut n_skip = 0usize;
    let mut failures: Vec<(String, String)> = Vec::new();

    for case in &all_cases {
        let out = run_case(case);
        let kind = if out.detail.starts_with("SKIP") {
            n_skip += 1;
            "skip"
        } else if out.pass {
            n_pass += 1;
            "pass"
        } else {
            n_fail += 1;
            failures.push((case.label.clone(), out.detail.clone()));
            "FAIL"
        };
        println!("  [{kind}] {:<28}  {}", case.label, out.detail);
    }

    println!(
        "\nsummary: pass={n_pass}  fail={n_fail}  skip={n_skip}  total={}",
        all_cases.len()
    );

    if !failures.is_empty() {
        println!("\nFAILURES (Phase B / additional Phase A scope needed):");
        for (label, detail) in &failures {
            println!("  - {label}: {detail}");
        }
        std::process::exit(1);
    } else if n_pass == 0 {
        println!(
            "\nNo cases ran (corpus likely missing). Re-run with the matrix corpus available."
        );
        std::process::exit(2);
    } else {
        println!(
            "\nALL CLEAR: historical CB-on regressions do not reproduce at HEAD. \
             Phase 0.4 (default flip) is unblocked."
        );
    }
}
