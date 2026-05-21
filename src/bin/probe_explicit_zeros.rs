//! Issue #47 — do explicit-zero entries still cost a factor-time penalty?
//!
//! POUNCE fixes its CHO `parmest` KKT sparsity pattern once and refills
//! values each IPM iterate; at points where a multiplier is zero those
//! slots arrive as an explicit `0.0`. `from_triplets` keeps explicit
//! zeros, so they reach the symbolic phase and can perturb ordering /
//! fill / supernode formation. POUNCE works around this by stripping
//! explicit zeros in its `pounce-feral` backend before every `factor()`.
//!
//! This probe factors the same CHO iter-0 KKT two ways on the current
//! HEAD — after Fix 1 (fine-grained delayed pivoting, `42434a5`) and
//! Fix 2 (cancellation-free 2×2 inertia, `80c05f5`):
//!
//!   stripped  — `cho_iter0_kkt.mtx`, 205512 nnz, no explicit zeros.
//!   kept      — `cho_iter0_kkt_with_explicit_zeros.txt`, 326245 nnz,
//!               the pattern POUNCE actually builds (120733 explicit 0).
//!
//! For each it reports factor wall time, factor nnz (fill), inertia,
//! and the solve residual against the dumped RHS, so we can see whether
//! the ~2× aggregate POUNCE slowdown (issue #47) still reproduces on a
//! single iterate post-fix.
//!
//! Usage: cargo run --release --bin probe_explicit_zeros

use std::io::BufRead;
use std::path::Path;
use std::time::Instant;

use feral::{read_mtx, CscMatrix, Solver};

const DIR: &str = "/Users/jkitchin/projects/pounce/benchmarks/cho/feral_repro";

fn norm_inf(v: &[f64]) -> f64 {
    v.iter().fold(0.0_f64, |m, &x| m.max(x.abs()))
}

/// Read the custom explicit-zeros dump: first line `n nnz ntriplets`,
/// then `nnz` lines of `row col value`, 0-based lower triangle.
fn read_triplet_dump(path: &Path) -> CscMatrix {
    let f = std::fs::File::open(path).expect("open triplet dump");
    let mut lines = std::io::BufReader::new(f).lines();
    let header = lines.next().expect("header line").expect("header read");
    let mut hdr = header.split_whitespace();
    let n: usize = hdr.next().expect("n").parse().expect("parse n");
    let nnz: usize = hdr.next().expect("nnz").parse().expect("parse nnz");
    let mut rows = Vec::with_capacity(nnz);
    let mut cols = Vec::with_capacity(nnz);
    let mut vals = Vec::with_capacity(nnz);
    for line in lines {
        let line = line.expect("data line");
        if line.trim().is_empty() {
            continue;
        }
        let mut t = line.split_whitespace();
        let r: usize = t.next().expect("row").parse().expect("parse row");
        let c: usize = t.next().expect("col").parse().expect("parse col");
        let v: f64 = t.next().expect("val").parse().expect("parse val");
        rows.push(r);
        cols.push(c);
        vals.push(v);
    }
    CscMatrix::from_triplets(n, &rows, &cols, &vals)
        .expect("triplet dump is a valid lower triangle")
}

fn read_rhs(path: &Path, n: usize) -> Vec<f64> {
    let f = std::fs::File::open(path).expect("open rhs");
    let v: Vec<f64> = std::io::BufReader::new(f)
        .lines()
        .filter_map(|l| {
            let l = l.expect("rhs line");
            let t = l.trim();
            if t.is_empty() {
                None
            } else {
                Some(t.parse().expect("parse rhs value"))
            }
        })
        .collect();
    assert_eq!(v.len(), n, "rhs length must match matrix order");
    v
}

/// Count stored entries whose value is exactly 0.0.
fn explicit_zeros(csc: &CscMatrix) -> usize {
    csc.values.iter().filter(|&&v| v == 0.0).count()
}

/// Factor `csc` `repeat` times on a *single warm* `Solver` — the IPM
/// pattern: analyze once, refactor on the same pattern many times. The
/// per-call time exposes whether the symbolic + MC64 caches engage.
fn run(label: &str, csc: &CscMatrix, rhs: &[f64], repeat: usize) {
    let in_nnz = csc.col_ptr[csc.n];
    let ez = explicit_zeros(csc);
    let mut s = Solver::new();
    println!("--- {label}: in_nnz={in_nnz} explicit_zeros={ez} ---");
    for call in 0..repeat {
        let t = Instant::now();
        let status = s.factor(csc, None);
        let ms = t.elapsed().as_secs_f64() * 1e3;
        let fnnz = s.factors().map(|f| f.factor_nnz()).unwrap_or(0);
        let inertia = s
            .inertia()
            .map(|i| format!("({},{},{})", i.positive, i.negative, i.zero))
            .unwrap_or_else(|| "-".to_string());
        let rel = match s.solve(rhs) {
            Ok(x) => {
                let mut ax = vec![0.0; csc.n];
                csc.symv(&x, &mut ax);
                let r: Vec<f64> = ax.iter().zip(rhs).map(|(&a, &b)| a - b).collect();
                norm_inf(&r) / norm_inf(rhs).max(1.0)
            }
            Err(_) => f64::NAN,
        };
        let tag = if call == 0 { "cold" } else { "warm" };
        println!(
            "  call {call} ({tag:<4}) factor_ms={ms:>8.1}  factor_nnz={fnnz:<9}  \
             symbolic_calls={}  mc64_cache_hits={}  inertia={inertia:<16} rel_res={rel:.2e}  {status:?}",
            s.symbolic_call_count(),
            s.mc64_cache_hit_count(),
        );
    }
    println!();
}

fn main() {
    let stripped_path = Path::new(DIR).join("cho_iter0_kkt.mtx");
    let kept_path = Path::new(DIR).join("cho_iter0_kkt_with_explicit_zeros.txt");
    let rhs_path = Path::new(DIR).join("cho_iter0_rhs.txt");
    for p in [&stripped_path, &kept_path, &rhs_path] {
        if !p.exists() {
            eprintln!("SKIP: {} not present", p.display());
            std::process::exit(2);
        }
    }

    let stripped = read_mtx(&stripped_path)
        .and_then(|m| m.to_csc())
        .expect("load stripped mtx");
    let kept = read_triplet_dump(&kept_path);
    assert_eq!(stripped.n, kept.n, "both matrices must share order n");
    let rhs = read_rhs(&rhs_path, stripped.n);

    println!(
        "CHO parmest iter-0 KKT  n={}  (issue #47: explicit-zero penalty)",
        stripped.n
    );
    println!("Each matrix factored 4x on one warm Solver (analyze-once / refactor-many).\n");
    run("stripped", &stripped, &rhs, 4);
    run("explicit zeros kept", &kept, &rhs, 4);
}
