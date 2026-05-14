//! Diagnostic: per-supernode cascade-break firing-ratio distribution
//! across families. Validates or refutes the issue #15 claim that
//! "marine_1600 / pinene_3200 supernodes fire at 0.85–0.95+ while
//! qcqp1000-1nc fires at ~0.5".
//!
//! For each iterate we factor with `cascade_break_ratio = None` so
//! cascade-break never fires and delays propagate naturally up the
//! supernode tree. Then we read `SparseFactors.node_factors` and
//! compute, for every non-root supernode, the ratio
//! `n_delayed_in / ncol` — exactly the trigger expression at
//! `src/numeric/factorize.rs:1823`. The maximum-over-supernodes ratio
//! per iterate characterizes how badly an unbroken cascade would
//! pile up.
//!
//! Roots are identified from `SymbolicFactorization.supernodes[*]
//! .children` (a node is root iff it appears in no other node's
//! children list). This matches the `is_root` array computed at
//! `factorize.rs:1251` and `factorize.rs:1450`.
//!
//! Per-family report:
//!   - max ratio, p95 ratio, p50 ratio across all non-root supernodes
//!     across all iterates
//!   - count of supernodes in bins {[0.0,0.5), [0.5,0.65),
//!     [0.65,0.75), [0.75,0.85), [0.85,0.95), [0.95,1.0]}
//!   - the largest supernode (by ncol) that would fire at threshold
//!     0.5, listed with (matrix, snode_idx, ncol, n_delayed_in, ratio)
//!
//! Usage:
//!     cargo run --release --bin diag_cascade_ratio_distribution

use std::path::{Path, PathBuf};
use std::time::Instant;

use feral::numeric::factorize::{
    factorize_multifrontal_parallel_with_workspace, FactorWorkspace, NumericParams,
};
use feral::read_mtx;
use feral::symbolic::{symbolic_factorize, SupernodeParams};

#[derive(Debug, Clone)]
struct Sample {
    matrix: String,
    snode_idx: usize,
    ncol: usize,
    n_delayed_in: usize,
    ratio: f64,
}

fn iterate_mtx_paths(family_dir: &Path, limit: usize) -> Vec<PathBuf> {
    let mut paths: Vec<PathBuf> = Vec::new();
    if let Ok(rd) = std::fs::read_dir(family_dir) {
        for e in rd.flatten() {
            let p = e.path();
            if p.extension().and_then(|x| x.to_str()) == Some("mtx") {
                paths.push(p);
            }
        }
    }
    paths.sort();
    paths.truncate(limit);
    paths
}

fn collect_family(family_name: &str, family_dir: &Path, limit: usize) -> Vec<Sample> {
    let paths = iterate_mtx_paths(family_dir, limit);
    let mut samples: Vec<Sample> = Vec::new();

    eprintln!(
        "{}: scanning {} iterates from {}",
        family_name,
        paths.len(),
        family_dir.display()
    );

    let params = NumericParams {
        cascade_break_ratio: None,
        cascade_break_eps: None,
        ..NumericParams::default()
    };
    let snode_params = SupernodeParams::default();

    for path in &paths {
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("?")
            .to_string();

        let mtx = match read_mtx(path) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("  {}: read_mtx failed: {:?}", stem, e);
                continue;
            }
        };
        let csc = match mtx.to_csc() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("  {}: to_csc failed: {:?}", stem, e);
                continue;
            }
        };
        drop(mtx);

        let sym = match symbolic_factorize(&csc, &snode_params) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("  {}: symbolic_factorize failed: {:?}", stem, e);
                continue;
            }
        };

        // is_root[i] iff i is not anyone's child.
        let n_snodes = sym.supernodes.len();
        let mut is_root = vec![true; n_snodes];
        for parent in &sym.supernodes {
            for &child_idx in &parent.children {
                if child_idx < n_snodes {
                    is_root[child_idx] = false;
                }
            }
        }

        let mut ws = FactorWorkspace::new();
        let t = Instant::now();
        let (factors, _inertia) =
            match factorize_multifrontal_parallel_with_workspace(&csc, &sym, &params, &mut ws) {
                Ok(f) => f,
                Err(e) => {
                    eprintln!("  {}: factor failed: {:?}", stem, e);
                    continue;
                }
            };
        let dt = t.elapsed();

        let mut local_max = 0.0_f64;
        for (i, nf) in factors.node_factors.iter().enumerate() {
            if is_root[i] {
                continue;
            }
            if nf.ncol == 0 {
                continue;
            }
            let r = (nf.n_delayed_in as f64) / (nf.ncol as f64);
            if r > local_max {
                local_max = r;
            }
            samples.push(Sample {
                matrix: stem.clone(),
                snode_idx: i,
                ncol: nf.ncol,
                n_delayed_in: nf.n_delayed_in,
                ratio: r,
            });
        }
        eprintln!(
            "  {}: factor {:>6} ms, n_snodes={:>6}, max_ratio={:.3}",
            stem,
            dt.as_millis(),
            n_snodes,
            local_max
        );
    }
    samples
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn report_family(family_name: &str, samples: &[Sample]) {
    println!("\n=== {} ===", family_name);
    println!(
        "non-root supernodes (across all iterates): {}",
        samples.len()
    );
    if samples.is_empty() {
        return;
    }
    let mut ratios: Vec<f64> = samples.iter().map(|s| s.ratio).collect();
    ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let p50 = percentile(&ratios, 50.0);
    let p90 = percentile(&ratios, 90.0);
    let p95 = percentile(&ratios, 95.0);
    let p99 = percentile(&ratios, 99.0);
    let max = *ratios.last().unwrap();

    println!(
        "ratio percentiles:  p50={:.3}  p90={:.3}  p95={:.3}  p99={:.3}  max={:.3}",
        p50, p90, p95, p99, max
    );

    let bins = [
        ("[0.00, 0.50)", 0.00, 0.50),
        ("[0.50, 0.65)", 0.50, 0.65),
        ("[0.65, 0.75)", 0.65, 0.75),
        ("[0.75, 0.85)", 0.75, 0.85),
        ("[0.85, 0.95)", 0.85, 0.95),
        ("[0.95, 1.01)", 0.95, 1.01),
    ];
    println!("bin counts:");
    for (label, lo, hi) in &bins {
        let c = samples
            .iter()
            .filter(|s| s.ratio >= *lo && s.ratio < *hi)
            .count();
        println!("  {} : {}", label, c);
    }

    // Largest supernodes that would fire at thresholds 0.5, 0.65, 0.85.
    for thr in [0.50, 0.65, 0.85] {
        let mut fired: Vec<&Sample> = samples.iter().filter(|s| s.ratio >= thr).collect();
        fired.sort_by_key(|s| std::cmp::Reverse(s.ncol));
        let topn = 3;
        println!("top {} fired @ threshold {:.2} by ncol:", topn, thr);
        for s in fired.iter().take(topn) {
            println!(
                "  {:<28}  snode={:>6}  ncol={:>5}  n_delayed_in={:>5}  ratio={:.3}",
                s.matrix, s.snode_idx, s.ncol, s.n_delayed_in, s.ratio
            );
        }
        if fired.is_empty() {
            println!("  (none fire at this threshold)");
        }
    }

    // Per-matrix max — sometimes the headline.
    let mut per_matrix: std::collections::BTreeMap<String, f64> = std::collections::BTreeMap::new();
    for s in samples {
        let e = per_matrix.entry(s.matrix.clone()).or_insert(0.0);
        if s.ratio > *e {
            *e = s.ratio;
        }
    }
    let mut per_matrix_vec: Vec<(String, f64)> = per_matrix.into_iter().collect();
    per_matrix_vec.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("per-iterate max ratio (top 10):");
    for (m, r) in per_matrix_vec.iter().take(10) {
        println!("  {:<28}  max_ratio={:.3}", m, r);
    }
}

fn main() {
    let families: [(&str, &str, usize); 3] = [
        (
            "qcqp1000-1nc",
            "data/matrices/kkt-mittelmann/qcqp1000-1nc",
            30,
        ),
        (
            "marine_1600",
            "data/matrices/kkt-mittelmann/marine_1600",
            20,
        ),
        (
            "pinene_3200",
            "data/matrices/kkt-mittelmann/pinene_3200",
            20,
        ),
    ];

    let mut all: Vec<(String, Vec<Sample>)> = Vec::new();
    for (name, dir, limit) in &families {
        let p = Path::new(dir);
        if !p.exists() {
            eprintln!("skip {}: dir {} missing", name, dir);
            continue;
        }
        let s = collect_family(name, p, *limit);
        all.push((name.to_string(), s));
    }

    for (name, samples) in &all {
        report_family(name, samples);
    }
}
