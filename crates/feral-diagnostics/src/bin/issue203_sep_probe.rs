//! Issue #203 — nested-dissection separator-quality probe.
//!
//! Runs `feral_metis::metis_order_full` on one matrix under a sweep of
//! `MetisOptions`, and for each resulting permutation reports:
//!
//!   * `top_sep` — the size of the **top-level separator**, recovered
//!     from the permutation alone: nested dissection numbers the top
//!     separator last, so `top_sep` is `n - p` for the largest prefix
//!     length `p` whose induced subgraph still has two components of at
//!     least `n/8` vertices each. Computed with incremental union-find,
//!     so it works on any permutation, including one imported from
//!     another solver.
//!   * `nnz_L` / `flops` — the permutation replayed through feral's own
//!     symbolic pipeline as `OrderingMethod::External`, so the numbers
//!     are directly comparable to `issue203_fill_probe`.
//!
//! Extra arguments are permutation files (one 0-based original index
//! per line, new-to-old) measured the same way, for importing a
//! reference ordering.
//!
//! Usage:
//!   cargo run --release -p feral-diagnostics --bin issue203_sep_probe \
//!       -- MATRIX.mtx [PERM.txt ...]

use feral::read_mtx;
use feral::sparse::csc::CscMatrix;
use feral::symbolic::{symbolic_factorize_with_method, OrderingMethod, SupernodeParams};
use feral_metis::MetisOptions;
use feral_ordering_core::CscPattern;

/// Expand a lower-triangle CSC matrix into the full-symmetric pattern
/// (diagonal included) that the ordering crates consume.
fn full_pattern(m: &CscMatrix) -> (Vec<i32>, Vec<i32>) {
    let n = m.n;
    let mut deg = vec![0usize; n];
    for j in 0..n {
        for k in m.col_ptr[j]..m.col_ptr[j + 1] {
            let i = m.row_idx[k];
            deg[j] += 1;
            if i != j {
                deg[i] += 1;
            }
        }
    }
    let mut col_ptr = vec![0i32; n + 1];
    for j in 0..n {
        col_ptr[j + 1] = col_ptr[j] + deg[j] as i32;
    }
    let mut fill = col_ptr.clone();
    let mut row_idx = vec![0i32; col_ptr[n] as usize];
    for j in 0..n {
        for k in m.col_ptr[j]..m.col_ptr[j + 1] {
            let i = m.row_idx[k];
            row_idx[fill[j] as usize] = i as i32;
            fill[j] += 1;
            if i != j {
                row_idx[fill[i] as usize] = j as i32;
                fill[i] += 1;
            }
        }
    }
    for j in 0..n {
        let lo = col_ptr[j] as usize;
        let hi = col_ptr[j + 1] as usize;
        row_idx[lo..hi].sort_unstable();
    }
    (col_ptr, row_idx)
}

struct Dsu {
    parent: Vec<u32>,
    size: Vec<u32>,
}

impl Dsu {
    fn new(n: usize) -> Self {
        Dsu {
            parent: (0..n as u32).collect(),
            size: vec![1; n],
        }
    }
    fn find(&mut self, mut v: u32) -> u32 {
        while self.parent[v as usize] != v {
            let g = self.parent[self.parent[v as usize] as usize];
            self.parent[v as usize] = g;
            v = g;
        }
        v
    }
    fn union(&mut self, a: u32, b: u32) -> bool {
        let (mut ra, mut rb) = (self.find(a), self.find(b));
        if ra == rb {
            return false;
        }
        if self.size[ra as usize] < self.size[rb as usize] {
            std::mem::swap(&mut ra, &mut rb);
        }
        self.parent[rb as usize] = ra;
        self.size[ra as usize] += self.size[rb as usize];
        true
    }
}

/// Size of the top-level separator implied by `perm` (new-to-old).
///
/// Adds vertices in permutation order under union-find and remembers
/// the last prefix at which two components of at least `n/8` vertices
/// coexist. The separator is everything after that prefix. Returns `n`
/// when the prefix never splits (no recognisable top separator, e.g. a
/// band ordering).
fn top_separator(col_ptr: &[i32], row_idx: &[i32], perm: &[usize]) -> usize {
    let n = perm.len();
    let big = (n / 8).max(1);
    let mut dsu = Dsu::new(n);
    let mut added = vec![false; n];
    let mut n_big = 0usize;
    let mut last_split = 0usize;
    for (p, &old) in perm.iter().enumerate() {
        added[old] = true;
        if 1 >= big {
            n_big += 1;
        }
        let lo = col_ptr[old] as usize;
        let hi = col_ptr[old + 1] as usize;
        for &u in &row_idx[lo..hi] {
            let u = u as usize;
            if u == old || !added[u] {
                continue;
            }
            let (ra, rb) = (dsu.find(old as u32), dsu.find(u as u32));
            if ra == rb {
                continue;
            }
            let (sa, sb) = (
                dsu.size[ra as usize] as usize,
                dsu.size[rb as usize] as usize,
            );
            let was = (sa >= big) as usize + (sb >= big) as usize;
            dsu.union(old as u32, u as u32);
            n_big = n_big + ((sa + sb >= big) as usize) - was;
        }
        if n_big >= 2 {
            last_split = p + 1;
        }
    }
    if last_split == 0 {
        n
    } else {
        n - last_split
    }
}

fn measure(matrix: &CscMatrix, perm: &[usize]) -> Option<(u64, f64, u64)> {
    let params = SupernodeParams::default();
    let sym =
        symbolic_factorize_with_method(matrix, &params, OrderingMethod::External(perm.to_vec()))
            .ok()?;
    let mut flops = 0.0f64;
    let mut wide = 0u64;
    for &c in &sym.col_counts {
        let below = c.saturating_sub(1) as f64;
        flops += below * below;
        wide = wide.max(c as u64);
    }
    Some((sym.factor_nnz_estimate as u64, flops, wide))
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(mtx_path) = args.first() else {
        eprintln!("usage: issue203_sep_probe MATRIX.mtx [PERM.txt ...]");
        std::process::exit(2);
    };
    let mtx = match read_mtx(std::path::Path::new(mtx_path)) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("error: cannot read {mtx_path}: {e}");
            std::process::exit(1);
        }
    };
    let matrix = match mtx.to_csc() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: csc conversion failed: {e}");
            std::process::exit(1);
        }
    };
    let (col_ptr, row_idx) = full_pattern(&matrix);
    let pattern = match CscPattern::new(matrix.n, &col_ptr, &row_idx) {
        Some(p) => p,
        None => {
            eprintln!("error: malformed pattern");
            std::process::exit(1);
        }
    };

    println!("matrix n={} nnz_lower={}", matrix.n, matrix.row_idx.len());
    println!(
        "{:<34} {:>9} {:>13} {:>13} {:>10} {:>9}",
        "arm", "top_sep", "nnz_L", "flops", "front_max", "ms"
    );

    let base = MetisOptions::default();
    let mut node = base.clone();
    node.node_refine = true;
    let mut arms: Vec<(String, MetisOptions)> = vec![
        ("default".to_string(), base.clone()),
        ("node_refine".to_string(), node.clone()),
    ];
    for passes in [20u32, 40] {
        let mut o = node.clone();
        o.fm_passes = passes;
        arms.push((format!("node_refine+fm_passes={passes}"), o));
    }
    if std::env::var("SEP_PROBE_SWEEP").is_err() {
        // Default run compares the two refinement structures only; the
        // full knob sweep is opt-in because it costs ~10 orderings.
        return run(&matrix, &col_ptr, &row_idx, &pattern, arms, &args[1..]);
    }
    for niparts in [16u32, 64] {
        let mut o = base.clone();
        o.niparts = niparts;
        arms.push((format!("niparts={niparts}"), o));
    }
    for passes in [40u32, 200] {
        let mut o = base.clone();
        o.fm_passes = passes;
        arms.push((format!("fm_passes={passes}"), o));
    }
    for sw in [50u32, 800] {
        let mut o = base.clone();
        o.nd_to_amd_switch = sw;
        arms.push((format!("nd_to_amd_switch={sw}"), o));
    }
    for floor in [30u32, 500] {
        let mut o = base.clone();
        o.coarsen_floor = floor;
        arms.push((format!("coarsen_floor={floor}"), o));
    }
    for imb in [0.03f64, 0.4] {
        let mut o = base.clone();
        o.max_imbalance = imb;
        arms.push((format!("max_imbalance={imb}"), o));
    }

    run(&matrix, &col_ptr, &row_idx, &pattern, arms, &args[1..]);
}

fn run(
    matrix: &CscMatrix,
    col_ptr: &[i32],
    row_idx: &[i32],
    pattern: &CscPattern<'_>,
    arms: Vec<(String, MetisOptions)>,
    perm_files: &[String],
) {
    for (label, opts) in arms {
        let t = std::time::Instant::now();
        let perm_i32 = match feral_metis::metis_order_full(pattern, &opts) {
            Ok((p, _, _)) => p,
            Err(e) => {
                println!("{label:<34} FAILED: {e:?}");
                continue;
            }
        };
        let ms = t.elapsed().as_secs_f64() * 1e3;
        let perm: Vec<usize> = perm_i32.iter().map(|&v| v as usize).collect();
        if let Ok(dir) = std::env::var("SEP_PROBE_DUMP") {
            let safe: String = label
                .chars()
                .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
                .collect();
            let path = std::path::Path::new(&dir).join(format!("perm_{safe}.txt"));
            let body: String = perm.iter().map(|v| format!("{v}\n")).collect();
            if let Err(e) = std::fs::write(&path, body) {
                eprintln!("dump {}: {e}", path.display());
            }
        }
        let ts = top_separator(col_ptr, row_idx, &perm);
        match measure(matrix, &perm) {
            Some((nnz, flops, wide)) => {
                println!("{label:<34} {ts:>9} {nnz:>13} {flops:>13.4e} {wide:>10} {ms:>9.0}")
            }
            None => println!("{label:<34} {ts:>9}  symbolic FAILED"),
        }
    }

    for path in perm_files {
        let text = match std::fs::read_to_string(path) {
            Ok(t) => t,
            Err(e) => {
                println!("{path:<34} read failed: {e}");
                continue;
            }
        };
        let perm: Vec<usize> = text
            .split_ascii_whitespace()
            .filter_map(|t| t.parse::<usize>().ok())
            .collect();
        let stem = std::path::Path::new(path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(path);
        let ts = top_separator(col_ptr, row_idx, &perm);
        match measure(matrix, &perm) {
            Some((nnz, flops, wide)) => println!(
                "{:<34} {ts:>9} {nnz:>13} {flops:>13.4e} {wide:>10} {:>9}",
                format!("ext:{stem}"),
                "-"
            ),
            None => println!("ext:{stem:<30} {ts:>9}  symbolic FAILED"),
        }
    }
}
