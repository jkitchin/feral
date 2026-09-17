//! Issue #203 — recover the nested-dissection separator tree from a
//! permutation, and report its per-depth profile.
//!
//! Nested dissection numbers each block's separator last, so the tree
//! can be read back out of any permutation without the ordering code:
//! for a block occupying positions `[lo, hi)`, add its vertices in
//! permutation order under union-find; the last prefix at which two
//! components of at least an eighth of the block coexist ends the
//! block's two halves, and everything after it is the block's
//! separator. Recurse on each half.
//!
//! For each depth the probe prints the number of blocks, the total
//! separator weight, and `sum sep^2` — the quantity that dominates the
//! factorization flop count, since a separator of size `s` contributes
//! a dense `s x s` elimination at its own level.
//!
//! Usage:
//!   cargo run --release -p feral-diagnostics --bin issue203_sep_tree \
//!       -- MATRIX.mtx PERM.txt [PERM.txt ...]
//!
//! With no permutation file the probe orders the matrix with
//! `feral_metis` and profiles that.

use feral::read_mtx;
use feral::sparse::csc::CscMatrix;
use feral_ordering_core::CscPattern;

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
    fn union(&mut self, a: u32, b: u32) {
        let (mut ra, mut rb) = (self.find(a), self.find(b));
        if ra == rb {
            return;
        }
        if self.size[ra as usize] < self.size[rb as usize] {
            std::mem::swap(&mut ra, &mut rb);
        }
        self.parent[rb as usize] = ra;
        self.size[ra as usize] += self.size[rb as usize];
    }
}

#[derive(Default, Clone)]
struct Level {
    blocks: u64,
    sep_total: u64,
    sep_sq: f64,
    sep_max: u64,
}

/// Split the block at positions `[lo, hi)` into (halves, separator).
/// Returns the position ranges of the halves and the separator size.
/// `in_block[v]` must be true exactly for the block's vertices.
fn split_block(
    col_ptr: &[i32],
    row_idx: &[i32],
    perm: &[usize],
    lo: usize,
    hi: usize,
    in_block: &[bool],
) -> (Vec<(usize, usize)>, usize) {
    let len = hi - lo;
    let big = (len / 8).max(1);
    let n = perm.len();
    let mut dsu = Dsu::new(n);
    let mut added = vec![false; n];
    let mut n_big = 0usize;
    let mut last_split = 0usize;
    for (off, &old) in perm[lo..hi].iter().enumerate() {
        added[old] = true;
        if 1 >= big {
            n_big += 1;
        }
        for &u in &row_idx[col_ptr[old] as usize..col_ptr[old + 1] as usize] {
            let u = u as usize;
            if u == old || !added[u] || !in_block[u] {
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
            last_split = off + 1;
        }
    }
    if last_split == 0 {
        if std::env::var("SEP_TREE_DEBUG").is_ok() {
            eprintln!("  unparsed block lo={lo} hi={hi} len={len}: never split");
        }
        return (Vec::new(), 0);
    }
    // Component position ranges at the splitting prefix. Nested
    // dissection lays the halves out contiguously, so min/max position
    // per component is enough; a non-contiguous component means the
    // permutation is not nested-dissection-shaped and the block is
    // reported as a leaf.
    let mut dsu2 = Dsu::new(n);
    let mut added2 = vec![false; n];
    for &old in perm[lo..lo + last_split].iter() {
        added2[old] = true;
    }
    for &old in perm[lo..lo + last_split].iter() {
        for &u in &row_idx[col_ptr[old] as usize..col_ptr[old + 1] as usize] {
            let u = u as usize;
            if u != old && added2[u] && in_block[u] {
                dsu2.union(old as u32, u as u32);
            }
        }
    }
    let mut ranges: std::collections::HashMap<u32, (usize, usize, usize)> =
        std::collections::HashMap::new();
    for (off, &old) in perm[lo..lo + last_split].iter().enumerate() {
        let r = dsu2.find(old as u32);
        let e = ranges.entry(r).or_insert((off, off, 0));
        e.0 = e.0.min(off);
        e.1 = e.1.max(off);
        e.2 += 1;
    }
    let mut out: Vec<(usize, usize)> = Vec::new();
    for (_, (mn, mx, cnt)) in ranges {
        if mx - mn + 1 != cnt {
            if std::env::var("SEP_TREE_DEBUG").is_ok() {
                eprintln!(
                    "  non-contiguous block lo={lo} hi={hi} len={len} last_split={last_split} comp mn={mn} mx={mx} cnt={cnt}"
                );
            }
            return (Vec::new(), len - last_split);
        }
        out.push((lo + mn, lo + mx + 1));
    }
    out.sort_unstable();
    (out, len - last_split)
}

fn profile(col_ptr: &[i32], row_idx: &[i32], perm: &[usize], leaf: usize) -> Vec<Level> {
    let n = perm.len();
    let mut levels: Vec<Level> = Vec::new();
    let mut work: Vec<(usize, usize, usize)> = vec![(0, n, 0)];
    while let Some((lo, hi, depth)) = work.pop() {
        if hi - lo <= leaf {
            continue;
        }
        let mut in_block = vec![false; n];
        for &v in &perm[lo..hi] {
            in_block[v] = true;
        }
        let (halves, sep) = split_block(col_ptr, row_idx, perm, lo, hi, &in_block);
        if levels.len() <= depth {
            levels.resize(depth + 1, Level::default());
        }
        let l = &mut levels[depth];
        l.blocks += 1;
        l.sep_total += sep as u64;
        l.sep_sq += (sep as f64) * (sep as f64);
        l.sep_max = l.sep_max.max(sep as u64);
        for (a, b) in halves {
            work.push((a, b, depth + 1));
        }
    }
    levels
}

fn report(label: &str, levels: &[Level]) {
    println!("\n{label}");
    println!(
        "{:>6} {:>8} {:>12} {:>10} {:>14}",
        "depth", "blocks", "sep_total", "sep_max", "sum sep^2"
    );
    let mut tot = 0u64;
    let mut sq = 0.0f64;
    for (d, l) in levels.iter().enumerate() {
        if l.blocks == 0 {
            continue;
        }
        println!(
            "{:>6} {:>8} {:>12} {:>10} {:>14.4e}",
            d, l.blocks, l.sep_total, l.sep_max, l.sep_sq
        );
        tot += l.sep_total;
        sq += l.sep_sq;
    }
    println!(
        "{:>6} {:>8} {:>12} {:>10} {:>14.4e}",
        "all", "", tot, "", sq
    );
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(mtx_path) = args.first() else {
        eprintln!("usage: issue203_sep_tree MATRIX.mtx [PERM.txt ...]");
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
    let leaf = 200usize;
    println!("matrix n={} leaf={}", matrix.n, leaf);

    if args.len() == 1 {
        let pattern = match CscPattern::new(matrix.n, &col_ptr, &row_idx) {
            Some(p) => p,
            None => {
                eprintln!("error: malformed pattern");
                std::process::exit(1);
            }
        };
        match feral_metis::metis_order(&pattern) {
            Ok(p) => {
                let perm: Vec<usize> = p.iter().map(|&v| v as usize).collect();
                let lv = profile(&col_ptr, &row_idx, &perm, leaf);
                report("feral-metis (default)", &lv);
            }
            Err(e) => eprintln!("metis failed: {e:?}"),
        }
        return;
    }

    for path in &args[1..] {
        let text = match std::fs::read_to_string(path) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("{path}: {e}");
                continue;
            }
        };
        let perm: Vec<usize> = text
            .split_ascii_whitespace()
            .filter_map(|t| t.parse::<usize>().ok())
            .collect();
        if perm.len() != matrix.n {
            eprintln!("{path}: length {} != n {}", perm.len(), matrix.n);
            continue;
        }
        let lv = profile(&col_ptr, &row_idx, &perm, leaf);
        report(path, &lv);
    }
}
