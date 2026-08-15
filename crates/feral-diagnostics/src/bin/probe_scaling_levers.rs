//! Diagnostic for the two open scaling levers: #153's warm-path scaling
//! and #134 item B's lower-triangle-blind router gates.
//!
//! Read-only. Answers three questions with numbers rather than
//! reasoning-from-the-source:
//!
//!   1. What does `pick_scaling_strategy` actually route the six #153
//!      downstream KKTs to? (Decides whether the existing MC64 warm
//!      cache is even applicable to them.)
//!   2. How many Knight-Ruiz sweeps does `compute_infnorm` burn from
//!      `d = 1`, and how many would it burn warm-started from the
//!      previous iterate's `d`? (Bounds the #153 scaling win.)
//!   3. Over the KKT corpus, how many matrices would the router send
//!      somewhere else if its gates counted *symmetric* degrees
//!      instead of stored lower-triangle degrees — and does the route
//!      flip when an arrow KKT is symmetrically reversed? (Sizes #134
//!      item B.)

use feral::scaling::{pick_scaling_strategy, ScalingStrategy};
use feral::{read_mtx, CscMatrix, Solver};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// The router's gate inputs, computed two ways.
struct Gates {
    diag_only: usize,
    max_deg: usize,
}

impl Gates {
    /// `>32` dense column AND `>=30%` degree-1 diagonal columns.
    fn route(&self, n: usize) -> ScalingStrategy {
        if n == 0 {
            return ScalingStrategy::InfNorm;
        }
        if self.max_deg > 32 && self.diag_only as f64 / n as f64 >= 0.3 {
            ScalingStrategy::Mc64Symmetric
        } else {
            ScalingStrategy::InfNorm
        }
    }
}

/// What `pick_scaling_strategy` counts today: stored lower-triangle
/// entries per column, value-aware (issue #47).
fn stored_gates(m: &CscMatrix) -> Gates {
    let mut diag_only = 0;
    let mut max_deg = 0;
    for j in 0..m.n {
        let mut nnz = 0;
        let mut diag = false;
        for k in m.col_ptr[j]..m.col_ptr[j + 1] {
            if m.values[k] == 0.0 {
                continue;
            }
            nnz += 1;
            if m.row_idx[k] == j {
                diag = true;
            }
        }
        max_deg = max_deg.max(nnz);
        if nnz == 1 && diag {
            diag_only += 1;
        }
    }
    Gates { diag_only, max_deg }
}

/// What #134 item B proposes: the full symmetric degree of each
/// column — stored entries plus the entries that symmetry puts in the
/// unstored upper triangle. Same O(n + nnz), one extra n-vector.
fn symmetric_gates(m: &CscMatrix) -> Gates {
    let mut deg = vec![0usize; m.n];
    let mut has_diag = vec![false; m.n];
    for j in 0..m.n {
        for k in m.col_ptr[j]..m.col_ptr[j + 1] {
            if m.values[k] == 0.0 {
                continue;
            }
            let i = m.row_idx[k];
            deg[j] += 1;
            if i == j {
                has_diag[j] = true;
            } else {
                deg[i] += 1;
            }
        }
    }
    let mut diag_only = 0;
    let mut max_deg = 0;
    for j in 0..m.n {
        max_deg = max_deg.max(deg[j]);
        if deg[j] == 1 && has_diag[j] {
            diag_only += 1;
        }
    }
    Gates { diag_only, max_deg }
}

/// Symmetrically reverse the index order: `P(i) = n-1-i`. The matrix
/// is the same up to a symmetric permutation, so every
/// permutation-invariant property (including whether MC64 helps) is
/// unchanged. A router that changes its mind here is keying on
/// storage, not on the matrix.
fn reverse_permute(m: &CscMatrix) -> Result<CscMatrix, feral::FeralError> {
    let n = m.n;
    let mut rows = Vec::with_capacity(m.row_idx.len());
    let mut cols = Vec::with_capacity(m.row_idx.len());
    let mut vals = Vec::with_capacity(m.row_idx.len());
    for j in 0..n {
        for k in m.col_ptr[j]..m.col_ptr[j + 1] {
            let i = m.row_idx[k];
            // (i, j) with i >= j maps to (n-1-i, n-1-j) with
            // n-1-i <= n-1-j, so swap back into the lower triangle.
            rows.push(n - 1 - j);
            cols.push(n - 1 - i);
            vals.push(m.values[k]);
        }
    }
    CscMatrix::from_triplets(n, &rows, &cols, &vals)
}

/// Knight-Ruiz ∞-norm equilibration, instrumented for sweep count and
/// optionally warm-started. Mirrors `src/scaling/infnorm.rs`
/// (`max_iter = 10`, `tol = 1e-8`) — this is a measurement copy, not a
/// second implementation to be kept in sync.
/// Result of one instrumented Knight-Ruiz run.
struct KrRun {
    d: Vec<f64>,
    sweeps: usize,
    converged: bool,
    final_dev: f64,
    guard_hits: usize,
    subnormals: usize,
}

/// Knight-Ruiz oo-norm equilibration, instrumented for sweep count and
/// optionally warm-started. Mirrors `src/scaling/infnorm.rs`
/// (`max_iter = 10`, `tol = 1e-8`, the same guarded update) — this is a
/// measurement copy, not a second implementation to be kept in sync.
fn kr_sweeps(m: &CscMatrix, init: Option<&[f64]>) -> KrRun {
    kr_sweeps_capped(m, init, 10)
}

/// Same, with an explicit sweep budget.
fn kr_sweeps_capped(m: &CscMatrix, init: Option<&[f64]>, max_iter: usize) -> KrRun {
    let n = m.n;
    let mut d = match init {
        Some(prev) if prev.len() == n => prev.to_vec(),
        _ => vec![1.0f64; n],
    };
    let tol = 1e-8;
    let mut row_max = vec![0.0f64; n];
    let mut sweeps = 0;
    let mut converged = false;
    let mut final_dev = f64::INFINITY;
    let mut guard_hits = 0;
    for _ in 0..max_iter {
        sweeps += 1;
        for r in row_max.iter_mut() {
            *r = 0.0;
        }
        for j in 0..n {
            let dj = d[j];
            let mut col_max = row_max[j];
            for k in m.col_ptr[j]..m.col_ptr[j + 1] {
                let i = m.row_idx[k];
                let v = (d[i] * m.values[k] * dj).abs();
                if i != j && v > row_max[i] {
                    row_max[i] = v;
                }
                if v > col_max {
                    col_max = v;
                }
            }
            row_max[j] = col_max;
        }
        let mut max_dev = 0.0f64;
        for i in 0..n {
            let mv = row_max[i];
            if mv > 0.0 {
                // Exactly `crate::scaling::kr_guarded_update`.
                let cand = d[i] / mv.sqrt();
                if cand.is_finite() && cand > 0.0 {
                    d[i] = cand;
                } else {
                    guard_hits += 1;
                }
                let dev = (mv - 1.0).abs();
                if dev > max_dev {
                    max_dev = dev;
                }
            }
        }
        final_dev = max_dev;
        if max_dev < tol {
            converged = true;
            break;
        }
    }
    let subnormals = d.iter().filter(|x| x.is_subnormal()).count();
    KrRun {
        d,
        sweeps,
        converged,
        final_dev,
        guard_hits,
        subnormals,
    }
}

/// Collect the iterate dumps for one problem, in order.
fn iterates(dir: &Path, limit: usize) -> Vec<(String, CscMatrix)> {
    let mut paths: Vec<PathBuf> = match std::fs::read_dir(dir) {
        Ok(rd) => rd
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "mtx"))
            .collect(),
        Err(_) => return Vec::new(),
    };
    paths.sort();
    paths
        .into_iter()
        .take(limit)
        .filter_map(|p| {
            let name = p.file_stem()?.to_string_lossy().to_string();
            let csc = read_mtx(&p).ok()?.to_csc().ok()?;
            Some((name, csc))
        })
        .collect()
}

/// Part 1 + 2: the #153 six.
fn issue_153() {
    println!("=== #153: what the six downstream KKTs route to, and what KR actually costs ===\n");
    let base = Path::new("data/matrices/kkt-mittelmann");
    for prob in [
        "clnlbeam",
        "dtoc1nd",
        "steering_12800",
        "rocket_12800",
        "marine_1600",
        "dtoc2",
    ] {
        let its = iterates(&base.join(prob), 3);
        if its.is_empty() {
            println!("{:<16} (no .mtx dumps found)", prob);
            continue;
        }
        let (_, m0) = &its[0];
        println!(
            "{}  n={} nnz={}  Auto -> {:?}",
            prob,
            m0.n,
            m0.nnz(),
            pick_scaling_strategy(m0)
        );
        let t = Instant::now();
        let r0 = kr_sweeps(m0, None);
        let ms0 = t.elapsed().as_secs_f64() * 1e3;
        println!(
            "    iter0 cold : {:>2} sweeps {:>7.2} ms  converged={} final_dev={:.3e} guard={} subnormal_d={}",
            r0.sweeps, ms0, r0.converged, r0.final_dev, r0.guard_hits, r0.subnormals
        );
        for (idx, (_, m)) in its.iter().enumerate().skip(1) {
            let t = Instant::now();
            let rc = kr_sweeps(m, None);
            let msc = t.elapsed().as_secs_f64() * 1e3;
            let t = Instant::now();
            let rw = kr_sweeps(m, Some(&r0.d));
            let msw = t.elapsed().as_secs_f64() * 1e3;
            println!(
                "    iter{} cold : {:>2} sweeps {:>7.2} ms  converged={} final_dev={:.3e} guard={} subnormal_d={}",
                idx, rc.sweeps, msc, rc.converged, rc.final_dev, rc.guard_hits, rc.subnormals
            );
            println!(
                "    iter{} warm : {:>2} sweeps {:>7.2} ms  converged={} final_dev={:.3e} guard={} subnormal_d={}   (started from iter0 d)",
                idx, rw.sweeps, msw, rw.converged, rw.final_dev, rw.guard_hits, rw.subnormals
            );
        }
        // Does the shipped router's gate see an arrow head here once
        // degrees are counted symmetrically? (#134B on the inputs
        // pounce actually sends.)
        let gs = stored_gates(m0);
        let gy = symmetric_gates(m0);
        println!(
            "    gates      : stored d1={} md={} -> {:?}   symmetric d1={} md={}",
            gs.diag_only,
            gs.max_deg,
            gs.route(m0.n),
            gy.diag_only,
            gy.max_deg,
        );
        println!();
    }
}

/// Part 3: the router gates, stored vs symmetric, over the corpus.
fn issue_134b() {
    println!("\n\n=== #134B: router gates, stored lower-triangle vs symmetric degree ===\n");

    // (a) The reversal experiment on the calibration panel.
    println!("(a) symmetric reversal P(i) = n-1-i on the calibration panel:\n");
    println!(
        "{:<18} {:>7} {:>26} {:>26}",
        "matrix", "n", "as stored (route)", "reversed (route)"
    );
    let kkt = Path::new("data/matrices/kkt");
    for fam in [
        "VESUVIO", "VESUVIOU", "MUONSINE", "CRESC132", "ACOPP30", "AVION2",
    ] {
        let its = iterates(&kkt.join(fam), 1);
        let Some((_, m)) = its.first() else {
            println!("{:<18} (no dump)", fam);
            continue;
        };
        let Ok(rev) = reverse_permute(m) else {
            println!("{:<18} (permute failed)", fam);
            continue;
        };
        let g = stored_gates(m);
        let gr = stored_gates(&rev);
        println!(
            "{:<18} {:>7} {:>26} {:>26}",
            fam,
            m.n,
            format!(
                "d1={} md={} {:?}",
                g.diag_only,
                g.max_deg,
                pick_scaling_strategy(m)
            ),
            format!(
                "d1={} md={} {:?}",
                gr.diag_only,
                gr.max_deg,
                pick_scaling_strategy(&rev)
            ),
        );
    }

    // (b) Corpus sweep: where do the two gate definitions disagree?
    println!("\n(b) corpus sweep over data/matrices/kkt (one iterate per family):\n");
    let mut total = 0;
    let mut agree = 0;
    let mut flips: Vec<(String, usize, usize, usize, usize, usize)> = Vec::new();
    let mut dirs: Vec<PathBuf> = match std::fs::read_dir(kkt) {
        Ok(rd) => rd
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.is_dir())
            .collect(),
        Err(e) => {
            println!("  cannot read corpus: {}", e);
            return;
        }
    };
    dirs.sort();
    for dir in dirs {
        let its = iterates(&dir, 1);
        let Some((_, m)) = its.first() else {
            continue;
        };
        total += 1;
        let gs = stored_gates(m);
        let gy = symmetric_gates(m);
        let rs = gs.route(m.n);
        let ry = gy.route(m.n);
        if rs == ry {
            agree += 1;
        } else {
            let name = dir
                .file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default();
            flips.push((
                name,
                m.n,
                gs.diag_only,
                gs.max_deg,
                gy.diag_only,
                gy.max_deg,
            ));
        }
    }
    println!(
        "  {} families read, {} route the same either way, {} flip",
        total,
        agree,
        flips.len()
    );
    if !flips.is_empty() {
        println!(
            "\n  {:<20} {:>8} {:>10} {:>8} {:>10} {:>8}",
            "family", "n", "stored d1", "stored md", "sym d1", "sym md"
        );
        for (name, n, sd, sm, yd, ym) in flips.iter().take(60) {
            println!(
                "  {:<20} {:>8} {:>10} {:>8} {:>10} {:>8}",
                name, n, sd, sm, yd, ym
            );
        }
        if flips.len() > 60 {
            println!("  ... and {} more", flips.len() - 60);
        }
    }
}

/// Part 4: does the MC64 warm scaling cache actually hit across a
/// real IPM iterate sequence? `marine_1600` routes to
/// `Mc64Symmetric` and has 18 dumped iterates, so it is the fixture
/// that can answer "warm-path scaling never warms" for the MC64 half.
/// `dtoc1nd` routes to InfNorm and is the control: the cache is
/// gated off by design there, so it must report zero hits.
fn cache_hits() {
    println!(
        "\n\n=== #153: does the MC64 warm scaling cache hit across an iterate sequence? ===\n"
    );
    let base = Path::new("data/matrices/kkt-mittelmann");
    for prob in ["marine_1600", "dtoc1nd", "rocket_12800"] {
        let its = iterates(&base.join(prob), 20);
        if its.is_empty() {
            continue;
        }
        let route = pick_scaling_strategy(&its[0].1);
        println!("{}  ({} iterates, Auto -> {:?})", prob, its.len(), route);
        let mut solver = Solver::new().with_parallel(false);
        let mut prev_hits = 0;
        for (i, (_, m)) in its.iter().enumerate() {
            let t = Instant::now();
            let st = solver.factor(m, None);
            let ms = t.elapsed().as_secs_f64() * 1e3;
            let hits = solver.mc64_cache_hit_count();
            println!(
                "    iter {:>2}: {:>8.1} ms  cache_hit={}  status={:?}",
                i,
                ms,
                if hits > prev_hits { "YES" } else { "no " },
                st
            );
            prev_hits = hits;
        }
        println!(
            "    total cache hits: {} / {}",
            solver.mc64_cache_hit_count(),
            its.len()
        );
        println!();
    }
}

/// Part 5: KR is cap-bound, not tolerance-bound, on working iterates.
/// So the question is not "does warm start converge sooner" but "how
/// good a scaling does each budget buy". Sweep the cap, cold vs
/// warm-started, and report the equilibration quality reached.
fn kr_budget() {
    println!("\n=== #153: KR quality per sweep budget (cold vs warm-started) ===\n");
    let base = Path::new("data/matrices/kkt-mittelmann");
    for prob in ["clnlbeam", "dtoc1nd", "marine_1600", "rocket_12800"] {
        let its = iterates(&base.join(prob), 2);
        if its.len() < 2 {
            continue;
        }
        let d0 = kr_sweeps_capped(&its[0].1, None, 10).d;
        println!(
            "{}  (quality = max_i |rowmax_i - 1| after the budget)",
            prob
        );
        println!(
            "    {:>6}  {:>12}  {:>12}",
            "budget", "cold dev", "warm dev"
        );
        for cap in [1usize, 2, 3, 5, 10] {
            let c = kr_sweeps_capped(&its[1].1, None, cap);
            let w = kr_sweeps_capped(&its[1].1, Some(&d0), cap);
            println!(
                "    {:>6}  {:>12.3e}  {:>12.3e}",
                cap, c.final_dev, w.final_dev
            );
        }
        println!();
    }
}

fn main() {
    issue_153();
    cache_hits();
    kr_budget();
    issue_134b();
}
