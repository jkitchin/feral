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

/// The permutation-invariant candidate for #134B.
///
/// Both gates are computed on the *symmetric* degree — the degree of
/// column `j` in the full matrix, not in the stored lower triangle —
/// so a symmetric relabeling cannot change either one.
///
/// The head gate is the max symmetric degree. The slack-mass gate
/// counts columns whose symmetric degree is at most `slack_max` and
/// whose diagonal is present and nonzero: in an arrow KKT a slack
/// column carries its diagonal plus its single coupling to the arrow
/// head, so its symmetric degree is 2 regardless of whether the head
/// leads or trails. That is the invariant the shipped `nnz_col == 1`
/// test is a storage-dependent special case of.
fn invariant_gates(m: &CscMatrix, slack_max: usize) -> Gates {
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
        if deg[j] <= slack_max && has_diag[j] {
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
    calibrate();
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
    for prob in [
        "clnlbeam",
        "dtoc1nd",
        "steering_12800",
        "rocket_12800",
        "marine_1600",
        "dtoc2",
    ] {
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

/// Part 5: the one KR question `dev/research/scaling-warm-start-2026-08-09.md`
/// did not measure.
///
/// That note falsified "warm-start reduces the sweep count" (it does
/// not — KR is cap-bound) and separately rejected "lower the cap"
/// (cold KR at cap 5 is ~26x worse equilibrated than at cap 10). What
/// it did not try is the *combination*: carry `d` forward across the
/// IPM sequence AND spend fewer sweeps per factorization, so the
/// iteration becomes one long continued run rather than N truncated
/// restarts.
///
/// The experiment is the shipping comparison, not a single pair:
///   - cold chain  = today. Every iterate restarts from `d = 1`, 10 sweeps.
///   - warm chain  = candidate. Iterate 0 pays 10 sweeps; every later
///     iterate starts from the previous iterate's `d` and spends `k`.
///
/// Both are scored by the equilibration they actually deliver
/// (`max_i |rowmax_i - 1|` of the matrix the factorization then sees).
/// Only pairs with an unchanged pattern are compared, since a pattern
/// change is where a real solver would drop the carried `d` anyway.
fn kr_chain(label: &str, dir: &Path, limit: usize) {
    let its = iterates(dir, limit);
    if its.len() < 3 {
        println!("{}: fewer than 3 usable iterates, skipping", label);
        return;
    }
    // Restrict to the longest run of iterates sharing iterate 1's
    // pattern (iterate 0 is routinely a different pattern).
    let base_nnz = its[1].1.nnz();
    let seq: Vec<&CscMatrix> = its
        .iter()
        .map(|(_, m)| m)
        .filter(|m| m.nnz() == base_nnz)
        .collect();
    if seq.len() < 3 {
        println!("{}: no stable-pattern run, skipping", label);
        return;
    }
    println!(
        "{}  n={} nnz={}  {} stable-pattern iterates",
        label,
        seq[0].n,
        base_nnz,
        seq.len()
    );
    println!(
        "    {:>4}  {:>12}  {:>12}  {:>12}  {:>7}",
        "k", "cold@10 geo", "warm@k geo", "warm@10 geo", "warm@k wins"
    );
    // Cold chain: independent of k.
    let cold: Vec<f64> = seq
        .iter()
        .map(|m| kr_sweeps_capped(m, None, 10).final_dev)
        .collect();
    let geo = |v: &[f64]| -> f64 {
        let s: f64 = v.iter().map(|x| x.max(1e-300).ln()).sum();
        (s / v.len() as f64).exp()
    };
    for k in [2usize, 3, 5, 10] {
        // Warm chain: iterate 0 pays the full 10, the rest pay k.
        let mut d = kr_sweeps_capped(seq[0], None, 10).d;
        let mut warm = Vec::with_capacity(seq.len());
        warm.push(cold[0]);
        for m in seq.iter().skip(1) {
            let r = kr_sweeps_capped(m, Some(&d), k);
            warm.push(r.final_dev);
            d = r.d;
        }
        let wins = warm
            .iter()
            .zip(cold.iter())
            .skip(1)
            .filter(|(w, c)| w < c)
            .count();
        let warm10 = if k == 10 { geo(&warm) } else { f64::NAN };
        println!(
            "    {:>4}  {:>12.3e}  {:>12.3e}  {:>12}  {:>3}/{:<3}",
            k,
            geo(&cold),
            geo(&warm),
            if warm10.is_nan() {
                "-".to_string()
            } else {
                format!("{:.3e}", warm10)
            },
            wins,
            seq.len() - 1
        );
    }
    println!();
}

fn kr_budget() {
    println!("\n=== #153: warm chain at budget k vs the shipped cold chain at 10 ===");
    println!("    (geo = geometric mean of max_i |rowmax_i - 1| over the run;");
    println!("     lower is better; 'wins' counts iterates where warm@k beat cold@10)\n");
    let mm = Path::new("data/matrices/kkt-mittelmann");
    let kkt = Path::new("data/matrices/kkt");
    let exp = Path::new("data/matrices/kkt-expansion");
    // The two matrices that actually run InfNorm at steady state.
    kr_chain("clnlbeam       [InfNorm route]", &mm.join("clnlbeam"), 20);
    kr_chain(
        "steering_12800 [InfNorm route]",
        &mm.join("steering_12800"),
        20,
    );
    // The 2026-08-09 note's own hard fixtures, now with real IPM
    // drift instead of the note's synthetic +/-5% perturbation.
    kr_chain("HYDCAR20       [note fixture]", &kkt.join("HYDCAR20"), 40);
    kr_chain("TWIRISM1       [note fixture]", &exp.join("TWIRISM1"), 40);
    // MC64-routed, for contrast: KR is not on their path.
    kr_chain("marine_1600    [MC64 route]", &mm.join("marine_1600"), 20);
}

/// Every corpus root the scaling router is validated against.
/// `kkt-mittelmann` is where the "MC64 hurts clnlbeam" evidence lives
/// (journal 2026-05-16), so a sweep that omits it does not cover the
/// panel the shipped thresholds were calibrated on.
const CORPUS_ROOTS: &[&str] = &[
    "data/matrices/kkt",
    "data/matrices/kkt-mittelmann",
    "data/matrices/kkt-expansion",
];

/// #134B calibration: does the invariant gate reproduce the shipped
/// route on the 2026-05-17 panel, and how many corpus families move?
fn calibrate() {
    println!("\n(c) invariant gate (symmetric degree, slack_max=2) vs the shipped router:\n");
    let kkt = Path::new("data/matrices/kkt");
    println!("  calibration panel (expected: VESUVIO-class YES, clnlbeam/ACOPP30 NO):");
    println!(
        "  {:<12} {:>7} {:>26} {:>30} {:>16}",
        "matrix", "n", "shipped (stored)", "invariant (symmetric)", "reversed inv."
    );
    for fam in [
        "VESUVIO", "VESUVIOU", "VESUVIA", "MUONSINE", "CRESC132", "ACOPP30", "ACOPR30", "AIRPORT",
        "HAIFAM",
    ] {
        let its = iterates(&kkt.join(fam), 1);
        let Some((_, m)) = its.first() else { continue };
        let gs = stored_gates(m);
        let gi = invariant_gates(m, 2);
        let rev = reverse_permute(m).ok();
        let gir = rev.as_ref().map(|r| invariant_gates(r, 2));
        println!(
            "  {:<12} {:>7} {:>26} {:>30} {:>16}",
            fam,
            m.n,
            format!("d1={} md={} {:?}", gs.diag_only, gs.max_deg, gs.route(m.n)),
            format!("s2={} md={} {:?}", gi.diag_only, gi.max_deg, gi.route(m.n)),
            match &gir {
                Some(g) => format!("{:?}", g.route(m.n)),
                None => "-".into(),
            },
        );
    }

    // Corpus: how many families change route, and is the invariant
    // gate actually invariant?
    let mut dirs: Vec<PathBuf> = Vec::new();
    for root in CORPUS_ROOTS {
        if let Ok(rd) = std::fs::read_dir(Path::new(root)) {
            dirs.extend(
                rd.filter_map(|e| e.ok())
                    .map(|e| e.path())
                    .filter(|p| p.is_dir()),
            );
        }
    }
    dirs.sort();
    for slack_max in [1usize, 2, 3] {
        let (mut total, mut changed, mut inv_ok, mut shipped_not_inv) = (0, 0, 0, 0);
        let mut gained = Vec::new();
        let mut lost = Vec::new();
        for dir in &dirs {
            let its = iterates(dir, 1);
            let Some((_, m)) = its.first() else { continue };
            total += 1;
            let rs = stored_gates(m).route(m.n);
            let gi = invariant_gates(m, slack_max);
            let ri = gi.route(m.n);
            let name = dir
                .file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default();
            if rs != ri {
                changed += 1;
                if ri == ScalingStrategy::Mc64Symmetric {
                    gained.push(name.clone())
                } else {
                    lost.push(name.clone())
                }
            }
            if let Ok(rev) = reverse_permute(m) {
                // Is the candidate actually permutation-invariant?
                if invariant_gates(&rev, slack_max).route(m.n) == ri {
                    inv_ok += 1;
                }
                // Is the shipped router invariant? (control)
                if stored_gates(&rev).route(m.n) != rs {
                    shipped_not_inv += 1;
                }
            }
        }
        println!(
            "\n  slack_max={}: {} families | route changes vs shipped: {} ({} gain MC64, {} lose it)",
            slack_max, total, changed, gained.len(), lost.len()
        );
        println!(
            "    invariant under reversal: candidate {}/{}   shipped router NOT invariant on {}",
            inv_ok, total, shipped_not_inv
        );
        if !gained.is_empty() {
            println!("    gain MC64: {}", gained.join(" "));
        }
        if !lost.is_empty() {
            println!("    lose MC64: {}", lost.join(" "));
        }
    }
}

/// The 8 families whose route changes under the invariant gate
/// (slack_max=2). This is the whole risk surface of #134B: every
/// other family routes exactly where it does today.
/// Discover the families whose route changes under the shippable
/// variant: gate (a) unchanged (stored `nnz_col == 1 && diag`), gate
/// (b) on symmetric max degree. Monotone -- every change is a gain.
fn movers_gate_b() -> Vec<(String, PathBuf)> {
    let mut out = Vec::new();
    let mut dirs: Vec<PathBuf> = Vec::new();
    for root in CORPUS_ROOTS {
        if let Ok(rd) = std::fs::read_dir(Path::new(root)) {
            dirs.extend(
                rd.filter_map(|e| e.ok())
                    .map(|e| e.path())
                    .filter(|p| p.is_dir()),
            );
        }
    }
    dirs.sort();
    for d in dirs {
        let Some((_, m)) = iterates(&d, 1).into_iter().next() else {
            continue;
        };
        let st = stored_gates(&m);
        let sy = invariant_gates(&m, 2);
        let base = st.route(m.n);
        let new = Gates {
            diag_only: st.diag_only,
            max_deg: sy.max_deg,
        }
        .route(m.n);
        if base != new {
            let name = d
                .file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default();
            out.push((name, d));
        }
    }
    out
}

/// Factor every iterate of `m` under a forced strategy and report
/// (median factor ms, max scaled residual, inertia signature, failures).
fn run_route(mats: &[(String, CscMatrix)], strat: ScalingStrategy) -> (f64, f64, Vec<i64>, usize) {
    let mut times = Vec::new();
    let mut worst = 0.0f64;
    let mut sig = Vec::new();
    let mut fails = 0usize;
    for (_, m) in mats {
        let mut s = Solver::new().with_scaling(strat.clone());
        // 3 repeats, take the min: factor time here is a few hundred
        // microseconds, so timer noise dominates a single sample.
        let mut best = f64::INFINITY;
        let mut ok = false;
        for _ in 0..3 {
            s = Solver::new().with_scaling(strat.clone());
            let t = Instant::now();
            let st = s.factor(m, None);
            let el = t.elapsed().as_secs_f64() * 1e3;
            ok = matches!(st, feral::FactorStatus::Success);
            best = best.min(el);
        }
        if !ok {
            fails += 1;
            sig.push(i64::MIN);
            continue;
        }
        times.push(best);
        sig.push(match s.inertia() {
            Some(i) => i.negative as i64,
            None => -1,
        });
        // rhs = A * ones, so the exact solution is all ones.
        let n = m.n;
        let mut b = vec![0.0f64; n];
        for j in 0..n {
            for k in m.col_ptr[j]..m.col_ptr[j + 1] {
                let i = m.row_idx[k];
                let v = m.values[k];
                b[i] += v;
                if i != j {
                    b[j] += v;
                }
            }
        }
        if let Ok(x) = s.solve(&b) {
            let err = x.iter().map(|v| (v - 1.0).abs()).fold(0.0f64, f64::max);
            if err.is_finite() {
                worst = worst.max(err);
            } else {
                worst = f64::INFINITY;
            }
        }
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let med = if times.is_empty() {
        f64::NAN
    } else {
        times[times.len() / 2]
    };
    (med, worst, sig, fails)
}

/// Price the route changes: does the new route cost time or accuracy
/// on the 8 families that move?
fn price_movers() {
    println!("\n(d) pricing the 8 route changes (all iterates, min-of-3 factor, rhs = A*1):\n");
    println!(
        "  {:<11} {:>5} {:>4} {:>11} {:>11} {:>7} {:>11} {:>11} {:>9}",
        "family",
        "n",
        "its",
        "InfNorm ms",
        "MC64 ms",
        "t ratio",
        "InfNorm err",
        "MC64 err",
        "inertia"
    );
    for (fam, dir) in movers_gate_b() {
        let fam = fam.as_str();
        let dir_note = &"gain MC64";
        let mats = iterates(&dir, 40);
        if mats.is_empty() {
            println!("  {:<11} (no iterates)", fam);
            continue;
        }
        let n = mats[0].1.n;
        let (t_inf, e_inf, s_inf, f_inf) = run_route(&mats, ScalingStrategy::InfNorm);
        let (t_m64, e_m64, s_m64, f_m64) = run_route(&mats, ScalingStrategy::Mc64Symmetric);
        // Which of the two is the NEW route?
        let new_is_mc64 = *dir_note == "gain MC64";
        let (t_old, t_new) = if new_is_mc64 {
            (t_inf, t_m64)
        } else {
            (t_m64, t_inf)
        };
        println!(
            "  {:<11} {:>5} {:>4} {:>11.3} {:>11.3} {:>7.2} {:>11.2e} {:>11.2e} {:>9} {} {}",
            fam,
            n,
            mats.len(),
            t_inf,
            t_m64,
            t_new / t_old,
            e_inf,
            e_m64,
            if s_inf == s_m64 { "same" } else { "DIFFER" },
            dir_note,
            if f_inf + f_m64 > 0 {
                format!("[fail inf={} mc64={}]", f_inf, f_m64)
            } else {
                String::new()
            },
        );
    }
    println!(
        "\n  't ratio' is new-route time / old-route time; <1 means the change is free or faster."
    );
}

/// How marginal is each `slack_max` decision? Prints the slack-mass
/// ratio at slack_max = 1,2,3 against the 0.30 gate for the panel and
/// the movers, so the threshold is chosen on data rather than taste.
fn marginality() {
    println!("\n(e) slack-mass ratio vs the 0.30 gate, by slack_max:\n");
    println!(
        "  {:<11} {:>6} {:>6} {:>8} {:>8} {:>8}  gate b (md>32)",
        "family", "n", "sym md", "s<=1/n", "s<=2/n", "s<=3/n"
    );
    let kkt = Path::new("data/matrices/kkt");
    let fams = [
        "clnlbeam",
        "CLNLBEAM",
        "steering",
        "VESUVIO",
        "VESUVIOU",
        "VESUVIA",
        "MUONSINE",
        "CRESC132",
        "ACOPP30",
        "ACOPR14",
        "CRESC100",
        "DECONVBNE",
        "AIRPORT",
        "GOFFIN",
        "HAIFAM",
        "LHAIFAM",
        "MAKELA4",
        "ACOPR30",
    ];
    for fam in fams {
        let dir = CORPUS_ROOTS
            .iter()
            .map(|r| Path::new(r).join(fam))
            .find(|p| p.is_dir())
            .unwrap_or_else(|| kkt.join(fam));
        let its = iterates(&dir, 1);
        let Some((_, m)) = its.first() else {
            println!("  {:<11} (missing)", fam);
            continue;
        };
        let g = |k| invariant_gates(m, k);
        let md = g(1).max_deg;
        println!(
            "  {:<11} {:>6} {:>6} {:>8.3} {:>8.3} {:>8.3}  {}",
            fam,
            m.n,
            md,
            g(1).diag_only as f64 / m.n as f64,
            g(2).diag_only as f64 / m.n as f64,
            g(3).diag_only as f64 / m.n as f64,
            if md > 32 {
                "pass"
            } else {
                "FAIL -> InfNorm regardless"
            },
        );
    }
}

/// For families the shipped router sends to MC64, what is the
/// symmetric degree of the columns it counts as "slack"? If those
/// degrees are small and tightly clustered a single `slack_max`
/// works; if they spread, the slack-mass gate needs a different
/// invariant formulation.
fn slack_degree_spectrum() {
    let fams = [
        "VESUVIO",
        "MUONSINE",
        "CRESC132",
        "ACOPR30",
        "marine_1600",
        "rocket_12800",
        "ROCKET",
        "STEERING",
        "robot_1600",
        "pinene_3200",
        "gasoil_3200",
        "TWIRISM1",
        "MSQRTA",
        "LEUVEN3",
        "ACOPP118",
        "MPC1",
        "CMPC2",
    ];
    println!("\n(f) symmetric degree of the columns the SHIPPED gate counts as slack:\n");
    println!(
        "  {:<13} {:>7} {:>8} {:>7}  {:>5} {:>5} {:>5} {:>5}   cumulative frac of n with sym deg <= k",
        "family", "n", "stored/n", "sym md", "min", "med", "p90", "max"
    );
    println!(
        "  {:<13} {:>7} {:>8} {:>7}  {:>5} {:>5} {:>5} {:>5}   {:>6} {:>6} {:>6} {:>6} {:>6}",
        "", "", "", "", "", "", "", "", "k=2", "k=3", "k=4", "k=6", "k=8"
    );
    for fam in fams {
        let dir = CORPUS_ROOTS
            .iter()
            .map(|r| Path::new(r).join(fam))
            .find(|p| p.is_dir());
        let Some(dir) = dir else {
            println!("  {:<13} (missing)", fam);
            continue;
        };
        let its = iterates(&dir, 1);
        let Some((_, m)) = its.first() else { continue };
        let n = m.n;
        // symmetric degrees
        let mut deg = vec![0usize; n];
        for j in 0..n {
            for k in m.col_ptr[j]..m.col_ptr[j + 1] {
                if m.values[k] == 0.0 {
                    continue;
                }
                let i = m.row_idx[k];
                deg[j] += 1;
                if i != j {
                    deg[i] += 1;
                }
            }
        }
        // degrees of the columns the shipped gate counts
        let mut counted = Vec::new();
        #[allow(clippy::needless_range_loop)]
        for j in 0..n {
            let mut nnz = 0;
            let mut d = false;
            for k in m.col_ptr[j]..m.col_ptr[j + 1] {
                if m.values[k] == 0.0 {
                    continue;
                }
                nnz += 1;
                if m.row_idx[k] == j {
                    d = true;
                }
            }
            if nnz == 1 && d {
                counted.push(deg[j]);
            }
        }
        if counted.is_empty() {
            println!(
                "  {:<13} {:>7} {:>8.3} {:>7}  (shipped gate counts none)",
                fam,
                n,
                0.0,
                deg.iter().max().copied().unwrap_or(0)
            );
            continue;
        }
        counted.sort_unstable();
        let pick = |q: f64| counted[((counted.len() as f64 - 1.0) * q) as usize];
        let cum = |k: usize| deg.iter().filter(|&&d| d <= k).count() as f64 / n as f64;
        println!(
            "  {:<13} {:>7} {:>8.3} {:>7}  {:>5} {:>5} {:>5} {:>5}   {:>6.3} {:>6.3} {:>6.3} {:>6.3} {:>6.3}",
            fam,
            n,
            counted.len() as f64 / n as f64,
            deg.iter().max().copied().unwrap_or(0),
            counted[0],
            pick(0.5),
            pick(0.9),
            counted[counted.len() - 1],
            cum(2),
            cum(3),
            cum(4),
            cum(6),
            cum(8),
        );
    }
}

/// The complete design space: gate (a) and gate (b) can each be
/// computed on stored or symmetric degrees, independently. Which
/// combinations are permutation-invariant, and what does each cost in
/// route churn?
fn decision_table() {
    println!("\n(g) gate (a) x gate (b) design table over the full corpus:\n");
    let mut dirs: Vec<PathBuf> = Vec::new();
    for root in CORPUS_ROOTS {
        if let Ok(rd) = std::fs::read_dir(Path::new(root)) {
            dirs.extend(
                rd.filter_map(|e| e.ok())
                    .map(|e| e.path())
                    .filter(|p| p.is_dir()),
            );
        }
    }
    dirs.sort();
    let mats: Vec<CscMatrix> = dirs
        .iter()
        .filter_map(|d| iterates(d, 1).into_iter().next().map(|(_, m)| m))
        .collect();
    let revs: Vec<Option<CscMatrix>> = mats.iter().map(|m| reverse_permute(m).ok()).collect();

    // route(a_sym, b_sym): gate (a) counts slack as sym deg <= 2 when
    // a_sym, else stored nnz == 1; gate (b) uses sym max deg when
    // b_sym, else stored max nnz.
    fn route(m: &CscMatrix, a_sym: bool, b_sym: bool) -> ScalingStrategy {
        let st = stored_gates(m);
        let sy = invariant_gates(m, 2);
        let g = Gates {
            diag_only: if a_sym { sy.diag_only } else { st.diag_only },
            max_deg: if b_sym { sy.max_deg } else { st.max_deg },
        };
        g.route(m.n)
    }

    println!(
        "  {:<22} {:>9} {:>9} {:>9}  {:>26}",
        "variant", "changes", "gain", "lose", "invariant under reversal"
    );
    for (label, a_sym, b_sym) in [
        ("shipped (stored,stored)", false, false),
        ("(stored a, sym b)", false, true),
        ("(sym a, stored b)", true, false),
        ("(sym a, sym b)", true, true),
    ] {
        let (mut ch, mut gain, mut lose, mut inv) = (0, 0, 0, 0);
        for (i, m) in mats.iter().enumerate() {
            let base = route(m, false, false);
            let r = route(m, a_sym, b_sym);
            if r != base {
                ch += 1;
                if r == ScalingStrategy::Mc64Symmetric {
                    gain += 1
                } else {
                    lose += 1
                }
            }
            if let Some(rev) = &revs[i] {
                if route(rev, a_sym, b_sym) == r {
                    inv += 1;
                }
            }
        }
        println!(
            "  {:<22} {:>9} {:>9} {:>9}  {:>19}/{}",
            label,
            ch,
            gain,
            lose,
            inv,
            mats.len()
        );
    }

    // Does each variant hold the duals-last case (reversed VESUVIO)?
    println!("\n  reversed-VESUVIO check (the pounce/discopt duals-last shape):");
    let kkt = Path::new("data/matrices/kkt");
    for fam in ["VESUVIO", "MUONSINE", "CRESC132"] {
        let its = iterates(&kkt.join(fam), 1);
        let Some((_, m)) = its.first() else { continue };
        let Ok(rev) = reverse_permute(m) else {
            continue;
        };
        print!("    {:<10}", fam);
        for (label, a_sym, b_sym) in [
            ("stored,stored", false, false),
            ("stored,sym", false, true),
            ("sym,stored", true, false),
            ("sym,sym", true, true),
        ] {
            let ok = route(&rev, a_sym, b_sym) == ScalingStrategy::Mc64Symmetric;
            print!("  {}={}", label, if ok { "MC64" } else { "InfNorm" });
        }
        println!();
    }
}

/// Post-fix verification: call the SHIPPED `pick_scaling_strategy`
/// (not a local reimplementation) on every family and its symmetric
/// reversal, and count invariance and route changes for real.
fn verify_shipped() {
    println!("\n(h) shipped pick_scaling_strategy, measured directly:\n");
    let mut dirs: Vec<PathBuf> = Vec::new();
    for root in CORPUS_ROOTS {
        if let Ok(rd) = std::fs::read_dir(Path::new(root)) {
            dirs.extend(
                rd.filter_map(|e| e.ok())
                    .map(|e| e.path())
                    .filter(|p| p.is_dir()),
            );
        }
    }
    dirs.sort();
    let (mut total, mut inv, mut vs_old, mut gain, mut lose) = (0, 0, 0, 0, 0);
    let mut noninv = Vec::new();
    for d in &dirs {
        let Some((_, m)) = iterates(d, 1).into_iter().next() else {
            continue;
        };
        total += 1;
        let r = pick_scaling_strategy(&m);
        // the pre-fix router, reproduced locally
        let old = stored_gates(&m).route(m.n);
        if r != old {
            vs_old += 1;
            if r == ScalingStrategy::Mc64Symmetric {
                gain += 1
            } else {
                lose += 1
            }
        }
        if let Ok(rev) = reverse_permute(&m) {
            if pick_scaling_strategy(&rev) == r {
                inv += 1;
            } else if noninv.len() < 12 {
                noninv.push(
                    d.file_name()
                        .map(|s| s.to_string_lossy().to_string())
                        .unwrap_or_default(),
                );
            }
        }
    }
    println!("  families:                     {}", total);
    println!("  invariant under reversal:     {}/{}", inv, total);
    println!(
        "  route changes vs pre-fix:     {} ({} gain MC64, {} lose)",
        vs_old, gain, lose
    );
    println!(
        "  still order-dependent (gate a, sample): {}",
        noninv.join(" ")
    );
}

fn main() {
    let which: Vec<String> = std::env::args().skip(1).collect();
    let want = |k: &str| which.is_empty() || which.iter().any(|a| a == k);
    if want("153") {
        issue_153();
    }
    if want("cache") {
        cache_hits();
    }
    if want("kr") {
        kr_budget();
    }
    if want("134b") {
        issue_134b();
    }
    if want("price") {
        price_movers();
    }
    if want("marg") {
        marginality();
    }
    if want("spec") {
        slack_degree_spectrum();
    }
    if want("table") {
        decision_table();
    }
    if want("verify") {
        verify_shipped();
    }
}
