use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::time::Instant;

use feral::numeric::factorize::factorize_multifrontal;
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::{
    factor, factor_single_front, read_mtx, read_sidecar, solve, solve_refined,
    solve_sparse_refined, BunchKaufmanParams, CscMatrix, Inertia, KktSidecar, SymmetricMatrix,
    ZeroPivotAction,
};

/// A KKT matrix that failed inertia or residual on a given solver path.
#[derive(Clone)]
struct Failure {
    name: String,
    n: usize,
    expected: Inertia,
    actual: Inertia,
    inertia_ok: bool,
    residual: f64,
    residual_ok: bool,
}

/// Extract the problem family from a matrix name like "POLAK6_0021" → "POLAK6".
/// Strips the trailing `_<digits>` if present.
fn family_of(name: &str) -> &str {
    if let Some(idx) = name.rfind('_') {
        let suffix = &name[idx + 1..];
        if !suffix.is_empty() && suffix.chars().all(|c| c.is_ascii_digit()) {
            return &name[..idx];
        }
    }
    name
}

fn print_failure_analysis(label: &str, failures: &[Failure]) {
    if failures.is_empty() {
        println!("\n{} failure analysis: no failures", label);
        return;
    }
    println!(
        "\n--- {} failure analysis ({} failures) ---",
        label,
        failures.len()
    );

    // Group by problem family
    let mut by_family: HashMap<&str, (usize, usize, f64, usize)> = HashMap::new();
    for f in failures {
        let fam = family_of(&f.name);
        let entry = by_family.entry(fam).or_insert((0, 0, 0.0, 0));
        entry.3 += 1;
        if !f.inertia_ok {
            entry.0 += 1;
        }
        if !f.residual_ok {
            entry.1 += 1;
        }
        if f.residual > entry.2 {
            entry.2 = f.residual;
        }
    }

    let mut families: Vec<_> = by_family.into_iter().collect();
    families.sort_by_key(|(_, v)| std::cmp::Reverse(v.3));

    println!(
        "\n{:<22} {:>8} {:>10} {:>10} {:>14}",
        "family", "total", "inertia", "residual", "worst_res"
    );
    for (fam, (ifail, rfail, worst, total)) in families.iter().take(25) {
        println!(
            "{:<22} {:>8} {:>10} {:>10} {:>14.2e}",
            fam, total, ifail, rfail, worst
        );
    }
    if families.len() > 25 {
        println!("  ... and {} more families", families.len() - 25);
    }

    // Top 20 worst by residual
    let mut by_residual: Vec<&Failure> = failures.iter().collect();
    by_residual.sort_by(|a, b| {
        b.residual
            .partial_cmp(&a.residual)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    println!("\nTop 15 worst residuals:");
    println!(
        "{:<28} {:>5} {:>12} {:>14} {:>14}",
        "name", "n", "residual", "expected", "actual"
    );
    for f in by_residual.iter().take(15) {
        println!(
            "{:<28} {:>5} {:>12.2e} {:>14} {:>14}",
            f.name,
            f.n,
            f.residual,
            format!("{}", f.expected),
            format!("{}", f.actual),
        );
    }
}

fn print_cross_comparison(dense: &[Failure], sparse: &[Failure]) {
    let dense_by_name: HashMap<&str, &Failure> =
        dense.iter().map(|f| (f.name.as_str(), f)).collect();
    let sparse_by_name: HashMap<&str, &Failure> =
        sparse.iter().map(|f| (f.name.as_str(), f)).collect();

    let dense_names: HashSet<&str> = dense_by_name.keys().copied().collect();
    let sparse_names: HashSet<&str> = sparse_by_name.keys().copied().collect();
    let shared_names: Vec<&str> = dense_names.intersection(&sparse_names).copied().collect();
    let both = shared_names.len();
    let dense_only = dense_names.len() - both;
    let sparse_only = sparse_names.len() - both;

    println!("\n--- Dense ∩ Sparse failure overlap ---");
    println!("Failed in BOTH dense and sparse:  {}", both);
    println!("Failed in dense only:             {}", dense_only);
    println!("Failed in sparse only:            {}", sparse_only);

    if shared_names.is_empty() {
        return;
    }

    // Build the joined shared records so we can bucket by failure mode.
    struct Shared<'a> {
        d: &'a Failure,
        s: &'a Failure,
    }
    let shared: Vec<Shared> = shared_names
        .iter()
        .map(|name| Shared {
            d: dense_by_name[name],
            s: sparse_by_name[name],
        })
        .collect();

    // Bucket 1: by failure mode. A "mode" is the pair (dense inertia ok?,
    // dense residual ok?, sparse inertia ok?, sparse residual ok?).
    // Simplified to 3 buckets that are easy to interpret:
    //   INERTIA_BOTH   — both paths disagree with MUMPS on inertia
    //   RESIDUAL_BOTH  — both paths match inertia but have bad residual
    //   MIXED          — one path has inertia issue, the other residual-only
    let mut inertia_both = 0usize;
    let mut residual_both = 0usize;
    let mut mixed = 0usize;
    for Shared { d, s } in &shared {
        let d_ine = !d.inertia_ok;
        let s_ine = !s.inertia_ok;
        let d_res = !d.residual_ok;
        let s_res = !s.residual_ok;
        if d_ine && s_ine {
            inertia_both += 1;
        } else if d_res && s_res && !d_ine && !s_ine {
            residual_both += 1;
        } else {
            mixed += 1;
        }
    }
    println!("\nShared failure mode breakdown:");
    println!(
        "  Inertia mismatch on BOTH paths:        {:>6}",
        inertia_both
    );
    println!(
        "  Residual-only fail on BOTH paths:      {:>6}",
        residual_both
    );
    println!("  Mixed (one inertia, other residual):   {:>6}", mixed);

    // Bucket 2: by size class.
    let mut small = 0usize; // n <= 100
    let mut med = 0usize; // 100 < n <= 1000
    let mut large = 0usize; // n > 1000
    for Shared { d, .. } in &shared {
        match d.n {
            0..=100 => small += 1,
            101..=1000 => med += 1,
            _ => large += 1,
        }
    }
    println!("\nShared failure size class breakdown:");
    println!("  n <=  100:  {:>6}", small);
    println!("  n <= 1000:  {:>6}", med);
    println!("  n >  1000:  {:>6}", large);

    // Bucket 3: top families among shared failures.
    let mut fam_counts: HashMap<&str, (usize, usize, usize, f64)> = HashMap::new();
    // value: (total, inertia-both count, residual-both count, worst dense+sparse max residual)
    for Shared { d, s } in &shared {
        let fam = family_of(&d.name);
        let entry = fam_counts.entry(fam).or_insert((0, 0, 0, 0.0));
        entry.0 += 1;
        if !d.inertia_ok && !s.inertia_ok {
            entry.1 += 1;
        }
        if !d.residual_ok && !s.residual_ok && d.inertia_ok && s.inertia_ok {
            entry.2 += 1;
        }
        let worst = d.residual.max(s.residual);
        if worst > entry.3 {
            entry.3 = worst;
        }
    }
    let mut fams: Vec<_> = fam_counts.into_iter().collect();
    fams.sort_by_key(|(_, v)| std::cmp::Reverse(v.0));
    println!("\nTop 25 families in shared failures:");
    println!(
        "{:<22} {:>8} {:>10} {:>10} {:>14}",
        "family", "total", "inertia", "residual", "worst_res"
    );
    for (fam, (total, ine, res, worst)) in fams.iter().take(25) {
        println!(
            "{:<22} {:>8} {:>10} {:>10} {:>14.2e}",
            fam, total, ine, res, worst
        );
    }
    if fams.len() > 25 {
        println!("  ... and {} more families", fams.len() - 25);
    }

    // Bucket 4: top 15 worst shared residuals (max of dense/sparse).
    let mut by_worst: Vec<&Shared> = shared.iter().collect();
    by_worst.sort_by(|a, b| {
        let aw = a.d.residual.max(a.s.residual);
        let bw = b.d.residual.max(b.s.residual);
        bw.partial_cmp(&aw).unwrap_or(std::cmp::Ordering::Equal)
    });
    println!("\nTop 15 worst shared residuals:");
    println!(
        "{:<28} {:>5} {:>12} {:>12} {:>14} {:>14}",
        "name", "n", "dense_res", "sparse_res", "expected", "actual(sp)"
    );
    for sh in by_worst.iter().take(15) {
        println!(
            "{:<28} {:>5} {:>12.2e} {:>12.2e} {:>14} {:>14}",
            sh.d.name,
            sh.d.n,
            sh.d.residual,
            sh.s.residual,
            format!("{}", sh.d.expected),
            format!("{}", sh.s.actual),
        );
    }
}

/// Phase 2.1.7 — emit feral-vs-oracle timing comparisons for a single path.
///
/// Joins per-matrix feral timings against the MUMPS and SSIDS timings that
/// `load_kkt_dir` pulled out of the `*.mumps.json` / `*.ssids.json` sidecars,
/// computes ratio = feral_time / oracle_time, and reports:
///
///   1. Overall distribution of factor and solve ratios (geomean, p50, p90,
///      p99, max) per oracle.
///   2. Per-family geomean factor ratio against MUMPS, sorted by matrix count,
///      so families with many matrices dominate the summary in proportion to
///      their representation in the corpus.
///   3. Top 10 worst factor-ratio matrices versus MUMPS, as a triage hook for
///      Phase 2.4/2.5 optimization work.
///
/// `label` is printed in the section header ("Dense" or "Sparse"). Matrices
/// with no oracle sidecar are counted but excluded from ratio stats — they
/// simply cannot contribute to the comparison.
fn print_perf_comparison(label: &str, timings: &[MatrixTiming], entries: &[KktEntry]) {
    let entry_by_name: HashMap<&str, &KktEntry> =
        entries.iter().map(|e| (e.name.as_str(), e)).collect();

    // Build the joined row set. We only keep rows where feral has a timing
    // AND at least one oracle sidecar has a timing — otherwise there is
    // nothing to compare against. mumps_factor_ratio and friends are
    // per-matrix, per-(factor/solve), per-oracle.
    struct Row<'a> {
        family: String,
        timing: &'a MatrixTiming,
        mumps: Option<OracleTiming>,
        ssids: Option<OracleTiming>,
    }
    let mut rows: Vec<Row> = Vec::new();
    for t in timings {
        let Some(entry) = entry_by_name.get(t.name.as_str()) else {
            continue;
        };
        if entry.mumps_timing.is_none() && entry.ssids_timing.is_none() {
            continue;
        }
        rows.push(Row {
            family: family_of(&t.name).to_string(),
            timing: t,
            mumps: entry.mumps_timing,
            ssids: entry.ssids_timing,
        });
    }

    if rows.is_empty() {
        println!(
            "\n--- {} perf vs oracles: no matrices have oracle timings ---",
            label
        );
        return;
    }

    // Ratio helper. Both feral and oracle microsecond counts are clamped to
    // `>= 1` so that sub-microsecond matrices (which hit the clock-resolution
    // floor on both sides) produce a meaningful ratio of 1.0 rather than
    // collapsing the geometric mean in log space. The clamp is symmetric by
    // design: ratios at the noise floor are uninformative but must not poison
    // the aggregate statistic.
    let ratio = |feral: u128, oracle: u64| -> f64 {
        let num = feral.max(1) as f64;
        let denom = oracle.max(1) as f64;
        num / denom
    };

    // Pull out per-row ratio vectors. A matrix with no MUMPS sidecar is
    // skipped for the MUMPS lists but still contributes to the SSIDS lists
    // (and vice versa).
    let mut mumps_factor: Vec<f64> = Vec::new();
    let mut mumps_solve: Vec<f64> = Vec::new();
    let mut ssids_factor: Vec<f64> = Vec::new();
    let mut ssids_solve: Vec<f64> = Vec::new();
    for r in &rows {
        if let Some(m) = r.mumps {
            mumps_factor.push(ratio(r.timing.factor_us, m.factor_us));
            mumps_solve.push(ratio(r.timing.solve_us, m.solve_us));
        }
        if let Some(s) = r.ssids {
            ssids_factor.push(ratio(r.timing.factor_us, s.factor_us));
            ssids_solve.push(ratio(r.timing.solve_us, s.solve_us));
        }
    }

    // Distribution summary: geomean is the right central tendency for
    // multiplicative quantities like speed ratios. Arithmetic mean would
    // overweight slow matrices. Percentiles use linear interpolation-free
    // nearest-rank to keep the output integer-stable across reruns.
    fn quantiles(vals: &mut [f64]) -> Option<(f64, f64, f64, f64, f64)> {
        if vals.is_empty() {
            return None;
        }
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = vals.len();
        // Geometric mean in log space: avoids overflow on large ratios.
        let log_sum: f64 = vals.iter().map(|v| v.max(1e-300).ln()).sum();
        let geomean = (log_sum / n as f64).exp();
        let idx = |q: f64| -> usize {
            ((n as f64 * q).ceil() as usize)
                .saturating_sub(1)
                .min(n - 1)
        };
        let p50 = vals[idx(0.50)];
        let p90 = vals[idx(0.90)];
        let p99 = vals[idx(0.99)];
        let max = vals[n - 1];
        Some((geomean, p50, p90, p99, max))
    }

    println!(
        "\n=== {} perf vs canonical oracles ({} matrices with oracle timings) ===",
        label,
        rows.len()
    );
    println!(
        "\n{:<18} {:>6} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "ratio", "count", "geomean", "p50", "p90", "p99", "max"
    );
    let emit = |name: &str, vals: &mut Vec<f64>| {
        if let Some((g, p50, p90, p99, mx)) = quantiles(vals) {
            println!(
                "{:<18} {:>6} {:>10.2} {:>10.2} {:>10.2} {:>10.2} {:>10.2}",
                name,
                vals.len(),
                g,
                p50,
                p90,
                p99,
                mx,
            );
        }
    };
    emit("factor/MUMPS", &mut mumps_factor);
    emit("solve/MUMPS", &mut mumps_solve);
    emit("factor/SSIDS", &mut ssids_factor);
    emit("solve/SSIDS", &mut ssids_solve);

    // Per-family factor geomean vs MUMPS. Families with <3 MUMPS-ratio
    // samples are still shown — small families are often the most
    // informative for ACOPP-style outliers — but flagged via count.
    let mut by_family: HashMap<&str, Vec<f64>> = HashMap::new();
    for r in &rows {
        if let Some(m) = r.mumps {
            by_family
                .entry(r.family.as_str())
                .or_default()
                .push(ratio(r.timing.factor_us, m.factor_us));
        }
    }
    let mut fam_rows: Vec<(&str, Vec<f64>)> = by_family.into_iter().collect();
    fam_rows.sort_by(|a, b| b.1.len().cmp(&a.1.len()));
    println!("\nPer-family factor geomean vs MUMPS (top 25 families by count):");
    println!(
        "{:<22} {:>6} {:>10} {:>10} {:>10}",
        "family", "count", "geomean", "p50", "max"
    );
    for (fam, mut vals) in fam_rows.into_iter().take(25) {
        if let Some((g, p50, _, _, mx)) = quantiles(&mut vals) {
            println!(
                "{:<22} {:>6} {:>10.2} {:>10.2} {:>10.2}",
                fam,
                vals.len(),
                g,
                p50,
                mx,
            );
        }
    }

    // Top 10 worst factor-time slowdowns vs MUMPS. These are the matrices
    // to triage when a Phase 2.4/2.5 optimization lands and we want to
    // check whether it moved the tail.
    let mut worst: Vec<(f64, &Row)> = rows
        .iter()
        .filter_map(|r| r.mumps.map(|m| (ratio(r.timing.factor_us, m.factor_us), r)))
        .collect();
    worst.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    println!("\nTop 10 worst factor-ratio vs MUMPS:");
    println!(
        "{:<28} {:>5} {:>12} {:>12} {:>10}",
        "name", "n", "feral(μs)", "mumps(μs)", "ratio"
    );
    for (r, row) in worst.iter().take(10) {
        let m = row.mumps.unwrap();
        println!(
            "{:<28} {:>5} {:>12} {:>12} {:>10.2}",
            row.timing.name, row.timing.n, row.timing.factor_us, m.factor_us, r
        );
    }
}

/// Phase 2.8.1 spec exit partition against the canonical MUMPS factor ratio.
///
/// The Phase 2 exit criterion in `FERAL-PROJECT-SPEC.md` §1747 asks
/// "within 2× of MUMPS on the small-frontal KKT set; within 3× on the
/// medium set". Concrete bucket definitions come from
/// `dev/plans/phase-2-planning.md` §2.8.1:
///
/// - **small-frontal:** max frontal dim < 200 AND problem n <= 10^3,
///   target `factor/MUMPS p90 <= 2.0`
/// - **medium:** max frontal dim < 500 AND problem n <= 10^4,
///   target `factor/MUMPS p90 <= 3.0`
///
/// "Max frontal dim" is `n` for the dense single-front path and
/// `max(supernode.nrow)` from the symbolic factorization for the
/// sparse multifrontal path; both are stored on `MatrixTiming` at
/// push time. Matrices without a MUMPS oracle sidecar are excluded —
/// they cannot contribute to a ratio statistic.
fn print_phase28_partition(label: &str, timings: &[MatrixTiming], entries: &[KktEntry]) {
    let entry_by_name: HashMap<&str, &KktEntry> =
        entries.iter().map(|e| (e.name.as_str(), e)).collect();

    let ratio = |feral: u128, oracle: u64| -> f64 {
        let num = feral.max(1) as f64;
        let denom = oracle.max(1) as f64;
        num / denom
    };

    let mut small: Vec<f64> = Vec::new();
    let mut medium: Vec<f64> = Vec::new();
    for t in timings {
        let Some(entry) = entry_by_name.get(t.name.as_str()) else {
            continue;
        };
        let Some(m) = entry.mumps_timing else {
            continue;
        };
        let r = ratio(t.factor_us, m.factor_us);
        if t.max_front < 200 && t.n <= 1_000 {
            small.push(r);
        }
        if t.max_front < 500 && t.n <= 10_000 {
            medium.push(r);
        }
    }

    fn p90(vals: &mut [f64]) -> Option<f64> {
        if vals.is_empty() {
            return None;
        }
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((vals.len() as f64 * 0.90).ceil() as usize)
            .saturating_sub(1)
            .min(vals.len() - 1);
        Some(vals[idx])
    }

    println!(
        "\n--- {} Phase 2.8.1 exit partition (factor ratio vs MUMPS) ---",
        label
    );
    println!(
        "{:<24} {:>6} {:>8} {:>10} {:>8}",
        "bucket", "count", "p90", "target", "verdict"
    );
    let emit = |name: &str, target: f64, vals: &mut Vec<f64>| {
        if let Some(p) = p90(vals) {
            let verdict = if p <= target { "PASS" } else { "FAIL" };
            println!(
                "{:<24} {:>6} {:>8.2} {:>10} {:>8}",
                name,
                vals.len(),
                p,
                format!("<= {:.1}", target),
                verdict,
            );
        } else {
            println!(
                "{:<24} {:>6} {:>8} {:>10} {:>8}",
                name,
                0,
                "-",
                format!("<= {:.1}", target),
                "N/A",
            );
        }
    };
    emit("small-frontal (<200)", 2.0, &mut small);
    emit("medium (<500)", 3.0, &mut medium);
}

/// Simple deterministic PRNG for benchmark matrix generation.
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        let t = (self.next_u64() as f64) / (u64::MAX as f64);
        lo + t * (hi - lo)
    }
}

/// Generate a random SPD matrix: A = M·Mᵀ + δI
fn random_spd(n: usize, rng: &mut Rng) -> SymmetricMatrix {
    let mut mat = SymmetricMatrix::zeros(n);
    let mut m = vec![0.0; n * n];
    for j in 0..n {
        for i in j..n {
            m[j * n + i] = rng.uniform(-1.0, 1.0);
        }
    }
    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in 0..n {
                sum += m[k * n + i] * m[k * n + j];
            }
            mat.set(i, j, sum + if i == j { 0.01 } else { 0.0 });
        }
    }
    mat
}

/// Generate a random KKT matrix
fn random_kkt(n_var: usize, n_con: usize, rng: &mut Rng) -> SymmetricMatrix {
    let n = n_var + n_con;
    let mut mat = SymmetricMatrix::zeros(n);

    for i in 0..n_var {
        mat.set(i, i, rng.uniform(1.0, 5.0) + n_var as f64 * 0.5);
        for j in 0..i {
            mat.set(i, j, rng.uniform(-0.3, 0.3));
        }
    }
    for i in 0..n_con {
        for j in 0..n_var {
            mat.set(n_var + i, j, rng.uniform(-2.0, 2.0));
        }
        mat.set(n_var + i, n_var + i, -1e-8);
    }
    mat
}

struct BenchResult {
    name: String,
    n: usize,
    factor_us: u128,
    solve_us: u128,
    inertia: String,
}

fn bench_matrix(
    name: &str,
    mat: &SymmetricMatrix,
    params: &BunchKaufmanParams,
    rhs: &[f64],
) -> Option<BenchResult> {
    let t0 = Instant::now();
    let (factors, inertia) = match factor(mat, params) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("  {}: factor failed: {}", name, e);
            return None;
        }
    };
    let factor_us = t0.elapsed().as_micros();

    let t1 = Instant::now();
    match solve(&factors, rhs) {
        Ok(_) => {}
        Err(e) => {
            eprintln!("  {}: solve failed: {}", name, e);
            return None;
        }
    };
    let solve_us = t1.elapsed().as_micros();

    Some(BenchResult {
        name: name.to_string(),
        n: mat.n,
        factor_us,
        solve_us,
        inertia: format!("{}", inertia),
    })
}

/// A loaded KKT matrix with its sidecar metadata.
struct KktEntry {
    name: String,
    /// Path to the .mtx file. Used to write `.feral.json` sidecars next to it.
    mtx_path: std::path::PathBuf,
    matrix: SymmetricMatrix,
    csc: CscMatrix,
    sidecar: KktSidecar,
    /// Canonical Fortran MUMPS 5.8.2 oracle timing from `*.mumps.json` if present.
    mumps_timing: Option<OracleTiming>,
    /// Canonical SPRAL SSIDS oracle timing from `*.ssids.json` if present.
    ssids_timing: Option<OracleTiming>,
}

/// Factor and solve microseconds from a canonical oracle sidecar.
///
/// Populated from `*.mumps.json` / `*.ssids.json` written by the oracle runners
/// in `external_benchmarks/`. Used by `print_perf_comparison` to compute
/// feral/oracle ratios for the Phase 2.1.7 baseline report.
#[derive(Clone, Copy)]
struct OracleTiming {
    factor_us: u64,
    solve_us: u64,
}

/// Per-matrix feral factor and solve timing on a single path.
///
/// Phase 2.1.7 perf harness collects these alongside the existing pass/fail
/// tallies so that `print_perf_comparison` can join them against the oracle
/// timings stored in `KktEntry::{mumps_timing, ssids_timing}` and emit
/// feral/MUMPS and feral/SSIDS ratios grouped by problem family.
#[derive(Clone)]
struct MatrixTiming {
    name: String,
    n: usize,
    /// Maximum frontal-matrix dimension used by the path that produced this
    /// timing. For the dense single-front path this equals `n`. For the
    /// sparse multifrontal path it is the maximum `supernode.nrow` across
    /// all supernodes in the symbolic factorization. Used by Phase 2.8.1
    /// to partition the corpus into "small-frontal" and "medium" sets
    /// against the spec exit criterion.
    max_front: usize,
    factor_us: u128,
    solve_us: u128,
}

/// Parse `factor_us` and `solve_us` from a canonical-oracle JSON sidecar.
///
/// Returns `None` if the file does not exist, cannot be parsed, or is missing
/// either timing field. Matches the schema written by MUMPS and SSIDS runners
/// (see `data/matrices/kkt/ACOPP30/ACOPP30_0000.mumps.json` for an example).
fn read_oracle_timing(path: &Path) -> Option<OracleTiming> {
    let contents = std::fs::read_to_string(path).ok()?;
    let v: serde_json::Value = serde_json::from_str(&contents).ok()?;
    let factor_us = v.get("factor_us")?.as_u64()?;
    let solve_us = v.get("solve_us")?.as_u64()?;
    Some(OracleTiming {
        factor_us,
        solve_us,
    })
}

/// Write a canonical `.feral.json` sidecar next to the matrix file.
/// Schema matches dev/plans/phase-1b-consensus-exit.md and the MUMPS/SSIDS
/// oracle outputs in external_benchmarks/.
#[allow(clippy::too_many_arguments)]
fn write_feral_sidecar(
    mtx_path: &Path,
    name: &str,
    n: usize,
    nnz: usize,
    factor_us: u128,
    solve_us: u128,
    inertia: &Inertia,
    residual: f64,
    needs_refinement: bool,
    path_label: &str,
) -> Result<(), std::io::Error> {
    let suffix = format!("{}.json", path_label);
    let mut canonical = mtx_path.to_path_buf();
    canonical.set_extension(suffix);

    let json = format!(
        "{{\"solver\":\"{}\",\"version\":\"0.1.0\",\"matrix\":\"{}\",\
         \"n\":{},\"nnz\":{},\"factor_us\":{},\"solve_us\":{},\
         \"inertia\":{{\"positive\":{},\"negative\":{},\"zero\":{}}},\
         \"rhs_source\":\"sidecar\",\"residual_2norm_relative\":{:.17e},\
         \"factorization_status\":\"ok\",\
         \"solver_info\":{{\"needs_refinement\":{}}}}}\n",
        path_label,
        name,
        n,
        nnz,
        factor_us,
        solve_us,
        inertia.positive,
        inertia.negative,
        inertia.zero,
        residual,
        needs_refinement,
    );
    std::fs::write(canonical, json)
}

/// Load all KKT matrices from `dir`, returning them sorted by name.
/// Returns an empty vec if the directory does not exist.
fn load_kkt_dir(dir: &Path) -> Vec<KktEntry> {
    if !dir.is_dir() {
        return Vec::new();
    }

    let mut entries = Vec::new();

    // Walk subdirectories (one per problem)
    let mut subdirs: Vec<_> = match std::fs::read_dir(dir) {
        Ok(d) => d.filter_map(|e| e.ok()).collect(),
        Err(_) => return Vec::new(),
    };
    subdirs.sort_by_key(|e| e.file_name());

    for subdir in subdirs {
        let subdir_path = subdir.path();
        if !subdir_path.is_dir() {
            continue;
        }

        // Find all .mtx files in this subdirectory
        let mut mtx_files: Vec<_> = match std::fs::read_dir(&subdir_path) {
            Ok(d) => d
                .filter_map(|e| e.ok())
                .filter(|e| e.path().extension().is_some_and(|ext| ext == "mtx"))
                .collect(),
            Err(_) => continue,
        };
        mtx_files.sort_by_key(|e| e.file_name());

        for mtx_entry in mtx_files {
            let mtx_path = mtx_entry.path();
            let stem = mtx_path.file_stem().unwrap().to_string_lossy().to_string();
            let json_path = mtx_path.with_extension("json");

            if !json_path.exists() {
                eprintln!("  SKIP {} (no .json sidecar)", stem);
                continue;
            }

            let mtx = match read_mtx(&mtx_path) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("  SKIP {} (mtx parse error: {})", stem, e);
                    continue;
                }
            };

            // Phase 1a hard-coded a `mtx.n > 500` skip here. Phase 2
            // lifts it: the sparse multifrontal path has no reason to
            // skip larger matrices, and the Phase 1b validation that
            // never ran on n > 500 is the #1 known scope gap
            // (see dev/plans/phase-2-planning.md §2.1.1). Dense BK is
            // still O(n^3) and painful above a few thousand, so the
            // dense loop below has its own inline cutoff — but this
            // load-time filter is gone.

            let sidecar = match read_sidecar(&json_path) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("  SKIP {} (json parse error: {})", stem, e);
                    continue;
                }
            };

            // Skip matrices with NaN/Inf in RHS or matrix data (diverged IPM)
            if sidecar.finite_rhs().is_none() {
                continue;
            }
            if mtx.entries.iter().any(|(_, _, v)| !v.is_finite()) {
                continue;
            }

            // Validate dimension consistency
            let expected_dim = sidecar.n + sidecar.m;
            if mtx.n != expected_dim {
                eprintln!(
                    "  SKIP {} (mtx dim {} != sidecar n+m={}+{}={})",
                    stem, mtx.n, sidecar.n, sidecar.m, expected_dim
                );
                continue;
            }

            let csc = match mtx.to_csc() {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("  SKIP {} (csc conversion: {})", stem, e);
                    continue;
                }
            };

            // Oracle timing sidecars written by external_benchmarks/{mumps,ssids}_oracle.
            // Schema: see OracleTiming and read_oracle_timing above. Missing files are OK
            // and simply leave the timing fields as None.
            let mumps_timing = read_oracle_timing(&mtx_path.with_extension("mumps.json"));
            let ssids_timing = read_oracle_timing(&mtx_path.with_extension("ssids.json"));

            entries.push(KktEntry {
                name: stem,
                mtx_path: mtx_path.clone(),
                matrix: mtx.to_dense(),
                csc,
                sidecar,
                mumps_timing,
                ssids_timing,
            });
        }
    }

    entries
}

fn main() {
    println!("FERAL benchmark harness");

    let config_path = Path::new("data/benchmark-config.toml");
    print!("Loading matrices from {} ... ", config_path.display());

    if config_path.exists() {
        println!("found");
    } else {
        println!("not found");
    }

    // Built-in dense benchmarks
    let mut rng = Rng::new(42);
    let params_spd = BunchKaufmanParams::default();
    // Dense KKT path: pivot_threshold = 0.0 because the dense
    // kernel does not implement delayed pivoting — a non-zero
    // threshold here sends rejected pivots through ForceAccept
    // and zeros out structural pivots on e.g. HYDCAR20, METHANL8,
    // DEGENLPA, HS118.
    let params_kkt_dense = BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        ..BunchKaufmanParams::default()
    };
    // Sparse KKT path (Phase 2.3): pivot_threshold = 0.01
    // (SSIDS/MUMPS default) because delayed pivoting
    // (may_delay=true at non-roots) gives rejected pivots a
    // landing zone at the parent supernode. The column-relative
    // test |d| >= u*col_max bounds the L growth factor by
    // 1/u = 100.
    let params_kkt_sparse = BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        pivot_threshold: 0.01,
        ..BunchKaufmanParams::default()
    };

    let benchmarks: Vec<(&str, SymmetricMatrix, &BunchKaufmanParams)> = vec![
        ("spd_10", random_spd(10, &mut rng), &params_spd),
        ("spd_50", random_spd(50, &mut rng), &params_spd),
        ("spd_100", random_spd(100, &mut rng), &params_spd),
        ("spd_200", random_spd(200, &mut rng), &params_spd),
        ("kkt_10_3", random_kkt(10, 3, &mut rng), &params_kkt_dense),
        ("kkt_30_10", random_kkt(30, 10, &mut rng), &params_kkt_dense),
        ("kkt_50_15", random_kkt(50, 15, &mut rng), &params_kkt_dense),
        (
            "kkt_100_30",
            random_kkt(100, 30, &mut rng),
            &params_kkt_dense,
        ),
    ];

    println!(
        "\n{:<15} {:>5} {:>12} {:>12} {:>14}",
        "name", "n", "factor(μs)", "solve(μs)", "inertia"
    );
    println!("{}", "-".repeat(62));

    let mut count = 0;
    for (name, mat, params) in &benchmarks {
        let n = mat.n;
        let rhs: Vec<f64> = (0..n).map(|i| (i + 1) as f64 * 0.1).collect();

        if let Some(result) = bench_matrix(name, mat, params, &rhs) {
            println!(
                "{:<15} {:>5} {:>12} {:>12} {:>14}",
                result.name, result.n, result.factor_us, result.solve_us, result.inertia
            );
            count += 1;
        }
    }

    println!("\n{} matrices benchmarked", count);

    // --- Real KKT matrices from data/matrices/kkt/ ---
    let kkt_dir = Path::new("data/matrices/kkt");
    print!("\nLoading KKT matrices from {} ... ", kkt_dir.display());

    let kkt_entries = load_kkt_dir(kkt_dir);
    if kkt_entries.is_empty() {
        println!("not found (run collect_kkt from ripopt to generate)");
        return;
    }
    println!("{} matrices loaded", kkt_entries.len());

    let mut n_total = 0usize;
    let mut n_inertia_pass = 0usize;
    let mut n_residual_pass = 0usize;
    let mut n_factor_fail = 0usize;
    let mut worst_residual = 0.0f64;
    let mut worst_residual_name = String::new();
    let mut dense_failures: Vec<Failure> = Vec::new();
    // Phase 2.1.7 perf harness: collect per-matrix dense timings so
    // print_perf_comparison can cross-reference against the canonical
    // MUMPS and SSIDS oracle timings loaded into KktEntry.
    let mut dense_timings: Vec<MatrixTiming> = Vec::new();

    let emit_sidecars = std::env::var("FERAL_EMIT_SIDECARS").is_ok();

    // Dense BK is O(n^3). Above ~1000 it starts to dominate bench runtime
    // (a single n=5314 problem is ~150 GFLOPs scalar-unblocked). The sparse
    // loop below has no such cutoff. FERAL_DENSE_MAX overrides the default.
    let dense_max: usize = std::env::var("FERAL_DENSE_MAX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1000);
    let mut n_dense_skipped_large = 0usize;

    for entry in &kkt_entries {
        n_total += 1;
        let n = entry.matrix.n;
        let nnz = entry.csc.values.len();

        if n > dense_max {
            n_dense_skipped_large += 1;
            continue;
        }

        // Factor
        let t0 = Instant::now();
        let (factors, inertia) = match factor_single_front(&entry.matrix, &params_kkt_sparse) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("  {}: factor failed: {}", entry.name, e);
                n_factor_fail += 1;
                continue;
            }
        };
        let factor_us = t0.elapsed().as_micros();

        // Check inertia against sidecar
        let expected_inertia = Inertia {
            positive: entry.sidecar.inertia.positive,
            negative: entry.sidecar.inertia.negative,
            zero: entry.sidecar.inertia.zero,
        };
        let inertia_ok = inertia == expected_inertia;
        if inertia_ok {
            n_inertia_pass += 1;
        }

        // Solve with sidecar RHS (guaranteed finite by load_kkt_dir filter)
        let rhs = entry.sidecar.finite_rhs().unwrap();
        // Phase 1b solve convention (FERAL-PROJECT-SPEC.md §1709): use
        // solve_refined for all KKT solves to recover machine precision on
        // matrices flagged with needs_refinement under ForceAccept.
        let t1 = Instant::now();
        let x = match solve_refined(&entry.matrix, &factors, &rhs) {
            Ok(x) => x,
            Err(e) => {
                eprintln!("  {}: solve failed: {}", entry.name, e);
                continue;
            }
        };
        let solve_us = t1.elapsed().as_micros();
        dense_timings.push(MatrixTiming {
            name: entry.name.clone(),
            n,
            max_front: n,
            factor_us,
            solve_us,
        });

        // Compute residual: ||Ax - b|| / ||b||
        let mut ax = vec![0.0; n];
        entry.matrix.symv(&x, &mut ax);
        let mut res_norm_sq = 0.0;
        let mut b_norm_sq = 0.0;
        for i in 0..n {
            let r = ax[i] - rhs[i];
            res_norm_sq += r * r;
            b_norm_sq += rhs[i] * rhs[i];
        }
        let relative_residual = if b_norm_sq > 0.0 {
            (res_norm_sq / b_norm_sq).sqrt()
        } else {
            res_norm_sq.sqrt()
        };

        // Residual tolerance: n * eps * condition estimate
        // Use a generous threshold; real KKT matrices can be ill-conditioned
        let residual_tol = (n as f64) * f64::EPSILON * 1e6;
        let residual_ok = relative_residual <= residual_tol;
        if residual_ok {
            n_residual_pass += 1;
        }

        if emit_sidecars {
            let _ = write_feral_sidecar(
                &entry.mtx_path,
                &entry.name,
                n,
                nnz,
                factor_us,
                solve_us,
                &inertia,
                relative_residual,
                factors.needs_refinement,
                "feral",
            );
        }

        if relative_residual > worst_residual {
            worst_residual = relative_residual;
            worst_residual_name = entry.name.clone();
        }

        if !inertia_ok || !residual_ok {
            dense_failures.push(Failure {
                name: entry.name.clone(),
                n,
                expected: expected_inertia,
                actual: inertia,
                inertia_ok,
                residual: relative_residual,
                residual_ok,
            });
        }
    }

    // Summary
    let n_dense_ran = n_total - n_dense_skipped_large;
    println!(
        "\nKKT summary: {} matrices ({} dense-eligible n <= {}, {} skipped n > {})",
        n_total, n_dense_ran, dense_max, n_dense_skipped_large, dense_max
    );
    println!(
        "  Inertia match: {}/{} ({:.1}%)",
        n_inertia_pass,
        n_dense_ran,
        100.0 * n_inertia_pass as f64 / n_dense_ran.max(1) as f64
    );
    println!(
        "  Residual pass: {}/{} ({:.1}%)",
        n_residual_pass,
        n_dense_ran,
        100.0 * n_residual_pass as f64 / n_dense_ran.max(1) as f64
    );
    if n_factor_fail > 0 {
        println!("  Factor failures: {}", n_factor_fail);
    }
    println!(
        "  Worst residual: {:.2e} ({})",
        worst_residual, worst_residual_name
    );

    // --- Sparse solver validation ---
    println!("\n--- Sparse solver validation ---");
    // Phase 2.2.3: the previous nemin=10000 override was added to
    // mask an amalgamation bug that claimed non-contiguous column
    // ranges as if they were contiguous supernodes. That bug is
    // fixed in commit 91e808b (adjacency check in find_supernodes),
    // so the bench now runs with the default multi-supernode
    // configuration and reports the true rate. Expect a large drop
    // from the historical 99.8% number — that number was an
    // artifact of the single-supernode override, not a real rate.
    let snode_params = SupernodeParams::default();

    let mut sp_total = 0usize;
    let mut sp_inertia_pass = 0usize;
    let mut sp_residual_pass = 0usize;
    let mut sp_factor_fail = 0usize;
    let mut sp_solve_fail = 0usize;
    let mut sp_worst_res = 0.0f64;
    let mut sp_worst_name = String::new();
    let mut sparse_failures: Vec<Failure> = Vec::new();
    // Phase 2.1.7 perf harness: sparse-side per-matrix timings. The sparse
    // factor timing is the sum of symbolic + numeric because both are part
    // of the work feral has to do for a one-shot factor call. MUMPS/SSIDS
    // oracle JSONs report a single `factor_us` that covers their equivalent
    // phases, so this matches the comparison semantics.
    let mut sparse_timings: Vec<MatrixTiming> = Vec::new();

    for entry in &kkt_entries {
        sp_total += 1;
        let n = entry.csc.n;

        let expected_inertia = Inertia {
            positive: entry.sidecar.inertia.positive,
            negative: entry.sidecar.inertia.negative,
            zero: entry.sidecar.inertia.zero,
        };

        // Symbolic factorization. Timed together with the numeric phase so
        // the reported `factor_us` is apples-to-apples with the single
        // `factor_us` field MUMPS and SSIDS emit for their equivalent work.
        let tf = Instant::now();
        let sym = match symbolic_factorize(&entry.csc, &snode_params) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("  {}: symbolic failed: {}", entry.name, e);
                sp_factor_fail += 1;
                continue;
            }
        };
        // Phase 2.8.1 partition key: the largest dense block factored by
        // the multifrontal path. Falls back to `n` if the symbolic phase
        // produced no supernodes (degenerate n=0 case).
        let sp_max_front = sym.supernodes.iter().map(|s| s.nrow).max().unwrap_or(n);

        // Numeric factorization
        let (sp_factors, sp_inertia) =
            match factorize_multifrontal(&entry.csc, &sym, &params_kkt_sparse) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("  {}: sparse factor failed: {}", entry.name, e);
                    sp_factor_fail += 1;
                    continue;
                }
            };
        let sp_factor_us = tf.elapsed().as_micros();

        let inertia_ok = sp_inertia == expected_inertia;
        if inertia_ok {
            sp_inertia_pass += 1;
        }

        // Solve
        let rhs = match entry.sidecar.finite_rhs() {
            Some(r) => r,
            None => continue,
        };
        let ts = Instant::now();
        let x = match solve_sparse_refined(&entry.csc, &sp_factors, &rhs) {
            Ok(x) => x,
            Err(e) => {
                eprintln!("  {}: sparse solve failed: {}", entry.name, e);
                sp_solve_fail += 1;
                continue;
            }
        };
        let sp_solve_us = ts.elapsed().as_micros();
        sparse_timings.push(MatrixTiming {
            name: entry.name.clone(),
            n,
            max_front: sp_max_front,
            factor_us: sp_factor_us,
            solve_us: sp_solve_us,
        });

        // Residual
        let mut ax = vec![0.0; n];
        entry.csc.symv(&x, &mut ax);
        let mut res_sq = 0.0;
        let mut b_sq = 0.0;
        for i in 0..n {
            let r = ax[i] - rhs[i];
            res_sq += r * r;
            b_sq += rhs[i] * rhs[i];
        }
        let rel_res = if b_sq > 0.0 {
            (res_sq / b_sq).sqrt()
        } else {
            res_sq.sqrt()
        };

        let tol = (n as f64) * f64::EPSILON * 1e6;
        let residual_ok = rel_res <= tol;
        if residual_ok {
            sp_residual_pass += 1;
        }
        if rel_res > sp_worst_res {
            sp_worst_res = rel_res;
            sp_worst_name = entry.name.clone();
        }

        if !inertia_ok || !residual_ok {
            sparse_failures.push(Failure {
                name: entry.name.clone(),
                n,
                expected: expected_inertia,
                actual: sp_inertia,
                inertia_ok,
                residual: rel_res,
                residual_ok,
            });
        }
    }

    println!("Sparse solver: {}/{} total", sp_total, sp_total);
    println!(
        "  Inertia match vs MUMPS: {}/{} ({:.1}%)",
        sp_inertia_pass,
        sp_total,
        100.0 * sp_inertia_pass as f64 / sp_total.max(1) as f64
    );
    println!(
        "  Residual pass: {}/{} ({:.1}%)",
        sp_residual_pass,
        sp_total,
        100.0 * sp_residual_pass as f64 / sp_total.max(1) as f64
    );
    if sp_factor_fail > 0 {
        println!("  Factor failures: {}", sp_factor_fail);
    }
    if sp_solve_fail > 0 {
        println!("  Solve failures: {}", sp_solve_fail);
    }
    println!("  Worst residual: {:.2e} ({})", sp_worst_res, sp_worst_name);

    // ============ Failure analysis ============
    print_failure_analysis("Dense", &dense_failures);
    print_failure_analysis("Sparse", &sparse_failures);
    print_cross_comparison(&dense_failures, &sparse_failures);

    // ============ Phase 2.1.7 perf vs oracles ============
    print_perf_comparison("Dense", &dense_timings, &kkt_entries);
    print_perf_comparison("Sparse", &sparse_timings, &kkt_entries);

    // ============ Phase 2.8.1 spec exit partition ============
    print_phase28_partition("Dense", &dense_timings, &kkt_entries);
    print_phase28_partition("Sparse", &sparse_timings, &kkt_entries);
}
