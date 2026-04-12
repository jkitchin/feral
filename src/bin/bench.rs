use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::time::Instant;

use feral::numeric::factorize::factorize_multifrontal;
use feral::symbolic::{symbolic_factorize, SupernodeParams};
use feral::{
    factor, read_mtx, read_sidecar, solve, solve_refined, solve_sparse_refined, BunchKaufmanParams,
    CscMatrix, Inertia, KktSidecar, SymmetricMatrix, ZeroPivotAction,
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
    let dense_names: HashSet<&str> = dense.iter().map(|f| f.name.as_str()).collect();
    let sparse_names: HashSet<&str> = sparse.iter().map(|f| f.name.as_str()).collect();
    let both = dense_names.intersection(&sparse_names).count();
    let dense_only = dense_names.len() - both;
    let sparse_only = sparse_names.len() - both;

    println!("\n--- Dense ∩ Sparse failure overlap ---");
    println!("Failed in BOTH dense and sparse:  {}", both);
    println!("Failed in dense only:             {}", dense_only);
    println!("Failed in sparse only:            {}", sparse_only);
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

            // Skip matrices too large for the dense solver (Phase 1a)
            if mtx.n > 500 {
                continue;
            }

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

            entries.push(KktEntry {
                name: stem,
                mtx_path: mtx_path.clone(),
                matrix: mtx.to_dense(),
                csc,
                sidecar,
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
    let params_kkt = BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        ..BunchKaufmanParams::default()
    };

    let benchmarks: Vec<(&str, SymmetricMatrix, &BunchKaufmanParams)> = vec![
        ("spd_10", random_spd(10, &mut rng), &params_spd),
        ("spd_50", random_spd(50, &mut rng), &params_spd),
        ("spd_100", random_spd(100, &mut rng), &params_spd),
        ("spd_200", random_spd(200, &mut rng), &params_spd),
        ("kkt_10_3", random_kkt(10, 3, &mut rng), &params_kkt),
        ("kkt_30_10", random_kkt(30, 10, &mut rng), &params_kkt),
        ("kkt_50_15", random_kkt(50, 15, &mut rng), &params_kkt),
        ("kkt_100_30", random_kkt(100, 30, &mut rng), &params_kkt),
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

    let emit_sidecars = std::env::var("FERAL_EMIT_SIDECARS").is_ok();

    for entry in &kkt_entries {
        n_total += 1;
        let n = entry.matrix.n;
        let nnz = entry.csc.values.len();

        // Factor
        let t0 = Instant::now();
        let (factors, inertia) = match factor(&entry.matrix, &params_kkt) {
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
    println!("\nKKT summary: {}/{} total", n_total, n_total);
    println!(
        "  Inertia match: {}/{} ({:.1}%)",
        n_inertia_pass,
        n_total,
        100.0 * n_inertia_pass as f64 / n_total.max(1) as f64
    );
    println!(
        "  Residual pass: {}/{} ({:.1}%)",
        n_residual_pass,
        n_total,
        100.0 * n_residual_pass as f64 / n_total.max(1) as f64
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
    // Use large nemin to force single-supernode for correctness validation.
    // Multi-supernode solve has a known issue with contribution block assembly.
    let snode_params = SupernodeParams { nemin: 10000 };

    let mut sp_total = 0usize;
    let mut sp_inertia_pass = 0usize;
    let mut sp_residual_pass = 0usize;
    let mut sp_factor_fail = 0usize;
    let mut sp_solve_fail = 0usize;
    let mut sp_worst_res = 0.0f64;
    let mut sp_worst_name = String::new();
    let mut sparse_failures: Vec<Failure> = Vec::new();

    for entry in &kkt_entries {
        sp_total += 1;
        let n = entry.csc.n;

        let expected_inertia = Inertia {
            positive: entry.sidecar.inertia.positive,
            negative: entry.sidecar.inertia.negative,
            zero: entry.sidecar.inertia.zero,
        };

        // Symbolic factorization
        let sym = match symbolic_factorize(&entry.csc, &snode_params) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("  {}: symbolic failed: {}", entry.name, e);
                sp_factor_fail += 1;
                continue;
            }
        };

        // Numeric factorization
        let (sp_factors, sp_inertia) = match factorize_multifrontal(&entry.csc, &sym, &params_kkt) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("  {}: sparse factor failed: {}", entry.name, e);
                sp_factor_fail += 1;
                continue;
            }
        };

        let inertia_ok = sp_inertia == expected_inertia;
        if inertia_ok {
            sp_inertia_pass += 1;
        }

        // Solve
        let rhs = match entry.sidecar.finite_rhs() {
            Some(r) => r,
            None => continue,
        };
        let x = match solve_sparse_refined(&entry.csc, &sp_factors, &rhs) {
            Ok(x) => x,
            Err(e) => {
                eprintln!("  {}: sparse solve failed: {}", entry.name, e);
                sp_solve_fail += 1;
                continue;
            }
        };

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
}
