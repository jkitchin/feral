use std::path::Path;
use std::time::Instant;

use feral::{
    factor, read_mtx, read_sidecar, solve, BunchKaufmanParams, Inertia, KktSidecar,
    SymmetricMatrix, ZeroPivotAction,
};

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
    matrix: SymmetricMatrix,
    sidecar: KktSidecar,
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
                .filter(|e| {
                    e.path()
                        .extension()
                        .is_some_and(|ext| ext == "mtx")
                })
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

            entries.push(KktEntry {
                name: stem,
                matrix: mtx.to_dense(),
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
        println!(
            "not found (run collect_kkt from ripopt to generate)"
        );
        return;
    }
    println!("{} matrices loaded", kkt_entries.len());

    println!(
        "\n{:<30} {:>5} {:>10} {:>10} {:>14} {:>8} {:>12}",
        "name", "n", "fac(μs)", "sol(μs)", "inertia", "inertia", "residual"
    );
    println!(
        "{:<30} {:>5} {:>10} {:>10} {:>14} {:>8} {:>12}",
        "", "", "", "", "", "match?", "||Ax-b||/||b||"
    );
    println!("{}", "-".repeat(95));

    let mut n_total = 0usize;
    let mut n_inertia_pass = 0usize;
    let mut n_residual_pass = 0usize;
    let mut n_factor_fail = 0usize;
    let mut worst_residual = 0.0f64;
    let mut worst_residual_name = String::new();

    for entry in &kkt_entries {
        n_total += 1;
        let n = entry.matrix.n;

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
        let t1 = Instant::now();
        let x = match solve(&factors, &rhs) {
            Ok(x) => x,
            Err(e) => {
                eprintln!("  {}: solve failed: {}", entry.name, e);
                println!(
                    "{:<30} {:>5} {:>10} {:>10} {:>14} {:>8} {:>12}",
                    entry.name,
                    n,
                    factor_us,
                    "-",
                    format!("{}", inertia),
                    if inertia_ok { "PASS" } else { "FAIL" },
                    "solve_err"
                );
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

        if relative_residual > worst_residual {
            worst_residual = relative_residual;
            worst_residual_name = entry.name.clone();
        }

        let inertia_tag = if inertia_ok { "PASS" } else { "FAIL" };
        let res_tag = if residual_ok {
            format!("{:.2e}", relative_residual)
        } else {
            format!("{:.2e} FAIL", relative_residual)
        };

        println!(
            "{:<30} {:>5} {:>10} {:>10} {:>14} {:>8} {:>12}",
            entry.name, n, factor_us, solve_us, format!("{}", inertia), inertia_tag, res_tag
        );
    }

    // Summary
    println!("{}", "-".repeat(95));
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
}
