use std::path::Path;
use std::time::Instant;

use feral::{factor, solve, BunchKaufmanParams, SymmetricMatrix, ZeroPivotAction};

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
}
