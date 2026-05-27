//! Synthetic Thomson-problem KKT throughput probe (issue #56).
//!
//! The Thomson problem minimizes `Σ_{i<j} 1/||p_i - p_j||` over `n`
//! points `p_i ∈ R^3` constrained to the unit sphere. The Lagrangian
//! Hessian is a dense `3n × 3n` block (all-pairs Coulomb coupling)
//! plus a diagonal `2 λ_i I` contribution from the sphere
//! multipliers. The constraint Jacobian rows are `2 p_i` placed in
//! columns `3i..3i+3`.
//!
//! We assemble the KKT system `[H A^T; A 0]` (lower-triangle CSC),
//! factor it under several FERAL configurations, and report per-config
//! wall time + `FactorStats`. Goal: localize the per-iter throughput
//! gap reported in #56 (1.57x on elec50, 2.34x on elec100 vs MUMPS).
//!
//! Run:
//!     cargo run --release --bin probe_thomson_hessian -- 50
//!     cargo run --release --bin probe_thomson_hessian -- 100
//!
//! The probe is intentionally self-contained — no GAMS / pounce
//! dependency. It uses a Fibonacci-spiral point placement and
//! lambda_i = 1 so the matrix is determined entirely by `n`.

use feral::numeric::factorize::NumericParams;
use feral::symbolic::supernode::SupernodeParams;
use feral::{BunchKaufmanParams, CscMatrix, FactorStats, FactorStatus, Solver};
use std::time::Instant;

const N_REPS: usize = 9;

/// Place `n` points on the unit sphere via a Fibonacci spiral.
/// Deterministic, well-spread, no two-point coincidences for any
/// `n >= 2`.
fn fibonacci_sphere(n: usize) -> Vec<[f64; 3]> {
    let phi = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt()); // golden angle
    (0..n)
        .map(|i| {
            // y in [-1, 1], evenly spaced
            let y = 1.0 - (i as f64) / ((n - 1).max(1) as f64) * 2.0;
            let r = (1.0 - y * y).sqrt();
            let theta = phi * (i as f64);
            [r * theta.cos(), y, r * theta.sin()]
        })
        .collect()
}

/// Build the Thomson KKT matrix at the Fibonacci-spiral configuration
/// with sphere multipliers `lambda_i = 1`. Order is `(3n + n) = 4n`:
/// the leading `3n` block is the dense Lagrangian Hessian, the
/// trailing `n` rows/cols are the sphere constraint multipliers.
///
/// The objvar / objective-definition constraint is intentionally
/// omitted (KKT order would be `4n + 2` otherwise). The dense-H ×
/// sphere-A structure is what drives factor cost; the objvar adds
/// only one zero-row and one sparse-A row.
fn build_thomson_kkt(n: usize) -> CscMatrix {
    let p = fibonacci_sphere(n);
    let nv = 3 * n; // primal variables
    let nc = n; // constraints
    let kkt_n = nv + nc;

    // Build via triplet — lower triangle only (row >= col).
    let mut rows: Vec<usize> = Vec::new();
    let mut cols: Vec<usize> = Vec::new();
    let mut vals: Vec<f64> = Vec::new();

    // Lagrangian Hessian H (3n x 3n dense block).
    // For each pair (i, j) with i != j:
    //   B_ij = (3 r r^T) / |r|^5  -  I / |r|^3,  with r = p_i - p_j
    // Block layout:
    //   H[3i..3i+3, 3i..3i+3] += sum_{j != i} B_ij
    //   H[3i..3i+3, 3j..3j+3] += -B_ij   (i != j)
    // Sphere multiplier contribution (lambda_i = 1):
    //   H[3i..3i+3, 3i..3i+3] += 2 I
    //
    // Accumulate into a dense buffer first, then emit triplets.
    let mut h = vec![0.0f64; nv * nv];
    let idx = |r: usize, c: usize| r * nv + c;

    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let r = [p[i][0] - p[j][0], p[i][1] - p[j][1], p[i][2] - p[j][2]];
            let r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
            let r_norm = r2.sqrt();
            let inv_r3 = 1.0 / (r2 * r_norm);
            let inv_r5 = inv_r3 / r2;
            // B_ij = 3*rr^T*inv_r5 - I*inv_r3
            let mut b = [[0.0f64; 3]; 3];
            for (a, b_row) in b.iter_mut().enumerate() {
                for (b_idx, b_v) in b_row.iter_mut().enumerate() {
                    *b_v = 3.0 * r[a] * r[b_idx] * inv_r5;
                }
                b_row[a] -= inv_r3;
            }
            // Diagonal block: H[3i+a, 3i+b] += B_ij[a][b]
            for a in 0..3 {
                for b_idx in 0..3 {
                    h[idx(3 * i + a, 3 * i + b_idx)] += b[a][b_idx];
                }
            }
            // Off-diagonal block (i, j): H[3i+a, 3j+b] -= B_ij[a][b].
            // Only emit when 3*i + a >= 3*j + b (lower triangle).
            for a in 0..3 {
                for b_idx in 0..3 {
                    h[idx(3 * i + a, 3 * j + b_idx)] -= b[a][b_idx];
                }
            }
        }
        // Sphere multiplier contribution
        for a in 0..3 {
            h[idx(3 * i + a, 3 * i + a)] += 2.0;
        }
    }

    // Emit H lower triangle into triplets.
    for c in 0..nv {
        for r in c..nv {
            let v = h[idx(r, c)];
            if v != 0.0 {
                rows.push(r);
                cols.push(c);
                vals.push(v);
            }
        }
    }

    // A block: sphere constraint Jacobian. Row `i` (in the KKT
    // numbering: `nv + i`) has columns `3i, 3i+1, 3i+2` with values
    // `2 p_i`. Since rows >= cols for A (the constraint rows sit
    // below the primal columns), emit directly.
    for (i, p_i) in p.iter().enumerate() {
        let row = nv + i;
        for (a, p_ia) in p_i.iter().enumerate() {
            rows.push(row);
            cols.push(3 * i + a);
            vals.push(2.0 * *p_ia);
        }
    }

    // No (2,2) block — KKT has zero block where the constraint rows
    // meet the constraint columns. CSC supports an empty column.

    let _ = nc;
    CscMatrix::from_triplets(kkt_n, &rows, &cols, &vals)
        .expect("Thomson KKT triplet construction must succeed")
}

#[derive(Default)]
struct Timing {
    factor_us: Vec<u64>,
    solve_us: Vec<u64>,
}

impl Timing {
    fn record(&mut self, factor: u64, solve: u64) {
        self.factor_us.push(factor);
        self.solve_us.push(solve);
    }
    fn summary(&self) -> (u64, u64, u64, u64) {
        // returns (factor_min, factor_med, solve_min, solve_med)
        let mut f = self.factor_us.clone();
        let mut s = self.solve_us.clone();
        f.sort();
        s.sort();
        let mid = f.len() / 2;
        (f[0], f[mid], s[0], s[mid])
    }
}

fn run_config(
    label: &str,
    matrix: &CscMatrix,
    np: NumericParams,
    sp: SupernodeParams,
    use_refined: bool,
) {
    run_config_full(label, matrix, np, sp, use_refined, true, false);
}

fn run_config_full(
    label: &str,
    matrix: &CscMatrix,
    np: NumericParams,
    sp: SupernodeParams,
    use_refined: bool,
    parallel: bool,
    cold_solver: bool,
) {
    let make_solver = || Solver::with_params(np.clone(), sp.clone()).with_parallel(parallel);
    let mut solver = make_solver();
    let rhs = vec![1.0f64; matrix.n];

    // Warm-up factor (builds symbolic, primes workspace, JITs branch
    // predictors). Not timed.
    match solver.factor(matrix, None) {
        FactorStatus::Success => {}
        FactorStatus::WrongInertia { .. } => {} // accept; inertia oracle is N/A here
        other => {
            println!("  [{}] warm-up factor failed: {:?}", label, other);
            return;
        }
    }
    let _ = solver.solve(&rhs);

    let mut t = Timing::default();
    for _ in 0..N_REPS {
        if cold_solver {
            // Simulate a fresh-solver-per-iter caller: discard all
            // cached symbolic/factor state. This is the per-iter
            // cost shape if the caller doesn't keep a Solver across
            // IPM iterations.
            solver = make_solver();
        }
        let t0 = Instant::now();
        let status = solver.factor(matrix, None);
        let factor_us = t0.elapsed().as_micros() as u64;
        if !matches!(
            status,
            FactorStatus::Success | FactorStatus::WrongInertia { .. }
        ) {
            println!("  [{}] factor failed mid-loop: {:?}", label, status);
            return;
        }
        let t1 = Instant::now();
        let _ = if use_refined {
            solver.solve_refined(matrix, &rhs)
        } else {
            solver.solve(&rhs)
        };
        let solve_us = t1.elapsed().as_micros() as u64;
        t.record(factor_us, solve_us);
    }

    let (fmin, fmed, smin, smed) = t.summary();
    let stats: FactorStats = solver
        .last_factor_stats()
        .expect("stats available post-factor");
    println!(
        "  {:32}  factor min/med = {:>7}/{:>7} µs   solve min/med = {:>6}/{:>6} µs   \
         nnz_L = {:>9}  fill = {:>5.2}  n_tiny = {:>3}  pivot_range = [{:.2e}, {:.2e}]",
        label,
        fmin,
        fmed,
        smin,
        smed,
        stats.nnz_l,
        stats.fill_ratio,
        stats.n_tiny,
        stats.min_abs_pivot,
        stats.max_abs_pivot
    );
}

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);

    let kkt = build_thomson_kkt(n);
    println!(
        "Thomson KKT: n_electrons = {}, kkt_order = {}, nnz_lower = {}, density = {:.2}%",
        n,
        kkt.n,
        kkt.nnz(),
        100.0 * (kkt.nnz() as f64) / (kkt.n as f64 * (kkt.n as f64 + 1.0) / 2.0)
    );
    println!(
        "N_REPS = {} (warm-up factor + warm-up solve excluded)\n",
        N_REPS
    );

    let sp = SupernodeParams::default();
    let bk = BunchKaufmanParams::default();

    // Phase B default: CB armed (cascade_break_ratio = Some(0.5),
    // cascade_break_eps = Some(1e-10)), FMA whatever the default is,
    // parallel on (Solver::new() routes through the parallel driver
    // when supernode count >= N_PAR_MIN).
    let np_default = NumericParams {
        bk: bk.clone(),
        ..NumericParams::default()
    };
    run_config(
        "(1) Phase B default (CB armed, FMA off)",
        &kkt,
        np_default.clone(),
        sp.clone(),
        false,
    );

    // CB disarmed — tests hypothesis (4): CB-armed per-pivot
    // bookkeeping overhead on cascade-free dense fronts.
    let np_cb_off = NumericParams {
        bk: bk.clone(),
        cascade_break_ratio: None,
        cascade_break_eps: None,
        ..NumericParams::default()
    };
    run_config("(2) CB disarmed", &kkt, np_cb_off, sp.clone(), false);

    // IR on — tests hypothesis (3): iterative refinement back-solve
    // is a significant fraction of per-iter cost.
    run_config(
        "(3) Phase B default + IR (solve_refined)",
        &kkt,
        np_default.clone(),
        sp.clone(),
        true,
    );

    // FMA on — tests hypothesis (1) at the kernel level. NumericParams.fma = true
    // dispatches the FMA-pair panel/trailing-update kernels.
    let np_fma = NumericParams {
        bk: bk.clone(),
        fma: true,
        ..NumericParams::default()
    };
    run_config(
        "(4) Phase B default + FMA on",
        &kkt,
        np_fma,
        sp.clone(),
        false,
    );

    // FMA on + CB off — combined to confirm CB-off + FMA-on is additive (or not)
    let np_fma_cb_off = NumericParams {
        bk: bk.clone(),
        fma: true,
        cascade_break_ratio: None,
        cascade_break_eps: None,
        ..NumericParams::default()
    };
    run_config(
        "(5) FMA on + CB off",
        &kkt,
        np_fma_cb_off,
        sp.clone(),
        false,
    );

    // Parallel driver off — tests hypothesis (2): rayon dispatch
    // overhead on small dense fronts where the work-stealing cost
    // exceeds the per-supernode work.
    run_config_full(
        "(6) Phase B default, parallel OFF",
        &kkt,
        np_default.clone(),
        sp.clone(),
        false,
        false,
        false,
    );

    // Cold-solver path — tests whether per-call symbolic re-analysis
    // dominates. If the gap is here, the caller side either isn't
    // caching the Solver or is invalidating the pattern fingerprint
    // every iter (the symbolic cache is keyed on pattern, not values).
    run_config_full(
        "(7) Cold solver (fresh symbolic each rep)",
        &kkt,
        np_default,
        sp,
        false,
        true,
        true,
    );
}
