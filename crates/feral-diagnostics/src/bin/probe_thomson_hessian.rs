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

use feral::dense::factor::{phase_timing, PHASE_TIMING_ENABLED};
use feral::numeric::factorize::{
    factorize_multifrontal_supernodal_with_workspace, FactorWorkspace, NumericParams, Profiler,
    SupernodeTiming,
};
use feral::symbolic::supernode::SupernodeParams;
use feral::symbolic::{symbolic_factorize_with_method, OrderingMethod};
use feral::{BunchKaufmanParams, CscMatrix, FactorStats, FactorStatus, Solver};
use std::sync::atomic::Ordering as AtomicOrdering;
use std::sync::{Arc, Mutex};
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
        sp.clone(),
        false,
        true,
        true,
    );

    // ===== Phase 2: per-phase breakdown of the warm factor =====
    //
    // Pounce-side data (pounce#65) confirmed the symbolic cache reuses
    // on 99.4–99.6% of per-iter factor calls, so the +50–65% cold-solver
    // gap from above does NOT explain the per-iter Thomson penalty. The
    // gap is therefore real kernel throughput at front-size 400-800.
    // Localize it: split the warm `factor()` wall into prologue +
    // per-supernode (assembly / panel / Schur / scalar-tail) + epilogue,
    // run on the sequential driver with the existing Phase 2.10
    // Profiler and the issue-#44 phase counters.
    println!();
    println!("==== Phase 2: per-phase breakdown (sequential, warm) ====");
    per_phase_breakdown(
        &kkt,
        BunchKaufmanParams::default(),
        SupernodeParams::default(),
    );

    // ===== Phase 3: per-phase breakdown via Solver =====
    //
    // Phase 2 drives the raw multifrontal driver, which bypasses
    // `Solver::mc64_scaling_cache` (issue #38 Track B2). The remaining
    // scaling cost in the Phase 2 prologue could either be (a) real
    // FERAL work that no caching layer exists for, or (b) work that
    // `Solver` already caches on warm IPM-trajectory calls. Phase 3
    // exercises the same matrix through `Solver::factor()` with
    // `with_profiling(true)` so we can read the Solver-level prologue
    // breakdown + `mc64_cache_hit_count()` and compare scaling_us
    // against Phase 2.
    println!();
    println!("==== Phase 3: per-phase breakdown via Solver (mc64_scaling_cache active) ====");
    per_phase_breakdown_via_solver(&kkt);
}

/// Drop into the sequential multifrontal driver with a Profiler attached
/// and `PHASE_TIMING_ENABLED=true`, then split the warm factor wall.
fn per_phase_breakdown(matrix: &CscMatrix, bk: BunchKaufmanParams, sp: SupernodeParams) {
    let symbolic = match symbolic_factorize_with_method(matrix, &sp, OrderingMethod::Auto) {
        Ok(s) => s,
        Err(e) => {
            println!("  symbolic_factorize_with_method failed: {:?}", e);
            return;
        }
    };
    println!(
        "  symbolic: {} supernodes, max nrow {}, max ncol {}, sum nrow*ncol {}",
        symbolic.supernodes.len(),
        symbolic
            .supernodes
            .iter()
            .map(|s| s.nrow)
            .max()
            .unwrap_or(0),
        symbolic
            .supernodes
            .iter()
            .map(|s| s.ncol)
            .max()
            .unwrap_or(0),
        symbolic
            .supernodes
            .iter()
            .map(|s| s.nrow * s.ncol)
            .sum::<usize>(),
    );

    let nparams = NumericParams::with_bk(bk);
    let mut ws = FactorWorkspace::new();

    // Warm up workspace + branch predictors (matches the warm-up done
    // by the config sweep above). Not timed.
    {
        let mut nparams_warm = nparams.clone();
        nparams_warm.profiler = None;
        let _ = factorize_multifrontal_supernodal_with_workspace(
            matrix,
            &symbolic,
            &nparams_warm,
            &mut ws,
        );
    }

    // Aggregate per-supernode phase deltas across reps. The phase
    // counters are process-global; the sequential driver snapshots
    // them before/after each `factor_one_supernode` so the per-snode
    // `assembly_us` / `panelfactor_us` / `schur_us` / `scalartail_us`
    // fields are exact for the supernode that emitted them.
    let mut sum_total: u128 = 0;
    let mut sum_prologue: u128 = 0;
    let mut sum_epilogue: u128 = 0;
    let mut sum_loop_us: u128 = 0;
    let mut sum_assembly: u128 = 0;
    let mut sum_densefactor: u128 = 0;
    let mut sum_panel: u128 = 0;
    let mut sum_schur: u128 = 0;
    let mut sum_scalartail: u128 = 0;
    let mut last_timings: Vec<SupernodeTiming> = Vec::new();
    let mut last_prologue_breakdown = feral::numeric::factorize::PrologueBreakdown::default();

    // Sub-phase counters (process-global; same accumulator each rep
    // since `phase_timing::reset()` zeros them at the top of the loop).
    let mut sum_buildrow: u128 = 0;
    let mut sum_scatter: u128 = 0;
    let mut sum_extendadd: u128 = 0;
    let mut sum_lextract: u128 = 0;
    let mut sum_contribextract: u128 = 0;
    let mut sum_contribzerofill: u128 = 0;

    PHASE_TIMING_ENABLED.store(true, AtomicOrdering::Relaxed);

    for rep in 0..N_REPS {
        phase_timing::reset();
        let prof = Arc::new(Mutex::new(Profiler::new()));
        let mut np = nparams.clone();
        np.profiler = Some(prof.clone());
        // Issue #56 Lever A.2: simulate Solver's `pattern_reused` signal
        // — the warm-up call above populated `ws.permute_cache`, so all
        // timed reps can take the cache fast path.
        np.pattern_reused_hint = true;

        let t0 = Instant::now();
        let result =
            factorize_multifrontal_supernodal_with_workspace(matrix, &symbolic, &np, &mut ws);
        let total_us = t0.elapsed().as_micros() as u64;
        if let Err(e) = result {
            println!("  numeric driver failed: {:?}", e);
            PHASE_TIMING_ENABLED.store(false, AtomicOrdering::Relaxed);
            return;
        }

        let prof_guard = prof.lock().expect("profiler poisoned");
        let report = prof_guard.report();
        let timings = prof_guard.timings();
        let loop_us: u64 = timings.iter().map(|t| t.us).sum();
        let assembly_us: u64 = timings.iter().map(|t| t.assembly_us).sum();
        let densefactor_us: u64 = timings.iter().map(|t| t.densefactor_us).sum();
        let panel_us: u64 = timings.iter().map(|t| t.panelfactor_us).sum();
        let schur_us: u64 = timings.iter().map(|t| t.schur_us).sum();
        let scalartail_us: u64 = timings.iter().map(|t| t.scalartail_us).sum();

        sum_total += total_us as u128;
        sum_prologue += report.prologue_us as u128;
        sum_epilogue += report.epilogue_us as u128;
        sum_loop_us += loop_us as u128;
        sum_assembly += assembly_us as u128;
        sum_densefactor += densefactor_us as u128;
        sum_panel += panel_us as u128;
        sum_schur += schur_us as u128;
        sum_scalartail += scalartail_us as u128;

        if rep == N_REPS - 1 {
            last_timings = timings.to_vec();
            last_prologue_breakdown = report.prologue_breakdown.clone();
        }

        // Sample process-global sub-phase counters before
        // `phase_timing::reset()` runs at the top of the next iter.
        let detail = phase_timing::snapshot_detail();
        sum_buildrow += detail.0 as u128;
        sum_scatter += detail.1 as u128;
        sum_extendadd += detail.2 as u128;
        sum_lextract += detail.3 as u128;
        sum_contribextract += detail.4 as u128;
        sum_contribzerofill += phase_timing::snapshot_contrib_zerofill() as u128;
    }

    PHASE_TIMING_ENABLED.store(false, AtomicOrdering::Relaxed);

    let n = N_REPS as u128;
    let avg = |s: u128| -> u64 { (s / n) as u64 };
    let pct = |x: u128, total: u128| -> f64 {
        if total == 0 {
            0.0
        } else {
            100.0 * (x as f64) / (total as f64)
        }
    };

    let total_avg = avg(sum_total);
    let loop_avg = avg(sum_loop_us);
    let prologue_avg = avg(sum_prologue);
    let epilogue_avg = avg(sum_epilogue);
    let assembly_avg = avg(sum_assembly);
    let densefactor_avg = avg(sum_densefactor);
    let panel_avg = avg(sum_panel);
    let schur_avg = avg(sum_schur);
    let scalartail_avg = avg(sum_scalartail);
    let densefactor_other_avg =
        densefactor_avg.saturating_sub(panel_avg + schur_avg + scalartail_avg);

    println!(
        "  averaged over {} warm reps (sequential driver, PHASE_TIMING_ENABLED):",
        N_REPS
    );
    println!("    total wall              = {:>7} µs", total_avg);
    println!(
        "    prologue                = {:>7} µs   ({:>4.1}%)",
        prologue_avg,
        pct(sum_prologue, sum_total)
    );
    {
        // Prologue sub-phase breakdown from the last rep — these
        // fields are populated when a profiler is attached.
        let pb = last_prologue_breakdown.clone();
        println!("      row_map               = {:>7} µs", pb.row_map_us);
        println!("      scaling (MC64/InfNorm)= {:>7} µs", pb.scaling_us);
        println!(
            "      scaling_pivot_order   = {:>7} µs",
            pb.scaling_pivot_order_us
        );
        println!("      permute (P A P^T)     = {:>7} µs", pb.permute_us);
        println!(
            "        from_triplets       = {:>7} µs",
            pb.permute_from_triplets_us
        );
        println!("      infnorm + tol         = {:>7} µs", pb.infnorm_tol_us);
        println!(
            "      symmetric_pattern     = {:>7} µs",
            pb.symmetric_pattern_us
        );
        println!("      setup (alloc, is_root)= {:>7} µs", pb.setup_us);
    }
    println!(
        "    epilogue                = {:>7} µs   ({:>4.1}%)",
        epilogue_avg,
        pct(sum_epilogue, sum_total)
    );
    println!(
        "    per-supernode loop sum  = {:>7} µs   ({:>4.1}%)",
        loop_avg,
        pct(sum_loop_us, sum_total)
    );
    println!(
        "      assembly              = {:>7} µs   ({:>4.1}% of loop)",
        assembly_avg,
        pct(sum_assembly, sum_loop_us)
    );
    println!(
        "      dense factor          = {:>7} µs   ({:>4.1}% of loop)",
        densefactor_avg,
        pct(sum_densefactor, sum_loop_us)
    );
    println!(
        "        panel/diag BK       = {:>7} µs   ({:>4.1}% of loop, {:>4.1}% of dense)",
        panel_avg,
        pct(sum_panel, sum_loop_us),
        pct(sum_panel, sum_densefactor)
    );
    println!(
        "        Schur trailing      = {:>7} µs   ({:>4.1}% of loop, {:>4.1}% of dense)",
        schur_avg,
        pct(sum_schur, sum_loop_us),
        pct(sum_schur, sum_densefactor)
    );
    println!(
        "        scalar tail         = {:>7} µs   ({:>4.1}% of loop, {:>4.1}% of dense)",
        scalartail_avg,
        pct(sum_scalartail, sum_loop_us),
        pct(sum_scalartail, sum_densefactor)
    );
    println!(
        "        dense bookkeeping   = {:>7} µs   (= dense - panel - schur - scalartail)",
        densefactor_other_avg
    );
    // Sub-phase drill-down — these are process-global counters in
    // nanoseconds; divide by 1000 to convert to µs, then by N_REPS.
    let ns_to_us_avg = |s: u128| -> u64 { (s / 1000 / n) as u64 };
    println!(
        "          lextract          = {:>7} µs   (subset of dense bookkeeping)",
        ns_to_us_avg(sum_lextract)
    );
    println!(
        "          contribextract    = {:>7} µs   (subset of dense bookkeeping)",
        ns_to_us_avg(sum_contribextract)
    );
    println!(
        "            zerofill        = {:>7} µs   (subset of contribextract — dead work)",
        ns_to_us_avg(sum_contribzerofill)
    );
    println!(
        "        assembly drill-down: buildrow = {:>5} µs, scatter = {:>5} µs, extendadd = {:>5} µs",
        ns_to_us_avg(sum_buildrow),
        ns_to_us_avg(sum_scatter),
        ns_to_us_avg(sum_extendadd),
    );

    // Top-3 slowest supernodes from the last rep — Thomson's KKT is
    // dominated by one wide root supernode after AMD/METIS so this
    // typically prints one giant entry + tiny tail.
    let mut top: Vec<&SupernodeTiming> = last_timings.iter().collect();
    top.sort_by_key(|t| std::cmp::Reverse(t.us));
    println!("  top supernodes by wall (last rep):");
    println!(
        "    {:>5}  {:>5}  {:>5}  {:>7}  {:>7}  {:>7}  {:>7}  {:>7}",
        "snode", "nrow", "ncol", "us", "asm_us", "panel", "schur", "tail"
    );
    for t in top.iter().take(3) {
        println!(
            "    {:>5}  {:>5}  {:>5}  {:>7}  {:>7}  {:>7}  {:>7}  {:>7}",
            t.snode_idx,
            t.nrow,
            t.ncol,
            t.us,
            t.assembly_us,
            t.panelfactor_us,
            t.schur_us,
            t.scalartail_us,
        );
    }
}

/// Phase 3: drive `factor()` through `Solver::with_profiling(true)` so the
/// per-call MC64 scaling cache (issue #38 Track B2) is exercised. Reports
/// the same prologue breakdown shape as Phase 2 plus aggregated cache hits,
/// so the scaling line can be compared head-to-head with the raw-driver path.
fn per_phase_breakdown_via_solver(matrix: &CscMatrix) {
    let mut solver = Solver::new().with_profiling(true);

    // Warm-up factor — primes symbolic cache, MC64 cache, workspace,
    // branch predictors. Not timed.
    match solver.factor(matrix, None) {
        FactorStatus::Success => {}
        FactorStatus::WrongInertia { .. } => {}
        other => {
            println!("  warm-up factor failed: {:?}", other);
            return;
        }
    }
    let baseline_hits = solver.mc64_cache_hit_count();

    let mut sum_total: u128 = 0;
    let mut sum_prologue: u128 = 0;
    let mut sum_epilogue: u128 = 0;
    let mut sum_row_map: u128 = 0;
    let mut sum_scaling: u128 = 0;
    let mut sum_scaling_pivot_order: u128 = 0;
    let mut sum_permute: u128 = 0;
    let mut sum_permute_triplets: u128 = 0;
    let mut sum_infnorm_tol: u128 = 0;
    let mut sum_sym_pattern: u128 = 0;
    let mut sum_setup: u128 = 0;
    let mut reps_with_report: usize = 0;
    let mut last_pattern_reused: Option<bool> = None;
    let mut last_scaling_info: Option<String> = None;

    PHASE_TIMING_ENABLED.store(true, AtomicOrdering::Relaxed);

    for _ in 0..N_REPS {
        let t0 = Instant::now();
        let status = solver.factor(matrix, None);
        let total_us = t0.elapsed().as_micros() as u64;
        if !matches!(
            status,
            FactorStatus::Success | FactorStatus::WrongInertia { .. }
        ) {
            println!("  factor failed mid-loop: {:?}", status);
            PHASE_TIMING_ENABLED.store(false, AtomicOrdering::Relaxed);
            return;
        }

        sum_total += total_us as u128;
        if let Some(report) = solver.profile_report() {
            // The Solver routes Thomson KKTs through the supernodal
            // driver (well above the tiny-path / dense-fast-path
            // thresholds), so `n_supernodes > 0` and `total_us > 0`
            // here. Defensive guard: only aggregate prologue counters
            // when the report actually represents a supernodal factor.
            if report.total_us > 0 {
                sum_prologue += report.prologue_us as u128;
                sum_epilogue += report.epilogue_us as u128;
                sum_row_map += report.prologue_breakdown.row_map_us as u128;
                sum_scaling += report.prologue_breakdown.scaling_us as u128;
                sum_scaling_pivot_order += report.prologue_breakdown.scaling_pivot_order_us as u128;
                sum_permute += report.prologue_breakdown.permute_us as u128;
                sum_permute_triplets += report.prologue_breakdown.permute_from_triplets_us as u128;
                sum_infnorm_tol += report.prologue_breakdown.infnorm_tol_us as u128;
                sum_sym_pattern += report.prologue_breakdown.symmetric_pattern_us as u128;
                sum_setup += report.prologue_breakdown.setup_us as u128;
                reps_with_report += 1;
            }
        }
        if let Some(stats) = solver.last_factor_stats() {
            last_pattern_reused = Some(stats.pattern_reused);
        }
        if let Some(info) = solver.scaling_info() {
            last_scaling_info = Some(format!("{:?}", info));
        }
    }

    PHASE_TIMING_ENABLED.store(false, AtomicOrdering::Relaxed);

    let total_hits = solver.mc64_cache_hit_count();
    let new_hits = total_hits.saturating_sub(baseline_hits);

    let n = N_REPS as u128;
    let avg = |s: u128| -> u64 { (s / n) as u64 };
    let avg_r = |s: u128| -> u64 {
        if reps_with_report == 0 {
            0
        } else {
            (s / reps_with_report as u128) as u64
        }
    };
    let pct = |x: u128, total: u128| -> f64 {
        if total == 0 {
            0.0
        } else {
            100.0 * (x as f64) / (total as f64)
        }
    };

    println!(
        "  averaged over {} warm reps (Solver::factor, with_profiling=true):",
        N_REPS
    );
    println!("    total wall              = {:>7} µs", avg(sum_total));
    println!(
        "    prologue                = {:>7} µs   ({:>4.1}%)",
        avg_r(sum_prologue),
        pct(sum_prologue, sum_total)
    );
    println!("      row_map               = {:>7} µs", avg_r(sum_row_map));
    println!("      scaling (MC64/InfNorm)= {:>7} µs", avg_r(sum_scaling));
    println!(
        "      scaling_pivot_order   = {:>7} µs",
        avg_r(sum_scaling_pivot_order)
    );
    println!("      permute (P A P^T)     = {:>7} µs", avg_r(sum_permute));
    println!(
        "        from_triplets       = {:>7} µs",
        avg_r(sum_permute_triplets)
    );
    println!(
        "      infnorm + tol         = {:>7} µs",
        avg_r(sum_infnorm_tol)
    );
    println!(
        "      symmetric_pattern     = {:>7} µs",
        avg_r(sum_sym_pattern)
    );
    println!("      setup (alloc, is_root)= {:>7} µs", avg_r(sum_setup));
    println!(
        "    epilogue                = {:>7} µs   ({:>4.1}%)",
        avg_r(sum_epilogue),
        pct(sum_epilogue, sum_total)
    );
    println!(
        "  mc64_cache hits across {} reps: {} (baseline {} before, {} after)",
        N_REPS, new_hits, baseline_hits, total_hits
    );
    println!(
        "  last_factor_stats.pattern_reused = {:?}, reps_with_supernodal_report = {}/{}",
        last_pattern_reused, reps_with_report, N_REPS
    );
    println!("  scaling_info = {:?}", last_scaling_info);
}
