//! Profile FERAL's numeric multifrontal kernel on CHAINWOO_0000 to
//! identify hot-loop bottlenecks vs MUMPS/SSIDS reference timings.
//!
//! Reference (CHAINWOO_0000, METIS-ND, n=4000, nnz=7999):
//! - MUMPS: factor 726 µs, nnz_L = 51,964
//! - SSIDS: factor 3564 µs, nnz_L = 123,447
//! - feral: ~25 ms, nnz_L = 281,526
//!
//! We use the existing Phase-2.10 `Profiler` to capture per-supernode
//! timings (with prologue/epilogue split) at the multifrontal level,
//! and additionally call the dense BK kernel directly via
//! `factor_frontal_with_profile` on synthetic frontals matching the
//! observed front-size histogram so we can decompose the per-frontal
//! cost into alloc/copy, setup, pivot-loop, extract phases.
//!
//! Run:
//!     cargo run --release --bin diag_chainwoo_profile

use feral::dense::factor::{factor_frontal_with_profile, FrontalProfile};
use feral::dense::matrix::SymmetricMatrix;
use feral::numeric::factorize::{
    factorize_multifrontal_supernodal_with_workspace, FactorWorkspace, NumericParams, Profiler,
};
use feral::symbolic::{symbolic_factorize_with_method, OrderingMethod, SupernodeParams};
use feral::{read_mtx, BunchKaufmanParams};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

const N_REPS: usize = 7;

fn percentiles(vals: &mut [u64]) -> (u64, u64, u64, u64, u64) {
    if vals.is_empty() {
        return (0, 0, 0, 0, 0);
    }
    vals.sort();
    let n = vals.len();
    let idx = |p: f64| -> usize { ((n as f64 * p) as usize).min(n - 1) };
    (
        vals[0],
        vals[idx(0.5)],
        vals[idx(0.9)],
        vals[idx(0.99)],
        vals[n - 1],
    )
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "data/matrices/kkt-expansion/CHAINWOO/CHAINWOO_0000.mtx".into());
    let mtx = read_mtx(Path::new(&path)).expect("read_mtx");
    let csc = mtx.to_csc().expect("to_csc");
    println!("matrix: {}, n={}, nnz={}", path, csc.n, csc.row_idx.len());

    let snode_params = SupernodeParams::default();
    let bk = BunchKaufmanParams::default();
    let symbolic = symbolic_factorize_with_method(&csc, &snode_params, OrderingMethod::MetisND)
        .expect("metis symbolic");
    println!(
        "symbolic: {} supernodes, max_nrow {}, sum nrow*ncol {}",
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
            .map(|s| s.nrow * s.ncol)
            .sum::<usize>()
    );

    // Front-size histogram from the symbolic factorization. The "32"
    // upper bound mentioned in the task hint refers to ncol; nrow is
    // larger because trailing rows pull in fill.
    let mut nrow_hist: std::collections::BTreeMap<usize, usize> = Default::default();
    let mut ncol_hist: std::collections::BTreeMap<usize, usize> = Default::default();
    for s in &symbolic.supernodes {
        *nrow_hist.entry(s.nrow).or_default() += 1;
        *ncol_hist.entry(s.ncol).or_default() += 1;
    }
    let nrow_total: usize = symbolic.supernodes.iter().map(|s| s.nrow).sum();
    let nrow_max = *nrow_hist.keys().last().unwrap_or(&0);
    let ncol_max = *ncol_hist.keys().last().unwrap_or(&0);
    println!(
        "  supernode nrow:  min={}, mean={:.1}, max={}",
        nrow_hist.keys().next().copied().unwrap_or(0),
        nrow_total as f64 / symbolic.supernodes.len() as f64,
        nrow_max
    );
    println!(
        "  supernode ncol:  min={}, mean={:.1}, max={}",
        ncol_hist.keys().next().copied().unwrap_or(0),
        symbolic.supernodes.iter().map(|s| s.ncol).sum::<usize>() as f64
            / symbolic.supernodes.len() as f64,
        ncol_max
    );
    println!("  ncol histogram (top 10 bins):");
    let mut ncols: Vec<(usize, usize)> = ncol_hist.iter().map(|(&k, &v)| (k, v)).collect();
    ncols.sort_by_key(|&(_, v)| std::cmp::Reverse(v));
    for &(k, v) in ncols.iter().take(10) {
        println!("    ncol={:>3}: {}", k, v);
    }
    println!("  nrow histogram (top 10 bins):");
    let mut nrows: Vec<(usize, usize)> = nrow_hist.iter().map(|(&k, &v)| (k, v)).collect();
    nrows.sort_by_key(|&(_, v)| std::cmp::Reverse(v));
    for &(k, v) in nrows.iter().take(10) {
        println!("    nrow={:>3}: {}", k, v);
    }

    // ===== Phase 1: multifrontal driver level =====
    println!("\n==== Phase 1: multifrontal driver-level Profiler ====");
    let mut total_us_runs = Vec::new();
    let mut prologue_runs = Vec::new();
    let mut epilogue_runs = Vec::new();
    let mut loop_runs = Vec::new();
    // Aggregate per-supernode timings across runs
    let mut last_timings: Vec<feral::numeric::factorize::SupernodeTiming> = Vec::new();
    let mut ws = FactorWorkspace::new();
    // warm-up
    {
        let nparams = NumericParams::with_bk(bk.clone());
        let _ =
            factorize_multifrontal_supernodal_with_workspace(&csc, &symbolic, &nparams, &mut ws);
    }
    for rep in 0..N_REPS {
        let prof = Arc::new(Mutex::new(Profiler::new()));
        let mut nparams = NumericParams::with_bk(bk.clone());
        nparams.profiler = Some(prof.clone());
        let t0 = Instant::now();
        let result =
            factorize_multifrontal_supernodal_with_workspace(&csc, &symbolic, &nparams, &mut ws);
        let total_us = t0.elapsed().as_micros() as u64;
        let _ = result.expect("numeric");

        let p = prof.lock().unwrap();
        let report = p.report();
        total_us_runs.push(total_us);
        prologue_runs.push(report.prologue_us);
        epilogue_runs.push(report.epilogue_us);
        loop_runs.push(report.loop_us);

        if rep == N_REPS - 1 {
            last_timings = p.timings().to_vec();
        }
    }

    let med = |v: &[u64]| -> u64 {
        let mut x: Vec<u64> = v.to_vec();
        x.sort();
        x[x.len() / 2]
    };
    let total_med = med(&total_us_runs);
    let prologue_med = med(&prologue_runs);
    let epilogue_med = med(&epilogue_runs);
    let loop_med = med(&loop_runs);
    println!(
        "  total_us median = {}, prologue = {}, epilogue = {}, loop = {} (sum_per_snode)",
        total_med, prologue_med, epilogue_med, loop_med
    );
    println!(
        "  driver overhead (prologue+epilogue) = {} us  ({:.1}%)",
        prologue_med + epilogue_med,
        100.0 * (prologue_med + epilogue_med) as f64 / total_med.max(1) as f64
    );
    let unaccounted =
        total_med as i64 - loop_med as i64 - prologue_med as i64 - epilogue_med as i64;
    println!(
        "  driver-level unaccounted (between loop sample sites) = {} us  ({:.1}%)",
        unaccounted,
        100.0 * unaccounted as f64 / total_med.max(1) as f64
    );

    // ===== Phase 1b: per-supernode time distribution =====
    println!("\n==== Phase 1b: per-supernode timing distribution ====");
    let mut us_all: Vec<u64> = last_timings.iter().map(|t| t.us).collect();
    let (us_min, us_p50, us_p90, us_p99, us_max) = percentiles(&mut us_all);
    println!(
        "  per-snode us:  min={}, p50={}, p90={}, p99={}, max={}",
        us_min, us_p50, us_p90, us_p99, us_max
    );
    let total_loop: u64 = last_timings.iter().map(|t| t.us).sum();
    println!("  sum per-snode us = {}", total_loop);

    // Bucketed by ncol
    let mut by_ncol: std::collections::BTreeMap<usize, (usize, u64)> = Default::default();
    for t in &last_timings {
        let e = by_ncol.entry(t.ncol).or_insert((0, 0));
        e.0 += 1;
        e.1 += t.us;
    }
    println!("  per-supernode time bucketed by ncol (top by total time):");
    let mut buckets: Vec<(usize, usize, u64)> =
        by_ncol.iter().map(|(&k, &(c, s))| (k, c, s)).collect();
    buckets.sort_by_key(|&(_, _, s)| std::cmp::Reverse(s));
    for &(ncol, count, sum_us) in buckets.iter().take(12) {
        println!(
            "    ncol={:>3}  count={:>5}  sum_us={:>6}  avg_us={:>5.2}",
            ncol,
            count,
            sum_us,
            sum_us as f64 / count.max(1) as f64
        );
    }

    // Bucketed by nrow
    let mut by_nrow: std::collections::BTreeMap<usize, (usize, u64)> = Default::default();
    for t in &last_timings {
        let e = by_nrow.entry(t.nrow).or_insert((0, 0));
        e.0 += 1;
        e.1 += t.us;
    }
    println!("  per-supernode time bucketed by nrow (top by total time):");
    let mut buckets: Vec<(usize, usize, u64)> =
        by_nrow.iter().map(|(&k, &(c, s))| (k, c, s)).collect();
    buckets.sort_by_key(|&(_, _, s)| std::cmp::Reverse(s));
    for &(nrow, count, sum_us) in buckets.iter().take(12) {
        println!(
            "    nrow={:>3}  count={:>5}  sum_us={:>6}  avg_us={:>5.2}",
            nrow,
            count,
            sum_us,
            sum_us as f64 / count.max(1) as f64
        );
    }

    // ===== Phase 2: dense kernel internal breakdown via FrontalProfile =====
    // Generate one synthetic frontal at each commonly-occurring (nrow, ncol)
    // bucket and time the dense BK kernel directly.
    println!("\n==== Phase 2: dense BK kernel internal breakdown ====");
    println!(
        "  synthetic frontals at observed (nrow, ncol) buckets, calling factor_frontal_with_profile"
    );

    // Build representative buckets weighted by how many real supernodes hit them.
    let mut joint: std::collections::BTreeMap<(usize, usize), usize> = Default::default();
    for s in &symbolic.supernodes {
        *joint.entry((s.nrow, s.ncol)).or_default() += 1;
    }
    let mut joint_v: Vec<((usize, usize), usize)> = joint.into_iter().collect();
    joint_v.sort_by_key(|&(_, c)| std::cmp::Reverse(c));

    // Profile aggregate weighted by frequency for the observed distribution.
    let mut total_alloc_copy: u128 = 0;
    let mut total_setup: u128 = 0;
    let mut total_pivot: u128 = 0;
    let mut total_extract: u128 = 0;
    let mut total_calls: u64 = 0;

    println!(
        "  per-(nrow,ncol) micro-bench (median over {} reps):",
        N_REPS
    );
    println!(
        "    {:>4}{:>4}{:>6}{:>10}{:>10}{:>10}{:>10}{:>10}",
        "nrow", "ncol", "freq", "total_ns", "alloc_ns", "setup_ns", "pivot_ns", "extract_ns"
    );
    for &((nrow, ncol), freq) in joint_v.iter().take(15) {
        if nrow == 0 || ncol == 0 {
            continue;
        }
        // Build a deterministic SPD-ish symmetric test matrix
        let mat = make_test_frontal(nrow);
        let mut totals: Vec<u128> = Vec::new();
        let mut alloc_v: Vec<u128> = Vec::new();
        let mut setup_v: Vec<u128> = Vec::new();
        let mut pivot_v: Vec<u128> = Vec::new();
        let mut extract_v: Vec<u128> = Vec::new();
        for _ in 0..N_REPS {
            let mut prof = FrontalProfile::default();
            let t0 = Instant::now();
            let res = factor_frontal_with_profile(&mat, ncol, false, &bk, Some(&mut prof));
            let elapsed = t0.elapsed().as_nanos();
            let _ = res.expect("dense");
            totals.push(elapsed);
            alloc_v.push(prof.alloc_copy_ns);
            setup_v.push(prof.setup_ns);
            pivot_v.push(prof.pivot_loop_ns);
            extract_v.push(prof.extract_ns);
        }
        let pick = |mut v: Vec<u128>| -> u128 {
            v.sort();
            v[v.len() / 2]
        };
        let med_total = pick(totals);
        let med_alloc = pick(alloc_v);
        let med_setup = pick(setup_v);
        let med_pivot = pick(pivot_v);
        let med_extract = pick(extract_v);
        println!(
            "    {:>4}{:>4}{:>6}{:>10}{:>10}{:>10}{:>10}{:>10}",
            nrow, ncol, freq, med_total, med_alloc, med_setup, med_pivot, med_extract
        );
        // Weighted aggregate over the observed front distribution.
        let f = freq as u128;
        total_alloc_copy += med_alloc * f;
        total_setup += med_setup * f;
        total_pivot += med_pivot * f;
        total_extract += med_extract * f;
        total_calls += freq as u64;
    }

    if total_calls > 0 {
        println!(
            "\n  weighted aggregate over top {} buckets ({} fronts):",
            15.min(joint_v.len()),
            total_calls
        );
        let sum = total_alloc_copy + total_setup + total_pivot + total_extract;
        let pct = |x: u128| 100.0 * x as f64 / sum.max(1) as f64;
        println!(
            "    alloc_copy = {:>10} ns ({:>5.1}%)",
            total_alloc_copy,
            pct(total_alloc_copy)
        );
        println!(
            "    setup      = {:>10} ns ({:>5.1}%)",
            total_setup,
            pct(total_setup)
        );
        println!(
            "    pivot_loop = {:>10} ns ({:>5.1}%)",
            total_pivot,
            pct(total_pivot)
        );
        println!(
            "    extract    = {:>10} ns ({:>5.1}%)",
            total_extract,
            pct(total_extract)
        );
        println!(
            "    sum        = {:>10} ns = {:.1} us  (vs measured loop_us = {})",
            sum,
            sum as f64 / 1000.0,
            loop_med
        );
    }

    // ===== Phase 3: assembly cost characterization =====
    println!("\n==== Phase 3: assembly-side cost ====");
    // Time the symmetric_pattern + permute_csc_values pieces by re-running the
    // multifrontal call with a profiler and looking at the prologue split.
    println!(
        "  prologue (median over {} runs) = {} us — covers MC64/scaling, permute_csc_values, full_pattern, is_root build, contrib_blocks alloc",
        N_REPS, prologue_med
    );
    println!("  epilogue                         = {} us", epilogue_med);
    println!("  driver-level loop sum            = {} us", loop_med);
    println!(
        "  driver-level unaccounted         = {} us  (likely time spent calling Instant::now inside the loop and supernode dispatch overhead)",
        unaccounted
    );

    // Quantify the ncol=0 / 1 / 2 tail
    println!("\n==== Phase 4: tiny-front tail ====");
    let mut tiny_count = 0usize;
    let mut tiny_us = 0u64;
    for t in &last_timings {
        if t.ncol <= 2 {
            tiny_count += 1;
            tiny_us += t.us;
        }
    }
    println!(
        "  fronts with ncol<=2: count={}, sum_us={}, avg_us={:.2}",
        tiny_count,
        tiny_us,
        tiny_us as f64 / tiny_count.max(1) as f64
    );

    // ===== Phase 5: dense kernel sensitivity to BK branches =====
    // The synthetic SPD frontals above never hit 2x2 / rook / delay
    // paths. Real CHAINWOO frontals are KKT (indefinite). Re-run the
    // 32x32 timing with an indefinite saddle-point matrix and with
    // may_delay=true to see how much the BK pivot complexity costs.
    println!("\n==== Phase 5: dense kernel — indefinite vs SPD, may_delay sensitivity ====");
    {
        let n = 32;
        for label in ["spd", "kkt_indef"] {
            let mat = if label == "spd" {
                make_test_frontal(n)
            } else {
                make_kkt_frontal(n)
            };
            for &may_delay in &[false, true] {
                let mut totals: Vec<u128> = Vec::new();
                let mut alloc_v: Vec<u128> = Vec::new();
                let mut pivot_v: Vec<u128> = Vec::new();
                let mut extract_v: Vec<u128> = Vec::new();
                for _ in 0..N_REPS {
                    let mut prof = FrontalProfile::default();
                    let t0 = Instant::now();
                    let res = factor_frontal_with_profile(&mat, n, may_delay, &bk, Some(&mut prof));
                    let elapsed = t0.elapsed().as_nanos();
                    let _ = res;
                    totals.push(elapsed);
                    alloc_v.push(prof.alloc_copy_ns);
                    pivot_v.push(prof.pivot_loop_ns);
                    extract_v.push(prof.extract_ns);
                }
                let pick = |mut v: Vec<u128>| -> u128 {
                    v.sort();
                    v[v.len() / 2]
                };
                println!(
                    "    {} 32x32 may_delay={}: total={} ns,  alloc={}, pivot={}, extract={}",
                    label,
                    may_delay,
                    pick(totals),
                    pick(alloc_v),
                    pick(pivot_v),
                    pick(extract_v),
                );
            }
        }
    }

    // ===== Phase 6: quantify extend_add via FactorWorkspace assembly =====
    // We can't directly call factor_one_supernode (it's private), but we can
    // test the difference between two configurations: one warm cache (real
    // run already populated everything), and use the Profiler's per-snode
    // breakdown to see which 32x32 fronts are slowest. We already have it.
    println!("\n==== Phase 6: top-N slowest 32x32 supernodes (last-run timings) ====");
    let mut top: Vec<&feral::numeric::factorize::SupernodeTiming> = last_timings
        .iter()
        .filter(|t| t.nrow == 32 && t.ncol == 32)
        .collect();
    top.sort_by_key(|t| std::cmp::Reverse(t.us));
    for (i, t) in top.iter().take(10).enumerate() {
        let snode = &symbolic.supernodes[t.snode_idx];
        let n_children = snode.children.len();
        // estimate fan-in: sum of child contrib dims (we don't have the actual contribs here,
        // so use child nrow as proxy for contrib dim size)
        let total_child_contrib_nrow: usize = snode
            .children
            .iter()
            .map(|&c| {
                symbolic.supernodes[c]
                    .nrow
                    .saturating_sub(symbolic.supernodes[c].ncol)
            })
            .sum();
        println!(
            "    #{} snode={}: {} us, n_children={}, sum(child trailing rows)={}",
            i, t.snode_idx, t.us, n_children, total_child_contrib_nrow
        );
    }

    let n_32_with_kids: usize = symbolic
        .supernodes
        .iter()
        .filter(|s| s.nrow == 32 && s.ncol == 32 && !s.children.is_empty())
        .count();
    let total_kids: usize = symbolic
        .supernodes
        .iter()
        .filter(|s| s.nrow == 32 && s.ncol == 32)
        .map(|s| s.children.len())
        .sum();
    println!(
        "  32x32 fronts: {} have children, {} total children across all 32x32 fronts",
        n_32_with_kids, total_kids
    );

    // ===== Phase 7: actual frontal sizes (with delays) — synthetic frontals
    // assume actual_nrow == snode.nrow but the driver may inflate it via
    // delayed pivots. Check NodeFactors::nrow from a real factorization.
    println!("\n==== Phase 7: actual factored frontal sizes (incl. delays) ====");
    {
        let nparams = NumericParams::with_bk(bk.clone());
        let (factors, _) =
            feral::numeric::factorize::factorize_multifrontal_supernodal(&csc, &symbolic, &nparams)
                .expect("numeric");
        let mut size_hist: std::collections::BTreeMap<usize, usize> = Default::default();
        let mut nrow_x_ncol_total: usize = 0;
        for nf in &factors.node_factors {
            *size_hist.entry(nf.nrow).or_default() += 1;
            nrow_x_ncol_total += nf.nrow * nf.ncol;
        }
        println!("  actual frontal nrow histogram:");
        let mut pairs: Vec<(usize, usize)> = size_hist.iter().map(|(&k, &v)| (k, v)).collect();
        pairs.sort_by_key(|&(_, v)| std::cmp::Reverse(v));
        for &(nrow, count) in pairs.iter().take(10) {
            println!("    actual_nrow={:>3}: {}", nrow, count);
        }
        let largest_actual = factors
            .node_factors
            .iter()
            .map(|n| n.nrow)
            .max()
            .unwrap_or(0);
        let mean_actual = factors.node_factors.iter().map(|n| n.nrow).sum::<usize>() as f64
            / factors.node_factors.len() as f64;
        println!(
            "  largest actual frontal nrow = {}, mean actual nrow = {:.1}",
            largest_actual, mean_actual
        );
        println!("  sum nrow*ncol over all fronts = {}", nrow_x_ncol_total);

        // Now time the dense kernel for the largest actual frontal we found
        for &target_nrow in &[32usize, 64, 96, 128, 192, 256] {
            if target_nrow > largest_actual + 64 {
                break;
            }
            let mat = make_test_frontal(target_nrow);
            let mut totals: Vec<u128> = Vec::new();
            let mut pivot_v: Vec<u128> = Vec::new();
            let mut alloc_v: Vec<u128> = Vec::new();
            for _ in 0..N_REPS {
                let mut prof = FrontalProfile::default();
                let t0 = Instant::now();
                // ncol = target_nrow ⇒ all columns eliminated — same shape
                // as a root supernode; gives a worst-case BK loop time.
                let res =
                    factor_frontal_with_profile(&mat, target_nrow, false, &bk, Some(&mut prof));
                let elapsed = t0.elapsed().as_nanos();
                let _ = res;
                totals.push(elapsed);
                pivot_v.push(prof.pivot_loop_ns);
                alloc_v.push(prof.alloc_copy_ns);
            }
            let pick = |mut v: Vec<u128>| -> u128 {
                v.sort();
                v[v.len() / 2]
            };
            println!(
                "    n={:>4}: total={} ns, alloc_copy={}, pivot={}",
                target_nrow,
                pick(totals),
                pick(alloc_v),
                pick(pivot_v),
            );
        }
    }

    // ===== Phase 8: dense kernel cost with ncol << nrow (trailing dominant) =====
    println!("\n==== Phase 8: dense kernel — trailing-dominant fronts (ncol << nrow) ====");
    println!("  matches CHAINWOO actual fronts: ncol=32, nrow={{32, 67, 100, 256, 1024, 1984}}");
    for &nrow_test in &[32usize, 67, 100, 256, 1024, 1984] {
        let mat = make_test_frontal(nrow_test);
        let ncol_test = 32.min(nrow_test);
        let mut totals: Vec<u128> = Vec::new();
        let mut pivot_v: Vec<u128> = Vec::new();
        let mut alloc_v: Vec<u128> = Vec::new();
        let mut extract_v: Vec<u128> = Vec::new();
        for _ in 0..N_REPS {
            let mut prof = FrontalProfile::default();
            let t0 = Instant::now();
            let res = factor_frontal_with_profile(&mat, ncol_test, false, &bk, Some(&mut prof));
            let elapsed = t0.elapsed().as_nanos();
            let _ = res;
            totals.push(elapsed);
            pivot_v.push(prof.pivot_loop_ns);
            alloc_v.push(prof.alloc_copy_ns);
            extract_v.push(prof.extract_ns);
        }
        let pick = |mut v: Vec<u128>| -> u128 {
            v.sort();
            v[v.len() / 2]
        };
        println!(
            "    nrow={:>4} ncol={:>3}: total={:>10} ns, alloc={:>7}, pivot={:>10}, extract={:>8}",
            nrow_test,
            ncol_test,
            pick(totals),
            pick(alloc_v),
            pick(pivot_v),
            pick(extract_v),
        );
    }

    println!("\nDone.");
}

/// Construct a representative dense frontal matrix of size `n×n` with
/// well-conditioned 1×1 pivots so factor_frontal runs without delays.
/// SPD case — does not exercise BK 2×2 / rook / delay branches.
fn make_test_frontal(n: usize) -> SymmetricMatrix {
    let mut data = vec![0.0; n * n];
    for j in 0..n {
        // diagonal dominance
        data[j * n + j] = 4.0 + 0.13 * j as f64;
        for i in (j + 1)..n {
            // small off-diagonal coupling, deterministic
            let x = ((i.wrapping_mul(31) ^ j.wrapping_mul(7)) % 17) as f64 / 17.0;
            data[j * n + i] = 0.05 * (x - 0.5);
        }
    }
    SymmetricMatrix { n, data }
}

/// Construct a saddle-point KKT-style indefinite frontal: top-left
/// positive-definite block, bottom-right zero block (constraint
/// rows), strong cross coupling. This exercises BK 2×2 pivots, the
/// path that actually fires on CHAINWOO_0000.
#[allow(dead_code)]
fn make_kkt_frontal(n: usize) -> SymmetricMatrix {
    let mut data = vec![0.0; n * n];
    let n1 = n / 2;
    for j in 0..n1 {
        data[j * n + j] = 1.0 + 0.05 * j as f64;
        for i in (j + 1)..n1 {
            let x = ((i.wrapping_mul(31) ^ j.wrapping_mul(7)) % 13) as f64 / 13.0;
            data[j * n + i] = 0.02 * (x - 0.5);
        }
        // strong cross block — constraint Jacobian rows
        for i in n1..n {
            data[j * n + i] = if (i + j) % 3 == 0 { 1.0 } else { 0.0 };
        }
    }
    // bottom-right block: small regularization, near-zero diagonal
    // to force 2×2 pivots
    for j in n1..n {
        data[j * n + j] = -1e-8;
    }
    SymmetricMatrix { n, data }
}
