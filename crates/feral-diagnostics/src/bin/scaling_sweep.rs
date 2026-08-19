//! scaling_sweep — phase-attributed scaling-law data collection.
//!
//! Step 1 of the systematic super-linear (O(n^2)) audit
//! (`dev/plans/...` / journal 2026-06-06). For each input matrix this
//! runs `factor()` with profiling enabled, forces a symbolic cache
//! miss so the symbolic phase is timed, takes the per-field median of
//! K cold factors to beat timer noise, and emits one CSV row with the
//! full per-phase breakdown plus the structural control variates
//! (`max_col_degree`, `sum_d_logd`) that distinguish "true O(n^2) in n"
//! from "this family's densest coupling column grows with n".
//!
//! The Rust side is DATA COLLECTION ONLY. The exponent fit (OLS
//! log-log per phase, leave-one-out, alpha>1.3 flag, d_max control
//! variate) is done outside in Python over the CSV.
//!
//! Usage:
//!   cargo run -p feral-diagnostics --bin scaling_sweep -- \
//!       --family rocket_12800 [--scaling mc64] [--repeats 5] [--out f.csv]
//!   cargo run -p feral-diagnostics --bin scaling_sweep -- \
//!       --manifest list.txt           # lines: mtx_path[ rhs_path(ignored)]
//!   cargo run -p feral-diagnostics --bin scaling_sweep -- \
//!       --generated spd --sizes 100,300,1000,3000,10000
//!   cargo run -p feral-diagnostics --bin scaling_sweep -- \
//!       --generated kkt --sizes 100,300,1000,3000
//!
//! Output: CSV to `--out PATH` or `$SCALING_SWEEP_DUMP`, else stdout.

use feral::scaling::{pick_scaling_strategy, ScalingStrategy};
use feral::symbolic::supernode::SupernodeParams;
use feral::{read_mtx, CscMatrix, FactorStatus, NumericParams, Solver};
use std::path::{Path, PathBuf};

/// Canonical symbolic-stage order (matches the `record_stage` call
/// order in `src/symbolic/mod.rs`). Fixing the order keeps the CSV
/// header stable across runs even though `SymbolicProfileReport.stages`
/// only contains the stages that actually ran.
const SYMBOLIC_STAGES: [&str; 17] = [
    "symmetric_pattern",
    "pick_preprocess",
    "ordering",
    "ldlt_compress",
    "compress_pattern",
    "expand_perm",
    "permute1",
    "etree_initial",
    "postorder",
    "perm_compose",
    "permute2",
    "etree_relabel",
    "col_counts",
    "renumber",
    "find_supernodes",
    "small_leaf_groups",
    "peak_contrib",
];

/// Per-phase timing + structural metrics for one factor call. All
/// timings are microseconds. `*_us == u64::MAX` is never used; missing
/// reports default to 0 so medians are well-defined.
#[derive(Clone, Default)]
struct Metrics {
    // numeric ProfileReport
    prologue_us: u64,
    pb_row_map_us: u64,
    pb_scaling_us: u64,
    pb_scaling_pivot_order_us: u64,
    pb_permute_us: u64,
    pb_permute_from_triplets_us: u64,
    pb_symmetric_pattern_us: u64,
    pb_infnorm_tol_us: u64,
    pb_setup_us: u64,
    loop_us: u64,
    epilogue_us: u64,
    total_us: u64,
    n_supernodes: u64,
    // symbolic stages, indexed by SYMBOLIC_STAGES
    sym_total_us: u64,
    sym_overhead_pct_milli: u64, // overhead_pct * 1000, integer for median
    sym_stage_us: [u64; 17],
}

fn median_u64(xs: &mut [u64]) -> u64 {
    if xs.is_empty() {
        return 0;
    }
    xs.sort_unstable();
    xs[xs.len() / 2]
}

/// Median of each field independently across the K repeats. Per-field
/// median is robust to a single slow/fast run and avoids picking one
/// "representative" run whose breakdown is internally inconsistent.
fn median_metrics(runs: &[Metrics]) -> Metrics {
    let pick = |f: fn(&Metrics) -> u64| {
        let mut v: Vec<u64> = runs.iter().map(f).collect();
        median_u64(&mut v)
    };
    let mut m = Metrics {
        prologue_us: pick(|r| r.prologue_us),
        pb_row_map_us: pick(|r| r.pb_row_map_us),
        pb_scaling_us: pick(|r| r.pb_scaling_us),
        pb_scaling_pivot_order_us: pick(|r| r.pb_scaling_pivot_order_us),
        pb_permute_us: pick(|r| r.pb_permute_us),
        pb_permute_from_triplets_us: pick(|r| r.pb_permute_from_triplets_us),
        pb_symmetric_pattern_us: pick(|r| r.pb_symmetric_pattern_us),
        pb_infnorm_tol_us: pick(|r| r.pb_infnorm_tol_us),
        pb_setup_us: pick(|r| r.pb_setup_us),
        loop_us: pick(|r| r.loop_us),
        epilogue_us: pick(|r| r.epilogue_us),
        total_us: pick(|r| r.total_us),
        n_supernodes: pick(|r| r.n_supernodes),
        sym_total_us: pick(|r| r.sym_total_us),
        sym_overhead_pct_milli: pick(|r| r.sym_overhead_pct_milli),
        sym_stage_us: [0; 17],
    };
    for s in 0..17 {
        let mut v: Vec<u64> = runs.iter().map(|r| r.sym_stage_us[s]).collect();
        m.sym_stage_us[s] = median_u64(&mut v);
    }
    m
}

/// Structural control variates computed straight off the CSC pattern
/// (cheap O(nnz)): max column degree and `sum_j d_j * ln(d_j)`. These
/// are the discriminators for the densest-column hypothesis — a phase
/// whose cost tracks `sum_d_logd` rather than `n` is "per-column sort
/// on a growing coupling column", not asymptotic-in-n.
fn structural_stats(csc: &CscMatrix) -> (usize, f64) {
    let mut max_deg = 0usize;
    let mut sum_d_logd = 0.0f64;
    for j in 0..csc.n {
        let d = csc.col_ptr[j + 1] - csc.col_ptr[j];
        if d > max_deg {
            max_deg = d;
        }
        if d > 0 {
            sum_d_logd += d as f64 * (d as f64).ln();
        }
    }
    (max_deg, sum_d_logd)
}

/// Run one cold factor (symbolic invalidated beforehand) and extract
/// the phase timings. Returns `None` if the factor failed fatally.
fn measure_one(solver: &mut Solver, csc: &CscMatrix) -> Option<Metrics> {
    solver.invalidate_symbolic_cache();
    let status = solver.factor(csc, None);
    if let FactorStatus::FatalError(_) = status {
        return None;
    }
    let mut m = Metrics::default();
    if let Some(rep) = solver.profile_report() {
        let bd = &rep.prologue_breakdown;
        m.prologue_us = rep.prologue_us;
        m.pb_row_map_us = bd.row_map_us;
        m.pb_scaling_us = bd.scaling_us;
        m.pb_scaling_pivot_order_us = bd.scaling_pivot_order_us;
        m.pb_permute_us = bd.permute_us;
        m.pb_permute_from_triplets_us = bd.permute_from_triplets_us;
        m.pb_symmetric_pattern_us = bd.symmetric_pattern_us;
        m.pb_infnorm_tol_us = bd.infnorm_tol_us;
        m.pb_setup_us = bd.setup_us;
        m.loop_us = rep.loop_us();
        m.epilogue_us = rep.epilogue_us;
        m.total_us = rep.total_us;
        m.n_supernodes = rep.n_supernodes as u64;
    }
    if let Some(srep) = solver.symbolic_profile_report() {
        m.sym_total_us = srep.total_us;
        m.sym_overhead_pct_milli = (srep.overhead_pct * 1000.0).round() as u64;
        for (i, name) in SYMBOLIC_STAGES.iter().enumerate() {
            if let Some(st) = srep.stages.iter().find(|s| s.name == *name) {
                m.sym_stage_us[i] = st.us;
            }
        }
    }
    Some(m)
}

/// Effective scaling label for the CSV: the pinned strategy, or
/// `Auto->X` recording what the picker resolves to on this matrix.
fn scaling_label(pinned: &Option<ScalingStrategy>, csc: &CscMatrix) -> String {
    match pinned {
        Some(ScalingStrategy::InfNorm) => "InfNorm".into(),
        Some(ScalingStrategy::Mc64Symmetric) => "Mc64Symmetric".into(),
        Some(ScalingStrategy::Identity) => "Identity".into(),
        Some(ScalingStrategy::External(_)) => "External".into(),
        Some(ScalingStrategy::Auto) | None => format!("Auto->{:?}", pick_scaling_strategy(csc)),
    }
}

fn csv_header() -> String {
    let mut h = String::from(
        "family,name,n,nnz,max_col_degree,sum_d_logd,scaling_strategy,\
         sym_total_us,sym_overhead_pct",
    );
    for s in SYMBOLIC_STAGES {
        h.push_str(&format!(",sym_{s}_us"));
    }
    h.push_str(
        ",prologue_us,pb_row_map_us,pb_scaling_us,pb_scaling_pivot_order_us,\
         pb_permute_us,pb_permute_from_triplets_us,pb_symmetric_pattern_us,\
         pb_infnorm_tol_us,pb_setup_us,loop_us,epilogue_us,total_us,n_supernodes",
    );
    h
}

fn csv_row(family: &str, name: &str, csc: &CscMatrix, scaling: &str, m: &Metrics) -> String {
    let (max_deg, sum_d_logd) = structural_stats(csc);
    let mut row = format!(
        "{family},{name},{},{},{max_deg},{sum_d_logd:.3},{scaling},{},{:.4}",
        csc.n,
        csc.row_idx.len(),
        m.sym_total_us,
        m.sym_overhead_pct_milli as f64 / 1000.0,
    );
    for s in 0..17 {
        row.push_str(&format!(",{}", m.sym_stage_us[s]));
    }
    row.push_str(&format!(
        ",{},{},{},{},{},{},{},{},{},{},{},{},{}",
        m.prologue_us,
        m.pb_row_map_us,
        m.pb_scaling_us,
        m.pb_scaling_pivot_order_us,
        m.pb_permute_us,
        m.pb_permute_from_triplets_us,
        m.pb_symmetric_pattern_us,
        m.pb_infnorm_tol_us,
        m.pb_setup_us,
        m.loop_us,
        m.epilogue_us,
        m.total_us,
        m.n_supernodes,
    ));
    row
}

// ---------------------------------------------------------------------
// Matrix sources
// ---------------------------------------------------------------------

/// Locate a family directory under the known corpus roots.
fn find_family_dir(name: &str) -> Option<PathBuf> {
    for root in [
        "data/matrices/kkt-mittelmann",
        "data/matrices/kkt",
        "data/matrices/kkt-expansion",
    ] {
        let p = Path::new(root).join(name);
        if p.is_dir() {
            return Some(p);
        }
    }
    None
}

/// Collect `(name, csc)` for every `.mtx` in a family directory, sorted
/// by filename so IPM-trajectory members are in iterate order.
fn load_family(dir: &Path) -> Vec<(String, CscMatrix)> {
    let mut entries: Vec<PathBuf> = match std::fs::read_dir(dir) {
        Ok(rd) => rd
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().map(|x| x == "mtx").unwrap_or(false))
            .collect(),
        Err(_) => return Vec::new(),
    };
    entries.sort();
    entries
        .iter()
        .filter_map(|p| {
            let name = p.file_stem()?.to_string_lossy().into_owned();
            let csc = read_mtx(p).ok()?.to_csc().ok()?;
            Some((name, csc))
        })
        .collect()
}

fn load_manifest(path: &Path) -> Vec<(String, CscMatrix)> {
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("cannot read manifest {}: {e}", path.display());
            return Vec::new();
        }
    };
    text.lines()
        .filter(|l| !l.trim().is_empty() && !l.trim_start().starts_with('#'))
        .filter_map(|line| {
            let mtx = line.split_whitespace().next()?;
            let p = Path::new(mtx);
            let name = p.file_stem()?.to_string_lossy().into_owned();
            let csc = read_mtx(p).ok()?.to_csc().ok()?;
            Some((name, csc))
        })
        .collect()
}

/// Deterministic LCG (no rng dependency, reproducible across runs).
struct Lcg(u64);
impl Lcg {
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn below(&mut self, bound: usize) -> usize {
        (self.next_u64() >> 33) as usize % bound.max(1)
    }
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Bandwidth for the generated families. A *constant* band keeps fill
/// near-linear in `n`, so a generated ladder isolates per-phase scaling
/// instead of being swamped by the pathological fill of a full-range
/// random pattern (which makes the numeric loop ~n^2.5, a generator
/// artifact rather than a solver scaling property).
const GEN_BAND: usize = 16;

/// Symmetric diagonally-dominant SPD matrix: `deg` random off-diagonals
/// per column drawn from a constant-width band below the diagonal, plus
/// a diagonal set to dominate. Valid SPD for any seed; both nnz and
/// fill grow linearly in n.
fn gen_spd(n: usize, deg: usize, seed: u64) -> Option<CscMatrix> {
    let mut rng = Lcg(seed);
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    let mut absrowsum = vec![0.0f64; n];
    for c in 0..n {
        let span = GEN_BAND.min(n - c - 1);
        for _ in 0..deg {
            if span == 0 {
                break;
            }
            let r = c + 1 + rng.below(span); // r in (c, c+band]
            let v = (rng.unit() - 0.5) * 2.0; // [-1, 1]
            rows.push(r);
            cols.push(c);
            vals.push(v);
            absrowsum[r] += v.abs();
            absrowsum[c] += v.abs();
        }
    }
    for (i, &s) in absrowsum.iter().enumerate() {
        rows.push(i);
        cols.push(i);
        vals.push(s + 1.0); // strict diagonal dominance
    }
    CscMatrix::from_triplets(n, &rows, &cols, &vals).ok()
}

/// Symmetric-indefinite KKT `[[H, A^T],[A, 0]]` (lower triangle): SPD
/// `H` block on `0..n_var`, dense-ish constraint rows `A` at
/// `(n_var + i, j)`, and a structurally-zero `(2,2)` block. Total
/// dimension `n_var + m`. Parametrized by a single size `s`:
/// `n_var = s`, `m = s/3`.
fn gen_kkt(s: usize, deg: usize, seed: u64) -> Option<CscMatrix> {
    let n_var = s;
    let m = (s / 3).max(1);
    let n = n_var + m;
    let mut rng = Lcg(seed);
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    let mut absrowsum = vec![0.0f64; n_var];
    // H block off-diagonals (banded lower triangle within 0..n_var).
    for c in 0..n_var {
        let span = GEN_BAND.min(n_var - c - 1);
        for _ in 0..deg {
            if span == 0 {
                break;
            }
            let r = c + 1 + rng.below(span);
            let v = (rng.unit() - 0.5) * 2.0;
            rows.push(r);
            cols.push(c);
            vals.push(v);
            absrowsum[r] += v.abs();
            absrowsum[c] += v.abs();
        }
    }
    for (i, &s) in absrowsum.iter().enumerate() {
        rows.push(i);
        cols.push(i);
        vals.push(s + 1.0);
    }
    // A block: constraint row (n_var + i) couples to deg+1 variables in
    // a local window, mirroring a discretized-PDE constraint Jacobian
    // (keeps fill near-linear; a global random A would densify the root).
    for i in 0..m {
        let r = n_var + i;
        let base = (i * n_var) / m; // spread constraints across variables
        for _ in 0..(deg + 1) {
            let span = GEN_BAND.min(n_var);
            let c = (base + rng.below(span)) % n_var;
            // r = n_var + i >= n_var > c, so always lower triangle.
            let v = (rng.unit() - 0.5) * 2.0;
            rows.push(r);
            cols.push(c);
            vals.push(v);
        }
    }
    // (2,2) block intentionally zero (genuine KKT indefiniteness).
    CscMatrix::from_triplets(n, &rows, &cols, &vals).ok()
}

// ---------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------

fn parse_scaling(s: &str) -> Option<ScalingStrategy> {
    match s.to_ascii_lowercase().as_str() {
        "mc64" | "mc64symmetric" => Some(ScalingStrategy::Mc64Symmetric),
        "infnorm" => Some(ScalingStrategy::InfNorm),
        "identity" => Some(ScalingStrategy::Identity),
        "auto" => Some(ScalingStrategy::Auto),
        _ => None,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut family: Option<String> = None;
    let mut manifest: Option<String> = None;
    let mut generated: Option<String> = None;
    let mut sizes: Vec<usize> = Vec::new();
    let mut pinned: Option<ScalingStrategy> = None;
    let mut repeats: usize = feral::env::usize_var("SCALING_SWEEP_REPEATS").unwrap_or(5);
    let mut out: Option<String> = std::env::var("SCALING_SWEEP_DUMP").ok();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--family" => {
                i += 1;
                family = args.get(i).cloned();
            }
            "--manifest" => {
                i += 1;
                manifest = args.get(i).cloned();
            }
            "--generated" => {
                i += 1;
                generated = args.get(i).cloned();
            }
            "--sizes" => {
                i += 1;
                if let Some(list) = args.get(i) {
                    sizes = list
                        .split(',')
                        .filter_map(|x| x.trim().parse().ok())
                        .collect();
                }
            }
            "--scaling" => {
                i += 1;
                if let Some(s) = args.get(i) {
                    match parse_scaling(s) {
                        Some(strat) => pinned = Some(strat),
                        None => eprintln!("unknown --scaling '{s}', ignoring"),
                    }
                }
            }
            "--repeats" => {
                i += 1;
                if let Some(k) = args.get(i).and_then(|v| v.parse().ok()) {
                    repeats = k;
                }
            }
            "--out" => {
                i += 1;
                out = args.get(i).cloned();
            }
            other => eprintln!("unknown arg '{other}', ignoring"),
        }
        i += 1;
    }
    let repeats = repeats.max(1);

    // Resolve the job list: (family_label, name, csc).
    let jobs: Vec<(String, String, CscMatrix)> = if let Some(fam) = &family {
        match find_family_dir(fam) {
            Some(dir) => load_family(&dir)
                .into_iter()
                .map(|(name, csc)| (fam.clone(), name, csc))
                .collect(),
            None => {
                eprintln!("family dir for '{fam}' not found under data/matrices/*");
                Vec::new()
            }
        }
    } else if let Some(mf) = &manifest {
        load_manifest(Path::new(mf))
            .into_iter()
            .map(|(name, csc)| ("manifest".to_string(), name, csc))
            .collect()
    } else if let Some(kind) = &generated {
        if sizes.is_empty() {
            eprintln!("--generated requires --sizes a,b,c");
            Vec::new()
        } else {
            sizes
                .iter()
                .filter_map(|&s| {
                    let csc = match kind.as_str() {
                        "spd" => gen_spd(s, 4, 0xC0FFEE),
                        "kkt" => gen_kkt(s, 4, 0xC0FFEE),
                        _ => {
                            eprintln!("unknown --generated kind '{kind}' (use spd|kkt)");
                            None
                        }
                    }?;
                    Some((format!("gen_{kind}"), format!("{kind}_{s}"), csc))
                })
                .collect()
        }
    } else {
        eprintln!(
            "usage: scaling_sweep --family NAME | --manifest FILE | \
             --generated spd|kkt --sizes a,b,c  [--scaling mc64|infnorm|identity|auto] \
             [--repeats K] [--out CSV]"
        );
        return;
    };

    if jobs.is_empty() {
        eprintln!("no matrices to sweep");
        return;
    }

    // Sink: file if requested, else stdout. Errors fall back to stdout.
    let mut lines: Vec<String> = vec![csv_header()];
    eprintln!(
        "scaling_sweep: {} matrices, repeats={repeats}, scaling={}",
        jobs.len(),
        pinned
            .as_ref()
            .map(|s| format!("{s:?}"))
            .unwrap_or_else(|| "Auto".into()),
    );

    for (fam, name, csc) in &jobs {
        let mut solver = Solver::with_params(NumericParams::default(), SupernodeParams::default())
            .with_profiling(true)
            .with_parallel(false);
        if let Some(strat) = &pinned {
            solver = solver.with_scaling(strat.clone());
        }
        // Warmup (build pools / allocations); not measured.
        let _ = measure_one(&mut solver, csc);
        let mut runs = Vec::with_capacity(repeats);
        for _ in 0..repeats {
            if let Some(m) = measure_one(&mut solver, csc) {
                runs.push(m);
            }
        }
        if runs.is_empty() {
            eprintln!("  {name}: all factors failed; skipping");
            continue;
        }
        let med = median_metrics(&runs);
        let label = scaling_label(&pinned, csc);
        let row = csv_row(fam, name, csc, &label, &med);
        eprintln!(
            "  {name:<24} n={:>7} nnz={:>8} total={:>8.1}ms prologue={:>8.1}ms \
             scaling={:>8.1}ms sym={:>7.1}ms loop={:>7.1}ms",
            csc.n,
            csc.row_idx.len(),
            med.total_us as f64 / 1e3,
            med.prologue_us as f64 / 1e3,
            med.pb_scaling_us as f64 / 1e3,
            med.sym_total_us as f64 / 1e3,
            med.loop_us as f64 / 1e3,
        );
        lines.push(row);
    }

    let body = lines.join("\n") + "\n";
    match &out {
        Some(path) => match std::fs::write(path, &body) {
            Ok(()) => eprintln!("wrote {} rows to {path}", lines.len() - 1),
            Err(e) => {
                eprintln!("failed to write {path}: {e}; dumping to stdout");
                print!("{body}");
            }
        },
        None => print!("{body}"),
    }
}
