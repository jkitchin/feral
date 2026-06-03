//! Reproducer for the slow Poisson optimal-control KKT factorization
//! flagged by ripopt benchmarks (Poisson 50K: 169s feral vs 0.12s
//! Ipopt+MUMPS).
//!
//! Builds the discrete 2D Poisson optimal-control KKT matrix directly
//! in lower-triangle CSC (no ripopt dependency), then runs feral's
//! symbolic + numeric multifrontal factorization with the same
//! NumericParams ripopt's `feral_direct.rs` uses today, and prints a
//! timing breakdown.
//!
//! Problem statement (matches `ripopt/benchmarks/large_scale/problems.rs::PoissonControl`):
//!   variables  : u[0..K²], f[0..K²]              (n = 2K²)
//!   constraints: -Δh u - f = 0      (5-point stencil, Dirichlet u=0 on ∂Ω)
//!                m = K²
//!   objective  : 0.5·h²·Σ(u_ij - u_d(x,y))² + (α/2)·h²·Σ f_ij²
//!                with α = 0.01, h = 1/(K+1)
//!
//! KKT layout (lower triangle):
//!   row/col 0          .. K²       — u block       diag h²
//!   row/col K²         .. 2K²      — f block       diag α·h²
//!   row     2K²        .. 3K²      — λ block       J entries (lower since rows > x cols)
//!   diag    2K²        .. 3K²      — δ_c·I        (defaults to 0; pass --delta_c to enable)
//!
//! KKT dimension n_kkt = 3K². K=158 → n_kkt = 74,892, the case where
//! `factorize_multifrontal_with_workspace` clocked at ~83 s/call in
//! the ripopt probe (98.6% of the 169 s end-to-end time).
//!
//! Run:
//!     cargo run --release --bin diag_poisson_kkt -- 50            # n_kkt = 7500    (small)
//!     cargo run --release --bin diag_poisson_kkt -- 158           # n_kkt = 74892   (the slow case)
//!     cargo run --release --bin diag_poisson_kkt -- 50 --metis    # force METIS-ND
//!     cargo run --release --bin diag_poisson_kkt -- 158 --infnorm # InfNorm scaling
//!
//! Knobs to vary (edit constants below or thread CLI flags):
//!   - OrderingMethod : Amd | Amf | MetisND | ScotchND | KahipND | Auto
//!   - ScalingStrategy: Identity | InfNorm | Mc64
//!   - SupernodeParams: relax thresholds, amalgamation
//!   - BunchKaufmanParams: pivot_threshold, on_zero_pivot, may_delay
//!
//! What ripopt currently sets (for parity with the slow case):
//!   numeric_params.scaling = ScalingStrategy::Identity
//!   numeric_params.bk.on_zero_pivot = ZeroPivotAction::ForceAccept
//!   numeric_params.bk.zero_tol      = 1e-10
//!   numeric_params.bk.pivot_threshold = NumericParams::default() (= 1e-8 in current feral)
//!   ordering = OrderingMethod::default() (Amd)

use feral::numeric::factorize::{
    factorize_multifrontal_supernodal_with_workspace, FactorWorkspace, NumericParams,
};
use feral::numeric::solve::solve_sparse;
use feral::scaling::ScalingStrategy;
use feral::symbolic::{symbolic_factorize_with_method, OrderingMethod, SupernodeParams};
use feral::{BunchKaufmanParams, CscMatrix, ZeroPivotAction};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let k: usize = args
        .iter()
        .skip(1)
        .find(|s| s.parse::<usize>().is_ok())
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);

    let mut method = OrderingMethod::Amd;
    let mut scaling = ScalingStrategy::Identity;
    let mut delta_c: f64 = 0.0;
    let mut delta_w: f64 = 0.0;
    let mut pivot_threshold: f64 = 1e-8;
    let mut nrhs: usize = 1;
    let mut reps: usize = 2;
    let mut nemin_override: Option<usize> = None;

    for arg in &args[1..] {
        match arg.as_str() {
            "--amd" => method = OrderingMethod::Amd,
            "--amf" => method = OrderingMethod::Amf,
            "--metis" => method = OrderingMethod::MetisND,
            "--scotch" => method = OrderingMethod::ScotchND,
            "--kahip" => method = OrderingMethod::KahipND,
            "--auto" => method = OrderingMethod::Auto,
            "--identity" => scaling = ScalingStrategy::Identity,
            "--infnorm" => scaling = ScalingStrategy::InfNorm,
            "--mc64" => scaling = ScalingStrategy::Mc64Symmetric,
            "--auto-scaling" => scaling = ScalingStrategy::Auto,
            s if s.starts_with("--pivtol=") => {
                pivot_threshold = s[9..].parse().unwrap_or(1e-8);
            }
            s if s.starts_with("--delta_c=") => {
                delta_c = s[10..].parse().unwrap_or(0.0);
            }
            s if s.starts_with("--delta_w=") => {
                delta_w = s[10..].parse().unwrap_or(0.0);
            }
            s if s.starts_with("--reps=") => {
                reps = s[7..].parse().unwrap_or(2);
            }
            s if s.starts_with("--nrhs=") => {
                nrhs = s[7..].parse().unwrap_or(1);
            }
            s if s.starts_with("--nemin=") => {
                nemin_override = s[8..].parse().ok();
            }
            _ => {}
        }
    }

    let n_kkt = 3 * k * k;
    let m = k * k;
    let n_x = 2 * k * k;
    let h = 1.0 / (k as f64 + 1.0);
    let alpha = 0.01;
    let inv_h2 = 1.0 / (h * h);

    eprintln!(
        "PoissonControl K={}  n_x={}  m={}  n_kkt={}  h={:.6e}  alpha={}",
        k, n_x, m, n_kkt, h, alpha
    );
    eprintln!(
        "  ordering={:?}  scaling={:?}  pivtol={}  delta_c={}  delta_w={}  reps={}  nrhs={}  nemin={:?}",
        method, scaling, pivot_threshold, delta_c, delta_w, reps, nrhs, nemin_override
    );

    // ===== Build lower-triangle triplets =====
    let t0 = Instant::now();
    let mut rows: Vec<usize> = Vec::new();
    let mut cols: Vec<usize> = Vec::new();
    let mut vals: Vec<f64> = Vec::new();

    // (1,1) Hessian block — diagonal only (separable quadratic objective).
    // u block: i ∈ [0, K²), diag h² + δ_w (positive definite)
    for i in 0..m {
        rows.push(i);
        cols.push(i);
        vals.push(h * h + delta_w);
    }
    // f block: i ∈ [K², 2K²), diag α·h² + δ_w
    for i in 0..m {
        let idx = m + i;
        rows.push(idx);
        cols.push(idx);
        vals.push(alpha * h * h + delta_w);
    }

    // (2,1) Jacobian block: constraint row = 2K² + c, c = i*K + j
    //   ∂c/∂u_{ij}    = 4/h²       (col = c)
    //   ∂c/∂u_{i±1,j} = -1/h²      (col = c ± K, if in-bounds)
    //   ∂c/∂u_{i,j±1} = -1/h²      (col = c ± 1, if in-bounds)
    //   ∂c/∂f_{ij}    = -1         (col = K² + c)
    // All entries below have row = 2K² + c > col (≤ 2K² - 1), so they
    // automatically belong to the lower triangle.
    for i in 0..k {
        for j in 0..k {
            let c = i * k + j;
            let con_row = 2 * m + c;

            // Center u
            rows.push(con_row);
            cols.push(c);
            vals.push(4.0 * inv_h2);

            // u_{i-1, j}
            if i > 0 {
                let nbr = (i - 1) * k + j;
                rows.push(con_row);
                cols.push(nbr);
                vals.push(-inv_h2);
            }
            // u_{i+1, j}
            if i + 1 < k {
                let nbr = (i + 1) * k + j;
                rows.push(con_row);
                cols.push(nbr);
                vals.push(-inv_h2);
            }
            // u_{i, j-1}
            if j > 0 {
                let nbr = i * k + (j - 1);
                rows.push(con_row);
                cols.push(nbr);
                vals.push(-inv_h2);
            }
            // u_{i, j+1}
            if j + 1 < k {
                let nbr = i * k + (j + 1);
                rows.push(con_row);
                cols.push(nbr);
                vals.push(-inv_h2);
            }

            // f coupling
            rows.push(con_row);
            cols.push(m + c);
            vals.push(-1.0);
        }
    }

    // (2,2) δ_c block — diagonal of -δ_c on constraint rows (default 0)
    if delta_c != 0.0 {
        for c in 0..m {
            let idx = 2 * m + c;
            rows.push(idx);
            cols.push(idx);
            vals.push(-delta_c);
        }
    }

    let triplet_us = t0.elapsed().as_micros();
    eprintln!(
        "build triplets: {} us  ({} entries)",
        triplet_us,
        rows.len()
    );

    // ===== CSC =====
    let t0 = Instant::now();
    let csc = match CscMatrix::from_triplets(n_kkt, &rows, &cols, &vals) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("csc build failed: {}", e);
            std::process::exit(1);
        }
    };
    let csc_us = t0.elapsed().as_micros();
    eprintln!("csc_build: {} us  (nnz_lower = {})", csc_us, csc.nnz());

    // Drop the intermediate vectors so they don't bias the heap during factor.
    drop(rows);
    drop(cols);
    drop(vals);

    // ===== Symbolic =====
    let mut snode_params = SupernodeParams::default();
    if let Some(n) = nemin_override {
        snode_params.nemin = n;
    }
    let t0 = Instant::now();
    let symbolic = match symbolic_factorize_with_method(&csc, &snode_params, method) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("symbolic failed: {}", e);
            std::process::exit(1);
        }
    };
    let symb_us = t0.elapsed().as_micros();
    let max_nrow = symbolic
        .supernodes
        .iter()
        .map(|s| s.nrow)
        .max()
        .unwrap_or(0);
    let sum_work: usize = symbolic.supernodes.iter().map(|s| s.nrow * s.ncol).sum();
    let sum_col_counts: usize = symbolic.col_counts.iter().sum();
    eprintln!(
        "symbolic: {} us  ({} supernodes, max_nrow={}, Σ nrow·ncol={}, Σ col_counts={}, factor_nnz_estimate={}, resolved_method={:?})",
        symb_us,
        symbolic.supernodes.len(),
        max_nrow,
        sum_work,
        sum_col_counts,
        symbolic.factor_nnz_estimate,
        symbolic.resolved_method
    );

    // ===== Numeric =====
    let bk = BunchKaufmanParams {
        on_zero_pivot: ZeroPivotAction::ForceAccept,
        zero_tol: 1e-10,
        zero_tol_2x2: 1e-20,
        pivot_threshold,
        ..BunchKaufmanParams::default()
    };

    let mut nparams = NumericParams::with_bk(bk.clone());
    nparams.scaling = scaling.clone();

    let mut ws = FactorWorkspace::new();

    // Warm-up + timed reps. Workspace is reused so we measure
    // amortized factor cost (matches what ripopt sees on the
    // second factor in `factor_with_inertia_correction`).
    let _ = factorize_multifrontal_supernodal_with_workspace(&csc, &symbolic, &nparams, &mut ws);

    let mut factor_us: Vec<u128> = Vec::with_capacity(reps);
    let mut last_factors_summary = (0usize, 0usize, 0usize); // (n_factor_nz, pos, neg)
    for _ in 0..reps {
        let t0 = Instant::now();
        let res =
            factorize_multifrontal_supernodal_with_workspace(&csc, &symbolic, &nparams, &mut ws);
        let us = t0.elapsed().as_micros();
        factor_us.push(us);
        match res {
            Ok((factors, inertia)) => {
                let nz = factors.factor_nnz();
                last_factors_summary = (nz, inertia.positive, inertia.negative);
                let needs_refinement = factors.needs_refinement;
                let total_delayed: usize =
                    factors.node_factors.iter().map(|nf| nf.n_delayed_in).sum();
                let max_delayed: usize = factors
                    .node_factors
                    .iter()
                    .map(|nf| nf.n_delayed_in)
                    .max()
                    .unwrap_or(0);
                let total_actual_nrow: usize = factors
                    .node_factors
                    .iter()
                    .map(|nf| nf.frontal_factors.nrow)
                    .sum();
                let total_actual_nrow_x_ncol: usize = factors
                    .node_factors
                    .iter()
                    .map(|nf| nf.frontal_factors.nrow * nf.frontal_factors.ncol)
                    .sum();
                eprintln!(
                    "  numeric: total_n_delayed_in={}  max_n_delayed_in={}  Σ actual_nrow={}  Σ actual_nrow·ncol={}",
                    total_delayed, max_delayed, total_actual_nrow, total_actual_nrow_x_ncol
                );
                // factor_nnz gap decomposition:
                //   (1) Σ col_counts                    — textbook GnP L-fill, no amalgamation
                //   (2) symbolic supernodal prediction  — col_counts grouped into supernode dense blocks
                //   (3) numeric factor_nnz              — actual storage incl. delayed pivots
                // Δ_amalgamation = (2) - (1) is supernodal padding (zeros added to share row pattern).
                // Δ_delayed      = (3) - (2) is delayed-pivot fill (rows pulled into parents).
                let sum_col_counts: usize = symbolic.col_counts.iter().sum();
                let sym_supernodal_nnz: usize = symbolic
                    .supernodes
                    .iter()
                    .map(|s| {
                        let nelim = s.ncol;
                        let trailing = s.nrow.saturating_sub(nelim) * nelim;
                        nelim * (nelim + 1) / 2 + trailing
                    })
                    .sum();
                let amalg_pad = sym_supernodal_nnz.saturating_sub(sum_col_counts);
                let delayed_fill = nz.saturating_sub(sym_supernodal_nnz);
                eprintln!(
                    "  factor_nnz breakdown: Σcc={} | sym_supernodal={} (Δ_amalg=+{}, +{:.1}%) | numeric={} (Δ_delayed=+{}, +{:.1}%)",
                    sum_col_counts,
                    sym_supernodal_nnz,
                    amalg_pad,
                    100.0 * amalg_pad as f64 / sum_col_counts.max(1) as f64,
                    nz,
                    delayed_fill,
                    100.0 * delayed_fill as f64 / sym_supernodal_nnz.max(1) as f64,
                );
                // Decompose numeric factor_nnz vs symbolic supernodal:
                //   per-supernode: Δ = padded_dense - true_L_nnz_in_own_cols
                //   where:
                //     padded_dense = nelim*(nelim+1)/2 + (nrow - nelim)*nelim   (numeric)
                //     true_L_nnz   = Σ_{k=0..own_ncol-1} col_counts[first_col + k]
                //   Δ splits into:
                //     (a) delayed-pivot inflation: nelim > own_ncol contributes own block growth + extra trailing
                //     (b) pass-through padding: rows in trailing that don't appear in own-col L
                let mut delta_delayed_only = 0i64; // sym-predicted padded - true L
                let mut delta_passthrough = 0i64; // numeric padded - sym-predicted padded
                let mut zero_delayed_passthrough = 0i64; // pass-through on nodes with n_delayed_in=0 only
                let mut zero_delayed_count = 0usize;
                for (i, nf) in factors.node_factors.iter().enumerate() {
                    let s = &symbolic.supernodes[i];
                    let own_ncol = s.ncol;
                    let first_col = s.first_col;
                    let true_l_nnz: usize = (0..own_ncol)
                        .map(|k| symbolic.col_counts[first_col + k])
                        .sum();
                    let sym_padded =
                        own_ncol * (own_ncol + 1) / 2 + s.nrow.saturating_sub(own_ncol) * own_ncol;
                    let nelim = nf.frontal_factors.nelim;
                    let nrow = nf.frontal_factors.nrow;
                    let num_padded = nelim * (nelim + 1) / 2 + nrow.saturating_sub(nelim) * nelim;
                    delta_delayed_only += sym_padded as i64 - true_l_nnz as i64;
                    delta_passthrough += num_padded as i64 - sym_padded as i64;
                    if nf.n_delayed_in == 0 {
                        zero_delayed_passthrough += num_padded as i64 - sym_padded as i64;
                        zero_delayed_count += 1;
                    }
                }
                eprintln!(
                    "  gap source: amalg_padding={:+}, passthrough_padding={:+}, of which {:+} is on n_delayed_in==0 nodes ({} of {} nodes)",
                    delta_delayed_only,
                    delta_passthrough,
                    zero_delayed_passthrough,
                    zero_delayed_count,
                    factors.node_factors.len(),
                );
                // Per-supernode comparison: symbolic.supernodes[i].nrow vs node_factors[i].frontal_factors.nrow
                // and symbolic.supernodes[i].ncol vs node_factors[i].frontal_factors.ncol.
                let mut max_nrow_growth = 0.0_f64;
                let mut max_ncol_growth_idx = 0usize;
                let mut count_nrow_grew = 0usize;
                let mut count_ncol_grew = 0usize;
                let mut total_sym_nrow = 0usize;
                let mut total_sym_ncol = 0usize;
                for (i, nf) in factors.node_factors.iter().enumerate() {
                    let s = &symbolic.supernodes[i];
                    total_sym_nrow += s.nrow;
                    total_sym_ncol += s.ncol;
                    if nf.frontal_factors.nrow > s.nrow {
                        count_nrow_grew += 1;
                        let g = nf.frontal_factors.nrow as f64 / s.nrow.max(1) as f64;
                        if g > max_nrow_growth {
                            max_nrow_growth = g;
                            max_ncol_growth_idx = i;
                        }
                    }
                    if nf.frontal_factors.ncol > s.ncol {
                        count_ncol_grew += 1;
                    }
                }
                eprintln!(
                    "  cmp: Σ sym_nrow={}, Σ sym_ncol={}, n_supernodes={}, count_nrow_grew={}, count_ncol_grew={}, max_nrow_growth={:.2}× at idx={}",
                    total_sym_nrow, total_sym_ncol, factors.node_factors.len(), count_nrow_grew, count_ncol_grew, max_nrow_growth, max_ncol_growth_idx
                );
                if let Some(nf) = factors.node_factors.get(max_ncol_growth_idx) {
                    let s = &symbolic.supernodes[max_ncol_growth_idx];
                    eprintln!(
                        "  worst-case[{}]: sym(first_col={}, ncol={}, nrow={}) vs num(ncol={}, nrow={}, n_delayed_in={}, n_children={})",
                        max_ncol_growth_idx, s.first_col, s.ncol, s.nrow, nf.frontal_factors.ncol, nf.frontal_factors.nrow, nf.n_delayed_in, s.children.len()
                    );
                    let mut row_below = 0usize;
                    let mut row_above_or_eq = 0usize;
                    for &r in &nf.row_indices {
                        if r < s.first_col {
                            row_below += 1;
                        } else {
                            row_above_or_eq += 1;
                        }
                    }
                    eprintln!(
                        "  row indices: {} < first_col, {} >= first_col (own_range=[{}..{}))",
                        row_below,
                        row_above_or_eq,
                        s.first_col,
                        s.first_col + s.ncol
                    );
                    // Sum of children's contrib trailing — should be ≈ (numeric nrow - own_ncol) at this node.
                    let mut total_child_contrib_trailing = 0usize;
                    for &c in &s.children {
                        if let Some(cb) = factors.node_factors.get(c) {
                            // contrib block was consumed; can't read it. Use frontal_factors.nrow - nelim of child.
                            let child_trailing = cb
                                .frontal_factors
                                .nrow
                                .saturating_sub(cb.frontal_factors.nelim);
                            total_child_contrib_trailing += child_trailing;
                        }
                    }
                    let nroots = symbolic
                        .supernodes
                        .iter()
                        .enumerate()
                        .filter(|(_, s2)| {
                            // s is root if no other supernode has it in children
                            !symbolic
                                .supernodes
                                .iter()
                                .any(|p| p.children.contains(&s2.first_col))
                        })
                        .count();
                    eprintln!(
                        "  root_idx_check: Σ children trailing = {}, n_roots_in_forest ≈ {}",
                        total_child_contrib_trailing, nroots
                    );
                }
                // Solve once on the last rep
                let mut rhs = vec![1.0; n_kkt];
                for (i, v) in rhs.iter_mut().enumerate() {
                    *v = 1.0 + 0.001 * (i as f64);
                }
                let t0_solve = Instant::now();
                for _ in 0..nrhs {
                    let _ = solve_sparse(&factors, &rhs);
                }
                let solve_us = t0_solve.elapsed().as_micros();
                eprintln!(
                    "  factor: {} us  factor_nnz={}  inertia=(+{}, -{}, 0={})  needs_refinement={}  solve x{}: {} us",
                    us, nz, inertia.positive, inertia.negative, n_kkt - inertia.positive - inertia.negative, needs_refinement, nrhs, solve_us
                );
            }
            Err(e) => {
                eprintln!("  factor: {} us — FAILED: {}", us, e);
            }
        }
    }

    factor_us.sort();
    let med = factor_us[factor_us.len() / 2];
    println!(
        "RESULT  K={}  n_kkt={}  ordering={:?}  scaling={:?}  pivtol={}  triplet_us={}  csc_us={}  symb_us={}  factor_med_us={}  factor_nnz={}  inertia=(+{}, -{})",
        k,
        n_kkt,
        method,
        scaling,
        pivot_threshold,
        triplet_us,
        csc_us,
        symb_us,
        med,
        last_factors_summary.0,
        last_factors_summary.1,
        last_factors_summary.2
    );
}
