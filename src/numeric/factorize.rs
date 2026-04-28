#[cfg(test)]
use crate::dense::factor::factor;
use crate::dense::factor::{factor_frontal_blocked_in_place, BunchKaufmanParams, FrontalFactors};
use crate::dense::matrix::SymmetricMatrix;
use crate::error::FeralError;
use crate::inertia::Inertia;
use crate::scaling::{compute_scaling, compute_scaling_with_cache, ScalingStrategy};
use crate::sparse::csc::CscMatrix;
use crate::symbolic::SymbolicFactorization;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Numeric-phase parameters bundle.
///
/// Groups the dense Bunch-Kaufman pivot configuration with the
/// global symmetric scaling strategy. Both are numeric-time
/// choices — they depend on the matrix values, not the sparsity
/// pattern. Keeping them together at the numeric entry point
/// (rather than splitting `bk` into the BK call and `scaling`
/// into the symbolic call) lets the symbolic factorization stay
/// value-agnostic and therefore reusable across multiple numeric
/// factorizations of structurally identical KKTs (the IPM use
/// case). See `dev/research/pounce-integration-interface.md` and
/// `dev/plans/scaling-in-numeric.md` (β refactor).
#[derive(Debug, Clone, Default)]
pub struct NumericParams {
    /// Dense BK kernel parameters.
    pub bk: BunchKaufmanParams,
    /// Global symmetric scaling strategy applied at the start of
    /// numeric factorization.
    pub scaling: ScalingStrategy,
    /// Phase 2.9 small-leaf-subtree batching gate. Default `Off`
    /// preserves the reference per-supernode driver. When `On` the
    /// driver processes `SymbolicFactorization::small_leaf_groups`
    /// via `factor_one_small_leaf` instead of the generic
    /// `factor_one_supernode`, skipping the per-leaf
    /// `build_row_indices` call. See
    /// `dev/plans/phase-2.9-small-leaf-subtree.md`.
    pub small_leaf: SmallLeafBatch,
    /// Phase 2.10 per-supernode profiler. When `Some`, the sequential
    /// driver records per-supernode timings, plus prologue/epilogue
    /// costs, into the shared `Profiler`. When `None` (default), no
    /// timing work runs — zero overhead in production. See
    /// `dev/plans/phase-2.10-supernode-profiler.md`.
    pub profiler: Option<Arc<Mutex<Profiler>>>,
}

/// Gate for Phase 2.9 small-leaf-subtree batching.
///
/// When `Off` (default), `factorize_multifrontal_supernodal_with_
/// workspace` runs the generic per-supernode body on every
/// supernode. When `On`, leaf supernodes that were grouped at
/// symbolic time are routed through the batched path.
///
/// Default is `Off`. Phase 2.11 attempted a default flip after a
/// single-run measurement appeared to show 24-27% reduction on the
/// tiny-IPM tail; a 5-run repeat showed the effect was within
/// ~5% measurement noise (see `dev/tried-and-rejected.md` Phase
/// 2.11). The flip was reverted; the tail gap is structural
/// (bushy elimination tree) and needs a column-renumbering
/// refactor (see `dev/plans/phase-2.12-*` once written).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SmallLeafBatch {
    #[default]
    Off,
    On,
}

/// One supernode's timing record. Phase 2.10
/// (`dev/plans/phase-2.10-supernode-profiler.md`).
#[derive(Debug, Clone, serde::Serialize)]
pub struct SupernodeTiming {
    pub snode_idx: usize,
    pub nrow: usize,
    pub ncol: usize,
    pub us: u64,
}

/// Per-invocation profiler for `factorize_multifrontal_supernodal_with_workspace`.
///
/// Attached to `NumericParams::profiler` as `Some(Arc<Mutex<Profiler>>)`
/// to record per-supernode timings, prologue and epilogue costs. When
/// the field is `None` the driver does no timing work — zero overhead.
///
/// The profiler is a diagnostic, not a correctness path. A poisoned
/// mutex (only possible if a panic happened while holding the lock,
/// which the driver code paths do not do) is silently ignored: the
/// affected sample is dropped, factorization continues, and the
/// `report()` validation invariants surface the gap.
#[derive(Debug, Clone, Default)]
pub struct Profiler {
    timings: Vec<SupernodeTiming>,
    prologue_us: u64,
    epilogue_us: u64,
    total_us: u64,
}

impl Profiler {
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of supernode timing samples recorded.
    pub fn len(&self) -> usize {
        self.timings.len()
    }

    pub fn is_empty(&self) -> bool {
        self.timings.is_empty()
    }

    /// Raw per-supernode timings in driver order.
    pub fn timings(&self) -> &[SupernodeTiming] {
        &self.timings
    }

    /// Compute the bucketed report from accumulated samples.
    pub fn report(&self) -> ProfileReport {
        const RANGES: &[(&str, usize, usize)] = &[
            ("<=8", 0, 8),
            ("9-16", 9, 16),
            ("17-32", 17, 32),
            ("33-64", 33, 64),
            ("65-128", 65, 128),
            (">128", 129, usize::MAX),
        ];

        let mut buckets: Vec<BucketStats> = RANGES
            .iter()
            .map(|&(range, _, _)| BucketStats {
                range,
                count: 0,
                sum_us: 0,
                pct_of_total: 0.0,
                avg_us: 0.0,
            })
            .collect();

        for t in &self.timings {
            for (i, &(_, lo, hi)) in RANGES.iter().enumerate() {
                if t.nrow >= lo && t.nrow <= hi {
                    buckets[i].count += 1;
                    buckets[i].sum_us += t.us;
                    break;
                }
            }
        }

        let loop_us: u64 = buckets.iter().map(|b| b.sum_us).sum();

        let mut warnings: Vec<String> = Vec::new();
        let count_sum: usize = buckets.iter().map(|b| b.count).sum();
        if count_sum != self.timings.len() {
            warnings.push(format!(
                "bucket count sum {} != timings len {}",
                count_sum,
                self.timings.len()
            ));
        }
        if self.total_us > 0 && loop_us + self.prologue_us + self.epilogue_us > self.total_us {
            warnings.push(format!(
                "loop+prologue+epilogue ({}) exceeds total ({})",
                loop_us + self.prologue_us + self.epilogue_us,
                self.total_us
            ));
        }

        for b in &mut buckets {
            if loop_us > 0 {
                b.pct_of_total = (b.sum_us as f64) * 100.0 / (loop_us as f64);
            }
            if b.count > 0 {
                b.avg_us = (b.sum_us as f64) / (b.count as f64);
            }
        }

        let overhead_pct = if self.total_us > 0 {
            ((self.prologue_us + self.epilogue_us) as f64) * 100.0 / (self.total_us as f64)
        } else {
            0.0
        };

        ProfileReport {
            n_supernodes: self.timings.len(),
            prologue_us: self.prologue_us,
            epilogue_us: self.epilogue_us,
            loop_us,
            total_us: self.total_us,
            overhead_pct,
            buckets,
            validation_warnings: warnings,
        }
    }
}

/// One front-size bucket in the profile histogram.
#[derive(Debug, Clone, serde::Serialize)]
pub struct BucketStats {
    pub range: &'static str,
    pub count: usize,
    pub sum_us: u64,
    pub pct_of_total: f64,
    pub avg_us: f64,
}

/// Aggregated profile report. Serializable for diagnostic dumps.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ProfileReport {
    pub n_supernodes: usize,
    pub prologue_us: u64,
    pub epilogue_us: u64,
    /// Sum of per-supernode timings — the inner-loop wallclock.
    pub loop_us: u64,
    /// Total wallclock for the entire driver call.
    pub total_us: u64,
    pub overhead_pct: f64,
    pub buckets: Vec<BucketStats>,
    pub validation_warnings: Vec<String>,
}

impl NumericParams {
    /// Construct a `NumericParams` from a `BunchKaufmanParams`,
    /// using the default scaling strategy. Convenience for
    /// callers that only customize BK behavior.
    pub fn with_bk(bk: BunchKaufmanParams) -> Self {
        Self {
            bk,
            scaling: ScalingStrategy::default(),
            small_leaf: SmallLeafBatch::default(),
            profiler: None,
        }
    }
}

/// Dense Schur complement block returned by
/// [`factorize_multifrontal_with_schur`] (F3.2b).
///
/// Layout: column-major full-square `dim × dim` (the `dim²` buffer is
/// dense; both upper and lower triangles are populated by mirroring the
/// computed lower triangle, per `dev/research/schur-complement.md` D5).
/// Row/column ordering matches the user-supplied `schur_indices` exactly.
///
/// The mathematical content is `S = A_SS − A_FS^T A_FF^{-1} A_FS` where
/// `A_FF` is the eliminated (non-Schur) block, `A_FS` is the coupling,
/// and `A_SS` is the Schur block. Inertia is *not* computed for `S` —
/// callers wanting an inertia-correct read of the full system must
/// account for the Schur block separately (see F3.0 D7 prominent doc).
#[derive(Debug, Clone)]
pub struct SchurBlock {
    /// Side length of the Schur block (`= schur_indices.len()`).
    pub dim: usize,
    /// `dim × dim` column-major full-square dense buffer.
    pub data: Vec<f64>,
}

impl SchurBlock {
    /// Read the `(i, j)` entry. `0 <= i, j < dim`.
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[j * self.dim + i]
    }
}

/// Stored factors from a sparse multifrontal LDL^T factorization.
#[derive(Debug)]
pub struct SparseFactors {
    /// Matrix dimension.
    pub n: usize,

    /// Fill-reducing permutation (new-to-old).
    pub perm: Vec<usize>,
    /// Inverse permutation (old-to-new).
    pub perm_inv: Vec<usize>,

    /// Per-supernode factor data. Each entry contains:
    /// - L factor columns (nrow × ncol column-major, unit diagonal implicit)
    /// - D block diagonal values (ncol entries for 1×1 blocks)
    /// - D block subdiagonal values (for 2×2 blocks)
    /// - Pivot sequence (which columns used 1×1 vs 2×2 pivots)
    /// - Row indices of the frontal matrix
    pub node_factors: Vec<NodeFactors>,

    /// Whether iterative refinement is recommended.
    pub needs_refinement: bool,

    /// Global symmetric scaling vector in **user-order** indexing.
    /// Length `n`. The matrix actually factored is `D · A · D` with
    /// `D = diag(scaling)`, so solve must pre-scale the RHS and
    /// post-scale the solution with the same vector. Cloned from
    /// `SymbolicFactorization::scaling` at the end of
    /// `factorize_multifrontal` so the solve path can reach it
    /// without a back-pointer to the symbolic analysis.
    pub scaling: Vec<f64>,

    /// Diagnostic info about how `scaling` was produced. Mirrored
    /// from `SymbolicFactorization::scaling_info` for telemetry.
    pub scaling_info: crate::scaling::ScalingInfo,

    /// Concrete fill-reducing ordering method actually used. Mirrored
    /// from `SymbolicFactorization::resolved_method`. Resolves
    /// `OrderingMethod::Auto` to the dispatched method.
    pub resolved_method: crate::symbolic::OrderingMethod,
    /// Concrete amalgamation strategy actually used. Mirrored
    /// from `SymbolicFactorization::resolved_amalgamation`.
    pub resolved_amalgamation: crate::symbolic::AmalgamationStrategy,
    /// Concrete ordering preprocessor actually used. Mirrored
    /// from `SymbolicFactorization::resolved_preprocess`.
    pub resolved_preprocess: crate::symbolic::OrderingPreprocess,
}

impl SparseFactors {
    /// One-line diagnostic summary of the strategies and pivot counts
    /// that produced these factors. Suitable for logging one record
    /// per factorization in monitoring drivers.
    ///
    /// Format:
    /// `n=<n> | <ordering> | <amalg> | preproc=<preproc> |
    ///  scaling=<scaling_info> | n_supernodes=<k> | nnz_L=<nL> |
    ///  n_2x2=<n2> | n_delayed=<nd> | inertia=(p,n,z)`
    ///
    /// Aggregated from `node_factors` so it is O(supernodes) and
    /// allocation-light. The inertia summed here equals the
    /// `Inertia` returned from `factorize_multifrontal`.
    pub fn summary(&self) -> String {
        let mut n_2x2 = 0usize;
        let mut n_delayed = 0usize;
        let mut nnz_l = 0usize;
        let mut inertia = crate::inertia::Inertia::new(0, 0, 0);
        for nf in &self.node_factors {
            let ff = &nf.frontal_factors;
            n_delayed += ff.n_delayed;
            // Match factor_nnz() accounting (lower-tri inc diag of
            // eliminated block + trailing rect).
            let trailing = ff.nrow.saturating_sub(ff.nelim) * ff.nelim;
            nnz_l += ff.nelim * (ff.nelim + 1) / 2 + trailing;
            let nelim = ff.nelim;
            let mut k = 0;
            while k < nelim {
                let two_by_two = k + 1 < nelim && ff.d_subdiag[k] != 0.0;
                if two_by_two {
                    n_2x2 += 1;
                    k += 2;
                } else {
                    k += 1;
                }
            }
            inertia.positive += nf.inertia.positive;
            inertia.negative += nf.inertia.negative;
            inertia.zero += nf.inertia.zero;
        }
        format!(
            "n={} | ord={:?} | amalg={:?} | preproc={:?} | scaling={:?} | n_supernodes={} | nnz_L={} | n_2x2={} | n_delayed={} | inertia=({},{},{})",
            self.n,
            self.resolved_method,
            self.resolved_amalgamation,
            self.resolved_preprocess,
            self.scaling_info,
            self.node_factors.len(),
            nnz_l,
            n_2x2,
            n_delayed,
            inertia.positive,
            inertia.negative,
            inertia.zero,
        )
    }

    /// Total real entries used in the L factor across all supernodes.
    ///
    /// Per supernode the L block is `nrow × nelim` column-major with
    /// unit-lower-triangular structure in the leading `nelim × nelim`
    /// eliminated block. The strict-upper triangle of that block is
    /// structurally zero and excluded from the count. The unit
    /// diagonal *is* counted.
    ///
    /// Per-supernode count:
    /// `nelim * (nelim + 1) / 2 + (nrow - nelim) * nelim`
    ///   = (eliminated lower-tri inc diagonal) + (trailing rect rows).
    ///
    /// This matches SSIDS's `inform%num_factor` accounting exactly at
    /// the median across the kkt corpus (verified by
    /// `src/bin/diag_factor_nnz_accounting.rs`). MUMPS's `INFOG(9)`
    /// uses a different accounting that includes additional entries
    /// for delayed pivots and pre-allocation; nnzL/MUMPS ratios will
    /// therefore be < 1 typically.
    ///
    /// The D entries are not counted here (`nelim + n_2x2` extra
    /// scalars; negligible for fill-ratio analysis on large fronts).
    ///
    /// Use case: fill-ratio diagnostics. `factor_nnz() / csc.nnz()` is
    /// a quick proxy for ordering quality. Values <10× on KKT-style
    /// matrices indicate a healthy ordering; values >50× suggest the
    /// resolved `OrderingMethod` is mismatched to the structure.
    pub fn factor_nnz(&self) -> usize {
        self.node_factors
            .iter()
            .map(|nf| {
                let nrow = nf.frontal_factors.nrow;
                let nelim = nf.frontal_factors.nelim;
                let trailing = nrow.saturating_sub(nelim) * nelim;
                let eliminated_lower_with_diag = nelim * (nelim + 1) / 2;
                eliminated_lower_with_diag + trailing
            })
            .sum()
    }

    /// Minimum eigenvalue of D over all eliminated pivots.
    ///
    /// 1×1 pivots contribute `d_diag[k]` directly. 2×2 blocks
    /// contribute the smaller eigenvalue of
    /// `[[d_diag[k], d_subdiag[k]], [d_subdiag[k], d_diag[k+1]]]`,
    /// computed as `(trace - sqrt(trace^2 - 4*det)) / 2`.
    ///
    /// 2×2 detection follows the solve-path convention
    /// (`src/numeric/solve.rs:217`): `d_subdiag[k] != 0.0` with the
    /// bounds check `k + 1 < nelim`.
    ///
    /// Returns `None` when no pivots were eliminated (n=0 or every
    /// supernode skipped). Used by ipopt-style unconstrained
    /// inertia correction (`-min_d + eps` as a direct delta_w).
    pub fn min_diagonal(&self) -> Option<f64> {
        let mut min_d = f64::INFINITY;
        let mut any = false;
        for nf in &self.node_factors {
            let ff = &nf.frontal_factors;
            let nelim = ff.nelim;
            let mut k = 0;
            while k < nelim {
                let two_by_two = k + 1 < nelim && ff.d_subdiag[k] != 0.0;
                let eig = if two_by_two {
                    let a = ff.d_diag[k];
                    let b = ff.d_subdiag[k];
                    let c = ff.d_diag[k + 1];
                    let trace = a + c;
                    let det = a * c - b * b;
                    let disc = (trace * trace - 4.0 * det).max(0.0).sqrt();
                    (trace - disc) * 0.5
                } else {
                    ff.d_diag[k]
                };
                if eig < min_d {
                    min_d = eig;
                }
                any = true;
                k += if two_by_two { 2 } else { 1 };
            }
        }
        if any {
            Some(min_d)
        } else {
            None
        }
    }
}

/// Factor data for a single supernode.
#[derive(Debug)]
pub struct NodeFactors {
    /// First column index (in permuted numbering).
    pub first_col: usize,
    /// Attempted column count (`snode.ncol() + n_delayed_in`). This is
    /// the `ncol` argument that was passed to `factor_frontal` and may
    /// exceed the supernode's native column count when children delayed
    /// pivots up into this node. Solve paths that iterate over
    /// eliminated columns must use `frontal_factors.nelim`, not `ncol`.
    pub ncol: usize,
    /// Number of pivots actually eliminated at this node
    /// (`ncol - n_delayed_out`). Mirror of `frontal_factors.nelim` for
    /// convenience in the solve path.
    pub nelim: usize,
    /// Number of delayed columns that entered this node from its
    /// children during parent assembly (sum of `child.contrib.n_delayed`
    /// over all children). These occupy positions
    /// `[snode.ncol() .. snode.ncol() + n_delayed_in)` of `row_indices`
    /// and are fed to `factor_frontal` as additional fully-summed
    /// columns on top of the supernode's native column count.
    pub n_delayed_in: usize,
    /// Total number of rows in the frontal.
    pub nrow: usize,
    /// Row indices of the frontal (length nrow).
    pub row_indices: Vec<usize>,
    /// The frontal factors from partial BK factorization.
    pub frontal_factors: FrontalFactors,
    /// Inertia of this node's eliminated pivots.
    pub inertia: Inertia,
}

/// Caller-owned scratch pool for sparse numeric factorization.
///
/// Reusing a single workspace across multiple calls of
/// [`factorize_multifrontal_with_workspace`] amortises per-call
/// allocation — the alloc-probe evidence in
/// `dev/research/sparse-tail-perf-2026-04-19.md` §9 shows 17–23
/// allocations per supernode, many of which are scratch buffers
/// that can be pooled.
///
/// Each field grows monotonically: the first call sizes the field
/// to what the matrix needs; subsequent calls on larger matrices
/// grow via `resize`, and subsequent calls on smaller matrices
/// reuse the existing capacity without shrinking.
///
/// The scratch buffers are NOT populated across calls — every call
/// clears them to a well-defined initial state on entry. The
/// workspace exists purely to retain heap capacity between calls,
/// not to carry data.
///
/// Invariant for `row_map`: at function entry every entry is
/// `usize::MAX`. The per-supernode loop in
/// `factorize_multifrontal_with_workspace` writes and then clears
/// exactly `row_indices.len()` entries per iteration, preserving
/// the invariant between iterations. At call entry the invariant
/// is re-established unconditionally by clearing and re-filling
/// `row_map` so prior error paths (which skip the clear) cannot
/// corrupt subsequent calls.
#[derive(Debug, Default)]
pub struct FactorWorkspace {
    /// Global→local row-index map. Length grows to `matrix.n`;
    /// entries are maintained in the all-`usize::MAX` state outside
    /// the per-supernode critical section.
    row_map: Vec<usize>,
    /// Pooled storage for the per-supernode frontal
    /// `SymmetricMatrix::data` buffer. Length resized per supernode
    /// to `nrow * nrow`; the allocation is reused across supernodes
    /// and across calls. Left empty when ownership is temporarily
    /// borrowed by an in-flight `SymmetricMatrix`.
    frontal_values: Vec<f64>,
    /// Scratch for `build_row_indices`: delayed-column globals
    /// accumulated from children of the current supernode.
    build_delayed: Vec<usize>,
    /// Scratch for `build_row_indices`: trailing (non-fully-summed)
    /// row globals for the current supernode, collected via a
    /// `build_seen`-based dedup and sorted at the end to match the
    /// pre-pool BTreeSet traversal order.
    build_trailing: Vec<usize>,
    /// Scratch for `build_row_indices`: global→`bool` membership
    /// marker. Length grows to `matrix.n`; entries are maintained
    /// in the all-`false` state outside the call (touched indices
    /// are cleared before return).
    build_seen: Vec<bool>,
    /// Pooled `n * n` f64 storage for the D.3/D.4 dense fast-path
    /// densify of the input `CscMatrix`. Reused across calls via
    /// `std::mem::take` + `CscMatrix::to_dense_into` so the
    /// fast-path no longer reallocates `n * n` doubles per call.
    /// Left empty when ownership is temporarily borrowed by an
    /// in-flight `SymmetricMatrix`.
    dense_values: Vec<f64>,
}

impl FactorWorkspace {
    /// Construct an empty workspace. Equivalent to `default()`.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Gate predicate for the D.3 dense fast-path.
///
/// Returns `true` when the input qualifies for the dense fast-path.
///
/// Two disjuncts:
///   1. **D.4 tiny-n** — `n ≤ N_TINY` unconditionally (density is
///      irrelevant; the multifrontal scaffolding cost dominates at
///      these sizes).
///   2. **D.3 small-dense** — `n ≤ N_MAX` and
///      `nnz_lower / (n * (n + 1) / 2) ≥ ρ_MIN`. The density threshold
///      is expressed as the integer inequality
///      `nnz_lower * ρ_DEN ≥ n * (n + 1) / 2 * ρ_NUM` so the check
///      costs a handful of integer ops with no division or FP.
///
/// Authoritative entry point for the gate; callers must not
/// roll their own. Thresholds may be tuned post-measurement
/// (see `dev/plans/sparse-tail-d3.md` stage 2 for D.3 and
/// `dev/plans/sparse-tail-d4.md` stage 2 for D.4).
///
/// Thresholds (`N_TINY = 16`, `N_MAX = 128`, `ρ_MIN = 1/4`) are
/// initial values from the research note
/// `dev/research/sparse-tail-d3-d4-2026-04-19.md`. Update all three
/// together if a future sweep tunes them.
#[inline]
pub fn should_use_dense_fast_path(n: usize, nnz_lower: usize) -> bool {
    // D.4 tiny-n: unconditional.
    const N_TINY: usize = 16;
    // D.3 small-dense: density-gated.
    const N_MAX: usize = 128;
    // ρ_MIN = ρ_NUM / ρ_DEN = 1/4 = 0.25
    const RHO_NUM: usize = 1;
    const RHO_DEN: usize = 4;
    if n == 0 {
        return false;
    }
    if n <= N_TINY {
        return true;
    }
    if n > N_MAX {
        return false;
    }
    let lower_cells = n * (n + 1) / 2;
    // nnz_lower / lower_cells >= RHO_NUM / RHO_DEN, i.e.
    // nnz_lower * RHO_DEN >= lower_cells * RHO_NUM.
    nnz_lower * RHO_DEN >= lower_cells * RHO_NUM
}

/// Fast-path factorization for small-and-dense matrices.
///
/// Skips symbolic analysis entirely: densifies the CSC into a
/// `SymmetricMatrix`, applies the usual global symmetric scaling,
/// runs the dense BK kernel on all `n` columns, and wraps the
/// `FrontalFactors` in a single-supernode `SparseFactors` that is
/// shape-compatible with `solve_sparse`.
///
/// Should only be called on matrices for which
/// [`should_use_dense_fast_path`] returns `true`. The production
/// dispatch path in `factorize_multifrontal_with_workspace` enforces
/// this; direct callers (tests, benches) must observe it themselves.
///
pub fn dense_fast_factor(
    matrix: &CscMatrix,
    params: &NumericParams,
) -> Result<(SparseFactors, Inertia), FeralError> {
    let mut ws = FactorWorkspace::new();
    dense_fast_factor_with_workspace(matrix, params, &mut ws)
}

/// Pooled-buffer variant of [`dense_fast_factor`].
///
/// The `n * n` dense-densify buffer is drawn from (and returned
/// to) `ws.dense_values`, so repeated calls across a single
/// `FactorWorkspace` lifetime amortise the `n * n` f64 allocation.
/// See `dev/research/phase-2.5.x-to-dense-pooling.md`.
pub fn dense_fast_factor_with_workspace(
    matrix: &CscMatrix,
    params: &NumericParams,
    ws: &mut FactorWorkspace,
) -> Result<(SparseFactors, Inertia), FeralError> {
    let n = matrix.n;
    if n == 0 {
        return Err(FeralError::InvalidInput(
            "dense_fast_factor: matrix dimension is zero".to_string(),
        ));
    }

    // Global symmetric scaling — same contract as the multifrontal
    // path. Perm is identity here so user-order == pivot-order.
    let (scaling, scaling_info) = compute_scaling(matrix, &params.scaling)?;
    if let crate::scaling::ScalingInfo::PartialSingular { n_unmatched } = &scaling_info {
        eprintln!(
            "warning: MC64 matching left {} of {} variables unmatched; \
             scaling is identity on those rows/columns",
            n_unmatched, n
        );
    }

    // Densify the CSC into a SymmetricMatrix (lower-triangle populated
    // at data[j*n + i] for i >= j) then apply D · A · D in place.
    // Pool the `n * n` buffer: hand the caller-owned Vec to
    // `to_dense_into`, use it, then return it to `ws.dense_values`
    // before falling out of the function.
    let dense_buf = std::mem::take(&mut ws.dense_values);
    let mut sym = matrix.to_dense_into(dense_buf);
    for (j, &s_j) in scaling.iter().enumerate() {
        let col = j * n;
        for (i, &s_i) in scaling.iter().enumerate().skip(j) {
            sym.data[col + i] *= s_i * s_j;
        }
    }

    // Factor the full n columns. `may_delay = false` matches the
    // multifrontal root-supernode behavior: ForceAccept absorbs any
    // unstable pivot instead of carrying it forward (there is no
    // ancestor in a single-node factorization).
    // Factor in place into `sym.data` (W-3a). `sym.data` content is
    // undefined on return, but the buffer itself is reusable; return it
    // to the pool.
    let ff = factor_frontal_blocked_in_place(&mut sym, n, false, &params.bk)?;
    ws.dense_values = sym.data;

    let inertia = ff.inertia.clone();
    let needs_refinement = ff.needs_refinement;

    // Synthesize a single-supernode SparseFactors with identity perm.
    // `solve_sparse` iterates node_factors applying each node's
    // FrontalFactors to its slice; with row_indices = 0..n and
    // perm/perm_inv identity, this reduces exactly to the dense solve.
    let perm: Vec<usize> = (0..n).collect();
    let perm_inv: Vec<usize> = (0..n).collect();
    let row_indices: Vec<usize> = (0..n).collect();

    let node = NodeFactors {
        first_col: 0,
        ncol: n,
        nelim: ff.nelim,
        n_delayed_in: 0,
        nrow: n,
        row_indices,
        frontal_factors: ff,
        inertia: inertia.clone(),
    };

    Ok((
        SparseFactors {
            n,
            perm,
            perm_inv,
            node_factors: vec![node],
            needs_refinement,
            scaling,
            scaling_info,
            // Dense fast-path skips symbolic analysis; no ordering /
            // amalgamation / preprocess actually ran. Record the
            // concrete "did-nothing" values rather than the
            // `Auto` sentinels (which are dispatch tokens, not
            // resolutions). The single-supernode
            // `node_factors.len() == 1` is the identifying signal
            // that the fast path was taken.
            resolved_method: crate::symbolic::OrderingMethod::Amd,
            resolved_amalgamation: crate::symbolic::AmalgamationStrategy::Adjacency,
            resolved_preprocess: crate::symbolic::OrderingPreprocess::None,
        },
        inertia,
    ))
}

/// Forced-supernodal variant of [`factorize_multifrontal`].
///
/// Bypasses the D.3 dense fast-path gate and runs the multifrontal
/// supernodal path regardless of input shape. Intended for test
/// oracles (the solve-parity suite in `tests/dense_fast_path.rs`)
/// that need to compare the dense-path factor against the
/// multifrontal factor on an in-gate matrix.
pub fn factorize_multifrontal_supernodal(
    matrix: &CscMatrix,
    symbolic: &SymbolicFactorization,
    params: &NumericParams,
) -> Result<(SparseFactors, Inertia), FeralError> {
    let mut ws = FactorWorkspace::new();
    factorize_multifrontal_supernodal_with_workspace(matrix, symbolic, params, &mut ws)
}

/// Perform multifrontal numeric factorization.
///
/// Takes the original sparse matrix and the symbolic factorization,
/// performs numeric factorization by traversing supernodes in postorder:
///
/// 1. Assemble original matrix entries into the frontal matrix
/// 2. Assemble child contribution blocks (extend-add)
/// 3. Factor the frontal with the dense BK kernel
/// 4. Extract the contribution block (Schur complement)
/// 5. Accumulate inertia
///
/// This entry point allocates a fresh `FactorWorkspace` on every
/// call. Callers amortising factorization across multiple
/// invocations (e.g. IPM iterations) should use
/// [`factorize_multifrontal_with_workspace`] instead and retain
/// the workspace between calls.
pub fn factorize_multifrontal(
    matrix: &CscMatrix,
    symbolic: &SymbolicFactorization,
    params: &NumericParams,
) -> Result<(SparseFactors, Inertia), FeralError> {
    let mut ws = FactorWorkspace::new();
    factorize_multifrontal_with_workspace(matrix, symbolic, params, &mut ws)
}

/// Numeric multifrontal factorization with a partial Schur extraction (F3.2b).
///
/// `symbolic` must have been produced by
/// [`crate::symbolic::symbolic_factorize_with_schur`]; otherwise this
/// returns `InvalidInput`. The matching invariant — `is_schur_tail ==
/// Some(n_schur) > 0` — is the only structural precondition.
///
/// Pipeline divergence from [`factorize_multifrontal`]:
///
/// 1. Per-supernode `nvschur[s]` is computed from `is_schur_tail` and
///    the supernode column ranges. Only supernodes whose column range
///    intersects `[n - n_schur, n)` have `nvschur > 0`. Those
///    supernodes are necessarily root(s) of the etree post-F3.2a (Schur
///    columns occupy the highest etree-index positions).
///
/// 2. At each Schur-bearing root, the Bunch-Kaufman pivot loop
///    eliminates only `expanded_ncol − nvschur` columns; the remaining
///    `nvschur` Schur columns end up un-eliminated in the contribution
///    block at positions `[0, nvschur) × [0, nvschur)` (col-major
///    lower-triangle dense). This matches MUMPS
///    `dfac_front_LDLT_type1.F:193-205`'s `NPIV ≤ NASS − NVSCHUR`
///    bound (see dev/research/schur-complement.md D4-D6).
///
/// 3. After the postorder loop, the dense `n_schur × n_schur` Schur
///    block is read out of the root supernode's `ContribBlock`,
///    mirrored lower→upper, and returned as a [`SchurBlock`].
///
/// **Constraint**: the Schur columns must form a single contiguous tail
/// and the tail-bearing supernode must be a single root whose last
/// column is at position `n - 1`. The F3.2a symbolic pipeline now
/// guarantees the single-supernode invariant by force-merging any
/// Schur-bearing supernodes before this entry point sees them
/// (see [`crate::symbolic::symbolic_factorize_with_schur`] step 8b,
/// mirroring MUMPS's HALO-SCHUR amalgamation in
/// `ana_orderings.F:9187-9220`). Forest-structured Schur sets (the
/// matrix has multiple connected components, each contributing a Schur
/// root) are rejected at the symbolic phase with `InvalidInput`.
///
/// Returned `Inertia` reflects the inertia of the *eliminated* block
/// `A_FF` only — the Schur block's spectrum is not factored.
pub fn factorize_multifrontal_with_schur(
    matrix: &CscMatrix,
    symbolic: &SymbolicFactorization,
    params: &NumericParams,
) -> Result<(SparseFactors, Inertia, SchurBlock), FeralError> {
    let n_schur = symbolic.is_schur_tail.ok_or_else(|| {
        FeralError::InvalidInput(
            "factorize_multifrontal_with_schur requires symbolic produced by \
             symbolic_factorize_with_schur (is_schur_tail is None)"
                .to_string(),
        )
    })?;
    if n_schur == 0 {
        return Err(FeralError::InvalidInput(
            "is_schur_tail = Some(0); use factorize_multifrontal instead".to_string(),
        ));
    }

    let n = symbolic.n;
    let n_snodes = symbolic.supernodes.len();

    // Per-supernode nvschur. Schur columns occupy global perm positions
    // [n - n_schur, n), so a supernode's nvschur is the size of its
    // column-range intersection with that interval. The
    // F3.2a postorder pins these positions to the tail of the supernode
    // sequence, so only the last contiguous run of supernodes has
    // nvschur > 0.
    let mut nvschur_per_snode = vec![0usize; n_snodes];
    let schur_lo = n - n_schur;
    for (s, snode) in symbolic.supernodes.iter().enumerate() {
        let col_lo = snode.first_col;
        let col_hi = col_lo + snode.ncol();
        if col_hi <= schur_lo || col_lo >= n {
            continue;
        }
        let lo = col_lo.max(schur_lo);
        let hi = col_hi.min(n);
        nvschur_per_snode[s] = hi - lo;
    }

    // F3.2b scope guard: require the Schur tail to live entirely in
    // one supernode whose last column is at position n - 1. Multi-
    // supernode Schur tails are deferred to F3.3.
    let last_snode = n_snodes
        .checked_sub(1)
        .ok_or_else(|| FeralError::InvalidInput("symbolic has zero supernodes".to_string()))?;
    let last = &symbolic.supernodes[last_snode];
    if last.first_col + last.ncol() != n {
        return Err(FeralError::InvalidInput(
            "Schur path expects last supernode to end at n-1".to_string(),
        ));
    }
    if nvschur_per_snode[last_snode] != n_schur {
        return Err(FeralError::InvalidInput(format!(
            "F3.2b scope: Schur tail must lie in a single root supernode \
             (last snode covers {} of {} Schur columns); see \
             dev/research/schur-complement.md F3.3",
            nvschur_per_snode[last_snode], n_schur
        )));
    }
    for &k in &nvschur_per_snode[..last_snode] {
        debug_assert_eq!(k, 0, "nvschur > 0 outside last snode violates F3.2b scope");
    }

    let mut ws = FactorWorkspace::new();
    factorize_multifrontal_with_schur_inner(matrix, symbolic, params, &mut ws, &nvschur_per_snode)
}

/// F3.2b inner driver: a Schur-aware specialization of
/// [`factorize_multifrontal_supernodal_with_workspace`]. Sequential.
/// Skips the dense fast-path (incompatible with partial elimination)
/// and the small-leaf batch path (leaves cannot be Schur-bearing under
/// the F3.2a layout, but we route everything through the generic
/// `factor_one_supernode` to keep the nvschur threading explicit).
fn factorize_multifrontal_with_schur_inner(
    matrix: &CscMatrix,
    symbolic: &SymbolicFactorization,
    params: &NumericParams,
    ws: &mut FactorWorkspace,
    nvschur_per_snode: &[usize],
) -> Result<(SparseFactors, Inertia, SchurBlock), FeralError> {
    let n = symbolic.n;
    let n_snodes = symbolic.supernodes.len();

    ws.row_map.clear();
    ws.row_map.resize(n, usize::MAX);

    let (scaling_user, scaling_info) = crate::scaling::compute_scaling(matrix, &params.scaling)?;
    let scaling_pivot_order: Vec<f64> =
        symbolic.perm.iter().map(|&old| scaling_user[old]).collect();

    let permuted = permute_csc_values(matrix, &symbolic.perm, &symbolic.perm_inv)?;
    let full_pattern = permuted.symmetric_pattern();

    let mut is_root = vec![true; n_snodes];
    for snode in &symbolic.supernodes {
        for &child_idx in &snode.children {
            if child_idx < n_snodes {
                is_root[child_idx] = false;
            }
        }
    }

    let mut contrib_blocks: Vec<Option<ContribBlock>> = (0..n_snodes).map(|_| None).collect();
    let mut node_factors: Vec<NodeFactors> = Vec::with_capacity(n_snodes);
    let mut total_inertia = Inertia {
        positive: 0,
        negative: 0,
        zero: 0,
    };
    let mut needs_refinement = false;

    for (snode_idx, &nvschur) in nvschur_per_snode.iter().enumerate() {
        let node = factor_one_supernode(
            snode_idx,
            symbolic,
            &permuted,
            &full_pattern,
            &scaling_pivot_order,
            &is_root,
            params,
            ws,
            &mut contrib_blocks,
            nvschur,
        )?;
        total_inertia.positive += node.inertia.positive;
        total_inertia.negative += node.inertia.negative;
        total_inertia.zero += node.inertia.zero;
        if node.frontal_factors.needs_refinement {
            needs_refinement = true;
        }
        node_factors.push(node);
    }

    // Extract the dense Schur block from the last (Schur-bearing) root
    // supernode's contribution block. The first nvschur rows/cols of
    // contrib are the Schur columns in user-supplied order — see
    // factor_one_supernode (nvschur > 0) plus the BK pivot gate at
    // src/dense/factor.rs:1670 (positions ≥ ncol_eff are never swapped).
    let n_schur = nvschur_per_snode[n_snodes - 1];
    debug_assert!(n_schur > 0);
    let contrib = contrib_blocks[n_snodes - 1].take().ok_or_else(|| {
        FeralError::InvalidInput(
            "Schur path: root supernode produced no contribution block".to_string(),
        )
    })?;
    if contrib.dim < n_schur {
        return Err(FeralError::InvalidInput(format!(
            "Schur extraction: root contrib dim {} < n_schur {}",
            contrib.dim, n_schur
        )));
    }

    // Extract leading n_schur × n_schur subblock. ContribBlock data is
    // col-major dim × dim with valid data in the lower triangle (per
    // factor_frontal_blocked_in_place / factor_frontal). Mirror to a
    // full-square output buffer.
    let mut out = vec![0.0f64; n_schur * n_schur];
    for j in 0..n_schur {
        for i in 0..n_schur {
            let val = if i >= j {
                contrib.data[j * contrib.dim + i]
            } else {
                contrib.data[i * contrib.dim + j]
            };
            out[j * n_schur + i] = val;
        }
    }

    let factors = SparseFactors {
        n,
        perm: symbolic.perm.clone(),
        perm_inv: symbolic.perm_inv.clone(),
        node_factors,
        needs_refinement,
        scaling: scaling_user,
        scaling_info,
        resolved_method: symbolic.resolved_method,
        resolved_amalgamation: symbolic.resolved_amalgamation,
        resolved_preprocess: symbolic.resolved_preprocess,
    };
    let schur = SchurBlock {
        dim: n_schur,
        data: out,
    };

    Ok((factors, total_inertia, schur))
}

/// Gated dispatcher: routes to the D.3 dense fast-path when
/// [`should_use_dense_fast_path`] fires, otherwise runs the
/// multifrontal supernodal body in
/// [`factorize_multifrontal_supernodal_with_workspace`].
///
/// Semantics are byte-identical to `factorize_multifrontal`: the
/// returned `SparseFactors` and `Inertia` are the same for the
/// same inputs. Scratch allocations are drawn from (and returned
/// to) `ws` instead of the global allocator, so repeated calls
/// with different matrices amortise heap traffic.
///
/// On a gate hit the dense path draws its `n * n` densify buffer
/// from `ws.dense_values` (pooled via `to_dense_into`) — see
/// `dev/research/phase-2.5.x-to-dense-pooling.md`.
pub fn factorize_multifrontal_with_workspace(
    matrix: &CscMatrix,
    symbolic: &SymbolicFactorization,
    params: &NumericParams,
    ws: &mut FactorWorkspace,
) -> Result<(SparseFactors, Inertia), FeralError> {
    if should_use_dense_fast_path(matrix.n, matrix.row_idx.len()) {
        return dense_fast_factor_with_workspace(matrix, params, ws);
    }
    factorize_multifrontal_supernodal_with_workspace(matrix, symbolic, params, ws)
}

/// Workspace-reusing supernodal body (un-gated).
///
/// See [`factorize_multifrontal_supernodal`] for the entry point
/// that bypasses the D.3 gate. Directly callable from tests that
/// need forced-multifrontal behavior on an in-gate matrix.
///
/// See `dev/plans/factor-workspace.md` for the rollout plan and
/// `tests/factor_workspace_parity.rs` for the guardrail tests
/// enforcing bit-level equivalence with the no-workspace path.
pub fn factorize_multifrontal_supernodal_with_workspace(
    matrix: &CscMatrix,
    symbolic: &SymbolicFactorization,
    params: &NumericParams,
    ws: &mut FactorWorkspace,
) -> Result<(SparseFactors, Inertia), FeralError> {
    // Phase 2.10 profiler. When `params.profiler.is_none()`, every
    // `Instant::now()` below is gated out, so the production path
    // does no timing work.
    let t_total = params.profiler.as_ref().map(|_| Instant::now());
    let t_prologue = params.profiler.as_ref().map(|_| Instant::now());

    let n = symbolic.n;
    let n_snodes = symbolic.supernodes.len();

    // Re-establish the `row_map` invariant (all entries `usize::MAX`,
    // length >= n) unconditionally, so a prior error-exit that
    // skipped the per-supernode clear cannot leak state into this
    // call. `clear()` keeps capacity; `resize` rewrites entries —
    // cost is O(n), not O(n_snodes * n) as the pre-workspace code
    // paid.
    ws.row_map.clear();
    ws.row_map.resize(n, usize::MAX);

    // β refactor: scaling is a numeric-phase concern, computed
    // here against the live matrix values, not cached on the
    // value-agnostic `SymbolicFactorization`. Returns the user-
    // order scaling vector and a diagnostic info enum.
    //
    // Phase 2.4.4: if the symbolic phase ran `LdltCompress`, it
    // already produced an `Mc64Cache` that we reuse here when the
    // scaling strategy also resolves to MC64 — O(n) post-processing
    // instead of a second Hungarian.
    let (scaling_user, scaling_info) =
        compute_scaling_with_cache(matrix, &params.scaling, symbolic.cached_mc64.as_ref())?;
    if let crate::scaling::ScalingInfo::PartialSingular { n_unmatched } = &scaling_info {
        // No project-wide logging framework yet; mirror the Phase 1
        // convention of eprintln! for unusual diagnostics so this is
        // visible in bench output without being a hard failure.
        // Structurally singular matrices are allowed to proceed —
        // they typically surface the issue as a zero pivot during
        // numeric factorization, the right layer to reject.
        eprintln!(
            "warning: MC64 matching left {} of {} variables unmatched; \
             scaling is identity on those rows/columns",
            n_unmatched, n
        );
    }
    // Pivot-order cache of `scaling_user`: for each pivot index k,
    // `scaling_pivot_order[k] == scaling_user[symbolic.perm[k]]`.
    // This matches the assembly-time lookup pattern below where the
    // permuted CSC is indexed in pivot positions.
    let scaling_pivot_order: Vec<f64> =
        symbolic.perm.iter().map(|&old| scaling_user[old]).collect();
    debug_assert_eq!(scaling_pivot_order.len(), n);

    // Permute the matrix values into the new ordering
    let permuted = permute_csc_values(matrix, &symbolic.perm, &symbolic.perm_inv)?;

    // Full symmetric pattern for correct row index computation
    let full_pattern = permuted.symmetric_pattern();

    // Phase 2.3 Step 5: identify root supernodes (no parent in the etree
    // forest). A node is a root iff no other supernode lists it as a
    // child. Roots must run with `may_delay = false` so
    // `ZeroPivotAction::ForceAccept` absorbs any unstable pivots instead
    // of delaying them to a non-existent ancestor. On disconnected
    // matrices the forest has multiple roots — this handles them
    // uniformly.
    let mut is_root = vec![true; n_snodes];
    for snode in &symbolic.supernodes {
        for &child_idx in &snode.children {
            if child_idx < n_snodes {
                is_root[child_idx] = false;
            }
        }
    }

    // Storage for contribution blocks (one per supernode, freed after parent assembly)
    let mut contrib_blocks: Vec<Option<ContribBlock>> = (0..n_snodes).map(|_| None).collect();

    let mut node_factors: Vec<NodeFactors> = Vec::with_capacity(n_snodes);
    let mut total_inertia = Inertia {
        positive: 0,
        negative: 0,
        zero: 0,
    };
    let mut needs_refinement = false;

    // Process supernodes in postorder (children before parents).
    // Phase 2.5.2 Step B: per-supernode body extracted into
    // `factor_one_supernode` so a parallel task-graph driver (Step C)
    // can invoke it independently per-supernode. Sequential behaviour
    // is bit-exact against the pre-extraction loop — the helper is a
    // direct lift of the original loop body.
    //
    // Phase 2.9 (`dev/plans/phase-2.9-small-leaf-subtree.md`): when
    // `params.small_leaf == On`, leaf supernodes that were grouped
    // at symbolic time are dispatched to `factor_one_small_leaf` in
    // a single batched sweep per group. Group members are
    // postorder-consecutive indices, so we advance `snode_idx` past
    // the whole group after processing it; non-grouped supernodes
    // take the generic path exactly as before. The gate is `Off`
    // by default.
    let use_small_leaf =
        params.small_leaf == SmallLeafBatch::On && !symbolic.small_leaf_groups.is_empty();
    let mut snode_idx = 0usize;

    let prologue_us = t_prologue.map(|t| t.elapsed().as_micros() as u64);

    while snode_idx < n_snodes {
        if use_small_leaf {
            if let Some(gid) = symbolic.snode_group[snode_idx] {
                let group = &symbolic.small_leaf_groups[gid];
                debug_assert_eq!(
                    group.members.first(),
                    Some(&snode_idx),
                    "group members must start at current snode_idx"
                );
                for (i, &m) in group.members.iter().enumerate() {
                    let t_snode = params.profiler.as_ref().map(|_| Instant::now());
                    let node = factor_one_small_leaf(
                        m,
                        &group.member_rows[i],
                        symbolic,
                        &permuted,
                        &scaling_pivot_order,
                        &is_root,
                        params,
                        ws,
                        &mut contrib_blocks,
                    )?;
                    if let (Some(arc), Some(t)) = (params.profiler.as_ref(), t_snode) {
                        let snode = &symbolic.supernodes[m];
                        let timing = SupernodeTiming {
                            snode_idx: m,
                            nrow: snode.nrow,
                            ncol: snode.ncol,
                            us: t.elapsed().as_micros() as u64,
                        };
                        if let Ok(mut prof) = arc.lock() {
                            prof.timings.push(timing);
                        }
                    }
                    total_inertia.positive += node.inertia.positive;
                    total_inertia.negative += node.inertia.negative;
                    total_inertia.zero += node.inertia.zero;
                    if node.frontal_factors.needs_refinement {
                        needs_refinement = true;
                    }
                    node_factors.push(node);
                }
                snode_idx += group.members.len();
                continue;
            }
        }

        let t_snode = params.profiler.as_ref().map(|_| Instant::now());
        let node = factor_one_supernode(
            snode_idx,
            symbolic,
            &permuted,
            &full_pattern,
            &scaling_pivot_order,
            &is_root,
            params,
            ws,
            &mut contrib_blocks,
            0, // nvschur: standard path has no Schur tail
        )?;
        if let (Some(arc), Some(t)) = (params.profiler.as_ref(), t_snode) {
            let snode = &symbolic.supernodes[snode_idx];
            let timing = SupernodeTiming {
                snode_idx,
                nrow: snode.nrow,
                ncol: snode.ncol,
                us: t.elapsed().as_micros() as u64,
            };
            if let Ok(mut prof) = arc.lock() {
                prof.timings.push(timing);
            }
        }

        total_inertia.positive += node.inertia.positive;
        total_inertia.negative += node.inertia.negative;
        total_inertia.zero += node.inertia.zero;
        if node.frontal_factors.needs_refinement {
            needs_refinement = true;
        }
        node_factors.push(node);
        snode_idx += 1;
    }

    let t_epilogue = params.profiler.as_ref().map(|_| Instant::now());

    let result = Ok((
        SparseFactors {
            n,
            perm: symbolic.perm.clone(),
            perm_inv: symbolic.perm_inv.clone(),
            node_factors,
            needs_refinement,
            // β refactor: scaling vector + diagnostic info are
            // produced by `compute_scaling` at the top of this
            // function (no longer cached on `SymbolicFactorization`).
            // Solve operates at the user API boundary so it needs
            // user-order indexing, not the pivot-order cache used
            // at assembly time.
            scaling: scaling_user,
            scaling_info,
            resolved_method: symbolic.resolved_method,
            resolved_amalgamation: symbolic.resolved_amalgamation,
            resolved_preprocess: symbolic.resolved_preprocess,
        },
        total_inertia,
    ));

    if let Some(arc) = params.profiler.as_ref() {
        if let Ok(mut prof) = arc.lock() {
            prof.prologue_us = prologue_us.unwrap_or(0);
            prof.epilogue_us = t_epilogue
                .map(|t| t.elapsed().as_micros() as u64)
                .unwrap_or(0);
            prof.total_us = t_total.map(|t| t.elapsed().as_micros() as u64).unwrap_or(0);
        }
    }

    result
}

/// Factor a single supernode in isolation.
///
/// Phase 2.5.2 Step B: extracted from
/// [`factorize_multifrontal_supernodal_with_workspace`]'s per-supernode
/// loop body so the same code path can be reused by a future parallel
/// task-graph driver (Step C). Preserves the exact semantics of the
/// original loop iteration:
///
/// * Takes child contribution blocks out of `contrib_blocks` (via
///   `Option::take`) — children must have been produced by a prior
///   call for this same `contrib_blocks`.
/// * Writes the produced contribution block (if any) into
///   `contrib_blocks[snode_idx]`.
/// * Uses `ws.row_map`, `ws.frontal_values`, `ws.build_delayed`,
///   `ws.build_trailing`, `ws.build_seen` as scratch; respects the
///   same entry/exit invariants (row_map all `usize::MAX`, build_seen
///   all `false`).
///
/// Returns the `NodeFactors` for the supernode. The caller accumulates
/// inertia / `needs_refinement` from it.
#[allow(clippy::too_many_arguments)]
fn factor_one_supernode(
    snode_idx: usize,
    symbolic: &SymbolicFactorization,
    permuted: &CscMatrix,
    full_pattern: &crate::sparse::csc::CscPattern,
    scaling_pivot_order: &[f64],
    is_root: &[bool],
    params: &NumericParams,
    ws: &mut FactorWorkspace,
    contrib_blocks: &mut [Option<ContribBlock>],
    nvschur: usize,
) -> Result<NodeFactors, FeralError> {
    let snode = &symbolic.supernodes[snode_idx];
    let own_ncol = snode.ncol();
    let nrow = snode.nrow;

    if nrow == 0 || own_ncol == 0 {
        return Ok(NodeFactors {
            first_col: snode.first_col,
            ncol: 0,
            nelim: 0,
            n_delayed_in: 0,
            nrow: 0,
            row_indices: Vec::new(),
            frontal_factors: FrontalFactors {
                nrow: 0,
                ncol: 0,
                nelim: 0,
                l: Vec::new(),
                d_diag: Vec::new(),
                d_subdiag: Vec::new(),
                perm: Vec::new(),
                perm_inv: Vec::new(),
                contrib: Vec::new(),
                contrib_dim: 0,
                n_delayed: 0,
                inertia: Inertia {
                    positive: 0,
                    negative: 0,
                    zero: 0,
                },
                needs_refinement: false,
                n_rook_rescues: 0,
                zero_tol: params.bk.zero_tol,
                zero_tol_2x2: params.bk.zero_tol_2x2,
            },
            inertia: Inertia {
                positive: 0,
                negative: 0,
                zero: 0,
            },
        });
    }

    // Phase 2.3 Step 5: count delayed columns arriving from each
    // child. Children that were processed under `may_delay = true`
    // may have left `n_delayed` fully-summed columns un-eliminated
    // in the top-left of their contribution block; these re-enter
    // pivot search at this node as additional fully-summed columns
    // on top of `snode.ncol()`.
    let n_delayed_in: usize = snode
        .children
        .iter()
        .filter_map(|&c| contrib_blocks[c].as_ref())
        .map(|c| c.n_delayed)
        .sum();
    let expanded_ncol = own_ncol + n_delayed_in;

    // Build the row indices for this frontal. The default layout is
    // [own native cols (own_ncol) | delayed cols from children (n_delayed_in) | trailing rows].
    let mut row_indices = build_row_indices(
        snode,
        full_pattern,
        contrib_blocks,
        &mut ws.build_delayed,
        &mut ws.build_trailing,
        &mut ws.build_seen,
    );
    let actual_nrow = row_indices.len();
    debug_assert!(
        actual_nrow >= expanded_ncol,
        "row_indices ({}) must cover the expanded fully-summed block ({})",
        actual_nrow,
        expanded_ncol
    );
    // F3.2b layout fix: when this is a Schur supernode (`nvschur > 0`)
    // that received delayed columns from descendants (`n_delayed_in > 0`),
    // the default layout above places own (Schur) cols at frontal
    // positions [0, own_ncol) and delayed cols at [own_ncol, expanded_ncol).
    // The BK pivot loop in `factor_frontal_blocked_in_place` eliminates
    // positions [0, ncol_eff) where ncol_eff = expanded_ncol - nvschur =
    // n_delayed_in, so without this swap the Schur cols sit inside the
    // eliminable range and get factored out — which is exactly the
    // opposite of what we want. Swap so delayed cols come first
    // (eliminable) and Schur cols come after (excluded from pivoting,
    // per the BK gate at src/dense/factor.rs:1670).
    let own_col_offset = if nvschur > 0 && n_delayed_in > 0 {
        let mut swapped = Vec::with_capacity(actual_nrow);
        swapped.extend_from_slice(&row_indices[own_ncol..expanded_ncol]);
        swapped.extend_from_slice(&row_indices[..own_ncol]);
        swapped.extend_from_slice(&row_indices[expanded_ncol..]);
        row_indices = swapped;
        n_delayed_in
    } else {
        0
    };

    // Populate the pooled `ws.row_map`. Invariant on entry: every entry
    // is `usize::MAX`. Mirror-clear at the end restores it.
    for (local, &global) in row_indices.iter().enumerate() {
        ws.row_map[global] = local;
    }

    // Step 1: Assemble original matrix entries into frontal, applying
    // symmetric scaling D·A·D in place. Own cols sit at frontal positions
    // [own_col_offset, own_col_offset + own_ncol); for the standard path
    // own_col_offset = 0, for the Schur swap path it's n_delayed_in.
    let scaling = scaling_pivot_order;
    let mut frontal_buf = std::mem::take(&mut ws.frontal_values);
    frontal_buf.clear();
    frontal_buf.resize(actual_nrow * actual_nrow, 0.0);
    let mut frontal = SymmetricMatrix {
        n: actual_nrow,
        data: frontal_buf,
    };
    for (k_local, &gj) in row_indices[own_col_offset..own_col_offset + own_ncol]
        .iter()
        .enumerate()
    {
        let local_j = own_col_offset + k_local;
        let s_j = scaling[gj];
        for k in permuted.col_ptr[gj]..permuted.col_ptr[gj + 1] {
            let gi = permuted.row_idx[k];
            let local_i = ws.row_map[gi];
            if local_i != usize::MAX {
                let val = permuted.values[k] * scaling[gi] * s_j;
                frontal.set(local_i, local_j, val);
            }
        }
    }

    // Step 2: Assemble child contribution blocks (extend-add).
    for &child_idx in &snode.children {
        if let Some(contrib) = contrib_blocks[child_idx].take() {
            extend_add(&contrib, &ws.row_map, &mut frontal);
        }
    }

    // Step 3: Factor the frontal in place (W-3a). `frontal.data`
    // content is undefined on return; the buffer goes back to the pool.
    //
    // F3.2b: `nvschur` Schur columns at positions
    // `[expanded_ncol - nvschur, expanded_ncol)` are excluded from the
    // eliminable range. The BK pivot loop only swaps within
    // `[0, ncol_eff)` (see `dense::factor` r_is_fully_summed gate at
    // src/dense/factor.rs:1670), so Schur columns stay at their
    // original positions and end up in the contribution block in the
    // user-supplied order. nvschur > 0 implies is_root (Schur tail at
    // top of etree post-F3.2a), so may_delay is forced to false.
    debug_assert!(
        nvschur == 0 || is_root[snode_idx],
        "nvschur > 0 only valid at root supernodes (Schur tail invariant)"
    );
    debug_assert!(nvschur <= expanded_ncol);
    let may_delay = !is_root[snode_idx];
    let eliminable = expanded_ncol - nvschur;
    let mut ff = factor_frontal_blocked_in_place(&mut frontal, eliminable, may_delay, &params.bk)?;
    ws.frontal_values = frontal.data;

    let node_inertia = ff.inertia.clone();
    let node_nelim = ff.nelim;
    let node_n_delayed = ff.n_delayed;

    // Step 4: Store contribution block for parent. Move
    // `ff.contrib` directly into `ContribBlock::data` (W-3b: avoid the
    // 30 MB clone on CHAINWOO root). After this move,
    // `frontal_factors.contrib` in the saved `NodeFactors` is empty —
    // production solve paths only read `l`, `d_diag`, `d_subdiag`,
    // `perm`, `perm_inv` from `frontal_factors`; `contrib` is consumed
    // by the parent supernode during assembly and is dead data
    // afterward.
    if ff.contrib_dim > 0 {
        let cdim = ff.contrib_dim;
        let mut contrib_row_indices = Vec::with_capacity(cdim);
        for cj in 0..cdim {
            contrib_row_indices.push(row_indices[ff.perm[node_nelim + cj]]);
        }
        let contrib_data = std::mem::take(&mut ff.contrib);
        contrib_blocks[snode_idx] = Some(ContribBlock {
            row_indices: contrib_row_indices,
            data: contrib_data,
            dim: cdim,
            n_delayed: node_n_delayed,
        });
    }

    // Restore the `row_map` invariant.
    for &global in &row_indices {
        ws.row_map[global] = usize::MAX;
    }

    Ok(NodeFactors {
        first_col: snode.first_col,
        ncol: expanded_ncol,
        nelim: node_nelim,
        n_delayed_in,
        nrow: actual_nrow,
        row_indices,
        frontal_factors: ff,
        inertia: node_inertia,
    })
}

/// Factor a single true-leaf supernode that was pre-qualified for the
/// SmallLeafSubtree batched path (phase 2.9).
///
/// This is a leaf specialisation of `factor_one_supernode` — true leaves
/// have no children, so:
///
/// * `n_delayed_in == 0`, `expanded_ncol == own_ncol`.
/// * No extend-add pass (the children loop is empty anyway).
/// * `row_indices` is passed in pre-computed at symbolic time
///   (`SmallLeafGroup::member_rows`), saving the per-front
///   `build_row_indices` call and its `build_delayed`/`build_seen`
///   scratch churn on every leaf.
///
/// All other semantics — scaling, `factor_frontal_blocked`, contribution
/// block deposit, `row_map` write/restore — match the generic path
/// byte-for-byte, which is what the parity tests in
/// `tests/small_leaf_parity.rs` verify.
#[allow(clippy::too_many_arguments)]
fn factor_one_small_leaf(
    snode_idx: usize,
    precomputed_rows: &[usize],
    symbolic: &SymbolicFactorization,
    permuted: &CscMatrix,
    scaling_pivot_order: &[f64],
    is_root: &[bool],
    params: &NumericParams,
    ws: &mut FactorWorkspace,
    contrib_blocks: &mut [Option<ContribBlock>],
) -> Result<NodeFactors, FeralError> {
    let snode = &symbolic.supernodes[snode_idx];
    debug_assert!(
        snode.children.is_empty(),
        "factor_one_small_leaf called on non-leaf supernode {}",
        snode_idx
    );

    let own_ncol = snode.ncol();
    let nrow = snode.nrow;

    if nrow == 0 || own_ncol == 0 {
        return Ok(NodeFactors {
            first_col: snode.first_col,
            ncol: 0,
            nelim: 0,
            n_delayed_in: 0,
            nrow: 0,
            row_indices: Vec::new(),
            frontal_factors: FrontalFactors {
                nrow: 0,
                ncol: 0,
                nelim: 0,
                l: Vec::new(),
                d_diag: Vec::new(),
                d_subdiag: Vec::new(),
                perm: Vec::new(),
                perm_inv: Vec::new(),
                contrib: Vec::new(),
                contrib_dim: 0,
                n_delayed: 0,
                inertia: Inertia {
                    positive: 0,
                    negative: 0,
                    zero: 0,
                },
                needs_refinement: false,
                n_rook_rescues: 0,
                zero_tol: params.bk.zero_tol,
                zero_tol_2x2: params.bk.zero_tol_2x2,
            },
            inertia: Inertia {
                positive: 0,
                negative: 0,
                zero: 0,
            },
        });
    }

    let row_indices = precomputed_rows.to_vec();
    let actual_nrow = row_indices.len();
    let expanded_ncol = own_ncol;
    debug_assert!(actual_nrow >= expanded_ncol);

    for (local, &global) in row_indices.iter().enumerate() {
        ws.row_map[global] = local;
    }

    let scaling = scaling_pivot_order;
    let mut frontal_buf = std::mem::take(&mut ws.frontal_values);
    frontal_buf.clear();
    frontal_buf.resize(actual_nrow * actual_nrow, 0.0);
    let mut frontal = SymmetricMatrix {
        n: actual_nrow,
        data: frontal_buf,
    };
    for (local_j, &gj) in row_indices[..own_ncol].iter().enumerate() {
        let s_j = scaling[gj];
        for k in permuted.col_ptr[gj]..permuted.col_ptr[gj + 1] {
            let gi = permuted.row_idx[k];
            let local_i = ws.row_map[gi];
            if local_i != usize::MAX {
                let val = permuted.values[k] * scaling[gi] * s_j;
                frontal.set(local_i, local_j, val);
            }
        }
    }

    // No extend-add: leaves have no children.

    // W-3a: factor in place; pool returns the (now-undefined) buffer.
    let may_delay = !is_root[snode_idx];
    let mut ff =
        factor_frontal_blocked_in_place(&mut frontal, expanded_ncol, may_delay, &params.bk)?;
    ws.frontal_values = frontal.data;

    let node_inertia = ff.inertia.clone();
    let node_nelim = ff.nelim;
    let node_n_delayed = ff.n_delayed;

    // W-3b: move `ff.contrib` rather than clone (see internal variant
    // for the full contract).
    if ff.contrib_dim > 0 {
        let cdim = ff.contrib_dim;
        let mut contrib_row_indices = Vec::with_capacity(cdim);
        for cj in 0..cdim {
            contrib_row_indices.push(row_indices[ff.perm[node_nelim + cj]]);
        }
        let contrib_data = std::mem::take(&mut ff.contrib);
        contrib_blocks[snode_idx] = Some(ContribBlock {
            row_indices: contrib_row_indices,
            data: contrib_data,
            dim: cdim,
            n_delayed: node_n_delayed,
        });
    }

    for &global in &row_indices {
        ws.row_map[global] = usize::MAX;
    }

    Ok(NodeFactors {
        first_col: snode.first_col,
        ncol: expanded_ncol,
        nelim: node_nelim,
        n_delayed_in: 0,
        nrow: actual_nrow,
        row_indices,
        frontal_factors: ff,
        inertia: node_inertia,
    })
}

/// Minimum supernode count below which the parallel driver falls
/// through to sequential. Phase 2.5.2 Step D. Tentative value
/// (conservative): reassess after the corpus bench in Step E shows
/// where per-task overhead breaks even.
pub const N_PAR_MIN: usize = 32;

/// Predicate: does the symbolic factorization present enough structure
/// for the parallel driver to win?
///
/// Two conditions:
/// 1. `n_snodes >= N_PAR_MIN` — enough tasks to amortise thread-pool
///    overhead.
/// 2. At least one supernode has ≥ 2 children — a pure postorder
///    chain has zero sibling parallelism, so the parallel driver
///    would add only overhead.
pub fn should_parallelize_assembly(symbolic: &SymbolicFactorization) -> bool {
    if symbolic.supernodes.len() < N_PAR_MIN {
        return false;
    }
    symbolic.supernodes.iter().any(|s| s.children.len() >= 2)
}

/// Gated parallel entry point. Phase 2.5.2 Step D.
///
/// Dispatch order:
/// 1. [`should_use_dense_fast_path`] — dense fast-path takes precedence.
/// 2. If [`should_parallelize_assembly`] returns true, dispatch to
///    the rayon driver ([`factorize_multifrontal_supernodal_parallel`]).
/// 3. Otherwise, sequential multifrontal.
///
/// The workspace `ws` is used by the dense fast-path and the
/// sequential fall-through; the parallel driver owns its own
/// per-thread scratch internally.
pub fn factorize_multifrontal_parallel_with_workspace(
    matrix: &CscMatrix,
    symbolic: &SymbolicFactorization,
    params: &NumericParams,
    ws: &mut FactorWorkspace,
) -> Result<(SparseFactors, Inertia), FeralError> {
    if should_use_dense_fast_path(matrix.n, matrix.row_idx.len()) {
        return dense_fast_factor_with_workspace(matrix, params, ws);
    }
    if should_parallelize_assembly(symbolic) {
        return factorize_multifrontal_supernodal_parallel(matrix, symbolic, params);
    }
    factorize_multifrontal_supernodal_with_workspace(matrix, symbolic, params, ws)
}

/// Fresh-workspace variant of [`factorize_multifrontal_parallel_with_workspace`].
pub fn factorize_multifrontal_parallel(
    matrix: &CscMatrix,
    symbolic: &SymbolicFactorization,
    params: &NumericParams,
) -> Result<(SparseFactors, Inertia), FeralError> {
    let mut ws = FactorWorkspace::new();
    factorize_multifrontal_parallel_with_workspace(matrix, symbolic, params, &mut ws)
}

/// Rayon task-graph parallel driver for the multifrontal assembly tree.
///
/// Phase 2.5.2 Step C. Bit-exact parity with
/// [`factorize_multifrontal_supernodal_with_workspace`] on each
/// supernode, because:
///
/// * Each supernode's assembly (extend-add over children) happens in
///   a single task with children iterated in `snode.children` order —
///   the same FP sum order as sequential.
/// * Each task uses a per-thread `FactorWorkspace` drawn from
///   `thread_ws[rayon::current_thread_index()]`, so scratch buffers
///   are never shared across threads.
/// * The shared contribution-block store is mutex-protected; each
///   task only locks it briefly to stage its own children into a
///   local `Vec<Option<ContribBlock>>` and, later, to deposit its
///   own block. No mutex is held during the dense kernel.
///
/// Entry points that dispatch to this driver must check
/// `n_snodes >= N_PAR_MIN` first and otherwise fall through to the
/// sequential path (see `factorize_multifrontal_parallel_with_workspace`).
///
/// The `FactorWorkspace` passed in by the caller is **not** used for
/// per-task scratch — it is reserved for the caller's amortisation
/// semantics (future extension). Per-task workspaces are owned
/// internally, one per rayon worker thread.
pub fn factorize_multifrontal_supernodal_parallel(
    matrix: &CscMatrix,
    symbolic: &SymbolicFactorization,
    params: &NumericParams,
) -> Result<(SparseFactors, Inertia), FeralError> {
    use std::sync::atomic::AtomicUsize;
    use std::sync::Mutex;

    let n = symbolic.n;
    let n_snodes = symbolic.supernodes.len();

    // Setup — mirrors the sequential driver. Reuse the symbolic-phase
    // MC64 cache if present (see the sequential driver for details).
    let (scaling_user, scaling_info) =
        compute_scaling_with_cache(matrix, &params.scaling, symbolic.cached_mc64.as_ref())?;
    if let crate::scaling::ScalingInfo::PartialSingular { n_unmatched } = &scaling_info {
        eprintln!(
            "warning: MC64 matching left {} of {} variables unmatched; \
             scaling is identity on those rows/columns",
            n_unmatched, n
        );
    }
    let scaling_pivot_order: Vec<f64> =
        symbolic.perm.iter().map(|&old| scaling_user[old]).collect();
    let permuted = permute_csc_values(matrix, &symbolic.perm, &symbolic.perm_inv)?;
    let full_pattern = permuted.symmetric_pattern();

    let mut is_root = vec![true; n_snodes];
    for snode in &symbolic.supernodes {
        for &child_idx in &snode.children {
            if child_idx < n_snodes {
                is_root[child_idx] = false;
            }
        }
    }

    // Parent table: parents[c] == i iff i's children include c.
    let mut parents: Vec<Option<usize>> = vec![None; n_snodes];
    for (i, snode) in symbolic.supernodes.iter().enumerate() {
        for &c in &snode.children {
            if c < n_snodes {
                parents[c] = Some(i);
            }
        }
    }

    // Pending-children atomic counter per supernode. A supernode is
    // ready to process when its counter hits zero.
    let pending: Vec<AtomicUsize> = symbolic
        .supernodes
        .iter()
        .map(|s| {
            let cnt = s.children.iter().filter(|&&c| c < n_snodes).count();
            AtomicUsize::new(cnt)
        })
        .collect();

    // Shared state: contrib blocks, result slots, first error.
    let contrib_blocks: Mutex<Vec<Option<ContribBlock>>> =
        Mutex::new((0..n_snodes).map(|_| None).collect());
    let node_factors_out: Mutex<Vec<Option<NodeFactors>>> =
        Mutex::new((0..n_snodes).map(|_| None).collect());
    let first_error: Mutex<Option<FeralError>> = Mutex::new(None);

    // Per-thread workspaces. Provision one workspace per rayon
    // worker PLUS one extra slot for the calling thread, which may
    // also execute tasks inside `rayon::scope` and has
    // `current_thread_index() == None`. Bin the calling thread's
    // tasks into the extra slot (index `num_threads`) to avoid
    // mutex-serializing it against worker 0 — and to prevent the
    // caller and worker 0 from time-sharing a single workspace.
    let num_threads = rayon::current_num_threads().max(1);
    let thread_ws: Vec<Mutex<FactorWorkspace>> = (0..num_threads + 1)
        .map(|_| {
            let mut w = FactorWorkspace::new();
            w.row_map.resize(n, usize::MAX);
            w.build_seen.resize(n, false);
            Mutex::new(w)
        })
        .collect();

    // Collect the true leaves (supernodes with no children) BEFORE
    // entering the scope. Using `pending[i].load() == 0` as the
    // seeding predicate is unsound: once the scope is live, workers
    // execute previously-spawned tasks concurrently with this loop
    // and decrement parents' counters. A non-leaf whose pending
    // counter just hit zero would then be spawned twice — once here
    // by the caller, and again by the final child via the
    // fetch_sub==1 trampoline in `run_parallel_task`.
    let leaves: Vec<usize> = symbolic
        .supernodes
        .iter()
        .enumerate()
        .filter_map(|(i, s)| {
            if s.children.iter().all(|&c| c >= n_snodes) {
                Some(i)
            } else {
                None
            }
        })
        .collect();

    rayon::scope(|scope| {
        for &leaf_idx in &leaves {
            run_parallel_task(
                scope,
                leaf_idx,
                symbolic,
                &permuted,
                &full_pattern,
                &scaling_pivot_order,
                &is_root,
                params,
                &parents,
                &pending,
                &contrib_blocks,
                &node_factors_out,
                &first_error,
                &thread_ws,
            );
        }
    });

    // Surface any first-error that the tasks captured.
    let err_opt = match first_error.into_inner() {
        Ok(v) => v,
        Err(p) => p.into_inner(),
    };
    if let Some(err) = err_opt {
        return Err(err);
    }

    // Collect node_factors in postorder (same order as sequential).
    let nodes_vec = match node_factors_out.into_inner() {
        Ok(v) => v,
        Err(p) => p.into_inner(),
    };
    let mut final_nodes: Vec<NodeFactors> = Vec::with_capacity(n_snodes);
    let mut total_inertia = Inertia {
        positive: 0,
        negative: 0,
        zero: 0,
    };
    let mut needs_refinement = false;
    for opt in nodes_vec.into_iter() {
        let node = match opt {
            Some(n) => n,
            None => {
                return Err(FeralError::InvalidInput(
                    "parallel driver: supernode was not processed (graph stall)".to_string(),
                ));
            }
        };
        total_inertia.positive += node.inertia.positive;
        total_inertia.negative += node.inertia.negative;
        total_inertia.zero += node.inertia.zero;
        if node.frontal_factors.needs_refinement {
            needs_refinement = true;
        }
        final_nodes.push(node);
    }

    Ok((
        SparseFactors {
            n,
            perm: symbolic.perm.clone(),
            perm_inv: symbolic.perm_inv.clone(),
            node_factors: final_nodes,
            needs_refinement,
            scaling: scaling_user,
            scaling_info,
            resolved_method: symbolic.resolved_method,
            resolved_amalgamation: symbolic.resolved_amalgamation,
            resolved_preprocess: symbolic.resolved_preprocess,
        },
        total_inertia,
    ))
}

/// Spawn a single supernode factorization task into the rayon scope.
///
/// On completion, decrements the parent's pending counter and — if
/// the parent becomes ready — recursively spawns the parent into the
/// same scope. The top-level call seeds all leaf supernodes.
#[allow(clippy::too_many_arguments)]
fn run_parallel_task<'a>(
    scope: &rayon::Scope<'a>,
    snode_idx: usize,
    symbolic: &'a SymbolicFactorization,
    permuted: &'a CscMatrix,
    full_pattern: &'a crate::sparse::csc::CscPattern,
    scaling_pivot_order: &'a [f64],
    is_root: &'a [bool],
    params: &'a NumericParams,
    parents: &'a [Option<usize>],
    pending: &'a [std::sync::atomic::AtomicUsize],
    contrib_blocks: &'a std::sync::Mutex<Vec<Option<ContribBlock>>>,
    node_factors_out: &'a std::sync::Mutex<Vec<Option<NodeFactors>>>,
    first_error: &'a std::sync::Mutex<Option<FeralError>>,
    thread_ws: &'a [std::sync::Mutex<FactorWorkspace>],
) {
    use std::sync::atomic::Ordering;
    scope.spawn(move |s| {
        // Fast-exit if a prior task errored; the scope will still
        // drain, we just skip actual work.
        {
            let err_guard = match first_error.lock() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            if err_guard.is_some() {
                return;
            }
        }

        let snode = &symbolic.supernodes[snode_idx];
        let n_snodes = symbolic.supernodes.len();

        // Stage child contributions into a task-local vec so the
        // helper can read/take them without holding the shared lock.
        // Most entries stay None; only the children's slots are
        // populated. The helper writes its own slot into this same
        // vec, which we extract after the helper returns.
        let mut local_contribs: Vec<Option<ContribBlock>> = (0..n_snodes).map(|_| None).collect();
        {
            let mut shared = match contrib_blocks.lock() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            for &c in &snode.children {
                if c < n_snodes {
                    local_contribs[c] = shared[c].take();
                }
            }
        }

        // Pick a per-thread workspace slot. `current_thread_index`
        // returns Some(worker_idx) when this task runs on a rayon
        // worker, and None when it runs on the calling thread
        // (rayon::scope donates the caller's thread to execute
        // tasks while it waits). The caller gets the last slot
        // (`thread_ws.len() - 1`) so it does not contend with any
        // worker's slot.
        let thread_idx = rayon::current_thread_index().unwrap_or(thread_ws.len() - 1);
        let thread_idx = thread_idx.min(thread_ws.len() - 1);
        let ws_mtx = &thread_ws[thread_idx];

        let result = {
            let mut ws_guard = match ws_mtx.lock() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            factor_one_supernode(
                snode_idx,
                symbolic,
                permuted,
                full_pattern,
                scaling_pivot_order,
                is_root,
                params,
                &mut ws_guard,
                &mut local_contribs,
                0, // nvschur: parallel path is not used by Schur API
            )
        };

        match result {
            Ok(node) => {
                let own_contrib = local_contribs[snode_idx].take();
                {
                    let mut shared = match contrib_blocks.lock() {
                        Ok(g) => g,
                        Err(p) => p.into_inner(),
                    };
                    shared[snode_idx] = own_contrib;
                }
                {
                    let mut nf = match node_factors_out.lock() {
                        Ok(g) => g,
                        Err(p) => p.into_inner(),
                    };
                    nf[snode_idx] = Some(node);
                }
                if let Some(parent_idx) = parents[snode_idx] {
                    let prev = pending[parent_idx].fetch_sub(1, Ordering::AcqRel);
                    if prev == 1 {
                        run_parallel_task(
                            s,
                            parent_idx,
                            symbolic,
                            permuted,
                            full_pattern,
                            scaling_pivot_order,
                            is_root,
                            params,
                            parents,
                            pending,
                            contrib_blocks,
                            node_factors_out,
                            first_error,
                            thread_ws,
                        );
                    }
                }
            }
            Err(e) => {
                let mut err_guard = match first_error.lock() {
                    Ok(g) => g,
                    Err(p) => p.into_inner(),
                };
                if err_guard.is_none() {
                    *err_guard = Some(e);
                }
            }
        }
    });
}

/// Permute a CSC matrix: compute the lower triangle of P·A·Pᵀ.
fn permute_csc_values(
    matrix: &CscMatrix,
    _perm: &[usize],
    perm_inv: &[usize],
) -> Result<CscMatrix, FeralError> {
    let n = matrix.n;

    // Collect permuted entries in lower triangle
    let mut triplets: Vec<(usize, usize, f64)> = Vec::with_capacity(matrix.nnz());

    for old_j in 0..n {
        let new_j = perm_inv[old_j];
        for k in matrix.col_ptr[old_j]..matrix.col_ptr[old_j + 1] {
            let old_i = matrix.row_idx[k];
            let new_i = perm_inv[old_i];
            let val = matrix.values[k];

            // Store in lower triangle of permuted matrix
            if new_i >= new_j {
                triplets.push((new_i, new_j, val));
            } else {
                triplets.push((new_j, new_i, val));
            }
        }
    }

    let rows: Vec<usize> = triplets.iter().map(|t| t.0).collect();
    let cols: Vec<usize> = triplets.iter().map(|t| t.1).collect();
    let vals: Vec<f64> = triplets.iter().map(|t| t.2).collect();

    CscMatrix::from_triplets(n, &rows, &cols, &vals)
}

/// Build row indices for a frontal matrix.
///
/// Returns indices laid out as:
///
/// ```text
/// [own native cols (own_ncol)]
/// [delayed cols inherited from children (n_delayed_in)]
/// [trailing non-fully-summed rows, sorted]
/// ```
///
/// The first two regions together form the fully-summed block that
/// `factor_frontal` is allowed to pivot over. Delayed column global
/// indices come from each child's `ContribBlock.row_indices[..n_delayed]`
/// in child-iteration order; duplicates across children cannot arise
/// because each matrix column belongs to exactly one supernode.
/// Trailing rows are deduplicated against the fully-summed set so a
/// delayed column that also shows up as a pattern row of a parent
/// column (via the full symmetric pattern) does not appear twice.
fn build_row_indices(
    snode: &crate::symbolic::supernode::Supernode,
    full_pattern: &crate::sparse::csc::CscPattern,
    contrib_blocks: &[Option<ContribBlock>],
    build_delayed: &mut Vec<usize>,
    build_trailing: &mut Vec<usize>,
    build_seen: &mut Vec<bool>,
) -> Vec<usize> {
    let own_ncol = snode.ncol();
    let first_col = snode.first_col;
    let n = full_pattern.n;

    // Grow `build_seen` on demand; caller maintains the all-`false`
    // invariant outside this function.
    if build_seen.len() < n {
        build_seen.resize(n, false);
    }

    // Collect delayed columns from each child, preserving child-iteration
    // order. Bit-for-bit equivalent to the old `Vec::new() + extend` path;
    // the Vec is pooled across supernodes so only its capacity growth
    // allocates.
    build_delayed.clear();
    for &child_idx in &snode.children {
        if let Some(contrib) = &contrib_blocks[child_idx] {
            build_delayed.extend_from_slice(&contrib.row_indices[..contrib.n_delayed]);
        }
    }

    // Mark own native + delayed columns as "fully summed" in the seen
    // bitmap so the trailing scan skips them. Duplicates across children
    // cannot arise (each matrix column belongs to exactly one supernode).
    for seen in build_seen.iter_mut().skip(first_col).take(own_ncol) {
        *seen = true;
    }
    for &c in build_delayed.iter() {
        build_seen[c] = true;
    }

    // Trailing row set via seen-based dedup. Same role as the previous
    // BTreeSet<usize> but with O(1) insert and a single O(m log m) sort
    // at the end to match the BTreeSet iteration order that callers
    // (and the parity tests) depend on.
    build_trailing.clear();
    for j in first_col..first_col + own_ncol {
        for k in full_pattern.col_ptr[j]..full_pattern.col_ptr[j + 1] {
            let r = full_pattern.row_idx[k];
            if !build_seen[r] {
                build_seen[r] = true;
                build_trailing.push(r);
            }
        }
    }
    for &child_idx in &snode.children {
        if let Some(contrib) = &contrib_blocks[child_idx] {
            for &r in &contrib.row_indices[contrib.n_delayed..] {
                if !build_seen[r] {
                    build_seen[r] = true;
                    build_trailing.push(r);
                }
            }
        }
    }
    build_trailing.sort_unstable();

    let total = own_ncol + build_delayed.len() + build_trailing.len();
    let mut result = Vec::with_capacity(total);
    result.extend(first_col..first_col + own_ncol);
    result.extend_from_slice(build_delayed);
    result.extend_from_slice(build_trailing);

    // Restore the all-`false` invariant on `build_seen` by clearing
    // only the entries we touched. Cheaper than a full `resize` and
    // keeps the invariant auditable.
    for seen in build_seen.iter_mut().skip(first_col).take(own_ncol) {
        *seen = false;
    }
    for &c in build_delayed.iter() {
        build_seen[c] = false;
    }
    for &r in build_trailing.iter() {
        build_seen[r] = false;
    }

    result
}

/// Contribution block from a child supernode.
///
/// Under delayed pivoting the top-left `n_delayed × n_delayed` block
/// holds the child's un-eliminated fully-summed columns (which must
/// re-enter pivot search at the parent as additional fully-summed
/// columns), and the bottom-right `(dim - n_delayed) × (dim - n_delayed)`
/// block is the classic Schur complement over the non-fully-summed
/// trailing rows. The cross block (rows = trailing, cols = delayed)
/// carries the mixed interactions. `row_indices[..n_delayed]` are
/// the global row indices of the delayed columns in the parent's
/// numbering; `row_indices[n_delayed..]` are the trailing rows.
#[derive(Debug)]
struct ContribBlock {
    /// Row indices of the contribution block (global).
    /// First `n_delayed` entries are delayed fully-summed columns;
    /// the remainder are the trailing non-fully-summed rows (sorted).
    row_indices: Vec<usize>,
    /// Dense symmetric matrix data (lower triangle, column-major).
    /// Dimension: row_indices.len() × row_indices.len()
    data: Vec<f64>,
    /// Dimension of the contribution block.
    dim: usize,
    /// Number of delayed fully-summed columns carried in this block
    /// (top-left `n_delayed × n_delayed` sub-matrix). Zero for nodes
    /// whose BK sweep succeeded on every attempted column. Consumed
    /// by the parent's `build_row_indices` and the Step 5 assembly
    /// which places these columns in the parent's fully-summed region.
    n_delayed: usize,
}

/// Extend-add: assemble a child's contribution block into the parent frontal.
fn extend_add(contrib: &ContribBlock, parent_row_map: &[usize], frontal: &mut SymmetricMatrix) {
    let cdim = contrib.dim;
    for cj in 0..cdim {
        let parent_j = parent_row_map[contrib.row_indices[cj]];
        if parent_j == usize::MAX {
            continue;
        }
        for ci in cj..cdim {
            let parent_i = parent_row_map[contrib.row_indices[ci]];
            if parent_i == usize::MAX {
                continue;
            }
            let val = contrib.data[cj * cdim + ci];
            if val == 0.0 {
                continue;
            }
            // Place in lower triangle of parent frontal
            if parent_i >= parent_j {
                frontal.set(parent_i, parent_j, frontal.get(parent_i, parent_j) + val);
            } else {
                frontal.set(parent_j, parent_i, frontal.get(parent_j, parent_i) + val);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dense::factor::ZeroPivotAction;
    use crate::symbolic::{symbolic_factorize, SupernodeParams};

    fn make_params() -> NumericParams {
        NumericParams::with_bk(BunchKaufmanParams {
            on_zero_pivot: ZeroPivotAction::ForceAccept,
            ..BunchKaufmanParams::default()
        })
    }

    #[test]
    fn test_factorize_diagonal() {
        let m = CscMatrix::from_triplets(3, &[0, 1, 2], &[0, 1, 2], &[2.0, 3.0, 5.0]).unwrap();

        let sym = symbolic_factorize(&m, &SupernodeParams::default()).unwrap();
        let (factors, inertia) = factorize_multifrontal(&m, &sym, &make_params()).unwrap();

        assert_eq!(inertia.positive, 3);
        assert_eq!(inertia.negative, 0);
        assert_eq!(inertia.zero, 0);
        assert_eq!(factors.n, 3);
    }

    #[test]
    fn test_summary_one_liner() {
        // Tridiagonal n=32 — large enough to bypass the dense
        // fast-path (N_TINY=16, N_MAX=128 with density gate) so the
        // multifrontal path runs and the resolved-field mirroring
        // from `SymbolicFactorization` is exercised.
        let n = 32usize;
        let mut rows: Vec<usize> = Vec::new();
        let mut cols: Vec<usize> = Vec::new();
        let mut vals: Vec<f64> = Vec::new();
        for i in 0..n {
            rows.push(i);
            cols.push(i);
            vals.push(2.0);
            if i + 1 < n {
                rows.push(i + 1);
                cols.push(i);
                vals.push(-1.0);
            }
        }
        let m = CscMatrix::from_triplets(n, &rows, &cols, &vals).unwrap();

        let sym = symbolic_factorize(&m, &SupernodeParams::default()).unwrap();
        let (factors, inertia) = factorize_multifrontal(&m, &sym, &make_params()).unwrap();

        let s = factors.summary();
        assert!(s.contains("ord="), "summary missing ord field: {}", s);
        assert!(s.contains("amalg="), "summary missing amalg field: {}", s);
        assert!(
            s.contains("preproc="),
            "summary missing preproc field: {}",
            s
        );
        assert!(
            s.contains("scaling="),
            "summary missing scaling field: {}",
            s
        );
        let nnz_l = factors.factor_nnz();
        assert!(nnz_l > 0, "tridiagonal factor_nnz must be > 0");
        assert!(
            s.contains(&format!("nnz_L={}", nnz_l)),
            "summary nnz_L mismatch: got {}, want nnz_L={}",
            s,
            nnz_l
        );
        let expected = format!(
            "inertia=({},{},{})",
            inertia.positive, inertia.negative, inertia.zero
        );
        assert!(
            s.contains(&expected),
            "summary inertia mismatch: got {}, want substring {}",
            s,
            expected
        );
        // `Auto` is a dispatch sentinel, never a resolved value.
        assert_ne!(
            factors.resolved_amalgamation,
            crate::symbolic::AmalgamationStrategy::Auto
        );
        assert_ne!(
            factors.resolved_method,
            crate::symbolic::OrderingMethod::Auto
        );
        assert_ne!(
            factors.resolved_preprocess,
            crate::symbolic::OrderingPreprocess::Auto
        );
        // Mirror invariant: the numeric factors agree with the
        // symbolic factorization on the resolved strategies.
        assert_eq!(factors.resolved_method, sym.resolved_method);
        assert_eq!(factors.resolved_amalgamation, sym.resolved_amalgamation);
        assert_eq!(factors.resolved_preprocess, sym.resolved_preprocess);
    }

    #[test]
    fn test_factorize_tridiagonal() {
        // [2 -1  0]
        // [-1 2 -1]
        // [0 -1  2]
        let m = CscMatrix::from_triplets(
            3,
            &[0, 1, 1, 2, 2],
            &[0, 0, 1, 1, 2],
            &[2.0, -1.0, 2.0, -1.0, 2.0],
        )
        .unwrap();

        let sym = symbolic_factorize(&m, &SupernodeParams::default()).unwrap();
        let (factors, inertia) = factorize_multifrontal(&m, &sym, &make_params()).unwrap();

        // This matrix is SPD
        assert_eq!(inertia.positive, 3);
        assert_eq!(inertia.negative, 0);
        assert_eq!(inertia.zero, 0);
        assert_eq!(factors.n, 3);
    }

    #[test]
    fn test_factorize_matches_dense() {
        // Factor a small matrix with both dense and sparse, compare inertia
        // [2 -1  0]
        // [-1 3 -1]
        // [0 -1  4]
        let m = CscMatrix::from_triplets(
            3,
            &[0, 1, 1, 2, 2],
            &[0, 0, 1, 1, 2],
            &[2.0, -1.0, 3.0, -1.0, 4.0],
        )
        .unwrap();

        // Dense factorization
        let dense_mat = m.to_dense();
        let params = make_params();
        let (_, dense_inertia) = factor(&dense_mat, &params.bk).unwrap();

        // Sparse factorization
        let sym = symbolic_factorize(&m, &SupernodeParams::default()).unwrap();
        let (_, sparse_inertia) = factorize_multifrontal(&m, &sym, &params).unwrap();

        assert_eq!(sparse_inertia, dense_inertia);
    }

    #[test]
    fn test_factorize_kkt() {
        // KKT matrix: [H A^T; A -delta*I]
        // H = [[2,0],[0,3]], A = [1,1], delta = 1e-8
        let m = CscMatrix::from_triplets(
            3,
            &[0, 1, 2, 2, 2],
            &[0, 1, 0, 1, 2],
            &[2.0, 3.0, 1.0, 1.0, -1e-8],
        )
        .unwrap();

        let sym = symbolic_factorize(&m, &SupernodeParams::default()).unwrap();
        let params = make_params();
        let (_, inertia) = factorize_multifrontal(&m, &sym, &params).unwrap();

        // Should have 2 positive (H block), 1 negative (constraint block)
        assert_eq!(inertia.positive, 2);
        assert_eq!(inertia.negative, 1);
        assert_eq!(inertia.zero, 0);
    }

    #[test]
    fn test_factorize_indefinite() {
        // Indefinite: [[1,2],[2,1]]
        let m = CscMatrix::from_triplets(2, &[0, 1, 1], &[0, 0, 1], &[1.0, 2.0, 1.0]).unwrap();

        let sym = symbolic_factorize(&m, &SupernodeParams::default()).unwrap();
        let params = make_params();
        let (_, inertia) = factorize_multifrontal(&m, &sym, &params).unwrap();

        // Eigenvalues: 3, -1 → 1 positive, 1 negative
        assert_eq!(inertia.positive, 1);
        assert_eq!(inertia.negative, 1);
        assert_eq!(inertia.zero, 0);
    }

    /// Structural goal of the β refactor: a single SymbolicFactorization
    /// is reusable across NumericParams that select different scaling
    /// strategies. The same `sym` factors twice — once with InfNorm,
    /// once with Identity — and both calls succeed and produce the
    /// expected inertia (1 positive, 2 negative for a saddle-point
    /// system with one constraint).
    #[test]
    fn factorize_multifrontal_with_two_strategies_on_one_symbolic() {
        use crate::scaling::ScalingStrategy;

        // Saddle-point KKT: [[2 0 -1], [0 2 -1], [-1 -1 0]].
        // Inertia: H = 2I_2 contributes 2 positive; constraint Schur
        // is -[-1 -1]·(I/2)·[-1 -1]^T = -1, so 1 negative.
        let m = CscMatrix::from_triplets(
            3,
            &[0, 2, 1, 2, 2],
            &[0, 0, 1, 1, 2],
            &[2.0, -1.0, 2.0, -1.0, 0.0],
        )
        .unwrap();

        let sym = symbolic_factorize(&m, &SupernodeParams::default()).unwrap();

        let infnorm = NumericParams {
            bk: BunchKaufmanParams {
                on_zero_pivot: ZeroPivotAction::ForceAccept,
                ..BunchKaufmanParams::default()
            },
            scaling: ScalingStrategy::InfNorm,
            small_leaf: Default::default(),
            profiler: None,
        };
        let identity = NumericParams {
            bk: infnorm.bk.clone(),
            scaling: ScalingStrategy::Identity,
            small_leaf: Default::default(),
            profiler: None,
        };

        let (_, i_inf) = factorize_multifrontal(&m, &sym, &infnorm).unwrap();
        let (_, i_id) = factorize_multifrontal(&m, &sym, &identity).unwrap();

        assert_eq!(i_inf.positive, 2);
        assert_eq!(i_inf.negative, 1);
        assert_eq!(i_id.positive, 2);
        assert_eq!(i_id.negative, 1);
    }

    /// 6×6 KKT for F3.2b Schur extraction tests. Same shape as the
    /// hand-built oracle in src/symbolic tests:
    /// - Non-Schur block diag(1,2,3,4) at positions 0..4
    /// - Schur block (positions 4,5):
    ///     [1.5, 0.2; 0.2, 2.5]
    /// - Coupling A_FS:
    ///     col 4 has rows {0:0.5, 2:0.7}; col 5 has rows {1:0.3, 3:0.9}
    fn small_kkt_6x6_for_schur() -> CscMatrix {
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for i in 0..4 {
            rows.push(i);
            cols.push(i);
            vals.push((i + 1) as f64);
        }
        rows.push(4);
        cols.push(0);
        vals.push(0.5);
        rows.push(4);
        cols.push(2);
        vals.push(0.7);
        rows.push(5);
        cols.push(1);
        vals.push(0.3);
        rows.push(5);
        cols.push(3);
        vals.push(0.9);
        rows.push(4);
        cols.push(4);
        vals.push(1.5);
        rows.push(5);
        cols.push(4);
        vals.push(0.2);
        rows.push(5);
        cols.push(5);
        vals.push(2.5);
        CscMatrix::from_triplets(6, &rows, &cols, &vals).unwrap()
    }

    /// Hand-computed Schur complement S = A_SS − A_FS^T A_FF^{-1} A_FS:
    ///   A_FF = diag(1, 2, 3, 4) ⇒ A_FF^{-1} = diag(1, 0.5, 1/3, 0.25)
    ///   A_FS^T A_FF^{-1} A_FS:
    ///     (4,4) = 0.5²·1 + 0.7²·(1/3) = 0.25 + 0.49/3
    ///     (4,5) = 0  (no shared row between col 4 and col 5)
    ///     (5,5) = 0.3²·0.5 + 0.9²·0.25 = 0.045 + 0.2025
    ///   S = [[1.5 − (0.25 + 0.49/3), 0.2],
    ///        [0.2, 2.5 − (0.045 + 0.2025)]]
    fn hand_computed_schur_2x2() -> [[f64; 2]; 2] {
        let s00 = 1.5 - (0.25 + 0.49 / 3.0);
        let s11 = 2.5 - (0.045 + 0.2025);
        let s01 = 0.2;
        [[s00, s01], [s01, s11]]
    }

    #[test]
    fn schur_block_matches_hand_computed_for_small_kkt() {
        let m = small_kkt_6x6_for_schur();
        let params = crate::symbolic::SupernodeParams::default();
        // ScalingStrategy::Identity is required because compute_scaling
        // is called against the original matrix in the Schur driver,
        // and the hand-computed S assumes no scaling. Default Auto on
        // a 6-row matrix would route to InfNorm and rescale entries.
        let sym = crate::symbolic::symbolic_factorize_with_schur(&m, &params, &[4, 5]).unwrap();
        let nparams = NumericParams {
            scaling: crate::scaling::ScalingStrategy::Identity,
            ..NumericParams::default()
        };
        let (_factors, _inertia, schur) =
            factorize_multifrontal_with_schur(&m, &sym, &nparams).unwrap();
        assert_eq!(schur.dim, 2);
        let expected = hand_computed_schur_2x2();
        let tol = 1e-12;
        for i in 0..2 {
            for j in 0..2 {
                let got = schur.get(i, j);
                let want = expected[i][j];
                assert!(
                    (got - want).abs() < tol,
                    "S({},{}) got {} want {} diff {}",
                    i,
                    j,
                    got,
                    want,
                    (got - want).abs()
                );
            }
        }
    }

    #[test]
    fn schur_block_full_square_storage() {
        // S(i,j) == S(j,i) at every entry — full-square storage with
        // mirror to upper triangle.
        let m = small_kkt_6x6_for_schur();
        let params = crate::symbolic::SupernodeParams::default();
        let sym = crate::symbolic::symbolic_factorize_with_schur(&m, &params, &[4, 5]).unwrap();
        let nparams = NumericParams {
            scaling: crate::scaling::ScalingStrategy::Identity,
            ..NumericParams::default()
        };
        let (_, _, schur) = factorize_multifrontal_with_schur(&m, &sym, &nparams).unwrap();
        for i in 0..schur.dim {
            for j in 0..schur.dim {
                assert!((schur.get(i, j) - schur.get(j, i)).abs() < 1e-15);
            }
        }
    }

    #[test]
    fn schur_user_order_preserved_when_reversed() {
        // Reverse user order: schur_indices = [5, 4]. Then S
        // reported has rows/cols permuted so that out(0,0) corresponds
        // to original index 5 (= S_hand(1,1)).
        let m = small_kkt_6x6_for_schur();
        let params = crate::symbolic::SupernodeParams::default();
        let sym = crate::symbolic::symbolic_factorize_with_schur(&m, &params, &[5, 4]).unwrap();
        let nparams = NumericParams {
            scaling: crate::scaling::ScalingStrategy::Identity,
            ..NumericParams::default()
        };
        let (_, _, schur) = factorize_multifrontal_with_schur(&m, &sym, &nparams).unwrap();
        let hand = hand_computed_schur_2x2();
        // Mapping: out(i,j) = hand(map(i), map(j)) with map = [1, 0]
        let map = [1usize, 0usize];
        let tol = 1e-12;
        for i in 0..2 {
            for j in 0..2 {
                let got = schur.get(i, j);
                let want = hand[map[i]][map[j]];
                assert!(
                    (got - want).abs() < tol,
                    "reversed S({},{}) got {} want {}",
                    i,
                    j,
                    got,
                    want
                );
            }
        }
    }

    #[test]
    fn schur_rejects_symbolic_without_schur_tail() {
        let m = small_kkt_6x6_for_schur();
        let params = crate::symbolic::SupernodeParams::default();
        let sym = crate::symbolic::symbolic_factorize(&m, &params).unwrap(); // no Schur
        assert_eq!(sym.is_schur_tail, None);
        let nparams = NumericParams::default();
        let r = factorize_multifrontal_with_schur(&m, &sym, &nparams);
        assert!(matches!(r, Err(FeralError::InvalidInput(_))));
    }

    /// F3.2b multi-supernode Schur tail. Builds a problem where the
    /// pre-merge symbolic phase produces multiple Schur-bearing
    /// supernodes (size > nemin and with structurally distinct row
    /// patterns); after the symbolic merge step (see
    /// `merge_schur_tail_supernodes` in `symbolic/mod.rs`), the numeric
    /// driver must accept the symbolic and produce the correct Schur
    /// block. Verified against an oracle that solves
    /// `A_FF * X = A_FS` densely and computes
    /// `S = A_SS - A_FS^T * X`.
    #[test]
    fn schur_multi_supernode_tail_matches_oracle() {
        // Two coupled subblocks A and B with their own dense Schur
        // tail, plus a tridiagonal cross-link across the entire Schur
        // set so the etree has a single Schur root (forest Schur is
        // unsupported per F3.2a).
        let half = 25usize;
        let k_each = 40usize;
        let n = 2 * half + 2 * k_each;

        let mut rows: Vec<usize> = Vec::new();
        let mut cols: Vec<usize> = Vec::new();
        let mut vals: Vec<f64> = Vec::new();
        for i in 0..n {
            rows.push(i);
            cols.push(i);
            vals.push(2.0 + i as f64);
        }
        for i in 0..half {
            for s in 0..k_each {
                let j = 2 * half + s;
                rows.push(j);
                cols.push(i);
                vals.push(0.1);
            }
        }
        for i in half..2 * half {
            for s in 0..k_each {
                let j = 2 * half + k_each + s;
                rows.push(j);
                cols.push(i);
                vals.push(0.1);
            }
        }
        for s in 0..k_each {
            for t in 0..s {
                rows.push(2 * half + s);
                cols.push(2 * half + t);
                vals.push(0.05);
                rows.push(2 * half + k_each + s);
                cols.push(2 * half + k_each + t);
                vals.push(0.05);
            }
        }
        for s in (2 * half + 1)..n {
            rows.push(s);
            cols.push(s - 1);
            vals.push(0.03);
        }
        let m = CscMatrix::from_triplets(n, &rows, &cols, &vals).unwrap();
        let schur: Vec<usize> = (2 * half..n).collect();
        let n_schur = schur.len();

        let params = crate::symbolic::SupernodeParams::default();
        let sym = crate::symbolic::symbolic_factorize_with_schur(&m, &params, &schur).unwrap();
        let nparams = NumericParams {
            scaling: crate::scaling::ScalingStrategy::Identity,
            ..NumericParams::default()
        };
        let (_, _, schur_block) = factorize_multifrontal_with_schur(&m, &sym, &nparams).unwrap();
        assert_eq!(schur_block.dim, n_schur);

        // Build A as dense and the f-index list.
        let mut is_schur = vec![false; n];
        for &i in &schur {
            is_schur[i] = true;
        }
        let f_indices: Vec<usize> = (0..n).filter(|i| !is_schur[*i]).collect();
        let nf = f_indices.len();
        let mut f_inv = vec![usize::MAX; n];
        for (k, &i) in f_indices.iter().enumerate() {
            f_inv[i] = k;
        }
        let mut a = vec![0.0f64; n * n];
        for j in 0..n {
            for k in m.col_ptr[j]..m.col_ptr[j + 1] {
                let i = m.row_idx[k];
                a[j * n + i] = m.values[k];
                if i != j {
                    a[i * n + j] = m.values[k];
                }
            }
        }

        // Factor A_FF (sparse).
        let mut tr = (Vec::new(), Vec::new(), Vec::new());
        for j in 0..n {
            if is_schur[j] {
                continue;
            }
            for k in m.col_ptr[j]..m.col_ptr[j + 1] {
                let i = m.row_idx[k];
                if !is_schur[i] {
                    tr.0.push(f_inv[i]);
                    tr.1.push(f_inv[j]);
                    tr.2.push(m.values[k]);
                }
            }
        }
        let a_ff = CscMatrix::from_triplets(nf, &tr.0, &tr.1, &tr.2).unwrap();
        let sym_ff = crate::symbolic::symbolic_factorize(&a_ff, &params).unwrap();
        let (factors_ff, _) = factorize_multifrontal(&a_ff, &sym_ff, &nparams).unwrap();

        // S = A_SS - A_FS^T A_FF^{-1} A_FS via column-by-column solve.
        let mut s_ref = vec![0.0f64; n_schur * n_schur];
        for (si, &i) in schur.iter().enumerate() {
            for (sj, &j) in schur.iter().enumerate() {
                s_ref[sj * n_schur + si] = a[j * n + i];
            }
        }
        for (sj, &j) in schur.iter().enumerate() {
            let mut rhs = vec![0.0f64; nf];
            for &fi in &f_indices {
                rhs[f_inv[fi]] = a[j * n + fi];
            }
            let x = crate::numeric::solve::solve_sparse(&factors_ff, &rhs).unwrap();
            for (si, &i) in schur.iter().enumerate() {
                let mut acc = 0.0;
                for &fi in &f_indices {
                    acc += a[i * n + fi] * x[f_inv[fi]];
                }
                s_ref[sj * n_schur + si] -= acc;
            }
        }

        let mut max_rel = 0.0f64;
        for sj in 0..n_schur {
            for si in 0..n_schur {
                let want = s_ref[sj * n_schur + si];
                let got = schur_block.get(si, sj);
                let denom = want.abs().max(1e-14);
                let rel = (got - want).abs() / denom;
                if rel > max_rel {
                    max_rel = rel;
                }
            }
        }
        assert!(
            max_rel < 1e-10,
            "Schur block max relative error {} exceeds 1e-10",
            max_rel
        );
    }
}
