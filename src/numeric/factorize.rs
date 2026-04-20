#[cfg(test)]
use crate::dense::factor::factor;
use crate::dense::factor::{factor_frontal, BunchKaufmanParams, FrontalFactors};
use crate::dense::matrix::SymmetricMatrix;
use crate::error::FeralError;
use crate::inertia::Inertia;
use crate::scaling::{compute_scaling, ScalingStrategy};
use crate::sparse::csc::CscMatrix;
use crate::symbolic::SymbolicFactorization;

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
}

impl NumericParams {
    /// Construct a `NumericParams` from a `BunchKaufmanParams`,
    /// using the default scaling strategy. Convenience for
    /// callers that only customize BK behavior.
    pub fn with_bk(bk: BunchKaufmanParams) -> Self {
        Self {
            bk,
            scaling: ScalingStrategy::default(),
        }
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
}

impl FactorWorkspace {
    /// Construct an empty workspace. Equivalent to `default()`.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Gate predicate for the D.3 dense fast-path.
///
/// Returns `true` when the input satisfies `n ≤ N_MAX` and
/// `nnz_lower / (n * (n + 1) / 2) ≥ ρ_MIN`. The density threshold
/// is expressed as the integer inequality
/// `nnz_lower * ρ_DEN ≥ n * (n + 1) / 2 * ρ_NUM` so the check costs
/// a handful of integer ops with no division or FP.
///
/// Authoritative entry point for the gate; callers must not
/// roll their own. Thresholds may be tuned post-measurement
/// (see `dev/plans/sparse-tail-d3.md` stage 2).
///
/// Thresholds (`N_MAX = 128`, `ρ_MIN = 1/4`) are initial values from
/// the research note `dev/research/sparse-tail-d3-d4-2026-04-19.md`.
/// The stage-2 measurement sweep in `dev/plans/sparse-tail-d3.md` may
/// tune them; update both `N_MAX` and the numerator/denominator pair
/// together if tuned.
#[inline]
pub fn should_use_dense_fast_path(n: usize, nnz_lower: usize) -> bool {
    const N_MAX: usize = 128;
    // ρ_MIN = ρ_NUM / ρ_DEN = 1/4 = 0.25
    const RHO_NUM: usize = 1;
    const RHO_DEN: usize = 4;
    if n == 0 || n > N_MAX {
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
    let mut sym = matrix.to_dense();
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
    let ff = factor_frontal(&sym, n, false, &params.bk)?;

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
/// On a gate hit `ws` is pass-through — the dense path allocates
/// its own dense buffer (pooling it is a follow-up; see
/// `dev/plans/sparse-tail-d3.md`).
pub fn factorize_multifrontal_with_workspace(
    matrix: &CscMatrix,
    symbolic: &SymbolicFactorization,
    params: &NumericParams,
    ws: &mut FactorWorkspace,
) -> Result<(SparseFactors, Inertia), FeralError> {
    if should_use_dense_fast_path(matrix.n, matrix.row_idx.len()) {
        return dense_fast_factor(matrix, params);
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
    let (scaling_user, scaling_info) = compute_scaling(matrix, &params.scaling)?;
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

    // Process supernodes in postorder (children before parents)
    for snode_idx in 0..n_snodes {
        let snode = &symbolic.supernodes[snode_idx];
        let own_ncol = snode.ncol();
        let nrow = snode.nrow;

        if nrow == 0 || own_ncol == 0 {
            node_factors.push(NodeFactors {
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
                    zero_tol: params.bk.zero_tol,
                    zero_tol_2x2: params.bk.zero_tol_2x2,
                },
                inertia: Inertia {
                    positive: 0,
                    negative: 0,
                    zero: 0,
                },
            });
            continue;
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

        // Build the row indices for this frontal. With delays the layout is
        // [own native cols (own_ncol) | delayed cols from children (n_delayed_in) | trailing rows].
        let row_indices = build_row_indices(
            snode,
            &full_pattern,
            &contrib_blocks,
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

        // Populate the pooled `ws.row_map`. Invariant on entry to
        // this block: every entry is `usize::MAX` (either from the
        // top-of-function reset or from the end-of-loop clear of the
        // previous iteration). We write exactly `row_indices.len()`
        // entries here; the mirror-clear at the end of the iteration
        // restores the invariant.
        for (local, &global) in row_indices.iter().enumerate() {
            ws.row_map[global] = local;
        }

        // Step 1: Assemble original matrix entries into frontal.
        // Scan only the supernode's own native columns' CSC entries. The
        // delayed columns at positions `[own_ncol..expanded_ncol)` have
        // already had their raw A values baked into each child's contrib
        // block during the child's own Step 1; scanning them here would
        // double-count. The CSC stores lower-triangle entries (row >= col),
        // so each entry A(row, col) is found by scanning column col.
        // Entries where our columns appear as ROWS arrive via child
        // contribution blocks.
        //
        // Phase 2.2.1 Step 6: Apply MC64 symmetric scaling in-place
        // as `D · A · D` where `D = diag(scaling_pivot_order)`. The
        // permuted CSC produced above is indexed in pivot positions,
        // and `scaling_pivot_order` is also in pivot indexing (see
        // src/symbolic/mod.rs and the Step 5 commit 67954d9), so the
        // lookup is direct — no indirection through `perm`. Diagonal
        // entries receive `s[i]^2`; off-diagonal entries receive
        // `s[i] * s[j]`. Identity strategy fills the vector with 1.0,
        // so this multiply is a no-op when scaling is disabled.
        let scaling = &scaling_pivot_order;
        // Pool the frontal's `data` Vec via the workspace. `std::mem::take`
        // hands the Vec to us (leaving `ws.frontal_values` empty); we
        // clear-then-resize to `actual_nrow * actual_nrow` zeros, wrap
        // it in a `SymmetricMatrix`, use it, and return the buffer to
        // the workspace before falling out of the iteration. If
        // `factor_frontal` errors out the buffer is dropped along with
        // the `frontal` local — acceptable because an error aborts the
        // whole call anyway.
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

        // Step 2: Assemble child contribution blocks (extend-add). The
        // child's `row_indices` cover both its delayed columns (top) and
        // its trailing rows (bottom); `row_map` maps both classes to
        // their new positions in the parent frontal — delayed cols land
        // in `[own_ncol..expanded_ncol)` and trailing rows land
        // in `[expanded_ncol..actual_nrow)`.
        for &child_idx in &snode.children {
            if let Some(contrib) = contrib_blocks[child_idx].take() {
                extend_add(&contrib, &ws.row_map, &mut frontal);
            }
        }

        // Step 3: Factor the frontal, eliminating up to `expanded_ncol`
        // fully-summed columns. Pivot search is restricted to the first
        // `expanded_ncol` rows. Rows `expanded_ncol..actual_nrow` are
        // never swapped, preserving contribution block row ordering.
        //
        // Phase 2.3 Step 5: non-root supernodes pass `may_delay = true`
        // so the BK kernel can break on the first un-pivotable column
        // and carry the leftover fully-summed columns forward as
        // delayed pivots in the contribution block's top-left region.
        // Root supernodes force-accept (via `ZeroPivotAction::ForceAccept`)
        // because they have no ancestor to absorb a delay.
        let may_delay = !is_root[snode_idx];
        let ff = factor_frontal(&frontal, expanded_ncol, may_delay, &params.bk)?;

        // Return the frontal's data buffer to the pool. `factor_frontal`
        // takes its input by `&SymmetricMatrix` and copies into a local
        // work array, so `frontal.data` is still the zeroed buffer we
        // allocated above. Reclaim it for the next supernode's frontal.
        ws.frontal_values = frontal.data;

        // Extract what we need before moving ff
        let node_inertia = ff.inertia.clone();
        let node_needs_ref = ff.needs_refinement;
        let node_nelim = ff.nelim;
        let node_n_delayed = ff.n_delayed;

        // Step 4: Store contribution block for parent.
        //
        // The kernel's contrib block rows `0..cdim` correspond to
        // kernel positions `[nelim..nrow)`. The first `n_delayed`
        // of those are the un-eliminated fully-summed columns
        // (post-BK-swap ordering); the remainder are the unchanged
        // trailing rows. Unpermute with `ff.perm` to recover each
        // contrib row's global index:
        //
        //     contrib_row_indices[cj] = row_indices[ff.perm[nelim + cj]]
        //
        // Kernel swaps only touch positions `[0..expanded_ncol)`, so
        // `ff.perm[i] = i` for `i >= expanded_ncol` — the formula is
        // uniformly correct for delayed and trailing positions. When
        // `n_delayed == 0` it reduces to the pre-Phase-2.3 slice
        // `row_indices[expanded_ncol..]`.
        if ff.contrib_dim > 0 {
            let cdim = ff.contrib_dim;
            let mut contrib_row_indices = Vec::with_capacity(cdim);
            for cj in 0..cdim {
                contrib_row_indices.push(row_indices[ff.perm[node_nelim + cj]]);
            }
            contrib_blocks[snode_idx] = Some(ContribBlock {
                row_indices: contrib_row_indices,
                data: ff.contrib.clone(),
                dim: cdim,
                n_delayed: node_n_delayed,
            });
        }

        // Accumulate inertia
        total_inertia.positive += node_inertia.positive;
        total_inertia.negative += node_inertia.negative;
        total_inertia.zero += node_inertia.zero;

        if node_needs_ref {
            needs_refinement = true;
        }

        // Clear the pooled row map. This restores the
        // all-`usize::MAX` invariant for the next iteration (and, via
        // the last iteration, for the next `factorize_multifrontal*`
        // call that shares this workspace).
        for &global in &row_indices {
            ws.row_map[global] = usize::MAX;
        }

        node_factors.push(NodeFactors {
            first_col: snode.first_col,
            ncol: expanded_ncol,
            nelim: node_nelim,
            n_delayed_in,
            nrow: actual_nrow,
            row_indices,
            frontal_factors: ff,
            inertia: node_inertia,
        });
    }

    Ok((
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
        },
        total_inertia,
    ))
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
        };
        let identity = NumericParams {
            bk: infnorm.bk.clone(),
            scaling: ScalingStrategy::Identity,
        };

        let (_, i_inf) = factorize_multifrontal(&m, &sym, &infnorm).unwrap();
        let (_, i_id) = factorize_multifrontal(&m, &sym, &identity).unwrap();

        assert_eq!(i_inf.positive, 2);
        assert_eq!(i_inf.negative, 1);
        assert_eq!(i_id.positive, 2);
        assert_eq!(i_id.negative, 1);
    }
}
