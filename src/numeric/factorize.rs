#[cfg(test)]
use crate::dense::factor::factor;
use crate::dense::factor::{factor_frontal, BunchKaufmanParams, FrontalFactors};
use crate::dense::matrix::SymmetricMatrix;
use crate::error::FeralError;
use crate::inertia::Inertia;
use crate::sparse::csc::CscMatrix;
use crate::symbolic::SymbolicFactorization;

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
    /// children during parent assembly. Populated by Step 5 (currently
    /// always zero because `build_row_indices` has not yet been
    /// taught to expand the fully-summed count).
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
pub fn factorize_multifrontal(
    matrix: &CscMatrix,
    symbolic: &SymbolicFactorization,
    params: &BunchKaufmanParams,
) -> Result<(SparseFactors, Inertia), FeralError> {
    let n = symbolic.n;
    let n_snodes = symbolic.supernodes.len();

    // Permute the matrix values into the new ordering
    let permuted = permute_csc_values(matrix, &symbolic.perm, &symbolic.perm_inv)?;

    // Full symmetric pattern for correct row index computation
    let full_pattern = permuted.symmetric_pattern();

    // Transpose is precomputed but currently unused — kept for future
    // amap-based assembly optimization.
    let _ = build_csc_transpose(&permuted);

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
        let ncol = snode.ncol();
        let nrow = snode.nrow;

        if nrow == 0 || ncol == 0 {
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
                    zero_tol: params.zero_tol,
                    zero_tol_2x2: params.zero_tol_2x2,
                },
                inertia: Inertia {
                    positive: 0,
                    negative: 0,
                    zero: 0,
                },
            });
            continue;
        }

        // Build the row indices for this frontal
        let row_indices = build_row_indices(snode, &full_pattern, &contrib_blocks);
        let actual_nrow = row_indices.len();

        // Build a map from global row index to local frontal row index
        let mut row_map = vec![usize::MAX; n];
        for (local, &global) in row_indices.iter().enumerate() {
            row_map[global] = local;
        }

        // Step 1: Assemble original matrix entries into frontal.
        // Scan only our eliminated columns' CSC entries. The CSC stores
        // lower-triangle entries (row >= col), so each entry A(row, col)
        // is found by scanning column col. Entries where our columns appear
        // as ROWS arrive via child contribution blocks.
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
        debug_assert_eq!(symbolic.scaling_pivot_order.len(), symbolic.n);
        let scaling = &symbolic.scaling_pivot_order;
        let mut frontal = SymmetricMatrix::zeros(actual_nrow);
        for (local_j, &gj) in row_indices[..ncol].iter().enumerate() {
            let s_j = scaling[gj];
            for k in permuted.col_ptr[gj]..permuted.col_ptr[gj + 1] {
                let gi = permuted.row_idx[k];
                let local_i = row_map[gi];
                if local_i != usize::MAX {
                    let val = permuted.values[k] * scaling[gi] * s_j;
                    frontal.set(local_i, local_j, val);
                }
            }
        }

        // Step 2: Assemble child contribution blocks (extend-add)
        for &child_idx in &snode.children {
            if let Some(contrib) = contrib_blocks[child_idx].take() {
                extend_add(&contrib, &row_map, &mut frontal);
            }
        }

        // Step 3: Factor the frontal, eliminating only ncol columns.
        // Pivot search is restricted to the first ncol rows. Rows ncol..nrow
        // are never swapped, preserving contribution block row ordering.
        //
        // Phase 2.3 Step 4: `may_delay` stays `false` at this step.
        // The original plan's assumption — "only `pivot_threshold > 0`
        // can trigger delays" — was wrong: the 2×2 Duff-Reid growth
        // bound and the det-floor stability checks reject pivots
        // independently of the column-relative threshold and DO
        // produce the `Delayed` outcome under `may_delay = true`.
        // Enabling delays before Step 5's parent-side assembly lands
        // causes bench sparse validation (SWOPF et al) to hit the
        // `debug_assert(n_delayed == 0)` on a legitimate growth-bound
        // rejection. The `is_root` computation and flag flip to
        // `!is_root[snode_idx]` are deferred to Step 5's commit,
        // which will also correctly unpermute and scatter the
        // delayed columns via `ContribBlock::n_delayed`.
        let ff = factor_frontal(&frontal, ncol, false, params)?;

        // Extract what we need before moving ff
        let node_inertia = ff.inertia.clone();
        let node_needs_ref = ff.needs_refinement;
        let node_nelim = ff.nelim;
        let node_n_delayed = ff.n_delayed;

        // Step 4: Store contribution block for parent.
        //
        // Phase 2.3 invariant (Step 4 level): with `pivot_threshold = 0.0`
        // the column-relative test never fails, so `n_delayed == 0` and
        // the contrib block is the pure Schur complement over trailing
        // rows — the pre-Phase-2.3 layout. Step 5 replaces this branch
        // with a delayed-column-aware scatter that (a) unpermutes the
        // kernel's internal BK swaps within 0..ncol to recover each
        // delayed column's original global index and (b) lays the
        // delayed fully-summed columns at the head of `row_indices`.
        // Fail loudly if a delay sneaks through before Step 5 is in
        // place — a silent misassembly would be much worse than a
        // debug assert.
        debug_assert_eq!(
            node_n_delayed, 0,
            "Step 5 (parent-side delay assembly) must land before delays can fire"
        );
        if ff.contrib_dim > 0 {
            contrib_blocks[snode_idx] = Some(ContribBlock {
                row_indices: row_indices[ncol..].to_vec(),
                data: ff.contrib.clone(),
                dim: ff.contrib_dim,
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

        // Clear the row map
        for &global in &row_indices {
            row_map[global] = usize::MAX;
        }

        node_factors.push(NodeFactors {
            first_col: snode.first_col,
            ncol,
            nelim: node_nelim,
            // Step 5 will populate this once `build_row_indices`
            // incorporates each child's `n_delayed` into the
            // expanded fully-summed column count.
            n_delayed_in: 0,
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
            // Phase 2.2.1 Step 6: propagate the user-order scaling
            // vector into the numeric factors so Step 7 (solve-side
            // pre/post-scaling) can reach it without a back-pointer
            // to `SymbolicFactorization`. Solve operates at the
            // user API boundary, so it needs user-order indexing,
            // not the pivot-order cache used at assembly time.
            scaling: symbolic.scaling.clone(),
            scaling_info: symbolic.scaling_info.clone(),
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
/// Returns indices with eliminated columns FIRST (positions 0..ncol),
/// followed by non-eliminated rows sorted. This ordering is required
/// because factor_frontal treats positions 0..ncol as fully-summed
/// columns and ncol..nrow as the contribution block region.
fn build_row_indices(
    snode: &crate::symbolic::supernode::Supernode,
    full_pattern: &crate::sparse::csc::CscPattern,
    contrib_blocks: &[Option<ContribBlock>],
) -> Vec<usize> {
    let ncol = snode.ncol();
    let first_col = snode.first_col;

    let mut all_rows = std::collections::BTreeSet::new();

    // All rows connected to eliminated columns via the full symmetric pattern
    for j in first_col..first_col + ncol {
        for k in full_pattern.col_ptr[j]..full_pattern.col_ptr[j + 1] {
            all_rows.insert(full_pattern.row_idx[k]);
        }
    }

    // Row indices from child contribution blocks
    for &child_idx in &snode.children {
        if let Some(contrib) = &contrib_blocks[child_idx] {
            for &row in &contrib.row_indices {
                all_rows.insert(row);
            }
        }
    }

    // Build result: eliminated columns first, then non-eliminated rows
    let elim_cols: Vec<usize> = (first_col..first_col + ncol).collect();
    let non_elim: Vec<usize> = all_rows
        .iter()
        .copied()
        .filter(|&r| r < first_col || r >= first_col + ncol)
        .collect();

    let mut result = elim_cols;
    result.extend(non_elim);
    result
}

/// Build the transpose of a lower-triangle CSC matrix (excluding diagonal).
///
/// For each off-diagonal entry (row, col) with row > col in the CSC,
/// records that row `row` has an entry at column `col`.
///
/// Returns (trans_ptr, trans_col, trans_src) where:
/// - trans_ptr[k]..trans_ptr[k+1] = range of entries for row k
/// - trans_col[idx] = the column c < k
/// - trans_src[idx] = position in the original CSC values array
fn build_csc_transpose(csc: &CscMatrix) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let n = csc.n;

    // Count entries per row (excluding diagonal)
    let mut counts = vec![0usize; n];
    for col in 0..n {
        for k in csc.col_ptr[col]..csc.col_ptr[col + 1] {
            let row = csc.row_idx[k];
            if row > col {
                counts[row] += 1;
            }
        }
    }

    // Build ptr
    let mut trans_ptr = vec![0usize; n + 1];
    for i in 0..n {
        trans_ptr[i + 1] = trans_ptr[i] + counts[i];
    }
    let total = trans_ptr[n];
    let mut trans_col = vec![0usize; total];
    let mut trans_src = vec![0usize; total];

    // Fill entries
    let mut offsets = trans_ptr[..n].to_vec();
    for col in 0..n {
        for k in csc.col_ptr[col]..csc.col_ptr[col + 1] {
            let row = csc.row_idx[k];
            if row > col {
                let pos = offsets[row];
                trans_col[pos] = col;
                trans_src[pos] = k;
                offsets[row] += 1;
            }
        }
    }

    (trans_ptr, trans_col, trans_src)
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
    /// by Step 5's parent-side delay-aware assembly; until that lands
    /// no reader touches the field, hence the allow.
    #[allow(dead_code)]
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

    fn make_params() -> BunchKaufmanParams {
        BunchKaufmanParams {
            on_zero_pivot: ZeroPivotAction::ForceAccept,
            ..BunchKaufmanParams::default()
        }
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
        let (_, dense_inertia) = factor(&dense_mat, &params).unwrap();

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
}
