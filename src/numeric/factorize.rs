use crate::dense::factor::{factor_frontal, BunchKaufmanParams, FrontalFactors};
#[cfg(test)]
use crate::dense::factor::factor;
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
}

/// Factor data for a single supernode.
#[derive(Debug)]
pub struct NodeFactors {
    /// First column index (in permuted numbering).
    pub first_col: usize,
    /// Number of eliminated columns.
    pub ncol: usize,
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
    let permuted = permute_csc_values(matrix, &symbolic.perm, &symbolic.perm_inv);

    // Full symmetric pattern for correct row index computation
    let full_pattern = permuted.symmetric_pattern();

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
                nrow: 0,
                row_indices: Vec::new(),
                frontal_factors: FrontalFactors {
                    nrow: 0, ncol: 0, l: Vec::new(),
                    d_diag: Vec::new(), d_subdiag: Vec::new(),
                    perm: Vec::new(), perm_inv: Vec::new(),
                    contrib: Vec::new(), contrib_dim: 0,
                    inertia: Inertia { positive: 0, negative: 0, zero: 0 },
                    needs_refinement: false,
                },
                inertia: Inertia { positive: 0, negative: 0, zero: 0 },
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
        // Each CSC entry (row, col) with row >= col represents A(row,col).
        // This entry should be assembled at the frontal that eliminates
        // column min(row,col) = col. We assemble it here if col is one of
        // our eliminated columns.
        //
        // Additionally, entries where `row` is one of our eliminated columns
        // (but `col` is not) represent A(col, row) by symmetry and should
        // also be assembled here.
        let mut frontal = SymmetricMatrix::zeros(actual_nrow);

        // Determine which global columns are eliminated at this supernode
        let mut is_eliminated = vec![false; n];
        for local_j in 0..ncol {
            is_eliminated[row_indices[local_j]] = true;
        }

        // Scan all CSC entries
        for col in 0..n {
            for k in permuted.col_ptr[col]..permuted.col_ptr[col + 1] {
                let row = permuted.row_idx[k];
                // Assemble if col OR row is one of our eliminated columns
                let col_elim = is_eliminated[col];
                let row_elim = is_eliminated[row];
                if !col_elim && !row_elim {
                    continue;
                }
                let local_col = row_map[col];
                let local_row = row_map[row];
                if local_col == usize::MAX || local_row == usize::MAX {
                    continue;
                }
                let val = permuted.values[k];
                // Place in the frontal's lower triangle
                if local_row >= local_col {
                    frontal.set(local_row, local_col, frontal.get(local_row, local_col) + val);
                } else {
                    frontal.set(local_col, local_row, frontal.get(local_col, local_row) + val);
                }
            }
        }

        // Clean up
        for local_j in 0..ncol {
            is_eliminated[row_indices[local_j]] = false;
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
        let ff = factor_frontal(&frontal, ncol, params)?;

        // Extract what we need before moving ff
        let node_inertia = ff.inertia.clone();
        let node_needs_ref = ff.needs_refinement;

        // Step 4: Store contribution block (Schur complement) for parent
        if ff.contrib_dim > 0 {
            contrib_blocks[snode_idx] = Some(ContribBlock {
                row_indices: row_indices[ncol..].to_vec(),
                data: ff.contrib.clone(),
                dim: ff.contrib_dim,
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
        },
        total_inertia,
    ))
}

/// Permute a CSC matrix: compute the lower triangle of P·A·Pᵀ.
fn permute_csc_values(
    matrix: &CscMatrix,
    _perm: &[usize],
    perm_inv: &[usize],
) -> CscMatrix {
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
        .expect("permute_csc_values: failed to build CSC")
}

/// Build row indices for a frontal matrix.
///
/// Uses the full symmetric pattern to find all rows connected to
/// the eliminated columns, plus rows from child contribution blocks.
fn build_row_indices(
    snode: &crate::symbolic::supernode::Supernode,
    full_pattern: &crate::sparse::csc::CscPattern,
    contrib_blocks: &[Option<ContribBlock>],
) -> Vec<usize> {
    let ncol = snode.ncol();
    let first_col = snode.first_col;

    let mut rows = std::collections::BTreeSet::new();

    // All rows connected to eliminated columns via the full symmetric pattern
    // (includes both lower and upper triangle connections)
    for j in first_col..first_col + ncol {
        for k in full_pattern.col_ptr[j]..full_pattern.col_ptr[j + 1] {
            rows.insert(full_pattern.row_idx[k]);
        }
    }

    // Row indices from child contribution blocks
    for &child_idx in &snode.children {
        if let Some(contrib) = &contrib_blocks[child_idx] {
            for &row in &contrib.row_indices {
                rows.insert(row);
            }
        }
    }

    rows.into_iter().collect()
}

/// Contribution block from a child supernode.
#[derive(Debug)]
struct ContribBlock {
    /// Row indices of the contribution block (global, sorted).
    row_indices: Vec<usize>,
    /// Dense symmetric matrix data (lower triangle, column-major).
    /// Dimension: row_indices.len() × row_indices.len()
    data: Vec<f64>,
    /// Dimension of the contribution block.
    dim: usize,
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
        let m = CscMatrix::from_triplets(
            3,
            &[0, 1, 2],
            &[0, 1, 2],
            &[2.0, 3.0, 5.0],
        )
        .unwrap();

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
        let m = CscMatrix::from_triplets(
            2,
            &[0, 1, 1],
            &[0, 0, 1],
            &[1.0, 2.0, 1.0],
        )
        .unwrap();

        let sym = symbolic_factorize(&m, &SupernodeParams::default()).unwrap();
        let params = make_params();
        let (_, inertia) = factorize_multifrontal(&m, &sym, &params).unwrap();

        // Eigenvalues: 3, -1 → 1 positive, 1 negative
        assert_eq!(inertia.positive, 1);
        assert_eq!(inertia.negative, 1);
        assert_eq!(inertia.zero, 0);
    }
}
