use crate::dense::factor::{factor, BunchKaufmanParams};
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
    /// The dense factors from BK factorization of this frontal.
    /// Stored as the full factor output from the dense kernel.
    pub dense_factors: crate::dense::factor::Factors,
    /// Inertia of this node.
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
                dense_factors: empty_factors(),
                inertia: Inertia { positive: 0, negative: 0, zero: 0 },
            });
            continue;
        }

        // Build the row indices for this frontal
        let row_indices = build_row_indices(snode, &permuted, &symbolic.supernodes, &contrib_blocks);
        let actual_nrow = row_indices.len();

        // Build a map from global row index to local frontal row index
        let mut row_map = vec![usize::MAX; n];
        for (local, &global) in row_indices.iter().enumerate() {
            row_map[global] = local;
        }

        // Step 1: Assemble original matrix entries into frontal
        let mut frontal = SymmetricMatrix::zeros(actual_nrow);
        assemble_original(&permuted, &row_indices, &row_map, ncol, &mut frontal);

        // Step 2: Assemble child contribution blocks (extend-add)
        for &child_idx in &snode.children {
            if let Some(contrib) = contrib_blocks[child_idx].take() {
                extend_add(&contrib, &row_map, &mut frontal);
            }
        }

        // Step 3: Factor the frontal with the dense BK kernel
        let (factors, inertia) = factor(&frontal, params)?;

        // Step 4: Extract contribution block
        if actual_nrow > ncol {
            let contrib = extract_contribution(&factors, ncol, actual_nrow, &row_indices);
            contrib_blocks[snode_idx] = Some(contrib);
        }

        // Accumulate inertia
        total_inertia.positive += inertia.positive;
        total_inertia.negative += inertia.negative;
        total_inertia.zero += inertia.zero;

        if inertia.zero > 0 {
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
            dense_factors: factors,
            inertia,
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

/// Build row indices for a frontal matrix by combining:
/// - The eliminated columns of this supernode
/// - Row indices from the original matrix for those columns
/// - Row indices from child contribution blocks
fn build_row_indices(
    snode: &crate::symbolic::supernode::Supernode,
    permuted: &CscMatrix,
    all_snodes: &[crate::symbolic::supernode::Supernode],
    contrib_blocks: &[Option<ContribBlock>],
) -> Vec<usize> {
    let ncol = snode.ncol();
    let first_col = snode.first_col;

    let mut rows = std::collections::BTreeSet::new();

    // The eliminated columns themselves
    for j in first_col..first_col + ncol {
        rows.insert(j);
    }

    // Row indices from the original matrix (lower triangle)
    for j in first_col..first_col + ncol {
        for k in permuted.col_ptr[j]..permuted.col_ptr[j + 1] {
            rows.insert(permuted.row_idx[k]);
        }
    }

    // Row indices from child contribution blocks
    for &child_idx in &snode.children {
        if let Some(contrib) = &contrib_blocks[child_idx] {
            for &row in &contrib.row_indices {
                rows.insert(row);
            }
        }
        // Also add rows from child supernodes' non-eliminated rows
        let child = &all_snodes[child_idx];
        for &row in &child.row_indices {
            if row >= first_col + ncol || row < first_col {
                // This row is outside our eliminated columns — it may
                // need to be in our frontal
            }
        }
    }


    rows.into_iter().collect()
}

/// Assemble original matrix entries into the frontal matrix.
///
/// Scans the permuted CSC matrix for all entries where both row and column
/// are in this frontal's row index set. Places each entry in the lower
/// triangle of the frontal.
fn assemble_original(
    permuted: &CscMatrix,
    _row_indices: &[usize],
    row_map: &[usize],
    _ncol: usize,
    frontal: &mut SymmetricMatrix,
) {
    let n = permuted.n;

    for col in 0..n {
        let local_col = row_map[col];
        if local_col == usize::MAX {
            continue;
        }
        for k in permuted.col_ptr[col]..permuted.col_ptr[col + 1] {
            let row = permuted.row_idx[k];
            let local_row = row_map[row];
            if local_row == usize::MAX {
                continue;
            }
            let val = permuted.values[k];

            // CSC entry is (row, col) with row >= col (lower triangle).
            // Place in the frontal's lower triangle.
            if local_row >= local_col {
                frontal.set(local_row, local_col, frontal.get(local_row, local_col) + val);
            } else {
                frontal.set(local_col, local_row, frontal.get(local_col, local_row) + val);
            }
        }
    }
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

/// Extract the contribution block (Schur complement) from the factored frontal.
///
/// After BK factorization of an nrow×nrow frontal with ncol pivots,
/// the contribution block is the (nrow-ncol)×(nrow-ncol) Schur complement
/// in the lower-right corner.
///
/// The Schur complement S = A22 - A21 * D^{-1} * L^{-1} * ... but we can
/// extract it more simply: the dense factor already computed the full
/// factorization. The contribution block entries are:
///   S(i,j) = frontal(ncol+i, ncol+j) - sum of updates from eliminated pivots
///
/// Actually, the simplest correct approach: reconstruct the Schur complement
/// from the factored frontal. The L and D factors contain the full information.
/// S = A22 - L21 * D * L21^T where L21 is the off-diagonal part of L.
fn extract_contribution(
    factors: &crate::dense::factor::Factors,
    ncol: usize,
    nrow: usize,
    row_indices: &[usize],
) -> ContribBlock {
    let cdim = nrow - ncol;
    let mut data = vec![0.0f64; cdim * cdim];

    // The Schur complement is stored in the lower-right (nrow-ncol)×(nrow-ncol)
    // block of the factored matrix. After LDL^T factorization of the full
    // frontal, the trailing block contains A22 - L21*D*L21^T.
    //
    // In our dense factor, the work array contains the modified matrix.
    // The Schur complement is the unfactored trailing block.
    // We can access it through the factors' internal data.
    //
    // Since the dense factor works on the full matrix and we need the
    // Schur complement, we reconstruct it: S_ij = original_A_ij - sum of
    // updates from the first ncol pivots.
    //
    // Simpler approach: the dense BK kernel modifies the matrix in-place.
    // After factoring ncol pivots, the remaining (nrow-ncol)×(nrow-ncol)
    // block in the work array IS the Schur complement.
    // Access it from factors.work_data (the modified matrix storage).

    // The factors store L in the lower triangle and D on the diagonal.
    // The Schur complement is in the unfactored trailing block of the
    // work matrix. We can reconstruct it by looking at what's left.
    //
    // For now, compute S = A22 - L21 * D * L21^T explicitly.
    // L21 is rows ncol..nrow, cols 0..ncol of the L factor.
    // D is the block diagonal from the first ncol pivots.

    // Get L21 and D from the factors
    let l = &factors.l;
    let d_diag = &factors.d_diag;
    let d_subdiag = &factors.d_subdiag;
    let n_full = factors.n;

    // L21: rows ncol..nrow, cols 0..ncol (in permuted order)
    // Compute L21 * D * L21^T
    for cj in 0..cdim {
        for ci in cj..cdim {
            let mut s = 0.0;
            let pi = ncol + ci;
            let pj = ncol + cj;

            // Walk through pivots
            let mut k = 0;
            while k < ncol {
                if k + 1 < ncol && d_subdiag[k] != 0.0 {
                    // 2×2 pivot at (k, k+1)
                    let d11 = d_diag[k];
                    let d21 = d_subdiag[k];
                    let d22 = d_diag[k + 1];

                    let l_i_k = l[k * n_full + pi];
                    let l_i_k1 = l[(k + 1) * n_full + pi];
                    let l_j_k = l[k * n_full + pj];
                    let l_j_k1 = l[(k + 1) * n_full + pj];

                    // (L21 * D)_ik = L_ik * D_kk + L_i(k+1) * D_(k+1)k
                    let ld_i_k = l_i_k * d11 + l_i_k1 * d21;
                    let ld_i_k1 = l_i_k * d21 + l_i_k1 * d22;

                    s += ld_i_k * l_j_k + ld_i_k1 * l_j_k1;
                    k += 2;
                } else {
                    // 1×1 pivot at k
                    let d_k = d_diag[k];
                    let l_ik = l[k * n_full + pi];
                    let l_jk = l[k * n_full + pj];
                    s += l_ik * d_k * l_jk;
                    k += 1;
                }
            }

            data[cj * cdim + ci] = -s; // Schur complement = A22 - L21*D*L21^T
            // But we need to ADD the original A22 entries, which were already
            // assembled in the frontal. The factored matrix's trailing block
            // is A22 - L21*D*L21^T, which is exactly what we want.
            // However, the dense factor overwrites the full matrix.
            // We need to get the trailing block from the factor storage.
        }
    }

    // Actually, the simplest correct approach: we factored the full nrow×nrow
    // frontal. The trailing (nrow-ncol)×(nrow-ncol) block after the first
    // ncol pivots IS the Schur complement. It's stored in the factors'
    // internal L storage in the lower-right corner.
    //
    // In our dense BK implementation, after factoring, the work matrix
    // contains: L in the strict lower triangle, D on diagonal/subdiagonal.
    // The trailing block (rows/cols ncol..nrow) of the work matrix is the
    // Schur complement — it was updated by all ncol rank-1/rank-2 updates.
    //
    // Access it directly from l (which stores the modified matrix).
    for cj in 0..cdim {
        for ci in cj..cdim {
            let pi = ncol + ci;
            let pj = ncol + cj;
            // The trailing block of l contains the Schur complement
            // (the updates from factoring the first ncol pivots have been
            // applied to this region but it hasn't been factored itself)
            data[cj * cdim + ci] = l[pj * n_full + pi];
        }
    }

    ContribBlock {
        row_indices: row_indices[ncol..].to_vec(),
        data,
        dim: cdim,
    }
}

/// Create empty factors for a zero-size node.
fn empty_factors() -> crate::dense::factor::Factors {
    crate::dense::factor::Factors {
        n: 0,
        l: Vec::new(),
        d_diag: Vec::new(),
        d_subdiag: Vec::new(),
        perm: Vec::new(),
        perm_inv: Vec::new(),
        d_eq: Vec::new(),
        needs_refinement: false,
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
