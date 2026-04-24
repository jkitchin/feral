use crate::ordering::elimination_tree::EliminationTree;
use crate::symbolic::small_leaf::SmallLeafParams;

/// Parameters controlling supernode amalgamation.
///
/// β refactor (`dev/plans/scaling-in-numeric.md`): the
/// `scaling_strategy` field used to live here, but scaling is a
/// numeric-time concern and now lives on
/// [`crate::numeric::factorize::NumericParams`]. This struct
/// covers only the symbolic phase.
pub struct SupernodeParams {
    /// Minimum number of eliminated columns in a supernode. Nodes with
    /// fewer eliminations are candidates for merging with their parent.
    /// Default: 32 (matching SSIDS). MUMPS uses 5.
    /// Setting nemin=1 effectively disables amalgamation.
    pub nemin: usize,

    /// Opt-in ordering preprocessing. Default `None`.
    ///
    /// Set to `OrderingPreprocess::LdltCompress` to run MC64 symmetric
    /// matching and collapse each matched pair into one super-variable
    /// before handing the graph to AMD/METIS/SCOTCH. Matches MUMPS's
    /// `ICNTL(12) = 2` for SYM=2. Opt-in while the corpus bench is
    /// collected; see `dev/plans/phase-2.6.5-ldlt-compressed-graph.md`.
    pub preprocess: OrderingPreprocess,

    /// Small-leaf-subtree grouping parameters (Phase 2.9). Controls
    /// which true-leaf supernodes are packed into batch groups for
    /// the numeric small-leaf fast path. The detection runs
    /// unconditionally at symbolic time; whether the numeric phase
    /// uses the groups is gated by
    /// [`crate::numeric::factorize::NumericParams::small_leaf`].
    /// See `dev/research/phase-2.9-small-leaf-subtree.md`.
    pub small_leaf: SmallLeafParams,
}

/// Ordering-stage preprocessing flag.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum OrderingPreprocess {
    /// No preprocessing. The fill-reducing ordering runs directly on
    /// the symmetric pattern.
    None,
    /// Duff-Pralet symmetric matching + quotient-graph compression.
    /// See `crate::symbolic::ldlt_compress`.
    LdltCompress,
    /// Shape-dispatched: run `LdltCompress` when cheap shape predicates
    /// predict a benefit, else run `None`. See
    /// `crate::symbolic::pick_ordering_preprocess` for the rule.
    ///
    /// Parallels `ScalingStrategy::Auto`. Default since Phase 2.4.4.
    #[default]
    Auto,
}

impl Default for SupernodeParams {
    fn default() -> Self {
        Self {
            nemin: 32,
            preprocess: OrderingPreprocess::Auto,
            small_leaf: SmallLeafParams::default(),
        }
    }
}

/// A supernode in the assembly tree.
#[derive(Debug, Clone)]
pub struct Supernode {
    /// Range of columns eliminated in this supernode (in the postordered numbering).
    /// The number of eliminated columns is `cols.end - cols.start`.
    pub first_col: usize,
    pub ncol: usize,
    /// Total number of rows in the frontal matrix (nrow >= ncol).
    pub nrow: usize,
    /// Row indices of the frontal matrix (length nrow).
    /// The first `ncol` entries are the eliminated columns themselves;
    /// the remaining `nrow - ncol` are the non-eliminated rows that form
    /// the contribution block.
    pub row_indices: Vec<usize>,
    /// Children supernode indices.
    pub children: Vec<usize>,
}

impl Supernode {
    /// Number of eliminated columns.
    #[inline]
    pub fn ncol(&self) -> usize {
        self.ncol
    }

    /// Number of rows in the contribution block.
    #[inline]
    pub fn contrib_nrow(&self) -> usize {
        self.nrow - self.ncol
    }

    /// Size of the contribution block in f64 entries.
    #[inline]
    pub fn contrib_size(&self) -> usize {
        let cn = self.contrib_nrow();
        cn * cn
    }
}

/// Detect fundamental supernodes and apply amalgamation.
///
/// A fundamental supernode is a maximal set of consecutive columns j, j+1, ..., j+k
/// where each column's row structure is identical (the same set of row indices,
/// minus the column being eliminated). This is detected by checking that:
/// 1. Column j+1 has exactly one more nonzero than column j (the new diagonal).
/// 2. The parent of j in the elimination tree is j+1.
///
/// After detecting fundamental supernodes, amalgamation merges small nodes
/// using the SSIDS merge rule:
/// 1. Trivial chain: parent has exactly 1 column AND parent nrow == child nrow - child ncol + parent ncol.
///    (i.e., same row structure minus the eliminated columns)
/// 2. Size-based: both parent AND child have < nemin columns.
///
/// `col_row_indices` provides the actual row indices for each column of L
/// (used to build correct frontal row index sets).
///
/// Returns supernodes in postorder (children before parents).
pub fn find_supernodes(
    etree: &EliminationTree,
    col_counts: &[usize],
    params: &SupernodeParams,
) -> Vec<Supernode> {
    let n = etree.n;
    if n == 0 {
        return Vec::new();
    }

    // Step 1: Find fundamental supernodes
    // snode_id[j] = which supernode column j belongs to
    let mut snode_id = vec![0usize; n];
    let mut snode_starts: Vec<usize> = Vec::new();

    // Count how many children each node has in the etree
    let mut n_children = vec![0usize; n];
    for j in 0..n {
        if let Some(p) = etree.parent[j] {
            n_children[p] += 1;
        }
    }

    snode_starts.push(0);
    snode_id[0] = 0;

    for j in 1..n {
        // Column j starts a new supernode unless ALL conditions hold:
        // 1. parent[j-1] == j (j-1 is a child of j in the etree)
        // 2. col_counts[j] == col_counts[j-1] - 1 (same row structure minus one row)
        // 3. j has exactly one child in the etree (j-1 is its only child)
        //    This prevents chaining across disconnected components where
        //    col_count conditions happen to match spuriously.
        let same_snode = etree.parent[j - 1] == Some(j)
            && col_counts[j] + 1 == col_counts[j - 1]
            && n_children[j] == 1;

        if same_snode {
            snode_id[j] = snode_id[j - 1];
        } else {
            snode_id[j] = snode_starts.len();
            snode_starts.push(j);
        }
    }

    let n_snodes = snode_starts.len();

    // Compute supernode sizes and parent relationships
    let mut snode_ncols = vec![0usize; n_snodes];
    let mut snode_parent: Vec<Option<usize>> = vec![None; n_snodes];

    for j in 0..n {
        snode_ncols[snode_id[j]] += 1;
    }

    // Parent of a supernode = supernode containing the parent of its last column
    for s in 0..n_snodes {
        let last_col = snode_starts[s] + snode_ncols[s] - 1;
        if let Some(p) = etree.parent[last_col] {
            snode_parent[s] = Some(snode_id[p]);
        }
    }

    // Step 2: Amalgamation
    // Track which supernodes are merged (absorbed into parent)
    let mut merged_into = vec![None::<usize>; n_snodes];
    // Track the actual first column of each supernode (may change during merging)
    let mut snode_first_col: Vec<usize> = snode_starts.clone();

    for (s, sp) in snode_parent.iter().enumerate() {
        if let Some(p) = sp {
            let p = *p;
            if find_root(s, &merged_into) != s {
                continue; // already merged into another node
            }

            let root_s = find_root(s, &merged_into);
            let root_p = find_root(p, &merged_into);
            if root_s == root_p {
                continue;
            }

            // Adjacency check: merging is only valid when the child's
            // effective column range [s_first, s_first+s_ncol) is
            // immediately followed by the parent's column range
            // [p_first, p_first+p_ncol). Otherwise the merged
            // supernode's `first_col..first_col+ncol` would no longer
            // be a contiguous block of the column numbering, and
            // downstream code (row-index construction, A-assembly, L
            // storage, solve gather/scatter) would silently claim
            // columns that belong to *other* supernodes.
            //
            // In a postorder-column-numbered elimination tree every
            // parent's columns come after all its descendants', so in
            // a multi-child parent at most one child is adjacent —
            // the one whose last column is parent_first - 1. Merging
            // any other child breaks contiguity. The arrow matrix
            // (variables 0..n-2 all parented by variable n-1) is the
            // archetype: only child n-2 is adjacent to parent n-1.
            //
            // SSIDS side-steps this by emitting a permutation that
            // renumbers columns so merged supernodes are contiguous
            // by construction (`core_analyse.f90:644-685`). That's a
            // strictly better amalgamation policy (merges more
            // children, reduces fill on arrow-like trees) but is a
            // larger refactor. For now the adjacency check is the
            // minimal correctness fix; see
            // `dev/research/phase-2.2.3-plateau.md` for the full
            // analysis.
            let s_first = snode_first_col[root_s];
            let s_ncol = snode_ncols[root_s];
            let p_first = snode_first_col[root_p];
            if s_first + s_ncol != p_first {
                continue;
            }

            let child_ncol = snode_ncols[root_s];
            let parent_ncol = snode_ncols[root_p];

            // SSIDS merge rule:
            // 1. Trivial chain: parent has exactly 1 col AND parent's column
            //    count == child's last column count - 1 (same row structure
            //    minus one eliminated column)
            let trivial_chain = parent_ncol == 1 && {
                let child_last = s_first + s_ncol - 1;
                col_counts[p_first] + 1 == col_counts[child_last]
            };

            // 2. Size-based: both have < nemin columns
            let size_based = child_ncol < params.nemin && parent_ncol < params.nemin;

            if trivial_chain || size_based {
                merged_into[root_s] = Some(root_p);
                // Transfer columns to parent and update first column.
                // Adjacency invariant guarantees s_first < p_first,
                // so the merged range is [s_first, p_first+p_ncol).
                snode_ncols[root_p] = child_ncol + parent_ncol;
                snode_first_col[root_p] = s_first;
            }
        }
    }

    // Step 3: Build final supernode list
    // Collect non-merged supernodes
    let mut final_snodes: Vec<Supernode> = Vec::new();
    let mut new_snode_id = vec![0usize; n_snodes]; // old → new supernode index

    for s in 0..n_snodes {
        if merged_into[s].is_some() {
            continue;
        }

        let first_col = snode_first_col[s];
        let ncol = snode_ncols[s];
        // nrow = col_counts[first_col]: number of rows in L for the first
        // column of this supernode, which gives the frontal matrix height
        let nrow = col_counts[first_col].max(ncol);

        // Row indices: the first_col..first_col+ncol are the eliminated columns,
        // plus the remaining rows from col_counts
        // For now, store just the column range — actual row indices are
        // determined during symbolic factorization with the full pattern
        let row_indices = (first_col..first_col + nrow).collect();

        new_snode_id[s] = final_snodes.len();

        final_snodes.push(Supernode {
            first_col,
            ncol,
            nrow,
            row_indices,
            children: Vec::new(),
        });
    }

    // Set children relationships
    for s in 0..n_snodes {
        if merged_into[s].is_some() {
            continue;
        }
        if let Some(p) = snode_parent[s] {
            let root_p = find_root(p, &merged_into);
            if root_p != s {
                let new_child = new_snode_id[s];
                let new_parent = new_snode_id[root_p];
                final_snodes[new_parent].children.push(new_child);
            }
        }
    }

    final_snodes
}

/// Find the root of the merge chain for supernode s.
fn find_root(s: usize, merged_into: &[Option<usize>]) -> usize {
    let mut node = s;
    while let Some(parent) = merged_into[node] {
        node = parent;
    }
    node
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::csc::CscMatrix;
    use crate::symbolic::column_counts::column_counts;

    #[test]
    fn test_supernodes_tridiagonal() {
        // Tridiagonal 4x4: col_counts = [2, 2, 2, 1]
        // Columns 2,3 form a fundamental supernode (parent[2]=3, counts[3]+1=counts[2])
        // Columns 0 and 1 are singletons
        let m =
            CscMatrix::from_triplets(4, &[0, 1, 1, 2, 2, 3, 3], &[0, 0, 1, 1, 2, 2, 3], &[1.0; 7])
                .unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);
        let counts = column_counts(&pat, &etree);

        // With nemin=1, we get 3 supernodes: {0}, {1}, {2,3}
        let params = SupernodeParams {
            nemin: 1,
            ..Default::default()
        };
        let snodes = find_supernodes(&etree, &counts, &params);
        assert_eq!(snodes.len(), 3);

        let total_cols: usize = snodes.iter().map(|s| s.ncol()).sum();
        assert_eq!(total_cols, 4);
    }

    #[test]
    fn test_supernodes_tridiagonal_amalgamated() {
        // With large nemin, all singletons should be amalgamated into one
        let m =
            CscMatrix::from_triplets(4, &[0, 1, 1, 2, 2, 3, 3], &[0, 0, 1, 1, 2, 2, 3], &[1.0; 7])
                .unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);
        let counts = column_counts(&pat, &etree);

        let params = SupernodeParams {
            nemin: 32,
            ..Default::default()
        };
        let snodes = find_supernodes(&etree, &counts, &params);

        // All 4 columns should be amalgamated into 1 supernode
        let total_cols: usize = snodes.iter().map(|s| s.ncol()).sum();
        assert_eq!(total_cols, 4);
        assert_eq!(snodes.len(), 1);
    }

    #[test]
    fn test_supernodes_dense() {
        // Dense 3x3: col_counts = [3, 2, 1]
        // Fundamental: column 1 chains into column 0 (parent[0]=1, counts[1]=counts[0]-1)
        // Column 2 chains into column 1 (parent[1]=2, counts[2]=counts[1]-1)
        // So all 3 columns form one fundamental supernode
        let m = CscMatrix::from_triplets(3, &[0, 1, 2, 1, 2, 2], &[0, 0, 0, 1, 1, 2], &[1.0; 6])
            .unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);
        let counts = column_counts(&pat, &etree);

        let params = SupernodeParams {
            nemin: 1,
            ..Default::default()
        };
        let snodes = find_supernodes(&etree, &counts, &params);

        // Should be 1 supernode with 3 columns (fundamental)
        assert_eq!(snodes.len(), 1);
        assert_eq!(snodes[0].ncol(), 3);
        assert_eq!(snodes[0].nrow, 3);
        assert_eq!(snodes[0].contrib_size(), 0); // no contribution block
    }

    #[test]
    fn test_supernodes_block_diagonal() {
        // Two 2x2 dense blocks: two independent supernodes
        let m = CscMatrix::from_triplets(4, &[0, 1, 1, 2, 3, 3], &[0, 0, 1, 2, 2, 3], &[1.0; 6])
            .unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);
        let counts = column_counts(&pat, &etree);

        let params = SupernodeParams {
            nemin: 1,
            ..Default::default()
        };
        let snodes = find_supernodes(&etree, &counts, &params);

        // Two fundamental supernodes of size 2
        assert_eq!(snodes.len(), 2);
        assert_eq!(snodes[0].ncol(), 2);
        assert_eq!(snodes[1].ncol(), 2);
    }

    #[test]
    fn test_supernodes_diagonal_no_amalg() {
        // Diagonal 4x4 with nemin=1: 4 singletons, no merging possible
        let m = CscMatrix::from_triplets(4, &[0, 1, 2, 3], &[0, 1, 2, 3], &[1.0; 4]).unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);
        let counts = column_counts(&pat, &etree);

        let params = SupernodeParams {
            nemin: 1,
            ..Default::default()
        };
        let snodes = find_supernodes(&etree, &counts, &params);

        // Each column is independent (no parents), so 4 supernodes
        assert_eq!(snodes.len(), 4);
    }

    #[test]
    fn test_supernodes_total_columns() {
        // For any matrix, the total columns across all supernodes should equal n
        let m = CscMatrix::from_triplets(
            5,
            &[0, 1, 2, 3, 4, 1, 2, 3, 4],
            &[0, 0, 0, 0, 0, 1, 2, 3, 4],
            &[1.0; 9],
        )
        .unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);
        let counts = column_counts(&pat, &etree);

        for nemin in [1, 5, 32] {
            let params = SupernodeParams {
                nemin,
                ..Default::default()
            };
            let snodes = find_supernodes(&etree, &counts, &params);
            let total: usize = snodes.iter().map(|s| s.ncol()).sum();
            assert_eq!(total, 5, "nemin={}: total columns {} != 5", nemin, total);
        }
    }

    #[test]
    fn test_supernode_children_valid() {
        // Verify all child indices are valid
        let m = CscMatrix::from_triplets(
            5,
            &[0, 1, 2, 3, 4, 1, 2, 3, 4],
            &[0, 0, 0, 0, 0, 1, 2, 3, 4],
            &[1.0; 9],
        )
        .unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);
        let counts = column_counts(&pat, &etree);

        let params = SupernodeParams {
            nemin: 1,
            ..Default::default()
        };
        let snodes = find_supernodes(&etree, &counts, &params);

        for (i, s) in snodes.iter().enumerate() {
            for &child in &s.children {
                assert!(child < snodes.len(), "invalid child index");
                assert!(
                    child < i,
                    "child {} should come before parent {} in postorder",
                    child,
                    i
                );
            }
        }
    }
}
