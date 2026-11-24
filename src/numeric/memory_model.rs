//! A-priori memory model of `Solver::factor` (issue #249).
//!
//! [`crate::symbolic::WorkEstimate::peak_bytes`] used to be the slacked
//! column-count factor size plus the simulated contribution-block peak.
//! Measured on the Primal2_1000 LP KKT (n = 3.86M) that left out, in
//! order of size (`dev/research/issue-249-memory-gap.md`):
//!
//! 1. **amalgamation fill.** Each supernode stores a dense `nrow × ncol`
//!    block, `Σ nrow·ncol` entries. The column counts are taken before
//!    amalgamation; on Primal2 (AMF) the stored block sum is 1.50× their
//!    total, more than the 1.2 slack covers.
//! 2. **per-node index data.** Row indices, the in-front BK permutation
//!    and its inverse (all of length `nrow`), the `D` vectors, and the
//!    `NodeFactors` record itself: 0.78 GB on Primal2.
//! 3. **the analysis**, which the `Solver` keeps cached (0.77 GB).
//! 4. **numeric workspace and `Solver` caches**, proportional to the
//!    input: the permuted copy of the matrix, the permute cache, the
//!    pattern kept for the fingerprint check, `row_map`, scaling vectors.
//! 5. **under the parallel driver,** the blocks of finished subtree tasks
//!    that wait for their parent task (the frontier), and one workspace
//!    per worker.
//!
//! Every term assumes no delayed pivots: a delay moves columns into a
//! larger parent front and grows the factor beyond this model.

use std::mem::size_of;

use crate::numeric::contrib::{ContribBlock, Packed, Square};
use crate::numeric::factorize::NodeFactors;
use crate::symbolic::supernode::Supernode;
use crate::symbolic::SymbolicFactorization;

const VB: usize = size_of::<f64>();
const IB: usize = size_of::<usize>();

fn vec_bytes<T>(v: &Vec<T>) -> usize {
    v.capacity() * size_of::<T>()
}

/// Bytes the numeric phase keeps for one supernode's factor values: the
/// dense `nrow × ncol` block, absent delayed pivots.
pub(crate) fn node_value_bytes(sn: &Supernode) -> usize {
    sn.nrow.saturating_mul(sn.ncol).saturating_mul(VB)
}

/// Bytes the numeric phase keeps for one supernode beside its values:
/// the front's row indices, the in-front BK permutation and its inverse
/// (each of length `nrow`), `d_diag` / `d_subdiag` (length `ncol`), and
/// the `NodeFactors` record in the factor's node vector.
pub(crate) fn node_index_bytes(sn: &Supernode) -> usize {
    3 * sn.nrow * IB + 2 * sn.ncol * VB + size_of::<NodeFactors>()
}

/// Bytes of one live contribution block: its values (counted square, as
/// [`Supernode::contrib_size`]) and its row indices.
fn contrib_block_bytes(sn: &Supernode) -> usize {
    sn.contrib_size() * VB + sn.contrib_nrow() * IB
}

/// Heap bytes held by an analysis, by capacity. A `Solver` keeps its
/// analysis cached for the next factorization of the same pattern.
pub(crate) fn analysis_heap_bytes(sym: &SymbolicFactorization) -> usize {
    let mut b = vec_bytes(&sym.perm) + vec_bytes(&sym.perm_inv) + vec_bytes(&sym.supernodes);
    for sn in &sym.supernodes {
        b += vec_bytes(&sn.row_indices) + vec_bytes(&sn.children);
    }
    b += vec_bytes(&sym.contrib_sizes)
        + vec_bytes(&sym.etree.parent)
        + vec_bytes(&sym.permuted_pattern.col_ptr)
        + vec_bytes(&sym.permuted_pattern.row_idx)
        + vec_bytes(&sym.col_counts)
        + vec_bytes(&sym.small_leaf_groups)
        + vec_bytes(&sym.snode_group)
        + vec_bytes(&sym.static_row_offsets)
        + vec_bytes(&sym.static_row_flat);
    for g in &sym.small_leaf_groups {
        b += vec_bytes(&g.members) + vec_bytes(&g.member_rows) + vec_bytes(&g.offsets);
        for r in &g.member_rows {
            b += vec_bytes(r);
        }
    }
    if let crate::symbolic::OrderingMethod::External(p) = &sym.resolved_method {
        b += vec_bytes(p);
    }
    if let Some(c) = &sym.cached_mc64 {
        b += vec_bytes(&c.perm) + vec_bytes(&c.u) + vec_bytes(&c.v) + vec_bytes(&c.cmax);
    }
    b
}

/// Stored entries of the input matrix, recovered from the analysis: the
/// permuted pattern holds both triangles, so a lower-triangle input with
/// a full diagonal stores `(nnz_full + n) / 2`.
fn input_stored_nnz(sym: &SymbolicFactorization) -> usize {
    (sym.permuted_pattern.row_idx.len() + sym.n) / 2
}

/// Numeric workspace and `Solver` caches of the sequential path, sized
/// from the input: the permuted copy `P·A·Pᵀ` (kept by the permute cache
/// after the call), the cache's copy of the input pattern, inverse
/// permutation and value route, the pattern the `Solver` keeps for its
/// fingerprint check, `row_map`, `build_seen`, the two scaling vectors,
/// the per-supernode contribution-slot vector and `is_root`.
pub(crate) fn workspace_bytes(sym: &SymbolicFactorization) -> usize {
    let n = sym.n;
    let s = sym.supernodes.len();
    let nnz = input_stored_nnz(sym);
    let csc_pattern = (n + 1) * IB + nnz * IB;
    let permuted = csc_pattern + nnz * VB;
    let permute_cache = csc_pattern + n * IB + nnz * IB;
    let solver_pattern = csc_pattern;
    let scratch = n * IB + n + 2 * n * VB;
    let slots = s * (size_of::<Option<ContribBlock<Packed>>>() + 1);
    permuted + permute_cache + solver_pattern + scratch + slots
}

/// The pooled frontal buffer's capacity (entries) after a sequence of
/// fronts, by the one growth rule of `dense::matrix::grow_pooled_front`.
fn grow_pool(cap: usize, nrow: usize) -> usize {
    let needed = nrow * nrow;
    if cap < needed {
        needed.max(2 * cap)
    } else {
        cap
    }
}

/// Sequential driver's numeric peak: walk the postorder, adding each
/// node's stored factor and contribution block and the frontal pool's
/// capacity, freeing children's blocks after their parent, and keep the
/// largest total. Blocks are counted square (an upper bound on the
/// packed blocks the sequential driver holds).
pub(crate) fn sequential_numeric_peak(sym: &SymbolicFactorization) -> usize {
    let sns = &sym.supernodes;
    let mut stored = 0usize;
    let mut live = 0usize;
    let mut cap = 0usize;
    let mut peak = 0usize;
    for sn in sns {
        cap = grow_pool(cap, sn.nrow);
        stored += node_value_bytes(sn) + node_index_bytes(sn);
        live += contrib_block_bytes(sn);
        peak = peak.max(stored + live + cap * VB);
        for &c in &sn.children {
            if let Some(child) = sns.get(c) {
                live -= contrib_block_bytes(child);
            }
        }
    }
    peak
}

/// The parallel task-graph driver's extra memory over the factor.
pub(crate) struct ParallelModel {
    /// Held once whatever the thread count: the shared contribution-slot
    /// and result-slot vectors, the task plan.
    pub shared_bytes: usize,
    /// Largest total of task-root contribution blocks that can be
    /// finished and waiting at once: the maximum-weight antichain of the
    /// task tree. Bounds the frontier under any schedule.
    pub frontier_bytes: usize,
    /// Per running task: a worker's workspace (`row_map`, `build_seen`,
    /// its contribution-slot vector, a frontal pool as large as the
    /// sequential driver's) and the largest live-block peak inside one
    /// task's inline subtree.
    pub per_thread_bytes: usize,
}

/// Build the parallel model from the task plan's shape. `tasks` lists,
/// for each task root, its inline nodes in postorder followed by the
/// root itself; `parent_task[t]` is the task that consumes `t`'s block.
pub(crate) fn parallel_model(
    sym: &SymbolicFactorization,
    tasks: &[(usize, &[usize])],
    parent_task: &[Option<usize>],
) -> ParallelModel {
    let sns = &sym.supernodes;
    let s = sns.len();

    // Largest live-block peak inside one task: its inline nodes in
    // postorder, freeing a child's block once its parent (inline or the
    // root) has run. Children that are task roots of their own are
    // frontier blocks, counted there.
    let mut is_inline_of = vec![false; s];
    let mut max_task_peak = 0usize;
    for &(root, owned) in tasks {
        for &k in owned {
            is_inline_of[k] = true;
        }
        let mut live = 0usize;
        let mut peak = 0usize;
        for &k in owned.iter().chain(std::iter::once(&root)) {
            let sn = &sns[k];
            live += contrib_block_bytes(sn);
            peak = peak.max(live);
            for &c in &sn.children {
                if c < s && is_inline_of[c] {
                    live -= contrib_block_bytes(&sns[c]);
                }
            }
        }
        for &k in owned {
            is_inline_of[k] = false;
        }
        max_task_peak = max_task_peak.max(peak);
    }

    // Maximum-weight antichain of task-root blocks over the task tree:
    // best(t) = max(block(t), Σ best(child tasks)). Task roots appear in
    // ascending (post)order, so children precede parents.
    let mut child_sum = vec![0usize; s];
    let mut frontier = 0usize;
    for &(root, _) in tasks {
        let best = contrib_block_bytes(&sns[root]).max(child_sum[root]);
        match parent_task.get(root).copied().flatten() {
            Some(p) if p < s => child_sum[p] += best,
            _ => frontier += best,
        }
    }

    let mut pool = 0usize;
    for sn in sns {
        pool = grow_pool(pool, sn.nrow);
    }
    let slot_sq = size_of::<Option<ContribBlock<Square>>>();
    let per_thread = sym.n * IB + sym.n + s * slot_sq + pool * VB + max_task_peak;
    // Shared: contribution slots, result slots, and the task plan
    // (owned lists, parent/pending/flop arrays, is_root).
    let shared = s * (slot_sq + size_of::<Option<NodeFactors>>()) + s * (3 * IB + 2 * IB + 1 + 8);

    ParallelModel {
        shared_bytes: shared,
        frontier_bytes: frontier,
        per_thread_bytes: per_thread,
    }
}
