//! CSR graph representation for the SCOTCH partitioning pipeline.
//!
//! The layout is the same shape every multilevel partitioner uses
//! (Karypis & Kumar 1998, Pellegrini 1996): an offset array `xadj`
//! into a flat neighbor array `adjncy`, with parallel vertex- and
//! edge-weight arrays. Indices stay `i32` end-to-end to match the
//! ordering-crate contract and keep cache density high.
//!
//! Diagonal entries (self-loops) of the input matrix are dropped on
//! intake; they carry no fill-reduction information and would only
//! pollute the compression and FM kernels.
//!
//! This is intentionally a near-mirror of `feral_metis::graph::Graph`
//! and will be lifted into `feral-ordering-core` (or shared via
//! feral-metis's `internals` module) once the SCOTCH driver in S5
//! needs the rest of feral-metis's coarsening / FM infrastructure.
//! For S1 (graph compression alone) it stays local.

use feral_ordering_core::{CscPattern, OrderingError};

/// CSR graph for SCOTCH partitioning.
///
/// Adjacency lists are not required to be sorted by the kernels in
/// this crate. [`Graph::from_csc_pattern`] happens to emit them sorted
/// because it reads a CSC with sorted row indices, and `compress`
/// relies on this for its closed-neighborhood hash. After future
/// coarsening passes (S2+), sortedness will need re-establishing if
/// any kernel requires it.
#[derive(Debug, Clone)]
pub(crate) struct Graph {
    /// Number of vertices. Must equal `xadj.len() - 1` and fit in
    /// `i32`.
    pub nvtxs: i32,
    /// Adjacency offsets, length `nvtxs + 1`. `xadj[0] == 0`,
    /// non-decreasing. `xadj[nvtxs] == adjncy.len()` and equals
    /// `2 * |E|` for undirected graphs without self-loops.
    pub xadj: Vec<i32>,
    /// Neighbor lists, length `2 * |E|`. Each undirected edge `{u,v}`
    /// appears twice (once in `u`'s list, once in `v`'s).
    pub adjncy: Vec<i32>,
    /// Vertex weights, length `nvtxs`. Default: all 1.
    pub vwgt: Vec<i32>,
    /// Edge weights, aligned with `adjncy`. Default: all 1.
    pub adjwgt: Vec<i32>,
}

impl Graph {
    /// Build a [`Graph`] from a full-symmetric [`CscPattern`].
    ///
    /// Diagonal entries are dropped. Duplicate row indices within a
    /// column (which `CscPattern::new` does not yet reject) are
    /// likewise dropped via a running "last-seen" check; this
    /// exploits the contract that row indices within each column are
    /// sorted ascending.
    ///
    /// Complexity: `O(nnz)` time, `O(nnz + n)` space.
    pub(crate) fn from_csc_pattern(pattern: &CscPattern<'_>) -> Result<Self, OrderingError> {
        let n = pattern.n;
        if n > i32::MAX as usize {
            return Err(OrderingError::IndexOverflow);
        }
        let nvtxs = n as i32;
        let mut xadj: Vec<i32> = Vec::with_capacity(n + 1);
        let mut adjncy: Vec<i32> = Vec::with_capacity(pattern.row_idx.len());
        xadj.push(0);
        for j in 0..n {
            let lo = pattern.col_ptr[j] as usize;
            let hi = pattern.col_ptr[j + 1] as usize;
            if hi > pattern.row_idx.len() || lo > hi {
                return Err(OrderingError::MalformedInput);
            }
            let jj = j as i32;
            let mut last: i32 = -1;
            for &r in &pattern.row_idx[lo..hi] {
                if r == jj {
                    continue; // drop diagonal
                }
                if r < 0 || r >= nvtxs {
                    return Err(OrderingError::MalformedInput);
                }
                if r == last {
                    continue; // drop duplicate
                }
                adjncy.push(r);
                last = r;
            }
            if adjncy.len() > i32::MAX as usize {
                return Err(OrderingError::IndexOverflow);
            }
            xadj.push(adjncy.len() as i32);
        }
        let vwgt = vec![1i32; n];
        let adjwgt = vec![1i32; adjncy.len()];
        Ok(Graph {
            nvtxs,
            xadj,
            adjncy,
            vwgt,
            adjwgt,
        })
    }

    /// Borrow the adjacency slice of vertex `v`.
    pub(crate) fn neighbors(&self, v: i32) -> &[i32] {
        let lo = self.xadj[v as usize] as usize;
        let hi = self.xadj[(v + 1) as usize] as usize;
        &self.adjncy[lo..hi]
    }

    /// Borrow the edge-weight slice aligned with [`Self::neighbors`].
    pub(crate) fn edge_weights(&self, v: i32) -> &[i32] {
        let lo = self.xadj[v as usize] as usize;
        let hi = self.xadj[(v + 1) as usize] as usize;
        &self.adjwgt[lo..hi]
    }

    /// Degree of vertex `v`.
    #[cfg(test)]
    pub(crate) fn degree(&self, v: i32) -> i32 {
        self.xadj[(v + 1) as usize] - self.xadj[v as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::csc_from_edges;

    #[test]
    fn diagonal_drops_and_yields_isolated_vertices() {
        // n=3, no off-diagonal edges. Just include diagonal entries.
        let cp: Vec<i32> = vec![0, 1, 2, 3];
        let ri: Vec<i32> = vec![0, 1, 2];
        let pat = CscPattern::new(3, &cp, &ri).expect("ok");
        let g = Graph::from_csc_pattern(&pat).expect("ok");
        assert_eq!(g.nvtxs, 3);
        assert_eq!(g.xadj, vec![0, 0, 0, 0]);
        assert!(g.adjncy.is_empty());
        assert_eq!(g.vwgt, vec![1, 1, 1]);
    }

    #[test]
    fn path_graph_from_csc() {
        // P_4: 0-1-2-3. Diagonal also present.
        let mut edges = vec![(0, 1), (1, 2), (2, 3)];
        for i in 0..4 {
            edges.push((i, i));
        }
        let (cp, ri) = csc_from_edges(4, &edges);
        let pat = CscPattern::new(4, &cp, &ri).expect("ok");
        let g = Graph::from_csc_pattern(&pat).expect("ok");
        assert_eq!(g.nvtxs, 4);
        // 0 -> {1}, 1 -> {0,2}, 2 -> {1,3}, 3 -> {2}
        assert_eq!(g.degree(0), 1);
        assert_eq!(g.degree(1), 2);
        assert_eq!(g.degree(2), 2);
        assert_eq!(g.degree(3), 1);
        assert_eq!(g.neighbors(1), &[0, 2]);
        assert_eq!(g.edge_weights(1), &[1, 1]);
    }
}
