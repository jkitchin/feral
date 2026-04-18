//! Initial bisection at the coarsest level.
//!
//! Implements two simple schemes matched to METIS 5.2.0's
//! `Init2WayPartition`:
//!
//! - **GGP** (Greedy Graph Growing): seed one vertex in part 0,
//!   repeatedly pull in the boundary vertex whose inclusion minimises
//!   the added cut. Stops when part-0 weight reaches half the total.
//! - **Random BFS**: seed one vertex, assign vertices to part 0 in
//!   BFS order until the half-weight mark is reached.
//!
//! Each call runs one trial. The multi-trial driver (with post-FM
//! scoring) lives in `node_nd.rs` where the FM module is already in
//! scope.

use crate::graph::Graph;
use crate::rng::SplitMix;

pub const PART_A: u8 = 0;
pub const PART_B: u8 = 1;

/// Compute the edge-cut of a bisection.
pub fn cut_size(graph: &Graph, labels: &[u8]) -> i32 {
    debug_assert_eq!(labels.len(), graph.nvtxs as usize);
    let mut cut: i64 = 0;
    for v in 0..graph.nvtxs as usize {
        let lv = labels[v];
        let lo = graph.xadj[v] as usize;
        let hi = graph.xadj[v + 1] as usize;
        for k in lo..hi {
            let u = graph.adjncy[k] as usize;
            if u > v && labels[u] != lv {
                cut += graph.adjwgt[k] as i64;
            }
        }
    }
    cut.min(i32::MAX as i64) as i32
}

/// Total vertex weight of part `p`.
pub fn part_weight(graph: &Graph, labels: &[u8], p: u8) -> i64 {
    let mut s: i64 = 0;
    for (v, &l) in labels.iter().enumerate() {
        if l == p {
            s += graph.vwgt[v] as i64;
        }
    }
    s
}

/// GGP trial. Grows part 0 from a random seed until its weight
/// reaches `target_a`. All unvisited vertices go to part 1.
pub fn initial_bisect_ggp(graph: &Graph, rng: &mut SplitMix, target_a: i64) -> Vec<u8> {
    let n = graph.nvtxs as usize;
    let mut labels: Vec<u8> = vec![PART_B; n];
    if n == 0 {
        return labels;
    }
    let seed = rng.gen_range(n as u64) as usize;
    labels[seed] = PART_A;
    let mut a_weight: i64 = graph.vwgt[seed] as i64;

    // Track each unassigned vertex's "gain" = (edges to part 0) - (edges to part 1).
    // A priority-ish structure without a full PQ: we pick the vertex
    // with largest gain via linear scan over the boundary set. On
    // coarsest graphs (n <= coarsen_floor = 120) this is fine.
    let mut gain: Vec<i32> = vec![0; n];
    let mut in_boundary: Vec<bool> = vec![false; n];
    let mut boundary: Vec<i32> = Vec::new();

    // Populate gains for seed's neighbors.
    let push_neighbors = |v: usize,
                          graph: &Graph,
                          labels: &[u8],
                          gain: &mut [i32],
                          in_boundary: &mut [bool],
                          boundary: &mut Vec<i32>,
                          adding_to_a: bool| {
        let lo = graph.xadj[v] as usize;
        let hi = graph.xadj[v + 1] as usize;
        for k in lo..hi {
            let u = graph.adjncy[k] as usize;
            if labels[u] != PART_B {
                continue;
            }
            // v is in A, u is in B. Edge (v,u) contributes +adjwgt to
            // u's gain to move into A (one more edge on the A side).
            let w = graph.adjwgt[k];
            if adding_to_a {
                gain[u] = gain[u].saturating_add(w);
            }
            if !in_boundary[u] {
                in_boundary[u] = true;
                boundary.push(u as i32);
            }
        }
    };

    push_neighbors(
        seed,
        graph,
        &labels,
        &mut gain,
        &mut in_boundary,
        &mut boundary,
        true,
    );

    while a_weight < target_a && !boundary.is_empty() {
        // Pick best-gain boundary vertex.
        let mut best_i: usize = 0;
        let mut best_g: i32 = i32::MIN;
        for (i, &v) in boundary.iter().enumerate() {
            if labels[v as usize] != PART_B {
                continue;
            }
            let g = gain[v as usize];
            if g > best_g {
                best_g = g;
                best_i = i;
            }
        }
        let v = boundary.swap_remove(best_i);
        let vu = v as usize;
        if labels[vu] != PART_B {
            continue;
        }
        labels[vu] = PART_A;
        in_boundary[vu] = false;
        a_weight += graph.vwgt[vu] as i64;
        // Update neighbors' gains: v was in B, now in A; each
        // B-neighbor u of v sees +adjwgt(v,u) toward A.
        push_neighbors(
            vu,
            graph,
            &labels,
            &mut gain,
            &mut in_boundary,
            &mut boundary,
            true,
        );
    }
    labels
}

/// Random BFS trial. Grows part 0 in BFS order from a random seed
/// until its weight reaches `target_a`.
pub fn initial_bisect_bfs(graph: &Graph, rng: &mut SplitMix, target_a: i64) -> Vec<u8> {
    let n = graph.nvtxs as usize;
    let mut labels: Vec<u8> = vec![PART_B; n];
    if n == 0 {
        return labels;
    }
    let seed = rng.gen_range(n as u64) as usize;
    let mut queue: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
    queue.push_back(seed);
    let mut visited: Vec<bool> = vec![false; n];
    visited[seed] = true;
    let mut a_weight: i64 = 0;
    while let Some(v) = queue.pop_front() {
        if a_weight >= target_a {
            break;
        }
        labels[v] = PART_A;
        a_weight += graph.vwgt[v] as i64;
        let lo = graph.xadj[v] as usize;
        let hi = graph.xadj[v + 1] as usize;
        for k in lo..hi {
            let u = graph.adjncy[k] as usize;
            if !visited[u] {
                visited[u] = true;
                queue.push_back(u);
            }
        }
    }
    labels
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coarsen::{coarsen_level, CoarsenCounters};
    use feral_ordering_core::CscPattern;
    use std::collections::BTreeSet;

    fn csc_from_triples(n: usize, triples: &[(usize, usize)]) -> (Vec<i32>, Vec<i32>) {
        let mut set: BTreeSet<(usize, usize)> = BTreeSet::new();
        for &(i, j) in triples {
            set.insert((i, j));
            set.insert((j, i));
        }
        let mut cols: Vec<Vec<i32>> = vec![Vec::new(); n];
        for &(r, c) in &set {
            cols[c].push(r as i32);
        }
        for col in &mut cols {
            col.sort();
        }
        let mut col_ptr: Vec<i32> = vec![0];
        let mut row_idx: Vec<i32> = Vec::new();
        for col in &cols {
            for &r in col {
                row_idx.push(r);
            }
            col_ptr.push(row_idx.len() as i32);
        }
        (col_ptr, row_idx)
    }

    fn grid(m: usize, n: usize) -> Graph {
        let idx = |r: usize, c: usize| r * n + c;
        let total = m * n;
        let mut t = Vec::new();
        for r in 0..m {
            for c in 0..n {
                let k = idx(r, c);
                t.push((k, k));
                if r + 1 < m {
                    t.push((k, idx(r + 1, c)));
                }
                if c + 1 < n {
                    t.push((k, idx(r, c + 1)));
                }
            }
        }
        let (cp, ri) = csc_from_triples(total, &t);
        let pat = CscPattern::new(total, &cp, &ri).unwrap();
        Graph::from_csc_pattern(&pat).unwrap()
    }

    fn total_weight(graph: &Graph) -> i64 {
        graph.vwgt.iter().map(|&w| w as i64).sum()
    }

    #[test]
    fn cut_size_on_known_bisection() {
        let g = grid(2, 2); // four vertices, four edges
                            // Split into {0,1} and {2,3}. Cut edges: (0,2) and (1,3) → 2.
        let labels = vec![PART_A, PART_A, PART_B, PART_B];
        assert_eq!(cut_size(&g, &labels), 2);
        // All together → zero cut.
        let zero = vec![PART_A; 4];
        assert_eq!(cut_size(&g, &zero), 0);
    }

    #[test]
    fn ggp_produces_valid_bisection() {
        let g = grid(6, 6);
        let total = total_weight(&g);
        let target = total / 2;
        let mut rng = SplitMix::new(1);
        let labels = initial_bisect_ggp(&g, &mut rng, target);
        assert_eq!(labels.len(), g.nvtxs as usize);
        let a = part_weight(&g, &labels, PART_A);
        let b = part_weight(&g, &labels, PART_B);
        assert_eq!(a + b, total);
        assert!(
            a > 0 && b > 0,
            "both parts non-empty (got a={}, b={})",
            a,
            b
        );
        // GGP stops at the first vertex that tips a_weight past target,
        // so |a - target| is bounded by max vertex weight (1 here).
        assert!((a - target).abs() <= 1, "GGP weight balance: a={}", a);
    }

    #[test]
    fn bfs_produces_valid_bisection() {
        let g = grid(6, 6);
        let total = total_weight(&g);
        let target = total / 2;
        let mut rng = SplitMix::new(3);
        let labels = initial_bisect_bfs(&g, &mut rng, target);
        let a = part_weight(&g, &labels, PART_A);
        let b = part_weight(&g, &labels, PART_B);
        assert_eq!(a + b, total);
        assert!(a > 0 && b > 0);
        assert!((a - target).abs() <= 1, "BFS weight balance: a={}", a);
    }

    #[test]
    fn ggp_on_coarsened_graph_is_balanced() {
        // Exercise the coarsening + bisection sequence end-to-end
        // on a small grid. The coarse graph still has to accept a
        // balanced bisection.
        let g = grid(8, 8);
        let mut rng = SplitMix::new(5);
        let mut ctr = CoarsenCounters::default();
        let cg = coarsen_level(&g, &mut rng, 0.85, &mut ctr);
        let total = total_weight(&cg.graph);
        let target = total / 2;
        let labels = initial_bisect_ggp(&cg.graph, &mut rng, target);
        let a = part_weight(&cg.graph, &labels, PART_A);
        let b = part_weight(&cg.graph, &labels, PART_B);
        assert_eq!(a + b, total);
        assert!(a > 0 && b > 0);
    }

    #[test]
    fn ggp_is_deterministic() {
        let g = grid(5, 5);
        let total = total_weight(&g);
        let target = total / 2;
        let mut r1 = SplitMix::new(11);
        let mut r2 = SplitMix::new(11);
        let a = initial_bisect_ggp(&g, &mut r1, target);
        let b = initial_bisect_ggp(&g, &mut r2, target);
        assert_eq!(a, b);
    }
}
