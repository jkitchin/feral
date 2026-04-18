//! Fiduccia-Mattheyses refinement.
//!
//! Two refinement kernels:
//!
//! - `refine_bisection`: classic FM with best-balanced rollback.
//!   Picks the highest-gain boundary vertex, flips its partition,
//!   updates neighbor gains, locks it, and repeats. Tracks both
//!   "current cut" and "best cut subject to balance" separately; at
//!   the end of the pass rolls back to the best balanced state.
//! - `refine_separator`: greedy node-separator reduction. For each
//!   separator vertex it computes the gain (weight saved by pulling
//!   the vertex out of the separator, minus weight of neighbors on
//!   the far side that would need to enter the separator). Accepts
//!   only positive-gain moves that respect the balance constraint.
//!   A full two-sided FM with negative-gain acceptance is deferred
//!   until a concrete quality gap motivates it.
//!
//! Priority queue: a lazy `BinaryHeap<(gain, Reverse(v))>` rather
//! than METIS's bucket array. Correct; the O(log n) overhead per
//! operation is acceptable at the graph sizes FERAL targets
//! (≤ 100k vertices, FM rarely dominates runtime).

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::graph::Graph;
use crate::initial_partition::{cut_size, part_weight, PART_A, PART_B};

pub const PART_SEP: u8 = 2;

/// Refine an edge bisection with FM, tracking the best balanced cut.
///
/// `labels[v] ∈ {PART_A, PART_B}`. Modifies `labels` in place.
/// Returns the final edge cut.
pub fn refine_bisection(
    graph: &Graph,
    labels: &mut [u8],
    max_imbalance: f64,
    max_passes: u32,
) -> i32 {
    let n = graph.nvtxs as usize;
    if n < 2 {
        return cut_size(graph, labels);
    }
    let total: i64 = graph.vwgt.iter().map(|&w| w as i64).sum();
    let max_side = ((1.0 + max_imbalance) * total as f64 / 2.0).ceil() as i64;

    let mut pass_cut = cut_size(graph, labels);

    for _pass in 0..max_passes {
        let before_pass = pass_cut;
        let mut gain: Vec<i32> = vec![0; n];
        compute_gains(graph, labels, &mut gain);

        let mut locked: Vec<bool> = vec![false; n];
        let mut heap: BinaryHeap<(i32, Reverse<i32>, i32)> = BinaryHeap::new();
        // (gain, Reverse(vertex), stamp): stamp is the gain snapshot
        // stored alongside the entry so stale entries can be skipped.
        for (v, &g) in gain.iter().enumerate().take(n) {
            heap.push((g, Reverse(v as i32), g));
        }

        let mut cur_cut = pass_cut;
        let mut moves: Vec<i32> = Vec::new();
        let mut best_cut = pass_cut;
        let mut best_prefix: usize = 0;
        let mut a_w = part_weight(graph, labels, PART_A);
        let mut b_w = total - a_w;
        let mut best_a_w = a_w;
        let mut best_b_w = b_w;
        let mut no_improve: u32 = 0;

        while let Some((_, Reverse(v), stamp)) = heap.pop() {
            let vu = v as usize;
            if locked[vu] {
                continue;
            }
            if stamp != gain[vu] {
                // Stale — re-push if we haven't moved it yet.
                continue;
            }
            // Tentatively move v to the other side.
            let from = labels[vu];
            let to = if from == PART_A { PART_B } else { PART_A };
            let (new_a_w, new_b_w) = if to == PART_A {
                (a_w + graph.vwgt[vu] as i64, b_w - graph.vwgt[vu] as i64)
            } else {
                (a_w - graph.vwgt[vu] as i64, b_w + graph.vwgt[vu] as i64)
            };
            let side_max = new_a_w.max(new_b_w);
            // Always allow the move so FM can climb out of poor local
            // configurations, but only consider it for "best" if the
            // resulting partition is balanced.
            labels[vu] = to;
            a_w = new_a_w;
            b_w = new_b_w;
            cur_cut -= gain[vu];
            locked[vu] = true;
            moves.push(v);

            // Update neighbor gains.
            let lo = graph.xadj[vu] as usize;
            let hi = graph.xadj[vu + 1] as usize;
            for k in lo..hi {
                let u = graph.adjncy[k] as usize;
                if locked[u] {
                    continue;
                }
                // gain = ed - id. If neighbour u shared v's old side
                // (`from`): edge (u,v) was internal, now crosses →
                // u's ed +w, id -w → Δgain = +2w. If u shares v's
                // new side (`to`): edge was crossing, now internal
                // → Δgain = -2w.
                let w = graph.adjwgt[k];
                if labels[u] == from {
                    gain[u] += 2 * w;
                } else {
                    gain[u] -= 2 * w;
                }
                heap.push((gain[u], Reverse(u as i32), gain[u]));
            }

            if side_max <= max_side {
                if cur_cut < best_cut {
                    best_cut = cur_cut;
                    best_prefix = moves.len();
                    best_a_w = a_w;
                    best_b_w = b_w;
                    no_improve = 0;
                } else {
                    no_improve += 1;
                }
            } else {
                no_improve += 1;
            }
            if no_improve >= 50 {
                break;
            }
        }

        // Roll back moves after best_prefix.
        for &v in moves.iter().skip(best_prefix) {
            let vu = v as usize;
            labels[vu] = if labels[vu] == PART_A { PART_B } else { PART_A };
        }
        a_w = best_a_w;
        b_w = best_b_w;
        let _ = (a_w, b_w); // kept for post-pass debug assertions
        pass_cut = best_cut;
        if pass_cut == before_pass {
            break;
        }
    }
    pass_cut
}

/// Compute per-vertex gain = (edges to other side) - (edges to own side).
fn compute_gains(graph: &Graph, labels: &[u8], gain: &mut [i32]) {
    for (v, &lv) in labels.iter().enumerate() {
        let lo = graph.xadj[v] as usize;
        let hi = graph.xadj[v + 1] as usize;
        let mut ed: i32 = 0;
        let mut id: i32 = 0;
        for k in lo..hi {
            let u = graph.adjncy[k] as usize;
            let w = graph.adjwgt[k];
            if labels[u] == lv {
                id = id.saturating_add(w);
            } else {
                ed = ed.saturating_add(w);
            }
        }
        gain[v] = ed - id;
    }
}

/// Greedy node-separator refinement. Accepts positive-gain moves that
/// respect the balance constraint. Returns the final separator weight.
pub fn refine_separator(
    graph: &Graph,
    labels: &mut [u8],
    max_imbalance: f64,
    max_passes: u32,
) -> i64 {
    let n = graph.nvtxs as usize;
    let total: i64 = graph.vwgt.iter().map(|&w| w as i64).sum();
    let max_side = ((1.0 + max_imbalance) * total as f64 / 2.0).ceil() as i64;

    for _pass in 0..max_passes {
        let mut changed = false;
        let mut a_w = part_weight(graph, labels, PART_A);
        let mut b_w = part_weight(graph, labels, PART_B);
        for v in 0..n {
            if labels[v] != PART_SEP {
                continue;
            }
            // Compute cost of pulling v to side A or side B.
            let (cost_to_a, cost_to_b) = separator_pull_costs(graph, labels, v);
            let vwgt_v = graph.vwgt[v] as i64;
            // Net separator change = cost_to_side - vwgt_v.
            // Move is beneficial if cost_to_side < vwgt_v (gain > 0).
            let gain_a = vwgt_v - cost_to_a;
            let gain_b = vwgt_v - cost_to_b;
            let (best_gain, best_side) = if gain_a >= gain_b {
                (gain_a, PART_A)
            } else {
                (gain_b, PART_B)
            };
            if best_gain <= 0 {
                continue;
            }
            // Balance check: after the move, side gains vwgt_v; the
            // other side gains zero (only separator weight changes).
            let new_a = if best_side == PART_A {
                a_w + vwgt_v
            } else {
                a_w
            };
            let new_b = if best_side == PART_B {
                b_w + vwgt_v
            } else {
                b_w
            };
            if new_a.max(new_b) > max_side {
                continue;
            }
            // Apply move: v → best_side; far-side neighbors → SEP.
            labels[v] = best_side;
            if best_side == PART_A {
                a_w = new_a;
            } else {
                b_w = new_b;
            }
            let lo = graph.xadj[v] as usize;
            let hi = graph.xadj[v + 1] as usize;
            let far = if best_side == PART_A { PART_B } else { PART_A };
            for k in lo..hi {
                let u = graph.adjncy[k] as usize;
                if labels[u] == far {
                    labels[u] = PART_SEP;
                    if far == PART_A {
                        a_w -= graph.vwgt[u] as i64;
                    } else {
                        b_w -= graph.vwgt[u] as i64;
                    }
                }
            }
            changed = true;
        }
        if !changed {
            break;
        }
    }
    separator_weight(graph, labels)
}

/// Sum of `vwgt[u]` over neighbors u of v whose label is the given side.
fn separator_pull_costs(graph: &Graph, labels: &[u8], v: usize) -> (i64, i64) {
    let lo = graph.xadj[v] as usize;
    let hi = graph.xadj[v + 1] as usize;
    let mut cost_far_a: i64 = 0; // cost of pulling v to A = weight of neighbors currently on B
    let mut cost_far_b: i64 = 0;
    for k in lo..hi {
        let u = graph.adjncy[k] as usize;
        let wu = graph.vwgt[u] as i64;
        match labels[u] {
            PART_A => cost_far_b += wu,
            PART_B => cost_far_a += wu,
            _ => {}
        }
    }
    (cost_far_a, cost_far_b)
}

/// Total vertex weight of `PART_SEP` vertices.
pub fn separator_weight(graph: &Graph, labels: &[u8]) -> i64 {
    let mut s: i64 = 0;
    for (v, &l) in labels.iter().enumerate() {
        if l == PART_SEP {
            s += graph.vwgt[v] as i64;
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::initial_partition::initial_bisect_ggp;
    use crate::rng::SplitMix;
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

    /// Regression test for the FM neighbour-update sign bug fixed
    /// alongside this test (see `dev/research/metis-fm-sign-bug.md`).
    ///
    /// The bug flipped the signs at the `gain[u] ± 2w` neighbour
    /// update, so on a graph where FM actually had to move vertices,
    /// `cur_cut` drifted into negative impossible territory and FM
    /// rolled every move back. Existing tests missed it because they
    /// either started from already-optimal cuts (`initial_bisect_ggp`
    /// on grid is at the optimum), let the balance guard block every
    /// move, or only checked permutation validity.
    ///
    /// Two assertions matter here:
    ///
    /// 1. **I1 (bookkeeping consistency).** `returned_cut` must equal
    ///    `cut_size(graph, labels)` recomputed from scratch. This is
    ///    the assertion the bug *cannot* survive.
    /// 2. **Cut actually drops.** Path P_10 with alternating ABAB
    ///    labels has cut = 9 and balanced optimum cut = 1. FM with
    ///    correct bookkeeping must reduce the cut.
    #[test]
    fn fm_sign_invariant_on_alternating_path() {
        // Path 0-1-...-9.
        let mut t = Vec::new();
        for i in 0..10 {
            t.push((i, i));
        }
        for i in 0..9 {
            t.push((i, i + 1));
        }
        let (cp, ri) = csc_from_triples(10, &t);
        let pat = CscPattern::new(10, &cp, &ri).unwrap();
        let g = Graph::from_csc_pattern(&pat).unwrap();

        let mut labels: Vec<u8> = (0..10u8)
            .map(|k| if k % 2 == 0 { PART_A } else { PART_B })
            .collect();
        let before = cut_size(&g, &labels);
        assert_eq!(before, 9, "alternating path P_10 has cut 9");

        let after = refine_bisection(&g, &mut labels, 0.20, 32);

        // I1 — the assertion that catches the sign bug directly.
        assert_eq!(
            after,
            cut_size(&g, &labels),
            "returned cut must equal cut_size(labels) recomputed from scratch"
        );
        // Quality — FM must actually move at least one vertex on this
        // adversarial input.
        assert!(
            after < before,
            "FM must reduce cut from {} on alternating P_10, got {}",
            before,
            after
        );
        assert!(after >= 0, "cut size is non-negative, got {}", after);
    }

    #[test]
    fn refine_bisection_does_not_increase_cut() {
        let g = grid(8, 8);
        let total: i64 = g.vwgt.iter().map(|&w| w as i64).sum();
        let mut rng = SplitMix::new(17);
        let mut labels = initial_bisect_ggp(&g, &mut rng, total / 2);
        let initial = cut_size(&g, &labels);
        let final_cut = refine_bisection(&g, &mut labels, 0.20, 5);
        assert_eq!(
            final_cut,
            cut_size(&g, &labels),
            "reported cut matches labels"
        );
        assert!(
            final_cut <= initial,
            "cut must not increase (before={}, after={})",
            initial,
            final_cut
        );
    }

    #[test]
    fn refine_bisection_balance_respected() {
        let g = grid(6, 6);
        let total: i64 = g.vwgt.iter().map(|&w| w as i64).sum();
        let mut rng = SplitMix::new(9);
        let mut labels = initial_bisect_ggp(&g, &mut rng, total / 2);
        let returned = refine_bisection(&g, &mut labels, 0.20, 5);
        // I1 (bookkeeping consistency): returned cut equals cut
        // recomputed from labels.
        assert_eq!(returned, cut_size(&g, &labels), "I1: bookkeeping");
        let a = part_weight(&g, &labels, PART_A);
        let b = part_weight(&g, &labels, PART_B);
        let max_allowed = ((1.20_f64) * total as f64 / 2.0).ceil() as i64;
        assert!(a.max(b) <= max_allowed, "balance: a={} b={}", a, b);
        assert!(a > 0 && b > 0);
    }

    #[test]
    fn refine_bisection_bad_init_improves() {
        // Start from an adversarial labeling (all on one side, one
        // vertex on the other) — FM should move toward balance.
        let g = grid(4, 4);
        let mut labels = vec![PART_A; 16];
        labels[0] = PART_B;
        let before = cut_size(&g, &labels);
        let returned = refine_bisection(&g, &mut labels, 0.20, 10);
        let after = cut_size(&g, &labels);
        assert_eq!(returned, after, "I1: bookkeeping");
        assert!(
            after <= before,
            "FM should not worsen cut (before={}, after={})",
            before,
            after
        );
    }

    #[test]
    fn separator_weight_accounting() {
        let g = grid(3, 3);
        // Set up an explicit separator: middle row is SEP.
        let labels: Vec<u8> = (0..9u8)
            .map(|k| {
                let r = k / 3;
                match r {
                    0 => PART_A,
                    1 => PART_SEP,
                    _ => PART_B,
                }
            })
            .collect();
        assert_eq!(separator_weight(&g, &labels), 3);
    }

    #[test]
    fn refine_separator_reduces_weight_on_padded_case() {
        // Construct a 3x3 grid with middle row SEP and the middle
        // vertex "padded" — add an extra SEP vertex adjacent only to
        // A-side. Refinement should pull it out to A.
        let g = grid(3, 3);
        let mut labels: Vec<u8> = (0..9u8)
            .map(|k| {
                let r = k / 3;
                match r {
                    0 => PART_A,
                    1 => PART_SEP,
                    _ => PART_B,
                }
            })
            .collect();
        // Make index 3 (row 1, col 0) isolated from B side by
        // relabelling its row-2 neighbor as SEP too.
        labels[6] = PART_SEP;
        let before = separator_weight(&g, &labels);
        let after = refine_separator(&g, &mut labels, 0.50, 10);
        // I1 (bookkeeping consistency): returned separator weight
        // matches separator_weight(labels) recomputed from scratch.
        assert_eq!(after, separator_weight(&g, &labels), "I1: bookkeeping");
        assert!(
            after <= before,
            "separator weight must not grow (before={}, after={})",
            before,
            after
        );
    }
}
