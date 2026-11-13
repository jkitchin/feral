//! Contribution blocks and their assembly into parent fronts.
//!
//! A supernode's contribution block (the Schur complement of its front on
//! the rows it does not eliminate) waits until its parent is assembled.
//! Its storage is a type parameter ([`BlockLayout`]):
//!
//! - **Square** ([`Square`]): `c²` entries, the lower triangle carrying
//!   the values. The parallel task-graph driver holds blocks this way, as
//!   it always has.
//! - **Packed** ([`Packed`]): the lower triangle packed column-major,
//!   `c·(c+1)/2` entries. The sequential drivers (the plain one, which the
//!   parallel entry point also uses for a tree with too little task
//!   parallelism, and the Schur one) hold blocks this way; the dense
//!   kernel writes them so (`ContribLayout::PackedLower`).
//!
//! **Accumulators** (sequential drivers only, opt-in
//! `EarlyAssembly::WideParents`; `dev/research/wide-tree-contribution-memory.md`).
//! Under a parent whose children's blocks add up to more than twice its
//! own front — a wide star, thousands of children whose blocks land in
//! one small front — holding every block until the parent is assembled
//! costs all of them at once.
//! At the child position `symbolic::accumulate_from` names, the parent
//! gets an *accumulator*: its frontal over its static rows `[own |
//! trailing]`, seeded with its original entries. That child and the
//! siblings held before it are extend-added into it, and so is every later
//! sibling as it is deposited. When the parent is assembled the
//! accumulator becomes its frontal.
//!
//! **Memory.** Everything else stays as the classic scheme has it, so the
//! peak live heap is the classic one plus bookkeeping
//! (`symbolic::accumulate_from`): an absorbed block, or a remainder, is
//! freed rather than pooled for the next kernel output (a pooled buffer
//! only grows, and the parent's own block would keep its capacity
//! alive), and once an adopted accumulator's front is factored it is
//! freed and the idle frontal pool grows by the same rule
//! (`dense::matrix::grow_pooled_front`) as if it had held that front.
//! No returned factor keeps a contribution buffer (a root's empty block
//! takes none from the pool), so the retained factor is the same.
//!
//! **Bit-identity.** Assembly is unchanged arithmetic in an unchanged
//! order. Every frontal cell receives its original entry by assignment,
//! then one `+=` per child that reaches it, children in `children`
//! order, zeros skipped. The classic scheme extend-adds the children in
//! that order after the original entries are scattered. An accumulator
//! is seeded with the original entries when it opens, and the sequential
//! drivers deposit children in ascending supernode order, which is
//! `children` order (a parent whose children are not ascending gets no
//! accumulator). Should deposits ever leave that order, the plans they
//! pass over are abandoned (`Accumulators::deposit`): the accumulator
//! keeps the prefix of children it absorbed, the rest are held whole and
//! added after it in children order, so the order still holds. Columns a child
//! *delays* stay behind in a packed remainder (the leading `n_delayed`
//! columns of the packed block hold exactly the cells with a delayed
//! index): a delayed column belongs to one child, and the original matrix
//! (lower triangle, own columns only) has no entries in it, so those
//! cells are zero until that one remainder adds into them — `0.0 + v`,
//! as in the classic scheme. The parent's trailing rows are its static
//! trailing rows whether or not pivots were delayed (a delayed column is
//! a descendant's column, below the parent's first column, so it never
//! joins the trailing set), which is what lets the accumulator be laid
//! out before the delays are known.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::marker::PhantomData;

use crate::dense::factor::{packed_lower_len, ContribLayout, FrontalFactors};
use crate::error::FeralError;
use crate::numeric::factorize::{EarlyAssembly, NumericParams};
use crate::sparse::csc::CscMatrix;
use crate::symbolic::SymbolicFactorization;

/// Storage of a held contribution block: where the lower part of each
/// column (rows `j..dim` of column `j`) starts.
pub(crate) trait BlockLayout {
    /// The layout the dense kernel is asked to write.
    const KERNEL: ContribLayout;
    /// Entries in a whole block of order `dim`.
    fn len(dim: usize) -> usize;
    /// Offset of row `j` of column `j` in a block of order `dim`.
    fn column_start(dim: usize, j: usize) -> usize;
}

/// `dim × dim` column-major; the strict upper triangle is zero.
#[derive(Debug)]
pub(crate) enum Square {}

/// Lower triangle packed column-major.
#[derive(Debug)]
pub(crate) enum Packed {}

impl BlockLayout for Square {
    const KERNEL: ContribLayout = ContribLayout::Square;
    #[inline]
    fn len(dim: usize) -> usize {
        dim * dim
    }
    #[inline]
    fn column_start(dim: usize, j: usize) -> usize {
        j * dim + j
    }
}

impl BlockLayout for Packed {
    const KERNEL: ContribLayout = ContribLayout::PackedLower;
    #[inline]
    fn len(dim: usize) -> usize {
        packed_lower_len(dim, dim)
    }
    #[inline]
    fn column_start(dim: usize, j: usize) -> usize {
        packed_lower_len(dim, j)
    }
}

/// Contribution block from a child supernode, stored in layout `L`.
///
/// Under delayed pivoting the leading `n_delayed` rows/columns are the
/// child's un-eliminated fully-summed columns (which re-enter pivot search
/// at the parent), and the rest is the Schur complement over the trailing
/// rows. `row_indices[..n_delayed]` are the global indices of the delayed
/// columns; `row_indices[n_delayed..]` the trailing rows (sorted).
#[derive(Debug)]
pub(crate) struct ContribBlock<L: BlockLayout> {
    /// Global row indices; first `n_delayed` delayed columns, then the
    /// trailing rows.
    pub(crate) row_indices: Vec<usize>,
    /// The lower triangle in layout `L`. A remainder (a packed block
    /// whose non-delayed columns went into an accumulator) keeps only its
    /// first `n_delayed` columns.
    data: Vec<f64>,
    /// Order of the block (`row_indices.len()`).
    pub(crate) dim: usize,
    /// Number of delayed fully-summed columns carried in this block.
    pub(crate) n_delayed: usize,
    /// Symbolic frontal area (`ncol * nrow`) summed over this node's
    /// subtree (issue #221 chain-cost backstop).
    pub(crate) sub_sym_area: usize,
    /// Extra frontal area spent by the delay chain ending at this node
    /// (issue #221). Only meaningful when `n_delayed > 0`.
    pub(crate) chain_excess: usize,
    layout: PhantomData<L>,
}

impl<L: BlockLayout> ContribBlock<L> {
    /// Take the dense kernel's contribution block out of `ff`, which the
    /// driver asked for in layout `L` (`L::KERNEL`). `front_rows` are the
    /// global indices of the front's rows. `None` when the front left no
    /// contribution; an error if the block's length does not match `L`.
    pub(crate) fn from_factor(
        ff: &mut FrontalFactors,
        front_rows: &[usize],
        sub_sym_area: usize,
        chain_excess: usize,
    ) -> Result<Option<Self>, FeralError> {
        let dim = ff.contrib_dim;
        if dim == 0 {
            return Ok(None);
        }
        if ff.contrib.len() != L::len(dim) {
            return Err(FeralError::InvalidInput(format!(
                "contribution block of order {dim} has {} entries, not the {:?} {}",
                ff.contrib.len(),
                L::KERNEL,
                L::len(dim)
            )));
        }
        let row_indices = (0..dim)
            .map(|cj| front_rows[ff.perm[ff.nelim + cj]])
            .collect();
        Ok(Some(Self {
            row_indices,
            data: std::mem::take(&mut ff.contrib),
            dim,
            n_delayed: ff.n_delayed,
            sub_sym_area,
            chain_excess,
            layout: PhantomData,
        }))
    }

    /// Rows `j..dim` of column `j`.
    #[inline]
    fn column(&self, j: usize) -> &[f64] {
        let start = L::column_start(self.dim, j);
        &self.data[start..start + (self.dim - j)]
    }

    /// Lower-triangle entry `(i, j)`, `i >= j`.
    #[inline]
    pub(crate) fn get(&self, i: usize, j: usize) -> f64 {
        debug_assert!(i >= j && i < self.dim);
        self.column(j)[i - j]
    }

    /// Hand the storage back for reuse as a kernel output buffer.
    pub(crate) fn into_buffer(self) -> Vec<f64> {
        self.data
    }
}

impl ContribBlock<Packed> {
    /// Drop the columns already assembled into an accumulator, keeping
    /// the delayed ones.
    fn keep_delayed_columns(&mut self) {
        self.data
            .truncate(packed_lower_len(self.dim, self.n_delayed));
        self.data.shrink_to_fit();
    }
}

/// Add the first `cols` columns of `block` into the lower triangle of the
/// `dest_n × dest_n` column-major `dest`: block-local index `i` lands at
/// position `dest_of(i)` (`usize::MAX`: not in `dest`, skipped). Zeros are
/// skipped, as they always were.
#[inline]
pub(crate) fn extend_add<L: BlockLayout>(
    block: &ContribBlock<L>,
    cols: usize,
    dest_of: impl Fn(usize) -> usize,
    dest: &mut [f64],
    dest_n: usize,
) {
    for cj in 0..cols {
        let dest_j = dest_of(cj);
        if dest_j == usize::MAX {
            continue;
        }
        for (ci, &val) in (cj..block.dim).zip(block.column(cj)) {
            let dest_i = dest_of(ci);
            if dest_i == usize::MAX || val == 0.0 {
                continue;
            }
            let (row, col) = if dest_i >= dest_j {
                (dest_i, dest_j)
            } else {
                (dest_j, dest_i)
            };
            dest[col * dest_n + row] += val;
        }
    }
}

/// A parent's frontal under construction, over its static rows
/// `[own | trailing]` (`SymbolicFactorization::static_rows`), in the
/// `SymmetricMatrix` layout so it can become the frontal unchanged.
#[derive(Debug)]
pub(crate) struct Accumulator {
    /// `n × n` column-major, lower triangle authoritative.
    pub(crate) data: Vec<f64>,
    /// `sub_sym_area` of the children absorbed whole (a child that left
    /// a remainder keeps its own).
    pub(crate) sub_sym_area: usize,
    /// The children it absorbed: the first `absorbed` of the parent's
    /// `children`. All of them, unless its plan was abandoned
    /// ([`Accumulators::deposit`]); the blocks of the rest are held whole.
    pub(crate) absorbed: usize,
}

/// Position of global row `r` in a supernode's static rows `[own |
/// sorted trailing]`; `None` for a row outside them (a delayed column).
#[inline]
fn static_position(
    static_rows: &[usize],
    first_col: usize,
    own_ncol: usize,
    r: usize,
) -> Option<usize> {
    if r < first_col {
        return None;
    }
    if r < first_col + own_ncol {
        return Some(r - first_col);
    }
    static_rows[own_ncol..]
        .binary_search(&r)
        .ok()
        .map(|t| own_ncol + t)
}

/// Read-only inputs to assembling into an accumulator.
pub(crate) struct AssemblyInputs<'a> {
    pub(crate) symbolic: &'a SymbolicFactorization,
    /// `P·A·Pᵀ`, lower triangle.
    pub(crate) permuted: &'a CscMatrix,
    /// Symmetric scaling in pivot order.
    pub(crate) scaling: &'a [f64],
}

/// A parent with an accumulator in this factorization's scheme.
#[derive(Debug)]
struct Plan {
    parent: usize,
    /// Child position at which the accumulator opens.
    open_at: usize,
    /// Children deposited in children order so far (a prefix of
    /// `children`).
    deposited: usize,
    /// Opened by the child at `open_at`.
    accumulator: Option<Accumulator>,
}

/// The accumulators of one sequential factorization: which parents get
/// one under its [`EarlyAssembly`] scheme, and how far each has got. Its
/// bookkeeping is a few words per planned parent; nothing per child.
#[derive(Debug)]
pub(crate) struct Accumulators {
    /// Ascending by parent.
    plans: Vec<Plan>,
    /// `(child, plan)`: the child each unfinished plan expects next, least
    /// first; a plan leaves it when finished or abandoned. The sequential drivers deposit in ascending
    /// supernode order and a planned parent's children ascend, so a
    /// deposit that belongs to a plan is the least expected child.
    expected: BinaryHeap<Reverse<(usize, usize)>>,
    /// Scratch: the positions of an absorbed block's rows in its parent's
    /// static rows.
    map: Vec<usize>,
}

impl Accumulators {
    /// The accumulators `params.early_assembly` opens on `symbolic`
    /// ([`SymbolicFactorization::wide_parent_plan`]), derived here for
    /// this factorization: one pass over the supernodes and their
    /// children, cheap next to the factorization. An accumulator is laid
    /// out over its parent's static rows, so none opens when
    /// `params.use_static_row_indices` is off.
    pub(crate) fn new(symbolic: &SymbolicFactorization, params: &NumericParams) -> Self {
        let plans_from = |pairs: &mut dyn Iterator<Item = (usize, usize)>| -> Vec<Plan> {
            pairs
                .map(|(parent, open_at)| Plan {
                    parent,
                    open_at,
                    deposited: 0,
                    accumulator: None,
                })
                .collect()
        };
        let mut plans = match params.early_assembly {
            _ if !params.use_static_row_indices => Vec::new(),
            EarlyAssembly::Off => Vec::new(),
            EarlyAssembly::WideParents => plans_from(&mut symbolic.wide_parent_plan()),
            #[cfg(feature = "__early-assembly-always")]
            EarlyAssembly::Always => plans_from(&mut symbolic.accumulator_plan(|_| Some(0))),
        };
        // The bookkeeping lives for the whole factorization: exact
        // capacity, nothing per child.
        plans.shrink_to_fit();
        let supernodes = &symbolic.supernodes;
        let mut expected = Vec::with_capacity(plans.len());
        for (index, plan) in plans.iter().enumerate() {
            if let Some(&first) = supernodes[plan.parent].children.first() {
                expected.push(Reverse((first, index)));
            }
        }
        Self {
            plans,
            expected: BinaryHeap::from(expected),
            map: Vec::new(),
        }
    }

    /// Record that `child` has been factored, its block (if any) in
    /// `blocks[child]`. Under no plan, or before its parent's opening
    /// child, the block stays there until the parent is assembled. At the
    /// opening child the parent's accumulator opens and absorbs the
    /// siblings held so far and the block; after it, the block is
    /// absorbed at once. A block with delayed columns leaves its
    /// remainder in `blocks`; a block absorbed whole is freed (not pooled:
    /// a pooled buffer only grows, and a later block of the parent would
    /// keep its capacity alive).
    ///
    /// The drivers deposit in ascending supernode order; a deposit out of
    /// that order abandons the plans it passes over (debug builds assert).
    pub(crate) fn deposit(
        &mut self,
        blocks: &mut [Option<ContribBlock<Packed>>],
        child: usize,
        inputs: &AssemblyInputs<'_>,
    ) -> Result<(), FeralError> {
        let in_order = self.deposit_in_any_order(blocks, child, inputs)?;
        debug_assert!(
            in_order,
            "early assembly: supernode {child} deposited after a planned child it should follow"
        );
        Ok(())
    }

    /// [`Self::deposit`] without the order assertion; `false` if `child`
    /// came after a child some plan still expected.
    ///
    /// A plan expecting a child below `child` cannot keep children order:
    /// that child either came earlier, out of order, or comes later. It is
    /// abandoned: its accumulator (if open) keeps the children it absorbed
    /// — always a prefix of `children` — and its later children's blocks
    /// stay whole in `blocks`, as under the classic scheme, so the
    /// parent's assembly adds them after the accumulator, still in
    /// children order (`factorize::assemble_children`). Every cell then
    /// receives the same additions in the same order as under `Off`, so
    /// the factors are unchanged; only the memory bound is lost for that
    /// parent. Popping every passed-over plan here keeps the least
    /// expected child at the top of `expected`, so one abandoned plan
    /// never stalls the others.
    fn deposit_in_any_order(
        &mut self,
        blocks: &mut [Option<ContribBlock<Packed>>],
        child: usize,
        inputs: &AssemblyInputs<'_>,
    ) -> Result<bool, FeralError> {
        let mut in_order = true;
        let plan_index = loop {
            let Some(&Reverse((expected, plan_index))) = self.expected.peek() else {
                return Ok(in_order);
            };
            if expected < child {
                // Abandoned: it never enters `expected` again.
                self.expected.pop();
                in_order = false;
            } else if expected == child {
                self.expected.pop();
                break plan_index;
            } else {
                // Under no plan (or a repeated deposit): held.
                return Ok(in_order);
            }
        };
        let plan = &mut self.plans[plan_index];
        let parent = plan.parent;
        let children = &inputs.symbolic.supernodes[parent].children;
        let position = plan.deposited;
        plan.deposited += 1;
        if let Some(&next) = children.get(plan.deposited) {
            self.expected.push(Reverse((next, plan_index)));
        }
        if position < plan.open_at {
            return Ok(in_order);
        }
        let first = if position == plan.open_at {
            0
        } else {
            position
        };
        let accumulator = plan
            .accumulator
            .get_or_insert_with(|| open_accumulator(parent, inputs));
        for &c in &children[first..=position] {
            if let Some(held) = blocks[c].take() {
                absorb(parent, c, held, inputs, &mut self.map, accumulator, blocks)?;
            }
        }
        accumulator.absorbed = position + 1;
        Ok(in_order)
    }

    /// Supernode `p`'s accumulator, if its scheme opened one. The blocks
    /// its children left are in `blocks`, where `p`'s assembly finds them
    /// as it finds held blocks: remainders of the children the
    /// accumulator absorbed ([`Accumulator::absorbed`]), and the whole
    /// blocks of any later ones — children of a plan its deposits
    /// abandoned, or (a driver error) not yet deposited, which the
    /// deposit of `p` itself then abandons and asserts on.
    pub(crate) fn take(&mut self, p: usize) -> Option<Accumulator> {
        let index = self.plans.binary_search_by_key(&p, |pl| pl.parent).ok()?;
        self.plans[index].accumulator.take()
    }
}

/// Extend-add child `c`'s `block` into `p`'s accumulator `acc`. A block
/// with delayed columns stays behind in `blocks` as its remainder;
/// otherwise it is freed.
fn absorb(
    p: usize,
    c: usize,
    mut block: ContribBlock<Packed>,
    inputs: &AssemblyInputs<'_>,
    map: &mut Vec<usize>,
    acc: &mut Accumulator,
    blocks: &mut [Option<ContribBlock<Packed>>],
) -> Result<(), FeralError> {
    let sn = &inputs.symbolic.supernodes[p];
    let rows = inputs.symbolic.static_rows(p);
    map.clear();
    map.extend(
        block
            .row_indices
            .iter()
            .map(|&r| static_position(rows, sn.first_col, sn.ncol(), r).unwrap_or(usize::MAX)),
    );
    if map[block.n_delayed..].contains(&usize::MAX) {
        return Err(FeralError::InvalidInput(format!(
            "early assembly: a trailing row of supernode {c} is not a static row of its parent {p}"
        )));
    }
    extend_add(&block, block.dim, |i| map[i], &mut acc.data, rows.len());
    if block.n_delayed == 0 {
        acc.sub_sym_area = acc.sub_sym_area.saturating_add(block.sub_sym_area);
        return Ok(());
    }
    block.keep_delayed_columns();
    blocks[c] = Some(block);
    Ok(())
}

/// A fresh accumulator for supernode `p`, holding its scaled original
/// entries — assigned exactly as a classic front's step 1 assigns them.
fn open_accumulator(p: usize, inputs: &AssemblyInputs<'_>) -> Accumulator {
    let sn = &inputs.symbolic.supernodes[p];
    let rows = inputs.symbolic.static_rows(p);
    let n = rows.len();
    let own_ncol = sn.ncol();
    let (permuted, scaling) = (inputs.permuted, inputs.scaling);
    let mut data = vec![0.0f64; n * n];
    for k in 0..own_ncol {
        let gj = sn.first_col + k;
        let s_j = scaling[gj];
        for idx in permuted.col_ptr[gj]..permuted.col_ptr[gj + 1] {
            let gi = permuted.row_idx[idx];
            if let Some(i) = static_position(rows, sn.first_col, own_ncol, gi) {
                let val = permuted.values[idx] * scaling[gi] * s_j;
                let (row, col) = if i >= k { (i, k) } else { (k, i) };
                data[col * n + row] = val;
            }
        }
    }
    Accumulator {
        data,
        sub_sym_area: 0,
        absorbed: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbolic::{symbolic_factorize, SupernodeParams};

    /// A star of two-variable leaves (coupled to each other, one to each
    /// of two shared variables), analysed without amalgamation. Returns
    /// the analysis and the supernode with the most children (at least
    /// as many as [`VALUES`]).
    fn star() -> (SymbolicFactorization, usize) {
        let leaves = 4;
        let n = 2 + 2 * leaves;
        let (mut r, mut c, mut v) = (vec![0, 1], vec![0, 1], vec![1.0, 1.0]);
        for leaf in 0..leaves {
            let (a, b) = (2 + 2 * leaf, 3 + 2 * leaf);
            r.extend([a, b, b, a, b]);
            c.extend([a, b, a, 0, 1]);
            v.extend([4.0, 4.0, 0.5, 0.5, 0.5]);
        }
        let a = CscMatrix::from_triplets(n, &r, &c, &v).unwrap();
        let params = SupernodeParams {
            nemin: 1,
            ..SupernodeParams::default()
        };
        let symbolic = symbolic_factorize(&a, &params).unwrap();
        let root = (0..symbolic.supernodes.len())
            .max_by_key(|&p| symbolic.supernodes[p].children.len())
            .unwrap();
        assert!(symbolic.supernodes[root].children.len() >= VALUES.len());
        (symbolic, root)
    }

    /// A packed block over `rows`, the first `n_delayed` delayed, every
    /// cell `value`.
    fn packed(rows: Vec<usize>, n_delayed: usize, value: f64) -> ContribBlock<Packed> {
        let dim = rows.len();
        ContribBlock {
            row_indices: rows,
            data: vec![value; packed_lower_len(dim, dim)],
            dim,
            n_delayed,
            sub_sym_area: 0,
            chain_excess: 0,
            layout: PhantomData,
        }
    }

    /// Every parent accumulates (`EarlyAssembly::Always`).
    fn always() -> NumericParams {
        NumericParams {
            early_assembly: EarlyAssembly::Always,
            ..NumericParams::default()
        }
    }

    /// The first children's values. Absorbing the third before the
    /// second changes the sum even when the first goes first:
    /// `(1e16 + 1) − 1e16 = 0` but `(1e16 − 1e16) + 1 = 1`.
    const VALUES: [f64; 3] = [1e16, 1.0, -1e16];

    /// Inputs to assembling into the star's root (no original entries).
    struct Fixture {
        symbolic: SymbolicFactorization,
        root: usize,
        permuted: CscMatrix,
        scaling: Vec<f64>,
    }

    impl Fixture {
        fn new() -> Self {
            let (symbolic, root) = star();
            let permuted = CscMatrix::from_triplets(symbolic.n, &[], &[], &[]).unwrap();
            let scaling = vec![1.0; symbolic.n];
            Self {
                symbolic,
                root,
                permuted,
                scaling,
            }
        }

        fn inputs(&self) -> AssemblyInputs<'_> {
            AssemblyInputs {
                symbolic: &self.symbolic,
                permuted: &self.permuted,
                scaling: &self.scaling,
            }
        }

        fn child(&self, k: usize) -> usize {
            self.symbolic.supernodes[self.root].children[k]
        }

        fn n_children(&self) -> usize {
            self.symbolic.supernodes[self.root].children.len()
        }

        /// A block over the root's static rows, every cell `value`.
        fn block(&self, value: f64) -> ContribBlock<Packed> {
            packed(self.symbolic.static_rows(self.root).to_vec(), 0, value)
        }

        /// [`Self::block`] led by child `k`'s first column, delayed.
        fn delaying_block(&self, k: usize, value: f64) -> ContribBlock<Packed> {
            let mut rows = vec![self.symbolic.supernodes[self.child(k)].first_col];
            rows.extend_from_slice(self.symbolic.static_rows(self.root));
            packed(rows, 1, value)
        }

        fn blocks(&self) -> Vec<Option<ContribBlock<Packed>>> {
            (0..self.symbolic.supernodes.len()).map(|_| None).collect()
        }

        /// Deposit supernodes `0..=last` in ascending order, as the
        /// sequential drivers do: `make(k)` gives the root's child `k`'s
        /// block, every other supernode leaves none.
        fn deposit_through(
            &self,
            accumulators: &mut Accumulators,
            blocks: &mut [Option<ContribBlock<Packed>>],
            last: usize,
            make: impl Fn(usize) -> Option<ContribBlock<Packed>>,
        ) -> Result<(), FeralError> {
            let children = &self.symbolic.supernodes[self.root].children;
            for s in 0..=last {
                blocks[s] = children.iter().position(|&c| c == s).and_then(&make);
                accumulators.deposit(blocks, s, &self.inputs())?;
            }
            Ok(())
        }

        /// Deposit every supernode below the root, `make(k)` giving the
        /// root's child `k`'s block, then take the root's accumulator.
        fn assemble(
            &self,
            make: impl Fn(usize) -> Option<ContribBlock<Packed>>,
        ) -> (Accumulator, Vec<Option<ContribBlock<Packed>>>) {
            let mut accumulators = Accumulators::new(&self.symbolic, &always());
            let mut blocks = self.blocks();
            self.deposit_through(&mut accumulators, &mut blocks, self.root - 1, make)
                .unwrap();
            let acc = accumulators.take(self.root).unwrap();
            (acc, blocks)
        }
    }

    /// The children enter the accumulator in children order: every cell
    /// is the in-order sum of [`VALUES`], bit for bit, which a different
    /// order would not give.
    #[test]
    fn children_accumulate_in_children_order() {
        let fixture = Fixture::new();
        let (acc, _) = fixture.assemble(|k| VALUES.get(k).map(|&v| fixture.block(v)));
        let in_order = VALUES.iter().fold(0.0, |s, &v| s + v);
        let swapped = (VALUES[0] + VALUES[2]) + VALUES[1];
        assert_ne!(in_order.to_bits(), swapped.to_bits());
        let n = fixture.symbolic.static_rows(fixture.root).len();
        for j in 0..n {
            for i in j..n {
                assert_eq!(acc.data[j * n + i].to_bits(), in_order.to_bits());
            }
        }
    }

    /// A deposit out of children order abandons the plan instead of
    /// failing: the accumulator keeps the children it absorbed, the later
    /// ones stay whole, and adding those after it in children order gives
    /// every cell the in-order sum of [`VALUES`], bit for bit. The
    /// abandoned plan leaves nothing expected behind to stall other plans.
    #[test]
    fn a_deposit_out_of_children_order_falls_back_to_held_blocks() {
        let fixture = Fixture::new();
        let mut accumulators = Accumulators::new(&fixture.symbolic, &always());
        let mut blocks = fixture.blocks();
        // Ascending, as the drivers deposit, except that child 1 comes
        // right after child 2: the next supernode after it passes over it.
        let (one, two) = (fixture.child(1), fixture.child(2));
        let mut order: Vec<usize> = (0..fixture.root).filter(|&s| s != one).collect();
        let at = order.iter().position(|&s| s == two).unwrap() + 1;
        order.insert(at, one);
        let children = &fixture.symbolic.supernodes[fixture.root].children;
        for s in order {
            blocks[s] = children
                .iter()
                .position(|&c| c == s)
                .and_then(|k| VALUES.get(k).map(|&v| fixture.block(v)));
            let in_order = accumulators
                .deposit_in_any_order(&mut blocks, s, &fixture.inputs())
                .unwrap();
            assert_eq!(in_order, s != one + 1, "supernode {s}");
        }
        // Nothing below the root is still expected: no plan stalls.
        assert!(accumulators
            .expected
            .iter()
            .all(|&Reverse((c, _))| c >= fixture.root));
        let mut acc = accumulators.take(fixture.root).unwrap();
        assert_eq!(acc.absorbed, 1);
        let rows = fixture.symbolic.static_rows(fixture.root);
        let n = rows.len();
        for k in acc.absorbed..fixture.n_children() {
            if let Some(block) = blocks[fixture.child(k)].take() {
                extend_add(&block, block.dim, |i| i, &mut acc.data, n);
            }
        }
        let in_order = VALUES.iter().fold(0.0, |s, &v| s + v);
        for j in 0..n {
            for i in j..n {
                assert_eq!(acc.data[j * n + i].to_bits(), in_order.to_bits());
            }
        }
    }

    /// The accumulator carries the `sub_sym_area` of the children it
    /// absorbed whole (the issue-#221 chain backstop reads it); a child
    /// that delayed keeps its own on its remainder, which holds its one
    /// delayed column.
    #[test]
    fn accumulator_sums_the_area_of_children_absorbed_whole() {
        let fixture = Fixture::new();
        let areas = [3, 5, 7];
        let (acc, blocks) = fixture.assemble(|k| {
            let mut b = if k == 1 {
                fixture.delaying_block(k, 1.0)
            } else {
                fixture.block(1.0)
            };
            b.sub_sym_area = areas.get(k).copied()?;
            Some(b)
        });
        assert_eq!(acc.sub_sym_area, areas[0] + areas[2]);
        let remainder = blocks[fixture.child(1)].as_ref().unwrap();
        assert_eq!(remainder.sub_sym_area, areas[1]);
        assert_eq!(remainder.data.len(), packed_lower_len(remainder.dim, 1));
    }

    /// A block whose trailing row is not one of the parent's static rows
    /// cannot be absorbed: an error.
    #[test]
    fn a_trailing_row_outside_the_parent_fails() {
        let fixture = Fixture::new();
        let mut accumulators = Accumulators::new(&fixture.symbolic, &always());
        let mut blocks = fixture.blocks();
        let bad = || {
            let mut bad = fixture.block(1.0);
            bad.row_indices[0] = fixture.symbolic.supernodes[fixture.child(0)].first_col;
            Some(bad)
        };
        let deposited =
            fixture.deposit_through(&mut accumulators, &mut blocks, fixture.child(0), |_| bad());
        assert!(matches!(deposited, Err(FeralError::InvalidInput(_))));
    }

    /// A square and a packed block with the same values extend-add the
    /// same cells.
    #[test]
    fn square_and_packed_blocks_extend_add_alike() {
        let dim = 4;
        let value = |i: usize, j: usize| (1 + i * dim + j) as f64;
        let mut square = vec![0.0; dim * dim];
        let mut packed_data = Vec::new();
        for j in 0..dim {
            for i in j..dim {
                square[j * dim + i] = value(i, j);
                packed_data.push(value(i, j));
            }
        }
        let rows: Vec<usize> = (0..dim).collect();
        let sq = ContribBlock::<Square> {
            row_indices: rows.clone(),
            data: square,
            dim,
            n_delayed: 0,
            sub_sym_area: 0,
            chain_excess: 0,
            layout: PhantomData,
        };
        let pk = ContribBlock::<Packed> {
            data: packed_data,
            ..packed(rows, 0, 0.0)
        };
        let dest_of = |i: usize| [2, usize::MAX, 0, 4][i];
        let (mut a, mut b) = (vec![0.0; 25], vec![0.0; 25]);
        extend_add(&sq, dim, dest_of, &mut a, 5);
        extend_add(&pk, dim, dest_of, &mut b, 5);
        assert_eq!(a, b);
        assert!(a.iter().any(|&x| x != 0.0));
    }
}
