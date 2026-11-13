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

use std::marker::PhantomData;

use crate::dense::factor::{ContribLayout, FrontalFactors};
use crate::error::FeralError;

/// Entries in the first `cols` columns of a packed lower triangle of
/// order `dim` (`Σ_{k<cols} (dim − k)`); also the offset of column `cols`.
#[inline]
fn packed_len(dim: usize, cols: usize) -> usize {
    cols * (2 * dim - cols + 1) / 2
}

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
        packed_len(dim, dim)
    }
    #[inline]
    fn column_start(dim: usize, j: usize) -> usize {
        packed_len(dim, j)
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

/// Add `block` into the lower triangle of the `dest_n × dest_n`
/// column-major `dest`: block-local index `i` lands at position
/// `dest_of(i)` (`usize::MAX`: not in `dest`, skipped). Zeros are
/// skipped, as they always were.
#[inline]
pub(crate) fn extend_add<L: BlockLayout>(
    block: &ContribBlock<L>,
    dest_of: impl Fn(usize) -> usize,
    dest: &mut [f64],
    dest_n: usize,
) {
    for cj in 0..block.dim {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A square and a packed block with the same values extend-add the
    /// same cells.
    #[test]
    fn square_and_packed_blocks_extend_add_alike() {
        let dim = 4;
        let value = |i: usize, j: usize| (1 + i * dim + j) as f64;
        let mut square = vec![0.0; dim * dim];
        let mut packed = Vec::new();
        for j in 0..dim {
            for i in j..dim {
                square[j * dim + i] = value(i, j);
                packed.push(value(i, j));
            }
        }
        fn block<L: BlockLayout>(dim: usize, data: Vec<f64>) -> ContribBlock<L> {
            ContribBlock {
                row_indices: (0..dim).collect(),
                data,
                dim,
                n_delayed: 0,
                sub_sym_area: 0,
                chain_excess: 0,
                layout: PhantomData,
            }
        }
        let sq: ContribBlock<Square> = block(dim, square);
        let pk: ContribBlock<Packed> = block(dim, packed);
        let dest_of = |i: usize| [2, usize::MAX, 0, 4][i];
        let (mut a, mut b) = (vec![0.0; 25], vec![0.0; 25]);
        extend_add(&sq, dest_of, &mut a, 5);
        extend_add(&pk, dest_of, &mut b, 5);
        assert_eq!(a, b);
        assert!(a.iter().any(|&x| x != 0.0));
    }
}
