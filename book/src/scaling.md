# Scaling

Symmetric scaling replaces the system `A·x = b` with a congruent system
that is better conditioned, then maps the solution back. feral applies a
single per-variable scaling vector `s` symmetrically:

```text
    (D·A·D)·(D⁻¹·x) = D·b ,   D = diag(s)
```

so the **same** vector `s` pre-scales the right-hand side and post-scales
the solution — the `D⁻¹` cancels algebraically. Concretely, each matrix
entry `a[i,j]` is multiplied by `s[i]·s[j]` during frontal assembly, the
RHS is multiplied by `s` before the forward sweep, and the solution is
multiplied by `s` after the backward sweep. Good scaling pulls the
matrix entries toward unit magnitude, which improves pivot quality and
keeps the certified inertia trustworthy on KKT-shaped systems.

## Choosing a strategy

The strategy is a `feral::scaling::ScalingStrategy`, set on the `Solver`:

```rust,ignore
use feral::Solver;
use feral::scaling::ScalingStrategy;

let solver = Solver::new().with_scaling(ScalingStrategy::Auto);
```

The variants:

| Variant | What it computes |
|---------|------------------|
| `Auto` (default) | Picks `Mc64Symmetric` or `InfNorm` per matrix; see below. |
| `InfNorm` | Knight–Ruiz iterative ∞-norm equilibration. |
| `Mc64Symmetric` | Matching-based (Duff–Koster) symmetric scaling. |
| `Identity` | No scaling (all ones) — the fast path. |
| `External(Vec<f64>)` | A scaling vector you supply, in user ordering. |

### InfNorm — equilibration

`InfNorm` runs an iterative ∞-norm equilibration in the symmetry-preserving
style of Ruiz: it computes a diagonal `D` so that every row (and, by
symmetry, every column) of `D·A·D` has ∞-norm near one. It needs only a
handful of sparse mat-vec passes, requires no matching solve, and
preserves symmetry exactly. It is the right default for matrices that are
already roughly balanced.

See Ruiz (2001) and Knight, Ruiz & Uçar (2014) below.

### Mc64Symmetric — matching-based

`Mc64Symmetric` builds the scaling from a maximum-weight bipartite
matching on the log-magnitude graph (the MC64 family). The asymmetric
dual variables `u`, `v` of the matching are symmetrized into a single
per-variable factor

```text
    s_i = exp((u_i + v_i) / 2)
```

which is exactly the scaling MUMPS and SPRAL SSIDS use for symmetric
indefinite matrices. It is more expensive than equilibration (it solves a
matching) but dramatically better on arrow-shaped KKT systems, where a
few dense coupling rows otherwise dominate the norms.

See Duff & Koster (1999, 2001) and Duff & Pralet (2005) below.

### Auto — adaptive routing

`Auto` (the default) inspects the sparsity pattern and routes to
`Mc64Symmetric` only when the matrix looks like an arrow/saddle-point
KKT — specifically when **both**:

- a large fraction of columns are structurally diagonal-only (slack
  columns), and
- at least one column is dense relative to the rest (an arrow head).

Otherwise it routes to `InfNorm`. This split matters: arrow-KKT matrices
can improve by one to two orders of magnitude under MC64, while banded
KKTs can *regress* under it, so the router keeps each on the strategy
that helps. `Auto` additionally applies fallback guards — if the matrix
is already well equilibrated, or if the matching produces a degenerate
(over-wide) scaling, it falls back to `InfNorm`.

> The exact thresholds are implementation details that have been tuned
> against the benchmark corpus and may change between releases; treat
> `Auto` as "feral picks a reasonable scaling for this matrix" and reach
> for an explicit variant only when you have measured a better choice.

## References

Matching-based scaling (MC64 family):

- I. S. Duff and J. Koster. *The Design and Use of Algorithms for
  Permuting Large Entries to the Diagonal of Sparse Matrices.* SIAM J.
  Matrix Anal. Appl. 20(4):889–901, 1999.
  [doi:10.1137/S0895479897317661](https://doi.org/10.1137/S0895479897317661)
- I. S. Duff and J. Koster. *On Algorithms for Permuting Large Entries to
  the Diagonal of a Sparse Matrix.* SIAM J. Matrix Anal. Appl.
  22(4):973–996, 2001.
  [doi:10.1137/S0895479899358443](https://doi.org/10.1137/S0895479899358443)
- I. S. Duff and S. Pralet. *Strategies for Scaling and Pivoting for
  Sparse Symmetric Indefinite Problems.* SIAM J. Matrix Anal. Appl.
  27(2):313–340, 2005.
  [doi:10.1137/04061043X](https://doi.org/10.1137/04061043X)

Equilibration (Ruiz / Knight–Ruiz family):

- D. Ruiz. *A Scaling Algorithm to Equilibrate Both Rows and Columns
  Norms in Matrices.* Tech. Report RAL-TR-2001-034, Rutherford Appleton
  Laboratory, 2001.
  [PDF](https://www.numerical.rl.ac.uk/media/reports/drRAL2001034.pdf)
- P. A. Knight, D. Ruiz, and B. Uçar. *A Symmetry Preserving Algorithm
  for Matrix Scaling.* SIAM J. Matrix Anal. Appl. 35(3):931–955, 2014.
  [doi:10.1137/110825753](https://doi.org/10.1137/110825753)
