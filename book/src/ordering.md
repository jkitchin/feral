# Fill-reducing ordering

Before factoring, feral permutes the matrix to reduce **fill-in** — the
new nonzeros created during elimination. A good ordering can change the
factor size and factorization time by orders of magnitude, so the choice
of ordering is the single most important tuning knob for large sparse
problems. All orderings are clean-room implementations in pure Rust
(workspace crates `feral_amd`, `feral_amf`, `feral_metis`,
`feral_scotch`, `feral_kahip`); none link the original C/Fortran
libraries.

## Choosing a method

The method is a `feral::symbolic::OrderingMethod`, set on the `Solver`:

```rust,ignore
use feral::Solver;
use feral::symbolic::OrderingMethod;

let solver = Solver::new().with_ordering(OrderingMethod::Auto);
```

The variants:

| Variant | Algorithm |
|---------|-----------|
| `Amd` | Approximate Minimum Degree. |
| `Amf` | Approximate Minimum Fill (HAMF, the MUMPS variant). |
| `MetisND` | METIS-style multilevel nested dissection. |
| `ScotchND` | Scotch-style nested dissection. |
| `KahipND` | KaHIP flow-based nested dissection with data reduction. |
| `Auto` | Adaptive: pick a method from cheap pattern metrics. |
| `AutoRace` | Run several candidates and keep the smallest factor. |

### Local (greedy) orderings

- **`Amd`** — Approximate Minimum Degree eliminates the vertex of
  smallest *approximate* degree at each step, using a quotient-graph
  model with element absorption and supervariable detection. It is fast,
  low-memory, and the strongest general-purpose choice for small and
  medium matrices. This is the default for `symbolic_factorize`.
- **`Amf`** — Approximate Minimum Fill scores by an estimate of the
  *fill* a pivot would create rather than its degree. It is the HAMF4
  variant used by MUMPS, and tends to beat AMD slightly on the medium
  KKT systems that dominate interior-point workloads.

### Nested dissection orderings

Nested dissection recursively finds a small vertex **separator**, numbers
it last, and recurses on the two halves. It produces asymptotically
better orderings than greedy methods on large, mesh-like graphs.

- **`MetisND`** — multilevel nested dissection: coarsen the graph by
  heavy-edge matching, bisect at the coarsest level, then uncoarsen with
  Fiduccia–Mattheyses refinement, falling back to AMD on small leaves.
- **`ScotchND`** — multilevel nested dissection with graph compression
  and direct vertex-separator refinement.
- **`KahipND`** — flow-based nested dissection: it computes separators
  with max-flow/min-cut local improvement and applies exhaustive data
  reduction (degree-1/2, twin, and indistinguishable-vertex rules) before
  dissecting. Highest quality and highest setup cost; it is never chosen
  automatically.

### Automatic selection

- **`Auto`** chooses a method from a few cheap pattern metrics: very
  large, very sparse graphs are sent to `Amd` (nested dissection's
  separator search does not pay off there), and everything else follows
  the size-based default — `Amf` for smaller matrices, `MetisND` for
  larger ones. The crossover and the "very large, very sparse" gate are
  tuned against the benchmark corpus and may change between releases.
- **`AutoRace`** runs the symbolic analysis for several candidate
  orderings (`Amd`, `MetisND`, `ScotchND`, `KahipND`) and keeps the one
  whose factor is predicted smallest. It costs roughly the sum of those
  symbolic passes — worth it when one factorization is reused for many
  solves, since symbolic analysis is amortized and the numeric phase
  dominates.

> Symbolic analysis (ordering + supernode detection) runs once and is
> cached on the `Solver`; refactorizing the same pattern with new values
> reuses it. So an expensive ordering like `KahipND` or `AutoRace` is
> cheap in amortized terms across a Newton run or a refactorization loop.

## References

Greedy orderings:

- P. R. Amestoy, T. A. Davis, and I. S. Duff. *An Approximate Minimum
  Degree Ordering Algorithm.* SIAM J. Matrix Anal. Appl. 17(4):886–905,
  1996.
  [doi:10.1137/S0895479894278952](https://doi.org/10.1137/S0895479894278952)
- P. R. Amestoy, T. A. Davis, and I. S. Duff. *Algorithm 837: AMD, an
  Approximate Minimum Degree Ordering Algorithm.* ACM Trans. Math. Softw.
  30(3):381–388, 2004.
  [doi:10.1145/1024074.1024081](https://doi.org/10.1145/1024074.1024081)
- A. George and J. W. H. Liu. *Computer Solution of Large Sparse Positive
  Definite Systems.* Prentice-Hall, 1981. ISBN 0-13-165274-5. (Minimum-fill
  and elimination-tree foundations.)

Nested dissection:

- A. George. *Nested Dissection of a Regular Finite Element Mesh.* SIAM J.
  Numer. Anal. 10(2):345–363, 1973.
  [doi:10.1137/0710032](https://doi.org/10.1137/0710032)
- R. J. Lipton, D. J. Rose, and R. E. Tarjan. *Generalized Nested
  Dissection.* SIAM J. Numer. Anal. 16(2):346–358, 1979.
  [doi:10.1137/0716027](https://doi.org/10.1137/0716027)
- G. Karypis and V. Kumar. *A Fast and High Quality Multilevel Scheme for
  Partitioning Irregular Graphs.* SIAM J. Sci. Comput. 20(1):359–392,
  1998.
  [doi:10.1137/S1064827595287997](https://doi.org/10.1137/S1064827595287997)
- F. Pellegrini and J. Roman. *Scotch: A Software Package for Static
  Mapping by Dual Recursive Bipartitioning of Process and Architecture
  Graphs.* HPCN Europe, LNCS 1067:493–498, 1996.
  [doi:10.1007/3-540-61142-8_588](https://doi.org/10.1007/3-540-61142-8_588)
- P. Sanders and C. Schulz. *Engineering Multilevel Graph Partitioning
  Algorithms.* ESA 2011, LNCS 6942:469–480.
  [doi:10.1007/978-3-642-23719-5_40](https://doi.org/10.1007/978-3-642-23719-5_40)
- W. Ost, C. Schulz, and D. Strash. *Engineering Data Reduction for Nested
  Dissection.* ALENEX 2021, 113–127.
  [doi:10.1137/1.9781611976472.9](https://doi.org/10.1137/1.9781611976472.9)

Refinement primitive:

- C. M. Fiduccia and R. M. Mattheyses. *A Linear-Time Heuristic for
  Improving Network Partitions.* 19th Design Automation Conference,
  175–181, 1982.
  [doi:10.1109/DAC.1982.1585498](https://doi.org/10.1109/DAC.1982.1585498)
