# MC64 matching-based scaling for symmetric indefinite factorization

## Purpose

This is the research note for Phase 2.2.1 of the FERAL project plan
(`dev/plans/phase-2-planning.md`). The goal is to implement pure-Rust
MC64-style matching-based global scaling in feral's sparse multifrontal
pipeline, matching the behavior that canonical Fortran MUMPS and
SPRAL/SSIDS apply by default to symmetric indefinite KKT matrices.

Phase 2.1.2's sanity check run (commit `c01235f`,
`examples/triage_large_cresc132.rs`) established that feral produces
residuals of order `1e+03` to `1e+14` on matrices with `n > 500` while
canonical MUMPS and SSIDS produce residuals at machine precision on
the same inputs. Both the `dev/phase1-retrospective.org` ACOPP30
residual gap finding and this sanity check point to the same root
cause: feral does not apply a global scaling to the matrix before
factorization, while MUMPS and SSIDS both apply a matching-based
scaling by default. This note is the design document for closing
that gap.

## Canonical references

Primary sources that must be on the desk while implementing:

- citet:duff2001mc64 — "On Algorithms for Permuting Large Entries to
  the Diagonal of a Sparse Matrix" (SIMAX 22(4), 2001). Sections 3
  and 4 specify the six MC64 strategies; strategy 5 (maximum product
  of diagonal entries, with dual variables returned) is the one both
  MUMPS and SSIDS default to for symmetric indefinite matrices. The
  Hungarian/shortest-augmenting-path algorithm in §4 is the kernel
  to implement.
- citet:duff1999permuting — "The Design and Use of Algorithms for
  Permuting Large Entries..." (SIMAX 20(4), 1999). Motivation paper;
  gives the theoretical background and the argument for why matching
  preprocessing matters for stability of direct solvers.
- citet:duff2005symmetric — "Strategies for Scaling and Pivoting for
  Sparse Symmetric Indefinite Problems" (SIMAX 27(2), 2005). This is
  the paper that defines the symmetric average
  `s_i = exp((u_i + v_i) / 2)` that transforms the asymmetric MC64
  matching output into a single symmetric scaling vector. Both MUMPS
  and SSIDS implement this.
- citet:hogg2013pivoting — background on how matching-based
  preprocessing plugs into the SSIDS pivoting pipeline.

Secondary sources (concrete reference implementations that feral's
Rust port must match):

- `ref/mumps/src/dana_mtrans.F` — MUMPS 5.8.2 MC64 kernel, a direct
  translation of Jacko Koster's HSL MC64. 1196 lines. The routines
  of interest are `DMUMPS_MTRANSW` (lines 791–1081, strategy 5/6 with
  dual variables) and `DMUMPS_MTRANS_DRIVER` (the dispatcher).
- `ref/mumps/src/dana_aux.F` lines 1316–1984 — `DMUMPS_ANA_O`, the
  analysis-phase driver. Shows how MC64 is called, how the duals are
  exponentiated into row/column scalings, and how symmetric averaging
  is applied for `SYM=2`.
- `ref/mumps/src/dana_LDLT_preprocess.F` lines 270–460 —
  `DMUMPS_SYM_MWM`, the 2×2 pivot pair detection from the symmetric
  matching cycles. Not needed for the minimal port.
- `ref/mumps/src/dfac_dist_arrowheads_omp.F` lines 1021–1024 — where
  MUMPS actually applies the scaling: inside the frontal-assembly
  loop, as `VAL *= ROWSCA(IOLD) * COLSCA(JOLD)`. Not a separate
  pre-transform of the matrix.
- `ref/mumps/src/dsol_driver.F` lines 4144–4252 and 6540–6577 — how
  MUMPS pre-scales the RHS and post-scales the solution.
- `ref/spral/src/scaling.f90` — SPRAL's own BSD-licensed reimplementation
  of the Hungarian algorithm. `hungarian_match` at lines 938–1171 is
  the kernel, `hungarian_wrapper` at 597–801 is the log-transform +
  column-max normalization wrapper, and `hungarian_scale_sym_int64`
  at 134–170 is the entry point that returns a symmetric scaling
  vector. Reads as fewer lines than MUMPS because SPRAL is a clean
  reimplementation rather than a literal port.
- `ref/spral/src/ssids/cpu/kernels/assemble.hxx` lines 50–80 — the
  SSIDS analog of the MUMPS assembly-time scaling.
- `ref/spral/src/ssids/fkeep.F90` lines 253–308 — SSIDS's solve-time
  pre-scale and post-scale.
- `ref/spral/src/match_order.f90` — SPRAL's ordering path that
  compresses matched pairs; not required for the minimal port but
  worth understanding as a Phase 2.6 alternative.

## The algorithm in one paragraph

Given a sparse symmetric matrix `A`, form the weighted bipartite graph
where edge `(i, j)` has weight `log |A_{ij}|` (with edges for explicit
zero entries dropped). Solve the maximum-weight bipartite matching
problem on this graph by the Hungarian / shortest-augmenting-path
algorithm, which also produces optimal dual variables `u_i` (one per
row) and `v_j` (one per column) satisfying `u_i + v_j ≤ log |A_{ij}|`
for every edge with equality on the matched edges. Exponentiating the
duals gives row and column scaling vectors
`d^r_i = exp u_i` and `d^c_j = exp v_j`. For symmetric matrices,
these are averaged into a single per-variable scaling
`s_i = exp((u_i + v_i) / 2)`. Applying
`A ↦ diag(s) · A · diag(s)` before factorization produces a scaled
matrix whose largest entries in magnitude lie on the diagonal and
whose off-diagonals are bounded above by `1` in absolute value. The
factorization is then performed on the scaled matrix, and the solve
applies `b ↦ diag(s) · b` before the forward sweep and
`x ↦ diag(s) · x` after the backward sweep.

## What MUMPS and SSIDS do, precisely

The two expert reports (`mumps-expert` and `spral-expert` agents run
in parallel) agreed on the following framework. Differences are
called out explicitly.

### Analysis-phase plumbing

**MUMPS** (`ref/mumps/src/dana_aux.F:1316–1984`, `DMUMPS_ANA_O`):

1. Convert the user's input COO arrays into a column-compressed
   internal form, expanding the pattern to both triangles for
   `SYM=2` (`dana_aux.F:1526–1679`).
2. Drop duplicate entries via `DMUMPS_SUPPRESS_DUPPLI_VAL`, summing
   the absolute values (`dana_aux.F:1684`). This is important: the
   matching is on the absolute-value graph, so duplicates sharing an
   `(i,j)` must be combined in magnitude before taking the log.
3. Inject a zero entry for any missing structural diagonal so the
   matching can self-match in degenerate cases (`dana_aux.F:1666–1673`).
4. Call `DMUMPS_MTRANS_DRIVER` with strategy code `5` for "maximum
   product with dual variables returned" (`dana_aux.F:1428–1441,
   1696`).
5. Receive the matching `PERM[1..n]` and the dual variables in
   two scratch arrays.
6. Exponentiate the duals into `ROWSCA[i] = exp(u_i)` and
   `COLSCA[j] = exp(v_j)`, with an overflow guard that clamps any
   dual exceeding `log(huge)` to zero (which then maps to scaling 1
   via a defensive rewrite at `dana_aux.F:1809–1811`).
7. For `SYM=2`, average: `ROWSCA[j] = COLSCA[j] = exp((u_j + v_j)/2)`
   at `dana_aux.F:1865–1870`.
8. Call `DMUMPS_SYM_MWM` (`dana_LDLT_preprocess.F:270–460`) to walk
   the cycles of the matching permutation and identify 1×1 and 2×2
   pivot pairs. This is a pivoting optimization, not a correctness
   requirement for the scaling. **A minimal port does not need this
   step.**

The effective control parameters for MUMPS's default symmetric
indefinite path are `ICNTL(6)=7`, `ICNTL(8)=77`, `ICNTL(12)=0`, which
the driver resolves to `KEEP(23)=5`, `KEEP(52)=-2`, `KEEP(95)=2`
(`dana_driver.F:3680–3900`, `dmumps_driver.F:1287–1395`). `KEEP(23)=5`
selects MC64 strategy 5 and `KEEP(52)=-2` means "scaling precomputed
at analyze, apply at factorize".

**SSIDS** (`ref/spral/src/scaling.f90`):

1. SSIDS does **not** link to HSL MC64. It has a BSD-licensed
   reimplementation in `scaling.f90` lines 938–1171 (`hungarian_match`)
   with wrappers at 597–801 (`hungarian_wrapper`) and 134–170
   (`hungarian_scale_sym_int64`). The comment at `match_order.f90:1–4`
   notes the code is a "heavily modified version of HSL_MC80".
2. SSIDS's defaults for indefinite matrices are `options%ordering=1`
   (METIS only) and `options%scaling=0` (no scaling). **The user
   must explicitly request scaling** via `options%scaling=1` or
   `=3`. This is different from MUMPS, which applies matching-based
   scaling by default.
3. When scaling is requested, the input preprocessing is the same in
   spirit but differs in one detail: SSIDS normalizes the
   log-transformed values by subtracting the per-column maximum
   (`hungarian_wrapper` at `scaling.f90:652–657`). This makes the
   cost matrix non-negative, which is what `hungarian_match`
   expects. The column-max is then unwound after the matching runs
   so the returned duals are in the correct scale. MUMPS does the
   same thing internally in `DMUMPS_MTRANSW`.
4. For `sym=true`, SSIDS returns the scaling as
   `scaling[i] = exp((dualu[i] + dualv[i]) / 2)`
   (`scaling.f90:169`) — identical to MUMPS's formula.

### Application at factorization

Both solvers apply the scaling **during frontal assembly**, not as a
separate pre-transform of the matrix. When an original-matrix entry
`A_{ij}` is being scattered into its slot in a frontal, it is
multiplied by `s_i * s_j` on the fly.

- MUMPS: `VAL = A_loc(IZ8); IF (LSCAL) VAL = VAL * ROWSCA(IOLD) * COLSCA(JOLD)`
  in `dfac_dist_arrowheads_omp.F:1021–1024`.
- SSIDS: `node.lcol[k] = rscale * aval[src] * cscale;` in
  `assemble.hxx:64`.

This is not just a convenience — it avoids allocating a second copy
of the matrix values, and it means the scaling appears in the
factorization without requiring any subsequent code in the numerical
kernel to know about it. The pivot threshold operates on the scaled
entries, which is exactly the point: post-scaling, the off-diagonals
are bounded by 1 and the diagonal is close to 1 in magnitude, so a
threshold like `u = 0.01` rejects only entries whose magnitude is
genuinely 100× below the column max.

### Application at solve time

Both solvers pre-scale the RHS before the forward sweep and post-scale
the solution after the backward sweep, using the **same scaling
vector** (not its inverse). The derivation:

  Solve `A x = b` where `A` has been factored as `L D Lᵀ` but under the
  scaled representation `Ã = diag(s) · A · diag(s)`. The factorization
  inside the solver is `L D Lᵀ = Ã`, so
  ```
  Ã · (diag(s)⁻¹ · x) = diag(s) · b
  ```
  Define `y = diag(s)⁻¹ · x` and `c = diag(s) · b`. Solve `Ã y = c`
  via the stored factors, then recover `x = diag(s) · y`. Both the
  pre-scale and the post-scale multiply by `diag(s)` — the same
  vector.

- MUMPS: pre-scale at `dsol_driver.F:4144–4166`
  (`RHS(KDEC+I) *= ROWSCA(I)`), post-scale in the backward solve
  kernels via `SCALING_RHSINTR_BWD` (`dsol_driver.F:6540–6577`).
- SSIDS: pre-scale at `fkeep.F90:253–260`
  (`x2(i,r) = x(invp(i),r) * scaling(i)`), post-scale at
  `fkeep.F90:300–308` (`x(invp(i),r) = x2(i,r) * scaling(i)`).

For iterative refinement, the residual `r = b - A x` is computed
against the **unscaled** `A` and `x` — the scaling is invisible to
the outer refinement loop because the inner solve handles it
transparently. This is important for feral's `solve_sparse_refined`:
the residual computation must use the original CSC, not the scaled
one.

## The Hungarian algorithm kernel

Strategy 5 of citet:duff2001mc64 is a weighted bipartite matching
problem solved by a shortest-augmenting-path / Hungarian algorithm.
The cost matrix is formed from the log-absolute values of the original
matrix, transformed so that the minimum cost is zero and the goal is
to *minimize* total matching cost. The dual variables from the linear
programming formulation of this problem turn out to be exactly what
we need for scaling.

### Input transformation

Given input sparse matrix `A` with entries `a_{ij}`, build the cost
matrix `c_{ij}` over the same sparsity pattern as follows, omitting
explicit zero entries:

1. Drop any entry with `a_{ij} = 0` (implemented as `val(k) == 0.0`
   check in `scaling.f90:640`).
2. Take absolute values and logarithms: `ℓ_{ij} = log |a_{ij}|`.
3. For each column `j`, compute `C_j = max_i ℓ_{ij}` (the log of the
   column's largest absolute entry).
4. Replace each entry by `c_{ij} = C_j − ℓ_{ij}`. This is non-negative
   by construction, equals zero on the column-maximum entries, and
   makes the minimum-cost matching problem equivalent to the
   maximum-product matching (because we've subtracted a constant per
   column, which doesn't change the argmin).

For a symmetric matrix with only the lower triangle stored, the
input preprocessing must first expand to a full symmetric pattern
before building the cost matrix. SPRAL calls `half_to_full` at
`scaling.f90:650` to do this. A feral port must do the same —
matching the lower triangle alone would produce a different matching
than the full symmetric matrix.

### The Hungarian iteration

The core loop in `hungarian_match` (`scaling.f90:938–1171`) is a
textbook shortest-augmenting-path Hungarian algorithm:

1. Initialize with a greedy heuristic matching
   (`hungarian_init_heurisitic`) that matches rows to columns via
   whatever low-cost edges are immediately available.
2. For each unmatched column `j`, build a shortest-path tree rooted
   at `j` through the bipartite graph, using the reduced costs
   `c_{ij} - u_i - v_j ≥ 0` (where `u` and `v` are the current dual
   variables, initially zero).
3. Grow the tree until an unmatched row `i*` is reached. The path
   from `j` to `i*` is an augmenting path.
4. Update the dual variables on all nodes visited by the tree to
   maintain the complementary slackness conditions of the LP dual.
5. Flip the matching along the augmenting path (alternate edges
   swap in/out of the matching).
6. Repeat until every column is matched.

At termination, the dual variables satisfy
`u_i + v_j ≤ c_{ij}` for all edges with equality on the matched
edges. Unwinding the column-max normalization from step (4) of the
input transform gives dual variables that satisfy
`u_i + v_j ≤ log |a_{ij}|` for the *original* matrix, which is the
form citet:duff2001mc64 states.

The shortest-path search uses a binary heap over visited nodes. The
heap operations are the expensive part — each augmenting path costs
`O((m + n) log n)` worst case, and there are up to `n` augmenting
paths, giving `O(n · (m + n) · log n)` total, which for a sparse
matrix with `m ≈ O(n)` nonzeros is about `O(n² log n)`. This is
slower than the factorization itself for large matrices but
acceptable for correctness validation at the sizes feral currently
targets.

### Output: dual variables → scaling

After the matching converges, the SPRAL wrapper computes
`rscaling[i] = u_i` and `cscaling[j] = v_j - C_j` (unwinding the
column-max normalization — see `scaling.f90:169` and the preceding
wrapper logic). The final symmetric scaling is then
`s_i = exp((rscaling[i] + cscaling[i]) / 2)`.

For the symmetric case with `rscaling = u` (row duals) and `cscaling`
(col duals unwound), this is an exp-domain average of the dual
solutions. citet:duff2005symmetric §3 shows this is the scaling that
minimizes a symmetric cost functional related to the matrix
conditioning.

## Degenerate cases and correctness guards

Both reference implementations have accumulated a small but important
set of defensive checks that a Rust port must replicate.

### Structurally singular matrices

If the bipartite graph has no perfect matching (i.e., the matrix is
structurally rank-deficient), the Hungarian algorithm terminates
without a complete matching. In this case:

- MUMPS returns a hard error `INFO(1) = -6` (`dana_aux.F:1732–1740`).
  No fallback; the user must decide whether to proceed.
- SSIDS has an `options%scale_if_singular` flag. If `true`, it
  returns a warning and computes a partial scaling (the unmatched
  rows/columns get scaling `0`, which SSIDS then exponentiates to
  `1`). If `false`, it returns `ERROR_SINGULAR` and writes `0` into
  all scaling entries (`scaling.f90:668–676`).

For feral's port, the right default is **proceed with a warning and
fall back to identity scaling for the unmatched rows/columns**. KKT
matrices from IPOPT are often structurally rank-deficient in edge
cases (rank-deficient constraint Jacobians) and hard-failing on
those would regress feral's current `ForceAccept` behavior. The
warning should be surfaced in `SymbolicFactorization.scaling_warning:
Option<ScalingWarning>` for the bench harness to report.

### Near-zero matches

The column-max normalization assumes every column has at least one
non-zero entry. For a column of all zeros, `C_j = -∞` and the cost
matrix is undefined. Both MUMPS and SSIDS handle this by skipping
such columns in the matching — but feral is unlikely to encounter
them in a well-formed KKT input, and the preprocessing step that
dropped explicit zeros earlier ensures only structural zeros remain.
If a column with only structural zeros slips through (e.g., a
trivially-deletable variable), the port should treat it the same as
an unmatched column and fall back to identity scaling for that index.

### Dual variable overflow

The dual variables can grow large on matrices with a very wide
dynamic range of entry magnitudes. MUMPS and SSIDS both guard
against `exp(u_i) = +∞` by clamping dual variables whose exponent
would overflow `huge(double)` to zero (which exponentiates to 1,
i.e., identity scaling for that row or column). MUMPS's guard is
at `dana_aux.F:1797–1816`; SSIDS relies on its Hungarian kernel to
produce reasonable duals in the first place. A Rust port should
clamp `u_i` and `v_i` to `[-log(f64::MAX), log(f64::MAX)]` before
exponentiation. In practice the bound is `|u_i| ≤ ~709` before the
exponential overflows.

### Exact unit-matched entries

MUMPS has a defensive rewrite at `dana_aux.F:1809–1811`: if any
dual variable is exactly zero after the algorithm, it is rewritten
to `1` so that a zero scaling never reaches the factorization (a
zero scaling would zero out a whole row/column and destroy
symmetry). SSIDS relies on its Hungarian kernel not producing
exact zeros on the matched edges. A Rust port should follow MUMPS's
more defensive approach: check `s_i > 0` before storing; if not,
rewrite to `1.0`.

### Duplicate entries in CSC input

Before taking logs, duplicates must be combined. MUMPS sums their
**absolute values** (`DMUMPS_SUPPRESS_DUPPLI_VAL`). SSIDS expects
unique entries in the input CSC and does not combine duplicates.
feral's existing `CscMatrix::from_triplets` already sums duplicates
with signs, which is the wrong operation for matching purposes
(sum of signed values could be zero while max absolute value is
large). For the MC64 path we need a separate pass that computes
the max absolute value per `(i, j)` pair. Alternatively, and more
simply: the matching can be run on the existing feral CSC (which
has the *signed sum* of duplicates), and the documentation should
note that this differs subtly from MUMPS for matrices with
duplicates that cancel. For matrices produced by `collect_kkt`
(our target workload) this is not an issue because ipopt does not
generate duplicates.

### Heuristic initialization

The SPRAL `hungarian_init_heurisitic` at `scaling.f90:983` is a
cheap greedy first-pass that often matches a large fraction of
rows before the expensive shortest-path search begins. Skipping
this initialization does not affect correctness but doubles or
triples the runtime on typical inputs. A Rust port should
implement the heuristic initialization as a non-trivial
optimization — it is cheap to add and has a large impact on the
total matching time.

## How feral integrates this

### Data model changes

`SymbolicFactorization` in `src/symbolic/mod.rs` gains a new field:

```rust
pub struct SymbolicFactorization {
    // ... existing fields ...

    /// Symmetric global scaling from MC64 matching, in
    /// pivot-order indexing (consistent with `perm` and
    /// `col_counts`). Multiply frontal entries by
    /// `scaling[perm_inv[i]] * scaling[perm_inv[j]]` during
    /// assembly; pre-scale RHS by `scaling[perm_inv[i]]`; post-scale
    /// solution by `scaling[perm_inv[i]]`.
    pub scaling: Vec<f64>,

    /// Diagnostic: did the matching run, and did it succeed?
    pub scaling_info: ScalingInfo,
}

pub enum ScalingInfo {
    /// MC64 matching applied successfully.
    Applied,
    /// Matrix was structurally singular; partial scaling applied,
    /// identity fallback used for unmatched rows.
    PartialSingular { n_unmatched: usize },
    /// Scaling was requested but not applied (e.g., SupernodeParams
    /// opted out).
    NotApplied,
}
```

`SupernodeParams` (or a new `SymbolicParams` that wraps it) gains a
`scaling: ScalingStrategy` field with variants:
- `ScalingStrategy::Mc64Symmetric` — default
- `ScalingStrategy::Identity` — for regression testing and for inputs
  where matching would fail
- `ScalingStrategy::External(Vec<f64>)` — user-supplied pre-computed
  scaling (for Phase 3 POUNCE integration where the scaling may
  come from a higher level)

### Symbolic factorization pipeline

`symbolic_factorize` changes from:

```
AMD → permute → etree → postorder → column counts → supernodes
```

to:

```
MC64 matching on the original matrix → compute scaling vector
→ AMD (on the unscaled structure, since scaling doesn't change
  the sparsity pattern) → permute → etree → postorder
→ column counts → supernodes
→ permute scaling into pivot-order indexing
```

The AMD ordering is computed on the **structural** sparsity pattern,
which the scaling does not affect, so scaling and ordering are
independent and can run in any order. We run scaling first for
clarity and because the scaling vector needs to be permuted into the
final pivot order anyway.

### Numeric factorization

`factorize_multifrontal` applies the scaling when assembling each
frontal matrix. The existing `build_row_indices` / assembly loop in
`src/numeric/factorize.rs` gains a scaling lookup per original-matrix
entry:

```rust
let val = entry_value * scaling[row] * scaling[col];
frontal[local_i * nrow + local_j] = val;
```

This exactly mirrors `dfac_dist_arrowheads_omp.F:1021–1024` and
`assemble.hxx:64`. The scaling is applied once per entry as it is
scattered into the frontal — not in a separate pre-pass.

### Solve

`solve_sparse` and `solve_sparse_refined` in `src/numeric/solve.rs`
pre-scale the RHS before the forward sweep and post-scale the
solution after the backward sweep. The existing `perm` / `perm_inv`
handling is unchanged; the scaling is applied in pivot-order
indexing, adjacent to the permutation.

For `solve_sparse_refined` specifically: the residual computation
(`r = b - A x`) must use the **original unscaled** `A` and `x`.
Feral's existing refinement loop already passes `matrix: &CscMatrix`
which is the unscaled CSC, so this is correct by default — the
scaling is invisible to the outer refinement loop.

### Test plan

- **Hand-computed 3×3 tests.** Small matrices with analytic scaling
  vectors, checked exactly. Example: the matrix
  `[[1e-10, 1, 0], [1, 0, 1e10], [0, 1e10, 1e-5]]` — the scaling
  should produce a matrix whose entries are all within an order of
  magnitude of 1. Compute the expected scaling by hand from the
  Hungarian algorithm output and assert on it.
- **Hand-computed ACOPP30_0000 triage.** Run the existing
  `examples/triage_acopp30.rs` (or a new `triage_mc64.rs`) and
  verify that feral's post-scaling residual is within 2–3 orders
  of magnitude of MUMPS's `5.0e-14`. Pre-MC64 baseline is feral's
  current `3.15e-2`.
- **7-matrix sanity check panel.** Re-run
  `examples/triage_large_cresc132.rs` on the
  `{CHWIRUT1, HAHN1, GAUSS2, CRESC100, MUONSINE, VESUVIO, CRESC132}`
  panel and expect residuals within 2–3 orders of magnitude of
  MUMPS and SSIDS across the panel. If any single matrix is still
  off by more than 5 orders of magnitude, there is a second bug
  that MC64 did not catch and we triage it before declaring 2.2.1
  complete.
- **Full corpus consensus re-run.** Run the full 169k-matrix
  consensus framework with MC64 scaling enabled. Target: the 26
  Definitive feral failures reported after the rmumps deprecation
  (`519f9cc`) should drop substantially — ACOPP30 and DEVGLA2 are
  the dominant failure families and both have residual signatures
  consistent with the scaling hypothesis.
- **Inertia regression check.** The scaling should not change the
  inertia that feral reports (scaling is a congruence transformation
  `A ↦ D A D` which preserves eigenvalue signs). Confirm that the
  existing `tests/sparse_postorder.rs` and `tests/threshold_consistency.rs`
  all still pass with scaling enabled.
- **Round-trip test.** For a small SPD matrix, verify that
  `s_i · A_{ij} · s_j` has the unit-diagonal property that the
  scaling is supposed to produce, up to a small tolerance.

### Risk register (for the implementation plan)

- **R1: Hungarian implementation is slower than expected.** The
  `O(n² log n)` complexity is fine at `n = 5314` (worst case in
  the sanity panel) but may become a bottleneck at `n = 10⁴` or
  beyond. Mitigation: the implementation plan includes a timing
  measurement on CRESC132 and a go/no-go decision point. If the
  matching exceeds 10% of the total factorization time, optimize
  the heap and the shortest-path inner loop before proceeding.

- **R2: Symmetric matrices with non-square matching patterns.**
  Feral's CSC stores only the lower triangle. The matching must be
  run on the full symmetric pattern, which requires a pattern
  expansion (analogous to SPRAL's `half_to_full`). This is not
  hard but must be done carefully so the matching operates on the
  correct graph.

- **R3: MC64 changes the effective pivot threshold interaction.**
  Post-scaling, all off-diagonals are bounded by 1, which means
  the existing BK pivot threshold `alpha = (1+√17)/8 ≈ 0.6404`
  operates on a very different magnitude regime than it did
  pre-scaling. The interaction with `ZeroPivotAction::ForceAccept`
  may produce a different set of force-accepted pivots, which
  could change the inertia on some matrices. Mitigation: after
  implementing, compare the pre- and post-scaling inertia on every
  sanity-check matrix and investigate any change.

- **R4: The ±1 inertia errors from the Phase 2.1.2 sanity check
  may not go away.** The inertia ±1 pattern is a separate bug (the
  deferred 2×2 trace fix), and it is possible that fixing the
  scaling exposes it more clearly rather than hiding it. Mitigation:
  this is expected — after 2.2.1 lands, Phase 2.2.2 re-evaluates
  the trace fix against canonical MUMPS with the new scaling in
  place. The two fixes are deliberately sequenced.

- **R5: Duplicate-entry handling diverges from MUMPS.** See
  §"Duplicate entries in CSC input" above. For our target workload
  this does not matter, but a future regression test that includes
  matrices with duplicates might expose the divergence. Document
  in the research note; decide later whether to implement
  MUMPS-compatible absolute-sum duplicate handling.

## Minimum viable port (checklist)

For the implementation plan that follows this note:

- [ ] `src/scaling/mod.rs` module with public `ScalingStrategy`,
  `ScalingInfo`, and a `compute_mc64_scaling(csc: &CscMatrix) ->
  Result<Vec<f64>, FeralError>` function.
- [ ] `src/scaling/hungarian.rs` with the
  `hungarian_match(cost_csc: &CostGraph) -> (Vec<usize>, Vec<f64>, Vec<f64>)`
  kernel (matching permutation, row duals, column duals).
- [ ] `src/scaling/mc64.rs` with the wrapper that does input
  preprocessing (pattern expansion, log transform, column-max
  normalization), calls the Hungarian kernel, unwinds the
  normalization, and returns the symmetric scaling vector.
- [ ] `SymbolicFactorization::scaling` field and wiring in
  `symbolic_factorize`.
- [ ] `factorize_multifrontal` scaling application during assembly.
- [ ] `solve_sparse` pre/post scaling.
- [ ] `solve_sparse_refined` uses unscaled `matrix` for residual,
  which it already does by default.
- [ ] Test suite: hand-computed 3×3, ACOPP30_0000 regression,
  7-matrix sanity panel regression, full corpus consensus re-run.
- [ ] Documentation: update `README.md` Status section once the
  sanity-check panel residuals come within the target, and update
  `CHANGELOG.md` under `[Unreleased]` to note that the n>500
  correctness gap is closed.

## Estimated effort (from the Phase 2 plan)

Research note (this document): **2 hours**, done.

Implementation plan (next document, `dev/plans/mc64-scaling.md`):
**1 hour**.

Implementation:
- Hungarian kernel with heuristic init: **6–10 hours**. This is the
  hardest piece. The SPRAL implementation is about 230 lines of
  Fortran; the Rust port will be longer because of borrow-checker
  considerations and because we want explicit types on the heap
  operations. Budget 8 hours plus 4 hours of debugging.
- MC64 wrapper (input transform, call kernel, output unwind):
  **2 hours**. Small and straightforward.
- Integration into `symbolic_factorize`, `factorize_multifrontal`,
  and `solve_sparse`: **2–4 hours**. Mostly plumbing.
- Test suite: **3–5 hours**. Hand-computed tests take the most
  time; the existing integration tests auto-regress.
- Validation (sanity panel, consensus re-run, investigation of
  anything surprising): **2–4 hours**.

**Total realistic: 16–25 hours**, split across 3–4 focused sessions.

## Open questions carried into the implementation plan

1. **Scale factor storage: pivot-order or user-order?** SSIDS stores
   the scaling in pivot-order indexing (`fkeep%scaling`), while MUMPS
   stores it in user-order indexing (`ROWSCA` / `COLSCA`). Each has
   trade-offs: pivot-order is faster during assembly (direct lookup)
   but requires an extra permute at the end of `symbolic_factorize`;
   user-order is simpler conceptually but requires an indirection
   through `perm` during assembly. Recommend pivot-order for
   consistency with SPRAL and because feral's assembly already has
   all the permutation bookkeeping in pivot-order.

2. **Should the matching run on the fully-expanded symmetric pattern
   or on just the lower triangle?** Running on just the lower
   triangle produces a different (asymmetric) matching that would
   then need a different symmetrization. Running on the full pattern
   matches what MUMPS and SSIDS do. Recommend full pattern; the
   extra memory for the pattern expansion is `2x nnz` which is
   negligible.

3. **Do we need MC64 strategy 6** (maximum product with scaling and
   also explicit row permutation applied to the matrix)? MUMPS uses
   strategy 5 for `SYM=2` and does not apply the matching permutation
   to the matrix; the matching is consumed only by the 2×2 pair
   detection. For feral's minimal port, strategy 5 is sufficient.
   Strategy 6 would be needed only if we also implement the 2×2
   pair detection, which is Phase 2.3 work, not Phase 2.2.1.

4. **Fall back to Ruiz equilibration if Hungarian fails?** Current
   recommendation: no. If Hungarian fails, fall back to identity
   scaling and warn — Ruiz is a different algorithm and may or may
   not help. Adding a second scaling path complicates the code
   without clear benefit. The Phase 2.2.1 exit criterion is "MC64
   lands and residuals improve"; alternative scalings are a Phase
   2.5 performance optimization at most.

## References

bibliography:../references.bib
bibliographystyle:plain
