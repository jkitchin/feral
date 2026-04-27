# F3.0 — Schur Complement Extraction: Research Note

**Date:** 2026-04-27
**Phase:** F3.0 of `dev/plans/kkt-feature-gaps.md`
**Goal:** Settle the API, the ordering-constraint mechanism, and
the per-front numeric stopping rule before F3.1 code lands.

## Why we want this

Two distinct ripopt-side use cases (per
`dev/plans/ripopt-integration.md` §F3):

1. **Structured KKT elimination.** For NLPs whose KKT block has a
   block-diagonal slack tail (bounds-only problems, two-stage
   stochastic NLP), eliminating the slack block first and solving
   the smaller dense Schur in `(primal, dual)` space is faster than
   factoring the full augmented system every Newton step.

2. **Sensitivity analysis.** sIPOPT-style parametric sensitivities
   reduce to a Schur system over the parameter block. Solving
   against the reduced Schur is much smaller than the full system;
   Pirnay-López 2012 documents the speedup for large NLPs.

Neither use case is on feral's critical path — they unlock new
ripopt features rather than improving existing ones. F3 is the
biggest of the three F-phases (4-6 sessions per the roadmap) and
should ship after F1 and F2 land.

## Reference solver behavior — what MUMPS does

A focused MUMPS-source review (see `dev/journal/2026-04-27-09.org`
21:55 entry and below) yields six anchor facts that this note
cites. File:line references point into the MUMPS 5.8.2 tree at
`../ripopt/ref/mumps/src/`.

### Public API surface

`SIZE_SCHUR` (count) and `LISTVAR_SCHUR(:)` (1-based original-
index list) are user-supplied. `SCHUR(:)` is the host-side output
buffer the user allocates. `ICNTL(19)` selects the output mode:

- `1`: centralized dense, full square symmetric (both triangles).
- `2`: distributed 2D block-cyclic, lower triangle only.
- `3`: distributed 2D block-cyclic, full square.

(`dini_defaults.F:471-485`, `dmumps_driver.F:1462-1479`.)

For a shared-memory clean-room port, `ICNTL(19)=1` is the only
mode that matters; `2`/`3` are MPI/BLACS-specific.

### Validation rules

`dana_driver.F:3453-3499`:

- `SIZE_SCHUR=0` ⇒ silently disabled.
- `SIZE_SCHUR < 0` or `≥ N` ⇒ INFO(1)=-49.
- `LISTVAR_SCHUR` not associated or too small ⇒ INFO(1)=-22.
- Schur is incompatible with parallel analysis, max-transversal
  (`KEEP(23)`), and analysis-time scaling (`KEEP(52)=-2`).

For feral, validation is much simpler — no MPI, no analysis-time
scaling controls.

### Ordering constraint (the key insight)

MUMPS forces Schur variables to the **end of the postorder** by
the AMD/QAMD driver itself, not as a post-processing step. The
mechanism (`ana_orderings.F:2909-2942` HALO-SCHUR; `:9187-9220`
amalgamation):

1. All Schur variables get an internal `PERM(i) = N + 1` so the
   ordering algorithm leaves them for last.
2. They are amalgamated into a **single supervariable** rooted at
   `LISTVAR_SCHUR(1)` via `PE(LISTVAR_SCHUR(I)) = -LISTVAR_SCHUR(1)`.
3. The elimination tree therefore has a single artificial root
   whose fully-summed block is exactly the Schur set.

The output user-visible row/col order of the returned Schur
matches `LISTVAR_SCHUR` — preserved from the input list, *not*
from any internal permutation.

### Numeric stopping rule

Per-front, MUMPS computes `NVSCHUR = #{i in front : PERM(i) ≥
N - SIZE_SCHUR + 1}` (`dfac_front_aux.F:2562-2582`). The
elimination loop bounds `NPIV ≤ NASS - NVSCHUR`
(`dfac_front_LDLT_type1.F:193-205, 430-459`). The trailing
NVSCHUR rows/columns of the front accumulate descendant updates
but are not eliminated; they are precisely the partial Schur.

There is **no** "is this the root?" branch in the inner factor
loop. The mechanism is purely the bound on the eliminable
region. This is much cleaner than gating on a node-type flag.

### Output layout

`DMUMPS_GATHER_SCHUR` (`dfac_driver.F:4625-4757`) copies the
Schur block out of the workspace `S` into `id%SCHUR` as a
contiguous `SIZE_SCHUR × SIZE_SCHUR` column-major Fortran
matrix. For symmetric (`SYM=2`) the centralized dense mode
returns the **full square** (both triangles), via a prior
`DMUMPS_SYMMETRIZE` call (`dtype3_root.F:1258-1294`).

### Pivoting in symmetric case

The descendants of the Schur root use the standard MUMPS
threshold-pivoting LDLᵀ scheme (1×1 / 2×2 pivots, threshold
`CNTL(1)`). Schur presence does not change pivoting strategy
inside descendant fronts.

The Schur block itself is **not pivoted, scaled, or compressed**
— it is the partially-formed Schur returned to the user as-is.
Numerically:
```
S = A_SS - A_SF · (L_F D_F L_F^T)^{-1} · A_FS
```
where F is the eliminated set and S is the Schur set.

One subtlety: variables that get *delayed* during descendant
elimination must arrive at the root and be eliminated **before**
the Schur. `DMUMPS_PARPIVT1_SET_NVSCHUR_MAX`
(`dfac_asm.F:1002-1045`) recomputes the Schur bound at the root
to keep delayed pivots out of the Schur tail.

## Decisions

### D1. Output: dense column-major, full square, single mode

feral's first cut implements only **MUMPS `ICNTL(19)=1`-equivalent
behavior**: a single dense `n_schur × n_schur` column-major buffer
with **full symmetric entries** populated (both triangles). This
matches MUMPS's centralized symmetric output and is the only mode
ripopt's structured-KKT and sensitivity-analysis use cases need.

Distributed/sparse output modes are explicitly out of scope —
feral has no MPI runtime and no caller asks for sparse Schur.

```rust
impl Solver {
    pub fn factor_with_schur(
        &mut self,
        matrix: &CscMatrix,
        schur_indices: &[usize],
    ) -> FactorStatus;

    /// Returns Some(&[f64]) of length n_schur*n_schur, column-major,
    /// full symmetric (both triangles) when factor_with_schur completed.
    /// None if no Schur factor is stored.
    pub fn schur_complement(&self) -> Option<&[f64]>;

    /// n_schur for the most recent factor_with_schur, or 0.
    pub fn schur_size(&self) -> usize;
}
```

### D2. Variable-selection semantics

`schur_indices: &[usize]` is the user's **original-index list**
(0-based since this is Rust), arbitrary subset of `0..n`.
Duplicates and out-of-range indices return `FactorStatus::Failed`
with an `InvalidInput` cause (or a new `SchurInvalid` variant).

The output Schur matrix's rows/columns are in the same order as
`schur_indices` — preserved from the user's input, not from
feral's internal permutation. (Matches MUMPS's contract.)

### D3. Ordering constraint mechanism

Two candidate mechanisms, each shipped solvers use one:

a) **Bias the AMD score** so Schur variables are ordered last
   (the MUMPS HALO-SCHUR + amalgamation route, `ana_orderings.F`).
b) **Run AMD on the non-Schur subgraph, then append Schur tail.**

Pick (b). Justification:

- (a) requires modifying `feral-amd` (the external crate
  implementing the default ordering). We don't own that crate's
  API surface; bolting a "tail-pinned variable list" parameter
  onto its driver is invasive.
- (b) is composable and works for any ordering method: AMD,
  AMF, MetisND, ScotchND. Just run the chosen ordering on
  `pattern.restrict_to(non_schur_vars)`, append `schur_indices`,
  and feed the composed permutation through the rest of the
  pipeline.
- The amalgamation step from MUMPS (`PE[schur[i]] =
  -schur[0]` collapsing all Schur vars into one supervariable)
  is implicit in (b): in the elimination tree built from the
  composed permutation, the Schur variables form a contiguous
  postorder block and the column-counts pass naturally amalgamates
  them into a single supernode at the tree root.

Cost: an extra graph-restriction pass during analysis, O(nnz).

### D4. Numeric stopping rule

Adopt MUMPS's mechanism in (`dfac_front_aux.F` /
`dfac_front_LDLT_type1.F`): in `numeric/factorize.rs`, when
processing a frontal matrix, compute `nvschur` = count of
front rows whose post-permutation index lies in the Schur tail
`[n - n_schur, n)`. Bound the elimination loop's `nelim` so
`nelim ≤ nass - nvschur`.

For the **root supernode** specifically, `nvschur = nass` and
`nelim = 0`: the entire fully-summed block is the Schur. For
non-root descendant fronts, `nvschur = 0` (the Schur tail is the
single root supernode) and the existing factor code runs
unchanged.

This means: for the postorder layout produced by D3's mechanism,
the *only* front that has `nvschur > 0` is the root. The check
is one line in the front's setup code; the inner factor loop is
unchanged.

### D5. Schur assembly and symmetrization

The root front, having received all descendant updates, holds
the partial Schur in its lower triangle (since feral's frontal
storage is lower-only following the input `CscMatrix` convention).
Two output options:

a) Lower-only buffer of size `n_schur·(n_schur+1)/2`.
b) Full square buffer of size `n_schur²` with both triangles
   populated.

Pick (b). Reasons:
- ripopt's downstream consumers (dense factor, eigenvalue
  routine) want the full square so they can call any LAPACK
  routine without packed-format adapters.
- The `n_schur²` storage is the same as MUMPS centralized.
- Memory cost is negligible — n_schur is bounded by the
  caller's slack-block size, typically a few hundred.

Implementation: after the root supernode's update phase, copy
its lower triangle into the output buffer at column-major
positions, then mirror to the upper triangle.

### D6. Pivoting in the Schur tail

Match MUMPS exactly: **no pivoting on the Schur block**. Delayed
pivots from descendants that arrive at the root supernode must
be eliminated **before** the Schur, not consumed into it.

For feral's current Bunch-Kaufman path, this means: at the root
supernode's pivot loop, additionally bound the candidate-pivot
set so columns in the Schur tail are excluded from selection.
Pivots that are delayed up to the root from earlier supernodes
*are* eliminated (they reduce nelim of the root by the delay
count); the Schur tail width remains exactly `n_schur`.

The implementation surface here lives in
`numeric/bunch_kaufman/factor.rs` (or wherever the per-front
pivot loop is) — add an optional `schur_tail_size: usize`
parameter that the loop respects when scoring candidates.

### D7. Cross-validation oracle (F3.3)

For correctness verification on real-world matrices, extend the
existing MUMPS oracle with a Schur driver:

`external_benchmarks/mumps_schur_oracle/run_mumps_schur.py`:
For each KKT sidecar matrix, pick a Schur block (e.g., the
trailing `n/4` indices, or the indices marked `dual` in the
sidecar metadata), call MUMPS with `ICNTL(19)=1` and store the
returned dense Schur in verdict.json (compressed, as a base64-
encoded f64 buffer or a separate `.schur.bin` sidecar).

feral's `tests/schur_corpus_oracle.rs` then loads the same
matrix, factors with `factor_with_schur`, and compares the
output Schur entry-wise against the MUMPS reference. Acceptance:
max relative error ≤ 1e-10 on N ≥ 100 corpus matrices.

## Acceptance gates (from kkt-feature-gaps.md)

- F3.2: hand-computed Schur on a 10×10 KKT example matches
  feral's output to ≤ 100·ε relative error.
- F3.3: cross-validation N ≥ 100 corpus matrices, max relative
  entry-wise error ≤ 1e-10.
- F3.x (all phases): no regression on the Phase 2.8.1 bench
  partition. The new `factor_with_schur` path is opt-in; the
  default `factor` path is unchanged.

## Phase plan

**F3.1 — Ordering hook (estimated 1-2 sessions).** Modify
`symbolic/mod.rs` to accept an optional Schur-tail variable
list. The `OrderingMethod` dispatch runs on the non-Schur
subgraph; the Schur indices are appended in `LISTVAR_SCHUR`
order. The elimination tree, supernode detection, and column-
counts run unchanged on the composed permutation. Tests on
small KKT examples with hand-computed expected ordering.

**F3.2 — Numeric hook (estimated 2-3 sessions).** Modify
`numeric/factorize.rs` per D4: compute `nvschur` per front,
bound `nelim`, route Schur block to the public output buffer.
Bunch-Kaufman pivot loop excludes Schur columns from selection
(D6). Tests: D6's hand-computed 10×10 KKT.

**F3.3 — Cross-validation (estimated 1 session).** Build the
MUMPS Schur oracle (D7) and the feral comparison harness.

**F3.4 (optional) — `solve_against_schur(s_matrix, rhs)`
convenience.** Defer until a caller (sIPOPT-style sensitivity
in ripopt) actually asks.

## Open questions (close before F3.1)

1. **Should `factor_with_schur` accept an `OrderingMethod`
   override, or use the global default?** Answer: use the
   global default. The Schur-tail mechanism (D3) is composable
   with any ordering, so `factor_with_schur(matrix, schur_idx)`
   uses whatever `pick_default_method` chose. Callers wanting
   override use a new `factor_with_schur_method(matrix, idx,
   method)` form analogous to `symbolic_factorize_with_method`.

2. **What if `schur_indices` is empty?** Treat as
   `factor(matrix)` — return the standard FactorStatus and
   `schur_size() == 0`, `schur_complement() == None`.

3. **What if `schur_indices.len() == n`?** Empty elimination
   set: the Schur "block" is the whole matrix unchanged. We
   could either error (`InvalidInput`) or return the matrix as
   the Schur. Decision: error out — the user almost certainly
   meant a partial Schur, and `n_schur == n` indicates a logic
   bug upstream.

4. **Iterative refinement against the Schur output?** Not in
   F3. Refinement is per-system and the Schur is a *new* matrix;
   refinement would happen in the user's solver of the dense
   Schur, not feral.

5. **Inertia of the partial factor?** The inertia returned by
   `factor_with_schur` is the inertia of the **eliminated
   block** (the L D L^T factor of A_FF), not of A as a whole.
   Document this prominently — ripopt's inertia-correction
   loop must be aware that the Schur block contributes to
   total inertia separately.

## References

### MUMPS source (clean-room reference only)
- `dini_defaults.F:471-485` — ICNTL(19) semantics.
- `dana_driver.F:3453-3499, 3651-3671` — validation, ordering
  compatibility check.
- `dana_driver.F:1163, 1179` — `LISTVAR_SCHUR` passed into
  `DMUMPS_ANA_F`.
- `ana_orderings.F:2909-2942, 8301-8408, 9187-9220` —
  HALO-SCHUR, `MUMPS_SYMQAMD` Schur contract, amalgamation.
- `dfac_front_aux.F:2562-2582` — `DMUMPS_GET_SIZE_SCHUR_IN_FRONT`.
- `dfac_front_aux.F:115-125` — BLAS-3 update bounded by Schur.
- `dfac_front_LDLT_type1.F:193-205, 430-459` — front-level
  Schur stop in LDLᵀ.
- `dfac_asm.F:1002-1045` — `DMUMPS_PARPIVT1_SET_NVSCHUR_MAX`.
- `dfac_driver.F:4625-4838` — `DMUMPS_GATHER_SCHUR`.
- `dtype3_root.F:463-1294` — root-node distributed factorization
  with Schur (symmetrize, do-not-eliminate path).

### Papers
- Duff & Reid 1983: "The multifrontal solution of indefinite
  sparse symmetric linear equations." ACM TOMS 9(3) — the
  multifrontal method this builds on.
- Pirnay-López 2012: "Optimal sensitivity based on IPOPT"
  Math. Prog. Comput. 4(4), 307-331 — sIPOPT use case (D7
  oracle alternative).
- Gould, Scott, Hu 2007: "A numerical evaluation of sparse
  direct solvers for the solution of large sparse symmetric
  linear systems." ACM TOMS 33(2) — Schur API survey across
  HSL_MA57, MUMPS, SuperLU.

### feral-side companion docs
- `dev/plans/kkt-feature-gaps.md` §F3 — phase plan.
- `dev/plans/ripopt-integration.md` §F3 — adoption sketch.
- `dev/research/condition-estimate.md` — F2.0 (sibling note).
