# Issue #57 — Row-major working buffer for the multi-RHS sparse solve

**Date:** 2026-05-30
**Issue:** #57 — `solve_sparse_core_many_into: column-major RHS layout
prevents inner-loop vectorization`
**Builds on:** `dev/research/multi-rhs.md` (F1.0, the original multi-RHS
API + kernel-shape decision)
**Goal:** Decide and justify the internal working-buffer layout change
before editing `solve_sparse_core_many_into`, and bound the
correctness/performance risk.

## Symptom

The supernodal multi-RHS forward/back-substitution in
`src/numeric/solve.rs::solve_sparse_core_many_into` (lines ~448–624)
lays out its per-supernode working buffer `w` (`ws.w`, length
`max_nrow * nrhs`) **column-major**: column `c` of the supernode-local
rows occupies `w[c*nrow .. (c+1)*nrow]`, addressed as `w[c*nrow + i]`.

The innermost per-RHS update is therefore a stride-`nrow` access. The
forward kernel (~490–498):

```rust
for j in 0..nelim {
    for i in (j + 1)..nrow {
        let l_ij = ff.l[j * nrow + i];
        for c in 0..nrhs {
            let off = c * nrow;            // stride = nrow (often 100s)
            w[off + i] -= l_ij * w[off + j];
        }
    }
}
```

LLVM cannot auto-vectorize the inner `c`-loop (dependent strided
loads), so it emits scalar code. The single-RHS path
`solve_sparse_core_into` (a *separate* function, lines 144–294) is
contiguous and fast; multi-RHS therefore costs ~the same per RHS as
looping single-RHS. Measured per-RHS ratios (issue #57) hover at
0.71–1.07× — no amortization — against ~5–8× for dense GETRS.

The optimistic comment at lines 488–489 ("compiler auto-vectorizes for
small constant or runtime `nrhs`") was the F1.0 design intent (D3 in
`multi-rhs.md`, lines 116–138) and holds for small `nrhs ≤ 8`; it fails
for the pounce workload, which uses `nrhs = n` (hundreds–thousands).

## Decision: flip the internal `w` buffer to row-major

Store element `(i, c)` at `w[i*nrhs + c]` instead of `w[c*nrow + i]`.
The inner `for c in 0..nrhs` loop then runs over **contiguous**
addresses (stride 1) and auto-vectorizes. Allocation size is unchanged
(`max_nrow * nrhs`); only the indexing convention changes.

### What stays column-major (the public contract — do NOT change)

The F1.0 decision D1 (`multi-rhs.md` lines 73–88) fixed the
caller-visible RHS/solution layout as **column-major `n × nrhs`**, to
match MUMPS (`ICNTL(20)=0`) and SSIDS (`ldx`). That contract is part of
the public API doc (`solve.rs:345-346`) and the C ABI (`feral_solve` in
`src/capi.rs`), and the Python `Solver.solve(2-D)` path
(`python/src/lib.rs`) packs/unpacks against it. It MUST be preserved:

- `rhs`, `x_out`, the workspace `y` and `scaled_rhs` buffers all stay
  column-major (`column c at [c*n .. (c+1)*n]`).
- The permute/unpermute loops (`solve.rs` ~460–466, ~616–623) and the
  MC64 pre/post-scaling in the caller `solve_sparse_many_into`
  (~413–437) are untouched.

The column-major ↔ row-major transpose is absorbed **entirely** by the
per-supernode gather (load `y`-column-major → `w`-row-major) and
scatter (store `w`-row-major → `y`-column-major) loops, which already
exist as the mediation layer. No new transpose pass, no signature
change, no `SolveManyWorkspace` field/size change (only the `w` field
doc comment).

### Sites that change (all inside `solve_sparse_core_many_into`)

1. Phase-1 forward gather (~479–485): `w_col` slice → `w[i*nrhs + c]`.
2. Phase-1 forward kernel (~490–498): `w[c*nrow+i]/[+j]` →
   `w[i*nrhs+c]/[j*nrhs+c]`, `for c` innermost & contiguous.
3. Phase-1 scatter (~501–507): slice → `w[i*nrhs + c]`.
4. Phase-2 D-block gather (~521–527): slice → `w[i*nrhs + c]`.
5. Phase-2 D-solve (~529–565): the one non-mechanical spot. Loses the
   contiguous `w_col` slice; 1×1 and 2×2 (incl. force-accepted-zero)
   pivot arithmetic stays per-column and bit-for-bit identical, only
   element addresses change to `w[k*nrhs + c]`.
6. Phase-2 scatter (~567–573): slice → `w[i*nrhs + c]`.
7. Phase-3 back gather (~586–592): slice → `w[i*nrhs + c]`.
8. Phase-3 back kernel (~596–605): dot-product reduction; keep `i`
   iteration order (→ identical float accumulation order), only
   `w[c*nrow+i]` → `w[i*nrhs+c]`.
9. Phase-3 scatter (~607–613): slice → `w[i*nrhs + c]`.

## Risk analysis

**Capability / correctness: low.** This is a memory-layout + loop-nest
transform, not an algorithm change. The forward kernel and D-solve do
no cross-`c` reduction, so per-element results are independent of `c`
layout → bit-identical. The back-sub `sum` reduction iterates `i` in
the same order under either layout (layout changes *where* `w[i]` lives,
not the iteration order) → accumulation order preserved → bit-identical.
Permutation, scaling, 1×1/2×2 pivots, force-accepted zeros, and
iterative refinement are all outside or unchanged by the edited loops.

**Single-RHS / refinement hot path: zero impact.** `solve_sparse`,
`solve_sparse_into_ws`, `solve_sparse_refined` and
`Solver::solve_many_refined` all route through the *separate*
`solve_sparse_core_into`, which is physically untouched. The
IPM-critical refinement path does not change shape. (Note: `nrhs == 1`
through the many path degenerates to `w[i*1+0] = w[i]` — contiguous,
identical to today.)

**Public API / downstream: zero impact.** No signature change; the
column-major contract is preserved, so pounce's `kkt_solve_many`, the C
ABI, and the Python bindings need no changes (the issue confirms the
pounce plumbing is already in place).

## Oracle (satisfies the CLAUDE.md "no co-authored oracle" rule)

The oracle is the **pre-existing, independent single-RHS path**
`solve_sparse`: `solve_sparse_many(k-stacked B)` must equal `k`
independent `solve_sparse` calls, per entry. This is already enforced
by `tests/multi_rhs.rs::solve_many_matches_k_independent_solves`
(nrhs ≤ 3 on 5×5). That test predates this change and is not modified,
so the oracle is external. We add a larger case (`n ≥ 120`, multiple
supernodes, `nrhs ≥ 16`) — large enough to exercise vectorized codegen
and catch a row-major indexing slip at scale — asserting equality to a
tight tolerance (`1e-12`, not loosened).

## Acceptance criteria

1. `cargo test --test multi_rhs` green before *and* after (existing +
   new larger case), tight tolerance.
2. `cargo test --lib` ≥ 317 pass; clippy `-D warnings` clean; fmt clean.
3. New `src/bin/bench_multirhs.rs`: at `nrhs = 64` and `256` the
   multi-RHS per-RHS time is a clear fraction of single-RHS per-RHS
   (ratio well below 1.0). The F1.2 target in `multi-rhs.md` (lines
   245–249) was per-column ≤ 0.75× of single-RHS; report measured
   numbers and state plainly if the target is not met.
4. Results bit-identical to the single-RHS oracle (expected); if not,
   report max abs/rel diff and justify.

## Follow-up (out of scope here)

Fix #2 from the issue — BLAS-3 `dtrsm`/`dgemm` per supernode panel
(`multi-rhs.md` D3 `nrhs > 8` regime; `src/bin/blas3_prototype.rs` is
early groundwork) — targets the 5–10× regime and is deferred until this
layout fix is measured.
