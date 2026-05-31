# Lever 2.1 — parallel-across-RHS multi-RHS solve

**STATUS: DEFERRED (2026-05-31).** Design complete; not implemented. Deferred in
favour of Lever 2.2 (symbolic speedups), which targets the actual measured tail.
Rationale below.

Source: `dev/research/perf-review-2026-05-31.md` §2 Tier-2 #1.

## Idea

The multi-RHS solve `solve_sparse_core_many_into` (`src/numeric/solve.rs:495`)
processes all `nrhs` columns through one pass of the supernodal forward / D /
backward substitution. RHS columns are mathematically independent, so the column
set could be split into groups and solved on separate rayon workers.

The internal working buffers `y` (`n*nrhs`) and `w` (`max_nrow*nrhs`) are
**row-major** (issue #57), so a contiguous column *range* is not contiguous —
but a worker could own a disjoint column range and run the full node sequence
for just those columns with its own `w`/`acc` scratch.

## Why it is deferred (not implemented)

1. **Bit-exactness is entangled with the dispatch threshold.** The solve picks
   its kernel by *total* nrhs: `use_blas3 = nrhs >= BLAS3_NRHS_THRESHOLD (32)`
   (`solve.rs:509`). Splitting a 64-column solve into, say, 4 groups of 16 makes
   every group fall **below** the threshold and flip to the rank-1 path.
   Critically, the existing tests document that the BLAS-3 back-substitution is
   **not** bit-identical to the rank-1 / single-RHS path — only "close" at
   ~1e-15 (`tests/multi_rhs.rs:205-211`: "back-sub reorders the panel vs trailing
   reduction, so a ~1e-15 drift is expected"). So a naive column split changes
   the result vs the serial solve. Staying bit-exact requires threading a
   *forced-path* selector **and** a column-range through all six solve kernels
   (`fwd_rank1`/`fwd_blas3`/`back_rank1`/`back_blas3`/`dsolve_node` + the
   gather/scatter) — real surgery on the bit-exact-tested numeric core.

2. **Narrow payoff.** The solve is already faster than MUMPS (the perf-review
   pain is the *factor* tail, not the solve). Parallel-across-RHS only helps
   **large-nrhs** solves; the primary consumer — the IPM predictor-corrector —
   uses **nrhs = 2**, which gets nothing. The perf-review itself rates this
   "Tier-2, lower leverage for single-RHS, real for many-RHS / heavy
   refinement."

3. **Better alternative available now.** Lever 2.2 (symbolic-phase speedups)
   targets the population the 154k-matrix p90-vs-MUMPS tail actually lives in
   (symbolic-bound small matrices; KIRBY2 8.76× is analysis time), is
   load-robust (removes work rather than adding threads), and does not touch the
   numeric bit-exact path.

## If revisited

Implement as: an outer `solve_sparse_many_into` that, when `nrhs` is large and a
worker pool is available, partitions columns into `g` groups, **each group sized
to keep its own dispatch decision identical to the serial whole** (i.e. either
keep all groups ≥ threshold, or force every group onto the same kernel the
serial call would use), runs `solve_sparse_core_many_into` per group on disjoint
column slices with per-group `w`/`acc` scratch, via `rayon::scope`. Gate on a
min-nrhs and min-work threshold. Verify against `tests/multi_rhs.rs` parity plus
a new "parallel == serial bit-exact" test that pins the forced path. A/B via
`bench_multirhs`.
