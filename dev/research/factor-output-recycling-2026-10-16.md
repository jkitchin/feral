# Recycling per-front factor outputs across warm refactors (issue #153)

## Evidence that motivates it

From `dev/journal/2026-10-16-01.org`, on an x86_64 KVM guest with glibc 2.39:

- **Allocation count.** A warm refactor of the hanging-chain KKT
  (11,875 supernodes) makes 93,138 allocations (16.2 MB), or
  7.84 per supernode. dhat attributes almost all of them to per-front
  *outputs* that are rebuilt on every refactor even though the pattern
  is reused:
  - `dense/factor.rs`, eager path: `perm` (1835), `l` (1905), `d_diag`
    (1906), `perm_inv` (1982), `d_subdiag` (1995);
  - `dense/factor.rs`, blocked path: 2237, 2447, 2448, 2510, 2526;
  - `numeric/factorize.rs`: `row_indices` = `static_rows.to_vec()`
    (2535), and `contrib_row_indices` (2756).
- **Allocator sensitivity.** Swapping glibc for mimalloc speeds up the
  warm refactor on 71/77 corpus matrices: geomean 0.849, 0.70-0.78 in
  the 0.1-10 ms buckets, 0.54-0.56 on small-front-heavy matrices. On
  large-front matrices the effect is neutral. That is the ceiling a
  code-side fix should approach.

## Prior work, and why this is not a re-proposal

- **Lever D.1 (2026-04-19, `sparse-tail-perf-2026-04-19.md` §9)** pooled
  the *scratch* (row_map, frontal, subdiag, d_panel, build_*) and took
  allocations from about 19 to about 8 per supernode. Its items 3 and 4
  (output buffers and row indices) were never done. The leftover 7.8
  per supernode is exactly that remainder.
- **Phase C multi-slot contribution pool (2026-05-13, rejected).**
  `Vec<Vec<f64>>` push/pop bookkeeping cost more than it saved, measured
  on macOS. That pool recycled *contribution blocks* through a LIFO
  stack. This proposal recycles *outputs* by supernode index, with no
  stack: one `Option::take` per front.
- **#200 bump-allocator test.** Shown on 2026-10-16 to be an invalid
  instrument (no memory reuse makes it slower than glibc). Its "ruled
  out" conclusion does not apply.

## Mechanism

1. `Solver::factor` already drops `last_factors` on every error path and
   replaces it on success. Right before the numeric driver runs, move
   `last_factors` (same pattern, so the same supernode numbering) into
   `FactorWorkspace` as per-supernode buffer sets.
2. Before calling the kernel for supernode `i`, the driver installs
   set `i` into `FactorScratch` output slots.
3. The kernels `take` from those slots and re-initialize with
   `clear()` plus `resize`/`extend`. The contents are therefore
   identical to `vec![0.0; n]`, `(0..n).collect()` and `to_vec()`, so
   the result is byte-exact by construction.
4. Any slot left unused is dropped at the end of the call.

## Properties

- **Byte-exact**, because every reused buffer is fully re-initialized
  exactly as the fresh allocation was.
- **Peak memory goes down.** Today, factor N-1 stays alive while factor
  N is built. With recycling, N-1's buffers become N's buffers.
- **Idle memory is unchanged.** Only one factor stays resident between
  calls.
- **Capacity high-water.** Buffers keep their largest capacity per node.
  This only matters if delays inflate a node once and never again.
  Accepted.
