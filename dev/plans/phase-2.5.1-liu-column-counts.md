# Phase 2.5.1 — Plan: Liu's row-subtree column counts

**Date:** 2026-04-20
**Research note:** `dev/research/phase-2.5.1-liu-column-counts.md`

## Goal

Replace the current `O(n²)` `column_counts` in
`src/symbolic/column_counts.rs` with the Gilbert–Ng–Peyton
`O(nnz(A) + n·α(n))` algorithm (Liu's row-subtree flavor via
DSU with path compression).

## Exit criteria

From the research note §Acceptance:

1. All 118 lib tests pass.
2. Corpus-sweep test reports **zero** per-matrix column-count
   differences across 154588 KKT matrices vs the old
   implementation.
3. Symbolic throughput on the large-n subset (n > 500)
   improves ≥ 3× per matrix.
4. (soft) Overall sparse factor/MUMPS p90 improves (any
   measurable amount).

## Step 0 — Gate check

**Before implementing,** measure symbolic-vs-numeric time
split on the KKT corpus. If `symbolic_factorize` is under 5%
of total factor time on every size class, abort Phase 2.5.1
and move to **Phase 2.5.2** (Rayon on the assembly tree) or
**Phase 2.5.3** (scratch-buffer preallocation in
`solve_sparse`).

Mechanism: add `sym_factor_us` to `MatrixTiming` in
`src/bin/bench.rs`, wrap the existing `symbolic_factorize`
call in `Instant::now()`, emit an extra ratio line.

Commit as `phase-2.5.1: instrument symbolic-vs-numeric split`.

## Step 1 — Postorder + first descendant

Add to `src/ordering/elimination_tree.rs`:

- `pub fn postorder(&self) -> Vec<usize>` — returns nodes in
  postorder of the forest rooted at the tree roots.
- `pub fn first_descendants(&self, post: &[usize]) -> Vec<usize>`
  — returns `first[i]` = postorder index of the first (smallest
  postorder-number) descendant of i.

Tests:
- Path graph (chain): postorder is 0..n−1; first[i] = i.
- Star (1 root, n−1 leaves): postorder leaves first, root last;
  first[root] = 0; first[leaf] = its own postorder index.
- Existing etree unit tests continue to pass.

Commit as `phase-2.5.1: etree postorder + first_descendants`.

## Step 2 — GNP column counts implementation

Add `pub fn column_counts_gnp(pattern: &CscPattern, etree:
&EliminationTree) -> Vec<usize>` in
`src/symbolic/column_counts.rs`. Body follows the research
note pseudocode. Inline path-compressed `find` with `parent`
array. No new module.

Keep the existing `column_counts` as-is for now so both
coexist during verification.

Unit tests: the 5 existing tests, reused verbatim against
`column_counts_gnp`. All must pass bit-exact.

Commit as `phase-2.5.1: GNP column counts (parallel
implementation)`.

## Step 3 — Cross-check example

Write `examples/verify_column_counts.rs`:

- Load every KKT matrix (same pattern as
  `examples/triage_sparse_kernel_diff.rs`).
- Compute `col_counts_old = column_counts(&pat, &etree)`.
- Compute `col_counts_new = column_counts_gnp(&pat, &etree)`.
- For each matrix, assert equality; collect mismatches with
  name + first-differing index + (old, new) values.
- Print aggregate: "Matched: X/N. Mismatches: Y."

Run it. If Y > 0, do NOT proceed — debug.

Commit as `phase-2.5.1: corpus cross-check example`.

## Step 4 — Switch production call site + deprecate old

If Step 3 shows zero mismatches:

1. In `src/symbolic/mod.rs:334`, replace the call to
   `column_counts(...)` with `column_counts_gnp(...)`.
2. Rename the old `column_counts` to
   `column_counts_reference` behind `#[cfg(test)]` so the
   supernode unit tests keep a known-good oracle.
3. Keep `pub use column_counts::{column_counts_gnp as
   column_counts, total_factor_nnz}` in
   `src/symbolic/mod.rs` so external API is unchanged.

Commit as `phase-2.5.1: switch symbolic to GNP column counts`.

## Step 5 — Validation

1. `cargo test --release --lib` — must pass 118/118.
2. `cargo run --bin bench --release` — must pass Phase 2.8
   partition verdicts, no inertia/residual regression vs the
   Step 0 baseline.
3. Record before/after factor-ratio numbers for sparse and
   for the large-n (n > 500) subset.
4. Write `dev/validation/phase-2.5.1-liu-column-counts.md`
   with exit criteria table, timing split, and KKT bench
   deltas.

Commit as `phase-2.5.1: validation report + close`.

## Step 6 — Session wrap

- Session checkpoint `dev/sessions/2026-04-20-09.md`.
- Append journal 2026-04-20-09.org.
- `dev/decisions.md` entry if any decisions made (unlikely —
  pure refactor).
- Refresh `dev/context.md`.

## Abandon criteria

- Step 0 gate fails (symbolic < 5% everywhere) → abandon this
  plan, pick a different Phase 2.5.x item.
- Step 3 cross-check shows per-matrix mismatches we can't
  explain by end of session → abandon GNP for now, record in
  `tried-and-rejected.md` with the specific counterexample.
- Any test in Step 5 regresses (inertia or residual) →
  `git revert` and record.

## Estimated effort

- Step 0: 30 min
- Step 1: 60 min
- Step 2: 90 min
- Step 3: 30 min
- Step 4: 20 min
- Step 5: 60 min
- Step 6: 20 min

**Total:** ~5 hours (matches research-note and Phase 2 plan
estimate).
