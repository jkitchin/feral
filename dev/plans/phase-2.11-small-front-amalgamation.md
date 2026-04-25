# Phase 2.11 plan — Tighten small-leaf grouping on bushy IPM KKTs

**Status:** Pre-implementation plan (locked).
**Date:** 2026-04-25
**Research:** `dev/research/phase-2.11-small-front-amalgamation.md`

Phase 2.11 commits to **Option B** from the research note: tighten
`find_small_leaf_groups` so it produces non-degenerate batches on
the tiny-IPM tail. Phase 2.12+ defers Options A/C (SSIDS-style
column renumbering) and D (numeric small-front fast path).

The starting evidence for "what's breaking" is the diagnostic in
`src/bin/diag_amalgamation.rs`:

| matrix         | leaves | n_groups | leaves/group |
|----------------|-------:|---------:|--------------|
| ACOPR30_0067   |    235 |     105  | 2.24         |
| CRESC100_0000  |    411 |     189  | 2.17         |
| LAKES_0000     |    181 |      34  | 5.32         |
| NELSON_0000    |    129 |     127  | 1.02         |
| SWOPF_0000     |     35 |       6  | 5.83         |

NELSON's 1.02 leaves/group is the worst case. We do not yet know
whether this is driven by `arena_budget` overflow, `nrow_max` /
`ncol_max` qualification, or non-qualifying snodes breaking groups
in postorder. **Step A diagnoses this, then Step B applies the
appropriate parameter or algorithmic change.**

---

## Files touched

- `src/bin/diag_amalgamation.rs` — add Step A counters.
- `src/symbolic/small_leaf.rs` — add `SmallLeafParams` doc updates,
  retune defaults, lift the non-qualifying-break rule if Step A
  identifies it as the dominant breaker.
- `tests/profiler_smoke.rs` — extend if any new invariant lands.
- (possibly) `src/numeric/factorize.rs` small_leaf branch — only
  if the algorithmic change requires it.

---

## Step A — Identify the dominant breaker (instrument the diagnostic)

Add three counters to the existing `diag_amalgamation` binary:

1. `n_qualifying`: snodes that satisfy
   `children.is_empty() && ncol≤ncol_max && nrow≤nrow_max && nrow>0`.
2. `n_group_closes_by_arena`: number of times a group flushes
   because `arena_size + leaf_size > arena_budget`.
3. `n_group_closes_by_nonqualifying`: number of times a group
   flushes because the next snode does not qualify.
4. `n_group_closes_by_end`: terminal flush.
5. The `nrow_actual` distribution among qualifying snodes
   (mean, p50, p95, max).

**Acceptance for Step A.** Output reproduces the per-matrix
leaves/group ratio above and identifies the dominant
group-closure reason for each tail matrix. **Decision gate:**

- If `arena_budget` overflow dominates → Step B1.
- If `nrow_max` rejection dominates (qualifying < leaves) →
  Step B2.
- If non-qualifying interleaving dominates → Step B3.
- Mixed: combine.

Decision logged in the journal before proceeding to Step B.

---

## Step B — Apply the targeted fix

### B1 (arena_budget): retune defaults

If Step A shows arena overflow dominates, the change is one line in
`SmallLeafParams::default()`:

```rust
arena_budget: 16384,  // was 4096
```

A 4× bump is safe: we already accept up to a `nrow_max² = 256`-byte
leaf, so the largest single leaf is 256 of 16384. At numeric time
the arena is heap-allocated once per group; 16K per group is
trivial.

### B2 (nrow_max): widen qualification

If Step A shows `nrow_max` rejection dominates, raise:

```rust
nrow_max: 32,  // was 16
```

This requires a parity audit because the numeric small_leaf path
sizes some scratch by `nrow_max`. Check for hard-coded `16` in
`numeric::factorize::small_leaf_*` paths before changing.

### B3 (non-qualifying breaker): suspend-and-resume groups

If Step A shows non-qualifying interleaving dominates, lift the
strict-postorder-adjacency requirement on
`find_small_leaf_groups`. The change:

- Allow a group to continue across a non-qualifying snode.
- Track the postorder index range covered by the group; the
  numeric driver processes intervening non-grouped snodes first,
  then "comes back" to the group when its last member's
  postorder position is reached.

This is invasive in the numeric driver. **If Step A points here,
re-evaluate scope before implementing — this may be too large
for Phase 2.11 and we may pivot to Option A in Phase 2.12.**

---

## Step C — Tests

### C1 — small_leaf_groups improvements (unit tests)

Add to `tests/profiler_smoke.rs` (or a new `tests/small_leaf_packing.rs`):

- `small_leaf_packs_more_after_phase_2_11`: build a synthetic
  bushy tree (N small leaves under one parent) and assert
  `n_groups < N` (currently it's `≈ N` for bushy patterns).
- `small_leaf_no_regression_on_chain`: ensure the chain
  (LAKES-style) case still produces ≤ original `n_groups`.

### C2 — parity (must not regress)

The existing parity suites must pass unchanged:

- `tests/small_leaf_parity.rs`
- `tests/delayed_pivoting.rs`
- `tests/threshold_consistency.rs`
- `tests/sparse_postorder.rs`

`cargo test --release`: 0 failures, 0 ignored that weren't already
ignored.

### C3 — profiler smoke

`tests/profiler_smoke.rs` continues to pass. The bucket-sum and
total-bound invariants are independent of amalgamation strategy.

---

## Step D — Verification

Re-run `cargo run --release --bin profile_supernode_distribution`.
Compare `total_us` and the ≤8 bucket count to the Phase 2.10
baseline.

**Success criteria** (any one suffices for the phase to land,
all of them ideal):

| metric                                    | baseline | target  |
|-------------------------------------------|---------:|--------:|
| ACOPR30_0067 `total_us`                   |     1222 |   ≤ 800 |
| CRESC100_0000 `total_us`                  |     1304 |   ≤ 850 |
| NELSON_0000 leaves/group (avg)            |     1.02 |  ≥ 10.0 |
| Any tail matrix `total_us` regressing     |        — | forbidden |

A 30%+ improvement on one tail matrix with no regressions is the
landing bar.

**Rejection criteria.** Any of:

- A parity test fails or an existing test flakes.
- Any Phase 2.10 acceptance invariant breaks (validation_warnings,
  bucket sum mismatch, etc.).
- A test matrix that previously passed regresses on `total_us`.
- The Step A decision lands at B3 (suspend-and-resume) and the
  numeric-driver scope is too large for one phase. Document and
  defer.

If rejected, record findings in `tried-and-rejected.md` and write
a Phase 2.12 follow-up (likely Option A — SSIDS-style column
renumbering).

---

## Out of scope

- Anything from Options A/C/D in the research note.
- AMF ordering changes.
- Numeric kernel changes outside the small_leaf branch.
- Public-API additions beyond retuning `SmallLeafParams` defaults.

---

## Step ordering & checkpoints

1. Step A: instrument diagnostic, run, log decision (one commit).
2. Step B: apply chosen fix (one commit).
3. Step C: add/update tests (one commit if separable, else fold
   into Step B).
4. Step D: run profiler binary, log results to journal,
   final commit + checkpoint.

Each step reaches a "green tests + measurable evidence in the
journal" gate before the next step starts.
