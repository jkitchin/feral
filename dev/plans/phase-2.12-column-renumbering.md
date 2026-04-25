# Phase 2.12 — Column-renumbering amalgamation: implementation plan

**Status:** Implementation plan for Phase 2.12.
**Date:** 2026-04-25
**Research:** `dev/research/phase-2.12-column-renumbering.md`

---

## Goal

Replace the adjacency-gated merge in `find_supernodes` with a
merge-biased postorder so that all merges satisfying the SSIDS size
rule become naturally adjacent. Targets the 128-410 sibling-merges
the diagnostic shows are blocked on the tiny-IPM tail.

## Acceptance criteria

(See research note §7.) Summary: parity preserved; ≥50% front-count
reduction on at least one tail matrix; ≥20% total_us reduction on
ACOPR30 or CRESC100 (5-run median); no corpus regression.

---

## Steps

### Step A — Test fixtures (write tests first)

**File:** `tests/column_renumbering.rs` (new).

Two structural tests that fail under the current `Adjacency`
strategy and should pass under `Renumber`:

1. **Arrow matrix.** `n=8`, all off-diagonals in column 7 (or row 7
   stored). With `nemin=32`, 7 leaf children + 1 parent. Currently
   produces ≥7 supernodes (only one merge possible). Renumber
   target: 1 supernode covering all 8 columns.

2. **Bushy fan.** `n=33`, columns 0..31 all parented by 32 with no
   inter-leaf coupling. With `nemin=32`, currently produces 32
   leaves + 1 root. Renumber target: 1 supernode.

Add a parity test:

3. **Parity on tridiagonal.** A tridiagonal n=8 matrix is already
   chain-postordered. Both `Adjacency` and `Renumber` strategies
   must produce identical supernode lists (and identical perm).

Compile and run; expect tests 1-2 to fail until implementation lands.

### Step B — `predict_merges` function

**File:** `src/symbolic/supernode.rs` (extend).

Public function (or `pub(super)`): given `etree`, `col_counts`,
`SupernodeParams`, return:

- `fundamental_supernodes: Vec<(first_col, ncol)>` — fundamental
  supernodes from the existing Step 1 logic.
- `desired_merges: Vec<Option<usize>>` — for each fundamental
  supernode, the parent supernode index it would merge into under
  the SSIDS size rule (ignoring adjacency), or `None`.

Implementation: lift the Step 1 fundamental-supernode detection from
`find_supernodes` into a private helper, call it twice (once from
`find_supernodes`, once from `predict_merges`).

Tests: in-module test that `predict_merges` on the arrow matrix
returns 7 desired-merge entries pointing to the root supernode.

### Step C — `biased_postorder` function

**File:** `src/ordering/postorder.rs` (extend).

New signature:

```rust
pub fn biased_postorder(
    etree: &EliminationTree,
    bias: &[bool],  // bias[child_node] = true → emit late (adjacent to parent)
) -> (Vec<usize>, Vec<usize>);
```

Per-node bias is convenient: when descending into a parent's
children, partition them into `bias=false` (emit first) and
`bias=true` (emit last), recurse postorder within each partition.
Within each partition, retain the existing subtree-size ordering for
peak-memory minimization.

`bias[k] == false` for all k must equal `postorder(etree)`.

Tests: parity with `postorder` when bias is all-false; arrow matrix
with `bias[0..n-1] = true` (and `bias[n-1] = false` as the parent)
emits children 0..n-2 last (adjacent to the parent's column n-1).

### Step D — Pipeline wiring

**File:** `src/symbolic/mod.rs::symbolic_factorize_with_method`.

Add an `amalgamation_strategy: AmalgamationStrategy` field to
`SupernodeParams`, with variants `Adjacency` (default for now) and
`Renumber`.

When `Renumber`:

1. After Step 4 (re-permute → etree₁, col_counts₁), call
   `predict_merges`.
2. Build per-column `bias[c] = true` iff column `c`'s fundamental
   supernode wants to merge with its parent.
3. `post2 = biased_postorder(etree₁, &bias)`.
4. If `post2` is identity (no nonzero bias OR same as identity
   relabeling), skip the second pass.
5. Else compose: `perm₂[k] = perm[post2[k]]`, re-permute matrix,
   rebuild etree₂ and col_counts₂.
6. Continue with `find_supernodes` on (etree₂, col_counts₂). The
   adjacency check inside is still active (correctness invariant);
   it now passes naturally for desired merges.

The `Adjacency` strategy is the literal current behavior — all new
code is dead under that gate.

### Step E — Tests (re-run)

The Step A fixtures should now pass. Add:

- **Existing parity-test invariant:** all existing tests still pass
  with `Adjacency` (default) and pass with `Renumber` to within
  numeric parity. The default stays `Adjacency` so the existing
  suite is unchanged.
- **Cross-strategy parity:** for each existing test fixture, factor
  with both strategies and assert the *numeric output* matches to
  bit-equal (LDLᵀ on the same A always produces the same factor up
  to permutation; both strategies factor `Pᵀ A P` for some valid P).

### Step F — Benchmark

Re-use `src/bin/diag_amalgamation.rs` and
`src/bin/diag_small_leaf_gate.rs` to compare `Adjacency` vs
`Renumber`:

- `diag_amalgamation`: number of supernodes, multi-child counts.
  Expect dramatic reduction in supernode count on NELSON.
- `diag_small_leaf_gate`: 5-run median total_us per strategy per
  matrix. Expect ≥20% reduction on ACOPR30 or CRESC100.

Add a third small binary `diag_strategy_compare` if needed to drive
the side-by-side comparison cleanly.

### Step G — Decision gate

If Step F shows the success criteria met (research note §7):

- Flip `AmalgamationStrategy` default `Adjacency` → `Renumber`.
- Update doc-comments to point at this phase as the canonical
  amalgamation strategy.
- Run full corpus bench (`cargo run --release --bin bench`) to
  verify no broad regression.
- Commit. Append decision to `dev/decisions.md`.

If Step F shows failure:

- Record in `dev/tried-and-rejected.md` with the per-matrix numbers.
- Keep the implementation behind the `Renumber` gate (don't delete
  it — useful infrastructure for future phases).
- Do not flip the default. Commit anyway with the negative result
  documented.

---

## Out of scope

- AMF ordering (would be a separate phase).
- Numeric-side small-front fast-path (separate phase, deferred).
- Generalizing the merge rule beyond `nemin`-based SSIDS.
- Refactoring `find_supernodes` to remove the adjacency check
  entirely (it stays as a correctness invariant; renumbering simply
  ensures it always passes).

## Risks (per research note §5, abbreviated)

- Etree invariance under within-subtree relabeling — relies on the
  CHOLMOD/SSIDS invariant. Tested by parity.
- Permutation composition direction — easy to invert by mistake.
  Tested by sorted-bijection asserts.
- small_leaf_groups behavior change — expect *improvement*, but
  watch for unexpected drops via the Phase 2.10 profiler binary.
- No-bias-needed fast-path: if it misfires (skips when it should
  bias), no correctness loss but performance regression. Tested by
  exact equality between "all-false bias" and identity postorder.
