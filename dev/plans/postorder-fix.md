# Plan: Wire Postorder Into the Symbolic Pipeline

## Goal

Fix the orphan-postorder bug in `symbolic_factorize` so that supernode
detection and amalgamation operate on a postordered elimination tree.
This is the Phase 1b correctness fix that closes the remaining gap on
the 153k KKT corpus.

Research: `dev/research/postorder-pipeline.md`.

## Scope

In scope:
- Modify `src/symbolic/mod.rs::symbolic_factorize` to apply the
  CHOLMOD-style composition `final_perm = amd_perm ∘ post`.
- Add an invariant test on supernode contiguity.
- Add a regression test that loads a small bordered KKT (or MGH10S
  itself) and asserts sparse inertia == dense inertia.
- Re-run the 153k bench, record new sparse numbers in the session
  checkpoint.

Out of scope:
- Folding the two etree builds into one (Phase 2 perf optimization).
- Refinement (`solve_sparse_refined` — separate next step in the
  current Phase 1b push).
- Phase 1b exit validation document (separate next step).
- AMD itself, METIS, anything else in the analysis pipeline.

## Test-First Order

Per spec §5.1, tests come before implementation. The agent must NOT
write the implementation and the test oracle in the same session
without an external oracle. For this fix the oracles are:

1. **Hand-built bordered KKT (3×3 or 4×4)** with known inertia,
   small enough that the answer is computed by inspection.
2. **MGH10S_0000** with sidecar inertia (35, 16, 0) from MUMPS — an
   external oracle.
3. **Dense factorization on the same matrix** via `factor()` — an
   independent in-repo oracle.

### Test 1: Supernode contiguity invariant
Location: new test in `src/symbolic/mod.rs::tests` (or extend
`test_symbolic_factorize_basic`).

```rust
#[test]
fn supernode_columns_form_etree_subtrees() {
    // For each existing fixture, after symbolic_factorize:
    //   for every supernode s and every j in s.first_col..s.first_col+s.ncol,
    //     all parents up to the next supernode boundary are inside the same range.
    // This is a property of postordered etrees that the buggy version violates.
}
```

Must FAIL on `main`, PASS after fix.

### Test 2: Bordered KKT inertia match
Location: new file `tests/sparse_postorder.rs`.

A 4×4 bordered matrix:
```
[ 1  0  0  -1 ]
[ 0  1  0  -1 ]
[ 0  0  1  -1 ]
[-1 -1 -1   0 ]
```
Inertia is `(3, 1, 0)` (3 positive Hs, 1 negative pivot from the
constraint). Hand-verifiable, mirrors MGH10S structure at minimum
size.

```rust
#[test]
fn small_bordered_kkt_sparse_inertia_matches_dense() {
    // Build the matrix as CSC
    // Run dense factor → assert (3, 1, 0)
    // Run symbolic_factorize + factorize_multifrontal → assert (3, 1, 0)
}
```

Must FAIL on `main` (sparse will say (4, 0, 0) or similar), PASS after fix.

### Test 3: MGH10S regression
Location: same `tests/sparse_postorder.rs`.

```rust
#[test]
fn mgh10s_sparse_inertia_matches_sidecar() {
    // read_mtx + read_sidecar
    // sparse factor
    // assert sparse_inertia == sidecar_inertia
}
```

Must FAIL on `main` (sparse returns (50, 1, 0)), PASS after fix.
This test depends on the data directory; gate with
`#[ignore]` if `data/matrices/kkt/MGH10S/MGH10S_0000.mtx` is
absent? — instead use `Path::exists()` and skip with a printed
warning, since CI may not have data. Check existing pattern in
`tests/kkt_hardening.rs` for how the repo gates data-dependent
tests.

### Test 4: Triage example re-run as smoke test
After the fix, re-run `cargo run --release --example triage_mgh10s`.
The reported sparse inertia should match the dense inertia
(35, 16, 0) and the residual should be < 1e-6.

## Implementation Steps

### Step 1: Read existing API surface (no edits)
- `src/symbolic/mod.rs::symbolic_factorize` — current pipeline
- `src/ordering/postorder.rs::postorder` — orphan API
- `src/ordering/amd.rs::permute_pattern` — for the second permutation
- `src/symbolic/column_counts.rs::column_counts` — confirm signature
- `src/numeric/factorize.rs` — verify it only reads `sym.perm` /
  `sym.perm_inv` opaquely

### Step 2: Write Test 1 (supernode contiguity invariant)
Add to `src/symbolic/mod.rs::tests`. Run the test, confirm it fails
against `main` for an asymmetric tree fixture. If existing fixtures
all happen to be naturally postordered, hand-build a small bordered
matrix in the test itself.

### Step 3: Write Test 2 (small bordered KKT)
Create `tests/sparse_postorder.rs`. Confirm it fails.

### Step 4: Write Test 3 (MGH10S regression)
Add to `tests/sparse_postorder.rs`. Confirm it fails.

### Step 5: Implement the postorder composition
Edit `src/symbolic/mod.rs::symbolic_factorize`:

```rust
pub fn symbolic_factorize(
    matrix: &CscMatrix,
    snode_params: &SupernodeParams,
) -> Result<SymbolicFactorization, FeralError> {
    let n = matrix.n;

    // Step 1: Fill-reducing ordering (AMD)
    let full_pattern = matrix.symmetric_pattern();
    let amd_perm = amd_order(&full_pattern);

    // Step 2a: Build etree on the AMD-permuted pattern
    let amd_pattern = permute_pattern(&full_pattern, &amd_perm);
    let amd_etree = EliminationTree::from_pattern(&amd_pattern);

    // Step 2b: Postorder the etree (in AMD numbering)
    let (post, _post_inv) = postorder(&amd_etree);

    // Step 2c: Compose AMD perm with postorder
    //   final_perm[k] = amd_perm[post[k]]
    let final_perm: Vec<usize> = post.iter().map(|&p| amd_perm[p]).collect();
    let mut final_perm_inv = vec![0usize; n];
    for (new, &old) in final_perm.iter().enumerate() {
        final_perm_inv[old] = new;
    }

    // Step 2d: Re-permute on the composed permutation and rebuild etree
    let permuted_pattern = permute_pattern(&full_pattern, &final_perm);
    let etree = EliminationTree::from_pattern(&permuted_pattern);

    // Step 3: Column counts on the postordered pattern
    let col_counts = column_counts(&permuted_pattern, &etree);
    let factor_nnz = total_factor_nnz(&col_counts);

    // Step 4: Supernode detection on the postordered etree
    let supernodes = find_supernodes(&etree, &col_counts, snode_params);

    // Step 5: contribution sizes, peak memory, return
    // (unchanged below)
    ...

    Ok(SymbolicFactorization {
        n,
        perm: final_perm,
        perm_inv: final_perm_inv,
        ...
    })
}
```

Key choices:
- `import postorder` from `crate::ordering::postorder::postorder`.
- The struct fields `perm` / `perm_inv` keep their names but their
  semantics change from "AMD perm" to "composed perm". Add a doc
  comment update on the struct fields to make this explicit.
- The `etree` field in `SymbolicFactorization` now references the
  *composed-permutation* etree (not the AMD-only etree). Already
  consistent with the docstring "Elimination tree of the permuted
  matrix" — but worth re-confirming.

### Step 6: Run the new tests
- Test 1, 2, 3 should all flip from FAIL to PASS.
- The full `cargo test` suite should still pass (107 tests + the 3
  new ones).

### Step 7: Run clippy
`cargo clippy -- -D warnings` must be clean.

### Step 8: Re-run the triage example
`cargo run --release --example triage_mgh10s` — confirm sparse
inertia (35, 16, 0) and residual at machine precision.

### Step 9: Re-run the 153k bench
`cargo run --release --bin bench` — record new sparse numbers.
Expected outcome:
- Sparse inertia match should jump from 98.6% toward 99.2% (matching
  dense) or higher.
- Sparse residual pass should follow.
- If sparse > dense, that means dense has its own bug that postorder
  exposed — investigate before celebrating.

If the new numbers are lower than expected, the fix may be incomplete
or there may be a second bug; do NOT commit until the triage example
passes at machine precision.

### Step 10: Commit
Atomic commits in this order:
1. `Add postorder regression tests for sparse symbolic` — Test 1, 2, 3.
   Tests fail on this commit (new tests added without the fix).
2. `Wire postorder into symbolic_factorize (CHOLMOD-style composition)` —
   the implementation. All tests now pass.
3. (Optional) `Update docstrings on SymbolicFactorization fields` if the
   semantic change to `perm` warrants its own commit.

Per CLAUDE.md, the working tree must `cargo test` clean before each
commit — so commit 1 must be combined with commit 2 (red→green in one
commit) OR the tests must be marked `#[ignore]` in commit 1 and
un-ignored in commit 2. Choice: combine them. The split would be
artificial and the protocol "one commit per logical change" is best
served by treating the test+fix as one logical change.

Actually, on further thought: the protocol's "tests first" rule
applies to the *development order*, not the commit history. Run the
tests against `main` to confirm they fail (red), implement the fix,
confirm they pass (green), then commit the test + fix together as one
atomic change. Document the red→green transition in the commit body.

Final commit plan:
1. `Wire postorder into symbolic_factorize (with regression tests)`
   — combines Test 1, 2, 3 and the implementation.

### Step 11: Update session journal and decisions
- Append to `dev/journal/2026-04-12-NN.org`: a finding entry with the
  triage output before/after, and the bench delta.
- Append to `dev/decisions.md`: the architectural decision to use
  CHOLMOD-style postorder composition over alternatives (e.g.
  postorder embedded in find_supernodes, or a single re-permutation
  pass).

### Step 12: Hand off to next step
The Phase 1b solver convention work (`solve_sparse_refined` + dense
bench switching to `solve_refined` + Phase 1b exit validation
document) follows in the same session if time permits, or in a
follow-up session.

## Acceptance Criteria

This fix is done when ALL of the following hold:

1. Test 1 (contiguity invariant), Test 2 (bordered KKT), Test 3
   (MGH10S regression) all pass.
2. The full `cargo test` suite passes (no regressions in the 107
   existing tests).
3. `cargo clippy -- -D warnings` is clean.
4. `cargo run --release --example triage_mgh10s` reports sparse
   inertia (35, 16, 0) with residual < 1e-6.
5. `cargo run --release --bin bench` shows sparse inertia ≥ dense
   inertia on the 153k corpus, and the worst sparse residual is
   reasonable (≤ 1.0, not 1e21).

Phase 1b exit (100% inertia) is NOT a requirement of this fix — that
is the goal of the broader Phase 1b push. This fix only needs to
demonstrate that the structural bug is closed.

## Estimated Cost

- Reading existing code: small (already done in the triage)
- Writing tests: ~30 minutes
- Implementing the fix: ~30 minutes (the change is ~10 lines)
- Re-running the bench: 5–10 minutes
- Journaling and commit: ~15 minutes

Total: under 2 hours of focused work, assuming no second bug surfaces.

## Rollback Plan

If the fix causes regressions in any existing test, revert via
`git revert <commit>`. The change is contained to one function in
`src/symbolic/mod.rs` plus new test files; revert is safe.

If the fix passes all tests but the bench numbers do NOT improve
(or get worse), do NOT commit — the fix is wrong somewhere. Open a
new triage on the worst remaining failure and update this plan.
