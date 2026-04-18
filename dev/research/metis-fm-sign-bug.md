## Test gap: feral-metis FM sign bug + bookkeeping invariants

**Date:** 2026-04-17
**Status:** Bug confirmed, fix and test hardening pending
**Affected file:** `crates/feral-metis/src/fm_refine.rs`
**Affected function:** `refine_bisection` (edge-bisection FM)

## 1. Bug

Lines 113–118 update neighbour gains with the wrong sign for the
`gain = ed - id` convention used everywhere else in the function
(`compute_gains` line 172; cut update `cur_cut -= gain[v]` line 97).

When vertex `v` moves from→to, for each unlocked neighbour `u`:

| `labels[u]` before | edge before | edge after | Δ(ed−id) for u | correct update | code (lines 114–118) |
|--------------------|-------------|------------|----------------|----------------|----------------------|
| `from` (same as v's old side) | internal  | crossing | `+2w`          | `gain[u] += 2w` | `gain[u] -= 2w` ❌ |
| `to`   (same as v's new side) | crossing  | internal | `−2w`          | `gain[u] -= 2w` | `gain[u] += 2w` ❌ |

The comment block at lines 109–112 also rationalises in the wrong
direction (claims "one more edge crossing → gain decreases," which
is the opposite of `gain = ed − id`).

The corrected version landed in `feral-scotch/src/halo_fm.rs` and
`feral-scotch/src/band_fm.rs`; those modules use the right signs and
pass invariant checks.

## 2. Symptom

Adversarial path P_10 with alternating labels `[A,B,A,B,A,B,A,B,A,B]`
(true cut = 9, optimum cut = 1):

```
metis FM adversarial: before=9 after=-1143 labels=[0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
```

`cur_cut` is updated via `cur_cut -= gain[v]`, but `gain[v]` was
populated with the corrupted neighbour-update direction during prior
iterations of the same pass, so it drifts arbitrarily negative.
`best_cut` ends at −1143 (impossible, cut size is non-negative);
`best_prefix` stays at 0 because the negative `cur_cut` never
coincides with a balanced state in the right window. All moves are
rolled back and labels return to the input.

Net behaviour: FM is effectively a no-op on graphs where it actually
needs to move vertices.

## 3. Why the existing `fm_refine.rs` tests miss it

Every test in `crates/feral-metis/src/fm_refine.rs` falls into one
of four categories that fail to trigger the bug:

1. **Already-optimal initial cut.** `refine_bisection_does_not_increase_cut`
   uses `initial_bisect_ggp(grid(8,8), seed=17)` which produces cut = 8,
   the minimum bisection of an 8×8 grid. FM has nothing to do.
   Verified by debug print: `initial=8 returned_cut=8 actual_cut=8`.
   The assertion `final ≤ initial` is satisfied trivially (`8 ≤ 8`).

2. **Balance constraint blocks every move.** `refine_bisection_bad_init_improves`
   starts from `[B, A, A, A, A, A, A, A, A]` (cut = 2). With
   `max_imbalance = 0.20` on n = 9, no FM move satisfies the balance
   guard, so `best_prefix` stays at 0 and the function returns 2.
   Trivially `after ≤ 2`.

3. **Wrong invariant.** `refine_bisection_balance_respected` checks
   only the partition weights, not the cut quality.

4. **Permutation-only checks.** `nd_order_*` validate that the
   returned permutation is bijective and the inverse is consistent;
   they say nothing about the bisection cut along the way.

None of the tests assert the **bookkeeping-consistency invariant**
`returned_cut == cut_size(graph, labels)`, which is the one check the
bug cannot survive.

## 4. Test design — invariants every FM-style refiner must satisfy

The following invariants apply to any FM-style refinement routine,
including `feral_metis::fm_refine::{refine_bisection, refine_separator}`,
`feral_scotch::halo_fm::halo_fm_refine`, `feral_scotch::band_fm::band_fm_refine`,
and `feral_scotch::vertex_separator::compute_vertex_separator`. These
should be lifted into a shared test helper or repeated per crate.

### I1. Bookkeeping consistency (catches sign bugs directly)

```text
let after = refine(graph, &mut labels, ...);
assert_eq!(after as i64, cut_size(graph, &labels));        // edge-cut
assert_eq!(after as i64, separator_weight(graph, &labels)); // vertex-sep
```

This single assertion would have caught the bug on day one. **Add to
every existing FM test** — it is cheap and load-bearing.

### I2. Cut never grows

```text
let before = cut_size(graph, &labels);
let after  = refine(graph, &mut labels, ...);
assert!(after as i64 <= before);
```

Already present, but only meaningful when paired with I1; without I1
a runaway negative `after` satisfies the inequality vacuously.

### I3. Labels stay in {PART_A, PART_B} (or {…, PART_SEP})

Trivial sanity check; catches off-by-one writes through `labels`.

### I4. Balance respected at exit

`max(part_weight(A), part_weight(B)) <= ((1 + max_imbalance) * total / 2).ceil()`.

### I5. No-op on optimal input

If `cut_size(graph, &labels) == known_optimum`, then refinement
must leave both `labels` and the returned cut unchanged.

### I6. Determinism

Same `(graph, labels, options)` ⇒ same `(returned_cut, labels)`
across two runs.

### I7. Pass-cap honoured

For `max_passes = 0`, refinement is a no-op (returns `cut_size(labels)`,
labels unchanged).

## 5. Concrete adversarial cases to add

These cases exercise *non-trivial* FM work — the optimal cut is
substantially better than the input and FM must actually move
vertices to reach it. Each should be checked against I1, I2, I4, I6.

| ID  | Graph                 | Initial labels                 | Initial cut | Optimum | Notes                                                     |
|-----|-----------------------|--------------------------------|-------------|---------|-----------------------------------------------------------|
| A1  | P_10 path             | alternating ABABABABAB         | 9           | 1       | The minimal bug witness (Section 2).                      |
| A2  | P_20 path             | alternating                    | 19          | 1       | Tests that long sequences of moves bookkeep correctly.    |
| A3  | C_12 cycle            | alternating                    | 12          | 2       | Cycle forces ≥ 2 cut; checks even-length cycle handling.  |
| A4  | 4×4 grid              | columns ABABABAB / ABABABAB    | 24          | 4       | 2D analogue, rules out 1D-only flukes.                    |
| A5  | 6×6 grid              | random labels (fixed seed)     | varies      | 6       | Stress test with mixed boundary topology.                 |
| A6  | K_{4,4}               | one A, seven B                 | 4           | 4       | Already-optimal cut but unbalanced; FM should rebalance.  |
| A7  | Two K_4 + bridge edge | both K_4 in A                  | 0           | 0       | Disconnected components shouldn't trigger spurious moves. |
| A8  | P_10                  | all A (degenerate empty side)  | 0           | —       | Edge case: refinement must not panic on empty side.       |
| A9  | Single vertex         | all A                          | 0           | —       | Degenerate; must return 0 instantly.                      |
| A10 | Empty edge set        | half A, half B                 | 0           | 0       | No edges → no gains → no moves.                           |

Compute `initial cut` and `optimum` from the construction, not from
running the solver under test (oracle independence per CLAUDE.md
"never write both implementation and oracle in the same session").

For A6, "FM should rebalance" means: after FM with `max_imbalance = 0.10`,
both sides have weight 4 (the only balanced configuration), and the
cut equals 16 (or the minimum reachable under the balance guard).
This is one place where the current code's bookkeeping inconsistency
would show even after the sign fix — worth recording the expected
post-fix value here once the fix lands.

## 6. Ordering-side invariants (for `nd_order` / `scotch_order`)

In addition to permutation validity, every contract producer must
satisfy:

- **N1.** `inverse(perm)[perm[i]] == i` for all i.
- **N2.** Permutation is a bijection on `0..n`.
- **N3.** Determinism (same input → same perm).
- **N4.** Stats counters monotone non-decreasing across recursion.
- **N5.** **Symbolic-fill upper bound.** For a small fixed graph
  (say 5×5 grid) where the elimination tree fill can be hand-counted
  for the natural ordering and an externally-computed AMD ordering,
  assert `nnz(L) <= K` with K from the reference. This is the
  ordering-quality analogue of I1 and would catch a bisector that
  silently degrades to a poor ordering even if all permutations
  validate.

We do not currently have N5 anywhere in feral-metis or feral-scotch
tests; both crates only check N1–N3.

## 7. Where to put these tests

Three options, listed in increasing scope:

1. **Inline per crate.** Add I1–I7 to each FM-style function's existing
   tests in feral-metis and feral-scotch. Lowest friction; highest
   duplication.

2. **Shared test crate (`feral-refine-testkit`).** A `dev-dependencies`-only
   crate exporting `assert_fm_invariants(graph, labels_before, labels_after, returned_cut, max_imbalance)`
   and the A1–A10 case constructors. Both feral-metis and feral-scotch
   depend on it under `[dev-dependencies]`. Eliminates duplication;
   one new crate to maintain.

3. **Integration crate.** Add an `feral-ordering-tests` crate that
   exercises every contract producer through `feral_ordering_core`
   and runs the full I1–I7 + N1–N5 suite. Catches contract drift
   too. Largest scope; defer until we have ≥ 3 ordering crates
   actively shipping.

Recommend (2) once the metis sign fix lands. Until then, add I1 +
A1 inline to feral-metis as the regression test for this specific
bug.

## 8. Action items

1. **Fix:** swap the signs at `feral-metis/src/fm_refine.rs:115` and
   `:117`, and rewrite the comment block at `:109–112` to reflect
   `gain = ed − id`.
2. **Regression test:** add `adversarial_alternating_path_should_improve`
   with the I1 invariant `assert_eq!(after as i64, cut_size(&g, &labels))`.
   Without I1 the test passes for the wrong reason.
3. **Hardening sweep:** add I1 to every existing FM test in
   feral-metis (4 tests) and feral-scotch (8 + 6 + 6 = 20 tests).
   Cheap; high regression value.
4. **Adversarial set:** add A1–A10 to feral-metis. After the fix,
   verify each `after` matches the table.
5. **(Deferred)** Extract shared `feral-refine-testkit` per Section 7
   option (2) once the fix has landed and the API is stable.

## 9. Cross-references

- `dev/research/scotch-halo-fm.md` — corrected sign convention used
  in feral-scotch (the signs that should land in feral-metis too).
- `dev/research/scotch-band-fm.md` — same correction inside the band
  inner FM; same invariants apply.
- `dev/research/adversarial-testing.md` §1 — general motivation for
  directed adversarial inputs over random fuzzing.
