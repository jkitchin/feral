## Phase 2.2.3 — Supernode amalgamation adjacency fix

**Date:** 2026-04-13
**Commits:** `cccd640` (research note + diagnostic), `91e808b` (fix),
`fcf3c57` (test comments + bench update)
**Research note:** `dev/research/phase-2.2.3-plateau.md`

---

### 1. Summary

A single bug in `src/symbolic/supernode.rs::find_supernodes`
caused every failure observed in the Phase 2.2.2 validation
plateau, *and* was quietly corrupting results across the entire
154k-matrix KKT corpus. The symptom set previously attributed to
three independent root causes (MC64 interaction, multi-supernode
contribution-block assembly, iterative refinement stagnation)
was a single failure mode in supernode amalgamation.

The fix is a five-line adjacency check that refuses to merge a
child into its parent unless the child's column range is
immediately adjacent to the parent's column range in the
postorder column numbering.

### 2. How it was found

The diagnostic binary `examples/triage_plateau.rs` showed that
CHWIRUT1, CRESC100, CRESC132, and ACOPP30 *all* converged under
`nemin=10000` (single-supernode-forcing) but all diverged under
`nemin=32`. The bench's `nemin: 10000` override (commit `81e686c`,
"Multi-supernode solve has a known issue with contribution block
assembly") was acknowledged as a known issue but the scope was
understood to be "solve-side bug"; the real bug was in the
analyse-phase amalgamation.

The minimal reproducer (`examples/min_multisupernode.rs`) is a
6×6 arrow matrix. Under the pre-fix `nemin=1` path, the root
supernode was reported as `first_col=0, ncol=2` while another
supernode simultaneously held `first_col=1, ncol=1` — the two
supernodes both claimed ownership of column 1. The output of the
diagnostic made the bug immediately obvious.

### 3. The bug

`find_supernodes` has two passes:

1. **Fundamental supernode detection** (step 1) — groups
   consecutive columns that share a column-count chain and
   single-child elimination-tree structure into fundamental
   supernodes. Requires `n_children[j] == 1`, so chain-like
   subtrees merge but multi-child parents do not.
2. **Amalgamation** (step 2) — merges child fundamental
   supernodes into parents per SSIDS rules:
   * `trivial_chain`: parent has `ncol == 1` and the col-count
     pattern matches (child_last has one more row than parent_first).
   * `size_based`: both child and parent have `< nemin` columns.

Step 2 then updated the parent's column range with:

```rust
snode_ncols[root_p] += child_ncol;
snode_first_col[root_p] = snode_first_col[root_p]
    .min(snode_first_col[root_s]);
```

**The bug**: downstream code (`build_row_indices`, the A-scan,
`elim_cols`) assumes `first_col..first_col+ncol` is a contiguous
column range. The amalgamation code never checked that the child
and parent were adjacent in that range. For any parent with
multiple children, the *first* child processed would get merged
and the merged range would become `[child.first, parent.first+1)`
— skipping over every column that belonged to another supernode
between them.

On AMD-ordered matrices with chain-like elimination trees,
adjacency holds naturally, so the bug almost always stayed benign.
On unordered arrow-like matrices (or at strategic points in the
AMD-ordered elimination trees of ACOPP30, CHWIRUT1, CRESC100,
CRESC132), adjacency fails and variables get eliminated twice
with inconsistent state.

### 4. The fix

In the step-2 merge loop, before running the `trivial_chain` or
`size_based` tests:

```rust
let s_first = snode_first_col[root_s];
let s_ncol = snode_ncols[root_s];
let p_first = snode_first_col[root_p];
if s_first + s_ncol != p_first {
    continue;
}
```

And on merge, use unconditional arithmetic instead of `min`
(since adjacency guarantees `s_first < p_first`):

```rust
snode_ncols[root_p] = child_ncol + parent_ncol;
snode_first_col[root_p] = s_first;
```

See `src/symbolic/supernode.rs:155-220` post-fix.

### 5. Before / after on the regression panel

Command: `cargo test --test mc64_regression -- --ignored`.

| Matrix         |    n | Pre-fix residual | Post-fix residual | Canonical MUMPS | Target   | Status     |
|----------------|-----:|-----------------:|------------------:|----------------:|---------:|------------|
| CHWIRUT1_0000  |  645 |          8.50e+2 |          8.69e−14 |         9.51e−13 | < 1e−8   | **PASS**  |
| CRESC100_0000  |  806 |          1.43e+2 |          1.75e−16 |         6.15e−15 | < 1e−8   | **PASS**  |
| CRESC132_0000  | 5314 |          1.37e+5 |          4.43e−15 |         2.48e−11 | < 1e−6   | **PASS**  |
| ACOPP30_0000   |  209 |          1.08e−1 |          1.66e+5  |         5.01e−14 | < 1e−8   | **FAIL**  |

CHWIRUT1 beats canonical MUMPS by half an order of magnitude.
CRESC100 beats canonical by 2 orders. CRESC132 beats canonical by
4 orders. CRESC132's previously-observed ±2 inertia mismatch is
*also* closed — it was a symptom of the same amalgamation bug,
not a trace-rule issue as previously conjectured.

ACOPP30 regresses because the fix produces 117 fine-grained
supernodes where the pre-fix amalgamation bug accidentally fused
16 of them. With 1–2 columns per supernode the BK kernel has no
room to pivot and 14–31 pivots get `ForceAccept`'d as zero
(inertia (56–58, 122–137, 14–31) vs MUMPS (71, 137, 1)).

### 6. Bench impact

Historical sparse bench rates were an artifact of the `nemin=10000`
override, not a real capability. Dropping the override (`fcf3c57`)
reveals the true rates:

|                                  | Phase 2.2.2 (reported) | Phase 2.2.3 (honest) |
|----------------------------------|-----------------------:|---------------------:|
| Dense inertia match vs rmumps    | 97.2%                  | (unchanged)          |
| Dense residual pass vs rmumps    | 97.9%                  | (unchanged)          |
| Sparse inertia match vs MUMPS    | 99.0%                  | 74.2%                |
| Sparse residual pass vs MUMPS    | 99.8%                  | 77.9%                |

The 22-point drop is **not** a regression in the strict sense —
the pre-fix 99.8% was a number computed on a
single-supernode-forcing configuration that didn't actually
exercise the multi-supernode code. The new 77.9% is the first
honest measurement of the multi-supernode path's real capability.

Matrices that fail under the honest rate fall into three buckets:

1. **ACOPP30-class.** Small indefinite KKT matrices where the BK
   kernel's local pivoting can't maintain numerical stability in
   fine-grained supernodes. Full recovery needs delayed pivoting
   (Phase 2.3) and/or SSIDS-style column renumbering to produce
   coarser supernodes.
2. **SWOPF-class.** Persistent 1e+10-class residuals with
   sizable inertia drift. Needs investigation; likely a
   combination of (1) and a second, distinct issue.
3. **HYDCAR20-class.** Catastrophic inertia failures (e.g.
   (99,99,0) → (23,23,152)). Probably a pivot-acceptance edge
   case we haven't seen before.

All three buckets are Phase 2.3 / Phase 2.4 work.

### 7. What the plateau story teaches

The Phase 2.2.1 and Phase 2.2.2 work (MC64 scaling, column-relative
pivot rejection) remain valid and valuable — they are prerequisites
for delayed pivoting in Phase 2.3 and they closed real bugs. But
the *specific* failure mode Phase 2.2.2 was trying to attack
(MC64 driving pivots below the absolute zero floor) was neither
the root cause of CHWIRUT1/CRESC100/CRESC132's plateau, nor the
root cause of ACOPP30's 47-order recovery (which was credited to
the pivot threshold). The entire picture is messier and more
interesting than the previous session's narrative: ACOPP30 was
improved by Phase 2.2.2 via a different mechanism than intended,
and the other three matrices were waiting on Phase 2.2.3.

**Lesson for future phases:** when a fix "works" on some matrices
but not others, the mechanism may not be what you think — and
particularly, "bench shows pass rate X" with X near 100% is not
trustworthy evidence in the presence of known-issue overrides.

### 8. Follow-up work

1. **SSIDS-style column renumbering** (SPRAL
   `src/core_analyse.f90:644-685`). Allow non-adjacent children
   to merge by emitting a permutation `sperm` that renumbers
   columns so every amalgamated supernode is contiguous by
   construction. Strictly better for fill-in and flops on
   arrow-like trees; likely recovers most of ACOPP30's gap
   without touching the numerical kernel.
2. **Delayed pivoting** (Phase 2.3, already planned). Required
   for ACOPP30's full closure regardless of (1).
3. **SWOPF + HYDCAR20 triage.** Individual investigations once
   Phase 2.3 infrastructure exists.
4. **`nemin` default tuning.** The previous default (32) was
   selected under the nemin=10000 override regime where it
   didn't matter much; now that multi-supernode runs for real,
   the default should be benchmarked. SSIDS uses 8 and MUMPS
   uses 5; 32 may be too aggressive.
