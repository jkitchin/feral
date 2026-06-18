# Bump-elimination speedup: sub-diagonal pivot index + dense bump workspace

Date: 2026-06-18
Scope: accelerate the **existing** in-place, partial-pivoting `SparseLu`
Forrest–Tomlin bump elimination (`src/lu/sparse_update.rs:eliminate_bump`)
**without changing its numerics**. Two safe optimizations only (steps 1+2).
The asymptotic redesign is a separate research spike — see
[[asymptotic-bump-update-spike-2026-06-18]].

Builds on [[unsymmetric-lu]] §4.3 (the FT update design) and the implementation
history in `dev/journal/2026-06-08-01.org`.

---

## 1. Problem

discopt#229: on the MINLPLib `casctanks` root McCormick LP relaxation
(basis `m = 2169`, `na = 5238`, ~5757 nnz, well-conditioned, ~1243 clean pivots),
a single spatial-B&B node LP solve takes ~14.7 s vs HiGHS's 0.02 s. The result is
correct (optimum −167.751 to 1e-12) — purely performance. **94% (6.0 s) is inside
`SparseLu::update`**; pricing/ftran/btran/ratio-test/refactor are all <0.3 s
combined.

Inside the update (phase 2, 792 pivots):
- bump width `h − r`: avg ≈ 750, max = 2144 ≈ m;
- `eta_ops` per update tiny (0–600) ⇒ eta replay in `compute_spike` is **not** the cost;
- ~5.4 s in the pivot-selection scan, ~5.7 s in row elimination.

This is the **non-localized-spike regime** that the FT update was explicitly *not*
tuned for. The 2026-06-08 journal already documented this as the inherent worst
case ("for a DENSE-spike basis the bump spans the tail and FT degrades to
product-form-like cost ... the FT win is for localized spikes (the realistic LP
regime)"). `casctanks` is the counterexample to "realistic = localized."

## 2. Root cause (line-level, rev 5ab8074)

`eliminate_bump` (`src/lu/sparse_update.rs:305`):

1. **Pivot-selection scan, ~5.4 s.** Lines 321–327: for every pivot column
   `k ∈ [r, h]`, the inner loop `for i in k..=h { get_col(&u_rows[i], k) }` visits
   *every* tail row with a binary search, regardless of how many actually carry a
   column-`k` entry. That is O(bump²·log) of pure overhead independent of fill.

2. **Row elimination, ~5.7 s.** Lines 343–353: `row_sub` (`:423`) sparse-merges two
   column-sorted rows per eliminated entry. On a densifying bump this is O(bump)
   per row × O(bump) rows.

Note `set_column_r` (`:291`) overwrites column `r` in place with the dense spike —
there is **no cyclic column permutation** (deliberately; see §4). So the first
elimination step densifies the tail and the fill is real, not a storage artifact.

## 3. Prior art — what is OFF the table

The textbook FT cyclic-permutation approach (move the leaving column to the end →
upper-Hessenberg → eliminate the single subdiagonal) was **implemented and reverted**
(journal 2026-06-08-01.org, 19:00 + 21:00). Reason: in a sparse U the Hessenberg's
diagonal pivots are the old superdiagonal entries `U[k,k+1]`, which are *frequently
zero* → zero pivot → division by zero. The current in-place + **partial-pivoting**
scheme exists precisely to dodge that landmine (a sub-diagonal spike entry is a
nonzero pivot; the swap goes into the eta so the unit-lower base L is never
permuted). **Any change in steps 1+2 must preserve this partial-pivoting structure.**

## 4. The two safe optimizations

### Step 1 — sub-diagonal pivot index (removes the ~5.4 s scan)

Maintain, for the duration of one `eliminate_bump` call, an index of *which rows
carry a nonzero entry in column k* over the bump `[r, h]`. Pivot selection then
iterates only the actual candidates instead of `k..=h`.

- The existing `u_above[c]` index (`src/lu/sparse_factor.rs`) is **strict-upper**
  (rows `i < c` with an entry in column `c`); it does **not** cover the sub-diagonal
  candidates (`i > k`, entry in column `k`) the pivot scan needs. A new, bump-local
  sub-diagonal index is required.
- Candidate structure: a per-column list (or a scatter into a reused workspace keyed
  by column) of bump rows that currently hold a column-`k` entry, updated as `row_sub`
  creates/destroys column-`k` fill.

**Correctness argument (no numeric change):** pivot selection picks the
max-magnitude entry among rows `[k, h]` with a column-`k` entry. The index lists
*exactly that set*. Same set ⇒ same pivot row ⇒ same swaps and multipliers ⇒
**identical `FtOp` sequence**. The change only skips rows whose column-`k` entry is
absent (where `get_col` returns `None` today anyway).

### Step 2 — dense bump workspace (replaces sparse-merge elimination)

Scatter the bump rows into a reused dense workspace, run the *same* partial-pivoting
elimination with contiguous SAXPY, gather back to column-sorted sparse rows. Record
the identical `FtOp::Swap`/`Axpy` sequence.

- Constant-factor win (no per-row `Vec` alloc — already partly addressed by the L12
  scratch reuse, journal 2026-06-10; the remaining cost is the merge itself), and
  cache-friendly SAXPY over the bump column range.
- Does **not** change the O(bump²) asymptotic — the fill is genuine. The asymptotic
  improvement is the separate spike.
- Gate behind a width threshold: narrow bumps (the localized regime) stay on the
  sparse path where it is already cheap; only wide bumps pay the scatter/gather.

## 5. Reproducer (extract the real casctanks update trace)

A single basis matrix will **not** reproduce avg-bump-750 — the O(bump²) cost lives
in the *update chain*, not any one factorization. The fixture must be the
**update trace**: the initial basis + the ordered `(leaving_slot, entering_col)`
stream, plus a refactor event marker.

Extraction site (discopt `claude/root-setup-perf`,
`crates/discopt-core/src/lp/simplex/primal.rs`):
- `primal.rs:529` `self.lu.update(slot, &col)` — dump `(slot, col)` per call;
- `refactorize` (`:203`/`:278`) — dump the basis column set at each refactor (trace
  resets here).

Plan: instrument behind an env var (`FERAL_DUMP_LU_TRACE=path`) so it is a no-op in
normal builds, drive the `casctanks.nl` McCormick LP node solve through the Python
`MccormickLPRelaxer(backend="simplex")` path, and serialize the trace. Save under
`tests/data/lu_trace/casctanks/` (mirrors the `tests/data/parity/*.mtx` convention).
A feral replay harness reconstructs the basis, factors, and applies the update
stream.

## 6. Validation

Differential test: dense path (step 2) vs. sparse path vs. `DenseLu` oracle over the
**chained** casctanks trace. Assert, beyond final-`U` equality:

- **Eta-sequence equivalence, not just final U** — `compute_spike` (`:198`) replays
  `FtOp::Swap`/`Axpy`; test ftran AND btran residuals after every update in the chain.
- **Partial pivoting preserved** — the bump always takes the max-magnitude pivot
  (`eliminate_bump` lines 314–318 deliberately ignore `pivot_threshold`).
- **`NeedsRefactor` paths preserved** — vanishing bump pivot (`:329`,
  `ztol = zero_pivot_tol · u_max0`) and growth-budget exceed (`:153`) still trigger
  refactor with clean rollback.
- **Growth monitor still fed** — `changed_max / u_max0` high-water (`:146–152`) must
  cover the dense path's touched rows.

End-to-end (via `[patch]`-override of the feral git dep in discopt): `casctanks` LP
optimum unchanged (−167.751), global correctness gate (`incorrect_count ≤ 0`) holds,
re-measure the `casctanks` node solve and report the speedup honestly (expect a large
fraction from step 1 alone; step 2 on top; the residual O(bump²) is the spike's job).

## 7. Sequencing

Step 1 as a standalone, test-first PR (smallest blast radius, numerics-identical,
≈half the time). Step 2 second. Benchmark after each. Do not touch the partial-pivoting
math or attempt the cyclic permutation — that is [[asymptotic-bump-update-spike-2026-06-18]].
