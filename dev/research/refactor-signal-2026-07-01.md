# Research/design note: richer `update()` instability signal (issue #95)

Date: 2026-07-01
Author: agent session 2026-07-01-01
Status: design → implement

## 1. Problem

`SparseLu::update` / `DenseLu::update` collapse every instability path into a
single payload-free `Err(FeralError::NeedsRefactor)` (`src/error.rs:69`). A
caller (discopt#364's framework-level LP-error handling) cannot tell:

- an **ill-conditioning** failure (growth / tiny pivot) — where iterative
  refine-and-retry is the right response — from
- a **bookkeeping-budget** trip (update-count cap) — where a plain refactor is
  all that is needed and refine-and-retry is wasteful.

The actual magnitude (growth ratio / pivot size) is discarded.

Separately, the advisory `should_refactor()` is **cost-based only** and exists
**only on `SparseLu`**; `DenseLu` has no equivalent. discopt forces the dense LU
for small bases (`m ≤ 256`), exactly the regime that needs the parity.

## 2. Failure paths (the four causes)

Inventory from the issue, verified in code:

| cause          | sparse site                                   | dense site                              |
|----------------|-----------------------------------------------|-----------------------------------------|
| update budget  | `sparse_update.rs:93`                          | `dense_update.rs:50`                     |
| growth budget  | `sparse_update.rs:176`                         | `dense_update.rs:121`                    |
| tiny/zero/∞ piv| `sparse_update.rs:448,451,490` (elim sweep)    | `dense_update.rs:95` (bump) + `:132` (final) |
| singular repl. | `sparse_update.rs:118` (no pivot at/below diag)| — (dense folds into tiny-pivot/final)    |

The dense path has no distinct "singular" branch: a linearly dependent
replacement drives the final `U` diagonal to ~0 and trips the tiny-pivot check.
Only sparse can cheaply detect "the spike has nothing at or below its own
diagonal in rank order" (`h_rank < r_rank`) before eliminating, so `Singular`
is a sparse-only cause; dense reports `TinyPivot` for the same situation. That
asymmetry is inherent and documented, not a gap to close.

## 3. Design — additive (non-breaking), the issue's preferred option

### 3.1 `RefactorCause` + `last_refactor()`

New enum in `src/lu/mod.rs`, re-exported from the crate root:

```rust
pub enum RefactorCause { Growth, UpdateBudget, TinyPivot, Singular }
```

New field on both `SparseLu` and `DenseLu`:
`last_refactor: Option<(RefactorCause, f64)>` and accessor
`pub fn last_refactor(&self) -> Option<(RefactorCause, f64)>`.

- Set on **every** `NeedsRefactor` return path in `update`/`update_sparse`.
- `None` after a fresh `factor`/`refactor` (clean slate).
- Left untouched by a **successful** update — it records the most recent
  refactor-triggering event, which a caller consults only after an `Err`.
- `Err(NeedsRefactor)` is kept as-is: additive, non-breaking. Callers that
  ignore the accessor see no change.

Magnitude (`f64`) semantics per cause:

- `Growth`       → the element-growth high-water ratio that tripped (> `max_growth`).
- `UpdateBudget` → the update count that hit the cap (`= max_updates`).
- `TinyPivot`    → `|pivot|` of the offending diagonal (≤ `zero_pivot_tol·u_max0`).
- `Singular`     → `0.0` (no pivot; the replacement column is dependent).

### 3.2 Growth-aware refactor recommendation + parity

- `growth()` getter on both (exposes the monitor; supports the recommendation).
- `should_refactor_growth()` on both: `true` once the growth high-water reaches
  the geometric midpoint (log-space) between 1 and `max_growth`, i.e.
  `growth >= max_growth.sqrt()` (only when `max_growth` is finite and > 1).
  This lets a caller **pre-empt** a growth trip instead of discovering it on the
  update that fails.
- `should_refactor()` parity on `DenseLu` (cost-based analog): `true` once
  `updates_since_refactor() >= m`. A dense update is `O(m²)` (Hessenberg
  elimination + `O(m²)` clones) and a fresh factor is `O(m³)`, so ~`m` updates
  cost about one refactor — mirroring sparse's `update_work_total >= factor_nnz()`.
- `updates_since_refactor()` already exists on both (parity already met there).

## 4. Testing (no external oracle needed)

Each cause is reached by a deterministically constructed input; the assertions
are behavioral (which cause) and self-consistent (the magnitude relation the
path itself guarantees), not a numerical result requiring an external solver:

- `UpdateBudget`: `max_updates = 1`, second update → `Some((UpdateBudget, 1.0))`.
- `TinyPivot`: replace the last slot of a 2×2 identity with `e_0`
  (`[e0, e0]` singular ⇒ final pivot exactly 0) → magnitude `≤ ztol`.
- `Singular` (sparse): replace a slot with the **zero** column (empty spike
  support ⇒ `h_rank` is `None`) → `Some((Singular, 0.0))`.
- `Growth`: tiny `max_growth`, a growth-inducing update → cause `Growth`,
  magnitude `> max_growth` (the value that tripped the cap).
- `should_refactor` (dense): `false` fresh; `true` after `m` last-slot updates.
- `last_refactor()` is `None` right after `factor`.
