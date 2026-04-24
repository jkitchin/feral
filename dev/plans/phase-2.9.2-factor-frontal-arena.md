# Phase 2.9.2 — `factor_frontal` arena refactor

Prior phase: `dev/plans/phase-2.9-small-leaf-subtree.md` (SmallLeafSubtree
batching). Profile evidence: `dev/journal/2026-04-24-01.org` entries
16:10 / 16:15 and commit b65bb5f.

## Motivation

The Phase 2.9 batched-leaf path shipped gated Off because it produced
no measurable speedup (geomean ~1.00× on archetype long-tail IPM
matrices). The per-leaf profiler (`src/bin/diag_leaf_profile.rs`) then
showed the outer-memset hypothesis was wrong: `frontal_buf.resize`
is only 3.8% of per-leaf time. The dominant cost (42.6%, or ~66%
after correcting for `Instant::now()` overhead) lives **inside**
`factor_frontal`:

| site                                           | lines                    |
|------------------------------------------------|--------------------------|
| `let mut a = vec![0.0; nrow * nrow]`           | `src/dense/factor.rs:670` |
| copy matrix.data → a                           | `src/dense/factor.rs:671-675` |
| `let mut perm: Vec<usize> = (0..nrow).collect()` | `src/dense/factor.rs:677` |
| `let mut subdiag = vec![0.0; nrow]`            | `src/dense/factor.rs:678` |
| plus `factor_frontal_blocked` variants         | `:869`, `:876`, `:877`, `:883` |

All leaf fronts (ncol ≤ 8) land in `factor_frontal` because
`block_size` defaults to 64 (`factor_frontal_blocked` delegates at
`:864-866`). Every leaf call therefore does one nrow² heap
allocation + copy, two small Vec allocs, and a `d_panel` alloc in
the blocked variant. At 1832 leaves per archetype matrix this is
measurable dead weight.

The refactor: introduce a caller-supplied scratch struct so that the
dense kernel does **zero** heap allocations during the hot loop
inside a leaf group. The existing owning signatures stay as thin
wrappers so no call site breaks.

## Expected outcome

- Leaf-path speedup ≥ 1.5× on ACOPR30 / CRESC100 / HAIFAM (from
  removing ~35% of the current 177 ns-per-leaf `bk_kernel` cost,
  leaving ~115 ns per leaf).
- Bulk matrices show ≤ 5% change (the wrapper path is semantically
  identical to today).
- Gate `SmallLeafBatch::default()` can flip to `On` if Step H
  confirms no regression.

## Rejection criteria

Abort at Step A if the sub-phase profile shows the internal
`vec!+copy` + perm/subdiag allocs together are < 25% of
`bk_kernel`. That would mean the cost is dominated by actual
pivot-search / `scalar_pivot_step` arithmetic, which this refactor
does not touch.

## Step A — Bound the win (instrument before refactoring)

File: `src/bin/diag_leaf_profile.rs` (extend).

1. Add a diagnostic-only public entry point (or `#[cfg(test)]` hook)
   that re-exposes the *internals* of `factor_frontal` with
   `Instant::now()` checkpoints around:
   - The `vec![0.0; nrow*nrow]` + copy loop.
   - The `perm`/`subdiag` Vec allocations.
   - The `while k < ncol` pivot-search driver.
   - The return-struct assembly (`l`, `d_diag`, `d_subdiag`,
     `contrib` allocations).

   Acceptable alternative: a one-shot private fork of `factor_frontal`
   inside `diag_leaf_profile.rs` that duplicates the logic with
   timers inserted. Less invasive to production code.

2. Re-run the profiler on the same four archetype matrices. Report
   `%bk_kernel` for each internal sub-phase.

3. **Gate**: if `alloc+copy` phases sum to < 25% of `bk_kernel`,
   stop and reconsider. Record the finding in
   `dev/tried-and-rejected.md` and close the phase.

4. Otherwise, record baseline numbers in
   `dev/journal/YYYY-MM-DD-NN.org` under tag `:phase-2.9.2:baseline:`.

## Step B — Design `FrontalScratch`

File: `src/dense/factor.rs`.

Add:

```rust
/// Caller-supplied working storage for `factor_frontal_into` and
/// `factor_frontal_blocked_into`. Reusable across calls; the kernels
/// `clear()` and extend the Vecs as needed. Capacity is retained so
/// a warm scratch avoids reallocations.
#[derive(Default, Debug)]
pub struct FrontalScratch {
    /// Column-major nrow × nrow working array. On entry: lower triangle
    /// populated with the frontal matrix; upper triangle ignored. On
    /// exit: contains L factor (lower) and the trailing contribution
    /// block.
    pub a: Vec<f64>,
    /// Row permutation, length nrow.
    pub perm: Vec<usize>,
    /// Subdiagonal of D (packed), length nrow.
    pub subdiag: Vec<f64>,
    /// Panel-local D values for the blocked path, length `block_size`.
    pub d_panel: Vec<f64>,
}

impl FrontalScratch {
    pub fn new() -> Self { Self::default() }
    /// Ensure capacity for a single call on `nrow × nrow` with the
    /// given block size. Does not zero — the kernel fills on entry.
    pub fn reserve(&mut self, nrow: usize, block_size: usize) {
        self.a.clear(); self.a.reserve(nrow * nrow);
        self.perm.clear(); self.perm.reserve(nrow);
        self.subdiag.clear(); self.subdiag.reserve(nrow);
        self.d_panel.clear(); self.d_panel.reserve(block_size.max(1));
    }
}
```

## Step C — `factor_frontal_into` (unblocked path)

Add a new public function:

```rust
pub fn factor_frontal_into(
    matrix: &crate::dense::matrix::SymmetricMatrix,
    ncol: usize,
    may_delay: bool,
    params: &BunchKaufmanParams,
    scratch: &mut FrontalScratch,
) -> Result<FrontalFactors, FeralError>
```

Body: same as current `factor_frontal` except:
- `scratch.a.clear()`, `scratch.a.resize(nrow*nrow, 0.0)`, then
  copy-fill lower triangle from `matrix.data`.
- `scratch.perm.clear(); scratch.perm.extend(0..nrow);`
- `scratch.subdiag.clear(); scratch.subdiag.resize(nrow, 0.0);`
- Pass `&mut scratch.a`, `&mut scratch.perm`, `&mut scratch.subdiag`
  into `scalar_pivot_step` (which already takes `&mut` slices).

Return-struct assembly (building `FrontalFactors { l, d_diag, d_subdiag,
perm, perm_inv, contrib, ... }`) is unchanged — those remain owning
Vecs carried out of the scratch by clone-extraction. (Future phase
2.9.3 could make `FrontalFactors` borrow or swap, but not now.)

Existing `factor_frontal` becomes:

```rust
pub fn factor_frontal(matrix: &SymmetricMatrix, ncol, may_delay, params)
    -> Result<FrontalFactors, FeralError>
{
    let mut scratch = FrontalScratch::new();
    factor_frontal_into(matrix, ncol, may_delay, params, &mut scratch)
}
```

This preserves every call site in `src/`, `tests/`, examples.

## Step D — `factor_frontal_blocked_into`

Same pattern for `factor_frontal_blocked`. Signature:

```rust
pub fn factor_frontal_blocked_into(
    matrix: &SymmetricMatrix,
    ncol: usize,
    may_delay: bool,
    params: &BunchKaufmanParams,
    scratch: &mut FrontalScratch,
) -> Result<FrontalFactors, FeralError>
```

The small-ncol delegation (`if bs < 2 || ncol <= bs`) now calls
`factor_frontal_into(..., scratch)` — same scratch, no new alloc.

Existing `factor_frontal_blocked` becomes a wrapper as in Step C.

**Subtle point**: the `FORCE_SCALAR_FRONTAL` diagnostic hook (line
857) must also take a `&mut scratch` when we go via the wrapper.
Wire it as `factor_frontal_into(..., scratch)`.

## Step E — Wire the leaf-batch path

File: `src/numeric/factorize.rs`.

1. Add `frontal_scratch: FrontalScratch` to `FactorWorkspace`.
2. In `factor_one_small_leaf`, replace
   ```rust
   factor_frontal_blocked(&frontal, own_ncol, true, &bk_params)
   ```
   with
   ```rust
   factor_frontal_blocked_into(&frontal, own_ncol, true, &bk_params,
                               &mut ws.frontal_scratch)
   ```
3. Do **not** change `factor_one_supernode` in this phase — keep the
   blast radius narrow. If the refactor works for leaves, a follow-up
   session can thread the scratch through the general supernode path.

## Step F — (Optional extension) scatter directly into scratch

Today: scatter writes into `ws.frontal_values`, wraps in
`SymmetricMatrix`, passes to `factor_frontal_blocked_into` which
copy-fills `scratch.a` from `matrix.data`.

If Step A shows the copy-in is a big chunk of the saved cost, add
a variant that takes pre-populated scratch:

```rust
pub fn factor_frontal_in_place(
    nrow: usize, ncol: usize, may_delay: bool,
    params: &BunchKaufmanParams,
    scratch: &mut FrontalScratch,  // scratch.a is the matrix, not copy
) -> Result<FrontalFactors, FeralError>
```

The outer scatter loop writes directly into `scratch.a` instead of
`ws.frontal_values`. Eliminates one buffer + one copy.

**Defer decision to after Step D numbers.** If Steps C-E already hit
the 1.5× target, Step F is gravy and can land in a follow-up.

## Step G — Parity tests

File: `tests/factor_frontal_scratch_parity.rs` (new).

Bit-exact comparison between:
- `factor_frontal(A, ncol, may_delay, params)` — owning wrapper
- `factor_frontal_into(A, ncol, may_delay, params, &mut scratch)` —
  with a fresh `FrontalScratch::new()`
- `factor_frontal_into(...)` with a scratch that has been warmed by a
  previous unrelated call (exercises the `clear()`+reuse path)

For every combination, assert bit-exact `l`, `d_diag`, `d_subdiag`,
`perm`, `perm_inv`, `contrib`, `inertia`, `n_delayed`, and
`needs_refinement`. Matrices:
- The 4×4 Bunch-Kaufman paper fixture (already in `tests/`).
- A 32×32 random indefinite (fixed-seed).
- A 128×128 semi-dense (exercises the blocked path).

Existing `tests/small_leaf_parity.rs` should pass unchanged (the
leaf-batch path now uses the scratch variant but must still produce
bit-exact factors vs the scalar path).

## Step H — Benchmark

1. Rebuild and re-run `src/bin/diag_leaf_profile.rs`. Confirm
   `bk_kernel` ns/leaf drops by ≥ 25%.
2. Re-run `src/bin/diag_small_leaf.rs` (Off vs On). Success if
   geomean On/Off speedup ≥ 1.5× across the archetype matrices and
   no bulk matrix regresses > 5%.
3. Run `cargo run --release --bin bench`. Compare geomean factor_us
   vs the prior session checkpoint. No regressions > 5%.

Record numbers in the session checkpoint.

## Step I — Flip the gate

If Step H passes:
- `SmallLeafBatch::default()` → `On` in `src/numeric/factorize.rs`.
- Update `dev/decisions.md` with the rationale.
- Update `CHANGELOG.md` Unreleased section.

If Step H misses the target but the refactor itself is correct and
neutral, land the refactor anyway (cleaner code, lower alloc churn)
but leave the gate Off.

## Files touched (estimate)

| file                                           | LOC delta |
|------------------------------------------------|-----------|
| `src/dense/factor.rs`                           | +150 / -10 |
| `src/numeric/factorize.rs`                      | +20 / -2  |
| `tests/factor_frontal_scratch_parity.rs` (new)  | +180     |
| `src/bin/diag_leaf_profile.rs` (Step A, optional) | +50   |
| `dev/journal/YYYY-MM-DD-NN.org`                 | +appendix |
| `dev/sessions/YYYY-MM-DD-NN.md`                 | +checkpoint |
| `CHANGELOG.md`                                  | +1 entry if Step I fires |

## Risks

1. **Hidden alloc sites**: the returned `FrontalFactors` still owns
   `l`, `d_diag`, `d_subdiag`, `perm`, `perm_inv`, `contrib`. If
   Step A shows those return-struct allocs dominate, the refactor
   won't hit 1.5×. Mitigation: Step A is precisely to measure this.

2. **Rejection / delay paths**: `may_delay == true` + pivot rejection
   can unwind mid-kernel. The scratch must be safe to reuse on the
   next call — `clear()` + resize in the kernel prologue handles
   this. Verify with a test that constructs an indefinite matrix
   forcing delays.

3. **`FORCE_SCALAR_FRONTAL` diagnostic**: the delegation from
   `_blocked_into` to `_into` must pass the same scratch. Easy to
   get wrong; covered by parity tests in Step G.

4. **Supernode path interaction**: keeping `factor_one_supernode` on
   the owning wrapper in this phase means two sites (leaves vs
   interior) use different APIs. Accept this as transitional — a
   follow-up phase threads the scratch through the supernode path.

## Estimated effort

One focused session for Steps A-E + G + H, possibly two if Step A
surprises or Step F becomes necessary.
