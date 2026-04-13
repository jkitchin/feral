## Phase 2.3 — Delayed pivoting

**Status:** Pre-implementation research note for Phase 2.3.
**Date:** 2026-04-13
**Related:**
- Prior research: `dev/research/dense-ldlt.md`,
  `dev/research/scaling-aware-pivot-rejection.md`,
  `dev/research/phase-2.2.3-plateau.md`.
- Prior validation: `dev/validation/phase-2.2.3-supernode-adjacency.md`.
- Frontier evidence: `examples/dense_vs_sparse.rs`,
  `examples/parity_config_sweep.rs`, `tests/parity.rs` (the
  17 currently-`#[ignore]`'d frontier tests).
**Key references:**
- citet:bunch1977stable (Bunch-Kaufman pivoting).
- citet:duff1983multifrontal (multifrontal method, delayed pivots).
- citet:hogg2013pivoting (SSIDS pivoting strategy).
- citet:duff2007approximate (MUMPS pivot postponement).
- SPRAL source: `ref/spral/src/ssids/cpu/factor.hxx`,
  `ref/spral/src/ssids/cpu/kernels/ldlt_tpp.cxx`,
  `ref/spral/src/ssids/cpu/kernels/assemble.hxx`.
- MUMPS source: `ref/mumps/src/dfac_front_aux.F`,
  `ref/mumps/src/dfac_asm.F`.

---

### 1. Problem statement

The Phase 2.2.3 frontier — the 17 `#[ignore]`'d parity tests after
the supernode adjacency fix and the InfNorm scaling default — is
dominated by **inertia drift on small indefinite KKT matrices**.
The dense BK path with Knight-Ruiz equilibration solves all of
HYDCAR20, METHANL8, SWOPF, HATFLDG, HATFLDBNE, HATFLDF to machine
precision (see `examples/dense_vs_sparse.rs` evidence in commit
`610b213`); the sparse multifrontal path with the *same* scaling
gets the inertia wrong by 1–150 pivots and the residual blows up
to 10²–10¹². The remaining gap is **not scaling-related**.

The diagnostic in `examples/parity_config_sweep.rs` showed that
even after porting Knight-Ruiz to the sparse path
(commit `b87bff1`), the same matrices still fail with sparse
inertia (98, 99, 1) on HYDCAR20 vs the dense path's (99, 99, 0)
on the same input with the same scaling. The gap is the
**interaction of AMD reordering + multifrontal partial-view BK
pivoting**: at each frontal, BK only sees a slice of the matrix
and must decide pivot acceptance from local information, with no
mechanism to defer an unstable pivot to a context where more
candidates are available.

This is exactly the failure mode that **delayed pivoting** is
designed to fix. Both MUMPS and SSIDS use it; feral does not. The
column-relative threshold from Phase 2.2.2 (`pivot_threshold = 0.01`)
is the precondition for delayed pivoting — a pivot fails the
threshold test, gets delayed, and gets a second chance at the
parent. Without delays, the threshold is harmful and we
correctly removed it as a default in commit `6056740`. With
delays, the threshold becomes a precondition for correctness.

**Failure mode in one line.** Today's `factor_frontal` is
**all-or-nothing**: it tries to eliminate `ncol` columns and
either succeeds (counting `pos`/`neg` for each pivot) or marks
unstable pivots as zero via `try_reject_1x1_frontal`. There is
no path for an unstable pivot to escape the current frontal and
re-attempt at a context where elimination order, scaling
contributions, and trailing rows differ.

---

### 2. Canonical reference: SSIDS

The SPRAL/SSIDS source is the cleanest reference. Below summarizes
the model verified by the spral-expert reading
(`ref/spral/src/ssids/cpu/`).

#### 2.1 Failure within a supernode

In `ldlt_tpp.cxx:239–260` (`ldlt_tpp_factor`), the kernel walks
columns `p = nelim+1..n-1` looking for a 1×1 or 2×2 pivot that
satisfies `|a_pp| >= u * max(|col p|)` (and the analogous 2×2
det test). When neither test fires for any candidate and the
"last-resort" `p = nelim` fallback at lines 243–259 also fails,
the kernel `break`s out of the `while (nelim < n)` loop and
returns. **It does not retry within the same supernode.** The
caller (`factor.hxx:87–119`) records `node.ndelay_out = n - node.nelim`
— the number of columns the kernel could not eliminate.

Inside `ldlt_app.cxx`, the block-append (APP) pass has its own
"append failed pivots to the tail of the block column" mechanism,
but from the parent's perspective the contract is the same:
`nelim` is what was eliminated; `n - nelim` is what wasn't.

#### 2.2 Data structure for the delayed tail

There is **no separate delayed-pivot list**. The delayed columns
live in place at the trailing end of the child's own `lcol` and
`perm` arrays. Three integer fields per `NumericNode`
(`NumericNode.hxx:67–72`) carry the protocol across the tree
boundary:

- `nelim`: columns successfully eliminated at this node.
- `ndelay_out`: columns the kernel failed to eliminate (= n - nelim
  if all native, or more if some were inherited and re-failed).
- `ndelay_in`: columns delivered to this node by all children
  combined.

The child's frontal layout after factorization:

```
lcol layout for a child with nelim < n:
  [   L11 (nelim x nelim) | tail (delayed, n - nelim x n - nelim) ]
  [   L21 (m - n x nelim) | tail (m - n x n - nelim)               ]
```

The "tail" is what gets passed up. The non-fully-summed rows
`[n, m)` participate as before.

#### 2.3 Assembly at the parent

`assemble.hxx:143–351` handles parent assembly. The parent's own
`ncol` and `nrow` are pre-expanded by the sum of all children's
`ndelay_out`:

```fortran
node.ndelay_in = sum over children of child.ndelay_out
nrow = snode.nrow + node.ndelay_in
ncol = snode.ncol + node.ndelay_in
```

The parent's `lcol` is allocated at the expanded size and `perm`
at length `ncol`. Delayed columns from all children land at
columns `[snode.ncol, snode.ncol + ndelay_in)` of the parent's
fully-summed block, in arrival order. The relevant code is
`assemble.hxx:227–270`:

```cpp
int delay_col = snode.ncol;
for each child:
  for (int i = 0; i < child->ndelay_out; i++) {
    node.lcol[delay_col*(ldl+1) + j] = child->lcol[(child->nelim+i)*(lds+1) + j];
    node.perm[delay_col] = child->perm[child->nelim + i];
    // scatter the non-fully-summed rows via map[csnode.rlist[j]]
    delay_col++;
  }
```

The delayed columns then **participate in the parent's BK pivot
search just like native columns** (`factor.hxx:49,60,87`). They
can be swapped, accepted, or delayed again.

#### 2.4 Solve-side semantics

The L factor for a delayed pivot is stored at the node where it
was finally eliminated — wherever in the tree that turns out to
be. The `perm` array preserves the original variable index, so
the gather pattern at each node is `m + ndelay_in` and reads
`x[node.perm[i] - 1]` directly via `perm`.
(`NumericSubtree.hxx:286–339` and `:341–406`.)

This means a variable delayed from node A all the way to the
root contributes to k frontal matrices' below-diagonal L rows
and to k solve-phase gather/scatters, where k is the depth from A
to the root. That is the runtime cost of delays.

#### 2.5 Numerical robustness

The threshold `u = 0.01` (canonical default) bounds the L
off-diagonal entries to `1/u = 100`, which bounds growth to the
classical Bunch-Kaufman bound. Delaying is what enforces this:
if no candidate satisfies the threshold, the column is deferred
rather than accepted with a small pivot.

Worst case: a pivot delayed all the way to the root and still
unsatisfiable. The root has no parent to forward to. SSIDS has
two outcomes (`ldlt_tpp.cxx:179–204`):

- `options.action = true`: accept as zero pivot, count
  `num_zero`, mark factorization as singular.
- `options.action = false`: throw `SingularError`.

There is no infinite loop — at each ascent the parent's
`ncol` strictly grows but the total problem size is finite.

#### 2.6 `num_delay` accounting

`stats.num_delay += node.ndelay_out` at every node
(`factor.hxx:119–120`). A pivot delayed across k levels
contributes k to the count, not 1. Useful for telemetry: the
existing MUMPS `.mumps.json` and SSIDS `.ssids.json` sidecars
in `data/matrices/kkt/` report this field, so feral can directly
compare.

---

### 3. MUMPS perspective (cross-validation)

MUMPS uses the same data-structure model — delayed pivots live
in place at the trailing rows/columns of the frontal's dense
workspace, and the parent's `NASS` (fully-summed columns) is
expanded by `sum_children NASS - NPIV` at assembly time
(`ref/mumps/src/dfac_asm_master_m.F:243`). The header field at
`IW(IOLDPS+1+KEEP(IXSZ))` is overwritten with `NASS - NPIV`
(the delay count) when the frontal is stacked
(`ref/mumps/src/dfac_mem_stack.F:615`), and parent assembly
in `DMUMPS_ASM_SLAVE_TO_MASTER_*` reads it back as the count of
"fully-summed columns to absorb from this child."

`INFOG(13)` is the canonical "delayed pivots" counter; it
accumulates `NASS - NPIV` at every node (`dfac_b.F:423`,
`dfac_mem_stack.F:179`). **Same semantics as SSIDS `num_delay`:
counts events, not unique pivots.** A pivot delayed across k
levels contributes k. Useful as a cost proxy, not as a pivot
count.

#### 3.1 Where MUMPS diverges from SSIDS

There is one significant algorithmic difference that affects
how aggressive a feral implementation should be.

**SSIDS:** stops at the first column that fails the threshold.
The remaining columns of the frontal are all marked delayed
together (`ldlt_tpp.cxx:239–260` — `break` out of the
`while (nelim < n)` loop on first failure).

**MUMPS:** does *not* stop at the first failure. The pivot
search loop in `DMUMPS_FAC_I_LDLT`
(`ref/mumps/src/dfac_front_aux.F:857–1145`) iterates `IPIV_SHIFT`
across the entire block, skips failed columns via `GO TO 460`,
keeps trying subsequent columns, and uses `Inextpiv = KEEP(206)`
to cycle back to columns previously rejected as 1×1 candidates
in case they can later be paired into a 2×2 with a partner that
became available after intermediate updates. Only when the *entire
block* is exhausted without finding *any* successful pivot does
MUMPS set `INOPV=1` (line 1120) and fall through to the delay
path. The result: a single MUMPS factorization can find usable
pivots from anywhere in the block, accumulate them, and only
delay the irreducible remainder *en masse*.

A second difference is MUMPS's **STATICMODE** fallback
(`dfac_front_LDLT_type1.F:434`): when `CNTL(4) > 0` and even
the opportunistic search fails, MUMPS will force a pivot through
with a small static perturbation (`CSEUIL`) rather than delay it.
This is similar to feral's existing `ZeroPivotAction::ForceAccept`,
but applied earlier in the pivot search.

#### 3.2 Implication for feral

Two design points to choose:

1. **In-block opportunistic search vs first-failure stop.** SSIDS
   is simpler and easier to verify; MUMPS is more aggressive and
   produces fewer delays in practice. For a first implementation,
   **match SSIDS**: break on first failure and let the parent
   absorb the tail. Revisit only if the parity panel reveals
   matrices where the in-block opportunistic scan would have made
   a difference.

2. **Static fallback.** Keep `ZeroPivotAction::ForceAccept` as
   the option to force a pivot through at the *root* supernode
   only, when delays have nowhere left to go. This is feral's
   existing behavior and matches MUMPS's STATICMODE in spirit.

---

### 4. Current feral state

#### 4.1 `factor_frontal` (`src/dense/factor.rs:433–806`)

Today's contract:
- Input: `SymmetricMatrix` of size `nrow`, target `ncol` to
  eliminate.
- Output: `FrontalFactors` with always exactly `ncol` D entries
  and `nrow × ncol` L block; a separate `(nrow - ncol) × (nrow - ncol)`
  contribution block.

When a 1×1 pivot fails the column-relative threshold,
`try_reject_1x1_frontal` (`src/dense/factor.rs:818–855`) zeroes
the L column and the diagonal, increments `zero`, and sets
`needs_refinement`. The pivot is permanently lost — there is no
mechanism to leave it un-eliminated and pass it up.

When a 2×2 pivot's Duff-Reid growth bound fails (line 685),
the kernel falls back to a single 1×1 at `k`, which goes through
the same reject path.

#### 4.2 `FrontalFactors` (`src/dense/factor.rs:390–422`)

Fields: `nrow`, `ncol`, `l: Vec<f64>`, `d_diag: Vec<f64>`,
`d_subdiag: Vec<f64>`, `perm`, `perm_inv`, `contrib: Vec<f64>`,
`contrib_dim`, `inertia`, `needs_refinement`, `zero_tol`,
`zero_tol_2x2`.

The `contrib` field is **separate** from `l` (unlike SSIDS
where they share storage in `lcol`). `contrib` holds the
`(nrow - ncol)²` Schur complement.

#### 4.3 `factorize_multifrontal` (`src/numeric/factorize.rs:73–249`)

For each supernode in postorder:
1. `build_row_indices` constructs `[fully_summed_cols, trailing_rows]`
   from the symbolic structure. Currently
   `fully_summed_cols = first_col..first_col+ncol` — a contiguous
   range.
2. Assemble original A entries (with InfNorm scaling).
3. `extend_add` scatters child contribution blocks.
4. `factor_frontal(&frontal, ncol, params)`.
5. Save the contribution block keyed on `row_indices[ncol..]`.

The contribution block carries only the trailing (non-fully-summed)
rows. There is no representation of "delayed pivots" in the
inter-supernode contract.

#### 4.4 `solve_sparse_core` (`src/numeric/solve.rs:85–219`)

Three phases, all iterating supernodes via `for node in node_factors`:
forward sub in postorder, D-block in postorder, backward sub in
reverse postorder. Each phase gathers `w[i] = y[row_indices[perm[i]]]`,
does its work, and scatters back. The contract assumes `row_indices`
exactly matches the structure used at factor time.

#### 4.5 Symbolic/post-2.2.3 supernode adjacency invariant

`find_supernodes` (`src/symbolic/supernode.rs`) only merges
supernodes whose column ranges are adjacent in the postorder
column numbering. Delayed pivoting **does not interact** with
this invariant — the symbolic supernode structure is still
contiguous-column; delayed pivots are a *runtime* expansion of
each frontal's ncol/nrow during numerical factorization. The
amalgamation logic doesn't need to know about delays.

---

### 5. Design space

#### 5.1 Where to store the delayed tail

**Option A: Fold into the existing `contrib` field.**
After factorization, redefine `contrib` to be `(nrow - nelim) × (nrow - nelim)`
where the first `(ncol - nelim)` rows/columns are delayed pivots
and the remaining `(nrow - ncol)` are non-fully-summed. The
parent's assembly differentiates the two by counting how many
delays each child has.

- **Pro:** Reuses existing storage path; minimal new fields.
- **Con:** The contrib block is currently formed as
  `A22 - L21·D·L21^T`, which is the Schur complement of the
  *eliminated* block. With delays, the "trailing" block hasn't
  been reduced and is just the sub-block of `a` at rows/cols
  `[nelim, nrow)`. This requires changing the contrib semantics
  and the `extract_contrib` step in `factor_frontal`.

**Option B: Add a separate delayed-pivot block to `FrontalFactors`.**
New fields: `nelim: usize`, `delayed_l: Vec<f64>` (the L21 rows
of the delayed columns, indexed by trailing rows), `delayed_diag:
Vec<f64>` (the un-eliminated diagonal entries), `delayed_off: Vec<f64>`
(the un-eliminated off-diagonals between delayed columns).

- **Pro:** Cleaner separation; `contrib` keeps its existing
  Schur-complement semantics.
- **Con:** More fields, more complex assembly logic at the parent
  to recompose the delayed block into the parent frontal.

**Option C: SSIDS-style merge `l` and `contrib` into a single `lcol`.**
Refactor `FrontalFactors` to store everything in one column-major
`lcol` of size `nrow × nrow` (or `m × n` where `m = nrow + ndelay_in`,
`n = ncol + ndelay_in`), with the L block, D, and
delayed/contribution tail all addressed by offsets.

- **Pro:** Closest to the SSIDS reference; the inter-supernode
  contract becomes "give me a slab of `lcol` to copy into my own
  slab".
- **Con:** Largest refactor of the data structures; touches every
  `factor_frontal` consumer.

**Recommendation: Option A.** It's the smallest change that
preserves the existing solve and assembly machinery. The `contrib`
field expands semantically (it now carries delayed pivots in its
top-left corner) but the wire format stays the same: a square
dense block with row indices. The assembly side learns to treat
the first `n_delayed` row/column positions of an incoming child's
contrib as "fully-summed columns to add to my own ncol" rather
than "trailing rows to add to my non-fully-summed region".

If Option A turns out to be hard to reason about (e.g., the
Schur complement vs raw sub-block distinction creates subtle
bugs), fall back to Option B as a clearer separation.

#### 5.2 Where to detect a delayed pivot

The detection happens inside `factor_frontal`. Currently
`try_reject_1x1_frontal` returns `true` when a pivot fails the
threshold and zeros the L column. The new behavior:

- Return `Delayed` (a new state) when the pivot fails and the
  caller has indicated it can handle delays (i.e., this is not
  the root supernode).
- The kernel `break`s out of the `while k < ncol` loop on the
  first delayed pivot, leaving `nelim = k` and the trailing
  rows/columns intact.
- Return `FrontalFactors { nelim: k, .. }` with `nelim < ncol`.

The 2×2 path needs the same treatment — when the Duff-Reid
growth bound fails, the kernel breaks instead of falling back to
1×1.

**Design detail: do we delay one pivot at a time or break on the
first failure?** SSIDS breaks immediately. This is the simplest
and matches "the kernel can't make progress here." Within a
single supernode, the pivots that come AFTER a failed pivot may
also be unstable in the new (still-unreduced) submatrix, so
attempting them is risky.

**Recommendation: break on first failure**, matching SSIDS.

#### 5.3 Where to flag "this is the root, no further delays"

The root supernode has no parent, so a delayed pivot at the root
must be either:
- accepted as zero (`ZeroPivotAction::ForceAccept`), or
- raise an error (`ZeroPivotAction::Fail`).

This means `factor_frontal` needs an extra flag: `is_root: bool`.
At the root, the kernel does *not* break on failure but instead
falls through the existing `try_reject_1x1_frontal` path with
`ForceAccept` semantics.

#### 5.4 Solve-side changes

For each supernode the gather is `w[i] = y[row_indices[perm[i]]]`
for `i in 0..nrow`. With delayed pivoting, the `nrow` includes
delayed columns from children. The solve sees a uniform "ncol
columns to eliminate, nrow total" picture that already includes
the delays, because they were merged into the parent's frontal at
factor time.

**The solve side requires no protocol changes** as long as
`SparseFactors::node_factors[i].row_indices` and `nrow` reflect
the post-delay-merge dimensions. The gather/scatter pattern is
identical; only the sizes per supernode change.

#### 5.5 Threshold default

When delayed pivoting lands, the `pivot_threshold = 0.01` default
that we removed in commit `6056740` should be **restored** —
it's the precondition for delays to fire correctly. This is the
opposite of the parity_config_sweep evidence today (which showed
threshold = 0.0 is best), but only because today's threshold
fires `ForceAccept` instead of `Delayed`.

---

### 6. Recommended approach

**Phase 2.3 implements SSIDS-style delayed pivoting as Option A
on top of feral's existing supernode machinery.** Specifically:

1. **Mechanism in `factor_frontal`:** add a `nelim` output
   distinct from the input `ncol`, and a control flag for "may
   delay". When the threshold test fails (1×1 column-relative
   or 2×2 Duff-Reid), break out of the `while k < ncol` loop
   instead of zeroing the pivot. Return `FrontalFactors` with
   `nelim < ncol` and the trailing rows/columns intact in
   `contrib`.

2. **Assembly in `factorize_multifrontal`:** the parent's
   `build_row_indices` accounts for `sum(child.n_delayed)` extra
   "fully-summed" columns coming from children. Each child's
   `n_delayed` rows of its `contrib` block are scattered into the
   parent's expanded fully-summed region (cols
   `[snode.ncol, snode.ncol + sum_delays)`). The `n_delayed`
   columns are added to the parent's effective `ncol` for the
   call to `factor_frontal`.

3. **Per-node `n_delayed_in` and `n_delayed_out` tracking** in
   `NodeFactors`. After factorization, `n_delayed_out = ncol_attempted - nelim`.
   The contribution block carries the delayed columns at its top.

4. **Root handling:** the symbolic root supernode (the one
   without any parent in the etree) is factored with `is_root = true`,
   which disables the delay path and falls through to the
   existing `ForceAccept` zeros.

5. **Solve-side:** *no protocol change*. The solve already
   iterates `row_indices[..]` and `perm[..]` of the stored
   `NodeFactors`; if those reflect the post-delay-merge
   dimensions, the existing gather/scatter handles delays
   transparently.

6. **Default `pivot_threshold` restored to 0.01** once the
   delay path is in place. This is the SSIDS / MUMPS default and
   is the precondition for delays to actually fire. Update
   `BunchKaufmanParams::default()` and the parity test
   regeneration template.

7. **Unit tests:**
   - `factor_frontal` with `is_root = false` on a small
     constructed frontal where one column is unstable; verify
     `nelim < ncol` and the trailing rows match a hand
     computation.
   - `factor_frontal` with `is_root = true` on the same frontal;
     verify the existing `ForceAccept` behavior.
   - `factorize_multifrontal` on a small KKT matrix where one
     pivot is known to fail at a leaf supernode and succeed at
     the parent; verify the root inertia matches a dense LDL^T
     reference.

8. **Parity panel:** rerun `parity_config_sweep.rs` and
   `select_parity_panel.rs` after the change. Expect HYDCAR20,
   METHANL8, SWOPF, HATFLDG, HATFLDBNE, HATFLDF, ACOPP30 to
   flip to passing — these are the matrices the dense path
   solves with KR equilibration alone (delayed pivoting in the
   sparse path is the multifrontal analog of "BK with full
   pivot visibility").

---

### 7. Risks and unknowns

**Risk: contrib semantics change is subtle.** Option A redefines
the contrib block to carry both delayed pivots and trailing
rows. The existing `extend_add` function assumes everything in
contrib is a trailing-row contribution. The fix is to split
contrib into "delayed-pivot section" and "trailing-row section"
during assembly at the parent. A clean implementation might add
a `n_delayed` field to the `ContribBlock` struct and have
`extend_add` consult it.

**Risk: `factor_frontal` is shared between dense and sparse paths.**
The dense BK in `dense::factor::factor()` uses `factor_frontal`
indirectly via the same kernel. We need to make sure the dense
path can opt out of delayed pivoting (`is_root = true` always)
so dense behavior doesn't change.

**Risk: tail growth blows up on pathological matrices.** A worst
case is a column that gets delayed all the way from a leaf to
the root, increasing the root's effective `ncol` by
`O(tree_depth)`. For very tall trees this could quadratically
inflate root memory. SSIDS and MUMPS both accept this risk; it's
bounded by `n` total delays and is rare in practice. Phase 2.3
inherits the same risk profile.

**Unknown: interaction with the supernode adjacency invariant.**
The Phase 2.2.3 fix requires that supernodes' column ranges are
contiguous. Delayed pivoting expands each frontal's `ncol`
*at runtime* to include delayed pivots from children. The
*symbolic* supernode structure is unchanged (every symbolic
supernode still owns a contiguous range). But the *numeric*
factor block at each supernode is now larger, and the L factor
indexes columns by the runtime-expanded position, not the
symbolic position. The solve must be careful to use runtime
column indices.

**Unknown: `nelim` vs `ncol` naming.** Today `ncol` means "number
of columns to eliminate at this supernode" and is also "number
actually eliminated". With delays, these split. We need to pick
clear names: SSIDS uses `nelim` for actually-eliminated and
`ncol` for the symbolic upper bound (pre-delay). I'll match
SSIDS naming.

**Unknown: how aggressive should the delay be?** SSIDS breaks on
the first failure within a supernode. An alternative is to keep
trying subsequent columns and only delay the failed ones — but
this would require re-checking pivot stability after each later
elimination, which is what BK normally does already. Breaking on
first failure is simpler and SSIDS-canonical.

---

### 8. Out of scope for Phase 2.3

- **Reusing factor_frontal for the dense path with delays.** The
  dense path is `is_root = true` always; no delay machinery
  exposed.
- **GPU offload.** Not relevant.
- **Adaptive threshold tuning.** Use the SSIDS default `u = 0.01`.
- **Parity tests for matrices that even the dense path fails on
  (e.g. ACOPP30).** ACOPP30 is a known-hard problem the dense
  path also fails on; delayed pivoting alone may not close it.
  Track it but don't gate Phase 2.3 on closing it.

---

### 9. Exit criteria

Phase 2.3 is done when:

1. `cargo test` is green (all 160+ tests pass).
2. `cargo run --release --example dense_vs_sparse` shows the
   sparse path matching the dense path on HYDCAR20, METHANL8,
   SWOPF, HATFLDG (the four matrices where dense-with-KR
   succeeds today).
3. The parity panel is regenerated with the delayed-pivoting
   default and the passing-test count strictly increases. Target:
   at least 7 of the currently-17 frontier matrices flip to
   passing. Stretch: 12 of 17.
4. The `pivot_threshold = 0.01` default is restored and the
   `tests/pivot_rejection.rs` tests still pass.
5. A new unit test in `factor_frontal` verifies the
   `nelim < ncol` path on a hand-constructed unstable frontal.
6. A new integration test in `factorize_multifrontal` verifies
   delays propagate from a leaf to the parent and elimination
   succeeds on a small constructed KKT matrix.
7. The session checkpoint reports the parity-panel pass-count
   delta as the headline progress metric (per the user
   directive in the prior session).
