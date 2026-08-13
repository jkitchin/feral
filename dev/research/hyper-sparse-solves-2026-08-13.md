# Hyper-sparse triangular solves (issue #161B)

Date: 2026-08-13
Issue: [#161](https://github.com/jkitchin/feral/issues/161) part B
Prior art in tree: `dev/research/markowitz-133-measure-2026-07-10.md` (§ #130),
`src/bin/probe_lu_phases.rs` (which explicitly punts on the hyper-sparse case).

## The claim to be checked

Issue #161 reports that `SparseLu::ftran`/`btran` cost the same whether the
solution has one nonzero or `m`:

```
QPLIB_1157 basis (m=3937, factor_nnz=190654)
  sparse rhs (unit vector): solution nnz = 1     ftran =  98.4 us
  dense rhs:                solution nnz = 3937  ftran = 132.7 us
  --> 0.74x the cost for 0.0003x the output
```

The 0.74x — rather than 1.00x — is the tell. Exactly half of each solve already
skips zeros; the other half does not.

## Code inspection: which half is not work-proportional

`ftran` is `P`-gather → `lsolve` → etas forward → `usolve` → `Q`-scatter
(`src/lu/sparse_solve.rs:186-247`). `btran` is the transpose in reverse.

The four triangular kernels split into two shapes:

| kernel | form | zero handling | cost |
|---|---|---|---|
| `lsolve` (`L y = s`) | **scatter** | `if sk == 0.0 { continue }` before touching `L(:,k)` | `O(m + flops)` |
| `ut_solve` (`Uᵀ z = s`) | **scatter** | `continue` only *after* `row.first()` + the divide | `O(m` row touches` + flops)` |
| `usolve` (`U w = s`) | **gather** | none — every row's entry list is read | `O(nnz(U))` always |
| `lt_solve` (`Lᵀ v = s`) | **gather** | none — every column of `L` is read | `O(nnz(L))` always |

So the scatter forms are already flop-proportional and the gather forms are not.
That is exactly the measured 0.74x: the solve halves that skip zeros go to ~0,
the halves that do not stay at full cost.

`ut_solve` is a third case: it is a scatter form, but it dereferences
`u_rows[i]` (a separate heap allocation per row) *before* testing `s[i]`, so it
pays `m` cache misses regardless of sparsity even though its flop count is
already proportional.

## What the reach needs, and what is already in the tree

A reach-limited gather solve needs, for each kernel, the predecessor graph of
the substitution:

- `usolve`: `w[k]` can be nonzero when `s[k] != 0` **or** some `w[c] != 0` with
  `U[k,c] != 0`. Edges are `c -> {k : U[k,c] != 0}` — which is exactly
  `u_above[c]`, the column-wise index of `U`'s off-diagonal entries that the
  Forrest–Tomlin update already builds and maintains
  (`sparse_factor.rs:108-115`, `sparse_update.rs:236-276`). **Already present.**
- `lt_solve`: `v[k]` can be nonzero when `s[k] != 0` **or** some `v[i] != 0`
  with `L[i,k] != 0`. Edges are `i -> {k : L[i,k] != 0}` — the *row*-wise
  structure of `L`. `L` is stored CSC. **Not present; must be built.** `L` is
  never touched by the FT update ("the base `L` is never touched",
  `sparse_factor.rs:41-46`), so a CSR index of `L` built once at factor time
  stays valid for the whole update chain.

Traversal order: both graphs' edges run strictly one way down a triangular
order, so sorting the reach is a valid topological order — descending
`uperm[k]` (triangular rank) for `usolve`, descending pivot position for
`lt_solve`. `U` is upper triangular *in `uperm` order*, so it must be the rank
and not the position index that orders the `U` sweep; `L`, `P` and `Q` stay in
fixed pivot-position coordinates.

`compute_spike` (`sparse_update.rs:335-424`) already does precisely this shape
of thing for the `L` graph — mark array, explicit stack, `sort_unstable`, clear
the marks over the touched list. The new code follows it rather than inventing
a second idiom.

## Design

### 1. `ut_solve`: hoist the zero test (unconditional)

Test `s[i] == 0.0` before dereferencing `u_rows[i]`. This is what `lsolve`
already does. No new parameter, no new state, no change to the flop sequence —
purely the removal of `m` pointer chases on rows that contribute nothing.

### 2. `usolve` / `lt_solve`: reach-limited sweep behind a density gate

```
sources = { k : s[k] != 0 }                      (one O(m) scan)
if |sources| > cap:  dense sweep                 (reach ⊇ sources, so it would abort anyway)
R = DFS(sources), abort to the dense sweep if |R| > cap
sort R into topological order
sweep only R
```

with `cap = hyper_sparse_max_density * m`.

**Why the gate is one knob and not two.** The reach always contains the
sources, so an rhs already denser than the cap cannot produce a reach below it.
Testing the source count first is therefore free pruning, not a second
heuristic — it just moves the inevitable abort earlier.

**Worst case.** A *sparse* rhs whose solution fills in walks the DFS up to the
cap and then does the full dense sweep anyway: `1 + cap` relative cost. At the
default `cap = 0.25` that is a bounded ≤25% regression on the fill-in case, paid
to buy the order-of-magnitude win on the hyper-sparse case. This is the cost the
issue's "route on measured solution density, not rhs density" note is describing
— you cannot know the solution density without paying for part of the reach.

### 3. What stays `O(m)`

The `P`-gather, the `Q`-scatter and the source scan are `O(m)` and stay that
way: `ftran(&mut self, rhs: &mut [f64])` takes a *dense* rhs, so reading it is
`Omega(m)` no matter what the kernels do. Going below that needs a sparse-rhs
entry point (`ftran_sparse(&[(usize, f64)]) -> &[(usize, f64)]`), which is an
API addition and is deliberately **not** in this change. The prize here is
`O(nnz(factor))` → `O(m + reach work)`, not `O(1)`.

## Semantics that change

The diagonal guard in `usolve`/`ut_solve` (`SingularBasis` on an absent, zero,
non-finite, or out-of-order stored diagonal — L10) is currently evaluated on
**every** row of `U` on **every** solve. After this change it is evaluated on
the rows the solution actually depends on:

- `ut_solve` skips rows with `s[i] == 0.0`,
- `usolve` on the reach-limited route skips rows outside the reach.

This is a real narrowing and is recorded as such rather than glossed. The
argument that it is sound, not merely cheap: a row `k` outside the reach has
`s[k] == 0` and no reached predecessor, so the substitution assigns it
`0 / U[k,k]`. If `U[k,k]` is a healthy pivot that is `0` — which is what the
sparse route leaves in place. If `U[k,k]` is zero the row is `0 = 0`, the system
is consistent and underdetermined there, and `0` remains *a* correct solution
component. So no wrong answer is returned; what is lost is the *diagnostic*
that the factor is degenerate somewhere the caller did not ask about. A caller
that wants that diagnostic unconditionally has it at factor and update time,
where singularity is detected against the pivot tolerance rather than
incidentally during a solve.

The dense fallback route keeps the full every-row check, so a dense-rhs solve
(the case where a caller would notice) is unchanged.

## Rejected alternatives

- **Track the nonzero pattern through the whole `ftran` chain** (reach out of
  `lsolve`, incremental marking through the etas) instead of re-scanning `s`
  before `usolve`. Correct, and it would save one `O(m)` pass — but the
  `P`-gather is already `O(m)`, so it buys no asymptotic improvement and costs a
  second marker array plus per-eta-op bookkeeping in the hot loop. Re-scanning
  is one branch-predictable linear pass over a buffer already in cache.
- **Priority-queue sweep instead of DFS + sort.** Avoids the `O(r log r)` sort
  by pulling positions in rank order on the fly. Same asymptotics with a worse
  constant and much more code; `compute_spike` already established
  DFS + `sort_unstable` as the in-tree idiom.
- **Building the `L` CSR index unconditionally at factor time.** Rejected in
  favour of building it only when `hyper_sparse_max_density > 0.0`, so the
  feature-off path allocates nothing new.

## Open question deferred to measurement

The default `hyper_sparse_max_density`. The break-even is where
`reach work + DFS + sort` meets `nnz(factor)`, which depends on how the reach's
rows compare in length to the average row. Set from the A/B in
`examples/hyper_sparse_solve.rs`, not from theory.

---

# Measurements

Machine: the session container (Linux x86-64, VM). All numbers are interleaved
A/B on the *same* basis: two `SparseLu` built from one `SparseColMatrix` and one
`SparseLuSymbolic`, differing only in `hyper_sparse_max_density`, with the arm
order alternating rep by rep. `examples/hyper_sparse_solve.rs` exits non-zero if
the reach route never fires, so none of this can be a flattering measurement of
the dense path against itself.

Fixture: `m = 4000`, 1.90 nonzeros/column (issue #161 reports 2.27/col for
QPLIB_3852), a 400-column bump, `nnz(LU) = 30778` (fill 4.05x). Unit-vector
right-hand sides, 1280 calls per arm.

## Headline

```
                  ftran                     btran
  dense sweep     mean 68.3  p50 60.1 us    mean 57.5  p50 51.3 us
  reach-limited   mean 35.0  p50 23.5 us    mean 33.5  p50 23.9 us
  -->             mean 1.95x p50 2.56x      mean 1.72x p50 2.15x
  dense-rhs fallback: 0.97x           max |reach − dense| = 0.0
```

The two routes agree **bit for bit** on this fixture (`max |diff| = 0.000e0`),
which is the expected outcome and not a coincidence: the reach route evaluates
the same substitution over a subset of rows whose omitted terms are exactly
`v · 0.0`.

Median and mean are both reported because this basis's solution density is
bimodal — `ftran(eᵢ)` has a p50 of **3** nonzeros and a p90 of **366**. The mean
is set by the minority of solves that reach through the whole bump, so it
*understates* the win on the hyper-sparse case; the median overstates it for a
mixed workload. Neither alone is the honest number.

## Choosing the default

Sweeping the cap on the same fixture:

| `hyper_sparse_max_density` | ftran (mean) | positions/sweep | dense-rhs fallback |
|---|---|---|---|
| 0.05 | 2.08x | 3.1 | 0.98x |
| 0.10 | 2.06x | 32.8 | 0.97x |
| **0.25** | **2.04x** | **39.5** | **0.98x** |
| 0.50 | 2.04x | 39.5 | 0.93x |
| 1.00 | 1.98x | 133.8 | 0.67x |

The win is flat — every setting captures it, because the solves that matter have
reaches far below any of these caps. What the cap actually controls is the
**downside**, and the table shows the abort mechanism doing its job: at 1.00 the
cap can never be exceeded, nothing ever aborts, and a dense right-hand side pays
a full wasted reach (0.67x). Between 0.05 and 0.25 the fallback cost is inside
noise of 1.00x.

**0.25 shipped.** It buys the full win on this fixture at ≤2% on the dense-rhs
case, and leaves headroom for bases whose solutions are moderately rather than
extremely sparse — which 0.05 would exclude for no measured benefit.

## The residual `O(m)` floor is real and is not the marshalling

A phase split of a hyper-sparse `ftran` (`m = 4000`, no bump, reach = 3.4
positions/sweep, 4000 reps):

```
  loop base (fill of the rhs)                    0.2 us
  ftran_partial (P-gather + lsolve)             10.3 us
  full ftran − ftran_partial (usolve + Q)       13.2 us
  full ftran                                    22.1 us
```

With a 3.4-position reach, `usolve`'s arithmetic is nothing; the 22 us is ~6
linear passes over 32 KB buffers at 2-3 us each on this machine. It is spread
evenly, not concentrated — `ftran_partial`, which is only the `P`-gather and
`lsolve`, is already half of it.

I tried removing the largest-looking of those terms (sparse permuted
marshalling) and it changed nothing; that is written up in
`dev/tried-and-rejected.md`. The conclusion stands as the design section
predicted, for a different reason than the design section gave: the floor is not
one removable term, it is the dense-vector API itself. Every `ftran(&mut [f64])`
must read `m` values and write `m` values. Lifting it needs
`ftran_sparse(&[(usize, f64)]) -> &[(usize, f64)]`, an API addition, and is the
natural follow-up to this change.

So: issue #161B's `O(nnz(factor))` per solve is now `O(m + reach work)`. It is
not yet `O(reach work)`, and this note should not be read as claiming it is.

## Cross-check against the issue's numbers

Issue #161 measured `ftran` at 0.74x the dense-rhs cost for a 1-of-3937
solution — "2918x more work than it performs". The mechanism that predicts 0.74x
rather than 1.00x is the scatter/gather split in the code-inspection table above:
two of the four kernels already skipped zeros, two did not. After this change the
same measurement on the in-tree fixture is 0.97x dense-rhs vs 2.56x p50 on a
sparse rhs — i.e. the sparse case moved and the dense case did not, which is the
signature of having fixed the gather halves specifically.
