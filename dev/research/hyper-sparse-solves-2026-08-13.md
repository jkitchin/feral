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

---

# Part 2: removing the `O(m)` floor

The section above closes with "the floor is not one removable term, it is the
dense-vector API itself" and names `ftran_sparse` as the follow-up. This is that
follow-up, measured.

## What the floor actually was

The phase probe showed 22 us on a 3.4-position reach at `m = 4000`, spread over
~6 linear passes with no dominant term. Each of those passes exists because
`ftran(&mut self, rhs: &mut [f64])` hands the solve a dense vector: the `P`
gather, the source scan, the `Q` scatter, and `lsolve`'s sweep all walk `0..m`
because that is the only description of the input they have.

No amount of work inside the kernels removes that. The signature has to change.

## The new entry points

```rust
pub fn ftran_sparse(&mut self, rhs: &[(usize, f64)], out: &mut Vec<(usize, f64)>)
    -> Result<(), FeralError>;
pub fn btran_sparse(&mut self, rhs: &[(usize, f64)], out: &mut Vec<(usize, f64)>)
    -> Result<(), FeralError>;
```

| step | cost |
|---|---|
| scatter the rhs (with row scaling) into pivot-position space | `O(nnz(rhs))` |
| `L`-solve over the Gilbert–Peierls reach | `O(reach work)` |
| replay the Forrest–Tomlin etas | `O(eta ops)` |
| `U`-solve over the reach | `O(reach work)` |
| gather the solution (with column scaling), sorted | `O(nnz(x) log nnz(x))` |

Nothing is proportional to `m`. The eta term is proportional to the update
chain, which is what `compute_spike` in the Forrest–Tomlin update already pays.

The design rests on a dense accumulator that is **all zero between calls** and is
restored in `O(touched)`, never `O(m)` — the same `ft_work`/`scratch_mark`
convention the FT update uses. That invariant is the one genuinely dangerous
thing here: a leak does not corrupt the solve that caused it, it corrupts the
*next* one. So the reset runs on every exit path including the error paths, and
`failed_solves_leave_the_accumulator_clean` interleaves failing solves with
succeeding ones and checks the succeeding ones against the dense oracle.

## Measured: the operation count is flat in `m`

Same fixture generator, `bump = 0`, `band = 3`, unit-vector right-hand sides.
"work" is `last_sparse_solve_work()` — positions swept plus factor entries
traversed:

| `m` | dense sweep ftran (p50) | reach-limited (p50) | **sparse API (p50)** | sparse work |
|---|---|---|---|---|
| 1,000 | 7.8 us | 4.7 us | **0.30 us** | 12.5 |
| 4,000 | 40.8 us | 20.3 us | **1.07 us** | 12.1 |
| 16,000 | 183.4 us | 89.6 us | **2.47 us** | 13.1 |
| 64,000 | 1251.0 us | 616.7 us | **3.34 us** | 11.0 |

**The work column is the result.** `m` grows 64x; the operation count does not
move (12.5 → 11.0). The dense sweep grows 160x over the same range, which is
what an `O(m)` term looks like.

Against the dense sweep at `m = 64000`: **277x on ftran, 234x on btran**.

## Why the *time* still grows a little, and why that is not an `O(m)` term

Time goes 0.30 → 3.34 us (11x) while `m` goes 64x and the work count stays flat.
An `O(m)` term would give 64x. The growth is also decelerating — 4x in `m` buys
2.3x in time from 4k to 16k but only 1.35x from 16k to 64k — which is the shape
of a cache/TLB effect reaching its plateau, not of a linear term.

The mechanism is the dense accumulator: at `m = 64000` it is 512 KB, past L2, so
each of the ~12 scattered accesses is a miss. That is `O(work × miss cost)`, not
`O(m)`. It is inherent to the scatter/gather sparse-solver design and is the
standard trade; the alternative (a hash or a compacted workspace) costs more per
access than it saves.

This is exactly why `last_sparse_solve_work()` exists and is asserted on rather
than a timing. An asymptotic claim cannot be pinned by a wall clock: a
reintroduced `O(m)` term at these sizes would look like a constant factor, which
is indistinguishable from machine noise. The counter is deterministic, so
`sparse_solve_work_does_not_grow_with_m` can pin it directly across an 8x range,
and the Python suite pins it again through the binding.

## On the mixed-density case

On the bump fixture (`m = 4000`, solution density bimodal at p50 = 3 / p90 = 366)
the sparse API measures ftran p50 **0.83 us** against 57.3 us for the dense
sweep, but its *mean* is 9.6 us — 6.5x rather than 69x. The minority of solves
that reach through the whole bump dominate the mean, and for those the sparse API
is doing the same work as the dense sweep plus a sort. That is the honest shape
of the win: enormous where the solution is genuinely sparse, roughly neutral
where it is not.

Callers with dense solutions should keep using `ftran`. The doc comments say so.

## What is still not done

- **The etas are `O(eta ops)`, not `O(work)`.** Every op is walked whether or not
  its source is nonzero. Skipping zero sources is a two-line change and was
  deliberately not made: it would diverge from `FtEta::apply_forward` on a
  non-finite multiplier (`mult * 0.0` is `NaN`, not `0.0`), and bit-for-bit
  agreement with the dense path is worth more than a branch that does not change
  the asymptotics. An indexed eta structure would, but that is a different piece
  of work.
- **The reach is sorted, not merged.** A dense-solution sparse solve pays
  `O(r log r)` where the dense sweep pays `O(r)`. Bounded and only bites the case
  the caller should not have used this API for.
- **Still unverified on the real QPLIB bases** (they are `.npz` in the discopt
  tree, not in this repo). The fixture reproduces the structural signature; the
  transfer is untested from here.

---

# Part 3: the transfer test, run by the maintainer on real bases

Everything above was measured on a synthetic fixture, with the transfer to real
QPLIB bases flagged as unverified. The maintainer ran that test (PR #162 review)
and it changed two conclusions. Recorded here rather than in a reply, because
one of them is a defect in what this note recommended.

## What held

The mechanism transfers. On both real bases the route fired (2370–2550 of 2560
solves, asserted), `max |off − on| = 0.000e0`, `factor_nnz()` identical across
arms. `ftran` improved 1.74x on QPLIB_3852 and 10.60x on QPLIB_1157 — the latter
well above anything the fixture predicted.

## What did not: the 0.25 default was wrong

| cap | `ftran` | `btran` | route fired |
|---|---|---|---|
| 0.05 | 10.59x | 0.98x | 1195 |
| **0.10** | **10.81x** | **0.97x** | **1195** |
| 0.25 (shipped) | 11.01x | **0.69x** | 2370 |
| 1.00 | 10.75x | **0.69x** | 2560 |

The sweep in Part 1 concluded "the win is flat from 0.05 to 1.00". That was
measured on `ftran` and `btran` together and reported as one number, and it is
true for `ftran` alone. `btran` is not flat: it falls off a cliff between 0.10
and 0.25, exactly where the fired count doubles. Those extra ~1175 solves are
`btran` reaches sitting in the 10–25% band, taking the route and losing 1.45x to
it — the `O(r log r)` sort this note listed under "what is still not done",
biting a population the fixture does not contain.

**Why the fixture could not show it.** `lp_basis()` produces solutions that are
either tiny (p50 = 3) or nearly `m` (p90 = 366 of 4000, ~9%) — bimodal, with
nothing in the 10–25% band. QPLIB_1157's mean solution density is 14.7% of `m`,
squarely inside it. A cap can only be validated against a basis whose solution
density straddles it, and the generator was tuned to QPLIB_3852's 2.27 nnz/col,
where `btran` is 1.02x and the cap never bites at all.

This is the concrete failure of the "measure the threshold, don't guess it"
discipline in Part 1: the sweep was real, the fixture was not representative,
and reporting one blended ratio hid a regression in half of it. **Two lessons
worth carrying: sweep each direction separately when they can diverge, and a
threshold measured only on generated data is a guess wearing a table.**

Default lowered to **0.10** — the largest measured cap that still excludes the
losing band — and pinned by
`default_density_cap_is_pinned_to_the_measured_value`, which explains that
raising it requires a basis whose solutions live in the admitted band and win
there.

## The end-to-end result, which reframes the whole issue

Measured through discopt on the root LP of QPLIB_1157, discopt held fixed and
only feral varied:

| feral arm | wall | vs crates.io |
|---|---|---|
| 0.15.1 | 6.088 s | 1.00x |
| `main` + `dense_bump_max_dim = 4096` | 3.567 s | 1.71x |
| this work + bump + cap 0.10 | 3.607 s | 1.69x |

*(The 1.71x row is the cap's value and stands. What it costs is settled
elsewhere: the cap only fires on a triangularized symbolic, so taking it means
taking the peel, and on the bchoco06 root LP that loses a dual bound — issue
#163, and issue #168 for the retracted claim that the dense route escaped it.
Speedup and trajectory change are one lever.)*

**~1.00x end-to-end**, against 10.6x on the component. The maintainer has
retracted issue #161's "93.1% of wall in the LU layer" as a profile-attribution
artifact; triangular solves are ~1% of that solve. Four other QPLIB instances
show 4–13%.

A 10.6x speedup that moves the total by 1% is a direct measurement that the
component was 1% of the total. The work here is sound and free, but **the 476x
in Part 2 is a component ratio and must not be quoted as a downstream
prediction** — and the `ftran_sparse`/`btran_sparse` follow-up in particular
should not be sized off it. On this evidence the LU cost that matters is numeric
factorization (`dense_bump_max_dim`, ≥41% of that solve's wall), not the solves.

## Standing gap

The two `.npz`/`.mtx` bases and the `real_basis_hyper.rs` harness were offered
as a PR attachment but did not arrive — neither comment body carries an
attachment link, and they are not in discopt's git history (the directory issue
#161 names, `scratchpad/issue1008/`, does not exist; `scratchpad/i1008/` holds
only Python harnesses). `capture.py` there cannot regenerate them either: it
needs a QPLIB corpus under `~/Dropbox` and a hardcoded macOS worktree path, and
it captures LP relaxations rather than simplex bases.

So the cap change here is adopted **on the maintainer's measurement, not
reproduced in-tree**. It is the conservative direction — a smaller cap strictly
reduces the population taking the new route — but until those two files are in
the repository as fixtures, no in-tree test can catch a regression of this kind.
Carrying them is the single highest-value follow-up in this note.


---

# Part 4: reproduced in-tree, and the stated mechanism was wrong

The bases arrived. Both `.mtx` and `.npz` were checked to describe the same
matrix (`max|npz − mtx| = 0.0`, dimensions and nnz as documented), and the two
`.mtx` now live in `tests/data/lu_bases/`.

## The cap finding reproduces exactly

Re-running the sweep here, on a different machine:

| cap | `ftran` | `btran` | fired |
|---|---|---|---|
| 0.02 | 7.68x | 0.96x | 1195 |
| 0.05 | 7.44x | 0.97x | 1195 |
| **0.10** | 7.18x | **0.96x** | 1195 |
| 0.25 | 7.84x | **0.62x** | 2370 |
| 0.50 | 7.50x | **0.62x** | 2370 |
| 1.00 | 7.95x | **0.61x** | 2560 |

The fired counts are identical to the maintainer's (1195 / 2370 / 2560) — they
are deterministic. The solution-density profile matches to the digit (p50 = 1,
p90 = 3593, max = 3621, mean = 580.4, 14.7% of `m`). The wall-clock ratios differ
(this machine is slower on `ftran`, and the `btran` regression is *worse* here:
0.62x, i.e. 1.6x) but the cliff is in the same place.

## The mechanism I gave was wrong

Both the PR and this note blamed the `O(r log r)` sort. That is not what costs.
Sorting 400 / 1000 / 2000 positions measures **2.8 / 9.0 / 18.3 us**, against a
regression of ~105 us per solve. The sort is at most a sixth of it.

Instrumenting the reach to count graph edges walked, on QPLIB_1157 `btran`:

| cap | fired | reach nodes/sweep | **DFS edges/sweep** | vs `nnz(L) = 75,084` |
|---|---|---|---|---|
| 0.10 | 21/256 | 2 | 3,662 | **5%** |
| 0.25 | 256/256 | 584 | 68,925 | **92%** |

**The reach DFS traverses the same factor entries the numeric sweep then
traverses again.** At 0.25 it walks 92% of `nnz(L)` just to decide which columns
to visit, and then visits them. The route pays ~1.9x to avoid 1x, which predicts
~0.53x and measures 0.62x.

That reframes the cap. It is not a workaround for an implementation detail that
a cleverer sweep could remove — it is intrinsic to reach-based solving: finding
the reach is the same order of work as the solve, so the route can only win when
the reach is a small fraction of the whole. Replacing the sort would not have
helped, and the "known limit" this note listed pointed at the wrong thing.

Corrected in the parameter's doc comment, the changelog, the PR body, and above.

## What is now guarded in-tree

`tests/lu_real_bases.rs`:

- `default_cap_excludes_the_band_where_btran_reaches_are_near_dense` — asserts on
  *reach sizes*, which are deterministic, not on timings. Verified to fail with
  the intended message when the default is put back to 0.25.
- `sparse_entry_points_agree_with_dense_on_real_bases` — `ftran_sparse` /
  `btran_sparse` over every column of both bases against the dense entry point.
  This was the one piece neither party had run on real data.
- `reach_route_is_bit_identical_to_the_dense_sweep_on_real_bases` — `assert_eq!`
  on the `f64` vectors, not a tolerance.

The standing gap from Part 3 is closed.
