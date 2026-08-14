# Is the sparse LU's cost a pivot-ordering question? (issue #161, reframed)

Date: 2026-08-13
Fixtures: `tests/data/lu_bases/` (real discopt simplex bases, added in PR #162)
Harness: `examples/lu_fill_orderings.rs`, `examples/basis_refactor.rs`

## The question

Issue #161's original premise — "93.1% of wall in the LU layer, and triangular
solves are the thing to fix" — was retracted by its author after an end-to-end
measurement. What replaced it:

> Numeric factorization is **≥41%** of the motivating LP's wall. The factor
> shows 5.7x fill on a basis averaging 6.8 nonzeros per column. Is the fill
> itself the cost, i.e. is this a pivot-ordering question?

Now answerable, because the basis is in the repository.

## Method

`examples/lu_fill_orderings.rs` holds the basis, the pivot rule, and the numeric
kernel fixed, and varies only the column permutation fed to
`SparseLuSymbolic::with_order`. `with_order` reports the whole basis as bump, so
the dense-bump route cannot fire on any ordering arm — every arm runs the same
sparse scatter kernel, and the comparison is ordering against ordering.

The last two rows then hold the *ordering* fixed and vary the **kernel**, which
is the other half of the question.

## Result: QPLIB_1157 (m = 3937, 7.46 nnz/col — the instance #1008 is about)

```
  ordering        nnz(LU)     fill   numeric(ms)   ns/factor-nnz
  natural          913860    31.11x      630.21       689.6
  AMD              190654     6.49x       74.82       392.4
  AMF              150001     5.11x       35.70       238.0
  METIS            195068     6.64x       83.33       427.2
  peel+sparse      197937     6.74x       90.81       458.8   (bump=611)
  peel+denseBump   198702     6.76x       20.23       101.8   (bump=611)
```

QPLIB_3852 (m = 1760, 2.27 nnz/col) is uninteresting by comparison — every
ordering lands at fill 1.01–1.09x and 0.31–0.40 ms. It has almost no fill to
save. All of the following is about the dense-per-column basis.

## Answer: yes to both halves, and they are independent

**Ordering matters, and feral is not using its best one.** AMF gives **21% less
fill than AMD** (150,001 vs 190,654) and factors **2.10x faster** (35.70 vs
74.82 ms). METIS is slightly worse than AMD. The LU symbolic hardcodes AMD
(`sparse_symbolic.rs`, `amd_permutation`), while the LDLᵀ side has had AMF
available as `OrderingMethod::Amf` all along. AMF costs no more to compute here
(21.16 vs 21.32 ms on the whole basis).

**The kernel matters more.** At essentially identical fill (197,937 vs 198,702)
the dense-bump route is **4.5x faster** than the sparse scatter kernel (20.23 vs
90.81 ms) — 101.8 vs 458.8 ns per factor-nonzero. That is not a fill difference
at all; it is the same arithmetic run through a blocked dense kernel instead of
a scalar scatter.

So the honest answer to "is the fill itself the cost" is: **partly, but the
larger factor is what computes the fill, not how much of it there is.**

## They compound

Re-running the three-arm `basis_refactor` A/B with the bump ordering switched
between AMD and AMF (temporary patch, reverted; not in the tree):

| arm | AMD | AMF |
|---|---|---|
| whole-basis ordering, sparse kernel | 101.40 ms, nnz 190,654 | **59.43 ms**, nnz 150,001 |
| peel + sparse bump | 97.45 ms, nnz 197,937 | 54.36 ms, nnz 165,559 |
| peel + dense bump | 23.72 ms, nnz 198,702 | **21.97 ms**, nnz 167,682 |

Two independent levers, both already implemented in this repository:

| lever | effect on numeric factorization | state |
|---|---|---|
| `dense_bump_max_dim` > 0 | **4.28x** (97.45 → 23.72 ms) | merged in #160, **defaults to 0** |
| AMD → AMF for the LU ordering | **1.79x** (97.45 → 54.36 ms) at today's defaults; 1.08x on top of the dense route | not wired up at all |

**At today's shipped defaults the factorization costs 97.45 ms. With both levers
it costs 21.97 ms — 4.44x.** Since numeric factorization is ≥41% of that LP's
wall, this is the part of feral where end-to-end time actually is.

## What this does *not* establish

- **Two bases is not a panel.** AMF wins on both here, but the maintainer has
  already measured `dense_bump_max_dim = 4096` regressing QPLIB_2017 to 0.80x,
  which is exactly the kind of instance-dependence that makes a two-point sample
  worthless for setting a default. Neither lever should be flipped on this
  evidence; both deserve the QPLIB panel.
- **Why AMF wins here is not explained.** AMF (approximate minimum fill) and AMD
  (approximate minimum degree) optimize different proxies, and minimum-fill
  proxies are known to do better on some structures and worse on others. Nothing
  here says which side an arbitrary LP basis falls on.
- **The dense-route cap is still dimension-based.** `dense_bump_max_dim` bounds
  the bump's *dimension*, which is a memory bound, not a performance predictor.
  The route wins when the bump's factor is dense and should lose when a large
  bump factors sparsely — which is a plausible mechanism for the QPLIB_2017
  regression, and is checkable against that instance if it can be obtained.

## Suggested next step

A QPLIB-basis panel — more dumped bases across instances — measuring the 2x2 of
{AMD, AMF} x {sparse bump, dense bump}, reporting fill and numeric time per
instance. That is the evidence needed to set both defaults, and it needs data
this repository does not have rather than code it does not have.

---

# Issue #163: the peel is reverted to opt-in

Date: 2026-08-13 (same session, after the above)

## The report

discopt's `bchoco06_illcond_scaled_path_recovers_bound_649` — an ill-conditioned
root LP that had been certifying `Optimal` — started returning `Numerical`, i.e.
losing its dual bound, against a feral carrying PR #160. `dense_bump_max_dim` was
at its default of `0`, so the route the peel exists to enable was switched off.

## Bisect

discopt held at `8513cfa` with `[patch.crates-io] feral = { path = ... }`, three
arms, temporary env vars in feral:

| arm | ordering | result |
|---|---|---|
| A | peel + AMD on bump (#160 `analyze`) | **FAILED** — `Numerical` |
| B | peel, bump left in peel order | **FAILED** — `Numerical` |
| C | whole-basis AMD (pre-#160) | **PASSED** — `Optimal` |

The bump's AMD is not the lever. **The peel is.** Reproduced in both directions
after the fix landed: `analyze → analyze_amd_only` passes, flipping it back to
`analyze_triangularized` fails, same message.

## The mechanism is *not* stability — measured

The obvious reading ("the peel decides pivots structurally, so it takes a bad
pivot") is wrong, and it was worth an hour to find that out.

Method: `analyze` was temporarily instrumented to dump every basis it was handed
as Matrix Market, and the discopt test run under both orderings — 46 bases on the
passing (AMD) trajectory, 30 on the failing (peel) one. Each basis was then
re-factored both ways in feral and scored on backward error (relative residual of
`A x = b` and `Aᵀ y = c`) and forward error (against a known `x_ref`), by
`examples/probe_illcond_ordering.rs`.

**Result: the peel is never the worse ordering.** On the 30 bases of the failing
trajectory:

- backward error ~1e-16 under both orderings, on every basis;
- forward error rises to 2.6e-11 (the LP really is ill-conditioned) — and the
  peel's ratio to whole-basis AMD is **0.0x–1.0x, never above 1.0x**;
- fill is slightly *better* under the peel here (2519 vs 2588 on the worst basis),
  the opposite of QPLIB_1157 above.

Sample (failing trajectory, tail):

```
  basis          amd_resid  amd_fwderr  peel_resid  peel_fwderr  resid_x  fwd_x   nnz amd/peel
  basis_000025    1.93e-16    4.55e-13    1.82e-16    9.94e-15     0.9x    0.0x   2496/2494
  basis_000026    9.65e-17    2.58e-11    1.82e-16    2.59e-11     1.9x    1.0x   2588/2519
  basis_000029    1.82e-16    5.53e-12    1.82e-16    2.08e-12     1.0x    0.4x   2730/2683
```

So what changes is the **trajectory**, not the accuracy. At forward errors of
~1e-11 the two orderings' solves disagree in exactly the bits discopt's ratio
test reads; the two runs then take different pivot sequences (46 refactorizations
against 30), and this LP is conditioned badly enough that one path certifies and
the other trips discopt's numerical guard.

## Why revert anyway

Not "the peel is unstable" — that claim is refuted above. The argument is:

1. **The peel has no standalone payoff.** On the real QPLIB_1157 basis it is
   *worse on fill* (197,937 vs 190,654) and 1.04x on time. That is not a result
   worth perturbing a downstream solver's arithmetic for.
2. **Its real payoff, 4.28x, is the dense-bump route** — and that route is off by
   default, so nobody taking `..LuParams::default()` was getting any of it.
3. **It cost a downstream regression that had been green.** Against a benefit of
   ~nothing, that settles it.

So `analyze` returns to whole-basis AMD and the peel becomes
`analyze_triangularized`, to be opted into *together with* `dense_bump_max_dim`
— which is the only configuration where it pays for itself.

**What this does not establish:** nothing here says whole-basis AMD is the better
trajectory in general. It is the one that was in place when the downstream test
was green. A different ill-conditioned LP could just as easily prefer the peel;
that is the nature of a trajectory-sensitivity result, and it is the honest
reading.

## Does the fix strand the 1.71x?

**Yes. Corrected 2026-08-14 — the original answer here was "No", and it was
wrong.** See issue #168, filed from the discopt side, and the independent
re-measurement below which reproduces it.

#163's stated motivation is taking `dense_bump_max_dim = 4096`, which under the
new pairing also means taking the peel — so the worry was that the configuration
worth 1.71x is exactly the one that fails. It is.

Held fixed: discopt at `bce881ff`, unmodified, `cargo test -p discopt-core --lib
bchoco06`. Only feral varies, via `[patch.crates-io] feral = { path = … }`. Each
arm patches `LuParams::default()`'s `dense_bump_max_dim` and redirects
`SparseLuSymbolic::analyze` to the triangularizing path.

| feral rev | ordering | cap | `bchoco06_..._recovers_bound_649` | dense-bump firings |
|---|---|---|---|---|
| `e00aa70` | whole-basis AMD | 0 | **ok** | 0 |
| `e00aa70` | peel | 0 | **FAILED** — `Numerical` | 0 |
| `e00aa70` | peel | 4096 | **FAILED** — `Numerical` | **26** |
| `895ef65` | peel | 4096 | **FAILED** — `Numerical` | **26** |

All three failures are the same assertion — the test's ground truth, not its
subject:

```
assertion `left == right` failed: unscaled cold solve of the bchoco06 root LP must be Optimal
  left: Numerical
 right: Optimal
```

**The arm is not vacuous.** Both routes are silent fallbacks, so a test that only
checks the answer passes whether or not the new code ran. A counter patched into
`sparse_factor.rs` immediately after `want_dense_bump` is computed —

```rust
if want_dense_bump { eprintln!("PROBE_DENSE_BUMP bump_dim={}", bump_dim); }
```

— fires **26 times** in both cap-4096 arms and **0 times** in both cap-0 arms.
The failing configuration is one where the dense route demonstrably ran, and a
patch that silently failed to apply would have made the cap-4096 arm identical to
peel-no-cap, which it is not.

Row 4 is the same claim at the commit it was authored on, so this is not a
regression introduced by anything that landed after it. Why the original run
reported `PASSED` is undetermined; vacuity is ruled out, and stale build
artifacts or an arm mix-up are the two candidates that cannot be distinguished
after the fact. The measurement was recorded without a firing counter, which is
what made it unfalsifiable at the time — the counter above is the practice that
should have been in place.

**What follows.** `sparse_factor.rs` makes the peel a hard precondition for the
route:

```rust
let want_dense_bump = symbolic.triangularized
    && (peeled || bump_dim <= params.dense_threshold)
    && ...
```

so there is no cap-without-peel configuration to retreat to. The 1.71x and the
bchoco06 bound loss are the same lever, and #163's original coupling argument
stands unchanged: on this LP, opting into the speedup means opting into the lost
bound. Nothing downstream is exposed — `dense_bump_max_dim` defaults to `0` and
`analyze` is whole-basis AMD, so both halves of the lever are off unless a caller
asks — but a caller who does ask should not expect the dense route to recover the
bound, and the doc comment on `LuParams::dense_bump_max_dim` now says so.

**What this retracts.** The paragraph that used to follow this table argued that
because the failing configuration was the *middle* of three, "no ordering-quality
story survives that", and the result was coin-flip sensitivity to any
perturbation of the bits the ratio test reads. That argument was built entirely
on the third row. With the corrected table there is no middle: both peel arms
fail and only whole-basis AMD passes, and the failure tracks the ordering. That
is *not* a demonstration that whole-basis AMD is the better trajectory in general
— #166 has QPLIB_3225 apparently going the other way, and the caveat two sections
above still holds — but the specific "reordering the bump again lands back on a
certifying path" claim is withdrawn.

## Regression coverage

`tests/lu_default_ordering.rs` — a *contract* test, since the measurement above
says there is no numerical difference to assert:

- `analyze` must not triangularize and must equal `analyze_amd_only` exactly,
  with `analyze_triangularized` on the same fixture as the non-vacuity witness;
- `dense_bump_max_dim` must be inert under the default ordering and must fire
  under `analyze_triangularized`, pinning the two as a pair.

Both fail with their intended messages if `analyze` is flipped back to the peel.
End-to-end behavior stays pinned downstream, by the discopt test that reported it.

## Reproducing the dump

Not in the tree — restore it temporarily at the top of
`SparseLuSymbolic::analyze` (and/or `analyze_triangularized`):

```rust
if let Ok(dir) = std::env::var("FERAL_DUMP_BASES") {
    use std::io::Write;
    use std::sync::atomic::{AtomicUsize, Ordering};
    static N: AtomicUsize = AtomicUsize::new(0);
    let k = N.fetch_add(1, Ordering::Relaxed);
    if let Ok(f) = std::fs::File::create(format!("{dir}/basis_{k:06}.mtx")) {
        let mut w = std::io::BufWriter::new(f);
        let _ = writeln!(w, "%%MatrixMarket matrix coordinate real general");
        let _ = writeln!(w, "{} {} {}", a.m, a.col_ptr.len() - 1, a.nnz());
        for j in 0..a.col_ptr.len() - 1 {
            for idx in a.col_ptr[j]..a.col_ptr[j + 1] {
                let _ = writeln!(
                    w, "{} {} {:.17e}", a.row_idx[idx] + 1, j + 1, a.values[idx]
                );
            }
        }
    }
}
```

Then, from a discopt checkout with `[patch.crates-io] feral = { path = ... }`:

```
FERAL_DUMP_BASES=<dir> cargo test -p discopt-core --release \
    bchoco06_illcond_scaled_path_recovers_bound_649
cargo run --release --example probe_illcond_ordering -- <dir>
```

The worst basis of that set is carried in-tree as
`tests/data/lu_bases/bchoco06_illcond_basis.mtx`, so the scorer runs on
`tests/data/lu_bases` with no discopt checkout at all.

---

# Correction: the peel's payoff is in the *symbolic* step, which I never measured

Date: 2026-08-13, after the maintainer's review of PR #162.

Everything above about the peel's cost-benefit is wrong in one specific,
consequential way, and the corrected numbers change the argument (though not the
decision).

## What I measured, and what I should have

The whole "Result: QPLIB_1157" table at the top of this note is **numeric
factorization only**. `examples/lu_fill_orderings.rs` times `SparseLu::factor` and
reports ns per factor-nonzero; the symbolic step is not in it. I then wrote "the
peel is roughly break-even on speed" and later "no standalone payoff" from that
column alone.

`examples/basis_refactor.rs` — in this repository, and used earlier in the same
session — times *both* phases separately. Running it on all three in-tree fixtures
(20 reps, release):

| basis | m | nnz | symbolic `analyze` | symbolic peel | ratio | total ratio |
|---|---|---|---|---|---|---|
| QPLIB_1157 | 3937 | 29376 | 21.280 ± 0.895 ms | 2.166 ± 0.176 ms | **9.83x** | 1.03x |
| QPLIB_3852 | 1760 | 4003 | 0.857 ± 0.083 ms | 0.130 ± 0.018 ms | **6.61x** | **2.73x** |
| bchoco06 | 833 | 2404 | 0.536 ± 0.044 ms | 0.127 ± 0.014 ms | **4.22x** | **2.26x** |

This is not subtle and it is not new information. `CHANGELOG.md`'s #160 entry —
which I edited in this session — already said the peel cuts the ordering "from
9.837 ± 0.295 ms to 0.851 ± 0.037 ms". That is the same 9.8x.

## Why it matters more than the numeric column

`analyze` is re-run on **every refactorization**, so its cost multiplies by the
refactorization count instead of amortizing. The maintainer's end-to-end run, 14
QPLIB relaxations, four arms in one binary:

| arm | geomean | median | max |
|---|---|---|---|
| `api` (sparse-rhs entry points) | 1.067x | 1.090x | 1.363x |
| `tri` (constructor switch only) | **1.306x** | 1.469x | 1.674x |
| `all` | **1.497x** | 1.729x | 2.584x |

On QPLIB_3775: `LuSymbolic` 1048.5 ms across 64 factorizations against
`LuNumeric` 184.6 ms — 5.7x the numeric factorization spent choosing a column
order, 64 times over.

## Two compounding errors, and the pattern

1. **Measured one phase and reported it as the total.** The harness that reports
   both was already written and already used.
2. **Generalized from the fixture with the smallest effect.** QPLIB_1157's total
   ratio is 1.03x; I quoted it as "1.04x" and stopped. The other two fixtures are
   2.73x and 2.26x and were sitting in the same directory.

That is the fourth instance of the same failure mode in one session — asserting a
mechanism or a magnitude that fits the observation without testing it — and the
worst of the four, because the disconfirming evidence was in a file I was editing.

## The decision survives, on a different argument

Not "the peel is free to give up" — it is not — but "neither ordering dominates,
so the caller must choose". Against the speedup:

- issue #163: the peel is a different rounding trajectory and it cost an
  ill-conditioned LP its dual bound;
- QPLIB_2055 under `tri` is **0.389x**, a 2.6x slowdown, objective moving in the
  9th significant figure — a longer pivot path, not a slower kernel.

`analyze` stays whole-basis AMD because it is the trajectory the downstream suite
was green against. The rustdoc now states the cost of that choice next to the
accuracy argument, in both directions, which is what the review asked for.

## Still open

- Make the ordering a parameter with a documented default rather than two
  separately named constructors, so it is A/B-able without a code change. Filed
  as a follow-up; it is an API-shape call.
- QPLIB_3225 solves under `main` but fails under the #162 branch in all three
  solve arms, which points at the #163 ordering revert. Not isolated under a
  controlled arm=main/arm=branch rerun. Filed.
