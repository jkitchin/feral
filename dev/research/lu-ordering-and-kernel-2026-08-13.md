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
