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
