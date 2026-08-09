# The factorization gap after 0.15.0, and a chain regression it exposed

**Date:** 2026-08-09
**Machine:** Apple M2, `Mac14,2`, 4 performance + 4 efficiency cores, macOS.
**Status:** PARTIALLY SUPERSEDED by `chain-kkt-corpus-2026-08-09.md`.
Result 1 of this note stands. Results 2 and 3 are superseded.

> **Result 1 of this note was right.** An earlier revision of this
> banner said feral was 8x-23x *faster* than MA57 on chain KKTs. That
> came from an oracle running `ICNTL(15) = 1`, which computes MC64
> scaling inside the timed `MA57BD` region while feral's MC64 sits in
> untimed analysis. Corrected (`ICNTL(15) = 0`, 15 pairs), MA57 is
> faster on all six real matrices by 1.59x to 5.53x — the same
> direction, and close to the magnitude, this note reported from the
> proxies (1.4x-2.4x). The proxies were vindicated by the real corpus,
> not contradicted by it.
>
> **What is genuinely superseded:**
>
> * the regression is **not** on the large chains — it is on `dtoc1nd`
>   (n = 9,685), where #150 costs 25%. On the four largest matrices #150
>   *gains* 1.20x to 2.05x.
> * the efficiency-core hypothesis in "Result 3" is **refuted**.
>   Thread count moves nothing on the real matrices, on hardware with
>   more efficiency cores than the machine the hypothesis came from.
>
> So the proxies got the gap right and the regression's location
> exactly backwards.

## Why

0.15.0 was cut specifically so the pounce#552 factorization comparison
could be re-taken: the 3.5-4.8x gap in that report describes the
pre-SIMD kernel. Nobody knew the current gap, which made every remaining
optimization item a bet at unknown odds.

The five #552 models are Pyomo NMPC problems needing idaes/prommis, and
this machine has neither them nor `data/matrices/`. What it does have is
the CoinHSL v2023.11.17 bundle, so MA57 is available as an oracle. The
substitute is block-tridiagonal KKT proxies at the reported geometry:
see `external_benchmarks/chain_proxy/README.md` for the construction,
the analytic inertia oracle, and the limits.

## Method

Paired alternating A/B per `decisions.md` (2026-08-09): every arm timed
once per pair, `min` over 15 pairs, exact two-sided sign test. All arms
read the same manifest, matrices and synthesized RHS.

Correctness first: both feral and MA57 reproduce the analytic inertia
(`T*nx` positive, `T*nc` negative, zero zeros) exactly on all five
proxies, feral residuals 3.7e-16 to 2.3e-15, MA57 2.0e-16 to 2.4e-16.
The matrices are sound; the timings are on well-conditioned,
correctly-solved systems.

**Arm naming caveat:** the "new" arm throughout is `main` at `7a31ff6`,
which is three commits past the `v0.15.0` tag (it includes the
amalgamation guard from #155, opt-in and documented as bit-identical by
default). A worktree at the exact `v0.15.0` tag was built but the
bisect that would separate tag from main **did not run**. Everything
below therefore says "main", not "0.15.0", and attributing the
regression to a specific PR is still open.

**Main has since moved, in a way that matters here.** #156 (issue #154,
`4f2fad6`) landed after these runs: `Solver::with_params` no longer
hardcodes `use_parallel: true` but derives it from
`available_parallelism() > 1`, and all four dispatch sites now fall back
to the sequential path when pool construction fails. That changes which
driver a default-constructed solver takes, which is exactly the axis the
regression sits on. `7a31ff6` (measured) predates it; `f7a152a` does
not. The bisect below must therefore treat current main as its own
point rather than assuming it behaves like the measured `7a31ff6`.

## Result 1 — the gap against MA57

`min factor_us` over 15 pairs; ratio > 1 means feral is faster.

| proxy | n | main | ma57 | ratio | wins | p |
|---|---|---|---|---|---|---|
| hicks_like | 1,806 | 1,082 | 603 | 0.557 | 2/15 | 0.0074 |
| cart_pole_like | 2,709 | 1,590 | 1,081 | 0.680 | 1/15 | 0.0010 |
| quad_tank_like | 3,010 | 1,635 | 968 | 0.592 | 0/15 | 0.0001 |
| double_column_like | 25,389 | 36,150 | 26,210 | 0.725 | 5/15 | 0.3018 |
| prommis_sx_like | 28,303 | 56,519 | 23,470 | 0.415 | 0/15 | 0.0001 |

**feral is slower than MA57 on all five, by 1.4x to 2.4x.** Four of the
five are significant; `double_column_like` is not (5/15, p = 0.30).

This is narrower than the 3.5-4.8x in pounce#552, but the two numbers
are not comparable: different matrices, different machine, and this one
is numeric factorization only where the report is end-to-end. The honest
statement is that on chain-structured proxies, post-SIMD feral is still
materially behind MA57, and the largest, most chain-like proxy is the
worst case.

## Result 2 — main is slower than 0.14.0 on the two big chains

Same run, validity control. Ratio > 1 means main is faster than 0.14.0.

| proxy | ratio | wins | p |
|---|---|---|---|
| hicks_like | 1.045 | 13/15 | 0.0074 |
| quad_tank_like | 1.026 | 13/15 | 0.0074 |
| cart_pole_like | 1.016 | 9/15 | 0.6072 |
| double_column_like | 0.914 | 4/15 | 0.1185 |
| **prommis_sx_like** | **0.833** | **2/15** | **0.0074** |

The three small ODE-shaped proxies gain a little. **The two large
chain-structured ones lose**, and `prommis_sx_like` loses 20%
significantly. This contradicts the release's "wins all twelve arms",
which was measured on real matrices on a 4-core homogeneous x86_64
container.

It also reproduces something already on the record: session 2026-08-09-02
found the old per-node driver beating the new one by ~20% on a wide-block
chain proxy (`chainW`), could not explain it, and accepted it as a proxy
quirk; the pounce-side reviewer on PR #150 reframed it as a probable
regression on exactly `double_column` / `prommis_sx` geometry. An
independent proxy at the reported geometry, on different hardware, now
shows the same thing. It should stop being treated as a quirk.

## Result 3 — mechanism not yet identified

15 pairs, two large proxies, arms that isolate the two levers 0.15.0
shipped. Ratio > 1 means faster than 0.14.0.

`prommis_sx_like` (0.14.0 = 53,237 us):

| arm | min_us | vs 0.14.0 | wins | p |
|---|---|---|---|---|
| main default | 67,376 | 0.790 | 1/15 | 0.0010 |
| main `PAR_MIN_SEEDS=u64::MAX` | 66,689 | 0.798 | 0/15 | 0.0001 |
| main `RAYON_NUM_THREADS=1` | 73,479 | 0.725 | 1/15 | 0.0010 |
| main `PACKED_SIMD=0` | 72,228 | 0.737 | 0/15 | 0.0001 |

`double_column_like` (0.14.0 = 37,602 us):

| arm | min_us | vs 0.14.0 | wins | p |
|---|---|---|---|---|
| main default | 40,382 | 0.931 | 3/15 | 0.0352 |
| main `PAR_MIN_SEEDS=u64::MAX` | 59,551 | 0.631 | 0/15 | 0.0001 |
| main `RAYON_NUM_THREADS=1` | 66,584 | 0.565 | 0/15 | 0.0001 |
| main `PACKED_SIMD=0` | 45,276 | 0.831 | 0/15 | 0.0001 |

Neither lever explains it. Disabling the packed SIMD kernel makes things
**worse** on both, so the kernel is helping, not hurting. Forcing the
sequential fallback does not recover the loss on `prommis_sx_like` and
is much worse on `double_column_like`.

Two things this does establish:

1. **These chains do not collapse to `seeds = 1`.** `FERAL_DEBUG_TASK_PLAN=1`
   reports `prommis_sx_like: n_snodes=2625 n_tasks=15 seeds=8` and
   `double_column_like: n_snodes=2591 n_tasks=17 seeds=9`. The task
   graph is active on both. The release note's reasoning that
   "chain-shaped trees collapse to a single task and take the sequential
   driver outright" does not hold for this geometry, which is precisely
   the geometry pounce cares about.
2. On `prommis_sx_like`, running the task graph (67,376) is worth almost
   nothing against forcing it off (66,689), while 0.14.0's per-supernode
   spawning got 53,237. Coarsening 2,625 supernodes into 15 unequal
   tasks appears to give up parallelism that the finer-grained driver
   was extracting.

**Untested hypothesis:** on a 4P+4E M2 rayon treats efficiency cores as
equivalent to performance cores, so a coarse task landing on an E-core
stalls the whole factorization, where per-node granularity let
work-stealing rebalance. This predicts the loss shrinks or inverts at
`RAYON_NUM_THREADS=4`. That arm was written (`probe_bisect.py`, not
committed since it never ran) but not executed.

## What to run on the corpus machine

In order. The first two are cheap and decide whether the third matters.

1. **Bisect the regression** across `c05eb77` (0.14.0), `e8e1c5a`
   (#149 SIMD), `fad5670` (#150 parallel), `808babb` (v0.15.0) and
   current `main`, on the real chain KKTs. This attributes it to a PR,
   and separates the v0.15.0 tag from post-tag `main` — which this
   measurement cannot do.
2. **Thread sweep** (1 / 2 / 4 / all) on the same matrices, on
   homogeneous-core hardware and on Apple silicon, to test the E-core
   hypothesis.
3. **Re-run the gap against MA57** on the real corpus. Only that number
   should be quoted to pounce; the proxy ratios above are a hypothesis.

Do not act on the scaling-reuse item until 1 and 2 land. If the
regression is real on the corpus, recovering 20% on the exact matrix
class pounce cares about is both cheaper and better-evidenced than a
design project that changes numerics and needs residual gates.
