# Amalgamation after the 0.15.0 kernel work: `nemin` re-sweep and a cost-model merge rule

Post-release queue item 2 (`dev/plans/release-0.15.0-checklist.md` §4)
was "amalgamation `nemin` sweep — 90% of clnlbeam's supernodes are ≤8
columns". This note reports the re-measurement and concludes that the
item's premise points the wrong way: **raising `nemin` is already
falsified on that exact family, and the measured win is in the opposite
direction.** It then identifies what is actually wrong with the merge
rule and proposes a fix.

## 1. Why re-measure something already in tried-and-rejected

`nemin` has been swept three times (2026-04-19, 2026-05-03 Phase A and
Phase B, 2026-05-16 issue #10 lever 5). The most recent sweep targeted
clnlbeam by name and found `nemin ∈ {32, 64, 128}` at geomean
1.032 / 1.356 / 1.989 × the `nemin=16` baseline, with `factor_nnz`
inflating 1.23–1.33×. `nemin ∈ {256, MAX}` hung. Issue #10 closed as
"hardware floor".

That rejection rests on one sentence: *"chain-link merges blow trailing
fill faster than the wider panel can amortize."* The amortization rate
is exactly what 0.15.0 changed — the trailing update went from an
SSE2-baseline scalar tile walk to explicit AVX2/NEON (n=2955:
4.24 s → 1.52 s x86; 2.25–3.19× on NEON), and per-supernode fixed
overhead fell twice over (pack-buffer pooling, env-read caching:
−5.7% / −9.0%). Both terms of the trade moved, in opposite directions.
Re-measuring is warranted; re-litigating the conclusion without new
measurement would not be.

**Pre-registered criterion, fixed before the first run:** a `nemin`
default change ships only if it wins ≥5% geomean with a ≥8/10 paired
sign test on ≥2 fixture classes and regresses no fixture by >2%.

## 2. Method

`crates/feral-diagnostics/src/bin/diag_nemin_post_simd.rs`. Paired
alternating A/B per `dev/decisions.md` 2026-08-09: every arm is timed
once per pair, in order, so drift hits all arms equally; `min_us` per
arm; sign test over pairs. Symbolic is computed once per arm and only
the numeric phase is timed, isolating the lever. Inertia and the true
∞-norm relative residual are reported per arm — `nemin` changes
numerics, so byte-parity does not apply.

Fixtures: 61 CUTEst KKT snapshots in `tests/data/parity/**` (n = 3 …
5314), plus four structured KKTs built in earlier sessions
(clnlbeam_like n=100000, grid250 n=62500, sparseqp_kkt n=26000,
chain12000_kkt n=12000). 10–15 pairs each. x86_64, 4 cores, AVX2.

## 3. Result

Geomean vs `nemin = 16`:

| arm | 61 parity fixtures |        | 4 structured KKTs |        |
|-----|--------------------|--------|-------------------|--------|
|     | time               | nnz    | time              | nnz    |
| 1   | 1.21               | 0.67   | 1.330             | 0.373  |
| 4   | 1.02               | 0.83   | 0.986             | 0.506  |
| 8   | **0.986**          | 0.89   | **0.925**         | 0.678  |
| 16  | 1.000              | 1.000  | 1.000             | 1.000  |
| 32  | 1.02               | 1.19   | 1.125             | 1.600  |
| 64  | 1.15               | 1.65   | 1.533             | 2.742  |

Three things are settled by this table.

1. **The queue item's direction is dead.** Every arm above 16 loses on
   time and inflates fill, on every fixture class. The 2026-05-16
   rejection survives the kernel rewrite intact — the faster kernel did
   not buy back the fill. Reported here as a confirmation, not a
   discovery.
2. **`nemin = 8` is better than the shipped default on both axes**, and
   most clearly where it matters: clnlbeam_like 0.925× time at 0.579×
   fill, chain12000 0.815× at 0.640×, sparseqp 0.950× at 0.676×. Inertia
   is identical across all arms on all fixtures; residuals are equal or
   better.
3. **It does not clear the pre-registered bar.** The parity geomean is
   1.4%, not 5%, and individual fixtures regress well past 2%:
   CERI651A_0000 1.163 (2/15 wins, 178 → 207 µs — a real effect, not
   noise), DEGENLPB_0046 1.119, BQPGASIM_0012 1.100, HS85_0176 1.056.
   By the criterion set before the run, **the default does not change.**

The criterion was written as a speed criterion and is therefore
under-specified: it has nothing to say about the uniform 10–42% fill
reduction, which is a deterministic symbolic quantity — identical on
every platform, and not subject to the timing noise that has twice
produced wrong conclusions on this container. That is a real result on
an axis the criterion did not cover, and it is the reason the next
section exists rather than the note ending here.

## 4. What is actually wrong: the merge rule has no fill guard

`find_supernodes` (`src/symbolic/supernode.rs:340-374`) accepts a merge
on either of two conditions:

```rust
let trivial_chain = parent_ncol == 1 && col_counts[p_first] + 1 == col_counts[child_last];
let size_based    = child_ncol < params.nemin && parent_ncol < params.nemin;
```

`trivial_chain` is the structurally free merge: the row patterns already
match, so it costs nothing. `size_based` asks only whether two
supernodes are *small*. It never asks what the merge costs. The whole
`nemin` sweep is therefore a search for one global number that stands in
for a per-merge decision, which is why every sweep finds a different
optimum per matrix family and why no single value has ever satisfied the
corpus.

The cost is exactly computable at the point of decision. A supernode's
front is `nrow = col_counts[first_col].max(ncol)`
(`supernode.rs:399`), so merging child into parent gives
`nrow_m = col_counts[s_first].max(ncol_c + ncol_p)`. The
`.max(ncol)` is where the damage happens: once the merged width exceeds
the natural row count, every additional column adds a triangle of pure
fill that no pattern justifies.

clnlbeam is the extreme case. Its chain links have `col_counts = 2`. At
`nemin = 16`, sixteen 1-column fronts of `nrow = 2` (LDLᵀ ≈ 4 flops
each, 64 total) merge into one `ncol = 16, nrow = 16` front:
`Σ_{j=1}^{16} j² = 1496` flops, a 23× inflation, and the measured
`factor_nnz` per column moves 2.0 → 9.5, matching the model's 4.25×
prediction to within the model's own crudeness. The size rule cannot see
any of this; it sees only `1 < 16` and `1 < 16`.

But pure fill minimisation is equally wrong: `nemin = 1` has the lowest
fill of every arm (0.373× on the structured set) and is the **slowest**
(1.330×), because 99999 one-column fronts pay 99999 lots of per-front
overhead. The objective is neither fill nor front count. It is time, and
time is roughly

```
  T  ≈  τ · flops  +  Ω · n_fronts
```

so a merge is worth taking exactly when the flops it adds cost less than
the one front's overhead it removes:

```
  Δflops · τ  <  Ω        ⟺        Δflops  <  Ω/τ  ≡  MERGE_FLOP_BUDGET
```

with `flops(ncol, nrow) = Σ_{k=0}^{ncol-1} (nrow-k)² = S(nrow) - S(nrow-ncol)`,
`S(x) = x(x+1)(2x+1)/6`.

`Ω/τ` is a single number with a physical meaning — the flop-equivalent
of one front's fixed overhead — and it is a *hardware* constant, not a
per-matrix tuning parameter. That is the substantive difference from
`nemin`: one calibration should serve the corpus, and when the kernel
gets faster (τ falls) or the per-front overhead falls (Ω), the rule
adapts by recalibration rather than by re-deriving a shape heuristic.
It also subsumes `trivial_chain`, whose Δflops is ≈ 0 by construction.

## 5. Measured: the cost model works, and then fails on accuracy

Implemented as `SupernodeParams::merge_flop_budget: Option<u128>`
(`None` = today's rule exactly). Budget sweep, same harness.

Structured KKTs, geomean vs the shipped default:

| budget | time  | nnz   |
|--------|-------|-------|
| 5      | 1.018 | 0.492 |
| 15     | 0.967 | 0.507 |
| 30     | 0.945 | 0.549 |
| 60     | 0.935 | 0.619 |
| 125    | 0.960 | 0.723 |
| 250    | 0.970 | 0.827 |
| 500    | 0.986 | 0.919 |
| 1000+  | 1.003 | 1.000 |

A clean interior optimum at 30–60, and it dominates the `nemin` sweep on
the axis that motivated it: on sparseqp, budget 30 holds fill at 0.493×
for the same 0.947× time that `nemin = 8` bought at 0.676× fill. The
per-front adaptivity is doing real work. Above ~500 the guard becomes a
no-op — each incremental merge is individually cheap even when the chain
of merges is ruinous, which is why the first probe (1000 … 1e6) was
indistinguishable from `off` and why the useful range is two orders of
magnitude lower than a naive Ω/τ estimate suggests.

On the 61 parity fixtures: 0.980–0.997× time, 0.866–0.940× fill. No
speed regression of the kind that sank `nemin = 8`.

**And then the residuals.** Reported per arm by the harness (single
unrefined `Solver::solve`, `b = 1`, so the ∞-norm residual is also the
relative one). Degradations >10× at budget ∈ {15, 30, 60, 125}:

| fixture | default | guarded | factor |
|---------|---------|---------|--------|
| HATFLDG_0003..0006 | 7.1e-15 | up to 7.4e-08 | up to 9e6× |
| VESUVIOU_0030 | 1.9e-06 | 5.4e-03 | 2789× |
| MEYER3NE_0220 | 2.7e-08 | 8.0e-07 | 30× |

Seven digits on HATFLDG, from machine precision to 1e-8. Inertia is
unchanged everywhere, and on HATFLDG/VESUVIOU the fill is *identical* to
the default — so this is not a fill effect. Thinner supernodes give
Bunch–Kaufman a smaller candidate pool inside each front, and on
ill-conditioned fixtures it takes worse pivots.

The same defect is in the `nemin` lever, which is what makes this a
property of the direction rather than of this particular rule:
`nemin = 8` degrades VESUVIOU_0030 by the same 2789× and MEYER3NE_0220
by 16×; `nemin = 4` degrades MEYER3NE_0220 by 83×.

**Conclusion: neither lever ships.** "Correctness before performance,
always" is a hard constraint in `CLAUDE.md`, and 2–7% of factor time plus
11–45% of fill does not buy seven digits of residual. The direction is
closed for both the global constant and the cost model — not because the
speed and memory wins are absent, but because they are paid for on an
axis neither the queue item nor my own pre-registered criterion thought
to check.

## 6. Plan

`merge_flop_budget` stays in the tree as an opt-in knob, defaulting to
`None` = today's rule exactly, so the default path is bit-identical and
no corpus gate is needed to carry it. It is the apparatus that produced
§5 and lets anyone reproduce it. It is **not recommended for use**: the
doc comment carries the accuracy result, and flipping the default would
need the full corpus run (inertia 100%, Phase 2.8.1 partitions, residual
envelope) that this container does not have — and would have to answer
the residual question first, which no amount of corpus timing does.

Nothing else here ships. Post-release queue item 2 is closed.

## 7. What this redirects to

Published after this work was done, `pounce#552`'s re-measurement against
a released 0.15.0 moves the target
([comment 5232409020](https://github.com/jkitchin/pounce/issues/552#issuecomment-5232409020)):

| matrix | 0.15.0 | MA57 | ratio was | ratio now |
|--------|-------:|-----:|----------:|----------:|
| dtoc1nd | 13.67 | 3.63 | 4.85× | **3.77×** |
| clnlbeam | 25.74 | 7.27 | 8.05× | 3.54× |
| marine_1600 | 35.50 | 13.50 | 3.05× | 2.63× |
| rocket_12800 | 19.11 | 8.79 | 2.87× | 2.17× |
| steering_12800 | 27.44 | 18.96 | 2.53× | 1.45× |
| dtoc2 | 109.01 | 84.21 | 1.70× | 1.29× |

clnlbeam — the matrix this whole queue item was aimed at, and the
nominated upstream target at 8.05× — is more than halved and **is no
longer the worst case**. `dtoc1nd` is, and it is a *dense-front* matrix:
nnz/dim = 23.0, fronts of 33–64 columns. The ratio ordering is no longer
the nnz/dim ordering, so "the gap tracks sparsity" is dead as a guiding
intuition.

Amalgamation was a chain-KKT lever aimed at a chain-KKT problem that has
substantially receded. The medium-front regime (33–64 columns) is where
the worst remaining ratio lives, and it is a different mechanism — one
the 0.15.0 packed kernel touches only above `PACKED_SIMD_MIN_WORK`. That
is the next target, not this one.

Note also that the other item pounce names as upstream — warm-path
scaling, "57–81% of the warm prologue" — was falsified earlier the same
day: the ∞-norm equilibration converges linearly at ratio ½ and never
reaches its tolerance, so the iteration count is fixed at the cap and no
warm start can reduce it
(`dev/research/scaling-warm-start-2026-08-09.md`). Cheapening it is a
conditioning trade, not a free win.
