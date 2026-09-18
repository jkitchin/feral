# Issue #203 — why fill grows superlinearly on long-horizon collocation KKTs

**Date:** 2026-09-17
**Machine:** Apple M2, `Mac14,2`, macOS. CoinHSL v2023.11.17 (MA57).
**Status:** diagnosis complete; no code change made.

Issue #203 (filed from pounce#947) reports that on a transient gas-network
optimal-control KKT, `dim K` and `nnz(K)` are exactly linear in the horizon
length `nfe` while symbolic fill grows as `nfe^1.37` and factorization time as
`nfe^2.3`. It offers a hypothesis — the matrix is a block chain, a front-to-back
elimination order would give linear fill, and none of AMD / METIS / SCOTCH /
KaHIP is finding it — and asks three questions:

1. why the orderings do not recover near-linear fill;
2. whether `OrderingMethod::External` plus a hand-built chain permutation does;
3. whether a chain-detection heuristic belongs in `Auto`.

Plus a fourth, from the original report: why the same systems go 1.2–1.9x faster
per iteration through IPOPT + MA57.

**Answers, in one paragraph.** The matrix is not a chain of small blocks; it is a
2-D product of a sparse 775-node spatial network with a time path of `3*nfe`
points, and its spatial extent exceeds its temporal extent at every size
measured. The chain permutation does produce exactly linear fill and is
90x worse than the default in `nnz_L` and 3254x worse in flops at nfe=6, so
question 2 is answered with the opposite sign to the one expected, and a
chain-detection heuristic (question 3) would be actively harmful. But the
superlinearity *is* a real ordering deficit: MA57's bundled real METIS, replayed
through feral's own symbolic pipeline, needs 1.70x fewer flops than feral's best
at nfe=48 and 2.89x fewer at nfe=96, with the gap widening as the horizon grows.
That gap alone covers the MA57 wall-clock advantage in question 4 without
invoking any kernel difference.

## The structure, read off the reproducer

pounce#947 ships a self-contained numpy reproducer that rebuilds the exact
Jacobian and Hessian patterns. It regenerates the reported geometry exactly:

| nfe | dim K | nnz(K) lower |
|---|---|---|
| 6 | 28,086 | 156,858 |
| 12 | 56,166 | 317,364 |
| 24 | 112,326 | 638,376 |
| 48 | 224,646 | 1,280,400 |
| 96 | 449,286 | 2,564,448 |

Its index arithmetic gives the graph:

* `NS = 775` states per collocation point, `NCP = 3` points per finite element,
  so `npt = 3*nfe` points and `nz = npt*NS` state variables;
* `NC = 6` controls with one knot per element (`nk = nfe+1`), `NE = 6` path
  constraints per point, `NC*(nk-1)` ramp constraints;
* the embedded spatial coupling `S` is **775 x 775 with 1977 nonzeros** — 2.55
  per row. That is a near-tree network (GasLib-40, 10 finite volumes per pipe),
  not a dense block.

So `K` is a **2-D product graph**: a sparse spatial network of 775 nodes crossed
with a time path of `3*nfe` points, with constraint rows roughly doubling each
slice. The decisive number is the aspect ratio. Spatial extent is fixed at 775;
temporal extent is 18 at nfe=6, 288 at nfe=96, 648 at nfe=216. **Space is wider
than time everywhere in the reported sweep**, so a time-slice separator (775
states, ~1550 rows with the constraints) is the *expensive* cut, not the cheap
one, and cutting along time repeatedly is the worst thing an ordering can do.

This is why "`nnz(K)` is exactly linear, so fill should be too" does not follow.
The same is true of a 2-D mesh refined in one direction: the input stays linear
while nested-dissection fill goes as `n log n` and flops as `n^1.5`, and neither
is a defect. Linear fill becomes available only once the time axis is the *long*
axis, i.e. `3*nfe >> 775`, or `nfe >> 258` — just past the largest size in the
report.

## Question 2 — the chain permutation, measured

`gen_perm.py` builds exactly the ordering the issue asks for: every row and
column of `K` sorted by its time key (state variables and their residual rows by
collocation point, path constraints by point, controls and ramp constraints at
their knot's point), in two within-slab variants — all-variables-then-all-
constraints, and interleaved variable/constraint pairs. It is fed to feral as
`OrderingMethod::External`, so the only thing that differs from a default run is
the permutation.

| nfe | Auto `nnz_L` | chain `nnz_L` | ratio | Auto flops | chain flops | ratio |
|---|---|---|---|---|---|---|
| 6 | 0.90M | 80.5M | 90x | 5.36e7 | 1.745e11 | 3254x |
| 12 | 2.37M | 165.1M | 70x | 2.97e8 | 3.609e11 | 1214x |
| 24 | 6.45M | 334.3M | 52x | 1.83e9 | 7.336e11 | 401x |
| 48 | 17.6M | 672.7M | 38x | 1.09e10 | 1.479e12 | 135x |

The chain ordering **is exactly linear** — 2.01x per doubling in fill, 2.02x in
flops. The issue's arithmetic is right. The constant is what kills it:
`front_max` pins at 3299 at every size, the width of one element's slab, because
every one of the `3*nfe` cuts densifies a 775-state interface. The interleaved
variant changes nothing (79.5M vs 80.5M at nfe=6), so this is not a tie-breaking
detail. Extrapolating both curves, the crossover is past `nfe ~ 1000`.

**Question 3 follows directly.** A chain / near-banded detector in `Auto`
dispatch would fire on this pattern — it is exactly the shape such a detector
looks for — and give up two orders of magnitude. It should not be built.

## Questions 1 and 4 — where the real gap is

If the chain order is not the answer, is a better ordering available at all?
`ma57_analyse.F` runs MA57ID + MA57AD (analysis only, pattern only, no numeric
factorization) and reports `INFO(5)` reals-in-factors, `INFO(7)` max front and
`RINFO(2)` elimination flops. MA57's automatic choice (`ICNTL(6)=5`) takes AMD at
nfe<=12 and switches to its bundled **real METIS** at nfe>=24 — and real METIS is
far ahead of every feral backend at scale. At nfe=96:

| ordering | `nnz_L` / reals | flops | max front |
|---|---|---|---|
| feral `Amd` | 53.7M | 5.61e10 | 3136 |
| feral `Amf` (= `Auto`) | 46.4M | 5.06e10 | 3453 |
| feral `MetisND` | 69.5M | 6.89e10 | 3204 |
| feral `ScotchND` | 75.5M | 8.22e10 | 3215 |
| MA57 + real METIS | 33.3M | 1.80e10 | 1605 |

To rule out storage conventions and kernel differences, MA57's pivot order
(`KEEP(1:N)` after MA57AD) was dumped, inverted to feral's new-to-old
convention, and replayed through feral's *own* symbolic pipeline as `External`.
Same amalgamation, same column counts, same flop metric — only the permutation
differs:

| nfe | feral `Auto` flops | MA57-METIS perm through feral | ratio |
|---|---|---|---|
| 6 | 5.36e7 | 8.99e7 | 0.60 |
| 12 | 2.97e8 | 4.18e8 | 0.71 |
| 24 | 1.83e9 | 1.72e9 | 1.06 |
| 48 | 1.09e10 | 6.42e9 | **1.70** |
| 96 | 5.06e10 | 1.75e10 | **2.89** |

The replayed numbers track MA57's own forecast closely (6.42e9 vs 6.68e9 at
nfe=48; 1.75e10 vs 1.80e10 at nfe=96), which validates the cross-solver metric
and confirms that nothing but the permutation is being compared.

Growth per doubling of `nfe`:

* feral `Auto` flops: 5.5x, 6.2x, 6.0x, 4.6x — exponent 2.2–2.6, flat;
* real METIS flops: 4.7x, 4.1x, 3.7x, 2.7x — exponent falling toward ~1.4.

**Real METIS is bending toward the near-linear asymptote the issue expects, and
feral's backends are not.** `front_max` says the same thing in one number: real
METIS's top separator is 1605, about one time slice of states plus constraints;
feral's best is 3453, about twice that. Feral's own `MetisND` is 3.9x worse in
flops than real METIS at nfe=96 despite being the same algorithm family, so the
deficit is in `feral-metis`'s separator quality, not in the choice of backend —
`Auto` is right to prefer `Amf` over the ND backends *as they currently stand*.

This also answers question 4. At the sizes pounce runs (nfe=48–216) the ordering
gap alone is 1.7x–2.9x in flops, which covers the reported 1.2–1.9x wall-clock
advantage of IPOPT + MA57 without any appeal to multifrontal-vs-other or to
kernel speed. That is a different conclusion from
`dev/research/chain-kkt-ma57-gap-2026-08-09.md`, which measured a numeric-
factorization gap on *proxy* chains where the orderings were comparable; both
can be true, and on these matrices ordering dominates.

## What this suggests, in priority order

1. **Close the `feral-metis` separator-quality gap.** It is the whole of the
   reported superlinearity, it is measurable without pounce in the loop, and the
   target is concrete: match real METIS's `front_max` of 1605 at nfe=96. The
   suspects to check first are graph compression before bisection, the number of
   initial-bisection trials, and two-sided vs one-sided FM separator refinement.
2. **Nothing chain-shaped.** Do not build the chain detector; see
   `dev/tried-and-rejected.md` (2026-09-17).
3. **The `Auto` dispatch is not the problem** and should not be retuned until
   (1) lands: it currently picks the best of the five backends available to it at
   every size measured.

## Reproduction

```sh
# pattern generator: the "kkt_pattern_repro.py" details block in pounce#947
python3 kkt_pattern_repro.py mtx 6 12 24 48 96
python3 gen_perm.py perms 6 12 24 48        # chain permutations

cargo build --release -p feral-diagnostics --bin issue203_fill_probe
./target/release/issue203_fill_probe mtx/gaslib40T_nseg10_nfe048.mtx \
    perms/perm_blocked_nfe048.txt

# MA57 side (needs the CoinHSL bundle; see external_benchmarks/ma57_oracle)
gfortran -O2 -ffixed-line-length-none -fallow-argument-mismatch \
    -o ma57_analyse ma57_analyse.F -L$HSL/lib -Wl,-rpath,$HSL/lib \
    -lhsl -lopenblas -lm
./ma57_analyse mtx/gaslib40T_nseg10_nfe048.mtx 4 keep048.txt
# invert KEEP to feral's new-to-old convention, then replay:
./target/release/issue203_fill_probe mtx/gaslib40T_nseg10_nfe048.mtx \
    perms/perm_ma57metisinv_nfe048.txt
```

`gen_perm.py` and `ma57_analyse.F` are small and are reproduced in the session
scratch; `issue203_fill_probe.rs` is committed under
`crates/feral-diagnostics/src/bin/`.

## Caveats

* Everything here is **symbolic**. `sum_j (c_j - 1)^2` is the standard dense
  column-update flop model and it tracks factorization time, but no numeric
  factorization was timed. The predicted 1.7x–2.9x speedup from a better
  ordering is a forecast, not a measurement.
* The reproducer writes unit values (diagonal 4, off-diagonal 1). That is correct
  for symbolic analysis and says nothing about pivoting: a real run may delay
  pivots and exceed these counts.
* `nfe=216` was not run. The trend at 96 is already unambiguous and 216 costs
  ~50 MB of matrix and minutes of ND per backend; it is the obvious next
  confirmation if the ordering work goes ahead.
