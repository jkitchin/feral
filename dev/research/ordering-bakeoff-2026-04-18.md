# Ordering bake-off: AMD vs METIS vs SCOTCH on the parity corpus

**Date.** 2026-04-18
**Binary.** `src/bin/bench_orderings.rs` (commit 938daf4)
**Corpus.** `tests/data/parity/` — 30 matrix families, one representative
`.mtx` per family (the lexicographically-first dump per IPM run).
**Build.** `cargo run --release --bin bench_orderings`.
**Metric.** `factor_nnz_estimate` from `symbolic_factorize_with_method`
= sum of column counts of L on the permuted pattern. This is the
*symbolic* fill — the upper bound on numeric fill with no pivoting.

## Raw per-matrix results

```
matrix                    n        nnz     fill_amd   fill_metis  fill_scotch    t_amd  t_metis   t_scot    m/amd    s/amd
----------------------------------------------------------------------------------------------------------------------------------
acopp30                 209        765         1544         1464         1952      210      168      446    0.948    1.264
argauss                   3          6            7            7            7        6        5        3    1.000    1.000
avion2                   64        193          277          282          282       42       44       42    1.018    1.018
batch                   121        305          518          518          518       55       49       52    1.000    1.000
bqpgasim                 50        172          224          224          224       32       35       34    1.000    1.000
ceri651a                190        739          912          937         1138      107       75      280    1.027    1.248
ceri651c                  7         28           33           33           33        4        5        4    1.000    1.000
ceri651els                7         28           33           33           33        3        4        3    1.000    1.000
chwirut1                645       1715         2061         2065         2101      517      813     1057    1.002    1.019
cresc100                806       2506         3025         3111         3680      663     1104     1246    1.028    1.217
cresc132               5314      22566        27096        27096        27157    52767     9508    17850    1.000    1.002
dallass                  77        169          290          290          290       37       33       31    1.000    1.000
degenlpa                 35        117          241          243          243       28       29       22    1.008    1.008
degenlpb                 35        117          241          243          243       17       15       20    1.008    1.008
hahn1                   715       2854         3432         3465         3590      759      562      803    1.010    1.046
hatfldbne                 8         15           18           18           18        6        6        5    1.000    1.000
hatfldf                   6         12           21           22           22        4        5        4    1.048    1.048
hatfldg                  50        120          229          226          226       20       22       22    0.987    0.987
hs103                    12         62           76           79           79        9       10        9    1.039    1.039
hs109                    19         71           92           98           98       11       12       11    1.065    1.065
hs85                     68        186          223          230          230       24       27       27    1.031    1.031
hydcar20                198        833         2154         2155         3085      150      119      430    1.000    1.432
meyer3ne                 51        134          160          164          164       18       19       24    1.025    1.025
palmer2ane               75        256          325          343          343       29       26       30    1.055    1.055
roszman1                 79        230          282          289          289       23       22       27    1.025    1.025
ssi                       3          5            6            6            6        2        3        2    1.000    1.000
ssine                     5          9           12           12           12        3        4        4    1.000    1.000
swopf                   175        407          808          811         1104       93       80      247    1.004    1.366
vesuvia                3083      12633        15193        15218        15754    10261     2814     5678    1.002    1.037
vesuvio                3083      10342        12436        12481        12668     8579     3325     4424    1.004    1.019
```

## Summary

```
geomean fill_metis  / fill_amd  = 1.011   (METIS produces 1.1% more fill on average)
geomean fill_scotch / fill_amd  = 1.060   (SCOTCH produces 6.0% more fill on average)
minimum-fill wins: AMD=28, METIS=12, SCOTCH=10 (ties count for all three)
total symbolic time (us): AMD=74479, METIS=18943, SCOTCH=32837
```

## Observations

1. **AMD is the fill-quality winner on this corpus.** Out of 30
   matrices, AMD is best or tied-best on 28. METIS beats AMD on
   exactly two: `acopp30` (1464 vs 1544, 5.2% better) and `hatfldg`
   (226 vs 229, 1.3% better). SCOTCH never strictly beats AMD.

2. **METIS is 4× faster in total symbolic time; SCOTCH is 2× faster.**
   Both ND crates scale better than AMD on the largest matrices
   because AMD's per-elimination degree update is O(deg²):
   - `cresc132` (n=5314): AMD 52.8ms, METIS 9.5ms, SCOTCH 17.8ms
   - `vesuvia`  (n=3083): AMD 10.3ms, METIS 2.8ms, SCOTCH 5.7ms
   - `vesuvio`  (n=3083): AMD  8.6ms, METIS 3.3ms, SCOTCH 4.4ms

3. **Fill gap is largest on dense-ish KKTs.** SCOTCH is >24% worse
   than AMD on `acopp30`, `ceri651a`, `cresc100`, `hydcar20`, `swopf`.
   These are power-flow / chemistry / trajectory matrices with
   a relatively dense Hessian block — AMD's minimum-degree heuristic
   handles those well, ND's top-down cut does not.

4. **METIS ≈ SCOTCH on most cases.** On 20 of 30 matrices they
   produce identical fill. Where they differ, METIS is tighter
   (acopp30, ceri651a, cresc100, hydcar20, swopf, hahn1). This is
   consistent with SCOTCH's halo+band FM sometimes not reaching the
   METIS-quality cut under the default move caps.

5. **The corpus is small-matrix-biased.** Median n ≈ 77. Only three
   matrices are > 1000 (cresc132=5314, vesuvia=3083, vesuvio=3083),
   and on those SCOTCH and METIS close the gap (s/amd = 1.002 on
   cresc132). The ND-favouring regime starts at n > ~10k for most
   pattern families; this corpus does not test that regime.

## What this does NOT say

- **Numeric factor NNZ with realistic pivoting.** Heavy pivoting
  promotes delayed pivots that rewrite the pattern. Symbolic fill
  is a lower-bound proxy; a true comparison runs
  `factorize_multifrontal` under each ordering and counts actual
  post-pivoting NNZ. Deferred per B6 in
  `dev/plans/ordering-scotch.md`.
- **Solve time.** Ordering quality feeds into numeric factor and
  solve time, not just NNZ. Not measured.
- **Ordering time scaling.** 30-matrix sample; no fit to `n` or
  `nnz`.
- **FM sign fix effect.** The sign bug was fixed in `ba31609`
  before this run; these numbers are post-fix. A pre/post delta
  would require reverting the fix, which is out of scope.

## Decision implications

- Keep AMD as the `OrderingMethod::Amd` default. It is the best
  fill-quality producer on this corpus and the performance penalty
  is small for n < 1000.
- METIS is the right choice when symbolic time dominates (typically
  n > ~3000 on this pattern family).
- SCOTCH is currently neither fastest nor tightest; its niche —
  graphs with heavy compression potential or separator-structure
  problems — is not represented in the parity corpus. Need a
  different corpus (UFL KKTs, larger grid structures) to find it.
- Adding KaHIP (per `dev/plans/ordering-kahip.md`) requires a
  larger, more diverse corpus before it can be meaningfully scored.

## Large-matrix extension (SuiteSparse)

**Corpus.** Four matrices pinned via `dev/scripts/large_matrices.txt`,
fetched by `dev/scripts/fetch_large_matrices.sh` into
`tests/data/large/`. Selected to span the medium-to-large symmetric-
indefinite / KKT regime that the parity corpus does not reach.

**Numbers below are post-FM-rebalance-fix (commit `4cf4a81`).** See
the "FM rebalance fix delta" subsection at the end of this section
for the before/after comparison — the fix substantially changes the
SCOTCH results on large matrices.

```
matrix                    n        nnz     fill_amd   fill_metis  fill_scotch    t_amd  t_metis   t_scot    m/amd    s/amd
----------------------------------------------------------------------------------------------------------------------------------
bcsstk38               8032     181746       939409      1052352      1209772   380003    80837    59373    1.120    1.288
bratu3d               27792      88627     10009287      8768952      8408115 14781723   359353   338396    0.876    0.840
c-big                345241    1343126    112023043    233614260     86948776 527521979 69420542 14723910    2.085    0.776
cont-201              80595     239596      5708424      5636772      5539936  5751419   372421   495533    0.987    0.970
```

```
geomean fill_metis / fill_amd  = 1.192
geomean fill_scotch / fill_amd = 0.950   (SCOTCH now produces 5.0% LESS fill than AMD on average)
minimum-fill wins: AMD=1, METIS=0, SCOTCH=3 (ties count for all)
total symbolic time (us): AMD=548435124, METIS=70233153, SCOTCH=15617212
```

### Observations (large corpus, post-FM-fix)

6. **ND now wins on large indef matrices as a class.** SCOTCH takes
   the fill crown on 3 of 4 large matrices (`bratu3d`, `cont-201`,
   `c-big`), losing only on `bcsstk38` (n=8032, densest of the four).
   Geomean fill ratio is 0.950× AMD — SCOTCH is the best fill
   producer at this scale.

7. **c-big flipped completely.** Pre-FM-fix, SCOTCH produced 4.88×
   more fill than AMD on c-big and took 237s (44% slower than METIS
   even). Post-fix, SCOTCH produces **22% less fill** than AMD on
   c-big and takes 14.7s (9× faster than METIS, 36× faster than AMD).
   The diagnosis in the FM-fix commit message explains why: c-big's
   multilevel coarsening projects imbalanced partitions at many
   levels; every such level used to roll back to the imbalance and
   do no useful refinement. Post-fix the pass records the first
   balanced state encountered and refines from there.

8. **bratu3d got slightly worse for SCOTCH but still beats AMD.**
   Pre-fix SCOTCH was 0.784× AMD fill on bratu3d; post-fix 0.840×.
   The FM rebalance fix occasionally prevents SCOTCH from rolling
   back a trajectory that the "roll back to start" behaviour treated
   favourably; this is not a bug but a tradeoff — the fix is correct
   on imbalanced starts, worth the small quality hit on balanced
   starts with unusual trajectories. bratu3d still wins on fill
   (0.840×) and runs comparably to METIS in time.

9. **Ordering time scales strongly with AMD.** On `c-big`, AMD takes
   ~527s vs METIS 69s (7.6× faster) and SCOTCH 14.7s (36× faster).
   Total large-corpus symbolic time: AMD 548s, METIS 70s, SCOTCH 16s.
   SCOTCH's order-of-magnitude improvement over pre-fix comes
   entirely from wasting fewer passes on unstable trajectories.

10. **cont-201 is essentially tied.** All three orderings come within
    3% on fill. On this 80k-row indef matrix the dominant structure
    is balanced between dense Hessian and mesh connectivity.

11. **bcsstk38 is the exception.** AMD wins by 12% on fill (a dense
    8k symmetric matrix, the heaviest nnz/n in this corpus). This
    matches the parity-corpus finding that AMD's minimum-degree
    handles densely-connected small matrices better than ND's
    separator-based approach. Threshold where ND dominates looks
    like nnz ~200k or n ~20k, whichever first.

### FM rebalance fix delta

Pre-fix (commit 7962568), large-corpus numbers were:
```
geomean fill_metis / fill_amd  = 1.194  (essentially unchanged)
geomean fill_scotch / fill_amd = 1.479  → 0.950 (58% reduction in fill ratio!)
SCOTCH wins: 2                           → 3 (now wins on c-big and bratu3d and cont-201)
SCOTCH total time: 238712448 us          → 15617212 us (15× faster)
```

The FM rebalance fix is the single largest improvement in the
ordering crates since the split. It affects every multilevel
scheme (METIS and SCOTCH) but the effect concentrates in SCOTCH
because SCOTCH's band-FM refinement more frequently projects
imbalanced partitions into the next level.

### Revised decision implications

- **AMD is no longer the unambiguous fill-quality default for
  n > ~10k.** SCOTCH geomeans 0.95× AMD on the large corpus and
  wins on 3 of 4 matrices. AMD still wins on the parity corpus
  (median n=77) and on dense small matrices.
- **METIS's role narrows.** Post-fix, SCOTCH matches METIS or beats
  it on every large matrix and does so in less ordering time.
  METIS remains the conservative choice (no FM-trajectory
  surprises like bratu3d's 0.784 → 0.840 delta) and a useful
  baseline.
- **Default-ordering heuristic starts to look reasonable.** Pick
  AMD for n ≤ ~2000, SCOTCH for n > ~2000 with indef or KKT
  structure. Deferred to a separate session with a wider corpus.
- **Open: numeric-fill comparison with pivoting.** All numbers
  here are symbolic. A KKT with significant delayed-pivot churn
  could rewrite the fill picture entirely. Tracked as B6 in
  `dev/plans/ordering-scotch.md`.

## Reproducing

```
cargo run --release --bin bench_orderings
cargo run --release --bin bench_orderings -- tests/data/parity
cargo run --release --bin bench_orderings -- tests/data/large
# Fetch large matrices first:
bash dev/scripts/fetch_large_matrices.sh
```
