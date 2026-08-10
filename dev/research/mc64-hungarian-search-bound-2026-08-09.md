# MC64 Hungarian search bound — the lever does not exist; the cost is memory layout

Date: 2026-08-09
Status: negative result on the stated lever + a small verified win
Follows: `dev/research/mc64-condition1-cost-share-2026-08-09.md`

## Correction to the prior note

`mc64-condition1-cost-share-2026-08-09.md` closes by recommending a
Hungarian search bound and describes it as

> "same-output-less-work, needs no gate and no residual argument"

**That characterization is wrong and should not be relied on.** Two
independent reasons, both found by inspection before any code was
written:

1. The classic shortest-augmenting-path bound is **already
   implemented**. `csp` (cost of the best augmenting path found so far)
   prunes the root scan (`src/scaling/hungarian.rs:541`), terminates the
   main loop (`:570`), and prunes the inner column scan (`:591`). There
   is no missing standard bound to add.
2. Any *further* truncation would end a search before its shortest
   augmenting path is proven, yielding a suboptimal matching, hence a
   different scaling vector. That is a numerics change requiring a
   corpus inertia/residual study and human approval — not a free win.

`dev/tried-and-rejected.md:2313` (2026-06-06-03) additionally proves
that no per-column reduced-cost bound can ever prune the inner scan
(`vj + lb_tight = dq0`, and `q0` was popped only because `dq0 < csp`),
and instructs future sessions not to retry it. That directive stands.

## Where the time actually goes

Per-edge-scan cost differs 4.5x between the two families that dominate
the corpus, on identical code:

| matrix          | match_us  | edge_scans | ns/scan | searches | touched/search |
|-----------------|-----------|------------|---------|----------|----------------|
| pinene_3200_0006|   206,907 | 70,268,885 |  2.94   | 77,994   |     87         |
| nql180_0000     | 1,214,332 | 91,979,446 | 13.2    |    893   | 24,020         |
| nql180_0002     | 3,197,295 |222,147,205 | 14.4    |    895   | 53,411         |

The two families are in opposite regimes. pinene's init heuristic
leaves ~61% of columns unmatched (77,994 searches on n=127,995), so it
runs many tiny searches whose working set stays in L1. nql180's init
matches 99.66% (893 searches on n=259,681), so it runs a few enormous
searches, each touching 9-21% of all rows — a working set far past L1.
The 4.5x per-scan gap tracks working-set size, not algorithm.

`build_cost_graph` is not the lever either: 8-12 ms/iterate, which is
20% of pinene's cheapest iterate but 0.4% of nql180's.

## What was done

The inner loop reads five *random-index* arrays per edge
(`visited[i]`, `u[i]`, `iperm[i]`, `d[i]`, heap position), and every
heap sift level did `d[self.heap[parent]]` — a load from the heap array
followed by a **dependent random load** into `d`.

`IndexHeap` now stores the key inline (`HeapEntry { key, idx }`), so
sifting stays within the heap array, which parent/child arithmetic
already walks with locality. The comparisons performed are unchanged,
so the matching is bit-identical by construction. This also removes the
heap's `&[f64]` borrow of `d`, which is what previously prevented `d`
from being fused with the other per-row arrays.

## Evidence

Bit-identity oracle: `diag_mc64_scaling_fingerprint` hashes the raw
IEEE bits of every entry of `SparseFactors::scaling`, so *any*
deviation in the matching is detected, not merely one large enough to
move a residual. Verified deterministic across repeat runs first.

- **51 matrices across 39 families: all fingerprints unchanged.**

Speed (median of 3, standalone matching, same binary path):

| matrix           | before us | after us | speedup |
|------------------|-----------|----------|---------|
| nql180_0000      | 1,174,045 | 1,128,533|  1.040x |
| nql180_0002      | 3,083,529 | 2,917,491|  1.057x |
| pinene_3200_0001 |    44,381 |   45,279 |  0.98x  |
| pinene_3200_0006 |   202,205 |  202,969 |  1.00x  |
| marine_1600_0000 |    29,911 |   29,820 |  1.00x  |

4-5% on the family that dominates corpus MC64 time, nothing elsewhere —
exactly the predicted split, since pinene's searches were already
L1-resident. This is smaller than hoped and is reported as such.

`sample` profile of nql180_0002: 100/108 samples in the inlined main
loop, 7 in `IndexHeap::update`. Decrease-key sift-*up* is ~6.5% — most
sift-ups terminate immediately — so the win comes from the sift-*down*
path inside `pop`, as intended.

## What is left, and what it is worth

A microbenchmark of the access pattern alone (5 split arrays vs 1 fused
struct, nql180's n) shows the split/fused gap is entirely a
working-set effect:

    ws=   500  split=0.70 ns  fused=0.70 ns  speedup=0.99x   <- pinene regime
    ws= 50000  split=2.12 ns  fused=1.12 ns  speedup=1.90x
    ws=259681  split=2.27 ns  fused=1.35 ns  speedup=1.68x   <- nql180 regime

So fusing `visited`/`u`/`iperm`/`d` into one per-row struct is worth
~1.7x **on the array-access portion only**, which is ~2.3 ns of
nql180's ~13 ns/scan. Expected end-to-end gain ~7%, bit-identical, now
unblocked by the inline-key change. Not yet done.

Honest bound on this whole direction: the Hungarian on nql180 is doing
893 genuinely long shortest-path searches, and their length is a
property of the matrix, not of the implementation. Constant-factor
memory work is worth tens of percent, not multiples. A multiple
requires changing the matching (auction/eps-optimal, or skipping MC64
on these KKTs) — which is the gated, approval-requiring path, not this
one.
