# ORBIT2 cluster regression — root cause: unhandled near-dense KKT columns

Date: 2026-04-27
Author: investigation agent (Opus 4.7), under user direction
Status: research note only — no production code edits yet

## TL;DR

The Phase 2.8.1 sparse partition `medium (<500)` p90 regressed to **39.26×**,
driven by the **ORBIT2** cluster (top-10 ratios 304×–384× vs MUMPS) and a
broader set of CUTEst-derived KKT systems with the same structural fingerprint:
**one (or a few) near-dense column(s) — typically a single objective-coupled
variable that appears in nearly every constraint row.** Neither feral's AMD
nor its `feral-metis` clean-room ND port has dense-column quotient logic, so
the dense column either (a) sits at the *root frontal* and explodes nnz_L by
5–47×, or (b) survives ordering at moderate fill but triggers a Bunch–Kaufman
delay cascade once `pivot_threshold = 0.01` is in play. MUMPS handles the
exact same matrices in 1.3–6 ms with 60–110k nnz_L by detecting dense rows
during its analysis (`ICNTL(6)`, `ICNTL(12)`) and pushing them to the end of
the elimination order.

The CHAINWOO fix (extend `pick_default_method` to route low-`avg_deg` n≥2000
to `MetisND`) does **not** fix ORBIT2. ORBIT2 already routes to MetisND
(n=4795, avg_deg=3.06, both branches of the rule fire). MetisND on its own
still produces 14× MUMPS's nnz_L because the dense column is a structural
feature ND cannot quotient out.

## 1. ORBIT2 characterization

Header (`ORBIT2_0000.mtx`):

```
4795 4795 14669
```

`n=4795`, stored `nnz=14669`, `avg_deg = 14669/4795 ≈ 3.06`. From the MUMPS
sidecar, the KKT block partition is 2698 primal × 2097 dual.

Block off-diagonal counts:

| block                       | count |
| --------------------------- | ----- |
| primal–primal (H off-diag)  | **0** |
| primal–dual (Jacobian)      | 11668 |
| dual–dual (B off-diag)      | **0** |

So ORBIT2 is a *pure bipartite KKT*: H is purely diagonal (regularised
identity), B is purely diagonal, and the only off-diagonal mass is the
Jacobian J coupling primals to duals. Six diagonal entries in the primal
block carry `2.0e15` regularisation (`hugeP_diag=6` in the count run); the
other 2992 primal diagonals are O(1).

Degree distribution (off-diagonal-only) by block:

```
primal_deg=2   count=4
primal_deg=3   count=898
primal_deg=4   count=985
primal_deg=2698 count=1     <— column 2698 has off-degree 1794
dual_deg=1     count=2
dual_deg=2     count=5
dual_deg=3     count=300
dual_deg=4     count=806
dual_deg=6     count=1794
```

**The structural fingerprint is column 2698 (the last primal): off-diagonal
degree 1794, i.e. that single primal variable couples to ~85% of all 2097
constraints.** Every dual that has degree 6 inherits its sixth coupling
through that one column. This is the "objective row" / "free regularisation
slack" pattern that shows up in roughly half of CUTEst's least-squares and
NLP families.

## 2. Per-ordering profile (`diag_chainwoo` on ORBIT2_0000)

```
matrix: ORBIT2_0000.mtx, n=4795, nnz=14669
  method   n_snodes  sym_nnz_est  num_nnz_l  max_front  sym_us  num_us
    Auto       594       98357     2001319         42   17277   287739
     Amd       797       88482     5147360         36    2602   969079
 MetisND       578       95549     1544349         39    7593   182086
 ScotchND      962      105298     2346925         34    5955   375066
 KahipND       594       98357     2001319         42   17349   260275
```

`Auto` resolves to `KahipND` (`choose_adaptive` rule
`n < 10_000 && avg_deg < 15.0 → KahipND`, see `src/symbolic/mod.rs:107-110`).

`pick_default_method(4795, 14669)` returns `MetisND` because
`n>=2000 && avg_deg<4.0` fires (`src/symbolic/mod.rs:226`). The bench path
calls `symbolic_factorize` (line 305 of the same file) and therefore picks
**MetisND**, which is the best of the four available orderings here.

## 3. MUMPS comparison

ORBIT2_0000 MUMPS sidecar (`ORBIT2_0000.mumps.json`):

```
factor_us = 2158
factor_nnz = 109782   (== INFOG(9))
inertia = (2698+, 2097-, 0)
```

| solver | nnz_L      | factor_us |  ratio nnz_L | ratio time |
| ------ | ---------- | --------- | ------------ | ---------- |
| MUMPS  |    109,782 |     2,158 |         1.0× |       1.0× |
| feral MetisND (best) | 1,544,349 |   ~190k (sym+num) | **14.1×** | ~88× |
| feral Auto (KahipND) | 2,001,319 |   ~305k          | **18.2×** | ~141× |
| feral AMD            | 5,147,360 |   ~972k          | **46.9×** | ~450× |

The bench reports **factor_us = 652,577 µs** for ORBIT2_0016 (worst in the
top-10), 3.4× the diag's MetisND-only number. The diag uses
`BunchKaufmanParams::default()` which has `pivot_threshold = 0.0`; the bench
uses `pivot_threshold = 0.01` (`src/bin/bench.rs:1198-1202`). The 3.4× gap is
the BK delay cascade firing on the dense column — the first 6 diagonals are
2e15 (huge regulariser), but the rest of the H_diag is O(1); when the dense
column 2698 is pivoted late, every elimination on it exposes 1794 candidate
rows and the column-relative pivot test `|d| ≥ 0.01·col_max` rejects most of
them, generating delayed-pivot postings.

## 4. Diagnosis — ordering, not numeric

Even with a perfect numeric kernel, **MetisND alone produces 14× MUMPS's
nnz_L on ORBIT2.** That is structurally impossible for the BK kernel to
recover: at 14× the work and 14× the L footprint, even a zero-overhead
delay-free factor would still be ~14× MUMPS at best. The pivot-cascade
multiplier on top is real (3.4×) but it is the *secondary* effect.

Root cause (one sentence): **`feral-metis::metis_order` does not quotient out
near-dense rows/columns before recursive bisection, so the dense column at
index 2698 sits inside a separator and inflates fill in every subtree.**

MUMPS-5.x default is METIS-NodeND with `ICNTL(6)=7` (automatic permutation
including dense-row detection) and a quasi-dense column threshold around
`max(40, 10√n)` ≈ 700 for n=4795. Column 2698 (degree 1794) is well above
that and gets pulled out before ND.

Supporting evidence from other CUTEst families with the same pattern:

| family       | n    | avg_deg | max_deg | feral-best nnz_L | MUMPS nnz_L | fill ratio |
| ------------ | ---- | ------- | ------- | ---------------- | ----------- | ---------- |
| ORBIT2       | 4795 | 3.06    | 1794    | 1,544,349 (MET)  |     109,782 | **14.1×** |
| COSHFUN      | 8001 | 2.00    | 2000    |   507,002 (MET)  |      85,929 | 5.9× |
| CATENA       | 5999 | 2.00    | (low)†  |   406,634 (MET)  |      78,725 | 5.2× |
| ARWHEAD      | 5000 | 2.00    | 4999    |   169,440 (any)  |      59,964 | 2.8× |
| BIGBANK      | 3034 | 2.27    | (low)   |   203,113 (AMD)  |      74,669 | 2.7× |
| A5NSSNSM     | 4508 | 2.80    |  501    |   161,159 (MET)  |      74,278 | 2.2× |
| EXPQUAD      | 1200 | 2.00    | 1099    |     9,500 (AMD)  |      14,400 | **0.66×** |
| EDENSCH      | 2000 | 2.00    |  (low)  |    41,133 (MET)  |      23,973 | 1.7× |

† CATENA has many ~3-degree columns plus 2697 huge regularisation diagonals.

ARWHEAD is striking: every ordering produces *exactly* nnz_L=169,440 because
column 0 is a perfect arrow (degree 4999). Neither AMD nor any ND variant in
feral can do anything about it without a dense-quotient pass. MUMPS gets
59,964 — 2.8× lower fill — by treating the arrow column as a delayed/late
pivot.

EXPQUAD (n=1200) is in the bench's small-frontal bucket and shows the
inverse: when `pick_default_method` correctly returns **AMD** (because n<2000
falls out of the dispatch rule), AMD gives 9,500 nnz_L vs MUMPS's 14,400 —
feral *beats* MUMPS by 1.5×. So the issue is *specifically* that for n≥2000
with a single dense column, ND's recursive bisection over-fills.

## 5. Other medium-bucket offenders

Beyond ORBIT2, the medium-bucket p90=39× is sustained by the families above
plus a longer tail. Sampling families with `max_deg/n > 0.1` and
`1000 ≤ n ≤ 10000` (`bash` walk over kkt-expansion/) returned 30+ matches in
under a minute. Some are quasi-dense (BDQRTIC max_deg/n=1.0, GILBERT 1.0,
FMINSURF 1.0); many are arrow-shaped (ARWHEAD, COSHFUN, EXPQUAD); and a few
are truly dense in the H block (ARGLBLE/CLE, CHANDHEU, DIXCHLNV with avg_deg
> 50). The common feature for the ones in the medium bucket is a small
max_front (≤ 500) yet structurally large fill due to the dense column.

## 6. Proposed fix — dense-column quotient in `feral-metis` ordering driver

Two complementary changes, ordered by leverage:

### Fix A (high leverage, surgical) — quasi-dense quotient before ND

In `crates/feral-metis/src/lib.rs::metis_order_full`, pre-process the input
pattern by partitioning columns into a "dense set" and a "sparse set":

```
let dense_thresh = max(40.0, 10.0 * (n as f64).sqrt()) as usize;
let mut dense_cols: Vec<i32> = ...;  // columns with deg > dense_thresh
let mut sparse_cols: Vec<i32> = ...;
```

Run the existing M1–M7 ND pipeline on the **sparse-induced subgraph only**.
Append the `dense_cols` (in their natural order, or by descending degree) to
the *end* of the returned permutation. The output is a valid permutation of
`[0, n)`.

This is the same technique HSL_MC68/MA77/MUMPS use (see Davis & Hager,
"Dynamic supernodes in sparse Cholesky update/downdate and triangular
solves", 2009 §3.2; or MUMPS userguide §3.6 on `ICNTL(6)`). It is a 30-line
change with no allocation or numerical impact.

For ORBIT2_0000: column 2698 (deg 1794 > thresh ≈ 690) gets pulled out;
the remaining graph has avg_deg ≈ 2.5 with a clean bipartite block structure
that ND splits cleanly. Predicted nnz_L drops into the same neighbourhood as
MUMPS (~110–200k) — i.e., 7–14× win on ORBIT2 alone, and the pivot cascade
disappears because column 2698 is at the root frontal where it is supposed
to live.

### Fix B (broader, medium leverage) — same quotient inside `feral-amd`

`crates/feral-amd/src/lib.rs::amd_order` does not implement AMD's standard
"`dense` parameter" (degrees > `Alpha * sqrt(n)` are ignored during quotient
graph updates and appended to the end). On ARWHEAD/COSHFUN this would
recover ~3× fill on its own, regardless of whether `pick_default_method`
routes to AMD or to METIS. Because ARWHEAD-shaped graphs return identical
nnz_L for every ordering (demonstrated above), the fill-reducing dispatch
rule cannot help; the dense-quotient step is the *only* lever.

This is a larger change because feral-amd is a clean-room AMD implementation
and the dense-handling is woven through quotient-graph updates. The Davis
1996 AMD paper §5 covers it in 2 pages.

### Fix C (no leverage but cheap fingerprint) — tighten dispatcher

Optionally, in `pick_default_method` (`src/symbolic/mod.rs:221`), add a
fingerprint that scans the matrix in O(n) time for a max-degree column and
short-circuits to a (future) "DenseColumnSplit" preprocessor when
`max_deg > 10·sqrt(n)`. Without Fix A or Fix B this fingerprint has nothing
to dispatch to, so this is a cosmetic change that should land *with* Fix A.

### Recommendation

**Implement Fix A first.** It is the smallest change, addresses the largest
number of cases (everything routed to MetisND), and the technique is
well-established. Re-run `diag_chainwoo` on the eight families above; if
nnz_L for the MetisND column drops to within 2× of MUMPS for ORBIT2,
COSHFUN, and CATENA, the fix is proven. Fix B then catches the AMD-routed
tail (BIGBANK, EXPQUAD-shaped problems with n<2000).

## 7. Validation plan

Step 1 (cheap, before code change): write a `diag_dense_quotient.rs` that
reads a matrix, splits dense columns by the threshold above, runs
`feral-metis` on the induced sparse graph, appends dense columns at the end,
runs `symbolic_factorize_with_method` with the resulting permutation under
`OrderingMethod::Amd` (i.e., feed the perm in as a fixed permutation),
factorises, and prints `(nnz_L, num_us)`. *This is a research binary, not a
production fix.* Compare to the diag_chainwoo baseline.

Step 2 (after Fix A lands): re-run `diag_chainwoo` on
{ORBIT2, COSHFUN, CATENA, A5NSSNSM, ARWHEAD, BIGBANK, EXPQUAD, EDENSCH}
and verify all `MetisND` nnz_L are ≤ 2× MUMPS. Acceptance gate before
running the full corpus.

Step 3 (only after Step 2 passes): run the bench against the full
kkt-expansion corpus and confirm `medium (<500)` p90 ≤ 3.0 and
`small-frontal (<200)` p90 ≤ 2.0. Expect `small-frontal` to also benefit
because n<2000 arrow problems (EXPQUAD-style) currently route to AMD which
also lacks dense quotient (Fix B prerequisite for full closure).

Risk: dense-column quotient can mis-handle truly bordered systems where the
"dense" column is actually a degenerate near-zero pivot that MUMPS would
*also* delay rather than eliminate at end. We treat this as a numerical
question separate from the symbolic ordering: the proposed fix only changes
the *order*, never the *pivot decision*, so the BK kernel's existing
`pivot_threshold` + delay-to-parent infrastructure remains the safety net.

## 8. Files cited

- `src/symbolic/mod.rs:97-114`  — `choose_adaptive` rule for `Auto`
- `src/symbolic/mod.rs:221-231` — `pick_default_method`
- `src/symbolic/mod.rs:301-307` — `symbolic_factorize` calls `pick_default_method`
- `src/bin/bench.rs:1198-1204`  — `params_kkt_sparse` with `pivot_threshold=0.01`
- `src/bin/bench.rs:1601-1605`  — bench dispatch path (uses `symbolic_factorize` by default)
- `src/bin/diag_chainwoo.rs:39-75` — diagnostic loop over orderings
- `crates/feral-metis/src/lib.rs:144-167` — `metis_order` / `metis_order_full`

## 9. Open questions / follow-ups (not for this session)

- Quantify the AMD dense-quotient win independently (Fix B). Worth it?
  Likely yes for n<2000 arrow problems.
- Does the dense-quotient invalidate any existing scaling/preprocess
  invariants? `Mc64Symmetric` already guards against dense columns
  (`src/scaling/hungarian.rs:270` "SPRAL's don't-assign-on-dense-cols
  guard"), so the matching layer is fine.
- ORBIT2 is one of 20 ORBIT2_xxxx files; spot-checking ORBIT2_0007 vs
  ORBIT2_0016 in the bench's top-10 shows they all share n=4795 and
  identical structure (KKT iterates of the same NLP), so a fix that works
  on ORBIT2_0000 should work on all 20.

## 10. Post-mortem (2026-04-27, after Fix A landed)

Fix A was implemented in `crates/feral-metis/src/lib.rs` (commit `ef366de`)
exactly as specified in §6: pull columns with off-degree > `max(40,
ceil(10*sqrt(n)))` out of the ND graph, run M1–M7 on the induced sparse
subgraph, append dense columns at the end.

**Empirical result on ORBIT2_0000 (`./target/release/diag_orbit2_quotient`):**

| ordering                                  | nnz_L      |
|-------------------------------------------|------------|
| baseline (Fix A disabled, MetisND)        | 1,544,349  |
| **Fix A enabled (default at commit time)**| 2,254,428  |
| MUMPS reference                           | 109,782    |

Fix A *increased* fill by 46%. The §6 prediction (110–200k) was wrong.

### Why Fix A was the wrong shape

Two reference-solver investigations (mumps-expert + spral-expert agents,
2026-04-27) found:

1. **MUMPS does NOT pre-strip dense columns.** `ICNTL(6)` is MC64
   matching, not dense-row removal — the §6 reading of the user-guide
   was a misinterpretation. MUMPS handles dense rows *inside* its
   AMD/AMF (`MUMPS_QAMD` in `ana_orderings.F:5226+`) via the `THRESM`
   parameter and the `HEAD(N)` quasi-dense list. The dense column is
   always in the graph passed to the ordering algorithm; the algorithm
   defers it during elimination.
2. **MUMPS auto-dispatch on ORBIT2 picks AMF, not METIS.** For
   `SYM=2, N=4795 ≤ SMALLSYM=10000, NBQD=1 < MAXQD=2`, the auto-choice
   in `ana_set_ordering.F:52-78` picks `IORD=2` (HAMF4). HAMF4 has no
   explicit dense detection but its bucketed-degree min-fill naturally
   places very-high-degree variables last. MUMPS's 109k nnz_L came
   from HAMF4, not METIS.
3. **SSIDS does NOT special-case dense columns at all.** Full pattern
   → METIS-ND → supernodal amalgamation with `nemin=32`. The dense
   column lands at the top separator naturally; zero-fill chain merges
   (`do_merge` in `core_analyse.f90:806-822`) collapse the chain into
   one dense BLAS-3 root frontal. SSIDS's `nnz_L` on ORBIT2-shape
   inputs is probably *not* small either — speed comes from BLAS-3
   kernel quality on the root, not fill reduction.
4. **Why pre-stripping regresses on bipartite KKT:** removing column
   2697 strips 1794 entries (~30% of off-diagonal mass). The residual
   graph is two diagonal blocks loosely glued by a thinned Jacobian —
   METIS-ND can't find a useful separator, invents bad ones, and the
   next-densest column dominates fill. The structural signal that made
   the dense column the natural top separator was destroyed by the
   pre-strip.

### Action taken

- Default flipped to `dense_quotient_enabled: false` (commit on top of
  `f073070`). The opt-in code path is preserved for diagnostic
  experimentation; `diag_orbit2_quotient` lets us continue measuring.
- Fix A is structurally correct (verified in unit tests) but
  empirically wrong for bipartite KKT. It may still help on
  *bordered* patterns where the dense column is genuinely separable
  from the rest of the graph; we have no fixture proving that yet.

### Next direction (Fix B, deferred to a future session)

Implement QAMD-style dense-row deferral inside `feral-amd` per the
MUMPS source:

- Add a `THRESM` parameter (`avg_deg*10 + (max_deg − avg_deg)/10 + 1`,
  or simpler `50*avg_deg`).
- During the AMD elimination loop, skip variables with degree > THRESM,
  store them in a quasi-dense list (analog of `HEAD(N)` in
  `ana_orderings.F:5500-5508`).
- After all non-dense pivots are exhausted, mass-eliminate the dense
  list at the end.
- Reference: Davis 1996 §5; MUMPS `ana_orderings.F:5226-5650`.

Estimated 100–200 LoC. Combined with a dispatcher fingerprint that
matches MUMPS's auto-choice (`AvgDens*50` + `NBQD≥2` test from
`dana_aux.F:3578-3585` and `ana_set_ordering.F:52-78`), this should
fix ORBIT2/COSHFUN/CATENA.

Before doing that, we should also answer: is feral's 1.54M nnz_L on
ORBIT2 already concentrated in one large root supernode (SSIDS-shape
outcome — meaning factor time is already dominated by BLAS-3 on the
root, and W-1+W-2's kernel speedups already help), or spread across
many supernodes (true ordering problem)? `diag_chainwoo_profile` on
ORBIT2_0000 will answer this in <1 minute.
