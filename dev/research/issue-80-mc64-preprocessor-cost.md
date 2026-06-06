# Issue #80 — MC64 preprocessor cost on large near-tree powerflow KKTs

**Date:** 2026-06-06
**Status:** **RESOLVED** by an MC64 kernel fix (§8), not the routing gate.
The ~O(n^1.9) cost was an avoidable per-iteration heap allocation; fixing it
drops MC64 on pf22 from 53s to 0.31s and the default first factor from 65.5s
to 2.6s, behavior-preserving. The §6 routing gate is **no longer needed** —
kept below for the record. Profiler-attribution fix landed (`36d847a`).
**Repro matrices:** `pounce/gams/nlpbench/feral_repro/powerflow22/
kkt_solve_iter2.bin` (pf22, n=2.8M) and `.../powerflow_profile_2026/
pf01_iter2.bin` (pf01, n=366k).
**Journal:** `dev/journal/2026-06-06-01.org`

---

## 1. TL;DR — issue #80's premise is wrong

Issue #80 reports a ~55s "ordering" stage on the pf22 KKT (n=2,813,976) and
blames the AMD minimum-degree kernel, asking for a faster AMD / bucketed
min-degree. **AMD is not the bottleneck.**

- `feral_amd::amd_order` (the production ordering — a full quotient-graph
  AMD with degree buckets in `feral_ordering_core::quotient_graph`) orders
  pf22 in **0.276s**.
- The ~53s is the **`LdltCompress` preprocessor's MC64 matching**
  (`scaling::compute_mc64_cache` → `mc64::compute_matching`, a SPRAL-style
  shortest-augmenting-path Hungarian). The per-stage symbolic profiler
  folded it into the `ordering` stage timer, which is the entire reason the
  issue mis-attributed the cost.

(The legacy `src/ordering/amd.rs::amd_order` *is* O(n²), but it is dead code
— only `permute_pattern` from that file is still used. Don't be misled by it.)

### Two repro traps that cost time

1. **Wrong function.** Production dispatches `OrderingMethod::Amd` to
   `feral_amd::amd_order` (`symbolic/mod.rs:569`), not the in-tree
   `ordering::amd::amd_order`.
2. **MC64 needs real values.** With `vals = 1.0` everywhere, MC64 matching is
   trivial and symbolic runs in ~1.5s. The 53s only appears with the real
   `f64` values from the dump. Any reproduction must load the real values.

---

## 2. Profiler-attribution fix (landed)

`symbolic/mod.rs` now records the preprocessor under its own `ldlt_compress`
stage (plus `compress_pattern` / `expand_perm` on the compressed-graph path)
and times only `run_external_ordering` under `ordering`. Confirmed on real
pf22: `ldlt_compress 63.06s`, `ordering 0.237s`. Test:
`tests/symbolic_profiler.rs::ldlt_compress_recorded_separately_from_ordering`.

This does not change any output — but it is the change that would have
prevented the misdiagnosis, and it lets callers budget the first factor
honestly.

---

## 3. Where MC64 actually runs — it is invoked twice

MC64 is reachable from two independent routing decisions, **both** of which
fire on the arrow/slack-KKT signature this matrix class has:

1. **`pick_ordering_preprocess` → `LdltCompress`** (symbolic). Predicate:
   `n >= 128 && low_degree_cols / n >= 0.30` (columns with stored nnz ≤ 2).
   Runs `compute_mc64_cache` to build the super-variable compression. The
   cache is kept and reused by scaling.
2. **`pick_scaling_strategy` → `Mc64Symmetric`** (numeric). Predicate:
   `diag_only / n >= 0.30`. If the `LdltCompress` cache exists it is reused
   (no second matching); if not, the numeric phase runs MC64 itself.

Consequence: **disabling only one leaves the cost.** The fast path requires
both `preprocess = None` *and* a non-MC64 scaling (`InfNorm`).

---

## 4. Experiment — full Solver first-factor on pf22 (n=2.8M)

`factor()` wall (symbolic + numeric), inertia, and relative residual on the
dumped RHS. Reference inertia: **940,248** negative eigenvalues.

| config | first factor | status | n_neg | resid | MC64? |
|---|---|---|---|---|---|
| **A** default: LdltCompress + Mc64Symmetric | **65.5s** | Success | 940248 ✓ | 5.3e-13 | yes (symbolic) |
| **B** None + InfNorm | **2.42s** | Success | 940248 ✓ | 3.8e-11 | **no** |
| C None + Mc64Symmetric | 63.4s | Success | 940248 ✓ | 3.0e-10 | yes (numeric) |
| D LdltCompress + InfNorm | 62.3s | Success | 940248 ✓ | 5.3e-13 | yes (symbolic) |

**Findings**

- **Inertia is exactly correct (940,248) in all four configs.** MC64 is *not*
  required for the inertia hard rule on this matrix.
- **Config B is 27× faster** (2.42s vs 65.5s) with a residual of 3.8e-11 —
  far inside any IPM tolerance (typically ~1e-8). MC64 buys 3.8e-11 → 5.3e-13,
  i.e. nothing that matters here.
- C and D isolate the cost to MC64 regardless of which decision invokes it,
  confirming §3.
- Fill is identical with/without compression (issue #80: 3.81× both ways), so
  `LdltCompress` buys **no fill** on this class.

## 5. Second size point — pf01 (n=366,288, avg_deg ≈ 5.76)

| config | first factor | status | n_neg | resid |
|---|---|---|---|---|
| A default | 1.73s | **Singular** | 126744 | 1.4e-9 |
| B None + InfNorm | **0.32s** | Success | 126744 | 2.7e-9 |
| C None + Mc64Symmetric | 1.60s | **Singular** | 126744 | 8.2e-9 |
| D LdltCompress + InfNorm | 1.52s | Success | 126744 | 2.9e-10 |

- All four configs agree on inertia (126,744) — a 4-way internal consistency
  check that the InfNorm path does not corrupt inertia on a second powerflow
  matrix.
- **MC64 scaling produces a spurious `Singular` status** here (A, C) while
  InfNorm returns clean `Success` (B, D), with all residuals fine. This is a
  second mark against MC64 on the powerflow class.
- At 366k, MC64 is only ~1.3s — tolerable. The cost is size-driven.

### MC64 cost scaling

`ldlt_compress` (MC64) stage: **1.3s @ 366k → 63s @ 2.8M**. That is 48× for a
7.7× increase in n, i.e. **≈ O(n^1.9)** — near-quadratic. This is the
shortest-augmenting-path matching degenerating on long augmenting paths in a
near-tree graph. It is a genuine algorithmic cost, not constant overhead.

Note pf01's `avg_deg 5.76` is *above* #50's `avg_deg < 5` powerflow predicate,
yet it still triggers `LdltCompress` (the arrow signature is degree-based, not
avg-deg-based). So the MC64 gate cannot simply reuse #50's predicate.

---

## 6. Proposed gate (NOT yet implemented)

Skip MC64 on **large arrow-signature KKTs** — route both the preprocessor and
the scaling away from MC64 when the matrix is big enough that the ~O(n^1.9)
matching dominates:

- `pick_ordering_preprocess`: return `None` when `n > N_GATE` *and* the arrow
  signature holds (the same `low_degree_cols / n >= 0.30` it already tests).
- `pick_scaling_strategy`: return `InfNorm` instead of `Mc64Symmetric` under
  the same `n > N_GATE` + arrow-signature condition.

`N_GATE` wants calibration against the powerflow size range (pf01 366k →
pf22 2.8M; intermediate pf10/15/21/23/24 at 0.6–2.4M were offered by the
issue author). A first cut: `N_GATE ≈ 500_000` — pf01 (366k, MC64 1.3s) stays
on the MC64 path; everything above pays InfNorm. The crossover where MC64
becomes "too expensive to justify 5e-13 vs 4e-11 residual" is between 366k
and 2.8M and should be pinned with the intermediate dumps.

### Why this is a `decisions.md`-level change, deferred

The 2026-04-19 lever-C diff made `Mc64Symmetric` the default *because* it
gave 8× tail-residual compression on the factor/MUMPS corpus and material
wins on VESUVIO/CRESC; `InfNorm` is the only thing that solves MSS1_0009 to
working precision. The gate must be **narrow** — large powerflow-class only —
so it does not regress those. Before flipping the default:

1. Run the full inertia/residual corpus A/B (the `tests/*_corpus_oracle` and
   the IPM bench) with the gate on, confirming no inertia regression and no
   material residual loss outside the gated class.
2. Add the intermediate powerflow sizes to pin `N_GATE`.
3. Record in `decisions.md` and `CHANGELOG.md` (changed default behavior).

### Alternative / complementary: speed up MC64

The ~O(n^1.9) matching is the root inefficiency. A near-linear matching (or a
cheap initial-matching heuristic that resolves most of the assignment before
the augmenting-path phase, à la MC64's `jdperm`/initial-extreme-matching) would
help every MC64 caller, not just the gated class. Larger effort; the gate is
the pragmatic near-term fix.

---

## 7. Bottom line

- AMD is fine (0.28s). Do **not** implement "bucketed min-degree" — it is
  already bucketed.
- The 53s is MC64, invoked twice (preprocessor + scaling) on the arrow
  signature, and is ~O(n^1.9).
- On pf22, dropping MC64 (config B) is **27× faster** with correct inertia and
  a residual that is irrelevant to the IPM.
- Land: profiler fix (done). Propose: a narrow large-powerflow MC64 gate,
  pending a corpus validation sweep before any default flip.

---

## 8. RESOLUTION — MC64 had an avoidable O(n·m) heap allocation

The ~O(n^1.9) was **not** inherent to shortest-augmenting-path matching. In
`hungarian_match` (`src/scaling/hungarian.rs`), the per-column augmenting loop
allocated a fresh `IndexHeap::new(m)` **inside** the loop:

```rust
for jord in 0..n {                 // up to n unmatched columns
    if jperm[jord] != NONE { continue; }
    ...
    let mut heap = IndexHeap::new(m);   // O(m) alloc + zero, EVERY iteration
    ...
}
```

`IndexHeap::new(m)` zeroes `2m+1` entries. Over up to `n` unmatched columns
that is **O(n·m) ≈ O(n²)** of pure allocation/zeroing — independent of how
short the actual augmenting paths are, and the one piece of per-iteration work
that was not already incremental (`d` and `visited` were reset over a tracked
`touched`/`visited_rows` set; only the heap was reallocated).

**Fix:** allocate the heap once before the loop and reset it incrementally at
iteration end via `IndexHeap::reset(touched)` — sets `pos[i] = 0` for the
touched rows and `len = 0`. Every heap member is in `touched` (an index is only
inserted right after `touched.push`), so this is a complete reset in
O(|touched|).

**Measured on real pf22** (`FERAL_MC64_TRACE=1`):

| quantity | before | after |
|---|---|---|
| MC64 matching | ~53s | **0.309s** (~170×) |
| default first factor (LdltCompress + Mc64Symmetric) | 65.5s | **2.594s** (~25×) |
| inertia (n_neg) | 940248 | 940248 ✓ |
| residual | 5.28e-13 | 5.28e-13 (identical) |

The matching output is bit-identical (same inertia and residual) — purely a
storage-reuse change. 48 scaling lib tests pass.

**Consequence for the §6 gate:** unnecessary. The default config is now 2.6s on
pf22 — as fast as the would-be-gated `None + InfNorm` path (2.42s) — while
keeping `Mc64Symmetric`'s conditioning and the `2026-04-19` default. No
default-behavior change, no corpus-validation risk. Do not implement the gate.

(The §6 analysis stands as the evidence that inertia is correct without MC64,
and the pf01 `Singular`-under-MC64 observation remains a separate latent issue
worth a look, but it is not a performance lever anymore.)
