# MC64 dense-column fast path

**Status:** Research note (precedes implementation, per spec §5.1).
**Date:** 2026-06-06
**Author:** session 2026-06-06-03
**Related:**
- Issue #80 (MC64 preprocessor cost); the heap-reuse fix `6699f09` closed the
  *near-tree heap-realloc* cost class but NOT this *dense-column* class.
- `dev/research/scaling-audit-2026-06-06.md` §4, §8.1 (the dense-column lead).
- `src/scaling/hungarian.rs` (`hungarian_match_instrumented`),
  `src/symbolic/mod.rs` (LdltCompress branch),
  `src/scaling/mod.rs` (`pick_ordering_preprocess`, `pick_scaling_strategy`).

---

## 1. The cost, localized

rocket_12800_0000: n=89601, cost_nnz=575985, **max_col_degree=38401** (one
coupling column touches 43% of rows). Instrumented Hungarian (release):

```
searches=29416  touched_total=2.238e8  heap_init=2.239e8  edge_scans=3.710e8  wall=3.69s
```

Mechanism (crisp): greedy init phase 2 deliberately refuses to claim the dense
column (`col_len > m/10` guard, hungarian.rs:336), so it enters the main loop
unmatched and gets matched there to a single row `m0`. Thereafter, in **every
augmenting search that pops `m0`**, the inner loop (hungarian.rs:585) re-scans
the dense column `iperm[m0]` — relaxing up to `dense_deg` rows. With
`touched_total / searches ≈ 7600`, roughly 1-in-5 searches pop `m0` and each
pays `O(dense_deg)`. That is the `O(searches × dense_deg)` law:
`6000 × 38401 ≈ 2.3e8 ≈ touched_total`, plus root scans and body columns make
up the `3.7e8` edge total.

This is a **one-time per-pattern** symbolic cost (LdltCompress runs MC64 once,
cached across IPM iterations), but 3.7s on a 90k KKT is a real inefficiency.

## 2. Options from the audit, evaluated against the hard rules

The hard rules forbid changing the matching/scaling vector without verifying
inertia/residuals on the corpus (human approval). So each option is judged on
whether it is **behavior-preserving** (produces a bit-identical matching).

### Option (b) — symbolic-side skip of the speculative MC64 — does NOT help rocket

`pick_ordering_preprocess` (mod.rs:485) selects `LdltCompress` purely on the
low-degree fraction (≥30% columns with nnz≤2) and `n≥128`; it **never inspects
max column degree**. rocket has both the arrow-KKT low-degree signature *and* a
near-dense column, so it routes to LdltCompress and runs the expensive MC64.

The resulting `Mc64Cache` is set **unconditionally** (mod.rs:779), even on the
`map.ncmp()==n` "no compression leverage" fall-through. It is reused for
`Mc64Symmetric` numeric scaling purely as an **optimization**: if absent, the
numeric phase recomputes the identical matching via `mc64::compute_symmetric`
(scaling/mod.rs:344). So the cache is *not* load-bearing for correctness.

But `pick_scaling_strategy` (mod.rs:653) returns `Mc64Symmetric` exactly when
`max_col_nnz > 32 && diag_only/n ≥ 0.3`. rocket's degree-38401 column trips the
arrow-head gate, so **rocket's numeric scaling resolves to MC64**. Skipping the
symbolic matching therefore saves nothing for rocket: the numeric phase would
recompute the same 3.7s matching. Option (b) only helps matrices whose scaling
is *not* MC64 (Identity/External/InfNorm/Auto→InfNorm), where the symbolic MC64
is purely speculative for compression — a real but orthogonal win, deferred.

### Option (a) — handle the dense column out of the inner loop

Investigated in depth (§3). **Result: no behavior-preserving inner-loop fast
path exists.** A column-level reduced-cost bound provably cannot prune the
scan, at any tightness (§3.2), and SPRAL confirms the cost is inherent to the
algorithm (§3.3). Closed.

## 3. Why a behavior-preserving inner-loop fast path is impossible

### 3.1 The idea tried (and why it was sound but useless)

`u` is monotonically non-increasing over the main loop. The only write to `u`
after initialization is hungarian.rs:640, `u[i] += d[i] - csp`, applied to rows
in `visited_rows`; such a row was popped (line 573) only while `d[top] < csp`
(line 570), so `d[i] - csp < 0` and each update *decreases* `u[i]`. Within a
search `u` is constant. Hence `lb[j] = min_k(cost[k] - u_init[row(k)])`, built
once after greedy init, is a permanent lower bound on column `j`'s minimum
reduced cost. In the inner loop, `min_k dnew = vj + min_k(cost[k]-u[i]) ≥
vj + lb[j2]`, so `vj + lb[j2] ≥ csp` ⟹ every edge has `dnew ≥ csp` ⟹ the scan
is a provable no-op and can be skipped bit-identically.

This was implemented (a `Vec<f64>` lb built in one O(nnz) pass, a one-line guard
before the inner scan) and **measured on rocket: it fired 0 times**
(`inner_scan_skips=0`, `edges_saved=0`, edge_scans and wall time unchanged at
3.7e8 / 3.96 s). All 7 Hungarian unit tests and the full 317-test lib suite
stayed green, confirming behavior preservation — but the optimization is inert.

### 3.2 The impossibility proof (column bounds can never prune)

The loose bound failing prompted analysis of the *tightest possible* column
bound. The inner loop scans `j2 = iperm[q0]`, the column matched to the just-
popped row `q0`. By the LP optimality conditions the kernel maintains:
- complementary slackness on the matched edge: `u[q0] + v[j2] = cost[jperm[j2]]`,
  i.e. the tight column dual is `v[j2] = cost[jperm[j2]] - u[q0]`;
- dual feasibility on all edges: `cost[k] - u[i] ≥ v[j2]`, so the matched edge
  *achieves* the column's minimum reduced cost — `v[j2] = min_k(cost[k]-u[i])`.

So the tightest valid bound is `lb_tight[j2] = cost[jperm[j2]] - u[q0]`. Plug
into the skip test (with `vj = dq0 - cost[jperm[j2]] + u[q0]`, hungarian.rs:583):

```
vj + lb_tight = (dq0 - cost[jperm[j2]] + u[q0]) + (cost[jperm[j2]] - u[q0]) = dq0.
```

The skip fires iff `vj + lb_tight ≥ csp`, i.e. `dq0 ≥ csp`. But `q0` was popped
only because `dq0 < csp` (line 570). **So the skip condition is never satisfied,
for any column, even with the tightest bound.** The matched edge always sits
exactly at the boundary `dq0`; the inner scan exists precisely to find improving
edges to *other* (unmatched or higher-`d`) rows, and a column-aggregate bound
carries no information about those. Column-level pruning is structurally
impossible. (This also explains why maintaining `v` live — the only thing that
would tighten the bound — would not help.)

### 3.3 SPRAL confirms the cost is inherent

The `spral-expert` read `ref/spral/src/scaling.f90::hungarian_match`
(lines 938-1171). Findings:
- The inner scan (lines 1082-1118) walks the **full** matched column on every
  settle, with only per-entry filters (`l(i) ≥ up` = closed-vertex skip, line
  1084; `dnew ≥ csp` upper-bound, line 1088) — **no range cut, no dense-column
  special case**.
- `dualv` is computed **once at the end** (lines 1158-1167), never maintained
  live, never used to shorten a scan. The running scalar `vj` (line 1081) is the
  feral `vj`.
- The *only* dense-aware logic in the file is the greedy-init claim guard
  (`ptr(j+1)-ptr(j) > m/10`, line 857) — which feral already mirrors
  (hungarian.rs:336) — and it does nothing for the matched-and-rescanned case.
- **Conclusion: SPRAL has the identical `O(searches × dense_deg)` cost and would
  be equally slow on rocket.** feral's port is faithful; there is no SPRAL trick
  to adopt.

## 4. Empirical result (summary)

| | edge_scans | inner_scan_skips | edges_saved | wall (release) |
|---|---|---|---|---|
| baseline | 3.71e8 | — | — | 3.69–3.96 s |
| with lb-skip | 3.71e8 | **0** | **0** | 3.96 s (unchanged) |

The lb-skip code was reverted (dead weight: never fires, adds an O(nnz) pass +
a branch per pop). The constraint hard rules (matching/scaling must not change
without inertia validation) are why a non-behavior-preserving variant was not
pursued.

## 5. Recommendation

1. **Accept the dense-column MC64 cost as inherent.** It is a one-time
   per-pattern symbolic cost (~3.7 s on a 90k KKT), matches the SPRAL/MC64
   reference, and cannot be reduced by any behavior-preserving change to the
   matching kernel. This closes the "dense-column fast path" lead from the
   scaling audit §8.1 option (a) with a definitive negative result.
2. **The only remaining lever is avoiding MC64 scaling on single-dense-column
   matrices** (audit §8.1 option (b)). For rocket this is blocked: its degree-
   38401 column trips the `pick_scaling_strategy` arrow-head gate
   (`max_col_nnz > 32`, mod.rs:653), so the matrix's scaling *resolves to* MC64,
   and the symbolic-side cache is reused rather than recomputed. Cheapening it
   would mean **changing the chosen scaling** (e.g. InfNorm) on such KKTs — a
   numerical-quality question (does the inertia/residual gate hold without MC64
   on a zero-(2,2)-block KKT with a dense coupling column?) that requires a
   corpus study and human approval per the constraints. Not pursued here.
3. **Option (b) for non-MC64-scaled matrices is a separate, safe win** (the
   symbolic LdltCompress MC64 is purely speculative for compression when the
   resolved scaling is Identity/External/InfNorm). Deferred; tracked for a
   future session.
