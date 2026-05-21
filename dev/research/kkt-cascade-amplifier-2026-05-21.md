# Research — the delayed-pivot cascade amplifier (Track A2)

Date: 2026-05-21
Session: 2026-05-21-01
Issue: #8 / #46 family (`pinene_3200` iter 6-9 factor-time explosion,
98 % of that problem's whole-solve wall).
Supersedes the framing in: `dev/plans/per-factor-cost-cluster.md`
Track A2 ("fix the 2×2 stability gate"), and the journal over-read
of 2026-05-21-01 §20:55 / §21:55.

## STATUS — pre-implementation research note

The fix is **not yet chosen**. This note establishes the mechanism
from instrumentation, lays out three candidate fixes, and recommends
a sequencing for human review. The kernel change is
correctness-critical (it touches inertia) — per the FERAL feature
lifecycle the implementation needs a plan + tests-first + an external
oracle, gated on this note.

## 1. The problem

`pinene_3200`'s IPM KKT iterates 6-9 each take 64-208 s to factor —
the production-path replay is 493.9 s total, 98 % of the whole-solve
wall. The matrix is a zero-(2,2)-block saddle KKT (issue #46 family):

    K = [ H   Bᵀ ]
        [ B   0  ]

Every constraint column has a structurally zero diagonal; it cannot
be a 1×1 pivot and must be paired into a 2×2 saddle pivot
`[[0,b],[b,a]]` (det `−b² < 0`, one positive / one negative
eigenvalue — inertia-exact by construction). The #46 fix
(`scalar_pivot_step` matching-aware 2×2 partner selection,
2026-05-20) is in `pinene`'s production path
(`preproc = LdltCompress` confirmed) yet `pinene` still cascades:
`factor_nnz = 165.7M` vs the symbolic no-delay estimate 2.39M — a
**69× fill blowup**.

## 2. Method

`probe_issue46_supernode pinene_3200_0009.mtx` (n=127995,
nnz_lower=732976, 63995 MC64 pairs) under two configs:

- **Config 1** — `LdltCompress` + `allow_delayed_pivots = true`: the
  production path.
- **Config 2** — `LdltCompress` + `allow_delayed_pivots = false`:
  static pivoting (force-accept, no delay).

Instrumentation added this session: profiling-gated atomic counters
in `panel_diag` (`src/dense/factor.rs`) at the two scalar delay
sites — `SCALAR_2X2_DELAY_{GROWTH,DET,BOTH,NEGDET}` at
`scalar_pivot_step`'s 2×2 gate, `SCALAR_1X1_DELAY{,_TINY}` at
`try_reject_1x1_frontal`'s 1×1 threshold gate.

## 3. Measurements

### 3.1 Config 1 — production path

| metric        | value           |
|---------------|-----------------|
| factor time   | 183 s           |
| `factor_nnz`  | 165 720 251     |
| blowup        | 69.48×          |
| `n_2x2`       | 62 302          |
| `n_delayed`   | **133 648**     |
| inertia       | (64000,63995,0) — **correct** |

Delay-cause counters:

| counter                   | value | meaning                              |
|---------------------------|------:|--------------------------------------|
| `scalar_2x2_delay_growth` |  1059 | 2×2 candidate failed Duff-Reid growth bound |
| `scalar_2x2_delay_both`   |    37 | failed growth bound AND SSIDS det floor |
| `scalar_2x2_delay_det`    |     0 | failed SSIDS det floor alone         |
| `scalar_2x2_delay_negdet` |   816 | of the 1096 2×2 delays, those with det < 0 |
| `scalar_1x1_delay`        |  2840 | 1×1 candidate below column-relative threshold |
| `scalar_1x1_delay_tiny`   |     0 | of the 2840, those with \|d\| ≤ zero_tol |

**Total delay events = 1059 + 37 + 2840 = 3936.**

### 3.2 Config 2 — static pivoting (force-accept)

| metric        | value           |
|---------------|-----------------|
| factor time   | 50 ms           |
| `factor_nnz`  | 2 988 497       |
| blowup        | **1.25×**       |
| `n_delayed`   | 0               |
| inertia       | (64001,63994,0) — **one wrong sign** |

## 4. The mechanism — an amplifier × two triggers

### 4.1 The amplifier — break-on-first-delay

The Bunch-Kaufman driver loop (`src/dense/factor.rs:1719-1849`):

```rust
while k < ncol {
    ... panel or scalar step ...
    PivotStepResult::Delayed => break,   // 1751, 1841
}
let nelim = k;
let n_delayed = ncol - nelim;
```

When **any** pivot delays, the loop `break`s and the supernode
forfeits its **entire remaining tail** — all `ncol − nelim` columns
become delayed pivots promoted to the parent front.

3936 delay events produce `n_delayed = 133 648` ⇒ **~34 columns
forfeited per event**. The distribution is bimodal: most supernodes
forfeit a short tail; the few near-root conduit supernodes forfeit
thousands each (session 2026-05-21-01 A1: conduit supernodes
10483/10484/10485 eliminate only 4 / 11 / 493 of their thousands of
columns). The tail-forfeit is the cascade multiplier.

**Config 2 is the proof the forfeited columns are recoverable.**
With delays disabled, force-accept eliminates every column in its
own supernode (`n_delayed = 0`) and produces a *healthy* 1.25× factor
in 50 ms. So when a stuck pivot is set aside, the **rest of the
supernode is pivotable** — the break-on-first forfeit is throwing
away pivotable work, not avoiding genuinely-stuck columns.

### 4.2 Trigger A — split MC64 pairs (2840 events, 1×1 delays)

The 2840 1×1 delays are **not** zero-diagonal saddle columns
(`scalar_1x1_delay_tiny = 0`): every one has
`zero_tol < |d| ≤ pivot_threshold·col_max`. They reach the
last-resort 1×1 because `scalar_pivot_step` found **no 2×2 partner**
(`a[k][k+1] == 0` ⇒ `partner = None`) — the matched partner is not
co-located at `k+1`.

These line up with the probe's **3781 split-across-supernodes MC64
pairs** (of 63995; 60214 same-supernode, 60122 of those adjacent).
#46's `LdltCompress` co-locates 93.9 % of pairs; the residual ~6 %
split across supernodes and delay until they reach a front holding
their partner. This is the **#46 ordering-gap residual**, not a
kernel bug — the kernel cannot 2×2 a column whose partner is not in
the front.

### 4.3 Trigger B — growth-bound rejection of co-located saddle 2×2s
(1096 events)

The 1096 `scalar_2x2_delay_{growth,both}` events are co-located
saddle pairs that **did** form a `{k,k+1}` 2×2 candidate (the #46
partner mechanism worked) but the candidate then failed the
**Duff-Reid growth bound** (`src/dense/factor.rs:3269-3270`):

```
reject iff (|d22|·rmax + amax·tmax)·u > |det|
        OR (|d11|·tmax + amax·rmax)·u > |det|
```

where `rmax`/`tmax` are the trailing column maxes of the two pivot
columns. The SSIDS scale-invariant det floor fires **zero** times
alone (`scalar_2x2_delay_det = 0`) — it is not involved. 816 of the
1096 rejected 2×2s are genuine indefinite saddles (`det < 0`).

The growth bound is a **stability** gate, not a correctness gate:
admitting a high-growth 2×2 produces large `L` entries → needs
iterative refinement, but `count_2x2_inertia_val` computes inertia
from the exact 2×2 eigenvalues regardless, so **admitting these 1096
keeps inertia exact**. The feedback loop is: cascade → dense fill →
larger `rmax`/`tmax` → more growth rejections → more cascade.

## 5. Candidate fixes

### Fix 1 — fine-grained delay (the amplifier). PRIMARY.

Replace break-on-first-delay with **swap-to-boundary**: when the
pivot at column `k` delays, swap it with the last fully-summed
column (`ncol_eff − 1`), decrement `ncol_eff`, and continue
eliminating at `k`. After the loop, columns `[nelim .. ncol)` are the
genuinely-stuck delayed columns (already the contribution block's
leading `n_delayed` rows — `factor.rs:1883-1891`, `factorize.rs:3258`).

- **Inertia-exact by construction**: delayed pivoting is the correct
  algorithm; a stuck column promoted to the parent is re-attempted
  with more context. No force-accept, no perturbation. Unlike
  `cascade_break`, which corrupts iter-9 inertia.
- **Payoff**: forfeits 1 column per stuck pivot instead of ~34.
  Config 2 is strong evidence the recovered columns are pivotable.
  Exact `n_delayed` after the fix is bounded below by ~3936 and
  cannot be read off config 1 (the 3936 event count is "supernodes
  that hit ≥1 delay", capped at one event per supernode by the
  current `break`); the true stuck-column count needs the
  implementation — or a swap-to-boundary probe — to measure.
- **Cost**: a real kernel change. The panel path
  (`lblt_panel_frontal`) also `break`s on `PanelStatus::Delayed`
  (`factor.rs:1844`) and would need the same treatment, though for
  `pinene` specifically `PANEL_DELAYED = 0` (all delays are scalar).
  `gamma0`/argmax searches must use `ncol_eff` consistently.

### Fix 2 — matching-aware growth-bound exemption (trigger B)

For a `{k,k+1}` 2×2 candidate that is a **co-located MC64-matched
saddle pair**, exempt it from the Duff-Reid growth bound (still
subject to the SSIDS det floor as a true-singularity guard). The
pair is inertia-exact by construction; the growth bound only governs
`L`-element growth, which `needs_refinement` already flags for
iterative refinement. Removes up to 1096 delay events.

Requires the kernel to know which `{k,k+1}` pairs are MC64-matched —
the matching is computed in the analysis phase but not currently
threaded into `scalar_pivot_step`. #46's note explicitly flagged
"the kernel never consults the MC64 pairing" as the structural gap.

### Fix 3 — tighter co-location for split pairs (trigger A)

Reduce the 3781 split-across-supernodes MC64 pairs by making
`LdltCompress` / the supernode amalgamation keep matched pairs in the
same front. This is analysis-phase work
(`src/symbolic/ldlt_compress.rs`). Removes up to 2840 delay events at
the source, but is the deepest change and may interact with
fill-reducing ordering quality.

## 6. Recommendation

Sequence by leverage and risk:

1. **Fix 1 (fine-grained delay)** first. It is the largest single
   lever (attacks the ×34 amplifier), is inertia-exact by
   construction (no new correctness surface), and is independent of
   the matching machinery. It will not fully close the gap if many
   columns are recursively stuck, but config 2 indicates few are.
2. Re-measure. If a residual cascade remains, add **Fix 2**
   (growth-bound exemption) — small, bounded, removes ≤1096 events.
3. **Fix 3** only if triggers A still dominate after 1+2.

Do **not** pursue `cascade_break` as the production fix: `CB=1` gives
the 100× speedup but force-accepts near-zero pivots and corrupts
iter-9 inertia (got (.,.,1), want (.,.,0)) — the same inadmissibility
#46 recorded for `allow_delayed_pivots = false`. The bounded-Δ CB
repair (scale `cascade_break_eps` by `‖A‖∞`, sign the perturbation)
remains a fallback only.

## 7. Oracles (test policy)

- Inertia on a non-singular synthetic saddle KKT — `(nv, nc, 0)` by
  the saddle-point inertia theorem (`H` SPD + `B` full row rank;
  Benzi, Golub & Liesen, Acta Numerica 2005, §3.4). External math.
- Residual `‖A·x − b‖ / ‖b‖` — a mathematical identity.
- `factor_nnz ≤ C × factor_nnz_estimate` — measurement / cascade
  guard. The existing `tests/issue_46_saddle_kkt_cascade.rs` uses
  `5×`; this note's fixes should tighten the synthetic-case bound.
- Bit-identical inertia + residual on the parity corpus for every
  non-cascading matrix — Fix 1 must not change well-conditioned
  factors.

## 8. Key files

- `src/dense/factor.rs` — `scalar_pivot_step` (delay sites
  ~3304/3471), the BK driver loop (1719-1849, the amplifier),
  `panel_diag` (the instrumentation), `lblt_panel_frontal` (panel
  delay path).
- `src/numeric/factorize.rs` — `n_delayed` aggregation (857), delayed
  columns into the parent front (2050-2109, 3258-3329).
- `src/bin/probe_issue46_supernode.rs` — the measurement probe
  (config 3 now opt-in behind `PROBE_ALL=1`).
- `dev/research/kkt-zero-2x2-block-cascade-2026-05-20.md` — the #46
  fix (partner selection); this note is its sequel.
